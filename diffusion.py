"""
TabDDPM-style diffusion model for synthetic tabular data generation.

Implements a Denoising Diffusion Probabilistic Model (DDPM) adapted for
tabular EHR data. Trains on the preprocessed patient feature matrix and
generates synthetic records in the same feature space as the SDV generators.

Reference: Kotelnikov et al. (2023) "TabDDPM: Modelling Tabular Data with
Diffusion Models" — https://arxiv.org/abs/2209.15421

Usage:
  python diffusion.py
  python diffusion.py --epochs 300 --steps 1000 --batch-size 1024

Output:
  data/synthetic/diffusion_synthetic.csv
  outputs/models/diffusion_model.pt
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------


class LinearNoiseSchedule:
    """
    Linear beta schedule from beta_start → beta_end over n_steps.
    Precomputes all quantities needed for forward and reverse diffusion.
    """

    def __init__(
        self,
        n_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: torch.device = torch.device("cpu"),
    ):
        self.n_steps = n_steps
        self.device = device

        betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.ones(1, device=device), alpha_bar[:-1]])

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = alpha_bar.sqrt()
        self.sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt()
        self.sqrt_recip_alphas = (1.0 / alphas).sqrt()
        # Posterior variance for reverse process
        self.posterior_variance = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: add noise to x0 at timestep t."""
        sqrt_ab = self.sqrt_alpha_bar[t].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
        return sqrt_ab * x0 + sqrt_one_minus * noise

    def p_sample(
        self, model: nn.Module, xt: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """One reverse diffusion step: denoise xt → x_{t-1}."""
        # Predict noise
        t_tensor = t * torch.ones(xt.shape[0], dtype=torch.long, device=self.device)
        predicted_noise = model(xt, t_tensor)

        # Compute mean of p(x_{t-1} | x_t)
        betas_t = self.betas[t]
        sqrt_one_minus_t = self.sqrt_one_minus_alpha_bar[t]
        sqrt_recip_t = self.sqrt_recip_alphas[t]

        mean = sqrt_recip_t * (xt - betas_t / sqrt_one_minus_t * predicted_noise)

        if t == 0:
            return mean
        posterior_var_t = self.posterior_variance[t]
        noise = torch.randn_like(xt)
        return mean + posterior_var_t.sqrt() * noise


# ---------------------------------------------------------------------------
# Denoising network
# ---------------------------------------------------------------------------


class SinusoidalTimestepEmbedding(nn.Module):
    """Encode scalar timestep t into a dense embedding vector."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResidualBlock(nn.Module):
    """MLP residual block: LayerNorm → Linear → SiLU → Linear + skip."""

    def __init__(self, dim: int, time_embed_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff1 = nn.Linear(dim, dim)
        self.ff2 = nn.Linear(dim, dim)
        self.time_proj = nn.Linear(time_embed_dim, dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.act(self.ff1(h)) + self.time_proj(self.act(t_emb))
        h = self.ff2(h)
        return x + h


class DenoisingMLP(nn.Module):
    """
    MLP that predicts the noise added at timestep t.
    Architecture: project input → stack of residual blocks → project output.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dims: list[int] = None,
        time_embed_dim: int = 128,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 1024, 1024, 512]

        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
        )

        # Input projection
        self.input_proj = nn.Linear(n_features, hidden_dims[0])

        # Residual blocks (all same dimension for simplicity)
        self.blocks = nn.ModuleList()
        dims = hidden_dims
        for i in range(len(dims) - 1):
            if dims[i] != dims[i + 1]:
                # Dimension change block (no residual)
                self.blocks.append(nn.Sequential(
                    nn.LayerNorm(dims[i]),
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.SiLU(),
                ))
            else:
                self.blocks.append(ResidualBlock(dims[i], time_embed_dim))

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], n_features),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        h = self.input_proj(x)
        for block in self.blocks:
            if isinstance(block, ResidualBlock):
                h = block(h, t_emb)
            else:
                h = block(h)
        return self.output_proj(h)


# ---------------------------------------------------------------------------
# TabDDPM model
# ---------------------------------------------------------------------------


class TabDDPM:
    """
    Tabular DDPM: wraps the noise schedule + denoising network into a
    sklearn-style fit/sample interface matching the SDV generators.
    """

    def __init__(
        self,
        n_steps: int = 1000,
        epochs: int = 300,
        batch_size: int = 1024,
        lr: float = 1e-3,
        hidden_dims: list[int] = None,
        time_embed_dim: int = 128,
        device: str = "auto",
    ):
        self.n_steps = n_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.hidden_dims = hidden_dims or [512, 1024, 1024, 512]
        self.time_embed_dim = time_embed_dim

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: DenoisingMLP | None = None
        self.schedule: LinearNoiseSchedule | None = None
        self.columns: list[str] | None = None
        self.n_features: int = 0

    def fit(self, df: pd.DataFrame) -> "TabDDPM":
        self.columns = list(df.columns)
        self.n_features = len(self.columns)

        print(f"\n[TabDDPM] Device: {self.device}")
        print(f"[TabDDPM] Features: {self.n_features}  |  Training rows: {len(df):,}")
        print(f"[TabDDPM] Steps: {self.n_steps}  |  Epochs: {self.epochs}  |  Batch: {self.batch_size}")

        # Prepare data tensor
        X = torch.tensor(df.values, dtype=torch.float32, device=self.device)
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Build model and schedule
        self.schedule = LinearNoiseSchedule(self.n_steps, device=self.device)
        self.model = DenoisingMLP(
            n_features=self.n_features,
            hidden_dims=self.hidden_dims,
            time_embed_dim=self.time_embed_dim,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[TabDDPM] Model parameters: {n_params:,}")

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )

        self.model.train()
        t0 = time.time()

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for (batch,) in loader:
                # Sample random timesteps
                t = torch.randint(0, self.n_steps, (batch.shape[0],), device=self.device)
                noise = torch.randn_like(batch)

                # Forward diffusion
                x_noisy = self.schedule.q_sample(batch, t, noise)

                # Predict noise and compute loss
                predicted_noise = self.model(x_noisy, t)
                loss = F.mse_loss(predicted_noise, noise)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()

            scheduler.step()

            if epoch % 50 == 0 or epoch == 1:
                avg_loss = epoch_loss / len(loader)
                elapsed = time.time() - t0
                print(f"  Epoch {epoch:4d}/{self.epochs}  loss={avg_loss:.6f}  elapsed={elapsed:.0f}s")

        elapsed = time.time() - t0
        print(f"[TabDDPM] Training complete in {elapsed:.1f}s")
        return self

    @torch.no_grad()
    def sample(self, num_rows: int) -> pd.DataFrame:
        """Generate synthetic rows via iterative denoising from pure noise."""
        assert self.model is not None, "Call fit() before sample()"
        self.model.eval()

        print(f"[TabDDPM] Sampling {num_rows:,} rows via {self.n_steps} denoising steps...")
        t0 = time.time()

        # Start from pure noise — all rows at once
        x = torch.randn(num_rows, self.n_features, device=self.device)

        # Reverse diffusion — vectorized over all rows, loop only over timesteps
        for step in tqdm(reversed(range(self.n_steps)), total=self.n_steps, desc="Denoising"):
            t_tensor = torch.full((num_rows,), step, dtype=torch.long, device=self.device)
            predicted_noise = self.model(x, t_tensor)

            betas_t = self.schedule.betas[step]
            sqrt_one_minus_t = self.schedule.sqrt_one_minus_alpha_bar[step]
            sqrt_recip_t = self.schedule.sqrt_recip_alphas[step]

            mean = sqrt_recip_t * (x - betas_t / sqrt_one_minus_t * predicted_noise)

            if step == 0:
                x = mean
            else:
                posterior_var_t = self.schedule.posterior_variance[step]
                x = mean + posterior_var_t.sqrt() * torch.randn_like(x)

        samples = x.cpu().numpy()
        elapsed = time.time() - t0
        print(f"[TabDDPM] Sampling done in {elapsed:.1f}s")

        return pd.DataFrame(samples, columns=self.columns)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "config": {
                "n_steps": self.n_steps,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "hidden_dims": self.hidden_dims,
                "time_embed_dim": self.time_embed_dim,
                "columns": self.columns,
                "n_features": self.n_features,
            }
        }, path)
        print(f"[TabDDPM] Model saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "TabDDPM":
        path = Path(path)
        checkpoint = torch.load(path, map_location="cpu")
        cfg = checkpoint["config"]
        instance = cls(
            n_steps=cfg["n_steps"],
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            hidden_dims=cfg["hidden_dims"],
            time_embed_dim=cfg["time_embed_dim"],
        )
        instance.columns = cfg["columns"]
        instance.n_features = cfg["n_features"]
        instance.schedule = LinearNoiseSchedule(instance.n_steps, device=instance.device)
        instance.model = DenoisingMLP(
            n_features=instance.n_features,
            hidden_dims=instance.hidden_dims,
            time_embed_dim=instance.time_embed_dim,
        ).to(instance.device)
        instance.model.load_state_dict(checkpoint["model_state"])
        return instance


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train TabDDPM and generate synthetic EHR data")
    parser.add_argument("--train-path",  default="data/processed/train.csv")
    parser.add_argument("--output-path", default="data/synthetic/diffusion_synthetic.csv")
    parser.add_argument("--model-path",  default="outputs/models/diffusion_model.pt")
    parser.add_argument("--target-col",  default="DECEASED")
    parser.add_argument("--epochs",      type=int, default=300)
    parser.add_argument("--steps",       type=int, default=1000)
    parser.add_argument("--batch-size",  type=int, default=1024)
    parser.add_argument("--lr",          type=float, default=1e-3)
    args = parser.parse_args()

    print("=" * 60)
    print("TabDDPM — Tabular Diffusion Model")
    print("=" * 60)

    # Load preprocessed train data
    print(f"\nLoading {args.train_path} ...")
    train_df = pd.read_csv(args.train_path)
    print(f"  {len(train_df):,} rows × {train_df.shape[1]} cols")
    print(f"  Target ({args.target_col}) positive rate: {train_df[args.target_col].mean()*100:.1f}%")

    # Train
    model = TabDDPM(
        n_steps=args.steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    model.fit(train_df)

    # Save model
    model.save(args.model_path)

    # Generate synthetic data matching train size
    print(f"\nGenerating {len(train_df):,} synthetic rows ...")
    synth_df = model.sample(num_rows=len(train_df))

    # Snap binary columns back to 0/1
    binary_cols = [
        c for c in train_df.select_dtypes(include="number").columns
        if c != args.target_col and set(train_df[c].dropna().unique()) <= {0, 1}
    ]
    for col in binary_cols:
        if col in synth_df.columns:
            synth_df[col] = synth_df[col].round().clip(0, 1).astype(int)

    # Save synthetic CSV
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    synth_df.to_csv(output_path, index=False)

    # Summary
    synth_pos_rate = synth_df[args.target_col].mean() * 100
    real_pos_rate = train_df[args.target_col].mean() * 100
    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"{'='*60}")
    print(f"  Synthetic rows  : {len(synth_df):,}")
    print(f"  Positive rate   : {synth_pos_rate:.1f}%  (real: {real_pos_rate:.1f}%)")
    print(f"  Saved to        : {output_path}")

    manifest = {
        "generator": "tabddpm",
        "train_rows": len(train_df),
        "synth_rows": len(synth_df),
        "target_col": args.target_col,
        "synth_positive_rate": round(synth_pos_rate / 100, 4),
        "real_positive_rate": round(real_pos_rate / 100, 4),
        "epochs": args.epochs,
        "n_steps": args.steps,
    }
    manifest_path = output_path.parent / "diffusion_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest        : {manifest_path}")


if __name__ == "__main__":
    main()
