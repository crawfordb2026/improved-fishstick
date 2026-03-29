"""
TabDDPM model classes for SageMaker training.
Imported by train.py when --generator diffusion is selected.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class LinearNoiseSchedule:
    def __init__(self, n_steps: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 0.02, device: torch.device = torch.device('cpu')):
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
        self.posterior_variance = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)

    def q_sample(self, x0, t, noise):
        sqrt_ab = self.sqrt_alpha_bar[t].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
        return sqrt_ab * x0 + sqrt_one_minus * noise


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, time_embed_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff1 = nn.Linear(dim, dim)
        self.ff2 = nn.Linear(dim, dim)
        self.time_proj = nn.Linear(time_embed_dim, dim)
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.norm(x)
        h = self.act(self.ff1(h)) + self.time_proj(self.act(t_emb))
        h = self.ff2(h)
        return x + h


class DenoisingMLP(nn.Module):
    def __init__(self, n_features: int, hidden_dims=None, time_embed_dim: int = 128):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 1024, 1024, 512]

        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
        )
        self.input_proj = nn.Linear(n_features, hidden_dims[0])

        self.blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            if hidden_dims[i] != hidden_dims[i + 1]:
                self.blocks.append(nn.Sequential(
                    nn.LayerNorm(hidden_dims[i]),
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.SiLU(),
                ))
            else:
                self.blocks.append(ResidualBlock(hidden_dims[i], time_embed_dim))

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1]),
            nn.Linear(hidden_dims[-1], n_features),
        )

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        h = self.input_proj(x)
        for block in self.blocks:
            if isinstance(block, ResidualBlock):
                h = block(h, t_emb)
            else:
                h = block(h)
        return self.output_proj(h)


class TabDDPM:
    def __init__(self, n_steps=1000, epochs=300, batch_size=1024, lr=1e-3,
                 hidden_dims=None, time_embed_dim=128):
        self.n_steps = n_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.hidden_dims = hidden_dims or [512, 1024, 1024, 512]
        self.time_embed_dim = time_embed_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.schedule = None
        self.columns = None
        self.n_features = 0

    def fit(self, df):
        import pandas as pd
        self.columns = list(df.columns)
        self.n_features = len(self.columns)

        print(f'[TabDDPM] Device: {self.device}')
        print(f'[TabDDPM] Features: {self.n_features}  Rows: {len(df):,}')
        print(f'[TabDDPM] Steps: {self.n_steps}  Epochs: {self.epochs}  Batch: {self.batch_size}')

        X = torch.tensor(df.values, dtype=torch.float32, device=self.device)
        loader = DataLoader(TensorDataset(X), batch_size=self.batch_size, shuffle=True)

        self.schedule = LinearNoiseSchedule(self.n_steps, device=self.device)
        self.model = DenoisingMLP(
            n_features=self.n_features,
            hidden_dims=self.hidden_dims,
            time_embed_dim=self.time_embed_dim,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f'[TabDDPM] Parameters: {n_params:,}')

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        self.model.train()
        t0 = time.time()
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for (batch,) in loader:
                t = torch.randint(0, self.n_steps, (batch.shape[0],), device=self.device)
                noise = torch.randn_like(batch)
                x_noisy = self.schedule.q_sample(batch, t, noise)
                predicted_noise = self.model(x_noisy, t)
                loss = F.mse_loss(predicted_noise, noise)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()

            if epoch % 50 == 0 or epoch == 1:
                print(f'  Epoch {epoch:4d}/{self.epochs}  loss={epoch_loss/len(loader):.6f}  elapsed={time.time()-t0:.0f}s')

        print(f'[TabDDPM] Training complete in {time.time()-t0:.1f}s')
        return self

    @torch.no_grad()
    def sample(self, num_rows: int):
        import pandas as pd
        self.model.eval()
        print(f'[TabDDPM] Sampling {num_rows:,} rows ...')
        t0 = time.time()

        x = torch.randn(num_rows, self.n_features, device=self.device)
        for step in reversed(range(self.n_steps)):
            t_tensor = torch.full((num_rows,), step, dtype=torch.long, device=self.device)
            predicted_noise = self.model(x, t_tensor)

            betas_t = self.schedule.betas[step]
            sqrt_one_minus_t = self.schedule.sqrt_one_minus_alpha_bar[step]
            sqrt_recip_t = self.schedule.sqrt_recip_alphas[step]
            mean = sqrt_recip_t * (x - betas_t / sqrt_one_minus_t * predicted_noise)

            if step == 0:
                x = mean
            else:
                x = mean + self.schedule.posterior_variance[step].sqrt() * torch.randn_like(x)

            if step % 200 == 0:
                print(f'  Denoising step {self.n_steps - step}/{self.n_steps} ...')

        print(f'[TabDDPM] Sampling done in {time.time()-t0:.1f}s')
        return pd.DataFrame(x.cpu().numpy(), columns=self.columns)
