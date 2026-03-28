"""
Realism evaluation: how closely does synthetic data resemble real data?

Metrics:
  - KS statistic per numeric column (lower = more similar)
  - Wasserstein distance per numeric column
  - Correlation matrix Frobenius distance
  - Real-vs-synthetic classifier AUROC (lower = more similar)
  - PCA and UMAP visualisations

A lower KS / Wasserstein / correlation distance is better.
A real-vs-synthetic AUROC closer to 0.5 means the classifier cannot tell them apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Column-level metrics
# ---------------------------------------------------------------------------


def ks_per_column(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    target_col: str = "Class",
) -> pd.DataFrame:
    """KS test for each numeric column. Returns DataFrame with stat and p-value."""
    cols = [c for c in real_df.select_dtypes(include=[np.number]).columns if c != target_col]
    rows = []
    for col in cols:
        stat, pval = stats.ks_2samp(
            real_df[col].dropna().values,
            synth_df[col].dropna().values,
        )
        rows.append({"column": col, "ks_stat": stat, "ks_pvalue": pval})
    return pd.DataFrame(rows).sort_values("ks_stat", ascending=False)


def wasserstein_per_column(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    target_col: str = "Class",
) -> pd.DataFrame:
    """Wasserstein-1 distance per numeric column."""
    cols = [c for c in real_df.select_dtypes(include=[np.number]).columns if c != target_col]
    rows = []
    for col in cols:
        dist = wasserstein_distance(
            real_df[col].dropna().values,
            synth_df[col].dropna().values,
        )
        rows.append({"column": col, "wasserstein": dist})
    return pd.DataFrame(rows).sort_values("wasserstein", ascending=False)


# ---------------------------------------------------------------------------
# Correlation distance
# ---------------------------------------------------------------------------


def correlation_distance(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    target_col: str = "Class",
) -> float:
    """Frobenius norm of the difference between real and synthetic correlation matrices."""
    cols = [c for c in real_df.select_dtypes(include=[np.number]).columns if c != target_col]
    corr_real = real_df[cols].corr().fillna(0).values
    corr_synth = synth_df[cols].corr().fillna(0).values
    return float(np.linalg.norm(corr_real - corr_synth, ord="fro"))


# ---------------------------------------------------------------------------
# Real-vs-synthetic classifier test
# ---------------------------------------------------------------------------


def real_vs_synthetic_auroc(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    target_col: str = "Class",
    n_splits: int = 5,
    random_state: int = 42,
) -> float:
    """
    Train a Random Forest to distinguish real (label=1) from synthetic (label=0).
    AUROC closer to 0.5 means the generator fools the classifier — better quality.
    """
    feature_cols = [c for c in real_df.select_dtypes(include=[np.number]).columns if c != target_col]

    n = min(len(real_df), len(synth_df))
    real_sample = real_df[feature_cols].sample(n=n, random_state=random_state)
    synth_sample = synth_df[feature_cols].sample(n=n, random_state=random_state)

    X = pd.concat([real_sample, synth_sample], ignore_index=True).values
    y = np.array([1] * n + [0] * n)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aurocs = []
    for train_idx, test_idx in cv.split(X, y):
        clf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
        clf.fit(X[train_idx], y[train_idx])
        proba = clf.predict_proba(X[test_idx])[:, 1]
        aurocs.append(roc_auc_score(y[test_idx], proba))

    mean_auroc = float(np.mean(aurocs))
    print(f"  Real-vs-Synthetic AUROC: {mean_auroc:.4f} (ideal ≈ 0.500)")
    return mean_auroc


# ---------------------------------------------------------------------------
# Summary scorecard
# ---------------------------------------------------------------------------


def realism_scorecard(
    real_df: pd.DataFrame,
    synth_datasets: dict[str, pd.DataFrame],
    target_col: str = "Class",
) -> pd.DataFrame:
    """Compute full realism scorecard for all generators."""
    rows = []
    for name, synth_df in synth_datasets.items():
        print(f"\nEvaluating realism: {name}")
        ks = ks_per_column(real_df, synth_df, target_col)
        wd = wasserstein_per_column(real_df, synth_df, target_col)
        corr_dist = correlation_distance(real_df, synth_df, target_col)
        disc_auroc = real_vs_synthetic_auroc(real_df, synth_df, target_col)

        rows.append({
            "generator": name,
            "ks_mean": float(ks["ks_stat"].mean()),
            "ks_max": float(ks["ks_stat"].max()),
            "wasserstein_mean": float(wd["wasserstein"].mean()),
            "wasserstein_max": float(wd["wasserstein"].max()),
            "correlation_distance": corr_dist,
            "discriminator_auroc": disc_auroc,
        })

    scorecard = pd.DataFrame(rows)
    print("\n--- REALISM SCORECARD ---")
    print(scorecard.to_string(index=False))
    return scorecard


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def plot_distributions(
    real_df: pd.DataFrame,
    synth_datasets: dict[str, pd.DataFrame],
    target_col: str = "Class",
    n_cols_to_plot: int = 6,
    output_dir: Optional[str | Path] = None,
) -> None:
    """Plot KDE overlays for the top n feature columns."""
    feature_cols = [
        c for c in real_df.select_dtypes(include=[np.number]).columns
        if c != target_col
    ][:n_cols_to_plot]

    n_generators = len(synth_datasets)
    fig, axes = plt.subplots(
        nrows=n_cols_to_plot,
        ncols=n_generators + 1,
        figsize=(4 * (n_generators + 1), 3 * n_cols_to_plot),
    )

    palette = sns.color_palette("tab10", n_generators + 1)

    for row_idx, col in enumerate(feature_cols):
        # Real
        ax = axes[row_idx][0]
        sns.kdeplot(real_df[col].dropna(), ax=ax, color=palette[0], fill=True, alpha=0.4)
        ax.set_title(f"Real — {col}", fontsize=8)
        ax.set_ylabel("")

        # Each generator
        for gen_idx, (name, synth_df) in enumerate(synth_datasets.items()):
            ax = axes[row_idx][gen_idx + 1]
            sns.kdeplot(real_df[col].dropna(), ax=ax, color=palette[0], fill=True, alpha=0.3, label="real")
            sns.kdeplot(synth_df[col].dropna(), ax=ax, color=palette[gen_idx + 1], fill=True, alpha=0.3, label=name)
            ax.set_title(f"{name} — {col}", fontsize=8)
            ax.set_ylabel("")
            if row_idx == 0:
                ax.legend(fontsize=7)

    plt.suptitle("Feature Distribution Comparison: Real vs Synthetic", fontsize=12, y=1.01)
    plt.tight_layout()

    if output_dir:
        path = Path(output_dir) / "distribution_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved distribution plot to {path}")
    plt.show()


def plot_correlation_heatmaps(
    real_df: pd.DataFrame,
    synth_datasets: dict[str, pd.DataFrame],
    target_col: str = "Class",
    output_dir: Optional[str | Path] = None,
) -> None:
    """Side-by-side correlation heatmaps."""
    cols = [c for c in real_df.select_dtypes(include=[np.number]).columns if c != target_col]
    n = len(synth_datasets) + 1
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))

    dfs = {"Real": real_df}
    dfs.update(synth_datasets)

    vmin, vmax = -1, 1
    for ax, (name, df) in zip(axes, dfs.items()):
        corr = df[cols].corr()
        sns.heatmap(corr, ax=ax, cmap="coolwarm", vmin=vmin, vmax=vmax,
                    square=True, linewidths=0, cbar=ax == axes[-1], xticklabels=False, yticklabels=False)
        ax.set_title(name, fontsize=10)

    plt.suptitle("Correlation Matrix Comparison", fontsize=12)
    plt.tight_layout()

    if output_dir:
        path = Path(output_dir) / "correlation_heatmaps.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved correlation heatmaps to {path}")
    plt.show()


def plot_pca_overlap(
    real_df: pd.DataFrame,
    synth_datasets: dict[str, pd.DataFrame],
    target_col: str = "Class",
    n_samples: int = 2000,
    output_dir: Optional[str | Path] = None,
) -> None:
    """PCA 2D scatter: real vs each generator."""
    from sklearn.decomposition import PCA

    cols = [c for c in real_df.select_dtypes(include=[np.number]).columns if c != target_col]
    rng = np.random.default_rng(42)

    n_gen = len(synth_datasets)
    fig, axes = plt.subplots(1, n_gen, figsize=(5 * n_gen, 5))
    if n_gen == 1:
        axes = [axes]

    real_sample = real_df[cols].sample(n=min(n_samples, len(real_df)), random_state=42)

    palette = {"real": "#1f77b4"}
    gen_colors = sns.color_palette("tab10", n_gen)

    for ax, (name, synth_df), color in zip(axes, synth_datasets.items(), gen_colors):
        synth_sample = synth_df[cols].sample(n=min(n_samples, len(synth_df)), random_state=42)
        combined = pd.concat([real_sample, synth_sample], ignore_index=True)

        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(StandardScaler().fit_transform(combined.values))

        n_real = len(real_sample)
        ax.scatter(coords[:n_real, 0], coords[:n_real, 1], alpha=0.3, s=5, label="real", color="#1f77b4")
        ax.scatter(coords[n_real:, 0], coords[n_real:, 1], alpha=0.3, s=5, label=name, color=color)
        var = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({var[0]:.1%})", fontsize=9)
        ax.set_ylabel(f"PC2 ({var[1]:.1%})", fontsize=9)
        ax.set_title(f"PCA: Real vs {name}", fontsize=10)
        ax.legend(fontsize=8, markerscale=3)

    plt.suptitle("PCA Overlap: Real vs Synthetic", fontsize=12)
    plt.tight_layout()

    if output_dir:
        path = Path(output_dir) / "pca_overlap.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved PCA plot to {path}")
    plt.show()
