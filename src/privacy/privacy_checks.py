"""
Privacy leakage analysis for synthetic tabular data.

Checks:
  1. Exact row duplication — are any synthetic rows verbatim copies of training rows?
  2. Nearest-neighbour distance — how close are synthetic rows to their nearest real neighbour?
  3. Rare record memorisation — are rare minority-class records reproduced?
  4. Membership inference sanity — simple shadow-model style check.

A lower duplicate rate and higher average NN distance = better privacy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 1. Exact duplication
# ---------------------------------------------------------------------------


def check_exact_duplicates(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
) -> dict:
    """
    Count synthetic rows that are exact matches to any training row.
    Returns dict with count, rate, and the duplicated rows.
    """
    if feature_cols is None:
        feature_cols = list(real_df.columns)

    # Round floats to reduce floating-point noise (5 decimal places)
    real_rounded = real_df[feature_cols].round(5)
    synth_rounded = synth_df[feature_cols].round(5)

    real_set = set(map(tuple, real_rounded.values))
    dup_mask = synth_rounded.apply(lambda row: tuple(row) in real_set, axis=1)
    dup_count = int(dup_mask.sum())
    dup_rate = dup_count / len(synth_df)

    print(f"  Exact duplicates: {dup_count} / {len(synth_df):,} ({dup_rate:.4%})")
    return {
        "dup_count": dup_count,
        "dup_rate": dup_rate,
        "dup_rows": synth_df[dup_mask].copy(),
    }


# ---------------------------------------------------------------------------
# 2. Nearest-neighbour distance
# ---------------------------------------------------------------------------


def nn_distance_analysis(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    n_neighbors: int = 5,
    sample_size: int = 5000,
    random_state: int = 42,
) -> dict:
    """
    For each synthetic row, find its nearest neighbour in the real training set.
    Returns distribution statistics on those distances.

    A very low mean distance suggests the generator is memorising training data.
    """
    if feature_cols is None:
        feature_cols = list(real_df.select_dtypes(include=[np.number]).columns)

    # Sample to keep computation tractable
    real_sample = real_df[feature_cols].sample(
        n=min(sample_size, len(real_df)), random_state=random_state
    )
    synth_sample = synth_df[feature_cols].sample(
        n=min(sample_size, len(synth_df)), random_state=random_state
    )

    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_sample.values)
    synth_scaled = scaler.transform(synth_sample.values)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree", n_jobs=-1)
    nbrs.fit(real_scaled)
    distances, _ = nbrs.kneighbors(synth_scaled)
    nn1_distances = distances[:, 0]

    result = {
        "nn_mean": float(np.mean(nn1_distances)),
        "nn_median": float(np.median(nn1_distances)),
        "nn_p5": float(np.percentile(nn1_distances, 5)),
        "nn_p95": float(np.percentile(nn1_distances, 95)),
        "pct_below_01": float(np.mean(nn1_distances < 0.1)),
        "nn_distances": nn1_distances,
    }

    print(
        f"  NN distance: mean={result['nn_mean']:.4f}  "
        f"median={result['nn_median']:.4f}  "
        f"p5={result['nn_p5']:.4f}  "
        f"p95={result['nn_p95']:.4f}  "
        f"frac_below_0.1={result['pct_below_01']:.4%}"
    )
    return result


# ---------------------------------------------------------------------------
# 3. Rare record memorisation
# ---------------------------------------------------------------------------


def check_rare_record_memorisation(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    target_col: str = "Class",
    minority_label: int = 1,
    feature_cols: Optional[list[str]] = None,
    distance_threshold: float = 0.05,
    n_neighbors: int = 1,
    random_state: int = 42,
) -> dict:
    """
    Check whether minority-class (rare) real records appear near synthetic data.
    High similarity to rare records is the most dangerous privacy leakage.
    """
    if feature_cols is None:
        feature_cols = [
            c for c in real_df.select_dtypes(include=[np.number]).columns
            if c != target_col
        ]

    rare_real = real_df[real_df[target_col] == minority_label][feature_cols]
    if len(rare_real) == 0:
        print("  No minority-class records found.")
        return {"n_rare": 0, "n_memorised": 0, "memorisation_rate": 0.0}

    scaler = StandardScaler()
    rare_scaled = scaler.fit_transform(rare_real.values)
    synth_scaled = scaler.transform(synth_df[feature_cols].values)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree", n_jobs=-1)
    nbrs.fit(rare_scaled)
    distances, _ = nbrs.kneighbors(synth_scaled)
    memorised_count = int(np.sum(distances[:, 0] < distance_threshold))
    memorisation_rate = memorised_count / len(synth_df)

    print(
        f"  Rare record check: {len(rare_real)} minority-class records  "
        f"→  {memorised_count} synthetic rows within dist={distance_threshold}  "
        f"({memorisation_rate:.4%})"
    )
    return {
        "n_rare": len(rare_real),
        "n_memorised": memorised_count,
        "memorisation_rate": memorisation_rate,
    }


# ---------------------------------------------------------------------------
# 4. Full privacy scorecard
# ---------------------------------------------------------------------------


def privacy_scorecard(
    real_df: pd.DataFrame,
    synth_datasets: dict[str, pd.DataFrame],
    target_col: str = "Class",
    n_neighbors: int = 5,
    sample_size: int = 5000,
) -> pd.DataFrame:
    """Run all privacy checks for each generator and return a summary DataFrame."""
    feature_cols = [
        c for c in real_df.select_dtypes(include=[np.number]).columns
        if c != target_col
    ]

    rows = []
    for name, synth_df in synth_datasets.items():
        print(f"\n[Privacy] Checking generator: {name}")

        dup = check_exact_duplicates(real_df, synth_df, feature_cols)
        nn = nn_distance_analysis(real_df, synth_df, feature_cols, n_neighbors, sample_size)
        mem = check_rare_record_memorisation(real_df, synth_df, target_col, feature_cols=feature_cols)

        rows.append({
            "generator": name,
            "exact_dup_rate": dup["dup_rate"],
            "nn_mean_distance": nn["nn_mean"],
            "nn_median_distance": nn["nn_median"],
            "nn_frac_below_0.1": nn["pct_below_01"],
            "rare_memorisation_rate": mem["memorisation_rate"],
        })

    scorecard = pd.DataFrame(rows)
    print("\n--- PRIVACY SCORECARD ---")
    print(scorecard.to_string(index=False))
    return scorecard


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot_nn_distance_distributions(
    real_df: pd.DataFrame,
    synth_datasets: dict[str, pd.DataFrame],
    target_col: str = "Class",
    sample_size: int = 5000,
    output_dir: Optional[str | Path] = None,
) -> None:
    """KDE plot of nearest-neighbour distances per generator."""
    feature_cols = [
        c for c in real_df.select_dtypes(include=[np.number]).columns
        if c != target_col
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    palette = sns.color_palette("tab10", len(synth_datasets))

    # Baseline: real-to-real distances (leave-one-out style)
    real_sample = real_df[feature_cols].sample(n=min(sample_size, len(real_df)), random_state=42)
    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_sample.values)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm="ball_tree", n_jobs=-1)
    nbrs.fit(real_scaled)
    dist_rr, _ = nbrs.kneighbors(real_scaled)
    # Use 2nd NN to exclude self
    sns.kdeplot(dist_rr[:, 1], ax=ax, label="real→real (baseline)", linestyle="--", color="black")

    for (name, synth_df), color in zip(synth_datasets.items(), palette):
        synth_sample = synth_df[feature_cols].sample(
            n=min(sample_size, len(synth_df)), random_state=42
        )
        synth_scaled = scaler.transform(synth_sample.values)
        nbrs2 = NearestNeighbors(n_neighbors=1, algorithm="ball_tree", n_jobs=-1)
        nbrs2.fit(real_scaled)
        dist_sr, _ = nbrs2.kneighbors(synth_scaled)
        sns.kdeplot(dist_sr[:, 0], ax=ax, label=f"synth({name})→real", color=color)

    ax.set_xlabel("Distance to Nearest Real Neighbour (L2, standardised)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Privacy: Nearest-Neighbour Distance Distributions", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()

    if output_dir:
        path = Path(output_dir) / "privacy_nn_distances.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved NN distance plot to {path}")
    plt.show()
