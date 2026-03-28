"""
Downstream ML utility evaluation.

Trains fraud classifiers (XGBoost, Random Forest, Logistic Regression) on:
  1. Real training data only
  2. Synthetic data only (from each generator)
  3. Mixed real + synthetic data (augmentation)

All evaluated on the same held-out real test set.

Metrics: AUROC, AUPRC, F1, Precision, Recall.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Downstream model zoo
# ---------------------------------------------------------------------------


def get_classifiers(random_state: int = 42) -> dict:
    return {
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=100,  # handles class imbalance
            eval_metric="aucpr",
            random_state=random_state,
            verbosity=0,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        ),
    }


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, y_proba)),
        "auprc": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------


def run_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    clf,
) -> dict:
    """Train a single classifier and evaluate on the test set."""
    # Logistic regression needs scaled features
    if model_name == "logistic_regression":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_proba)
    return metrics


# ---------------------------------------------------------------------------
# Full utility benchmark
# ---------------------------------------------------------------------------


def run_utility_benchmark(
    train_real: pd.DataFrame,
    test_real: pd.DataFrame,
    synth_datasets: dict[str, pd.DataFrame],
    target_col: str = "Class",
    aug_ratios: Optional[list[float]] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compare classifiers trained on:
      - real only
      - each synthetic dataset
      - real + synthetic (mixed)

    Returns a results DataFrame.
    """
    aug_ratios = aug_ratios or [1.0, 2.0]

    feature_cols = [c for c in train_real.select_dtypes(include=[np.number]).columns if c != target_col]

    X_test = test_real[feature_cols].values
    y_test = test_real[target_col].values

    classifiers = get_classifiers(random_state)
    results = []

    def _run(training_df: pd.DataFrame, training_label: str) -> None:
        X_tr = training_df[feature_cols].values
        y_tr = training_df[target_col].values
        for model_name, clf in classifiers.items():
            # Re-instantiate to avoid warm-start issues
            clf_fresh = get_classifiers(random_state)[model_name]
            metrics = run_experiment(X_tr, y_tr, X_test, y_test, model_name, clf_fresh)
            results.append({
                "training_data": training_label,
                "model": model_name,
                **metrics,
            })
            print(
                f"  [{training_label}] {model_name:22s}  "
                f"AUROC={metrics['auroc']:.4f}  AUPRC={metrics['auprc']:.4f}  "
                f"F1={metrics['f1']:.4f}  Recall={metrics['recall']:.4f}"
            )

    # 1. Real only
    print("\n=== Training on: REAL data ===")
    _run(train_real, "real_only")

    # 2. Synthetic only
    for gen_name, synth_df in synth_datasets.items():
        print(f"\n=== Training on: SYNTHETIC ({gen_name}) ===")
        _run(synth_df, f"synth_{gen_name}")

    # 3. Mixed real + synthetic
    for gen_name, synth_df in synth_datasets.items():
        for ratio in aug_ratios:
            n_synth = int(len(train_real) * ratio)
            synth_sample = synth_df.sample(
                n=min(n_synth, len(synth_df)),
                random_state=random_state,
                replace=n_synth > len(synth_df),
            )
            mixed_df = pd.concat([train_real, synth_sample], ignore_index=True)
            label = f"mixed_{gen_name}_x{ratio}"
            print(f"\n=== Training on: MIXED real + {gen_name} (aug x{ratio}) ===")
            print(f"    Train size: {len(mixed_df):,}  (real={len(train_real):,}  synth={len(synth_sample):,})")
            _run(mixed_df, label)

    df_results = pd.DataFrame(results)
    return df_results


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot_utility_comparison(
    results_df: pd.DataFrame,
    metric: str = "auroc",
    output_dir: Optional[str | Path] = None,
) -> None:
    """Bar chart comparing training strategies per model."""
    fig, ax = plt.subplots(figsize=(14, 6))

    pivot = results_df.pivot(index="training_data", columns="model", values=metric)
    pivot.plot(kind="bar", ax=ax, colormap="tab10", edgecolor="black", linewidth=0.5)

    ax.set_title(f"Downstream Utility: {metric.upper()} by Training Strategy", fontsize=13)
    ax.set_xlabel("Training Data", fontsize=11)
    ax.set_ylabel(metric.upper(), fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(title="Model", fontsize=9)
    ax.tick_params(axis="x", labelrotation=35)

    plt.tight_layout()

    if output_dir:
        path = Path(output_dir) / f"utility_{metric}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved utility plot to {path}")
    plt.show()


def plot_metric_heatmap(
    results_df: pd.DataFrame,
    output_dir: Optional[str | Path] = None,
) -> None:
    """Heatmap of all metrics for XGBoost across training strategies."""
    metric_cols = ["auroc", "auprc", "f1", "precision", "recall"]
    xgb_df = results_df[results_df["model"] == "xgboost"].copy()
    xgb_df = xgb_df.set_index("training_data")[metric_cols]

    fig, ax = plt.subplots(figsize=(8, max(4, len(xgb_df) * 0.5 + 2)))
    sns.heatmap(
        xgb_df.astype(float),
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Score"},
    )
    ax.set_title("XGBoost Metrics by Training Strategy", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()

    if output_dir:
        path = Path(output_dir) / "utility_heatmap.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved heatmap to {path}")
    plt.show()
