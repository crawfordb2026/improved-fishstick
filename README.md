# Synthetic Data Generator — Privacy-Safe ML Benchmark

A 24-hour hackathon prototype that generates synthetic tabular financial data and evaluates it across three dimensions: **realism**, **downstream ML utility**, and **privacy leakage**.

## The problem

Real financial datasets (fraud records, transaction histories) cannot be freely shared due to privacy regulations. Synthetic data generation offers a way to share privacy-conscious stand-ins — but only if the synthetic data is actually useful and not just a noisy copy.

## What this builds

A full benchmark pipeline that:

1. Generates synthetic data from multiple models
2. Proves whether each model's output is realistic
3. Proves whether it's useful for downstream fraud detection
4. Proves it doesn't leak real training records

```
Raw CSV  ──►  Profile  ──►  Preprocess  ──►  Train generators
                                                     │
                            ┌────────────────────────┤
                            │            │            │
                        Copula        CTGAN         TVAE
                            │            │            │
                            └────────────┴────────────┘
                                         │
                          ┌──────────────┼──────────────┐
                          │              │              │
                     Realism eval   Utility eval   Privacy checks
                     (KS, Wass,     (AUROC, F1     (dup rate,
                      corr dist,     on real        NN distance,
                      discriminator) test set)      memorisation)
                          │              │              │
                          └──────────────┴──────────────┘
                                         │
                                  Reports + CSV scorecards
```

---

## Dataset

**ULB Credit Card Fraud Detection** (Kaggle)

- 284,807 transactions over two days
- 492 fraud cases (0.172% positive rate)
- 28 PCA-anonymised features (V1–V28) + Amount + Time
- Target: `Class` (0 = legit, 1 = fraud)

Download:
```bash
# Kaggle CLI
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/raw/
```

Or download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in `data/raw/`.

---

## Generators

| Model | Type | Library | Notes |
|---|---|---|---|
| **Gaussian Copula** | Statistical | SDV | Fast classical baseline |
| **CTGAN** | GAN | SDV / ctgan | Strong tabular GAN |
| **TVAE** | VAE | SDV | Latent generative model |

---

## Evaluation metrics

### Realism
| Metric | Ideal | Interpretation |
|---|---|---|
| KS statistic (mean) | → 0 | Per-column distribution similarity |
| Wasserstein distance (mean) | → 0 | Earth-mover cost between distributions |
| Correlation matrix distance | → 0 | Feature correlation structure preserved |
| Discriminator AUROC | → 0.5 | Classifier cannot tell real from synthetic |

### Downstream ML utility
All models evaluated on the same held-out **real** test set.

Training strategies compared:
- Real data only (gold standard)
- Synthetic data only (each generator)
- Mixed real + synthetic (augmentation ratios 1×, 2×, 5×)

Downstream models: XGBoost · Random Forest · Logistic Regression

Metrics: AUROC · AUPRC · F1 · Precision · Recall

### Privacy leakage
| Check | What it catches |
|---|---|
| Exact duplicate rate | Verbatim copy of training rows |
| NN distance (synth → real) | How close synthetic rows are to training data |
| Rare record memorisation | Whether minority-class (fraud) records are reproduced |

---

## Project structure

```
improved-fishstick/
├── configs/
│   └── config.yaml               # All hyperparameters and paths
├── data/
│   ├── raw/                      # creditcard.csv goes here
│   ├── processed/                # train/val/test splits (auto-generated)
│   └── synthetic/                # generator outputs (auto-generated)
├── notebooks/
│   ├── 01_full_pipeline.ipynb    # Main demo notebook
│   └── 02_sagemaker_training.ipynb  # AWS SageMaker parallel training
├── src/
│   ├── preprocessing/
│   │   └── preprocess.py         # Load, profile, split, standardise
│   ├── generators/
│   │   └── train_generators.py   # Copula, CTGAN, TVAE training + generation
│   ├── evaluation/
│   │   ├── realism.py            # KS, Wasserstein, discriminator, PCA plots
│   │   └── utility.py            # Downstream fraud classifier benchmark
│   ├── privacy/
│   │   └── privacy_checks.py     # Dup rate, NN distance, memorisation
│   ├── utils/
│   │   └── s3_utils.py           # S3 upload/download helpers
│   └── pipeline.py               # End-to-end CLI runner
├── outputs/
│   └── models/                   # Saved generator .pkl files
├── reports/                      # All plots and CSV scorecards
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download dataset

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/raw/
```

### 3. Run the full pipeline

**Option A — Jupyter notebook** (recommended for demo):
```bash
jupyter notebook notebooks/01_full_pipeline.ipynb
```

**Option B — CLI**:
```bash
python -m src.pipeline --config configs/config.yaml
```

### 4. Find results in `reports/`

| File | Contents |
|---|---|
| `realism_scorecard.csv` | KS, Wasserstein, correlation, discriminator AUROC per generator |
| `utility_results.csv` | AUROC, AUPRC, F1, Precision, Recall per training strategy × model |
| `privacy_scorecard.csv` | Dup rate, NN distance, memorisation rate per generator |
| `distribution_comparison.png` | KDE overlays for real vs each synthetic generator |
| `correlation_heatmaps.png` | Side-by-side correlation matrices |
| `pca_overlap.png` | PCA scatter: real vs synthetic |
| `utility_auroc.png` | Bar chart: AUROC by training strategy |
| `privacy_nn_distances.png` | NN distance distributions |
| `pipeline_summary.json` | All results in one JSON |

---

## AWS architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        AWS Cloud                            │
│                                                             │
│  S3 Buckets                                                 │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ synthetic-ml-raw │  │synthetic-ml-      │                │
│  │  creditcard.csv  │  │curated/           │                │
│  └──────────────────┘  │  train.csv        │                │
│                         │  val.csv          │                │
│                         │  test.csv         │                │
│                         └──────────────────┘                │
│                                                             │
│  SageMaker                                                  │
│  ┌──────────────────────────────────────┐                  │
│  │   Training Jobs (parallel)           │                  │
│  │   ┌──────────┐ ┌───────┐ ┌────────┐ │                  │
│  │   │  Copula  │ │ CTGAN │ │  TVAE  │ │                  │
│  │   │ job      │ │ job   │ │ job    │ │                  │
│  │   └──────────┘ └───────┘ └────────┘ │                  │
│  └──────────────────────────────────────┘                  │
│                         │                                   │
│                         ▼                                   │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │synthetic-ml-      │  │synthetic-ml-     │                │
│  │synthetic/         │  │reports/          │                │
│  │  copula_synth.csv │  │  scorecards      │                │
│  │  ctgan_synth.csv  │  │  plots           │                │
│  │  tvae_synth.csv   │  │  summary.json    │                │
│  └──────────────────┘  └──────────────────┘                │
│                                                             │
│  CloudWatch — training logs and metrics                     │
└─────────────────────────────────────────────────────────────┘
```

For SageMaker-based parallel training, see `notebooks/02_sagemaker_training.ipynb`.

---

## Tune for speed (hackathon mode)

Reduce training time by lowering epochs in `configs/config.yaml`:

```yaml
generators:
  ctgan:
    epochs: 50    # default 300
  tvae:
    epochs: 50    # default 300
```

Copula trains in seconds. With 50 epochs CTGAN/TVAE finish in ~3-5 minutes on CPU.

---

## Pitch summary

> We built a synthetic tabular data benchmark pipeline on AWS that generates privacy-conscious synthetic financial data, compares multiple generative models, and evaluates each one on realism, downstream fraud detection utility, and privacy leakage risk. Instead of just generating fake rows, we built a full scorecard that tells you whether synthetic data is actually useful and safe enough to matter.

---

## FAQ

**Q: Why three generators instead of just one?**
A: Tabular generation is unstable. No single model dominates. CTGAN often wins on realism, Copula on speed, TVAE on utility. The scorecard decides.

**Q: How do you prove privacy safety?**
A: We don't claim it blindly. We run three empirical checks — exact duplication, nearest-neighbour distance, and minority-record memorisation — and report the results.

**Q: Why fraud detection for the utility test?**
A: It's the hardest downstream task on this dataset. AUPRC on a 0.17% positive rate stress-tests whether the synthetic data preserved the rare but important patterns.

**Q: Why AWS?**
A: Managed training at scale, clean S3 data lineage, easy experiment tracking, and a clear path to production synthetic data refresh pipelines.
