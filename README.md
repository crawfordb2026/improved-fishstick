# Synthara — Synthetic EHR Generation & Benchmark

A hackathon project that generates synthetic patient records from real EHR data and evaluates them across three dimensions: **realism**, **downstream ML utility**, and **privacy leakage**.

## The problem

Real patient data is locked behind HIPAA. Synthetic data generation offers a privacy-safe stand-in — but only if the synthetic records are actually statistically faithful and useful for training ML models.

## What this builds

An end-to-end benchmark pipeline that:

1. Flattens relational Synthea EHR tables into a single patient-level feature matrix
2. Trains four generative models in parallel on AWS SageMaker
3. Evaluates every generator on realism, ML utility, and privacy

```
Synthea CSVs ──► Flatten ──► Preprocess ──► Train generators (SageMaker)
                                                      │
                              ┌───────────────────────┼───────────────────────┐
                              │           │            │                       │
                           Copula      CTGAN         TVAE                Tabular Diffusion
                              │           │            │                       │
                              └───────────┴────────────┴───────────────────────┘
                                                      │
                               ┌──────────────────────┼──────────────────────┐
                               │                      │                      │
                          Realism eval           Utility eval          Privacy checks
                          (KS, Wasserstein,      (AUROC, AUPRC,        (dup rate,
                           corr distance,         F1 on real            NN distance,
                           discriminator)         test set)             memorisation)
                               │                      │                      │
                               └──────────────────────┴──────────────────────┘
                                                      │
                                           Reports + CSV scorecards
```

---

## Dataset

**Synthea COVID-19 Synthetic EHR** ([synthea.mitre.org](https://synthea.mitre.org/downloads))

- 124,150 patients with COVID-19 module
- Flattened from 8+ relational tables into 35 features per patient
- Target: `DECEASED` (0 = survived, 1 = died) — 19.5% positive rate

Download the 100k COVID-19 dataset from [synthea.mitre.org/downloads](https://synthea.mitre.org/downloads) and place the CSVs in `data/raw/100k_synthea_covid19_csv/`.

### Feature groups (35 total)

| Group | Count | Examples |
|---|---|---|
| Demographics | 5 | age, gender, race, hispanic, marital_status |
| Conditions | 9 | diabetes, hypertension, obesity, pneumonia, anemia |
| Encounters | 4 | inpatient, emergency, outpatient, total |
| Vitals | 10 | BMI, systolic/diastolic BP, O₂ saturation, heart rate |
| Financials | 7 | healthcare_expenses, healthcare_coverage |

---

## Generators

| Model | Type | Library | Epochs | Instance |
|---|---|---|---|---|
| **Gaussian Copula** | Statistical | SDV | — | ml.m5.4xlarge |
| **CTGAN** | GAN | SDV | 300 | ml.m5.4xlarge |
| **TVAE** | VAE | SDV | 300 | ml.m5.4xlarge |
| **Tabular Diffusion** | Diffusion | Custom (PyTorch) | 100 | ml.m5.4xlarge |

Tabular Diffusion is a custom implementation of Kotelnikov et al. (2023) — a denoising diffusion probabilistic model adapted for tabular data. The denoiser is a 3.34M parameter MLP with residual blocks and sinusoidal timestep embeddings.

---

## Results

### Realism (lower is better, discriminator → 0.5 is better)

| Generator | KS Mean | KS Max | Wasserstein Mean | Corr Distance | Discriminator AUROC |
|---|---|---|---|---|---|
| Copula | 0.111 | 0.678 | 0.126 | 3.237 | 1.000 |
| CTGAN | 0.100 | 0.678 | 0.075 | 1.652 | 1.000 |
| TVAE | 0.108 | 0.678 | 0.094 | 3.245 | 1.000 |
| **Tabular Diffusion** | **0.050** | **0.350** | **0.034** | **0.503** | 1.000 |

### Utility — train on synthetic, test on real (higher is better)

| Training Data | Best AUROC | Best AUPRC | Best F1 |
|---|---|---|---|
| Real only | 0.9926 | 0.9746 | 0.9155 |
| Copula | 0.9603 | 0.8956 | 0.7295 |
| CTGAN | 0.9680 | 0.8938 | 0.7903 |
| TVAE | 0.9769 | 0.9246 | 0.8097 |
| **Tabular Diffusion** | **0.9906** | **0.9679** | **0.8949** |
| Real + Tabular Diffusion | 0.9926 | 0.9744 | 0.9139 |

Tabular Diffusion closes to within **0.002 AUROC** of real data.

### Privacy — all generators pass

- 0 exact duplicates across all four generators
- 0 memorized rare records (deceased patients reproduced within distance 0.05)
- Nearest-neighbor distances well above the 0.1 danger threshold

---

## Project structure

```
synthara/
├── configs/
│   └── config.yaml                    # Hyperparameters and paths
├── data/
│   ├── raw/                           # Synthea CSV files go here
│   ├── processed/                     # train/val/test splits (auto-generated)
│   └── synthetic/                     # Generator outputs (auto-generated)
├── sagemaker/
│   ├── train.py                       # SageMaker training entry point
│   ├── launch_jobs.py                 # Submit + poll all 4 parallel jobs
│   ├── tab_ddpm.py                    # Tabular Diffusion model classes
│   └── requirements.txt              # SageMaker container dependencies
├── src/
│   ├── preprocessing/
│   │   ├── flatten_ehr.py             # Join Synthea tables → patient_features.csv
│   │   └── preprocess.py             # Split, log-transform, standardise
│   ├── generators/
│   │   └── train_generators.py        # Local Copula/CTGAN/TVAE training
│   ├── evaluation/
│   │   ├── realism.py                 # KS, Wasserstein, discriminator, PCA plots
│   │   └── utility.py                 # Downstream ML benchmark
│   ├── privacy/
│   │   └── privacy_checks.py          # Dup rate, NN distance, memorisation
│   └── pipeline.py                    # End-to-end CLI runner
├── diffusion.py                       # Standalone local Tabular Diffusion training script
├── outputs/
│   └── models/                        # Saved model files (.pkl / .pt)
└── reports/                           # All plots and CSV scorecards
```

---

## Quickstart

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Flatten the EHR data

```bash
python src/preprocessing/flatten_ehr.py
```

Reads from `data/raw/100k_synthea_covid19_csv/`, writes `data/raw/patient_features.csv`.

### 3. Run the full pipeline

```bash
python -m src.pipeline --config configs/config.yaml
```

This preprocesses, loads pre-built synthetic CSVs (if present), and runs all evaluation phases.

### 4. Train generators

**Option A — AWS SageMaker (all 4 in parallel):**
```bash
python sagemaker/launch_jobs.py
```

**Option B — Local SDV models only:**
```bash
python -m src.pipeline --config configs/config.yaml  # trains locally if no synthetic CSVs found
```

**Option C — Local Tabular Diffusion:**
```bash
python diffusion.py --epochs 100 --steps 200 --batch-size 1024
```

### 5. Find results in `reports/`

| File | Contents |
|---|---|
| `realism_scorecard.csv` | KS, Wasserstein, correlation distance, discriminator AUROC |
| `utility_results.csv` | AUROC, AUPRC, F1 per training strategy × classifier |
| `privacy_scorecard.csv` | Dup rate, NN distance, memorisation rate |
| `distribution_comparison.png` | KDE overlays: real vs each generator |
| `correlation_heatmaps.png` | Side-by-side correlation matrices |
| `pca_overlap.png` | PCA scatter: real vs synthetic |
| `utility_auroc.png` / `utility_auprc.png` | Bar charts by training strategy |
| `utility_heatmap.png` | Full metric heatmap across all conditions |
| `privacy_nn_distances.png` | NN distance distributions |
| `pipeline_summary.json` | All results in one JSON |

---

## AWS SageMaker setup

All four training jobs run in parallel. Each pulls `patient_features.csv` from S3 and pushes `{generator}_synthetic.csv` + the model artifact back to S3.

```
S3: synthetic-ml-{account}/synthea/data/patient_features.csv
         │
         ├──► synth-copula    (ml.m5.4xlarge)
         ├──► synth-ctgan     (ml.m5.4xlarge)
         ├──► synth-tvae      (ml.m5.4xlarge)
         └──► synth-diffusion (ml.m5.4xlarge)
                   │
                   ▼
S3: synthetic-ml-{account}/synthea/output/{job}/output/model.tar.gz
```

Configure `ACCOUNT`, `REGION`, `ROLE_ARN` in `sagemaker/launch_jobs.py` before running.

---

## Key engineering notes

- `flatten_ehr.py` uses `regex=False` on all `str.contains()` calls — condition keywords like `"Body mass index 30+"` contain regex metacharacters that silently zero out columns with the default `regex=True`
- All binary (0/1) columns are marked `sdtype='categorical'` in SDV metadata to prevent TVAE dimensional collapse
- Six zero-inflated count/monetary columns (`encounter_inpatient`, `encounter_emergency`, `encounter_total`, `condition_count`, `healthcare_expenses`, `healthcare_coverage`) are log1p-transformed before StandardScaler
- Discriminator AUROC = 1.0 across all generators is a known ceiling for zero-inflated tabular EHR data, not a bug — the encounter columns dominate the discriminator signal regardless of generator quality

---

## Team

- **Crawford Barnett**
- **Matthew Weber**
- **Tyler Hintz**
