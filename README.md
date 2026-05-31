# Traffic Demand Prediction

**Competition-grade Traffic Demand Prediction** — targeting R² ≥ 0.995

## Quick Start

```bash
pip install -r requirements.txt
```

1. Place datasets in `data/`:
   ```
   data/train.csv
   data/test.csv
   ```
2. Open and run `notebooks/TrafficDemandElite995.ipynb` top-to-bottom.

## Project Structure

```
TrafficDemandElite995/
├── data/
│   ├── train.csv          # Training data (add manually)
│   └── test.csv           # Test data (add manually)
├── notebooks/
│   └── TrafficDemandElite995.ipynb
├── src/
│   ├── utils.py           # Config, validation, EDA, cleaning
│   ├── feature_engineering.py
│   ├── encoding.py        # Frequency, count, target encoding
│   ├── aggregation.py     # Group statistics & memorization
│   ├── leakage_detection.py
│   ├── training.py        # CatBoost, LightGBM, Optuna HPO
│   └── ensemble.py        # Weight optimization & submission
├── outputs/
│   ├── models/
│   ├── predictions/
│   └── submissions/
├── requirements.txt
├── README.md
└── .gitignore
```

## Pipeline

| # | Section | Description |
|---|---------|-------------|
| 1 | Configuration | Project paths, CV settings, hardware |
| 2 | Validation | Check datasets exist and are valid |
| 3 | Loading | Read train.csv and test.csv |
| 4 | Schema Detection | Auto-detect target, categorical, numerical, datetime, geohash columns |
| 5 | EDA | Distribution, missing values, duplicates, visualizations |
| 6 | Leakage Detection | Train-test overlap, deterministic patterns, memorization |
| 7 | Cleaning | Duplicates, missing values, normalization |
| 8 | Feature Engineering | Timestamp, cyclic, geohash, interaction features |
| 9 | Aggregation | Group-based mean/median/std/count features |
| 10 | Encoding | Frequency, count, K-Fold target encoding |
| 11 | CatBoost | 5-Fold CV training (GPU) |
| 12 | LightGBM | 5-Fold CV training (GPU) |
| 13 | Optuna | 50 trials per model hyperparameter optimization |
| 14 | OOF Predictions | Out-of-fold predictions for ensemble |
| 15 | Ensemble | Automatic weight optimization (CatBoost + LightGBM) |
| 16 | Final Training | Retrain with best hyperparameters |
| 17 | Feature Importance | Combined CatBoost + LightGBM feature ranking |
| 18 | Submission | Generate submission.csv |

## Hardware

Optimized for:
- **CPU**: Intel Core 7 240H
- **GPU**: NVIDIA RTX 5050
- **RAM**: 24 GB

## Expected Runtime

| Stage | Time |
|-------|------|
| Feature Engineering | 1-3 min |
| CatBoost Optuna | 10-20 min |
| LightGBM Optuna | 5-15 min |
| Ensemble Optimization | 1-3 min |
| Final Training | 2-5 min |
| **Total** | **20-45 min** |
