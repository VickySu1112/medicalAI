# M2-A · Pre-registered 10-method horizontal benchmark

**Shared infrastructure**: data stacking, 5 mechanism blocks, StratifiedGroupKFold, episode-cluster bootstrap, per-landmark Platt — identical to M2-Base (see `../module2_v2_base/arch.md`).

## 10 methods (locked roster)

| # | Method | Category | Library | Hyperparameter grid (pre-registered) |
|:---:|:---|:---:|:---:|:---|
| 1 | L2-logistic supermodel | linear | sklearn | C ∈ {0.001, 0.01, 0.1, 1, 10} |
| 2 | Elastic-net logistic | linear regularized | sklearn | l1_ratio ∈ {0.1, 0.5, 0.9}, C ∈ {0.01, 0.1, 1} |
| 3 | GEE logistic (exchangeable) | clustered linear | statsmodels | (no tuning; family=Binomial, cov_struct=Exchangeable, groups=episode_id) |
| 4 | Random Forest | static tree ensemble | sklearn | n_estimators ∈ {200, 500}, max_depth ∈ {None, 8, 16}, min_samples_leaf ∈ {1, 5} |
| 5 | LightGBM | gradient boosting | lightgbm | num_leaves ∈ {15, 31}, learning_rate ∈ {0.05, 0.1}, n_estimators ∈ {200, 500} |
| 6 | XGBoost | gradient boosting | xgboost | max_depth ∈ {3, 6}, learning_rate ∈ {0.05, 0.1}, n_estimators ∈ {200, 500} |
| 7 | CatBoost | gradient boosting | catboost | depth ∈ {4, 6}, learning_rate ∈ {0.05, 0.1}, iterations ∈ {200, 500} |
| 8 | TabNet | attention tabular DL | pytorch-tabnet | n_d ∈ {8, 16}, n_steps ∈ {3, 5} |
| 9 | LSTM | sequence DL | torch | hidden ∈ {32, 64}, dropout ∈ {0.0, 0.2}; 4-step input per episode |
| 10 | Cox PH (landmark) | survival | lifelines | penalizer ∈ {0.0, 0.01, 0.1}; baseline_hazard='breslow' |

## Reporting

- **Leaderboard table**: methods ranked by temporal pooled ROC-AUC + cluster bootstrap CI; Δ vs M2-Base.
- **4×10 heatmap** of per-landmark ROC-AUC.
- **Calibration table**: intercept / slope per method per landmark.
- **Computational cost table**: training wall-clock, inference time, parameter count.
- **Reviewer-friendliness scorecard**: +/− on (interpretable, cluster-aware, calibration-honest, reproducible).

## Comparison fairness rules

- All 10 methods consume the same `module2_v2_shared.load_stacked()` output (rows + columns)
- All 10 use the same `StratifiedGroupKFold(5)` seed + episode-id groups
- All 10 CIs from same episode-cluster bootstrap × 1000
- All 10 calibrated with same per-landmark Platt on pooled outer-OOF (raw predictions → Platt → metrics)
- Dependency conflicts handled via subprocess isolation (LightGBM / XGBoost / CatBoost / TabNet) per existing M1·v1 boosting pattern

## Pre-registered exclusion rules

- If a method fails to fit (numerical / dependency issues), report N/A with reason; do not silently exclude.
- If a method's calibration slope < 0.3 or > 3.0, flag as "miscalibrated"; still include in leaderboard but exclude from recommendation candidates.

This file locks the M2-A scope. Any methods added or removed post-evaluation must be documented in §10 of the M2-A report.
