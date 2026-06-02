# Method setup — 6M EBM vs 8 baselines (locked)

All 9 methods share the SAME inputs:

- Same `X` = 16 live features from `build_feats_at_L(rows@dev6M)`
- Same `y` = `Y_24M_NHRH` (24-month no-hyper, no-hypo, no-relapse)
- Same `is_dev` / `episode_id`
- Same 5-fold `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=13)`
- Same temporal cohort (201 episodes, 6M landmark)
- Same paired episode-cluster bootstrap × 1000 (seed=7)

| # | Method | Hyperparameters |
|--:|:--|:--|
| 0 | **EBM** (anchor, locked) | `ExplainableBoostingClassifier(interactions=5, max_interaction_bins=16, random_state=2025)` via `ebm_oof_and_temporal(L=6)` |
| 1 | L2-Logistic | `StandardScaler` + `LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=2000, random_state=2025)` |
| 2 | Elastic-Net | `StandardScaler` + `LogisticRegression(C=1.0, penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=5000, random_state=2025)` |
| 3 | k-NN(15) | `StandardScaler` + `KNeighborsClassifier(n_neighbors=15, weights="distance")` |
| 4 | Gaussian NB | `StandardScaler` + `GaussianNB(var_smoothing=1e-9)` |
| 5 | SVM-RBF | `StandardScaler` + `SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=2025)` |
| 6 | RandomForest | `RandomForestClassifier(n_estimators=500, max_depth=None, class_weight="balanced_subsample", random_state=2025)` |
| 7 | XGBoost | `XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=2025)` |
| 8 | HistGB | `HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_depth=None, l2_regularization=1.0, random_state=2025)` |

EBM anchor protocol guarantees match to the atlas mainline number
(`m2v2_ebm_full_median/ebm_median_metrics.json` 6M temporal AUC reference);
script aborts on > 0.001 drift.
