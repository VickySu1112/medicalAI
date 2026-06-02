# Method setup — 6M EBM vs LR baseline + 8 sklearn classical (v2)

All 10 methods share the SAME inputs:

- Same `X` = 16 live features from `build_feats_at_L(rows@dev6M)`
- Same `y` = `Y_24M_NHRH`
- Same `is_dev` / `episode_id`
- Same 5-fold `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=13)`
- Same temporal cohort (201 episodes, 6M landmark)
- Same paired episode-cluster bootstrap × 1000 (seed=7)

| # | Method | Hyperparameters |
|--:|:--|:--|
| 0 | **EBM** (anchor, locked) | `ExplainableBoostingClassifier(interactions=5, max_interaction_bins=16, random_state=2025)` via `ebm_oof_and_temporal(L=6)` |
| 1 | L2-Logistic (baseline) | `StandardScaler` + `LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=2000, random_state=2025)` |
| 2 | RandomForest | `RandomForestClassifier(n_estimators=500, max_depth=None, class_weight="balanced_subsample", random_state=2025)` |
| 3 | ExtraTrees | `ExtraTreesClassifier(n_estimators=500, max_depth=None, class_weight="balanced_subsample", random_state=2025)` |
| 4 | GradientBoosting | `GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=2025)` |
| 5 | HistGB | `HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=2025)` |
| 6 | AdaBoost | `AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=2025)` (SAMME, tree stump) |
| 7 | DecisionTree | `DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, class_weight="balanced", random_state=2025)` |
| 8 | SVM-RBF | `StandardScaler` + `SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=2025)` |
| 9 | LDA | `StandardScaler` + `LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")` |

EBM anchor protocol guarantees match to the atlas mainline number
(`m2v2_ebm_full_median/ebm_median_metrics.json` 6M temporal AUC reference);
script aborts on > 0.003 drift.
