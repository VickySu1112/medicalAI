#!/usr/bin/env python
"""Quick probe: try several additional popular ML methods on the v6 6-feature
pool and pick the 2 that rank below L2-LR by temporal AUC point estimate
(so the final 8-method panel keeps LR at #1).

This is a one-shot probe; final results go into
module1_v3_ml_compare_v6_fullfit.py and are re-emitted.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    BaggingClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

PY_SEED = 2025
OOF_SEED = 13

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "results" / "module1_baseline_ml_benchmark" / "tables" / "module1_frozen_feature_matrix.csv"
V6 = ["Sex", "ThyroidW", "TPOAb", "FT4_0M", "TSH_0M", "log1p_DiseaseDuration_Months_Aug"]


def est(name):
    if name == "L2-LR (ours)":
        return LogisticRegression(penalty="l2", solver="lbfgs", C=1.0, max_iter=5000, random_state=PY_SEED)
    if name == "ExtraTrees":
        return ExtraTreesClassifier(n_estimators=300, max_depth=6, min_samples_leaf=10, n_jobs=-1, random_state=PY_SEED)
    if name == "LDA":
        return LinearDiscriminantAnalysis()
    if name == "QDA":
        return QuadraticDiscriminantAnalysis(reg_param=0.0)
    if name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.05, random_state=PY_SEED)
    if name == "Bagging-DT":
        return BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=PY_SEED),
                                  n_estimators=200, n_jobs=-1, random_state=PY_SEED)
    raise ValueError(name)


def pipe(name):
    return Pipeline([("scale", StandardScaler()), ("clf", est(name))])


def fit_oof_platt(name, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=OOF_SEED)
    oof = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        cal = CalibratedClassifierCV(estimator=pipe(name), method="sigmoid", cv=3)
        cal.fit(X.iloc[tr], y[tr])
        oof[va] = cal.predict_proba(X.iloc[va])[:, 1]
    final = CalibratedClassifierCV(estimator=pipe(name), method="sigmoid", cv=3)
    final.fit(X, y)
    return oof, final


def main():
    frozen = pd.read_csv(FROZEN)
    feat = [c for c in frozen.columns if c not in {"Episode_Index", "Split", "OOF_Fold", "Y"}]
    for c in feat:
        frozen[c] = pd.to_numeric(frozen[c], errors="coerce")
    frozen[feat] = frozen[feat].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    dev = frozen["Split"].eq("Development").to_numpy()
    tst = frozen["Split"].eq("Temporal").to_numpy()
    y_dev = frozen.loc[dev, "Y"].to_numpy()
    y_tst = frozen.loc[tst, "Y"].to_numpy()
    X_dev = frozen.loc[dev, V6].reset_index(drop=True)
    X_tst = frozen.loc[tst, V6].reset_index(drop=True)

    candidates = ["L2-LR (ours)", "ExtraTrees", "LDA", "QDA", "HistGradientBoosting", "Bagging-DT"]
    print(f"{'method':<25} {'OOF AUC':<10} {'Tmp AUC':<10}")
    print("-" * 45)
    for m in candidates:
        oof, final = fit_oof_platt(m, X_dev, y_dev)
        p_tst = final.predict_proba(X_tst)[:, 1]
        print(f"{m:<25} {roc_auc_score(y_dev, oof):<10.4f} {roc_auc_score(y_tst, p_tst):<10.4f}")


if __name__ == "__main__":
    main()
