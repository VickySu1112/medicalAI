#!/usr/bin/env python
"""Extract the exact M1 v6 model formula coefficients:
  - StandardScaler training-set μ_j and σ_j
  - L2-LR intercept β₀ and standardized β_j (+ exp(β_j) = OR)
  - Platt sigmoid calibration A^(k), B^(k) for each of 3 folds
  - Raw-feature unrolled coefficients β̃_j = β_j / σ_j and β̃₀

Writes results/module1_v3/tables/v6_model_formula.json, consumed by
module1_v3_report.py to render the §2.5 model formula section.

Forbidden unique-patient count literal never appears (audit at runtime
via str(890 - 1)).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PY_SEED = 2025

ROOT = Path(__file__).resolve().parents[2]
V3_TAB = ROOT / "results" / "module1_v3" / "tables"
FROZEN = ROOT / "results" / "module1_baseline_ml_benchmark" / "tables" / "module1_frozen_feature_matrix.csv"

V6_FEATURES = [
    "Sex", "ThyroidW", "TPOAb", "FT4_0M", "TSH_0M",
    "log1p_DiseaseDuration_Months_Aug",
]

PRETTY = {
    "Sex": "Sex",
    "ThyroidW": "Thyroid weight (g)",
    "TPOAb": "TPOAb",
    "FT4_0M": "FT4 at baseline",
    "TSH_0M": "TSH at baseline",
    "log1p_DiseaseDuration_Months_Aug": "log1p(disease duration in months)",
}


def main() -> None:
    forbidden = str(890 - 1)
    if forbidden in Path(__file__).read_text():
        raise RuntimeError("Forbidden literal present.")

    V3_TAB.mkdir(parents=True, exist_ok=True)
    f = pd.read_csv(FROZEN)
    needed = {"Episode_Index", "Split", "OOF_Fold", "Y"}
    feat = [c for c in f.columns if c not in needed]
    for c in feat:
        f[c] = pd.to_numeric(f[c], errors="coerce")
    f[feat] = f[feat].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    dev = f.Split.eq("Development").to_numpy()
    y_dev = f.loc[dev, "Y"].to_numpy()
    X_dev = f.loc[dev, V6_FEATURES].reset_index(drop=True)

    # 1. StandardScaler stats
    scaler = StandardScaler().fit(X_dev)
    mu = scaler.mean_
    sigma = scaler.scale_

    # 2. L2-LR on standardized features
    X_scaled = scaler.transform(X_dev)
    lr = LogisticRegression(penalty="l2", solver="lbfgs", C=1.0,
                            max_iter=5000, random_state=PY_SEED)
    lr.fit(X_scaled, y_dev)
    beta0 = float(lr.intercept_[0])
    betas = lr.coef_[0]

    # 3. Platt sigmoid (CalibratedClassifierCV cv=3, method=sigmoid)
    inner_pipe = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(penalty="l2", solver="lbfgs", C=1.0,
                                   max_iter=5000, random_state=PY_SEED)),
    ])
    cal = CalibratedClassifierCV(base_estimator=inner_pipe, method="sigmoid", cv=3)
    cal.fit(X_dev, y_dev)
    fold_A = []
    fold_B = []
    for c in cal.calibrated_classifiers_:
        # In sklearn 1.0.2, c.calibrators_ is a list of _SigmoidCalibration objects
        # (one per class for binary). With binary, length 1.
        sig = c.calibrators_[0]
        fold_A.append(float(sig.a_))
        fold_B.append(float(sig.b_))

    # 4. Raw-feature unrolled coefficients
    beta_raw_intercept = beta0 - float(np.sum(betas * mu / sigma))
    beta_raw = betas / sigma

    # Assemble structured output
    out = {
        "model_family": "L2-regularized logistic regression with Platt sigmoid calibration (cv=3)",
        "hyperparameters": {
            "penalty": "l2",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 5000,
            "random_state": PY_SEED,
            "calibration_method": "sigmoid",
            "calibration_cv": 3,
        },
        "features_in_order": V6_FEATURES,
        "n_features": len(V6_FEATURES),
        "pretty_labels": PRETTY,
        "standard_scaler": [
            {"feature": V6_FEATURES[i], "pretty": PRETTY[V6_FEATURES[i]],
             "mu_train": float(mu[i]), "sigma_train": float(sigma[i])}
            for i in range(len(V6_FEATURES))
        ],
        "l2_lr_standardized": {
            "intercept_beta0": beta0,
            "coefficients": [
                {"feature": V6_FEATURES[i], "pretty": PRETTY[V6_FEATURES[i]],
                 "beta": float(betas[i]), "OR_exp_beta": float(np.exp(betas[i]))}
                for i in range(len(V6_FEATURES))
            ],
        },
        "platt_sigmoid_calibration": {
            "note": "sklearn _SigmoidCalibration parameterization: p_cal = sigmoid(A * eta + B); A and B fitted per fold via Platt's method.",
            "folds": [
                {"fold": k + 1, "A": fold_A[k], "B": fold_B[k]}
                for k in range(3)
            ],
            "mean_A": float(np.mean(fold_A)),
            "mean_B": float(np.mean(fold_B)),
        },
        "raw_feature_unrolled": {
            "intercept_beta0_tilde": float(beta_raw_intercept),
            "coefficients": [
                {"feature": V6_FEATURES[i], "pretty": PRETTY[V6_FEATURES[i]],
                 "beta_raw": float(beta_raw[i])}
                for i in range(len(V6_FEATURES))
            ],
        },
    }
    (V3_TAB / "v6_model_formula.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Wrote {V3_TAB}/v6_model_formula.json")
    print(f"  Intercept β₀ = {beta0:.6f}")
    print(f"  Raw intercept β̃₀ = {beta_raw_intercept:.6f}")
    print(f"  3-fold Platt A/B means: A={np.mean(fold_A):.4f}, B={np.mean(fold_B):.4f}")


if __name__ == "__main__":
    main()
