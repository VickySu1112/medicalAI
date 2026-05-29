#!/usr/bin/env python
"""M1 v3 — v5 (drop TSH baseline) vs v6 head-to-head.

v6: Sex, ThyroidW, TPOAb, FT4_0M, TSH_0M, log_disease_duration
v5: same minus TSH_0M (TSH had p=0.39 in v6 OR forest — least significant)

Reports standalone perf, paired bootstrap ΔAUC (temporal), calibration,
and bootstrap LASSO selection frequencies. Outputs CSV + JSON. No HTML
generated — for chat-side comparison.

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
from sklearn.metrics import (
    average_precision_score, brier_score_loss, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PY_SEED = 2025
OOF_SEED = 13
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 7

ROOT = Path(__file__).resolve().parents[2]
V3_DIR = ROOT / "results" / "module1_v3"
V3_TAB = V3_DIR / "tables"
FROZEN = ROOT / "results" / "module1_baseline_ml_benchmark" / "tables" / "module1_frozen_feature_matrix.csv"

V6 = ["Sex", "ThyroidW", "TPOAb", "FT4_0M", "TSH_0M", "log1p_DiseaseDuration_Months_Aug"]
V5 = ["Sex", "ThyroidW", "TPOAb", "FT4_0M", "log1p_DiseaseDuration_Months_Aug"]

VARIANTS = [("v6", V6), ("v5_drop_TSH", V5)]

PRETTY = {
    "Sex": "Sex", "ThyroidW": "Thyroid weight", "TPOAb": "TPOAb",
    "FT4_0M": "FT4 (baseline)", "TSH_0M": "TSH (baseline)",
    "log1p_DiseaseDuration_Months_Aug": "Log disease duration",
}


def load():
    f = pd.read_csv(FROZEN)
    needed = {"Episode_Index", "Split", "OOF_Fold", "Y"}
    feat = [c for c in f.columns if c not in needed]
    for c in feat:
        f[c] = pd.to_numeric(f[c], errors="coerce")
    f[feat] = f[feat].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return f, f.Split.eq("Development").to_numpy(), f.Split.eq("Temporal").to_numpy()


def make_l2():
    return Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(penalty="l2", solver="lbfgs", C=1.0, max_iter=5000, random_state=PY_SEED)),
    ])


def fit_oof_platt(X_dev, y_dev):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=OOF_SEED)
    oof = np.zeros(len(y_dev), dtype=float)
    for tr, va in skf.split(X_dev, y_dev):
        cal = CalibratedClassifierCV(base_estimator=make_l2(), method="sigmoid", cv=3)
        cal.fit(X_dev.iloc[tr], y_dev[tr])
        oof[va] = cal.predict_proba(X_dev.iloc[va])[:, 1]
    final = CalibratedClassifierCV(base_estimator=make_l2(), method="sigmoid", cv=3)
    final.fit(X_dev, y_dev)
    return oof, final


def boot_auc_ci(y, p, *, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], p[idx]))
    return float(np.mean(out)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def paired_delta(y, p_a, p_b, *, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def calib_intercept_slope(y, p):
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs", max_iter=2000)
    lr.fit(z, y)
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


def or_table_wald(X_dev, y_dev, features):
    """Standardized OR + Wald CI for L2 logistic refit."""
    scaler = StandardScaler().fit(X_dev)
    Xs = scaler.transform(X_dev)
    lr = LogisticRegression(penalty="l2", solver="lbfgs", C=1.0, max_iter=5000, random_state=PY_SEED)
    lr.fit(Xs, y_dev)
    p_hat = lr.predict_proba(Xs)[:, 1]
    W = p_hat * (1 - p_hat)
    Xa = np.c_[np.ones(len(Xs)), Xs]
    XtWX = Xa.T @ (W[:, None] * Xa)
    cov = np.linalg.pinv(XtWX)
    se = np.sqrt(np.maximum(np.diag(cov)[1:], 0))
    coef = lr.coef_[0]
    rows = []
    from scipy import stats as st
    for i, f in enumerate(features):
        z = coef[i] / (se[i] if se[i] > 0 else 1e-12)
        rows.append({
            "feature": f, "PrettyLabel": PRETTY.get(f, f),
            "OR": float(np.exp(coef[i])),
            "CI_low": float(np.exp(coef[i] - 1.96 * se[i])),
            "CI_high": float(np.exp(coef[i] + 1.96 * se[i])),
            "p": float(2 * (1 - st.norm.cdf(abs(z)))),
        })
    return pd.DataFrame(rows)


def main():
    forbidden = str(890 - 1)
    if forbidden in Path(__file__).read_text():
        raise RuntimeError("Forbidden literal present.")

    V3_TAB.mkdir(parents=True, exist_ok=True)
    frozen, dev_mask, tst_mask = load()
    y_dev = frozen.loc[dev_mask, "Y"].to_numpy()
    y_tst = frozen.loc[tst_mask, "Y"].to_numpy()

    cached = {}
    print("=== Standalone ===")
    for tag, feats in VARIANTS:
        X_dev = frozen.loc[dev_mask, feats].reset_index(drop=True)
        X_tst = frozen.loc[tst_mask, feats].reset_index(drop=True)
        oof, final = fit_oof_platt(X_dev, y_dev)
        p_tst = final.predict_proba(X_tst)[:, 1]
        cached[tag] = {"oof": oof, "tst": p_tst, "feats": feats, "X_dev": X_dev, "X_tst": X_tst}
        m_o, lo_o, hi_o = boot_auc_ci(y_dev, oof)
        m_t, lo_t, hi_t = boot_auc_ci(y_tst, p_tst)
        intc, slp = calib_intercept_slope(y_tst, p_tst)
        brier = brier_score_loss(y_tst, p_tst)
        pr = average_precision_score(y_tst, p_tst)
        print(
            f"  {tag:<14}  k={len(feats):>2}  "
            f"OOF={m_o:.4f} [{lo_o:.3f}, {hi_o:.3f}]  "
            f"Tmp={m_t:.4f} [{lo_t:.3f}, {hi_t:.3f}]  "
            f"PR={pr:.3f}  Brier={brier:.3f}  intc={intc:.3f}  slope={slp:.2f}"
        )

    print("\n=== Paired ΔAUC v5 − v6 (temporal) ===")
    d_o = paired_delta(y_dev, cached["v5_drop_TSH"]["oof"], cached["v6"]["oof"])
    d_t = paired_delta(y_tst, cached["v5_drop_TSH"]["tst"], cached["v6"]["tst"])
    print(f"  Dev OOF Δ = {d_o[0]:+.4f} [{d_o[1]:+.4f}, {d_o[2]:+.4f}]")
    print(f"  Temporal Δ = {d_t[0]:+.4f} [{d_t[1]:+.4f}, {d_t[2]:+.4f}]")
    sig_o = "★" if (d_o[1] > 0 or d_o[2] < 0) else "(CI crosses 0)"
    sig_t = "★" if (d_t[1] > 0 or d_t[2] < 0) else "(CI crosses 0)"
    print(f"  → Dev: {sig_o} | Temporal: {sig_t}")

    print("\n=== v5 OR Forest ===")
    or_v5 = or_table_wald(cached["v5_drop_TSH"]["X_dev"], y_dev, V5)
    for _, r in or_v5.iterrows():
        print(f"  {PRETTY.get(r['feature']):<24}  OR={r['OR']:.3f}  CI=[{r['CI_low']:.3f}, {r['CI_high']:.3f}]  p={r['p']:.3g}")

    # Save summary
    summary = {
        "v6_temporal_AUC": float(boot_auc_ci(y_tst, cached["v6"]["tst"])[0]),
        "v5_temporal_AUC": float(boot_auc_ci(y_tst, cached["v5_drop_TSH"]["tst"])[0]),
        "v5_minus_v6_temporal_Δ": d_t[0],
        "v5_minus_v6_temporal_Δ_CI": [d_t[1], d_t[2]],
        "v5_minus_v6_Δ_CI_crosses_zero": bool(d_t[1] <= 0 <= d_t[2]),
        "v5_temporal_PR": float(average_precision_score(y_tst, cached["v5_drop_TSH"]["tst"])),
        "v5_temporal_Brier": float(brier_score_loss(y_tst, cached["v5_drop_TSH"]["tst"])),
        "v5_calib_intercept": float(calib_intercept_slope(y_tst, cached["v5_drop_TSH"]["tst"])[0]),
        "v5_calib_slope": float(calib_intercept_slope(y_tst, cached["v5_drop_TSH"]["tst"])[1]),
        "v5_OR_table": or_v5.to_dict(orient="records"),
    }
    (V3_TAB / "v5_vs_v6_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    or_v5.to_csv(V3_TAB / "v5_or_table.csv", index=False)
    print(f"\nSaved → {V3_TAB}/v5_vs_v6_summary.json + v5_or_table.csv")


if __name__ == "__main__":
    main()
