#!/usr/bin/env python
"""M1·v2a (augmented 10-feature) ablation: 10 vs 9 leave-one-feature-out.

For each of {Sex, HalfLife, TGAb, log1p_DiseaseDuration_Months_Aug}, remove
that single feature from the 10-feature curated pool, refit L2-logistic +
Platt on dev (5-fold OOF), score temporal, and report ΔAUC (Dev OOF and
Temporal) with paired-bootstrap 95% CIs (same bootstrap indices applied to
full and ablated predictions, Δ_b = AUC_full(b) − AUC_abl(b)).

The disease-duration row also directly answers "10 vs core 9" — removing
log disease duration from the augmented 10 reduces to the core 9 model.

Forbidden unique-patient count literal never appears in this source (audit
check at runtime via str(890 - 1)).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PY_SEED = 2025
OOF_SEED = 13
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 7

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "module1_v2_lasso_clean"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

SOURCE_TABLE_DIR = ROOT / "results" / "module1_baseline_ml_benchmark" / "tables"
FROZEN_MATRIX_PATH = SOURCE_TABLE_DIR / "module1_frozen_feature_matrix.csv"

M1_V2A_FULL = [
    "Sex",
    "ThyroidW",
    "Uptake24h",
    "HalfLife",
    "TRAb",
    "TGAb",
    "TPOAb",
    "FT4_0M",
    "TSH_0M",
    "log1p_DiseaseDuration_Months_Aug",
]

# Features to ablate, in display order. Includes the user-requested
# Log disease duration plus three additional "in selected core 9 but
# minor on prior PI/LOO" candidates.
ABLATIONS = [
    ("log1p_DiseaseDuration_Months_Aug", "Log disease duration (months) — also '10 → core 9'"),
    ("Sex", "Sex"),
    ("HalfLife", "Effective iodine half-life"),
    ("TGAb", "Thyroglobulin antibody"),
]

PRETTY = {f: lbl for f, lbl in ABLATIONS}
PRETTY.update(
    {
        "ThyroidW": "Thyroid weight",
        "Uptake24h": "24-hour thyroid RAI uptake",
        "TRAb": "TSH receptor antibody",
        "TPOAb": "Thyroid peroxidase antibody",
        "FT4_0M": "FT4 at baseline",
        "TSH_0M": "TSH at baseline",
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_inputs() -> dict:
    frozen = pd.read_csv(FROZEN_MATRIX_PATH)
    needed = {"Episode_Index", "Split", "OOF_Fold", "Y"}
    feature_cols = [c for c in frozen.columns if c not in needed]
    for col in feature_cols:
        frozen[col] = pd.to_numeric(frozen[col], errors="coerce")
    frozen[feature_cols] = (
        frozen[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    if len(frozen) != 1003:
        raise RuntimeError(f"Frozen matrix has {len(frozen)} rows; expected 1003")
    dev = frozen["Split"].eq("Development").to_numpy()
    test = frozen["Split"].eq("Temporal").to_numpy()
    return {"frozen": frozen, "dev_mask": dev, "test_mask": test}


def make_l2() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    penalty="l2", solver="lbfgs", C=1.0,
                    max_iter=5000, random_state=PY_SEED,
                ),
            ),
        ]
    )


def fit_oof_platt(
    X_dev: pd.DataFrame, y_dev: np.ndarray
) -> tuple[np.ndarray, CalibratedClassifierCV]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=OOF_SEED)
    oof = np.zeros(len(y_dev), dtype=float)
    for tr, va in skf.split(X_dev, y_dev):
        cal = CalibratedClassifierCV(base_estimator=make_l2(), method="sigmoid", cv=3)
        cal.fit(X_dev.iloc[tr], y_dev[tr])
        oof[va] = cal.predict_proba(X_dev.iloc[va])[:, 1]
    final = CalibratedClassifierCV(base_estimator=make_l2(), method="sigmoid", cv=3)
    final.fit(X_dev, y_dev)
    return oof, final


def calib_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs", max_iter=2000)
    lr.fit(z, y)
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


def auc_pr_brier(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    return (
        float(roc_auc_score(y, p)),
        float(average_precision_score(y, p)),
        float(brier_score_loss(y, p)),
    )


def paired_bootstrap_delta_auc(
    y: np.ndarray, p_full: np.ndarray, p_abl: np.ndarray,
    *, n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Paired bootstrap ΔAUC: same indices on both predictions, take diff."""
    rng = np.random.default_rng(seed)
    diffs: list[float] = []
    n_samp = len(y)
    for _ in range(n):
        idx = rng.integers(0, n_samp, size=n_samp)
        y_b = y[idx]
        if len(np.unique(y_b)) < 2:
            continue
        diffs.append(
            roc_auc_score(y_b, p_full[idx]) - roc_auc_score(y_b, p_abl[idx])
        )
    if not diffs:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.mean(diffs)),
        float(np.percentile(diffs, 2.5)),
        float(np.percentile(diffs, 97.5)),
    )


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------


def run_ablation(
    X_dev_full: pd.DataFrame, y_dev: np.ndarray,
    X_test_full: pd.DataFrame, y_test: np.ndarray,
    ablated_feature: str,
) -> dict:
    # Full 10-feature run
    oof_full, final_full = fit_oof_platt(X_dev_full, y_dev)
    prob_full_test = final_full.predict_proba(X_test_full)[:, 1]
    auc_oof_full, pr_oof_full, brier_oof_full = auc_pr_brier(y_dev, oof_full)
    auc_tmp_full, pr_tmp_full, brier_tmp_full = auc_pr_brier(y_test, prob_full_test)
    intc_oof_full, slp_oof_full = calib_intercept_slope(y_dev, oof_full)
    intc_tmp_full, slp_tmp_full = calib_intercept_slope(y_test, prob_full_test)

    # Ablated (drop one feature)
    feats_abl = [f for f in M1_V2A_FULL if f != ablated_feature]
    X_dev_abl = X_dev_full[feats_abl]
    X_test_abl = X_test_full[feats_abl]
    oof_abl, final_abl = fit_oof_platt(X_dev_abl, y_dev)
    prob_abl_test = final_abl.predict_proba(X_test_abl)[:, 1]
    auc_oof_abl, pr_oof_abl, brier_oof_abl = auc_pr_brier(y_dev, oof_abl)
    auc_tmp_abl, pr_tmp_abl, brier_tmp_abl = auc_pr_brier(y_test, prob_abl_test)
    intc_oof_abl, slp_oof_abl = calib_intercept_slope(y_dev, oof_abl)
    intc_tmp_abl, slp_tmp_abl = calib_intercept_slope(y_test, prob_abl_test)

    # Paired bootstrap ΔAUC on dev OOF and temporal
    d_oof_mean, d_oof_lo, d_oof_hi = paired_bootstrap_delta_auc(y_dev, oof_full, oof_abl)
    d_tmp_mean, d_tmp_lo, d_tmp_hi = paired_bootstrap_delta_auc(y_test, prob_full_test, prob_abl_test)

    return {
        "AblatedFeature": ablated_feature,
        "PrettyLabel": PRETTY.get(ablated_feature, ablated_feature),
        # Discrimination - full
        "Full_OOF_AUC": auc_oof_full,
        "Full_OOF_PR": pr_oof_full,
        "Full_OOF_Brier": brier_oof_full,
        "Full_Tmp_AUC": auc_tmp_full,
        "Full_Tmp_PR": pr_tmp_full,
        "Full_Tmp_Brier": brier_tmp_full,
        # Discrimination - ablated
        "Abl_OOF_AUC": auc_oof_abl,
        "Abl_OOF_PR": pr_oof_abl,
        "Abl_OOF_Brier": brier_oof_abl,
        "Abl_Tmp_AUC": auc_tmp_abl,
        "Abl_Tmp_PR": pr_tmp_abl,
        "Abl_Tmp_Brier": brier_tmp_abl,
        # Calibration - full
        "Full_OOF_CalibIntercept": intc_oof_full,
        "Full_OOF_CalibSlope": slp_oof_full,
        "Full_Tmp_CalibIntercept": intc_tmp_full,
        "Full_Tmp_CalibSlope": slp_tmp_full,
        # Calibration - ablated
        "Abl_OOF_CalibIntercept": intc_oof_abl,
        "Abl_OOF_CalibSlope": slp_oof_abl,
        "Abl_Tmp_CalibIntercept": intc_tmp_abl,
        "Abl_Tmp_CalibSlope": slp_tmp_abl,
        # Paired ΔAUC
        "Delta_OOF_AUC_mean": d_oof_mean,
        "Delta_OOF_AUC_CI_Low": d_oof_lo,
        "Delta_OOF_AUC_CI_High": d_oof_hi,
        "Delta_Tmp_AUC_mean": d_tmp_mean,
        "Delta_Tmp_AUC_CI_Low": d_tmp_lo,
        "Delta_Tmp_AUC_CI_High": d_tmp_hi,
    }


def plot_ablation(df: pd.DataFrame, out: Path) -> None:
    n = len(df)
    fig, axes = plt.subplots(n, 2, figsize=(12, 2.4 * n + 1), sharex=True)
    if n == 1:
        axes = axes.reshape(1, 2)
    for i, row in df.iterrows():
        for j, (split, mean_col, lo_col, hi_col) in enumerate(
            [
                ("Dev OOF", "Delta_OOF_AUC_mean", "Delta_OOF_AUC_CI_Low", "Delta_OOF_AUC_CI_High"),
                ("Temporal test", "Delta_Tmp_AUC_mean", "Delta_Tmp_AUC_CI_Low", "Delta_Tmp_AUC_CI_High"),
            ]
        ):
            ax = axes[i, j]
            mean = row[mean_col]
            lo = row[lo_col]
            hi = row[hi_col]
            color = "#a23b3b" if (lo > 0) else "#888"
            ax.errorbar(
                [mean], [0], xerr=[[mean - lo], [hi - mean]], fmt="o",
                color=color, capsize=6, markersize=8,
            )
            ax.axvline(0, color="#444", linestyle=":", linewidth=1)
            ax.set_yticks([])
            ax.set_xlim(-0.05, 0.20)
            if i == 0:
                ax.set_title(f"{split} ΔAUC (10 − 9, paired bootstrap CI)", fontsize=10)
            if j == 0:
                ax.text(
                    -0.045, 0.45, f"−{row['PrettyLabel']}", fontsize=10,
                    color="#1d4e89", weight="bold", va="center",
                )
            ax.text(
                hi + 0.005, 0, f"{mean:+.4f}  [{lo:+.4f}, {hi:+.4f}]",
                fontsize=8, color="#444", va="center",
            )
            for spine in ("top", "right", "left"):
                ax.spines[spine].set_visible(False)
            ax.tick_params(axis="y", which="both", left=False)
    fig.suptitle("Figure 12. M1·v2a 10 vs 9 ablation — paired bootstrap ΔAUC", fontsize=12)
    fig.text(0.5, 0.005, "ΔAUC = full(10) − ablated(9); CI fully above 0 → feature carries non-redundant signal", ha="center", fontsize=8, color="#666")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(out, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    forbidden = str(890 - 1)
    if forbidden in Path(__file__).read_text():
        raise RuntimeError("Forbidden unique-patient count literal present in source.")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs()
    frozen = inputs["frozen"]
    dev_mask = inputs["dev_mask"]
    test_mask = inputs["test_mask"]
    y_all = frozen["Y"].to_numpy()
    X_all = frozen[M1_V2A_FULL].copy()

    X_dev = X_all.loc[dev_mask].reset_index(drop=True)
    y_dev = y_all[dev_mask]
    X_test = X_all.loc[test_mask].reset_index(drop=True)
    y_test = y_all[test_mask]

    rows: list[dict] = []
    for feat, label in ABLATIONS:
        print(f"Ablating: {feat} ({label})")
        r = run_ablation(X_dev, y_dev, X_test, y_test, feat)
        rows.append(r)
        sig_oof = "*" if r["Delta_OOF_AUC_CI_Low"] > 0 else ""
        sig_tmp = "*" if r["Delta_Tmp_AUC_CI_Low"] > 0 else ""
        print(
            f"  ΔOOF AUC = {r['Delta_OOF_AUC_mean']:+.4f}{sig_oof} "
            f"[{r['Delta_OOF_AUC_CI_Low']:+.4f}, {r['Delta_OOF_AUC_CI_High']:+.4f}]"
        )
        print(
            f"  ΔTmp AUC = {r['Delta_Tmp_AUC_mean']:+.4f}{sig_tmp} "
            f"[{r['Delta_Tmp_AUC_CI_Low']:+.4f}, {r['Delta_Tmp_AUC_CI_High']:+.4f}]"
        )

    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "ablation_10_vs_9_augmented.csv", index=False)
    plot_ablation(df, FIG_DIR / "Figure_12_Ablation_10_vs_9.png")

    summary = {
        "ablations": [
            {
                "feature": r["AblatedFeature"],
                "pretty": r["PrettyLabel"],
                "delta_oof_auc": r["Delta_OOF_AUC_mean"],
                "delta_oof_auc_ci": [r["Delta_OOF_AUC_CI_Low"], r["Delta_OOF_AUC_CI_High"]],
                "delta_oof_auc_ci_above_zero": bool(r["Delta_OOF_AUC_CI_Low"] > 0),
                "delta_temporal_auc": r["Delta_Tmp_AUC_mean"],
                "delta_temporal_auc_ci": [r["Delta_Tmp_AUC_CI_Low"], r["Delta_Tmp_AUC_CI_High"]],
                "delta_temporal_auc_ci_above_zero": bool(r["Delta_Tmp_AUC_CI_Low"] > 0),
            }
            for r in rows
        ]
    }
    (TABLE_DIR / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print("\nDone. Ablation table + figure saved.")


if __name__ == "__main__":
    main()
