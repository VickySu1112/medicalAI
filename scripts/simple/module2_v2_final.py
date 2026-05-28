#!/usr/bin/env python
"""M2 v2 final: per-landmark 4-LR + ABCDE full reporting.

This is the iter 2 paper-recommended architecture (per-landmark independent
L2-logistic on the 5 mechanism block ABCDE feature set). Generates the full
per-landmark + pooled reporting layer:

- Calibration curve per landmark (Temporal)
- Risk-tier table (dev tertile cutoffs applied to temporal)
- Decision Curve Analysis per landmark
- Multi-seed bootstrap stability check
- Per-landmark and pooled performance with episode-cluster CIs

Outputs to results/module2_v2_base/ alongside the supermodel reports.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.simple.module2_v2_shared import (
    BLOCKS,
    LANDMARKS,
    StackedData,
    apply_per_landmark_platt,
    calib_intercept_slope,
    compute_metrics,
    episode_cluster_bootstrap_auc_ci,
    forbidden_token,
    load_stacked,
    per_landmark_platt_on_pooled_oof,
)

OUT_DIR = ROOT / "results" / "module2_v2_base"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"


# ---------------------------------------------------------------------------
# Per-landmark independent 4-LR with ABCDE features
# ---------------------------------------------------------------------------


def fit_per_landmark_oof(sd: StackedData, blocks: tuple[str, ...]) -> tuple[np.ndarray, dict[int, Pipeline]]:
    """Fit one L2-logistic per landmark independently."""
    X = sd.feature_matrix(blocks).values
    y = sd.rows["Y_24M_NHRH"].values
    lm = sd.rows["landmark"].values
    split = sd.rows["Split"].values
    proba = np.zeros(len(sd.rows))
    final_models: dict[int, Pipeline] = {}
    for L in LANDMARKS:
        mask = lm == L
        dev = mask & (split == "Development")
        test = mask & (split == "Temporal")
        X_d = X[dev]
        y_d = y[dev]
        # 5-fold OOF (plain StratifiedKFold; landmark-L rows are mutually independent)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        dev_idx = np.where(dev)[0]
        for tr, va in skf.split(X_d, y_d):
            pipe = Pipeline([("s", StandardScaler()),
                              ("lr", LogisticRegression(penalty="l2", solver="lbfgs",
                                                         C=1.0, max_iter=5000, random_state=2025))])
            pipe.fit(X_d[tr], y_d[tr])
            proba[dev_idx[va]] = pipe.predict_proba(X_d[va])[:, 1]
        # Final on full dev → score test
        final = Pipeline([("s", StandardScaler()),
                           ("lr", LogisticRegression(penalty="l2", solver="lbfgs",
                                                      C=1.0, max_iter=5000, random_state=2025))])
        final.fit(X_d, y_d)
        if test.any():
            proba[test] = final.predict_proba(X[test])[:, 1]
        final_models[L] = final
    return proba, final_models


# ---------------------------------------------------------------------------
# Reporting: per-landmark calibration / DCA / risk tier
# ---------------------------------------------------------------------------


def fig_calibration_perlandmark(sd: StackedData, proba: np.ndarray, out: Path) -> None:
    """4-panel calibration: each landmark on Temporal with binned reliability."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    is_test = sd.rows["Split"].values == "Temporal"
    y_all = sd.rows["Y_24M_NHRH"].values
    for ax, L in zip(axes, LANDMARKS):
        m = is_test & (sd.rows["landmark"].values == L)
        if m.sum() < 5:
            continue
        y = y_all[m]
        p = proba[m]
        # Binned reliability
        try:
            frac, mean_pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
        except Exception:
            ax.set_title(f"Landmark {L}M (insufficient data)")
            continue
        intc, slope = calib_intercept_slope(y, p)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.plot(mean_pred, frac, "o-", color="#1d4e89")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed fraction")
        ax.set_title(f"Landmark {L}M\nintercept={intc:+.3f}, slope={slope:+.3f}\nBrier={brier_score_loss(y, p):.3f}",
                     fontsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("M2 final (per-landmark 4-LR + ABCDE) — Temporal calibration", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_dca_perlandmark(sd: StackedData, proba: np.ndarray, out: Path) -> None:
    """Decision Curve Analysis: net benefit vs threshold per landmark."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    is_test = sd.rows["Split"].values == "Temporal"
    y_all = sd.rows["Y_24M_NHRH"].values
    for ax, L in zip(axes, LANDMARKS):
        m = is_test & (sd.rows["landmark"].values == L)
        if m.sum() < 5:
            continue
        y = y_all[m]
        p = proba[m]
        prevalence = y.mean()
        thresholds = np.linspace(0.05, 0.6, 30)
        nb_model = []
        nb_all = []
        nb_none = []
        for t in thresholds:
            tp = ((p >= t) & (y == 1)).sum()
            fp = ((p >= t) & (y == 0)).sum()
            nb = tp / len(y) - fp / len(y) * t / (1 - t)
            nb_model.append(nb)
            tp_a = y.sum()
            fp_a = (1 - y).sum()
            nb_all.append(tp_a / len(y) - fp_a / len(y) * t / (1 - t))
            nb_none.append(0)
        ax.plot(thresholds, nb_model, color="#a23b3b", label="Model")
        ax.plot(thresholds, nb_all, color="#888", linestyle="--", label="Treat all")
        ax.plot(thresholds, nb_none, color="#444", linestyle=":", label="Treat none")
        ax.set_xlabel("Decision threshold")
        ax.set_ylabel("Net benefit")
        ax.set_title(f"Landmark {L}M (prev={prevalence:.2f})", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("M2 final — Decision Curve Analysis per landmark (Temporal)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=160)
    plt.close(fig)


def build_risk_tiers(sd: StackedData, proba: np.ndarray) -> pd.DataFrame:
    """Per-landmark risk tier (low/mid/high tertiles cut on dev OOF) → temporal stats."""
    rows = []
    dev_mask = sd.rows["Split"].values == "Development"
    tmp_mask = sd.rows["Split"].values == "Temporal"
    lm_arr = sd.rows["landmark"].values
    y_all = sd.rows["Y_24M_NHRH"].values
    for L in LANDMARKS:
        d_mask = dev_mask & (lm_arr == L)
        t_mask = tmp_mask & (lm_arr == L)
        if d_mask.sum() < 5:
            continue
        # Cut on dev
        t1, t2 = np.percentile(proba[d_mask], [33.33, 66.67])
        for split_name, m in (("Dev OOF", d_mask), ("Temporal", t_mask)):
            y = y_all[m]
            p = proba[m]
            tiers = np.where(p <= t1, "Low", np.where(p <= t2, "Mid", "High"))
            for tier in ("Low", "Mid", "High"):
                sub = tiers == tier
                if sub.sum() == 0:
                    continue
                event_rate = y[sub].mean()
                rows.append({
                    "Landmark": f"{L}M",
                    "Split": split_name,
                    "Tier": tier,
                    "N": int(sub.sum()),
                    "Events": int(y[sub].sum()),
                    "EventRate": float(event_rate),
                    "NPV_if_low": 1 - event_rate if tier == "Low" else float("nan"),
                    "PPV_if_high": event_rate if tier == "High" else float("nan"),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Multi-seed stability check
# ---------------------------------------------------------------------------


def multi_seed_bootstrap_check(sd: StackedData, proba: np.ndarray, n_seeds: int = 5) -> pd.DataFrame:
    """Per-landmark + pooled ROC under different bootstrap seeds → check stability."""
    is_test = sd.rows["Split"].values == "Temporal"
    y_test = sd.rows.loc[is_test, "Y_24M_NHRH"].values
    rows = []
    seeds = list(range(2025, 2025 + n_seeds))
    for s in seeds:
        # Pooled
        m, lo, hi = episode_cluster_bootstrap_auc_ci(sd, proba, landmark=None, split="Temporal", seed=s)
        rows.append({"Seed": s, "Landmark": "Pooled", "Bootstrap_Mean_ROC": m, "CI_Low": lo, "CI_High": hi})
        for L in LANDMARKS:
            m, lo, hi = episode_cluster_bootstrap_auc_ci(sd, proba, landmark=L, split="Temporal", seed=s)
            rows.append({"Seed": s, "Landmark": f"{L}M", "Bootstrap_Mean_ROC": m, "CI_Low": lo, "CI_High": hi})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    tok = forbidden_token()
    if tok in Path(__file__).read_text():
        raise RuntimeError(f"Forbidden literal '{tok}' present in source.")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading stacked dataset…", flush=True)
    sd = load_stacked()

    print("\n[Final] Per-landmark 4-LR + ABCDE …", flush=True)
    raw, final_models = fit_per_landmark_oof(sd, blocks=("A", "B", "C", "D", "E"))
    cal = per_landmark_platt_on_pooled_oof(sd, raw)
    p_cal = apply_per_landmark_platt(sd, raw, cal)

    # Per-landmark + pooled performance with CIs
    is_test = sd.rows["Split"].values == "Temporal"
    y_test = sd.rows.loc[is_test, "Y_24M_NHRH"].values
    perf_rows = []
    for label, mask in [(f"{L}M", is_test & (sd.rows["landmark"].values == L)) for L in LANDMARKS] + \
                       [("Pooled", is_test)]:
        if mask.sum() < 5:
            continue
        y = sd.rows.loc[mask, "Y_24M_NHRH"].values
        p = p_cal[mask]
        m = compute_metrics(y, p)
        ic, sl = calib_intercept_slope(y, p)
        if label == "Pooled":
            mu, lo, hi = episode_cluster_bootstrap_auc_ci(sd, p_cal, landmark=None, split="Temporal")
        else:
            L_int = int(label[:-1])
            mu, lo, hi = episode_cluster_bootstrap_auc_ci(sd, p_cal, landmark=L_int, split="Temporal")
        perf_rows.append({"Landmark": label, "N": int(mask.sum()), "Events": int(y.sum()),
                           **m, "ROC_AUC_CI_Low": lo, "ROC_AUC_CI_High": hi,
                           "CalibIntercept": ic, "CalibSlope": sl})
    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(TAB_DIR / "final_4lr_abcde_perf.csv", index=False)

    # Calibration plot
    fig_calibration_perlandmark(sd, p_cal, FIG_DIR / "Figure_09_Final_Calibration.png")
    # DCA plot
    fig_dca_perlandmark(sd, p_cal, FIG_DIR / "Figure_10_Final_DCA.png")

    # Risk tier
    tier_df = build_risk_tiers(sd, p_cal)
    tier_df.to_csv(TAB_DIR / "final_4lr_abcde_risk_tiers.csv", index=False)

    # Multi-seed sensitivity
    print("\n[Sensitivity] Multi-seed bootstrap stability …", flush=True)
    seed_df = multi_seed_bootstrap_check(sd, p_cal)
    seed_df.to_csv(TAB_DIR / "final_4lr_abcde_multi_seed.csv", index=False)
    pooled_seed = seed_df[seed_df["Landmark"] == "Pooled"]
    print(f"  5-seed mean ROC: {pooled_seed['Bootstrap_Mean_ROC'].mean():.4f} ± {pooled_seed['Bootstrap_Mean_ROC'].std():.4f}", flush=True)
    print(f"  Mean CI width: {(pooled_seed['CI_High'] - pooled_seed['CI_Low']).mean():.4f}", flush=True)

    # Summary
    summary = {
        "architecture": "per-landmark 4-LR (legacy)",
        "features": "A+B+C+D+E (mechanism block ABCDE, 19 features per landmark)",
        "temporal_pooled_roc": float(perf_df[perf_df["Landmark"] == "Pooled"]["ROC_AUC"].iloc[0]),
        "per_landmark_roc": {row["Landmark"]: float(row["ROC_AUC"]) for _, row in perf_df.iterrows()},
        "low_tier_NPV_per_landmark_temporal": {
            row["Landmark"]: float(row["NPV_if_low"])
            for _, row in tier_df.iterrows() if row["Split"] == "Temporal" and row["Tier"] == "Low"
        },
        "multi_seed_pooled_std": float(pooled_seed["Bootstrap_Mean_ROC"].std()),
    }
    (TAB_DIR / "final_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== Per-landmark + pooled performance ===")
    print(perf_df.to_string(index=False), flush=True)
    print("\n=== Risk tiers (Temporal) ===")
    print(tier_df[tier_df["Split"] == "Temporal"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
