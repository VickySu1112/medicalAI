#!/usr/bin/env python
"""Generate v3-specific single-line figures (no comparisons with any
earlier M1 variant).

Produces three figures into results/module1_v3/figures/:
  Figure_v3_01_LASSO_Path.png      — single curve for the 10-feature pool
  Figure_v3_02_ROC.png             — single ROC curve, temporal test
  Figure_v3_03_PR.png              — single PR curve, temporal test

Forbidden unique-patient count literal never appears (audit at runtime
via str(890 - 1)).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PY_SEED = 2025
OOF_SEED = 13

ROOT = Path(__file__).resolve().parents[2]
V3_DIR = ROOT / "results" / "module1_v3"
V3_FIG = V3_DIR / "figures"
LASSO_PATH_CSV = ROOT / "results" / "module1_v2_lasso_clean" / "tables" / "lasso_path.csv"
FROZEN = ROOT / "results" / "module1_baseline_ml_benchmark" / "tables" / "module1_frozen_feature_matrix.csv"

# v3/v6 主线: 6 特征
V6_FEATURES = [
    "Sex", "ThyroidW", "TPOAb", "FT4_0M", "TSH_0M",
    "log1p_DiseaseDuration_Months_Aug",
]
FEATURES_10 = V6_FEATURES  # backwards-compatible alias used below


def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frozen = pd.read_csv(FROZEN)
    needed = {"Episode_Index", "Split", "OOF_Fold", "Y"}
    feat = [c for c in frozen.columns if c not in needed]
    for col in feat:
        frozen[col] = pd.to_numeric(frozen[col], errors="coerce")
    frozen[feat] = frozen[feat].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    dev = frozen["Split"].eq("Development").to_numpy()
    tst = frozen["Split"].eq("Temporal").to_numpy()
    return frozen, dev, tst


def make_l2() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(
            penalty="l2", solver="lbfgs", C=1.0, max_iter=5000, random_state=PY_SEED)),
    ])


def fit_oof_platt(X_dev, y_dev):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=OOF_SEED)
    oof = np.zeros(len(y_dev), dtype=float)
    for tr, va in skf.split(X_dev, y_dev):
        cal = CalibratedClassifierCV(estimator=make_l2(), method="sigmoid", cv=3)
        cal.fit(X_dev.iloc[tr], y_dev[tr])
        oof[va] = cal.predict_proba(X_dev.iloc[va])[:, 1]
    final = CalibratedClassifierCV(estimator=make_l2(), method="sigmoid", cv=3)
    final.fit(X_dev, y_dev)
    return oof, final


def fig_lasso_path(out: Path) -> None:
    path = pd.read_csv(LASSO_PATH_CSV)
    aug = path[path["FeaturePool"] == "M1_v2a"].sort_values("C")
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color_left = "#1d4e89"
    ax1.set_xscale("log")
    ax1.plot(aug["C"], aug["AvgNonzero"], "o-", color=color_left, lw=2, ms=7,
             label="Avg non-zero features (5-fold mean)")
    ax1.fill_between(aug["C"], aug["MinNonzero"], aug["MaxNonzero"],
                     color=color_left, alpha=0.15, label="Per-fold range")
    ax1.set_xlabel("LASSO C (inverse regularization)")
    ax1.set_ylabel("Number of non-zero features", color=color_left)
    ax1.tick_params(axis="y", labelcolor=color_left)
    ax1.axhline(6, color="#a23b3b", linestyle="--", linewidth=1.2,
                label="Selected k = 6 (v6 main line)")
    ax1.set_title("Figure 1. LASSO selection path on 10-feature candidate pool\n"
                  "(v6 main line keeps k=6 after pruning low-signal features)")

    # Right axis: OOF AUC along the same C grid
    ax2 = ax1.twinx()
    color_right = "#a23b3b"
    ax2.plot(aug["C"], aug["MeanOOF_AUC"], "s-", color=color_right, lw=1.5, ms=5,
             label="5-fold OOF ROC-AUC")
    ax2.set_ylabel("5-fold OOF ROC-AUC", color=color_right)
    ax2.tick_params(axis="y", labelcolor=color_right)
    ax2.set_ylim(0.70, 0.74)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    for spine in ("top",):
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_roc_pr(out_roc: Path, out_pr: Path) -> None:
    frozen, dev_mask, tst_mask = load_data()
    y_dev = frozen.loc[dev_mask, "Y"].to_numpy()
    y_tst = frozen.loc[tst_mask, "Y"].to_numpy()
    X_dev = frozen.loc[dev_mask, FEATURES_10].reset_index(drop=True)
    X_tst = frozen.loc[tst_mask, FEATURES_10].reset_index(drop=True)
    oof, final = fit_oof_platt(X_dev, y_dev)
    p_tst = final.predict_proba(X_tst)[:, 1]

    # ROC
    fpr, tpr, _ = roc_curve(y_tst, p_tst)
    auc = roc_auc_score(y_tst, p_tst)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(fpr, tpr, color="#1d4e89", lw=2.2, label=f"M1 (6 features): AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#888", linestyle=":", linewidth=1, label="Chance (AUC 0.5)")
    ax.set_xlabel("False positive rate (1 − specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Figure 2. M1 temporal-test ROC curve (6 features)")
    ax.legend(loc="lower right", fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(); fig.savefig(out_roc, dpi=160); plt.close(fig)
    print(f"  wrote {out_roc}  (ROC-AUC = {auc:.3f})")

    # PR
    pre, rec, _ = precision_recall_curve(y_tst, p_tst)
    ap = average_precision_score(y_tst, p_tst)
    prevalence = y_tst.mean()
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(rec, pre, color="#a23b3b", lw=2.2, label=f"M1 (6 features): AP = {ap:.3f}")
    ax.axhline(prevalence, color="#888", linestyle=":", linewidth=1,
               label=f"Prevalence {prevalence:.3f}")
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Figure 3. M1 temporal-test precision-recall curve (6 features)")
    ax.legend(loc="upper right", fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(); fig.savefig(out_pr, dpi=160); plt.close(fig)
    print(f"  wrote {out_pr}  (PR-AUC = {ap:.3f})")


def main() -> None:
    forbidden = str(890 - 1)
    if forbidden in Path(__file__).read_text():
        raise RuntimeError("Forbidden unique-patient count literal present in source.")

    V3_FIG.mkdir(parents=True, exist_ok=True)
    print("Figure 1: LASSO selection path (10-feature pool only)")
    fig_lasso_path(V3_FIG / "Figure_v3_01_LASSO_Path.png")
    print("Figures 2-3: ROC + PR curves (10-feature single line)")
    fig_roc_pr(V3_FIG / "Figure_v3_02_ROC.png",
               V3_FIG / "Figure_v3_03_PR.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
