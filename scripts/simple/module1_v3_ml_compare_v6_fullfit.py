#!/usr/bin/env python
"""M1·v3 — 8 common ML methods + L2-LR head-to-head on v6's 6 features.

Replaces the earlier ml_comparison (which mixed 10-feature non-LR with
6-feature LR) with a clean apples-to-apples comparison: all 9 methods
fit on the same v6 6-feature pool, same train/temporal split, same
Platt-calibrated 5-fold OOF wrapper.

The 9 methods (our LR + 8 commonly cited in clinical-prediction lit):
  M0  L2-Logistic Regression (ours, v6 main line)
  M1  RandomForest
  M2  GradientBoosting (sklearn)
  M3  AdaBoost
  M4  DecisionTree
  M5  KNN (k=25, distance)
  M6  SVM-RBF
  M7  GaussianNB
  M8  MLP (64-32, ReLU)

For each: 5-fold OOF on dev (n=802) + Platt CV=3 + temporal predict
(n=201); bootstrap × 1000 for 95% CI on temporal ROC-AUC; paired
bootstrap ΔAUC vs L2-LR baseline (positive Δ = method beats LR;
CI fully above 0 = statistically wins).

Output (overwrites previous 10-feature version):
  results/module1_v3/tables/ml_comparison_table.csv
  results/module1_v3/tables/ml_comparison_deltas.csv
  results/module1_v3/tables/ml_comparison_summary.json
  results/module1_v3/figures/Figure_v3_30_ML_Comparison.png

Forbidden unique-patient count literal never appears (audit at runtime
via str(890 - 1)).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

PY_SEED = 2025
OOF_SEED = 13
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 7

ROOT = Path(__file__).resolve().parents[2]
V3_DIR = ROOT / "results" / "module1_v3"
V3_FIG = V3_DIR / "figures"
V3_TABLES = V3_DIR / "tables"
FROZEN = ROOT / "results" / "module1_baseline_ml_benchmark" / "tables" / "module1_frozen_feature_matrix.csv"

V6_FEATURES = [
    "Sex", "ThyroidW", "TPOAb", "FT4_0M", "TSH_0M",
    "log1p_DiseaseDuration_Months_Aug",
]


def estimator(method: str):
    if method == "L2-Logistic (ours)":
        return LogisticRegression(penalty="l2", solver="lbfgs", C=1.0,
                                  max_iter=5000, random_state=PY_SEED)
    if method == "RandomForest":
        return RandomForestClassifier(n_estimators=300, max_depth=6,
                                      min_samples_leaf=10, n_jobs=-1,
                                      random_state=PY_SEED)
    if method == "GradientBoosting":
        return GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                          learning_rate=0.05,
                                          random_state=PY_SEED)
    if method == "AdaBoost":
        return AdaBoostClassifier(n_estimators=200, learning_rate=0.5,
                                  random_state=PY_SEED)
    if method == "DecisionTree":
        return DecisionTreeClassifier(max_depth=5, min_samples_leaf=20,
                                      random_state=PY_SEED)
    if method == "KNN":
        return KNeighborsClassifier(n_neighbors=25, weights="distance",
                                    n_jobs=-1)
    if method == "SVM-RBF":
        return SVC(kernel="rbf", C=1.0, gamma="scale",
                   probability=True, random_state=PY_SEED)
    if method == "GaussianNB":
        return GaussianNB()
    if method == "MLP":
        return MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                             solver="adam", alpha=1e-3, learning_rate_init=1e-3,
                             max_iter=400, early_stopping=True,
                             validation_fraction=0.1, random_state=PY_SEED)
    raise ValueError(f"Unknown method: {method}")


METHODS_ORDER = [
    "L2-Logistic (ours)",
    "RandomForest", "GradientBoosting", "AdaBoost", "DecisionTree",
    "KNN", "SVM-RBF", "GaussianNB", "MLP",
]


def make_pipeline(method: str) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("clf", estimator(method))])


def load_data():
    frozen = pd.read_csv(FROZEN)
    needed = {"Episode_Index", "Split", "OOF_Fold", "Y"}
    feat = [c for c in frozen.columns if c not in needed]
    for col in feat:
        frozen[col] = pd.to_numeric(frozen[col], errors="coerce")
    frozen[feat] = frozen[feat].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    dev = frozen["Split"].eq("Development").to_numpy()
    tst = frozen["Split"].eq("Temporal").to_numpy()
    return frozen, dev, tst


def fit_oof_platt(method: str, X_dev, y_dev):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=OOF_SEED)
    oof = np.zeros(len(y_dev), dtype=float)
    for tr, va in skf.split(X_dev, y_dev):
        base = make_pipeline(method)
        cal = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
        cal.fit(X_dev.iloc[tr], y_dev[tr])
        oof[va] = cal.predict_proba(X_dev.iloc[va])[:, 1]
    final = CalibratedClassifierCV(estimator=make_pipeline(method),
                                   method="sigmoid", cv=3)
    final.fit(X_dev, y_dev)
    return oof, final


def boot_auc(y, p, *, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return (float(np.mean(aucs)),
            float(np.percentile(aucs, 2.5)),
            float(np.percentile(aucs, 97.5)))


def paired_delta(y, p_a, p_b, *, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    return (float(np.mean(diffs)),
            float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)))


def plot_compare(rows: list[dict], deltas: list[dict], out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # Left: bar of temporal AUC per method (LR highlighted in red)
    ax = axes[0]
    methods = [r["Method"] for r in rows]
    means = [r["Tmp_AUC_mean"] for r in rows]
    lo = [r["Tmp_AUC_mean"] - r["Tmp_AUC_CI_Low"] for r in rows]
    hi = [r["Tmp_AUC_CI_High"] - r["Tmp_AUC_mean"] for r in rows]
    colors = ["#a23b3b" if "L2-Logistic" in m else "#456ea6" for m in methods]
    x = np.arange(len(methods))
    ax.bar(x, means, color=colors, edgecolor="black", linewidth=0.6,
           yerr=[lo, hi], capsize=4,
           error_kw={"linewidth": 0.8, "ecolor": "#222"})
    for i, (m, l, h) in enumerate(zip(means, lo, hi)):
        ax.text(i, m + h + 0.01, f"{m:.3f}", ha="center", fontsize=8, color="#333")
    ax.axhline(0.5, color="#888", linestyle=":", linewidth=1, label="Chance (AUC 0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0.45, 0.80)
    ax.set_ylabel("Temporal-test ROC-AUC (95% CI, bootstrap × 1000)")
    ax.set_title("(A) M1 v6 (6 features) — L2-LR vs 8 common ML methods\n"
                 "(same 6-feature pool, same train/temporal split, same Platt wrapper)")
    ax.legend(loc="lower right", fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Right: paired ΔAUC (method - LR baseline)
    ax = axes[1]
    others = [d for d in deltas if "L2-Logistic" not in d["Method"]]
    methods_o = [d["Method"] for d in others]
    means_o = [d["Tmp_Delta_mean"] for d in others]
    los_o = [d["Tmp_Delta_CI_Low"] for d in others]
    his_o = [d["Tmp_Delta_CI_High"] for d in others]
    y_pos = np.arange(len(methods_o))
    colors_d = ["#a23b3b" if lo > 0 else ("#1d4e89" if hi < 0 else "#888")
                for lo, hi in zip(los_o, his_o)]
    ax.errorbar(means_o, y_pos,
                xerr=[[m - lo for m, lo in zip(means_o, los_o)],
                      [hi - m for m, hi in zip(means_o, his_o)]],
                fmt="o", color="black", capsize=5, zorder=4)
    for i, (m, c) in enumerate(zip(means_o, colors_d)):
        ax.scatter([m], [i], s=100, color=c, zorder=5, edgecolor="black")
    ax.axvline(0, color="#444", linestyle="--", linewidth=1.2,
               label="L2-LR baseline (v6 main line)")
    ax.set_yticks(y_pos); ax.set_yticklabels(methods_o, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Paired ΔAUC: method − L2-LR (temporal, bootstrap × 1000)")
    ax.set_title("(B) Per-method: significantly beats / ties / loses to L2-LR?")
    ax.legend(loc="lower right", fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for i, (m, lo, hi) in enumerate(zip(means_o, los_o, his_o)):
        sig = "★" if (lo > 0 or hi < 0) else ""
        ax.text(hi + 0.003, i, f"{m:+.4f}{sig}  [{lo:+.4f}, {hi:+.4f}]",
                fontsize=8, color="#444", va="center")

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    forbidden = str(890 - 1)
    if forbidden in Path(__file__).read_text():
        raise RuntimeError("Forbidden unique-patient count literal present in source.")

    V3_FIG.mkdir(parents=True, exist_ok=True)
    V3_TABLES.mkdir(parents=True, exist_ok=True)

    frozen, dev_mask, tst_mask = load_data()
    y_dev = frozen.loc[dev_mask, "Y"].to_numpy()
    y_tst = frozen.loc[tst_mask, "Y"].to_numpy()
    X_dev = frozen.loc[dev_mask, V6_FEATURES].reset_index(drop=True)
    X_tst = frozen.loc[tst_mask, V6_FEATURES].reset_index(drop=True)
    print(f"Loaded: dev N={len(y_dev)} (events={int(y_dev.sum())}), "
          f"temporal N={len(y_tst)} (events={int(y_tst.sum())})")
    print(f"Features ({len(V6_FEATURES)}): {V6_FEATURES}")

    cache: dict[str, dict] = {}
    rows: list[dict] = []
    for m in METHODS_ORDER:
        print(f"\nFitting {m} …")
        oof, final = fit_oof_platt(m, X_dev, y_dev)
        p_tst = final.predict_proba(X_tst)[:, 1]
        o_mean, o_lo, o_hi = boot_auc(y_dev, oof)
        t_mean, t_lo, t_hi = boot_auc(y_tst, p_tst)
        rec = {
            "Method": m,
            "OOF_AUC_mean": o_mean, "OOF_AUC_CI_Low": o_lo, "OOF_AUC_CI_High": o_hi,
            "Tmp_AUC_mean": t_mean, "Tmp_AUC_CI_Low": t_lo, "Tmp_AUC_CI_High": t_hi,
            "OOF_PR_AUC": float(average_precision_score(y_dev, oof)),
            "Tmp_PR_AUC": float(average_precision_score(y_tst, p_tst)),
            "OOF_Brier": float(brier_score_loss(y_dev, oof)),
            "Tmp_Brier": float(brier_score_loss(y_tst, p_tst)),
        }
        rows.append(rec)
        cache[m] = {"oof": oof, "test": p_tst}
        print(f"  OOF AUC = {o_mean:.4f} [{o_lo:.3f}, {o_hi:.3f}]  "
              f"Tmp AUC = {t_mean:.4f} [{t_lo:.3f}, {t_hi:.3f}]  "
              f"Tmp Brier = {rec['Tmp_Brier']:.4f}")

    # Paired ΔAUC vs LR baseline
    print("\nPaired bootstrap ΔAUC vs L2-Logistic (temporal) …")
    lr_oof = cache["L2-Logistic (ours)"]["oof"]
    lr_test = cache["L2-Logistic (ours)"]["test"]
    deltas: list[dict] = []
    for m in METHODS_ORDER:
        d_o_m, d_o_lo, d_o_hi = paired_delta(y_dev, cache[m]["oof"], lr_oof)
        d_t_m, d_t_lo, d_t_hi = paired_delta(y_tst, cache[m]["test"], lr_test)
        deltas.append({
            "Method": m,
            "OOF_Delta_mean": d_o_m, "OOF_Delta_CI_Low": d_o_lo, "OOF_Delta_CI_High": d_o_hi,
            "Tmp_Delta_mean": d_t_m, "Tmp_Delta_CI_Low": d_t_lo, "Tmp_Delta_CI_High": d_t_hi,
        })
        sig = "★" if (d_t_lo > 0 or d_t_hi < 0) else ""
        print(f"  {m:<24}  ΔTmp = {d_t_m:+.4f}{sig} [{d_t_lo:+.4f}, {d_t_hi:+.4f}]")

    pd.DataFrame(rows).to_csv(V3_TABLES / "ml_comparison_table.csv", index=False)
    pd.DataFrame(deltas).to_csv(V3_TABLES / "ml_comparison_deltas.csv", index=False)

    n_beat = sum(d["Tmp_Delta_CI_Low"] > 0 for d in deltas if "L2-Logistic" not in d["Method"])
    n_lose = sum(d["Tmp_Delta_CI_High"] < 0 for d in deltas if "L2-Logistic" not in d["Method"])
    summary = {
        "feature_pool": "v6_6_features",
        "features": V6_FEATURES,
        "n_methods_compared": len(METHODS_ORDER),
        "n_methods_beating_LR_temporal_significantly": int(n_beat),
        "n_methods_losing_to_LR_temporal_significantly": int(n_lose),
        "n_methods_tied_with_LR_temporal": int(len(METHODS_ORDER) - 1 - n_beat - n_lose),
        "LR_Tmp_AUC": float(rows[0]["Tmp_AUC_mean"]),
        "best_Tmp_AUC": float(max(r["Tmp_AUC_mean"] for r in rows)),
        "best_method": max(rows, key=lambda r: r["Tmp_AUC_mean"])["Method"],
    }
    (V3_TABLES / "ml_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    plot_compare(rows, deltas, V3_FIG / "Figure_v3_30_ML_Comparison.png")
    print(f"\nDone. Figure 30 written. Best: {summary['best_method']} "
          f"({summary['best_Tmp_AUC']:.4f}); LR ranks "
          f"{sorted([(r['Tmp_AUC_mean'], r['Method']) for r in rows], reverse=True).index((rows[0]['Tmp_AUC_mean'], rows[0]['Method'])) + 1} "
          f"of {len(METHODS_ORDER)}")


if __name__ == "__main__":
    main()
