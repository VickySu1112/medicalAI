#!/usr/bin/env python
"""M1·v2 — non-LR robustness check: does "Thyroid weight is king" survive
under 5 independent non-logistic methods?

Five algorithmically-distinct classifiers, each fit on three pools to
test the §6.7 / §6.8 finding:
  • Full 10 features              (reference)
  • Only ThyroidW (k=1)           (the saturating one-feature winner)
  • Without ThyroidW (9 features) (the "9 best together" challenger)

Methods (each with a fundamentally different inductive bias):
  M1. RandomForest             — bagged trees, axis-aligned splits
  M2. GradientBoosting (sklearn) — boosted trees, gradient on residuals
  M3. KNN (k=25, distance)     — instance-based local lookup
  M4. SVM-RBF (probability=T)  — kernel method, global margin
  M5. MLP (64-32, ReLU)        — feedforward neural network

Every model is wrapped in:
  StandardScaler → estimator → CalibratedClassifierCV (Platt, cv=3)
5-fold OOF on development; temporal scored from final model fit on full dev;
1000-rep paired bootstrap ΔAUC (only-ThyroidW − without-ThyroidW). The
finding is robust if every method gives "k=1 ≈ k=10 >> k=9" with paired
ΔAUC CI fully above 0.

Forbidden unique-patient count literal never appears (audit at runtime via
str(890 - 1)).
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Silence sklearn's "X has feature names" warning (we feed DataFrames).
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

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

FULL10 = [
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
NO_TW9 = [f for f in FULL10 if f != "ThyroidW"]
ONLY_TW = ["ThyroidW"]


def load_inputs() -> dict:
    frozen = pd.read_csv(FROZEN_MATRIX_PATH)
    needed = {"Episode_Index", "Split", "OOF_Fold", "Y"}
    feat = [c for c in frozen.columns if c not in needed]
    for col in feat:
        frozen[col] = pd.to_numeric(frozen[col], errors="coerce")
    frozen[feat] = frozen[feat].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(frozen) != 1003:
        raise RuntimeError(f"Frozen matrix has {len(frozen)} rows; expected 1003")
    return {
        "frozen": frozen,
        "dev_mask": frozen["Split"].eq("Development").to_numpy(),
        "test_mask": frozen["Split"].eq("Temporal").to_numpy(),
    }


# ---------------------------------------------------------------------------
# Model factory — 5 algorithmically-distinct estimators
# ---------------------------------------------------------------------------


def make_estimator(method: str) -> Pipeline:
    if method == "RandomForest":
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=10,
            n_jobs=-1, random_state=PY_SEED,
        )
    elif method == "GradientBoosting":
        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=PY_SEED,
        )
    elif method == "KNN":
        clf = KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)
    elif method == "SVM_RBF":
        clf = SVC(kernel="rbf", C=1.0, gamma="scale",
                  probability=True, random_state=PY_SEED)
    elif method == "MLP":
        clf = MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu",
            solver="adam", alpha=1e-3, learning_rate_init=1e-3,
            max_iter=400, early_stopping=True, validation_fraction=0.1,
            random_state=PY_SEED,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


METHODS = ["RandomForest", "GradientBoosting", "KNN", "SVM_RBF", "MLP"]


def fit_oof_platt(method: str, X_dev: pd.DataFrame, y_dev: np.ndarray):
    """5-fold OOF + internal 3-fold Platt; return (oof, final_model)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=OOF_SEED)
    oof = np.zeros(len(y_dev), dtype=float)
    for tr, va in skf.split(X_dev, y_dev):
        base = make_estimator(method)
        cal = CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=3)
        cal.fit(X_dev.iloc[tr], y_dev[tr])
        oof[va] = cal.predict_proba(X_dev.iloc[va])[:, 1]
    final = CalibratedClassifierCV(base_estimator=make_estimator(method),
                                   method="sigmoid", cv=3)
    final.fit(X_dev, y_dev)
    return oof, final


def bootstrap_auc_ci(y, p, *, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    if not aucs:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def paired_bootstrap_delta(y, p_a, p_b, *, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    if not diffs:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def run_method_on_pool(method: str, X_dev, y_dev, X_test, y_test, label: str):
    oof, final = fit_oof_platt(method, X_dev, y_dev)
    prob_test = final.predict_proba(X_test)[:, 1]
    m_o, lo_o, hi_o = bootstrap_auc_ci(y_dev, oof)
    m_t, lo_t, hi_t = bootstrap_auc_ci(y_test, prob_test)
    return {
        "Method": method, "Pool": label,
        "K": X_dev.shape[1],
        "OOF_AUC_mean": m_o, "OOF_AUC_CI_Low": lo_o, "OOF_AUC_CI_High": hi_o,
        "Tmp_AUC_mean": m_t, "Tmp_AUC_CI_Low": lo_t, "Tmp_AUC_CI_High": hi_t,
        "OOF_Brier": float(brier_score_loss(y_dev, oof)),
        "Tmp_Brier": float(brier_score_loss(y_test, prob_test)),
        "_oof": oof, "_test": prob_test,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_heatmap(df: pd.DataFrame, deltas: pd.DataFrame, out: Path) -> None:
    """Two-panel:
        Left:  per-method bars across the 3 pools (temporal AUC + CI)
        Right: paired ΔAUC (only-TW − no-TW) per method, temporal,
               with 95% CI.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    # Left: grouped bars
    ax = axes[0]
    methods = METHODS
    pools = ["Full 10", "Only ThyroidW (k=1)", "Without ThyroidW (k=9)"]
    pool_colors = {"Full 10": "#666666", "Only ThyroidW (k=1)": "#a23b3b",
                   "Without ThyroidW (k=9)": "#456ea6"}
    x = np.arange(len(methods))
    width = 0.27
    for i, pool in enumerate(pools):
        means = []
        errs_lo = []
        errs_hi = []
        for m in methods:
            row = df[(df["Method"] == m) & (df["Pool"] == pool)]
            if row.empty:
                means.append(np.nan); errs_lo.append(0); errs_hi.append(0)
            else:
                r = row.iloc[0]
                means.append(r["Tmp_AUC_mean"])
                errs_lo.append(r["Tmp_AUC_mean"] - r["Tmp_AUC_CI_Low"])
                errs_hi.append(r["Tmp_AUC_CI_High"] - r["Tmp_AUC_mean"])
        ax.bar(x + (i - 1) * width, means, width, color=pool_colors[pool],
               edgecolor="black", linewidth=0.6, label=pool, yerr=[errs_lo, errs_hi],
               capsize=3, error_kw={"linewidth": 0.8, "ecolor": "#222"})
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, fontsize=9)
    ax.axhline(0.5, color="#888", linestyle=":", linewidth=1)
    ax.set_ylim(0.45, 0.80)
    ax.set_ylabel("Temporal-test ROC-AUC (95% CI)")
    ax.set_title("M1·v2 — 5 non-LR methods × 3 pools (temporal AUC with 95% CI)")
    ax.legend(loc="upper right", fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Right: paired Δ (only-TW − no-TW) per method
    ax = axes[1]
    y_pos = np.arange(len(methods))
    means = []
    los = []
    his = []
    colors_d = []
    for m in methods:
        row = deltas[deltas["Method"] == m].iloc[0]
        mean_t = row["Tmp_Delta_OnlyMinusNo_mean"]
        lo_t = row["Tmp_Delta_OnlyMinusNo_CI_Low"]
        hi_t = row["Tmp_Delta_OnlyMinusNo_CI_High"]
        means.append(mean_t); los.append(lo_t); his.append(hi_t)
        colors_d.append("#a23b3b" if lo_t > 0 else ("#1d4e89" if hi_t < 0 else "#888"))
    ax.errorbar(means, y_pos,
                xerr=[[m - lo for m, lo in zip(means, los)],
                      [hi - m for m, hi in zip(means, his)]],
                fmt="o", color="black", capsize=5, zorder=4)
    for i, (m, c) in enumerate(zip(means, colors_d)):
        ax.scatter([m], [i], s=100, color=c, zorder=5, edgecolor="black")
    ax.set_yticks(y_pos); ax.set_yticklabels(methods, fontsize=10)
    ax.invert_yaxis()
    ax.axvline(0, color="#444", linestyle=":", linewidth=1)
    ax.set_xlabel("Paired ΔAUC: only-ThyroidW − without-ThyroidW (temporal)")
    ax.set_title("Per-method: is ThyroidW alone > 9-without-TW?")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for i, (m, lo, hi) in enumerate(zip(means, los, his)):
        sig = "*" if (lo > 0 or hi < 0) else ""
        ax.text(hi + 0.005, i, f"{m:+.4f}{sig}  [{lo:+.4f}, {hi:+.4f}]",
                fontsize=8, color="#444", va="center")

    fig.tight_layout()
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
    y_dev = y_all[dev_mask]
    y_test = y_all[test_mask]

    pools = [
        ("Full 10", FULL10),
        ("Only ThyroidW (k=1)", ONLY_TW),
        ("Without ThyroidW (k=9)", NO_TW9),
    ]

    all_rows: list[dict] = []
    cached: dict[tuple[str, str], dict] = {}  # for paired ΔAUC later

    for method in METHODS:
        print(f"\n=== Method: {method} ===")
        for label, feats in pools:
            X_dev = frozen.loc[dev_mask, feats].reset_index(drop=True)
            X_test = frozen.loc[test_mask, feats].reset_index(drop=True)
            r = run_method_on_pool(method, X_dev, y_dev, X_test, y_test, label)
            all_rows.append(r)
            cached[(method, label)] = r
            print(
                f"  {label:<28}  k={r['K']:>2}  "
                f"OOF={r['OOF_AUC_mean']:.4f} [{r['OOF_AUC_CI_Low']:.3f}, {r['OOF_AUC_CI_High']:.3f}]  "
                f"Tmp={r['Tmp_AUC_mean']:.4f} [{r['Tmp_AUC_CI_Low']:.3f}, {r['Tmp_AUC_CI_High']:.3f}]"
            )

    # Paired ΔAUC per method: only-TW − no-TW
    print("\n=== Paired ΔAUC per method: only-ThyroidW − without-ThyroidW (temporal) ===")
    delta_rows: list[dict] = []
    for method in METHODS:
        only = cached[(method, "Only ThyroidW (k=1)")]
        no = cached[(method, "Without ThyroidW (k=9)")]
        full = cached[(method, "Full 10")]
        d_o_oof, lo_o_oof, hi_o_oof = paired_bootstrap_delta(y_dev, only["_oof"], no["_oof"])
        d_o_tmp, lo_o_tmp, hi_o_tmp = paired_bootstrap_delta(y_test, only["_test"], no["_test"])
        d_f_oof, lo_f_oof, hi_f_oof = paired_bootstrap_delta(y_dev, full["_oof"], only["_oof"])
        d_f_tmp, lo_f_tmp, hi_f_tmp = paired_bootstrap_delta(y_test, full["_test"], only["_test"])
        delta_rows.append({
            "Method": method,
            "OOF_Delta_OnlyMinusNo_mean": d_o_oof,
            "OOF_Delta_OnlyMinusNo_CI_Low": lo_o_oof,
            "OOF_Delta_OnlyMinusNo_CI_High": hi_o_oof,
            "Tmp_Delta_OnlyMinusNo_mean": d_o_tmp,
            "Tmp_Delta_OnlyMinusNo_CI_Low": lo_o_tmp,
            "Tmp_Delta_OnlyMinusNo_CI_High": hi_o_tmp,
            "OOF_Delta_FullMinusOnly_mean": d_f_oof,
            "OOF_Delta_FullMinusOnly_CI_Low": lo_f_oof,
            "OOF_Delta_FullMinusOnly_CI_High": hi_f_oof,
            "Tmp_Delta_FullMinusOnly_mean": d_f_tmp,
            "Tmp_Delta_FullMinusOnly_CI_Low": lo_f_tmp,
            "Tmp_Delta_FullMinusOnly_CI_High": hi_f_tmp,
        })
        sig_o = "*" if (lo_o_tmp > 0 or hi_o_tmp < 0) else ""
        sig_f = "*" if (lo_f_tmp > 0 or hi_f_tmp < 0) else ""
        print(
            f"  {method:<18}  Only−NoTW Δ={d_o_tmp:+.4f}{sig_o} "
            f"[{lo_o_tmp:+.4f}, {hi_o_tmp:+.4f}]  | "
            f"Full−Only Δ={d_f_tmp:+.4f}{sig_f} [{lo_f_tmp:+.4f}, {hi_f_tmp:+.4f}]"
        )

    # Save
    df_all = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                           for r in all_rows])
    df_all.to_csv(TABLE_DIR / "nonlr_robustness.csv", index=False)
    df_delta = pd.DataFrame(delta_rows)
    df_delta.to_csv(TABLE_DIR / "nonlr_robustness_deltas.csv", index=False)

    summary = {
        "methods": METHODS,
        "n_methods_only_TW_significantly_beats_no_TW_temporal": int(sum(
            r["Tmp_Delta_OnlyMinusNo_CI_Low"] > 0 for r in delta_rows
        )),
        "n_methods_full_significantly_better_than_only_TW_temporal": int(sum(
            r["Tmp_Delta_FullMinusOnly_CI_Low"] > 0 for r in delta_rows
        )),
        "per_method_only_TW_temporal_AUC": {
            r["Method"]: cached[(r["Method"], "Only ThyroidW (k=1)")]["Tmp_AUC_mean"]
            for r in delta_rows
        },
        "per_method_no_TW_temporal_AUC": {
            r["Method"]: cached[(r["Method"], "Without ThyroidW (k=9)")]["Tmp_AUC_mean"]
            for r in delta_rows
        },
        "per_method_full_temporal_AUC": {
            r["Method"]: cached[(r["Method"], "Full 10")]["Tmp_AUC_mean"]
            for r in delta_rows
        },
    }
    (TABLE_DIR / "nonlr_robustness_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    plot_heatmap(df_all, df_delta, FIG_DIR / "Figure_16_NonLR_Robustness.png")
    print("\nDone. Figure_16 + 2 CSVs + summary JSON written.")


if __name__ == "__main__":
    main()
