#!/usr/bin/env python
"""M1·v2 — mirror experiment: how far can the 9 features WITHOUT
Thyroid weight take us?

Three variants on the "Thyroid weight is forbidden" pool of 9 features:
  (a) All 9 together (L2 + Platt + 5-fold OOF, bootstrap CI).
  (b) Greedy backward elimination 9 → 1 (same LOO ΔAUC heuristic).
  (c) Pre-specified clinical subsets (Top-2/Top-3/Top-5 by prior SHAP
      rank in the augmented pool, excluding ThyroidW).

Compared head-to-head against:
  • only-ThyroidW (k=1), the saturating "one feature is enough" winner.
  • Full 10-feature model.

Goal: quantify how big the gap is between "everyone except ThyroidW" and
"only ThyroidW" — i.e., is gland weight really irreplaceable, or could 9
weaker features compensate when combined?

Forbidden unique-patient count literal never appears in this source
(audit at runtime via str(890 - 1)).
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
NINE_NO_TW = [f for f in FULL10 if f != "ThyroidW"]  # 9 features without ThyroidW

PRETTY = {
    "Sex": "Sex",
    "ThyroidW": "Thyroid weight",
    "Uptake24h": "24h RAI uptake",
    "HalfLife": "Effective iodine half-life",
    "TRAb": "TRAb",
    "TGAb": "TgAb",
    "TPOAb": "TPOAb",
    "FT4_0M": "FT4 (baseline)",
    "TSH_0M": "TSH (baseline)",
    "log1p_DiseaseDuration_Months_Aug": "Log disease duration",
}


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
    """Paired bootstrap on (AUC_a − AUC_b)."""
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


def score_subset(X_dev, y_dev, X_test, y_test, features, label):
    if not features:
        # Trivial "no features" → predict prevalence
        prev = y_dev.mean()
        oof = np.full_like(y_dev, prev, dtype=float)
        prob_test = np.full_like(y_test, prev, dtype=float)
    else:
        oof, final = fit_oof_platt(X_dev[features], y_dev)
        prob_test = final.predict_proba(X_test[features])[:, 1]
    m_oof, lo_oof, hi_oof = bootstrap_auc_ci(y_dev, oof)
    m_tmp, lo_tmp, hi_tmp = bootstrap_auc_ci(y_test, prob_test)
    return {
        "Label": label,
        "K": len(features),
        "Features": ";".join(features) if features else "(none)",
        "OOF_AUC_mean": m_oof, "OOF_AUC_CI_Low": lo_oof, "OOF_AUC_CI_High": hi_oof,
        "Tmp_AUC_mean": m_tmp, "Tmp_AUC_CI_Low": lo_tmp, "Tmp_AUC_CI_High": hi_tmp,
        "OOF_Brier": float(brier_score_loss(y_dev, oof)),
        "Tmp_Brier": float(brier_score_loss(y_test, prob_test)),
        "_oof": oof,
        "_test": prob_test,
    }


def loo_delta_oof_auc(X_dev, y_dev, features):
    oof_full, _ = fit_oof_platt(X_dev[list(features)], y_dev)
    auc_full = roc_auc_score(y_dev, oof_full)
    out = {}
    for f in features:
        rest = [g for g in features if g != f]
        if not rest:
            out[f] = 0.0
            continue
        oof_r, _ = fit_oof_platt(X_dev[rest], y_dev)
        out[f] = auc_full - roc_auc_score(y_dev, oof_r)
    return out


def greedy_backward(X_dev, y_dev, X_test, y_test, full_features):
    rows = []
    current = list(full_features)
    while len(current) >= 1:
        k = len(current)
        oof, final = fit_oof_platt(X_dev[current], y_dev)
        prob_test = final.predict_proba(X_test[current])[:, 1]
        m_oof, lo_oof, hi_oof = bootstrap_auc_ci(y_dev, oof)
        m_tmp, lo_tmp, hi_tmp = bootstrap_auc_ci(y_test, prob_test)
        rows.append(
            {
                "K": k,
                "Features": ";".join(current),
                "OOF_AUC_mean": m_oof, "OOF_AUC_CI_Low": lo_oof, "OOF_AUC_CI_High": hi_oof,
                "Tmp_AUC_mean": m_tmp, "Tmp_AUC_CI_Low": lo_tmp, "Tmp_AUC_CI_High": hi_tmp,
            }
        )
        print(
            f"  k={k:>2}  OOF={m_oof:.4f} [{lo_oof:.3f}, {hi_oof:.3f}]  "
            f"Tmp={m_tmp:.4f} [{lo_tmp:.3f}, {hi_tmp:.3f}]"
        )
        if k == 1:
            break
        loos = loo_delta_oof_auc(X_dev, y_dev, current)
        weakest = min(loos, key=loos.get)
        print(f"    → dropping {weakest!r} (ΔAUC={loos[weakest]:+.4f})")
        current.remove(weakest)
    return pd.DataFrame(rows)


def plot_compare(records, df_path, out):
    """Bar plot: temporal AUC + 95% CI for several subsets + the greedy
    path overlay.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: discrete subsets vs each other
    ax = axes[0]
    labels = [r["Label"] for r in records]
    means = [r["Tmp_AUC_mean"] for r in records]
    los = [r["Tmp_AUC_mean"] - r["Tmp_AUC_CI_Low"] for r in records]
    his = [r["Tmp_AUC_CI_High"] - r["Tmp_AUC_mean"] for r in records]
    colors = []
    for r in records:
        if "only ThyroidW" in r["Label"].lower() or "only thyroid" in r["Label"].lower():
            colors.append("#a23b3b")  # red — winning baseline
        elif "without thyroid" in r["Label"].lower() or "no thyroid" in r["Label"].lower():
            colors.append("#456ea6")  # blue — challenger
        elif "full 10" in r["Label"].lower():
            colors.append("#666666")  # grey — reference
        else:
            colors.append("#888888")
    y_pos = np.arange(len(labels))
    ax.errorbar(means, y_pos, xerr=[los, his], fmt="o", color="black", capsize=5, zorder=4)
    for i, (m, c) in enumerate(zip(means, colors)):
        ax.scatter([m], [i], s=120, color=c, zorder=5, edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0.5, color="#888", linestyle=":", linewidth=1)
    ax.set_xlim(0.45, 0.85)
    ax.set_xlabel("Temporal-test ROC-AUC (95% CI, bootstrap × 1000)")
    ax.set_title("M1·v2 mirror test: with vs without Thyroid weight")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for i, (m, lo, hi) in enumerate(zip(means, [r["Tmp_AUC_CI_Low"] for r in records],
                                       [r["Tmp_AUC_CI_High"] for r in records])):
        ax.text(hi + 0.005, i, f"{m:.3f}  [{lo:.3f}, {hi:.3f}]",
                fontsize=7, color="#444", va="center")

    # Right: greedy path overlay — "without ThyroidW" pool path 9→1
    ax = axes[1]
    k = df_path["K"].to_numpy()
    ax.fill_between(k, df_path["Tmp_AUC_CI_Low"], df_path["Tmp_AUC_CI_High"],
                    color="#456ea6", alpha=0.18, label="Temporal 95% CI")
    ax.plot(k, df_path["Tmp_AUC_mean"], "o-", color="#1d4e89", lw=2, ms=6,
            label="Temporal AUC (no ThyroidW pool)")
    ax.fill_between(k, df_path["OOF_AUC_CI_Low"], df_path["OOF_AUC_CI_High"],
                    color="#d28b18", alpha=0.15, label="Dev OOF 95% CI")
    ax.plot(k, df_path["OOF_AUC_mean"], "s-", color="#a87317", lw=2, ms=6,
            label="Dev OOF AUC (no ThyroidW pool)")
    # Horizontal reference: only-ThyroidW temporal
    ax.axhline(0.6825, color="#a23b3b", linestyle="--", linewidth=1.5,
               label="Only ThyroidW (k=1) Temporal = 0.683")
    ax.axhline(0.5, color="#888", linestyle=":", linewidth=1)
    ax.set_xticks(range(1, max(k) + 1))
    ax.set_xlim(0.5, max(k) + 0.5)
    ax.set_ylim(0.45, 0.85)
    ax.set_xlabel("Number of features k (from 9-pool, no ThyroidW)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Greedy backward elimination on the 9 features without ThyroidW")
    ax.legend(loc="lower right", fontsize=7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


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
    X_all = frozen[FULL10].copy()
    X_dev = X_all.loc[dev_mask].reset_index(drop=True)
    y_dev = y_all[dev_mask]
    X_test = X_all.loc[test_mask].reset_index(drop=True)
    y_test = y_all[test_mask]

    # Subset evaluations
    print("=== Head-to-head subsets ===")
    records = []
    records.append(score_subset(X_dev, y_dev, X_test, y_test, FULL10,
                                "Full 10 features (reference)"))
    records.append(score_subset(X_dev, y_dev, X_test, y_test, ["ThyroidW"],
                                "Only ThyroidW (k=1)"))
    records.append(score_subset(X_dev, y_dev, X_test, y_test, NINE_NO_TW,
                                f"Without ThyroidW — all 9 (k={len(NINE_NO_TW)})"))
    # Top-N without ThyroidW, by mean|SHAP| in augmented 10 pool (excluding ThyroidW):
    #   1. log1p_DiseaseDuration_Months_Aug   (0.196)
    #   2. TPOAb                               (0.183)
    #   3. Sex                                 (0.118)
    #   4. TRAb                                (0.075)
    #   5. FT4_0M                              (0.070)
    top2_no_tw = ["log1p_DiseaseDuration_Months_Aug", "TPOAb"]
    top3_no_tw = top2_no_tw + ["Sex"]
    top5_no_tw = top3_no_tw + ["TRAb", "FT4_0M"]
    records.append(score_subset(X_dev, y_dev, X_test, y_test, top2_no_tw,
                                "Top-2 without ThyroidW (LogDur+TPOAb)"))
    records.append(score_subset(X_dev, y_dev, X_test, y_test, top3_no_tw,
                                "Top-3 without ThyroidW (+Sex)"))
    records.append(score_subset(X_dev, y_dev, X_test, y_test, top5_no_tw,
                                "Top-5 without ThyroidW (+TRAb+FT4)"))

    for r in records:
        print(
            f"  {r['Label']:<48}  k={r['K']:>2}  "
            f"OOF={r['OOF_AUC_mean']:.4f} [{r['OOF_AUC_CI_Low']:.3f}, {r['OOF_AUC_CI_High']:.3f}]  "
            f"Tmp={r['Tmp_AUC_mean']:.4f} [{r['Tmp_AUC_CI_Low']:.3f}, {r['Tmp_AUC_CI_High']:.3f}]"
        )

    # Paired ΔAUC: ThyroidW alone − all-9-without-ThyroidW (Temporal)
    print("\n=== Paired ΔAUC: only-ThyroidW vs without-ThyroidW (9 features) ===")
    only_tw = next(r for r in records if r["Label"].startswith("Only ThyroidW"))
    no_tw_9 = next(r for r in records if "without ThyroidW — all 9" in r["Label"].lower() or "Without ThyroidW — all 9" in r["Label"])
    d_oof_m, d_oof_lo, d_oof_hi = paired_bootstrap_delta(y_dev, only_tw["_oof"], no_tw_9["_oof"])
    d_tmp_m, d_tmp_lo, d_tmp_hi = paired_bootstrap_delta(y_test, only_tw["_test"], no_tw_9["_test"])
    print(
        f"  Dev OOF Δ = {d_oof_m:+.4f} [{d_oof_lo:+.4f}, {d_oof_hi:+.4f}] "
        f"({'sig' if d_oof_lo > 0 else 'CI crosses 0'})"
    )
    print(
        f"  Temporal Δ = {d_tmp_m:+.4f} [{d_tmp_lo:+.4f}, {d_tmp_hi:+.4f}] "
        f"({'sig' if d_tmp_lo > 0 else 'CI crosses 0'})"
    )

    # Greedy backward on the no-ThyroidW pool (9 → 1)
    print("\n=== Greedy backward on 9 features without ThyroidW (9 → 1) ===")
    df_path = greedy_backward(X_dev, y_dev, X_test, y_test, NINE_NO_TW)
    df_path.to_csv(TABLE_DIR / "no_thyroidw_path.csv", index=False)

    # Save records (strip internal arrays for CSV)
    df_records = pd.DataFrame(
        [{k: v for k, v in r.items() if not k.startswith("_")} for r in records]
    )
    df_records.to_csv(TABLE_DIR / "no_thyroidw_subsets.csv", index=False)

    summary = {
        "only_thyroidw_temporal_AUC": only_tw["Tmp_AUC_mean"],
        "no_thyroidw_all9_temporal_AUC": no_tw_9["Tmp_AUC_mean"],
        "delta_temporal_only_minus_no": d_tmp_m,
        "delta_temporal_CI": [d_tmp_lo, d_tmp_hi],
        "delta_temporal_CI_above_zero": bool(d_tmp_lo > 0),
        "no_thyroidw_top5_temporal_AUC": next(
            r["Tmp_AUC_mean"] for r in records if r["Label"].startswith("Top-5")
        ),
    }
    (TABLE_DIR / "no_thyroidw_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    # Plot
    plot_compare(records, df_path, FIG_DIR / "Figure_14_No_ThyroidW.png")
    print("\nDone. Figure_14 + 2 CSVs + summary JSON written.")


if __name__ == "__main__":
    main()
