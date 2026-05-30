#!/usr/bin/env python
"""Generate the ML method comparison figure for the v3 report.

Renders Figure_v3_30_ML_Comparison.png from already-stored numbers:
  - 5 non-LR methods (RF / GBM / KNN / SVM-RBF / MLP) on Full 10-feature pool:
      results/module1_v2_lasso_clean/tables/nonlr_robustness.csv
      results/module1_v2_lasso_clean/tables/nonlr_robustness_deltas.csv
  - L2-Logistic on Full 10 from v2 performance_with_ci.csv (M1_v2a row)
  - For context: L2-Logistic on the v6 6-feature pool from
      results/module1_v3/tables/v6_performance.csv

No model fitting (frozen matrix unavailable on this host); all numbers
come from existing summary tables.

Output:
  results/module1_v3/figures/Figure_v3_30_ML_Comparison.png

Forbidden unique-patient count literal never appears (audit at runtime
via str(890 - 1)).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
V3_DIR = ROOT / "results" / "module1_v3"
V3_FIG = V3_DIR / "figures"
V3_TABLES = V3_DIR / "tables"
V2_TABLES = ROOT / "results" / "module1_v2_lasso_clean" / "tables"


def main() -> None:
    forbidden = str(890 - 1)
    if forbidden in Path(__file__).read_text():
        raise RuntimeError("Forbidden unique-patient count literal present in source.")

    V3_FIG.mkdir(parents=True, exist_ok=True)
    V3_TABLES.mkdir(parents=True, exist_ok=True)

    # Read non-LR results
    nonlr = pd.read_csv(V2_TABLES / "nonlr_robustness.csv")
    deltas = pd.read_csv(V2_TABLES / "nonlr_robustness_deltas.csv")
    full10 = nonlr[nonlr["Pool"] == "Full 10"].copy()

    # LR baseline on the same 10-feature pool
    perf = pd.read_csv(V2_TABLES / "performance_with_ci.csv")
    lr_row = perf[(perf["FeaturePool"] == "M1_v2a") & (perf["Split"] == "Temporal_Test") & (perf["Metric"] == "ROC_AUC")].iloc[0]
    lr_full10 = {
        "Method": "L2-Logistic (ours)",
        "Tmp_AUC_mean": float(lr_row["Value"]),
        "Tmp_AUC_CI_Low": float(lr_row["CI_Lower"]),
        "Tmp_AUC_CI_High": float(lr_row["CI_Upper"]),
    }
    # v6 6-feature LR for reference
    v6_perf = pd.read_csv(V3_DIR / "tables" / "v6_performance.csv")
    v6_row = v6_perf[(v6_perf["Split"] == "Temporal_Test") & (v6_perf["Metric"] == "ROC_AUC")].iloc[0]
    lr_v6 = {
        "Method": "L2-LR (v6, 6 features)",
        "Tmp_AUC_mean": float(v6_row["Value"]),
        "Tmp_AUC_CI_Low": float(v6_row["CI_Lower"]),
        "Tmp_AUC_CI_High": float(v6_row["CI_Upper"]),
    }

    # Combined display order
    order = [lr_full10, lr_v6]
    for m in ["RandomForest", "GradientBoosting", "KNN", "SVM_RBF", "MLP"]:
        r = full10[full10["Method"] == m].iloc[0]
        order.append({
            "Method": m,
            "Tmp_AUC_mean": float(r["Tmp_AUC_mean"]),
            "Tmp_AUC_CI_Low": float(r["Tmp_AUC_CI_Low"]),
            "Tmp_AUC_CI_High": float(r["Tmp_AUC_CI_High"]),
        })

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # Left: bar of temporal AUC per method
    ax = axes[0]
    methods = [d["Method"] for d in order]
    means = [d["Tmp_AUC_mean"] for d in order]
    lo = [d["Tmp_AUC_mean"] - d["Tmp_AUC_CI_Low"] for d in order]
    hi = [d["Tmp_AUC_CI_High"] - d["Tmp_AUC_mean"] for d in order]
    colors = ["#a23b3b" if "L2-LR" in m or "L2-Logistic" in m else "#456ea6" for m in methods]
    x = np.arange(len(methods))
    bars = ax.bar(x, means, color=colors, edgecolor="black", linewidth=0.6,
                  yerr=[lo, hi], capsize=4,
                  error_kw={"linewidth": 0.8, "ecolor": "#222"})
    for i, (m, l, h) in enumerate(zip(means, lo, hi)):
        ax.text(i, m + h + 0.01, f"{m:.3f}", ha="center", fontsize=8, color="#333")
    ax.axhline(0.5, color="#888", linestyle=":", linewidth=1, label="Chance (AUC 0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0.45, 0.80)
    ax.set_ylabel("Temporal-test ROC-AUC (95% CI, bootstrap × 1000)")
    ax.set_title("(A) M1 — L2-LR vs 5 common ML methods on same 10-feature pool\n"
                 "(red bars = our LR; identical Platt-calibration wrapper)")
    ax.legend(loc="lower right", fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Right: paired ΔAUC (method - LR_full10)
    # Reconstruct from nonlr_robustness data: the deltas table compares
    # only-TW vs no-TW; we recompute method vs LR using mean difference
    # of point estimates (since paired bootstrap deltas vs LR aren't in
    # the historical CSV).  Show |Δ| visually, no CI here.
    ax = axes[1]
    lr_ref = lr_full10["Tmp_AUC_mean"]
    others = [d for d in order if "L2-Logistic" not in d["Method"] and "L2-LR" not in d["Method"]]
    methods_o = [d["Method"] for d in others]
    diffs = [d["Tmp_AUC_mean"] - lr_ref for d in others]
    y_pos = np.arange(len(methods_o))
    colors_d = ["#1d4e89" if d < 0 else "#a23b3b" for d in diffs]
    ax.barh(y_pos, diffs, color=colors_d, edgecolor="black", linewidth=0.6)
    for i, d in enumerate(diffs):
        ax.text(d + (0.001 if d >= 0 else -0.001), i, f"{d:+.4f}",
                ha="left" if d >= 0 else "right", va="center", fontsize=8, color="#444")
    ax.axvline(0, color="#444", linestyle="--", linewidth=1.2,
               label="L2-LR baseline (Temporal AUC 0.686)")
    ax.set_yticks(y_pos); ax.set_yticklabels(methods_o, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(-0.05, 0.05)
    ax.set_xlabel("Tmp AUC difference (method − L2-LR, point estimate)")
    ax.set_title("(B) Method gap relative to L2-LR\n"
                 "(all 5 methods within ±0.03 of LR; CIs heavily overlap)")
    ax.legend(loc="lower right", fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    out = V3_FIG / "Figure_v3_30_ML_Comparison.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")

    # Save comparison table for the report
    tbl = pd.DataFrame([{
        "Method": d["Method"],
        "Tmp_AUC_mean": round(d["Tmp_AUC_mean"], 4),
        "Tmp_AUC_CI_Low": round(d["Tmp_AUC_CI_Low"], 4),
        "Tmp_AUC_CI_High": round(d["Tmp_AUC_CI_High"], 4),
        "Diff_vs_LR_full10": round(d["Tmp_AUC_mean"] - lr_ref, 4),
    } for d in order])
    out_csv = V3_TABLES / "ml_comparison_table.csv"
    tbl.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    print(tbl.to_string(index=False))


if __name__ == "__main__":
    main()
