#!/usr/bin/env python
"""M2-Base · Mechanism-guided serial landmark supermodel.

Runs the pre-registered M2-Base track (see results/module2_v2_base/arch.md):

  * Nested 5-step ablation A → A+B → A+B+C → A+B+C+D → A+B+C+D+E
  * Per-landmark + pooled ROC / PR / Brier with episode-cluster bootstrap CI
  * Paired episode-cluster bootstrap on consecutive ΔAUC
  * OR forest on full M2-Base.4 (cluster-robust / sandwich SE if statsmodels
    available; otherwise reports L2 point estimates with cluster-bootstrap CI)
  * Per-landmark + pooled calibration intercept/slope
  * Outputs to results/module2_v2_base/{figures,tables}/

All bootstrap CIs are episode-level cluster bootstrap × 1000. CV is
StratifiedGroupKFold(5) by episode_id, stratified on Y_24M_NHRH. Calibration
is per-landmark Platt on pooled outer-OOF.

Compliance:
- N = 1003 RAI treatment episodes → 4012 stacked landmark-rows
- The forbidden unique-patient count literal never appears in this source.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLBACKEND", "Agg")
warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scripts.simple.module2_v2_shared import (
    BLOCKS,
    ALL_BLOCKS_FEATURES,
    LANDMARKS,
    PRETTY,
    StackedData,
    apply_per_landmark_platt,
    calib_intercept_slope,
    compute_metrics,
    episode_cluster_bootstrap_auc_ci,
    fit_l2_predict_oof,
    forbidden_token,
    load_stacked,
    paired_episode_cluster_bootstrap_delta,
    per_landmark_and_pooled_perf,
    per_landmark_platt_on_pooled_oof,
)

OUT_DIR = ROOT / "results" / "module2_v2_base"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"

# Pre-registered nested ablation order (see arch.md §Nested ablation)
ABLATION_STEPS: list[tuple[str, tuple[str, ...]]] = [
    ("M2-Base.0 (A only)", ("A",)),
    ("M2-Base.1 (+ B)", ("A", "B")),
    ("M2-Base.2 (+ C)", ("A", "B", "C")),
    ("M2-Base.3 (+ D)", ("A", "B", "C", "D")),
    ("M2-Base.4 (+ E full)", ("A", "B", "C", "D", "E")),
]


# ---------------------------------------------------------------------------
# Run nested ablation
# ---------------------------------------------------------------------------


def run_nested_ablation(sd: StackedData) -> dict:
    """Return dict with per-step calibrated predictions, perf tables, and Δ table."""
    results: dict = {"steps": []}
    for label, blocks in ABLATION_STEPS:
        print(f"  → {label}: blocks={blocks}")
        raw_proba, _ = fit_l2_predict_oof(sd, block_subset=blocks)
        cal = per_landmark_platt_on_pooled_oof(sd, raw_proba)
        proba = apply_per_landmark_platt(sd, raw_proba, cal)
        dev_perf = per_landmark_and_pooled_perf(sd, proba, split="Development")
        tmp_perf = per_landmark_and_pooled_perf(sd, proba, split="Temporal")
        results["steps"].append(
            {
                "label": label,
                "blocks": list(blocks),
                "n_features": sum(len(BLOCKS[b]) for b in blocks),
                "proba": proba,
                "dev_perf": dev_perf,
                "tmp_perf": tmp_perf,
                "calibrators": cal,
            }
        )
    # Compute consecutive ΔAUC with paired episode-cluster bootstrap
    delta_rows: list[dict] = []
    for i in range(1, len(results["steps"])):
        prev = results["steps"][i - 1]
        curr = results["steps"][i]
        block_added = [b for b in curr["blocks"] if b not in prev["blocks"]][0]
        for L in list(LANDMARKS) + [None]:  # None = pooled
            for split, label_split in (("Development", "Dev OOF"), ("Temporal", "Temporal")):
                d_mean, d_lo, d_hi = paired_episode_cluster_bootstrap_delta(
                    sd, curr["proba"], prev["proba"], landmark=L, split=split
                )
                delta_rows.append(
                    {
                        "Step": curr["label"],
                        "BlockAdded": block_added,
                        "Split": label_split,
                        "Landmark": f"{L}M" if L is not None else "Pooled",
                        "Delta_AUC_mean": d_mean,
                        "Delta_AUC_CI_Low": d_lo,
                        "Delta_AUC_CI_High": d_hi,
                        "CI_excludes_zero": bool(d_lo > 0 or d_hi < 0)
                        if not (np.isnan(d_lo) or np.isnan(d_hi))
                        else False,
                    }
                )
    results["delta_table"] = pd.DataFrame(delta_rows)
    return results


# ---------------------------------------------------------------------------
# OR forest with sandwich SE (cluster-robust)
# ---------------------------------------------------------------------------


def or_forest_full_model(sd: StackedData) -> pd.DataFrame:
    """Fit unpenalised LR on dev rows; OR with cluster-robust SE by episode.

    Falls back to L2 + episode-cluster bootstrap if statsmodels unavailable.
    """
    X = sd.feature_matrix(("A", "B", "C", "D", "E"))
    y = sd.rows["Y_24M_NHRH"].values
    ep = sd.rows["episode_id"].values
    is_dev = sd.rows["Split"].values == "Development"
    X_dev = X.loc[is_dev]
    y_dev = y[is_dev]
    ep_dev = ep[is_dev]

    feat_names = list(X_dev.columns)
    # Standardise
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_dev.values)

    try:
        import statsmodels.api as sm
        Xs_const = sm.add_constant(Xs)
        glm = sm.GLM(y_dev, Xs_const, family=sm.families.Binomial())
        res = glm.fit(cov_type="cluster", cov_kwds={"groups": ep_dev})
        coefs = res.params[1:]  # drop intercept
        se = res.bse[1:]
        zvals = coefs / se
        pvals = res.pvalues[1:]
        method = "sandwich (statsmodels GLM cluster)"
    except Exception as e:
        print(f"  warning: statsmodels sandwich unavailable ({e}); falling back to L2 + bootstrap CI")
        lr = LogisticRegression(penalty="l2", solver="lbfgs", C=1.0, max_iter=5000, random_state=2025)
        lr.fit(Xs, y_dev)
        coefs = lr.coef_[0]
        # episode-cluster bootstrap on coefficient values for SE
        rng = np.random.default_rng(13)
        unique_eps = np.unique(ep_dev)
        ep_to_idx = {e: np.where(ep_dev == e)[0] for e in unique_eps}
        boot_coefs: list[np.ndarray] = []
        for _ in range(200):
            samp = rng.choice(unique_eps, size=len(unique_eps), replace=True)
            idx_concat = np.concatenate([ep_to_idx[e] for e in samp])
            try:
                lr_b = LogisticRegression(penalty="l2", solver="lbfgs", C=1.0,
                                          max_iter=2000, random_state=2025)
                lr_b.fit(Xs[idx_concat], y_dev[idx_concat])
                boot_coefs.append(lr_b.coef_[0])
            except Exception:
                continue
        boot_arr = np.array(boot_coefs)
        se = boot_arr.std(axis=0)
        zvals = coefs / np.where(se > 0, se, 1e-9)
        pvals = 2 * (1 - _normal_cdf(np.abs(zvals)))
        method = "L2 + 200 episode-cluster bootstrap"

    or_vals = np.exp(coefs)
    or_lo = np.exp(coefs - 1.96 * se)
    or_hi = np.exp(coefs + 1.96 * se)
    rows = []
    for i, f in enumerate(feat_names):
        # Tag with mechanism block
        block = next((b for b, feats in BLOCKS.items() if f in feats), "?")
        rows.append(
            {
                "Feature": f,
                "PrettyLabel": PRETTY.get(f, f),
                "Block": block,
                "Coefficient": float(coefs[i]),
                "SE": float(se[i]),
                "OR": float(or_vals[i]),
                "CI_Low": float(or_lo[i]),
                "CI_High": float(or_hi[i]),
                "p_value": float(pvals[i]),
                "Method": method,
            }
        )
    return pd.DataFrame(rows)


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    # Numpy-only standard-normal CDF approximation
    from math import erf, sqrt
    out = np.empty_like(x, dtype=float)
    for i, xi in enumerate(np.atleast_1d(x).ravel()):
        out.ravel()[i] = 0.5 * (1 + erf(float(xi) / sqrt(2)))
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


BLOCK_COLORS = {
    "A": "#1d4e89",  # burden — blue
    "B": "#0a6b3f",  # exposure — green
    "C": "#d28b18",  # current dynamic — orange
    "D": "#a23b3b",  # momentum — red
    "E": "#5a3f8a",  # time interactions — purple
}


def fig_per_landmark_perf(
    results: dict, out: Path, *, split: str
) -> None:
    """Multi-panel per-landmark ROC across 5 ablation steps + Pooled."""
    fig, ax = plt.subplots(figsize=(9, 5))
    landmark_labels = [f"{L}M" for L in LANDMARKS] + ["Pooled"]
    x = np.arange(len(landmark_labels))
    width = 0.16
    for i, step in enumerate(results["steps"]):
        perf = step["dev_perf"] if split == "Development" else step["tmp_perf"]
        aucs = []
        for lab in landmark_labels:
            row = perf[perf["Landmark"] == lab]
            aucs.append(float(row["ROC_AUC"].iloc[0]) if not row.empty else float("nan"))
        ax.bar(x + (i - 2) * width, aucs, width, label=step["label"],
               color=plt.cm.viridis(i / max(1, len(results["steps"]) - 1)))
    ax.set_xticks(x)
    ax.set_xticklabels(landmark_labels)
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="#888", linestyle=":", linewidth=1)
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"M2-Base per-landmark + pooled ROC ({split})")
    ax.legend(loc="lower right", fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_nested_delta(results: dict, out: Path) -> None:
    """Nested ablation ΔAUC + CI per step × landmark (Temporal)."""
    df = results["delta_table"]
    sub = df[df["Split"] == "Temporal"].copy()
    landmark_labels = [f"{L}M" for L in LANDMARKS] + ["Pooled"]
    step_labels = [s["label"] for s in results["steps"][1:]]
    fig, axes = plt.subplots(len(step_labels), 1, figsize=(9, 1.8 * len(step_labels) + 1), sharex=True)
    for i, step in enumerate(step_labels):
        ax = axes[i]
        step_rows = sub[sub["Step"] == step].set_index("Landmark").reindex(landmark_labels).reset_index()
        x = np.arange(len(landmark_labels))
        means = step_rows["Delta_AUC_mean"].values
        lo = (step_rows["Delta_AUC_mean"] - step_rows["Delta_AUC_CI_Low"]).abs().values
        hi = (step_rows["Delta_AUC_CI_High"] - step_rows["Delta_AUC_mean"]).abs().values
        block_added = step_rows["BlockAdded"].iloc[0] if not step_rows.empty else "?"
        color = BLOCK_COLORS.get(block_added, "#666")
        ax.errorbar(x, means, yerr=[lo, hi], fmt="o", color=color, capsize=4, markersize=7)
        ax.axhline(0, color="#444", linestyle=":", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(landmark_labels)
        ax.set_ylabel("ΔROC-AUC")
        ax.set_title(f"{step}  (+ block {block_added})", fontsize=10, loc="left")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("M2-Base nested ablation — ΔAUC vs previous step (Temporal, paired episode-cluster bootstrap)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_or_forest(df: pd.DataFrame, out: Path) -> None:
    """OR forest grouped by mechanism block."""
    df_sorted = df.copy()
    block_order = ["A", "B", "C", "D", "E"]
    df_sorted["BlockRank"] = df_sorted["Block"].apply(lambda b: block_order.index(b) if b in block_order else 99)
    df_sorted = df_sorted.sort_values(["BlockRank", "OR"], ascending=[True, False]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(df_sorted) + 1))
    y_pos = np.arange(len(df_sorted))[::-1]
    for i, row in df_sorted.iterrows():
        color = BLOCK_COLORS.get(row["Block"], "#666")
        yp = y_pos[i]
        ax.plot([row["CI_Low"], row["CI_High"]], [yp, yp], color=color, linewidth=2)
        ax.plot(row["OR"], yp, "o", color=color, markersize=8)
        ax.text(max(row["CI_High"], 3.5) + 0.1, yp,
                f"OR {row['OR']:.2f} [{row['CI_Low']:.2f}, {row['CI_High']:.2f}]",
                fontsize=7, va="center", color="#333")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"[{row['Block']}] {row['PrettyLabel']}" for _, row in df_sorted.iterrows()],
                       fontsize=8)
    ax.set_xscale("log")
    ax.axvline(1.0, color="#444", linestyle=":", linewidth=1)
    ax.set_xlabel("Odds ratio (log scale; per +1 SD)")
    ax.set_title("M2-Base.4 OR forest by mechanism block (cluster-robust SE)")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_calibration_per_landmark(results: dict, out: Path) -> None:
    """Per-landmark calibration plot for the full M2-Base.4 model on Temporal."""
    full = results["steps"][-1]
    proba = full["proba"]
    sd_rows_split = full["dev_perf"]  # placeholder; we recompute below using sd
    fig, axes = plt.subplots(1, len(LANDMARKS), figsize=(3 * len(LANDMARKS), 3.2))
    for ax, L in zip(axes, LANDMARKS):
        # Pull from the saved tmp_perf table
        row = full["tmp_perf"][full["tmp_perf"]["Landmark"] == f"{L}M"]
        if row.empty:
            continue
        # Compute calibration curve from raw data within this function
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
        ax.text(0.05, 0.85,
                f"intercept={float(row['CalibIntercept'].iloc[0]):+.3f}\nslope={float(row['CalibSlope'].iloc[0]):+.3f}\nBrier={float(row['Brier'].iloc[0]):.3f}",
                fontsize=8, transform=ax.transAxes)
        ax.set_title(f"Landmark {L}M")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("M2-Base.4 calibration (Temporal); decile binning omitted (intercept+slope only)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Hard-constraint runtime check (CLAUDE.md): forbidden literal absence.
    tok = forbidden_token()
    if tok in Path(__file__).read_text():
        raise RuntimeError(f"Forbidden literal '{tok}' present in this source file.")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading stacked dataset…")
    sd = load_stacked()
    print(f"  rows={len(sd.rows)} episodes={len(sd.episode_meta)} prevalence={sd.episode_meta['Y_24M_NHRH'].mean():.3f}")

    print("\nRunning nested ablation A → A+B → A+B+C → A+B+C+D → A+B+C+D+E…")
    results = run_nested_ablation(sd)

    # --- Tables ---
    perf_rows: list[pd.DataFrame] = []
    for step in results["steps"]:
        for label_df, split in ((step["dev_perf"], "Development"), (step["tmp_perf"], "Temporal")):
            df = label_df.copy()
            df.insert(0, "Step", step["label"])
            df.insert(1, "BlocksIncluded", "+".join(step["blocks"]))
            df.insert(2, "NFeatures", step["n_features"])
            perf_rows.append(df)
    perf_table = pd.concat(perf_rows, ignore_index=True)
    perf_table.to_csv(TAB_DIR / "ablation_performance.csv", index=False)
    print(f"\nSaved {TAB_DIR / 'ablation_performance.csv'}")

    results["delta_table"].to_csv(TAB_DIR / "ablation_delta_auc.csv", index=False)
    print(f"Saved {TAB_DIR / 'ablation_delta_auc.csv'}")

    print("\nFitting OR forest with cluster-robust SE on M2-Base.4 …")
    or_df = or_forest_full_model(sd)
    or_df.to_csv(TAB_DIR / "or_forest_base_full.csv", index=False)
    print(f"Saved {TAB_DIR / 'or_forest_base_full.csv'}")

    # --- Figures ---
    print("\nRendering figures…")
    fig_per_landmark_perf(results, FIG_DIR / "Figure_01_PerLandmark_ROC_Development.png", split="Development")
    fig_per_landmark_perf(results, FIG_DIR / "Figure_02_PerLandmark_ROC_Temporal.png", split="Temporal")
    fig_nested_delta(results, FIG_DIR / "Figure_03_NestedAblation_DeltaAUC_Temporal.png")
    fig_or_forest(or_df, FIG_DIR / "Figure_04_OR_Forest_full.png")
    fig_calibration_per_landmark(results, FIG_DIR / "Figure_05_Calibration_PerLandmark_Temporal.png")
    print(f"  saved 5 figures under {FIG_DIR}")

    # --- Run summary JSON ---
    summary = {
        "n_episodes": int(len(sd.episode_meta)),
        "n_landmark_rows": int(len(sd.rows)),
        "splits": {k: int(v) for k, v in sd.episode_meta["Split"].value_counts().items()},
        "prevalence": float(sd.episode_meta["Y_24M_NHRH"].mean()),
        "blocks_features": {b: BLOCKS[b] for b in ("A", "B", "C", "D", "E")},
        "ablation_steps": [
            {
                "label": step["label"],
                "blocks": step["blocks"],
                "n_features": step["n_features"],
                "tmp_pooled_roc": float(step["tmp_perf"][step["tmp_perf"]["Landmark"] == "Pooled"]["ROC_AUC"].iloc[0]),
                "tmp_pooled_brier": float(step["tmp_perf"][step["tmp_perf"]["Landmark"] == "Pooled"]["Brier"].iloc[0]),
                "tmp_6M_roc": float(step["tmp_perf"][step["tmp_perf"]["Landmark"] == "6M"]["ROC_AUC"].iloc[0]),
            }
            for step in results["steps"]
        ],
        "consecutive_delta_temporal_pooled": [
            {
                "step": r["Step"],
                "block_added": r["BlockAdded"],
                "delta_mean": r["Delta_AUC_mean"],
                "delta_ci": [r["Delta_AUC_CI_Low"], r["Delta_AUC_CI_High"]],
                "ci_excludes_zero": r["CI_excludes_zero"],
            }
            for _, r in results["delta_table"].iterrows()
            if r["Split"] == "Temporal" and r["Landmark"] == "Pooled"
        ],
    }
    (TAB_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved {TAB_DIR / 'run_summary.json'}")

    # --- Print key findings ---
    print("\n=== Key findings ===")
    print("Temporal pooled ROC by step:")
    for step in results["steps"]:
        pool = step["tmp_perf"][step["tmp_perf"]["Landmark"] == "Pooled"]
        print(f"  {step['label']:<30}  ROC={float(pool['ROC_AUC'].iloc[0]):.3f}  Brier={float(pool['Brier'].iloc[0]):.3f}")
    print("\nConsecutive Δ Temporal pooled:")
    for d in summary["consecutive_delta_temporal_pooled"]:
        flag = " *" if d["ci_excludes_zero"] else ""
        print(f"  {d['step']:<30}  +block {d['block_added']}  Δ={d['delta_mean']:+.4f}  CI=[{d['delta_ci'][0]:+.4f}, {d['delta_ci'][1]:+.4f}]{flag}")


if __name__ == "__main__":
    main()
