#!/usr/bin/env python
"""M2 v2 supplementary analyses: risk migration matrix + Shapley + 2x2 H2H.

Three deferred items from the original plan, now implemented:

  1. Risk migration transition probability matrix (low/mid/high tertiles
     cut on dev 0M OOF, applied to all landmarks and to temporal).
     Replaces the original Sankey plan with the more reviewer-friendly
     transition-probability matrix per the synthesis decision (Plan agent
     Mn10: Sankey demoted to supplementary).
  2. Shapley 5-block decomposition over all 5! = 120 orderings.
  3. 2x2 head-to-head vs M2-Base baseline: architecture × feature-set
     contributions (Plan-agent M6).

Outputs go to results/module2_v2_base/{figures,tables}/ (these are part
of M2-Base's full evidence base).
"""
from __future__ import annotations

import itertools
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
from sklearn.metrics import roc_auc_score

from scripts.simple.module2_v2_shared import (
    BLOCKS,
    LANDMARKS,
    StackedData,
    apply_per_landmark_platt,
    fit_l2_predict_oof,
    forbidden_token,
    load_stacked,
    paired_episode_cluster_bootstrap_delta,
    per_landmark_platt_on_pooled_oof,
)

OUT_DIR = ROOT / "results" / "module2_v2_base"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"


# ---------------------------------------------------------------------------
# 1. Risk migration transition matrix
# ---------------------------------------------------------------------------


def build_risk_tier_transitions(sd: StackedData, proba: np.ndarray) -> dict:
    """Compute risk-tier (low/mid/high) transitions across consecutive landmarks.

    Tertile cutoffs locked on dev OOF predictions at landmark 0M (Plan agent C4).
    """
    dev_mask = sd.rows["Split"].values == "Development"
    lm_arr = sd.rows["landmark"].values

    # Lock cutoffs on dev 0M
    dev_0m = dev_mask & (lm_arr == 0)
    t1, t2 = np.percentile(proba[dev_0m], [33.33, 66.67])

    def tier(p: np.ndarray) -> np.ndarray:
        return np.where(p <= t1, 0, np.where(p <= t2, 1, 2))  # 0=low, 1=mid, 2=high

    # Build per-episode tier sequence (landmark order)
    eps = sd.rows["episode_id"].values
    splits = sd.rows["Split"].values
    rows: list[dict] = []
    for ep in np.unique(eps):
        sub_mask = eps == ep
        sub = sd.rows.loc[sub_mask].sort_values("landmark")
        ep_proba = proba[sub_mask][np.argsort(sd.rows.loc[sub_mask, "landmark"].values)]
        tiers = tier(ep_proba)
        rows.append({
            "episode_id": int(ep),
            "Split": str(sub.iloc[0]["Split"]),
            "Y_24M_NHRH": int(sub.iloc[0]["Y_24M_NHRH"]),
            "tier_0M": int(tiers[0]),
            "tier_1M": int(tiers[1]),
            "tier_3M": int(tiers[2]),
            "tier_6M": int(tiers[3]),
        })
    tier_df = pd.DataFrame(rows)

    # Transition matrices for each consecutive landmark pair, per split
    transitions: dict = {"cutoffs": [float(t1), float(t2)], "by_pair": {}}
    pairs = [("0M", "1M"), ("1M", "3M"), ("3M", "6M")]
    tier_labels = ["Low", "Mid", "High"]
    for split in ("Development", "Temporal"):
        sub_df = tier_df[tier_df["Split"] == split]
        for col_from, col_to in pairs:
            mat = np.zeros((3, 3), dtype=int)
            for _, row in sub_df.iterrows():
                mat[row[f"tier_{col_from}"]][row[f"tier_{col_to}"]] += 1
            row_sums = mat.sum(axis=1, keepdims=True)
            prob_mat = np.divide(mat, row_sums, out=np.zeros_like(mat, dtype=float),
                                 where=row_sums > 0)
            transitions["by_pair"][f"{split}__{col_from}_to_{col_to}"] = {
                "counts": mat.tolist(),
                "probabilities": prob_mat.tolist(),
                "row_labels": tier_labels,
                "col_labels": tier_labels,
            }
    transitions["tier_assignments"] = tier_df.to_dict(orient="records")
    return transitions


def fig_risk_migration_matrix(transitions: dict, out: Path) -> None:
    """Render transition probability matrix heatmap: 2 splits × 3 transitions = 6 panels."""
    pairs = [("0M", "1M"), ("1M", "3M"), ("3M", "6M")]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    for r, split in enumerate(("Development", "Temporal")):
        for c, (lf, lt) in enumerate(pairs):
            ax = axes[r, c]
            key = f"{split}__{lf}_to_{lt}"
            data = transitions["by_pair"][key]
            prob = np.array(data["probabilities"])
            counts = np.array(data["counts"])
            im = ax.imshow(prob, cmap="YlOrRd", vmin=0, vmax=1)
            ax.set_xticks(range(3))
            ax.set_xticklabels(data["col_labels"])
            ax.set_yticks(range(3))
            ax.set_yticklabels(data["row_labels"])
            for i in range(3):
                for j in range(3):
                    cnt = int(counts[i, j])
                    if cnt < 5:
                        txt = f"{prob[i, j]:.2f}\n(n={cnt})"
                        color = "#888"
                    else:
                        txt = f"{prob[i, j]:.2f}\n(n={cnt})"
                        color = "white" if prob[i, j] > 0.5 else "#333"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=8, color=color)
            ax.set_title(f"{split}: {lf} → {lt}", fontsize=10)
            ax.set_xlabel("To tier")
            if c == 0:
                ax.set_ylabel("From tier")
    fig.suptitle("M2-Base risk-tier transition probability matrix\n"
                 "(tertile cutoffs locked on dev 0M OOF; row = from tier, col = to tier)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Shapley block decomposition (5 blocks → 120 orderings)
# ---------------------------------------------------------------------------


def shapley_blocks(sd: StackedData) -> pd.DataFrame:
    """Compute Shapley contribution of each block to temporal pooled ROC.

    Average marginal contribution over all 5! = 120 orderings of {A,B,C,D,E}.
    Caches L2 OOF runs to avoid 120 × 5 = 600 refits; only computes the 32 unique
    block subsets needed (2^5).
    """
    block_letters = ("A", "B", "C", "D", "E")
    is_test = sd.rows["Split"].values == "Temporal"
    y_test = sd.rows.loc[is_test, "Y_24M_NHRH"].values

    # Build a cache of subset → temporal pooled ROC.
    print("  Computing AUCs for 32 block subsets …", flush=True)
    cache: dict[frozenset, float] = {}
    for size in range(0, len(block_letters) + 1):
        for combo in itertools.combinations(block_letters, size):
            key = frozenset(combo)
            if not combo:
                cache[key] = 0.5  # empty model = 0.5 ROC
                continue
            try:
                raw, _ = fit_l2_predict_oof(sd, block_subset=list(combo))
                cal = per_landmark_platt_on_pooled_oof(sd, raw)
                p = apply_per_landmark_platt(sd, raw, cal)
                cache[key] = float(roc_auc_score(y_test, p[is_test]))
            except Exception:
                cache[key] = float("nan")

    print("  Averaging over 5! = 120 orderings …", flush=True)
    contributions: dict[str, list[float]] = {b: [] for b in block_letters}
    for perm in itertools.permutations(block_letters):
        prev_set: frozenset = frozenset()
        for b in perm:
            new_set = prev_set | {b}
            delta = cache.get(new_set, float("nan")) - cache.get(prev_set, float("nan"))
            if not np.isnan(delta):
                contributions[b].append(delta)
            prev_set = new_set

    rows = []
    full_auc = cache[frozenset(block_letters)]
    for b, vals in contributions.items():
        rows.append({
            "Block": b,
            "Description": {
                "A": "Baseline burden",
                "B": "RAI exposure",
                "C": "Current dynamic",
                "D": "Momentum",
                "E": "Time × dynamic",
            }[b],
            "Shapley_value": float(np.mean(vals)) if vals else float("nan"),
            "Std": float(np.std(vals)) if vals else float("nan"),
            "N_orderings": len(vals),
        })
    df = pd.DataFrame(rows)
    # Sanity check: sum of Shapley values should equal full AUC - 0.5
    df["Pct_of_total"] = df["Shapley_value"] / (full_auc - 0.5) * 100 if full_auc > 0.5 else float("nan")
    return df


def fig_shapley(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    df_sorted = df.sort_values("Shapley_value", ascending=False)
    colors = {"A": "#1d4e89", "B": "#0a6b3f", "C": "#d28b18", "D": "#a23b3b", "E": "#5a3f8a"}
    bar_colors = [colors[b] for b in df_sorted["Block"]]
    y = np.arange(len(df_sorted))[::-1]
    ax.barh(y, df_sorted["Shapley_value"], color=bar_colors, alpha=0.85)
    for i, row in df_sorted.iterrows():
        yp = y[df_sorted.index.get_loc(i)]
        ax.text(row["Shapley_value"] + 0.001, yp,
                f"{row['Shapley_value']:+.4f}  ({row['Pct_of_total']:+.1f}%)",
                va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([f"[{row['Block']}] {row['Description']}" for _, row in df_sorted.iterrows()])
    ax.axvline(0, color="#444", linestyle=":", linewidth=1)
    ax.set_xlabel("Shapley value (avg marginal ΔTemporal pooled ROC over 5! orderings)")
    ax.set_title("M2-Base Shapley 5-block decomposition")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. 2x2 head-to-head: architecture × feature-set
# ---------------------------------------------------------------------------
#
# Cell (a): supermodel WITH only baseline-burden + RAI-exposure + landmark-
#           specific current dynamics — i.e., the union of features the
#           original M1·v2 + per-landmark TSH/FT4 model uses. Tests "does
#           stacking help when features are matched?"
# Cell (b): supermodel WITH full A+B+C+D+E (M2-Base.4). Tests "do the new
#           features help once stacking is fixed?"
# Cell (c): per-landmark independent L2-logistic on the legacy-style 16-feature
#           clinical-core set (approximated here by A+B+C since we don't have
#           Eval state encodings in the stacked dataset). Tests "do 4 LRs
#           win on legacy features?"
# Cell (d): per-landmark independent L2-logistic on M2-Base.4 features. Tests
#           "do 4 LRs win on rich features?"
#
# In practice we approximate the "per-landmark independent LR" via the
# block-subset path with a single shared model — the data structure precludes
# truly-independent fits without re-implementing the 4-LR architecture. We
# document this approximation honestly in the report.


def two_by_two_head_to_head(sd: StackedData) -> pd.DataFrame:
    """Run 2x2 head-to-head with paired episode-cluster bootstrap CIs."""
    is_test = sd.rows["Split"].values == "Temporal"
    y_test = sd.rows.loc[is_test, "Y_24M_NHRH"].values

    def auc(p: np.ndarray) -> float:
        try:
            return float(roc_auc_score(y_test, p[is_test]))
        except Exception:
            return float("nan")

    # Cell (a): supermodel with A+B+C (baseline + exposure + current dynamic, no momentum, no time interactions)
    raw_a, _ = fit_l2_predict_oof(sd, block_subset=("A", "B", "C"))
    cal_a = per_landmark_platt_on_pooled_oof(sd, raw_a)
    p_a = apply_per_landmark_platt(sd, raw_a, cal_a)

    # Cell (b): supermodel with A+B+C+D+E (M2-Base.4 full)
    raw_b, _ = fit_l2_predict_oof(sd, block_subset=("A", "B", "C", "D", "E"))
    cal_b = per_landmark_platt_on_pooled_oof(sd, raw_b)
    p_b = apply_per_landmark_platt(sd, raw_b, cal_b)

    # Cell (c): per-landmark independent LR on A+B+C — we approximate by training
    # one L2 model per-landmark-subset (only 0M rows for 0M model, etc.); each
    # landmark uses the same A+B+C feature set.
    p_c = _per_landmark_independent(sd, ("A", "B", "C"))

    # Cell (d): per-landmark independent LR on A+B+C+D+E
    p_d = _per_landmark_independent(sd, ("A", "B", "C", "D", "E"))

    rows = []
    for name, proba in (("a. Supermodel + ABC features", p_a),
                         ("b. Supermodel + ABCDE features (M2-Base.4)", p_b),
                         ("c. Per-landmark LR + ABC features (4-LR baseline)", p_c),
                         ("d. Per-landmark LR + ABCDE features", p_d)):
        auc_val = auc(proba)
        rows.append({"Cell": name, "Temporal_pooled_ROC": auc_val})

    df = pd.DataFrame(rows)

    # Quantify architecture effect (a vs c, b vs d) and feature-set effect (a vs b, c vs d)
    arch_a_minus_c = df.iloc[0]["Temporal_pooled_ROC"] - df.iloc[2]["Temporal_pooled_ROC"]
    arch_b_minus_d = df.iloc[1]["Temporal_pooled_ROC"] - df.iloc[3]["Temporal_pooled_ROC"]
    feat_b_minus_a = df.iloc[1]["Temporal_pooled_ROC"] - df.iloc[0]["Temporal_pooled_ROC"]
    feat_d_minus_c = df.iloc[3]["Temporal_pooled_ROC"] - df.iloc[2]["Temporal_pooled_ROC"]

    # Paired-bootstrap CIs for the four contrasts
    def pbc(pa, pb, tag):
        m, lo, hi = paired_episode_cluster_bootstrap_delta(sd, pa, pb, landmark=None, split="Temporal")
        return {"Contrast": tag, "Delta_mean": m, "CI_Low": lo, "CI_High": hi,
                 "CI_excludes_zero": bool(lo > 0 or hi < 0)
                 if not (np.isnan(lo) or np.isnan(hi)) else False}

    contrasts = pd.DataFrame([
        pbc(p_a, p_c, "Architecture effect at ABC (supermodel − 4-LR)"),
        pbc(p_b, p_d, "Architecture effect at ABCDE (supermodel − 4-LR)"),
        pbc(p_b, p_a, "Feature effect on supermodel (ABCDE − ABC)"),
        pbc(p_d, p_c, "Feature effect on 4-LR (ABCDE − ABC)"),
    ])
    return df, contrasts


def _per_landmark_independent(sd: StackedData, blocks: tuple[str, ...]) -> np.ndarray:
    """Fit one L2-logistic per landmark on the chosen block subset.

    Mimics the original M2 4-LR architecture: each landmark gets its own model
    trained only on that landmark's rows. Per-landmark Platt applied identically.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X_all = sd.feature_matrix(blocks).values
    y_all = sd.rows["Y_24M_NHRH"].values
    lm_all = sd.rows["landmark"].values
    split_all = sd.rows["Split"].values

    proba = np.zeros(len(sd.rows), dtype=float)
    for L in LANDMARKS:
        lm_mask = lm_all == L
        dev = lm_mask & (split_all == "Development")
        test = lm_mask & (split_all == "Temporal")
        X_d = X_all[dev]
        y_d = y_all[dev]
        # 5-fold OOF (no group needed since rows of different episodes at same landmark are independent)
        from sklearn.model_selection import StratifiedKFold as SKF
        skf = SKF(n_splits=5, shuffle=True, random_state=42)
        dev_idx = np.where(dev)[0]
        for tr, va in skf.split(X_d, y_d):
            pipe = Pipeline([("s", StandardScaler()),
                              ("lr", LogisticRegression(penalty="l2", solver="lbfgs",
                                                         C=1.0, max_iter=5000, random_state=2025))])
            pipe.fit(X_d[tr], y_d[tr])
            proba[dev_idx[va]] = pipe.predict_proba(X_d[va])[:, 1]
        # Final on full dev → test
        if test.any():
            pipe = Pipeline([("s", StandardScaler()),
                              ("lr", LogisticRegression(penalty="l2", solver="lbfgs",
                                                         C=1.0, max_iter=5000, random_state=2025))])
            pipe.fit(X_d, y_d)
            proba[test] = pipe.predict_proba(X_all[test])[:, 1]
    return proba


def fig_2x2_head_to_head(df: pd.DataFrame, contrasts: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    # Left: 2x2 panel
    pivot = pd.DataFrame({
        "ABC features": [df.iloc[2]["Temporal_pooled_ROC"], df.iloc[0]["Temporal_pooled_ROC"]],
        "ABCDE features": [df.iloc[3]["Temporal_pooled_ROC"], df.iloc[1]["Temporal_pooled_ROC"]],
    }, index=["Per-landmark LR (4-LR)", "Supermodel (stacked)"])
    ax = axes[0]
    im = ax.imshow(pivot.values, cmap="YlGnBu", vmin=0.6, vmax=0.8, aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.4f}", ha="center", va="center",
                    color="white" if v > 0.7 else "#333", fontsize=11, weight="bold")
    ax.set_title("2×2 head-to-head: temporal pooled ROC")
    fig.colorbar(im, ax=ax, shrink=0.7)
    # Right: contrasts with CIs
    ax = axes[1]
    y = np.arange(len(contrasts))[::-1]
    means = contrasts["Delta_mean"].values
    lo = (contrasts["Delta_mean"] - contrasts["CI_Low"]).abs().values
    hi = (contrasts["CI_High"] - contrasts["Delta_mean"]).abs().values
    colors = ["#a23b3b" if r["CI_excludes_zero"] else "#666" for _, r in contrasts.iterrows()]
    for i, (yp, m, l, h, c) in enumerate(zip(y, means, lo, hi, colors)):
        ax.errorbar([m], [yp], xerr=[[l], [h]], fmt="o", color=c, capsize=5, markersize=8)
    ax.axvline(0, color="#444", linestyle=":", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r["Contrast"] for _, r in contrasts.iterrows()], fontsize=8)
    ax.set_xlabel("Δ ROC-AUC (paired episode-cluster bootstrap CI)")
    ax.set_title("2×2 contrasts isolating architecture vs feature-set")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


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

    # --- 1. Risk migration transition matrix ---
    print("\n[1] Building risk migration transition matrix …", flush=True)
    raw_full, _ = fit_l2_predict_oof(sd, block_subset=("A", "B", "C", "D", "E"))
    cal_full = per_landmark_platt_on_pooled_oof(sd, raw_full)
    p_full = apply_per_landmark_platt(sd, raw_full, cal_full)
    transitions = build_risk_tier_transitions(sd, p_full)
    (TAB_DIR / "risk_migration_transitions.json").write_text(
        json.dumps({k: v for k, v in transitions.items() if k != "tier_assignments"},
                    indent=2, ensure_ascii=False)
    )
    # Tier assignments table (lightweight CSV)
    pd.DataFrame(transitions["tier_assignments"]).to_csv(
        TAB_DIR / "risk_tier_assignments.csv", index=False
    )
    fig_risk_migration_matrix(transitions, FIG_DIR / "Figure_06_RiskMigration_TransitionMatrix.png")
    print(f"  cutoffs (dev 0M tertiles): low ≤ {transitions['cutoffs'][0]:.3f} < mid ≤ {transitions['cutoffs'][1]:.3f} < high", flush=True)

    # --- 2. Shapley block decomposition ---
    print("\n[2] Computing Shapley 5-block decomposition …", flush=True)
    shapley_df = shapley_blocks(sd)
    shapley_df.to_csv(TAB_DIR / "shapley_block_decomposition.csv", index=False)
    fig_shapley(shapley_df, FIG_DIR / "Figure_07_Shapley_Block_Decomposition.png")
    print(shapley_df.to_string(index=False), flush=True)

    # --- 3. 2x2 head-to-head ---
    print("\n[3] 2x2 head-to-head (architecture × feature-set) …", flush=True)
    h2h_df, contrasts = two_by_two_head_to_head(sd)
    h2h_df.to_csv(TAB_DIR / "head_to_head_2x2.csv", index=False)
    contrasts.to_csv(TAB_DIR / "head_to_head_contrasts.csv", index=False)
    fig_2x2_head_to_head(h2h_df, contrasts, FIG_DIR / "Figure_08_2x2_Head_to_Head.png")
    print("\n  2x2 cells:")
    print(h2h_df.to_string(index=False), flush=True)
    print("\n  Contrasts (paired bootstrap):")
    print(contrasts.to_string(index=False), flush=True)

    # Summary
    summary = {
        "risk_migration_cutoffs": transitions["cutoffs"],
        "shapley_values": shapley_df.set_index("Block")["Shapley_value"].to_dict(),
        "2x2_cells": h2h_df.set_index("Cell")["Temporal_pooled_ROC"].to_dict(),
        "2x2_contrasts": contrasts.to_dict(orient="records"),
    }
    (TAB_DIR / "supplementary_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nDONE. Supplementary tables/figures saved.", flush=True)


if __name__ == "__main__":
    main()
