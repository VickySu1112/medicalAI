#!/usr/bin/env python
"""M2 · B4 — Phase 5 high-value cross-landmark ideas (task5).

Reuses the corrected+LOCF EBM-OOF tracking (no new modelling beyond the per-
landmark EBM/persistence already in xland_shared) to land four near-zero-cost,
high-value cross-landmark analyses:

  ① 风险档迁移 3×3 矩阵 + 桑基 — how Low/Mid/High risk tiers reassign as the
     landmark advances (e.g. 3M→6M, 6M→12M), on the temporal split.
  ② persistence 增量随地标 — EBM − persistence ΔAUC per landmark with a paired
     episode-cluster bootstrap CI (how much the learned model adds over "just
     read today's TSH", and whether the gap's 95% CI excludes 0).
  ③ per-landmark 最优弃权率 — split the pooled selective_prediction into a
     per-landmark risk–coverage curve (label-free abstention on the least
     confident), so the optimal defer-rate can differ by landmark.
  ④ 错例 churn — Jaccard of consecutive-landmark error sets (consecutive-pair
     view complementing the full matrix in xland_trajectory).

口径 = corrected+LOCF, EBM-OOF dev / temporal read-out, naive+persistence,
time-safe. Temporal numbers are read-out only.

Outputs (results/module2_v2_vertical/b4_target_gru/):
    tables/xland_tier_migration.csv          — per consecutive-pair 3×3 counts
    tables/xland_persist_increment.csv       — EBM−persist ΔAUC + bootstrap CI
    tables/xland_per_landmark_abstain.csv    — risk-coverage per landmark
    tables/xland_consecutive_churn.csv       — consecutive-pair error Jaccard
    tables/xland_phase5_summary.json
    figures/xland_Figure_05_TierSankey.png
    figures/xland_Figure_06_PersistIncrement.png
    figures/xland_Figure_07_PerLandmarkAbstain.png
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

from scripts.simple.module2_v2_shared import (
    StackedData,
    paired_episode_cluster_bootstrap_delta,
)
from scripts.simple.module2_v2_b4_xland_shared import (
    TIER_NAMES,
    TBL,
    FIG,
    assign_tier,
    build_tracking_long,
    ensure_dirs,
)


# ---------------------------------------------------------------------------
# ① risk-tier migration 3×3 + sankey
# ---------------------------------------------------------------------------


def tier_migration(long_df, landmarks, split="Temporal"):
    sub = long_df[long_df["Split"] == split]
    Ls = sorted(landmarks)
    recs = []
    mats = {}
    for a, b in zip(Ls[:-1], Ls[1:]):
        ga = sub[sub["landmark"] == a].set_index("episode_id")["tier"]
        gb = sub[sub["landmark"] == b].set_index("episode_id")["tier"]
        common = ga.index.intersection(gb.index)
        mat = pd.DataFrame(0, index=list(TIER_NAMES), columns=list(TIER_NAMES))
        for e in common:
            mat.loc[ga[e], gb[e]] += 1
        mats[(a, b)] = mat
        for ti in TIER_NAMES:
            for tj in TIER_NAMES:
                recs.append({"from_landmark": f"{a}M", "to_landmark": f"{b}M",
                             "from_tier": ti, "to_tier": tj,
                             "n": int(mat.loc[ti, tj])})
    return pd.DataFrame(recs), mats


def _sankey(mats, landmarks):
    """Hand-drawn flow Sankey across the consecutive landmark tier states.

    Columns = landmarks; nodes = Low/Mid/High stacked; ribbons = counts. Kept in
    matplotlib so the report stays self-contained (base64 PNG, no plotly HTML).
    """
    Ls = sorted(landmarks)
    pairs = list(zip(Ls[:-1], Ls[1:]))
    tier_y = {"High": 2.0, "Mid": 1.0, "Low": 0.0}
    tier_color = {"High": "#c1121f", "Mid": "#e9c46a", "Low": "#2a9d8f"}
    fig, ax = plt.subplots(figsize=(2.4 + 2.2 * len(pairs), 4.6))

    # node totals per landmark
    col_tot = {}
    for li, L in enumerate(Ls):
        if li < len(Ls) - 1:
            m = mats[(L, Ls[li + 1])]
            col_tot[L] = {t: int(m.loc[t].sum()) for t in TIER_NAMES}
        else:
            m = mats[(Ls[li - 1], L)]
            col_tot[L] = {t: int(m[t].sum()) for t in TIER_NAMES}

    maxtot = max(max(d.values()) for d in col_tot.values()) or 1
    barw = 0.10
    xpos = {L: i for i, L in enumerate(Ls)}

    # draw nodes
    for L in Ls:
        for t in TIER_NAMES:
            h = 0.9 * col_tot[L][t] / maxtot
            ax.add_patch(plt.Rectangle((xpos[L] - barw / 2, tier_y[t] - h / 2), barw, h,
                                       color=tier_color[t], alpha=0.95, zorder=3))
            if col_tot[L][t] > 0:
                ax.text(xpos[L], tier_y[t] + h / 2 + 0.03, str(col_tot[L][t]),
                        ha="center", va="bottom", fontsize=7, color="#333")

    # draw ribbons
    for (a, b) in pairs:
        m = mats[(a, b)]
        x0, x1 = xpos[a] + barw / 2, xpos[b] - barw / 2
        # track vertical offset within each source/target node
        src_off = {t: -0.45 * col_tot[a][t] / maxtot * 0.9 for t in TIER_NAMES}
        tgt_off = {t: -0.45 * col_tot[b][t] / maxtot * 0.9 for t in TIER_NAMES}
        for ti in TIER_NAMES:
            for tj in TIER_NAMES:
                n = int(m.loc[ti, tj])
                if n == 0:
                    continue
                w = 0.9 * n / maxtot
                ys0 = tier_y[ti] + src_off[ti]
                ys1 = tier_y[tj] + tgt_off[tj]
                src_off[ti] += w
                tgt_off[tj] += w
                verts = [(x0, ys0), (x0 + (x1 - x0) * 0.4, ys0),
                         (x0 + (x1 - x0) * 0.6, ys1), (x1, ys1),
                         (x1, ys1 + w), (x0 + (x1 - x0) * 0.6, ys1 + w),
                         (x0 + (x1 - x0) * 0.4, ys0 + w), (x0, ys0 + w), (x0, ys0)]
                codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                         MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                         MplPath.CLOSEPOLY]
                col = tier_color[ti] if ti == tj else "#9aa0a6"
                ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=col,
                                       alpha=0.32 if ti != tj else 0.5, edgecolor="none", zorder=2))

    ax.set_xticks(list(xpos.values()))
    ax.set_xticklabels([f"{L}M" for L in Ls])
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["Low", "Mid", "High"])
    ax.set_xlim(-0.4, len(Ls) - 0.6); ax.set_ylim(-0.8, 2.8)
    ax.set_title("风险档迁移桑基(temporal;彩带=保持档, 灰带=改判)", fontsize=10)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "xland_Figure_05_TierSankey.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# ② persistence increment (EBM − persist ΔAUC) per landmark, paired bootstrap CI
# ---------------------------------------------------------------------------


def _shim_from_long(long_df):
    """Build a StackedData shim + aligned EBM/persist proba vectors from LONG."""
    df = long_df.sort_values(["episode_id", "landmark"]).reset_index(drop=True)
    rows = pd.DataFrame({
        "episode_id": df["episode_id"].values,
        "landmark": df["landmark"].values,
        "Split": df["Split"].values,
        "Y_24M_NHRH": df["Y"].values,
    })
    ep_meta = rows.drop_duplicates("episode_id")[["episode_id", "Split", "Y_24M_NHRH"]]
    sd = StackedData(rows=rows, episode_meta=ep_meta)
    p_ebm = df["p"].values
    p_persist = df["p_persist"].values
    p_naive = df["p_naive"].values
    return sd, p_ebm, p_persist, p_naive


def persist_increment(long_df, meta, landmarks):
    sd, p_ebm, p_persist, p_naive = _shim_from_long(long_df)
    recs = []
    for L in sorted(landmarks):
        a = meta["auc"][L]
        d_mean, d_lo, d_hi = paired_episode_cluster_bootstrap_delta(
            sd, p_ebm, p_persist, landmark=L, split="Temporal"
        )
        dn_mean, dn_lo, dn_hi = paired_episode_cluster_bootstrap_delta(
            sd, p_ebm, p_naive, landmark=L, split="Temporal"
        )
        recs.append({
            "Landmark": f"{L}M", "_L": L,
            "EBM_temporal": round(a["ebm_temporal"], 4),
            "Persist_temporal": round(a["persist_temporal"], 4),
            "dAUC_EBM_minus_Persist": round(d_mean, 4),
            "CI_low": round(d_lo, 4), "CI_high": round(d_hi, 4),
            "excludes_0": bool((d_lo > 0) or (d_hi < 0)),
            "dAUC_EBM_minus_Naive": round(dn_mean, 4),
        })
    return pd.DataFrame(recs)


def _plot_increment(inc):
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    x = inc["_L"].values
    ax.plot(x, inc["EBM_temporal"], marker="o", color="#1d4e89", label="EBM (temporal)")
    ax.plot(x, inc["Persist_temporal"], marker="s", color="#c1121f", label="persistence (今日TSH)")
    ax.fill_between(x, inc["Persist_temporal"], inc["EBM_temporal"], color="#1d4e89", alpha=0.10)
    for _, r in inc.iterrows():
        tag = "*" if r["excludes_0"] else ""
        ax.annotate(f"Δ{r['dAUC_EBM_minus_Persist']:+.3f}{tag}",
                    (r["_L"], (r["EBM_temporal"] + r["Persist_temporal"]) / 2),
                    fontsize=8, ha="center", color="#1d4e89")
    ax.set_xticks(x); ax.set_xticklabels([f"{int(v)}M" for v in x])
    ax.set_xlabel("地标 (landmark)"); ax.set_ylabel("ROC-AUC (temporal, read-out)")
    ax.set_title("学习模型相对 persistence 的增量随地标\n(Δ=EBM−persist;* = 95% bootstrap CI 排除 0)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "xland_Figure_06_PersistIncrement.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# ③ per-landmark optimal abstention (risk-coverage), label-free
# ---------------------------------------------------------------------------


def per_landmark_abstain(long_df, meta, landmarks):
    """Per-landmark risk–coverage: abstain on least-confident (|p−thr|) temporal
    cases; trace retained accuracy/NPV/PPV. Threshold = Youden@OOF (selection-safe).
    """
    recs = []
    for L in sorted(landmarks):
        thr = meta["thr"][L]
        g = long_df[(long_df["landmark"] == L) & (long_df["Split"] == "Temporal")]
        y = g["Y"].values
        p = g["p"].values
        conf = np.abs(p - thr)
        order = np.argsort(-conf)  # most confident first
        for abst in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
            keep = order[: int(round((1 - abst) * len(y)))]
            yk, pk = y[keep], p[keep]
            pred = (pk >= thr).astype(int)
            n = len(yk)
            tp = int(((pred == 1) & (yk == 1)).sum()); fp = int(((pred == 1) & (yk == 0)).sum())
            tn = int(((pred == 0) & (yk == 0)).sum()); fn = int(((pred == 0) & (yk == 1)).sum())
            acc = (tp + tn) / n if n else float("nan")
            npv = tn / (tn + fn) if (tn + fn) else float("nan")
            ppv = tp / (tp + fp) if (tp + fp) else float("nan")
            recs.append({"Landmark": f"{L}M", "_L": L,
                         "abstain_pct": int(abst * 100), "coverage_pct": int((1 - abst) * 100),
                         "n_retained": n, "accuracy": round(acc, 4),
                         "NPV": round(npv, 4) if np.isfinite(npv) else None,
                         "PPV": round(ppv, 4) if np.isfinite(ppv) else None})
    return pd.DataFrame(recs)


def _plot_abstain(rc, landmarks):
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    cmap = {1: "#90c2e7", 3: "#168aad", 6: "#1d4e89", 12: "#03045e"}
    for L in sorted(landmarks):
        s = rc[rc["_L"] == L].sort_values("abstain_pct")
        ax.plot(s["abstain_pct"], s["accuracy"], marker="o", color=cmap.get(L, "#333"),
                label=f"{L}M")
    ax.set_xlabel("弃权率 % (转交临床)"); ax.set_ylabel("保留病例准确率 (temporal)")
    ax.set_title("逐地标选择性预测:弃权率 vs 保留准确率\n(各地标最优弃权率可不同)", fontsize=10)
    ax.legend(fontsize=8, title="地标")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "xland_Figure_07_PerLandmarkAbstain.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# ④ consecutive-pair error churn (Jaccard)
# ---------------------------------------------------------------------------


def consecutive_churn(long_df, landmarks, split="Temporal"):
    sub = long_df[long_df["Split"] == split]
    Ls = sorted(landmarks)
    recs = []
    for a, b in zip(Ls[:-1], Ls[1:]):
        ea = set(sub.loc[(sub["landmark"] == a) & (sub["correct"] == 0), "episode_id"])
        eb = set(sub.loc[(sub["landmark"] == b) & (sub["correct"] == 0), "episode_id"])
        union = len(ea | eb)
        jac = len(ea & eb) / union if union else float("nan")
        recs.append({"split": split, "pair": f"{a}M→{b}M",
                     "n_err_a": len(ea), "n_err_b": len(eb),
                     "n_persisted": len(ea & eb), "n_resolved": len(ea - eb),
                     "n_new": len(eb - ea),
                     "jaccard": round(jac, 4) if union else None})
    return pd.DataFrame(recs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke: 6M only (pairs degenerate)")
    args = ap.parse_args()
    ensure_dirs()

    long_df, meta = build_tracking_long(quick=args.quick)
    landmarks = meta["landmarks"]

    # ① tier migration
    mig, mats = tier_migration(long_df, landmarks)
    mig.to_csv(TBL / "xland_tier_migration.csv", index=False)
    if len(landmarks) > 1:
        _sankey(mats, landmarks)

    # ② persistence increment
    inc = persist_increment(long_df, meta, landmarks)
    inc.drop(columns=["_L"]).to_csv(TBL / "xland_persist_increment.csv", index=False)
    if len(landmarks) > 1:
        _plot_increment(inc)

    # ③ per-landmark abstain
    rc = per_landmark_abstain(long_df, meta, landmarks)
    rc.drop(columns=["_L"]).to_csv(TBL / "xland_per_landmark_abstain.csv", index=False)
    if len(landmarks) > 1:
        _plot_abstain(rc, landmarks)

    # ④ consecutive churn
    cc = consecutive_churn(long_df, landmarks)
    cc.to_csv(TBL / "xland_consecutive_churn.csv", index=False)

    # ---- report ----
    print("=== ① 风险档迁移 3×3 (temporal, consecutive pairs) ===", flush=True)
    for (a, b), m in mats.items():
        print(f"\n  {a}M → {b}M (行=源档, 列=目标档):", flush=True)
        print(m.to_string(), flush=True)

    print("\n=== ② persistence 增量随地标 (EBM−persist ΔAUC, paired bootstrap CI) ===", flush=True)
    print(inc.drop(columns=["_L"]).to_string(index=False), flush=True)

    print("\n=== ③ per-landmark 最优弃权率 (risk-coverage, temporal) ===", flush=True)
    # show the no-abstain and 30%-abstain rows per landmark
    show = rc[rc["abstain_pct"].isin([0, 30])]
    print(show.drop(columns=["_L"]).to_string(index=False), flush=True)

    print("\n=== ④ 相邻地标错例 churn (Jaccard) ===", flush=True)
    print(cc.to_string(index=False), flush=True)

    summary = {
        "口径": "corrected+LOCF, EBM-OOF dev / temporal read-out, time-safe",
        "landmarks": [f"{L}M" for L in landmarks],
        "tier_migration": mig.to_dict(orient="records"),
        "persist_increment": inc.drop(columns=["_L"]).to_dict(orient="records"),
        "per_landmark_abstain": rc.drop(columns=["_L"]).to_dict(orient="records"),
        "consecutive_churn": cc.to_dict(orient="records"),
        "note": "Temporal numbers are read-out only; tiers cut at 0.30/0.60 on the EBM probability.",
    }
    (TBL / "xland_phase5_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(f"\nSaved → {TBL}  &  {FIG}", flush=True)


if __name__ == "__main__":
    main()
