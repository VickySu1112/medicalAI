#!/usr/bin/env python
"""M2 · B4 — cross-landmark prediction tracking (Phase 4 task1, script 1).

Builds the per-episode × per-landmark (1/3/6/12) EBM prediction TRACKING wide
table and a tracking heatmap of |p − Y| ordered by trajectory.

口径: corrected truth + LOCF, EBM-OOF (dev) / temporal read-out, with naive
(dev prevalence) + persistence (time-safe TSH_current L2-LR) baselines per
landmark. All time-safe. Temporal numbers are read-out only.

Outputs (results/module2_v2_vertical/b4_target_gru/):
    tables/xland_tracking_wide.csv   — one row per episode:
        episode_id, Split, Y, p@{L}, correct@{L}, tier@{L}, p_naive@{L},
        p_persist@{L}, state@{L}   for L in 1/3/6/12
    tables/xland_tracking_long.csv   — tidy long form (episode × landmark)
    tables/xland_landmark_auc.csv    — per-landmark EBM/persist/naive OOF+temporal AUC
    tables/xland_tracking_summary.json
    figures/xland_Figure_01_TrackingHeatmap.png
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
from matplotlib.colors import LinearSegmentedColormap

from scripts.simple.module2_v2_b4_xland_shared import (
    LANDMARKS,
    TBL,
    FIG,
    build_tracking_long,
    ensure_dirs,
    long_to_wide,
)


def _auc_table(meta) -> pd.DataFrame:
    recs = []
    for L in meta["landmarks"]:
        a = meta["auc"][L]
        recs.append(
            {
                "Landmark": f"{L}M",
                "N_dev": a["n_dev"],
                "N_temporal": a["n_temporal"],
                "Prevalence_dev": round(a["prevalence_dev"], 4),
                "EBM_OOF": round(a["ebm_oof"], 4),
                "EBM_Temporal": round(a["ebm_temporal"], 4),
                "Persist_OOF": round(a["persist_oof"], 4),
                "Persist_Temporal": round(a["persist_temporal"], 4),
                "Naive_OOF": round(a["naive_oof"], 4),
                "Naive_Temporal": round(a["naive_temporal"], 4),
                "state_source": meta["state_source"][L],
            }
        )
    return pd.DataFrame(recs)


def _tracking_heatmap(long_df: pd.DataFrame, landmarks) -> None:
    """|p − Y| heatmap (rows = episodes ordered by trajectory, cols = landmarks).

    Rows are sorted by the per-episode error trajectory (mean |p−Y| then the
    later-landmark error) so persistently-hard episodes sink to one band and
    signal-accumulation (early-wrong → late-right) episodes show a left→right
    fade. Built on the TEMPORAL split (held-out; read-out only) so the picture
    is the deployment-time error structure.
    """
    sub = long_df[long_df["Split"] == "Temporal"].copy()
    sub["abs_err"] = (sub["p"] - sub["Y"]).abs()
    piv = sub.pivot(index="episode_id", columns="landmark", values="abs_err")
    piv = piv.reindex(columns=sorted(landmarks))
    # order: mean error desc, then last-landmark error desc (hard band on top)
    last_L = sorted(landmarks)[-1]
    order_key = piv.mean(axis=1).fillna(0.0) + 0.001 * piv[last_L].fillna(0.0)
    piv = piv.loc[order_key.sort_values(ascending=False).index]

    cmap = LinearSegmentedColormap.from_list("err", ["#0d3b66", "#f4f1de", "#c1121f"])
    fig, ax = plt.subplots(figsize=(5.6, 7.2))
    im = ax.imshow(piv.values, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{int(c)}M" for c in piv.columns])
    ax.set_yticks([])
    ax.set_xlabel("地标 (landmark)")
    ax.set_ylabel(f"病例 (temporal, n={piv.shape[0]}，按轨迹误差排序)")
    ax.set_title("跨地标预测误差 |p−Y| 追踪热力图\n(顶部=顽固难例带，左高右低=信号积累型)", fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("|p − Y|  (0=完美, 1=最差)")
    fig.tight_layout()
    fig.savefig(FIG / "xland_Figure_01_TrackingHeatmap.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke: 6M only")
    args = ap.parse_args()
    ensure_dirs()

    long_df, meta = build_tracking_long(quick=args.quick)
    landmarks = meta["landmarks"]
    wide = long_to_wide(long_df)

    long_df.to_csv(TBL / "xland_tracking_long.csv", index=False)
    wide.to_csv(TBL / "xland_tracking_wide.csv", index=False)
    auc_df = _auc_table(meta)
    auc_df.to_csv(TBL / "xland_landmark_auc.csv", index=False)

    _tracking_heatmap(long_df, landmarks)

    # headline counts
    n_ep = int(wide["episode_id"].nunique())
    temporal_ep = int((wide["Split"] == "Temporal").sum())
    dev_ep = int((wide["Split"] == "Development").sum())

    print("=== 跨地标预测追踪宽表 (corrected+LOCF, EBM-OOF / temporal read-out) ===", flush=True)
    print(f"  分析单元 = 人次(疗程); N = {n_ep} 人次 (dev {dev_ep} / temporal {temporal_ep})", flush=True)
    print(f"  地标 = {[f'{L}M' for L in landmarks]}", flush=True)
    print("\n  逐地标判别 (EBM vs persistence vs naive; OOF 选择 / Temporal read-out):", flush=True)
    print(auc_df.to_string(index=False), flush=True)
    print(f"\n  宽表列 = {list(wide.columns)}", flush=True)

    summary = {
        "口径": "corrected truth (Current_Time) + LOCF impute, time-safe; EBM-OOF dev / temporal read-out",
        "analysis_unit": "treatment-episode (人次)",
        "n_episodes": n_ep,
        "n_dev_episodes": dev_ep,
        "n_temporal_episodes": temporal_ep,
        "landmarks": [f"{L}M" for L in landmarks],
        "youden_thr_per_landmark": {f"{L}M": round(meta["thr"][L], 4) for L in landmarks},
        "landmark_auc": auc_df.to_dict(orient="records"),
        "wide_columns": list(wide.columns),
        "note_temporal": "Temporal AUCs are read-out only (never used to select/tune).",
        "note_state_6_12": "Functional state at 6M/12M is derived from corrected hormone levels "
        "(clinical Eval one-hot is degenerate there); 1M/3M use the real clinical Eval.",
    }
    (TBL / "xland_tracking_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(f"\nSaved → {TBL}  &  {FIG}", flush=True)


if __name__ == "__main__":
    main()
