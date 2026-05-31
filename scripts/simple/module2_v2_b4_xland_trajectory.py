#!/usr/bin/env python
"""M2 · B4 — same-episode error trajectories across landmarks (Phase 4 task1, script 2).

Follows each episode's EBM correctness across the 1/3/6/12 landmarks and asks:

  * 对错模式 (correctness pattern): persistent-correct / persistent-wrong /
    turned-correct (wrong→right) / turned-wrong (right→wrong) / mixed — using
    only the landmarks where the episode is observed.
  * 顽固难例 vs 信号积累型: among episodes wrong at the FIRST landmark, who is
    fixed by accumulating signal (early-wrong → late-right) vs who stays wrong
    (stubborn / feature-ceiling).
  * |p − Y| 随地标曲线: mean absolute error vs landmark for EBM, with the
    persistence baseline overlaid (does watching today's TSH improve the same
    way the model does as labs accumulate?).
  * 错例 churn (Jaccard): overlap of the per-landmark error sets — are the same
    episodes wrong at every landmark, or does the wrong-set rotate?

口径 = corrected+LOCF, EBM-OOF (dev) / temporal read-out, naive+persistence,
time-safe. Trajectory patterns are computed on the TEMPORAL split (held-out;
deployment-time) and, separately, on dev OOF for the larger-N churn matrix.

Outputs (results/module2_v2_vertical/b4_target_gru/):
    tables/xland_trajectory_patterns.csv      — pattern counts (temporal & dev OOF)
    tables/xland_stubborn_vs_accumulation.csv — first-wrong fate split
    tables/xland_abs_err_curve.csv            — mean|p−Y| vs landmark (EBM & persist)
    tables/xland_error_churn_jaccard.csv      — pairwise Jaccard of error sets
    tables/xland_trajectory_summary.json
    figures/xland_Figure_02_AbsErrCurve.png
    figures/xland_Figure_03_ErrorChurnJaccard.png
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

from scripts.simple.module2_v2_b4_xland_shared import (
    TBL,
    FIG,
    build_tracking_long,
    ensure_dirs,
)


# ---------------------------------------------------------------------------
# Correctness pattern per episode
# ---------------------------------------------------------------------------


def _classify_pattern(correct_seq) -> str:
    """Label an ordered correctness sequence (over observed landmarks).

    persistent-correct : all 1 · persistent-wrong : all 0 ·
    turned-correct     : starts 0, ends 1, monotone-ish improvement (last is 1
                          and first is 0) · turned-wrong : first 1, last 0 ·
    mixed              : oscillates otherwise.
    """
    s = list(correct_seq)
    if len(s) == 0:
        return "none"
    if all(v == 1 for v in s):
        return "persistent-correct"
    if all(v == 0 for v in s):
        return "persistent-wrong"
    if s[0] == 0 and s[-1] == 1:
        return "turned-correct"
    if s[0] == 1 and s[-1] == 0:
        return "turned-wrong"
    return "mixed"


def pattern_table(long_df: pd.DataFrame, split: str, landmarks) -> pd.DataFrame:
    sub = long_df[long_df["Split"] == split]
    rows = []
    for e, g in sub.sort_values("landmark").groupby("episode_id"):
        seq = g["correct"].tolist()
        rows.append({"episode_id": e, "Y": int(g["Y"].iloc[0]),
                     "pattern": _classify_pattern(seq), "n_obs": len(seq)})
    pf = pd.DataFrame(rows)
    counts = pf["pattern"].value_counts().to_dict()
    out = pd.DataFrame(
        [{"split": split, "pattern": p, "n_episodes": int(counts.get(p, 0)),
          "pct": round(100 * counts.get(p, 0) / max(len(pf), 1), 1)}
         for p in ("persistent-correct", "turned-correct", "mixed",
                   "turned-wrong", "persistent-wrong")]
    )
    return out, pf


# ---------------------------------------------------------------------------
# Stubborn hard cases vs signal accumulation
# ---------------------------------------------------------------------------


def stubborn_vs_accumulation(long_df: pd.DataFrame, split: str, landmarks) -> pd.DataFrame:
    """Among episodes WRONG at their first observed landmark, split the fate:

    fixed_by_signal  : becomes correct at some later landmark (and stays correct
                       at the last) — signal-accumulation type.
    stubborn         : wrong at the first AND the last landmark — feature ceiling.
    partial          : flips correct then wrong again (unstable).
    """
    sub = long_df[long_df["Split"] == split]
    n_first_wrong = 0
    fixed = stubborn = partial = 0
    for e, g in sub.sort_values("landmark").groupby("episode_id"):
        seq = g["correct"].tolist()
        if len(seq) < 2 or seq[0] != 0:
            continue
        n_first_wrong += 1
        if seq[-1] == 1:
            fixed += 1
        elif all(v == 0 for v in seq):
            stubborn += 1
        else:
            partial += 1
    return pd.DataFrame([{
        "split": split,
        "n_first_landmark_wrong": n_first_wrong,
        "fixed_by_signal_accumulation": fixed,
        "stubborn_wrong_throughout": stubborn,
        "partial_unstable": partial,
        "pct_fixed": round(100 * fixed / max(n_first_wrong, 1), 1),
        "pct_stubborn": round(100 * stubborn / max(n_first_wrong, 1), 1),
    }])


# ---------------------------------------------------------------------------
# |p − Y| vs landmark (EBM vs persistence)
# ---------------------------------------------------------------------------


def abs_err_curve(long_df: pd.DataFrame, landmarks) -> pd.DataFrame:
    rows = []
    for split in ("Development", "Temporal"):
        sub = long_df[long_df["Split"] == split]
        for L in sorted(landmarks):
            g = sub[sub["landmark"] == L]
            if g.empty:
                continue
            rows.append({
                "split": split, "landmark": f"{L}M", "_L": L,
                "n": int(len(g)),
                "EBM_mean_abs_err": round(float((g["p"] - g["Y"]).abs().mean()), 4),
                "Persist_mean_abs_err": round(float((g["p_persist"] - g["Y"]).abs().mean()), 4),
                "Naive_mean_abs_err": round(float((g["p_naive"] - g["Y"]).abs().mean()), 4),
            })
    return pd.DataFrame(rows)


def _plot_abs_err_curve(curve: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=True)
    for ax, split in zip(axes, ("Development", "Temporal")):
        c = curve[curve["split"] == split].sort_values("_L")
        if c.empty:
            continue
        x = c["_L"].values
        ax.plot(x, c["EBM_mean_abs_err"], marker="o", color="#1d4e89", label="EBM (OOF/temporal)")
        ax.plot(x, c["Persist_mean_abs_err"], marker="s", color="#c1121f", label="persistence (今日TSH)")
        ax.plot(x, c["Naive_mean_abs_err"], marker="^", ls="--", color="#999", label="naive (患病率)")
        ax.set_xticks(x); ax.set_xticklabels([f"{int(v)}M" for v in x])
        ax.set_xlabel("地标 (landmark)")
        ax.set_title(f"{'dev OOF' if split=='Development' else 'temporal (read-out)'}", fontsize=10)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("平均 |p − Y|  (越低越好)")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("预测误差 |p−Y| 随地标:EBM vs persistence vs naive(信号积累)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / "xland_Figure_02_AbsErrCurve.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Error churn (Jaccard of per-landmark error sets)
# ---------------------------------------------------------------------------


def error_churn(long_df: pd.DataFrame, split: str, landmarks):
    sub = long_df[long_df["Split"] == split]
    err_sets = {}
    for L in sorted(landmarks):
        g = sub[sub["landmark"] == L]
        err_sets[L] = set(g.loc[g["correct"] == 0, "episode_id"].tolist())
    Ls = sorted(landmarks)
    mat = np.full((len(Ls), len(Ls)), np.nan)
    recs = []
    for i, Li in enumerate(Ls):
        for j, Lj in enumerate(Ls):
            a, b = err_sets[Li], err_sets[Lj]
            union = len(a | b)
            jac = (len(a & b) / union) if union else float("nan")
            mat[i, j] = jac
            if i < j:
                recs.append({
                    "split": split, "landmark_a": f"{Li}M", "landmark_b": f"{Lj}M",
                    "n_err_a": len(a), "n_err_b": len(b),
                    "n_overlap": len(a & b), "n_union": union,
                    "jaccard": round(jac, 4) if union else None,
                })
    return pd.DataFrame(recs), mat, Ls, err_sets


def _plot_churn(mat, Ls, split: str) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(Ls))); ax.set_xticklabels([f"{L}M" for L in Ls])
    ax.set_yticks(range(len(Ls))); ax.set_yticklabels([f"{L}M" for L in Ls])
    for i in range(len(Ls)):
        for j in range(len(Ls)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        color="white" if mat[i, j] < 0.55 else "black", fontsize=9)
    ax.set_title(f"错例集 Jaccard 重叠({'temporal' if split=='Temporal' else 'dev OOF'})\n"
                 "高=同一批人一直错(顽固);低=错例轮换", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Jaccard")
    fig.tight_layout()
    fig.savefig(FIG / "xland_Figure_03_ErrorChurnJaccard.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke: 6M only (trajectory degenerate)")
    args = ap.parse_args()
    ensure_dirs()

    long_df, meta = build_tracking_long(quick=args.quick)
    landmarks = meta["landmarks"]

    # patterns (temporal + dev OOF)
    pat_frames = []
    pat_detail = {}
    for split in ("Temporal", "Development"):
        pf, detail = pattern_table(long_df, split, landmarks)
        pat_frames.append(pf)
        pat_detail[split] = detail
    pat_df = pd.concat(pat_frames, ignore_index=True)
    pat_df.to_csv(TBL / "xland_trajectory_patterns.csv", index=False)

    # stubborn vs accumulation
    stub = pd.concat(
        [stubborn_vs_accumulation(long_df, s, landmarks) for s in ("Temporal", "Development")],
        ignore_index=True,
    )
    stub.to_csv(TBL / "xland_stubborn_vs_accumulation.csv", index=False)

    # abs-err curve
    curve = abs_err_curve(long_df, landmarks)
    curve.to_csv(TBL / "xland_abs_err_curve.csv", index=False)
    if len(landmarks) > 1:
        _plot_abs_err_curve(curve)

    # error churn
    churn_t, mat_t, Ls, _ = error_churn(long_df, "Temporal", landmarks)
    churn_d, mat_d, _, _ = error_churn(long_df, "Development", landmarks)
    churn = pd.concat([churn_t, churn_d], ignore_index=True)
    churn.to_csv(TBL / "xland_error_churn_jaccard.csv", index=False)
    if len(landmarks) > 1:
        _plot_churn(mat_t, Ls, "Temporal")

    # ---- report ----
    print("=== (1) 对错模式分布 (同人跨地标 EBM 正确性) ===", flush=True)
    for split in ("Temporal", "Development"):
        print(f"\n  [{split}]", flush=True)
        print(pat_df[pat_df["split"] == split].to_string(index=False), flush=True)

    print("\n=== (2) 顽固难例 vs 信号积累型 (首地标即错者的去向) ===", flush=True)
    print(stub.to_string(index=False), flush=True)

    print("\n=== (3) |p−Y| 随地标 (EBM vs persistence vs naive) ===", flush=True)
    print(curve.to_string(index=False), flush=True)

    print("\n=== (4) 错例 churn — Jaccard 重叠 (temporal) ===", flush=True)
    print(churn_t.to_string(index=False), flush=True)

    summary = {
        "口径": "corrected+LOCF, EBM-OOF dev / temporal read-out, time-safe",
        "analysis_unit": "treatment-episode (人次)",
        "landmarks": [f"{L}M" for L in landmarks],
        "patterns": pat_df.to_dict(orient="records"),
        "stubborn_vs_accumulation": stub.to_dict(orient="records"),
        "abs_err_curve": curve.to_dict(orient="records"),
        "error_churn_jaccard_temporal": churn_t.to_dict(orient="records"),
        "error_churn_jaccard_dev": churn_d.to_dict(orient="records"),
        "note": "Patterns/stubborn computed per observed-landmark sequence; "
        "Temporal numbers are read-out only.",
    }
    (TBL / "xland_trajectory_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(f"\nSaved → {TBL}  &  {FIG}", flush=True)


if __name__ == "__main__":
    main()
