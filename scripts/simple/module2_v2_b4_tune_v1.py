#!/usr/bin/env python
"""B4 · V1 vanilla GRU tuning probe.

Answers "can V1 be tuned further?" empirically with a 2×2 grid:
  {hidden 12, 24} × {no Eval, +Eval immune-state features}.

The +Eval arm appends per-landmark Eval_{1M,3M}_{Hyper,Normal,Hypo,Missing}
(step 1 = 1M, step 2 = 3M; 0M/6M zero — no Eval there) as a new dynamic block
"F", folded into the GRU's per-step input. Time-safe: step t sees only its own
landmark's Eval. This is the single feature that gives per-LR its 3M edge
(per-LR+Eval 3M ROC 0.847 vs B4 V1 3M 0.762).

All configs reuse the B4 harness (5-seed ensemble, augmentation, per-landmark
Platt on pooled OOF, episode-cluster CV). OOF (dev) for selection; temporal only
reported.
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.simple.module2_v2_shared import (
    STAGE2_LONG,
    apply_per_landmark_platt,
    load_stacked,
    per_landmark_and_pooled_perf,
    per_landmark_platt_on_pooled_oof,
)
from scripts.simple.module2_v2_vertical_b4_gru import (
    SEEDS,
    build_timeaware_tensors,
    place_step_predictions_into_rows,
    train_target_gru,
)

OUT = ROOT / "results" / "module2_v2_vertical" / "b4_target_gru" / "tables"
EVAL_CATS = ("Hyper", "Normal", "Hypo", "Missing")


def load_eval_per_step(episodes: np.ndarray) -> np.ndarray:
    """Return (n_ep, 4, 4) per-landmark Eval features aligned to `episodes`.

    step 1 <- Eval_1M_*, step 2 <- Eval_3M_*; steps 0/3 (0M/6M) zero.
    """
    df = pd.read_csv(STAGE2_LONG, low_memory=False)
    one = (df.drop_duplicates("Treatment_ID", keep="first")
             .sort_values("Treatment_Index").reset_index(drop=True))
    one["Episode_Index"] = one["Treatment_Index"].astype(int)
    cols1 = [f"Eval_1M_{c}" for c in EVAL_CATS]
    cols3 = [f"Eval_3M_{c}" for c in EVAL_CATS]
    for c in cols1 + cols3:
        if c not in one.columns:
            one[c] = 0.0
    idx = {int(e): i for i, e in enumerate(one["Episode_Index"].values)}
    v1 = one[cols1].fillna(0.0).values.astype(np.float32)
    v3 = one[cols3].fillna(0.0).values.astype(np.float32)
    ev = np.zeros((len(episodes), 4, 4), dtype=np.float32)
    for i, e in enumerate(episodes):
        r = idx[int(e)]
        ev[i, 1, :] = v1[r]   # 1M
        ev[i, 2, :] = v3[r]   # 3M
    return ev


def run_config(sd, tensors, *, use_eval: bool, hidden: int):
    seq_X, dt, mask, seq_y, is_dev, episodes, _, block_idx = tensors
    if use_eval:
        ev = load_eval_per_step(episodes)
        seq_X = np.concatenate([seq_X, ev], axis=2)
        base = block_idx_width(block_idx, tensors)
        block_idx = dict(block_idx)
        block_idx["F"] = [base + k for k in range(4)]
    proba_step = train_target_gru(
        seq_X, dt, mask, seq_y, is_dev, episodes, block_idx,
        hidden=hidden, use_decay=False, zoneout=0.1, augment=True, seeds=SEEDS,
    )
    raw = place_step_predictions_into_rows(sd, proba_step, episodes)
    cal = apply_per_landmark_platt(sd, raw, per_landmark_platt_on_pooled_oof(sd, raw))
    tmp = per_landmark_and_pooled_perf(sd, cal, split="Temporal")
    dev = per_landmark_and_pooled_perf(sd, cal, split="Development")
    g = lambda d, L, col="ROC_AUC": float(d.set_index("Landmark").loc[L, col])
    return {
        "use_eval": use_eval, "hidden": hidden,
        "dev_pooled": g(dev, "Pooled"),
        "tmp_pooled": g(tmp, "Pooled"),
        "tmp_3M": g(tmp, "3M"), "tmp_6M": g(tmp, "6M"),
        "tmp_0M": g(tmp, "0M"), "tmp_1M": g(tmp, "1M"),
        "calib_slope_pooled": g(tmp, "Pooled", "CalibSlope"),
    }


def block_idx_width(block_idx, tensors) -> int:
    # number of feature columns in seq_X before appending F
    return tensors[0].shape[2]


def main() -> None:
    print("Loading stacked dataset…", flush=True)
    sd = load_stacked()
    tensors = build_timeaware_tensors(sd)
    print(f"  tensor {tensors[0].shape}; running V1 2×2 tuning grid (5-seed each)…", flush=True)

    rows = []
    for use_eval in (False, True):
        for hidden in (12, 24):
            tag = f"V1 {'+' if use_eval else 'no'}Eval h{hidden}"
            print(f"\n[{tag}] …", flush=True)
            r = run_config(sd, tensors, use_eval=use_eval, hidden=hidden)
            r["config"] = tag
            rows.append(r)
            print(f"  dev {r['dev_pooled']:.4f} | tmp pooled {r['tmp_pooled']:.4f} | "
                  f"3M {r['tmp_3M']:.4f} | 6M {r['tmp_6M']:.4f} | slope {r['calib_slope_pooled']:.3f}",
                  flush=True)

    df = pd.DataFrame(rows)[
        ["config", "use_eval", "hidden", "dev_pooled", "tmp_pooled",
         "tmp_0M", "tmp_1M", "tmp_3M", "tmp_6M", "calib_slope_pooled"]
    ]
    df.to_csv(OUT / "v1_tuning_grid.csv", index=False)
    print("\n=== V1 tuning grid (temporal) ===", flush=True)
    print(df.to_string(index=False), flush=True)
    print(f"\nSaved → {OUT / 'v1_tuning_grid.csv'}", flush=True)


if __name__ == "__main__":
    main()
