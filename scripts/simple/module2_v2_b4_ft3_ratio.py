#!/usr/bin/env python
"""M2 · B4 — redo per-landmark importance WITH FT3, and test FT3/FT4 combinations.

The user asked: redo the per-month importance table including FT3, and is there a
way to *combine* FT3+FT4?

Key idea: FT3 and FT4 are ~0.92 correlated, so sum / mean / PCA-1 of (FT3,FT4)
just re-captures the shared "hormone level" axis (≈ FT4) and adds nothing. The
only combination that carries NEW information is the one orthogonal to that axis —
the FT3/FT4 RATIO (T3-predominance; clinically the T3-toxicosis marker). This
script:
  (1) per-landmark L2 importance on A+B+C+D + FT3_current + FT3_velocity +
      FT3/FT4 ratio (redo the table, see where FT3 / the ratio rank each month);
  (2) feature-patch test: ABCDE vs +FT3 vs +FT3/FT4-ratio vs +ratio+Eval.

All time-safe: current/ratio use only the landmark-L value.
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.simple.module2_v2_shared import BLOCKS, LANDMARKS, PRETTY, load_stacked, PY_SEED
from scripts.simple.module2_v2_b4_trab_patch import add_current_velocity, load_eval_cols, lr_oof_and_temporal, ABCDE

OUT = ROOT / "results" / "module2_v2_vertical" / "b4_target_gru"
PRETTY = {**PRETTY, "FT3_current": "FT3 at current landmark", "FT3_velocity": "FT3 velocity (per month)",
          "FT3_over_FT4": "FT3/FT4 ratio (T3 dominance)"}


def add_ratio(sd):
    rows = sd.rows
    ft3 = rows["FT3_current"].values.astype(float)
    ft4 = rows["FT4_current"].values.astype(float)
    denom = np.where(np.abs(ft4) < 0.1, np.nan, ft4)   # guard the 6M FT4≈0 quirk
    ratio = ft3 / denom
    dev = (rows["Split"] == "Development").values
    ratio = np.where(np.isfinite(ratio), ratio, np.nan)
    hi = np.nanpercentile(ratio[dev], 99)
    ratio = np.clip(ratio, 0, hi)
    ratio = np.where(np.isnan(ratio), np.nanmedian(ratio[dev]), ratio)
    rows["FT3_over_FT4"] = ratio
    return sd


def per_landmark_importance(sd, feats):
    rows = sd.rows
    y = rows["Y_24M_NHRH"].values
    lm = rows["landmark"].values
    is_dev = (rows["Split"] == "Development").values
    X = rows[feats].values
    out = {}
    for L in LANDMARKS:
        m = is_dev & (lm == L)
        pipe = Pipeline([("s", StandardScaler()),
                         ("lr", LogisticRegression(penalty="l2", solver="lbfgs", C=1.0,
                                                   max_iter=5000, random_state=PY_SEED))])
        pipe.fit(X[m], y[m])
        coef = pipe.named_steps["lr"].coef_[0]
        order = np.argsort(-np.abs(coef))
        out[L] = [(feats[i], round(float(coef[i]), 3)) for i in order]
    return out


def main() -> None:
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    sd = load_stacked()
    sd = add_current_velocity(sd, "FT3")
    sd = add_ratio(sd)
    rows = sd.rows
    y = rows["Y_24M_NHRH"].values
    ep = rows["episode_id"].values
    is_dev = (rows["Split"] == "Development").values
    lm = rows["landmark"].values

    # (1) redo per-landmark importance WITH FT3 + ratio
    feats = BLOCKS["A"] + BLOCKS["B"] + BLOCKS["C"] + BLOCKS["D"] + ["FT3_current", "FT3_velocity", "FT3_over_FT4"]
    imp = per_landmark_importance(sd, feats)
    print("=== (1) Per-landmark importance WITH FT3 + FT3/FT4 ratio (top-6 |coef|) ===", flush=True)
    recs = []
    for L in LANDMARKS:
        print(f"\n  {L}M:")
        for rank, (f, c) in enumerate(imp[L][:6], 1):
            tag = "  <==" if ("FT3" in f) else ""
            print(f"    {rank}. {PRETTY.get(f, f):<32} {c:+.3f}{tag}", flush=True)
            recs.append({"landmark": f"{L}M", "rank": rank, "feature": PRETTY.get(f, f), "coef": c})
        # also report explicit ranks of the FT3 features
        for f in ("FT3_current", "FT3_velocity", "FT3_over_FT4"):
            r = [k for k, (ff, _) in enumerate(imp[L], 1) if ff == f][0]
            print(f"      · {PRETTY.get(f, f)} 排第 {r}/{len(feats)}", flush=True)
    pd.DataFrame(recs).to_csv(OUT / "tables" / "per_landmark_importance_with_ft3.csv", index=False)

    # (2) feature-patch: does FT3 or the ratio add anything?
    Xbase = sd.feature_matrix(ABCDE).values
    ft3 = np.column_stack([rows["FT3_current"].values, rows["FT3_velocity"].values])
    ratio = rows["FT3_over_FT4"].values.reshape(-1, 1)
    eval_cols = load_eval_cols(sd)
    sets = {
        "ABCDE (base)": Xbase,
        "ABCDE + FT3 (level)": np.column_stack([Xbase, ft3]),
        "ABCDE + FT3/FT4 ratio": np.column_stack([Xbase, ratio]),
        "ABCDE + FT3 + ratio": np.column_stack([Xbase, ft3, ratio]),
        "ABCDE + FT3 + ratio + Eval": np.column_stack([Xbase, ft3, ratio, eval_cols]),
    }
    print("\n=== (2) Patch test — does combining FT3+FT4 (ratio) add anything? ===", flush=True)
    rows_out = []
    for name, X in sets.items():
        res = lr_oof_and_temporal(X, y, ep, is_dev, lm)
        rows_out.append({"feature_set": name, **{k: round(v, 4) for k, v in res.items()}})
    patch = pd.DataFrame(rows_out)
    patch.to_csv(OUT / "tables" / "ft3_ratio_patch.csv", index=False)
    print(patch.to_string(index=False), flush=True)

    (OUT / "tables" / "ft3_ratio_summary.json").write_text(json.dumps({
        "per_landmark_importance_with_ft3": recs,
        "patch": patch.to_dict(orient="records"),
        "ft3_ft4_corr_pooled": 0.916,
    }, indent=2, ensure_ascii=False))
    print(f"\nSaved → {OUT}", flush=True)


if __name__ == "__main__":
    main()
