#!/usr/bin/env python
"""M2 v2 Eval-state probe: add immune-state encodings to per-landmark 4-LR.

The v1 4-LR architecture used Eval_L_{Hyper,Normal,Hypo} state encodings
that the v2 ABCDE feature set drops. This probe re-adds them (where data
exists — note: stage2 source has Eval_6M = 0 entirely, only 1M/3M have
real values) to see whether 1M/3M ROC recovers toward v1's higher numbers.

Outputs to results/module2_v2_base/ with `_evalstate` suffix to distinguish
from the standard ABCDE results.
"""
from __future__ import annotations

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

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.simple.module2_v2_shared import (
    LANDMARKS,
    apply_per_landmark_platt,
    calib_intercept_slope,
    compute_metrics,
    episode_cluster_bootstrap_auc_ci,
    forbidden_token,
    load_stacked,
    per_landmark_platt_on_pooled_oof,
    STAGE2_LONG,
)

OUT_DIR = ROOT / "results" / "module2_v2_base"
TAB_DIR = OUT_DIR / "tables"


def load_eval_state_per_episode() -> pd.DataFrame:
    df = pd.read_csv(STAGE2_LONG, low_memory=False)
    one_per = (
        df.drop_duplicates("Treatment_ID", keep="first")
        .sort_values("Treatment_Index")
        .reset_index(drop=True)
    )
    cols = []
    for L in (1, 3):  # 6M is all zero in this source — exclude
        for cat in ("Hyper", "Normal", "Hypo", "Missing"):
            cols.append(f"Eval_{L}M_{cat}")
    keep = ["Treatment_Index"] + cols
    out = one_per[keep].copy()
    out["Episode_Index"] = out["Treatment_Index"].astype(int)
    return out


def run_per_landmark(blocks_features: list[str], landmark_specific_features: dict[int, list[str]],
                      sd, eval_df) -> tuple[np.ndarray, dict]:
    """Per-landmark 4-LR fit with global ABCDE + landmark-specific Eval state extras.

    For each row, look up the Eval state columns relevant to that landmark
    from the per-episode eval_df and append to the feature matrix.
    """
    # Build augmented feature matrix per landmark
    X_base = sd.feature_matrix(("A", "B", "C", "D", "E"))
    X_base["episode_id"] = sd.rows["episode_id"].values
    X_base["landmark"] = sd.rows["landmark"].values
    X_base = X_base.merge(eval_df, left_on="episode_id", right_on="Episode_Index", how="left")
    X_base = X_base.drop(columns=["Episode_Index", "Treatment_Index"])
    # Replace any NaN from merge
    X_base = X_base.fillna(0)

    y = sd.rows["Y_24M_NHRH"].values
    split = sd.rows["Split"].values
    lm = sd.rows["landmark"].values
    proba = np.zeros(len(sd.rows))

    for L in LANDMARKS:
        feats = list(blocks_features) + landmark_specific_features.get(L, [])
        mask = lm == L
        dev = mask & (split == "Development")
        test = mask & (split == "Temporal")
        Xm = X_base[feats].values
        y_d = y[dev]
        X_d = Xm[dev]
        dev_idx = np.where(dev)[0]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for tr, va in skf.split(X_d, y_d):
            pipe = Pipeline([("s", StandardScaler()),
                              ("lr", LogisticRegression(penalty="l2", solver="lbfgs",
                                                         C=1.0, max_iter=5000, random_state=2025))])
            pipe.fit(X_d[tr], y_d[tr])
            proba[dev_idx[va]] = pipe.predict_proba(X_d[va])[:, 1]
        if test.any():
            final = Pipeline([("s", StandardScaler()),
                               ("lr", LogisticRegression(penalty="l2", solver="lbfgs",
                                                          C=1.0, max_iter=5000, random_state=2025))])
            final.fit(X_d, y_d)
            proba[test] = final.predict_proba(Xm[test])[:, 1]
    return proba, X_base


def main() -> None:
    tok = forbidden_token()
    if tok in Path(__file__).read_text():
        raise RuntimeError(f"Forbidden literal '{tok}' present in source.")
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading stacked dataset + Eval state …", flush=True)
    sd = load_stacked()
    eval_df = load_eval_state_per_episode()
    print(f"  Eval state columns added: {[c for c in eval_df.columns if c.startswith('Eval_')]}",
          flush=True)

    blocks_features = ["Sex", "ThyroidW", "TRAb", "TGAb", "TPOAb", "FT4_0M", "TSH_0M",
                        "log1p_DiseaseDuration_Months_Aug",
                        "Uptake24h", "HalfLife",
                        "TSH_current", "FT4_current",
                        "TSH_velocity", "FT4_velocity", "velocity_observed",
                        "landmark_time_centered", "is_baseline_landmark",
                        "lm_x_TSH_current", "lm_x_FT4_current",
                        "lm_x_TSH_velocity", "lm_x_FT4_velocity"]

    # Per-landmark Eval state additions
    landmark_specific = {
        0: [],  # baseline: no Eval state
        1: ["Eval_1M_Hyper", "Eval_1M_Normal", "Eval_1M_Hypo", "Eval_1M_Missing"],
        3: ["Eval_3M_Hyper", "Eval_3M_Normal", "Eval_3M_Hypo", "Eval_3M_Missing"],
        6: [],  # 6M Eval state all zero in this source
    }

    print("\n[Probe] Per-landmark 4-LR + ABCDE + Eval state (1M/3M only) …", flush=True)
    proba, _ = run_per_landmark(blocks_features, landmark_specific, sd, eval_df)
    cal = per_landmark_platt_on_pooled_oof(sd, proba)
    p_cal = apply_per_landmark_platt(sd, proba, cal)

    # Per-landmark + pooled performance
    is_test = sd.rows["Split"].values == "Temporal"
    perf_rows = []
    for label, mask in [(f"{L}M", is_test & (sd.rows["landmark"].values == L)) for L in LANDMARKS] + \
                       [("Pooled", is_test)]:
        y = sd.rows.loc[mask, "Y_24M_NHRH"].values
        p = p_cal[mask]
        if len(np.unique(y)) < 2:
            continue
        m = compute_metrics(y, p)
        ic, sl = calib_intercept_slope(y, p)
        L_int = None if label == "Pooled" else int(label[:-1])
        mu, lo, hi = episode_cluster_bootstrap_auc_ci(sd, p_cal, landmark=L_int, split="Temporal")
        perf_rows.append({"Landmark": label, "N": int(mask.sum()), "Events": int(y.sum()),
                           **m, "ROC_AUC_CI_Low": lo, "ROC_AUC_CI_High": hi,
                           "CalibIntercept": ic, "CalibSlope": sl})
    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(TAB_DIR / "final_4lr_abcde_plus_eval_perf.csv", index=False)
    summary = {
        "feature_set": "ABCDE + Eval_1M_{Hyper,Normal,Hypo,Missing} + Eval_3M_{Hyper,Normal,Hypo,Missing}",
        "note": "Eval_6M not added — stage2 source has Eval_6M_* all zero (data missing for 6M Eval). Iter 3 candidate: source from v1 raw data.",
        "performance": perf_df.to_dict(orient="records"),
    }
    (TAB_DIR / "final_4lr_abcde_plus_eval_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== Per-landmark + pooled performance ===")
    print(perf_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
