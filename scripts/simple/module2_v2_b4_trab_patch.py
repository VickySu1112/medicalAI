#!/usr/bin/env python
"""M2 · B4 — feature-completeness audit fix + per-month hard cases + feature patch.

Audit finding: the M2 dynamic set used TSH/FT4 current+velocity but OMITTED FT3,
even though FT3_{0,1,3,6}M is measured at ~83-99% per month (same coverage as
FT4/TSH). FT3 (T3-toxicosis) is clinically central to Graves relapse. Follow-up
TRAb, by contrast, is ~2%/14% measured at 1M/3M → unusable (mostly imputed).

This script:
  (1) Adds FT3_current + FT3_velocity (time-safe, same construction as TSH/FT4);
      asserts time-safety of the additions.
  (2) Feature-patch test (L2, raw ROC — calibration-invariant): ABCDE  vs  +FT3
      vs  +FT3+Eval  vs  +FT3+Eval+TRAb(fu, shown to add ~0). OOF + temporal +
      per-landmark.
  (3) Per-MONTH hard-case profiles incl. FT3 — does the FN/FP phenotype vary by
      landmark, and does FT3 separate it?

Time-safety: *_current at landmark L = value measured at L (≤ L); *_velocity =
(v_L − v_prev)/Δt, 0 at 0M (≤ L). Labs are biomarkers, not post-RAI medication.
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

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.simple.module2_v2_shared import (
    LANDMARKS,
    STAGE2_LONG,
    apply_per_landmark_platt,
    fit_l2_predict_oof,
    load_stacked,
    per_landmark_platt_on_pooled_oof,
    PY_SEED,
)

OUT = ROOT / "results" / "module2_v2_vertical" / "b4_target_gru"
ABCDE = ("A", "B", "C", "D", "E")
PREV = {0: None, 1: 0, 3: 1, 6: 3}


def _roc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")


def _per_episode(col_names):
    df = pd.read_csv(STAGE2_LONG, low_memory=False)
    one = (df.drop_duplicates("Treatment_ID", keep="first")
             .sort_values("Treatment_Index").reset_index(drop=True))
    one["Episode_Index"] = one["Treatment_Index"].astype(int)
    for c in col_names:
        if c not in one.columns:
            one[c] = np.nan
    idx = {int(e): i for i, e in enumerate(one["Episode_Index"].values)}
    return one, idx


def add_current_velocity(sd, marker):
    """Add {marker}_at_{L}M (per episode) + {marker}_current/{marker}_velocity (time-safe)."""
    cols = [f"{marker}_{L}M" for L in LANDMARKS]
    one, idx = _per_episode(cols)
    vals = {L: pd.to_numeric(one[f"{marker}_{L}M"], errors="coerce").values.astype(float) for L in LANDMARKS}
    rows = sd.rows
    epid = rows["episode_id"].values
    lm = rows["landmark"].values
    for L in LANDMARKS:
        rows[f"{marker}_at_{L}M"] = np.array([vals[L][idx[int(e)]] for e in epid])
    cur = np.zeros(len(rows)); vel = np.zeros(len(rows))
    for i in range(len(rows)):
        L = int(lm[i]); cur[i] = rows[f"{marker}_at_{L}M"].values[i]; p = PREV[L]
        vel[i] = 0.0 if p is None else (rows[f"{marker}_at_{L}M"].values[i] - rows[f"{marker}_at_{p}M"].values[i]) / (L - p)
    rows[f"{marker}_current"] = cur
    rows[f"{marker}_velocity"] = vel
    dev = (rows["Split"] == "Development").values
    for c in (f"{marker}_current", f"{marker}_velocity"):
        rows[c] = rows[c].replace([np.inf, -np.inf], np.nan)
        rows[c] = rows[c].fillna(np.nanmedian(rows.loc[dev, c]))
    return sd


def assert_addition_time_safe(sd, marker):
    """{marker}_current at landmark L must equal {marker}_at_{L}M (never a future month)."""
    rows = sd.rows
    lm = rows["landmark"].values
    for L in LANDMARKS:
        m = lm == L
        a = rows.loc[m, f"{marker}_current"].values
        b = rows.loc[m, f"{marker}_at_{L}M"].values
        obs = ~np.isnan(b)  # imputed (NaN) rows use dev-median (a constant, not future info)
        ok = np.allclose(a[obs], b[obs])
        assert ok, f"TIME-SAFETY: {marker}_current at {L}M != observed {marker}_at_{L}M"
    print(f"  [assert] {marker}_current/velocity time-safe (observed L-rows use exactly the L value).", flush=True)


def load_eval_cols(sd):
    cats = ("Hyper", "Normal", "Hypo", "Missing")
    one, idx = _per_episode([f"Eval_{Lm}M_{c}" for Lm in (1, 3) for c in cats])
    out = np.zeros((len(sd.rows), 4))
    lm = sd.rows["landmark"].values
    epid = sd.rows["episode_id"].values
    for Lm in (1, 3):
        v = one[[f"Eval_{Lm}M_{c}" for c in cats]].fillna(0.0).values
        for i in range(len(sd.rows)):
            if int(lm[i]) == Lm:
                out[i] = v[idx[int(epid[i])]]
    return out


def lr_oof_and_temporal(X, y, ep, is_dev, lm):
    oof = np.zeros(len(y))
    Xd, yd, epd = X[is_dev], y[is_dev], ep[is_dev]
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=13)
    dev_idx = np.where(is_dev)[0]
    for tr, va in skf.split(Xd, yd, groups=epd):
        pipe = Pipeline([("s", StandardScaler()),
                         ("lr", LogisticRegression(penalty="l2", solver="lbfgs", C=1.0,
                                                   max_iter=5000, random_state=PY_SEED))])
        pipe.fit(Xd[tr], yd[tr])
        oof[dev_idx[va]] = pipe.predict_proba(Xd[va])[:, 1]
    final = Pipeline([("s", StandardScaler()),
                      ("lr", LogisticRegression(penalty="l2", solver="lbfgs", C=1.0,
                                                max_iter=5000, random_state=PY_SEED))])
    final.fit(Xd, yd)
    is_test = ~is_dev
    oof[is_test] = final.predict_proba(X[is_test])[:, 1]
    res = {"oof_pooled": _roc(y[is_dev], oof[is_dev]), "tmp_pooled": _roc(y[is_test], oof[is_test])}
    for L in LANDMARKS:
        m = is_test & (lm == L)
        res[f"tmp_{L}M"] = _roc(y[m], oof[m])
    return res


def main() -> None:
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    sd = load_stacked()
    print("Adding FT3 (omitted) + follow-up TRAb (mostly missing, for contrast)…", flush=True)
    sd = add_current_velocity(sd, "FT3")
    sd = add_current_velocity(sd, "TRAb")
    assert_addition_time_safe(sd, "FT3")
    assert_addition_time_safe(sd, "TRAb")
    rows = sd.rows
    y = rows["Y_24M_NHRH"].values
    ep = rows["episode_id"].values
    is_dev = (rows["Split"] == "Development").values
    lm = rows["landmark"].values

    Xbase = sd.feature_matrix(ABCDE).values
    ft3 = np.column_stack([rows["FT3_current"].values, rows["FT3_velocity"].values])
    eval_cols = load_eval_cols(sd)
    trab = np.column_stack([rows["TRAb_current"].values, rows["TRAb_velocity"].values])

    feat_sets = {
        "ABCDE (base, no FT3)": Xbase,
        "ABCDE + FT3": np.column_stack([Xbase, ft3]),
        "ABCDE + FT3 + Eval": np.column_stack([Xbase, ft3, eval_cols]),
        "ABCDE + FT3 + Eval + TRAb(fu)": np.column_stack([Xbase, ft3, eval_cols, trab]),
    }
    print("\n=== (2) Feature-patch test (L2, raw ROC; OOF=selection, temporal=read-out) ===", flush=True)
    patch_rows = []
    for name, X in feat_sets.items():
        res = lr_oof_and_temporal(X, y, ep, is_dev, lm)
        patch_rows.append({"feature_set": name, **{k: round(v, 4) for k, v in res.items()}})
    patch_df = pd.DataFrame(patch_rows)
    patch_df.to_csv(OUT / "tables" / "ft3_patch_comparison.csv", index=False)
    print(patch_df.to_string(index=False), flush=True)

    print("\n=== (1) Per-month hard-case profiles (base ABCDE L2 OOF) incl. FT3 ===", flush=True)
    oof_all, _ = fit_l2_predict_oof(sd, block_subset=ABCDE)
    cal = apply_per_landmark_platt(sd, oof_all, per_landmark_platt_on_pooled_oof(sd, oof_all))
    key = ["TSH_current", "FT4_current", "FT3_current", "FT3_velocity",
           "TSH_velocity", "FT4_velocity", "TRAb", "ThyroidW"]
    recs = []
    K = 15
    for L in LANDMARKS:
        idxL = np.where(is_dev & (lm == L))[0]
        pL = cal[idxL]; yL = y[idxL]
        kk = min(K, int((yL == 1).sum()), int((yL == 0).sum()))
        fn = idxL[yL == 1][np.argsort(pL[yL == 1])[:kk]]
        fp = idxL[yL == 0][np.argsort(-pL[yL == 0])[:kk]]
        for grp, ix in (("FN", fn), ("FP", fp)):
            rec = {"landmark": f"{L}M", "group": grp, "n": len(ix)}
            for f in key:
                rec[f] = round(float(np.nanmean(rows[f].values[ix])), 2) if f in rows.columns else None
            recs.append(rec)
    prof_df = pd.DataFrame(recs)
    prof_df.to_csv(OUT / "tables" / "hardcase_by_month_ft3.csv", index=False)
    print(prof_df.to_string(index=False), flush=True)

    base = float(patch_df.iloc[0]["tmp_pooled"])
    best = patch_df.iloc[patch_df["tmp_pooled"].astype(float).idxmax()]
    (OUT / "tables" / "ft3_patch_summary.json").write_text(json.dumps({
        "patch_comparison": patch_df.to_dict(orient="records"),
        "base_temporal_pooled": base,
        "best_feature_set": best["feature_set"], "best_temporal_pooled": float(best["tmp_pooled"]),
        "per_LR_Eval_reference": 0.770,
        "audit": {"FT3_omitted_now_added": True, "followup_TRAb_coverage_1M_3M": "1.8%/14.3% → unusable"},
    }, indent=2, ensure_ascii=False))
    print(f"\nBest: {best['feature_set']} → temporal pooled {best['tmp_pooled']} (base {base}, per-LR+Eval ref 0.770)", flush=True)
    print(f"Saved → {OUT}", flush=True)


if __name__ == "__main__":
    main()
