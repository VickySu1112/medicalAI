#!/usr/bin/env python
"""M2 · B4 — time-safe imputation-method experiment (corrected truth口径).

Compares four imputers for the genuinely-missing 6M/12M hormone levels (follow-up
drop-out: ~25% at 6M, ~40% at 12M) under the corrected truth口径, per landmark:

    (a) median   — dev-median of the column at L (baseline)
    (b) locf     — last-observation-carried-forward: carry the most recent EARLIER
                   landmark's true value (≤ L); residual NaN → dev-median
    (c) missforest — IterativeImputer(RandomForestRegressor), dev-fit, ≤L features
    (d) knn      — KNNImputer, dev-fit, ≤L features

Every method is strictly time-safe:
  * the imputer is FIT on Development rows at landmark L only;
  * the feature matrix at L contains only ≤L information (statics + current@L +
    velocity ending at L), so no column can leak a month > L;
  * LOCF only looks backward.

For each (method × landmark) we run the SAME validated EBM primitive
(ebm_oof_and_temporal on the orthogonal axes) and report:
  OOF AUC, temporal AUC, temporal Brier, + episode-cluster bootstrap 95% CI on
  temporal AUC, + paired bootstrap ΔAUC vs the median baseline (CI crossing /
  excluding 0 → recommendation language; never the word "significant").

The corrected current/velocity used here come from the data-layer single source
of truth (_corrected_landmark_values), but with the degenerate 6M/12M wide-column
ZERO-fallback removed so genuinely-missing cells are real NaN that the imputers
act on (the mainline load_stacked_x keeps the legacy zero-fill for byte-for-byte
reproduction of the 0.873/0.884 reference).

Outputs:
    results/module2_v2_vertical/m2v2_ebm_xland/tables/impute_method_comparison.csv
    results/module2_v2_vertical/m2v2_ebm_xland/tables/impute_method_summary.json
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
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import IterativeImputer, KNNImputer

from scripts.simple.module2_v2_shared import (
    STAGE2_LONG,
    LANDMARKS_X,
    _prev_landmark_map,
    brier_score_loss,
    episode_cluster_bootstrap_auc_ci,
    load_stacked_x,
    paired_episode_cluster_bootstrap_delta,
)
from scripts.simple.module2_v2_b4_ebm_axes import build_feats_at_L
from scripts.simple.module2_v2_b4_ebm_oof import ebm_oof_and_temporal, _roc

OUT = ROOT / "results" / "module2_v2_vertical" / "m2v2_ebm_xland"
MARKERS = ("FT3", "FT4", "TSH")
CUR_COLS = [f"{m}_current" for m in MARKERS]
VEL_COLS = [f"{m}_velocity" for m in MARKERS]
STATIC = ["ThyroidW", "TRAb", "TGAb", "TPOAb", "Sex", "FT4_0M", "TSH_0M",
          "log1p_DiseaseDuration_Months_Aug", "Uptake24h", "HalfLife"]
# Wide {marker}_{L}M is a real measurement on the first row only for L∈{0,1,3};
# at 6M/12M it is ZERO-degenerate and must be treated as missing.
WIDE_OK = {0: True, 1: True, 3: True, 6: False, 12: False}
# Core required methods (task A–B brief) + worklog-requested "let the model
# understand missingness" comparators (informative missingness hypothesis).
METHODS = ("median", "locf", "missforest", "knn")
EXTRA_METHODS = ("ebm_native", "median_indicator")
ALL_METHODS = METHODS + EXTRA_METHODS


# ---------------------------------------------------------------------------
# Proper-NaN corrected truth (no degenerate 6M/12M zero-fallback)
# ---------------------------------------------------------------------------

_TRUTH_CACHE: dict = {}
_SKELETON = None


def _skeleton_rows():
    """Cached corrected load_stacked_x() rows (read once; callers .copy())."""
    global _SKELETON
    if _SKELETON is None:
        _SKELETON = load_stacked_x().rows
    return _SKELETON


def _raw_truth_proper_nan(landmarks):
    """Per-(marker, L) corrected level with genuine NaN for follow-up drop-out.

    Identical to the data-layer corrected pull EXCEPT the wide-column fallback is
    only used where it is a real measurement (L∈{0,1,3}); at 6M/12M a missing
    Current_Time row stays NaN (the wide *_6M/*_12M cell is zero-degenerate).
    Returns ``{marker: {L: array indexed by episode-order}}`` + (episodes, split).

    Cached per landmark-set (the long table is re-read once, not per call) to keep
    the per-landmark/per-method loops from thrashing I/O.
    """
    key = tuple(sorted(set(landmarks)))
    if key in _TRUTH_CACHE:
        return _TRUTH_CACHE[key]
    df = pd.read_csv(STAGE2_LONG, low_memory=False)
    df["Treatment_Index"] = pd.to_numeric(df["Treatment_Index"], errors="coerce").astype(int)
    meta = (
        df.drop_duplicates("Treatment_Index", keep="first")
        .sort_values("Treatment_Index")[["Treatment_Index", "Split"]]
    )
    episodes = meta["Treatment_Index"].astype(int).values
    split = meta["Split"].values
    pos = {int(e): i for i, e in enumerate(episodes)}
    first = df.drop_duplicates("Treatment_Index", keep="first")
    out = {m: {} for m in MARKERS}
    for marker in MARKERS:
        for L in sorted(set(landmarks)):
            vals = np.full(len(episodes), np.nan, dtype=float)
            sub = df[df["Current_Time"].astype(str) == f"{L}M"].drop_duplicates(
                "Treatment_Index", keep="first"
            )
            cvals = pd.to_numeric(sub[f"{marker}_Current"], errors="coerce").values
            for e, v in zip(sub["Treatment_Index"].astype(int).values, cvals):
                if int(e) in pos:
                    vals[pos[int(e)]] = v
            if WIDE_OK.get(L, False) and f"{marker}_{L}M" in first.columns:
                wv = pd.to_numeric(first[f"{marker}_{L}M"], errors="coerce").values
                for e, v in zip(first["Treatment_Index"].astype(int).values, wv):
                    i = pos.get(int(e))
                    if i is not None and (np.isnan(vals[i]) or vals[i] == 0.0) and not np.isnan(v):
                        vals[i] = v
            out[marker][L] = vals
    _TRUTH_CACHE[key] = (out, episodes, split)
    return out, episodes, split


# ---------------------------------------------------------------------------
# Build a method-specific imputed `rows` (FT3/FT4/TSH current+velocity)
# ---------------------------------------------------------------------------


def build_rows_for_method(method: str, landmarks=LANDMARKS_X):
    """Return a stacked `rows` whose FT3/FT4/TSH current+velocity are imputed by
    `method`, all strictly time-safe (imputer dev-fit, ≤L features only).

    Skeleton (episode_id, landmark, Split, statics, audit columns, Block D/E) comes
    from load_stacked_x; only the hormone current/velocity are re-derived from the
    proper-NaN corrected truth and then imputed.

    Note: the skeleton always carries the full LANDMARKS_X set of rows, so current/
    velocity are constructed over LANDMARKS_X regardless of which landmarks the
    caller will later evaluate; `landmarks` only narrows the per-landmark imputer
    fit/transform loop (a subset still imputes its own rows correctly).
    """
    skeleton_landmarks = tuple(sorted(LANDMARKS_X))
    landmarks = tuple(sorted(landmarks))
    prev = _prev_landmark_map(skeleton_landmarks)
    truth, episodes, _split = _raw_truth_proper_nan(skeleton_landmarks)
    ep_pos = {int(e): i for i, e in enumerate(episodes)}

    rows = _skeleton_rows().copy()  # corrected skeleton (statics, Split, Y, etc.)
    epid = rows["episode_id"].values
    lm = rows["landmark"].values
    is_dev = (rows["Split"] == "Development").values

    # --- raw current (proper NaN) per marker at each row's own landmark ---
    # raw_at spans the full skeleton landmark set (rows include all of them).
    raw_cur = {m: np.full(len(rows), np.nan) for m in MARKERS}
    raw_at = {m: {L: np.array([truth[m][L][ep_pos[int(e)]] for e in epid]) for L in skeleton_landmarks} for m in MARKERS}
    for m in MARKERS:
        col = np.full(len(rows), np.nan)
        for i in range(len(rows)):
            col[i] = raw_at[m][int(lm[i])][i]
        raw_cur[m] = col

    # --- LOCF carry (only for method=locf): fill missing current at L from most
    #     recent EARLIER landmark's true value (≤L) ---
    if method == "locf":
        order = sorted(skeleton_landmarks)
        for m in MARKERS:
            for li, L in enumerate(order):
                if li == 0:
                    continue
                mask = lm == L
                miss = mask & np.isnan(raw_cur[m])
                # carry from nearest earlier landmark with a value
                for j in range(li - 1, -1, -1):
                    Lp = order[j]
                    src = raw_at[m][Lp]
                    fillable = miss & ~np.isnan(src)
                    raw_cur[m][fillable] = src[fillable]
                    miss = mask & np.isnan(raw_cur[m])
                    if not miss.any():
                        break

    # --- velocity (proper NaN) = (cur_L - true_prev) / dt ; uses the (possibly
    #     LOCF-carried) current at L and the true previous-landmark value ---
    raw_vel = {m: np.zeros(len(rows)) for m in MARKERS}
    for m in MARKERS:
        v = np.zeros(len(rows))
        for i in range(len(rows)):
            L = int(lm[i]); p = prev.get(L)
            if p is None:
                v[i] = 0.0
            else:
                cur = raw_cur[m][i]
                pv = raw_at[m][p][i]
                v[i] = (cur - pv) / (L - p) if (not np.isnan(cur) and not np.isnan(pv)) else np.nan
        raw_vel[m] = v

    # write raw (NaN-bearing) current/velocity into rows
    for m in MARKERS:
        rows[f"{m}_current"] = raw_cur[m]
        rows[f"{m}_velocity"] = raw_vel[m]

    # --- per-landmark imputation, dev-fit, ≤L feature matrix only ---
    feat_cols = STATIC + CUR_COLS + VEL_COLS
    for L in landmarks:
        mL = lm == L
        devL = mL & is_dev
        sub = rows.loc[mL, feat_cols].copy()
        sub_dev = rows.loc[devL, feat_cols].copy()
        if method in ("median", "locf"):
            # dev-median per column (residual NaN after LOCF also handled here)
            med = sub_dev.median(numeric_only=True)
            sub = sub.fillna(med)
        elif method == "missforest":
            imp = IterativeImputer(
                estimator=RandomForestRegressor(
                    n_estimators=100, max_depth=None, n_jobs=1, random_state=0
                ),
                max_iter=10, random_state=0, sample_posterior=False,
            )
            imp.fit(sub_dev.values)
            sub = pd.DataFrame(imp.transform(sub.values), index=sub.index, columns=feat_cols)
        elif method == "knn":
            imp = KNNImputer(n_neighbors=10, weights="distance")
            imp.fit(sub_dev.values)
            sub = pd.DataFrame(imp.transform(sub.values), index=sub.index, columns=feat_cols)
        else:
            raise ValueError(f"unknown method {method}")
        # write imputed current/velocity back (statics unchanged)
        for c in CUR_COLS + VEL_COLS:
            rows.loc[mL, c] = sub[c].values

    # final safety: no NaN remaining in the hormone cols (any residual → dev-median)
    for c in CUR_COLS + VEL_COLS:
        if rows[c].isna().any():
            rows[c] = rows[c].fillna(rows.loc[is_dev, c].median())
    return rows


# ---------------------------------------------------------------------------
# Per-method × per-landmark EBM + metrics + bootstrap
# ---------------------------------------------------------------------------


def run_method(method: str, landmarks, n_boot: int):
    rows = build_rows_for_method(method, landmarks)
    y = rows["Y_24M_NHRH"].values
    lm = rows["landmark"].values
    is_dev = (rows["Split"] == "Development").values
    preds = {}  # L -> full-length pred vector (for bootstrap reuse)
    recs = []
    # lightweight StackedData-like shim for the bootstrap helpers
    from scripts.simple.module2_v2_shared import StackedData
    ep_meta = rows.drop_duplicates("episode_id")[["episode_id", "Split", "Y_24M_NHRH"]]
    sd_shim = StackedData(rows=rows, episode_meta=ep_meta)
    for L in landmarks:
        pred, _ebm, live = ebm_oof_and_temporal(rows, y, lm, is_dev, L)
        preds[L] = pred
        devL = is_dev & (lm == L)
        tstL = (~is_dev) & (lm == L)
        oof_auc = _roc(y[devL], pred[devL])
        tmp_auc = _roc(y[tstL], pred[tstL])
        tmp_brier = float(brier_score_loss(y[tstL], pred[tstL])) if tstL.sum() else float("nan")
        am, alo, ahi = episode_cluster_bootstrap_auc_ci(
            sd_shim, pred, landmark=L, split="Temporal", n=n_boot
        )
        recs.append({
            "method": method, "landmark": f"{L}M",
            "n_truth_present": int((~np.isnan(_raw_present(method, rows, lm, L))).sum()),
            "OOF_AUC": round(oof_auc, 4),
            "Temporal_AUC": round(tmp_auc, 4),
            "Temporal_AUC_CI_low": round(alo, 4),
            "Temporal_AUC_CI_high": round(ahi, 4),
            "Temporal_Brier": round(tmp_brier, 4),
            "nlive": len(live),
        })
    return recs, preds, sd_shim


def _raw_present(method, rows, lm, L):
    """Placeholder used only for an informational count; truth-presence is the
    pre-impute non-NaN mask, recomputed cheaply from the proper-NaN source."""
    truth, episodes, _ = _raw_truth_proper_nan((L,))
    ep_pos = {int(e): i for i, e in enumerate(episodes)}
    mL = lm == L
    epid = rows.loc[mL, "episode_id"].values
    # all-three present
    pres = np.array([
        (not np.isnan(truth["FT3"][L][ep_pos[int(e)]]))
        and (not np.isnan(truth["FT4"][L][ep_pos[int(e)]]))
        and (not np.isnan(truth["TSH"][L][ep_pos[int(e)]]))
        for e in epid
    ])
    out = np.where(pres, 1.0, np.nan)
    return out


# ---------------------------------------------------------------------------
# Extra comparators: "let the model understand missingness"
#   ebm_native       — pass NaN current/velocity straight to EBM (own bin)
#   median_indicator — median-impute + binary *_is_missing columns
# These test the informative-missingness hypothesis (follow-up drop-out is not
# random: a missing 6M/12M lab is itself prognostic). All ≤L, dev-fit.
# ---------------------------------------------------------------------------

from interpret.glassbox import ExplainableBoostingClassifier  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from scripts.simple.module2_v2_shared import CV_SEED, PY_SEED, StackedData  # noqa: E402


def _axes_frame(rows, maskL, *, keep_nan: bool):
    """Orthogonal-axis feature frame (same axes as build_feats_at_L). z-fit on
    dev@L ignoring NaN. If keep_nan, NaN stays NaN in the output (EBM bins it
    natively); else NaN→0 after z-scoring. Strictly ≤L."""
    SQRT2 = np.sqrt(2.0)
    ft3 = rows["FT3_current"].values.astype(float); ft4 = rows["FT4_current"].values.astype(float)
    dft3 = rows["FT3_velocity"].values.astype(float); dft4 = rows["FT4_velocity"].values.astype(float)

    def zfit(v):
        mu = np.nanmean(v[maskL]); sd = np.nanstd(v[maskL])
        z = np.zeros_like(v) if (not np.isfinite(sd) or sd < 1e-9) else (v - mu) / sd
        return z if keep_nan else np.nan_to_num(z, nan=0.0)

    z3, z4 = zfit(ft3), zfit(ft4)
    dz3, dz4 = zfit(dft3), zfit(dft4)
    df = pd.DataFrame(index=rows.index)
    for c in STATIC + ["TSH_current", "TSH_velocity"]:
        df[c] = rows[c].values
    df["Hormone_load"] = (z3 + z4) / SQRT2
    df["T3T4_balance"] = (z3 - z4) / SQRT2
    df["Velocity_load"] = (dz3 + dz4) / SQRT2
    df["Velocity_balance"] = (dz3 - dz4) / SQRT2
    return df


def _ebm_oof_temporal_frame(feat_all, y, ep, lm, is_dev, L, seed=PY_SEED):
    """EBM OOF(dev@L) + temporal(final-fit) on a prebuilt feature frame.

    `ep` is the episode_id array aligned to feat_all rows (for grouped CV). NaN is
    allowed (interpret EBM bins it). Columns all-NaN or constant on dev@L dropped.
    """
    lm = np.asarray(lm); is_dev = np.asarray(is_dev); y = np.asarray(y); ep = np.asarray(ep)
    devL = is_dev & (lm == L); tstL = (~is_dev) & (lm == L)
    Xtr_full = feat_all.loc[devL]
    live = [c for c in feat_all.columns
            if Xtr_full[c].notna().any() and (np.nanstd(Xtr_full[c].values) > 1e-9)]
    feat_live = feat_all[live]
    Xd = feat_live.loc[devL].values; yd = y[devL]; epd = ep[devL]
    pred = np.zeros(len(feat_all), dtype=float)
    dev_idx = np.where(devL)[0]
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    for tr, va in skf.split(Xd, yd, groups=epd):
        ebm = ExplainableBoostingClassifier(random_state=seed, interactions=5)
        ebm.fit(Xd[tr], yd[tr])
        pred[dev_idx[va]] = ebm.predict_proba(Xd[va])[:, 1]
    final = ExplainableBoostingClassifier(random_state=seed, interactions=5)
    final.fit(Xd, yd)
    if tstL.any():
        pred[tstL] = final.predict_proba(feat_live.loc[tstL].values)[:, 1]
    return pred, live


def _build_rows_raw_nan(landmarks):
    """build_rows skeleton WITHOUT imputation: FT3/FT4/TSH current+velocity keep
    genuine NaN (for ebm_native and for indicator masks). Also returns per-marker
    raw-NaN current as `{marker}_current_rawnan` for indicator construction."""
    skeleton_landmarks = tuple(sorted(LANDMARKS_X))
    prev = _prev_landmark_map(skeleton_landmarks)
    truth, episodes, _ = _raw_truth_proper_nan(skeleton_landmarks)
    ep_pos = {int(e): i for i, e in enumerate(episodes)}
    rows = _skeleton_rows().copy()
    epid = rows["episode_id"].values; lm = rows["landmark"].values
    raw_at = {m: {L: np.array([truth[m][L][ep_pos[int(e)]] for e in epid]) for L in skeleton_landmarks} for m in MARKERS}
    for m in MARKERS:
        cur = np.array([raw_at[m][int(lm[i])][i] for i in range(len(rows))])
        vel = np.full(len(rows), np.nan)
        for i in range(len(rows)):
            L = int(lm[i]); p = prev.get(L)
            if p is None:
                vel[i] = 0.0
            else:
                pv = raw_at[m][p][i]
                vel[i] = (cur[i] - pv) / (L - p) if (not np.isnan(cur[i]) and not np.isnan(pv)) else np.nan
        rows[f"{m}_current"] = cur
        rows[f"{m}_velocity"] = vel
        rows[f"{m}_current_rawnan"] = cur  # alias for indicator masks
    return rows


def run_method_extra(method: str, landmarks, n_boot: int):
    """Runner for ebm_native / median_indicator (NaN-aware feature path)."""
    if method == "ebm_native":
        rows = _build_rows_raw_nan(landmarks)
        keep_nan = True
    elif method == "median_indicator":
        rows = build_rows_for_method("median", landmarks)          # median-imputed values
        raw = _build_rows_raw_nan(landmarks)                       # raw NaN for masks
        for m in MARKERS:
            rows[f"{m}_current_rawnan"] = raw[f"{m}_current"].values
        keep_nan = False
    else:
        raise ValueError(method)

    y = rows["Y_24M_NHRH"].values
    lm = rows["landmark"].values
    ep = rows["episode_id"].values
    is_dev = (rows["Split"] == "Development").values
    ep_meta = rows.drop_duplicates("episode_id")[["episode_id", "Split", "Y_24M_NHRH"]]
    sd_shim = StackedData(rows=rows, episode_meta=ep_meta)
    truth_cache = {L: _raw_truth_proper_nan((L,)) for L in landmarks}
    preds = {}; recs = []
    for L in landmarks:
        feat_all = _axes_frame(rows, is_dev & (lm == L), keep_nan=keep_nan)
        if method == "median_indicator":
            for m in MARKERS:
                feat_all[f"{m}_is_missing"] = np.isnan(rows[f"{m}_current_rawnan"].values).astype(float)
        pred, live = _ebm_oof_temporal_frame(feat_all, y, ep, lm, is_dev, L)
        preds[L] = pred
        devL = is_dev & (lm == L); tstL = (~is_dev) & (lm == L)
        tr2, eps2, _ = truth_cache[L]
        epp = {int(e): i for i, e in enumerate(eps2)}
        epid = rows.loc[lm == L, "episode_id"].values
        n_pres = int(np.sum([
            (not np.isnan(tr2["FT3"][L][epp[int(e)]])) and (not np.isnan(tr2["FT4"][L][epp[int(e)]]))
            and (not np.isnan(tr2["TSH"][L][epp[int(e)]])) for e in epid]))
        _, alo, ahi = episode_cluster_bootstrap_auc_ci(sd_shim, pred, landmark=L, split="Temporal", n=n_boot)
        recs.append({
            "method": method, "landmark": f"{L}M", "n_truth_present": n_pres,
            "OOF_AUC": round(_roc(y[devL], pred[devL]), 4),
            "Temporal_AUC": round(_roc(y[tstL], pred[tstL]), 4),
            "Temporal_AUC_CI_low": round(alo, 4),
            "Temporal_AUC_CI_high": round(ahi, 4),
            "Temporal_Brier": round(float(brier_score_loss(y[tstL], pred[tstL])) if tstL.sum() else float("nan"), 4),
            "nlive": len(live),
        })
    return recs, preds, sd_shim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke: 1 landmark (6M), 100 boot, median+locf only")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--methods", type=str, default="",
                    help="comma-list to restrict methods (default: all core + extras)")
    ap.add_argument("--no-extras", action="store_true", help="skip ebm_native/median_indicator")
    args = ap.parse_args()

    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    if args.quick:
        landmarks = (6,)
        methods = ("median", "locf")
        n_boot = 100
    else:
        landmarks = LANDMARKS_X
        methods = METHODS if args.no_extras else ALL_METHODS
        n_boot = args.boot
    if args.methods:
        methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())

    all_recs = []
    preds_by_method = {}
    sd_ref = None
    for method in methods:
        print(f"=== method={method} ===", flush=True)
        if method in EXTRA_METHODS:
            recs, preds, sd_shim = run_method_extra(method, landmarks, n_boot)
        else:
            recs, preds, sd_shim = run_method(method, landmarks, n_boot)
        for r in recs:
            print("  " + "  ".join(f"{k}={v}" for k, v in r.items()
                                   if k in ("landmark", "OOF_AUC", "Temporal_AUC",
                                            "Temporal_AUC_CI_low", "Temporal_AUC_CI_high",
                                            "Temporal_Brier")), flush=True)
        all_recs.extend(recs)
        preds_by_method[method] = preds
        if sd_ref is None:
            sd_ref = sd_shim

    df = pd.DataFrame(all_recs)
    df.to_csv(OUT / "tables" / "impute_method_comparison.csv", index=False)
    print(f"\nSaved comparison → {OUT/'tables'/'impute_method_comparison.csv'}", flush=True)

    # --- paired ΔAUC vs median baseline (CI cross/exclude 0) ---
    deltas = []
    if "median" in preds_by_method:
        y = sd_ref.rows["Y_24M_NHRH"].values
        lm = sd_ref.rows["landmark"].values
        for method in methods:
            if method == "median":
                continue
            for L in landmarks:
                # paired bootstrap needs both preds aligned to the SAME sd.rows;
                # preds differ only in imputation, rows skeleton identical → safe.
                dm, dlo, dhi = paired_episode_cluster_bootstrap_delta(
                    sd_ref,
                    preds_by_method[method][L],
                    preds_by_method["median"][L],
                    landmark=L, split="Temporal", n=n_boot,
                )
                crosses0 = (dlo <= 0 <= dhi)
                deltas.append({
                    "method": method, "vs": "median", "landmark": f"{L}M",
                    "dAUC_mean": round(dm, 4),
                    "dAUC_CI_low": round(dlo, 4), "dAUC_CI_high": round(dhi, 4),
                    "CI_excludes_0": (not crosses0),
                    "direction": "better" if dm > 0 else ("worse" if dm < 0 else "tie"),
                })
    ddf = pd.DataFrame(deltas)
    if not ddf.empty:
        ddf.to_csv(OUT / "tables" / "impute_delta_vs_median.csv", index=False)
        print("\n=== ΔAUC vs median (paired episode-cluster bootstrap) ===", flush=True)
        print(ddf.to_string(index=False), flush=True)

    # --- recommendation: best method per landmark by temporal AUC, with CI vs median ---
    rec_lines = []
    for L in landmarks:
        sub = df[df["landmark"] == f"{L}M"].copy()
        if sub.empty:
            continue
        best = sub.loc[sub["Temporal_AUC"].idxmax()]
        line = {"landmark": f"{L}M", "best_method": best["method"],
                "best_Temporal_AUC": float(best["Temporal_AUC"])}
        if not ddf.empty and best["method"] != "median":
            d = ddf[(ddf["landmark"] == f"{L}M") & (ddf["method"] == best["method"])]
            if not d.empty:
                line["dAUC_vs_median"] = float(d.iloc[0]["dAUC_mean"])
                line["CI_excludes_0"] = bool(d.iloc[0]["CI_excludes_0"])
        rec_lines.append(line)

    summary = {
        "口径": "corrected truth (proper-NaN, no 6M/12M zero-fallback)",
        "missing_rate_note": "6M ≈25%, 12M ≈40% follow-up drop-out (all-three-hormone)",
        "methods": list(methods),
        "landmarks": [f"{L}M" for L in landmarks],
        "n_boot": n_boot,
        "comparison": all_recs,
        "delta_vs_median": deltas,
        "per_landmark_best": rec_lines,
    }
    (OUT / "tables" / "impute_method_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(f"\nSaved summary → {OUT/'tables'/'impute_method_summary.json'}", flush=True)
    print("\n=== per-landmark best method ===", flush=True)
    for line in rec_lines:
        print("  " + "  ".join(f"{k}={v}" for k, v in line.items()), flush=True)


if __name__ == "__main__":
    main()
