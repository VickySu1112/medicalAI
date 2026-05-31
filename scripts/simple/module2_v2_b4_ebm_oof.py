#!/usr/bin/env python
"""M2 · B4 — per-landmark EBM OOF + temporal read-out, with naive baselines.

Phase 0 infrastructure for the cross-landmark deepening (adds the 12M landmark).
This module provides a UNIFIED, leakage-audited evaluation primitive so every
landmark L (including 12M) is scored the same way the mainline scores 0/1/3/6:

    dev  → 5-fold StratifiedGroupKFold(by episode_id) OOF probabilities
    temporal → final dev-fit EBM predicts the held-out temporal rows

plus two naive baselines required by the project's "always include a
persistence baseline" rule:

    naive_proba        — constant = dev event rate (no information)
    persistence_proba  — time-safe single-feature "look at current TSH" L2-LR

Feature construction reuses build_feats_at_L from module2_v2_b4_ebm_axes (the
orthogonal-axis parameterisation consistent with the group table). FT3/FT4
current+velocity are built landmark-aware here (so 12M chains off 6M) WITHOUT
touching the mainline trab patch, which is hard-wired to (0,1,3,6).

Time-safety gate: assert_no_future_feature(feats, L) raises if any feature name
references a month > L, so an accidental 12M column at L<12 (or any Stage1_*Risk)
is caught before it reaches the model.
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import re
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.simple.module2_v2_shared import (
    CV_SEED,
    PY_SEED,
    STAGE2_LONG,
    _corrected_landmark_values,
    _prev_landmark_map,
)
from scripts.simple.module2_v2_b4_ebm_axes import FEATS, build_feats_at_L


# ---------------------------------------------------------------------------
# Landmark-aware FT3/FT4 current+velocity (mainline-safe; supports 12M)
# ---------------------------------------------------------------------------


def _episode_landmark_value(df: pd.DataFrame, marker: str, L: int, episodes) -> np.ndarray:
    """Time-safe corrected level of `marker` at landmark L per episode.

    Thin wrapper around the single source of truth
    ``module2_v2_shared._corrected_landmark_values`` (kept for backward-compat
    callers). The corrected pull now lives in the data layer; this realigns its
    episode-ordered output to the caller-supplied ``episodes`` order. ``df`` is
    accepted for signature stability but ignored (the shared helper re-reads the
    long table itself).
    """
    episodes = [int(e) for e in episodes]
    corr = _corrected_landmark_values([L], markers=(marker,))[marker][L]
    # corr is indexed by Treatment_Index ascending (0..1002); the long table's
    # unique-episode order is exactly that ascending order, matching `episodes`
    # in every current caller. Realign defensively by position.
    if len(corr) == len(episodes):
        return corr.astype(float)
    # Fallback: build a position map (should not trigger in practice).
    base_order = list(range(len(corr)))
    pos = {e: i for i, e in enumerate(base_order)}
    out = np.full(len(episodes), np.nan, dtype=float)
    for i, e in enumerate(episodes):
        j = pos.get(e)
        if j is not None:
            out[i] = corr[j]
    return out


def ensure_marker_current_velocity(sd, marker: str, landmarks):
    """Add {marker}_at_{L}M + {marker}_current/{marker}_velocity for any landmark set.

    Parameterised by `landmarks` (the mainline trab patch is hard-wired to 0,1,3,6
    and cannot reach 12M). Per-landmark levels come from _episode_landmark_value,
    which sources the real 6M/12M hormone level from the Current_Time-matched row
    (the first-row wide *_12M column is all-zeros and must NOT be used). Velocity
    chains off the immediate predecessor; first landmark → 0. Imputation is
    dev-median (a constant, not future info), matching the mainline convention.
    """
    landmarks = tuple(sorted(landmarks))
    prev = _prev_landmark_map(landmarks)
    df = pd.read_csv(STAGE2_LONG, low_memory=False)
    df["Treatment_Index"] = pd.to_numeric(df["Treatment_Index"], errors="coerce").astype(int)
    episodes = (
        df.drop_duplicates("Treatment_Index", keep="first")
        .sort_values("Treatment_Index")["Treatment_Index"].astype(int).values
    )
    idx = {int(e): i for i, e in enumerate(episodes)}
    vals = {L: _episode_landmark_value(df, marker, L, episodes) for L in landmarks}
    rows = sd.rows
    epid = rows["episode_id"].values
    lm = rows["landmark"].values
    for L in landmarks:
        rows[f"{marker}_at_{L}M"] = np.array([vals[L][idx[int(e)]] for e in epid])
    cur = np.zeros(len(rows))
    vel = np.zeros(len(rows))
    for i in range(len(rows)):
        L = int(lm[i])
        at_L = rows[f"{marker}_at_{L}M"].values[i]
        cur[i] = at_L
        p = prev.get(L)
        if p is None:
            vel[i] = 0.0
        else:
            vel[i] = (at_L - rows[f"{marker}_at_{p}M"].values[i]) / (L - p)
    rows[f"{marker}_current"] = cur
    rows[f"{marker}_velocity"] = vel
    dev = (rows["Split"] == "Development").values
    for c in (f"{marker}_current", f"{marker}_velocity"):
        rows[c] = rows[c].replace([np.inf, -np.inf], np.nan)
        rows[c] = rows[c].fillna(np.nanmedian(rows.loc[dev, c]))
    return sd


def prepare_axis_inputs(sd, landmarks):
    """Ensure FT3/FT4/TSH current+velocity carry the corrected real source.

    As of the data-layer consolidation, ``load_stacked_x()`` already produces the
    corrected current/velocity (the single source of truth is
    ``module2_v2_shared._corrected_landmark_values``), so calling this on a
    corrected StackedData is an idempotent no-op (recomputes identical values).
    It is retained so callers built on the legacy ``load_stacked()`` /
    ``load_stacked_x(corrected=False)`` (degenerate wide columns at 6M/12M) can
    still upgrade in place to the real 6M/12M levels. Mutates sd.rows; returns it.
    """
    sd = ensure_marker_current_velocity(sd, "FT3", landmarks)
    sd = ensure_marker_current_velocity(sd, "FT4", landmarks)
    sd = ensure_marker_current_velocity(sd, "TSH", landmarks)
    return sd


# ---------------------------------------------------------------------------
# Time-safety gate
# ---------------------------------------------------------------------------

_MONTH_PATTERNS = (
    re.compile(r"_(\d+)M(?:$|[_x])"),     # *_12M, *_12M_x..., *_at... handled below
    re.compile(r"_at_(\d+)M"),            # *_at_12M
    re.compile(r"Eval_(\d+)M"),           # Eval_12M[_Hyper...]
    re.compile(r"_(\d+)M_to_\d+M"),       # Window_12M_to_18M (start month)
)


def _max_future_month(name: str) -> int | None:
    """Return the largest month index referenced by a feature name, or None.

    Scans several known naming conventions (_{L}M, _at_{L}M, Eval_{L}M,
    Window_{L}M_to_…). Used to detect any column that peeks past landmark L.
    """
    months: list[int] = []
    for pat in _MONTH_PATTERNS:
        for m in pat.finditer(name):
            try:
                months.append(int(m.group(1)))
            except (ValueError, IndexError):
                continue
    return max(months) if months else None


def assert_no_future_feature(feats, L: int) -> None:
    """Raise if any feature in `feats` references information after landmark L.

    Catches: *_{L'}M / *_at_{L'}M / Eval_{L'}M / Window_{L'}M_to_… with L' > L
    (e.g. an FT4_12M leaking into an L=6 model), and any Stage1_*Risk column
    (the 6M-built scalar risk is masked at earlier landmarks per CLAUDE.md).

    `feats` may be a list/Index of column names or a DataFrame.
    """
    if hasattr(feats, "columns"):
        names = list(feats.columns)
    else:
        names = list(feats)
    bad: list[str] = []
    for name in names:
        if re.search(r"Stage1_.*Risk", str(name)):
            bad.append(f"{name} (Stage1 scalar risk; not time-safe < its build month)")
            continue
        fut = _max_future_month(str(name))
        if fut is not None and fut > L:
            bad.append(f"{name} (references {fut}M > landmark {L}M)")
    if bad:
        raise RuntimeError(
            f"[time-safety] {len(bad)} feature(s) peek past landmark {L}M: " + "; ".join(bad)
        )


# ---------------------------------------------------------------------------
# Per-landmark EBM OOF + temporal
# ---------------------------------------------------------------------------


def _roc(y, p) -> float:
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")


def ebm_oof_and_temporal(rows, y, lm, is_dev, L, feats=FEATS, seed: int = PY_SEED,
                         interactions: int = 5, max_interaction_bins: int = 16):
    """Per-landmark EBM: dev OOF probabilities + temporal read-out from final fit.

    Parameters
    ----------
    rows : DataFrame of stacked landmark-rows (must contain FT3/FT4 current+velocity
        for build_feats_at_L — call prepare_axis_inputs first).
    y, lm, is_dev : aligned arrays (length = len(rows)) of label, landmark, dev mask.
    L : the landmark to fit at.
    feats : candidate feature list (default FEATS = orthogonal axes + statics).
    seed : EBM/LR random_state (default PY_SEED).
    interactions : number of automatic pairwise interactions the EBM may learn
        (default 5 = mainline GA2M; pass 0 for a pure additive GAM, 2 for the
        reduced configuration — used by the interaction ablation in
        module2_v2_ebm_interaction_diag.py to quantify the discrimination gain of
        interaction terms vs the no-interaction GAM).
    max_interaction_bins : interaction 网格分箱数 (default 16). 全项目已统一迁移到
        bin16:默认 (~62×62) 交互网格对 1003 人次过细 → 多数格无人落入 → 交互 2D 查表
        近乎全外推;降到 16×16 后网格 occupancy ~85–90%,每格有真实病人支撑、查表可信。
        单变量形状函数与判别 AUC 基本不受影响 (见 bins sweep)。

    Returns
    -------
    pred : full-length array; dev rows at L hold 5-fold OOF, temporal rows at L
        hold the final-fit prediction; all other rows are 0.
    final_ebm : EBM fitted on all dev rows at L.
    live : the non-constant feature subset actually used (velocity@0M is constant).
    """
    lm = np.asarray(lm)
    is_dev = np.asarray(is_dev)
    y = np.asarray(y)
    devL = is_dev & (lm == L)
    tstL = (~is_dev) & (lm == L)

    feat_all = build_feats_at_L(rows, devL)  # axes z-fit on dev@L, applied to all
    # Time-safety gate on the realised feature matrix (column names).
    assert_no_future_feature(feat_all, L)

    Xtr_full = feat_all.loc[devL]
    live = [c for c in feat_all.columns if Xtr_full[c].std() > 1e-9]
    feat_live = feat_all[live]

    Xd = feat_live.loc[devL].values
    yd = y[devL]
    ep = rows["episode_id"].values
    epd = ep[devL]

    pred = np.zeros(len(rows), dtype=float)
    dev_idx = np.where(devL)[0]
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    for tr, va in skf.split(Xd, yd, groups=epd):
        ebm = ExplainableBoostingClassifier(random_state=seed, interactions=interactions,
                                            max_interaction_bins=max_interaction_bins)
        ebm.fit(Xd[tr], yd[tr])
        pred[dev_idx[va]] = ebm.predict_proba(Xd[va])[:, 1]

    final_ebm = ExplainableBoostingClassifier(random_state=seed, interactions=interactions,
                                              max_interaction_bins=max_interaction_bins)
    final_ebm.fit(Xd, yd)
    if tstL.any():
        pred[tstL] = final_ebm.predict_proba(feat_live.loc[tstL].values)[:, 1]
    return pred, final_ebm, live


# ---------------------------------------------------------------------------
# Naive baselines (persistence rule)
# ---------------------------------------------------------------------------


def naive_proba(y_dev, n: int) -> np.ndarray:
    """Constant predictor = dev event rate, broadcast to length n (no information)."""
    return np.full(int(n), float(np.mean(y_dev)), dtype=float)


def persistence_proba(rows, y, lm, is_dev, L):
    """Time-safe "look at the current labs" baseline: single-feature L2-LR on TSH_current@L.

    The clinically naive comparator — predict relapse from the current functional
    state only (TSH measured at L), no momentum / burden / exposure. dev → OOF,
    temporal → final dev-fit prediction, exactly like ebm_oof_and_temporal so the
    learned model's increment over "just read today's TSH" is directly quantified.

    Returns full-length pred (dev@L = OOF, temporal@L = final-fit, else 0).
    """
    lm = np.asarray(lm)
    is_dev = np.asarray(is_dev)
    y = np.asarray(y)
    devL = is_dev & (lm == L)
    tstL = (~is_dev) & (lm == L)

    feat = pd.DataFrame({"TSH_current": rows["TSH_current"].values})
    assert_no_future_feature(feat, L)  # TSH_current is the value measured at L

    Xd = feat.loc[devL].values
    yd = y[devL]
    epd = rows["episode_id"].values[devL]

    pred = np.zeros(len(rows), dtype=float)
    dev_idx = np.where(devL)[0]
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    for tr, va in skf.split(Xd, yd, groups=epd):
        pipe = Pipeline(
            [
                ("s", StandardScaler()),
                ("lr", LogisticRegression(penalty="l2", solver="lbfgs", C=1.0,
                                          max_iter=5000, random_state=PY_SEED)),
            ]
        )
        pipe.fit(Xd[tr], yd[tr])
        pred[dev_idx[va]] = pipe.predict_proba(Xd[va])[:, 1]

    final = Pipeline(
        [
            ("s", StandardScaler()),
            ("lr", LogisticRegression(penalty="l2", solver="lbfgs", C=1.0,
                                      max_iter=5000, random_state=PY_SEED)),
        ]
    )
    final.fit(Xd, yd)
    if tstL.any():
        pred[tstL] = final.predict_proba(feat.loc[tstL].values)[:, 1]
    return pred
