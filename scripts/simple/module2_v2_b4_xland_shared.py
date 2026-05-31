#!/usr/bin/env python
"""M2 · B4 — shared scaffold for the cross-landmark error / transition mining
(Phase 4 task1 + Phase 5 task5).

Single source of truth for the four xland scripts (tracking / trajectory /
errtypes / report). It encapsulates the project口径 so every script scores the
same way:

    口径 = corrected truth (Current_Time rows) + LOCF impute (proper-NaN, no
           6M/12M zero-fallback), landmarks (1, 3, 6, 12).
    rows = module2_v2_impute_experiment.build_rows_for_method("locf", LANDMARKS)
    EBM  = ebm_oof_and_temporal  (dev → 5-fold StratifiedGroupKFold OOF;
           temporal → final dev-fit read-out — NEVER used to select/tune).
    baselines = naive (dev prevalence, constant) + persistence (time-safe
           single-feature L2-LR on TSH_current@L — "just look at today's TSH").

Everything here is **time-safe**: at landmark L only ≤L information is used.
``assert_no_future_feature`` runs inside ebm_oof_and_temporal. Functional state
at 1M/3M uses the real clinical ``Eval_{L}M`` one-hot (ground truth); at 6M/12M
the long-table Eval one-hot is degenerate (all-zero, same bug class as the wide
hormone columns), so state is DERIVED from the corrected hormone levels at L
(best-available, TSH-primary; reported as such). No post-RAI medication feature.

analysis unit = treatment-episode (疗程 / 人次); N = 1003. Repeat patients are
INDEPENDENT episodes (no patient grouping). The forbidden unique-patient count is
never computed or printed.
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

import matplotlib  # noqa: E402
import matplotlib.font_manager as _fm  # noqa: E402

for _fp in ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "/Library/Fonts/Arial Unicode.ttf"):
    if os.path.exists(_fp):
        _fm.fontManager.addfont(_fp)
        matplotlib.rcParams["font.family"] = "Arial Unicode MS"
        break
matplotlib.rcParams["axes.unicode_minus"] = False

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from scripts.simple.module2_v2_shared import STAGE2_LONG, forbidden_token  # noqa: E402
from scripts.simple.module2_v2_impute_experiment import (  # noqa: E402
    _raw_truth_proper_nan,
    build_rows_for_method,
)
from scripts.simple.module2_v2_b4_ebm_oof import (  # noqa: E402
    ebm_oof_and_temporal,
    naive_proba,
    persistence_proba,
)

# ---------------------------------------------------------------------------
# Constants (口径 already fixed by the plan / worklog)
# ---------------------------------------------------------------------------
LANDMARKS = (1, 3, 6, 12)          # 0M is M1's job; 12M added (PR#4 line)
OUT = ROOT / "results" / "module2_v2_vertical" / "b4_target_gru"
TBL = OUT / "tables"
FIG = OUT / "figures"

# risk tiers on the EBM probability (3 bins; cuts are clinical-flavoured but
# applied to the OOF/temporal probability, so they are label-free at apply time)
TIER_CUTS = (0.30, 0.60)           # Low < .30 ≤ Mid < .60 ≤ High
TIER_NAMES = ("Low", "Mid", "High")

# reference ranges for the DERIVED functional state at 6M/12M (clinical adult):
#   TSH 0.27–4.2 mIU/L · FT4 12–22 pmol/L · FT3 3.1–6.8 pmol/L
REF = {"TSH_lo": 0.27, "TSH_hi": 4.2, "FT4_lo": 12.0, "FT4_hi": 22.0, "FT3_hi": 6.8}

STATE_ORDER = ("Hyper", "Normal", "Hypo", "Unknown")


def ensure_dirs() -> None:
    TBL.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Functional state per landmark (time-safe)
# ---------------------------------------------------------------------------


def _derive_state(ft3: np.ndarray, ft4: np.ndarray, tsh: np.ndarray) -> np.ndarray:
    """Best-available functional state from corrected hormone levels at L.

    TSH-primary post-RAI convention: suppressed TSH with elevated FT ⇒ Hyper;
    high TSH with low FT4 ⇒ Hypo; otherwise Normal. ``Unknown`` only when all
    three markers are missing. Used at 6M/12M where the clinical ``Eval_{L}M``
    one-hot is degenerate; validated against the real 1M/3M Eval at ~0.76–0.80
    agreement (residual = missing-FT4 'Unknown' + borderline-TSH Hyper).
    """
    n = len(ft4)
    st = np.full(n, "Unknown", dtype=object)
    fin = np.isfinite(tsh) | np.isfinite(ft4) | np.isfinite(ft3)
    st[fin] = "Normal"
    hi_ft = (np.nan_to_num(ft4, nan=-1.0) > REF["FT4_hi"]) | (np.nan_to_num(ft3, nan=-1.0) > REF["FT3_hi"])
    lo_ft = np.nan_to_num(ft4, nan=1e9) < REF["FT4_lo"]
    hyper = ((tsh < REF["TSH_lo"]) & hi_ft) | hi_ft
    hypo = ((tsh > REF["TSH_hi"]) & lo_ft) | (tsh > 10.0)
    st[hyper & fin] = "Hyper"
    st[hypo & fin] = "Hypo"
    return st


def functional_state_by_landmark(episodes: np.ndarray) -> dict[int, np.ndarray]:
    """Per-landmark functional state, episode-ordered to `episodes`.

    1M/3M → real clinical ``Eval_{L}M`` one-hot (ground truth, time-safe).
    6M/12M → derived from the corrected hormone levels (Eval one-hot degenerate).
    Returns {L: array[str] aligned to `episodes`}.
    """
    df = pd.read_csv(STAGE2_LONG, low_memory=False)
    df["Treatment_Index"] = pd.to_numeric(df["Treatment_Index"], errors="coerce").astype(int)
    first = df.drop_duplicates("Treatment_Index", keep="first").sort_values("Treatment_Index")
    base_eps = first["Treatment_Index"].astype(int).values
    pos = {int(e): i for i, e in enumerate(base_eps)}
    realign = np.array([pos[int(e)] for e in episodes])

    truth, _eps, _split = _raw_truth_proper_nan(LANDMARKS)
    out: dict[int, np.ndarray] = {}
    for L in LANDMARKS:
        has_eval = any(
            pd.to_numeric(first[f"Eval_{L}M_{k}"], errors="coerce").fillna(0).sum() > 0
            for k in ("Hyper", "Normal", "Hypo")
        )
        if has_eval:
            st = np.full(len(first), "Missing", dtype=object)
            for k in ("Hyper", "Normal", "Hypo"):
                v = pd.to_numeric(first[f"Eval_{L}M_{k}"], errors="coerce").fillna(0).values
                st[v == 1] = k
            # episodes missing a clinical eval → fall back to derived
            der = _derive_state(truth["FT3"][L], truth["FT4"][L], truth["TSH"][L])
            miss = st == "Missing"
            st[miss] = der[miss]
        else:
            st = _derive_state(truth["FT3"][L], truth["FT4"][L], truth["TSH"][L])
        out[L] = st[realign]
    return out


def state_source_label(L: int) -> str:
    return "clinical Eval" if L in (1, 3) else "derived(corrected hormones)"


# ---------------------------------------------------------------------------
# Risk tiers
# ---------------------------------------------------------------------------


def assign_tier(p: np.ndarray) -> np.ndarray:
    """Bin a probability vector into Low/Mid/High by TIER_CUTS (label-free)."""
    lo, hi = TIER_CUTS
    t = np.where(p < lo, TIER_NAMES[0], np.where(p < hi, TIER_NAMES[1], TIER_NAMES[2]))
    return t.astype(object)


# ---------------------------------------------------------------------------
# Youden threshold (per landmark, on dev OOF — selection-safe)
# ---------------------------------------------------------------------------


def youden_threshold(y_dev: np.ndarray, p_dev: np.ndarray) -> float:
    """Sens+Spec−1 optimal cut on dev OOF (never uses temporal labels)."""
    ths = np.linspace(0.05, 0.95, 181)

    def j(t: float) -> float:
        pred = (p_dev >= t).astype(int)
        tp = int(((pred == 1) & (y_dev == 1)).sum())
        fn = int(((pred == 0) & (y_dev == 1)).sum())
        tn = int(((pred == 0) & (y_dev == 0)).sum())
        fp = int(((pred == 1) & (y_dev == 0)).sum())
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        return sens + spec - 1.0

    return float(max(ths, key=j))


# ---------------------------------------------------------------------------
# Core: build the per-episode × per-landmark prediction wide table
# ---------------------------------------------------------------------------


def _roc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def build_tracking_long(quick: bool = False, seed: int | None = None):
    """Run the EBM-OOF + baselines at every landmark and assemble a LONG frame.

    Returns (long_df, meta) where long_df has one row per (episode, landmark)
    that the episode actually contributes (all episodes appear at every landmark
    in LANDMARKS — the stacked design is balanced), with columns:

        episode_id, landmark, Split, Y,
        p (EBM prob), correct (0/1 at Youden@OOF), tier (Low/Mid/High),
        p_naive, p_persist, state (functional state at L), thr (Youden@OOF)

    meta carries per-landmark Youden thresholds + headline AUCs (dev/temporal)
    for EBM / persistence / naive and the per-landmark functional-state source.
    Temporal numbers are READ-OUT ONLY (never used to choose anything).
    """
    landmarks = (6,) if quick else LANDMARKS
    rows = build_rows_for_method("locf", landmarks)
    y = rows["Y_24M_NHRH"].values.astype(int)
    lm = rows["landmark"].values.astype(int)
    is_dev = (rows["Split"] == "Development").values
    split_col = np.where(is_dev, "Development", "Temporal")
    epid = rows["episode_id"].values.astype(int)

    # functional state per landmark, aligned to the episode order in `rows`
    episodes_sorted = (
        rows.drop_duplicates("episode_id").sort_values("episode_id")["episode_id"].astype(int).values
    )
    state_map = functional_state_by_landmark(episodes_sorted)
    ep_to_state_idx = {int(e): i for i, e in enumerate(episodes_sorted)}

    long_records: list[dict] = []
    meta: dict = {"landmarks": list(landmarks), "thr": {}, "auc": {}, "state_source": {}}

    for L in landmarks:
        kwargs = {} if seed is None else {"seed": seed}
        pred, _ebm, _live = ebm_oof_and_temporal(rows, y, lm, is_dev, L, **kwargs)
        p_persist = persistence_proba(rows, y, lm, is_dev, L)

        devL = is_dev & (lm == L)
        tstL = (~is_dev) & (lm == L)
        atL = lm == L

        p_naive_const = float(np.mean(y[devL]))  # dev prevalence (constant)
        thr = youden_threshold(y[devL], pred[devL])

        meta["thr"][L] = thr
        meta["state_source"][L] = state_source_label(L)
        meta["auc"][L] = {
            "ebm_oof": _roc(y[devL], pred[devL]),
            "ebm_temporal": _roc(y[tstL], pred[tstL]),
            "persist_oof": _roc(y[devL], p_persist[devL]),
            "persist_temporal": _roc(y[tstL], p_persist[tstL]),
            "naive_oof": _roc(y[devL], np.full(devL.sum(), p_naive_const)),
            "naive_temporal": _roc(y[tstL], np.full(tstL.sum(), p_naive_const)),
            "n_dev": int(devL.sum()),
            "n_temporal": int(tstL.sum()),
            "prevalence_dev": p_naive_const,
        }

        tier = assign_tier(pred)
        st = state_map[L]
        idxs = np.where(atL)[0]
        for i in idxs:
            e = int(epid[i])
            long_records.append(
                {
                    "episode_id": e,
                    "landmark": L,
                    "Split": split_col[i],
                    "Y": int(y[i]),
                    "p": float(pred[i]),
                    "correct": int((pred[i] >= thr) == bool(y[i])),
                    "pred_label": int(pred[i] >= thr),
                    "tier": tier[i],
                    "p_naive": p_naive_const,
                    "p_persist": float(p_persist[i]),
                    "state": st[ep_to_state_idx[e]],
                    "thr": float(thr),
                }
            )

    long_df = pd.DataFrame(long_records)
    # guard: the forbidden unique-patient count must never surface
    assert forbidden_token() not in str(long_df.shape), "forbidden token leaked into shape"
    return long_df, meta


PROFILE_FEATS = [
    "TSH_current", "FT3_current", "FT4_current",
    "TSH_velocity", "FT3_velocity", "FT4_velocity",
    "TRAb", "TGAb", "TPOAb", "TSH_0M", "FT4_0M",
    "ThyroidW", "log1p_DiseaseDuration_Months_Aug", "Uptake24h", "HalfLife", "Sex",
]


def feature_rows(quick: bool = False) -> pd.DataFrame:
    """Raw (corrected+LOCF imputed) feature rows keyed by (episode_id, landmark).

    Same rows the EBM sees (build_rows_for_method('locf', …)), restricted to the
    PROFILE_FEATS + keys. Used by the error-subtype profiling/clustering without
    re-running any model. All columns are ≤L (time-safe by construction).
    """
    landmarks = (6,) if quick else LANDMARKS
    rows = build_rows_for_method("locf", landmarks)
    cols = ["episode_id", "landmark", "Split", "Y_24M_NHRH"] + PROFILE_FEATS
    return rows[cols].copy()


def long_to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the LONG tracking frame to ONE row per episode with @L columns.

    Columns: episode_id, Split, Y, then for each landmark L in the data:
    p@L, correct@L, tier@L, p_naive@L, p_persist@L, state@L.
    Split/Y are episode-constant; taken from the first available landmark row.
    """
    landmarks = sorted(long_df["landmark"].unique())
    base = (
        long_df.sort_values(["episode_id", "landmark"])
        .drop_duplicates("episode_id")[["episode_id", "Split", "Y"]]
        .reset_index(drop=True)
    )
    wide = base.copy()
    for L in landmarks:
        sub = long_df[long_df["landmark"] == L].set_index("episode_id")
        for src, dst in (
            ("p", f"p@{L}"),
            ("correct", f"correct@{L}"),
            ("tier", f"tier@{L}"),
            ("p_naive", f"p_naive@{L}"),
            ("p_persist", f"p_persist@{L}"),
            ("state", f"state@{L}"),
        ):
            wide[dst] = wide["episode_id"].map(sub[src])
    return wide
