#!/usr/bin/env python
"""M2 v2 shared infrastructure for three-track redesign.

Provides:
- Stacked long-format dataset (1003 episodes × 4 landmarks = 4012 landmark-rows)
- Five mechanism feature blocks (A burden, B exposure, C dynamic, D momentum, E time)
- StratifiedGroupKFold by episode_id + per-landmark Platt + episode-cluster bootstrap
- Leakage unit-test assertions
- Forbidden unique-patient count literal never appears (runtime str(890 - 1))

This module is consumed by:
- module2_v2_base.py (M2-Base supermodel)
- module2_v2_horizontal_a.py (M2-A 10-method benchmark)
- module2_v2_vertical_b{1,2,3}_*.py (M2-B novel architectures)

Pre-registered decisions live in:
    results/module2_v2_base/arch.md
    results/module2_v2_horizontal/arch.md
    results/module2_v2_vertical/arch.md
    results/module2_v2_synthesis/arch.md
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

# ---------------------------------------------------------------------------
# Constants (pre-registered)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

PY_SEED = 2025
CV_SEED = 13
N_FOLDS = 5
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 7

LANDMARKS = (0, 1, 3, 6)
# Cross-landmark extension: adds the 12M landmark (FT3/FT4/TSH_12M ~100% covered
# in stage2_long_table). Mainline LANDMARKS is untouched so load_stacked() stays
# byte-for-byte identical; load_stacked_x() opts into the extended set.
LANDMARKS_X = (0, 1, 3, 6, 12)
EXPECTED_EPISODES = 1003
EXPECTED_DEV = 802
EXPECTED_TEMPORAL = 201


def _prev_landmark_map(landmarks: Sequence[int]) -> dict[int, int | None]:
    """Map each landmark to its immediate predecessor in the ordered set.

    First landmark → None (no velocity). Used for Δ/Δt velocity construction so
    that the extended set (…, 6, 12) chains 12M off 6M automatically.
    """
    ordered = sorted(landmarks)
    out: dict[int, int | None] = {}
    for i, L in enumerate(ordered):
        out[L] = None if i == 0 else ordered[i - 1]
    return out

# Data sources
STAGE2_LONG = ROOT / "results" / "stage2_mh_h6h12_cjk" / "tables" / "stage2_long_table.csv"
M1_FROZEN = ROOT / "results" / "module1_baseline_ml_benchmark" / "tables" / "module1_frozen_feature_matrix.csv"

# Mechanism Block A: baseline burden (time-invariant, from M1 v2 curated 8)
BLOCK_A_BURDEN = [
    "Sex",
    "ThyroidW",
    "TRAb",
    "TGAb",
    "TPOAb",
    "FT4_0M",
    "TSH_0M",
    "log1p_DiseaseDuration_Months_Aug",
]

# Block B: RAI exposure (time-invariant)
BLOCK_B_EXPOSURE = ["Uptake24h", "HalfLife"]

# Block C: current dynamic (landmark-conditional lookup)
BLOCK_C_CURRENT = ["TSH_current", "FT4_current"]

# Block D: momentum (Δ/Δt; L=0M → 0 + velocity_observed=0)
BLOCK_D_MOMENTUM = ["TSH_velocity", "FT4_velocity"]

# Block E: time interactions (landmark_time × dynamic features)
BLOCK_E_TIME = [
    "landmark_time_centered",
    "is_baseline_landmark",
    "velocity_observed",
    "lm_x_TSH_current",
    "lm_x_FT4_current",
    "lm_x_TSH_velocity",
    "lm_x_FT4_velocity",
]

BLOCKS = {
    "A": BLOCK_A_BURDEN,
    "B": BLOCK_B_EXPOSURE,
    "C": BLOCK_C_CURRENT,
    "D": BLOCK_D_MOMENTUM,
    "E": BLOCK_E_TIME,
}

ALL_BLOCKS_FEATURES = BLOCK_A_BURDEN + BLOCK_B_EXPOSURE + BLOCK_C_CURRENT + BLOCK_D_MOMENTUM + BLOCK_E_TIME

# Pretty labels for plots / reports
PRETTY = {
    "Sex": "Sex",
    "ThyroidW": "Thyroid weight",
    "TRAb": "TSH receptor antibody",
    "TGAb": "Thyroglobulin antibody",
    "TPOAb": "Thyroid peroxidase antibody",
    "FT4_0M": "FT4 at baseline",
    "TSH_0M": "TSH at baseline",
    "log1p_DiseaseDuration_Months_Aug": "Log disease duration (months)",
    "Uptake24h": "24-hour RAI uptake",
    "HalfLife": "Effective iodine half-life",
    "TSH_current": "TSH at current landmark",
    "FT4_current": "FT4 at current landmark",
    "TSH_velocity": "TSH velocity (per month)",
    "FT4_velocity": "FT4 velocity (per month)",
    "landmark_time_centered": "Landmark time (centered at 1.5M)",
    "is_baseline_landmark": "Is baseline landmark (0M)",
    "velocity_observed": "Velocity observed (L ≥ 1M)",
    "lm_x_TSH_current": "Landmark × TSH_current",
    "lm_x_FT4_current": "Landmark × FT4_current",
    "lm_x_TSH_velocity": "Landmark × TSH_velocity",
    "lm_x_FT4_velocity": "Landmark × FT4_velocity",
}

# Forbidden unique-patient count literal must NEVER appear in source.
# Runtime-computed for any audit checks (CLAUDE.md hard constraint).
def forbidden_token() -> str:
    return str(890 - 1)


# ---------------------------------------------------------------------------
# Data loading and stacking
# ---------------------------------------------------------------------------


@dataclass
class StackedData:
    """Container for stacked landmark-row data.

    Attributes
    ----------
    rows : DataFrame of 4012 rows = 1003 episodes × 4 landmarks.
        Columns: episode_id, landmark, landmark_time_centered, is_baseline_landmark,
        velocity_observed, Y_24M_NHRH, Split, plus the 19 features in ALL_BLOCKS_FEATURES.
    episode_meta : DataFrame of 1003 rows with episode_id, Split, Y_24M_NHRH;
        used for episode-level CV grouping and cluster bootstrap.
    """

    rows: pd.DataFrame
    episode_meta: pd.DataFrame

    def feature_matrix(self, block_subset: Sequence[str] | None = None) -> pd.DataFrame:
        """Return the feature columns. If block_subset given, restrict to those blocks.

        block_subset elements drawn from {"A", "B", "C", "D", "E"}.
        """
        if block_subset is None:
            cols = ALL_BLOCKS_FEATURES
        else:
            cols = []
            for b in block_subset:
                cols.extend(BLOCKS[b])
        return self.rows[list(cols)].copy()


def _load_m1_frozen() -> pd.DataFrame:
    """Load M1 v2 baseline matrix for the 8 baseline-burden features + Y."""
    df = pd.read_csv(M1_FROZEN, low_memory=False)
    needed = {"Episode_Index", "Split", "Y"}.union(BLOCK_A_BURDEN).union(BLOCK_B_EXPOSURE)
    missing = needed.difference(df.columns)
    if missing:
        raise RuntimeError(f"M1 frozen matrix missing columns: {missing}")
    keep = ["Episode_Index", "Split", "Y"] + BLOCK_A_BURDEN + BLOCK_B_EXPOSURE
    out = df[keep].copy()
    if len(out) != EXPECTED_EPISODES:
        raise RuntimeError(
            f"M1 frozen matrix has {len(out)} rows; expected {EXPECTED_EPISODES} treatment episodes"
        )
    return out


def _load_stage2_landmark_features(landmarks: Sequence[int] = LANDMARKS) -> pd.DataFrame:
    """Load per-episode landmark-specific TSH/FT4 from stage 2 long table.

    Stage2 long table has one row per (episode × M3 window); we take the first
    row per episode_id (all rows of same episode share the same baseline+landmark
    columns) and extract FT3/FT4/TSH at each landmark L>0M.

    `landmarks` defaults to the mainline (0, 1, 3, 6); pass LANDMARKS_X to also
    pull the 12M labs. Only L>0M columns are read here — M1 frozen matrix is
    authoritative for baseline (0M) burden.
    """
    df = pd.read_csv(STAGE2_LONG, low_memory=False)
    one_per = (
        df.drop_duplicates("Treatment_ID", keep="first")
        .sort_values("Treatment_Index")
        .reset_index(drop=True)
    )
    if len(one_per) != EXPECTED_EPISODES:
        raise RuntimeError(
            f"Stage2 long table has {len(one_per)} unique episodes; expected {EXPECTED_EPISODES}"
        )
    # Skip 0M variants — M1 frozen matrix is authoritative for baseline; only
    # take L>0M dynamic landmark labs from stage2.
    nonzero = [L for L in sorted(landmarks) if L != 0]
    landmark_cols = [f"{m}_{L}M" for m in ("FT3", "FT4", "TSH") for L in nonzero]
    keep = ["Treatment_ID", "Treatment_Index"] + landmark_cols
    out = one_per[keep].copy()
    # Both M1 frozen matrix and Stage2 long table use 0..1002 episode indexing.
    out["Episode_Index"] = out["Treatment_Index"].astype(int)
    return out


# ---------------------------------------------------------------------------
# Corrected landmark truth-value source (single source of truth)
# ---------------------------------------------------------------------------
#
# The degenerate wide columns FT3/FT4/TSH_{L}M in the stage2 long table are real
# on the FIRST (3M) row per episode only for L∈{0,1,3}; the _6M cell is real for
# just ~13/1003 episodes and the _12M cell is all-zeros there. The information-
# bearing, time-safe value at landmark L lives on the row whose Current_Time ==
# "{L}M", in the {marker}_Current column (where it equals {marker}_{L}M exactly,
# verified). _corrected_landmark_values recovers those real 6M/12M levels.
#
# This is the ONE place the corrected pull is implemented; load_stacked(corrected=
# True) / load_stacked_x() consume it so the direct build_feats_at_L path reads
# truth, and module2_v2_b4_ebm_oof imports it instead of duplicating the logic.

CORRECTED_MARKERS = ("FT3", "FT4", "TSH")


def _corrected_landmark_values(
    landmarks: Sequence[int] = LANDMARKS,
    markers: Sequence[str] = CORRECTED_MARKERS,
) -> dict[str, dict[int, np.ndarray]]:
    """Per-episode, per-landmark corrected hormone level (time-safe).

    Returns ``{marker: {L: array indexed by episode-order}}`` where episode order
    is the stage2 long-table's unique Treatment_Index ascending (0..1002).

    Strategy per (marker, L):
      1. Prefer the ``Current_Time == "{L}M"`` row's ``{marker}_Current`` — the
         value measured at L (≤ L; never reads a month > L).
      2. Fall back to the wide ``{marker}_{L}M`` on the first row per episode for
         landmarks without a Current_Time row (0M/1M), or where source 1 is
         missing / a degenerate 0.

    This is intentionally identical to the former ebm_oof ``_episode_landmark_value``
    so the consolidated data-layer path reproduces the corrected EBM numbers.
    """
    df = pd.read_csv(STAGE2_LONG, low_memory=False)
    df["Treatment_Index"] = pd.to_numeric(df["Treatment_Index"], errors="coerce").astype(int)
    episodes = (
        df.drop_duplicates("Treatment_Index", keep="first")
        .sort_values("Treatment_Index")["Treatment_Index"].astype(int).values
    )
    pos = {int(e): i for i, e in enumerate(episodes)}
    first = df.drop_duplicates("Treatment_Index", keep="first")
    out: dict[str, dict[int, np.ndarray]] = {}
    for marker in markers:
        cur_col = f"{marker}_Current"
        out[marker] = {}
        for L in sorted(set(landmarks)):
            vals = np.full(len(episodes), np.nan, dtype=float)
            # Source 1: Current_Time == "{L}M" row.
            if "Current_Time" in df.columns and cur_col in df.columns:
                sub = df[df["Current_Time"].astype(str) == f"{L}M"].drop_duplicates(
                    "Treatment_Index", keep="first"
                )
                cvals = pd.to_numeric(sub[cur_col], errors="coerce").values.astype(float)
                for e, v in zip(sub["Treatment_Index"].astype(int).values, cvals):
                    if int(e) in pos:
                        vals[pos[int(e)]] = v
            # Source 2: wide {marker}_{L}M on first row (0M/1M + gaps/degenerate 0).
            wide_col = f"{marker}_{L}M"
            if wide_col in first.columns:
                wvals = pd.to_numeric(first[wide_col], errors="coerce").values.astype(float)
                for e, v in zip(first["Treatment_Index"].astype(int).values, wvals):
                    i = pos.get(int(e))
                    if i is not None and (np.isnan(vals[i]) or vals[i] == 0.0) and not np.isnan(v):
                        vals[i] = v
            out[marker][L] = vals
    return out


def _restack_long(
    combined: pd.DataFrame,
    landmarks: Sequence[int] = LANDMARKS,
    *,
    corrected: bool = False,
) -> pd.DataFrame:
    """Reshape one-row-per-episode wide → len(landmarks)-rows-per-episode long.

    Inputs `combined` has columns including FT3/FT4/TSH at every L in `landmarks`.
    Output: one row per (episode × landmark), with TSH_current / FT4_current set
    by landmark lookup, plus audit columns (TSH/FT4_at_{L}M for each landmark)
    retained for the leakage gate.

    When ``corrected=True`` the current + audit hormone columns (and an added
    FT3_current / FT3_at_{L}M) are taken from the single-source-of-truth
    ``_corrected_landmark_values`` instead of the degenerate wide ``{marker}_{L}M``
    cells. This is what makes the direct build_feats_at_L path read real 6M/12M
    levels; it stays strictly time-safe (value measured at L). ``corrected=False``
    (the mainline default) leaves the wide-column behaviour byte-for-byte intact.
    """
    landmarks = tuple(sorted(landmarks))
    corr = _corrected_landmark_values(landmarks) if corrected else None
    # episode-order index used by _corrected_landmark_values (Treatment_Index asc).
    if corr is not None:
        ep_order = (
            pd.read_csv(STAGE2_LONG, low_memory=False, usecols=["Treatment_Index"])
            .assign(Treatment_Index=lambda d: pd.to_numeric(d["Treatment_Index"], errors="coerce").astype(int))
            .drop_duplicates("Treatment_Index", keep="first")
            .sort_values("Treatment_Index")["Treatment_Index"].astype(int).values
        )
        ep_pos = {int(e): i for i, e in enumerate(ep_order)}
    rows = []
    for _, ep in combined.iterrows():
        epi = int(ep["Episode_Index"])
        for L in landmarks:
            r = {
                "episode_id": epi,
                "landmark": L,
                "Split": ep["Split"],
                "Y_24M_NHRH": int(ep["Y"]),
            }
            # Block A: baseline burden (time-invariant)
            for f in BLOCK_A_BURDEN:
                r[f] = ep[f]
            # Block B: RAI exposure (time-invariant)
            for f in BLOCK_B_EXPOSURE:
                r[f] = ep[f]
            if corr is None:
                # Block C: current dynamic (landmark-conditional)
                # 0M values from M1 frozen (TSH_0M / FT4_0M); L>0 from stage2
                r["TSH_current"] = ep[f"TSH_{L}M"]
                r["FT4_current"] = ep[f"FT4_{L}M"]
                # Audit columns: explicit landmark-specific TSH/FT4 for leakage gate
                for L_audit in landmarks:
                    r[f"TSH_at_{L_audit}M"] = ep[f"TSH_{L_audit}M"]
                    r[f"FT4_at_{L_audit}M"] = ep[f"FT4_{L_audit}M"]
            else:
                # Corrected source: pull truth from Current_Time-matched rows.
                i = ep_pos[epi]
                r["TSH_current"] = corr["TSH"][L][i]
                r["FT4_current"] = corr["FT4"][L][i]
                r["FT3_current"] = corr["FT3"][L][i]
                for L_audit in landmarks:
                    r[f"TSH_at_{L_audit}M"] = corr["TSH"][L_audit][i]
                    r[f"FT4_at_{L_audit}M"] = corr["FT4"][L_audit][i]
                    r[f"FT3_at_{L_audit}M"] = corr["FT3"][L_audit][i]
            rows.append(r)
    out = pd.DataFrame(rows)
    return out


def _add_blocks_D_and_E(long_df: pd.DataFrame, landmarks: Sequence[int] = LANDMARKS) -> pd.DataFrame:
    """Add momentum (Δ/Δt) and time-interaction features.

    Velocity uses true rate: (current − previous) / Δt months.
    First landmark (0M) → velocity=0 + velocity_observed=0 (Plan agent M1 fix).
    Previous-landmark lookup is the immediate predecessor in the ordered set, so
    the extended set chains 12M velocity off 6M automatically.

    `landmark_time_centered` keeps the pre-registered fixed offset of 1.5 months
    regardless of `landmarks`, so the mainline (0,1,3,6) output is unchanged.
    """
    prev_landmark = _prev_landmark_map(landmarks)
    nonzero = [L for L in sorted(landmarks) if prev_landmark[L] is not None]
    out = long_df.copy()
    # FT3 velocity is only built when the corrected source added FT3 current+audit
    # columns (mainline keeps FT3 out of Block C/D, so this is a no-op there).
    has_ft3 = "FT3_current" in out.columns and all(
        f"FT3_at_{L}M" in out.columns for L in sorted(landmarks)
    )
    out["TSH_velocity"] = 0.0
    out["FT4_velocity"] = 0.0
    if has_ft3:
        out["FT3_velocity"] = 0.0
    out["velocity_observed"] = 0
    for L in nonzero:
        prev_L = prev_landmark[L]
        dt = L - prev_L
        mask = out["landmark"] == L
        out.loc[mask, "TSH_velocity"] = (
            out.loc[mask, f"TSH_at_{L}M"] - out.loc[mask, f"TSH_at_{prev_L}M"]
        ) / dt
        out.loc[mask, "FT4_velocity"] = (
            out.loc[mask, f"FT4_at_{L}M"] - out.loc[mask, f"FT4_at_{prev_L}M"]
        ) / dt
        if has_ft3:
            out.loc[mask, "FT3_velocity"] = (
                out.loc[mask, f"FT3_at_{L}M"] - out.loc[mask, f"FT3_at_{prev_L}M"]
            ) / dt
        out.loc[mask, "velocity_observed"] = 1
    # Block E: time + interactions. Offset 1.5 is a fixed pre-registered constant
    # (NOT recomputed from `landmarks`) to keep the mainline byte-for-byte stable.
    out["landmark_time_centered"] = out["landmark"].astype(float) - 1.5
    out["is_baseline_landmark"] = (out["landmark"] == 0).astype(int)
    out["lm_x_TSH_current"] = out["landmark_time_centered"] * out["TSH_current"]
    out["lm_x_FT4_current"] = out["landmark_time_centered"] * out["FT4_current"]
    out["lm_x_TSH_velocity"] = out["landmark_time_centered"] * out["TSH_velocity"]
    out["lm_x_FT4_velocity"] = out["landmark_time_centered"] * out["FT4_velocity"]
    return out


def _impute_simple(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Train-only median imputation (default for fast smoke tests).

    For the full M2 v2 pipeline, MICE m=10 is the pre-registered choice (see
    arch.md); this simple median path is the complete-case sensitivity baseline.
    A full MICE implementation may be added in module2_v2_sensitivity.py.
    """
    out = df.copy()
    dev_mask = out["Split"] == "Development"
    for c in columns:
        if c not in out.columns:
            continue
        med = out.loc[dev_mask, c].median()
        out[c] = out[c].replace([np.inf, -np.inf], np.nan)
        out[c] = out[c].fillna(med)
    return out


def load_stacked(
    use_simple_impute: bool = True,
    landmarks: Sequence[int] = LANDMARKS,
    *,
    corrected: bool = False,
) -> StackedData:
    """Build the M2 v2 stacked landmark-row dataset.

    Returns
    -------
    StackedData with `rows` shape ≈ (1003·len(landmarks), ~30) and `episode_meta`
    shape (1003, 3).

    `landmarks` defaults to the mainline (0, 1, 3, 6) → 4012 rows, byte-for-byte
    stable; pass LANDMARKS_X (or call load_stacked_x) for the 5015-row extension.
    Includes all 5 mechanism blocks' features. Calls run_leakage_assertions()
    to verify column conventions match arch.md.

    ``corrected`` (default False — mainline is byte-for-byte unchanged) switches
    the current/audit hormone columns to the single-source-of-truth corrected pull
    (``_corrected_landmark_values``) and additionally materialises FT3_current /
    FT3_velocity, so the direct build_feats_at_L path reads real 6M/12M levels
    rather than the degenerate wide columns. ``load_stacked_x`` defaults it on.
    """
    landmarks = tuple(sorted(landmarks))
    n_lm = len(landmarks)
    # Load M1 frozen baseline matrix + stage2 long landmark data
    m1 = _load_m1_frozen()
    s2 = _load_stage2_landmark_features(landmarks=landmarks)
    # Inner join on Episode_Index to combine baseline burden + landmark dynamics
    combined = m1.merge(s2, on="Episode_Index", how="inner", validate="one_to_one")
    if len(combined) != EXPECTED_EPISODES:
        raise RuntimeError(
            f"Combined episode count {len(combined)} != expected {EXPECTED_EPISODES}"
        )
    # Reshape wide → long (n_lm landmark-rows per episode)
    long_df = _restack_long(combined, landmarks=landmarks, corrected=corrected)
    if len(long_df) != EXPECTED_EPISODES * n_lm:
        raise RuntimeError(
            f"Stacked row count {len(long_df)} != {EXPECTED_EPISODES}×{n_lm}"
        )
    # Run leakage assertions on raw stacked data BEFORE any imputation.
    # The assertions check lookup correctness (audit_TSH == TSH_current) and
    # are only meaningful on raw values; imputation legitimately changes both.
    ep_meta = (
        long_df.drop_duplicates("episode_id")[["episode_id", "Split", "Y_24M_NHRH"]]
        .sort_values("episode_id")
        .reset_index(drop=True)
    )
    if len(ep_meta) != EXPECTED_EPISODES:
        raise RuntimeError("Episode meta count mismatch")
    sd_raw = StackedData(rows=long_df, episode_meta=ep_meta)
    run_leakage_assertions(sd_raw, landmarks=landmarks)
    # Now add momentum + time-interaction features (uses raw audit columns).
    long_df = _add_blocks_D_and_E(long_df, landmarks=landmarks)
    # Impute (simple train-only median for now); audit columns kept raw.
    if use_simple_impute:
        impute_cols = list(ALL_BLOCKS_FEATURES)
        if corrected:
            # FT3 current+velocity were added by the corrected path; impute them
            # too (dev-median; a constant, not future info) so build_feats_at_L
            # sees no NaN. Audit *_at_{L}M columns stay raw for the leakage gate.
            impute_cols += ["FT3_current", "FT3_velocity"]
        long_df = _impute_simple(long_df, impute_cols)
    sd_final = StackedData(rows=long_df, episode_meta=ep_meta)
    # Re-run with full=True now that Block D/E features exist.
    run_leakage_assertions(sd_final, full=True, landmarks=landmarks)
    return sd_final


def load_stacked_x(use_simple_impute: bool = True, *, corrected: bool = True) -> StackedData:
    """Cross-landmark extension of load_stacked over LANDMARKS_X = (0,1,3,6,12).

    Identical logic to load_stacked but stacks 5 landmark-rows per episode
    (5015 rows total). The 12M FT3/FT4/TSH labs are ~100% covered in
    stage2_long_table, so the 12M rows are real (not imputed) for those.

    ``corrected`` defaults to **True** here: the extension exists to study the
    real 6M/12M hormone levels, so the current/audit columns use the corrected
    single-source-of-truth pull and FT3 current+velocity are materialised. Pass
    ``corrected=False`` to reproduce the legacy degenerate-wide-column behaviour.
    """
    return load_stacked(
        use_simple_impute=use_simple_impute, landmarks=LANDMARKS_X, corrected=corrected
    )


# ---------------------------------------------------------------------------
# Leakage assertions (Mn1: automated unit-test on lookup correctness)
# ---------------------------------------------------------------------------


def run_leakage_assertions(
    sd: StackedData, *, full: bool = False, landmarks: Sequence[int] = LANDMARKS
) -> None:
    """Pytest-style assertions on stacked data integrity.

    Raises RuntimeError on any violation. Pre-registered checks:
    1. Row count = 1003 × len(landmarks).
    2. Episode count = 1003.
    3. Each episode appears exactly len(landmarks) times (one per landmark).
    4. TSH_current at landmark L equals TSH_at_{L}M for that row.
    5. FT4_current at landmark L equals FT4_at_{L}M for that row.
    6. (full only) velocity_observed = 0 iff landmark = first landmark (0M).
    7. (full only) is_baseline_landmark = 1 iff landmark = 0M.
    8. (full only) landmark_time_centered = landmark − 1.5.
    9. Y_24M_NHRH constant within each episode (same value across all rows).
    10. Split assignment respects episode boundaries (same Split for all rows of an episode).

    Set `full=True` once Block D/E features have been added to also check those.
    `landmarks` defaults to the mainline (0,1,3,6); row-count and per-episode
    repeat checks scale to its length.
    """
    landmarks = tuple(sorted(landmarks))
    n_lm = len(landmarks)
    rows = sd.rows
    if len(rows) != EXPECTED_EPISODES * n_lm:
        raise RuntimeError(f"[leak-1] Row count {len(rows)} != {EXPECTED_EPISODES * n_lm}")
    n_ep = rows["episode_id"].nunique()
    if n_ep != EXPECTED_EPISODES:
        raise RuntimeError(f"[leak-2] Episode count {n_ep} != {EXPECTED_EPISODES}")
    counts = rows.groupby("episode_id").size()
    if not (counts == n_lm).all():
        bad = counts[counts != n_lm]
        raise RuntimeError(f"[leak-3] {len(bad)} episodes have != {n_lm} rows")
    # Check lookup correctness on a sample of 50 random episodes (full check
    # expensive). Only meaningful on raw (pre-imputation) data — `full=False`
    # path. After imputation `full=True` skips these because imputed
    # TSH_current may legitimately differ from raw audit_TSH (NaN→median).
    if not full:
        rng = np.random.default_rng(0)
        sample_ep = rng.choice(rows["episode_id"].unique(), size=50, replace=False)
        for ep in sample_ep:
            sub = rows[rows["episode_id"] == ep]
            for _, r in sub.iterrows():
                L = int(r["landmark"])
                audit_tsh = r[f"TSH_at_{L}M"]
                cur_tsh = r["TSH_current"]
                if not np.isclose(audit_tsh, cur_tsh, equal_nan=True):
                    raise RuntimeError(
                        f"[leak-4] ep={ep} L={L}: TSH_current={cur_tsh} != TSH_at_{L}M={audit_tsh}"
                    )
                audit_ft4 = r[f"FT4_at_{L}M"]
                cur_ft4 = r["FT4_current"]
                if not np.isclose(audit_ft4, cur_ft4, equal_nan=True):
                    raise RuntimeError(
                        f"[leak-5] ep={ep} L={L}: FT4_current={cur_ft4} != FT4_at_{L}M={audit_ft4}"
                    )
    if full:
        vobs_0 = rows.loc[rows["landmark"] == 0, "velocity_observed"]
        if not (vobs_0 == 0).all():
            raise RuntimeError("[leak-6a] velocity_observed != 0 at landmark=0M")
        vobs_gt0 = rows.loc[rows["landmark"] > 0, "velocity_observed"]
        if not (vobs_gt0 == 1).all():
            raise RuntimeError("[leak-6b] velocity_observed != 1 at landmark>0M")
        if not (
            rows.loc[rows["landmark"] == 0, "is_baseline_landmark"].eq(1).all()
            and rows.loc[rows["landmark"] > 0, "is_baseline_landmark"].eq(0).all()
        ):
            raise RuntimeError("[leak-7] is_baseline_landmark mismatch")
        expected_lt = rows["landmark"].astype(float) - 1.5
        if not np.allclose(rows["landmark_time_centered"], expected_lt):
            raise RuntimeError("[leak-8] landmark_time_centered != landmark - 1.5")
    y_by_ep = rows.groupby("episode_id")["Y_24M_NHRH"].nunique()
    if not (y_by_ep == 1).all():
        raise RuntimeError("[leak-9] Y_24M_NHRH not constant within episode")
    split_by_ep = rows.groupby("episode_id")["Split"].nunique()
    if not (split_by_ep == 1).all():
        raise RuntimeError("[leak-10] Split varies within episode")
    # Forbidden-token audit
    tok = forbidden_token()
    n_ep_check = rows["episode_id"].nunique()
    if str(n_ep_check) == tok:
        raise RuntimeError(f"[leak-forbidden] episode count == forbidden token; report '1003 episodes'")


# ---------------------------------------------------------------------------
# CV + Bootstrap + Calibration helpers (all episode-aware)
# ---------------------------------------------------------------------------


def make_episode_stratified_group_kfold() -> StratifiedGroupKFold:
    """Construct the canonical CV splitter for all 3 M2 v2 tracks."""
    return StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)


def fit_l2_predict_oof(
    sd: StackedData,
    block_subset: Sequence[str] | None = None,
    *,
    C: float = 1.0,
) -> tuple[np.ndarray, LogisticRegression]:
    """Run 5-fold StratifiedGroupKFold OOF L2-logistic on stacked rows.

    Returns
    -------
    oof_proba : array length 4012 of OOF predictions on dev rows + train-fit
        predictions on temporal rows.
    final_model : fitted on all dev rows.

    Splits: dev rows (Split=Development) participate in CV; temporal rows scored
    by the final dev-fit model.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X_all = sd.feature_matrix(block_subset).values
    y_all = sd.rows["Y_24M_NHRH"].values
    ep_all = sd.rows["episode_id"].values
    is_dev = (sd.rows["Split"] == "Development").values
    is_test = ~is_dev

    X_dev = X_all[is_dev]
    y_dev = y_all[is_dev]
    ep_dev = ep_all[is_dev]

    oof_all = np.zeros(len(sd.rows), dtype=float)

    splitter = make_episode_stratified_group_kfold()
    for tr_idx, va_idx in splitter.split(X_dev, y_dev, groups=ep_dev):
        pipe = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        penalty="l2", solver="lbfgs", C=C,
                        max_iter=5000, random_state=PY_SEED,
                    ),
                ),
            ]
        )
        pipe.fit(X_dev[tr_idx], y_dev[tr_idx])
        oof_dev_proba = pipe.predict_proba(X_dev[va_idx])[:, 1]
        dev_indices = np.where(is_dev)[0]
        oof_all[dev_indices[va_idx]] = oof_dev_proba

    final = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    penalty="l2", solver="lbfgs", C=C,
                    max_iter=5000, random_state=PY_SEED,
                ),
            ),
        ]
    )
    final.fit(X_dev, y_dev)
    if is_test.any():
        oof_all[is_test] = final.predict_proba(X_all[is_test])[:, 1]
    return oof_all, final


def per_landmark_platt_on_pooled_oof(
    sd: StackedData, oof_proba: np.ndarray
) -> dict[int, tuple[float, float]]:
    """Fit one Platt logistic per landmark on pooled outer-OOF dev predictions.

    Returns dict landmark → (intercept, slope) on logit(p) scale.
    Apply via: calibrated_p = 1 / (1 + exp(-(intercept + slope * logit(raw))))
    """
    out: dict[int, tuple[float, float]] = {}
    is_dev = sd.rows["Split"].values == "Development"
    for L in LANDMARKS:
        mask = is_dev & (sd.rows["landmark"].values == L)
        if mask.sum() < 20:
            out[L] = (0.0, 1.0)  # identity
            continue
        p = np.clip(oof_proba[mask], 1e-6, 1 - 1e-6)
        z = np.log(p / (1 - p)).reshape(-1, 1)
        y = sd.rows.loc[mask, "Y_24M_NHRH"].values
        platt = LogisticRegression(solver="lbfgs", max_iter=2000)
        platt.fit(z, y)
        out[L] = (float(platt.intercept_[0]), float(platt.coef_[0][0]))
    return out


def apply_per_landmark_platt(
    sd: StackedData, raw_proba: np.ndarray, calibrators: dict[int, tuple[float, float]]
) -> np.ndarray:
    """Apply per-landmark Platt scalers to raw predictions."""
    out = np.zeros_like(raw_proba)
    for L in LANDMARKS:
        mask = sd.rows["landmark"].values == L
        if not mask.any():
            continue
        a, b = calibrators[L]
        p = np.clip(raw_proba[mask], 1e-6, 1 - 1e-6)
        z = np.log(p / (1 - p))
        out[mask] = 1.0 / (1.0 + np.exp(-(a + b * z)))
    return out


def compute_metrics(
    y: np.ndarray, proba: np.ndarray
) -> dict[str, float]:
    """ROC-AUC / PR-AUC / Brier on a single subset (e.g., one landmark)."""
    if len(np.unique(y)) < 2:
        return {"ROC_AUC": float("nan"), "PR_AUC": float("nan"), "Brier": float("nan")}
    return {
        "ROC_AUC": float(roc_auc_score(y, proba)),
        "PR_AUC": float(average_precision_score(y, proba)),
        "Brier": float(brier_score_loss(y, proba)),
    }


def calib_intercept_slope(y: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    p = np.clip(proba, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs", max_iter=2000)
    lr.fit(z, y)
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


def episode_cluster_bootstrap_auc_ci(
    sd: StackedData,
    proba: np.ndarray,
    landmark: int | None = None,
    *,
    split: str = "Temporal",
    n: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Episode-level cluster bootstrap: resample episodes, include all 4 rows.

    Returns (mean, CI_low, CI_high) for ROC-AUC on the given split (and optional landmark).
    """
    is_split = sd.rows["Split"].values == split
    is_lm = (
        sd.rows["landmark"].values == landmark if landmark is not None else np.ones(len(sd.rows), dtype=bool)
    )
    keep = is_split & is_lm
    if keep.sum() < 5:
        return float("nan"), float("nan"), float("nan")
    eps = sd.rows.loc[keep, "episode_id"].values
    y_keep = sd.rows.loc[keep, "Y_24M_NHRH"].values
    p_keep = proba[keep]
    unique_eps = np.unique(eps)
    ep_to_indices = {e: np.where(eps == e)[0] for e in unique_eps}
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    for _ in range(n):
        sampled_eps = rng.choice(unique_eps, size=len(unique_eps), replace=True)
        idx_concat = np.concatenate([ep_to_indices[e] for e in sampled_eps])
        y_b = y_keep[idx_concat]
        p_b = p_keep[idx_concat]
        if len(np.unique(y_b)) < 2:
            continue
        aucs.append(roc_auc_score(y_b, p_b))
    if not aucs:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.mean(aucs)),
        float(np.percentile(aucs, 2.5)),
        float(np.percentile(aucs, 97.5)),
    )


def paired_episode_cluster_bootstrap_delta(
    sd: StackedData,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    *,
    landmark: int | None = None,
    split: str = "Temporal",
    n: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Paired episode-cluster bootstrap on ΔAUC = AUC_a − AUC_b.

    Same resampled episodes applied to both prediction vectors; Δ computed per
    resample then percentiles reported.
    """
    is_split = sd.rows["Split"].values == split
    is_lm = (
        sd.rows["landmark"].values == landmark if landmark is not None else np.ones(len(sd.rows), dtype=bool)
    )
    keep = is_split & is_lm
    if keep.sum() < 5:
        return float("nan"), float("nan"), float("nan")
    eps = sd.rows.loc[keep, "episode_id"].values
    y_keep = sd.rows.loc[keep, "Y_24M_NHRH"].values
    pa = proba_a[keep]
    pb = proba_b[keep]
    unique_eps = np.unique(eps)
    ep_to_indices = {e: np.where(eps == e)[0] for e in unique_eps}
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(n):
        sampled_eps = rng.choice(unique_eps, size=len(unique_eps), replace=True)
        idx_concat = np.concatenate([ep_to_indices[e] for e in sampled_eps])
        y_b = y_keep[idx_concat]
        if len(np.unique(y_b)) < 2:
            continue
        deltas.append(roc_auc_score(y_b, pa[idx_concat]) - roc_auc_score(y_b, pb[idx_concat]))
    if not deltas:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.mean(deltas)),
        float(np.percentile(deltas, 2.5)),
        float(np.percentile(deltas, 97.5)),
    )


# ---------------------------------------------------------------------------
# Convenience: per-landmark + pooled performance summary
# ---------------------------------------------------------------------------


def per_landmark_and_pooled_perf(
    sd: StackedData, proba: np.ndarray, split: str = "TemporalTest"
) -> pd.DataFrame:
    """Build a per-landmark + pooled performance table on a single split."""
    rows: list[dict] = []
    is_split = sd.rows["Split"].values == split
    y_all = sd.rows["Y_24M_NHRH"].values
    for L in LANDMARKS:
        m = is_split & (sd.rows["landmark"].values == L)
        if m.sum() < 5:
            continue
        metrics = compute_metrics(y_all[m], proba[m])
        ic, sl = calib_intercept_slope(y_all[m], proba[m])
        auc_mean, auc_lo, auc_hi = episode_cluster_bootstrap_auc_ci(
            sd, proba, landmark=L, split=split
        )
        rows.append(
            {
                "Split": split,
                "Landmark": f"{L}M",
                "N": int(m.sum()),
                "Events": int(y_all[m].sum()),
                **metrics,
                "ROC_AUC_CI_Low": auc_lo,
                "ROC_AUC_CI_High": auc_hi,
                "CalibIntercept": ic,
                "CalibSlope": sl,
            }
        )
    # Pooled
    m = is_split
    if m.sum() >= 5:
        metrics = compute_metrics(y_all[m], proba[m])
        ic, sl = calib_intercept_slope(y_all[m], proba[m])
        auc_mean, auc_lo, auc_hi = episode_cluster_bootstrap_auc_ci(
            sd, proba, landmark=None, split=split
        )
        rows.append(
            {
                "Split": split,
                "Landmark": "Pooled",
                "N": int(m.sum()),
                "Events": int(y_all[m].sum()),
                **metrics,
                "ROC_AUC_CI_Low": auc_lo,
                "ROC_AUC_CI_High": auc_hi,
                "CalibIntercept": ic,
                "CalibSlope": sl,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Smoke test entry point
# ---------------------------------------------------------------------------


def _smoke_test() -> None:
    """Quick sanity check: load + stack + assertions + L2 fit + per-landmark perf."""
    print("Loading stacked dataset…")
    sd = load_stacked()
    print(f"  rows shape: {sd.rows.shape}")
    print(f"  episodes:   {sd.episode_meta.shape[0]}")
    print(f"  Split:      {sd.episode_meta['Split'].value_counts().to_dict()}")
    print(f"  Prevalence: {sd.episode_meta['Y_24M_NHRH'].mean():.3f}")
    print("  leakage assertions: PASS")

    print("\nFitting L2 supermodel (full A+B+C+D+E)…")
    raw, model = fit_l2_predict_oof(sd, block_subset=("A", "B", "C", "D", "E"))
    print(f"  raw proba shape: {raw.shape}")

    print("\nFitting per-landmark Platt on pooled OOF…")
    cal = per_landmark_platt_on_pooled_oof(sd, raw)
    for L, (a, b) in cal.items():
        print(f"  landmark {L}M: intercept={a:+.3f}, slope={b:+.3f}")

    proba_cal = apply_per_landmark_platt(sd, raw, cal)
    print("\nPer-landmark + pooled performance:")
    print("  -- Dev OOF --")
    print(per_landmark_and_pooled_perf(sd, proba_cal, split="Development").to_string(index=False))
    print("  -- Temporal --")
    print(per_landmark_and_pooled_perf(sd, proba_cal, split="Temporal").to_string(index=False))


if __name__ == "__main__":
    _smoke_test()
