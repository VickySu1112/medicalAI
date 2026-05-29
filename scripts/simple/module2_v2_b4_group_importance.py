#!/usr/bin/env python
"""M2 · B4 — collinearity-robust per-landmark GROUP importance drift.

Raw LR coefficients are not a valid importance ranking under collinearity
(FT3≈FT4, r=0.92 → sign-flipping "coefficient cancellation"; magnitude ≠
importance, and the ranking reshuffles whenever you add a derived feature).

Correct method (primary): per-landmark GROUP drop-AUC — remove a whole clinical
group, refit, measure the OOF AUC drop. This is prediction-grounded and
collinearity-robust (the entire correlated group is removed together). To make
the hormone groups cleanly separable we rotate the collinear (FT3,FT4) pair into
two ORTHOGONAL axes per landmark:

    Hormone_load    = (zFT3 + zFT4)/√2      # overall thyroid-hormone level
    T3T4_balance    = (zFT3 − zFT4)/√2      # T3-vs-T4 dominance (T3-toxicosis axis)
    Velocity_load   = (zΔFT3 + zΔFT4)/√2    # overall hormone momentum
    Velocity_balance= (zΔFT3 − zΔFT4)/√2    # T3-vs-T4 momentum

z-scoring is done within each landmark's dev rows (project convention is dev-only
standardisation). Raw-coefficient ranking is reported only as a DIAGNOSTIC.
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.simple.module2_v2_shared import BLOCKS, LANDMARKS, load_stacked, PY_SEED
from scripts.simple.module2_v2_b4_trab_patch import add_current_velocity

OUT = ROOT / "results" / "module2_v2_vertical" / "b4_target_gru"
SQRT2 = np.sqrt(2.0)


def _z(v, m):
    """z-score column v using dev-mask m's mean/std (constant -> zeros)."""
    mu = np.nanmean(v[m]); sd = np.nanstd(v[m])
    if not np.isfinite(sd) or sd < 1e-9:
        return np.zeros_like(v)
    z = (v - mu) / sd
    return np.nan_to_num(z, nan=0.0)


def build_axes_for_landmark(rows, mask_devL):
    """Return a DataFrame of grouped features for the rows, axes z-scored on dev@L."""
    ft3 = rows["FT3_current"].values.astype(float); ft4 = rows["FT4_current"].values.astype(float)
    dft3 = rows["FT3_velocity"].values.astype(float); dft4 = rows["FT4_velocity"].values.astype(float)
    zft3, zft4 = _z(ft3, mask_devL), _z(ft4, mask_devL)
    zdft3, zdft4 = _z(dft3, mask_devL), _z(dft4, mask_devL)
    df = pd.DataFrame(index=rows.index)
    df["Hormone_load"] = (zft3 + zft4) / SQRT2
    df["T3T4_balance"] = (zft3 - zft4) / SQRT2
    df["Velocity_load"] = (zdft3 + zdft4) / SQRT2
    df["Velocity_balance"] = (zdft3 - zdft4) / SQRT2
    for c in ["ThyroidW", "TRAb", "TGAb", "TPOAb", "Sex", "log1p_DiseaseDuration_Months_Aug",
              "Uptake24h", "HalfLife", "TSH_current", "TSH_velocity"]:
        df[c] = rows[c].values
    return df


GROUPS = {
    "Goiter (ThyroidW)": ["ThyroidW"],
    "Antibodies": ["TRAb", "TGAb", "TPOAb"],
    "Chronicity/Sex": ["log1p_DiseaseDuration_Months_Aug", "Sex"],
    "RAI exposure": ["Uptake24h", "HalfLife"],
    "Hormone level": ["Hormone_load", "TSH_current"],
    "T3/T4 balance": ["T3T4_balance"],
    "Momentum (load)": ["Velocity_load", "TSH_velocity"],
    "T3/T4 momentum balance": ["Velocity_balance"],
}
ALL_COLS = [c for cols in GROUPS.values() for c in cols]


def _oof_auc(X, y, seed=13):
    if X.shape[1] == 0:
        return 0.5
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y):
        pipe = Pipeline([("s", StandardScaler()),
                         ("lr", LogisticRegression(penalty="l2", solver="lbfgs", C=1.0,
                                                   max_iter=5000, random_state=PY_SEED))])
        pipe.fit(X[tr], y[tr]); oof[va] = pipe.predict_proba(X[va])[:, 1]
    return roc_auc_score(y, oof) if len(np.unique(y)) > 1 else 0.5


def main() -> None:
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    sd = load_stacked()
    sd = add_current_velocity(sd, "FT3")
    rows = sd.rows
    y_all = rows["Y_24M_NHRH"].values
    lm = rows["landmark"].values
    is_dev = (rows["Split"] == "Development").values

    drift = {g: {} for g in GROUPS}
    full_auc = {}
    for L in LANDMARKS:
        maskL = is_dev & (lm == L)
        feat_df = build_axes_for_landmark(rows, maskL)
        Xdf = feat_df.loc[maskL, ALL_COLS]
        # drop axis groups that are degenerate (all-constant, e.g. velocity at 0M)
        live_cols = [c for c in ALL_COLS if Xdf[c].std() > 1e-9]
        y = y_all[maskL]
        auc_full = _oof_auc(Xdf[live_cols].values, y)
        full_auc[L] = auc_full
        for g, cols in GROUPS.items():
            keep = [c for c in live_cols if c not in cols]
            auc_wo = _oof_auc(Xdf[keep].values, y)
            drift[g][L] = round(auc_full - auc_wo, 4)  # ΔAUC = importance of group

    # Drift table: group × landmark ΔAUC
    table = pd.DataFrame({f"{L}M": {g: drift[g][L] for g in GROUPS} for L in LANDMARKS})
    table.insert(0, "group", table.index)
    table.to_csv(OUT / "tables" / "group_dropauc_drift.csv", index=False)
    print("OOF AUC (full model) per landmark:",
          {f"{L}M": round(full_auc[L], 4) for L in LANDMARKS}, flush=True)
    print("\n=== Group drop-AUC (ΔAUC = AUC_full − AUC_without_group), per landmark ===", flush=True)
    print(table.to_string(index=False), flush=True)

    print("\n=== Per-landmark group leaderboard (top by ΔAUC) ===", flush=True)
    lead = {}
    for L in LANDMARKS:
        ranked = sorted(GROUPS, key=lambda g: drift[g][L], reverse=True)
        lead[f"{L}M"] = [(g, drift[g][L]) for g in ranked[:4]]
        print(f"  {L}M: " + "  |  ".join(f"{g} {drift[g][L]:+.3f}" for g in ranked[:4]), flush=True)

    # Heatmap
    mat = np.array([[drift[g][L] for L in LANDMARKS] for g in GROUPS], dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=max(0.02, np.nanmax(mat)), aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{L}M" for L in LANDMARKS])
    ax.set_yticks(range(len(GROUPS))); ax.set_yticklabels(list(GROUPS))
    for i, g in enumerate(GROUPS):
        for j, L in enumerate(LANDMARKS):
            v = drift[g][L]
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=8,
                    color="white" if v > max(0.02, np.nanmax(mat)) * 0.6 else "#222")
    ax.set_title("Per-landmark GROUP importance (drop-AUC) — collinearity-robust")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="ΔAUC (OOF)")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "Figure_07_GroupImportanceDrift.png", dpi=160); plt.close(fig)

    (OUT / "tables" / "group_importance_summary.json").write_text(json.dumps({
        "method": "per-landmark group drop-AUC (OOF), orthogonal hormone axes",
        "full_oof_auc": {f"{L}M": round(full_auc[L], 4) for L in LANDMARKS},
        "drift": {g: {f"{L}M": drift[g][L] for L in LANDMARKS} for g in GROUPS},
        "leaderboard_top4": lead,
    }, indent=2, ensure_ascii=False))
    print(f"\nSaved → {OUT}", flush=True)


if __name__ == "__main__":
    main()
