#!/usr/bin/env python
"""M2 · B4 v2 report builder (fork) — regenerate the clean, renamed master figures
and a consolidated summary for the M2-v2 interpretability report.

Naming is final & explicit (no "load"/"balance" jargon in user-facing output):
  Hormone_load     -> FT3+FT4 level     (the two peripheral hormones' joint level)
  Velocity_load    -> FT3+FT4 velocity
  T3T4_balance     -> FT3-FT4 gap       (T3-vs-T4 dominance)
  Velocity_balance -> FT3-FT4 vel gap
  TSH kept separate (pituitary feedback). ThyroidW = goiter weight.

Regenerates: per-landmark GROUP drop-AUC heatmap (collinearity-robust importance
drift) + a consolidated v2 summary pulling EBM / tuning / selective-prediction
results already on disk.
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

from scripts.simple.module2_v2_shared import LANDMARKS, load_stacked, PY_SEED
from scripts.simple.module2_v2_b4_trab_patch import add_current_velocity
from scripts.simple.module2_v2_b4_ebm_axes import build_feats_at_L, FEATS

OUT = ROOT / "results" / "module2_v2_vertical" / "b4_target_gru"

# clean, explicit group labels (no "load"/"balance")
GROUPS = {
    "ThyroidW (goiter)": ["ThyroidW"],
    "Antibodies": ["TRAb", "TGAb", "TPOAb"],
    "Baseline labs / duration": ["FT4_0M", "TSH_0M", "log1p_DiseaseDuration_Months_Aug", "Sex"],
    "RAI exposure": ["Uptake24h", "HalfLife"],
    "FT3,FT4 combined level & TSH": ["Hormone_load", "TSH_current"],
    "FT3,FT4 gap (T3 dominance)": ["T3T4_balance"],
    "FT3,FT4 combined velocity & TSH": ["Velocity_load", "TSH_velocity"],
    "FT3,FT4 velocity gap": ["Velocity_balance"],
}


def _roc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5


def _oof_auc(X, y, seed=13):
    if X.shape[1] == 0:
        return 0.5
    oof = np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        pipe = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=5000, random_state=PY_SEED))]).fit(X[tr], y[tr])
        oof[va] = pipe.predict_proba(X[va])[:, 1]
    return _roc(y, oof)


def main() -> None:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    sd = load_stacked(); sd = add_current_velocity(sd, "FT3")
    rows = sd.rows
    y = rows["Y_24M_NHRH"].values; lm = rows["landmark"].values
    is_dev = (rows["Split"] == "Development").values

    drift = {g: {} for g in GROUPS}
    full = {}
    for L in LANDMARKS:
        maskL = is_dev & (lm == L)
        feat = build_feats_at_L(rows, maskL)
        live = [c for c in FEATS if feat.loc[maskL, c].std() > 1e-9]
        X = feat.loc[maskL, live].values; yy = y[maskL]
        afull = _oof_auc(X, yy); full[L] = afull
        for g, cols in GROUPS.items():
            keep = [c for c in live if c not in cols]
            drift[g][L] = round(afull - _oof_auc(feat.loc[maskL, keep].values, yy), 4)

    tbl = pd.DataFrame({f"{L}M": {g: drift[g][L] for g in GROUPS} for L in LANDMARKS})
    tbl.insert(0, "group", tbl.index)
    tbl.to_csv(OUT / "tables" / "v2_group_importance_drift.csv", index=False)
    print("Full OOF AUC:", {f"{L}M": round(full[L], 4) for L in LANDMARKS}, flush=True)
    print(tbl.to_string(index=False), flush=True)

    # Master heatmap (clean English labels — no load/balance)
    mat = np.array([[drift[g][L] for L in LANDMARKS] for g in GROUPS])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=max(0.02, mat.max()), aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{L}M" for L in LANDMARKS])
    ax.set_yticks(range(len(GROUPS))); ax.set_yticklabels(list(GROUPS))
    for i, g in enumerate(GROUPS):
        for j, L in enumerate(LANDMARKS):
            v = drift[g][L]
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=8,
                    color="white" if v > max(0.02, mat.max()) * 0.6 else "#222")
    ax.set_title("M2-v2 group importance drift (LR drop-AUC, OOF) — goiter→level→velocity")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="ΔAUC")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "Figure_08_v2_ImportanceDrift.png", dpi=160); plt.close(fig)

    # consolidate prior results
    def _load(p):
        f = OUT / "tables" / p
        return json.loads(f.read_text()) if f.exists() else None
    summary = {
        "naming": {"Hormone_load": "FT3+FT4 level", "Velocity_load": "FT3+FT4 velocity",
                   "T3T4_balance": "FT3-FT4 gap", "Velocity_balance": "FT3-FT4 vel gap"},
        "full_oof_auc": {f"{L}M": round(full[L], 4) for L in LANDMARKS},
        "group_drift": {g: drift[g] for g in GROUPS},
        "ebm_axes": _load("ebm_axes_summary.json"),
        "ebm_tuning": _load("ebm_tuning.json"),
        "selective_prediction": _load("subtypes_abstain_summary.json"),
    }
    (OUT / "tables" / "v2_report_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved master figure + v2_report_summary.json → {OUT}", flush=True)


if __name__ == "__main__":
    main()
