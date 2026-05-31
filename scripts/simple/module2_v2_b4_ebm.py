#!/usr/bin/env python
"""M2 · B4 — EBM (Explainable Boosting Machine) per-landmark: does a cooler,
nonlinear glassbox make MOMENTUM (velocity) show up more than LR, and does it
beat the LR ceiling?

EBM = GAM with automatic pairwise interactions; fully interpretable (each feature
gets a nonlinear shape + native importance), so velocity is a first-class term
(not absorbed like in a GRU). We report per-landmark:
  * EBM temporal AUC vs LR (does nonlinearity break the ceiling?)
  * group permutation importance on temporal (same method as LR/GRU drift) — is
    Momentum bigger under EBM?
  * EBM native term importances (top terms incl. velocity / interactions)
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
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from interpret.glassbox import ExplainableBoostingClassifier

from scripts.simple.module2_v2_shared import LANDMARKS, load_stacked, PY_SEED
from scripts.simple.module2_v2_b4_trab_patch import add_current_velocity

OUT = ROOT / "results" / "module2_v2_vertical" / "b4_target_gru"

FEATS = ["ThyroidW", "TRAb", "TGAb", "TPOAb", "Sex", "FT4_0M", "TSH_0M",
         "log1p_DiseaseDuration_Months_Aug", "Uptake24h", "HalfLife",
         "TSH_current", "FT4_current", "FT3_current",
         "TSH_velocity", "FT4_velocity", "FT3_velocity"]
GROUPS = {
    "Goiter (ThyroidW)": ["ThyroidW"],
    "Antibodies": ["TRAb", "TGAb", "TPOAb"],
    "Baseline labs/chronicity": ["FT4_0M", "TSH_0M", "log1p_DiseaseDuration_Months_Aug", "Sex"],
    "RAI exposure": ["Uptake24h", "HalfLife"],
    "Current level (C)": ["TSH_current", "FT4_current", "FT3_current"],
    "Momentum (velocity)": ["TSH_velocity", "FT4_velocity", "FT3_velocity"],
}


def _roc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")


def main() -> None:
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    sd = load_stacked()
    sd = add_current_velocity(sd, "FT3")
    rows = sd.rows
    y = rows["Y_24M_NHRH"].values
    lm = rows["landmark"].values
    is_dev = (rows["Split"] == "Development").values
    col = {f: i for i, f in enumerate(FEATS)}

    rng = np.random.default_rng(7)
    auc_cmp = []
    drift = {g: {} for g in GROUPS}
    native = {}
    for L in LANDMARKS:
        devL = is_dev & (lm == L); tstL = (~is_dev) & (lm == L)
        Xtr = rows.loc[devL, FEATS].values; ytr = y[devL]
        Xte = rows.loc[tstL, FEATS].values; yte = y[tstL]

        # LR baseline
        lr = Pipeline([("s", StandardScaler()),
                       ("lr", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                                 max_iter=5000, random_state=PY_SEED))]).fit(Xtr, ytr)
        lr_auc = _roc(yte, lr.predict_proba(Xte)[:, 1])

        # EBM
        ebm = ExplainableBoostingClassifier(random_state=PY_SEED, interactions=5, max_interaction_bins=16)
        ebm.fit(Xtr, ytr)
        base = ebm.predict_proba(Xte)[:, 1]
        ebm_auc = _roc(yte, base)
        auc_cmp.append({"landmark": f"{L}M", "LR_AUC": round(lr_auc, 4), "EBM_AUC": round(ebm_auc, 4),
                        "EBM_minus_LR": round(ebm_auc - lr_auc, 4)})

        # group permutation importance on temporal
        for g, cols in GROUPS.items():
            idx = [col[c] for c in cols if c in col]
            drops = []
            for _ in range(10):
                Xp = Xte.copy(); perm = rng.permutation(Xp.shape[0])
                Xp[:, idx] = Xp[perm][:, idx]
                drops.append(ebm_auc - _roc(yte, ebm.predict_proba(Xp)[:, 1]))
            drift[g][L] = round(float(np.mean(drops)), 4)

        # EBM native term importances (top 6)
        g_exp = ebm.explain_global()
        names = g_exp.data()["names"]; scores = g_exp.data()["scores"]
        top = sorted(zip(names, scores), key=lambda t: -t[1])[:6]
        native[f"{L}M"] = [(n, round(float(s), 4)) for n, s in top]

    cmp_df = pd.DataFrame(auc_cmp)
    print("=== EBM vs LR per-landmark temporal AUC ===", flush=True)
    print(cmp_df.to_string(index=False), flush=True)

    table = pd.DataFrame({f"{L}M": {g: drift[g][L] for g in GROUPS} for L in LANDMARKS})
    table.insert(0, "group", table.index)
    table.to_csv(OUT / "tables" / "ebm_perm_importance_drift.csv", index=False)
    print("\n=== EBM group permutation importance (ΔAUC, temporal) — is Momentum bigger? ===", flush=True)
    print(table.to_string(index=False), flush=True)

    print("\n=== EBM native top terms per landmark (incl. nonlinear shapes & interactions) ===", flush=True)
    for L in LANDMARKS:
        print(f"  {L}M: " + "  |  ".join(f"{n} {s:.3f}" for n, s in native[f'{L}M']), flush=True)

    (OUT / "tables" / "ebm_summary.json").write_text(json.dumps({
        "auc_vs_lr": auc_cmp,
        "perm_drift": {g: drift[g] for g in GROUPS},
        "native_top_terms": native,
    }, indent=2, ensure_ascii=False))
    print(f"\nSaved → {OUT}", flush=True)


if __name__ == "__main__":
    main()
