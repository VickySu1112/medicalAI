#!/usr/bin/env python
"""M2 · B4 — EBM on AGGREGATED orthogonal axes (consistent with the group table).

Unifies the parameterisation: no independent FT3/FT4 (which re-introduce
collinear attribution). The collinear (FT3,FT4) pair and its velocity are rotated
into orthogonal axes per landmark (z-scored on dev@L, applied to all rows):

    Hormone_load     = (zFT3 + zFT4)/√2
    T3T4_balance     = (zFT3 − zFT4)/√2
    Velocity_load    = (zΔFT3 + zΔFT4)/√2
    Velocity_balance = (zΔFT3 − zΔFT4)/√2

TSH kept separate (pituitary feedback axis, inversely related). Per landmark we
report EBM native top-3 terms (now in axis terms), EBM vs LR temporal AUC, and
group permutation importance — directly comparable to the LR group drop-AUC table.
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
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.simple.module2_v2_shared import LANDMARKS, load_stacked, PY_SEED
from scripts.simple.module2_v2_b4_trab_patch import add_current_velocity

OUT = ROOT / "results" / "module2_v2_vertical" / "b4_target_gru"
SQRT2 = np.sqrt(2.0)
STATIC = ["ThyroidW", "TRAb", "TGAb", "TPOAb", "Sex", "FT4_0M", "TSH_0M",
          "log1p_DiseaseDuration_Months_Aug", "Uptake24h", "HalfLife"]
AXES = ["TSH_current", "TSH_velocity", "Hormone_load", "T3T4_balance", "Velocity_load", "Velocity_balance"]
FEATS = STATIC + AXES
GROUPS = {
    "Goiter (ThyroidW)": ["ThyroidW"],
    "Antibodies": ["TRAb", "TGAb", "TPOAb"],
    "Baseline labs/chronicity": ["FT4_0M", "TSH_0M", "log1p_DiseaseDuration_Months_Aug", "Sex"],
    "RAI exposure": ["Uptake24h", "HalfLife"],
    "Hormone level (load)": ["Hormone_load", "TSH_current"],
    "T3/T4 balance": ["T3T4_balance"],
    "Momentum (load)": ["Velocity_load", "TSH_velocity"],
    "T3/T4 momentum balance": ["Velocity_balance"],
}


def _roc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")


def _zfit(v, m):
    mu = np.nanmean(v[m]); sd = np.nanstd(v[m])
    if not np.isfinite(sd) or sd < 1e-9:
        return np.zeros_like(v)
    return np.nan_to_num((v - mu) / sd, nan=0.0)


def build_feats_at_L(rows, maskL):
    ft3 = rows["FT3_current"].values.astype(float); ft4 = rows["FT4_current"].values.astype(float)
    dft3 = rows["FT3_velocity"].values.astype(float); dft4 = rows["FT4_velocity"].values.astype(float)
    z3, z4 = _zfit(ft3, maskL), _zfit(ft4, maskL)
    dz3, dz4 = _zfit(dft3, maskL), _zfit(dft4, maskL)
    df = pd.DataFrame(index=rows.index)
    for c in STATIC + ["TSH_current", "TSH_velocity"]:
        df[c] = rows[c].values
    df["Hormone_load"] = (z3 + z4) / SQRT2
    df["T3T4_balance"] = (z3 - z4) / SQRT2
    df["Velocity_load"] = (dz3 + dz4) / SQRT2
    df["Velocity_balance"] = (dz3 - dz4) / SQRT2
    return df[FEATS]


def main() -> None:
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    sd = load_stacked(); sd = add_current_velocity(sd, "FT3")
    rows = sd.rows
    y = rows["Y_24M_NHRH"].values; lm = rows["landmark"].values
    is_dev = (rows["Split"] == "Development").values
    rng = np.random.default_rng(7)

    auc_cmp = []; drift = {g: {} for g in GROUPS}; native = {}
    for L in LANDMARKS:
        devL = is_dev & (lm == L); tstL = (~is_dev) & (lm == L)
        feat_all = build_feats_at_L(rows, devL)   # axes z-fit on dev@L, applied to all
        Xtr = feat_all.loc[devL]; ytr = y[devL]
        Xte = feat_all.loc[tstL]; yte = y[tstL]
        live = [c for c in FEATS if Xtr[c].std() > 1e-9]   # drop constant (velocity@0M)
        Xtr, Xte = Xtr[live], Xte[live]

        lr = Pipeline([("s", StandardScaler()),
                       ("lr", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                                 max_iter=5000, random_state=PY_SEED))]).fit(Xtr.values, ytr)
        lr_auc = _roc(yte, lr.predict_proba(Xte.values)[:, 1])
        ebm = ExplainableBoostingClassifier(random_state=PY_SEED, interactions=5, max_interaction_bins=16).fit(Xtr, ytr)
        base = ebm.predict_proba(Xte)[:, 1]; ebm_auc = _roc(yte, base)
        auc_cmp.append({"landmark": f"{L}M", "LR_AUC": round(lr_auc, 4), "EBM_AUC": round(ebm_auc, 4)})

        for g, cols in GROUPS.items():
            idx = [c for c in cols if c in live]
            if not idx:
                drift[g][L] = 0.0; continue
            drops = []
            for _ in range(10):
                Xp = Xte.copy(); perm = rng.permutation(len(Xp))
                Xp[idx] = Xp[idx].values[perm]
                drops.append(ebm_auc - _roc(yte, ebm.predict_proba(Xp)[:, 1]))
            drift[g][L] = round(float(np.mean(drops)), 4)

        d = ebm.explain_global().data(); terms = sorted(zip(d["names"], d["scores"]), key=lambda t: -t[1])
        mains = [(n, round(float(s), 3)) for n, s in terms if "&" not in n][:3]
        native[f"{L}M"] = mains

    cmp_df = pd.DataFrame(auc_cmp)
    print("=== EBM(轴) vs LR(轴) temporal AUC ===", flush=True)
    print(cmp_df.to_string(index=False), flush=True)
    table = pd.DataFrame({f"{L}M": {g: drift[g][L] for g in GROUPS} for L in LANDMARKS})
    table.insert(0, "group", table.index)
    table.to_csv(OUT / "tables" / "ebm_axes_perm_drift.csv", index=False)
    print("\n=== EBM(轴) group permutation importance ΔAUC ===", flush=True)
    print(table.to_string(index=False), flush=True)
    print("\n=== EBM(轴) native top-3 单特征 / 月 ===", flush=True)
    for L in LANDMARKS:
        print(f"  {L}M: " + "  |  ".join(f"{n} {s:.3f}" for n, s in native[f'{L}M']), flush=True)
    (OUT / "tables" / "ebm_axes_summary.json").write_text(json.dumps(
        {"auc": auc_cmp, "perm_drift": {g: drift[g] for g in GROUPS}, "native_top3": native},
        indent=2, ensure_ascii=False))
    print(f"\nSaved → {OUT}", flush=True)


if __name__ == "__main__":
    main()
