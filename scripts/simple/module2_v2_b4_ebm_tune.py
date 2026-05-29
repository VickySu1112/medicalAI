#!/usr/bin/env python
"""M2 · B4 — EBM hyperparameter tuning (per landmark, OOF-selected).

Tries to push EBM past the LR ceiling by sweeping interactions / learning rate /
tree size / bagging. OOF (3-fold within dev@L) selects the config; temporal is
read out only. Aggregated orthogonal axes (consistent naming, no independent
FT3/FT4).
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
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.simple.module2_v2_shared import LANDMARKS, load_stacked, PY_SEED
from scripts.simple.module2_v2_b4_ebm_axes import build_feats_at_L, FEATS

OUT = ROOT / "results" / "module2_v2_vertical" / "b4_target_gru"

CONFIGS = {
    "default": dict(interactions=5, learning_rate=0.01),
    "no_inter": dict(interactions=0, learning_rate=0.02),
    "more_inter": dict(interactions=10, learning_rate=0.02),
    "reg_slow": dict(interactions=5, learning_rate=0.005, outer_bags=16, min_samples_leaf=4),
    "shallow": dict(interactions=3, learning_rate=0.05, max_leaves=2),
    "coarse_bins": dict(interactions=5, learning_rate=0.02, max_bins=64, min_samples_leaf=8),
}


def _roc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")


def _oof_auc_ebm(X, y, kw, seed=13):
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y):
        m = ExplainableBoostingClassifier(random_state=PY_SEED, **kw).fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return _roc(y, oof)


def main() -> None:
    sd = load_stacked()
    from scripts.simple.module2_v2_b4_trab_patch import add_current_velocity
    sd = add_current_velocity(sd, "FT3")
    rows = sd.rows
    y = rows["Y_24M_NHRH"].values; lm = rows["landmark"].values
    is_dev = (rows["Split"] == "Development").values

    out = []
    for L in LANDMARKS:
        devL = is_dev & (lm == L); tstL = (~is_dev) & (lm == L)
        feat = build_feats_at_L(rows, devL)
        live = [c for c in FEATS if feat.loc[devL, c].std() > 1e-9]
        Xtr, ytr = feat.loc[devL, live], y[devL]
        Xte, yte = feat.loc[tstL, live], y[tstL]
        # LR ref
        lr = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=5000, random_state=PY_SEED))]).fit(Xtr.values, ytr)
        lr_auc = _roc(yte, lr.predict_proba(Xte.values)[:, 1])
        # sweep
        best, best_oof = None, -1
        oof_scores = {}
        for name, kw in CONFIGS.items():
            try:
                o = _oof_auc_ebm(Xtr, ytr, kw)
            except Exception as e:
                o = float("nan")
            oof_scores[name] = round(o, 4)
            if np.isfinite(o) and o > best_oof:
                best_oof, best = o, name
        # temporal for default + best
        def tmp_auc(kw):
            m = ExplainableBoostingClassifier(random_state=PY_SEED, **kw).fit(Xtr, ytr)
            return _roc(yte, m.predict_proba(Xte)[:, 1])
        ebm_def = tmp_auc(CONFIGS["default"])
        ebm_best = tmp_auc(CONFIGS[best])
        out.append({"landmark": f"{L}M", "LR": round(lr_auc, 4), "EBM_default": round(ebm_def, 4),
                    "EBM_best_cfg": best, "EBM_best_OOF": round(best_oof, 4), "EBM_best_temporal": round(ebm_best, 4),
                    "oof_by_cfg": oof_scores})
    df = pd.DataFrame(out)
    print(df[["landmark", "LR", "EBM_default", "EBM_best_cfg", "EBM_best_OOF", "EBM_best_temporal"]].to_string(index=False), flush=True)
    print("\nOOF by config:")
    for r in out:
        print(f"  {r['landmark']}: {r['oof_by_cfg']}", flush=True)
    (OUT / "tables" / "ebm_tuning.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nSaved → {OUT / 'tables' / 'ebm_tuning.json'}", flush=True)


if __name__ == "__main__":
    main()
