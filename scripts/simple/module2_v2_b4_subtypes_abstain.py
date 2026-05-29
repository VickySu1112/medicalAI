#!/usr/bin/env python
"""M2 · B4 — per-month importance shift, error-subtype drift, selective prediction.

Q1: Are the error-prone subtypes (hidden stratification, Oakden-Rayner 2020) the
    same across months? -> per-landmark feature-importance shift + the per-month
    hard-case profile (computed in module2_v2_b4_trab_patch).
Q2: Selective prediction / reject option — abstain on the uncertain cases (defer
    to a human expert) and trace abstention-rate vs accuracy/NPV on the retained
    (the risk-coverage curve).

Model for Q2 = L2 on ABCDE + FT3 + Eval (best available feature set), per-landmark
Platt-calibrated. Abstention rule is label-free (distance from the decision
threshold), so applying it to the temporal split does not peek at labels.
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.simple.module2_v2_shared import (
    BLOCKS,
    LANDMARKS,
    PRETTY,
    apply_per_landmark_platt,
    load_stacked,
    per_landmark_platt_on_pooled_oof,
    PY_SEED,
)
from scripts.simple.module2_v2_b4_trab_patch import add_current_velocity, load_eval_cols, ABCDE

OUT = ROOT / "results" / "module2_v2_vertical" / "b4_target_gru"


def _lr():
    return Pipeline([("s", StandardScaler()),
                     ("lr", LogisticRegression(penalty="l2", solver="lbfgs", C=1.0,
                                               max_iter=5000, random_state=PY_SEED))])


# ---------------------------------------------------------------------------
# Q1: per-landmark feature-importance shift
# ---------------------------------------------------------------------------


def per_landmark_importance(sd):
    rows = sd.rows
    y = rows["Y_24M_NHRH"].values
    lm = rows["landmark"].values
    is_dev = (rows["Split"] == "Development").values
    feats = BLOCKS["A"] + BLOCKS["B"] + BLOCKS["C"] + BLOCKS["D"]  # per-landmark: drop E (constant within L)
    X = rows[feats].values
    out = {}
    for L in LANDMARKS:
        m = is_dev & (lm == L)
        pipe = _lr()
        pipe.fit(X[m], y[m])
        coef = pipe.named_steps["lr"].coef_[0]
        order = np.argsort(-np.abs(coef))
        out[L] = [(feats[i], round(float(coef[i]), 3)) for i in order[:6]]
    return feats, out


# ---------------------------------------------------------------------------
# Q2: best-model predictions + selective prediction (risk-coverage)
# ---------------------------------------------------------------------------


def oof_temporal_preds(X, y, ep, is_dev):
    """Return full-length prediction vector: OOF on dev, final-fit on temporal."""
    pred = np.zeros(len(y))
    Xd, yd, epd = X[is_dev], y[is_dev], ep[is_dev]
    dev_idx = np.where(is_dev)[0]
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=13)
    for tr, va in skf.split(Xd, yd, groups=epd):
        p = _lr(); p.fit(Xd[tr], yd[tr]); pred[dev_idx[va]] = p.predict_proba(Xd[va])[:, 1]
    final = _lr(); final.fit(Xd, yd)
    pred[~is_dev] = final.predict_proba(X[~is_dev])[:, 1]
    return pred


def _metrics_at(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    tn = ((pred == 0) & (y == 0)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    n = len(y)
    acc = (tp + tn) / n if n else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    roc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    return {"n": int(n), "acc": acc, "NPV": npv, "PPV": ppv, "ROC": roc}


def selective_prediction(sd, p_cal):
    rows = sd.rows
    y = rows["Y_24M_NHRH"].values
    is_dev = (rows["Split"] == "Development").values
    # Youden threshold on dev OOF
    yd, pd_ = y[is_dev], p_cal[is_dev]
    ths = np.linspace(0.05, 0.95, 181)
    jbest = max(ths, key=lambda t: (_metrics_at(yd, pd_, t)["acc"]))  # accuracy-optimal thr
    # also Youden for sens+spec
    def youden(t):
        m = _metrics_at(yd, pd_, t)
        pred = (pd_ >= t).astype(int)
        tp = ((pred == 1) & (yd == 1)).sum(); fn = ((pred == 0) & (yd == 1)).sum()
        tn = ((pred == 0) & (yd == 0)).sum(); fp = ((pred == 1) & (yd == 0)).sum()
        sens = tp / (tp + fn) if (tp + fn) else 0; spec = tn / (tn + fp) if (tn + fp) else 0
        return sens + spec - 1
    thr = float(max(ths, key=youden))

    # Risk-coverage on TEMPORAL (abstain on least-confident = closest to thr; label-free)
    is_test = ~is_dev
    yt, pt = y[is_test], p_cal[is_test]
    conf = np.abs(pt - thr)
    order = np.argsort(-conf)  # most confident first
    rows_rc = []
    for abst in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        keep = order[: int(round((1 - abst) * len(yt)))]
        m = _metrics_at(yt[keep], pt[keep], thr)
        rows_rc.append({"abstain_pct": int(abst * 100), "coverage_pct": int((1 - abst) * 100),
                        "n_retained": m["n"], "accuracy": round(m["acc"], 4),
                        "NPV": round(m["NPV"], 4), "PPV": round(m["PPV"], 4),
                        "ROC_retained": round(m["ROC"], 4)})
    return thr, pd.DataFrame(rows_rc)


def main() -> None:
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    sd = load_stacked()
    sd = add_current_velocity(sd, "FT3")

    print("=== (Q1) Per-landmark feature-importance shift (std L2 coef, top-6) ===", flush=True)
    feats, imp = per_landmark_importance(sd)
    imp_records = []
    for L in LANDMARKS:
        print(f"\n  {L}M:")
        for rank, (f, c) in enumerate(imp[L], 1):
            print(f"    {rank}. {PRETTY.get(f, f):<32} {c:+.3f}", flush=True)
            imp_records.append({"landmark": f"{L}M", "rank": rank, "feature": PRETTY.get(f, f), "coef": c})
    pd.DataFrame(imp_records).to_csv(OUT / "tables" / "per_landmark_importance.csv", index=False)

    print("\n=== (Q2) Selective prediction / abstention (model = ABCDE+FT3+Eval) ===", flush=True)
    rows = sd.rows
    y = rows["Y_24M_NHRH"].values
    ep = rows["episode_id"].values
    is_dev = (rows["Split"] == "Development").values
    X = np.column_stack([sd.feature_matrix(ABCDE).values,
                         rows["FT3_current"].values, rows["FT3_velocity"].values,
                         load_eval_cols(sd)])
    raw = oof_temporal_preds(X, y, ep, is_dev)
    p_cal = apply_per_landmark_platt(sd, raw, per_landmark_platt_on_pooled_oof(sd, raw))
    thr, rc = selective_prediction(sd, p_cal)
    rc.to_csv(OUT / "tables" / "selective_prediction_riskcoverage.csv", index=False)
    print(f"  decision threshold (Youden on dev OOF) = {thr:.3f}", flush=True)
    print(rc.to_string(index=False), flush=True)

    # Figure: abstention% vs accuracy & NPV (retained, temporal)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(rc["abstain_pct"], rc["accuracy"], marker="o", label="Accuracy (retained)", color="#1d4e89")
    ax.plot(rc["abstain_pct"], rc["NPV"], marker="s", label="NPV (retained)", color="#2a7f5f")
    ax.set_xlabel("abstention rate % (deferred to clinician)")
    ax.set_ylabel("temporal performance on retained cases")
    ax.set_title("Selective prediction: abstain on the uncertain → retained accuracy/NPV ↑")
    ax.legend(fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "figures" / "Figure_06_RiskCoverage.png", dpi=160); plt.close(fig)

    base = rc.iloc[0]; a30 = rc[rc["abstain_pct"] == 30].iloc[0]
    (OUT / "tables" / "subtypes_abstain_summary.json").write_text(json.dumps({
        "per_landmark_importance": imp_records,
        "selective_threshold": thr,
        "risk_coverage": rc.to_dict(orient="records"),
        "headline": {"no_abstain": {"acc": base["accuracy"], "NPV": base["NPV"]},
                     "abstain_30pct": {"acc": a30["accuracy"], "NPV": a30["NPV"]}},
    }, indent=2, ensure_ascii=False))
    print(f"\nNo-abstain acc {base['accuracy']} / NPV {base['NPV']}  →  "
          f"abstain 30%: acc {a30['accuracy']} / NPV {a30['NPV']}", flush=True)
    print(f"Saved → {OUT}", flush=True)


if __name__ == "__main__":
    main()
