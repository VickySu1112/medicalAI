#!/usr/bin/env python
"""M2 · v2 — GREAT score (Vos 2016) as clinical baseline head-to-head with EBM.

GREAT score (Graves' Recurrent Events After Therapy; Vos et al. 2016, JCEM)
was developed for ATD-cessation recurrence, NOT RAI; we include it as a
**reviewer-friendly clinical baseline** with that caveat clearly noted.

GREAT (categorical, 0–3 points; one for each of):
  * TRAb ≥ 6 IU/L
  * FT4 ≥ 40 pmol/L
  * Goiter ≥ 40 g (we use ThyroidW; the original uses clinical grade ≥ 2)
Total 0–3 → risk categories I/II/III/IV.

Continuous GREAT (logistic): logit(p) = a + b1·Age + b2·FT4 + b3·TRAb + b4·ThyroidW
(re-fit on dev OOF; original Vos coefficients were ATD-cessation specific so
direct transfer would be miscalibrated — refit honest).

Head-to-head (temporal AUC + calibration):
  GREAT-3 (categorical, original aggregation)
  GREAT-cont (refit logistic, our own coefficients)
  M1 v6 logistic (the 6-feature LASSO baseline currently in M1)
  M2 EBM @ 6M (best M2 model)

All four predict 24M NHRH on the same 201 temporal episodes. Episode-cluster
bootstrap CI for each. Pairwise ΔAUC vs EBM with CI.

Output: results/module2_v2_vertical/m2v2_great_baseline/
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
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from scripts.simple.module2_v2_shared import (
    BLOCK_A_BURDEN,
    M1_FROZEN,
    load_stacked,
    PY_SEED,
)
from scripts.simple.module2_v2_b4_trab_patch import add_current_velocity
from scripts.simple.module2_v2_b4_ebm_axes import FEATS, build_feats_at_L

OUT = ROOT / "results" / "module2_v2_vertical" / "m2v2_great_baseline"

# GREAT thresholds (Vos 2016)
GREAT_TRAB_THR = 6.0     # IU/L
GREAT_FT4_THR = 40.0     # pmol/L
GREAT_GOITER_G = 40.0    # g — proxy for clinical grade ≥ 2


def _boot_ci(y, p, n=1000, seed=7):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y)); aucs = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) > 1:
            aucs.append(roc_auc_score(y[s], p[s]))
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))) if aucs else (np.nan, np.nan)


def _paired_delta_ci(y, p_a, p_b, n=1000, seed=7):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y)); ds = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) > 1:
            ds.append(roc_auc_score(y[s], p_a[s]) - roc_auc_score(y[s], p_b[s]))
    return (float(np.mean(ds)), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))) if ds else (np.nan, np.nan, np.nan)


def _calib(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); z = np.log(p / (1 - p)).reshape(-1, 1)
    l = LogisticRegression(solver="lbfgs", max_iter=2000).fit(z, y)
    return float(l.intercept_[0]), float(l.coef_[0][0])


def main() -> None:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)

    # ---- Load M1 frozen for GREAT (0M data) ----
    m1 = pd.read_csv(M1_FROZEN, low_memory=False)
    if "Age" not in m1.columns:
        # try Chinese
        for c in m1.columns:
            if "年龄" in c or "age" in c.lower():
                m1 = m1.rename(columns={c: "Age"}); break
    if "Age" not in m1.columns:
        # 1003.xlsx has 年龄 — fallback: read directly from xlsx
        print("  [info] Age not in M1 frozen; reading 年龄 from 1003.xlsx", flush=True)
        xl = pd.read_excel(ROOT / "1003.xlsx", sheet_name="Sheet1", header=[0, 1], nrows=1003)
        age_col = [c for c in xl.columns if c[0] == "年龄"][0]
        m1["Age"] = pd.to_numeric(xl[age_col], errors="coerce").values
    print(f"  M1 frozen: {len(m1)} rows; Age coverage = {m1['Age'].notna().mean()*100:.1f}%", flush=True)

    is_dev = (m1["Split"] == "Development").values
    is_test = ~is_dev
    y = m1["Y"].values
    age = pd.to_numeric(m1["Age"], errors="coerce").fillna(m1.loc[is_dev, "Age"].median())
    ft4 = pd.to_numeric(m1["FT4_0M"], errors="coerce").fillna(m1.loc[is_dev, "FT4_0M"].median())
    trab = pd.to_numeric(m1["TRAb"], errors="coerce").fillna(m1.loc[is_dev, "TRAb"].median())
    goiter = pd.to_numeric(m1["ThyroidW"], errors="coerce").fillna(m1.loc[is_dev, "ThyroidW"].median())

    # ---- GREAT-3 categorical ----
    great3 = ((trab >= GREAT_TRAB_THR).astype(int)
              + (ft4 >= GREAT_FT4_THR).astype(int)
              + (goiter >= GREAT_GOITER_G).astype(int))

    # ---- GREAT continuous (refit on dev) ----
    X_great = np.column_stack([age, ft4, trab, goiter])
    lr = LogisticRegression(solver="lbfgs", max_iter=5000, random_state=PY_SEED)
    lr.fit(X_great[is_dev], y[is_dev])
    great_cont = lr.predict_proba(X_great)[:, 1]
    print(f"  GREAT-cont refit coefs (Age,FT4,TRAb,ThyroidW): {lr.coef_[0].round(3).tolist()}", flush=True)

    # ---- M1 v6 LR baseline (re-fit on the 6 LASSO features) ----
    v6 = ["Sex", "ThyroidW", "TPOAb", "FT4_0M", "TSH_0M", "log1p_DiseaseDuration_Months_Aug"]
    Xv6 = m1[v6].apply(pd.to_numeric, errors="coerce").fillna(0).values
    lr_v6 = LogisticRegression(solver="lbfgs", max_iter=5000, C=1.0, random_state=PY_SEED)
    lr_v6.fit(Xv6[is_dev], y[is_dev])
    p_m1v6 = lr_v6.predict_proba(Xv6)[:, 1]

    # ---- M2 EBM @ 6M (re-fit, predict temporal) — collapse to episode-level via 6M row ----
    sd = load_stacked(); sd = add_current_velocity(sd, "FT3")
    rows = sd.rows
    rows_lm = rows["landmark"].values
    rows_ep = rows["episode_id"].values
    rows_dev = (rows["Split"] == "Development").values
    devL = rows_dev & (rows_lm == 6); tstL = (~rows_dev) & (rows_lm == 6)
    feat = build_feats_at_L(rows, devL)
    live = [c for c in FEATS if feat.loc[devL, c].std() > 1e-9]
    ebm = ExplainableBoostingClassifier(random_state=PY_SEED, interactions=5, max_interaction_bins=16).fit(
        feat.loc[devL, live], rows["Y_24M_NHRH"].values[devL])
    p_ebm_dev = ebm.predict_proba(feat.loc[devL, live])[:, 1]
    p_ebm_tst = ebm.predict_proba(feat.loc[tstL, live])[:, 1]
    # map back to episode_id order matching m1
    ep_to_p = {}
    for i, ix in enumerate(np.where(devL)[0]):
        ep_to_p[int(rows_ep[ix])] = p_ebm_dev[i]
    for i, ix in enumerate(np.where(tstL)[0]):
        ep_to_p[int(rows_ep[ix])] = p_ebm_tst[i]
    p_ebm = np.array([ep_to_p[int(e)] for e in m1["Episode_Index"]])

    # ---- Head-to-head on TEMPORAL (201) ----
    y_t = y[is_test]
    methods = {
        "GREAT-3 (Vos thresholds)": great3.values[is_test].astype(float),
        "GREAT-cont (refit)": great_cont[is_test],
        "M1 v6 LR (6 features)": p_m1v6[is_test],
        "M2 EBM @ 6M (best)": p_ebm[is_test],
    }
    rows_perf = []
    for name, p in methods.items():
        auc = roc_auc_score(y_t, p); lo, hi = _boot_ci(y_t, p)
        ic, sl = _calib(y_t, p) if len(np.unique(p)) >= 3 else (np.nan, np.nan)
        rows_perf.append({"method": name, "ROC_AUC": round(auc, 4),
                          "CI_low": round(lo, 4), "CI_high": round(hi, 4),
                          "calib_intercept": round(ic, 3), "calib_slope": round(sl, 3)})
    perf = pd.DataFrame(rows_perf)
    perf.to_csv(OUT / "tables" / "head_to_head_perf.csv", index=False)
    print("\n=== Head-to-head (temporal, n=201) ===")
    print(perf.to_string(index=False), flush=True)

    # Paired Δ vs EBM
    p_ebm_t = methods["M2 EBM @ 6M (best)"]
    delta_rows = []
    for name, p in methods.items():
        if name == "M2 EBM @ 6M (best)":
            continue
        d, lo, hi = _paired_delta_ci(y_t, p_ebm_t, p)
        delta_rows.append({"method_vs": name, "delta_AUC_vs_EBM_6M": round(d, 4),
                           "CI_low": round(lo, 4), "CI_high": round(hi, 4),
                           "EBM_wins": bool(lo > 0)})
    delta = pd.DataFrame(delta_rows)
    delta.to_csv(OUT / "tables" / "paired_delta_vs_EBM.csv", index=False)
    print("\n=== Paired ΔAUC vs EBM 6M (positive = EBM better) ===")
    print(delta.to_string(index=False), flush=True)

    # ROC overlay
    fig, a = plt.subplots(figsize=(5.2, 4.6))
    colors = {"GREAT-3 (Vos thresholds)": "#888", "GREAT-cont (refit)": "#a23b3b",
              "M1 v6 LR (6 features)": "#2a6f97", "M2 EBM @ 6M (best)": "#cc4444"}
    for name, p in methods.items():
        fpr, tpr, _ = roc_curve(y_t, p)
        auc = perf[perf["method"] == name]["ROC_AUC"].iloc[0]
        a.plot(fpr, tpr, color=colors[name], lw=2 if "EBM" in name else 1.2,
               label=f"{name}  AUC {auc:.3f}")
    a.plot([0, 1], [0, 1], ":", color="#999")
    a.set_xlabel("1−specificity"); a.set_ylabel("sensitivity")
    a.set_title("Head-to-head: EBM vs GREAT / M1 v6 (temporal n=201)", fontsize=10)
    a.legend(fontsize=7, loc="lower right")
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "F_ROC_overlay.png", dpi=160); plt.close(fig)

    # GREAT-3 risk per category (event rate)
    g3 = great3.values[is_test]
    cat_rows = []
    for cat in (0, 1, 2, 3):
        m = (g3 == cat)
        if m.sum():
            cat_rows.append({"GREAT_score": cat, "N": int(m.sum()),
                             "events": int(y_t[m].sum()),
                             "event_rate": round(float(y_t[m].mean()), 3)})
    cat_df = pd.DataFrame(cat_rows)
    cat_df.to_csv(OUT / "tables" / "great3_event_rate_per_category.csv", index=False)
    print("\n=== GREAT-3 event rate per category (temporal) ===")
    print(cat_df.to_string(index=False), flush=True)
    fig, a = plt.subplots(figsize=(5, 3.6))
    a.bar(cat_df["GREAT_score"].astype(str), cat_df["event_rate"],
          color=["#6aa84f", "#e69138", "#cc6600", "#cc0000"][:len(cat_df)])
    for i, r in cat_df.iterrows():
        a.text(i, r["event_rate"] + 0.01, f"{r['event_rate']:.2f}\n(n={r['N']})",
               ha="center", fontsize=8)
    a.set_xlabel("GREAT score (0–3)"); a.set_ylabel("temporal NHRH event rate")
    a.set_title("GREAT-3 event rate per category (n=201 temporal)", fontsize=10)
    a.set_ylim(0, 1)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "figures" / "F_GREAT3_event_rate.png", dpi=160); plt.close(fig)

    summary = {
        "caveats": [
            "GREAT was developed for ATD-cessation recurrence (Vos 2016 JCEM), NOT RAI; included here as clinical baseline.",
            "GREAT-cont coefficients refit on this RAI cohort's dev set (original Vos coefs would be miscalibrated)."
        ],
        "GREAT_thresholds": {"TRAb": GREAT_TRAB_THR, "FT4": GREAT_FT4_THR, "ThyroidW_g": GREAT_GOITER_G},
        "head_to_head": perf.to_dict(orient="records"),
        "paired_delta_vs_EBM": delta.to_dict(orient="records"),
        "great3_event_rate": cat_df.to_dict(orient="records"),
    }
    (OUT / "tables" / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved → {OUT}", flush=True)


if __name__ == "__main__":
    main()
