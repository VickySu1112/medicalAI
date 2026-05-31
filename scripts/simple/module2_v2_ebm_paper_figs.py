#!/usr/bin/env python
"""M2 · v2 — EBM paper figures (glassbox interpretability).

Per-landmark EBM (Explainable Boosting Machine) on the aggregated orthogonal
hormone axes. Produces publication figures:
  E1  per-landmark term importance (top terms) — the importance drift.
  E2  EBM shape functions for the per-landmark lead feature (the nonlinear
      risk relationships — the glassbox signature; esp. the 6M velocity shape).
  E3  per-landmark calibration (reliability) — EBM native probabilities.
  E4  per-landmark ROC with episode-bootstrap CI.
Plus a summary JSON (AUC/CI/calibration/top-terms per landmark).

Output: results/module2_v2_vertical/m2v2_ebm_paper/.
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
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression

from scripts.simple.module2_v2_shared import LANDMARKS, load_stacked, PY_SEED
from scripts.simple.module2_v2_b4_trab_patch import add_current_velocity
from scripts.simple.module2_v2_b4_ebm_axes import build_feats_at_L, FEATS

OUT = ROOT / "results" / "module2_v2_vertical" / "m2v2_ebm_paper"
DISP = {
    "ThyroidW": "Thyroid weight", "TRAb": "TRAb", "TGAb": "TGAb", "TPOAb": "TPOAb",
    "Sex": "Sex", "FT4_0M": "FT4 (0M)", "TSH_0M": "TSH (0M)",
    "log1p_DiseaseDuration_Months_Aug": "Disease duration (log,mo)",
    "Uptake24h": "24h uptake", "HalfLife": "Iodine half-life",
    "TSH_current": "TSH (current)", "TSH_velocity": "TSH velocity",
    "Hormone_load": "FT3,FT4 level", "T3T4_balance": "FT3,FT4 gap",
    "Velocity_load": "FT3,FT4 velocity", "Velocity_balance": "FT3,FT4 vel gap",
}
LEAD = {0: "ThyroidW", 1: "Hormone_load", 3: "Hormone_load", 6: "Velocity_load"}


def _disp(n):
    return " × ".join(DISP.get(p, p) for p in n.split(" & ")) if " & " in n else DISP.get(n, n)


def _boot_auc_ci(y, p, n=1000, seed=7):
    rng = np.random.default_rng(seed)
    aucs = []
    idx = np.arange(len(y))
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) > 1:
            aucs.append(roc_auc_score(y[s], p[s]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def _calib(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs", max_iter=2000).fit(z, y)
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


def main() -> None:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    sd = load_stacked(); sd = add_current_velocity(sd, "FT3")
    rows = sd.rows
    y = rows["Y_24M_NHRH"].values; lm = rows["landmark"].values
    is_dev = (rows["Split"] == "Development").values

    models, preds, summ = {}, {}, {}
    fig1, ax1 = plt.subplots(2, 2, figsize=(11, 7))
    fig2, ax2 = plt.subplots(2, 2, figsize=(11, 7))
    fig3, ax3 = plt.subplots(2, 2, figsize=(10, 7))
    for k, L in enumerate(LANDMARKS):
        devL = is_dev & (lm == L); tstL = (~is_dev) & (lm == L)
        feat = build_feats_at_L(rows, devL)
        live = [c for c in FEATS if feat.loc[devL, c].std() > 1e-9]
        Xtr, ytr = feat.loc[devL, live], y[devL]
        Xte, yte = feat.loc[tstL, live], y[tstL]
        ebm = ExplainableBoostingClassifier(random_state=PY_SEED, interactions=5, max_interaction_bins=16).fit(Xtr, ytr)
        models[L] = ebm
        p = ebm.predict_proba(Xte)[:, 1]; preds[L] = (yte, p)
        auc = roc_auc_score(yte, p); lo, hi = _boot_auc_ci(yte, p)
        ic, sl = _calib(yte, p); br = brier_score_loss(yte, p)
        g = ebm.explain_global(); names, scores = g.data()["names"], g.data()["scores"]
        top = sorted(zip(names, scores), key=lambda t: -t[1])[:6]
        summ[f"{L}M"] = {"AUC": round(auc, 4), "AUC_CI": [round(lo, 4), round(hi, 4)],
                          "calib_intercept": round(ic, 3), "calib_slope": round(sl, 3),
                          "Brier": round(br, 4),
                          "top6": [{"term": _disp(n), "importance": round(float(s), 4)} for n, s in top]}

        # E1 importance barh
        a = ax1[k // 2][k % 2]
        labs = [_disp(n) for n, _ in top][::-1]; vals = [s for _, s in top][::-1]
        a.barh(range(len(vals)), vals, color="#2a6f97")
        a.set_yticks(range(len(vals))); a.set_yticklabels(labs, fontsize=8)
        a.set_title(f"{L}M  (EBM, temporal AUC {auc:.3f})", fontsize=10)
        a.set_xlabel("EBM importance (mean |contribution|)", fontsize=8)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)

        # E2 shape function of the lead feature
        feat_lead = LEAD[L] if LEAD[L] in ebm.term_names_ else ebm.term_names_[0]
        idx = ebm.term_names_.index(feat_lead)
        dd = g.data(idx)
        b = ax2[k // 2][k % 2]
        xs = np.array(dd["names"], dtype=float); ys = np.array(dd["scores"], dtype=float)
        if len(xs) == len(ys) + 1:  # continuous: bin edges
            mid = (xs[:-1] + xs[1:]) / 2
            b.step(mid, ys, where="mid", color="#a23b3b", lw=2)
            b.fill_between(mid, ys, step="mid", alpha=0.12, color="#a23b3b")
        else:
            b.plot(xs[:len(ys)], ys, "o-", color="#a23b3b")
        b.axhline(0, color="#999", lw=0.8, ls=":")
        b.set_title(f"{L}M lead: {_disp(feat_lead)}", fontsize=10)
        b.set_xlabel("feature value (standardized axis)", fontsize=8)
        b.set_ylabel("contribution to log-odds", fontsize=8)
        for s in ("top", "right"):
            b.spines[s].set_visible(False)

        # E3 calibration reliability
        c = ax3[k // 2][k % 2]
        bins = np.quantile(p, np.linspace(0, 1, 6))
        bins[0], bins[-1] = -0.001, 1.001
        bi = np.digitize(p, bins) - 1
        xs_c, ys_c = [], []
        for j in range(5):
            m = bi == j
            if m.sum() >= 5:
                xs_c.append(p[m].mean()); ys_c.append(yte[m].mean())
        c.plot([0, 1], [0, 1], ":", color="#999")
        c.plot(xs_c, ys_c, "o-", color="#2a7f5f")
        c.set_title(f"{L}M calib  slope {sl:.2f}, intc {ic:.2f}", fontsize=10)
        c.set_xlabel("predicted", fontsize=8); c.set_ylabel("observed", fontsize=8)
        c.set_xlim(0, 1); c.set_ylim(0, 1)
        for s in ("top", "right"):
            c.spines[s].set_visible(False)

    fig1.suptitle("EBM term importance by landmark — drift: goiter → hormone level → hormone velocity", fontsize=12)
    fig1.tight_layout(rect=[0, 0, 1, 0.96]); fig1.savefig(OUT / "figures" / "Figure_E1_Importance.png", dpi=160); plt.close(fig1)
    fig2.suptitle("EBM shape functions (glassbox) — nonlinear risk relationships per landmark", fontsize=12)
    fig2.tight_layout(rect=[0, 0, 1, 0.96]); fig2.savefig(OUT / "figures" / "Figure_E2_Shapes.png", dpi=160); plt.close(fig2)
    fig3.suptitle("EBM calibration (temporal, native probabilities)", fontsize=12)
    fig3.tight_layout(rect=[0, 0, 1, 0.96]); fig3.savefig(OUT / "figures" / "Figure_E3_Calibration.png", dpi=160); plt.close(fig3)

    # E4 ROC
    fig4, ax4 = plt.subplots(figsize=(5.5, 5))
    for L in LANDMARKS:
        yy, pp = preds[L]
        fpr, tpr, _ = roc_curve(yy, pp)
        ax4.plot(fpr, tpr, lw=2, label=f"{L}M (AUC {summ[f'{L}M']['AUC']:.3f})")
    ax4.plot([0, 1], [0, 1], ":", color="#999")
    ax4.set_xlabel("1 − specificity"); ax4.set_ylabel("sensitivity")
    ax4.set_title("EBM per-landmark ROC (temporal)"); ax4.legend(fontsize=8, loc="lower right")
    for s in ("top", "right"):
        ax4.spines[s].set_visible(False)
    fig4.tight_layout(); fig4.savefig(OUT / "figures" / "Figure_E4_ROC.png", dpi=160); plt.close(fig4)

    (OUT / "tables" / "ebm_paper_summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False))
    print("EBM paper figures saved. Per-landmark summary:")
    for L in LANDMARKS:
        s = summ[f"{L}M"]
        print(f"  {L}M: AUC {s['AUC']} CI{s['AUC_CI']} | slope {s['calib_slope']} intc {s['calib_intercept']} | "
              f"top: {s['top6'][0]['term']} {s['top6'][0]['importance']}", flush=True)
    print(f"Saved → {OUT}", flush=True)


if __name__ == "__main__":
    main()
