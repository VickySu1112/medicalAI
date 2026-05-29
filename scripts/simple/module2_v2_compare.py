#!/usr/bin/env python
"""M2 · v2 — EBM vs other ML methods, comprehensive comparison figures.

12 methods (incl. naive baseline) fit per landmark on the same aggregated-axis
features; temporal evaluated on ROC-AUC (+ bootstrap CI), PR-AUC, Brier,
calibration slope, fit time. Many comparison figures across months × metrics ×
granularities (heatmaps / grouped bars / ROC overlays / calibration overlays /
DCA / ranking / ΔAUC-vs-EBM / cost-benefit).

Output: results/module2_v2_vertical/m2v2_compare/.
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import sys
import time
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
for _fp in ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "/Library/Fonts/Arial Unicode.ttf"):
    if os.path.exists(_fp):
        _fm.fontManager.addfont(_fp); plt.rcParams["font.family"] = "Arial Unicode MS"; break
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from scripts.simple.module2_v2_shared import LANDMARKS, load_stacked, PY_SEED
from scripts.simple.module2_v2_b4_trab_patch import add_current_velocity
from scripts.simple.module2_v2_b4_ebm_axes import build_feats_at_L, FEATS

OUT = ROOT / "results" / "module2_v2_vertical" / "m2v2_compare"
FIGD = OUT / "figures"
manifest = []
_n = [0]


def save(fig, slug, cap):
    _n[0] += 1; name = f"C{_n[0]:02d}_{slug}.png"
    fig.savefig(FIGD / name, dpi=150, bbox_inches="tight"); plt.close(fig)
    manifest.append({"n": _n[0], "file": name, "caption": cap})


def methods():
    return {
        "Naive(prevalence)": DummyClassifier(strategy="prior"),
        "L2-Logistic": Pipeline([("s", StandardScaler()), ("m", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=5000, random_state=PY_SEED))]),
        "Elastic-net": Pipeline([("s", StandardScaler()), ("m", LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=1.0, solver="saga", max_iter=5000, random_state=PY_SEED))]),
        "GaussianNB": Pipeline([("s", StandardScaler()), ("m", GaussianNB())]),
        "kNN(15)": Pipeline([("s", StandardScaler()), ("m", KNeighborsClassifier(15))]),
        "SVM-RBF": Pipeline([("s", StandardScaler()), ("m", SVC(probability=True, random_state=PY_SEED))]),
        "RandomForest": RandomForestClassifier(n_estimators=400, random_state=PY_SEED, n_jobs=1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=400, random_state=PY_SEED, n_jobs=1),
        "HistGBM": HistGradientBoostingClassifier(random_state=PY_SEED),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, eval_metric="logloss", random_state=PY_SEED, n_jobs=1),
        "LightGBM": LGBMClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=PY_SEED, n_jobs=1, verbose=-1),
        "EBM": ExplainableBoostingClassifier(random_state=PY_SEED, interactions=5),
    }
REPRESENT = ["EBM", "L2-Logistic", "RandomForest", "HistGBM", "XGBoost"]


def _boot(y, p, n=1000, seed=7):
    rng = np.random.default_rng(seed); idx = np.arange(len(y)); a = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) > 1:
            a.append(roc_auc_score(y[s], p[s]))
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if a else (np.nan, np.nan)


def _calib_slope(y, p):
    if len(np.unique(p)) < 3 or len(np.unique(y)) < 2:
        return np.nan
    p = np.clip(p, 1e-6, 1 - 1e-6); z = np.log(p / (1 - p)).reshape(-1, 1)
    return float(LogisticRegression(solver="lbfgs", max_iter=2000).fit(z, y).coef_[0][0])


def main() -> None:
    FIGD.mkdir(parents=True, exist_ok=True)
    sd = load_stacked(); sd = add_current_velocity(sd, "FT3")
    rows = sd.rows; y = rows["Y_24M_NHRH"].values; lm = rows["landmark"].values
    is_dev = (rows["Split"] == "Development").values
    mlist = list(methods().keys())

    R = {m: {} for m in mlist}  # R[method][L] = dict(p,auc,lo,hi,pr,brier,slope,t,y)
    table = []
    for L in LANDMARKS:
        devL = is_dev & (lm == L); tstL = (~is_dev) & (lm == L)
        feat = build_feats_at_L(rows, devL); live = [c for c in FEATS if feat.loc[devL, c].std() > 1e-9]
        Xtr, ytr = feat.loc[devL, live].values, y[devL]; Xte, yte = feat.loc[tstL, live].values, y[tstL]
        for name, est in methods().items():
            try:
                t0 = time.time(); est.fit(Xtr, ytr); ft = time.time() - t0
                p = est.predict_proba(Xte)[:, 1]
                auc = roc_auc_score(yte, p); lo, hi = _boot(yte, p)
                pr = average_precision_score(yte, p); br = brier_score_loss(yte, p); sl = _calib_slope(yte, p)
            except Exception as e:
                print(f"  !! {name}@{L}M failed: {e}", flush=True); continue
            R[name][L] = dict(p=p, y=yte, auc=auc, lo=lo, hi=hi, pr=pr, brier=br, slope=sl, t=ft)
            table.append({"method": name, "landmark": f"{L}M", "ROC_AUC": round(auc, 4),
                          "CI_lo": round(lo, 4), "CI_hi": round(hi, 4), "PR_AUC": round(pr, 4),
                          "Brier": round(br, 4), "calib_slope": round(sl, 4) if sl == sl else None,
                          "fit_s": round(ft, 3)})
        print(f"  {L}M done", flush=True)
    import pandas as pd
    pd.DataFrame(table).to_csv(OUT / "metrics_table.csv", index=False)

    # order methods by mean AUC desc
    mean_auc = {m: np.nanmean([R[m][L]["auc"] for L in LANDMARKS if L in R[m]]) for m in mlist}
    order = sorted([m for m in mlist if R[m]], key=lambda m: -mean_auc[m])

    def heat(metric, slug, cap, cmap="YlGnBu", lower_better=False):
        mat = np.array([[R[m][L].get(metric, np.nan) if L in R[m] else np.nan for L in LANDMARKS] for m in order])
        f, a = plt.subplots(figsize=(6.5, 0.42 * len(order) + 1.4))
        im = a.imshow(mat, cmap=cmap + ("_r" if lower_better else ""), aspect="auto")
        a.set_xticks(range(4)); a.set_xticklabels([f"{L}M" for L in LANDMARKS]); a.set_yticks(range(len(order))); a.set_yticklabels(order, fontsize=8)
        for i in range(len(order)):
            for j in range(4):
                if mat[i, j] == mat[i, j]:
                    a.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=7)
        a.set_title(cap, fontsize=10); f.colorbar(im, ax=a, fraction=.025)
        save(f, slug, cap)

    heat("auc", "Heatmap_ROC", "ROC-AUC: 方法 × 地标", "YlGnBu")
    heat("pr", "Heatmap_PR", "PR-AUC: 方法 × 地标", "YlGnBu")
    heat("brier", "Heatmap_Brier", "Brier(越低越好): 方法 × 地标", "YlOrRd", lower_better=True)
    heat("slope", "Heatmap_CalibSlope", "校准斜率(理想 1): 方法 × 地标", "PuBuGn")

    # per-landmark ROC-AUC bar with CI
    for L in LANDMARKS:
        ms = [m for m in order if L in R[m]]; v = [R[m][L]["auc"] for m in ms]
        err = [[R[m][L]["auc"] - R[m][L]["lo"] for m in ms], [R[m][L]["hi"] - R[m][L]["auc"] for m in ms]]
        f, a = plt.subplots(figsize=(6, 4)); yp = np.arange(len(ms))[::-1]
        cols = ["#cc0000" if m == "EBM" else ("#bbbbbb" if m == "Naive(prevalence)" else "#2a6f97") for m in ms]
        a.barh(yp, v, xerr=err, color=cols, capsize=2)
        a.set_yticks(yp); a.set_yticklabels(ms, fontsize=8); a.set_xlim(0.45, 0.9)
        a.set_title(f"{L}M ROC-AUC(95%CI),EBM 红", fontsize=10); a.set_xlabel("ROC-AUC")
        save(f, f"Bar_ROC_{L}M", f"{L}M 各方法 ROC-AUC(含95%CI)")

    # ROC overlays per landmark
    for L in LANDMARKS:
        f, a = plt.subplots(figsize=(4.6, 4.2))
        for m in order:
            if L not in R[m] or m == "Naive(prevalence)":
                continue
            fpr, tpr, _ = roc_curve(R[m][L]["y"], R[m][L]["p"])
            a.plot(fpr, tpr, lw=2 if m == "EBM" else 1, label=f"{m} {R[m][L]['auc']:.3f}", alpha=.9 if m == "EBM" else .6)
        a.plot([0, 1], [0, 1], ":", color="#999"); a.legend(fontsize=6, loc="lower right")
        a.set_title(f"{L}M ROC 叠加", fontsize=10); a.set_xlabel("1−spec"); a.set_ylabel("sens")
        save(f, f"ROCoverlay_{L}M", f"{L}M 各方法 ROC 叠加")

    # calibration overlays (representative)
    for L in LANDMARKS:
        f, a = plt.subplots(figsize=(4.6, 4.2)); a.plot([0, 1], [0, 1], ":", color="#999")
        for m in REPRESENT:
            if L not in R[m]:
                continue
            p = R[m][L]["p"]; yy = R[m][L]["y"]; bn = np.quantile(p, np.linspace(0, 1, 6)); bn[0], bn[-1] = -1e-3, 1 + 1e-3
            bi = np.digitize(p, bn) - 1; xs, ys = [], []
            for j in range(5):
                mm = bi == j
                if mm.sum() >= 4:
                    xs.append(p[mm].mean()); ys.append(yy[mm].mean())
            a.plot(xs, ys, "o-", lw=2 if m == "EBM" else 1, label=m, alpha=.9 if m == "EBM" else .6)
        a.legend(fontsize=7); a.set_title(f"{L}M 校准对比", fontsize=10); a.set_xlabel("predicted"); a.set_ylabel("observed"); a.set_xlim(0, 1); a.set_ylim(0, 1)
        save(f, f"Caliboverlay_{L}M", f"{L}M 代表方法校准对比")

    # mean ROC-AUC ranking with CI (pooled across landmarks via mean)
    f, a = plt.subplots(figsize=(6, 4.2)); ms = order
    mv = [mean_auc[m] for m in ms]; yp = np.arange(len(ms))[::-1]
    cols = ["#cc0000" if m == "EBM" else ("#bbbbbb" if m == "Naive(prevalence)" else "#2a6f97") for m in ms]
    a.barh(yp, mv, color=cols); a.set_yticks(yp); a.set_yticklabels(ms, fontsize=8); a.set_xlim(0.45, 0.85)
    for i, m in enumerate(ms):
        a.text(mv[i] + .003, yp[i], f"{mv[i]:.3f}", va="center", fontsize=7)
    a.set_title("各方法平均 ROC-AUC(4 地标均值),EBM 红", fontsize=10); a.set_xlabel("mean ROC-AUC")
    save(f, "Ranking_meanAUC", "各方法 4 地标平均 ROC-AUC 排名")

    # ΔAUC vs EBM (per landmark mean)
    f, a = plt.subplots(figsize=(6, 4.2)); ms = [m for m in order if m != "EBM"]
    d = [np.nanmean([R[m][L]["auc"] - R["EBM"][L]["auc"] for L in LANDMARKS if L in R[m] and L in R["EBM"]]) for m in ms]
    yp = np.arange(len(ms))[::-1]; a.barh(yp, d, color=["#2a7f5f" if x > 0 else "#a23b3b" for x in d])
    a.axvline(0, color="#333", lw=1); a.set_yticks(yp); a.set_yticklabels(ms, fontsize=8)
    a.set_title("ΔROC-AUC vs EBM(>0 表示优于 EBM)", fontsize=10); a.set_xlabel("mean ΔAUC vs EBM")
    save(f, "Delta_vs_EBM", "各方法相对 EBM 的平均 ΔROC-AUC")

    # DCA per landmark (representative)
    for L in LANDMARKS:
        f, a = plt.subplots(figsize=(4.6, 4)); pts = np.linspace(.05, .6, 40)
        for m in REPRESENT:
            if L not in R[m]:
                continue
            p = R[m][L]["p"]; yy = R[m][L]["y"]; nb = []
            for pt in pts:
                pr_ = (p >= pt).astype(int); tp = ((pr_ == 1) & (yy == 1)).sum(); fp = ((pr_ == 1) & (yy == 0)).sum()
                nb.append(tp / len(yy) - fp / len(yy) * pt / (1 - pt))
            a.plot(pts, nb, lw=2 if m == "EBM" else 1, label=m, alpha=.9 if m == "EBM" else .6)
        prev = R["EBM"][L]["y"].mean(); a.plot(pts, [prev - (1 - prev) * pt / (1 - pt) for pt in pts], "--", color="#888", label="treat-all")
        a.axhline(0, color="#bbb", lw=.8); a.legend(fontsize=6); a.set_title(f"{L}M 决策曲线对比", fontsize=10)
        a.set_xlabel("threshold"); a.set_ylabel("net benefit")
        save(f, f"DCA_{L}M", f"{L}M 代表方法决策曲线(DCA)对比")

    # cost vs performance
    f, a = plt.subplots(figsize=(5.5, 4))
    for m in order:
        ft = np.nanmean([R[m][L]["t"] for L in LANDMARKS if L in R[m]])
        a.scatter(ft, mean_auc[m], s=60, color="#cc0000" if m == "EBM" else "#2a6f97")
        a.annotate(m, (ft, mean_auc[m]), textcoords="offset points", xytext=(4, 3), fontsize=7)
    a.set_xscale("log"); a.set_xlabel("平均拟合耗时 (s, log)"); a.set_ylabel("mean ROC-AUC")
    a.set_title("性能 vs 计算成本(EBM 红)", fontsize=10)
    save(f, "Cost_vs_Perf", "性能 vs 拟合耗时(成本-收益)")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\n生成对比图 {_n[0]} 张 → {FIGD}", flush=True)
    print("平均 ROC-AUC 排名:", {m: round(mean_auc[m], 4) for m in order}, flush=True)


if __name__ == "__main__":
    main()
