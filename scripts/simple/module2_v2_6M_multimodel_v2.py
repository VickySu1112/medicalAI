#!/usr/bin/env python
"""M2 · v2 — 6M EBM vs LR baseline + 8 sklearn classical methods (10 total).

Extension of `module2_v2_6M_multimodel.py` — same lock conditions, different
baseline set:

  Anchor:    EBM (locked, must match atlas ≈ 0.858)
  Baseline:  L2-Logistic Regression
  Methods:
    1. RandomForest
    2. ExtraTrees
    3. GradientBoosting   (classic full-fit GBM)
    4. HistGradientBoosting
    5. AdaBoost           (SAMME, default base estimator)
    6. DecisionTree       (max_depth=5, min_samples_leaf=20)
    7. SVM-RBF
    8. LinearDiscriminantAnalysis  (shrinkage="auto", solver="lsqr")

Same 5 immovable lock conditions as v1:
1. EBM via ebm_oof_and_temporal(L=6, interactions=5, max_interaction_bins=16,
   seed=PY_SEED=2025); abort if observed AUC drifts > 0.003 from atlas 0.8582.
2. All methods share SAME X = build_feats_at_L on rows@dev6M (16 live), same y,
   same is_dev, same episode_id. Rows built ONCE via build_rows_for_method("median").
3. Same 5-fold StratifiedGroupKFold(seed=CV_SEED=13, groups=episode_id).
4. AUC reported = temporal (201 ep); dev OOF only for paired bootstrap.
5. Untouched: m2v2_ebm_full_median/, m2v2_6M_multimodel/ (v1).

Output → results/module2_v2_vertical/m2v2_6M_multimodel_v2/
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import hashlib
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
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from scripts.simple.module2_v2_shared import (
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    CV_SEED,
    PY_SEED,
)
from scripts.simple.module2_v2_impute_experiment import build_rows_for_method
from scripts.simple.module2_v2_b4_ebm_axes import FEATS, build_feats_at_L
from scripts.simple.module2_v2_b4_ebm_oof import ebm_oof_and_temporal

LANDMARK = 6
# Atlas snapshot 0.8582; current rerun ≈ 0.8562 (interpret/sklearn lib refresh).
EBM_LOCKED_AUC = 0.8582
TOL = 0.003
OUT = ROOT / "results" / "module2_v2_vertical" / "m2v2_6M_multimodel_v2"


# ---------------------------------------------------------------------------
# Method factories (each returns a fresh estimator per fold)
# ---------------------------------------------------------------------------


def make_methods():
    def lr_l2():
        return Pipeline([
            ("s", StandardScaler()),
            ("c", LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
                                     max_iter=2000, random_state=PY_SEED)),
        ])

    def rf():
        return RandomForestClassifier(
            n_estimators=500, max_depth=None,
            class_weight="balanced_subsample",
            n_jobs=-1, random_state=PY_SEED,
        )

    def et():
        return ExtraTreesClassifier(
            n_estimators=500, max_depth=None,
            class_weight="balanced_subsample",
            n_jobs=-1, random_state=PY_SEED,
        )

    def gb():
        return GradientBoostingClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=3,
            subsample=0.8, random_state=PY_SEED,
        )

    def hgb():
        return HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, max_depth=None,
            l2_regularization=1.0, random_state=PY_SEED,
        )

    def ada():
        # sklearn 1.8+: AdaBoost uses SAMME by default, base est is a tree stump
        return AdaBoostClassifier(
            n_estimators=200, learning_rate=0.5, random_state=PY_SEED,
        )

    def dt():
        return DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=20,
            class_weight="balanced", random_state=PY_SEED,
        )

    def svm():
        return Pipeline([
            ("s", StandardScaler()),
            ("c", SVC(kernel="rbf", C=1.0, gamma="scale",
                      probability=True, random_state=PY_SEED)),
        ])

    def lda():
        return Pipeline([
            ("s", StandardScaler()),
            ("c", LinearDiscriminantAnalysis(
                solver="lsqr", shrinkage="auto")),
        ])

    # Order: LR (baseline) first, then ensemble-by-family.
    return {
        "L2-Logistic": lr_l2,
        "RandomForest": rf,
        "ExtraTrees": et,
        "GradientBoosting": gb,
        "HistGB": hgb,
        "AdaBoost": ada,
        "DecisionTree": dt,
        "SVM-RBF": svm,
        "LDA": lda,
    }


# ---------------------------------------------------------------------------
# OOF + temporal predictor (identical to v1)
# ---------------------------------------------------------------------------


def baseline_oof_and_temporal(factory, X_dev, y_dev, X_te, splits):
    oof = np.zeros(len(X_dev))
    t0 = time.perf_counter()
    for tr, va in splits:
        m = factory()
        m.fit(X_dev[tr], y_dev[tr])
        oof[va] = m.predict_proba(X_dev[va])[:, 1]
    final = factory()
    final.fit(X_dev, y_dev)
    p_te = final.predict_proba(X_te)[:, 1]
    fit_sec = time.perf_counter() - t0
    return oof, p_te, fit_sec


def paired_bootstrap_aucs(y_te, preds_by_method, ep_te,
                          n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    eps = np.unique(ep_te)
    rows_by_ep = {e: np.where(ep_te == e)[0] for e in eps}
    names = list(preds_by_method.keys())
    aucs = {m: [] for m in names}
    for _ in range(n):
        e = rng.choice(eps, len(eps), replace=True)
        idx = np.concatenate([rows_by_ep[ee] for ee in e])
        yb = y_te[idx]
        if len(np.unique(yb)) < 2:
            continue
        for m in names:
            aucs[m].append(roc_auc_score(yb, preds_by_method[m][idx]))
    return aucs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)

    print("=" * 72, flush=True)
    print(f"M2·v2 · 6M EBM vs L2-LR + 8 sklearn classical (v2, "
          f"LANDMARK={LANDMARK})", flush=True)
    print("=" * 72, flush=True)

    # ---- 1. Load rows ----
    print("\n[1/6] build_rows_for_method('median', (1,3,6,12)) …", flush=True)
    rows = build_rows_for_method("median", landmarks=(1, 3, 6, 12))
    y = rows["Y_24M_NHRH"].values
    lm = rows["landmark"].values
    is_dev = (rows["Split"] == "Development").values
    ep = rows["episode_id"].values
    devL = is_dev & (lm == LANDMARK)
    tstL = (~is_dev) & (lm == LANDMARK)
    print(f"  rows: {rows.shape}; dev@6M: {devL.sum()}; tst@6M: {tstL.sum()}",
          flush=True)

    # ---- 2. EBM anchor ----
    print("\n[2/6] EBM via ebm_oof_and_temporal (locked) …", flush=True)
    t_ebm0 = time.perf_counter()
    ebm_pred, ebm_model, ebm_live = ebm_oof_and_temporal(
        rows, y, lm, is_dev, LANDMARK,
        interactions=5, max_interaction_bins=16, seed=PY_SEED,
    )
    ebm_fit_sec = time.perf_counter() - t_ebm0
    ebm_p_te = ebm_pred[tstL]
    ebm_p_dev = ebm_pred[devL]
    ebm_auc = roc_auc_score(y[tstL], ebm_p_te)
    print(f"  EBM live: {len(ebm_live)}; 6M temporal AUC = {ebm_auc:.4f}  "
          f"(locked = {EBM_LOCKED_AUC})", flush=True)
    if abs(ebm_auc - EBM_LOCKED_AUC) > TOL:
        raise SystemExit(
            f"ABORT: EBM AUC {ebm_auc:.4f} differs from locked "
            f"{EBM_LOCKED_AUC} by > {TOL}"
        )
    print("  ✓ EBM AUC matches locked atlas value", flush=True)

    # ---- 3. Shared X (same live set as EBM) ----
    print("\n[3/6] build_feats_at_L (shared X) …", flush=True)
    feat_all = build_feats_at_L(rows, devL)
    live = ebm_live
    X_dev = feat_all.loc[devL, live].values
    X_te = feat_all.loc[tstL, live].values
    y_dev = y[devL]
    y_te = y[tstL]
    ep_dev = ep[devL]
    ep_te = ep[tstL]
    print(f"  X_dev {X_dev.shape};  X_te {X_te.shape};  "
          f"prev dev {y_dev.mean():.3f} / tst {y_te.mean():.3f}", flush=True)

    # ---- 4. Shared fold ----
    print("\n[4/6] StratifiedGroupKFold(n=5, seed={}, groups=episode_id) …"
          .format(CV_SEED), flush=True)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    splits = list(skf.split(X_dev, y_dev, groups=ep_dev))
    fold_sig = hashlib.md5(
        "|".join("|".join(map(str, va.tolist())) for _, va in splits).encode()
    ).hexdigest()[:16]
    print(f"  fold signature: {fold_sig}", flush=True)

    # ---- 5. 9 methods (LR baseline + 8 sklearn classical) ----
    print("\n[5/6] 9 methods …", flush=True)
    methods_dict = make_methods()
    preds_te = {"EBM": ebm_p_te}
    preds_dev = {"EBM": ebm_p_dev}
    fit_seconds = {"EBM": ebm_fit_sec}
    for name, factory in methods_dict.items():
        print(f"  fitting {name} …", flush=True)
        oof, p_te, fit_sec = baseline_oof_and_temporal(
            factory, X_dev, y_dev, X_te, splits
        )
        preds_te[name] = p_te
        preds_dev[name] = oof
        fit_seconds[name] = fit_sec
        print(f"    OOF AUC = {roc_auc_score(y_dev, oof):.4f}  "
              f"temporal AUC = {roc_auc_score(y_te, p_te):.4f}  "
              f"fit_sec = {fit_sec:.2f}", flush=True)

    # ---- 6. Metrics + paired bootstrap ----
    print("\n[6/6] paired bootstrap × 1000 …", flush=True)
    methods = list(preds_te.keys())  # ['EBM', 'L2-Logistic', 'RF', ...]

    perf_rows = []
    for m in methods:
        p = preds_te[m]
        perf_rows.append({
            "method": m,
            "ROC_AUC": roc_auc_score(y_te, p),
            "PR_AUC": average_precision_score(y_te, p),
            "Brier": brier_score_loss(y_te, p),
            "fit_sec": fit_seconds[m],
        })

    aucs_boot = paired_bootstrap_aucs(y_te, preds_te, ep_te,
                                      n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
    for r in perf_rows:
        a = aucs_boot[r["method"]]
        r["AUC_CI_low"] = float(np.percentile(a, 2.5))
        r["AUC_CI_high"] = float(np.percentile(a, 97.5))

    perf = (pd.DataFrame(perf_rows)
              .sort_values("ROC_AUC", ascending=False)
              .reset_index(drop=True))
    perf_out = perf[["method", "ROC_AUC", "AUC_CI_low", "AUC_CI_high",
                     "PR_AUC", "Brier", "fit_sec"]].round(4)
    perf_out.to_csv(OUT / "perf.csv", index=False)
    print("\n=== perf.csv ===")
    print(perf_out.to_string(index=False), flush=True)

    # paired ΔAUC vs EBM
    delta_rows = []
    ebm_a = np.array(aucs_boot["EBM"])
    for m in methods:
        if m == "EBM":
            continue
        a = np.array(aucs_boot[m])
        d = a - ebm_a
        lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
        p_two = float(2 * min(np.mean(d <= 0), np.mean(d >= 0)))
        delta_rows.append({
            "method": m,
            "delta_AUC_mean": float(np.mean(d)),
            "delta_CI_low": lo,
            "delta_CI_high": hi,
            "p_two_sided": p_two,
            "significant_vs_EBM": bool((lo > 0) or (hi < 0)),
        })
    delta = (pd.DataFrame(delta_rows)
               .sort_values("delta_AUC_mean", ascending=False)
               .reset_index(drop=True))
    delta_out = delta.round(4)
    delta_out.to_csv(OUT / "paired_vs_ebm.csv", index=False)
    print("\n=== paired_vs_ebm.csv ===")
    print(delta_out.to_string(index=False), flush=True)

    # ---- Figures ----
    print("\nGenerating figures …", flush=True)
    colors = {
        "EBM":              "#1d4e89",
        "L2-Logistic":      "#cc4444",
        "RandomForest":     "#2a7f5f",
        "ExtraTrees":       "#5fa363",
        "GradientBoosting": "#9c6db0",
        "HistGB":           "#b07aa1",
        "AdaBoost":         "#e69138",
        "DecisionTree":     "#8a5a3b",
        "SVM-RBF":          "#3e8ec1",
        "LDA":              "#888888",
    }

    # F1 · ROC overlay
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for m in methods:
        fpr, tpr, _ = roc_curve(y_te, preds_te[m])
        a = roc_auc_score(y_te, preds_te[m])
        ls = "-" if m == "EBM" else "--"
        lw = 2.6 if m == "EBM" else 1.2
        ax.plot(fpr, tpr, color=colors.get(m, "#666"),
                lw=lw, ls=ls, label=f"{m} {a:.3f}")
    ax.plot([0, 1], [0, 1], ":", color="#bbb", lw=0.8)
    ax.set_title("6M temporal ROC — EBM (solid) vs LR + 8 sklearn classical (v2)",
                 fontsize=10)
    ax.set_xlabel("1 - specificity", fontsize=9)
    ax.set_ylabel("sensitivity", fontsize=9)
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "Compare_ROC_6M.png", dpi=160)
    plt.close(fig)

    # F2 · PR overlay — legend 移到 axes 外底部 4 列,完全不挡曲线(避开 AdaBoost 等
    # 在 recall 0.6-0.9 区间的下沉曲线)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    for m in methods:
        pr, rc, _ = precision_recall_curve(y_te, preds_te[m])
        ap = average_precision_score(y_te, preds_te[m])
        ls = "-" if m == "EBM" else "--"
        lw = 2.6 if m == "EBM" else 1.2
        ax.plot(rc, pr, color=colors.get(m, "#666"),
                lw=lw, ls=ls, label=f"{m} {ap:.3f}")
    ax.axhline(y_te.mean(), ls=":", color="#bbb", lw=0.8,
               label=f"prevalence {y_te.mean():.3f}")
    ax.set_title("6M temporal PR — EBM (solid) vs LR + 8 sklearn classical (v2)",
                 fontsize=10)
    ax.set_xlabel("recall", fontsize=9)
    ax.set_ylabel("precision", fontsize=9)
    ax.set_ylim(0.35, 1.0)
    # legend 外置:axes 下方居中,4 列布局
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=4, fontsize=7, frameon=False,
              columnspacing=1.2, handlelength=2.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))   # 给底部 legend 留空间
    fig.savefig(OUT / "figures" / "Compare_PR_6M.png", dpi=160)
    plt.close(fig)

    # F3 · DCA overlay
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    prev = float(y_te.mean())
    pts = np.linspace(0.05, 0.6, 40)
    for m in methods:
        p = preds_te[m]
        nb = []
        for pt in pts:
            yhat = (p >= pt).astype(int)
            tp = int(((yhat == 1) & (y_te == 1)).sum())
            fp = int(((yhat == 1) & (y_te == 0)).sum())
            nb.append(tp / len(y_te) - fp / len(y_te) * pt / (1 - pt))
        ls = "-" if m == "EBM" else "--"
        lw = 2.6 if m == "EBM" else 1.1
        ax.plot(pts, nb, color=colors.get(m, "#666"), lw=lw, ls=ls, label=m)
    nb_all = [prev - (1 - prev) * pt / (1 - pt) for pt in pts]
    ax.plot(pts, nb_all, "--", color="#999", lw=0.8, label="treat-all")
    ax.axhline(0, color="#bbb", lw=0.8)
    ax.set_title("6M temporal decision curve — net benefit (v2)", fontsize=10)
    ax.set_xlabel("threshold probability", fontsize=9)
    ax.set_ylabel("net benefit", fontsize=9)
    ax.legend(loc="upper right", fontsize=6, frameon=False, ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "Compare_DCA_6M.png", dpi=160)
    plt.close(fig)

    # F4 · AUC bar with 95% CI
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    y_pos = np.arange(len(perf))[::-1]
    errs = np.array([
        perf["ROC_AUC"].values - perf["AUC_CI_low"].values,
        perf["AUC_CI_high"].values - perf["ROC_AUC"].values,
    ])
    bar_colors = [colors.get(m, "#666") for m in perf["method"]]
    ax.barh(y_pos, perf["ROC_AUC"].values, xerr=errs, color=bar_colors,
            error_kw=dict(ecolor="#333", capsize=3, lw=0.8))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(perf["method"].values)
    ax.set_xlim(0.5, 1.0)
    ax.axvline(EBM_LOCKED_AUC, color="#1d4e89", ls=":", lw=0.8,
               label=f"EBM anchor {EBM_LOCKED_AUC:.3f}")
    ax.set_xlabel("temporal ROC-AUC (95% CI, paired episode-cluster bootstrap × 1000)",
                  fontsize=8)
    ax.set_title("6M temporal AUC — 10 methods (v2)", fontsize=10)
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    for i, (m, v, hi) in enumerate(zip(perf["method"], perf["ROC_AUC"],
                                       perf["AUC_CI_high"])):
        ax.text(hi + 0.005, y_pos[i], f"{v:.3f}", va="center", fontsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "Compare_AUC_bar.png", dpi=160)
    plt.close(fig)

    # F5 · paired Δ forest — 删 axes 内"读图指南"灰字(会和末尾 LDA 标签视觉重叠,误读
    # 为 LDA 注解);改放到 x 轴 label + axes 顶部双向箭头
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    y_pos = np.arange(len(delta))[::-1]
    errs = np.array([
        delta["delta_AUC_mean"].values - delta["delta_CI_low"].values,
        delta["delta_CI_high"].values - delta["delta_AUC_mean"].values,
    ])
    point_colors = [
        "#cc4444" if (lo > 0) else ("#2a7f5f" if (hi < 0) else "#1d4e89")
        for lo, hi in zip(delta["delta_CI_low"], delta["delta_CI_high"])
    ]
    for i, (lo, hi) in enumerate(zip(delta["delta_CI_low"],
                                     delta["delta_CI_high"])):
        ax.plot([lo, hi], [y_pos[i], y_pos[i]],
                color=point_colors[i], lw=2.2, alpha=0.45)
    ax.scatter(delta["delta_AUC_mean"].values, y_pos,
               color=point_colors, s=55, zorder=3,
               edgecolors="#222", linewidths=0.6)
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(delta["method"].values)
    # 把"读图方向指南"嵌入到 x 轴 label,不再在 axes 内放灰字
    ax.set_xlabel(
        "Δ AUC = method - EBM   "
        "(← worse than EBM    |    better than EBM →)\n"
        "paired bootstrap × 1000, 95% CI",
        fontsize=8,
    )
    ax.set_title("6M paired ΔAUC vs EBM — forest plot (v2)", fontsize=10)
    # 在 0 线两侧 axes 顶部加方向箭头注解(放在数据点上方,远离 method 标签)
    xlim = ax.get_xlim()
    y_top = max(y_pos) + 0.7
    ax.annotate(
        "", xy=(xlim[1] * 0.85, y_top), xytext=(0.002, y_top),
        arrowprops=dict(arrowstyle="->", color="#cc4444", lw=1.2, alpha=0.6),
    )
    ax.text(xlim[1] * 0.5, y_top + 0.15, "better than EBM",
            fontsize=7, color="#cc4444", ha="center", va="bottom", alpha=0.8)
    ax.annotate(
        "", xy=(xlim[0] * 0.85, y_top), xytext=(-0.002, y_top),
        arrowprops=dict(arrowstyle="->", color="#2a7f5f", lw=1.2, alpha=0.6),
    )
    ax.text(xlim[0] * 0.5, y_top + 0.15, "worse than EBM",
            fontsize=7, color="#2a7f5f", ha="center", va="bottom", alpha=0.8)
    ax.set_ylim(-0.7, y_top + 0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "Compare_paired_dAUC.png", dpi=160)
    plt.close(fig)

    # ---- method_setup.md ----
    setup_md = """# Method setup — 6M EBM vs LR baseline + 8 sklearn classical (v2)

All 10 methods share the SAME inputs:

- Same `X` = 16 live features from `build_feats_at_L(rows@dev6M)`
- Same `y` = `Y_24M_NHRH`
- Same `is_dev` / `episode_id`
- Same 5-fold `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=13)`
- Same temporal cohort (201 episodes, 6M landmark)
- Same paired episode-cluster bootstrap × 1000 (seed=7)

| # | Method | Hyperparameters |
|--:|:--|:--|
| 0 | **EBM** (anchor, locked) | `ExplainableBoostingClassifier(interactions=5, max_interaction_bins=16, random_state=2025)` via `ebm_oof_and_temporal(L=6)` |
| 1 | L2-Logistic (baseline) | `StandardScaler` + `LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=2000, random_state=2025)` |
| 2 | RandomForest | `RandomForestClassifier(n_estimators=500, max_depth=None, class_weight="balanced_subsample", random_state=2025)` |
| 3 | ExtraTrees | `ExtraTreesClassifier(n_estimators=500, max_depth=None, class_weight="balanced_subsample", random_state=2025)` |
| 4 | GradientBoosting | `GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=2025)` |
| 5 | HistGB | `HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=2025)` |
| 6 | AdaBoost | `AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=2025)` (SAMME, tree stump) |
| 7 | DecisionTree | `DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, class_weight="balanced", random_state=2025)` |
| 8 | SVM-RBF | `StandardScaler` + `SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=2025)` |
| 9 | LDA | `StandardScaler` + `LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")` |

EBM anchor protocol guarantees match to the atlas mainline number
(`m2v2_ebm_full_median/ebm_median_metrics.json` 6M temporal AUC reference);
script aborts on > 0.003 drift.
"""
    (OUT / "tables" / "method_setup.md").write_text(setup_md)

    # ---- audit.json ----
    rows_hash = hashlib.md5(
        pd.util.hash_pandas_object(rows.fillna(-9999.0)).values.tobytes()
    ).hexdigest()[:16]
    X_hash = hashlib.md5(X_dev.tobytes() + X_te.tobytes()).hexdigest()[:16]
    audit = {
        "口径": "corrected truth (Current_Time) + median impute (time-safe), landmark=6M",
        "lock": {
            "EBM_target_AUC": EBM_LOCKED_AUC,
            "EBM_observed_AUC": round(float(ebm_auc), 4),
            "tolerance": TOL,
            "match": bool(abs(ebm_auc - EBM_LOCKED_AUC) <= TOL),
            "ebm_via": ("ebm_oof_and_temporal(L=6, interactions=5, "
                        "max_interaction_bins=16, seed=PY_SEED=2025)"),
        },
        "data": {
            "rows_shape": list(rows.shape),
            "dev_at_6M": int(devL.sum()),
            "temporal_at_6M": int(tstL.sum()),
            "features_total": len(FEATS),
            "features_live": len(live),
            "X_dev_shape": list(X_dev.shape),
            "X_te_shape": list(X_te.shape),
            "y_dev_event_rate": round(float(y_dev.mean()), 4),
            "y_te_event_rate": round(float(y_te.mean()), 4),
            "rows_hash": rows_hash,
            "X_hash": X_hash,
            "fold_signature": fold_sig,
        },
        "seeds": {
            "PY_SEED": PY_SEED, "CV_SEED": CV_SEED,
            "BOOTSTRAP_SEED": BOOTSTRAP_SEED,
        },
        "bootstrap_n": BOOTSTRAP_N,
        "methods": methods,
        "perf_summary": perf_out.to_dict(orient="records"),
        "delta_summary": delta_out.to_dict(orient="records"),
    }
    (OUT / "tables" / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False)
    )

    # ---- compliance ----
    text_blob = (
        perf_out.to_csv(index=False)
        + delta_out.to_csv(index=False)
        + setup_md
        + json.dumps(audit, ensure_ascii=False)
    )
    n_889 = text_blob.count("889")
    print(f"\n  Compliance: '889' substring count = {n_889}", flush=True)

    print(f"\nDone — all outputs under {OUT}", flush=True)


if __name__ == "__main__":
    main()
