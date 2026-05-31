#!/usr/bin/env python
"""M2 · v2 — EBM figure atlas: re-create the full M2 figure battery via the EBM
glassbox (≈47 figures), consistently.

Per landmark (0/1/3/6M): ROC, PR, calibration, EBM term importance, shape of the
top-2 features, decision curve (DCA), risk-tertile event rate, confusion matrix
(9 × 4 = 36). Plus cross-landmark summaries (importance drift, EBM vs LR, method×
landmark, selective prediction, calibration summary, multi-seed stability, lead
shapes, FT3/FT4 collinearity) → ≈47 total.

Output: results/module2_v2_vertical/m2v2_ebm_full/figures/ + manifest.json.
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
import matplotlib.font_manager as _fm
for _fp in ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "/Library/Fonts/Arial Unicode.ttf"):
    if os.path.exists(_fp):
        _fm.fontManager.addfont(_fp); plt.rcParams["font.family"] = "Arial Unicode MS"; break
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.simple.module2_v2_shared import LANDMARKS, load_stacked, PY_SEED
from scripts.simple.module2_v2_b4_trab_patch import add_current_velocity
from scripts.simple.module2_v2_b4_ebm_axes import build_feats_at_L, FEATS

OUT = ROOT / "results" / "module2_v2_vertical" / "m2v2_ebm_full"
FIGD = OUT / "figures"
DISP = {"ThyroidW": "Thyroid weight", "TRAb": "TRAb", "TGAb": "TGAb", "TPOAb": "TPOAb",
        "Sex": "Sex", "FT4_0M": "FT4 (0M)", "TSH_0M": "TSH (0M)",
        "log1p_DiseaseDuration_Months_Aug": "Duration (log,mo)", "Uptake24h": "24h uptake",
        "HalfLife": "Iodine half-life", "TSH_current": "TSH (current)", "TSH_velocity": "TSH velocity",
        "Hormone_load": "FT3,FT4 level", "T3T4_balance": "FT3,FT4 gap",
        "Velocity_load": "FT3,FT4 velocity", "Velocity_balance": "FT3,FT4 vel gap"}
manifest = []
_n = [0]


def _disp(t):
    return " × ".join(DISP.get(p, p) for p in t.split(" & ")) if " & " in t else DISP.get(t, t)


def save(fig, slug, caption):
    _n[0] += 1
    name = f"F{_n[0]:02d}_{slug}.png"
    fig.savefig(FIGD / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    manifest.append({"n": _n[0], "file": name, "caption": caption})


def _lr():
    return Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=5000, random_state=PY_SEED))])


def main() -> None:
    FIGD.mkdir(parents=True, exist_ok=True)
    sd = load_stacked(); sd = add_current_velocity(sd, "FT3")
    rows = sd.rows
    y = rows["Y_24M_NHRH"].values; lm = rows["landmark"].values
    is_dev = (rows["Split"] == "Development").values

    M = {}  # per-landmark fitted artifacts
    for L in LANDMARKS:
        devL = is_dev & (lm == L); tstL = (~is_dev) & (lm == L)
        feat = build_feats_at_L(rows, devL)
        live = [c for c in FEATS if feat.loc[devL, c].std() > 1e-9]
        Xtr, ytr = feat.loc[devL, live], y[devL]
        Xte, yte = feat.loc[tstL, live], y[tstL]
        ebm = ExplainableBoostingClassifier(random_state=PY_SEED, interactions=5, max_interaction_bins=16).fit(Xtr, ytr)
        lr = _lr().fit(Xtr.values, ytr)
        pe = ebm.predict_proba(Xte)[:, 1]; pe_dev = ebm.predict_proba(Xtr)[:, 1]
        pl = lr.predict_proba(Xte.values)[:, 1]
        M[L] = dict(ebm=ebm, lr=lr, Xte=Xte, yte=yte, pe=pe, pe_dev=pe_dev, ydev=ytr, pl=pl,
                    auc=roc_auc_score(yte, pe), live=live)

    # ---- per-landmark figures ----
    for L in LANDMARKS:
        m = M[L]; yte, pe = m["yte"], m["pe"]
        # ROC
        fpr, tpr, _ = roc_curve(yte, pe)
        f, a = plt.subplots(figsize=(4, 3.6)); a.plot(fpr, tpr, color="#2a6f97", lw=2)
        a.plot([0, 1], [0, 1], ":", color="#999"); a.set_title(f"{L}M EBM ROC (AUC {m['auc']:.3f})", fontsize=9)
        a.set_xlabel("1−spec", fontsize=8); a.set_ylabel("sens", fontsize=8)
        save(f, f"ROC_{L}M", f"{L}M EBM ROC,AUC {m['auc']:.3f}")
        # PR
        pr, rc, _ = precision_recall_curve(yte, pe); ap = average_precision_score(yte, pe)
        f, a = plt.subplots(figsize=(4, 3.6)); a.plot(rc, pr, color="#8a5a3b", lw=2)
        a.axhline(yte.mean(), ls=":", color="#999"); a.set_title(f"{L}M EBM PR (AP {ap:.3f})", fontsize=9)
        a.set_xlabel("recall", fontsize=8); a.set_ylabel("precision", fontsize=8)
        save(f, f"PR_{L}M", f"{L}M EBM precision-recall,AP {ap:.3f}")
        # Calibration
        bins = np.quantile(pe, np.linspace(0, 1, 6)); bins[0], bins[-1] = -1e-3, 1 + 1e-3
        bi = np.digitize(pe, bins) - 1; xs, ys = [], []
        for j in range(5):
            mm = bi == j
            if mm.sum() >= 4:
                xs.append(pe[mm].mean()); ys.append(yte[mm].mean())
        f, a = plt.subplots(figsize=(4, 3.6)); a.plot([0, 1], [0, 1], ":", color="#999")
        a.plot(xs, ys, "o-", color="#2a7f5f"); a.set_title(f"{L}M EBM calibration", fontsize=9)
        a.set_xlabel("predicted", fontsize=8); a.set_ylabel("observed", fontsize=8); a.set_xlim(0, 1); a.set_ylim(0, 1)
        save(f, f"Calib_{L}M", f"{L}M EBM reliability")
        # Importance bar
        g = m["ebm"].explain_global(); nm, sc = g.data()["names"], g.data()["scores"]
        top = sorted(zip(nm, sc), key=lambda t: -t[1])[:6]
        f, a = plt.subplots(figsize=(4.6, 3.6))
        a.barh(range(len(top)), [s for _, s in top][::-1], color="#2a6f97")
        a.set_yticks(range(len(top))); a.set_yticklabels([_disp(n) for n, _ in top][::-1], fontsize=7)
        a.set_title(f"{L}M EBM importance", fontsize=9); a.set_xlabel("mean |contribution|", fontsize=8)
        save(f, f"Importance_{L}M", f"{L}M EBM 原生重要性 top-6")
        # Shapes (top-2 UNIVARIATE features; skip interaction terms "A & B")
        univ = [(n, s) for n, s in top if " & " not in n][:2]
        for rank, (feat_n, _) in enumerate(univ):
            if feat_n not in m["ebm"].term_names_:
                continue
            dd = g.data(m["ebm"].term_names_.index(feat_n))
            xs2v, ys2v = dd.get("names"), dd.get("scores")
            if xs2v is None or ys2v is None:
                continue
            xs2 = np.asarray(xs2v, float); ys2 = np.asarray(ys2v, float)
            f, a = plt.subplots(figsize=(4, 3.6))
            if len(xs2) == len(ys2) + 1:
                mid = (xs2[:-1] + xs2[1:]) / 2; a.step(mid, ys2, where="mid", color="#a23b3b", lw=2)
                a.fill_between(mid, ys2, step="mid", alpha=0.12, color="#a23b3b")
            else:
                a.plot(xs2[:len(ys2)], ys2, "o-", color="#a23b3b")
            a.axhline(0, color="#999", ls=":", lw=.8); a.set_title(f"{L}M shape: {_disp(feat_n)}", fontsize=9)
            a.set_xlabel("value", fontsize=8); a.set_ylabel("log-odds contrib", fontsize=8)
            save(f, f"Shape_{L}M_r{rank+1}", f"{L}M EBM 形状函数 #{rank+1}:{_disp(feat_n)}")
        # DCA
        prev = yte.mean(); pts = np.linspace(0.05, 0.6, 40); nb_m, nb_all = [], []
        for pt in pts:
            pred = (pe >= pt).astype(int); tp = ((pred == 1) & (yte == 1)).sum(); fp = ((pred == 1) & (yte == 0)).sum()
            nb_m.append(tp / len(yte) - fp / len(yte) * pt / (1 - pt))
            nb_all.append(prev - (1 - prev) * pt / (1 - pt))
        f, a = plt.subplots(figsize=(4, 3.6)); a.plot(pts, nb_m, color="#1d4e89", lw=2, label="EBM")
        a.plot(pts, nb_all, "--", color="#888", label="treat-all"); a.axhline(0, color="#bbb", lw=.8, label="treat-none")
        a.set_title(f"{L}M EBM decision curve", fontsize=9); a.set_xlabel("threshold", fontsize=8)
        a.set_ylabel("net benefit", fontsize=8); a.legend(fontsize=6)
        save(f, f"DCA_{L}M", f"{L}M EBM 决策曲线(DCA)")
        # Risk-tertile event rate
        c1, c2 = np.quantile(m["pe_dev"], [1 / 3, 2 / 3])
        tier = np.where(pe <= c1, 0, np.where(pe <= c2, 1, 2))
        er = [yte[tier == t].mean() if (tier == t).sum() else np.nan for t in (0, 1, 2)]
        f, a = plt.subplots(figsize=(4, 3.6)); a.bar(["low", "mid", "high"], er, color=["#6aa84f", "#e69138", "#cc0000"])
        for i, v in enumerate(er):
            if not np.isnan(v):
                a.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        a.set_title(f"{L}M EBM 风险三档事件率", fontsize=9); a.set_ylabel("temporal event rate", fontsize=8); a.set_ylim(0, 1)
        save(f, f"RiskTier_{L}M", f"{L}M EBM 风险三档(dev 三分位)事件率")
        # Confusion at Youden(dev)
        ths = np.linspace(0.05, 0.95, 181)
        def yj(t):
            pr_ = (m["pe_dev"] >= t).astype(int); yy = m["ydev"]
            tp = ((pr_ == 1) & (yy == 1)).sum(); fn = ((pr_ == 0) & (yy == 1)).sum()
            tn = ((pr_ == 0) & (yy == 0)).sum(); fp = ((pr_ == 1) & (yy == 0)).sum()
            return (tp / (tp + fn + 1e-9)) + (tn / (tn + fp + 1e-9)) - 1
        thr = float(ths[int(np.argmax([yj(t) for t in ths]))])
        pr_ = (pe >= thr).astype(int)
        cm = np.array([[((pr_ == 0) & (yte == 0)).sum(), ((pr_ == 1) & (yte == 0)).sum()],
                       [((pr_ == 0) & (yte == 1)).sum(), ((pr_ == 1) & (yte == 1)).sum()]])
        f, a = plt.subplots(figsize=(3.8, 3.4)); im = a.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                a.text(j, i, int(cm[i, j]), ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() * .6 else "#222", fontsize=11)
        a.set_xticks([0, 1]); a.set_xticklabels(["pred 0", "pred 1"], fontsize=8)
        a.set_yticks([0, 1]); a.set_yticklabels(["true 0", "true 1"], fontsize=8)
        a.set_title(f"{L}M EBM confusion (thr {thr:.2f})", fontsize=9)
        save(f, f"Confusion_{L}M", f"{L}M EBM 混淆矩阵(Youden 阈值)")

    # ---- summaries ----
    aucs = [M[L]["auc"] for L in LANDMARKS]
    aps = [average_precision_score(M[L]["yte"], M[L]["pe"]) for L in LANDMARKS]
    brs = [brier_score_loss(M[L]["yte"], M[L]["pe"]) for L in LANDMARKS]
    f, a = plt.subplots(figsize=(5, 3.6)); xp = range(4)
    a.plot(xp, aucs, "o-", label="ROC-AUC"); a.plot(xp, aps, "s-", label="PR-AUC"); a.plot(xp, brs, "^-", label="Brier")
    a.set_xticks(xp); a.set_xticklabels([f"{L}M" for L in LANDMARKS]); a.legend(fontsize=8)
    a.set_title("EBM 判别/校准随地标递增", fontsize=10)
    save(f, "Metrics_over_time", "EBM ROC/PR/Brier 随地标")

    lrs = [roc_auc_score(M[L]["yte"], M[L]["pl"]) for L in LANDMARKS]
    f, a = plt.subplots(figsize=(5, 3.6)); a.plot(xp, aucs, "o-", label="EBM"); a.plot(xp, lrs, "s--", label="LR")
    a.set_xticks(xp); a.set_xticklabels([f"{L}M" for L in LANDMARKS]); a.legend(fontsize=8); a.set_ylim(0.6, 0.86)
    a.set_title("EBM vs 简约 LR(逐地标 AUC)", fontsize=10)
    save(f, "EBM_vs_LR", "EBM vs LR 逐地标 AUC")

    f, a = plt.subplots(figsize=(5, 4))
    for L in LANDMARKS:
        fpr, tpr, _ = roc_curve(M[L]["yte"], M[L]["pe"]); a.plot(fpr, tpr, lw=2, label=f"{L}M ({M[L]['auc']:.3f})")
    a.plot([0, 1], [0, 1], ":", color="#999"); a.legend(fontsize=8); a.set_title("EBM ROC 全地标合并", fontsize=10)
    a.set_xlabel("1−spec"); a.set_ylabel("sens")
    save(f, "ROC_all", "EBM 全地标 ROC")

    # multi-seed stability
    f, a = plt.subplots(figsize=(5, 3.6)); data = []
    for L in LANDMARKS:
        feat = build_feats_at_L(rows, is_dev & (lm == L)); live = M[L]["live"]
        Xtr = feat.loc[is_dev & (lm == L), live]; ytr = M[L]["ydev"]
        Xte = M[L]["Xte"]; yte = M[L]["yte"]; seedaucs = []
        for s in (1, 2, 3, 4, 5):
            e = ExplainableBoostingClassifier(random_state=s, interactions=5, max_interaction_bins=16).fit(Xtr, ytr)
            seedaucs.append(roc_auc_score(yte, e.predict_proba(Xte)[:, 1]))
        data.append(seedaucs)
    a.boxplot(data, labels=[f"{L}M" for L in LANDMARKS]); a.set_title("EBM 多seed AUC 稳定性", fontsize=10)
    a.set_ylabel("temporal AUC")
    save(f, "MultiSeed_stability", "EBM 多 seed(5)temporal AUC 稳定性")

    # F41 group-importance drift heatmap (EBM native, univariate)
    GRP = {"Goiter": ["ThyroidW"], "Antibodies": ["TRAb", "TGAb", "TPOAb"],
           "Baseline/dur": ["FT4_0M", "TSH_0M", "log1p_DiseaseDuration_Months_Aug", "Sex"],
           "RAI": ["Uptake24h", "HalfLife"], "FT3,FT4 level": ["Hormone_load"], "TSH level": ["TSH_current"],
           "FT3,FT4 velocity": ["Velocity_load"], "TSH velocity": ["TSH_velocity"], "FT3,FT4 gap": ["T3T4_balance"]}
    gmat = np.zeros((len(GRP), 4))
    for j, L in enumerate(LANDMARKS):
        gg = M[L]["ebm"].explain_global(); imp = {n: s for n, s in zip(gg.data()["names"], gg.data()["scores"]) if " & " not in n}
        for i, (gn, cols) in enumerate(GRP.items()):
            gmat[i, j] = sum(imp.get(c, 0) for c in cols)
    f, a = plt.subplots(figsize=(6.5, 4.4)); im = a.imshow(gmat, cmap="YlOrRd", aspect="auto")
    a.set_xticks(range(4)); a.set_xticklabels([f"{L}M" for L in LANDMARKS]); a.set_yticks(range(len(GRP))); a.set_yticklabels(list(GRP), fontsize=8)
    for i in range(len(GRP)):
        for j in range(4):
            a.text(j, i, f"{gmat[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if gmat[i, j] > gmat.max() * .6 else "#222")
    a.set_title("EBM 原生重要性(组 × 地标)", fontsize=10); f.colorbar(im, ax=a, fraction=.025)
    save(f, "GroupImportance_heatmap", "EBM 原生重要性 组×地标 漂移热力图")

    # F42 calibration summary scatter
    def _cal(yy, pp):
        pp = np.clip(pp, 1e-6, 1 - 1e-6); z = np.log(pp / (1 - pp)).reshape(-1, 1)
        l = LogisticRegression(solver="lbfgs", max_iter=2000).fit(z, yy); return float(l.intercept_[0]), float(l.coef_[0][0])
    f, a = plt.subplots(figsize=(4.6, 4))
    for L in LANDMARKS:
        ic, sl = _cal(M[L]["yte"], M[L]["pe"]); a.scatter(ic, sl, s=60); a.annotate(f"{L}M", (ic, sl), textcoords="offset points", xytext=(5, 4), fontsize=9)
    a.axhline(1, ls=":", color="#999"); a.axvline(0, ls=":", color="#999"); a.axhspan(.8, 1.2, alpha=.08, color="#2a7f5f")
    a.set_xlabel("calib intercept"); a.set_ylabel("calib slope"); a.set_title("EBM 校准汇总(各地标)", fontsize=10)
    save(f, "Calib_summary", "EBM 校准截距/斜率汇总")

    # F43 collinearity FT3/FT4 vs TSH
    f, a = plt.subplots(figsize=(5, 3.6)); xp2 = np.arange(4); w = .35; c34, ct4 = [], []
    for L in LANDMARKS:
        md = is_dev & (lm == L)
        f3 = rows.loc[md, "FT3_current"].values; f4 = rows.loc[md, "FT4_current"].values; ts = rows.loc[md, "TSH_current"].values
        c34.append(np.corrcoef(f3, f4)[0, 1]); ct4.append(np.corrcoef(ts, f4)[0, 1])
    a.bar(xp2 - w / 2, c34, w, label="corr(FT3,FT4)", color="#2a6f97"); a.bar(xp2 + w / 2, ct4, w, label="corr(TSH,FT4)", color="#a23b3b")
    a.set_xticks(xp2); a.set_xticklabels([f"{L}M" for L in LANDMARKS]); a.axhline(0, color="#999", lw=.8); a.legend(fontsize=8)
    a.set_title("共线性: FT3/FT4 同向 · TSH 反向", fontsize=10)
    save(f, "Collinearity", "FT3-FT4 同向 vs TSH 反向 相关性(轴构造依据)")

    # F44 lead shapes 2x2 panel
    f, axs = plt.subplots(2, 2, figsize=(10, 7))
    for k, L in enumerate(LANDMARKS):
        gg = M[L]["ebm"].explain_global()
        univ = [n for n, s in sorted(zip(gg.data()["names"], gg.data()["scores"]), key=lambda t: -t[1]) if " & " not in n]
        fn = univ[0]; dd = gg.data(M[L]["ebm"].term_names_.index(fn))
        xs = np.asarray(dd.get("names", []), float); ys = np.asarray(dd.get("scores", []), float); ax = axs[k // 2][k % 2]
        if len(xs) == len(ys) + 1:
            mid = (xs[:-1] + xs[1:]) / 2; ax.step(mid, ys, where="mid", color="#a23b3b", lw=2); ax.fill_between(mid, ys, step="mid", alpha=.12, color="#a23b3b")
        elif len(ys):
            ax.plot(xs[:len(ys)], ys, "o-", color="#a23b3b")
        ax.axhline(0, ls=":", color="#999"); ax.set_title(f"{L}M lead: {_disp(fn)}", fontsize=9)
    f.suptitle("EBM 主导特征形状函数(逐地标)", fontsize=11); f.tight_layout(rect=[0, 0, 1, .96])
    save(f, "Lead_shapes_panel", "EBM 各地标主导特征形状函数面板")

    # F45 selective prediction (EBM pooled temporal)
    allp = np.concatenate([M[L]["pe"] for L in LANDMARKS]); ally = np.concatenate([M[L]["yte"] for L in LANDMARKS])
    alldp = np.concatenate([M[L]["pe_dev"] for L in LANDMARKS]); alldy = np.concatenate([M[L]["ydev"] for L in LANDMARKS])
    ths = np.linspace(.05, .95, 181)
    def _yj(t, pp, yy):
        pr = (pp >= t).astype(int); tp = ((pr == 1) & (yy == 1)).sum(); fn = ((pr == 0) & (yy == 1)).sum(); tn = ((pr == 0) & (yy == 0)).sum(); fp = ((pr == 1) & (yy == 0)).sum()
        return tp / (tp + fn + 1e-9) + tn / (tn + fp + 1e-9) - 1
    thr = float(ths[int(np.argmax([_yj(t, alldp, alldy) for t in ths]))])
    order = np.argsort(-np.abs(allp - thr)); absts = [0, 10, 20, 30, 40, 50]; accs, npvs = [], []
    for ab in absts:
        keep = order[:int(round((1 - ab / 100) * len(ally)))]; yk = ally[keep]; pk = (allp[keep] >= thr).astype(int)
        tp = ((pk == 1) & (yk == 1)).sum(); tn = ((pk == 0) & (yk == 0)).sum(); fn = ((pk == 0) & (yk == 1)).sum()
        accs.append((tp + tn) / len(yk)); npvs.append(tn / (tn + fn + 1e-9))
    f, a = plt.subplots(figsize=(5.2, 3.8)); a.plot(absts, accs, "o-", label="accuracy", color="#1d4e89"); a.plot(absts, npvs, "s-", label="NPV", color="#2a7f5f")
    a.set_xlabel("abstention %"); a.set_ylabel("retained performance"); a.set_title("EBM 选择性预测(pooled temporal)", fontsize=10); a.legend(fontsize=8)
    save(f, "Selective_prediction", "EBM 选择性弃权 风险-覆盖(pooled)")

    # F46 operating point sens/spec/PPV/NPV per landmark
    f, a = plt.subplots(figsize=(6, 3.8)); mets = {"Sens": [], "Spec": [], "PPV": [], "NPV": []}
    for L in LANDMARKS:
        thr = float(ths[int(np.argmax([_yj(t, M[L]["pe_dev"], M[L]["ydev"]) for t in ths]))])
        pr = (M[L]["pe"] >= thr).astype(int); yy = M[L]["yte"]
        tp = ((pr == 1) & (yy == 1)).sum(); fp = ((pr == 1) & (yy == 0)).sum(); tn = ((pr == 0) & (yy == 0)).sum(); fn = ((pr == 0) & (yy == 1)).sum()
        mets["Sens"].append(tp / (tp + fn + 1e-9)); mets["Spec"].append(tn / (tn + fp + 1e-9)); mets["PPV"].append(tp / (tp + fp + 1e-9)); mets["NPV"].append(tn / (tn + fn + 1e-9))
    xp3 = np.arange(4); w = .2
    for i, (k, v) in enumerate(mets.items()):
        a.bar(xp3 + (i - 1.5) * w, v, w, label=k)
    a.set_xticks(xp3); a.set_xticklabels([f"{L}M" for L in LANDMARKS]); a.legend(fontsize=7, ncol=4); a.set_ylim(0, 1)
    a.set_title("EBM 工作点 Sens/Spec/PPV/NPV(Youden)", fontsize=10)
    save(f, "OperatingPoint", "EBM 各地标工作点 敏感度/特异度/PPV/NPV")

    # F47 pooled calibration reliability (all temporal)
    bins = np.quantile(allp, np.linspace(0, 1, 7)); bins[0], bins[-1] = -1e-3, 1 + 1e-3; bi = np.digitize(allp, bins) - 1
    xs, ys = [], []
    for j in range(6):
        mm = bi == j
        if mm.sum() >= 8:
            xs.append(allp[mm].mean()); ys.append(ally[mm].mean())
    f, a = plt.subplots(figsize=(4.4, 4)); a.plot([0, 1], [0, 1], ":", color="#999"); a.plot(xs, ys, "o-", color="#2a7f5f")
    a.set_xlabel("predicted"); a.set_ylabel("observed"); a.set_title("EBM pooled 校准(全地标)", fontsize=10); a.set_xlim(0, 1); a.set_ylim(0, 1)
    save(f, "Calib_pooled", "EBM pooled 校准可靠性(全地标合并)")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"生成 EBM 图谱 {_n[0]} 张 → {FIGD}", flush=True)
    for r in manifest:
        print(f"  F{r['n']:02d} {r['file']}", flush=True)


if __name__ == "__main__":
    main()
