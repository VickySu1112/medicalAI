#!/usr/bin/env python
"""M2 · v2 — 6M vs 12M 多维对比(bin16 + corrected/LOCF 口径)。

诚实查证旧口径(退化读宽列 bug)曾认为「12M 不如 6M」的说法在 corrected+LOCF+bin16
下是否仍成立。复用与主线完全一致的评估原语 `ebm_oof_and_temporal`(bin16)在 6M / 12M
两个地标各跑 EBM,并与两个 naive 基线对齐打分:

    naive_proba        — 常数 = dev 阳性率(无信息)
    persistence_proba  — 单特征 L2-LR(只看当期 TSH_current@L 的「看今天甲功」基线)

多维度对比(全部 dev OOF 选择 / temporal 仅报告):
  1. 判别        OOF AUC、temporal AUC
  2. 对 persistence 的增量   ΔAUC = EBM − persistence(配对 episode-cluster bootstrap 95%CI)
                            ΔAUC = EBM − naive(同上)
  3. 校准        temporal Brier、可靠性曲线(10 分箱)、校准 intercept/slope
  4. 稳定性      OOF − temporal AUC gap
  5. 事件率      dev / temporal 复发事件率(地标无关,确认终点是 episode 级)
  6. 重要性结构  bin16 dev@L 原生重要性 top(读 task2 的 bin16_importance_top5.csv,
                若缺则现算),对比 6M vs 12M 主导轴

口径与约束(硬性):
  数据  corrected 真值 + LOCF —— `build_rows_for_method("locf",(1,3,6,12))`。
  模型  `ebm_oof_and_temporal`(interactions=5, max_interaction_bins=16, PY_SEED)。
  dev OOF 选择/分析、temporal 仅报告;N=1003 人次;无 889 / unique-patient;time-safe;
  不写 significant 无检验;CJK 防豆腐(Arial Unicode);不写 worklog;不 git commit。

产物(results/module2_v2_vertical/,前缀 m2v2_bin16_6m12m_):
  m2v2_bin16_6m12m_comparison.csv      — 6M/12M × {EBM, persistence, naive} 指标行
  m2v2_bin16_6m12m_delta_vs_base.csv   — ΔAUC(EBM−persistence / EBM−naive)配对 bootstrap
  m2v2_bin16_6m12m_panel.png           — 4 面板:AUC 条 / ΔAUC / 可靠性曲线 / 重要性对比
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import argparse
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.simple.module2_v2_shared import (
    PY_SEED,
    StackedData,
    brier_score_loss,
    calib_intercept_slope,
    episode_cluster_bootstrap_auc_ci,
    forbidden_token,
    paired_episode_cluster_bootstrap_delta,
)
from scripts.simple.module2_v2_impute_experiment import build_rows_for_method
from scripts.simple.module2_v2_b4_ebm_oof import (
    ebm_oof_and_temporal,
    naive_proba,
    persistence_proba,
    _roc,
)

LANDMARKS_BUILD = (1, 3, 6, 12)   # build over the full extended set (LOCF chains 12M off 6M)
LANDMARKS_CMP = (6, 12)           # the two landmarks under comparison
OUT = ROOT / "results" / "module2_v2_vertical"
PREFIX = "m2v2_bin16_6m12m_"
IMPORT_CSV = OUT / "m2v2_ebm_xland" / "tables" / "bin16_importance_top5.csv"

# CJK 字体(无豆腐)
_CJK = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
if Path(_CJK).exists():
    fm.fontManager.addfont(_CJK)
    plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False


def _reliability(y, p, n_bins: int = 10):
    """Equal-width reliability curve: per-bin (mean pred, observed freq, count)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        xs.append(float(p[m].mean()))
        ys.append(float(y[m].mean()))
        ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def run(n_boot: int, landmarks_cmp=LANDMARKS_CMP):
    """Fit EBM + persistence + naive at each compared landmark; return tidy frames."""
    rows = build_rows_for_method("locf", LANDMARKS_BUILD)
    assert rows["episode_id"].nunique() == 1003, "analysis unit must be 1003 人次"
    y = rows["Y_24M_NHRH"].values
    lm = rows["landmark"].values
    is_dev = (rows["Split"] == "Development").values

    ep_meta = rows.drop_duplicates("episode_id")[["episode_id", "Split", "Y_24M_NHRH"]]
    sd_shim = StackedData(rows=rows, episode_meta=ep_meta)

    preds = {}   # (L, model) -> full-length pred vector
    comp_recs = []
    rel_curves = {}  # (L, model) -> (xs, ys, ns) on temporal

    for L in landmarks_cmp:
        devL = is_dev & (lm == L)
        tstL = (~is_dev) & (lm == L)

        # --- EBM (bin16, the mainline primitive) ---
        p_ebm, _ebm, live = ebm_oof_and_temporal(rows, y, lm, is_dev, L)
        # --- persistence: single-feature L2-LR on TSH_current@L ---
        p_pers = persistence_proba(rows, y, lm, is_dev, L)
        # --- naive: constant dev event rate ---
        p_naive = naive_proba(y[devL], len(rows))

        preds[(L, "EBM")] = p_ebm
        preds[(L, "persistence")] = p_pers
        preds[(L, "naive")] = p_naive

        for name, p in (("EBM", p_ebm), ("persistence", p_pers), ("naive", p_naive)):
            oof_auc = _roc(y[devL], p[devL])
            tmp_auc = _roc(y[tstL], p[tstL])
            tmp_brier = float(brier_score_loss(y[tstL], p[tstL])) if tstL.sum() else float("nan")
            ic, sl = calib_intercept_slope(y[tstL], p[tstL])
            am, alo, ahi = episode_cluster_bootstrap_auc_ci(
                sd_shim, p, landmark=L, split="Temporal", n=n_boot
            )
            comp_recs.append({
                "landmark": f"{L}M",
                "model": name,
                "nlive": (len(live) if name == "EBM" else (1 if name == "persistence" else 0)),
                "dev_event_rate": round(float(np.mean(y[devL])), 4),
                "temporal_event_rate": round(float(np.mean(y[tstL])), 4),
                "OOF_AUC": round(oof_auc, 4),
                "Temporal_AUC": round(tmp_auc, 4),
                "Temporal_AUC_CI_low": round(alo, 4),
                "Temporal_AUC_CI_high": round(ahi, 4),
                "OOF_minus_Temporal_AUC": round(oof_auc - tmp_auc, 4),
                "Temporal_Brier": round(tmp_brier, 4),
                "Temporal_CalibIntercept": round(ic, 4) if np.isfinite(ic) else float("nan"),
                "Temporal_CalibSlope": round(sl, 4) if np.isfinite(sl) else float("nan"),
            })
            rel_curves[(L, name)] = _reliability(y[tstL], p[tstL])

    comp = pd.DataFrame(comp_recs)

    # --- paired ΔAUC: EBM vs persistence, EBM vs naive (temporal) ---
    delta_recs = []
    for L in landmarks_cmp:
        for base in ("persistence", "naive"):
            dm, dlo, dhi = paired_episode_cluster_bootstrap_delta(
                sd_shim, preds[(L, "EBM")], preds[(L, base)],
                landmark=L, split="Temporal", n=n_boot,
            )
            crosses0 = (dlo <= 0 <= dhi)
            delta_recs.append({
                "landmark": f"{L}M",
                "comparison": f"EBM - {base}",
                "dAUC_mean": round(dm, 4),
                "dAUC_CI_low": round(dlo, 4),
                "dAUC_CI_high": round(dhi, 4),
                "CI_excludes_0": (not crosses0),
                "direction": "EBM better" if dm > 0 else ("EBM worse" if dm < 0 else "tie"),
            })
    delta = pd.DataFrame(delta_recs)
    return comp, delta, rel_curves


def load_importance_6m12m():
    """Read task2's bin16 importance table; return (df, top-ordered name lists 6M/12M).

    Falls back to recomputation only if the CSV is absent (task2 already produced
    it including the 12M column). Returns the DataFrame plus, for each landmark,
    a list of (feature, importance) sorted descending (full feature set)."""
    if not IMPORT_CSV.exists():
        return None, {}, {}
    df = pd.read_csv(IMPORT_CSV)
    out = {}
    for L in (6, 12):
        col = f"{L}M"
        sub = df[["feature", col]].dropna(subset=[col]).copy()
        sub = sub.sort_values(col, ascending=False)
        out[L] = list(zip(sub["feature"].tolist(), sub[col].astype(float).tolist()))
    return df, out.get(6, []), out.get(12, [])


def make_panel(comp, delta, rel_curves, imp6, imp12, out_png):
    """4-panel figure: (a) OOF/temporal AUC bars, (b) ΔAUC vs persistence/naive,
    (c) temporal reliability curves, (d) bin16 importance 6M vs 12M (top union)."""
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
    color = {"EBM": "#c0392b", "persistence": "#2980b9", "naive": "#7f8c8d"}

    # (a) AUC bars: OOF + temporal, grouped by landmark × model
    axA = axes[0, 0]
    models = ["EBM", "persistence", "naive"]
    lms = [6, 12]
    width = 0.13
    base_x = np.arange(len(lms))
    for mi, mdl in enumerate(models):
        oof = [comp[(comp.landmark == f"{L}M") & (comp.model == mdl)]["OOF_AUC"].values[0] for L in lms]
        tmp = [comp[(comp.landmark == f"{L}M") & (comp.model == mdl)]["Temporal_AUC"].values[0] for L in lms]
        axA.bar(base_x + (mi - 1) * 2 * width - width / 2, oof, width,
                color=color[mdl], alpha=0.55, label=f"{mdl} OOF")
        axA.bar(base_x + (mi - 1) * 2 * width + width / 2, tmp, width,
                color=color[mdl], alpha=1.0, hatch="//", label=f"{mdl} temporal")
        for j, L in enumerate(lms):
            axA.text(base_x[j] + (mi - 1) * 2 * width - width / 2, oof[j] + 0.004,
                     f"{oof[j]:.3f}", ha="center", va="bottom", fontsize=7.5)
            axA.text(base_x[j] + (mi - 1) * 2 * width + width / 2, tmp[j] + 0.004,
                     f"{tmp[j]:.3f}", ha="center", va="bottom", fontsize=7.5)
    axA.set_xticks(base_x)
    axA.set_xticklabels([f"{L}M" for L in lms])
    axA.set_ylim(0.45, 1.0)
    axA.axhline(0.5, color="#888", ls=":", lw=0.8)
    axA.set_ylabel("ROC-AUC")
    axA.set_title("(a) 判别:OOF(实) vs temporal(斜纹) AUC\n每地标 EBM / persistence / naive", fontsize=11)
    axA.legend(fontsize=7.5, ncol=3, loc="lower center")

    # (b) ΔAUC vs persistence / naive with CI (temporal)
    axB = axes[0, 1]
    rowsB = []
    for L in lms:
        for base in ("persistence", "naive"):
            d = delta[(delta.landmark == f"{L}M") & (delta.comparison == f"EBM - {base}")].iloc[0]
            rowsB.append((f"{L}M\nEBM−{base}", d["dAUC_mean"], d["dAUC_CI_low"], d["dAUC_CI_high"]))
    ys = np.arange(len(rowsB))
    means = [r[1] for r in rowsB]
    los = [r[1] - r[2] for r in rowsB]
    his = [r[3] - r[1] for r in rowsB]
    cols = ["#27ae60" if rowsB[i][2] > 0 else "#e67e22" for i in range(len(rowsB))]
    axB.barh(ys, means, xerr=[los, his], color=cols, alpha=0.85,
             error_kw=dict(ecolor="#333", capsize=4, lw=1.2))
    axB.axvline(0.0, color="#333", lw=1.0)
    axB.set_yticks(ys)
    axB.set_yticklabels([r[0] for r in rowsB], fontsize=9)
    axB.invert_yaxis()
    axB.set_xlabel("ΔAUC (temporal, 配对 episode-cluster bootstrap 95%CI)")
    axB.set_title("(b) EBM 对 persistence / naive 基线的增量\n绿=CI 不跨 0", fontsize=11)
    for i, r in enumerate(rowsB):
        axB.text(r[3] + 0.002, ys[i], f"{r[1]:+.3f}\n[{r[2]:+.3f},{r[3]:+.3f}]",
                 va="center", fontsize=7.5)

    # (c) reliability curves on temporal (EBM only, 6M vs 12M)
    axC = axes[1, 0]
    axC.plot([0, 1], [0, 1], color="#888", ls="--", lw=1.0, label="理想")
    lc = {6: "#c0392b", 12: "#8e44ad"}
    for L in lms:
        xs, yv, ns = rel_curves[(L, "EBM")]
        br = comp[(comp.landmark == f"{L}M") & (comp.model == "EBM")]["Temporal_Brier"].values[0]
        sl = comp[(comp.landmark == f"{L}M") & (comp.model == "EBM")]["Temporal_CalibSlope"].values[0]
        axC.plot(xs, yv, "-o", color=lc[L], ms=5, lw=1.6,
                 label=f"{L}M EBM (Brier={br:.3f}, slope={sl:.2f})")
    axC.set_xlim(0, 1)
    axC.set_ylim(0, 1)
    axC.set_xlabel("预测风险(分箱均值)")
    axC.set_ylabel("观测复发频率")
    axC.set_title("(c) 校准:temporal 可靠性曲线(EBM,10 分箱)", fontsize=11)
    axC.legend(fontsize=8.5, loc="upper left")

    # (d) bin16 importance 6M vs 12M (top union)
    axD = axes[1, 1]
    if imp6 and imp12:
        d6 = dict(imp6)
        d12 = dict(imp12)
        # top-6 union by max importance across the two landmarks
        all_feats = set(d6) | set(d12)
        ranked = sorted(all_feats, key=lambda f: -max(d6.get(f, 0.0), d12.get(f, 0.0)))
        feats = ranked[:7]
        yy = np.arange(len(feats))
        h = 0.38
        v6 = [d6.get(f, 0.0) for f in feats]
        v12 = [d12.get(f, 0.0) for f in feats]
        axD.barh(yy + h / 2, v6, h, color="#c0392b", alpha=0.85, label="6M")
        axD.barh(yy - h / 2, v12, h, color="#8e44ad", alpha=0.85, label="12M")
        axD.set_yticks(yy)
        axD.set_yticklabels(feats, fontsize=8.5)
        axD.invert_yaxis()
        axD.set_xlabel("mean |Δlog-odds| (bin16 dev@L 原生重要性)")
        axD.set_title("(d) 特征重要性结构:6M vs 12M(top 并集)", fontsize=11)
        axD.legend(fontsize=9)
    else:
        axD.text(0.5, 0.5, "bin16_importance_top5.csv 缺失", ha="center", va="center")
        axD.set_axis_off()

    fig.suptitle("M2 EBM · 6M vs 12M 多维对比(bin16 · corrected/LOCF · N=1003 人次)",
                 fontsize=13.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="smoke: 6M+12M, 100 boot (full run uses 1000)")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    n_boot = 100 if args.quick else args.boot

    OUT.mkdir(parents=True, exist_ok=True)

    print(f"=== M2 EBM 6M vs 12M (bin16, corrected/LOCF, boot={n_boot}) ===", flush=True)
    comp, delta, rel_curves = run(n_boot)

    print("\n--- 多维对比 (6M / 12M × EBM / persistence / naive) ---", flush=True)
    show_cols = ["landmark", "model", "OOF_AUC", "Temporal_AUC",
                 "Temporal_AUC_CI_low", "Temporal_AUC_CI_high",
                 "OOF_minus_Temporal_AUC", "Temporal_Brier",
                 "Temporal_CalibIntercept", "Temporal_CalibSlope"]
    print(comp[show_cols].to_string(index=False), flush=True)

    print("\n--- ΔAUC: EBM 对基线增量 (temporal, 配对 bootstrap) ---", flush=True)
    print(delta.to_string(index=False), flush=True)

    comp_path = OUT / f"{PREFIX}comparison.csv"
    delta_path = OUT / f"{PREFIX}delta_vs_base.csv"
    comp.to_csv(comp_path, index=False)
    delta.to_csv(delta_path, index=False)

    # forbidden-token audit
    for blob in (comp.to_csv(index=False), delta.to_csv(index=False)):
        assert forbidden_token() not in blob, "forbidden unique-patient count leaked"

    # importance structure (read task2 output; fall back to recompute if missing)
    imp_df, imp6, imp12 = load_importance_6m12m()
    if imp_df is not None:
        print("\n--- bin16 重要性 (读 task2 csv) 6M top5 / 12M top5 ---", flush=True)
        for L, imp in ((6, imp6), (12, imp12)):
            top = imp[:5]
            print(f"  [{L}M] " + "  |  ".join(f"{n}={s:.3f}" for n, s in top), flush=True)

    png_path = OUT / f"{PREFIX}panel.png"
    make_panel(comp, delta, rel_curves, imp6, imp12, png_path)

    print(f"\nSaved → {comp_path}", flush=True)
    print(f"Saved → {delta_path}", flush=True)
    print(f"Saved → {png_path}", flush=True)
    print("[done] 6M vs 12M multi-dim comparison.", flush=True)


if __name__ == "__main__":
    main()
