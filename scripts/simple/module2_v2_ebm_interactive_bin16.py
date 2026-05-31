#!/usr/bin/env python
"""M2 · v2 — EBM 交互式可视化(plotly 自包含单页,drill-down 增强版)·**bin16 口径**。

本版与 `module2_v2_ebm_interactive.py` 唯一差异:**fit EBM 时设
`max_interaction_bins=16`**(原脚本走默认 62 bin)。目的:bin16 下交互 2D 网格
16×16 不再稀疏(occupancy ~85–90%),交互项 2D 查表此时可信(对比默认 62 的
「全外推」)。其余口径完全一致——corrected 真值 + LOCF(激素延续),地标 1/3/6/12,
每地标一个 EBM,dev 5 折 StratifiedGroupKFold OOF + 时间外由 final dev-fit 预测;
AUC 报时间外。

每地标(1/3/6/12)一个 figure:
  · 左 = 原生重要性条形(全局一览)
  · 右 = 「形状函数放大镜」:下拉选任一特征 → 单独大图 + x 轴滑块拖动放大任意段
12M 交互探索:下拉选交互项 → 2D 查表热力图,悬停看每格的 g + 落在该格的病人数/事件率
            (12M 自动选出「FT3,FT4 综合水平 × 综合变化速度」——momentum 在 level 之上的增量)
6M / 12M 单病人 waterfall:预测分解(每特征 ± 贡献)

注:EBM 在 numpy array 上 fit,explain_global() 返回占位名 feature_NNNN,按 live 列序
解析回真名(_resolve);索引查 shape 仍用占位名,显示用真名。

OOF/temporal fit 逻辑 = 本地 `ebm_oof_temporal_bins`(复制 ebm_oof_and_temporal 的
split 协议:dev 5 折 StratifiedGroupKFold seed=CV_SEED OOF / temporal final dev-fit),
仅额外传 `max_interaction_bins=16`(参考 module2_v2_ebm_bins_sweep.py;不改 ebm_oof.py)。

输出 results/module2_v2_vertical/Module2v2_EBM_interactive_bin16.html (self-contained)
**新文件,不覆盖原 62-bin 版 Module2v2_EBM_interactive.html。**
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from scripts.simple.module2_v2_shared import CV_SEED, PY_SEED
from scripts.simple.module2_v2_impute_experiment import build_rows_for_method
from scripts.simple.module2_v2_b4_ebm_axes import FEATS, build_feats_at_L
from scripts.simple.module2_v2_b4_ebm_oof import assert_no_future_feature

# corrected + LOCF 口径,地标 1/3/6/12(0M 由 M1 承担;12M 新增)
LANDMARKS = (1, 3, 6, 12)
# 本版交互口径:max_interaction_bins=16(默认 62 → 16,使交互 2D 网格占用充分)
MAX_INTERACTION_BINS = 16
INTERACTIONS = 5  # 与主线 GA2M 一致
OUT = ROOT / "results" / "module2_v2_vertical" / "Module2v2_EBM_interactive_bin16.html"
DISP = {"ThyroidW": "甲状腺重量", "TRAb": "TRAb", "TGAb": "TGAb", "TPOAb": "TPOAb", "Sex": "性别",
        "FT4_0M": "FT4(0M)", "TSH_0M": "TSH(0M)", "log1p_DiseaseDuration_Months_Aug": "病程(log,月)",
        "Uptake24h": "24h 摄碘率", "HalfLife": "碘半衰期", "TSH_current": "当期 TSH", "TSH_velocity": "TSH 变化速度",
        "Hormone_load": "FT3,FT4 综合水平", "T3T4_balance": "FT3,FT4 落差",
        "Velocity_load": "FT3,FT4 综合变化速度", "Velocity_balance": "FT3,FT4 速度落差"}
RED = "#a23b3b"; BLUE = "#2a6f97"; POS = "#c0504d"; NEG = "#4a7ba6"
FONT = "Arial Unicode MS, PingFang SC, sans-serif"


def ebm_oof_temporal_bins(rows, y, lm, is_dev, L, *, max_interaction_bins=MAX_INTERACTION_BINS,
                          interactions=INTERACTIONS, seed=PY_SEED):
    """复制 ebm_oof_and_temporal 的 split 逻辑,额外传 max_interaction_bins。

    dev → 5 折 StratifiedGroupKFold(by episode_id, seed=CV_SEED) OOF;
    temporal → final dev-fit EBM 预测。返回 (pred 全长, final_ebm, live 列序)。
    与原函数唯一差异 = EBM 多了 `max_interaction_bins=B`(本版 16)。不修改 ebm_oof.py。
    """
    lm = np.asarray(lm)
    is_dev = np.asarray(is_dev)
    y = np.asarray(y)
    devL = is_dev & (lm == L)
    tstL = (~is_dev) & (lm == L)

    feat_all = build_feats_at_L(rows, devL)  # 轴在 dev@L z-fit,应用到全行
    assert_no_future_feature(feat_all, L)    # time-safety gate(沿用)

    Xtr_full = feat_all.loc[devL]
    live = [c for c in feat_all.columns if Xtr_full[c].std() > 1e-9]
    feat_live = feat_all[live]

    Xd = feat_live.loc[devL].values
    yd = y[devL]
    epd = rows["episode_id"].values[devL]

    pred = np.zeros(len(rows), dtype=float)
    dev_idx = np.where(devL)[0]
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    for tr, va in skf.split(Xd, yd, groups=epd):
        ebm = ExplainableBoostingClassifier(
            random_state=seed, interactions=interactions,
            max_interaction_bins=max_interaction_bins,
        )
        ebm.fit(Xd[tr], yd[tr])
        pred[dev_idx[va]] = ebm.predict_proba(Xd[va])[:, 1]

    final_ebm = ExplainableBoostingClassifier(
        random_state=seed, interactions=interactions,
        max_interaction_bins=max_interaction_bins,
    )
    final_ebm.fit(Xd, yd)
    if tstL.any():
        pred[tstL] = final_ebm.predict_proba(feat_live.loc[tstL].values)[:, 1]
    return pred, final_ebm, live


def _resolve(term: str, live) -> str:
    """占位名 feature_NNNN → 真名(按 live 列序);交互项逐侧解析后 ' & ' 连接。

    ebm_oof_temporal_bins 在 numpy array 上 fit,故 explain_global() / term_names_
    用占位名 feature_NNNN。index N 即 live[N]。返回真名供 disp / DISP 查表。
    """
    def one(tok: str) -> str:
        tok = tok.strip()
        if tok.startswith("feature_"):
            try:
                return live[int(tok.split("_")[1])]
            except (ValueError, IndexError):
                return tok
        return tok
    return " & ".join(one(p) for p in term.split(" & ")) if " & " in term else one(term)


def disp(t):
    return " × ".join(DISP.get(p, p) for p in t.split(" & ")) if " & " in t else DISP.get(t, t)


def shape_traces(g, ebm, term_n, real_n, visible):
    """该 univariate 形状的置信带 + step 线(3 个 trace)。term_n=占位名(索引),real_n=真名(显示)。"""
    dd = g.data(ebm.term_names_.index(term_n))
    edges = np.asarray(dd["names"], float); sc = np.asarray(dd["scores"], float)
    lo = np.asarray(dd.get("lower_bounds", sc), float); hi = np.asarray(dd.get("upper_bounds", sc), float)
    mid = (edges[:-1] + edges[1:]) / 2 if len(edges) == len(sc) + 1 else edges[:len(sc)]
    nm = disp(real_n)
    return [
        go.Scatter(x=mid, y=hi, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False, visible=visible),
        go.Scatter(x=mid, y=lo, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(162,59,59,0.12)",
                   hoverinfo="skip", showlegend=False, visible=visible),
        go.Scatter(x=mid, y=sc, mode="lines", line=dict(color=RED, width=2.4, shape="hv"), showlegend=False,
                   visible=visible, name=nm,
                   hovertemplate=f"{nm}=%{{x:.3g}}<br>log-odds 贡献=%{{y:+.2f}}<extra></extra>"),
    ]


def landmark_browser(L, ebm, live, auc):
    """左=重要性条形;右=下拉选特征的形状放大镜(带 x 轴滑块)。"""
    g = ebm.explain_global()
    # (占位名, 真名, 分数),按分数降序
    terms = sorted(((n, _resolve(n, live), s) for n, s in zip(g.data()["names"], g.data()["scores"])),
                   key=lambda t: -t[2])
    top8 = terms[:8]
    univ = [(n, rn) for n, rn, _ in terms if " & " not in n]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.34, 0.66],
                        subplot_titles=["原生重要性 top-8(mean |Δlog-odds|)",
                                        "形状放大镜:下拉选特征 · 拖下方滑块放大任意段"],
                        horizontal_spacing=0.11)
    # col1: importance bar (trace 0, 永远可见);交互项标 ✕
    names8 = [disp(rn) + ("  ✕" if " & " in n else "") for n, rn, _ in top8][::-1]
    vals8 = [s for _, _, s in top8][::-1]
    fig.add_trace(go.Bar(x=vals8, y=names8, orientation="h", marker_color=BLUE,
                         hovertemplate="%{y}<br>重要性=%{x:.3f}<extra></extra>", showlegend=False), row=1, col=1)
    # col2: 每个 univariate 特征 3 个 trace,默认仅第一个可见
    buttons = []
    for fi, (term_n, real_n) in enumerate(univ):
        for tr in shape_traces(g, ebm, term_n, real_n, visible=(fi == 0)):
            fig.add_trace(tr, row=1, col=2)
    for fi, (term_n, real_n) in enumerate(univ):
        vis = [True] + [False] * (len(univ) * 3)
        for k in range(3):
            vis[1 + fi * 3 + k] = True
        buttons.append(dict(label=disp(real_n), method="update",
                            args=[{"visible": vis}, {"xaxis2.autorange": True}]))
    fig.add_hline(y=0, line=dict(color="#bbb", width=1, dash="dot"), row=1, col=2)
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06), row=1, col=2)
    fig.update_yaxes(title_text="log-odds 贡献", row=1, col=2)
    fig.update_layout(height=500, margin=dict(l=10, r=10, t=92, b=10),
                      font=dict(family=FONT, size=11),
                      updatemenus=[dict(buttons=buttons, x=1.0, xanchor="right", y=1.30, yanchor="top",
                                        direction="down", showactive=True, bgcolor="#eef2f6")],
                      title=dict(text=f"<b>{L}M 地标</b> · 时间外 AUC {auc:.3f} · 形状=该指标对 log-odds 的贡献（局部 OR = exp(Δy)）",
                                 x=0.5, font=dict(size=14)))
    fig.update_annotations(font_size=11)
    return fig


def _gval_at(sc, le, re, x, y):
    i = int(np.clip(np.searchsorted(le, x) - 1, 0, sc.shape[0] - 1))
    j = int(np.clip(np.searchsorted(re, y) - 1, 0, sc.shape[1] - 1))
    return sc[i, j]


def interaction_explore(L, ebm, live, Xtr, ytr):
    """下拉选交互项 → 2D 查表热力图;悬停看每格 g + 该格病人数/事件率。

    Xtr 列为真名(build_feats_at_L 输出);交互项 term 用占位名 → 解析回真名取列。
    """
    g = ebm.explain_global()
    overall = {n: s for n, s in zip(g.data()["names"], g.data()["scores"])}
    pairs = [(i, n) for i, n in enumerate(ebm.term_names_) if " & " in n]

    def is_cont(c):
        return c in Xtr.columns and Xtr[c].nunique() > 6
    # 解析真名,仅保留两侧均为连续变量者
    rpairs = []
    for i, n in pairs:
        a_r, b_r = _resolve(n, live).split(" & ")
        if is_cont(a_r) and is_cont(b_r):
            rpairs.append((i, n, a_r, b_r))
    rpairs = sorted(rpairs, key=lambda t: -overall.get(t[1], 0))[:3]
    if not rpairs:
        return None
    N = 12
    fig = go.Figure()
    buttons = []
    for pi, (idx, name, a_n, b_n) in enumerate(rpairs):
        d = g.data(idx); sc = np.asarray(d["scores"], float)
        le = np.asarray(d["left_names"], float); re = np.asarray(d["right_names"], float)
        Av = Xtr[a_n].values.astype(float); Bv = Xtr[b_n].values.astype(float)
        ax = np.nanpercentile(Av, np.linspace(0, 100, N + 1)); ay = np.nanpercentile(Bv, np.linspace(0, 100, N + 1))
        ax = np.unique(ax); ay = np.unique(ay)
        xc = (ax[:-1] + ax[1:]) / 2; yc = (ay[:-1] + ay[1:]) / 2
        Z = np.array([[_gval_at(sc, le, re, xi, yj) for xi in xc] for yj in yc])  # (ny, nx)
        Hn, _, _ = np.histogram2d(Av, Bv, bins=[ax, ay])
        Hy, _, _ = np.histogram2d(Av[ytr == 1], Bv[ytr == 1], bins=[ax, ay])
        Cer = np.divide(Hy, Hn, out=np.full_like(Hn, np.nan), where=Hn > 0) * 100.0
        cd = np.dstack([Hn.T, Cer.T])  # (ny, nx, 2)
        vmax = float(np.nanmax(np.abs(Z))) or 1.0
        fig.add_trace(go.Heatmap(z=Z, x=xc, y=yc, customdata=cd, visible=(pi == 0),
                                 colorscale="RdBu_r", zmid=0, zmin=-vmax, zmax=vmax,
                                 colorbar=dict(title="g(log-odds)", thickness=14),
                                 hovertemplate=(f"{disp(a_n)}≈%{{x:.3g}}<br>{disp(b_n)}≈%{{y:.3g}}"
                                                "<br>g=%{z:+.2f}<br>该格病人 n=%{customdata[0]:.0f} · 事件率=%{customdata[1]:.0f}%<extra></extra>")))
        vis = [j == pi for j in range(len(rpairs))]
        buttons.append(dict(label=f"{disp(a_n)} × {disp(b_n)}", method="update",
                            args=[{"visible": vis},
                                  {"xaxis.title.text": disp(a_n), "yaxis.title.text": disp(b_n)}]))
    a0, b0 = rpairs[0][2], rpairs[0][3]
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=92, b=10), font=dict(family=FONT, size=11),
                      xaxis_title=disp(a0), yaxis_title=disp(b0),
                      updatemenus=[dict(buttons=buttons, x=1.0, xanchor="right", y=1.16, yanchor="top",
                                        direction="down", showactive=True, bgcolor="#eef2f6")],
                      title=dict(text=f"<b>{L}M 交互项 g(x_i, x_j) 探索</b>(max_interaction_bins=16) · 下拉换交互项 · 悬停看每格 g + 落在该格的病人数/事件率",
                                 x=0.5, font=dict(size=14)))
    return fig


def waterfall(L, ebm, live, Xte, yte, pe):
    hi_i, lo_i = int(np.argmax(pe)), int(np.argmin(pe))
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[f"高危例(预测 {pe[hi_i]:.2f}，真实 {int(yte[hi_i])})",
                                        f"低危例(预测 {pe[lo_i]:.2f}，真实 {int(yte[lo_i])})"],
                        horizontal_spacing=0.16)
    loc = ebm.explain_local(Xte.values, yte)
    for k, idx in enumerate([hi_i, lo_i]):
        d = loc.data(idx)
        ps = [(disp(_resolve(str(n), live)), float(s)) for n, s in zip(d["names"], d["scores"])
              if " & " not in str(n)]
        ps = sorted(ps, key=lambda t: abs(t[1]))[-9:]
        ys = [p[0] for p in ps]; xs = [p[1] for p in ps]
        fig.add_trace(go.Bar(x=xs, y=ys, orientation="h", marker_color=[POS if v >= 0 else NEG for v in xs],
                             hovertemplate="%{y}<br>对该病人 log-odds 贡献=%{x:+.2f}<extra></extra>",
                             showlegend=False), row=1, col=k + 1)
        fig.add_vline(x=0, line=dict(color="#888", width=1), row=1, col=k + 1)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=54, b=10), font=dict(family=FONT, size=11),
                      title=dict(text=f"<b>{L}M 单病人预测分解(waterfall)</b> · 每条 = 该指标对这位病人 log-odds 的 ± 贡献,相加 = 总 logit",
                                 x=0.5, font=dict(size=14)))
    fig.update_annotations(font_size=12)
    return fig


HEAD = """<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>M2·v2 EBM 交互式可视化(bin16)</title>
<style>
 body{font-family:"Arial Unicode MS","PingFang SC",sans-serif;max-width:1180px;margin:24px auto;padding:0 18px;color:#222;line-height:1.6}
 h1{font-size:22px} h2{font-size:18px;border-left:5px solid #2a6f97;padding-left:10px;margin-top:34px}
 .lead{background:#f5f8fb;border:1px solid #dce6ef;border-radius:8px;padding:14px 18px;font-size:14px}
 code{background:#eef2f6;padding:1px 5px;border-radius:4px}
 .note{color:#666;font-size:13px}
 .bins{background:#eef7ee;border:1px solid #cfe6cf;border-radius:6px;padding:8px 12px;margin-top:8px;font-size:13px}
</style></head><body>
<h1>放射性碘治疗后 Graves 病早期复发 · EBM 玻璃盒 <span style="color:#2a6f97">交互式</span>可视化 <span style="color:#3a7a3a">(bin16 口径)</span></h1>
<div class="lead">
<b>怎么玩这页(给临床合作者）</b><br>
EBM 把每个指标对复发风险的影响写成一条 <b>形状函数</b> <code>f(x)</code>(纵轴 = 对 <b>log-odds</b> 的贡献);预测 = 各指标贡献 + 交互项之和。<br>
• <b>下拉选特征</b> → 右侧大图单独看它;<b>拖下方滑块</b>(或在图上框选)放大任意一段细看;双击复位。<br>
• 曲线两点高度差 Δ 就是这段的对数 OR,<b>局部 OR = exp(Δ)</b>——斜率随位置变化处即“阈值/拐点”。<br>
• <b>交互探索(12M)</b>:下拉换交互项,鼠标悬停任一格 → 看该格的 g 值 + <b>真实落在该格的病人数与事件率</b>;12M 自动选出「<b>当期甲功水平 × 其变化速度</b>」——同样的当期水平,“仍在上升”比“已回落”风险更高(momentum 在 level 之上的增量)。<br>
• <b>waterfall(6M / 12M)</b>:单个病人的“可加账单”,每条 = 该指标对他 log-odds 的 ± 贡献。<br>
<div class="bins"><b>本版交互口径:<code>max_interaction_bins=16</code></b>。默认 62×62 交互网格对 1003 人次过细 → 多数格无人落入 → 交互查表近乎「全外推」;<b>降到 16×16 后网格占用充分(occupancy ~85–90%)</b>,交互项 2D 查表此时<b>可读、可信</b>(每格背后有真实病人支撑)。单变量形状函数与判别口径不受影响。</div>
<b>数据口径</b>:corrected 真值(<code>Current_Time</code> 列)+ LOCF(激素延续),地标 <b>1/3/6/12</b>(0M 由 M1 承担);dev 5 折 StratifiedGroupKFold OOF,时间外集报告 AUC(EBM final dev-fit 预测)。<b>重要性叙事:当期甲功水平(FT3,FT4 综合水平)自 3M 接管首位并主导 6M / 12M;激素动量退居 6M 第三、12M 第五,以「水平 × 速度」交互形式提供增量。</b><br>
<span class="note">分析单元 = 治疗疗程(N=1003 人次);置信带来自 EBM 的 bagging。默认 62-bin 交互版见 Module2v2_EBM_interactive(.html);静态论文版见 Module2v2_EBM_paper(.html),原理见 Module2v2_EBM_讲解(.html)。</span>
</div>
"""
TAIL = "</body></html>"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="smoke: 仅 6M 单地标,验证管线 + bin16 交互 occupancy,不写最终 html")
    args = ap.parse_args()

    landmarks = (6,) if args.quick else LANDMARKS

    # corrected 真值(real 6M/12M)+ LOCF 插值,严格 time-safe
    rows = build_rows_for_method("locf", LANDMARKS)
    assert rows["episode_id"].nunique() == 1003, "analysis unit must be 1003 人次"
    y = rows["Y_24M_NHRH"].values
    lm = rows["landmark"].values
    is_dev = (rows["Split"] == "Development").values

    blocks = []
    inc_done = False
    for L in landmarks:
        devL = is_dev & (lm == L); tstL = (~is_dev) & (lm == L)
        # EBM OOF(dev) + temporal(final dev-fit)·max_interaction_bins=16
        pred, ebm, live = ebm_oof_temporal_bins(rows, y, lm, is_dev, L)
        pe = pred[tstL]                       # 时间外预测
        auc = roc_auc_score(y[tstL], pe)
        feat = build_feats_at_L(rows, devL)   # 真名特征矩阵(轴已在 dev@L z-fit)
        Xtr = feat.loc[devL, live]
        Xte = feat.loc[tstL, live]

        figs = [("形状放大镜 + 重要性", landmark_browser(L, ebm, live, auc))]
        if L == 12:
            ie = interaction_explore(L, ebm, live, Xtr, np.asarray(y[devL]))
            if ie is not None:
                figs.append(("交互项 2D 查表探索(bin16)", ie))
        if L in (6, 12):
            figs.append(("单病人 waterfall", waterfall(L, ebm, live, Xte.reset_index(drop=True),
                                                       np.asarray(y[tstL]), pe)))

        for sub, fg in figs:
            inc = "inline" if not inc_done else False
            inc_done = True
            blocks.append(f"<h2>{L}M — {sub}</h2>")
            blocks.append(fg.to_html(full_html=False, include_plotlyjs=inc, config={"displaylogo": False}))
        print(f"{L}M done, AUC={auc:.3f}")

        # smoke: 顺带报 bin16 交互 occupancy(每地标取 top 交互项)
        if args.quick:
            g = ebm.explain_global()
            for i, n in enumerate(ebm.term_names_):
                if " & " not in n:
                    continue
                real = _resolve(n, live)
                a_n, b_n = real.split(" & ")
                if a_n not in feat.columns or b_n not in feat.columns:
                    continue
                d = g.data(i); sc = np.asarray(d["scores"], float)
                le = np.asarray(d["left_names"], float); re = np.asarray(d["right_names"], float)
                Av = feat.loc[devL, a_n].values.astype(float)
                Bv = feat.loc[devL, b_n].values.astype(float)
                # 用 EBM 的实际 bin 边界算占用格比例
                Hn, _, _ = np.histogram2d(Av, Bv, bins=[le, re])
                ncells = (sc.shape[0]) * (sc.shape[1])
                occ = 100.0 * np.sum(Hn > 0) / max(ncells, 1)
                print(f"  [bin16 occ] {disp(real)}: grid {sc.shape[0]}×{sc.shape[1]} "
                      f"occupancy={occ:.1f}% (importance={g.data()['scores'][g.data()['names'].index(n)]:.3f})")

    if args.quick:
        print("\n[quick] smoke 完成,未写最终 html。")
        return

    OUT.write_text(HEAD + "\n".join(blocks) + TAIL, encoding="utf-8")
    print(f"\n写出 {OUT.name} · {OUT.stat().st_size/1024:.0f} KB · 自包含")


if __name__ == "__main__":
    main()
