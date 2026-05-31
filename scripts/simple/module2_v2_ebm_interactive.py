#!/usr/bin/env python
"""M2 · v2 — EBM 交互式可视化(plotly 自包含单页,drill-down 增强版)。

每地标(0/1/3/6M)一个 figure:
  · 左 = 原生重要性条形(全局一览)
  · 右 = 「形状函数放大镜」:下拉选任一特征 → 单独大图 + x 轴滑块拖动放大任意段
0M 交互探索:下拉选交互项 → 2D 查表热力图,悬停看每格的 g + 落在该格的病人数/事件率
6M 单病人 waterfall:预测分解(每特征 ± 贡献)

输出 results/module2_v2_vertical/Module2v2_EBM_interactive.html (self-contained)
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

from scripts.simple.module2_v2_shared import LANDMARKS, load_stacked, PY_SEED
from scripts.simple.module2_v2_b4_trab_patch import add_current_velocity
from scripts.simple.module2_v2_b4_ebm_axes import build_feats_at_L, FEATS

OUT = ROOT / "results" / "module2_v2_vertical" / "Module2v2_EBM_interactive.html"
DISP = {"ThyroidW": "甲状腺重量", "TRAb": "TRAb", "TGAb": "TGAb", "TPOAb": "TPOAb", "Sex": "性别",
        "FT4_0M": "FT4(0M)", "TSH_0M": "TSH(0M)", "log1p_DiseaseDuration_Months_Aug": "病程(log,月)",
        "Uptake24h": "24h 摄碘率", "HalfLife": "碘半衰期", "TSH_current": "当期 TSH", "TSH_velocity": "TSH 变化速度",
        "Hormone_load": "FT3,FT4 综合水平", "T3T4_balance": "FT3,FT4 落差",
        "Velocity_load": "FT3,FT4 综合变化速度", "Velocity_balance": "FT3,FT4 速度落差"}
RED = "#a23b3b"; BLUE = "#2a6f97"; POS = "#c0504d"; NEG = "#4a7ba6"
FONT = "Arial Unicode MS, PingFang SC, sans-serif"


def disp(t):
    return " × ".join(DISP.get(p, p) for p in t.split(" & ")) if " & " in t else DISP.get(t, t)


def shape_traces(g, ebm, feat_n, visible):
    """该 univariate 形状的置信带 + step 线(3 个 trace)。"""
    dd = g.data(ebm.term_names_.index(feat_n))
    edges = np.asarray(dd["names"], float); sc = np.asarray(dd["scores"], float)
    lo = np.asarray(dd.get("lower_bounds", sc), float); hi = np.asarray(dd.get("upper_bounds", sc), float)
    mid = (edges[:-1] + edges[1:]) / 2 if len(edges) == len(sc) + 1 else edges[:len(sc)]
    nm = disp(feat_n)
    return [
        go.Scatter(x=mid, y=hi, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False, visible=visible),
        go.Scatter(x=mid, y=lo, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(162,59,59,0.12)",
                   hoverinfo="skip", showlegend=False, visible=visible),
        go.Scatter(x=mid, y=sc, mode="lines", line=dict(color=RED, width=2.4, shape="hv"), showlegend=False,
                   visible=visible, name=nm,
                   hovertemplate=f"{nm}=%{{x:.3g}}<br>log-odds 贡献=%{{y:+.2f}}<extra></extra>"),
    ]


def landmark_browser(L, ebm, auc):
    """左=重要性条形;右=下拉选特征的形状放大镜(带 x 轴滑块)。"""
    g = ebm.explain_global()
    terms = sorted(zip(g.data()["names"], g.data()["scores"]), key=lambda t: -t[1])
    top8 = terms[:8]
    univ = [n for n, _ in terms if " & " not in n]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.34, 0.66],
                        subplot_titles=["原生重要性 top-8(mean |Δlog-odds|)",
                                        "形状放大镜:下拉选特征 · 拖下方滑块放大任意段"],
                        horizontal_spacing=0.11)
    # col1: importance bar (trace 0, 永远可见)
    names8 = [disp(n) + ("  ✕" if " & " in n else "") for n, _ in top8][::-1]
    vals8 = [s for _, s in top8][::-1]
    fig.add_trace(go.Bar(x=vals8, y=names8, orientation="h", marker_color=BLUE,
                         hovertemplate="%{y}<br>重要性=%{x:.3f}<extra></extra>", showlegend=False), row=1, col=1)
    # col2: 每个 univariate 特征 3 个 trace,默认仅第一个可见
    ntr = 1  # 已有 bar
    buttons = []
    for fi, feat in enumerate(univ):
        for tr in shape_traces(g, ebm, feat, visible=(fi == 0)):
            fig.add_trace(tr, row=1, col=2)
    total = 1 + len(univ) * 3
    for fi, feat in enumerate(univ):
        vis = [True] + [False] * (len(univ) * 3)
        for k in range(3):
            vis[1 + fi * 3 + k] = True
        buttons.append(dict(label=disp(feat), method="update",
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


def interaction_explore(L, ebm, Xtr, ytr):
    """下拉选交互项 → 2D 查表热力图;悬停看每格 g + 该格病人数/事件率。"""
    g = ebm.explain_global()
    overall = dict(zip(g.data()["names"], g.data()["scores"]))
    pairs = [(i, n) for i, n in enumerate(ebm.term_names_) if " & " in n]

    def is_cont(c):
        return Xtr[c].nunique() > 6
    pairs = [(i, n) for i, n in pairs if all(is_cont(p) for p in n.split(" & "))]
    pairs = sorted(pairs, key=lambda t: -overall.get(t[1], 0))[:3]
    if not pairs:
        return None
    N = 12
    fig = go.Figure()
    buttons = []
    for pi, (idx, name) in enumerate(pairs):
        a_n, b_n = name.split(" & ")
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
        vis = [j == pi for j in range(len(pairs))]
        buttons.append(dict(label=f"{disp(a_n)} × {disp(b_n)}", method="update",
                            args=[{"visible": vis},
                                  {"xaxis.title.text": disp(a_n), "yaxis.title.text": disp(b_n)}]))
    a0, b0 = pairs[0][1].split(" & ")
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=92, b=10), font=dict(family=FONT, size=11),
                      xaxis_title=disp(a0), yaxis_title=disp(b0),
                      updatemenus=[dict(buttons=buttons, x=1.0, xanchor="right", y=1.16, yanchor="top",
                                        direction="down", showactive=True, bgcolor="#eef2f6")],
                      title=dict(text=f"<b>{L}M 交互项 g(x_i, x_j) 探索</b> · 下拉换交互项 · 悬停看每格 g + 落在该格的病人数/事件率",
                                 x=0.5, font=dict(size=14)))
    return fig


def waterfall(L, ebm, Xte, yte, pe):
    hi_i, lo_i = int(np.argmax(pe)), int(np.argmin(pe))
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[f"高危例(预测 {pe[hi_i]:.2f}，真实 {int(yte[hi_i])})",
                                        f"低危例(预测 {pe[lo_i]:.2f}，真实 {int(yte[lo_i])})"],
                        horizontal_spacing=0.16)
    loc = ebm.explain_local(Xte, yte)
    for k, idx in enumerate([hi_i, lo_i]):
        d = loc.data(idx)
        ps = [(disp(n), float(s)) for n, s in zip(d["names"], d["scores"]) if " & " not in str(n)]
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
<title>M2·v2 EBM 交互式可视化</title>
<style>
 body{font-family:"Arial Unicode MS","PingFang SC",sans-serif;max-width:1180px;margin:24px auto;padding:0 18px;color:#222;line-height:1.6}
 h1{font-size:22px} h2{font-size:18px;border-left:5px solid #2a6f97;padding-left:10px;margin-top:34px}
 .lead{background:#f5f8fb;border:1px solid #dce6ef;border-radius:8px;padding:14px 18px;font-size:14px}
 code{background:#eef2f6;padding:1px 5px;border-radius:4px}
 .note{color:#666;font-size:13px}
</style></head><body>
<h1>放射性碘治疗后 Graves 病早期复发 · EBM 玻璃盒 <span style="color:#2a6f97">交互式</span>可视化</h1>
<div class="lead">
<b>怎么玩这页(给临床合作者）</b><br>
EBM 把每个指标对复发风险的影响写成一条 <b>形状函数</b> <code>f(x)</code>(纵轴 = 对 <b>log-odds</b> 的贡献);预测 = 各指标贡献 + 交互项之和。<br>
• <b>下拉选特征</b> → 右侧大图单独看它;<b>拖下方滑块</b>(或在图上框选)放大任意一段细看;双击复位。<br>
• 曲线两点高度差 Δ 就是这段的对数 OR,<b>局部 OR = exp(Δ)</b>——斜率随位置变化处即“阈值/拐点”。<br>
• <b>交互探索</b>:下拉换交互项,鼠标悬停任一格 → 看该格的 g 值 + <b>真实落在该格的病人数与事件率</b>(格子越空越靠外推,别过度解读)。<br>
• <b>waterfall</b>:单个病人的“可加账单”,每条 = 该指标对他 log-odds 的 ± 贡献。<br>
<span class="note">分析单元 = 治疗疗程(N=1003 人次);开发集 fit、时间外集报告 AUC;置信带来自 EBM 的 bagging。静态论文版见 Module2v2_EBM_paper(.html),原理见 Module2v2_EBM_讲解(.html)。</span>
</div>
"""
TAIL = "</body></html>"


def main():
    sd = load_stacked(); sd = add_current_velocity(sd, "FT3")
    rows = sd.rows
    y = rows["Y_24M_NHRH"].values; lm = rows["landmark"].values
    is_dev = (rows["Split"] == "Development").values

    blocks = []
    inc_done = False
    for L in LANDMARKS:
        devL = is_dev & (lm == L); tstL = (~is_dev) & (lm == L)
        feat = build_feats_at_L(rows, devL)
        live = [c for c in FEATS if feat.loc[devL, c].std() > 1e-9]
        Xtr, ytr = feat.loc[devL, live], y[devL]
        Xte, yte = feat.loc[tstL, live], y[tstL]
        ebm = ExplainableBoostingClassifier(random_state=PY_SEED, interactions=5).fit(Xtr, ytr)
        pe = ebm.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, pe)

        figs = [("形状放大镜 + 重要性", landmark_browser(L, ebm, auc))]
        if L == 0:
            ie = interaction_explore(L, ebm, Xtr, np.asarray(ytr))
            if ie is not None:
                figs.append(("交互项 2D 查表探索", ie))
        if L == 6:
            figs.append(("单病人 waterfall", waterfall(L, ebm, Xte.reset_index(drop=True), np.asarray(yte), pe)))

        for sub, fg in figs:
            inc = "inline" if not inc_done else False
            inc_done = True
            blocks.append(f"<h2>{L}M — {sub}</h2>")
            blocks.append(fg.to_html(full_html=False, include_plotlyjs=inc, config={"displaylogo": False}))
        print(f"{L}M done, AUC={auc:.3f}")

    OUT.write_text(HEAD + "\n".join(blocks) + TAIL, encoding="utf-8")
    print(f"\n写出 {OUT.name} · {OUT.stat().st_size/1024:.0f} KB · 自包含")


if __name__ == "__main__":
    main()
