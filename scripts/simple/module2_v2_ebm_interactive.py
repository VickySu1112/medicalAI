#!/usr/bin/env python
"""M2 · v2 — EBM 交互式可视化(plotly 自包含单页)。

每地标(0/1/3/6M)一个 figure:原生重要性条形 + top-8 univariate 形状函数
(鼠标悬停读 bin 区间 + log-odds 贡献 + bagging 置信带、可框选缩放看尾部);
外加 6M 两位代表病人的预测分解 waterfall(可加账单的逐特征 ± 贡献)。

供医学合作者浏览器直接打开拨弄。输出:
results/module2_v2_vertical/Module2v2_EBM_interactive.html (self-contained)
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


def disp(t):
    return " × ".join(DISP.get(p, p) for p in t.split(" & ")) if " & " in t else DISP.get(t, t)


def shape_traces(g, ebm, feat_n):
    """该 univariate 形状的置信带 + step 线 traces(log-odds 尺度)。"""
    dd = g.data(ebm.term_names_.index(feat_n))
    edges = np.asarray(dd["names"], float); sc = np.asarray(dd["scores"], float)
    lo = np.asarray(dd.get("lower_bounds", sc), float); hi = np.asarray(dd.get("upper_bounds", sc), float)
    mid = (edges[:-1] + edges[1:]) / 2 if len(edges) == len(sc) + 1 else edges[:len(sc)]
    nm = disp(feat_n)
    t_hi = go.Scatter(x=mid, y=hi, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False)
    t_lo = go.Scatter(x=mid, y=lo, mode="lines", line=dict(width=0), fill="tonexty",
                      fillcolor="rgba(162,59,59,0.12)", hoverinfo="skip", showlegend=False)
    t_ln = go.Scatter(x=mid, y=sc, mode="lines", line=dict(color=RED, width=2, shape="hv"), showlegend=False,
                      hovertemplate=f"{nm}=%{{x:.3g}}<br>log-odds 贡献=%{{y:+.2f}}<extra></extra>")
    return [t_hi, t_lo, t_ln]


def landmark_figure(L, ebm, auc):
    g = ebm.explain_global()
    terms = sorted(zip(g.data()["names"], g.data()["scores"]), key=lambda t: -t[1])
    top8 = terms[:8]
    univ = [n for n, _ in terms if " & " not in n][:8]
    titles = ["原生重要性 top-8(mean |Δlog-odds|)"] + [disp(f) for f in univ]
    titles += [""] * (9 - len(titles))
    fig = make_subplots(rows=3, cols=3, subplot_titles=titles,
                        horizontal_spacing=0.07, vertical_spacing=0.11)
    # cell (1,1): importance bar
    names8 = [disp(n) + ("  ✕" if " & " in n else "") for n, _ in top8][::-1]
    vals8 = [s for _, s in top8][::-1]
    fig.add_trace(go.Bar(x=vals8, y=names8, orientation="h", marker_color=BLUE,
                         hovertemplate="%{y}<br>重要性=%{x:.3f}<extra></extra>", showlegend=False), row=1, col=1)
    # cells 2..9: univariate shapes
    for i, feat_n in enumerate(univ):
        r, c = (i + 1) // 3 + 1, (i + 1) % 3 + 1
        for tr in shape_traces(g, ebm, feat_n):
            fig.add_trace(tr, row=r, col=c)
        fig.add_hline(y=0, line=dict(color="#bbb", width=1, dash="dot"), row=r, col=c)
    fig.update_layout(height=820, margin=dict(l=10, r=10, t=56, b=10),
                      font=dict(family="Arial Unicode MS, PingFang SC, sans-serif", size=11),
                      title=dict(text=f"<b>{L}M 地标</b> · 时间外 AUC {auc:.3f} · 形状=该指标对 log-odds 的贡献曲线（局部 OR = exp(Δy)）",
                                 x=0.5, font=dict(size=14)))
    fig.update_annotations(font_size=11)
    return fig


def waterfall_figure(L, ebm, Xte, yte, pe):
    hi_i, lo_i = int(np.argmax(pe)), int(np.argmin(pe))
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[f"高危例(预测 {pe[hi_i]:.2f}，真实 {int(yte[hi_i])})",
                                        f"低危例(预测 {pe[lo_i]:.2f}，真实 {int(yte[lo_i])})"],
                        horizontal_spacing=0.16)
    loc = ebm.explain_local(Xte, yte)
    for k, idx in enumerate([hi_i, lo_i]):
        d = loc.data(idx)
        pairs = [(disp(n), float(s)) for n, s in zip(d["names"], d["scores"]) if " & " not in str(n)]
        pairs = sorted(pairs, key=lambda t: abs(t[1]))[-9:]
        ys = [p[0] for p in pairs]; xs = [p[1] for p in pairs]
        cols = [POS if v >= 0 else NEG for v in xs]
        fig.add_trace(go.Bar(x=xs, y=ys, orientation="h", marker_color=cols,
                             hovertemplate="%{y}<br>对该病人 log-odds 贡献=%{x:+.2f}<extra></extra>",
                             showlegend=False), row=1, col=k + 1)
        fig.add_vline(x=0, line=dict(color="#888", width=1), row=1, col=k + 1)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=54, b=10),
                      font=dict(family="Arial Unicode MS, PingFang SC, sans-serif", size=11),
                      title=dict(text=f"<b>{L}M 单病人预测分解(waterfall)</b> · 每条 = 该指标对这位病人 log-odds 的 ± 贡献，相加 = 总 logit",
                                 x=0.5, font=dict(size=14)))
    fig.update_annotations(font_size=12)
    return fig


HEAD = """<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>M2·v2 EBM 交互式可视化</title>
<style>
 body{{font-family:"Arial Unicode MS","PingFang SC",sans-serif;max-width:1180px;margin:24px auto;padding:0 18px;color:#222;line-height:1.6}}
 h1{{font-size:22px}} h2{{font-size:18px;border-left:5px solid #2a6f97;padding-left:10px;margin-top:34px}}
 .lead{{background:#f5f8fb;border:1px solid #dce6ef;border-radius:8px;padding:14px 18px;font-size:14px}}
 code{{background:#eef2f6;padding:1px 5px;border-radius:4px}}
 .note{{color:#666;font-size:13px}}
</style></head><body>
<h1>放射性碘治疗后 Graves 病早期复发 · EBM 玻璃盒 <span style="color:#2a6f97">交互式</span>可视化</h1>
<div class="lead">
<b>怎么读这页(给临床合作者）</b><br>
EBM 把每个指标对复发风险的影响写成一条 <b>形状函数</b> <code>f(x)</code>，纵轴是该指标对 <b>log-odds（对数几率）</b>的加项；
模型预测 = 各指标贡献 + 交互项之和：<code>logit(p)=β0+Σ f_j(x_j)+Σ f_jk(·)</code>。<br>
• <b>悬停</b>任一点读出精确的 log-odds 贡献；<b>框选</b>可放大看尾部；双击复位。<br>
• 曲线从 x₁ 到 x₂ 的高度差 Δ <b>就是这段的对数 OR</b>，<b>局部 OR = exp(Δ)</b>——斜率随位置变化处即“阈值/拐点”（线性 logistic 回归画不出）。<br>
• 重要性条形里带 <b>✕</b> 的是两两交互项。<br>
• 最后的 <b>waterfall</b> 把单个病人的“可加账单”摊开：每条是该指标对这位病人 log-odds 的 ± 贡献，相加即其预测 logit。<br>
<span class="note">分析单元 = 治疗疗程（N=1003 人次）；开发集 fit、时间外集报告 AUC；置信带来自 EBM 的 bagging。静态论文版见 Module2v2_EBM_paper(.html)。</span>
</div>
"""
TAIL = "</body></html>"


def main():
    sd = load_stacked(); sd = add_current_velocity(sd, "FT3")
    rows = sd.rows
    y = rows["Y_24M_NHRH"].values; lm = rows["landmark"].values
    is_dev = (rows["Split"] == "Development").values

    blocks = []
    plotly_done = False
    for L in LANDMARKS:
        devL = is_dev & (lm == L); tstL = (~is_dev) & (lm == L)
        feat = build_feats_at_L(rows, devL)
        live = [c for c in FEATS if feat.loc[devL, c].std() > 1e-9]
        Xtr, ytr = feat.loc[devL, live], y[devL]
        Xte, yte = feat.loc[tstL, live], y[tstL]
        ebm = ExplainableBoostingClassifier(random_state=PY_SEED, interactions=5).fit(Xtr, ytr)
        pe = ebm.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, pe)
        f1 = landmark_figure(L, ebm, auc)
        f2 = waterfall_figure(L, ebm, Xte.reset_index(drop=True), np.asarray(yte), pe) if L == 6 else None
        inc = "inline" if not plotly_done else False
        plotly_done = True
        blocks.append(f"<h2>{L}M 地标 — 重要性 + 形状函数</h2>")
        blocks.append(f1.to_html(full_html=False, include_plotlyjs=inc, config={"displaylogo": False}))
        if f2 is not None:
            blocks.append("<h2>6M — 单病人预测分解(waterfall)</h2>")
            blocks.append(f2.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False}))
        print(f"{L}M done, AUC={auc:.3f}, univ shapes plotted")

    OUT.write_text(HEAD + "\n".join(blocks) + TAIL, encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    print(f"\n写出 {OUT.name} · {kb:.0f} KB · 自包含")


if __name__ == "__main__":
    main()
