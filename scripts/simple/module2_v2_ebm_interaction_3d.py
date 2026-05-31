#!/usr/bin/env python
"""EBM 交互项 g(x_i,x_j) 的 3D 可旋转曲面(plotly,自包含)。

不是降维——交互项本就 2 输入→1 输出(log-odds),天然 3D(x,y 两特征轴 + z=贡献高度)。
热力图是俯视投影;此为立体版,叠训练样本黑点显示「哪里有数据 vs 外推假峰」。
corrected+LOCF 口径。两个 scene:12M「水平×速度」(核心) + 6M「TPOAb×落差」(教程例)。
输出 results/module2_v2_vertical/Module2v2_EBM_interaction_3D.html
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys, warnings
from pathlib import Path
warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scripts.simple.module2_v2_impute_experiment import build_rows_for_method
from scripts.simple.module2_v2_b4_ebm_oof import ebm_oof_and_temporal
from scripts.simple.module2_v2_b4_ebm_axes import build_feats_at_L

DISP = {"ThyroidW": "甲状腺重量", "TRAb": "TRAb", "TGAb": "TGAb", "TPOAb": "TPOAb", "Sex": "性别",
        "FT4_0M": "FT4(0M)", "TSH_0M": "TSH(0M)", "log1p_DiseaseDuration_Months_Aug": "病程(log,月)",
        "Uptake24h": "24h 摄碘率", "HalfLife": "碘半衰期", "TSH_current": "当期 TSH", "TSH_velocity": "TSH 变化速度",
        "Hormone_load": "FT3,FT4 综合水平", "T3T4_balance": "FT3,FT4 落差",
        "Velocity_load": "FT3,FT4 综合变化速度", "Velocity_balance": "FT3,FT4 速度落差"}
def disp(t): return DISP.get(t, t)
FONT = "Arial Unicode MS, PingFang SC, sans-serif"
OUT = ROOT / "results" / "module2_v2_vertical" / "Module2v2_EBM_interaction_3D.html"


def _resolve(term, live):
    def one(tok):
        tok = tok.strip()
        if tok.startswith("feature_"):
            try: return live[int(tok.split("_")[1])]
            except (ValueError, IndexError): return tok
        return tok
    return " & ".join(one(p) for p in term.split(" & ")) if " & " in term else one(term)


def build(L, prefset):
    rows = build_rows_for_method("locf", (L,))
    y = rows["Y_24M_NHRH"].values; lm = rows["landmark"].values
    is_dev = (rows["Split"] == "Development").values; devL = is_dev & (lm == L)
    _p, ebm, live = ebm_oof_and_temporal(rows, y, lm, is_dev, L)
    feat = build_feats_at_L(rows, devL); Xtr = feat.loc[devL, live]
    g = ebm.explain_global(); overall = dict(zip(g.data()["names"], g.data()["scores"]))
    inter = [(i, n, _resolve(n, live)) for i, n in enumerate(ebm.term_names_) if " & " in n]
    inter = sorted(inter, key=lambda t: -overall.get(t[1], 0))
    chosen = next((t for t in inter if _resolve(t[1], live) in prefset), inter[0])
    idx, name, rname = chosen; a, b = rname.split(" & ")
    d = g.data(idx); sc = np.asarray(d["scores"], float)
    le = np.asarray(d["left_names"], float); re = np.asarray(d["right_names"], float)
    le = le[np.isfinite(le)]; re = re[np.isfinite(re)]
    xc = (le[:-1] + le[1:]) / 2; yc = (re[:-1] + re[1:]) / 2
    Av = Xtr[a].values.astype(float); Bv = Xtr[b].values.astype(float)
    def gz(x, yv):
        i = int(np.clip(np.searchsorted(le, x) - 1, 0, sc.shape[0] - 1))
        j = int(np.clip(np.searchsorted(re, yv) - 1, 0, sc.shape[1] - 1)); return sc[i, j]
    zs = np.array([gz(x, yv) for x, yv in zip(Av, Bv)])
    # 视图裁到 1–99 分位(去极端外推边)
    xlo, xhi = np.nanpercentile(Av, 1), np.nanpercentile(Av, 99)
    ylo, yhi = np.nanpercentile(Bv, 1), np.nanpercentile(Bv, 99)
    return dict(a=disp(a), b=disp(b), xc=xc, yc=yc, sc=sc, Av=Av, Bv=Bv, zs=zs,
                imp=overall.get(name, 0.0), xr=[xlo, xhi], yr=[ylo, yhi])


def main():
    specs = [[{"type": "surface"}, {"type": "surface"}]]
    fig = make_subplots(rows=1, cols=2, specs=specs, horizontal_spacing=0.04,
                        subplot_titles=["6M:FT3,FT4 综合水平 × 综合变化速度(momentum,bin16 自动#1)",
                                        "12M:当期 TSH × FT3,FT4 综合水平(bin16 自动#1)"])
    for col, (L, pref) in enumerate([(6, {"Hormone_load & Velocity_load", "Velocity_load & Hormone_load"}),
                                     (12, {"TSH_current & Hormone_load", "Hormone_load & TSH_current"})], start=1):
        D = build(L, pref)
        vmax = float(np.nanmax(np.abs(D["sc"]))) or 1.0
        sid = "scene" if col == 1 else "scene2"
        fig.add_trace(go.Surface(x=D["xc"], y=D["yc"], z=D["sc"].T, colorscale="RdBu_r",
                                 cmin=-vmax, cmax=vmax, opacity=0.92, showscale=(col == 2),
                                 colorbar=dict(title="g(log-odds)", thickness=12, x=1.0),
                                 hovertemplate=f"{D['a']}=%{{x:.3g}}<br>{D['b']}=%{{y:.3g}}<br>g=%{{z:+.2f}}<extra></extra>"),
                      row=1, col=col)
        # 训练样本散点(黑),z=该样本落格的 g —— 看「哪里有数据」
        fig.add_trace(go.Scatter3d(x=D["Av"], y=D["Bv"], z=D["zs"], mode="markers",
                                   marker=dict(size=2.2, color="#111", opacity=0.45),
                                   hoverinfo="skip", showlegend=False), row=1, col=col)
        sc_layout = dict(xaxis_title=D["a"], yaxis_title=D["b"], zaxis_title="g (log-odds 贡献)",
                         xaxis=dict(range=D["xr"]), yaxis=dict(range=D["yr"]),
                         camera=dict(eye=dict(x=1.6, y=-1.5, z=0.9)))
        fig.update_layout(**{sid: sc_layout})
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=60, b=0), font=dict(family=FONT, size=11),
                      title=dict(text="<b>EBM 交互项 g 的 3D 曲面</b> · 旋转看「山脊/鞍部」;黑点=训练样本(样本聚处可信,角落翘起无点处=正则外推假峰)", x=0.5, font=dict(size=13)))
    head = ('<div style="font-family:Arial Unicode MS,PingFang SC,sans-serif;max-width:1180px;margin:16px auto;padding:0 16px">'
            '<h2>EBM 交互项 3D 曲面(可旋转)</h2>'
            '<p style="color:#555;font-size:14px">交互项 g(x_i,x_j) 本就 2 输入→1 输出(log-odds),<b>天然 3D,无需降维</b>;'
            '热力图是它的俯视投影,这里把 z 轴(贡献)抬起来看立体。<b>拖动旋转</b>、滚轮缩放。'
            '黑点=训练样本(落在该格的 g 高度):<b>样本密集处的曲面可信,角落翘起却没有黑点的「山脊」是正则化外推的假峰</b>(呼应"外推不可信")。'
            'corrected+LOCF 口径,12M 的「水平×速度」即 momentum-beyond-inertia 的交互增量。</p></div>')
    html = head + fig.to_html(full_html=True, include_plotlyjs="inline", config={"displaylogo": False})
    OUT.write_text(html, encoding="utf-8")
    print("写出", OUT.name, f"{OUT.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
