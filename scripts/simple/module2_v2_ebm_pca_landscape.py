#!/usr/bin/env python
"""M2 · v2 — EBM「模型在 PCA 嵌入空间的风险景观 / 决策面」可玩 HTML。

把高维 live 特征用 PCA 压到 2D/3D 鸟瞰整体风险分布(risk-landscape /
decision-surface over a PCA embedding)。地标 6M / 12M 各一 scene。

口径(与 module2_v2_ebm_bins_sweep.py 完全一致):
  数据 —— corrected 真值 + LOCF,`build_rows_for_method("locf",(L,))`。
  特征 —— `build_feats_at_L` / `FEATS`(正交轴),取 dev@L z-fit 后的 live 列。
  模型 —— `ExplainableBoostingClassifier(interactions=5,
            max_interaction_bins=16, random_state=PY_SEED)`。
  预测概率 —— dev = 5 折 StratifiedGroupKFold OOF(seed=CV_SEED);
            temporal = dev-final-fit EBM `predict_proba`(仅报告)。
  PCA —— live 特征(全 1003@L 行)再 StandardScaler 后
         `sklearn.decomposition.PCA(n_components=3)`,取 PC1/PC2/PC3;
         报告前 2–3 主成分解释方差比。

可玩(plotly,自包含 inline):
  2D —— PC1×PC2 散点,着色=预测复发概率(RdBu_r);dev=圆 / temporal=方;
        真实 Y=1 描黑边;hover 显示 真实Y / 预测p / PC 坐标 + top3 贡献特征
        (EBM explain_local 的 per-term log-odds 贡献绝对值前 3)。
  3D —— PC1×PC2 平面 + z=预测概率 → 立体「风险地形」,可旋转,同样着色+样本点。
  下拉切 6M / 12M。

诚实标注(页面顶部):PCA 是无监督投影,只用于鸟瞰样本流形上的整体风险分布;
  EBM 本身是加性玻璃盒、应以逐特征形状函数解读为准,此降维图仅作整体概览、
  不替代逐项解释,且不在样本流形外外推。

硬约束:1003 人次 / 无 889 / time-safe(沿用 assert_no_future_feature) /
  OOF 用于 dev 概率、temporal 仅报告 / 不写 significant / 自包含。
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import argparse
import html
import json
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from scripts.simple.module2_v2_shared import CV_SEED, PY_SEED, forbidden_token
from scripts.simple.module2_v2_impute_experiment import build_rows_for_method
from scripts.simple.module2_v2_b4_ebm_axes import FEATS, build_feats_at_L
from scripts.simple.module2_v2_b4_ebm_oof import assert_no_future_feature

LANDMARKS = (6, 12)
MAX_INTERACTION_BINS = 16
INTERACTIONS = 5
OUT = ROOT / "results" / "module2_v2_vertical"

# 中文特征显示名(hover 用),保持与正交轴语义一致。
DISP = {
    "ThyroidW": "甲状腺重量", "TRAb": "TRAb", "TGAb": "TGAb", "TPOAb": "TPOAb",
    "Sex": "性别", "FT4_0M": "基线FT4", "TSH_0M": "基线TSH",
    "log1p_DiseaseDuration_Months_Aug": "病程(log)", "Uptake24h": "24h摄碘率",
    "HalfLife": "有效半衰期", "TSH_current": "当期TSH", "TSH_velocity": "TSH速度",
    "Hormone_load": "激素负荷", "T3T4_balance": "T3/T4平衡",
    "Velocity_load": "动量负荷", "Velocity_balance": "动量平衡",
}


def _roc(y, p) -> float:
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")


_FEATIDX = __import__("re").compile(r"feature_(\d+)")


def _disp_term(raw: str, live: list[str]) -> str:
    """把 explain_local 的 term 名映射成中文显示名。

    interpret 0.7.8 在用 numpy array 训练后,explain_local 返回的是占位符
    `feature_0012`(及交互项 `feature_0008 × feature_0012`),其数字索引 == live
    列表位置。这里先把占位符还原成真实 live 列名,再过 DISP;同时兼容 `×`/`&`
    两种交互连接符与已是真名的情况。
    """
    def one(tok: str) -> str:
        m = _FEATIDX.fullmatch(tok.strip())
        real = live[int(m.group(1))] if m else tok.strip()
        return DISP.get(real, real)

    for sep in (" × ", " & "):
        if sep in raw:
            return " × ".join(one(p) for p in raw.split(sep))
    return one(raw)


def fit_landmark(L: int) -> dict:
    """单地标:dev OOF + temporal final-fit 预测概率,PCA 嵌入,逐样本 top3 贡献。

    返回供绘图用的数组字典。完全复用 bins_sweep 的 split / 口径,仅多 PCA。
    """
    rows = build_rows_for_method("locf", (L,))
    assert rows["episode_id"].nunique() == 1003, "analysis unit must be 1003 人次"
    y = rows["Y_24M_NHRH"].values.astype(int)
    lm = rows["landmark"].values
    is_dev = (rows["Split"] == "Development").values

    devL = is_dev & (lm == L)
    tstL = (~is_dev) & (lm == L)
    atL = lm == L

    feat_all = build_feats_at_L(rows, devL)   # 正交轴 z-fit on dev@L,应用到全行
    assert_no_future_feature(feat_all, L)     # time-safety gate(沿用)

    Xtr_full = feat_all.loc[devL]
    live = [c for c in feat_all.columns if Xtr_full[c].std() > 1e-9]
    feat_live = feat_all[live]

    Xd = feat_live.loc[devL].values
    yd = y[devL]
    epd = rows["episode_id"].values[devL]

    # --- dev OOF 概率 ---
    pred = np.full(len(rows), np.nan, dtype=float)
    dev_idx = np.where(devL)[0]
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    for tr, va in skf.split(Xd, yd, groups=epd):
        ebm = ExplainableBoostingClassifier(
            random_state=PY_SEED, interactions=INTERACTIONS,
            max_interaction_bins=MAX_INTERACTION_BINS,
        )
        ebm.fit(Xd[tr], yd[tr])
        pred[dev_idx[va]] = ebm.predict_proba(Xd[va])[:, 1]

    # --- temporal:dev-final-fit predict ---
    final_ebm = ExplainableBoostingClassifier(
        random_state=PY_SEED, interactions=INTERACTIONS,
        max_interaction_bins=MAX_INTERACTION_BINS,
    )
    final_ebm.fit(Xd, yd)
    if tstL.any():
        pred[tstL] = final_ebm.predict_proba(feat_live.loc[tstL].values)[:, 1]

    oof_auc = _roc(y[devL], pred[devL])
    tmp_auc = _roc(y[tstL], pred[tstL])

    # --- PCA 嵌入:全 1003@L 行的 live 特征,StandardScaler 后 PCA(3) ---
    X_atL = feat_live.loc[atL].values
    scaler = StandardScaler().fit(X_atL)
    Xz = scaler.transform(X_atL)
    pca = PCA(n_components=3, random_state=PY_SEED).fit(Xz)
    pcs = pca.transform(Xz)                       # (n@L, 3)
    evr = pca.explained_variance_ratio_

    # --- 监督降维对比:LDA(判别轴) + PCA-2D 可达判别力 ---------------------
    #   回答"降维一定丢判别力吗":PCA 按方差选向(无监督)会丢;LDA 按 y 选向
    #   (有监督),二分类只 1 个判别方向(C-1=1),1 维即保住大部分判别力。
    #   诚实口径:LDA / LR 都在 dev 上 5 折 OOF;temporal 用 dev-final 投影。
    Xz_dev = scaler.transform(feat_live.loc[devL].values)
    pcs_dev = pca.transform(Xz_dev)
    oof_ld1 = np.full(yd.shape, np.nan)
    oof_pca2d = np.full(yd.shape, np.nan)
    skf2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    for tr, va in skf2.split(Xz_dev, yd, groups=epd):
        lda = LinearDiscriminantAnalysis(n_components=1).fit(Xz_dev[tr], yd[tr])
        oof_ld1[va] = lda.transform(Xz_dev[va]).ravel()
        lr = LogisticRegression(max_iter=2000).fit(pcs_dev[tr][:, :2], yd[tr])
        oof_pca2d[va] = lr.predict_proba(pcs_dev[va][:, :2])[:, 1]
    sign = 1.0 if _roc(yd, oof_ld1) >= 0.5 else -1.0    # 让 LD1 与高危正相关
    oof_ld1 = sign * oof_ld1
    ld1_auc = _roc(yd, oof_ld1)
    pca2d_auc = _roc(yd, oof_pca2d)
    lda_full = LinearDiscriminantAnalysis(n_components=1).fit(Xz_dev, yd)
    sign_f = 1.0 if _roc(yd, lda_full.transform(Xz_dev).ravel()) >= 0.5 else -1.0
    ld1_at = sign_f * lda_full.transform(Xz).ravel()    # 全 atL 行判别分(dev-fit 投影)

    # --- 逐样本 top3 贡献特征(EBM explain_local,per-term log-odds 贡献绝对值)---
    #     用 dev-final-fit 解释全 1003@L 行,仅作 hover 文案,不参与建模/选择。
    #     传带列名的 DataFrame(而非 .values),否则 term names 退化成 feature_00xx 占位符。
    expl = final_ebm.explain_local(feat_live.loc[atL])
    top3_txt: list[str] = []
    n_at = int(atL.sum())
    for i in range(n_at):
        d = expl.data(i)
        pairs = sorted(zip(d["names"], d["scores"]),
                       key=lambda t: -abs(float(t[1])))
        parts = []
        for nm, sc in pairs[:3]:
            parts.append(f"{_disp_term(nm, live)} {float(sc):+.2f}")
        top3_txt.append(" / ".join(parts))

    # --- 排布:atL 顺序下的子数组 ---
    atL_idx = np.where(atL)[0]
    is_dev_at = is_dev[atL_idx]
    y_at = y[atL_idx]
    p_at = pred[atL_idx]

    return {
        "L": L,
        "evr": [round(float(v), 4) for v in evr],
        "oof_auc": round(float(oof_auc), 4),
        "tmp_auc": round(float(tmp_auc), 4),
        "n_live": len(live),
        "live": live,
        "pc1": pcs[:, 0], "pc2": pcs[:, 1], "pc3": pcs[:, 2],
        "ld1": ld1_at,
        "ld1_auc": round(float(ld1_auc), 4),
        "pca2d_auc": round(float(pca2d_auc), 4),
        "pc1_range": [float(np.nanpercentile(pcs[:, 0], 1)), float(np.nanpercentile(pcs[:, 0], 99))],
        "pc2_range": [float(np.nanpercentile(pcs[:, 1], 1)), float(np.nanpercentile(pcs[:, 1], 99))],
        "p": p_at, "y": y_at, "is_dev": is_dev_at,
        "top3": top3_txt,
        "n_at": n_at,
        "n_dev": int(is_dev_at.sum()),
        "n_tmp": int((~is_dev_at).sum()),
        "n_pos": int(y_at.sum()),
    }


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------

CSCALE = "RdBu_r"   # 蓝=低危、红=高危

# 预测对错四象限(阈值 0.5):TP/TN 正确(绿/浅蓝),FP/FN 错误(橙/红,醒目)
VERDICT_COL = {"TP": "#1a9850", "TN": "#a6cee3", "FP": "#fd8d3c", "FN": "#d73027"}
VERDICT_NAME = {"TP": "命中 TP(真复发·报复发)", "TN": "正确 TN(真未发·报未发)",
                "FP": "误报 FP(真未发·报复发)", "FN": "漏诊 FN(真复发·报未发)"}


def _verdict(d: dict) -> np.ndarray:
    """阈值 0.5 下每样本预测对错四象限标签(over atL 顺序)。"""
    pred1 = d["p"] >= 0.5
    pos = d["y"] == 1
    return np.where(pos, np.where(pred1, "TP", "FN"), np.where(pred1, "FP", "TN"))


def _hover(d: dict, sel: np.ndarray) -> list[str]:
    """逐样本 hover 文本。sel = 该 trace 选中的布尔掩码(over atL 顺序)。"""
    txt = []
    idx = np.where(sel)[0]
    for i in idx:
        split = "dev(OOF)" if d["is_dev"][i] else "temporal"
        pred1 = d["p"][i] >= 0.5
        right = pred1 == (d["y"][i] == 1)
        verdict = ("命中TP✓" if d["y"][i] == 1 else "正确TN✓") if right \
            else ("漏诊FN✗" if d["y"][i] == 1 else "误报FP✗")
        txt.append(
            f"真实Y={int(d['y'][i])} · 预测p={d['p'][i]:.3f}(阈0.5→{'复发' if pred1 else '未复发'}) · <b>{verdict}</b><br>"
            f"{split} · LD1={d['ld1'][i]:+.2f}<br>"
            f"PC1={d['pc1'][i]:+.2f} PC2={d['pc2'][i]:+.2f}<br>"
            f"top3贡献: {d['top3'][i]}"
        )
    return txt


def _scatter2d_traces(d: dict, visible: bool):
    """2D 散点:颜色=预测对错四象限,符号=dev(圆)/temporal(方);错的(FP/FN)放大加粗、后画叠上层。"""
    traces = []
    dev = d["is_dev"]
    cat = _verdict(d)
    for src, symbol in [("dev", "circle"), ("temporal", "square")]:
        smask = dev if src == "dev" else ~dev
        for c in ["TN", "TP", "FP", "FN"]:        # 错的(FP/FN)后画,叠上层更醒目
            sel = smask & (cat == c)
            if not sel.any():
                continue
            wrong = c in ("FP", "FN")
            traces.append(go.Scatter(
                x=d["pc1"][sel], y=d["pc2"][sel], mode="markers",
                name=f"{VERDICT_NAME[c]} · {src}", visible=visible, legendgroup=c,
                marker=dict(
                    size=9 if wrong else 5.5, symbol=symbol, color=VERDICT_COL[c],
                    line=dict(width=1.3 if wrong else 0.4,
                              color="#111" if wrong else "rgba(60,60,60,0.5)"),
                ),
                text=_hover(d, sel), hovertemplate="%{text}<extra></extra>",
            ))
    return traces


def _risk_contour2d(d: dict, visible: bool):
    """2D 决策面:dev OOF 概率在 PC1×PC2 上的插值等高线背景(仅凸包内、不外推)。

    与 3D 地形同源(同一 griddata 线性插值),凸包外 NaN→留白不填充;叠在散点
    之下,用半透明同色标把"高/低危区域"画成连续过渡 + 0.1 间隔等高线(含 p=0.5
    决策线),让区域一眼可分。注意:PC1+2 只占约 1/4 方差,此面必然是"软边界、
    红蓝交叠",恰好诚实暴露 2D 投影分不开 → 判别力在高维、应以形状函数为准。
    """
    from scipy.interpolate import griddata

    dev = d["is_dev"]
    gx = d["pc1"][dev]; gy = d["pc2"][dev]; gz = d["p"][dev]
    xi = np.linspace(np.percentile(d["pc1"], 1), np.percentile(d["pc1"], 99), 90)
    yi = np.linspace(np.percentile(d["pc2"], 1), np.percentile(d["pc2"], 99), 90)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata((gx, gy), gz, (XI, YI), method="linear")  # 凸包外 NaN→不填充
    return go.Contour(
        x=xi, y=yi, z=ZI, visible=visible, name="风险面(dev OOF)",
        colorscale=CSCALE, zmin=0.0, zmax=1.0, opacity=0.42,
        showscale=False, hoverinfo="skip", connectgaps=False,
        line=dict(width=0.6, color="rgba(70,70,70,0.35)"),
        contours=dict(start=0.1, end=0.9, size=0.1, showlines=True,
                      showlabels=True, labelfont=dict(size=9, color="#444")),
    )


def _lda_traces(d: dict, visible: bool):
    """LDA 监督判别轴 LD1 上 Y=0 / Y=1 两组的 OOF 分布(overlay 密度直方图)。

    二分类 LDA 只 1 个判别方向(C-1=1);两组沿这一根轴若明显分开,即"判别信号
    能压到 1 维"。对照 PCA 前 2 维分不开,说明丢的是无监督方差、不是判别力。
    """
    ld1 = d["ld1"]; y = d["y"]
    traces = []
    for cls, col, nm in [(0, "#4575b4", "真实未复发 Y=0"), (1, "#d73027", "真实复发 Y=1")]:
        sel = y == cls
        traces.append(go.Histogram(
            x=ld1[sel], visible=visible, name=nm, legendgroup=nm,
            marker=dict(color=col, line=dict(width=0.3, color="white")),
            opacity=0.6, nbinsx=44, histnorm="probability density",
        ))
    return traces


def _surface_and_points(d: dict, visible: bool):
    """3D 风险地形:dev OOF 概率插值面 + 全样本点(z=预测概率)。"""
    from scipy.interpolate import griddata

    traces = []
    dev = d["is_dev"]
    # 用 dev(OOF)样本插值出地形面(temporal 只叠点,不喂面,避免泄漏到面里)。
    gx = d["pc1"][dev]; gy = d["pc2"][dev]; gz = d["p"][dev]
    xi = np.linspace(np.percentile(d["pc1"], 1), np.percentile(d["pc1"], 99), 60)
    yi = np.linspace(np.percentile(d["pc2"], 1), np.percentile(d["pc2"], 99), 60)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata((gx, gy), gz, (XI, YI), method="linear")
    # 外推区 NaN → 不画(只在样本流形凸包内显示地形,呼应"不外推")
    traces.append(go.Surface(
        x=xi, y=yi, z=ZI, visible=visible, name="风险地形(dev OOF 插值)",
        colorscale=CSCALE, cmin=0.0, cmax=1.0, opacity=0.80,
        showscale=False, hoverinfo="skip",
        contours={"z": {"show": True, "usecolormap": True,
                        "highlightcolor": "#222", "project_z": True}},
    ))
    # 样本点(z=各自预测概率)
    cat = _verdict(d)
    for src, symbol in [("dev", "circle"), ("temporal", "diamond")]:
        smask = dev if src == "dev" else ~dev
        for c in ["TN", "TP", "FP", "FN"]:
            sel = smask & (cat == c)
            if not sel.any():
                continue
            wrong = c in ("FP", "FN")
            traces.append(go.Scatter3d(
                x=d["pc1"][sel], y=d["pc2"][sel], z=d["p"][sel],
                mode="markers", name=f"{VERDICT_NAME[c]} · {src}", visible=visible,
                showlegend=False,
                marker=dict(
                    size=4.6 if wrong else 2.8, symbol=symbol, color=VERDICT_COL[c],
                    line=dict(width=0.5, color="#111"),
                ),
                text=_hover(d, sel), hovertemplate="%{text}<extra></extra>",
            ))
    return traces


def build_html(results: dict, out_path: Path) -> None:
    L_list = list(results.keys())

    # ---- 2D figure(下拉切地标)----
    fig2d = go.Figure()
    seg2d = {}   # L -> (start, count)
    for L in L_list:
        start = len(fig2d.data)
        fig2d.add_trace(_risk_contour2d(results[L], visible=(L == L_list[0])))
        for t in _scatter2d_traces(results[L], visible=(L == L_list[0])):
            fig2d.add_trace(t)
        seg2d[L] = (start, len(fig2d.data) - start)
    n2d = len(fig2d.data)

    def _vis_mask(seg, L_on, total):
        m = [False] * total
        s, c = seg[L_on]
        for i in range(s, s + c):
            m[i] = True
        return m

    btn2d = []
    for L in L_list:
        d = results[L]
        btn2d.append(dict(
            label=f"{L}M", method="update",
            args=[{"visible": _vis_mask(seg2d, L, n2d)},
                  {"title": _title2d(d),
                   "xaxis.range": d["pc1_range"], "yaxis.range": d["pc2_range"]}],
        ))
    d0 = results[L_list[0]]
    fig2d.update_layout(
        title=_title2d(d0),
        xaxis=dict(title="PC1", range=d0["pc1_range"]),
        yaxis=dict(title="PC2", range=d0["pc2_range"]),
        template="plotly_white", height=660,
        font=dict(family="PingFang SC, Hiragino Sans GB, Microsoft YaHei, Noto Sans CJK SC, sans-serif", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11)),
        margin=dict(l=60, r=40, t=152, b=55),
        updatemenus=[dict(type="buttons", direction="right", buttons=btn2d,
                          x=0.0, y=1.24, xanchor="left", showactive=True,
                          pad=dict(t=4, b=4))],
    )

    # ---- 3D figure(下拉切地标)----
    fig3d = go.Figure()
    seg3d = {}
    for L in L_list:
        start = len(fig3d.data)
        tr = _surface_and_points(results[L], visible=(L == L_list[0]))
        for t in tr:
            fig3d.add_trace(t)
        seg3d[L] = (start, len(tr))
    n3d = len(fig3d.data)

    btn3d = []
    for L in L_list:
        d = results[L]
        btn3d.append(dict(
            label=f"{L}M", method="update",
            args=[{"visible": _vis_mask(seg3d, L, n3d)},
                  {"title": _title3d(d)}],
        ))
    fig3d.update_layout(
        title=_title3d(results[L_list[0]]),
        template="plotly_white", height=680,
        font=dict(family="PingFang SC, Hiragino Sans GB, Microsoft YaHei, Noto Sans CJK SC, sans-serif", size=13),
        scene=dict(
            xaxis_title="PC1", yaxis_title="PC2", zaxis_title="预测复发 p",
            zaxis=dict(range=[0, 1]),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.9)),
        ),
        margin=dict(l=10, r=10, t=110, b=10),
        updatemenus=[dict(type="buttons", direction="right", buttons=btn3d,
                          x=0.0, y=1.10, xanchor="left", showactive=True,
                          pad=dict(t=4, b=4))],
    )

    # ---- LDA figure(监督判别轴 LD1 分布,下拉切地标)----
    fig_lda = go.Figure()
    seg_lda = {}
    for L in L_list:
        start = len(fig_lda.data)
        for t in _lda_traces(results[L], visible=(L == L_list[0])):
            fig_lda.add_trace(t)
        seg_lda[L] = (start, len(fig_lda.data) - start)
    n_lda = len(fig_lda.data)
    btn_lda = [dict(label=f"{L}M", method="update",
                    args=[{"visible": _vis_mask(seg_lda, L, n_lda)},
                          {"title": _title_lda(results[L])}]) for L in L_list]
    fig_lda.update_layout(
        title=_title_lda(results[L_list[0]]), barmode="overlay",
        xaxis_title="LD1 — LDA 监督判别轴(OOF 判别分,越大越偏高危)",
        yaxis_title="概率密度", template="plotly_white", height=440,
        font=dict(family="PingFang SC, Hiragino Sans GB, Microsoft YaHei, Noto Sans CJK SC, sans-serif", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=60, r=40, t=120, b=55),
        updatemenus=[dict(type="buttons", direction="right", buttons=btn_lda,
                          x=0.0, y=1.22, xanchor="left", showactive=True,
                          pad=dict(t=4, b=4))],
    )

    # ---- 组装自包含 HTML ----
    div2d = pio.to_html(fig2d, include_plotlyjs="inline", full_html=False,
                        div_id="fig2d", config={"displaylogo": False})
    div3d = pio.to_html(fig3d, include_plotlyjs=False, full_html=False,
                        div_id="fig3d", config={"displaylogo": False})
    div_lda = pio.to_html(fig_lda, include_plotlyjs=False, full_html=False,
                          div_id="fig_lda", config={"displaylogo": False})

    evr_rows = "".join(
        f"<tr><td>{L}M</td><td>{results[L]['evr'][0]*100:.1f}%</td>"
        f"<td>{results[L]['evr'][1]*100:.1f}%</td>"
        f"<td>{results[L]['evr'][2]*100:.1f}%</td>"
        f"<td>{sum(results[L]['evr'][:2])*100:.1f}%</td>"
        f"<td>{sum(results[L]['evr'][:3])*100:.1f}%</td>"
        f"<td>{results[L]['n_live']}</td>"
        f"<td>{results[L]['oof_auc']:.3f}</td>"
        f"<td>{results[L]['tmp_auc']:.3f}</td></tr>"
        for L in L_list
    )
    cmp_rows = "".join(
        f"<tr><td>{L}M</td>"
        f"<td>{results[L]['pca2d_auc']:.3f}</td>"
        f"<td><b style='color:#1a7a3a'>{results[L]['ld1_auc']:.3f}</b></td>"
        f"<td>{results[L]['oof_auc']:.3f}</td>"
        f"<td>{results[L]['ld1_auc']-results[L]['pca2d_auc']:+.3f}</td></tr>"
        for L in L_list
    )

    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EBM · PCA / LDA 降维风险景观(bin16)</title>
<style>
  body {{ font-family: "PingFang SC","Hiragino Sans GB","Microsoft YaHei","Noto Sans CJK SC",sans-serif;
         margin: 0; padding: 22px 28px 60px; color:#1d2430; background:#fbfcfe; }}
  h1 {{ font-size: 23px; margin: 0 0 6px; }}
  h2 {{ font-size: 17px; margin: 30px 0 8px; color:#243; border-left:4px solid #4a7; padding-left:9px; }}
  .sub {{ color:#566; font-size:13px; margin:0 0 14px; }}
  .honest {{ background:#fff7e6; border:1px solid #f0d090; border-radius:8px;
             padding:12px 16px; font-size:13.5px; line-height:1.62; color:#5a4a1a; margin:14px 0 8px; }}
  .honest b {{ color:#7a5a00; }}
  table.meta {{ border-collapse:collapse; font-size:13px; margin:10px 0 4px; }}
  table.meta th, table.meta td {{ border:1px solid #d7dde6; padding:5px 11px; text-align:center; }}
  table.meta th {{ background:#eef3f8; }}
  .legend-note {{ font-size:12.5px; color:#667; margin:6px 0 0; line-height:1.55; }}
  .pill {{ display:inline-block; background:#eef3f8; border:1px solid #d7dde6; border-radius:12px;
           padding:1px 9px; margin:0 3px; font-size:12px; }}
  code {{ background:#eef1f4; padding:1px 5px; border-radius:4px; font-size:12px; }}
</style></head>
<body>
<h1>EBM · 模型在 PCA / LDA 降维空间的风险景观 + 决策面</h1>
<p class="sub">risk-landscape over PCA(无监督) &amp; LDA(有监督) embeddings ·
  口径:corrected 真值 + LOCF,<code>max_interaction_bins=16</code>,interactions=5 ·
  分析单元 <b>1003 人次</b>(repeat 疗程按独立人次)· 地标 6M / 12M</p>

<div class="honest">
  <b>诚实标注 — 怎么看这三张图。</b>
  <b>① PCA(无监督)</b>按"方差最大"选投影方向、<b>不看标签</b>,前 2 维只占约 1/4 方差,所以高/低危
  在 2D 上必然<b>交叠、分不开</b>——丢的是"无监督方差",不代表判别力没了。
  <b>③ LDA(有监督)</b>按"类间/类内方差比最大"选向、<b>朝标签 y 优化</b>,二分类只 1 个判别方向(C−1=1),
  看两类沿 LD1 一根轴能否分开:能分开就说明<b>判别信号可压到 1 维,但得监督才找得到</b>。代价是 LDA
  用了标签,故一律以 <b>OOF</b> 评估(否则乐观虚高)。两者都<b>不替代逐特征形状函数</b>(EBM 玻璃盒本体)。
  降维面/地形只在 dev 凸包内插值绘制,<b>不外推</b>。
</div>

<table class="meta">
  <tr><th>地标</th><th>PC1 解释方差</th><th>PC2</th><th>PC3</th>
      <th>PC1+2 累计</th><th>PC1–3 累计</th><th>live 特征数</th>
      <th>OOF AUC</th><th>temporal AUC</th></tr>
  {evr_rows}
</table>
<p class="sub" style="margin:16px 0 4px"><b>判别力对比(dev OOF AUC)——回答"降维一定丢吗"</b></p>
<table class="meta">
  <tr><th>地标</th><th>PCA 前2维<br>(无监督,LR)</th><th>LDA LD1<br>(有监督,1维)</th>
      <th>全模型 EBM<br>(16 维)</th><th>LDA − PCA2D</th></tr>
  {cmp_rows}
</table>
<p class="legend-note">
  PCA 前 2 维(占 ~1/4 方差)判别力明显低于全模型;<b>LDA 仅 1 维</b>就逼近全模型 → 判别信号能压到 1 维,
  但<b>要监督才找得到</b>,无监督 PCA 找不到。LDA 用了标签,此处全为 <b>OOF</b> 值(诚实,不乐观)。
</p>
<p class="legend-note">
  <span class="pill" style="color:#1a9850">● 命中 TP</span>
  <span class="pill" style="color:#3a7fb0">● 正确 TN</span>
  <span class="pill" style="color:#d9730d">● 误报 FP</span>
  <span class="pill" style="color:#c0392b">● 漏诊 FN(最危险)</span>
  <span class="pill">● 圆=dev / ■ 方=temporal</span>
  点色 = 阈 0.5 下预测对错四象限,错的(FP/FN)放大加粗;背景等高面 = OOF 概率(红高蓝低)。
</p>

<h2>① PCA 2D 鸟瞰(无监督)— 点色=预测对错,背景=风险决策面</h2>
<p class="sub"><b>点色</b>=阈 0.5 下预测对错(TP/TN/FP/FN),<b>符号</b>=dev(圆)/temporal(方),错的放大加粗;
  <b>背景</b>=dev OOF 概率插值决策面(含 p=0.5 决策线,仅凸包内不外推);视图已裁到 1–99 分位(去离群点)。
  注意 PC1+2 只占约 1/4 方差,红蓝必然交叠——这正是无监督降维"分不开"的诚实呈现。顶部按钮切 6M / 12M。</p>
{div2d}

<h2>② PCA 3D 风险地形(无监督)— PC1 × PC2 平面 + z = 预测复发概率(可旋转)</h2>
<p class="sub">连续曲面 = dev OOF 概率在嵌入平面上的插值地形(仅凸包内、不外推);散点为各样本(高度=其预测 p)。
  <b>拖拽旋转、滚轮缩放</b>(3D 用 WebGL,需在浏览器本地打开才能转)。</p>
{div3d}

<h2>③ LDA 监督判别轴 LD1(有监督)— 两类沿一根轴的分离</h2>
<p class="sub">二分类 LDA 只有 1 个判别方向(C−1=1)。下面是 <b>OOF LD1</b> 上真实未复发 / 复发两组的密度分布:
  两峰<b>错开</b>即"判别信号可压到 1 维"——对照 ① 的 PCA 2D 分不开,差别就在 LDA <b>看了 y</b>、PCA 没看。顶部按钮切 6M / 12M。</p>
{div_lda}

</body></html>
"""
    # 安全核验:无 forbidden unique-patient token。
    # 注意:整页内联了 plotly.js minified bundle(数值常量/SVG path/哈希)与运行时
    #   浮点(如 hover 里某样本 预测p=0.889),裸子串 "889" 会被这些第三方/数值内容误命中,
    #   并非患者计数泄漏。故只对「我撰写的语义内容」(标题/表格/标注/图例/章节文案,
    #   即所有人为写入的计数与说明)做核验,而非整页 JS bundle。authored 内容里的计数
    #   一律来自运行时(1003/802/201/374),不含任何 unique-patient count。
    authored = "\n".join([
        *[_title2d(results[L]) for L in L_list],
        *[_title3d(results[L]) for L in L_list],
        *[_title_lda(results[L]) for L in L_list],
        evr_rows, cmp_rows,
        page[page.index("<body>"):page.index(div2d)],   # 顶部全部人为文案(含 honest/表/图例/①标题)
    ])
    assert forbidden_token() not in authored, "forbidden unique-patient count leaked into authored HTML"
    out_path.write_text(page, encoding="utf-8")


def _title2d(d: dict) -> str:
    return (f"{d['L']}M · PCA 嵌入风险景观(2D)　"
            f"PC1+2={sum(d['evr'][:2])*100:.0f}% 方差　"
            f"n={d['n_at']}人次(dev {d['n_dev']} / temporal {d['n_tmp']}，复发 {d['n_pos']})　"
            f"OOF AUC={d['oof_auc']:.3f}")


def _title3d(d: dict) -> str:
    return (f"{d['L']}M · PCA 嵌入风险地形(3D)　"
            f"PC1+2={sum(d['evr'][:2])*100:.0f}% 方差　"
            f"z=预测复发 p　OOF AUC={d['oof_auc']:.3f}")


def _title_lda(d: dict) -> str:
    return (f"{d['L']}M · LDA 监督判别轴 LD1 的两类分布　"
            f"LD1(1维) OOF AUC={d['ld1_auc']:.3f}　"
            f"vs PCA前2维={d['pca2d_auc']:.3f}　vs 全模型(16维)={d['oof_auc']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="smoke: 仅 6M 一个地标,不写最终 HTML(只打印 EVR/AUC)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    L_run = (6,) if args.quick else LANDMARKS

    results = {}
    for L in L_run:
        print(f"[fit] landmark {L}M …", flush=True)
        d = fit_landmark(L)
        results[L] = d
        print(f"  EVR PC1/2/3 = {d['evr'][0]:.3f} / {d['evr'][1]:.3f} / {d['evr'][2]:.3f}"
              f"  (PC1+2={sum(d['evr'][:2]):.3f}, PC1-3={sum(d['evr'][:3]):.3f})", flush=True)
        print(f"  n@L={d['n_at']} (dev {d['n_dev']} / temporal {d['n_tmp']}, 复发 {d['n_pos']})"
              f"  live={d['n_live']}  OOF_AUC={d['oof_auc']}  temporal_AUC={d['tmp_auc']}", flush=True)

    if args.quick:
        print("[quick] smoke done (HTML 未写).", flush=True)
        return

    out_path = OUT / "Module2v2_EBM_pca_landscape.html"
    build_html(results, out_path)
    size_kb = out_path.stat().st_size / 1024
    print(f"\nSaved → {out_path}  ({size_kb:.0f} KB)", flush=True)
    print("[done] PCA risk-landscape HTML complete.", flush=True)


if __name__ == "__main__":
    main()
