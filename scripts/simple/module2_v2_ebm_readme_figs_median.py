#!/usr/bin/env python
"""M2 · v2 — README 用静态 PNG 图(matplotlib,带决策曲面,CJK 无豆腐)。

README 嵌 PNG(非 html),复用现有两个 plotly 脚本的口径与拟合逻辑,只把输出
换成 matplotlib 静态图:

  图1【最重要特征 3D 曲面】12M「激素负荷 Hormone_load × 动量负荷 Velocity_load」
       交互项 g(x_i,x_j) 的决策曲面(z = 对 log-odds 的贡献),Axes3D.plot_surface,
       叠训练样本散点(样本密集处可信、角落无样本处是正则外推)。即
       momentum-beyond-inertia 的交互增量。
       → results/module2_v2_vertical/m2v2_bin16_top_interaction_3d_median.png

  图2【降维整体风险景观】6M PCA PC1×PC2 散点 + dev OOF 概率插值决策面(contourf,
       仅样本凸包内、不外推,含 p=0.5 决策线),散点按预测对错四象限着色
       (TP/TN/FP/FN,FN 漏诊红色放大)。视图裁到 PC1/PC2 的 1–99 分位。
       → results/module2_v2_vertical/m2v2_bin16_pca_landscape_median.png

口径(硬约束):
  - 数据 corrected + median(主线):`build_rows_for_method("locf", (1,3,6,12))`。
  - dev OOF 分析;temporal 仅报告(此处两图均用 dev OOF / dev-fit 解释,不喂 temporal)。
  - 模型 EBM bin16:`ebm_oof_and_temporal` 默认 max_interaction_bins=16。
  - 分析单元 1003 人次;无 unique-patient 计数 / 无数字 889;time-safe。
  - 不写 significant 无检验。
  - 不写任何 worklog 文件(并发冲突)。
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import argparse
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import griddata

# --- CJK 字体(无豆腐)---
fm.fontManager.addfont("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False

from scripts.simple.module2_v2_shared import forbidden_token
from scripts.simple.module2_v2_impute_experiment import build_rows_for_method
from scripts.simple.module2_v2_b4_ebm_axes import build_feats_at_L
from scripts.simple.module2_v2_b4_ebm_oof import ebm_oof_and_temporal

# PCA 风险景观沿用 plotly 版本的 fit_landmark / _verdict / 配色,保持完全同源口径。
from scripts.simple.module2_v2_ebm_pca_landscape import (
    fit_landmark,
    _verdict,
    VERDICT_COL,
    VERDICT_NAME,
)

# === median 主线 patch:让 fit_landmark 内的 build_rows_for_method 也用 median ===
# fit_landmark 是从 module2_v2_ebm_pca_landscape 引入的,该模块内硬编码 "locf"。
# 重写它的 build_rows_for_method 引用,以便 fit_landmark 在本脚本下用 median 数据。
import scripts.simple.module2_v2_ebm_pca_landscape as _pca_mod
import scripts.simple.module2_v2_impute_experiment as _imp_mod
_orig_build = _imp_mod.build_rows_for_method
def _patched_build(method, landmarks=None):
    new = "median" if method == "locf" else method
    if landmarks is None:
        return _orig_build(new)
    return _orig_build(new, landmarks)
_imp_mod.build_rows_for_method = _patched_build
_pca_mod.build_rows_for_method = _patched_build
# === patch 结束 ===



OUT = ROOT / "results" / "module2_v2_vertical"

# 中文特征显示名(与正交轴语义一致)。
DISP = {
    "ThyroidW": "甲状腺重量", "TRAb": "TRAb", "TGAb": "TGAb", "TPOAb": "TPOAb",
    "Sex": "性别", "FT4_0M": "基线FT4", "TSH_0M": "基线TSH",
    "log1p_DiseaseDuration_Months_Aug": "病程(log)", "Uptake24h": "24h摄碘率",
    "HalfLife": "有效半衰期", "TSH_current": "当期TSH", "TSH_velocity": "TSH速度",
    "Hormone_load": "激素负荷", "T3T4_balance": "T3/T4平衡",
    "Velocity_load": "动量负荷", "Velocity_balance": "动量平衡",
}


def _disp(t: str) -> str:
    return DISP.get(t, t)


# ---------------------------------------------------------------------------
# 图1:12M 最重要两两交互项的 3D 决策曲面(matplotlib Axes3D.plot_surface)
# ---------------------------------------------------------------------------

def _resolve(term: str, live: list[str]) -> str:
    """把 interpret 的占位符 term 名(feature_NN / feature_NN & feature_MM)还原成真实 live 列名。"""
    def one(tok: str) -> str:
        tok = tok.strip()
        if tok.startswith("feature_"):
            try:
                return live[int(tok.split("_")[1])]
            except (ValueError, IndexError):
                return tok
        return tok
    return " & ".join(one(p) for p in term.split(" & ")) if " & " in term else one(term)


def build_interaction(L: int, prefset: set[str]) -> dict:
    """单地标:取最重要的(优先 prefset)两两交互项,返回 2D scores 网格 + 训练样本散点高度。

    完全复用 module2_v2_ebm_interaction_3d.build 的口径与查表逻辑,仅把输出留给 matplotlib。

    注意(bin16 GA2M 实况):interactions=5 自动选出的交互项集合随地标变化 —
    momentum-beyond-inertia 的「激素负荷 × 动量负荷」(Hormone_load &
    Velocity_load)在 6M 是自动选出的 #1 交互项,但在 12M 未被自动选入(12M 的 #1
    是「当期TSH × 激素负荷」)。`prefset` 命中则用之(诚实复用真实模型已学到的项),
    未命中则回退到该地标真正的 top-1 交互项;`hit` 标志记录是否命中 prefset,
    所有标题/轴名一律由「实际选中」的项派生,绝不张冠李戴。
    """
    rows = build_rows_for_method("median", (1, 3, 6, 12))
    assert rows["episode_id"].nunique() == 1003, "analysis unit must be 1003 人次"
    y = rows["Y_24M_NHRH"].values.astype(int)
    lm = rows["landmark"].values
    is_dev = (rows["Split"] == "Development").values
    devL = is_dev & (lm == L)

    # bin16:ebm_oof_and_temporal 默认 max_interaction_bins=16,取其 final_ebm。
    _pred, ebm, live = ebm_oof_and_temporal(rows, y, lm, is_dev, L)

    feat = build_feats_at_L(rows, devL)
    Xtr = feat.loc[devL, live]

    g = ebm.explain_global()
    overall = dict(zip(g.data()["names"], g.data()["scores"]))
    inter = [(i, n, _resolve(n, live)) for i, n in enumerate(ebm.term_names_) if " & " in n]
    inter = sorted(inter, key=lambda t: -overall.get(t[1], 0.0))
    preferred = next((t for t in inter if _resolve(t[1], live) in prefset), None)
    chosen = preferred if preferred is not None else inter[0]
    hit = preferred is not None
    idx, name, rname = chosen
    a, b = rname.split(" & ")

    d = g.data(idx)
    sc = np.asarray(d["scores"], float)                      # 2D 网格 g(log-odds 贡献)
    le = np.asarray(d["left_names"], float)                  # x 轴 bin 边
    re_ = np.asarray(d["right_names"], float)                # y 轴 bin 边
    le = le[np.isfinite(le)]
    re_ = re_[np.isfinite(re_)]
    xc = (le[:-1] + le[1:]) / 2                              # x 轴 bin 中心
    yc = (re_[:-1] + re_[1:]) / 2                            # y 轴 bin 中心

    Av = Xtr[a].values.astype(float)
    Bv = Xtr[b].values.astype(float)

    def gz(x: float, yv: float) -> float:
        i = int(np.clip(np.searchsorted(le, x) - 1, 0, sc.shape[0] - 1))
        j = int(np.clip(np.searchsorted(re_, yv) - 1, 0, sc.shape[1] - 1))
        return sc[i, j]

    zs = np.array([gz(x, yv) for x, yv in zip(Av, Bv)])      # 训练样本落格的 g 高度

    # 视图裁到 1–99 分位(去极端外推边),与 plotly 版一致。
    xlo, xhi = np.nanpercentile(Av, 1), np.nanpercentile(Av, 99)
    ylo, yhi = np.nanpercentile(Bv, 1), np.nanpercentile(Bv, 99)
    return {
        "L": L, "hit": hit, "term_raw": rname,
        "a": _disp(a), "b": _disp(b), "a_raw": a, "b_raw": b,
        "xc": xc, "yc": yc, "sc": sc, "Av": Av, "Bv": Bv, "zs": zs,
        "imp": overall.get(name, 0.0),
        "xr": (xlo, xhi), "yr": (ylo, yhi),
        "n_tr": int(devL.sum()),
    }


# 各正交轴的简短中文释义(轴标用,避免误标)。
AXIS_HINT = {
    "Hormone_load": "综合FT3/FT4 水平(↑越热)", "Velocity_load": "综合FT3/FT4 变化速度(↑回升)",
    "T3T4_balance": "FT3/FT4 落差", "Velocity_balance": "FT3/FT4 速度落差",
    "TSH_current": "当期 TSH(↑越受抑提示甲亢)", "TSH_velocity": "TSH 变化速度",
    "ThyroidW": "甲状腺重量", "TPOAb": "TPOAb", "Uptake24h": "24h 摄碘率",
}


def fig_interaction_3d(out_path: Path) -> dict:
    """图1:momentum-beyond-inertia「激素负荷 × 动量负荷」交互项的 3D 决策曲面 PNG。

    关键实况:interactions=5 的 bin16 GA2M 在 **6M** 把「Hormone_load &
    Velocity_load」自动选为 #1 交互项,在 12M 未自动选入(12M #1 是「当期TSH ×
    激素负荷」)。为忠实呈现 momentum-beyond-inertia 这一命名交互的真实学习结果、
    且不伪造一个被迫单交互的模型,本图取该命名项被自动选中的地标(6M)。所有标
    题/轴名一律由实际选中的项派生;若该项确实命中(hit)则在标题点明,否则报错。
    """
    L = 6
    pref = {"Hormone_load & Velocity_load", "Velocity_load & Hormone_load"}
    D = build_interaction(L, pref)
    assert D["hit"], (
        f"momentum 交互项未在 {L}M 被自动选入(实际选中 {D['term_raw']});"
        "请检查 interactions 配置"
    )

    XC, YC = np.meshgrid(D["xc"], D["yc"])      # (ny, nx)
    Z = D["sc"].T                               # 转置使行=y、列=x,与网格匹配
    vmax = float(np.nanmax(np.abs(D["sc"]))) or 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig = plt.figure(figsize=(11.0, 8.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_position((0.02, 0.04, 0.84, 0.80))   # 让曲面占满画布,减少留白
    try:
        ax.set_box_aspect((1.25, 1.25, 0.85))   # x/y 铺开、z 适度压扁,山脊更清晰
    except Exception:
        pass

    surf = ax.plot_surface(
        XC, YC, Z, cmap="RdBu_r", norm=norm,
        rstride=1, cstride=1, linewidth=0.15, edgecolor="0.5",
        antialiased=True, alpha=0.92, shade=False,
    )

    # 训练样本散点(黑),z = 各样本落格的 g 高度 —— 看「哪里有真实数据」。
    in_view = (
        (D["Av"] >= D["xr"][0]) & (D["Av"] <= D["xr"][1])
        & (D["Bv"] >= D["yr"][0]) & (D["Bv"] <= D["yr"][1])
    )
    ax.scatter(
        D["Av"][in_view], D["Bv"][in_view], D["zs"][in_view],
        s=7, c="#111111", alpha=0.40, depthshade=True,
        label=f"训练样本(dev@{L}M,可见 {int(in_view.sum())} / 总 {D['n_tr']};视图裁 1–99 分位去极端外推边)",
    )

    ax.set_xlim(D["xr"]); ax.set_ylim(D["yr"])
    ax.set_xlabel(f"\n{D['a']}\n{AXIS_HINT.get(D['a_raw'], '')}", fontsize=10.5, labelpad=12)
    ax.set_ylabel(f"\n{D['b']}\n{AXIS_HINT.get(D['b_raw'], '')}", fontsize=10.5, labelpad=12)
    ax.set_zlabel("g — 对复发 log-odds 的交互贡献", fontsize=10.5, labelpad=6)
    ax.view_init(elev=22, azim=-128)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=16, pad=0.10)
    cbar.set_label("g(log-odds 贡献)  红=推高复发风险 / 蓝=压低", fontsize=10)

    ax.set_title(
        f"EBM 交互项 3D 决策曲面 · {L}M「{D['a']} × {D['b']}」(bin16,GA2M 自动选入 · median 主线下 #3)\n"
        "momentum-beyond-inertia:在「当前水平」之外,叠加「变化速度」的交互增量\n"
        "z = 该两特征组合对复发 log-odds 的交互贡献 g;黑点 = 训练样本落格高度\n"
        "样本密集处曲面可信;角落翘起却无黑点的「山脊」是正则化外推(不可信)",
        fontsize=12, pad=16,
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)

    fig.text(
        0.015, 0.015,
        f"momentum-beyond-inertia 的交互增量 · corrected+median · 分析单元 1003 人次 · "
        f"importance={D['imp']:.4f} · |g|max={vmax:.2f} · "
        f"(该命名交互在 12M 未自动选入,故取其自动选入的 {L}M)",
        fontsize=8.5, color="#555555",
    )

    # 已用 ax.set_position 手动布局,不调 tight_layout(会覆盖)。
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {
        "L": L, "hit": D["hit"], "term": f"{D['a']} × {D['b']}",
        "vmax": vmax, "imp": D["imp"], "n_tr": D["n_tr"],
        "a": D["a"], "b": D["b"], "xr": D["xr"], "yr": D["yr"],
    }


# ---------------------------------------------------------------------------
# 图2:6M PCA PC1×PC2 风险景观 + dev OOF 概率插值决策面(matplotlib contourf)
# ---------------------------------------------------------------------------

def fig_pca_landscape(out_path: Path, L: int = 6) -> dict:
    """图2:PCA PC1×PC2 散点(预测对错四象限) + dev OOF 概率插值决策面 PNG。

    复用 plotly 版 fit_landmark(同源 OOF / PCA / verdict 口径);决策面用 dev OOF
    概率 griddata 线性插值,凸包外 NaN→留白不外推,含 p=0.5 决策线。
    """
    d = fit_landmark(L)
    cat = _verdict(d)              # 阈 0.5 下逐样本 TP/TN/FP/FN(over atL 顺序)
    dev = d["is_dev"]

    # --- 决策面:仅 dev(OOF)样本插值,凸包外留白(不外推)---
    gx = d["pc1"][dev]; gy = d["pc2"][dev]; gz = d["p"][dev]
    xlo, xhi = d["pc1_range"]
    ylo, yhi = d["pc2_range"]
    xi = np.linspace(xlo, xhi, 220)
    yi = np.linspace(ylo, yhi, 220)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata((gx, gy), gz, (XI, YI), method="linear")   # 凸包外 NaN→不填充

    fig, ax = plt.subplots(figsize=(11.0, 8.2))

    # 连续过渡背景(红=高危/蓝=低危)+ 0.1 间隔等高线
    levels = np.linspace(0.0, 1.0, 11)
    cf = ax.contourf(XI, YI, ZI, levels=levels, cmap="RdBu_r",
                     vmin=0.0, vmax=1.0, alpha=0.55, extend="neither")
    cl = ax.contour(XI, YI, ZI, levels=levels, colors="0.45",
                    linewidths=0.5, alpha=0.6)
    ax.clabel(cl, inline=True, fontsize=7, fmt="%.1f")
    # p=0.5 决策线(加粗黑色虚线)
    try:
        dec = ax.contour(XI, YI, ZI, levels=[0.5], colors="#111111",
                         linewidths=2.0, linestyles="--")
        ax.clabel(dec, inline=True, fontsize=9, fmt="决策线 p=0.5")
    except Exception:
        pass

    # --- 散点:预测对错四象限;错的(FP/FN)放大加粗、后画叠上层;FN 漏诊红色最大 ---
    order = ["TN", "TP", "FP", "FN"]
    size_map = {"TN": 22, "TP": 26, "FP": 60, "FN": 95}
    for c in order:
        sel = cat == c
        if not sel.any():
            continue
        wrong = c in ("FP", "FN")
        ax.scatter(
            d["pc1"][sel], d["pc2"][sel],
            s=size_map[c], c=VERDICT_COL[c],
            marker="o",
            edgecolors=("#111111" if wrong else "none"),
            linewidths=(1.1 if wrong else 0.0),
            alpha=(0.95 if wrong else 0.70),
            label=f"{VERDICT_NAME[c]}  (n={int(sel.sum())})",
            zorder=(5 if wrong else 3),
        )

    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.set_xlabel(f"PC1(解释方差 {d['evr'][0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2(解释方差 {d['evr'][1]*100:.1f}%)", fontsize=11)

    cbar = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("预测复发概率 p(dev OOF 插值)  红=高危 / 蓝=低危", fontsize=10)

    pc12 = sum(d["evr"][:2]) * 100
    ax.set_title(
        f"EBM 模型在 PCA 嵌入空间的风险景观 + 决策面 · {L}M(bin16)\n"
        f"PC1×PC2 散点(按阈 0.5 预测对错四象限着色) + dev OOF 概率插值决策面(仅凸包内、不外推)\n"
        f"PC1+2 仅占 {pc12:.0f}% 方差 → 红蓝必然交叠(无监督降维「分不开」的诚实呈现)",
        fontsize=12.5, pad=14,
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9, scatterpoints=1)

    fig.text(
        0.015, 0.012,
        f"corrected+median · 分析单元 1003 人次 · n@{L}M={d['n_at']}"
        f"(dev {d['n_dev']} / temporal {d['n_tmp']},复发 {d['n_pos']}) · OOF AUC={d['oof_auc']:.3f} · "
        "FN(漏诊,真复发却报未复发)红色放大 = 最危险的错误",
        fontsize=8.5, color="#555555",
    )

    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 四象限计数(用于报告)
    counts = {c: int((cat == c).sum()) for c in order}
    return {
        "evr": d["evr"], "oof_auc": d["oof_auc"], "n_at": d["n_at"],
        "n_dev": d["n_dev"], "n_tmp": d["n_tmp"], "n_pos": d["n_pos"],
        "counts": counts, "pc12": pc12,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="smoke:只跑图2(PCA 6M,小网格隐含在 fit_landmark),验证管线后再全跑")
    ap.add_argument("--only", choices=["3d", "pca"], default=None,
                    help="只生成其中一张图(调试用)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    forbidden = forbidden_token()  # "889" — 确保不出现在任何 authored 文案

    if args.quick:
        # smoke:只跑 PCA 6M(最快验证 fit_landmark + matplotlib + CJK 字体)。
        p2 = OUT / "m2v2_bin16_pca_landscape_SMOKE.png"
        info2 = fig_pca_landscape(p2, L=6)
        print(f"[quick] PCA landscape smoke → {p2.name} "
              f"({p2.stat().st_size/1024:.0f} KB)  OOF AUC={info2['oof_auc']:.3f} "
              f"counts={info2['counts']}", flush=True)
        assert forbidden not in str(info2), "forbidden unique-patient count leaked"
        print("[quick] smoke done.", flush=True)
        return

    results = {}
    if args.only in (None, "3d"):
        p1 = OUT / "m2v2_bin16_top_interaction_3d_median.png"
        info1 = fig_interaction_3d(p1)
        results["3d"] = (p1, info1)
        print(f"[done] 交互 3D 曲面 → {p1.name} ({p1.stat().st_size/1024:.0f} KB)  "
              f"地标={info1['L']}M  交互项=「{info1['term']}」(hit={info1['hit']}) "
              f"importance={info1['imp']:.4f} |g|max={info1['vmax']:.3f}  "
              f"训练样本 n={info1['n_tr']}", flush=True)

    if args.only in (None, "pca"):
        p2 = OUT / "m2v2_bin16_pca_landscape_median.png"
        info2 = fig_pca_landscape(p2, L=6)
        results["pca"] = (p2, info2)
        print(f"[done] PCA 风险景观 → {p2.name} ({p2.stat().st_size/1024:.0f} KB)  "
              f"6M PC1+2={info2['pc12']:.0f}% OOF AUC={info2['oof_auc']:.3f}  "
              f"四象限 {info2['counts']}", flush=True)

    # 安全核验:无 forbidden unique-patient token 出现在我撰写的运行时摘要里。
    assert forbidden not in str({k: v[1] for k, v in results.items()}), \
        "forbidden unique-patient count leaked into authored output"
    print("[all done] README 静态 PNG 生成完毕。", flush=True)


if __name__ == "__main__":
    main()
