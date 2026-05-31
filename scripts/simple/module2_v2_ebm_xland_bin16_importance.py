#!/usr/bin/env python
"""M2 · v2 — EBM 原生重要性 top5 × 地标(1/3/6/12)迁移表(bin16 口径)。

口径(与 Phase 3 诊断 / bins_sweep 完全一致):
  数据  corrected 真值 + LOCF —— `build_rows_for_method("locf", (L,))`(逐地标 build,
        skeleton 含全行,单地标 build 即正确)。
  特征  正交轴 —— `build_feats_at_L` / `FEATS`(TSH 单列 + Hormone_load / T3T4_balance /
        Velocity_load / Velocity_balance 正交轴 + 10 个静态/基线/暴露)。
  模型  `ExplainableBoostingClassifier(interactions=5, max_interaction_bins=16,
        random_state=PY_SEED)`,dev@L 全量 final-fit(重要性是 dev 量,无需 OOF)。

逐地标取 `explain_global().data()` 的原生重要性(= mean |Δlog-odds contribution|),
含交互项(真名,交互标 ×),取 top5,整理成「特征 × 地标」迁移表:
  值 = mean|Δlog-odds|;每地标内标注 rank1–5。

注:fit 时传 DataFrame(列名=真名),故 explain_global 直出真名;仍引 `_resolve` 作为
  占位名回退保险(传 numpy array 才出 feature_NNNN)。

产物:
  results/module2_v2_vertical/m2v2_ebm_xland/tables/bin16_importance_top5.csv
  results/module2_v2_vertical/m2v2_ebm_xland/figures/bin16_importance_top5.png

硬约束:N=1003 人次 / 无 889 / time-safe(沿用 assert_no_future_feature)。
  重要性是 dev 量,不涉及 temporal,不写 worklog / 不 git commit。
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

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
from interpret.glassbox import ExplainableBoostingClassifier

from scripts.simple.module2_v2_shared import PY_SEED, forbidden_token
from scripts.simple.module2_v2_impute_experiment import build_rows_for_method
from scripts.simple.module2_v2_b4_ebm_axes import FEATS, build_feats_at_L
from scripts.simple.module2_v2_b4_ebm_oof import assert_no_future_feature
from scripts.simple.module2_v2_ebm_interaction_diag import _resolve

LANDMARKS = (1, 3, 6, 12)
MAX_INTERACTION_BINS = 16
INTERACTIONS = 5
TOP_K = 5
OUT = ROOT / "results" / "module2_v2_vertical" / "m2v2_ebm_xland"

# CJK 字体(无 tofu)
_CJK = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
if Path(_CJK).exists():
    fm.fontManager.addfont(_CJK)
    plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False

# 轴组归类(用于迁移结论:当期水平 / 动量 / 腺体重 各在哪主导)
AXIS_GROUP = {
    "Hormone_load": "当期水平(load)",
    "TSH_current": "当期水平(load)",
    "T3T4_balance": "当期 T3/T4 平衡",
    "Velocity_load": "动量(velocity)",
    "TSH_velocity": "动量(velocity)",
    "Velocity_balance": "动量 T3/T4 平衡",
    "ThyroidW": "腺体重(Goiter)",
    "TRAb": "抗体",
    "TGAb": "抗体",
    "TPOAb": "抗体",
    "Sex": "基线/慢性",
    "FT4_0M": "基线/慢性",
    "TSH_0M": "基线/慢性",
    "log1p_DiseaseDuration_Months_Aug": "基线/慢性",
    "Uptake24h": "RAI 暴露",
    "HalfLife": "RAI 暴露",
}


def _label_term(name: str) -> str:
    """交互项(含 ' & ')标 ×;主效应原样返回。"""
    return f"{name} (×)" if " & " in name else name


def fit_importance_at_L(L: int) -> tuple[list[tuple[str, float]], float]:
    """逐地标 dev@L 全量 final-fit EBM(bin16),返回 (按重要性降序的 [(真名, score)],
    dev@L 阳性率)。score = explain_global 原生 mean |Δlog-odds contribution|。"""
    rows = build_rows_for_method("locf", (L,))  # 单地标 build 即正确(skeleton 含全行)
    assert rows["episode_id"].nunique() == 1003, "analysis unit must be 1003 人次"
    y = rows["Y_24M_NHRH"].values
    lm = rows["landmark"].values
    is_dev = (rows["Split"] == "Development").values
    devL = is_dev & (lm == L)

    feat_all = build_feats_at_L(rows, devL)   # 轴在 dev@L z-fit,应用到全行
    assert_no_future_feature(feat_all, L)     # time-safety gate(沿用)
    Xtr = feat_all.loc[devL]
    live = [c for c in FEATS if Xtr[c].std() > 1e-9]   # 丢常数列(0M 处 velocity)
    Xd = Xtr[live]                                      # 传 DataFrame → 真名直出
    yd = y[devL]

    ebm = ExplainableBoostingClassifier(
        interactions=INTERACTIONS,
        max_interaction_bins=MAX_INTERACTION_BINS,
        random_state=PY_SEED,
    )
    ebm.fit(Xd, yd)

    d = ebm.explain_global().data()
    # 占位名回退保险(DataFrame fit 时本已真名)
    pairs = [(_resolve(str(n), live), float(s)) for n, s in zip(d["names"], d["scores"])]
    pairs.sort(key=lambda t: -t[1])
    return pairs, float(np.mean(yd))


def main() -> None:
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)

    per_land_top: dict[int, list[tuple[str, float]]] = {}
    per_land_all: dict[int, dict[str, float]] = {}
    pos_rate: dict[int, float] = {}

    print(f"=== EBM 原生重要性 top{TOP_K} × 地标 (bin{MAX_INTERACTION_BINS}, "
          f"interactions={INTERACTIONS}, LOCF) ===", flush=True)
    for L in LANDMARKS:
        pairs, pr = fit_importance_at_L(L)
        per_land_all[L] = {n: s for n, s in pairs}
        per_land_top[L] = pairs[:TOP_K]
        pos_rate[L] = pr
        print(f"\n[{L}M]  dev 阳性率={pr:.3f}  top{TOP_K}:", flush=True)
        for r, (n, s) in enumerate(pairs[:TOP_K], 1):
            grp = AXIS_GROUP.get(n.split(" & ")[0], "交互") if " & " in n else AXIS_GROUP.get(n, "?")
            print(f"   rank{r}: {_label_term(n):<32} |Δlog-odds|={s:.4f}   [{grp}]", flush=True)

    # --- 迁移表:行=出现在任一地标 top5 的特征,列=各地标 ---
    feats_in_top = []
    for L in LANDMARKS:
        for n, _ in per_land_top[L]:
            if n not in feats_in_top:
                feats_in_top.append(n)

    # 每地标 rank 映射(仅 top5 内有 rank)
    rank_at: dict[int, dict[str, int]] = {}
    for L in LANDMARKS:
        rank_at[L] = {n: i + 1 for i, (n, _) in enumerate(per_land_top[L])}

    table_rows = []
    for n in feats_in_top:
        row = {"feature": _label_term(n),
               "axis_group": AXIS_GROUP.get(n.split(" & ")[0], "交互") if " & " in n
               else AXIS_GROUP.get(n, "")}
        for L in LANDMARKS:
            val = per_land_all[L].get(n, float("nan"))   # 全特征重要性(不止 top5)
            row[f"{L}M"] = round(val, 4) if np.isfinite(val) else float("nan")
            r = rank_at[L].get(n)
            row[f"{L}M_rank"] = int(r) if r is not None else ""
        table_rows.append(row)

    # 排序:按 4 地标重要性之和降序(展示主导特征在前)
    def _row_total(r):
        return np.nansum([r[f"{L}M"] for L in LANDMARKS])
    table_rows.sort(key=_row_total, reverse=True)

    cols = ["feature", "axis_group"]
    for L in LANDMARKS:
        cols += [f"{L}M", f"{L}M_rank"]
    tbl = pd.DataFrame(table_rows)[cols]
    csv_path = OUT / "tables" / "bin16_importance_top5.csv"
    tbl.to_csv(csv_path, index=False)
    print(f"\nSaved → {csv_path}", flush=True)
    print(tbl.to_string(index=False), flush=True)

    # 安全核验:无 forbidden token
    blob = tbl.to_csv(index=False)
    assert forbidden_token() not in blob, "forbidden unique-patient count leaked"

    # --- 热图:特征(行,top5 并集)× 地标(列),值=mean|Δlog-odds|,rank 角标 ---
    feat_labels = [_label_term(n) for n in feats_in_top]
    M = np.array([[per_land_all[L].get(n, np.nan) for L in LANDMARKS] for n in feats_in_top],
                 dtype=float)

    fig, ax = plt.subplots(figsize=(8.4, 0.52 * len(feats_in_top) + 2.2))
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad(color="#eeeeee")          # 该地标不存在该特征(如 0M velocity)→ 灰
    Mmask = np.ma.masked_invalid(M)
    im = ax.imshow(Mmask, aspect="auto", cmap=cmap, vmin=0.0,
                   vmax=float(np.nanmax(M)) if np.isfinite(np.nanmax(M)) else 1.0)
    ax.set_xticks(range(len(LANDMARKS)))
    ax.set_xticklabels([f"{L}M" for L in LANDMARKS], fontsize=11)
    ax.set_yticks(range(len(feats_in_top)))
    ax.set_yticklabels(feat_labels, fontsize=10)
    ax.set_xlabel("地标 (landmark)", fontsize=11)
    ax.set_title(f"EBM 原生重要性随地标迁移  (bin{MAX_INTERACTION_BINS} · LOCF · "
                 f"top{TOP_K} 并集 · N=1003 人次)\n值 = mean|Δlog-odds|;白字数字=该地标 top5 内 rank",
                 fontsize=12)
    # 单元格标注:重要性数值 + (若在该地标 top5)rank 角标
    vmax = float(np.nanmax(M)) if np.isfinite(np.nanmax(M)) else 1.0
    for i, n in enumerate(feats_in_top):
        for j, L in enumerate(LANDMARKS):
            v = M[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="#999999")
                continue
            r = rank_at[L].get(n)
            txt = f"{v:.3f}" + (f"\n#{r}" if r is not None else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                    color="white" if v > 0.55 * vmax else "black",
                    fontweight="bold" if r == 1 else "normal")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("mean |Δlog-odds|", fontsize=10)
    fig.tight_layout()
    png_path = OUT / "figures" / "bin16_importance_top5.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {png_path}", flush=True)
    print("[done] bin16 importance top5 × landmark.", flush=True)


if __name__ == "__main__":
    main()
