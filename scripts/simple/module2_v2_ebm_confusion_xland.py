#!/usr/bin/env python
"""M2 · task1 — 深挖混淆矩阵随地标 (1/3/6/12M) 的变迁 (bin16 口径)。

对每个地标，在 corrected 真值 + LOCF、EBM (interactions=5, max_interaction_bins=16,
即 bin16) 的 5 折 StratifiedGroupKFold dev OOF 概率上，用 dev OOF 上的 Youden's J
最优阈值 (同时报 0.5) 把概率二值化，然后：

  (1) 逐地标混淆矩阵 — TP/FP/TN/FN 计数 + Sens/Spec/PPV/NPV/Acc + FN率/FP率，
      在 Youden 与 0.5 两个阈值下；dev OOF (选阈/分析) 与 temporal (仅最终报告) 分开。
  (2) 误判亚型刻画 — 对比 FN(漏诊,真复发被判低危,临床最危险) / FP(误报) / TP / TN
      四组在关键特征上的均值 (+ FN−TN、FP−TP 的标准化差异 Cohen's d-like)，找 FN/FP
      集中区间；并用 silhouette + KMeans(k=2..4) 评估「误判样本 (FN∪FP) 能否概括成
      单一/少数易混淆亚型」。silhouette 低 → 诚实报告「错误弥散，无法概括成单一亚型」。
  (3) 跨地标变迁：
      (a) 混淆构成 (FN率/FP率/Sens/Spec/PPV) 随 1→3→6→12 的比例变化曲线；
      (b) 个体轨迹 — 同一 episode_id 在 1/3/6/12 的 dev OOF 对错演变，统计
          持续对 / 持续错 / 由错转对(信号积累越来越准) / 由对转错 / 混合 的人次与
          特征画像，并给整体趋势 (更对 vs 更错) 的量化 (末-首正确率、|p−Y| 斜率)。
      (c) 额外角度：
          · FN 特征签名是否跨地标稳定 (FN−TN 标准化差向量的地标间相关矩阵)；
          · 误判是否总是同一批人 (逐地标错例集合的 Jaccard churn 矩阵 + 持续错画像)；
          · 误判与决策边界的关系 (|p−thr| 在 误判 vs 正确 的分布；错例是否贴近 0.5/边界)；
          · 临床功能态 × 对错交叉 (Hyper/Normal/Hypo 下 FN/FP 的富集)。

口径 (硬约束)：corrected 真值 + LOCF；特征 = 正交轴 (build_feats_at_L 的 live 列)，
time-safe (ebm_oof 内含 assert_no_future_feature)；EBM bin16；dev OOF 选阈/分析，
temporal 仅最终报告。分析单元 = 治疗疗程，N = 1003 人次 (运行时计数，绝不报告唯一
患者数；禁用 token 由 forbidden_token 守门)。

产出 → results/module2_v2_vertical/  (dpi=150, 文件名前缀 m2v2_bin16_confusion_)
  m2v2_bin16_confusion_matrix.csv           逐地标混淆 (Youden & 0.5; dev & temporal)
  m2v2_bin16_confusion_subtype_profile.csv  FN/FP/TP/TN 四组特征均值 + 标准化差
  m2v2_bin16_confusion_separability.csv     误判池 silhouette / KMeans (各 k)
  m2v2_bin16_confusion_drift.csv            混淆构成跨地标变迁 (率)
  m2v2_bin16_confusion_trajectory.csv       个体对错轨迹模式计数 + 画像
  m2v2_bin16_confusion_churn_jaccard.csv    逐地标错例集合 Jaccard
  m2v2_bin16_confusion_fn_signature_corr.csv FN 特征签名跨地标相关
  m2v2_bin16_confusion_boundary.csv         |p−thr| 误判 vs 正确分布
  m2v2_bin16_confusion_state_cross.csv      功能态 × 对错交叉
  m2v2_bin16_confusion_summary.json         机器可读汇总
  m2v2_bin16_confusion_overview.png         混淆构成变迁 + 个体轨迹
  m2v2_bin16_confusion_subtypes.png         FN/FP 特征对比 + 误判可分性
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
import warnings
from collections import Counter
from itertools import combinations
from pathlib import Path

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
import matplotlib.font_manager as fm  # noqa: E402

for _fp in ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "/Library/Fonts/Arial Unicode.ttf"):
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
        break
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from scripts.simple.module2_v2_b4_xland_shared import (  # noqa: E402
    LANDMARKS,
    PROFILE_FEATS,
    build_tracking_long,
    feature_rows,
    youden_threshold,
)
from scripts.simple.module2_v2_shared import forbidden_token  # noqa: E402

OUT = ROOT / "results" / "module2_v2_vertical"
PFX = "m2v2_bin16_confusion_"

# 中文显示名 (沿用项目其它脚本的 DISP 习惯)
DISP = {
    "TSH_current": "当期TSH",
    "FT3_current": "当期FT3",
    "FT4_current": "当期FT4",
    "TSH_velocity": "TSH动量",
    "FT3_velocity": "FT3动量",
    "FT4_velocity": "FT4动量",
    "TRAb": "TRAb",
    "TGAb": "TGAb",
    "TPOAb": "TPOAb",
    "TSH_0M": "基线TSH",
    "FT4_0M": "基线FT4",
    "ThyroidW": "甲状腺重量",
    "log1p_DiseaseDuration_Months_Aug": "病程(log)",
    "Uptake24h": "24h摄取率",
    "HalfLife": "有效半衰期",
    "Sex": "性别",
}
# 子图/排序里重点展示的特征
KEY_FEATS = [
    "TSH_current", "FT4_current", "FT3_current", "TSH_velocity", "FT4_velocity",
    "TRAb", "ThyroidW", "Uptake24h", "HalfLife", "log1p_DiseaseDuration_Months_Aug",
]


# ---------------------------------------------------------------------------
# (1) 混淆矩阵
# ---------------------------------------------------------------------------


def _confusion(y: np.ndarray, pred_label: np.ndarray) -> dict:
    tp = int(((pred_label == 1) & (y == 1)).sum())
    fp = int(((pred_label == 1) & (y == 0)).sum())
    tn = int(((pred_label == 0) & (y == 0)).sum())
    fn = int(((pred_label == 0) & (y == 1)).sum())
    n = tp + fp + tn + fn
    npos = tp + fn
    nneg = tn + fp
    sens = tp / npos if npos else float("nan")
    spec = tn / nneg if nneg else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    acc = (tp + tn) / n if n else float("nan")
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn, "N": n,
        # FN率 = 漏诊率 = FN / 真复发数 = 1 − Sens；FP率 = 误报率 = FP / 真未复发 = 1 − Spec
        "FN_rate": round(fn / npos, 4) if npos else float("nan"),
        "FP_rate": round(fp / nneg, 4) if nneg else float("nan"),
        "Sens": round(sens, 4), "Spec": round(spec, 4),
        "PPV": round(ppv, 4), "NPV": round(npv, 4), "Acc": round(acc, 4),
    }


def confusion_table(long_df: pd.DataFrame, landmarks) -> pd.DataFrame:
    """逐地标混淆矩阵；两个阈值 (Youden@OOF, 0.5) × 两个 split (dev OOF, temporal)。"""
    recs = []
    for split in ("Development", "Temporal"):
        sub = long_df[long_df["Split"] == split]
        for L in sorted(landmarks):
            g = sub[sub["landmark"] == L]
            if g.empty:
                continue
            thr = float(g["thr"].iloc[0])
            for cut_name, cut in (("Youden@OOF", thr), ("0.5", 0.5)):
                lab = (g["p"].values >= cut).astype(int)
                recs.append({
                    "split": "Dev(OOF)" if split == "Development" else "Temporal",
                    "Landmark": f"{L}M", "_L": L, "cut": cut_name, "thr": round(cut, 3),
                    **_confusion(g["Y"].values, lab),
                })
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------
# (2) FN / FP / TP / TN 四组特征画像 + 标准化差
# ---------------------------------------------------------------------------


def _cell(g: pd.DataFrame) -> dict[str, np.ndarray]:
    """用 Youden 阈值把一组 (单地标单 split) 划成 FN/FP/TP/TN 特征子帧。"""
    lab = g["pred_label"].values
    y = g["Y"].values
    return {
        "FN": g[(lab == 0) & (y == 1)],
        "FP": g[(lab == 1) & (y == 0)],
        "TP": g[(lab == 1) & (y == 1)],
        "TN": g[(lab == 0) & (y == 0)],
    }


def subtype_profile(merged: pd.DataFrame, landmarks) -> pd.DataFrame:
    """每地标 (dev OOF) 四组特征均值 + 关键标准化差。

    FN_vs_TN_d / FP_vs_TP_d = (mean_err − mean_correct_same_class) / pooled_sd，
    用「同真值类别」做对照：FN 对照 TN(都是…其实 FN 真值=1，临床上最有意义的对照是
    它本应像 TP；这里同时给 FN−TP 与 FN−TN）。为简洁主报 FN−TP(漏诊 vs 抓到的复发)
    与 FP−TN(误报 vs 正确排除)。
    """
    recs = []
    for L in sorted(landmarks):
        g = merged[(merged["landmark"] == L) & (merged["Split"] == "Development")]
        cells = _cell(g)
        for f in PROFILE_FEATS:
            allv = g[f].values.astype(float)
            sd = np.nanstd(allv) or 1.0

            def m(name):
                v = cells[name][f].values.astype(float)
                return float(np.nanmean(v)) if len(v) else float("nan")

            fn_m, fp_m, tp_m, tn_m = m("FN"), m("FP"), m("TP"), m("TN")
            recs.append({
                "Landmark": f"{L}M", "_L": L, "feature": DISP.get(f, f), "_feat": f,
                "FN_mean": round(fn_m, 3), "FP_mean": round(fp_m, 3),
                "TP_mean": round(tp_m, 3), "TN_mean": round(tn_m, 3),
                # FN vs TP：漏诊的真复发 与 被抓到的真复发 差多少 (signal 弱在哪)
                "FN_vs_TP_d": round((fn_m - tp_m) / sd, 3),
                # FP vs TN：误报的真阴 与 正确排除的真阴 差多少 (为何被误报)
                "FP_vs_TN_d": round((fp_m - tn_m) / sd, 3),
                "n_FN": int(len(cells["FN"])), "n_FP": int(len(cells["FP"])),
                "n_TP": int(len(cells["TP"])), "n_TN": int(len(cells["TN"])),
            })
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------
# (2b) 误判可分性 — silhouette + KMeans (诚实判定能否概括成单一亚型)
# ---------------------------------------------------------------------------

SIL_GATE = 0.25  # 预注册门槛：silhouette ≥ 0.25 才认为误判成簇/可概括


def separability(merged: pd.DataFrame, landmarks) -> tuple[pd.DataFrame, dict]:
    """逐地标对误判池 (FN∪FP, dev OOF) 做标准化 KMeans(k=2..4)，报 silhouette。

    silhouette 低 (<SIL_GATE) → 误判在特征空间不成簇，无法概括成单一/少数易混淆亚型。
    另外报 FN 与 FP 是否本身就是两簇 (用 真实 FN/FP 标签做 silhouette，看误判两大类
    在特征上是否可分)。
    """
    recs = []
    verdict = {}
    feats = PROFILE_FEATS
    for L in sorted(landmarks):
        g = merged[(merged["landmark"] == L) & (merged["Split"] == "Development")]
        cells = _cell(g)
        err = pd.concat([cells["FN"], cells["FP"]], axis=0)
        if len(err) < 6:
            continue
        X = err[feats].values.astype(float)
        X = np.nan_to_num(X, nan=np.nanmean(X))
        Xs = StandardScaler().fit_transform(X)
        # FN/FP 天然两类的可分性 (label silhouette)
        fnfp = np.r_[np.zeros(len(cells["FN"])), np.ones(len(cells["FP"]))]
        sil_fnfp = (float(silhouette_score(Xs, fnfp))
                    if len(np.unique(fnfp)) > 1 and len(Xs) > 2 else float("nan"))
        best = {"k": None, "sil": -1.0}
        for k in (2, 3, 4):
            if len(Xs) <= k:
                continue
            km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)
            if len(np.unique(km.labels_)) < 2:
                continue
            sil = float(silhouette_score(Xs, km.labels_))
            sizes = sorted(Counter(km.labels_.tolist()).values(), reverse=True)
            recs.append({
                "Landmark": f"{L}M", "_L": L, "k": k, "silhouette": round(sil, 4),
                "n_err": int(len(Xs)), "cluster_sizes": "/".join(map(str, sizes)),
                "sil_FN_vs_FP_label": round(sil_fnfp, 4),
            })
            if sil > best["sil"]:
                best = {"k": k, "sil": sil}
        verdict[L] = {
            "best_k": best["k"], "best_silhouette": round(best["sil"], 4),
            "sil_FN_vs_FP_label": round(sil_fnfp, 4), "n_err": int(len(Xs)),
            "characterizable": bool(best["sil"] >= SIL_GATE),
        }
    return pd.DataFrame(recs), verdict


# ---------------------------------------------------------------------------
# (3a) 混淆构成跨地标变迁 (率)
# ---------------------------------------------------------------------------


def drift_table(conf: pd.DataFrame) -> pd.DataFrame:
    """从混淆表抽 Youden 口径的率，按 split × landmark 给出变迁 (长→宽友好的长表)。"""
    sub = conf[conf["cut"] == "Youden@OOF"].copy()
    keep = ["split", "Landmark", "_L", "FN_rate", "FP_rate", "Sens", "Spec",
            "PPV", "NPV", "Acc", "FN", "FP", "TP", "TN"]
    return sub[keep].sort_values(["split", "_L"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# (3b) 个体对错轨迹 (同 episode 跨地标)
# ---------------------------------------------------------------------------


def trajectory(long_df: pd.DataFrame, merged: pd.DataFrame, landmarks):
    """同 episode 在各地标的 dev OOF 对错序列 → 模式分类 + 画像 + 趋势量化。

    模式 (只用该 episode 被观测到的地标，本设计每 episode 在 4 个地标都有行)：
      persistent-correct  全对
      persistent-wrong    全错
      turned-correct      由错转对 (首错末对；信号积累越来越准)
      turned-wrong        由对转错 (首对末错)
      mixed               其余反复
    趋势量化：末地标正确率 − 首地标正确率；每 episode |p−Y| 对地标的线性斜率均值。
    """
    Ls = sorted(landmarks)
    dev = long_df[long_df["Split"] == "Development"].copy()
    # 透视成 episode × landmark 的 correct / p / Y
    piv_c = dev.pivot(index="episode_id", columns="landmark", values="correct")
    piv_p = dev.pivot(index="episode_id", columns="landmark", values="p")
    y_ep = dev.drop_duplicates("episode_id").set_index("episode_id")["Y"]
    piv_c = piv_c[Ls].dropna()
    piv_p = piv_p.loc[piv_c.index, Ls]

    def classify(seq):
        s = [int(x) for x in seq]
        if all(v == 1 for v in s):
            return "persistent-correct"
        if all(v == 0 for v in s):
            return "persistent-wrong"
        if s[0] == 0 and s[-1] == 1:
            return "turned-correct"
        if s[0] == 1 and s[-1] == 0:
            return "turned-wrong"
        return "mixed"

    pat = piv_c.apply(lambda r: classify(r.values), axis=1)
    cnt = Counter(pat.tolist())
    n_tot = len(pat)

    # 每模式的特征画像 (取首地标 features，作为「这批人长什么样」的稳定描述)
    L0 = Ls[0]
    fr0 = merged[(merged["landmark"] == L0) & (merged["Split"] == "Development")]
    fr0 = fr0.set_index("episode_id")
    pat_recs = []
    for p_name in ("persistent-correct", "persistent-wrong", "turned-correct",
                   "turned-wrong", "mixed"):
        eids = pat[pat == p_name].index
        rec = {"pattern": p_name, "n": int(len(eids)),
               "share": round(len(eids) / n_tot, 4) if n_tot else 0.0,
               "Y_relapse_rate": round(float(y_ep.loc[eids].mean()), 3) if len(eids) else float("nan")}
        for f in KEY_FEATS:
            v = fr0.loc[fr0.index.intersection(eids), f].values.astype(float)
            rec[DISP.get(f, f)] = round(float(np.nanmean(v)), 3) if len(v) else float("nan")
        pat_recs.append(rec)
    pat_df = pd.DataFrame(pat_recs)

    # 趋势量化
    correct_by_L = {L: float(dev[dev["landmark"] == L]["correct"].mean()) for L in Ls}
    abs_err_by_L = {L: float((dev[dev["landmark"] == L]["p"]
                              - dev[dev["landmark"] == L]["Y"]).abs().mean()) for L in Ls}
    # 每 episode |p−Y| vs landmark 斜率 (>0 = 越来越差)
    abs_err = (piv_p.values - y_ep.loc[piv_c.index].values.reshape(-1, 1)).__abs__()
    xs = np.array(Ls, dtype=float)
    xc = xs - xs.mean()
    slopes = (abs_err - abs_err.mean(axis=1, keepdims=True)) @ xc / (xc @ xc)
    trend = {
        "correct_rate_by_landmark": {f"{L}M": round(correct_by_L[L], 4) for L in Ls},
        "abs_err_by_landmark": {f"{L}M": round(abs_err_by_L[L], 4) for L in Ls},
        "delta_correct_last_minus_first": round(correct_by_L[Ls[-1]] - correct_by_L[Ls[0]], 4),
        "delta_abs_err_last_minus_first": round(abs_err_by_L[Ls[-1]] - abs_err_by_L[Ls[0]], 4),
        "mean_abs_err_slope_per_month": round(float(np.mean(slopes)), 5),
        "frac_episodes_improving": round(float((slopes < 0).mean()), 4),
        "frac_episodes_worsening": round(float((slopes > 0).mean()), 4),
        "n_episodes_tracked": int(n_tot),
        "pattern_counts": {k: int(v) for k, v in cnt.items()},
        "overall_direction": ("更对(随地标正确率上升)"
                              if correct_by_L[Ls[-1]] > correct_by_L[Ls[0]] else "更错或持平"),
    }
    return pat_df, trend, (piv_c, piv_p, y_ep)


# ---------------------------------------------------------------------------
# (3c-i) FN 特征签名跨地标相关 (是否稳定)
# ---------------------------------------------------------------------------


def fn_signature_corr(prof: pd.DataFrame, landmarks) -> tuple[pd.DataFrame, dict]:
    """FN−TP 标准化差向量 (按特征) 在各地标间的 Pearson 相关矩阵。

    高相关 = 「漏诊的复发相对被抓到的复发，弱在哪些特征」这个签名跨地标稳定。
    """
    Ls = sorted(landmarks)
    mat = prof.pivot(index="_feat", columns="_L", values="FN_vs_TP_d")
    mat = mat[Ls]
    corr = mat.corr(method="pearson")
    corr.index = [f"{L}M" for L in corr.index]
    corr.columns = [f"{L}M" for L in corr.columns]
    pairs = [corr.iloc[i, j] for i, j in combinations(range(len(Ls)), 2)]
    summ = {"mean_pairwise_corr": round(float(np.nanmean(pairs)), 4),
            "min_pairwise_corr": round(float(np.nanmin(pairs)), 4),
            "stable": bool(np.nanmean(pairs) >= 0.5)}
    out = corr.reset_index().rename(columns={"index": "FN_vs_TP_d_corr"})
    return out, summ


# ---------------------------------------------------------------------------
# (3c-ii) 误判 churn — 逐地标错例集合的 Jaccard (是否总是同一批人)
# ---------------------------------------------------------------------------


def churn_jaccard(long_df: pd.DataFrame, landmarks) -> tuple[pd.DataFrame, dict]:
    Ls = sorted(landmarks)
    dev = long_df[long_df["Split"] == "Development"]
    err_sets = {L: set(dev[(dev["landmark"] == L) & (dev["correct"] == 0)]["episode_id"])
                for L in Ls}
    recs = []
    for La in Ls:
        row = {"Landmark": f"{La}M"}
        for Lb in Ls:
            a, b = err_sets[La], err_sets[Lb]
            u = len(a | b)
            row[f"{Lb}M"] = round(len(a & b) / u, 3) if u else float("nan")
        recs.append(row)
    # 「持续错」集合 = 在全部地标都错的 episode
    always_wrong = set.intersection(*err_sets.values()) if err_sets else set()
    union_wrong = set.union(*err_sets.values()) if err_sets else set()
    offdiag = [recs[i][f"{Lb}M"] for i, La in enumerate(Ls)
               for Lb in Ls if La != Lb]
    summ = {
        "mean_offdiag_jaccard": round(float(np.nanmean(offdiag)), 4),
        "n_always_wrong": int(len(always_wrong)),
        "n_ever_wrong": int(len(union_wrong)),
        "frac_always_of_ever": round(len(always_wrong) / len(union_wrong), 4) if union_wrong else float("nan"),
    }
    return pd.DataFrame(recs), summ, always_wrong


# ---------------------------------------------------------------------------
# (3c-iii) 误判与决策边界关系 (|p − thr|)
# ---------------------------------------------------------------------------


def boundary_table(long_df: pd.DataFrame, landmarks) -> tuple[pd.DataFrame, dict]:
    """误判 vs 正确 的 |p−thr| 分布 (dev OOF)：错例是否贴近决策边界。"""
    recs = []
    near_frac = []
    for L in sorted(landmarks):
        g = long_df[(long_df["landmark"] == L) & (long_df["Split"] == "Development")]
        thr = float(g["thr"].iloc[0])
        d = (g["p"] - thr).abs()
        wrong = d[g["correct"] == 0]
        right = d[g["correct"] == 1]
        # 「贴近边界」= |p−thr| < 0.10
        near = (d < 0.10)
        nf_wrong = float(near[g["correct"] == 0].mean()) if (g["correct"] == 0).any() else float("nan")
        nf_right = float(near[g["correct"] == 1].mean()) if (g["correct"] == 1).any() else float("nan")
        near_frac.append(nf_wrong)
        recs.append({
            "Landmark": f"{L}M", "_L": L, "thr": round(thr, 3),
            "wrong_mean_dist": round(float(wrong.mean()), 4) if len(wrong) else float("nan"),
            "right_mean_dist": round(float(right.mean()), 4) if len(right) else float("nan"),
            "wrong_median_dist": round(float(wrong.median()), 4) if len(wrong) else float("nan"),
            "frac_wrong_within_0.10": round(nf_wrong, 4),
            "frac_right_within_0.10": round(nf_right, 4),
            "n_wrong": int(len(wrong)), "n_right": int(len(right)),
        })
    summ = {"mean_frac_wrong_near_boundary": round(float(np.nanmean(near_frac)), 4)}
    return pd.DataFrame(recs), summ


# ---------------------------------------------------------------------------
# (3c-iv) 功能态 × 对错交叉
# ---------------------------------------------------------------------------


def state_cross(long_df: pd.DataFrame, landmarks) -> pd.DataFrame:
    recs = []
    for L in sorted(landmarks):
        g = long_df[(long_df["landmark"] == L) & (long_df["Split"] == "Development")]
        lab = g["pred_label"].values
        y = g["Y"].values
        for st in ("Hyper", "Normal", "Hypo", "Unknown"):
            m = g["state"].values == st
            if not m.any():
                continue
            yy, ll = y[m], lab[m]
            tp = int(((ll == 1) & (yy == 1)).sum())
            fn = int(((ll == 0) & (yy == 1)).sum())
            tn = int(((ll == 0) & (yy == 0)).sum())
            fp = int(((ll == 1) & (yy == 0)).sum())
            npos = tp + fn
            nneg = tn + fp
            recs.append({
                "Landmark": f"{L}M", "_L": L, "state": st, "n": int(m.sum()),
                "relapse_rate": round(float(yy.mean()), 3),
                "FN": fn, "FP": fp, "TP": tp, "TN": tn,
                "FN_rate(missed/relapse)": round(fn / npos, 3) if npos else float("nan"),
                "FP_rate(alarm/nonrelapse)": round(fp / nneg, 3) if nneg else float("nan"),
            })
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------


def plot_overview(drift: pd.DataFrame, conf: pd.DataFrame, pat_df: pd.DataFrame,
                  trend: dict, churn: pd.DataFrame, landmarks):
    Ls = sorted(landmarks)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))

    # (A) 混淆率跨地标变迁 (dev OOF, Youden)
    ax = axes[0, 0]
    dd = drift[drift["split"] == "Dev(OOF)"].sort_values("_L")
    for col, lab, c, m in (("FN_rate", "FN率(漏诊=1−Sens)", "#c1121f", "o"),
                           ("FP_rate", "FP率(误报=1−Spec)", "#1d4e89", "s"),
                           ("PPV", "PPV(阳性预测值)", "#2a9d8f", "^"),
                           ("Acc", "准确率", "#6c757d", "D")):
        ax.plot(dd["_L"], dd[col], marker=m, color=c, label=lab, lw=2)
    ax.set_xticks(Ls); ax.set_xticklabels([f"{L}M" for L in Ls])
    ax.set_ylim(0, 1); ax.set_xlabel("地标"); ax.set_ylabel("率")
    ax.set_title("(A) 混淆构成跨地标变迁 (dev OOF, Youden@OOF)", fontsize=11)
    ax.legend(fontsize=8, loc="best"); ax.grid(alpha=0.25)

    # (B) dev vs temporal 的 Sens/Spec 跨地标 (Youden)
    ax = axes[0, 1]
    for split, ls in (("Dev(OOF)", "-"), ("Temporal", "--")):
        s = drift[drift["split"] == split].sort_values("_L")
        ax.plot(s["_L"], s["Sens"], marker="o", ls=ls, color="#c1121f",
                label=f"Sens {split}", lw=1.8)
        ax.plot(s["_L"], s["Spec"], marker="s", ls=ls, color="#1d4e89",
                label=f"Spec {split}", lw=1.8)
    ax.set_xticks(Ls); ax.set_xticklabels([f"{L}M" for L in Ls])
    ax.set_ylim(0, 1); ax.set_xlabel("地标"); ax.set_ylabel("率")
    ax.set_title("(B) 敏感度/特异度：dev OOF vs temporal", fontsize=11)
    ax.legend(fontsize=7.5, ncol=2, loc="lower center"); ax.grid(alpha=0.25)

    # (C) 个体对错轨迹模式占比 (dev OOF)
    ax = axes[1, 0]
    order = ["persistent-correct", "turned-correct", "mixed", "turned-wrong",
             "persistent-wrong"]
    zh = {"persistent-correct": "持续对", "turned-correct": "由错转对",
          "mixed": "反复(mixed)", "turned-wrong": "由对转错",
          "persistent-wrong": "持续错"}
    cols = {"persistent-correct": "#2a9d8f", "turned-correct": "#80b918",
            "mixed": "#adb5bd", "turned-wrong": "#f3722c", "persistent-wrong": "#9d0208"}
    pp = pat_df.set_index("pattern").reindex(order)
    bars = ax.bar([zh[o] for o in order], pp["share"].values,
                  color=[cols[o] for o in order])
    for b, n in zip(bars, pp["n"].values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                f"{int(n)}人次", ha="center", fontsize=8)
    ax.set_ylabel("占比 (dev episodes)")
    ax.set_title(f"(C) 个体对错轨迹模式 (同 episode 跨 {'/'.join(f'{L}' for L in Ls)}M)\n"
                 f"首→末正确率 {trend['correct_rate_by_landmark'][f'{Ls[0]}M']}→"
                 f"{trend['correct_rate_by_landmark'][f'{Ls[-1]}M']}  "
                 f"(Δ={trend['delta_correct_last_minus_first']:+.3f}, {trend['overall_direction']})",
                 fontsize=10)
    ax.grid(alpha=0.2, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=12)

    # (D) 错例 churn Jaccard 热图
    ax = axes[1, 1]
    M = churn.set_index("Landmark")[[f"{L}M" for L in Ls]].values.astype(float)
    im = ax.imshow(M, cmap="OrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(Ls))); ax.set_xticklabels([f"{L}M" for L in Ls])
    ax.set_yticks(range(len(Ls))); ax.set_yticklabels([f"{L}M" for L in Ls])
    for i in range(len(Ls)):
        for j in range(len(Ls)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] > 0.5 else "black", fontsize=9)
    ax.set_title("(D) 错例集合 Jaccard (dev OOF)\n高=同一批人持续错；低=错例换人", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("M2 EBM(bin16) 混淆矩阵跨地标变迁 · corrected+LOCF · N=1003 人次",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT / f"{PFX}overview.png", dpi=150)
    plt.close(fig)


def plot_subtypes(prof: pd.DataFrame, sep_df: pd.DataFrame, sep_verdict: dict,
                  landmarks):
    Ls = sorted(landmarks)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))

    # (A) FN−TP & FP−TN 标准化差 (6M 主报；用全地标均值排序)，热图: 特征 × 地标
    ax = axes[0]
    sub = prof[prof["_feat"].isin(KEY_FEATS)].copy()
    piv = sub.pivot(index="_feat", columns="_L", values="FN_vs_TP_d").reindex(KEY_FEATS)[Ls]
    M = piv.values.astype(float)
    vmax = np.nanmax(np.abs(M)) or 1.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(Ls))); ax.set_xticklabels([f"{L}M" for L in Ls])
    ax.set_yticks(range(len(KEY_FEATS)))
    ax.set_yticklabels([DISP.get(f, f) for f in KEY_FEATS], fontsize=9)
    for i in range(len(KEY_FEATS)):
        for j in range(len(Ls)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                        color="white" if abs(M[i, j]) > vmax * 0.6 else "black", fontsize=8)
    ax.set_title("(A) FN−TP 标准化差 (漏诊的复发 相对 抓到的复发)\n"
                 "红=FN 在该特征更高, 蓝=更低 (dev OOF)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (B) 误判可分性 silhouette 跨地标
    ax = axes[1]
    for k, c in ((2, "#1d4e89"), (3, "#2a9d8f"), (4, "#9d4edd")):
        s = sep_df[sep_df["k"] == k].sort_values("_L")
        if not s.empty:
            ax.plot(s["_L"], s["silhouette"], marker="o", color=c, label=f"KMeans k={k}", lw=2)
    s2 = sep_df.drop_duplicates("_L").sort_values("_L")
    ax.plot(s2["_L"], s2["sil_FN_vs_FP_label"], marker="x", ls="--", color="#e85d04",
            label="FN vs FP 标签", lw=1.8)
    ax.axhline(SIL_GATE, ls=":", color="#555", lw=1.2)
    ax.text(Ls[0], SIL_GATE + 0.01, f"可概括门槛 {SIL_GATE}", fontsize=8, color="#555")
    ax.axhline(0, ls="-", color="#bbb", lw=0.8)
    ax.set_xticks(Ls); ax.set_xticklabels([f"{L}M" for L in Ls])
    ax.set_xlabel("地标"); ax.set_ylabel("silhouette")
    char = [L for L, v in sep_verdict.items() if v["characterizable"]]
    ax.set_title(f"(B) 误判池(FN∪FP)可分性 silhouette\n"
                 f"达标地标: {('无' if not char else '/'.join(f'{L}M' for L in char))} "
                 f"→ {'可概括' if char else '错误弥散,无法概括成单一亚型'}", fontsize=10)
    ax.legend(fontsize=8, loc="best"); ax.grid(alpha=0.25)

    fig.suptitle("M2 EBM(bin16) 误判亚型刻画与可分性 · corrected+LOCF · N=1003 人次",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / f"{PFX}subtypes.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke: 6M only")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    long_df, meta = build_tracking_long(quick=args.quick)
    landmarks = meta["landmarks"]
    feat_df = feature_rows(quick=args.quick)
    merged = long_df.merge(feat_df, on=["episode_id", "landmark"], how="left",
                           suffixes=("", "_f"))

    # 守门：禁用 token 不得出现在任何形状/数据里
    n_episodes = long_df["episode_id"].nunique()  # 这是 episode(人次) 数，不是患者数
    assert forbidden_token() not in str(long_df.shape), "forbidden token in shape"

    # (1)
    conf = confusion_table(long_df, landmarks)
    conf.drop(columns=["_L"]).to_csv(OUT / f"{PFX}matrix.csv", index=False)

    # (2)
    prof = subtype_profile(merged, landmarks)
    prof.drop(columns=["_L", "_feat"]).to_csv(OUT / f"{PFX}subtype_profile.csv", index=False)

    # (2b)
    sep_df, sep_verdict = separability(merged, landmarks)
    if not sep_df.empty:
        sep_df.drop(columns=["_L"]).to_csv(OUT / f"{PFX}separability.csv", index=False)

    # (3a)
    drift = drift_table(conf)
    drift.drop(columns=["_L"]).to_csv(OUT / f"{PFX}drift.csv", index=False)

    # (3b)
    pat_df, trend, _piv = trajectory(long_df, merged, landmarks)
    pat_df.to_csv(OUT / f"{PFX}trajectory.csv", index=False)

    # (3c)
    multi = len(landmarks) > 1
    if multi:
        fn_corr, fn_summ = fn_signature_corr(prof, landmarks)
        fn_corr.to_csv(OUT / f"{PFX}fn_signature_corr.csv", index=False)
        churn, churn_summ, always_wrong = churn_jaccard(long_df, landmarks)
        churn.to_csv(OUT / f"{PFX}churn_jaccard.csv", index=False)
    else:
        fn_corr, fn_summ = pd.DataFrame(), {}
        churn, churn_summ, always_wrong = pd.DataFrame(), {}, set()
    bnd, bnd_summ = boundary_table(long_df, landmarks)
    bnd.drop(columns=["_L"]).to_csv(OUT / f"{PFX}boundary.csv", index=False)
    stx = state_cross(long_df, landmarks)
    stx.drop(columns=["_L"]).to_csv(OUT / f"{PFX}state_cross.csv", index=False)

    # 图
    if multi:
        plot_overview(drift, conf, pat_df, trend, churn, landmarks)
    plot_subtypes(prof, sep_df, sep_verdict, landmarks)

    # ---- 汇总 json ----
    summary = {
        "口径": "corrected真值+LOCF; EBM interactions=5 max_interaction_bins=16(bin16); "
              "dev 5-fold StratifiedGroupKFold OOF 选阈/分析, temporal 仅最终报告; time-safe",
        "analysis_unit": "treatment-episode(疗程/人次)",
        "n_episodes": int(n_episodes),
        "landmarks": [f"{L}M" for L in landmarks],
        "youden_thr": {f"{L}M": round(float(meta['thr'][L]), 4) for L in landmarks},
        "auc": {f"{L}M": meta["auc"][L] for L in landmarks},
        "confusion_youden_dev": conf[(conf["cut"] == "Youden@OOF") & (conf["split"] == "Dev(OOF)")]
        .drop(columns=["_L"]).to_dict(orient="records"),
        "confusion_youden_temporal": conf[(conf["cut"] == "Youden@OOF") & (conf["split"] == "Temporal")]
        .drop(columns=["_L"]).to_dict(orient="records"),
        "separability_verdict": {f"{L}M": v for L, v in sep_verdict.items()},
        "trajectory_trend": trend,
        "trajectory_patterns": pat_df.to_dict(orient="records"),
        "fn_signature_corr": fn_summ,
        "error_churn": churn_summ,
        "boundary": bnd_summ,
    }
    (OUT / f"{PFX}summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # ---- 控制台报告 ----
    print(f"\n=== N = {n_episodes} 人次 × {len(landmarks)} 地标 ; bin16 ; EBM-OOF ===", flush=True)
    print("\n=== (1) 逐地标混淆矩阵 (Youden@OOF) ===", flush=True)
    print(conf[conf["cut"] == "Youden@OOF"].drop(columns=["_L", "cut"]).to_string(index=False), flush=True)

    print("\n=== (2) FN/FP/TP/TN 关键特征画像 (dev OOF, 节选 6M) ===", flush=True)
    L_show = 6 if 6 in landmarks else landmarks[0]
    sl = prof[(prof["_L"] == L_show) & (prof["_feat"].isin(KEY_FEATS))]
    print(sl.drop(columns=["_L", "_feat"]).to_string(index=False), flush=True)

    print("\n=== (2b) 误判池 (FN∪FP) 可分性 (诚实判定) ===", flush=True)
    for L, v in sep_verdict.items():
        print(f"  {L}M: best_k={v['best_k']} silhouette={v['best_silhouette']} "
              f"(门槛{SIL_GATE}) | FN-vs-FP标签 sil={v['sil_FN_vs_FP_label']} "
              f"| n_err={v['n_err']} | 可概括={v['characterizable']}", flush=True)

    print("\n=== (3a) 混淆构成跨地标变迁 (dev OOF, Youden) ===", flush=True)
    print(drift[drift["split"] == "Dev(OOF)"].drop(columns=["_L", "FN", "FP", "TP", "TN"])
          .to_string(index=False), flush=True)

    print("\n=== (3b) 个体对错轨迹 (dev OOF, 同 episode 跨地标) ===", flush=True)
    print(pat_df[["pattern", "n", "share", "Y_relapse_rate", "当期TSH", "甲状腺重量", "TRAb"]]
          .to_string(index=False), flush=True)
    print(f"  趋势: 首→末正确率 {trend['delta_correct_last_minus_first']:+.3f} | "
          f"|p−Y|斜率/月 {trend['mean_abs_err_slope_per_month']:+.5f} | "
          f"改善 episode 占比 {trend['frac_episodes_improving']} vs 变差 {trend['frac_episodes_worsening']} | "
          f"方向: {trend['overall_direction']}", flush=True)

    if multi:
        print("\n=== (3c-i) FN−TP 特征签名跨地标相关 ===", flush=True)
        print(f"  平均成对相关 {fn_summ['mean_pairwise_corr']} (min {fn_summ['min_pairwise_corr']}) "
              f"→ 跨地标稳定={fn_summ['stable']}", flush=True)
        print("\n=== (3c-ii) 错例 churn Jaccard ===", flush=True)
        print(f"  平均非对角 Jaccard {churn_summ['mean_offdiag_jaccard']} | "
              f"全地标都错 {churn_summ['n_always_wrong']} 人次 / 曾错 {churn_summ['n_ever_wrong']} "
              f"({churn_summ['frac_always_of_ever']} 持续错占比)", flush=True)
    print("\n=== (3c-iii) 误判与决策边界 |p−thr| ===", flush=True)
    print(bnd.drop(columns=["_L"]).to_string(index=False), flush=True)
    print(f"  误判落在边界 ±0.10 内的平均占比 {bnd_summ['mean_frac_wrong_near_boundary']}", flush=True)
    print("\n=== (3c-iv) 功能态 × 对错交叉 (dev OOF) ===", flush=True)
    print(stx.drop(columns=["_L"]).to_string(index=False), flush=True)

    print(f"\nSaved → {OUT} (前缀 {PFX})", flush=True)


if __name__ == "__main__":
    main()
