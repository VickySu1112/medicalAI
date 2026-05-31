#!/usr/bin/env python
"""M2 · v2 — bin16 口径下 EBM「prune 到 ~12 项」实验(full vs pruned)。

问题:bin16 full(全 16 个 FEATS 主效应 + 自动 5 个交互 = 21 项)能否被裁到 ~12 项
(主效应 + 指定交互)而 **temporal AUC 不掉**?用配对 bootstrap 的 ΔAUC(pruned−full)
95% CI 是否含 0 来判定。

口径(与 module2_v2_ebm_bins_sweep.py / impute 实验一致):
  数据      —— corrected 真值 + LOCF:`build_rows_for_method("locf", (L,))`(逐地标)。
  特征      —— `build_feats_at_L` / `FEATS`(正交轴);prune = FEATS 子集。
  地标      —— 1 / 3 / 6 / 12M。
  OOF 协议  —— dev 5 折 StratifiedGroupKFold(by episode_id, seed=CV_SEED=13)OOF;
              temporal final dev-fit。EBM:`max_interaction_bins=16, random_state=PY_SEED`。
  配对检验  —— `module2_v2_shared.paired_episode_cluster_bootstrap_delta(..., split="Temporal")`。

两套配置:
  full   —— 主效应 = 全 16 个 FEATS;interactions=5(自动)→ 共 21 项(bin16 sweep 同口径)。
  pruned —— 主效应 = 8 个医学核心 + 指定 3 个交互 = 共 11 项(≈12)。
            主效应:ThyroidW, Hormone_load, TSH_current, Velocity_load, TRAb, TPOAb,
                    log1p_DiseaseDuration_Months_Aug, HalfLife(剂量学,二选一取 HalfLife)。
            交互  :ThyroidW×HalfLife(Marinelli 剂量学)、TPOAb×Velocity_load(桥本协同)、
                    Hormone_load×Velocity_load(水平×动量 momentum-beyond-inertia)。
            去掉  :Sex / T3T4_balance / Velocity_balance / FT4_0M / TSH_0M / TGAb /
                    Uptake24h / TSH_velocity 等弱/反直觉/后期冗余主效应,及非上述自动交互。

实现要点(interpret 0.7):`interactions=[(i,j),...]` 传**指定特征索引对**时,EBM 仍自动
拟合所传矩阵全部列的主效应 + 恰好这些交互对(已验证)。索引按所传特征矩阵列序解析。
地标 0M/1M 速度为常数会被 live 过滤丢弃;此处只跑 1/3/6/12M,且每地标对 pruned 交互对
按 live 列重映射索引、成员不 live 的交互自动剔除(time-safe 不变)。

产物:
  results/module2_v2_vertical/m2v2_ebm_xland/tables/bin16_prune12.csv
    列:landmark, full_temporal_AUC, pruned_temporal_AUC, full_n_terms, pruned_n_terms,
        dAUC_mean(pruned−full), dAUC_CI_low, dAUC_CI_high, CI_contains_0, no_loss
  (附)bin16_prune12_oof.csv —— 逐地标 OOF AUC(选择口径,仅供核对,不用于报告判定)
  bin16_prune12_summary.json

硬约束:1003 人次 / 无 889 / time-safe(沿用 assert_no_future_feature)/ OOF 选择、
  temporal 仅报告 / 不写 significant / 不写 worklog / 不 git commit。
"""
from __future__ import annotations

import os

# 单线程 BLAS(在导入 numpy/interpret 之前设好)。
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from scripts.simple.module2_v2_shared import (
    CV_SEED,
    PY_SEED,
    StackedData,
    forbidden_token,
    paired_episode_cluster_bootstrap_delta,
)
from scripts.simple.module2_v2_impute_experiment import build_rows_for_method
from scripts.simple.module2_v2_b4_ebm_axes import FEATS, build_feats_at_L
from scripts.simple.module2_v2_b4_ebm_oof import assert_no_future_feature

LANDMARKS = (1, 3, 6, 12)
MAX_INTERACTION_BINS = 16
FULL_INTERACTIONS = 5  # bin16 sweep / 主线 GA2M 同口径
OUT = ROOT / "results" / "module2_v2_vertical" / "m2v2_ebm_xland"

# --- pruned 配置(医学洞见;名字而非索引,便于审计与跨 live 重映射)---
PRUNED_MAINS = [
    "ThyroidW",                            # 甲状腺重量(goiter)
    "Hormone_load",                        # 当期综合甲功水平
    "TSH_current",                         # 当期 TSH(垂体反馈轴)
    "Velocity_load",                       # 综合变化速度(动量)
    "TRAb",                                # TSH 受体抗体(Graves 核心)
    "TPOAb",                               # 甲过氧化物酶抗体(桥本)
    "log1p_DiseaseDuration_Months_Aug",    # log 病程(慢性化)
    "HalfLife",                            # 有效碘半衰期(剂量学;二选一)
]
PRUNED_INTERACTIONS = [
    ("ThyroidW", "HalfLife"),       # Marinelli 剂量学:重量×半衰期 → 吸收剂量
    ("TPOAb", "Velocity_load"),     # 桥本协同:自身免疫×回落速度(跨地标稳定)
    ("Hormone_load", "Velocity_load"),  # 水平×动量(momentum-beyond-inertia)
]


def _roc(y, p) -> float:
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")


def _fit_oof_temporal(rows, y, lm, is_dev, L, *, feat_cols, interactions, seed=PY_SEED):
    """通用 EBM per-landmark:dev 5 折 SGKFold OOF + temporal final-fit。

    `feat_cols` 限定参与的主效应列(pruned 用子集);`interactions` 可为整数(自动)
    或指定索引对的列表(pruned)。指定对时索引按 **live 列序** 解析,故先算 live
    再把按名字给的对映射到 live 索引、剔除成员不 live 的对(0M/1M 速度常数场景)。
    所有配置统一 `max_interaction_bins=16`。
    """
    lm = np.asarray(lm)
    is_dev = np.asarray(is_dev)
    y = np.asarray(y)
    devL = is_dev & (lm == L)
    tstL = (~is_dev) & (lm == L)

    feat_all = build_feats_at_L(rows, devL)[list(feat_cols)]  # 轴在 dev@L z-fit,应用全行
    assert_no_future_feature(feat_all, L)                     # time-safety gate(沿用)

    Xtr_full = feat_all.loc[devL]
    live = [c for c in feat_all.columns if Xtr_full[c].std() > 1e-9]  # 丢常数(速度@早地标)
    feat_live = feat_all[live]
    live_idx = {c: i for i, c in enumerate(live)}

    # 指定交互:把按名字的对映射到 live 索引;成员缺失则剔除该对。
    if isinstance(interactions, list):
        inter_idx = []
        dropped_pairs = []
        for a, b in interactions:
            if a in live_idx and b in live_idx:
                inter_idx.append((live_idx[a], live_idx[b]))
            else:
                dropped_pairs.append((a, b))
        ebm_interactions = inter_idx
    else:
        ebm_interactions = interactions
        dropped_pairs = []

    Xd = feat_live.loc[devL].values
    yd = y[devL]
    epd = rows["episode_id"].values[devL]

    pred = np.zeros(len(rows), dtype=float)
    dev_idx = np.where(devL)[0]
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    for tr, va in skf.split(Xd, yd, groups=epd):
        ebm = ExplainableBoostingClassifier(
            random_state=seed, interactions=ebm_interactions,
            max_interaction_bins=MAX_INTERACTION_BINS,
        )
        ebm.fit(Xd[tr], yd[tr])
        pred[dev_idx[va]] = ebm.predict_proba(Xd[va])[:, 1]

    final_ebm = ExplainableBoostingClassifier(
        random_state=seed, interactions=ebm_interactions,
        max_interaction_bins=MAX_INTERACTION_BINS,
    )
    final_ebm.fit(Xd, yd)
    if tstL.any():
        pred[tstL] = final_ebm.predict_proba(feat_live.loc[tstL].values)[:, 1]

    n_terms = len(final_ebm.term_features_)
    n_mains = sum(1 for t in final_ebm.term_features_ if len(t) == 1)
    n_inters = sum(1 for t in final_ebm.term_features_ if len(t) == 2)
    return pred, {
        "n_terms": n_terms, "n_mains": n_mains, "n_inters": n_inters,
        "live": live, "dropped_pairs": dropped_pairs,
    }


def run_landmark(rows, y, lm, is_dev, L, n_boot):
    """单地标:full + pruned 各自 OOF/temporal,配对 ΔAUC(pruned−full)CI。"""
    sd_shim = StackedData(
        rows=rows,
        episode_meta=rows.drop_duplicates("episode_id")[["episode_id", "Split", "Y_24M_NHRH"]],
    )
    devL = is_dev & (np.asarray(lm) == L)
    tstL = (~is_dev) & (np.asarray(lm) == L)

    pred_full, info_full = _fit_oof_temporal(
        rows, y, lm, is_dev, L, feat_cols=FEATS, interactions=FULL_INTERACTIONS)
    pred_pru, info_pru = _fit_oof_temporal(
        rows, y, lm, is_dev, L, feat_cols=PRUNED_MAINS, interactions=PRUNED_INTERACTIONS)

    full_oof = _roc(np.asarray(y)[devL], pred_full[devL])
    full_tmp = _roc(np.asarray(y)[tstL], pred_full[tstL])
    pru_oof = _roc(np.asarray(y)[devL], pred_pru[devL])
    pru_tmp = _roc(np.asarray(y)[tstL], pred_pru[tstL])

    # 配对 bootstrap:Δ = pruned − full(同 resample 同时打分两向量),Temporal split。
    dm, dlo, dhi = paired_episode_cluster_bootstrap_delta(
        sd_shim, pred_pru, pred_full, landmark=L, split="Temporal", n=n_boot)
    contains0 = bool(dlo <= 0 <= dhi)

    return {
        "landmark": f"{L}M",
        "L": L,
        "full_OOF_AUC": round(float(full_oof), 4),
        "pruned_OOF_AUC": round(float(pru_oof), 4),
        "full_temporal_AUC": round(float(full_tmp), 4),
        "pruned_temporal_AUC": round(float(pru_tmp), 4),
        "full_n_terms": info_full["n_terms"],
        "full_n_mains": info_full["n_mains"],
        "full_n_inters": info_full["n_inters"],
        "pruned_n_terms": info_pru["n_terms"],
        "pruned_n_mains": info_pru["n_mains"],
        "pruned_n_inters": info_pru["n_inters"],
        "pruned_dropped_pairs": ["×".join(p) for p in info_pru["dropped_pairs"]],
        "dAUC_mean": round(float(dm), 4),
        "dAUC_CI_low": round(float(dlo), 4),
        "dAUC_CI_high": round(float(dhi), 4),
        "CI_contains_0": contains0,
        # 不掉性能 = ΔCI 含 0(无可检差异),或点估计 Δ≥0(pruned 不更差)。
        "no_loss": bool(contains0 or dm >= 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="smoke:单地标(6M),100 boot,不写汇总")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    landmarks = (6,) if args.quick else LANDMARKS
    n_boot = 100 if args.quick else args.boot

    print(f"[bin16 prune12] landmarks={landmarks}  n_boot={n_boot}", flush=True)
    print(f"  pruned mains({len(PRUNED_MAINS)}): {PRUNED_MAINS}", flush=True)
    print(f"  pruned inters({len(PRUNED_INTERACTIONS)}): "
          f"{['×'.join(p) for p in PRUNED_INTERACTIONS]}", flush=True)

    # corrected + LOCF;skeleton 含全行(LANDMARKS_X),build 一次即可覆盖全部地标。
    rows = build_rows_for_method("locf", LANDMARKS)
    assert rows["episode_id"].nunique() == 1003, "analysis unit must be 1003 人次"
    y = rows["Y_24M_NHRH"].values
    lm = rows["landmark"].values
    is_dev = (rows["Split"] == "Development").values

    results = []
    for L in landmarks:
        r = run_landmark(rows, y, lm, is_dev, L, n_boot)
        results.append(r)
        drop_note = (f" [dropped pairs: {r['pruned_dropped_pairs']}]"
                     if r["pruned_dropped_pairs"] else "")
        print(f"  [{r['landmark']:>3}] full_temporal={r['full_temporal_AUC']} "
              f"(terms={r['full_n_terms']}={r['full_n_mains']}m+{r['full_n_inters']}i)  "
              f"pruned_temporal={r['pruned_temporal_AUC']} "
              f"(terms={r['pruned_n_terms']}={r['pruned_n_mains']}m+{r['pruned_n_inters']}i)  "
              f"ΔAUC(pru−full)={r['dAUC_mean']:+.4f} "
              f"[{r['dAUC_CI_low']:+.4f},{r['dAUC_CI_high']:+.4f}] "
              f"CI∋0={r['CI_contains_0']} no_loss={r['no_loss']}{drop_note}", flush=True)

    if args.quick:
        print("[quick] smoke done (no summary written).", flush=True)
        return

    # --- 主表 bin16_prune12.csv ---
    main_cols = [
        "landmark", "full_temporal_AUC", "pruned_temporal_AUC",
        "full_n_terms", "pruned_n_terms",
        "dAUC_mean", "dAUC_CI_low", "dAUC_CI_high", "CI_contains_0", "no_loss",
    ]
    main_df = pd.DataFrame([{c: r[c] for c in main_cols} for r in results])
    main_csv = OUT / "tables" / "bin16_prune12.csv"
    main_df.to_csv(main_csv, index=False)
    print(f"\nSaved → {main_csv}", flush=True)
    print(main_df.to_string(index=False), flush=True)

    # --- 附:OOF 表(选择口径,仅核对)---
    oof_cols = ["landmark", "full_OOF_AUC", "pruned_OOF_AUC",
                "full_n_terms", "full_n_mains", "full_n_inters",
                "pruned_n_terms", "pruned_n_mains", "pruned_n_inters"]
    oof_df = pd.DataFrame([{c: r[c] for c in oof_cols} for r in results])
    oof_csv = OUT / "tables" / "bin16_prune12_oof.csv"
    oof_df.to_csv(oof_csv, index=False)
    print(f"Saved → {oof_csv}", flush=True)

    # --- 结论 ---
    n_no_loss = sum(r["no_loss"] for r in results)
    verdict = ("pruned ~12 项在所有地标均不掉性能(ΔAUC CI 含 0 或 Δ≥0)"
               if n_no_loss == len(results)
               else f"pruned 在 {n_no_loss}/{len(results)} 个地标不掉性能;"
                    "其余地标 ΔCI 偏负,需关注")
    summary = {
        "caliber": "corrected 真值 + LOCF; per-landmark EBM OOF(dev 5fold SGKFold seed13)"
                   "+ temporal final-fit; max_interaction_bins=16",
        "landmarks": [f"{L}M" for L in landmarks],
        "n_boot": n_boot,
        "n_episodes": 1003,
        "full_config": {"mains": FEATS, "interactions": FULL_INTERACTIONS,
                        "note": "全 16 主效应 + 5 自动交互(bin16 sweep 同口径)"},
        "pruned_config": {"mains": PRUNED_MAINS,
                          "interactions": ["×".join(p) for p in PRUNED_INTERACTIONS],
                          "n_terms_target": len(PRUNED_MAINS) + len(PRUNED_INTERACTIONS)},
        "dropped_mains": [f for f in FEATS if f not in PRUNED_MAINS],
        "per_landmark": results,
        "n_landmarks_no_loss": int(n_no_loss),
        "verdict": verdict,
    }
    sjson = OUT / "tables" / "bin16_prune12_summary.json"
    sjson.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    blob = json.dumps(summary, ensure_ascii=False)
    assert forbidden_token() not in blob, "forbidden unique-patient count leaked into summary"
    print(f"Saved → {sjson}", flush=True)
    print(f"\n[结论] {verdict}", flush=True)
    print("[done] bin16 prune12 complete.", flush=True)


if __name__ == "__main__":
    main()
