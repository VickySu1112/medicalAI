#!/usr/bin/env python
"""M2 · v2 — EBM 交互 bin 数扫描(max_interaction_bins sweep,并行)。

验证怀疑:62×62 默认交互网格对 1003 人次小样本过细 → 稀疏 → 诊断全判外推。
扫 `max_interaction_bins ∈ {8,16,24,32}`,逐地标(1/3/6/12)看:
  (a) 交互 occupancy 升多少(预期 8→高、32→低);
  (b) verdict 区分度是否改善(8 bin 是否不再全判外推);
  (c) temporal 判别 AUC 是否保持(交互本无判别增益,减 bin 应不损 AUC)。

口径(复用现有,与 Phase 3 诊断一致):
  数据 corrected 真值 + LOCF —— `build_rows_for_method("locf", (L,))`(逐地标)。
  特征 —— `build_feats_at_L` / `FEATS`(正交轴)。
  OOF 协议 —— 复制 `ebm_oof_and_temporal` 的 split 逻辑(dev 5 折 StratifiedGroupKFold
    seed=CV_SEED=13 OOF / temporal final dev-fit),但额外传 `max_interaction_bins=B`
    (原函数硬编码 interactions=5、无该参数;**不改 ebm_oof.py**,这里本地重写 fit)。
  交互 occupancy / support_energy / frac_var_residual / verdict —— 直接复用
    `module2_v2_ebm_interaction_diag.diagnose_interaction`(g.data(idx) 取 scores/
    left_names/right_names;np.histogram2d 算占用格;support_energy=占用格 |g| 能量/
    总能量;真2D ⇐ residual≥0.35 且 occ≥40% 且 support≥0.6,否则对冲伪/外推)。

并行:`concurrent.futures.ProcessPoolExecutor`,16 个 (B, L) 组合各一进程,每进程
  PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1(进程级并行 + 单线程 BLAS)。

产物:
  results/module2_v2_vertical/m2v2_ebm_xland/tables/bins_sweep.csv
    列:max_interaction_bins, landmark, temporal_AUC, n_interactions, mean_occ%,
        mean_support, n_真2D, n_对冲伪, n_外推
  results/module2_v2_vertical/m2v2_ebm_xland/tables/bins_sweep_interactions.csv
    (逐交互项明细,便于复核)
  results/module2_v2_vertical/m2v2_ebm_xland/tables/bins_sweep_summary.json

硬约束:1003 人次 / 无 889 / time-safe(沿用 assert_no_future_feature) /
  OOF 用于选择、temporal 仅报告 / 不写 significant。
"""
from __future__ import annotations

import os

# 进程级并行 + 每进程单线程 BLAS(在导入 numpy/interpret 之前设好)。
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
from concurrent.futures import ProcessPoolExecutor, as_completed
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

from scripts.simple.module2_v2_shared import CV_SEED, PY_SEED, forbidden_token
from scripts.simple.module2_v2_impute_experiment import build_rows_for_method
from scripts.simple.module2_v2_b4_ebm_axes import FEATS, build_feats_at_L
from scripts.simple.module2_v2_b4_ebm_oof import assert_no_future_feature
from scripts.simple.module2_v2_ebm_interaction_diag import (
    diagnose_interaction,
    _resolve,
    THR_RESIDUAL,
    THR_OCC,
    THR_SUPPORT,
)

LANDMARKS = (1, 3, 6, 12)
BINS = (8, 16, 24, 32)
INTERACTIONS = 5  # 与主线 GA2M / Phase 3 诊断一致
OUT = ROOT / "results" / "module2_v2_vertical" / "m2v2_ebm_xland"


def _roc(y, p) -> float:
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")


def ebm_oof_temporal_bins(rows, y, lm, is_dev, L, max_interaction_bins, *,
                          interactions=INTERACTIONS, seed=PY_SEED):
    """复制 ebm_oof_and_temporal 的 split 逻辑,额外传 max_interaction_bins。

    dev → 5 折 StratifiedGroupKFold(by episode_id, seed=CV_SEED) OOF；
    temporal → final dev-fit EBM 预测。返回 (pred 全长, final_ebm, live 列序)。
    与原函数唯一差异 = EBM 多了 `max_interaction_bins=B`。不修改 ebm_oof.py。
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


def run_one(B: int, L: int) -> dict:
    """单个 (B, L) 组合:dev OOF + temporal + 逐交互项诊断。在子进程里跑。"""
    # 子进程内再次确认单线程(以防继承缺失)。
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[k] = "1"

    rows = build_rows_for_method("locf", (L,))  # 单地标 build 即正确(skeleton 含全行)
    assert rows["episode_id"].nunique() == 1003, "analysis unit must be 1003 人次"
    y = rows["Y_24M_NHRH"].values
    lm = rows["landmark"].values
    is_dev = (rows["Split"] == "Development").values

    pred, ebm, live = ebm_oof_temporal_bins(rows, y, lm, is_dev, L, B)
    devL = is_dev & (lm == L)
    tstL = (~is_dev) & (lm == L)
    oof_auc = _roc(y[devL], pred[devL])
    tmp_auc = _roc(y[tstL], pred[tstL])

    # 逐交互项诊断(复用 Phase 3 的 diagnose_interaction,口径完全一致)
    g = ebm.explain_global()
    overall = dict(zip(g.data()["names"], g.data()["scores"]))
    feat_all = build_feats_at_L(rows, devL)
    Xd = feat_all.loc[devL, live]
    yd = y[devL]
    inter = [(i, n) for i, n in enumerate(ebm.term_names_) if " & " in n]

    diags = []
    for ti, raw in inter:
        real = _resolve(raw, live)
        a_n, b_n = real.split(" & ")
        Av = Xd[a_n].values.astype(float)
        Bv = Xd[b_n].values.astype(float)
        dd = diagnose_interaction(ebm, ti, real, Av, Bv, yd)
        dd.pop("_plot", None)  # 不跨进程回传画图中间量
        dd["importance"] = round(float(overall.get(raw, 0.0)), 4)
        diags.append(dd)
    diags.sort(key=lambda r: -r["importance"])

    occs = [d["occupancy_pct"] for d in diags]
    sups = [d["support_energy"] for d in diags]
    verdicts = [d["verdict"] for d in diags]
    return {
        "max_interaction_bins": B,
        "landmark": f"{L}M",
        "L": L,
        "OOF_AUC": round(float(oof_auc), 4),
        "temporal_AUC": round(float(tmp_auc), 4),
        "n_interactions": len(diags),
        "mean_occ%": round(float(np.mean(occs)), 2) if occs else float("nan"),
        "mean_support": round(float(np.mean(sups)), 3) if sups else float("nan"),
        "n_真2D": int(sum(v == "真2D" for v in verdicts)),
        "n_对冲伪": int(sum(v == "对冲伪" for v in verdicts)),
        "n_外推": int(sum(v == "外推不可信" for v in verdicts)),
        "_diags": diags,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="smoke: 单 bin(8)单 landmark(6M),串行,不写汇总")
    ap.add_argument("--workers", type=int, default=8,
                    help="并行进程数(默认 8;16 组合)")
    args = ap.parse_args()

    (OUT / "tables").mkdir(parents=True, exist_ok=True)

    if args.quick:
        combos = [(8, 6)]
        print(f"[quick] smoke 单组合 {combos}", flush=True)
        results = [run_one(B, L) for (B, L) in combos]
        for r in results:
            print(f"  B={r['max_interaction_bins']} {r['landmark']}: "
                  f"temporal_AUC={r['temporal_AUC']} n_int={r['n_interactions']} "
                  f"mean_occ%={r['mean_occ%']} mean_support={r['mean_support']} "
                  f"真2D/对冲伪/外推={r['n_真2D']}/{r['n_对冲伪']}/{r['n_外推']}", flush=True)
        return

    combos = [(B, L) for B in BINS for L in LANDMARKS]  # 16 个
    print(f"[run] {len(combos)} 个 (bin, landmark) 组合,并行 workers={args.workers}", flush=True)
    print(f"      bins={BINS}  landmarks={LANDMARKS}  interactions={INTERACTIONS}", flush=True)
    print(f"      真2D 判据: residual≥{THR_RESIDUAL} 且 occ≥{THR_OCC}% 且 support≥{THR_SUPPORT}", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, B, L): (B, L) for (B, L) in combos}
        for fut in as_completed(futs):
            B, L = futs[fut]
            try:
                r = fut.result()
            except Exception as e:  # 单组合失败不拖垮整批
                print(f"  [FAIL] B={B} {L}M: {type(e).__name__}: {e}", flush=True)
                raise
            results.append(r)
            print(f"  [ok] B={r['max_interaction_bins']:>2} {r['landmark']:>3}: "
                  f"OOF={r['OOF_AUC']} temporal={r['temporal_AUC']} "
                  f"n_int={r['n_interactions']} occ={r['mean_occ%']}% "
                  f"sup={r['mean_support']} 真2D/伪/外推="
                  f"{r['n_真2D']}/{r['n_对冲伪']}/{r['n_外推']}", flush=True)

    # 排序:bin 升序、landmark 升序
    results.sort(key=lambda r: (r["max_interaction_bins"], r["L"]))

    # --- 主表 bins_sweep.csv ---
    cols = ["max_interaction_bins", "landmark", "temporal_AUC", "n_interactions",
            "mean_occ%", "mean_support", "n_真2D", "n_对冲伪", "n_外推"]
    main_df = pd.DataFrame([{c: r[c] for c in cols} for r in results])
    main_csv = OUT / "tables" / "bins_sweep.csv"
    main_df.to_csv(main_csv, index=False)
    print(f"\nSaved → {main_csv}", flush=True)
    print(main_df.to_string(index=False), flush=True)

    # --- 逐交互项明细 bins_sweep_interactions.csv ---
    det_rows = []
    for r in results:
        for d in r["_diags"]:
            det_rows.append({
                "max_interaction_bins": r["max_interaction_bins"],
                "landmark": r["landmark"],
                "term": d["term"], "importance": d["importance"],
                "corr_AB": d["corr_AB"], "same_group": d["same_group"],
                "occupancy_pct": d["occupancy_pct"], "support_energy": d["support_energy"],
                "frac_var_residual": d["frac_var_residual"], "svd_r1_frac": d["svd_r1_frac"],
                "sign_consistency": d["sign_consistency"], "verdict": d["verdict"],
            })
    det_df = pd.DataFrame(det_rows)
    det_csv = OUT / "tables" / "bins_sweep_interactions.csv"
    det_df.to_csv(det_csv, index=False)
    print(f"Saved → {det_csv}", flush=True)

    # --- pooled 汇总(逐 bin 跨地标平均)---
    pooled = []
    for B in BINS:
        sub = [r for r in results if r["max_interaction_bins"] == B]
        n_int_tot = sum(r["n_interactions"] for r in sub)
        n_true = sum(r["n_真2D"] for r in sub)
        n_hedge = sum(r["n_对冲伪"] for r in sub)
        n_extrap = sum(r["n_外推"] for r in sub)
        pooled.append({
            "max_interaction_bins": B,
            "mean_temporal_AUC": round(float(np.nanmean([r["temporal_AUC"] for r in sub])), 4),
            "mean_occ%": round(float(np.nanmean([r["mean_occ%"] for r in sub])), 2),
            "mean_support": round(float(np.nanmean([r["mean_support"] for r in sub])), 3),
            "n_interactions_total": int(n_int_tot),
            "n_真2D_total": int(n_true),
            "n_对冲伪_total": int(n_hedge),
            "n_外推_total": int(n_extrap),
            "pct_真2D": round(100.0 * n_true / n_int_tot, 1) if n_int_tot else float("nan"),
        })
    pooled_df = pd.DataFrame(pooled)
    print("\n=== pooled(逐 bin 跨 4 地标汇总)===", flush=True)
    print(pooled_df.to_string(index=False), flush=True)

    summary = {
        "caliber": "corrected 真值 + LOCF; EBM per-landmark OOF(dev 5fold SGKFold seed13)+ temporal final-fit",
        "landmarks": [f"{L}M" for L in LANDMARKS],
        "bins_swept": list(BINS),
        "interactions": INTERACTIONS,
        "n_episodes": 1003,
        "preregistered_thresholds": {
            "frac_var_residual_>=": THR_RESIDUAL, "occupancy_pct_>=": THR_OCC,
            "support_energy_>=": THR_SUPPORT,
        },
        "per_combo": [{c: r[c] for c in cols} for r in results],
        "pooled_by_bin": pooled,
    }
    sjson = OUT / "tables" / "bins_sweep_summary.json"
    sjson.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    # 安全核验:无 forbidden token
    blob = json.dumps(summary, ensure_ascii=False)
    assert forbidden_token() not in blob, "forbidden unique-patient count leaked into summary"
    print(f"\nSaved → {sjson}", flush=True)
    print("[done] bins sweep complete.", flush=True)


if __name__ == "__main__":
    main()
