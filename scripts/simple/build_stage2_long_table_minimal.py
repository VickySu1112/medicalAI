#!/usr/bin/env python
"""Minimal stage2_long_table.csv builder from 1003.xlsx (no Stage 2 pipeline).

Reads `1003.xlsx` (multi-header: row 0 = visit group, row 1 = sub-item) and
produces only the columns `module2_v2_shared.load_stacked()` and downstream EBM
atlas / patient decomposition / 12M extension actually need:

    Treatment_ID, Treatment_Index, Episode_Index,
    FT3_{0,1,3,6,12,18,24}M, FT4_{...}, TSH_{...}, TRAb_{...},
    Eval_{1,3,6,12,18,24}M_{Hyper, Normal, Hypo, Missing}

Each Eval_*_Code is the raw 治疗评价 (1 甲亢/2 甲减/3 正常); one-hots are derived.

Output: results/stage2_mh_h6h12_cjk/tables/stage2_long_table.csv (one row per
episode; Treatment_Index = 0..1002 matches M1 frozen Episode_Index 0..1002).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

XLSX = ROOT / "1003.xlsx"
OUT_DIR = ROOT / "results" / "stage2_mh_h6h12_cjk" / "tables"
OUT = OUT_DIR / "stage2_long_table.csv"

VISIT_MAP = {  # xlsx visit-group label → landmark month
    "第一次复查（1个月）": 1,
    "第二次（3个月）": 3,
    "第3次（6个月）": 6,
    "第4次（1年）": 12,
    "第5次（1.5年）": 18,
    "第6次（2年）": 24,
}
LAB_KEYS = ("FT3", "FT4", "TSH", "TRAb")
EVAL_SUBLABELS = (
    "治疗评价（1甲亢，2甲减，3正常）（1甲亢，2甲减，3正常）",
    "治疗评价（1甲亢，2甲减，3正常）",
)


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def main() -> None:
    if not XLSX.exists():
        raise FileNotFoundError(f"1003.xlsx not found at {XLSX}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading {XLSX} (multi-header, nrows=1003)…", flush=True)
    # xlsx is physically ~1M rows (Excel-full with blank trailing rows);
    # actual data = rows 2..1004 → 1003 episodes. header=[0,1] correctly
    # expands merged-cell sub-headers (e.g. "第一次复查(1个月)" × "FT3").
    df = pd.read_excel(XLSX, sheet_name="Sheet1", header=[0, 1], nrows=1003)
    print(f"  data shape: {df.shape}", flush=True)
    if len(df) != 1003:
        raise RuntimeError(f"Expected 1003 episodes, got {len(df)}")

    n = len(df)
    out = pd.DataFrame({
        "Treatment_ID": np.arange(n),
        "Treatment_Index": np.arange(n),
        "Episode_Index": np.arange(n),
    })
    # Split 列从 M1 frozen 矩阵拉(episode 级 development/temporal 分组,m2-v3 loader 需要)
    m1_path = ROOT / "results" / "module1_baseline_ml_benchmark" / "tables" / "module1_frozen_feature_matrix.csv"
    if m1_path.exists():
        m1 = pd.read_csv(m1_path, low_memory=False, usecols=["Episode_Index", "Split"])
        out = out.merge(m1, on="Episode_Index", how="left", validate="one_to_one")
        if out["Split"].isna().any():
            raise RuntimeError("Split merge has NaN; episode 对齐错误")
        print(f"  Split: dev {(out['Split']=='Development').sum()} / temporal {(out['Split']=='Temporal').sum()}", flush=True)

    # Baseline labs (level-0 = simple header, level-1 = unit string)
    baseline_map = {"FT3": "FT3", "FT4": "FT4", "TSH": "TSH", "TRAb": "TRAb"}
    for col_l0, lab in baseline_map.items():
        matched = [c for c in df.columns if c[0] == col_l0]
        if matched:
            out[f"{lab}_0M"] = _to_num(df[matched[0]])
        else:
            out[f"{lab}_0M"] = np.nan

    # Per-visit follow-up labs + Eval
    for visit_label, L in VISIT_MAP.items():
        sub_cols = {c[1]: c for c in df.columns if c[0] == visit_label}
        for k in LAB_KEYS:
            out[f"{k}_{L}M"] = _to_num(df[sub_cols[k]]) if k in sub_cols else np.nan
        # Eval — find whichever sublabel exists
        eval_col = None
        for el in EVAL_SUBLABELS:
            if el in sub_cols:
                eval_col = sub_cols[el]; break
        if eval_col is not None:
            ec = _to_num(df[eval_col])
            out[f"Eval_{L}M_Code"] = ec
            out[f"Eval_{L}M_Hyper"] = (ec == 1).astype(int)
            out[f"Eval_{L}M_Hypo"] = (ec == 2).astype(int)
            out[f"Eval_{L}M_Normal"] = (ec == 3).astype(int)
            out[f"Eval_{L}M_Missing"] = ec.isna().astype(int)
        else:
            for cat in ("Code", "Hyper", "Hypo", "Normal", "Missing"):
                out[f"Eval_{L}M_{cat}"] = np.nan if cat == "Code" else 0

    # ---- 同时输出 LONG format(m2-v3 loader 期望 Current_Time + *_Current)----
    # 每个 episode 写 7 行,对应 0/1/3/6/12/18/24M 真值时点;wide 列保留以兼容老 loader
    landmarks = (0, 1, 3, 6, 12, 18, 24)
    long_records = []
    for _, row in out.iterrows():
        for L in landmarks:
            rec = {col: row[col] for col in out.columns}  # 保留所有 wide 列
            rec["Current_Time"] = f"{L}M"
            rec["FT3_Current"] = row.get(f"FT3_{L}M", np.nan)
            rec["FT4_Current"] = row.get(f"FT4_{L}M", np.nan)
            rec["TSH_Current"] = row.get(f"TSH_{L}M", np.nan)
            rec["TRAb_Current"] = row.get(f"TRAb_{L}M", np.nan)
            long_records.append(rec)
    out = pd.DataFrame(long_records)

    print(f"  built {len(out)} rows × {out.shape[1]} cols (long format,7 landmark × 1003)", flush=True)
    # Sanity
    for L in (0, 1, 3, 6, 12, 18, 24):
        m = out["Current_Time"] == f"{L}M"
        cov = out.loc[m, "FT4_Current"].notna().mean() * 100
        print(f"  FT4_Current @ {L}M non-null: {cov:.1f}%", flush=True)
    out.to_csv(OUT, index=False)
    print(f"\nSaved → {OUT}", flush=True)
    print(f"  size: {OUT.stat().st_size:,} bytes", flush=True)


if __name__ == "__main__":
    main()
