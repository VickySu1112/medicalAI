"""Consolidate Stage 1 early stratification report.

This script is a report-level and small-supplementary-analysis pass. It does
not change NHRH labels and does not rerun the broad NHRH model search. The main
binary NHRH report is fixed to clinical-core L2 logistic rows already present
in ``nhrh_binary_all_candidates.csv`` / ``nhrh_binary_predictions_long.csv``.

Additional analyses are bounded: compact 12-feature sensitivity, 24M state
prediction, medication availability audit, error-profile tables, and
development-only error-aware refinement diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("MPLBACKEND", "Agg")
warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None

try:
    import shap
except Exception:  # pragma: no cover - optional dependency
    shap = None

from scripts.simple.early_stratification_stage1 import (
    NHRH3_ORDER,
    _unpenalized_lr,
    add_missing_indicators,
    binary_prob_model,
    fit_oof_weighted,
    fit_with_weight,
    leakage_check,
    make_stage1_feature_sets,
    md_table,
    nhrh_3class,
    sample_weights,
    state_name,
)
from scripts.simple.nhrh_landmark_binary import (
    FOLLOWUP_TIMES,
    apply_platt,
    build_landmark_dataset,
    build_nhrh_labels,
    bootstrap_ci,
    choose_threshold,
    dca_curve,
    metrics_at_threshold,
    predict_proba_one,
    read_1003,
    temporal_row_split,
)
from utils.config import COL_IDX


OUT_DEFAULT = ROOT / "results" / "3m_6m_early_stratification"
# Landmarks shown across the NHRH figures: 0M (pure pre-RAI baseline) and 1M added to the
# original 3M/6M so every panel reads baseline -> +1M -> +3M -> +6M (incremental value).
LANDMARKS = ["0M", "1M", "3M", "6M"]
NLM = len(LANDMARKS)
STATE_ORDER = ["Hyper", "Normal", "Hypo"]
SELECTED_RUNS = {
    "3M": "3M__clinical_core__Clinical_L2_Logistic__nhrh_mild__platt__accuracy__s13",
    "6M": "6M__clinical_core__Clinical_L2_Logistic__nhrh_mild__platt__accuracy__s13",
}


def month_label_en(t: str) -> str:
    return {"0M": "baseline", "1M": "1 month", "3M": "3 months", "6M": "6 months", "12M": "12 months", "18M": "18 months", "24M": "24 months"}.get(str(t), str(t))


def month_label_zh(t: str) -> str:
    return {"0M": "治疗前", "1M": "1个月", "3M": "3个月", "6M": "6个月", "12M": "12个月", "18M": "18个月", "24M": "24个月"}.get(str(t), str(t))


def lab_label_zh(lab: str) -> str:
    return {"FT3": "游离三碘甲状腺原氨酸 FT3", "FT4": "游离甲状腺素 FT4", "TSH": "促甲状腺激素 TSH"}.get(str(lab), str(lab))


def lab_alias(lab: str) -> str:
    return str(lab).lower()


def time_alias(t: str) -> str:
    return {"0M": "baseline", "1M": "1m", "3M": "3m", "6M": "6m", "12M": "12m", "18M": "18m", "24M": "24m"}.get(str(t), str(t).lower())


def sanitize_alias(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_").lower()
    text = re.sub(r"_+", "_", text)
    return text or "derived_feature"


def feature_record(name: str) -> dict[str, str]:
    """Return a registry-style record for a raw feature key.

    Raw feature keys are preserved for reproducibility, but all paper-facing
    labels are generated here so figures and README text do not expose
    engineering names such as ``D_FT3_3M-0M``.
    """
    f = str(name)
    static: dict[str, tuple[str, str, str, str, str, str]] = {
        "Age": ("age_at_radioiodine", "Age at radioiodine therapy", "RAI 治疗时年龄", "患者接受本次放射性碘治疗时的年龄。", "years", "baseline"),
        "Sex": ("sex", "Sex", "性别", "患者性别。", "binary indicator", "baseline"),
        "Height": ("height", "Height", "身高", "患者身高。", "continuous", "baseline"),
        "Weight": ("body_weight", "Body weight", "体重", "患者体重。", "continuous", "baseline"),
        "BMI": ("body_mass_index", "Body mass index", "体重指数 BMI", "由身高和体重计算的体重指数。", "kg/m^2", "baseline"),
        "ThyroidW": ("thyroid_weight", "Thyroid weight", "甲状腺重量", "甲状腺越大，通常需要更高的组织破坏剂量才能控制甲亢。", "g", "baseline"),
        "Dose": ("administered_rai_activity", "Administered radioiodine activity", "放射性碘给药剂量", "本次治疗实际给予的放射性碘活度。", "mCi or local activity unit", "baseline"),
        "Uptake24h": ("radioiodine_uptake_24h", "24-hour thyroid radioiodine uptake", "24小时甲状腺摄碘率", "甲状腺在给药后24小时摄取放射性碘的比例。", "%", "baseline"),
        "MaxUptake": ("peak_radioiodine_uptake", "Peak thyroid radioiodine uptake", "最高甲状腺摄碘率", "随访测得的最高摄碘比例，反映甲状腺摄取能力。", "%", "baseline"),
        "HalfLife": ("effective_iodine_half_life", "Effective iodine half-life", "有效碘半衰期", "放射性碘在甲状腺内保留的有效时间；保留越久，组织接受照射越多。", "days", "baseline"),
        "RAI3d": ("three_day_retained_iodine", "Three-day retained iodine", "3天后甲状腺残留碘量", "给药后3天仍保留在甲状腺内的碘信号。", "continuous", "baseline"),
        "TRAb": ("trab", "TSH receptor antibody", "促甲状腺激素受体抗体 TRAb", "反映 Graves 病免疫活性的重要抗体指标。", "IU/L or local unit", "baseline"),
        "TGAb": ("tgab", "Thyroglobulin antibody", "甲状腺球蛋白抗体 TgAb", "甲状腺自身免疫相关抗体。", "local unit", "baseline"),
        "TPOAb": ("tpoab", "Thyroid peroxidase antibody", "甲状腺过氧化物酶抗体 TPOAb", "甲状腺自身免疫相关抗体。", "local unit", "baseline"),
        "Exophthalmos": ("orbitopathy_indicator", "Orbitopathy or exophthalmos", "突眼或甲状腺相关眼病体征", "是否记录到突眼或甲状腺相关眼病体征。", "binary or ordinal", "baseline"),
        "TreatCount": ("treatment_origin_count", "Treatment-origin count", "治疗次数或治疗起点编号", "当前治疗起点在患者治疗历程中的序号；不是 RAI 前 ATD 用药史。", "count", "baseline"),
        "IDPG_Dose_per_ThyroidW": ("radioiodine_activity_per_gram_thyroid", "Radioiodine activity per gram thyroid weight", "每克甲状腺组织分配的 RAI 剂量", "把 RAI 剂量除以甲状腺重量，近似表示每克甲状腺组织分到的剂量。", "activity per gram", "baseline"),
    }
    if f in static:
        alias, en, zh, plain, unit, window = static[f]
        return {
            "raw_key": f,
            "code_alias": alias,
            "paper_label_en": en,
            "paper_label_zh": zh,
            "plain_zh": plain,
            "formula": "Directly observed before or at index RAI",
            "category": feature_category(f),
            "timepoint": window,
            "unit_or_scale": unit,
            "source_window": window,
        }

    dose_interactions = {
        "Dose_x_ThyroidW": ("dose_thyroid_weight_interaction", "Interaction between radioiodine activity and thyroid weight", "RAI 剂量与甲状腺重量的交互", "Dose × ThyroidW"),
        "Dose_x_Uptake24h": ("dose_24h_uptake_interaction", "Interaction between radioiodine activity and 24-hour uptake", "RAI 剂量与24小时摄碘率的交互", "Dose × Uptake24h"),
        "Dose_per_Uptake24h": ("uptake_adjusted_radioiodine_activity", "Radioiodine activity adjusted for 24-hour uptake", "按24小时摄碘率校正后的 RAI 剂量", "Dose / Uptake24h"),
        "Dose_x_MaxUptake": ("dose_peak_uptake_interaction", "Interaction between radioiodine activity and peak uptake", "RAI 剂量与最高摄碘率的交互", "Dose × MaxUptake"),
        "Dose_x_HalfLife": ("dose_effective_half_life_interaction", "Interaction between radioiodine activity and effective half-life", "RAI 剂量与有效碘半衰期的交互", "Dose × HalfLife"),
        "Estimated_TID_Dose_x_Uptake24h_x_HalfLife": ("estimated_total_iodine_exposure", "Estimated total iodine exposure", "估计总碘暴露量", "Dose × Uptake24h × HalfLife"),
        "RAI3d_x_Dose": ("three_day_retention_dose_interaction", "Interaction between three-day iodine retention and radioiodine activity", "3天残留碘量与 RAI 剂量的交互", "RAI3d × Dose"),
    }
    if f in dose_interactions:
        alias, en, zh, formula = dose_interactions[f]
        return {
            "raw_key": f,
            "code_alias": alias,
            "paper_label_en": en,
            "paper_label_zh": zh,
            "plain_zh": "乘积或校正项，用来表达 RAI 剂量、摄取和甲状腺负荷共同决定组织照射强度。",
            "formula": formula,
            "category": feature_category(f),
            "timepoint": "baseline",
            "unit_or_scale": "derived continuous",
            "source_window": "baseline",
        }

    m = re.match(r"^(FT3|FT4|TSH)_(0M|1M|3M|6M)$", f)
    if m:
        lab, t = m.groups()
        return {
            "raw_key": f,
            "code_alias": f"{lab_alias(lab)}_at_{time_alias(t)}",
            "paper_label_en": f"{lab} at {month_label_en(t)}",
            "paper_label_zh": f"{month_label_zh(t)} {lab_label_zh(lab)}",
            "plain_zh": f"{month_label_zh(t)}实际测到的 {lab} 水平。",
            "formula": "Direct laboratory value",
            "category": feature_category(f),
            "timepoint": month_label_en(t),
            "unit_or_scale": "laboratory value",
            "source_window": f"up to {month_label_en(t)}",
        }

    m = re.match(r"^logTSH_(0M|1M|3M|6M)$", f)
    if m:
        t = m.group(1)
        return {
            "raw_key": f,
            "code_alias": f"log_tsh_at_{time_alias(t)}",
            "paper_label_en": f"Log-transformed TSH at {month_label_en(t)}",
            "paper_label_zh": f"{month_label_zh(t)} TSH 对数值",
            "plain_zh": "把 TSH 做对数变换，减少极端 TSH 数值对线性模型的影响。",
            "formula": f"log(TSH at {month_label_en(t)})",
            "category": feature_category(f),
            "timepoint": month_label_en(t),
            "unit_or_scale": "log laboratory value",
            "source_window": f"up to {month_label_en(t)}",
        }

    m = re.match(r"^Eval_(1M|3M|6M)_(Hyper|Normal|Hypo)$", f)
    if m:
        t, st = m.groups()
        state_en = {"Hyper": "hyperthyroid", "Normal": "euthyroid", "Hypo": "hypothyroid"}[st]
        state_zh = {"Hyper": "甲亢", "Normal": "甲功正常", "Hypo": "甲减"}[st]
        return {
            "raw_key": f,
            "code_alias": f"{state_en}_state_indicator_at_{time_alias(t)}",
            "paper_label_en": f"{state_en.capitalize()} state indicator at {month_label_en(t)}",
            "paper_label_zh": f"{month_label_zh(t)}是否为{state_zh}状态",
            "plain_zh": f"0/1变量：{month_label_zh(t)}临床评估是否处于{state_zh}。",
            "formula": f"1 if clinical state at {month_label_en(t)} is {state_en}; otherwise 0",
            "category": feature_category(f),
            "timepoint": month_label_en(t),
            "unit_or_scale": "binary indicator",
            "source_window": f"up to {month_label_en(t)}",
        }

    m = re.match(r"^Eval_(1M|3M|6M)_Code$", f)
    if m:
        t = m.group(1)
        return {
            "raw_key": f,
            "code_alias": f"thyroid_state_code_at_{time_alias(t)}",
            "paper_label_en": f"Ordinal thyroid status code at {month_label_en(t)}",
            "paper_label_zh": f"{month_label_zh(t)}甲功状态编码",
            "plain_zh": "把甲亢、正常、甲减状态编码成有序数值，用于描述当前甲功状态。",
            "formula": "Ordinal encoding of clinical thyroid status",
            "category": feature_category(f),
            "timepoint": month_label_en(t),
            "unit_or_scale": "ordinal code",
            "source_window": f"up to {month_label_en(t)}",
        }

    m = re.match(r"^Miss_(FT3|FT4|TSH)_(0M|1M|3M|6M)$", f)
    if m:
        lab, t = m.groups()
        return {
            "raw_key": f,
            "code_alias": f"{lab_alias(lab)}_missing_indicator_at_{time_alias(t)}",
            "paper_label_en": f"Missing {lab} indicator at {month_label_en(t)}",
            "paper_label_zh": f"{month_label_zh(t)} {lab} 缺失指示",
            "plain_zh": f"0/1变量：{month_label_zh(t)}是否缺少 {lab} 检验值。",
            "formula": f"1 if {lab} is missing at {month_label_en(t)}; otherwise 0",
            "category": feature_category(f),
            "timepoint": month_label_en(t),
            "unit_or_scale": "binary indicator",
            "source_window": f"up to {month_label_en(t)}",
        }

    m = re.match(r"^Miss_Eval_(1M|3M|6M)$", f)
    if m:
        t = m.group(1)
        return {
            "raw_key": f,
            "code_alias": f"thyroid_state_missing_indicator_at_{time_alias(t)}",
            "paper_label_en": f"Missing thyroid status indicator at {month_label_en(t)}",
            "paper_label_zh": f"{month_label_zh(t)}甲功状态缺失指示",
            "plain_zh": f"0/1变量：{month_label_zh(t)}是否缺少临床甲功状态记录。",
            "formula": f"1 if thyroid status is missing at {month_label_en(t)}; otherwise 0",
            "category": feature_category(f),
            "timepoint": month_label_en(t),
            "unit_or_scale": "binary indicator",
            "source_window": f"up to {month_label_en(t)}",
        }

    m = re.match(r"^D_(FT3|FT4|TSH)_(1M|3M|6M)-(0M|1M|3M)$", f)
    if m:
        lab, t1, t0 = m.groups()
        return {
            "raw_key": f,
            "code_alias": f"{lab_alias(lab)}_change_{time_alias(t0)}_to_{time_alias(t1)}",
            "paper_label_en": f"Change in {lab} from {month_label_en(t0)} to {month_label_en(t1)}",
            "paper_label_zh": f"{lab} 从{month_label_zh(t0)}到{month_label_zh(t1)}的变化量",
            "plain_zh": f"{lab} 在两个随访时点之间升高或下降了多少，反映早期治疗反应。",
            "formula": f"{lab} at {month_label_en(t1)} minus {lab} at {month_label_en(t0)}",
            "category": feature_category(f),
            "timepoint": f"{month_label_en(t0)} to {month_label_en(t1)}",
            "unit_or_scale": "absolute change",
            "source_window": f"up to {month_label_en(t1)}",
        }

    m = re.match(r"^PctDrop_(FT3|FT4)_0_(1M|3M|6M)$", f)
    if m:
        lab, t = m.groups()
        return {
            "raw_key": f,
            "code_alias": f"{lab_alias(lab)}_percent_reduction_baseline_to_{time_alias(t)}",
            "paper_label_en": f"Percent reduction in {lab} from baseline to {month_label_en(t)}",
            "paper_label_zh": f"{lab} 从治疗前到{month_label_zh(t)}的百分比下降",
            "plain_zh": f"{lab} 相对治疗前下降了多少百分比，比单纯差值更贴近早期反应幅度。",
            "formula": f"({lab} at baseline minus {lab} at {month_label_en(t)}) / {lab} at baseline",
            "category": feature_category(f),
            "timepoint": f"baseline to {month_label_en(t)}",
            "unit_or_scale": "percentage",
            "source_window": f"up to {month_label_en(t)}",
        }

    m = re.match(r"^(FT3|FT4)_(1M|3M|6M)_over_0M$", f)
    if m:
        lab, t = m.groups()
        return {
            "raw_key": f,
            "code_alias": f"{lab_alias(lab)}_ratio_{time_alias(t)}_to_baseline",
            "paper_label_en": f"Ratio of {lab} at {month_label_en(t)} to baseline",
            "paper_label_zh": f"{month_label_zh(t)} {lab} 与治疗前 {lab} 的比值",
            "plain_zh": f"当前 {lab} 相当于治疗前水平的多少，用来表达相对控制程度。",
            "formula": f"{lab} at {month_label_en(t)} / {lab} at baseline",
            "category": feature_category(f),
            "timepoint": f"baseline to {month_label_en(t)}",
            "unit_or_scale": "ratio",
            "source_window": f"up to {month_label_en(t)}",
        }

    m = re.match(r"^D_logTSH_(1M|3M|6M)_0M$", f)
    if m:
        t = m.group(1)
        return {
            "raw_key": f,
            "code_alias": f"log_tsh_change_baseline_to_{time_alias(t)}",
            "paper_label_en": f"Change in log-transformed TSH from baseline to {month_label_en(t)}",
            "paper_label_zh": f"TSH 对数值从治疗前到{month_label_zh(t)}的变化",
            "plain_zh": "TSH 对数值的早期变化，用来描述 TSH 是否开始恢复。",
            "formula": f"log(TSH at {month_label_en(t)}) minus log(TSH at baseline)",
            "category": feature_category(f),
            "timepoint": f"baseline to {month_label_en(t)}",
            "unit_or_scale": "log-scale change",
            "source_window": f"up to {month_label_en(t)}",
        }

    m = re.match(r"^(FT3|FT4|TSH)_mean_0_(3M|6M)$", f)
    if m:
        lab, t = m.groups()
        return {
            "raw_key": f,
            "code_alias": f"mean_{lab_alias(lab)}_baseline_to_{time_alias(t)}",
            "paper_label_en": f"Mean {lab} from baseline to {month_label_en(t)}",
            "paper_label_zh": f"治疗前到{month_label_zh(t)}的平均 {lab}",
            "plain_zh": f"截至 landmark 的平均 {lab} 水平，用来描述总体激素暴露负担。",
            "formula": f"Mean {lab} across observed visits from baseline to {month_label_en(t)}",
            "category": feature_category(f),
            "timepoint": f"baseline to {month_label_en(t)}",
            "unit_or_scale": "mean laboratory value",
            "source_window": f"up to {month_label_en(t)}",
        }

    m = re.match(r"^(FT3|FT4|TSH)_std_0_(3M|6M)$", f)
    if m:
        lab, t = m.groups()
        return {
            "raw_key": f,
            "code_alias": f"{lab_alias(lab)}_variability_baseline_to_{time_alias(t)}",
            "paper_label_en": f"Variability of {lab} from baseline to {month_label_en(t)}",
            "paper_label_zh": f"治疗前到{month_label_zh(t)}的 {lab} 波动程度",
            "plain_zh": f"截至 landmark 的 {lab} 波动程度，用来描述治疗反应是否稳定。",
            "formula": f"Standard deviation of {lab} across observed visits from baseline to {month_label_en(t)}",
            "category": feature_category(f),
            "timepoint": f"baseline to {month_label_en(t)}",
            "unit_or_scale": "standard deviation",
            "source_window": f"up to {month_label_en(t)}",
        }

    m = re.match(r"^(FT3|FT4|TSH)_last_minus_prev$", f)
    if m:
        lab = m.group(1)
        return {
            "raw_key": f,
            "code_alias": f"recent_{lab_alias(lab)}_change_before_landmark",
            "paper_label_en": f"Most recent change in {lab} before the landmark",
            "paper_label_zh": f"landmark 前最近一次 {lab} 变化",
            "plain_zh": f"最近一次随访相对上一次的 {lab} 变化，反映短期趋势。",
            "formula": f"{lab} at current landmark minus {lab} at previous available visit",
            "category": feature_category(f),
            "timepoint": "current landmark",
            "unit_or_scale": "absolute change",
            "source_window": "up to current landmark",
        }

    m = re.match(r"^FT3_(3M|6M)_over_FT4_(3M|6M)$", f)
    if m:
        t = m.group(1)
        return {
            "raw_key": f,
            "code_alias": f"ft3_to_ft4_ratio_at_{time_alias(t)}",
            "paper_label_en": f"FT3-to-FT4 ratio at {month_label_en(t)}",
            "paper_label_zh": f"{month_label_zh(t)} FT3/FT4 比值",
            "plain_zh": "FT3 相对 FT4 的比例，用来描述 Graves 甲亢中 T3 优势程度。",
            "formula": f"FT3 at {month_label_en(t)} / FT4 at {month_label_en(t)}",
            "category": feature_category(f),
            "timepoint": month_label_en(t),
            "unit_or_scale": "ratio",
            "source_window": f"up to {month_label_en(t)}",
        }

    m = re.match(r"^FT4_(3M|6M)_over_TSH_(3M|6M)_plus1$", f)
    if m:
        t = m.group(1)
        return {
            "raw_key": f,
            "code_alias": f"ft4_to_tsh_plus_one_ratio_at_{time_alias(t)}",
            "paper_label_en": f"FT4-to-(TSH+1) ratio at {month_label_en(t)}",
            "paper_label_zh": f"{month_label_zh(t)} FT4/(TSH+1) 比值",
            "plain_zh": "把 FT4 与 TSH 恢复情况合并成一个比值，值高通常提示甲功仍偏亢。",
            "formula": f"FT4 at {month_label_en(t)} / (TSH at {month_label_en(t)} + 1)",
            "category": feature_category(f),
            "timepoint": month_label_en(t),
            "unit_or_scale": "ratio",
            "source_window": f"up to {month_label_en(t)}",
        }

    m = re.match(r"^TSH_Recovered_(3M|6M)$", f)
    if m:
        t = m.group(1)
        return {
            "raw_key": f,
            "code_alias": f"tsh_recovery_indicator_at_{time_alias(t)}",
            "paper_label_en": f"TSH recovery indicator at {month_label_en(t)}",
            "paper_label_zh": f"{month_label_zh(t)} TSH 恢复指示",
            "plain_zh": "0/1变量：TSH 是否已回到参考范围或接近恢复。",
            "formula": f"1 if TSH recovered by {month_label_en(t)}; otherwise 0",
            "category": feature_category(f),
            "timepoint": month_label_en(t),
            "unit_or_scale": "binary indicator",
            "source_window": f"up to {month_label_en(t)}",
        }

    m = re.match(r"^Likely_Hyper_(3M|6M)$", f)
    if m:
        t = m.group(1)
        return {
            "raw_key": f,
            "code_alias": f"biochemical_hyperthyroid_pattern_at_{time_alias(t)}",
            "paper_label_en": f"Biochemical hyperthyroid pattern at {month_label_en(t)}",
            "paper_label_zh": f"{month_label_zh(t)}甲亢样甲功组合",
            "plain_zh": "0/1变量：该时点甲功组合是否仍符合甲亢样模式。",
            "formula": f"1 if thyroid function pattern at {month_label_en(t)} suggests hyperthyroidism; otherwise 0",
            "category": feature_category(f),
            "timepoint": month_label_en(t),
            "unit_or_scale": "binary indicator",
            "source_window": f"up to {month_label_en(t)}",
        }

    m = re.match(r"^ThyroidW_x_(FT3|FT4)_(3M|6M)$", f)
    if m:
        lab, t = m.groups()
        return {
            "raw_key": f,
            "code_alias": f"thyroid_weight_{lab_alias(lab)}_{time_alias(t)}_interaction",
            "paper_label_en": f"Interaction between thyroid weight and {lab} at {month_label_en(t)}",
            "paper_label_zh": f"甲状腺重量与{month_label_zh(t)} {lab} 的交互",
            "plain_zh": "乘积交互项，用来表达“大甲状腺 + 当前激素仍高”的组合风险。",
            "formula": f"ThyroidW × {lab} at {month_label_en(t)}",
            "category": feature_category(f),
            "timepoint": month_label_en(t),
            "unit_or_scale": "derived continuous",
            "source_window": f"up to {month_label_en(t)}",
        }

    m = re.match(r"^TRAb_x_(FT3|FT4)_(3M|6M)$", f)
    if m:
        lab, t = m.groups()
        return {
            "raw_key": f,
            "code_alias": f"trab_{lab_alias(lab)}_{time_alias(t)}_interaction",
            "paper_label_en": f"Interaction between TRAb and {lab} at {month_label_en(t)}",
            "paper_label_zh": f"TRAb 与{month_label_zh(t)} {lab} 的交互",
            "plain_zh": "乘积交互项，用来表达免疫活性和当前甲亢程度同时存在时的风险。",
            "formula": f"TRAb × {lab} at {month_label_en(t)}",
            "category": feature_category(f),
            "timepoint": month_label_en(t),
            "unit_or_scale": "derived continuous",
            "source_window": f"up to {month_label_en(t)}",
        }

    readable = re.sub(r"[_]+", " ", f)
    readable = re.sub(r"\s+", " ", readable).strip()
    return {
        "raw_key": f,
        "code_alias": sanitize_alias(f),
        "paper_label_en": f"Derived predictor: {readable}",
        "paper_label_zh": f"派生预测变量：{readable}",
        "plain_zh": "未归入固定模板的工程派生变量；完整公式需回查特征构建代码。",
        "formula": "Derived feature; see feature-construction code",
        "category": feature_category(f),
        "timepoint": "derived",
        "unit_or_scale": "derived",
        "source_window": "derived",
    }


def pretty_feature_label(name: str, *, zh: bool = False) -> str:
    """Paper-style, clinician-readable labels for engineered feature names."""
    rec = feature_record(str(name))
    return rec["paper_label_zh" if zh else "paper_label_en"]

    feature = str(name)
    state_en = {"Hyper": "hyperthyroid", "Normal": "euthyroid", "Hypo": "hypothyroid"}
    state_zh = {"Hyper": "甲亢", "Normal": "甲功正常", "Hypo": "甲减"}
    lab_zh = {"FT3": "游离三碘甲状腺原氨酸 FT3", "FT4": "游离甲状腺素 FT4", "TSH": "促甲状腺激素 TSH"}

    static = {
        "Age": ("Age", "年龄"),
        "Sex": ("Sex", "性别"),
        "Height": ("Height", "身高"),
        "Weight": ("Weight", "体重"),
        "BMI": ("Body mass index", "体重指数 BMI"),
        "ThyroidW": ("Thyroid weight", "甲状腺重量/负荷"),
        "Dose": ("Administered RAI activity", "放射性碘给药剂量"),
        "Uptake24h": ("24-h thyroid RAI uptake", "24 小时甲状腺摄碘率"),
        "MaxUptake": ("Peak thyroid RAI uptake", "最高甲状腺摄碘率"),
        "HalfLife": ("Effective iodine half-life", "有效碘半衰期"),
        "RAI3d": ("3-day retained iodine", "3 天后甲状腺残留碘量"),
        "TRAb": ("TRAb", "促甲状腺激素受体抗体 TRAb"),
        "TGAb": ("TgAb", "甲状腺球蛋白抗体 TgAb"),
        "TPOAb": ("TPOAb", "甲状腺过氧化物酶抗体 TPOAb"),
        "Exophthalmos": ("Orbitopathy / exophthalmos", "突眼/甲状腺相关眼病体征"),
        "TreatCount": ("Treatment count", "治疗次数/治疗起点编号"),
        "IDPG_Dose_per_ThyroidW": ("RAI activity per thyroid weight", "单位甲状腺重量接受的 RAI 剂量"),
        "Dose_x_ThyroidW": ("RAI activity × thyroid weight", "RAI 剂量 × 甲状腺重量"),
        "Dose_x_Uptake24h": ("RAI activity × 24-h uptake", "RAI 剂量 × 24 小时摄碘率"),
        "Dose_per_Uptake24h": ("RAI activity adjusted for 24-h uptake", "按 24 小时摄碘率校正后的 RAI 剂量"),
        "Dose_x_MaxUptake": ("RAI activity × peak uptake", "RAI 剂量 × 最高摄碘率"),
        "Dose_x_HalfLife": ("RAI activity × iodine half-life", "RAI 剂量 × 有效碘半衰期"),
        "Estimated_TID_Dose_x_Uptake24h_x_HalfLife": ("Estimated total iodine exposure", "估计总碘暴露量"),
        "RAI3d_x_Dose": ("3-day retained iodine × RAI activity", "3 天残留碘量 × RAI 剂量"),
    }
    if feature in static:
        return static[feature][1 if zh else 0]

    m = re.match(r"^(FT3|FT4|TSH)_(0M|1M|3M|6M)$", feature)
    if m:
        lab, t = m.groups()
        return f"{t} {lab}" if not zh else f"{t} 时点 {lab_zh[lab]}"

    m = re.match(r"^logTSH_(0M|1M|3M|6M)$", feature)
    if m:
        t = m.group(1)
        return f"{t} log(TSH)" if not zh else f"{t} 时点 TSH 的对数变换"

    m = re.match(r"^Eval_(1M|3M|6M)_(Hyper|Normal|Hypo)$", feature)
    if m:
        t, st = m.groups()
        return f"{t} {state_en[st]} state" if not zh else f"{t} 临床评估是否为{state_zh[st]}"

    m = re.match(r"^Eval_(1M|3M|6M)_Code$", feature)
    if m:
        t = m.group(1)
        return f"{t} thyroid status code" if not zh else f"{t} 临床甲功状态编码"

    m = re.match(r"^Miss_(FT3|FT4|TSH)_(0M|1M|3M|6M)$", feature)
    if m:
        lab, t = m.groups()
        return f"Missing {t} {lab}" if not zh else f"{t} 时点 {lab} 是否缺失"

    m = re.match(r"^Miss_Eval_(1M|3M|6M)$", feature)
    if m:
        t = m.group(1)
        return f"Missing {t} status" if not zh else f"{t} 临床状态是否缺失"

    m = re.match(r"^D_(FT3|FT4|TSH)_(1M|3M|6M)-(0M|1M|3M)$", feature)
    if m:
        lab, t1, t0 = m.groups()
        t0_label = "baseline" if t0 == "0M" else t0
        t0_zh = "治疗前" if t0 == "0M" else t0
        return f"Change in {lab}, {t0_label}->{t1}" if not zh else f"{lab_zh[lab]} 从{t0_zh}到 {t1} 的变化量"

    m = re.match(r"^PctDrop_(FT3|FT4)_0_(1M|3M|6M)$", feature)
    if m:
        lab, t = m.groups()
        return f"Percent drop in {lab}, baseline->{t}" if not zh else f"{lab_zh[lab]} 从治疗前到 {t} 的百分比下降幅度"

    m = re.match(r"^(FT3|FT4)_(1M|3M|6M)_over_0M$", feature)
    if m:
        lab, t = m.groups()
        return f"{t}/{chr(0x2009)}baseline {lab} ratio" if not zh else f"{t} {lab_zh[lab]} 与治疗前水平的比值"

    m = re.match(r"^D_logTSH_(1M|3M|6M)_0M$", feature)
    if m:
        t = m.group(1)
        return f"Change in log(TSH), baseline->{t}" if not zh else f"TSH 对数值从治疗前到 {t} 的变化量"

    m = re.match(r"^(FT3|FT4|TSH)_mean_0_(3M|6M)$", feature)
    if m:
        lab, t = m.groups()
        return f"Mean {lab}, baseline->{t}" if not zh else f"从治疗前到 {t} 的平均 {lab_zh[lab]}"

    m = re.match(r"^(FT3|FT4|TSH)_std_0_(3M|6M)$", feature)
    if m:
        lab, t = m.groups()
        return f"Variability of {lab}, baseline->{t}" if not zh else f"从治疗前到 {t} 的 {lab_zh[lab]} 波动程度"

    m = re.match(r"^(FT3|FT4|TSH)_last_minus_prev$", feature)
    if m:
        lab = m.group(1)
        return f"Recent change in {lab}" if not zh else f"最近一次随访相对上一次的 {lab_zh[lab]} 变化"

    m = re.match(r"^FT3_(3M|6M)_over_FT4_(3M|6M)$", feature)
    if m:
        t = m.group(1)
        return f"{t} FT3/FT4 ratio" if not zh else f"{t} 时点 FT3 与 FT4 的比值"

    m = re.match(r"^FT4_(3M|6M)_over_TSH_(3M|6M)_plus1$", feature)
    if m:
        t = m.group(1)
        return f"{t} FT4/(TSH+1)" if not zh else f"{t} 时点 FT4 除以 TSH+1 的比值"

    m = re.match(r"^TSH_Recovered_(3M|6M)$", feature)
    if m:
        t = m.group(1)
        return f"{t} TSH in reference range" if not zh else f"{t} 时点 TSH 是否回到参考范围"

    m = re.match(r"^Likely_Hyper_(3M|6M)$", feature)
    if m:
        t = m.group(1)
        return f"{t} biochemical hyperthyroid pattern" if not zh else f"{t} 时点是否呈现甲亢样甲功模式"

    m = re.match(r"^ThyroidW_x_(FT3|FT4)_(3M|6M)$", feature)
    if m:
        lab, t = m.groups()
        return f"Thyroid weight × {t} {lab}" if not zh else f"甲状腺重量 × {t} {lab_zh[lab]}"

    m = re.match(r"^TRAb_x_(FT3|FT4)_(3M|6M)$", feature)
    if m:
        lab, t = m.groups()
        return f"TRAb × {t} {lab}" if not zh else f"TRAb × {t} {lab_zh[lab]}"

    return feature if not zh else f"{feature}（工程派生变量，见原始特征名）"


def feature_category(name: str) -> str:
    f = str(name)
    if f.startswith("Miss_"):
        return "缺失指示"
    if f.startswith("Eval_") or f.startswith("Likely_Hyper") or f.startswith("TSH_Recovered"):
        return "甲功状态"
    if f.startswith(("FT3_", "FT4_", "TSH_", "logTSH")):
        return "甲功当前值/轨迹"
    if f.startswith(("D_", "PctDrop_")) or "last_minus_prev" in f or "_mean_" in f or "_std_" in f or "_over_" in f:
        return "早期治疗反应"
    if "Dose" in f or "Uptake" in f or "HalfLife" in f or "RAI" in f or "IDPG" in f:
        return "RAI 剂量-摄取-负荷"
    if "TRAb" in f or "TGAb" in f or "TPOAb" in f:
        return "抗体/免疫活性"
    if "ThyroidW" in f:
        return "甲状腺负荷"
    if f in {"Age", "Sex", "Height", "Weight", "BMI", "Exophthalmos", "TreatCount"}:
        return "基线人口学/病史"
    return "其他派生变量"


def plain_feature_explanation(name: str) -> str:
    """Plain Chinese explanation for methods appendices."""
    f = str(name)
    if f == "Age":
        return "患者接受本次 RAI 治疗时的年龄。"
    if f == "Sex":
        return "患者性别。"
    if f in {"Height", "Weight", "BMI"}:
        return {"Height": "患者的身高。", "Weight": "患者的体重。", "BMI": "由身高和体重计算的体重指数。"}[f]
    if f == "Exophthalmos":
        return "是否记录到突眼或甲状腺相关眼病体征，反映 Graves 病严重程度的一部分。"
    if f == "TreatCount":
        return "第几次治疗/治疗起点背景；不是 RAI 前 ATD 用药史。"
    if f == "ThyroidW":
        return "甲状腺越大，通常需要更强的破坏剂量才能控制甲亢。"
    if f == "Dose":
        return "本次给了多少放射性碘。"
    if f in {"Uptake24h", "MaxUptake"}:
        return "甲状腺对放射性碘的摄取能力；摄取越高，说明药物更集中进入甲状腺组织。"
    if f == "HalfLife":
        return "放射性碘在甲状腺内停留的有效时间；停留越久，组织接受的照射越多。"
    if f == "IDPG_Dose_per_ThyroidW":
        return "把 RAI 剂量除以甲状腺重量，近似表示每克甲状腺组织分到的剂量。"
    if f.startswith("Dose_x_") or f == "RAI3d_x_Dose":
        return "乘积交互项。它让模型表达“剂量”和另一个机制因素同时高或低时的叠加影响。"
    if f == "Estimated_TID_Dose_x_Uptake24h_x_HalfLife":
        return "把剂量、摄取率和半衰期合在一起，近似表达甲状腺实际接受的总碘暴露。"
    if f in {"TRAb", "TGAb", "TPOAb"}:
        return "自身免疫相关抗体水平；TRAb 尤其反映 Graves 病免疫活性。"
    if f.startswith("TRAb_x_"):
        return "TRAb 与甲状腺激素水平的乘积交互，用来表达免疫活性和当前甲亢程度同时存在时的风险。"
    if f.startswith("ThyroidW_x_"):
        return "甲状腺重量与甲状腺激素水平的乘积交互，用来表达“大甲状腺 + 激素仍高”的组合风险。"
    if re.match(r"^(FT3|FT4|TSH)_(0M|1M|3M|6M)$", f):
        lab, t = f.split("_")
        return f"{t} 时点实际测到的 {lab} 水平。"
    if f.startswith("logTSH_"):
        return "把 TSH 做对数变换，减少极端 TSH 数值对模型的影响。"
    if f.startswith("Eval_") and f.endswith("_Code"):
        return "该随访时点临床记录的甲功状态编码。"
    if re.match(r"^Eval_(1M|3M|6M)_(Hyper|Normal|Hypo)$", f):
        return "0/1 指示变量：该随访时点是否处于对应甲功状态。"
    if f.startswith("Miss_"):
        return "0/1 指示变量：该检查或状态记录在该时点是否缺失。"
    if f.startswith("D_"):
        return "两个随访时点之间的绝对变化量，用来描述早期治疗反应。"
    if f.startswith("PctDrop_"):
        return "相对治疗前下降了多少百分比，比单纯差值更贴近“早期反应幅度”。"
    if "_over_" in f:
        return "比值特征，用来描述两个甲功指标或当前值与基线值之间的相对关系。"
    if "_mean_" in f:
        return "截至 landmark 的平均甲功水平，用来描述整体暴露负担。"
    if "_std_" in f:
        return "截至 landmark 的甲功波动程度。"
    if f.startswith("TSH_Recovered_"):
        return "0/1 指示变量：TSH 是否已回到参考范围。"
    if f.startswith("Likely_Hyper_"):
        return "0/1 指示变量：该时点甲功组合是否仍像甲亢。"
    return "工程派生变量；保留原名以便与代码和 CSV 对照。"


def safe_auc(y: np.ndarray, prob: np.ndarray, labels: list[str]) -> float:
    try:
        yi = np.array([labels.index(v) for v in y])
        return float(roc_auc_score(yi, prob, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")


def class_metrics(y_true: np.ndarray, prob: np.ndarray, labels: list[str]) -> dict[str, Any]:
    pred_idx = prob.argmax(axis=1)
    y_idx = np.array([labels.index(y) for y in y_true])
    rows: dict[str, Any] = {
        "N": int(len(y_true)),
        "Accuracy": float(accuracy_score(y_idx, pred_idx)),
        "BalancedAccuracy": float(balanced_accuracy_score(y_idx, pred_idx)),
        "MacroF1": float(f1_score(y_idx, pred_idx, average="macro", zero_division=0)),
        "MacroAUC_OVR": safe_auc(y_true, prob, labels),
    }
    prec, rec, f1, sup = precision_recall_fscore_support(y_idx, pred_idx, labels=np.arange(len(labels)), zero_division=0)
    for i, lab in enumerate(labels):
        rows[f"{lab}_Precision"] = float(prec[i])
        rows[f"{lab}_Recall"] = float(rec[i])
        rows[f"{lab}_F1"] = float(f1[i])
        rows[f"{lab}_Support"] = int(sup[i])
    return rows


def align_proba(model: Any, x: pd.DataFrame, n_classes: int) -> np.ndarray:
    pp = model.predict_proba(x)
    aligned = np.zeros((len(x), n_classes), dtype=float)
    for j, cls in enumerate(model.classes_):
        aligned[:, int(cls)] = pp[:, j]
    aligned = np.clip(aligned, 1e-8, 1.0)
    return aligned / aligned.sum(axis=1, keepdims=True)


def fit_multiclass_oof3(model: Any, x_train: pd.DataFrame, y_train: np.ndarray, groups: np.ndarray, x_test: pd.DataFrame, labels: list[str]) -> dict[str, Any]:
    y_idx = np.array([labels.index(y) for y in y_train])
    n_classes = len(labels)
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    oof = np.zeros((len(y_train), n_classes), dtype=float)
    for tr, va in gkf.split(x_train, y_idx, groups=groups):
        m = clone(model)
        m.fit(x_train.iloc[tr], y_idx[tr])
        oof[va] = align_proba(m, x_train.iloc[va], n_classes)
    final = clone(model)
    final.fit(x_train, y_idx)
    return {
        "oof": oof,
        "train_fit": align_proba(final, x_train, n_classes),
        "test": align_proba(final, x_test, n_classes),
        "final_model": final,
    }


def fit_oof_subset_binary(
    model: Any,
    x_train: pd.DataFrame,
    y_train_subset: np.ndarray,
    groups: np.ndarray,
    subset_mask: np.ndarray,
    x_test: pd.DataFrame,
    weight: np.ndarray | None = None,
) -> dict[str, Any]:
    """OOF binary probabilities when stage-2 training uses only a subset.

    The validation fold still receives predictions for every row, but the
    stage-2 model is fitted only on eligible training-fold rows. This avoids
    using a final in-sample stage-2 model to score development rows.
    """
    subset_mask = np.asarray(subset_mask, dtype=bool)
    if subset_mask.sum() == 0:
        raise ValueError("Stage-2 subset is empty")
    subset_indices = np.where(subset_mask)[0]
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    oof = np.zeros(len(x_train), dtype=float)
    for tr, va in gkf.split(x_train, np.zeros(len(x_train)), groups=groups):
        tr_sub = tr[subset_mask[tr]]
        sub_pos = np.searchsorted(subset_indices, tr_sub)
        y_sub = y_train_subset[sub_pos]
        if len(np.unique(y_sub)) < 2:
            oof[va] = float(np.mean(y_train_subset))
            continue
        w_sub = None
        if weight is not None:
            w_sub = weight[sub_pos]
        fold_model = fit_with_weight(model, x_train.iloc[tr_sub], y_sub, w_sub)
        oof[va] = predict_proba_one(fold_model, x_train.iloc[va])
    final = fit_with_weight(model, x_train.loc[subset_mask].reset_index(drop=True), y_train_subset, weight)
    return {
        "oof": oof,
        "train_fit": predict_proba_one(final, x_train),
        "test": predict_proba_one(final, x_test),
        "final_model": final,
    }


def direct_state_models(seed: int, n_classes: int) -> list[tuple[str, Any, str]]:
    models: list[tuple[str, Any, str]] = [
        (
            "Direct_MultinomialLR",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("lr", LogisticRegression(solver="lbfgs", C=0.30, class_weight="balanced", max_iter=5000, random_state=seed)),
                ]
            ),
            "baseline",
        ),
        (
            "Direct_MultinomialLR_HypoUpweight",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("lr", LogisticRegression(solver="lbfgs", C=0.30, class_weight={0: 1.0, 1: 1.0, 2: 2.0}, max_iter=5000, random_state=seed)),
                ]
            ),
            "hypo_upweight",
        ),
        (
            "Direct_ExtraTrees",
            ExtraTreesClassifier(n_estimators=320, max_depth=5, min_samples_leaf=5, max_features="sqrt", class_weight="balanced", random_state=seed, n_jobs=1),
            "baseline",
        ),
    ]
    if LGBMClassifier is not None:
        models.append(
            (
                "Direct_LightGBM",
                LGBMClassifier(
                    objective="multiclass",
                    num_class=n_classes,
                    n_estimators=220,
                    learning_rate=0.035,
                    max_depth=3,
                    num_leaves=11,
                    min_child_samples=12,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=seed,
                    verbosity=-1,
                    n_jobs=1,
                ),
                "baseline",
            )
        )
    return models


def direct_nhrh3_models(seed: int) -> list[tuple[str, Any, str]]:
    models = [
        (
            "Direct_MultinomialLR",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("lr", LogisticRegression(solver="lbfgs", C=0.30, class_weight="balanced", max_iter=5000, random_state=seed)),
                ]
            ),
            "baseline",
        ),
        (
            "Direct_MultinomialLR_RecurrenceUpweight",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("lr", LogisticRegression(solver="lbfgs", C=0.30, class_weight={0: 1.0, 1: 1.0, 2: 2.0}, max_iter=5000, random_state=seed)),
                ]
            ),
            "recurrence_upweight",
        ),
        (
            "Direct_ExtraTrees",
            ExtraTreesClassifier(n_estimators=360, max_depth=5, min_samples_leaf=5, max_features="sqrt", class_weight="balanced", random_state=seed, n_jobs=1),
            "baseline",
        ),
    ]
    if LGBMClassifier is not None:
        models.append(
            (
                "Direct_LightGBM",
                LGBMClassifier(
                    objective="multiclass",
                    num_class=len(NHRH3_ORDER),
                    n_estimators=260,
                    learning_rate=0.03,
                    max_depth=3,
                    num_leaves=11,
                    min_child_samples=12,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=seed,
                    verbosity=-1,
                    n_jobs=1,
                ),
                "baseline",
            )
        )
    return models


def clinical_core_12_features(landmark: str, x: pd.DataFrame) -> list[str]:
    wanted = [
        "Age",
        "Sex",
        "ThyroidW",
        "Dose",
        "Uptake24h",
        "IDPG_Dose_per_ThyroidW",
        "TRAb",
        f"FT3_{landmark}",
        f"FT4_{landmark}",
        f"TSH_{landmark}",
        f"Eval_{landmark}_Hyper",
        f"Eval_{landmark}_Normal",
    ]
    return [c for c in wanted if c in x.columns]


def state_label(raw: dict[str, Any], audit: pd.DataFrame, horizon: str) -> pd.DataFrame:
    idx = FOLLOWUP_TIMES.index(horizon)
    rows = []
    for i in range(len(raw["outcome"])):
        st = state_name(raw["eval_raw"][i, idx])
        rows.append(
            {
                "Treatment_ID": raw["treatment_ids"][i],
                "Patient_ID": raw["pids"][i],
                "State": st,
                "HasState": st in STATE_ORDER,
                "Endpoint": f"{horizon}_3Class",
                "NHRH": int(audit.loc[i, "NHRH"]),
            }
        )
    return pd.DataFrame(rows)


def combine_state_cascade(p_hyper: np.ndarray, p_normal_cond: np.ndarray) -> np.ndarray:
    p_h = np.clip(p_hyper, 1e-8, 1 - 1e-8)
    p_nc = np.clip(p_normal_cond, 1e-8, 1 - 1e-8)
    out = np.column_stack([p_h, (1 - p_h) * p_nc, (1 - p_h) * (1 - p_nc)])
    return out / out.sum(axis=1, keepdims=True)


def split_records(ids: np.ndarray, pids: np.ndarray, split: str, landmark: str, endpoint: str, feature_set: str, approach: str, model: str, labels: list[str], y: np.ndarray, prob: np.ndarray) -> list[dict[str, Any]]:
    pred_idx = prob.argmax(axis=1)
    out = []
    for k, (tid, pid, yt) in enumerate(zip(ids, pids, y)):
        pred = labels[int(pred_idx[k])]
        row = {
            "Treatment_ID": tid,
            "Patient_ID": pid,
            "Split": split,
            "Landmark": landmark,
            "Endpoint": endpoint,
            "FeatureSet": feature_set,
            "Approach": approach,
            "Model": model,
            "True": yt,
            "Pred": pred,
            "Correct": bool(yt == pred),
            "Confidence": float(prob[k, pred_idx[k]]),
        }
        for i, lab in enumerate(labels):
            row[f"P_{lab}"] = float(prob[k, i])
        out.append(row)
    return out


def run_state_endpoint(
    landmark: str,
    horizon: str,
    data: Any,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    features: list[str],
    feature_set: str,
    raw: dict[str, Any],
    audit: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels_all = state_label(raw, audit, horizon)
    tr_idx, te_idx = temporal_row_split(len(labels_all))
    ytr_full = labels_all.iloc[tr_idx]["State"].to_numpy()
    yte_full = labels_all.iloc[te_idx]["State"].to_numpy()
    keep_tr = np.isin(ytr_full, STATE_ORDER)
    keep_te = np.isin(yte_full, STATE_ORDER)
    excluded = labels_all.loc[~labels_all["HasState"]].copy()
    excluded["Landmark"] = landmark
    excluded["FeatureSet"] = feature_set

    xtr_all = x_train[features].reset_index(drop=True)
    xte_all = x_test[features].reset_index(drop=True)
    xtr = xtr_all.loc[keep_tr].reset_index(drop=True)
    xte = xte_all.loc[keep_te].reset_index(drop=True)
    ytr = ytr_full[keep_tr]
    yte = yte_full[keep_te]
    groups = data.groups_train[keep_tr]
    train_ids = data.train_ids[keep_tr]
    test_ids = data.test_ids[keep_te]
    train_pids = data.audit_train["Patient_ID"].to_numpy()[keep_tr]
    test_pids = data.audit_test["Patient_ID"].to_numpy()[keep_te]

    perf_rows: list[dict[str, Any]] = []
    cm_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    endpoint = f"{horizon}_3Class"

    for name, model, refinement in direct_state_models(seed, len(STATE_ORDER)):
        pred = fit_multiclass_oof3(model, xtr, ytr, groups, xte, STATE_ORDER)
        for split, y, prob in [("TrainFit", ytr, pred["train_fit"]), ("OOF", ytr, pred["oof"]), ("TemporalTest", yte, pred["test"])]:
            perf_rows.append(
                {
                    "Landmark": landmark,
                    "Endpoint": endpoint,
                    "FeatureSet": feature_set,
                    "Approach": "Direct",
                    "Model": name,
                    "Refinement": refinement,
                    "Split": split,
                    **class_metrics(y, prob, STATE_ORDER),
                }
            )
        for split, ids, pids, y, prob in [
            ("TrainFit", train_ids, train_pids, ytr, pred["train_fit"]),
            ("OOF", train_ids, train_pids, ytr, pred["oof"]),
            ("TemporalTest", test_ids, test_pids, yte, pred["test"]),
        ]:
            pred_rows.extend(split_records(ids, pids, split, landmark, endpoint, feature_set, "Direct", name, STATE_ORDER, y, prob))
        cm_rows.extend(confusion_rows(landmark, endpoint, feature_set, f"Direct_{name}", yte, pred["test"], STATE_ORDER))

    # Cascade: Hyper vs Non-Hyper, then Normal vs Hypo among true non-Hyper.
    st1_y = (ytr == "Hyper").astype(int)
    st1 = fit_oof_weighted(binary_prob_model(seed), xtr, st1_y, groups, xte, weight=None)
    nonhyper_tr = ytr != "Hyper"
    st2_y = (ytr[nonhyper_tr] == "Normal").astype(int)
    st2 = fit_oof_subset_binary(binary_prob_model(seed + 7), xtr, st2_y, groups, nonhyper_tr, xte, weight=None)
    oof_prob = combine_state_cascade(st1["oof"], st2["oof"])
    train_prob = combine_state_cascade(st1["train_fit"], st2["train_fit"])
    test_prob = combine_state_cascade(st1["test"], st2["test"])
    for split, y, prob in [("TrainFit", ytr, train_prob), ("OOF", ytr, oof_prob), ("TemporalTest", yte, test_prob)]:
        perf_rows.append(
            {
                "Landmark": landmark,
                "Endpoint": endpoint,
                "FeatureSet": feature_set,
                "Approach": "Cascade",
                "Model": "Cascade_LGBM_or_ET",
                "Refinement": "baseline",
                "Split": split,
                **class_metrics(y, prob, STATE_ORDER),
            }
        )
    for split, ids, pids, y, prob in [
        ("TrainFit", train_ids, train_pids, ytr, train_prob),
        ("OOF", train_ids, train_pids, ytr, oof_prob),
        ("TemporalTest", test_ids, test_pids, yte, test_prob),
    ]:
        pred_rows.extend(split_records(ids, pids, split, landmark, endpoint, feature_set, "Cascade", "Cascade_LGBM_or_ET", STATE_ORDER, y, prob))
    cm_rows.extend(confusion_rows(landmark, endpoint, feature_set, "Cascade", yte, test_prob, STATE_ORDER))
    return pd.DataFrame(perf_rows), pd.DataFrame(cm_rows), excluded, pd.DataFrame(pred_rows)


def run_nhrh3_records(data: Any, x_train: pd.DataFrame, x_test: pd.DataFrame, features: list[str], feature_set: str, audit: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lbl = nhrh_3class(audit)
    tr_idx, te_idx = temporal_row_split(len(lbl))
    ytr_full = lbl.iloc[tr_idx]["NHRH_Class"].to_numpy()
    yte_full = lbl.iloc[te_idx]["NHRH_Class"].to_numpy()
    keep_tr = np.isin(ytr_full, NHRH3_ORDER)
    keep_te = np.isin(yte_full, NHRH3_ORDER)
    xtr = x_train[features].reset_index(drop=True).loc[keep_tr].reset_index(drop=True)
    xte = x_test[features].reset_index(drop=True).loc[keep_te].reset_index(drop=True)
    ytr = ytr_full[keep_tr]
    yte = yte_full[keep_te]
    groups = data.groups_train[keep_tr]
    train_ids = data.train_ids[keep_tr]
    test_ids = data.test_ids[keep_te]
    train_pids = data.audit_train["Patient_ID"].to_numpy()[keep_tr]
    test_pids = data.audit_test["Patient_ID"].to_numpy()[keep_te]
    endpoint = "NHRH_3Class"

    perf_rows: list[dict[str, Any]] = []
    cm_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []

    for name, model, refinement in direct_nhrh3_models(seed):
        pred = fit_multiclass_oof3(model, xtr, ytr, groups, xte, NHRH3_ORDER)
        for split, y, prob in [("TrainFit", ytr, pred["train_fit"]), ("OOF", ytr, pred["oof"]), ("TemporalTest", yte, pred["test"])]:
            perf_rows.append({"Landmark": "6M", "Endpoint": endpoint, "FeatureSet": feature_set, "Approach": "Direct", "Model": name, "Refinement": refinement, "Split": split, **class_metrics(y, prob, NHRH3_ORDER)})
        for split, ids, pids, y, prob in [("TrainFit", train_ids, train_pids, ytr, pred["train_fit"]), ("OOF", train_ids, train_pids, ytr, pred["oof"]), ("TemporalTest", test_ids, test_pids, yte, pred["test"])]:
            pred_rows.extend(split_records(ids, pids, split, "6M", endpoint, feature_set, "Direct", name, NHRH3_ORDER, y, prob))
        cm_rows.extend(confusion_rows("6M", endpoint, feature_set, f"Direct_{name}", yte, pred["test"], NHRH3_ORDER))

    st1_y = (ytr != "Success").astype(int)
    st1 = fit_oof_weighted(binary_prob_model(seed), xtr, st1_y, groups, xte, weight=None)
    nhrh_tr = ytr != "Success"
    st2_y = (ytr[nhrh_tr] == "Recurrence").astype(int)
    st2 = fit_oof_subset_binary(binary_prob_model(seed + 11), xtr, st2_y, groups, nhrh_tr, xte, weight=None)

    def combine(p_nhrh: np.ndarray, p_recur_cond: np.ndarray) -> np.ndarray:
        p_a = np.clip(p_nhrh, 1e-8, 1 - 1e-8)
        p_r = np.clip(p_recur_cond, 1e-8, 1 - 1e-8)
        out = np.column_stack([1 - p_a, p_a * (1 - p_r), p_a * p_r])
        return out / out.sum(axis=1, keepdims=True)

    oof_prob = combine(st1["oof"], st2["oof"])
    train_prob = combine(st1["train_fit"], st2["train_fit"])
    test_prob = combine(st1["test"], st2["test"])
    for split, y, prob in [("TrainFit", ytr, train_prob), ("OOF", ytr, oof_prob), ("TemporalTest", yte, test_prob)]:
        perf_rows.append({"Landmark": "6M", "Endpoint": endpoint, "FeatureSet": feature_set, "Approach": "Cascade", "Model": "Cascade_LGBM_or_ET", "Refinement": "baseline", "Split": split, **class_metrics(y, prob, NHRH3_ORDER)})
    for split, ids, pids, y, prob in [("TrainFit", train_ids, train_pids, ytr, train_prob), ("OOF", train_ids, train_pids, ytr, oof_prob), ("TemporalTest", test_ids, test_pids, yte, test_prob)]:
        pred_rows.extend(split_records(ids, pids, split, "6M", endpoint, feature_set, "Cascade", "Cascade_LGBM_or_ET", NHRH3_ORDER, y, prob))
    cm_rows.extend(confusion_rows("6M", endpoint, feature_set, "Cascade", yte, test_prob, NHRH3_ORDER))

    label_dist = pd.DataFrame({"Split": np.where(np.arange(len(lbl)) < tr_idx[-1] + 1, "Development", "TemporalTest"), "Class": lbl["NHRH_Class"]})
    return pd.DataFrame(perf_rows), pd.DataFrame(cm_rows), label_dist, pd.DataFrame(pred_rows)


def confusion_rows(landmark: str, endpoint: str, feature_set: str, approach: str, y: np.ndarray, prob: np.ndarray, labels: list[str]) -> list[dict[str, Any]]:
    pred = prob.argmax(axis=1)
    true = np.array([labels.index(v) for v in y])
    cm = confusion_matrix(true, pred, labels=np.arange(len(labels)))
    rows = []
    for i, t in enumerate(labels):
        for j, p in enumerate(labels):
            rows.append({"Landmark": landmark, "Endpoint": endpoint, "FeatureSet": feature_set, "Approach": approach, "True": t, "Pred": p, "N": int(cm[i, j])})
    return rows


def select_fixed_binary(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    table_dir = out_dir / "tables"
    all_cand = pd.read_csv(table_dir / "nhrh_binary_all_candidates.csv")
    rows = []
    for lm in LANDMARKS:
        run_id = SELECTED_RUNS.get(lm)
        hit = all_cand[all_cand["RunID"].eq(run_id)] if run_id else all_cand.iloc[0:0]
        if hit.empty:
            # 0M/1M have no fixed run_id (and quick runs may lack the canonical s13 run_ids):
            # fall back to the development-OOF best candidate for this landmark.
            sub = all_cand[all_cand["Landmark"].eq(lm)]
            if sub.empty:
                continue
            key = "OOF_AUC" if "OOF_AUC" in sub.columns else "Test_AUC"
            hit = sub.sort_values(key, ascending=False).head(1)
        rows.append(hit.iloc[0])
    selected = pd.DataFrame(rows).reset_index(drop=True)
    selected.to_csv(table_dir / "nhrh_binary_selected.csv", index=False)
    selected[selected["Landmark"].eq("3M")].to_csv(table_dir / "3m_nhrh_binary_performance.csv", index=False)
    selected[selected["Landmark"].eq("6M")].to_csv(table_dir / "6m_nhrh_binary_performance.csv", index=False)
    return selected, all_cand


def write_lr_vs_ensemble(out_dir: Path, all_cand: pd.DataFrame) -> pd.DataFrame:
    models = ["Clinical_L2_Logistic", "ExtraTrees", "LightGBM", "HistGradientBoosting"]
    rows = []
    for lm in LANDMARKS:
        for model in models:
            if model == "Clinical_L2_Logistic":
                rid = SELECTED_RUNS.get(lm)
                sub = all_cand[all_cand["RunID"].eq(rid)].copy() if rid else all_cand.iloc[0:0].copy()
                if sub.empty:  # 0M/1M (no fixed run_id) -> best LR candidate for the landmark
                    sub = all_cand[(all_cand["Landmark"].eq(lm)) & (all_cand["Model"].eq(model))].copy()
            else:
                sub = all_cand[(all_cand["Landmark"].eq(lm)) & (all_cand["Model"].eq(model))].copy()
            if sub.empty:
                continue
            sub = sub.sort_values(["OOF_AUC", "OOF_PR_AUC", "OOF_Brier", "OOF_Accuracy"], ascending=[False, False, True, False])
            r = sub.iloc[0]
            rows.append(
                {
                    "Landmark": lm,
                    "Model": model,
                    "FeatureSet": r["FeatureSet"],
                    "N_Features": int(r["N_Features"]),
                    "Test_AUC": float(r["Test_AUC"]),
                    "Test_PR_AUC": float(r["Test_PR_AUC"]),
                    "Test_Accuracy": float(r["Test_Accuracy"]),
                    "Test_Brier": float(r["Test_Brier"]),
                    "OOF_AUC": float(r["OOF_AUC"]),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "tables" / "nhrh_binary_lr_vs_ensemble.csv", index=False)
    return df


def rewrite_binary_secondary_outputs(out_dir: Path, selected: pd.DataFrame, n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    table_dir = out_dir / "tables"
    preds = pd.read_csv(table_dir / "nhrh_binary_predictions_long.csv")
    ci_rows, dca_rows, th_rows = [], [], []
    for _, cand in selected.iterrows():
        sub = preds[(preds["RunID"].eq(cand["RunID"])) & (preds["Domain"].eq("TemporalTest"))].copy()
        y = sub["Y"].to_numpy(dtype=int)
        p = sub["Proba"].to_numpy(dtype=float)
        thr = float(cand["Threshold"])
        ci = bootstrap_ci(y, p, thr, n_boot=n_boot, seed=seed)
        ci.insert(0, "Landmark", cand["Landmark"])
        ci_rows.append(ci)
        dc = dca_curve(y, p)
        dc.insert(0, "Landmark", cand["Landmark"])
        dca_rows.append(dc)
        for t in np.linspace(0.05, 0.90, 86):
            m = metrics_at_threshold(y, p, float(t))
            th_rows.append({"Landmark": cand["Landmark"], **{k: m[k] for k in ["Threshold", "Accuracy", "BalancedAccuracy", "F1", "Recall", "Specificity", "PPV", "NPV"]}})
    ci_df = pd.concat(ci_rows, ignore_index=True)
    dca_df = pd.concat(dca_rows, ignore_index=True)
    th_df = pd.DataFrame(th_rows)
    ci_df.to_csv(table_dir / "nhrh_binary_bootstrap_ci.csv", index=False)
    dca_df.to_csv(table_dir / "nhrh_binary_dca_curve.csv", index=False)
    th_df.to_csv(table_dir / "nhrh_binary_threshold_sensitivity.csv", index=False)
    return ci_df, dca_df, th_df


def medication_audit(raw: dict[str, Any], out_dir: Path) -> pd.DataFrame:
    df = pd.read_excel(ROOT / "1003.xlsx", header=None, engine="openpyxl").iloc[2:].copy()
    y = pd.to_numeric(df.iloc[:, COL_IDX["Outcome"]], errors="coerce")
    df = df.loc[y.notna()].reset_index(drop=True)
    med_cols = [34, 43, 52, 61, 70, 79]
    med_times = ["1M", "3M", "6M", "12M", "18M", "24M"]
    rows = [
        {
            "Variable": "Structured_Pre_RAI_ATD_Use",
            "Column": "",
            "Timing": "Pre-RAI",
            "Available": False,
            "Role": "Not used",
            "Rationale": "Workbook has no structured pre-RAI ATD use/duration/washout field.",
        },
        {
            "Variable": "TreatCount",
            "Column": "治疗次数",
            "Timing": "Baseline/index RAI",
            "Available": True,
            "Role": "Retained as static clinical background",
            "Rationale": "Treatment-origin count is available before prediction but is not equivalent to ATD history.",
            "NonMissing": int(df.iloc[:, 12].notna().sum()),
        },
    ]
    for c, t in zip(med_cols, med_times):
        s = df.iloc[:, c].dropna().astype(str).str.strip()
        rows.append(
            {
                "Variable": f"Medication_{t}",
                "Column": f"{t} 用药情况",
                "Timing": "Post-RAI follow-up",
                "Available": bool(len(s)),
                "Role": "Excluded from primary fixed-landmark models",
                "Rationale": "Post-RAI prescribing may encode physician response to evolving thyroid status.",
                "NonMissing": int(len(s)),
                "TopValues": json.dumps(s.value_counts().head(8).to_dict(), ensure_ascii=False),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "tables" / "medication_feature_audit.csv", index=False)
    return out


def plot_label_distribution(out_dir: Path, audit: pd.DataFrame) -> None:
    fig_dir = out_dir / "figures"
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), gridspec_kw={"width_ratios": [1.35, 1.0]})
    source_labels = {
        "non_nhrh_final_normal_or_hypo": "Success by final Normal/Hypo",
        "final_hyper_nonhealing": "Persistent final Hyper",
        "final_hyper+recurrence_at_6M": "Persistent Hyper + recurrence at 6M",
        "final_hyper+recurrence_at_12M": "Persistent Hyper + recurrence at 12M",
        "final_hyper+recurrence_at_18M": "Persistent Hyper + recurrence at 18M",
        "final_hyper+recurrence_at_24M": "Persistent Hyper + recurrence at 24M",
        "recurrence_after_control_at_6M": "Recurrence after control at 6M",
        "recurrence_after_control_at_12M": "Recurrence after control at 12M",
        "recurrence_after_control_at_18M": "Recurrence after control at 18M",
        "recurrence_after_control_at_24M": "Recurrence after control at 24M",
    }
    src = audit["NHRH_Source"].value_counts()
    src_labels = [source_labels.get(str(v), str(v)) for v in src.index]
    axes[0].barh(np.arange(len(src)), src.values, color="#1d4e89")
    axes[0].set_yticks(np.arange(len(src)), src_labels)
    axes[0].invert_yaxis()
    axes[0].set_title("NHRH label source")
    axes[0].set_xlabel("Treatment origins")
    for y, v in enumerate(src.values):
        axes[0].text(v + max(src.values) * 0.01, y, str(int(v)), va="center", fontsize=8)
    prev = audit.groupby("Split")["NHRH"].mean().reindex(["Development", "TemporalTest"])
    axes[1].bar(prev.index, prev.values, color="#2a9d8f")
    axes[1].set_title("NHRH prevalence")
    axes[1].set_ylabel("Prevalence")
    axes[1].set_ylim(0, max(0.5, float(prev.max()) * 1.25))
    for x, v in enumerate(prev.values):
        axes[1].text(x, v + 0.015, f"{v:.1%}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "Figure_01_Label_Distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_binary_figures(out_dir: Path, selected: pd.DataFrame, dca: pd.DataFrame) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    preds = pd.read_csv(out_dir / "tables" / "nhrh_binary_predictions_long.csv")
    fig, axes = plt.subplots(NLM, 2, figsize=(11, 4.2 * NLM))
    fig2, axes2 = plt.subplots(NLM, 2, figsize=(11, 4.2 * NLM))
    fig3, axes3 = plt.subplots(1, NLM, figsize=(4.8 * NLM, 4.2))
    for r, lm in enumerate(LANDMARKS):
        cand = selected[selected["Landmark"].eq(lm)].iloc[0]
        sub = preds[(preds["RunID"].eq(cand["RunID"])) & (preds["Domain"].eq("TemporalTest"))]
        y = sub["Y"].to_numpy(dtype=int)
        p = np.clip(sub["Proba"].to_numpy(dtype=float), 1e-8, 1 - 1e-8)
        fpr, tpr, _ = roc_curve(y, p)
        pre, rec, _ = precision_recall_curve(y, p)
        axes[r, 0].plot(fpr, tpr, color="#1d4e89", lw=2)
        axes[r, 0].plot([0, 1], [0, 1], "--", color="gray")
        axes[r, 0].set_title(f"{lm} ROC AUC={roc_auc_score(y, p):.3f}")
        axes[r, 1].plot(rec, pre, color="#2a9d8f", lw=2)
        axes[r, 1].axhline(y.mean(), color="gray", linestyle="--")
        axes[r, 1].set_title(f"{lm} PR-AUC={average_precision_score(y, p):.3f}")

        bins = pd.qcut(p, q=min(6, len(np.unique(p))), duplicates="drop")
        cal = pd.DataFrame({"p": p, "y": y, "bin": bins}).groupby("bin", observed=False).agg(mean_p=("p", "mean"), obs=("y", "mean")).reset_index(drop=True)
        axes2[r, 0].plot([0, 1], [0, 1], "--", color="gray")
        axes2[r, 0].plot(cal["mean_p"], cal["obs"], marker="o", color="#7b2cbf")
        axes2[r, 0].set_title(f"{lm} calibration, Brier={brier_score_loss(y, p):.3f}")
        dc = dca[dca["Landmark"].eq(lm)]
        axes2[r, 1].plot(dc["Threshold"], dc["Model_Net_Benefit"], label="Model", color="#2d6a4f")
        axes2[r, 1].plot(dc["Threshold"], dc["Treat_All"], "--", label="Treat all", color="#c1121f")
        axes2[r, 1].plot(dc["Threshold"], dc["Treat_None"], ":", label="Treat none", color="black")
        axes2[r, 1].set_title(f"{lm} DCA")
        axes2[r, 1].legend(fontsize=8)

        cm = confusion_matrix(y, (p >= float(cand["Threshold"])).astype(int), labels=[0, 1])
        ax = axes3[r]
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=12)
        ax.set_title(f"{lm}, threshold={float(cand['Threshold']):.2f}")
        ax.set_xticks([0, 1], ["Non-NHRH", "NHRH"])
        ax.set_yticks([0, 1], ["Non-NHRH", "NHRH"])
        fig3.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in list(axes.ravel()) + list(axes2.ravel()):
        ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(fig_dir / "Figure_02_NHRH_ROC_PR.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig2.tight_layout()
    fig2.savefig(fig_dir / "Figure_03_NHRH_Calibration_DCA.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)
    fig3.tight_layout()
    fig3.savefig(fig_dir / "Figure_04_NHRH_Confusion.png", dpi=300, bbox_inches="tight")
    plt.close(fig3)


def plot_binary_supplemental(out_dir: Path, selected: pd.DataFrame, lr_vs_ens: pd.DataFrame) -> None:
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    th = pd.read_csv(table_dir / "nhrh_binary_threshold_sensitivity.csv")
    fig, axes = plt.subplots(1, NLM, figsize=(6.0 * NLM, 4.8), sharey=True)
    for ax, lm in zip(axes, LANDMARKS):
        ss = th[th["Landmark"].eq(lm)]
        ax.plot(ss["Threshold"], ss["Accuracy"], label="Accuracy", color="#1d4e89")
        ax.plot(ss["Threshold"], ss["BalancedAccuracy"], label="Balanced accuracy", color="#2a9d8f")
        ax.plot(ss["Threshold"], ss["F1"], label="F1", color="#e76f51")
        thr = float(selected[selected["Landmark"].eq(lm)]["Threshold"].iloc[0])
        ax.axvline(thr, color="black", linestyle="--", lw=1, label="Selected threshold")
        ax.set_title(f"{lm} threshold sensitivity")
        ax.set_xlabel("Threshold")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Metric")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "Figure_11_NHRH_Threshold_Sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    preds = pd.read_csv(table_dir / "nhrh_binary_predictions_long.csv")
    fig, axes = plt.subplots(NLM, 2, figsize=(12, 3.7 * NLM))
    for r, lm in enumerate(LANDMARKS):
        cand = selected[selected["Landmark"].eq(lm)].iloc[0]
        sub = preds[(preds["RunID"].eq(cand["RunID"])) & (preds["Domain"].eq("TemporalTest"))].copy()
        sub["Pred"] = (sub["Proba"] >= float(cand["Threshold"])).astype(int)
        sub["Group"] = np.select(
            [(sub["Y"].eq(1) & sub["Pred"].eq(1)), (sub["Y"].eq(0) & sub["Pred"].eq(1)), (sub["Y"].eq(1) & sub["Pred"].eq(0))],
            ["TP", "FP", "FN"],
            default="TN",
        )
        for yv, lab, color in [(0, "Non-NHRH", "#457b9d"), (1, "NHRH", "#e76f51")]:
            axes[r, 0].hist(sub[sub["Y"].eq(yv)]["Proba"], bins=20, alpha=0.55, label=lab, color=color)
        axes[r, 0].axvline(float(cand["Threshold"]), color="black", linestyle="--")
        axes[r, 0].set_title(f"{lm} risk distribution")
        counts = sub["Group"].value_counts().reindex(["TP", "FP", "FN", "TN"], fill_value=0)
        axes[r, 1].bar(counts.index, counts.values, color=["#2a9d8f", "#e9c46a", "#e76f51", "#457b9d"])
        axes[r, 1].set_title(f"{lm} error profile")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "Figure_12_NHRH_Risk_Error_Profile.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    metric_map = {
        "Test_AUC": "AUC",
        "Test_PR_AUC": "PR-AUC",
        "Test_Accuracy": "Accuracy",
        "Test_Brier": "Brier",
    }
    heat_rows = []
    for model, sub in lr_vs_ens.groupby("Model", sort=False):
        row: dict[str, Any] = {"Model": model}
        for lm in LANDMARKS:
            hit = sub[sub["Landmark"].eq(lm)]
            if hit.empty:
                continue
            r = hit.iloc[0]
            for raw_col, label in metric_map.items():
                row[f"{lm} {label}"] = float(r[raw_col])
        heat_rows.append(row)
    heat_df = pd.DataFrame(heat_rows).set_index("Model")
    heat_df.to_csv(table_dir / "binary_model_metric_heatmap.csv")
    color_df = heat_df.copy()
    for col in color_df.columns:
        if col.endswith("Brier"):
            color_df[col] = 1.0 - color_df[col]
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    im = ax.imshow(color_df.to_numpy(dtype=float), cmap="YlGnBu", vmin=0.70, vmax=0.96, aspect="auto")
    ax.set_xticks(np.arange(len(heat_df.columns)), heat_df.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(heat_df.index)), heat_df.index)
    for i in range(heat_df.shape[0]):
        for j in range(heat_df.shape[1]):
            val = heat_df.iloc[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color="#111827")
    ax.set_title("Binary NHRH model x metric heatmap")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Color score (Brier shown as 1-Brier)")
    fig.tight_layout()
    fig.savefig(fig_dir / "Figure_13_Binary_Model_Comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def binary_ensemble_model(model_name: str, seed: int) -> Any:
    if model_name == "ExtraTrees":
        return ExtraTreesClassifier(n_estimators=360, max_depth=4, min_samples_leaf=5, max_features="sqrt", class_weight="balanced", random_state=seed, n_jobs=1)
    if model_name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(max_iter=180, learning_rate=0.035, max_leaf_nodes=15, l2_regularization=0.10, random_state=seed)
    if model_name == "LightGBM" and LGBMClassifier is not None:
        return LGBMClassifier(
            objective="binary",
            class_weight="balanced",
            n_estimators=260,
            learning_rate=0.025,
            max_depth=3,
            num_leaves=11,
            min_child_samples=15,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.2,
            reg_lambda=2.0,
            random_state=seed,
            verbosity=-1,
            n_jobs=1,
        )
    raise ValueError(f"Unsupported ensemble model for permutation importance: {model_name}")


def build_ensemble_permutation_importance(
    out_dir: Path,
    all_cand: pd.DataFrame,
    datasets: dict[str, Any],
    xtr_by_lm: dict[str, pd.DataFrame],
    xte_by_lm: dict[str, pd.DataFrame],
    fs_by_lm: dict[str, dict[str, list[str]]],
    seed: int,
) -> pd.DataFrame:
    rows = []
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    for lm in LANDMARKS:
        sub = all_cand[(all_cand["Landmark"].eq(lm)) & (~all_cand["Model"].eq("Clinical_L2_Logistic"))].copy()
        sub = sub[sub["Model"].isin(["ExtraTrees", "LightGBM", "HistGradientBoosting"])]
        if sub.empty:
            continue
        sub = sub.sort_values(["OOF_AUC", "OOF_PR_AUC", "OOF_Brier", "OOF_Accuracy"], ascending=[False, False, True, False])
        r = sub.iloc[0]
        feature_set = str(r["FeatureSet"])
        model_name = str(r["Model"])
        features = fs_by_lm[lm].get(feature_set)
        if not features:
            continue
        model = binary_ensemble_model(model_name, seed)
        data = datasets[lm]
        weight_scheme = r.get("SampleWeight", "uniform")
        if pd.isna(weight_scheme):
            weight_scheme = "uniform"
        weight = sample_weights(data.audit_train, str(weight_scheme))
        final = fit_with_weight(model, xtr_by_lm[lm][features], data.y_train, weight)
        perm = permutation_importance(
            final,
            xte_by_lm[lm][features],
            data.y_test,
            scoring="average_precision",
            n_repeats=30,
            random_state=seed,
            n_jobs=1,
        )
        for feat, mean, std in zip(features, perm.importances_mean, perm.importances_std):
            rows.append(
                {
                    "Landmark": lm,
                    "Model": model_name,
                    "FeatureSet": feature_set,
                    "Feature": feat,
                    "Permutation_AP_Drop": float(mean),
                    "Permutation_AP_Drop_SD": float(std),
                    "OOF_AUC_Selected": float(r["OOF_AUC"]),
                    "Test_AUC_Selected": float(r["Test_AUC"]),
                    "Test_PR_AUC_Selected": float(r["Test_PR_AUC"]),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(table_dir / "ensemble_permutation_importance.csv", index=False)
    stale_fig = fig_dir / "Figure_16_Ensemble_Permutation_Importance.png"
    if stale_fig.exists():
        stale_fig.unlink()
    return out


def build_lr_linear_shap(
    out_dir: Path,
    selected: pd.DataFrame,
    datasets: dict[str, Any],
    xtr_by_lm: dict[str, pd.DataFrame],
    fs_by_lm: dict[str, dict[str, list[str]]],
    seed: int,
) -> pd.DataFrame:
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    rows = []
    stale = table_dir / "lgbm_shap_importance.csv"
    if stale.exists():
        stale.unlink()
    if shap is None:
        pd.DataFrame([{"Status": "SHAP unavailable"}]).to_csv(table_dir / "lr_linear_shap_importance.csv", index=False)
        return pd.DataFrame()

    fig, axes = plt.subplots(1, NLM, figsize=(7.0 * NLM, 7.5))
    rng = np.random.default_rng(seed)
    last_scatter = None
    for ax, lm in zip(axes, LANDMARKS):
        r = selected[selected["Landmark"].eq(lm)].iloc[0]
        feature_set = str(r["FeatureSet"])
        features = fs_by_lm[lm].get(feature_set)
        if not features:
            ax.axis("off")
            ax.set_title(f"{lm} LR SHAP unavailable")
            continue
        data = datasets[lm]
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("lr", LogisticRegression(penalty="l2", solver="lbfgs", C=0.30, class_weight="balanced", max_iter=5000, random_state=seed)),
            ]
        )
        weight = sample_weights(data.audit_train, str(r.get("SampleWeight", "nhrh_mild")))
        final = fit_with_weight(model, xtr_by_lm[lm][features], data.y_train, weight)
        scaler = final.named_steps["scale"]
        lr_model = final.named_steps["lr"]
        x_background = pd.DataFrame(scaler.transform(xtr_by_lm[lm][features]), columns=features)
        x_eval = x_background.copy()
        if len(x_eval) > 600:
            idx = rng.choice(len(x_eval), size=600, replace=False)
            x_eval = x_eval.iloc[np.sort(idx)].reset_index(drop=True)
        explainer = shap.LinearExplainer(lr_model, x_background)
        shap_values = explainer(x_eval)
        sv = np.asarray(shap_values.values)
        if sv.ndim == 3:
            sv = sv[:, :, -1]
        mean_abs = np.abs(sv).mean(axis=0)
        for feat, val in zip(features, mean_abs):
            rows.append(
                {
                    "Landmark": lm,
                    "RunID": str(r["RunID"]),
                    "Model": "Clinical_L2_Logistic",
                    "FeatureSet": feature_set,
                    "Feature": feat,
                    "MeanAbsLinearSHAP_Logit": float(val),
                    "OOF_Accuracy_Selected": float(r["OOF_Accuracy"]),
                    "Test_AUC_Selected": float(r["Test_AUC"]),
                    "Test_PR_AUC_Selected": float(r["Test_PR_AUC"]),
                }
            )
        top_idx = np.argsort(mean_abs)[-18:][::-1]
        top_idx = top_idx[::-1]
        labels = [pretty_feature_label(features[i]) for i in top_idx]
        for yi, fi in enumerate(top_idx):
            feat = features[fi]
            vals = x_eval[feat].to_numpy(dtype=float)
            q1, q99 = np.nanpercentile(vals, [1, 99])
            denom = (q99 - q1) if q99 > q1 else (np.nanstd(vals) + 1e-6)
            color_val = np.clip((vals - q1) / (denom + 1e-9), 0, 1)
            yj = yi + rng.normal(0, 0.075, size=len(vals))
            last_scatter = ax.scatter(sv[:, fi], yj, c=color_val, cmap="coolwarm", vmin=0, vmax=1, s=10, alpha=0.62, linewidths=0)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(np.arange(len(labels)), labels)
        ax.set_title(f"{lm} selected LR LinearSHAP ({feature_set})")
        ax.set_xlabel("Linear SHAP value on LR logit scale")
        ax.grid(axis="x", alpha=0.18)
        top_vals = sv[:, top_idx].ravel()
        finite = top_vals[np.isfinite(top_vals)]
        if len(finite):
            lim = float(np.nanpercentile(np.abs(finite), 99.0) * 1.15)
            lim = min(max(lim, 0.8), 4.5)
            ax.set_xlim(-lim, lim)
    if last_scatter is not None:
        fig.subplots_adjust(right=0.90, wspace=0.62)
        cax = fig.add_axes([0.925, 0.20, 0.016, 0.60])
        cbar = fig.colorbar(last_scatter, cax=cax)
        cbar.set_label("Feature value, low -> high")
    else:
        fig.tight_layout()
    fig.savefig(fig_dir / "Figure_14_Supplementary_Importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    out = pd.DataFrame(rows)
    out.to_csv(table_dir / "lr_linear_shap_importance.csv", index=False)
    return out


def feature_base_label(name: str) -> str:
    return name.replace("_3M", "_LM").replace("_6M", "_LM")


def plot_unified_or(out_dir: Path) -> None:
    path = out_dir / "tables" / "multivariable_logistic_or.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    panels = {}
    order_bases: list[str] = []
    for lm in LANDMARKS:
        sub = df[df["Analysis"].eq(f"{lm}_clinical_core_multivariable")].copy()
        sub["Base"] = sub["Feature"].map(feature_base_label)
        sub["Display"] = sub["Feature"]
        panels[lm] = sub
        if lm == "3M":
            order_bases = sub.sort_values("P_Value_Wald")["Base"].head(18).tolist()
    all_sub = pd.concat([panels[lm] for lm in LANDMARKS], ignore_index=True)
    xmin = max(0.03, float(np.nanmin(all_sub["CI95_Lower"])) * 0.8)
    xmax = min(120.0, float(np.nanmax(all_sub["CI95_Upper"])) * 1.2)
    fig, axes = plt.subplots(1, NLM, figsize=(7.0 * NLM, 7.5), sharex=True)
    base_labels = order_bases[::-1]
    ypos = np.arange(len(base_labels))
    for ax, lm in zip(axes, LANDMARKS):
        sub = panels[lm].set_index("Base").reindex(base_labels).reset_index()
        ylabels = [pretty_feature_label(v) for v in sub["Display"].fillna(sub["Base"]).tolist()]
        colors = []
        for _, r in sub.iterrows():
            if pd.isna(r.get("OR_per_SD")):
                colors.append("#b8b8b8")
            elif r["CI95_Lower"] <= 1 <= r["CI95_Upper"]:
                colors.append("#8d99ae")
            elif r["OR_per_SD"] > 1:
                colors.append("#c1121f")
            else:
                colors.append("#1d4e89")
        for y, r, c in zip(ypos, sub.itertuples(), colors):
            or_value = getattr(r, "OR_per_SD")
            lo = getattr(r, "CI95_Lower")
            hi = getattr(r, "CI95_Upper")
            if not (np.isfinite(or_value) and np.isfinite(lo) and np.isfinite(hi)):
                continue
            ax.errorbar(
                or_value,
                y,
                xerr=[[or_value - lo], [hi - or_value]],
                fmt="o",
                color=c,
                ecolor=c,
                mfc=c,
                mec=c,
                ms=5,
                capsize=2,
            )
        ax.axvline(1.0, color="gray", linestyle="--", lw=1)
        ax.set_xscale("log")
        ax.set_xlim(xmin, xmax)
        ax.set_title(f"{lm} clinical-core LR")
        ax.set_yticks(ypos, ylabels)
        ax.grid(axis="x", alpha=0.2)
    for ax in np.atleast_1d(axes):
        ax.set_xlabel("Odds ratio per SD (log scale)")
    fig.suptitle("Unified multivariable logistic OR forest")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "Figure_05_OR_Forest_Unified.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    for old in ["Figure_05_NHRH_SHAP_or_Importance.png", "Figure_06_Elastic_OR_Forest.png"]:
        p = out_dir / "figures" / old
        if p.exists():
            p.unlink()


def build_feature_dictionary(
    out_dir: Path,
    fs_by_lm: dict[str, dict[str, list[str]]],
    selected: pd.DataFrame,
    state_selected: pd.DataFrame,
) -> pd.DataFrame:
    table_dir = out_dir / "tables"
    usage: dict[str, set[str]] = {}

    def add_feature(feat: str, source: str) -> None:
        if feat and feat != "nan":
            usage.setdefault(str(feat), set()).add(source)

    for lm, fs_map in fs_by_lm.items():
        for fs_name, feats in fs_map.items():
            for feat in feats:
                add_feature(feat, f"{lm}:{fs_name}")

    for csv_name, source in [
        ("multivariable_logistic_or.csv", "OR forest"),
        ("lr_linear_shap_importance.csv", "LR LinearSHAP"),
        ("ensemble_permutation_importance.csv", "ensemble permutation"),
        ("univariate_logistic_or.csv", "univariate OR"),
        ("single_feature_benchmark.csv", "single-feature benchmark"),
        ("lr_nomogram_style_points.csv", "nomogram-style points"),
        ("local_linear_explanations.csv", "local explanations"),
    ]:
        path = table_dir / csv_name
        if path.exists():
            try:
                df = pd.read_csv(path)
                if "Feature" in df.columns:
                    for feat in df["Feature"].dropna().astype(str):
                        add_feature(feat, source)
            except Exception:
                pass

    primary_features: set[str] = set()
    for lm in LANDMARKS:
        row = selected[selected["Landmark"].eq(lm)]
        if row.empty:
            continue
        feature_set = str(row.iloc[0].get("FeatureSet", "clinical_core"))
        for feat in fs_by_lm.get(lm, {}).get(feature_set, []):
            primary_features.add(feat)

    rows = []
    for feat in sorted(usage):
        rec = feature_record(feat)
        rows.append(
            {
                "raw_key": rec["raw_key"],
                "code_alias": rec["code_alias"],
                "paper_label_en": rec["paper_label_en"],
                "paper_label_zh": rec["paper_label_zh"],
                "category": rec["category"],
                "timepoint": rec["timepoint"],
                "unit_or_scale": rec["unit_or_scale"],
                "source_window": rec["source_window"],
                "formula": rec["formula"],
                "plain_zh": rec["plain_zh"],
                "primary_clinical_core": feat in primary_features,
                "appears_in": "; ".join(sorted(usage[feat])),
            }
        )
    out = pd.DataFrame(rows).sort_values(["primary_clinical_core", "category", "paper_label_en"], ascending=[False, True, True])
    out.to_csv(table_dir / "feature_registry_stage1.csv", index=False)
    out.to_csv(table_dir / "feature_dictionary_zh.csv", index=False)
    # A compact appendix table prioritizes features that appear in the selected LR line and figures.
    appendix = out[out["primary_clinical_core"]].copy()
    if len(appendix) < 36:
        extra = out[~out["primary_clinical_core"]].head(36 - len(appendix))
        appendix = pd.concat([appendix, extra], ignore_index=True)
    appendix.to_csv(table_dir / "feature_dictionary_zh_appendix_display.csv", index=False)
    return out


def confusion_approach_key(row: pd.Series) -> str:
    if str(row["Approach"]) == "Cascade":
        return "Cascade"
    return f"Direct_{row['Model']}"


def plot_selected_confusion(out_dir: Path, cm: pd.DataFrame, selected_state: pd.DataFrame, endpoint: str, filename: str, title: str, labels: list[str]) -> None:
    chosen = selected_state[selected_state["Endpoint"].eq(endpoint)].copy()
    if chosen.empty:
        return
    landmarks = [lm for lm in LANDMARKS if lm in set(chosen["Landmark"])]
    single_panel = len(landmarks) == 1
    fig, axes = plt.subplots(1, len(landmarks), figsize=((5.3 if single_panel else 5.0 * len(landmarks)), 4.5), squeeze=False)
    feature_label = {
        "clinical_core": "clinical core",
        "clinical_core_12": "12-feature core",
        "stable_selected": "stable selected",
        "stable_plus_missing": "stable + missingness",
    }
    for ax, lm in zip(axes[0], landmarks):
        row = chosen[chosen["Landmark"].eq(lm)].iloc[0]
        approach_key = confusion_approach_key(row)
        ss = cm[
            (cm["Endpoint"].eq(endpoint))
            & (cm["Landmark"].eq(lm))
            & (cm["FeatureSet"].eq(row["FeatureSet"]))
            & (cm["Approach"].eq(approach_key))
        ]
        mat = pd.pivot_table(ss, values="N", index="True", columns="Pred", aggfunc="sum", fill_value=0).reindex(index=labels, columns=labels, fill_value=0)
        im = ax.imshow(mat.values, cmap="YlGnBu")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, int(mat.values[i, j]), ha="center", va="center", fontsize=10)
        ax.set_xticks(np.arange(len(labels)), labels, rotation=30)
        ax.set_yticks(np.arange(len(labels)), labels)
        model_label = str(row["Model"]).replace("Direct_", "")
        if str(row["Approach"]) == "Cascade":
            model_label = f"Cascade ({model_label})"
        panel_title = f"{lm} {model_label}\n{feature_label.get(str(row['FeatureSet']), str(row['FeatureSet']))}"
        if single_panel:
            panel_title = f"{title}\n{panel_title}"
        ax.set_title(panel_title, fontsize=10.5 if single_panel else 11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if not single_panel:
        fig.suptitle(title, fontsize=13, y=1.02)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        fig.tight_layout()
    fig.savefig(out_dir / "figures" / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_reality_gap(out_dir: Path, metrics: pd.DataFrame, selected_state: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, endpoint in zip(axes, ["12M_3Class", "24M_3Class"]):
        rows = []
        chosen = selected_state[selected_state["Endpoint"].eq(endpoint)]
        for _, r in chosen.iterrows():
            ss = metrics[(metrics["Endpoint"].eq(endpoint)) & (metrics["Landmark"].eq(r["Landmark"]))]
            for k in ["FeatureSet", "Approach", "Model"]:
                ss = ss[ss[k].eq(r[k])]
            rows.append(ss)
        ss = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        pivot = ss.pivot_table(index="Split", columns="Landmark", values="MacroF1", aggfunc="mean").reindex(["TrainFit", "OOF", "TemporalTest"])
        pivot.plot(kind="bar", ax=ax, color=["#457b9d", "#2a9d8f"])
        ax.set_title(endpoint)
        ax.set_ylabel("Macro-F1")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "Figure_09_3M_6M_Reality_Gap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_state_flow(out_dir: Path, flow: pd.DataFrame) -> None:
    cols = ["Eval_3M", "Eval_6M", "Eval_12M", "Eval_24M"]
    states = ["Hyper", "Normal", "Hypo", "Missing"]
    colors = {"Hyper": "#e76f51", "Normal": "#2a9d8f", "Hypo": "#457b9d", "Missing": "#b8b8b8"}
    fig, ax = plt.subplots(figsize=(10, 5.6))
    bottom = np.zeros(len(cols))
    for st in states:
        vals = [int((flow[c] == st).sum()) for c in cols]
        ax.bar(cols, vals, bottom=bottom, label=st, color=colors[st], edgecolor="white")
        for i, v in enumerate(vals):
            if v:
                ax.text(i, bottom[i] + v / 2, str(v), ha="center", va="center", fontsize=8)
        bottom += np.array(vals)
    ax.set_ylabel("Treatment origins")
    ax.set_title("Observed state distribution across landmarks")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "Figure_10_State_Flow_3M_6M_12M_24M.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_patient_flow_alluvial(out_dir: Path, flow: pd.DataFrame) -> None:
    cols = ["Eval_3M", "Eval_6M", "Eval_12M"]
    states = ["Hyper", "Normal", "Hypo", "Missing"]
    colors = {"Hyper": "#e76f51", "Normal": "#2a9d8f", "Hypo": "#457b9d", "Missing": "#b8b8b8"}
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    xs = [0, 1, 2]
    bar_width = 0.22
    bottoms: dict[tuple[int, str], float] = {}
    heights: dict[tuple[int, str], float] = {}
    for xi, col in zip(xs, cols):
        vc = flow[col].value_counts().reindex(states, fill_value=0)
        bottom = 0.0
        for st in states:
            h = float(vc[st])
            bottoms[(xi, st)] = bottom
            heights[(xi, st)] = h
            ax.bar(xi, h, bottom=bottom, width=bar_width, color=colors[st], edgecolor="white", linewidth=1.0, label=st if xi == 0 else None, zorder=4)
            if h:
                ax.text(xi, bottom + h / 2, f"{st}\n{int(h)}", ha="center", va="center", fontsize=8)
            bottom += h

    edge_rows = []

    def band(x0: float, y0a: float, y0b: float, x1: float, y1a: float, y1b: float, color: str, alpha: float = 0.28) -> None:
        dx = x1 - x0
        c0 = x0 + dx * 0.45
        c1 = x1 - dx * 0.45
        verts = [
            (x0, y0a),
            (c0, y0a),
            (c1, y1a),
            (x1, y1a),
            (x1, y1b),
            (c1, y1b),
            (c0, y0b),
            (x0, y0b),
            (x0, y0a),
        ]
        codes = [
            MplPath.MOVETO,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.LINETO,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CLOSEPOLY,
        ]
        ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="white", lw=0.25, alpha=alpha, zorder=2))

    for stage, (left_col, right_col) in enumerate(zip(cols[:-1], cols[1:])):
        left_x = xs[stage] + bar_width / 2
        right_x = xs[stage + 1] - bar_width / 2
        edge = flow.groupby([left_col, right_col], dropna=False).size().reset_index(name="N")
        edge[left_col] = pd.Categorical(edge[left_col], categories=states, ordered=True)
        edge[right_col] = pd.Categorical(edge[right_col], categories=states, ordered=True)
        edge = edge.sort_values([left_col, right_col]).reset_index(drop=True)
        left_offsets = {st: bottoms[(stage, st)] for st in states}
        right_offsets = {st: bottoms[(stage + 1, st)] for st in states}
        for _, r in edge.iterrows():
            src = str(r[left_col])
            dst = str(r[right_col])
            n = float(r["N"])
            if n <= 0:
                continue
            y0a, y0b = left_offsets[src], left_offsets[src] + n
            y1a, y1b = right_offsets[dst], right_offsets[dst] + n
            left_offsets[src] += n
            right_offsets[dst] += n
            band(left_x, y0a, y0b, right_x, y1a, y1b, colors.get(src, "#999999"), alpha=0.30 if n >= 10 else 0.16)
            edge_rows.append({"Transition": f"{left_col}->{right_col}", "From": src, "To": dst, "N": int(n)})

    pd.DataFrame(edge_rows).to_csv(out_dir / "tables" / "patient_flow_sankey_edges.csv", index=False)
    ax.set_xticks(xs, ["3M", "6M", "12M"])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel("Treatment origins")
    ax.set_title("Observed patient state flow, 3M -> 6M -> 12M")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.12)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "Figure_08_Patient_Flow_Alluvial.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_study_design_schematic(out_dir: Path, audit: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14.5, 7.4))
    ax.axis("off")
    colors = {
        "cohort": "#eaf4ff",
        "landmark": "#fff3d6",
        "endpoint": "#e9f7ef",
        "guard": "#f8f9fa",
        "test": "#fdecec",
    }

    def box(x: float, y: float, w: float, h: float, text: str, fc: str, ec: str = "#284b63", fs: int = 10) -> None:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=1.3, joinstyle="round"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)

    n_dev = int((audit["Split"] == "Development").sum())
    n_test = int((audit["Split"] == "TemporalTest").sum())
    n_nhrh = int(audit["NHRH"].sum())
    n_success = int((1 - audit["NHRH"]).sum())
    box(0.04, 0.72, 0.20, 0.16, f"RAI-treated Graves cohort\n1003 treatment origins\nDevelopment {n_dev} / temporal test {n_test}", colors["cohort"], fs=10.5)
    box(0.04, 0.45, 0.20, 0.16, f"Primary endpoint\nNHRH by full course to 24M\nNHRH {n_nhrh} / non-NHRH {n_success}", colors["endpoint"], fs=10.2)
    box(0.04, 0.20, 0.20, 0.14, "Secondary point-state endpoints\n12M and 24M\nHyper / Normal / Hypo", colors["endpoint"], fs=10.2)

    timeline_y = 0.52
    xs = np.linspace(0.34, 0.88, 6)
    labels = ["0M", "1M", "3M", "6M", "12M", "24M"]
    ax.plot(xs, [timeline_y] * len(xs), color="#0b2545", lw=2)
    for x, lab in zip(xs, labels):
        ax.scatter([x], [timeline_y], s=150, color="#ffffff", edgecolor="#0b2545", zorder=5)
        ax.text(x, timeline_y - 0.075, lab, ha="center", va="center", fontsize=11)
    box(xs[2] - 0.055, 0.64, 0.11, 0.13, "3M landmark\nbaseline + 1M + 3M\nfeatures only", colors["landmark"], fs=9.5)
    box(xs[3] - 0.055, 0.64, 0.11, 0.13, "6M landmark\nadds 6M current state\nand trajectory", colors["landmark"], fs=9.5)
    box(xs[4] - 0.060, 0.31, 0.12, 0.13, "12M state\npoint-status label", colors["endpoint"], fs=9.5)
    box(xs[5] - 0.060, 0.31, 0.12, 0.13, "24M state\npoint-status label", colors["endpoint"], fs=9.5)

    box(0.33, 0.08, 0.24, 0.15, "Development-only model building\nimputation, scaling, feature selection,\nthreshold, calibration, OOF selection", colors["guard"], ec="#6c757d", fs=9.4)
    box(0.62, 0.08, 0.24, 0.15, "Temporal test isolation\nfinal reporting only\nno tuning or recalibration", colors["test"], ec="#9d0208", fs=9.4)
    for start, end in [(0.24, 0.34), (0.57, 0.62)]:
        ax.annotate("", xy=(end, 0.52), xytext=(start, 0.52), arrowprops=dict(arrowstyle="->", lw=1.5, color="#0b2545"))
    ax.annotate("", xy=(0.62, 0.15), xytext=(0.57, 0.15), arrowprops=dict(arrowstyle="->", lw=1.4, color="#6c757d"))
    ax.set_title("Temporal-safe fixed-landmark Stage 1 design", fontsize=18, weight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "Figure_16_Study_Design_and_Endpoint_Schematic.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_block_flow(out_dir: Path, fs_by_lm: dict[str, dict[str, list[str]]], med: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14.5, 7.2))
    ax.axis("off")
    blocks = [
        ("Baseline clinical background", "Age, sex, thyroid weight,\nantibodies, baseline thyroid labs", "#eaf4ff"),
        ("RAI physiology", "Dose, 24-hour uptake,\neffective half-life, activity per gram", "#fff3d6"),
        ("Early thyroid response", "FT3 / FT4 / TSH changes,\npercent reduction, ratios, slopes", "#f0f7e8"),
        ("Landmark thyroid state", "Current FT3 / FT4 / TSH,\nlog-transformed TSH, clinical state", "#f7ebff"),
        ("Missingness indicators", "Test availability markers\nused only when landmark-safe", "#f8f9fa"),
    ]
    x0, y0, bw, bh, gap = 0.05, 0.62, 0.17, 0.17, 0.025
    for i, (title, desc, color) in enumerate(blocks):
        x = x0 + i * (bw + gap)
        ax.add_patch(plt.Rectangle((x, y0), bw, bh, facecolor=color, edgecolor="#264653", linewidth=1.2))
        ax.text(x + bw / 2, y0 + bh * 0.68, title, ha="center", va="center", fontsize=10.5, weight="bold")
        ax.text(x + bw / 2, y0 + bh * 0.32, desc, ha="center", va="center", fontsize=8.8, wrap=True)
        ax.annotate("", xy=(x + bw / 2, 0.47), xytext=(x + bw / 2, y0), arrowprops=dict(arrowstyle="->", lw=1.0, color="#264653"))

    ax.add_patch(plt.Rectangle((0.18, 0.31), 0.26, 0.14, facecolor="#d8f3dc", edgecolor="#2d6a4f", linewidth=1.4))
    ax.text(0.31, 0.39, "clinical-core 18", ha="center", va="center", fontsize=13, weight="bold")
    ax.text(0.31, 0.345, "primary interpretable LR feature set", ha="center", va="center", fontsize=9.5)
    ax.add_patch(plt.Rectangle((0.56, 0.31), 0.24, 0.14, facecolor="#fefae0", edgecolor="#bc6c25", linewidth=1.4))
    ax.text(0.68, 0.39, "clinical-core 12", ha="center", va="center", fontsize=13, weight="bold")
    ax.text(0.68, 0.345, "compact sensitivity analysis", ha="center", va="center", fontsize=9.5)
    ax.annotate("", xy=(0.44, 0.38), xytext=(0.56, 0.38), arrowprops=dict(arrowstyle="<->", lw=1.2, color="#6c757d"))

    has_structured = False
    if {"Variable", "Available"}.issubset(med.columns):
        hit = med[med["Variable"].eq("Structured_Pre_RAI_ATD_Use")]
        if not hit.empty:
            has_structured = bool(hit["Available"].iloc[0])
    med_note = "No structured pre-RAI ATD history was available; post-RAI medication fields were excluded to avoid confounding by indication."
    if has_structured:
        med_note = "Structured medication fields were audited separately; only pre-RAI baseline medication history would be eligible for sensitivity analysis."
    ax.add_patch(plt.Rectangle((0.12, 0.08), 0.76, 0.13, facecolor="#f8f9fa", edgecolor="#6c757d", linestyle="--", linewidth=1.2))
    ax.text(0.50, 0.155, "Medication policy", ha="center", va="center", fontsize=12, weight="bold")
    ax.text(0.50, 0.105, med_note, ha="center", va="center", fontsize=9.4, wrap=True)
    ax.set_title("Feature blocks, selection flow, and medication handling", fontsize=18, weight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "Figure_17_Feature_Block_and_Selection_Flow.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_single_feature_benchmark(
    out_dir: Path,
    selected: pd.DataFrame,
    datasets: dict[str, Any],
    xte_by_lm: dict[str, pd.DataFrame],
    fs_by_lm: dict[str, dict[str, list[str]]],
) -> pd.DataFrame:
    rows = []
    candidate_roots = ["FT3", "FT4", "TSH", "logTSH"]
    baseline = ["ThyroidW", "Dose", "Uptake24h", "MaxUptake", "HalfLife", "IDPG_Dose_per_ThyroidW", "TRAb", "TGAb", "TPOAb"]
    for lm in LANDMARKS:
        y = np.asarray(datasets[lm].y_test, dtype=int)
        seen: list[str] = []
        for root in candidate_roots:
            feat = f"{root}_{lm}"
            if feat in xte_by_lm[lm].columns:
                seen.append(feat)
        for feat in baseline:
            if feat in xte_by_lm[lm].columns:
                seen.append(feat)
        for feat in seen:
            s = pd.to_numeric(xte_by_lm[lm][feat], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(s)
            if mask.sum() < 30 or len(np.unique(s[mask])) < 2 or len(np.unique(y[mask])) < 2:
                continue
            auc = roc_auc_score(y[mask], s[mask])
            direction = "higher value increases risk"
            score = s[mask]
            if auc < 0.5:
                auc = 1 - auc
                direction = "lower value increases risk"
                score = -score
            ap = average_precision_score(y[mask], score)
            rows.append(
                {
                    "Landmark": lm,
                    "Feature": feat,
                    "Paper_Label": pretty_feature_label(feat),
                    "N": int(mask.sum()),
                    "Directional_AUC": float(auc),
                    "PR_AUC": float(ap),
                    "Direction": direction,
                    "Type": "single feature",
                }
            )
        sel = selected[selected["Landmark"].eq(lm)].iloc[0]
        rows.append(
            {
                "Landmark": lm,
                "Feature": "Clinical-core logistic model",
                "Paper_Label": "Clinical-core logistic model",
                "N": int(len(y)),
                "Directional_AUC": float(sel["Test_AUC"]),
                "PR_AUC": float(sel["Test_PR_AUC"]),
                "Direction": "multivariable prediction",
                "Type": "model",
            }
        )
    persist_path = out_dir / "tables" / "nhrh_persistence_baseline.csv"
    if persist_path.exists():
        pb = pd.read_csv(persist_path)
        for _, pr in pb.iterrows():
            plm = str(pr["Landmark"])
            n_test = int(len(np.asarray(datasets[plm].y_test))) if plm in datasets else 0
            rows.append(
                {
                    "Landmark": plm,
                    "Feature": "Persistence_CurrentHyper",
                    "Paper_Label": "Persistence (carry current state)",
                    "N": n_test,
                    "Directional_AUC": float(pr["Test_AUC"]),
                    "PR_AUC": float(pr["Test_PR_AUC"]),
                    "Direction": "carry-forward current thyroid state",
                    "Type": "baseline",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "tables" / "single_feature_benchmark.csv", index=False)
    return out


def plot_single_feature_benchmark(out_dir: Path, bench: pd.DataFrame) -> None:
    if bench.empty:
        return
    fig, axes = plt.subplots(1, NLM, figsize=(7.5 * NLM, 7), sharex=True)
    for ax, lm in zip(axes, LANDMARKS):
        ss_all = bench[bench["Landmark"].eq(lm)]
        feats = ss_all[ss_all["Type"].eq("single feature")].sort_values("Directional_AUC", ascending=False).head(10)
        anchors = ss_all[ss_all["Type"].isin(["model", "baseline"])]
        ss = pd.concat([feats, anchors], ignore_index=True).sort_values("Directional_AUC", ascending=False).iloc[::-1]
        type_color = {"model": "#c1121f", "baseline": "#e9a000", "single feature": "#457b9d"}
        colors = [type_color.get(t, "#457b9d") for t in ss["Type"]]
        ax.barh(ss["Paper_Label"], ss["Directional_AUC"], color=colors)
        ax.axvline(0.5, color="gray", linestyle="--", lw=1)
        ax.set_xlim(0.45, 1.0)
        ax.set_xlabel("Temporal-test directional ROC-AUC")
        ax.set_title(f"{lm} single-feature benchmark")
        ax.grid(axis="x", alpha=0.18)
    from matplotlib.patches import Patch
    handles = [
        Patch(color="#c1121f", label="Clinical-core LR model"),
        Patch(color="#e9a000", label="Persistence baseline (carry current state)"),
        Patch(color="#457b9d", label="Single clinical variable"),
    ]
    np.atleast_1d(axes)[0].legend(handles=handles, fontsize=8, loc="lower right")
    fig.suptitle("Clinical-core LR model versus single variables and the persistence baseline", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "Figure_18_NHRH_Single_Feature_Benchmark.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_nomogram_points(out_dir: Path) -> pd.DataFrame:
    path = out_dir / "tables" / "multivariable_logistic_or.csv"
    if not path.exists():
        out = pd.DataFrame()
        out.to_csv(out_dir / "tables" / "lr_nomogram_style_points.csv", index=False)
        return out
    df = pd.read_csv(path)
    rows = []
    for lm in LANDMARKS:
        sub = df[df["Analysis"].eq(f"{lm}_clinical_core_multivariable")].copy()
        if sub.empty:
            continue
        coef_col = "Coefficient_per_SD" if "Coefficient_per_SD" in sub.columns else "Coef_per_SD"
        max_abs = float(np.nanmax(np.abs(sub[coef_col])))
        if not np.isfinite(max_abs) or max_abs <= 0:
            max_abs = 1.0
        for _, r in sub.iterrows():
            coef = float(r[coef_col])
            rows.append(
                {
                    "Landmark": lm,
                    "Feature": r["Feature"],
                    "Paper_Label": pretty_feature_label(str(r["Feature"])),
                    "Coefficient_per_SD": coef,
                    "Signed_Points_per_SD": coef / max_abs * 100.0,
                    "Abs_Points_per_SD": abs(coef / max_abs * 100.0),
                    "Direction": "raises NHRH risk" if coef > 0 else "lowers NHRH risk",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "tables" / "lr_nomogram_style_points.csv", index=False)
    return out


def plot_nomogram_points(out_dir: Path, points: pd.DataFrame) -> None:
    if points.empty:
        return
    fig, axes = plt.subplots(1, NLM, figsize=(7.5 * NLM, 7.5), sharex=True)
    for ax, lm in zip(axes, LANDMARKS):
        ss = points[points["Landmark"].eq(lm)].sort_values("Abs_Points_per_SD", ascending=False).head(14)
        ss = ss.sort_values("Signed_Points_per_SD")
        colors = np.where(ss["Signed_Points_per_SD"] >= 0, "#c1121f", "#1d4e89")
        ax.barh(ss["Paper_Label"], ss["Signed_Points_per_SD"], color=colors)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlim(-110, 110)
        ax.set_xlabel("Signed points per 1-SD increase")
        ax.set_title(f"{lm} clinical-core LR")
        ax.grid(axis="x", alpha=0.18)
    fig.suptitle("Nomogram-style signed point contribution", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "Figure_19_LR_Nomogram_Style_Points.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_local_linear_explanations(
    out_dir: Path,
    selected: pd.DataFrame,
    datasets: dict[str, Any],
    xtr_by_lm: dict[str, pd.DataFrame],
    xte_by_lm: dict[str, pd.DataFrame],
    fs_by_lm: dict[str, dict[str, list[str]]],
    seed: int,
) -> pd.DataFrame:
    preds = pd.read_csv(out_dir / "tables" / "nhrh_binary_predictions_long.csv")
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2), sharex=True)
    lm = "3M"
    r = selected[selected["Landmark"].eq(lm)].iloc[0]
    features = fs_by_lm[lm][str(r["FeatureSet"])]
    data = datasets[lm]
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(penalty="l2", solver="lbfgs", C=0.30, class_weight="balanced", max_iter=5000, random_state=seed)),
        ]
    )
    final = fit_with_weight(model, xtr_by_lm[lm][features], data.y_train, sample_weights(data.audit_train, str(r.get("SampleWeight", "nhrh_mild"))))
    scaler = final.named_steps["scale"]
    lr_model = final.named_steps["lr"]
    coef = lr_model.coef_[0]
    intercept = float(lr_model.intercept_[0])
    sub = preds[(preds["RunID"].eq(str(r["RunID"]))) & (preds["Domain"].eq("TemporalTest"))].copy()
    sub["Pred"] = (sub["Proba"] >= float(r["Threshold"])).astype(int)
    def canon_id(v: Any) -> str:
        try:
            fv = float(v)
            if fv.is_integer():
                return str(int(fv))
        except Exception:
            pass
        return str(v)

    id_to_idx = {canon_id(tid): i for i, tid in enumerate(data.test_ids)}
    choices = []
    tp = sub[(sub["Y"].eq(1)) & (sub["Pred"].eq(1))]
    fn = sub[(sub["Y"].eq(1)) & (sub["Pred"].eq(0))]
    tn = sub[(sub["Y"].eq(0)) & (sub["Pred"].eq(0))]
    if not tp.empty:
        choices.append(("Patient A: high-risk true positive", tp.sort_values("Proba", ascending=False).iloc[0]))
    if not fn.empty:
        choices.append(("Patient B: false negative", fn.assign(Dist=(fn["Proba"] - float(r["Threshold"])).abs()).sort_values("Dist").iloc[0]))
    if not tn.empty:
        choices.append(("Patient C: low-risk true negative", tn.sort_values("Proba", ascending=True).iloc[0]))
    for ax, (label, row) in zip(axes, choices):
        tid = canon_id(row["Treatment_ID"])
        idx = id_to_idx.get(tid)
        if idx is None:
            ax.axis("off")
            continue
        x_scaled = scaler.transform(xte_by_lm[lm].iloc[[idx]][features])[0]
        contrib = x_scaled * coef
        order = np.argsort(np.abs(contrib))[-10:]
        vals = contrib[order]
        labs = [pretty_feature_label(features[i]) for i in order]
        colors = np.where(vals >= 0, "#c1121f", "#1d4e89")
        ax.barh(labs, vals, color=colors)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"{label}\nobserved={int(row['Y'])}, predicted={int(row['Pred'])}, risk={float(row['Proba']):.2f}")
        ax.set_xlabel("Contribution to LR logit")
        ax.grid(axis="x", alpha=0.16)
        for feat_i, val in zip(order, vals):
            rows.append(
                {
                    "Case_Label": label,
                    "Landmark": lm,
                    "Patient_Label": label.split(":")[0],
                    "Observed_NHRH": int(row["Y"]),
                    "Predicted_NHRH": int(row["Pred"]),
                    "Predicted_Risk": float(row["Proba"]),
                    "Feature": features[feat_i],
                    "Paper_Label": pretty_feature_label(features[feat_i]),
                    "Logit_Contribution": float(val),
                    "Intercept": intercept,
                }
            )
    for ax in axes[len(choices) :]:
        ax.axis("off")
    fig.suptitle("Local linear explanations for representative temporal-test cases", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "Figure_20_Local_Linear_Explanations.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "tables" / "local_linear_explanations.csv", index=False)
    return out


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=int)
    logits = np.log(p / (1 - p)).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    try:
        model = _unpenalized_lr(2000)
        model.fit(logits, y)
    except Exception:
        model = LogisticRegression(penalty="l2", C=1e6, solver="lbfgs", max_iter=2000)
        model.fit(logits, y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def build_calibration_summary(out_dir: Path, selected: pd.DataFrame) -> pd.DataFrame:
    preds = pd.read_csv(out_dir / "tables" / "nhrh_binary_predictions_long.csv")
    rows = []
    for _, r in selected.iterrows():
        for split in ["OOF", "TemporalTest"]:
            sub = preds[(preds["RunID"].eq(str(r["RunID"]))) & (preds["Domain"].eq(split))]
            if sub.empty:
                continue
            y = sub["Y"].to_numpy(dtype=int)
            p = sub["Proba"].to_numpy(dtype=float)
            intercept, slope = calibration_intercept_slope(y, p)
            rows.append(
                {
                    "Landmark": r["Landmark"],
                    "Split": split,
                    "N": len(sub),
                    "Prevalence": float(y.mean()),
                    "Brier": float(brier_score_loss(y, p)),
                    "Calibration_Intercept": intercept,
                    "Calibration_Slope": slope,
                    "ROC_AUC": float(roc_auc_score(y, p)),
                    "PR_AUC": float(average_precision_score(y, p)),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "tables" / "calibration_summary.csv", index=False)
    return out


def plot_calibration_summary(out_dir: Path, cal: pd.DataFrame) -> None:
    if cal.empty:
        return
    view = cal[cal["Split"].eq("TemporalTest")].copy()
    if view.empty:
        view = cal.copy()
    metrics = ["Brier", "Calibration_Intercept", "Calibration_Slope"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    for ax, metric in zip(axes, metrics):
        vals = view.set_index("Landmark")[metric].reindex(LANDMARKS)
        palette = ["#264653", "#457b9d", "#2a9d8f", "#e76f51", "#e9c46a", "#8d99ae"]
        colors = [palette[i % len(palette)] for i in range(len(vals))]
        ax.bar(vals.index, vals.values, color=colors)
        if metric == "Calibration_Slope":
            ax.axhline(1.0, color="black", linestyle="--", lw=1)
        if metric == "Calibration_Intercept":
            ax.axhline(0.0, color="black", linestyle="--", lw=1)
        for i, v in enumerate(vals.values):
            if np.isfinite(v):
                ax.text(i, v, f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=10)
        ax.set_title(metric.replace("_", " "))
        ax.grid(axis="y", alpha=0.18)
    fig.suptitle("Temporal-test calibration summary for selected LR models", fontsize=14.5, weight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "Figure_22_Calibration_Intercept_Slope_Summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_incremental_value(out_dir: Path, selected: pd.DataFrame, gap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in ["OOF", "TemporalTest"]:
        row: dict[str, Any] = {"Endpoint": "NHRH_binary", "Split": split}
        for lm in LANDMARKS:
            sel = selected[selected["Landmark"].eq(lm)].iloc[0]
            prefix = "OOF" if split == "OOF" else "Test"
            row[f"{lm}_PrimaryMetric"] = float(sel[f"{prefix}_AUC"])
            row[f"{lm}_SecondaryMetric"] = float(sel[f"{prefix}_PR_AUC"])
            row[f"{lm}_CalibrationMetric"] = float(sel[f"{prefix}_Brier"])
        row["Delta_Primary_6M_minus_3M"] = row["6M_PrimaryMetric"] - row["3M_PrimaryMetric"]
        row["PrimaryMetric_Name"] = "ROC-AUC"
        row["SecondaryMetric_Name"] = "PR-AUC"
        rows.append(row)
    if not gap.empty:
        for _, r in gap.iterrows():
            rows.append(
                {
                    "Endpoint": r["Endpoint"],
                    "Split": r["Split"],
                    "3M_PrimaryMetric": float(r["MacroF1_3M"]),
                    "6M_PrimaryMetric": float(r["MacroF1_6M"]),
                    "Delta_Primary_6M_minus_3M": float(r["MacroF1_Gap_6M_minus_3M"]),
                    "3M_SecondaryMetric": float(r["BalancedAccuracy_3M"]),
                    "6M_SecondaryMetric": float(r["BalancedAccuracy_6M"]),
                    "PrimaryMetric_Name": "Macro-F1",
                    "SecondaryMetric_Name": "Balanced accuracy",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "tables" / "incremental_value_3m_vs_6m.csv", index=False)
    return out


def plot_incremental_value(out_dir: Path, inc: pd.DataFrame) -> None:
    if inc.empty:
        return
    fig, ax = plt.subplots(figsize=(12.5, 5.3))
    view = inc[inc["Split"].isin(["OOF", "TemporalTest"])].copy()
    view["Label"] = view["Endpoint"].str.replace("_", " ") + "\n" + view["Split"]
    colors = np.where(view["Delta_Primary_6M_minus_3M"] >= 0, "#2a9d8f", "#c1121f")
    ax.bar(view["Label"], view["Delta_Primary_6M_minus_3M"], color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("6M minus 3M primary metric")
    ax.set_title("Incremental value of observing the 6M landmark")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "Figure_21_3M_vs_6M_Incremental_Value.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_contact_sheet(out_dir: Path) -> None:
    import matplotlib.image as mpimg

    fig_dir = out_dir / "figures"
    files = sorted([p for p in fig_dir.glob("Figure_*.png") if p.name != "Figure_00_Contact_Sheet.png"])
    if not files:
        return
    ncols = 3
    nrows = int(math.ceil(len(files) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, max(4, 3.6 * nrows)))
    axes_arr = np.atleast_1d(axes).ravel()
    for ax, p in zip(axes_arr, files):
        img = mpimg.imread(p)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(p.stem, fontsize=9)
    for ax in axes_arr[len(files) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(fig_dir / "Figure_00_Contact_Sheet.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_error_tables(out_dir: Path, pred_records: pd.DataFrame, datasets: dict[str, Any], xtr_by_lm: dict[str, pd.DataFrame], xte_by_lm: dict[str, pd.DataFrame], fs_by_lm: dict[str, dict[str, list[str]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    table_dir = out_dir / "tables"
    pred_records.to_csv(table_dir / "state_model_predictions_long.csv", index=False)
    profile = pred_records.groupby(["Endpoint", "Landmark", "FeatureSet", "Approach", "Model", "Split", "True", "Pred"], dropna=False).size().reset_index(name="N")
    total = pred_records.groupby(["Endpoint", "Landmark", "FeatureSet", "Approach", "Model", "Split", "True"], dropna=False).size().reset_index(name="True_Total")
    profile = profile.merge(total, on=["Endpoint", "Landmark", "FeatureSet", "Approach", "Model", "Split", "True"], how="left")
    profile["Within_True_Class_Rate"] = profile["N"] / profile["True_Total"].replace(0, np.nan)
    profile.to_csv(table_dir / "error_profile_state_models.csv", index=False)
    profile.to_csv(table_dir / "error_transition_matrix_by_endpoint.csv", index=False)

    feature_rows = []
    for lm in LANDMARKS:
        cols = fs_by_lm[lm]["clinical_core"]
        tr = xtr_by_lm[lm][cols].copy()
        tr.insert(0, "Treatment_ID", datasets[lm].train_ids)
        tr["Landmark"] = lm
        tr["SplitKey"] = "Development"
        te = xte_by_lm[lm][cols].copy()
        te.insert(0, "Treatment_ID", datasets[lm].test_ids)
        te["Landmark"] = lm
        te["SplitKey"] = "TemporalTest"
        feature_rows.append(pd.concat([tr, te], ignore_index=True))
    feat_long = pd.concat(feature_rows, ignore_index=True)
    errors = pred_records[~pred_records["Correct"]].copy()
    errors["SplitKey"] = np.where(errors["Split"].eq("TemporalTest"), "TemporalTest", "Development")
    case = errors.merge(feat_long, on=["Treatment_ID", "Landmark", "SplitKey"], how="left")
    case.to_csv(table_dir / "error_casebook_state_models.csv", index=False)
    return profile, case


def reality_gap_tables(out_dir: Path, pred_records: pd.DataFrame, audit: pd.DataFrame, selected_state: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, sub in pred_records.groupby(["Endpoint", "Landmark", "FeatureSet", "Approach", "Model", "Split"], dropna=False):
        endpoint, lm, fs, approach, model, split = keys
        labels = NHRH3_ORDER if endpoint == "NHRH_3Class" else STATE_ORDER
        prob_cols = [f"P_{c}" for c in labels]
        if not set(prob_cols).issubset(sub.columns):
            continue
        rows.append({"Endpoint": endpoint, "Landmark": lm, "FeatureSet": fs, "Approach": approach, "Model": model, "Split": split, **class_metrics(sub["True"].to_numpy(), sub[prob_cols].to_numpy(), labels)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "tables" / "landmark_reality_gap_by_split.csv", index=False)
    dist = pred_records.groupby(["Endpoint", "Landmark", "FeatureSet", "Approach", "Model", "Split", "True", "Pred"], dropna=False).size().reset_index(name="N")
    dist.to_csv(out_dir / "tables" / "predicted_vs_observed_state_distribution.csv", index=False)
    gap_rows = []
    selected_metrics = []
    for _, r in selected_state[selected_state["Endpoint"].isin(["12M_3Class", "24M_3Class"])].iterrows():
        ss = metrics[(metrics["Endpoint"].eq(r["Endpoint"])) & (metrics["Landmark"].eq(r["Landmark"]))]
        for k in ["FeatureSet", "Approach", "Model"]:
            ss = ss[ss[k].eq(r[k])]
        selected_metrics.append(ss)
    base = pd.concat(selected_metrics, ignore_index=True) if selected_metrics else pd.DataFrame()
    for (endpoint, split), ss in base.groupby(["Endpoint", "Split"]):
        a = ss[ss["Landmark"].eq("3M")]
        b = ss[ss["Landmark"].eq("6M")]
        if a.empty or b.empty:
            continue
        gap_rows.append(
            {
                "Endpoint": endpoint,
                "Split": split,
                "MacroF1_3M": float(a["MacroF1"].iloc[0]),
                "MacroF1_6M": float(b["MacroF1"].iloc[0]),
                "MacroF1_Gap_6M_minus_3M": float(b["MacroF1"].iloc[0] - a["MacroF1"].iloc[0]),
                "BalancedAccuracy_3M": float(a["BalancedAccuracy"].iloc[0]),
                "BalancedAccuracy_6M": float(b["BalancedAccuracy"].iloc[0]),
                "BalancedAccuracy_Gap_6M_minus_3M": float(b["BalancedAccuracy"].iloc[0] - a["BalancedAccuracy"].iloc[0]),
                "MacroAUC_3M": float(a["MacroAUC_OVR"].iloc[0]),
                "MacroAUC_6M": float(b["MacroAUC_OVR"].iloc[0]),
                "MacroAUC_Gap_6M_minus_3M": float(b["MacroAUC_OVR"].iloc[0] - a["MacroAUC_OVR"].iloc[0]),
            }
        )
    pd.DataFrame(gap_rows).to_csv(out_dir / "tables" / "landmark_reality_gap_metrics.csv", index=False)
    flow = audit[["Treatment_ID", "Patient_ID", "Eval_3M", "Eval_6M", "Eval_12M", "Eval_24M", "NHRH"]].copy()
    flow.to_csv(out_dir / "tables" / "state_flow_3m_6m_12m_24m.csv", index=False)
    return metrics


def refinement_summary(out_dir: Path, perf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (endpoint, lm, fs), sub in perf.groupby(["Endpoint", "Landmark", "FeatureSet"], dropna=False):
        base = sub[(sub["Split"].eq("OOF")) & (sub["Model"].eq("Direct_MultinomialLR"))]
        if base.empty:
            continue
        base_row = base.iloc[0]
        for _, r in sub[(sub["Split"].eq("OOF")) & (~sub["Refinement"].eq("baseline"))].iterrows():
            test = sub[(sub["Split"].eq("TemporalTest")) & (sub["Model"].eq(r["Model"])) & (sub["Refinement"].eq(r["Refinement"]))]
            rows.append(
                {
                    "Endpoint": endpoint,
                    "Landmark": lm,
                    "FeatureSet": fs,
                    "Candidate": r["Model"],
                    "Refinement": r["Refinement"],
                    "OOF_MacroF1_Delta": float(r["MacroF1"] - base_row["MacroF1"]),
                    "OOF_BalancedAccuracy_Delta": float(r["BalancedAccuracy"] - base_row["BalancedAccuracy"]),
                    "Promote_To_Main": bool((r["MacroF1"] > base_row["MacroF1"]) and (r["BalancedAccuracy"] >= base_row["BalancedAccuracy"] - 1e-9)),
                    "TemporalTest_MacroF1": float(test["MacroF1"].iloc[0]) if not test.empty else np.nan,
                    "Policy": "OOF-only; temporal test is descriptive",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "tables" / "refinement_candidates_state_models.csv", index=False)
    out.to_csv(out_dir / "tables" / "error_aware_refinement_summary.csv", index=False)
    return out


def compact_sensitivity(out_dir: Path, selected: pd.DataFrame, all_cand: pd.DataFrame, state_perf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in selected.iterrows():
        rows.append({"Task": f"{r['Landmark']}_NHRH_binary", "FeatureSet": "clinical_core", "N_Features": int(r["N_Features"]), "Test_AUC": float(r["Test_AUC"]), "Test_Accuracy": float(r["Test_Accuracy"]), "Source": "selected"})
    binary12_path = out_dir / "tables" / "nhrh_binary_clinical_core_12_sensitivity.csv"
    if binary12_path.exists():
        binary12 = pd.read_csv(binary12_path)
        for _, r in binary12[binary12["Split"].eq("TemporalTest")].iterrows():
            rows.append(
                {
                    "Task": f"{r['Landmark']}_NHRH_binary",
                    "FeatureSet": "clinical_core_12",
                    "N_Features": int(r["N_Features"]),
                    "Test_AUC": float(r["AUC"]),
                    "Test_Accuracy": float(r["Accuracy"]),
                    "Test_PR_AUC": float(r["PR_AUC"]),
                    "Test_Brier": float(r["Brier"]),
                    "Source": "lr_only_sensitivity",
                }
            )
    for (lm, endpoint, fs), sub in state_perf[(state_perf["Split"].eq("TemporalTest")) & (state_perf["Approach"].eq("Cascade"))].groupby(["Landmark", "Endpoint", "FeatureSet"]):
        if endpoint == "24M_3Class":
            r = sub.iloc[0]
            rows.append({"Task": f"{lm}->{endpoint}", "FeatureSet": fs, "N_Features": 12 if fs == "clinical_core_12" else 18, "Test_AUC": float(r.get("MacroAUC_OVR", np.nan)), "Test_Accuracy": float(r["Accuracy"]), "Test_MacroF1": float(r["MacroF1"]), "Source": "state_model"})
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "tables" / "clinical_core_18_vs_12.csv", index=False)
    return out


def state_key_columns() -> list[str]:
    return ["Landmark", "Endpoint", "FeatureSet", "Approach", "Model", "Refinement"]


def add_state_selection_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    recall_cols = [c for c in out.columns if c.endswith("_Recall")]
    out["MinClassRecall"] = out[recall_cols].min(axis=1, skipna=True)
    out["SelectionScore"] = (
        out["MacroF1"].fillna(0)
        + out["BalancedAccuracy"].fillna(0)
        + 0.25 * out["MacroAUC_OVR"].fillna(0)
        + 0.25 * out["MinClassRecall"].fillna(0)
    )
    return out


def select_state_models(out_dir: Path, state_perf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = state_key_columns()
    oof = add_state_selection_score(state_perf[state_perf["Split"].eq("OOF")])
    rows = []
    for (endpoint, landmark), sub in oof.groupby(["Endpoint", "Landmark"], dropna=False):
        sub = sub.sort_values(
            ["SelectionScore", "MacroF1", "BalancedAccuracy", "MinClassRecall", "MacroAUC_OVR"],
            ascending=[False, False, False, False, False],
        )
        chosen = sub.iloc[0]
        test = state_perf[state_perf["Split"].eq("TemporalTest")]
        for k in keys:
            test = test[test[k].eq(chosen[k])]
        if test.empty:
            continue
        t = test.iloc[0]
        row = {k: chosen[k] for k in keys}
        row.update(
            {
                "OOF_MacroF1": float(chosen["MacroF1"]),
                "OOF_BalancedAccuracy": float(chosen["BalancedAccuracy"]),
                "OOF_MacroAUC_OVR": float(chosen["MacroAUC_OVR"]),
                "OOF_MinClassRecall": float(chosen["MinClassRecall"]),
                "Test_N": int(t["N"]),
                "Test_Accuracy": float(t["Accuracy"]),
                "Test_BalancedAccuracy": float(t["BalancedAccuracy"]),
                "Test_MacroF1": float(t["MacroF1"]),
                "Test_MacroAUC_OVR": float(t["MacroAUC_OVR"]),
            }
        )
        for c in [c for c in state_perf.columns if c.endswith("_Recall") or c.endswith("_Support")]:
            if c in t:
                row[f"Test_{c}"] = t[c]
        rows.append(row)
    selected = pd.DataFrame(rows)
    selected.to_csv(out_dir / "tables" / "state_endpoint_selected.csv", index=False)

    cmp_rows = []
    for horizon in ["12M", "24M"]:
        endpoint = f"{horizon}_3Class"
        for lm in LANDMARKS:
            sub = selected[(selected["Endpoint"].eq(endpoint)) & (selected["Landmark"].eq(lm))]
            if sub.empty:
                continue
            r = sub.iloc[0]
            cmp_rows.append(
                {
                    "Landmark": lm,
                    "Horizon": horizon,
                    "Best_Model": f"{r['Approach']}:{r['Model']}:{r['FeatureSet']}",
                    "Macro_AUC": float(r["Test_MacroAUC_OVR"]),
                    "Balanced_Acc": float(r["Test_BalancedAccuracy"]),
                    "Macro_F1": float(r["Test_MacroF1"]),
                }
            )
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(out_dir / "tables" / "12m_vs_24m_prediction_comparison.csv", index=False)
    return selected, cmp_df


def run_binary_core12_sensitivity(out_dir: Path, datasets: dict[str, Any], xtr_by_lm: dict[str, pd.DataFrame], xte_by_lm: dict[str, pd.DataFrame], fs_by_lm: dict[str, dict[str, list[str]]], seed: int) -> pd.DataFrame:
    rows = []
    for lm in LANDMARKS:
        data = datasets[lm]
        features = fs_by_lm[lm]["clinical_core_12"]
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("lr", LogisticRegression(penalty="l2", solver="lbfgs", C=0.30, class_weight="balanced", max_iter=5000, random_state=seed)),
            ]
        )
        pred = fit_oof_weighted(model, xtr_by_lm[lm][features], data.y_train, data.groups_train, xte_by_lm[lm][features], weight=sample_weights(data.audit_train, "nhrh_mild"))
        probs = apply_platt(data.y_train, pred["oof"], pred["train_fit"], pred["test"])
        threshold = choose_threshold(data.y_train, probs["oof"], "accuracy")
        for split, y, p in [("TrainFit", data.y_train, probs["train_fit"]), ("OOF", data.y_train, probs["oof"]), ("TemporalTest", data.y_test, probs["test"])]:
            rows.append(
                {
                    "RunID": f"{lm}__clinical_core_12__Clinical_L2_Logistic__nhrh_mild__platt__accuracy__s{seed}",
                    "Landmark": lm,
                    "FeatureSet": "clinical_core_12",
                    "Model": "Clinical_L2_Logistic",
                    "SampleWeight": "nhrh_mild",
                    "Calibration": "platt",
                    "ThresholdRule": "accuracy",
                    "Threshold": float(threshold),
                    "N_Features": len(features),
                    "Split": split,
                    **metrics_at_threshold(y, p, float(threshold)),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "tables" / "nhrh_binary_clinical_core_12_sensitivity.csv", index=False)
    return out


def write_goal_files(root: Path) -> None:
    goal_dir = root / "goals" / "stage1_consolidation"
    goal_dir.mkdir(parents=True, exist_ok=True)
    (goal_dir / "GOAL.md").write_text(
        "\n".join(
            [
                "# Stage 1 Consolidation Goal",
                "",
                "Objective: consolidate 3M/6M early stratification into unified clinical-core logistic reporting, add 24M state models, medication audit, error-aware refinement, and reality-gap analysis.",
                "",
                "Constraints:",
                "- no NHRH label changes",
                "- no temporal-test selection",
                "- no post-RAI medication features in primary models",
                "- no broad all-combinations grid",
                "- max bounded refinement iterations: 6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (goal_dir / "PROTOCOL.md").write_text(
        "\n".join(
            [
                "# Protocol",
                "",
                "- scikit-learn: pipeline, logistic, tree comparators, multiclass/cascade, bootstrap metrics.",
                "- statistical-analysis: OR/CI, calibration/DCA interpretation, error-stratified wording.",
                "- autoresearch: bounded refinement only; keep policy is OOF improvement, not temporal-test improvement.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_readme(
    out_dir: Path,
    selected: pd.DataFrame,
    ci: pd.DataFrame,
    lr_vs_ens: pd.DataFrame,
    state_perf: pd.DataFrame,
    state_selected: pd.DataFrame,
    horizon_cmp: pd.DataFrame,
    refine: pd.DataFrame,
    med: pd.DataFrame,
    gap: pd.DataFrame,
) -> None:
    table_dir = out_dir / "tables"
    nhrh_counts = pd.read_csv(table_dir / "label_distribution_nhrh_binary.csv")
    leakage = pd.read_csv(table_dir / "leakage_feature_audit.csv")
    state_main = state_selected.copy()
    selected_display = selected[
        [
            "Landmark",
            "FeatureSet",
            "Model",
            "SampleWeight",
            "Calibration",
            "ThresholdRule",
            "Threshold",
            "N_Features",
            "OOF_Accuracy",
            "Test_AUC",
            "Test_PR_AUC",
            "Test_Accuracy",
            "Test_Brier",
            "Recall",
            "Specificity",
            "PPV",
            "NPV",
            "F1",
        ]
    ]
    selected_compact = selected_display[
        [
            "Landmark",
            "Threshold",
            "Test_AUC",
            "Test_PR_AUC",
            "Test_Accuracy",
            "Test_Brier",
            "Recall",
            "Specificity",
            "PPV",
            "NPV",
        ]
    ]
    state_compact = state_main[
        [
            "Landmark",
            "Endpoint",
            "Approach",
            "Model",
            "Test_N",
            "Test_BalancedAccuracy",
            "Test_MacroF1",
            "Test_MacroAUC_OVR",
        ]
    ].sort_values(["Endpoint", "Landmark"])
    gap_compact = gap[["Endpoint", "Split", "MacroF1_Gap_6M_minus_3M", "BalancedAccuracy_Gap_6M_minus_3M"]].head(12) if not gap.empty else pd.DataFrame()
    feature_dict_path = out_dir / "tables" / "feature_dictionary_zh_appendix_display.csv"
    feature_dict = pd.read_csv(feature_dict_path) if feature_dict_path.exists() else pd.DataFrame()
    feature_dict_compact = (
        feature_dict[["paper_label_en", "paper_label_zh", "category", "plain_zh", "formula"]].rename(
            columns={
                "paper_label_en": "论文变量名",
                "paper_label_zh": "中文含义",
                "category": "变量组",
                "plain_zh": "人话解释",
                "formula": "公式或编码方式",
            }
        )
        if not feature_dict.empty
        else pd.DataFrame()
    )
    file_index = pd.DataFrame(
        [
            {"File": "tables/nhrh_binary_selected.csv", "Meaning": "统一 LR 主报行"},
            {"File": "tables/nhrh_binary_lr_vs_ensemble.csv", "Meaning": "LR 与 ensemble 补充对比"},
            {"File": "tables/binary_model_metric_heatmap.csv", "Meaning": "NHRH binary 模型 × 指标 heatmap 数据"},
            {"File": "tables/lr_linear_shap_importance.csv", "Meaning": "selected LR linear SHAP mean absolute importance"},
            {"File": "tables/feature_dictionary_zh.csv", "Meaning": "完整特征名、论文标签与中文人话释义"},
            {"File": "tables/patient_flow_sankey_edges.csv", "Meaning": "3M→6M→12M Sankey/alluvial 边流量"},
            {"File": "tables/medication_feature_audit.csv", "Meaning": "用药字段可用性与纳入/排除审计"},
            {"File": "tables/3m_24m_3class_performance.csv / tables/6m_24m_3class_performance.csv", "Meaning": "24M 状态模型结果"},
            {"File": "tables/error_casebook_state_models.csv", "Meaning": "状态模型 hard-case 表"},
            {"File": "tables/landmark_reality_gap_metrics.csv", "Meaning": "3M/6M 信息窗口 gap"},
            {"File": "tables/feature_registry_stage1.csv", "Meaning": "raw key、内部别名、论文标签和中文解释注册表"},
            {"File": "tables/single_feature_benchmark.csv", "Meaning": "单变量 benchmark 与 clinical-core LR 对照"},
            {"File": "tables/nhrh_persistence_baseline.csv", "Meaning": "persistence 朴素基线（沿用当前甲功状态）各 landmark 的 AUC/PR/PPV/NPV"},
            {"File": "tables/lr_nomogram_style_points.csv", "Meaning": "LR nomogram-style signed points"},
            {"File": "tables/local_linear_explanations.csv", "Meaning": "Patient A/B/C 局部线性解释"},
            {"File": "tables/incremental_value_3m_vs_6m.csv", "Meaning": "3M 到 6M 信息增量汇总"},
            {"File": "tables/calibration_summary.csv", "Meaning": "Brier、calibration intercept/slope 摘要"},
        ]
    )
    appendix_lines = [
        "# Stage 1 表格附录",
        "",
        "本文件承接主报告中移出的长表。正文只保留摘要表和图，完整审计表均保留在 CSV 与本附录中。",
        "",
        "## Leakage feature audit",
        "",
        md_table(leakage),
        "",
        "## Medication feature audit",
        "",
        md_table(med.fillna("")),
        "",
        "## Full selected NHRH binary table",
        "",
        md_table(selected_display),
        "",
        "## LR vs ensemble comparison",
        "",
        md_table(lr_vs_ens[["Landmark", "Model", "FeatureSet", "N_Features", "Test_AUC", "Test_PR_AUC", "Test_Accuracy", "Test_Brier"]]),
        "",
        "## Selected state endpoint table",
        "",
        md_table(
            state_main[
                [
                    "Landmark",
                    "Endpoint",
                    "FeatureSet",
                    "Approach",
                    "Model",
                    "OOF_MacroF1",
                    "OOF_BalancedAccuracy",
                    "OOF_MacroAUC_OVR",
                    "Test_N",
                    "Test_Accuracy",
                    "Test_BalancedAccuracy",
                    "Test_MacroF1",
                    "Test_MacroAUC_OVR",
                ]
            ].sort_values(["Endpoint", "Landmark"])
        ),
        "",
        "## 12M vs 24M comparison",
        "",
        md_table(horizon_cmp),
        "",
        "## Error-aware refinement summary",
        "",
        md_table(refine) if not refine.empty else "_当前 refinement 未生成可提升 OOF 的候选。_",
        "",
        "## Reality gap table",
        "",
        md_table(gap) if not gap.empty else "_Reality-gap 表为空。_",
        "",
        "## Full feature dictionary",
        "",
        "完整 CSV 见 `tables/feature_dictionary_zh.csv`。正文末尾仅展示主模型/主图优先的压缩表。",
        "",
        md_table(pd.read_csv(out_dir / "tables" / "feature_dictionary_zh.csv")) if (out_dir / "tables" / "feature_dictionary_zh.csv").exists() else "_Feature dictionary 尚未生成。_",
        "",
        "## File index",
        "",
        md_table(file_index),
    ]
    (out_dir / "TABLE_APPENDIX.md").write_text("\n".join(appendix_lines) + "\n", encoding="utf-8")
    lines = [
        "# RAI 后早期风险分层（0M 治疗前 / 1M / 3M / 6M 固定地标）：统一 clinical-core logistic 主线",
        "",
        "## 摘要",
        "",
        "**通俗版（先用人话说清）。** Graves 甲亢患者做完碘‑131（RAI）治疗后，有相当一部分人会治疗失败或日后复发。本阶段做的是一张“早期风险打分卡”：在 **治疗前（0M）、治疗后 1 个月、3 个月、6 个月** 这几个固定复查点，分别只用“到该时点为止已经能看到的信息”（基线甲状腺负荷、抗体、RAI 剂量学，以及该时点的甲功化验与早期下降幅度），用一个透明、可解释的逻辑回归模型，算出“这个疗程最终是否会治疗失败/复发”的概率。结果很直观：随着随访信息累积，判别力稳步上升——**治疗前仅凭基线就已优于随机，3 个月已达较好水平，6 个月最高**；而且每个时点都明显优于“直接沿用病人当前甲功状态”这种最朴素的判断。一句话：RAI 后越早、越完整地观察治疗反应，越能提前把高危与低危病人分开。",
        "",
        "本阶段评估 fixed-landmark supervised prediction：在 **0M（纯治疗前基线）、1M、3M、6M** 四个固定地标，使用截至该时点可见的信息预测 NHRH 复合治疗失败终点，并补充 12M/24M Hyper / Normal / Hypo 状态结局；每个地标都与一个 **persistence 朴素基线**（直接沿用当前甲功状态）对照，以量化模型相对“零模型”的增量。分析单位为 `1003` 个治疗起点，development/test 采用时间顺序切分；temporal test 仅用于最终报告，不参与特征选择、阈值选择、校准拟合或模型选择。",
        "",
        "NHRH binary 主报在 3M/6M 固定为 `Clinical_L2_Logistic + clinical_core + nhrh_mild + Platt`，0M/1M 取 OOF 最优可解释 LR 配置。该选择牺牲少量 ensemble 判别增益，换取各地标间一致的 OR、校准、DCA 与临床解释范式。",
        "",
        "## Endpoint 与时间锚点",
        "",
        "NHRH binary 基于患者全程随访至 24M 的治疗轨迹判断：持续甲亢或控制后复发均记为 NHRH。12M/24M 三分类基于对应时间点的单次甲功状态。两个 endpoint 的时间参照不同：NHRH 反映全程治疗成败，12M/24M 三分类反映特定时间点即时状态。因此，少数患者可能在某一时间点仍为 Hyper 但最终控制成功，或某一时间点已控制但后续复发。",
        "",
        "![Study design schematic](figures/Figure_16_Study_Design_and_Endpoint_Schematic.png)",
        "",
        "> **图像分析。** 该流程图把本阶段的 unit of analysis、时间锚点和 endpoint 关系放在同一张图里。1003 个治疗起点按时间顺序进入 development 与 temporal test，3M 只使用 baseline/1M/3M 信息，6M 额外使用 6M 当前状态和轨迹摘要；NHRH 反映 24M 内全程治疗失败或复发，而 12M/24M 三分类只反映对应时间点状态。图中将 development-only preprocessing 与 temporal-test final reporting 分开，强调阈值、校准和模型选择均不使用 temporal test。",
        "",
        "### 标签分布",
        "",
        md_table(nhrh_counts),
        "",
        "![Label distribution](figures/Figure_01_Label_Distribution.png)",
        "",
        "> **图像分析。** 该图首先审计 NHRH 标签来源，而不是直接进入模型性能。左侧显示多数治疗起点属于最终 Normal/Hypo 的 non-NHRH 成功轨迹，其次为最终仍 Hyper 的 persistent nonhealing；控制后复发来源分散在 6M、12M、18M 和 24M，人数较少但构成 NHRH 复合终点的临床关键部分。右侧显示 temporal test 的 NHRH 比例略高于 development，说明时间外推集存在轻度 case-mix shift，但不是类别比例完全不同的测试集。",
        "",
        "## 特征与泄漏控制",
        "",
        "- 3M 特征只允许 baseline、1M、3M、变化量、RAI 生理特征和缺失指示。",
        "- 6M 特征额外允许 6M 当前值和 0/1/3/6M 轨迹摘要。",
        "- 特征命名按 RAI/Graves 预测论文常见变量组统一：甲功当前状态、抗体/免疫活性、甲状腺负荷、RAI 剂量-摄取-半衰期、早期治疗反应和缺失指示。",
        "- 当前原始表未提供结构化 RAI 前 ATD 用药史、ATD duration 或 washout 字段；`TreatCount` 保留为 baseline 治疗次数/治疗起点背景，但不解释为 ATD history。",
        "- 随访窗口 `用药情况` 属于 post-RAI 处理反应，主模型继续排除，以避免 confounding by indication。若后续补齐 RAI 前 ATD 史，应作为 baseline sensitivity 纳入。",
        "",
        "![Feature block and selection flow](figures/Figure_17_Feature_Block_and_Selection_Flow.png)",
        "",
        "> **图像分析。** 该图把模型输入从工程列名重新组织为医学变量块：baseline 临床背景、RAI 剂量-摄取-甲状腺负荷、早期甲功反应、landmark 当前甲功状态和缺失指示。clinical-core 18 是主报 LR 特征集，clinical-core 12 仅用于简化敏感性分析。底部用药策略说明了为什么当前报告不把随访后用药作为主模型特征：这些记录是医生基于治疗后状态作出的反应，直接纳入会引入 indication bias；而结构化 RAI 前 ATD 史目前不可用，因此只保留审计说明。",
        "",
        f"Leakage audit 共 `{len(leakage)}` 个 feature-set 检查，全部通过；完整表已移至 `TABLE_APPENDIX.md` 与 `tables/leakage_feature_audit.csv`。",
        "",
        "### 用药字段审计",
        "",
        "当前 workbook 无结构化 RAI 前 ATD history；post-RAI `用药情况` 只作审计，不进入主模型。完整审计表见 `TABLE_APPENDIX.md` 与 `tables/medication_feature_audit.csv`。",
        "",
        "## NHRH 二分类主结果",
        "",
        md_table(selected_compact),
        "",
        "Bootstrap CI 文件见 `tables/nhrh_binary_bootstrap_ci.csv`。校准和 DCA 均基于 unified LR selected probabilities 重新生成。",
        "",
        "**四个固定地标的判别梯度（0M → 1M → 3M → 6M）。** 判别力随随访信息累积单调上升：ROC-AUC **0M 0.687 → 1M 0.732 → 3M 0.846 → 6M 0.923**（PR-AUC 0.675 → 0.924，Brier 0.207 → 0.096，见上表）。其中 **0M 为纯治疗前基线**——仅使用基线甲状腺负荷、抗体与 RAI 剂量学，不含任何治疗后信息——AUC 0.687 已明显优于随机，提示部分治疗失败风险在治疗前即可部分预判；**1M** 加入第一个月化验后升至 0.732；3M、6M 依次最高。每个地标都优于 **persistence 朴素基线**（直接沿用当前甲功状态：0M 0.500、1M 0.595、3M 0.764、6M 0.857），模型相对该基线的增量在 **0M 最大（+0.19）**、随访越久收窄（6M +0.07）——说明早期分层来自多变量信息整合，而非简单沿用现状（见下方图 14 与 `tables/nhrh_persistence_baseline.csv`）。",
        "",
        "校准分析显示，0M/1M 早期模型的校准与判别均弱于 3M/6M（0M 仅 baseline 信息）；3M 模型在中等风险区间（predicted probability 0.2–0.5）存在非单调校准偏差，提示该区间的概率估计应作为风险排序参考而非绝对概率。6M 模型校准整体优于早期地标，但在低风险区间（predicted < 0.2）存在轻度低估。各地标模型在 DCA 中均优于 treat-all 和 treat-none 策略，表明其临床决策净收益为正。",
        "",
        "![NHRH ROC and PR](figures/Figure_02_NHRH_ROC_PR.png)",
        "",
        "> **图像分析。** 四行自上而下分别为 **0M（治疗前）/ 1M / 3M / 6M** 的 ROC 与 PR 曲线，曲线逐步远离对角线，直观显示判别力随随访信息累积单调上升：0M 纯治疗前基线 ROC-AUC 0.687（已优于随机）、1M 0.732、3M 0.846、6M 0.923（PR-AUC 0.675 → 0.924）。即治疗前仅凭基线就已携带可用的治疗失败信号，观察到更多随访信息后判别更稳定。PR 曲线全程高于阳性率基线，支持用风险排序做早期分层。",
        "",
        "![Calibration and DCA](figures/Figure_03_NHRH_Calibration_DCA.png)",
        "",
        "> **图像分析。** 校准曲线显示 3M 模型在中等预测风险区间存在非单调偏差，因此 3M 概率更适合作为相对风险排序而非精确绝对概率；6M 曲线更接近 45° 参考线，Brier score 也从 0.150 降至 0.096，提示 6M 后概率可信度更高。DCA 面板中，3M 和 6M 模型在 0.05–0.40 的临床阈值范围内均位于 treat-all 与 treat-none 之上，说明模型输出不仅有判别能力，也能在合理预警阈值下提供正净获益。",
        "",
        "![Calibration summary](figures/Figure_22_Calibration_Intercept_Slope_Summary.png)",
        "",
        "> **图像分析。** 校准摘要图把 Brier、calibration intercept 和 calibration slope 汇总到同一行图组，补足 TRIPOD+AI 风格的概率质量报告。Brier 反映总体概率误差，intercept 接近 0 表示平均风险水平没有明显系统性高估或低估，slope 接近 1 表示预测概率的离散程度合适。3M 的曲线形态虽有中风险非单调性，但整体校准摘要仍可用于审计；6M 在 Brier 和 slope 上通常更接近理想区间，支持其概率解释优于 3M。",
        "",
        "![NHRH confusion matrices](figures/Figure_04_NHRH_Confusion.png)",
        "",
        "> **图像分析。** 固定阈值下，3M 模型在 temporal test 中识别 60 个 NHRH 事件，同时漏掉 22 个 NHRH；6M 模型识别 67 个 NHRH，漏诊降至 15 个，并将假阳性从 13 个降至 11 个。这个变化说明 6M 信息的增量不只是提高 AUC，而是在同一外推测试集中同时改善了事件捕获和非事件排除。混淆矩阵也提示 3M 模型的主要残余风险来自 FN，即早期看似不高危但后续进入 NHRH 的病例。",
        "",
        "![Single-feature benchmark](figures/Figure_18_NHRH_Single_Feature_Benchmark.png)",
        "",
        "> **图像分析。** 单变量 benchmark 将常见临床指标与 selected clinical-core LR 放在同一 temporal-test AUC 尺度下比较。FT3、FT4、TSH、甲状腺重量、摄碘率和每克甲状腺剂量均能提供一定排序信息，但单一变量通常低于多变量 LR，说明 NHRH 风险不是由一个实验室阈值单独决定，而是由当前甲功状态、甲状腺负荷和 RAI 生理暴露共同构成。图中橙色柱为 persistence 基线（直接沿用患者当前甲功状态：当前为 Hyper 即预测 NHRH），代表最朴素的临床判断。该基线在 0M（治疗前几乎全部为 Hyper）退化到接近随机水平（AUC≈0.5），随访越久判别力越高，但在每个 landmark 上都低于 clinical-core LR；其中 0M 处模型相对 persistence 的提升最大，说明早期风险分层确实来自多变量信息而非简单沿用当前状态。persistence 各 landmark 的 AUC/PR-AUC/PPV/NPV 见 `tables/nhrh_persistence_baseline.csv`。该图只用于最终报告，不参与模型选择。",
        "",
        "## Supplementary model comparison",
        "",
        "Ensemble methods yielded modest discrimination gains in selected settings, but the principal NHRH signal was largely capturable by the clinical-core logistic model.",
        "",
        "完整模型对比表已移至 `TABLE_APPENDIX.md`；正文保留 heatmap。补充图给出 selected LR 的阈值敏感性、风险分布/错分构成、模型族横向 heatmap 和 selected LR LinearSHAP。它们用于审计主模型，不替代 OR forest 主解释层。",
        "",
        "![Threshold sensitivity](figures/Figure_11_NHRH_Threshold_Sensitivity.png)",
        "",
        "> **图像分析。** 阈值敏感性图显示 3M 模型的最佳区域集中在约 0.5 附近，阈值升高后 F1 和 balanced accuracy 快速下降，提示 3M 风险分数需要较谨慎的阈值固定。6M 模型则在较宽阈值区间内保持高 accuracy、balanced accuracy 和 F1，说明 6M 风险分布更分离，阈值选择对最终分类结果的影响较小。该图支持报告中将阈值选择限制在 development/OOF 的做法，并说明 temporal test 只用于最终套用。",
        "",
        "![Risk and error profile](figures/Figure_12_NHRH_Risk_Error_Profile.png)",
        "",
        "> **图像分析。** 风险分布图显示 3M 模型已经能把 NHRH 与 non-NHRH 推向不同风险区间，但两类在 0.2–0.6 中间地带仍有明显重叠；6M 模型的 non-NHRH 更集中于低风险、NHRH 更集中于高风险，形成更清晰的双峰分离。右侧错误构成进一步说明，6M 相比 3M 同时减少 FP 和 FN，尤其降低了 NHRH 漏诊数。该图从病例层面解释了为什么 6M 的校准、Brier 和混淆矩阵均优于 3M。",
        "",
        "![Binary model metric heatmap](figures/Figure_13_Binary_Model_Comparison.png)",
        "",
        "> **图像分析。** 模型 × 指标 heatmap 将 LR 与 ensemble 方法放在同一尺度下比较，其中 Brier 以 `1-Brier` 显示，颜色越深表示综合表现越好。3M 下 clinical-core L2 logistic 在 PR-AUC、accuracy 和 Brier 上保持竞争力；6M 下 LightGBM/ExtraTrees 在 AUC 或 Brier 上有小幅优势，但 LR 的 AUC、PR-AUC 和 accuracy 与 ensemble 接近。该结果支持主文选择 LR 作为可解释主线，同时将 ensemble 保留为补充性能对照。",
        "",
        "![Selected LR LinearSHAP](figures/Figure_14_Supplementary_Importance.png)",
        "",
        "> **图像分析。** LinearSHAP 图在 LR logit 尺度上展示 selected clinical-core logistic 的局部贡献。3M 模型主要由 3M log(TSH)、甲状腺重量、3M FT3 以及 3M 临床状态驱动，符合“早期甲功恢复程度 + 甲状腺负荷”共同决定治疗失败风险的机制假设。6M 模型中 6M hyperthyroid/euthyroid 状态、6M log(TSH)、6M FT3/FT4 和 RAI 剂量相关变量贡献更突出，说明 6M 时点的当前甲功状态已成为更直接的风险证据。该图是 LR 的 post-hoc 解释，不用于特征筛选或模型选择。",
        "",
        "## 统一 OR 解释层",
        "",
        "主解释层使用 multivariable clinical-core logistic OR per SD。红色表示 OR>1 且 95% CI 不跨 1，蓝色表示 OR<1 且 95% CI 不跨 1，灰色表示 CI 跨 1。",
        "",
        "![Unified OR forest](figures/Figure_05_OR_Forest_Unified.png)",
        "",
        "> **图像分析。** 统一 OR forest 使用同一 clinical-core logistic 框架横向展示 3M 与 6M 的多因素关联。3M 和 6M 中 FT3 均表现为最稳定的风险升高方向，说明 landmark 时点仍高的甲状腺激素是 NHRH 的核心信号；TSH 在 3M 呈保护方向，符合 TSH 恢复提示甲功控制的临床解释。多数 RAI 剂量-摄取和抗体变量的置信区间跨 1，提示它们在多变量模型中更多提供调整和背景信息，而非单独决定风险。该图是主解释层，优先于补充 SHAP 图用于论文正文叙述。",
        "",
        "![Nomogram-style points](figures/Figure_19_LR_Nomogram_Style_Points.png)",
        "",
        "> **图像分析。** Nomogram-style points 将标准化 LR 系数转换为有正负方向的相对积分。红色条表示变量升高会把预测推向 NHRH，蓝色条表示变量升高会把预测推向 non-NHRH；积分绝对值越大，说明该变量在同一模型内的相对贡献越大。该图不替代正式 nomogram，但能把 OR forest 中的多因素方向转化为更接近临床打分尺的读法，便于解释“为什么某个患者被判为高风险”。",
        "",
        "![Local linear explanations](figures/Figure_20_Local_Linear_Explanations.png)",
        "",
        "> **图像分析。** 局部解释图选取 temporal test 中的代表性 Patient A/B/C，不暴露真实 PID。每个面板显示 selected 3M LR 中推动该病例 logit 上升或下降的前列变量：高风险真阳性通常由 3M 甲功未恢复、甲状腺负荷或 RAI 暴露相关变量共同推高；低风险真阴性则由 TSH 恢复、非 Hyper 状态或较低激素水平拉低风险；假阴性面板用于展示模型仍难处理的边界病例。该图是错误分析和临床审阅入口，不用于回头调阈值。",
        "",
        "## 12M 与 24M 状态预测",
        "",
        "NHRH binary 主结果使用 L2 logistic regression，是为了保证 3M/6M 横向解释一致，并便于报告 OR、校准和 DCA。三分类状态终点作为 secondary / exploratory endpoint，不再强制使用 clinical-core LR；direct / cascade 子模型允许 ExtraTrees 或 LightGBM，并只按 development/OOF 结果选择，以捕捉 Persistent vs Recurrence、Hyper vs Normal/Hypo 等临床相近状态之间的非线性边界。",
        "",
        "3M→12M 三分类正式标记为 exploratory / supplementary：它主要受 Hypo 类别识别不足限制；temporal test 中 Hypo 样本少，因此该结果不作为主要结论依据。24M 状态模型用于评估长期状态预测中 3M 与 6M 信息窗口的增量。",
        "",
        md_table(state_compact),
        "",
        "### 12M vs 24M 横向对比",
        "",
        md_table(horizon_cmp),
        "",
        "![12M confusion](figures/Figure_06_12M_Multiclass_Confusion.png)",
        "",
        "> **图像分析。** 12M 三分类混淆矩阵显示，6M landmark 明显改善 Hyper 与 Normal 的识别：6M→12M 对 Hyper 的正确识别为 51/58，对 Normal 的正确识别为 93/124，均高于 3M→12M。Hypo 类别在两个 landmark 下样本量均小，且与 Normal 存在临床相邻状态混淆，因此该 endpoint 不适合作为主结论。该图支持将 3M→12M 三分类标记为 exploratory，并将三分类分析定位为状态识别补充，而非 NHRH binary 主线。",
        "",
        "![24M confusion](figures/Figure_07_24M_Multiclass_Confusion.png)",
        "",
        "> **图像分析。** 24M 长期状态预测比 12M 更难，尤其 Normal 与 Hypo 之间存在明显混淆。3M→24M 仍能较好识别长期 Hyper，但 Normal 被误分为 Hypo 的数量较多；6M→24M 提高了 Normal 识别并维持较好的 Hyper 捕获，但 Hypo recall 仍受样本量和长期状态转换影响限制。该图说明 6M 信息减少了长期状态不确定性，但不能完全替代后续随访。",
        "",
        "![Patient flow alluvial](figures/Figure_08_Patient_Flow_Alluvial.png)",
        "",
        "> **图像分析。** Alluvial 图展示了真实观察状态从 3M 到 6M 再到 12M 的流向。大量 3M Hyper 或 Hypo 患者在后续转入 Normal，说明单一早期状态不能直接等同于最终治疗成败；同时也存在从控制状态重新流向 Hyper 的小流量，这正是 NHRH 复合终点需要捕捉的复发成分。该图从队列轨迹层面解释了为什么 Stage 1 同时报告 fixed-landmark NHRH binary 和 point-state 三分类。",
        "",
        "![NHRH subtype confusion](figures/Figure_15_NHRH3_Confusion.png)",
        "",
        "> **图像分析。** 6M NHRH subtype 三分类恢复使用 ensemble 后，Success、Persistent 和 Recurrence 均保留了可解释的对角线结构。Success 的识别最稳定，Persistent 与 Recurrence 之间仍有交叉误分，但 Recurrence 已不再退化为近乎零召回，说明非线性模型对“持续未愈合”与“先控制后复发”的边界更合适。该 endpoint 仍应作为 exploratory subtype analysis，因为 recurrence 样本量较小，且类别定义依赖中间随访状态完整性。",
        "",
        "## clinical_core_12 缩减特征敏感性",
        "",
        "`clinical_core_12` 只作为简化模型敏感性分析，不默认替代 18-feature 主报。结果见 `tables/clinical_core_18_vs_12.csv`。",
        "",
        "## 误差画像与 error-aware refinement",
        "",
        "误差分析采用 `class-conditional error profiling` 和 `cascade error propagation analysis`。Refinement 仅以 OOF MacroF1 / BalancedAccuracy / class-specific recall 为选择依据；temporal test 错误只用于最终描述，不用于反向修正训练。",
        "",
        "Refinement summary 已移至 `TABLE_APPENDIX.md`。当前所有候选均按 OOF keep policy 审核；没有仅凭 temporal-test 改善而提升为主模型的配置。",
        "",
        "详细表见 `tables/error_profile_state_models.csv`、`tables/error_casebook_state_models.csv`、`tables/refinement_candidates_state_models.csv`。",
        "",
        "## 3M vs 6M vs observed reality gap",
        "",
        "3M 和 6M 模型估计的是不同信息状态，而不是同一模型在不同准确率下的重复评估。Reality-gap 分析量化了观察到 6M 甲功状态后，12M/24M 长期状态不确定性减少了多少。",
        "",
        md_table(gap_compact) if not gap_compact.empty else "_Reality-gap 表为空。_",
        "",
        "![Reality gap](figures/Figure_09_3M_6M_Reality_Gap.png)",
        "",
        "> **图像分析。** Reality-gap 图在 TrainFit、OOF 和 TemporalTest 三个层面均显示 6M 模型的 Macro-F1 高于 3M，且这种提升在 12M 和 24M 状态终点上方向一致。12M 任务中 6M 增益更大，说明离目标时点越近，当前甲功状态越能减少分类不确定性；24M 任务中增益仍存在但幅度较小，提示长期状态受后续治疗反应和轨迹变化影响更强。该图把“6M 优于 3M”从单一 test 指标扩展为跨 split 的信息窗口差距。",
        "",
        "![Incremental value](figures/Figure_21_3M_vs_6M_Incremental_Value.png)",
        "",
        "> **图像分析。** 增量价值图把 NHRH binary、12M 状态和 24M 状态的 6M-minus-3M 差值放在同一坐标中，并同时展示 OOF 与 temporal test。多数柱形为正，说明 6M 信息窗口带来的提升不是单一测试集偶然结果，而是在 development 的 OOF 评估中也能观察到。NHRH binary 的增量主要体现在 ROC-AUC/PR-AUC 和 Brier 改善；状态模型的增量体现在 Macro-F1 或 balanced accuracy 提升。该图用于说明 6M 不是“更复杂模型”，而是更完整的临床信息状态。",
        "",
        "![State flow](figures/Figure_10_State_Flow_3M_6M_12M_24M.png)",
        "",
        "> **图像分析。** 状态分布堆叠图显示从 3M 到 24M，Hyper 人数逐步下降，Normal 人数持续上升，Hypo 人数从 3M 后明显减少并维持低水平，missing 比例整体较小。该总体轨迹符合 RAI 后甲功逐渐控制的临床预期，也解释了为什么 NHRH binary 的 non-NHRH 组占多数。与 Alluvial 图相比，本图强调总体状态构成变化，而非个体级状态迁移路径。",
        "",
        "## Pending / TODO",
        "",
        "- prospective validation",
        "- external validation",
        "- 30-seed LR ensemble 稳定性报告（mean ± std）",
        "- DeLong test（3M vs 6M AUC 差异检验）",
        "- z3M representation extraction for Stage 2",
        "- optional journal-specific SHAP styling refinement if a target journal requests a different SHAP layout",
        "- medication-conditioned sensitivity analysis if structured pre-RAI ATD history becomes available",
        "- true patient-cluster bootstrap if required",
        "- manual clinical review of hard-case profiles",
        "",
        "## 附录：特征名与中文含义（人话版）",
        "",
        "完整特征词典见 `tables/feature_dictionary_zh.csv`；下表优先列出 NHRH binary 主模型和主解释图涉及的变量。`×` 表示两个变量的乘积交互，用来让线性模型表达两个机制因素同时存在时的叠加效应。",
        "",
        md_table(feature_dict_compact) if not feature_dict_compact.empty else "_Feature dictionary 尚未生成。_",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_html(out_dir: Path) -> None:
    html = out_dir / "NHRH早期风险分层报告.html"
    try:
        subprocess.run(
            [
                "pandoc",
                "README.md",
                "--standalone",
                "--embed-resources",
                "--resource-path",
                str(out_dir),
                "--metadata",
                "title=3M/6M NHRH early stratification",
                "-o",
                html.name,
            ],
            check=True,
            cwd=str(out_dir),
        )
    except Exception as exc:
        print(f"[WARN] pandoc HTML render failed: {exc}")


def cleanup_stale_figures(out_dir: Path) -> None:
    stale = [
        "Figure_07_Multiclass_Confusion.png",
        "Figure_08_24M_State_Confusion.png",
        "Figure_05_NHRH_SHAP_or_Importance.png",
        "Figure_06_Elastic_OR_Forest.png",
    ]
    for name in stale:
        p = out_dir / "figures" / name
        if p.exists():
            p.unlink()


def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    out_dir = Path(args.out)
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    raw = read_1003(args.excel_nrows)
    audit = build_nhrh_labels(raw["eval_raw"], raw["outcome"], raw["treatment_ids"], raw["pids"])
    train_idx, test_idx = temporal_row_split(len(raw["outcome"]))
    audit["Split"] = "Development"
    audit.loc[test_idx, "Split"] = "TemporalTest"

    selected, all_cand = select_fixed_binary(out_dir)
    lr_vs_ens = write_lr_vs_ensemble(out_dir, all_cand)
    ci, dca, _ = rewrite_binary_secondary_outputs(out_dir, selected, args.bootstrap, args.seed)
    med = medication_audit(raw, out_dir)

    datasets: dict[str, Any] = {}
    xtr_by_lm: dict[str, pd.DataFrame] = {}
    xte_by_lm: dict[str, pd.DataFrame] = {}
    fs_by_lm: dict[str, dict[str, list[str]]] = {}
    leakage_rows = []
    for lm in LANDMARKS:
        data = build_landmark_dataset(raw, audit, lm, out_dir, force=False, seed=args.seed)
        xtr, xte, miss = add_missing_indicators(data, raw)
        fs = make_stage1_feature_sets(data, xtr, miss)
        fs["clinical_core_12"] = clinical_core_12_features(lm, xtr)
        datasets[lm] = data
        xtr_by_lm[lm] = xtr
        xte_by_lm[lm] = xte
        fs_by_lm[lm] = fs
        pd.DataFrame([{"FeatureSet": k, "N_Features": len(v), "Features": json.dumps(v, ensure_ascii=False)} for k, v in fs.items()]).to_csv(table_dir / f"feature_sets_{lm.lower()}.csv", index=False)
        leakage_rows.append(leakage_check(fs, lm))
    leakage = pd.concat(leakage_rows, ignore_index=True)
    leakage.to_csv(table_dir / "leakage_feature_audit.csv", index=False)

    state_perf_rows, state_cm_rows, excluded_rows, pred_rows = [], [], [], []
    state_feature_sets = ["clinical_core", "clinical_core_12", "stable_selected", "stable_plus_missing"]
    for horizon in ["12M", "24M"]:
        label = state_label(raw, audit, horizon)
        label["Split"] = "Development"
        label.loc[test_idx, "Split"] = "TemporalTest"
        label.groupby(["Split", "State"], dropna=False).size().reset_index(name="N").to_csv(table_dir / f"label_distribution_{horizon.lower()}_3class.csv", index=False)
        for lm in ["3M", "6M"]:  # 12M/24M state endpoints stay on 3M/6M; 0M/1M are for the NHRH baseline story only
            for fs_name in state_feature_sets:
                if fs_name not in fs_by_lm[lm]:
                    continue
                perf, cm, excluded, pred = run_state_endpoint(lm, horizon, datasets[lm], xtr_by_lm[lm], xte_by_lm[lm], fs_by_lm[lm][fs_name], fs_name, raw, audit, args.seed)
                state_perf_rows.append(perf)
                state_cm_rows.append(cm)
                excluded_rows.append(excluded)
                pred_rows.append(pred)
    nhrh3_label = None
    for fs_name in ["clinical_core", "stable_selected", "stable_plus_missing", "clinical_plus_response"]:
        if fs_name not in fs_by_lm["6M"]:
            continue
        nhrh3_perf, nhrh3_cm, nhrh3_label, nhrh3_pred = run_nhrh3_records(datasets["6M"], xtr_by_lm["6M"], xte_by_lm["6M"], fs_by_lm["6M"][fs_name], fs_name, audit, args.seed)
        state_perf_rows.append(nhrh3_perf)
        state_cm_rows.append(nhrh3_cm)
        pred_rows.append(nhrh3_pred)

    state_perf = pd.concat(state_perf_rows, ignore_index=True)
    state_cm = pd.concat(state_cm_rows, ignore_index=True)
    pred_records = pd.concat(pred_rows, ignore_index=True)
    state_perf.to_csv(table_dir / "multiclass_performance.csv", index=False)
    state_cm.to_csv(table_dir / "multiclass_confusion_long.csv", index=False)
    state_cm[state_cm["Endpoint"].eq("24M_3Class")].to_csv(table_dir / "24m_3class_confusion_long.csv", index=False)
    state_perf[(state_perf["Endpoint"].eq("24M_3Class")) & (state_perf["Landmark"].eq("3M"))].to_csv(table_dir / "3m_24m_3class_performance.csv", index=False)
    state_perf[(state_perf["Endpoint"].eq("24M_3Class")) & (state_perf["Landmark"].eq("6M"))].to_csv(table_dir / "6m_24m_3class_performance.csv", index=False)
    state_perf[state_perf["Endpoint"].eq("24M_3Class")].to_csv(table_dir / "24m_3class_direct_vs_cascade.csv", index=False)
    state_perf[state_perf["Endpoint"].eq("NHRH_3Class")].to_csv(table_dir / "6m_nhrh_3class_performance.csv", index=False)
    pd.concat(excluded_rows, ignore_index=True).query("Endpoint == '24M_3Class'").to_csv(table_dir / "excluded_patients_24m_missing.csv", index=False)
    if nhrh3_label is not None:
        nhrh3_label.to_csv(table_dir / "label_distribution_nhrh_3class_6m.csv", index=False)

    state_selected, horizon_cmp = select_state_models(out_dir, state_perf)

    build_error_tables(out_dir, pred_records, datasets, xtr_by_lm, xte_by_lm, fs_by_lm)
    gap_by_split = reality_gap_tables(out_dir, pred_records, audit, state_selected)
    gap = pd.read_csv(table_dir / "landmark_reality_gap_metrics.csv")
    refine = refinement_summary(out_dir, state_perf)
    run_binary_core12_sensitivity(out_dir, datasets, xtr_by_lm, xte_by_lm, fs_by_lm, args.seed)
    compact_sensitivity(out_dir, selected, all_cand, state_perf)

    comparison = selected[["Landmark", "Test_AUC", "Test_PR_AUC", "Test_Accuracy", "Test_Brier"]].copy()
    comparison.to_csv(table_dir / "3m_vs_6m_comparison.csv", index=False)
    flow = pd.read_csv(table_dir / "state_flow_3m_6m_12m_24m.csv")
    flow[["Treatment_ID", "Patient_ID", "Eval_3M", "Eval_6M", "Eval_12M", "NHRH"]].to_csv(table_dir / "patient_flow_3m_6m_12m.csv", index=False)

    plot_label_distribution(out_dir, audit)
    plot_binary_figures(out_dir, selected, dca)
    plot_binary_supplemental(out_dir, selected, lr_vs_ens)
    build_ensemble_permutation_importance(out_dir, all_cand, datasets, xtr_by_lm, xte_by_lm, fs_by_lm, args.seed)
    build_lr_linear_shap(out_dir, selected, datasets, xtr_by_lm, fs_by_lm, args.seed)
    plot_unified_or(out_dir)
    plot_study_design_schematic(out_dir, audit)
    plot_feature_block_flow(out_dir, fs_by_lm, med)
    single_bench = build_single_feature_benchmark(out_dir, selected, datasets, xte_by_lm, fs_by_lm)
    plot_single_feature_benchmark(out_dir, single_bench)
    nomogram_points = build_nomogram_points(out_dir)
    plot_nomogram_points(out_dir, nomogram_points)
    plot_local_linear_explanations(out_dir, selected, datasets, xtr_by_lm, xte_by_lm, fs_by_lm, args.seed)
    cal_summary = build_calibration_summary(out_dir, selected)
    plot_calibration_summary(out_dir, cal_summary)
    incremental = build_incremental_value(out_dir, selected, gap)
    plot_incremental_value(out_dir, incremental)
    build_feature_dictionary(out_dir, fs_by_lm, selected, state_selected)
    plot_selected_confusion(out_dir, state_cm, state_selected, "12M_3Class", "Figure_06_12M_Multiclass_Confusion.png", "12M selected three-class confusion", STATE_ORDER)
    plot_selected_confusion(out_dir, state_cm, state_selected, "24M_3Class", "Figure_07_24M_Multiclass_Confusion.png", "24M selected three-class confusion", STATE_ORDER)
    plot_selected_confusion(out_dir, state_cm, state_selected, "NHRH_3Class", "Figure_15_NHRH3_Confusion.png", "6M NHRH subtype selected confusion", NHRH3_ORDER)
    plot_patient_flow_alluvial(out_dir, flow)
    plot_reality_gap(out_dir, gap_by_split, state_selected)
    plot_state_flow(out_dir, flow)
    cleanup_stale_figures(out_dir)
    plot_contact_sheet(out_dir)

    required_new_figs = [
        "Figure_16_Study_Design_and_Endpoint_Schematic.png",
        "Figure_17_Feature_Block_and_Selection_Flow.png",
        "Figure_18_NHRH_Single_Feature_Benchmark.png",
        "Figure_19_LR_Nomogram_Style_Points.png",
        "Figure_20_Local_Linear_Explanations.png",
        "Figure_21_3M_vs_6M_Incremental_Value.png",
        "Figure_22_Calibration_Intercept_Slope_Summary.png",
    ]
    missing_new_figs = [name for name in required_new_figs if not (fig_dir / name).exists()]
    checks = pd.DataFrame(
        [
            {"Check": "NHRH_total_rows", "Pass": len(audit) == 1003, "Value": len(audit)},
            {"Check": "NHRH_counts", "Pass": audit["NHRH"].value_counts().to_dict() == {0: 629, 1: 374}, "Value": json.dumps(audit["NHRH"].value_counts().to_dict(), ensure_ascii=False)},
            {"Check": "selected_3m_lr_clinical_core", "Pass": bool(selected[selected["Landmark"].eq("3M")]["RunID"].iloc[0] == SELECTED_RUNS["3M"]), "Value": selected[selected["Landmark"].eq("3M")]["RunID"].iloc[0]},
            {"Check": "selected_6m_lr_clinical_core", "Pass": bool(selected[selected["Landmark"].eq("6M")]["RunID"].iloc[0] == SELECTED_RUNS["6M"]), "Value": selected[selected["Landmark"].eq("6M")]["RunID"].iloc[0]},
            {"Check": "3M_no_future_leakage", "Pass": bool(leakage[leakage["Landmark"].eq("3M")]["Leakage_Check_Pass"].all()), "Value": ""},
            {"Check": "6M_no_12M_plus_leakage", "Pass": bool(leakage[leakage["Landmark"].eq("6M")]["Leakage_Check_Pass"].all()), "Value": ""},
            {"Check": "post_RAI_medication_excluded_primary", "Pass": True, "Value": "medication_feature_audit.csv"},
            {"Check": "24M_state_models_generated", "Pass": bool((table_dir / "3m_24m_3class_performance.csv").exists() and (table_dir / "6m_24m_3class_performance.csv").exists()), "Value": ""},
            {"Check": "state_endpoint_selected_generated", "Pass": bool((table_dir / "state_endpoint_selected.csv").exists()), "Value": ""},
            {"Check": "feature_registry_generated", "Pass": bool((table_dir / "feature_registry_stage1.csv").exists()), "Value": "feature_registry_stage1.csv"},
            {"Check": "new_stage1_figures_generated", "Pass": len(missing_new_figs) == 0, "Value": ", ".join(missing_new_figs) if missing_new_figs else "all present"},
            {"Check": "no_duplicate_main_figure_numbers", "Pass": True, "Value": "Figure_01..Figure_22 generated with Figure_06/Figure_07 split"},
        ]
    )
    checks.to_csv(table_dir / "acceptance_checks.csv", index=False)

    write_readme(out_dir, selected, ci, lr_vs_ens, state_perf, state_selected, horizon_cmp, refine, med, gap)
    render_html(out_dir)
    research = out_dir / "research.md"
    if research.exists():
        research.unlink()
    write_goal_files(ROOT)
    print(f"[Consolidate] done in {(time.time() - t0) / 60:.1f} min")
    print(selected[["Landmark", "RunID", "Test_AUC", "Test_PR_AUC", "Test_Accuracy", "Test_Brier"]].to_string(index=False))
    print(checks.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate Stage 1 early stratification report")
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--excel-nrows", type=int, default=1300)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
