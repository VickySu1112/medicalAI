"""Build a Chinese paper-style report for the T5_gate005 dynamic model.

The report is intentionally isolated from the older teacher/frozen-fuse and
fixed-landmark report directories. It reads the selected 4-branch model
artifacts, recomputes metrics from predictions, summarizes matched training-time
branch removal ablations, and writes all outputs under results/t5_dynamic_paper/.
"""

from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from scripts import relapse_direct_threebranch_aug as direct
from scripts import relapse_threehead_landmark as base
from utils.config import STATIC_NAMES


SRC = Path("/tmp/medical_direct_4branch_timegate_focused_sweep/T5_gate005/seed_2025")
SWEEP = Path("/tmp/medical_direct_4branch_timegate_focused_sweep")
TRAIN_BRANCH_ABLATION_RUNS = {
    "full": Path("/tmp/medical_direct_4branch_timegate_focused_sweep/T5_gate005/seed_2025"),
    "drop_static_train": Path("/tmp/medical_direct_branch_drop_seed2025/T5_gate005_drop_static/seed_2025"),
    "drop_local_train": Path("/tmp/medical_direct_4branch_drop_local_seed2025/T5_gate005_drop_local/seed_2025"),
    "drop_global_train": Path("/tmp/medical_direct_branch_drop_seed2025/T5_gate005_drop_global/seed_2025"),
    "drop_z3M_train": Path("/tmp/medical_direct_4branch_drop_z3m_seed2025/T5_gate005_drop_z3m/seed_2025"),
}
OUT = ROOT / "results" / "t5_dynamic_paper"
FIG = OUT / "figures"
TAB = OUT / "tables"
THRESHOLD = 0.785
SPLITS = ["Train", "Validation", "TemporalTest"]
SPLIT_LABEL = {"Train": "Train", "Validation": "Validation", "TemporalTest": "Temporal test"}
COLORS = {
    "Train": "#3B6FB6",
    "Validation": "#D98C2B",
    "TemporalTest": "#4B9B6E",
    "accent": "#765AA6",
    "red": "#C44E52",
    "gray": "#68717D",
    "light": "#F3F6FA",
}


FEATURE_ZH = {
    "Patient_ID": "患者编号",
    "Interval_ID": "动态随访区间编号",
    "Interval_Name": "随访时间窗",
    "Start_Time": "当前 landmark 时间（月）",
    "Stop_Time": "下一随访时间（月）",
    "Y_Relapse": "下一时间窗是否复发",
    "DirectProb": "模型预测复发概率",
    "FT3_Current": "当前游离三碘甲状腺原氨酸 FT3",
    "FT4_Current": "当前游离甲状腺素 FT4",
    "logTSH_Current": "当前 TSH 对数变换",
    "Delta_FT4_1step": "最近一步 FT4 变化",
    "Delta_TSH_1step": "最近一步 TSH 变化",
    "Delta_FT4_k0": "当前 FT4 相对基线变化",
    "Delta_TSH_k0": "当前 TSH 相对基线变化",
    "Prev_State": "当前区间起点临床状态",
    "Time_In_Normal": "此前处于正常甲功状态的累计次数",
    "Prior_Relapse_Count": "此前复发次数",
    "Ever_Hyper_Before": "此前是否出现过甲亢状态",
    "Ever_Hypo_Before": "此前是否出现过甲减状态",
    "z3m_available": "3M 表征是否可用",
    "z3m_time_since_3m": "距离 3M 的归一化时间",
}


def ensure_dirs() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    for stale in [
        TAB / "branch_knockout_predictions.csv",
        TAB / "checkpoint_prediction_recompute_check.csv",
        TAB / "seed_stability.csv",
        TAB / "seed_stability_aggregated_configs.csv",
        TAB / "source_summary_seed2025.csv",
        TAB / "source_predictions_seed2025.csv",
        TAB / "source_history_seed2025.csv",
        TAB / "source_summary.csv",
        TAB / "source_predictions.csv",
        TAB / "source_history.csv",
        TAB / "model_config_seed2025.csv",
        FIG / "Figure_03_Seed2025_Metric_Bars.png",
        FIG / "Figure_11_Seed_Stability.png",
        OUT / "DirectThreeBranch_Summary.csv",
        OUT / "DirectThreeBranch_Predictions.csv",
        OUT / "DirectThreeBranch_History.csv",
        OUT / "DirectThreeBranch.pt",
        OUT / "run.log",
    ]:
        if stale.exists():
            stale.unlink()


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Arial Unicode MS",
                "PingFang SC",
                "Heiti TC",
                "SimHei",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "figure.dpi": 140,
            "savefig.dpi": 240,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_fig(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    eps = 1e-6
    logit = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps)).reshape(-1, 1)
    try:
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        lr.fit(logit, y.astype(int))
        return float(lr.intercept_[0]), float(lr.coef_[0, 0])
    except Exception:
        return float("nan"), float("nan")


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float = THRESHOLD) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prevalence = float(y.mean()) if len(y) else float("nan")
    intercept, slope = calibration_intercept_slope(y, p)
    ppv = precision_score(y, pred, zero_division=0)
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    return {
        "N": int(len(y)),
        "Events": int(y.sum()),
        "Prevalence": prevalence,
        "Threshold": float(threshold),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "AUC": safe_auc(y, p),
        "PR_AUC": safe_ap(y, p),
        "PR_AUC_Lift": safe_ap(y, p) / prevalence if prevalence > 0 else float("nan"),
        "Accuracy": float(accuracy_score(y, pred)),
        "Recall": float(recall_score(y, pred, zero_division=0)),
        "Specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "PPV": float(ppv),
        "NPV": float(npv),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y, pred)),
        "Brier": float(brier_score_loss(y, p)),
        "Calibration_Intercept": intercept,
        "Calibration_Slope": slope,
    }


def fmt(x: object, digits: int = 3) -> str:
    if isinstance(x, (float, np.floating)):
        if math.isnan(float(x)):
            return ""
        return f"{float(x):.{digits}f}"
    return str(x)


def md_table(df: pd.DataFrame, digits: int = 3, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    disp = df.copy()
    for col in disp.columns:
        disp[col] = disp[col].map(lambda v: fmt(v, digits))
    try:
        return disp.to_markdown(index=False)
    except Exception:
        cols = list(disp.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in disp.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(lines)


def load_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [p for p in [SRC / "DirectThreeBranch_Summary.csv", SRC / "DirectThreeBranch_Predictions.csv", SRC / "DirectThreeBranch_History.csv"] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing selected model artifacts: {missing}")
    summary = pd.read_csv(SRC / "DirectThreeBranch_Summary.csv")
    pred = pd.read_csv(SRC / "DirectThreeBranch_Predictions.csv")
    history = pd.read_csv(SRC / "DirectThreeBranch_History.csv")
    pred["Patient_ID"] = pred["Patient_ID"].astype(str)
    return summary, pred, history


def compute_core_tables(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, cm_rows = [], []
    for split in SPLITS:
        sub = pred[pred["Split"].eq(split)]
        m = binary_metrics(sub["Y_Relapse"].values, sub["DirectProb"].values, THRESHOLD)
        m["Split"] = split
        rows.append(m)
        cm_rows.append(
            {
                "Split": split,
                "Threshold": THRESHOLD,
                "TP": m["TP"],
                "FP": m["FP"],
                "TN": m["TN"],
                "FN": m["FN"],
            }
        )
    core = pd.DataFrame(rows)[
        [
            "Split",
            "N",
            "Events",
            "Prevalence",
            "Threshold",
            "TP",
            "FP",
            "TN",
            "FN",
            "AUC",
            "PR_AUC",
            "PR_AUC_Lift",
            "Accuracy",
            "Recall",
            "Specificity",
            "PPV",
            "NPV",
            "F1",
            "Balanced_Accuracy",
            "Brier",
            "Calibration_Intercept",
            "Calibration_Slope",
        ]
    ]
    cm = pd.DataFrame(cm_rows)
    core.to_csv(TAB / "core_binary_metrics.csv", index=False)
    cm.to_csv(TAB / "confusion_matrix.csv", index=False)
    return core, cm


def threshold_sensitivity(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    thresholds = np.round(np.arange(0.05, 0.951, 0.005), 3)
    for split in ["Validation", "TemporalTest"]:
        sub = pred[pred["Split"].eq(split)]
        y = sub["Y_Relapse"].values.astype(int)
        p = sub["DirectProb"].values.astype(float)
        for thr in thresholds:
            m = binary_metrics(y, p, float(thr))
            rows.append(
                {
                    "Split": split,
                    "Threshold": float(thr),
                    "Accuracy": m["Accuracy"],
                    "F1": m["F1"],
                    "Recall": m["Recall"],
                    "Specificity": m["Specificity"],
                    "PPV": m["PPV"],
                    "NPV": m["NPV"],
                    "Balanced_Accuracy": m["Balanced_Accuracy"],
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "threshold_sensitivity_validation_selected.csv", index=False)
    return out


def topk_capture(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        sub = pred[pred["Split"].eq(split)].sort_values("DirectProb", ascending=False)
        total_events = float(sub["Y_Relapse"].sum())
        prevalence = float(sub["Y_Relapse"].mean())
        for frac in [0.05, 0.10, 0.15, 0.20, 0.30]:
            n = max(1, int(math.ceil(len(sub) * frac)))
            selected = sub.head(n)
            events = float(selected["Y_Relapse"].sum())
            precision = events / n
            rows.append(
                {
                    "Split": split,
                    "Top_Fraction": frac,
                    "Selected_N": n,
                    "Captured_Events": int(events),
                    "Total_Events": int(total_events),
                    "Event_Capture": events / total_events if total_events else float("nan"),
                    "Precision": precision,
                    "Lift": precision / prevalence if prevalence > 0 else float("nan"),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "topk_interval_capture.csv", index=False)
    return out


def patient_maxrisk(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    patient = (
        pred.groupby(["Split", "Patient_ID"], as_index=False)
        .agg(Y_Relapse=("Y_Relapse", "max"), DirectProb=("DirectProb", "max"), N_Intervals=("Interval_ID", "count"))
    )
    rows = []
    for split in SPLITS:
        sub = patient[patient["Split"].eq(split)]
        if len(sub) == 0:
            continue
        m = binary_metrics(sub["Y_Relapse"].values, sub["DirectProb"].values, THRESHOLD)
        m["Split"] = split
        rows.append(m)
    metrics = pd.DataFrame(rows)
    top_rows = []
    for split in SPLITS:
        sub = patient[patient["Split"].eq(split)].sort_values("DirectProb", ascending=False)
        total_events = float(sub["Y_Relapse"].sum())
        prevalence = float(sub["Y_Relapse"].mean())
        for frac in [0.05, 0.10, 0.20, 0.30]:
            n = max(1, int(math.ceil(len(sub) * frac)))
            selected = sub.head(n)
            events = float(selected["Y_Relapse"].sum())
            precision = events / n
            top_rows.append(
                {
                    "Split": split,
                    "Top_Fraction": frac,
                    "Selected_Patients": n,
                    "Captured_Event_Patients": int(events),
                    "Total_Event_Patients": int(total_events),
                    "Patient_Event_Capture": events / total_events if total_events else float("nan"),
                    "Precision": precision,
                    "Lift": precision / prevalence if prevalence > 0 else float("nan"),
                }
            )
    top = pd.DataFrame(top_rows)
    patient.to_csv(TAB / "patient_level_predictions_maxrisk.csv", index=False)
    metrics.to_csv(TAB / "patient_level_maxrisk_metrics.csv", index=False)
    top.to_csv(TAB / "patient_level_maxrisk_warning.csv", index=False)
    return metrics, top


def dca_curve(pred: pd.DataFrame) -> pd.DataFrame:
    sub = pred[pred["Split"].eq("TemporalTest")]
    y = sub["Y_Relapse"].values.astype(int)
    p = sub["DirectProb"].values.astype(float)
    n = len(y)
    prevalence = float(y.mean())
    rows = []
    for pt in np.round(np.arange(0.05, 0.401, 0.01), 3):
        alert = p >= pt
        tp = float(((y == 1) & alert).sum())
        fp = float(((y == 0) & alert).sum())
        weight = pt / (1 - pt)
        rows.append(
            {
                "Threshold_Probability": float(pt),
                "Model_Net_Benefit": tp / n - fp / n * weight,
                "Treat_All_Net_Benefit": prevalence - (1 - prevalence) * weight,
                "Treat_None_Net_Benefit": 0.0,
                "Alert_Rate": float(alert.mean()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "decision_curve_005_040.csv", index=False)
    return out


def calibration_bins(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        sub = pred[pred["Split"].eq(split)].copy()
        try:
            sub["Risk_Decile"] = pd.qcut(sub["DirectProb"], 10, labels=False, duplicates="drop") + 1
        except ValueError:
            sub["Risk_Decile"] = 1
        for decile, part in sub.groupby("Risk_Decile"):
            rows.append(
                {
                    "Split": split,
                    "Risk_Decile": int(decile),
                    "N": int(len(part)),
                    "Events": int(part["Y_Relapse"].sum()),
                    "Mean_Predicted_Risk": float(part["DirectProb"].mean()),
                    "Observed_Risk": float(part["Y_Relapse"].mean()),
                    "Min_Predicted_Risk": float(part["DirectProb"].min()),
                    "Max_Predicted_Risk": float(part["DirectProb"].max()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "calibration_bins.csv", index=False)
    return out


def cluster_bootstrap_ci(pred: pd.DataFrame, n_boot: int = 1000, seed: int = 20260504) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    metrics = ["AUC", "PR_AUC", "Accuracy", "Recall", "Specificity", "PPV", "NPV", "F1", "Brier"]
    for split in SPLITS:
        sub = pred[pred["Split"].eq(split)].reset_index(drop=True)
        patient_to_idx = {pid: idx.values for pid, idx in sub.groupby("Patient_ID").groups.items()}
        patients = np.asarray(list(patient_to_idx.keys()))
        boot = {m: [] for m in metrics}
        for _ in range(n_boot):
            sampled = rng.choice(patients, size=len(patients), replace=True)
            idx = np.concatenate([patient_to_idx[pid] for pid in sampled])
            b = sub.iloc[idx]
            y = b["Y_Relapse"].values.astype(int)
            p = b["DirectProb"].values.astype(float)
            if len(np.unique(y)) < 2:
                continue
            bm = binary_metrics(y, p, THRESHOLD)
            for m in metrics:
                boot[m].append(bm[m])
        point = binary_metrics(sub["Y_Relapse"].values, sub["DirectProb"].values, THRESHOLD)
        for m in metrics:
            vals = np.asarray(boot[m], dtype=float)
            rows.append(
                {
                    "Split": split,
                    "Metric": m,
                    "Point": point[m],
                    "CI_Low": float(np.nanpercentile(vals, 2.5)) if len(vals) else float("nan"),
                    "CI_High": float(np.nanpercentile(vals, 97.5)) if len(vals) else float("nan"),
                    "Bootstrap_Unit": "Patient_ID",
                    "Bootstrap_N": int(len(vals)),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "cluster_bootstrap_ci_by_patient.csv", index=False)
    return out


def merge_feature_frame(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_tr, df_te, _, _, unique_pids = base.build_longitudinal_tables()
    train_patient_order = unique_pids[: int(len(unique_pids) * 0.8)]
    df_fit, df_val = base.split_train_validation(df_tr, train_patient_order)
    frames = []
    for split, df in [("Train", df_fit), ("Validation", df_val), ("TemporalTest", df_te)]:
        tmp = df.copy()
        tmp["Split"] = split
        frames.append(tmp)
    feature_df = pd.concat(frames, ignore_index=True)
    feature_df["Patient_ID"] = feature_df["Patient_ID"].astype(str)
    keys = ["Patient_ID", "Source_Row", "Interval_ID", "Interval_Name", "Start_Time", "Stop_Time", "Y_Relapse", "Split"]
    merged = pred.merge(feature_df, on=keys, how="left", validate="one_to_one", suffixes=("", "_feature"))
    merge_check = pd.DataFrame(
        [
            {
                "Rows": len(pred),
                "Merged_Rows": len(merged),
                "Rows_With_Features": int(merged["Prev_State"].notna().sum()) if "Prev_State" in merged else 0,
                "Feature_Merge_Complete": bool("Prev_State" in merged and merged["Prev_State"].notna().all()),
            }
        ]
    )
    merge_check.to_csv(TAB / "feature_merge_check.csv", index=False)
    return merged, feature_df


def error_casebook(merged: pd.DataFrame) -> pd.DataFrame:
    df = merged.copy()
    df["Pred_Label"] = (df["DirectProb"] >= THRESHOLD).astype(int)
    df["Error_Group"] = np.select(
        [
            (df["Y_Relapse"].eq(1) & df["Pred_Label"].eq(1)),
            (df["Y_Relapse"].eq(0) & df["Pred_Label"].eq(1)),
            (df["Y_Relapse"].eq(1) & df["Pred_Label"].eq(0)),
            (df["Y_Relapse"].eq(0) & df["Pred_Label"].eq(0)),
        ],
        ["TP", "FP", "FN", "TN"],
        default="NA",
    )
    cols = [
        "Patient_ID",
        "Split",
        "Interval_ID",
        "Interval_Name",
        "Start_Time",
        "Stop_Time",
        "Y_Relapse",
        "DirectProb",
        "Pred_Label",
        "Error_Group",
        "Prev_State",
        "FT3_Current",
        "FT4_Current",
        "logTSH_Current",
        "Delta_FT4_1step",
        "Delta_TSH_1step",
        "Time_In_Normal",
        "Prior_Relapse_Count",
        "Ever_Hyper_Before",
        "Ever_Hypo_Before",
    ]
    keep = [c for c in cols if c in df.columns]
    out = df[keep].sort_values(["Split", "Error_Group", "DirectProb"], ascending=[True, True, False])
    out.to_csv(TAB / "error_casebook.csv", index=False)
    group_rows = []
    feature_cols = [c for c in keep if c not in {"Patient_ID", "Split", "Interval_ID", "Interval_Name", "Prev_State", "Error_Group"}]
    for (split, group), part in df.groupby(["Split", "Error_Group"]):
        row = {
            "Split": split,
            "Error_Group": group,
            "N": int(len(part)),
            "Events": int(part["Y_Relapse"].sum()),
            "Mean_Probability": float(part["DirectProb"].mean()),
            "Median_Probability": float(part["DirectProb"].median()),
        }
        for col in feature_cols:
            if col in part:
                row[f"Mean_{col}"] = float(pd.to_numeric(part[col], errors="coerce").mean())
        group_rows.append(row)
    group_df = pd.DataFrame(group_rows)
    group_df.to_csv(TAB / "error_group_feature_profile.csv", index=False)
    return group_df


def build_feature_dictionary(feature_df: pd.DataFrame) -> pd.DataFrame:
    interval_cats = sorted(feature_df[feature_df["Split"].eq("Train")]["Interval_Name"].astype(str).unique())
    prev_state_cats = sorted(feature_df[feature_df["Split"].eq("Train")]["Prev_State"].astype(str).unique())
    static_df, local_df, global_df = base.build_feature_blocks(
        feature_df[feature_df["Split"].eq("Train")],
        interval_cats,
        prev_state_cats,
    )
    z3m_cols = [f"selected_z3m_{i:02d}" for i in range(12)] + ["z3m_available", "z3m_time_since_3m"]
    rows = []
    for branch, cols in [
        ("static", list(static_df.columns)),
        ("local", list(local_df.columns)),
        ("global", list(global_df.columns)),
        ("z3M", z3m_cols),
    ]:
        for col in cols:
            rows.append({"Branch": branch, "Feature": col, "中文含义": translate_feature(col)})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "feature_dictionary_zh.csv", index=False)
    return out


def translate_feature(name: str) -> str:
    if name in FEATURE_ZH:
        return FEATURE_ZH[name]
    if name in STATIC_NAMES:
        return f"基线静态特征：{name}"
    if name.startswith("Window_"):
        return "当前动态预测窗口指示变量：" + name.replace("Window_", "")
    if name.startswith("PrevState_"):
        return "当前窗口起点状态 one-hot：" + name.replace("PrevState_", "")
    if name.startswith("CoreWindow_"):
        return "高风险窗口指示变量：" + name.replace("CoreWindow_", "")
    if "_x_CoreWindow_" in name:
        left, right = name.split("_x_CoreWindow_", 1)
        return f"{translate_feature(left)} 与高风险窗口 {right} 的交互"
    if name.startswith("selected_z3m_"):
        return "3M early-response encoder 的第 " + name.rsplit("_", 1)[-1] + " 个嵌入维度"
    return name


def z3m_leakage_checks(feature_df: pd.DataFrame, cfg: direct.TrainConfig) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        df = feature_df[feature_df["Split"].eq(split)].reset_index(drop=True)
        z = direct.build_z3m_features(df, cfg)
        before = df["Start_Time"].values < 3.0
        comp_cols = [c for c in z.columns if c.startswith("selected_z3m_")]
        comp_ok = bool(np.allclose(z.loc[before, comp_cols].values, 0.0)) if before.any() else True
        avail_ok = bool(np.allclose(z.loc[before, "z3m_available"].values, 0.0)) if before.any() else True
        if "z3m_time_since_3m" in z:
            time_ok = bool(np.allclose(z.loc[before, "z3m_time_since_3m"].values, 0.0)) if before.any() else True
        else:
            time_ok = True
        rows.append(
            {
                "Split": split,
                "Rows": int(len(df)),
                "Rows_StartBefore3M": int(before.sum()),
                "Components_Zero_Before3M": comp_ok,
                "Availability_Zero_Before3M": avail_ok,
                "TimeFeature_Zero_Before3M": time_ok,
                "Z3M_Source_Rule": "fit uses OOF; validation uses fit-context; test uses final fit+val encoder",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "z3m_leakage_checks.csv", index=False)
    return out


def rebuild_model_and_datasets() -> tuple[direct.ThreeBranchHazardNet, dict[str, direct.DirectDataset], direct.TrainConfig, pd.DataFrame]:
    checkpoint = torch.load(SRC / "DirectThreeBranch.pt", map_location="cpu", weights_only=False)
    cfg_dict = dict(checkpoint["config"])
    cfg_dict["output_dir"] = Path(cfg_dict["output_dir"])
    cfg_dict["z3m_path"] = Path(cfg_dict["z3m_path"])
    if cfg_dict.get("feature_selection_json") not in (None, "", float("nan")):
        cfg_dict["feature_selection_json"] = Path(cfg_dict["feature_selection_json"])
    cfg = direct.TrainConfig(**cfg_dict)
    cfg.device = "cpu"
    cfg.batch_size = 512
    cfg.augment = False
    direct.seed_everything(cfg.seed)
    df_tr, df_te, _, _, unique_pids = base.build_longitudinal_tables()
    train_patient_order = unique_pids[: int(len(unique_pids) * 0.8)]
    df_fit, df_val = base.split_train_validation(df_tr, train_patient_order)
    _, fit_ds, val_ds, te_ds, _, _ = direct.build_datasets(df_fit, df_val, df_te, cfg)
    model = direct.ThreeBranchHazardNet(
        static_dim=fit_ds.static.shape[1],
        local_dim=fit_ds.local.shape[1],
        global_dim=fit_ds.global_x.shape[1],
        embed_dim=cfg.embed_dim,
        dropout=cfg.dropout,
        use_static_branch=not cfg.drop_static_branch,
        z3m_dim=fit_ds.z3m.shape[1],
        use_z3m_branch=cfg.z3m_mode in direct.Z3M_BRANCH_MODES,
        z3m_gate_init=cfg.z3m_gate_init,
        z3m_gate_mode=cfg.z3m_gate_mode,
        input_gate_init=cfg.input_gate_init,
        input_gates=cfg.input_gate_l1 > 0,
        branch_gate_init=cfg.branch_gate_init,
        branch_gates=cfg.branch_gate_l1 > 0,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    frames = []
    for split, df in [("Train", df_fit), ("Validation", df_val), ("TemporalTest", df_te)]:
        tmp = df.copy()
        tmp["Split"] = split
        frames.append(tmp)
    feature_df = pd.concat(frames, ignore_index=True)
    feature_df["Patient_ID"] = feature_df["Patient_ID"].astype(str)
    return model, {"Train": fit_ds, "Validation": val_ds, "TemporalTest": te_ds}, cfg, feature_df


def manual_forward_knockout(
    model: direct.ThreeBranchHazardNet,
    static_x: torch.Tensor,
    local_x: torch.Tensor,
    global_x: torch.Tensor,
    z3m_x: torch.Tensor,
    knockout: str | None,
) -> torch.Tensor:
    branches = []
    if model.use_input_gates:
        if model.use_static_branch:
            static_x = static_x * torch.sigmoid(model.static_input_gate_logit)
        local_x = local_x * torch.sigmoid(model.local_input_gate_logit)
        global_x = global_x * torch.sigmoid(model.global_input_gate_logit)
        if model.use_z3m_branch:
            z3m_x = z3m_x * torch.sigmoid(model.z3m_input_gate_logit)
    if model.use_static_branch:
        h = model.static_branch(static_x)
        if model.use_branch_gates:
            h = h * torch.sigmoid(model.static_branch_gate_logit)
        if knockout == "static":
            h = torch.zeros_like(h)
        branches.append(h)
    h = model.local_branch(local_x)
    if model.use_branch_gates:
        h = h * torch.sigmoid(model.local_branch_gate_logit)
    if knockout == "local":
        h = torch.zeros_like(h)
    branches.append(h)
    h = model.global_branch(global_x)
    if model.use_branch_gates:
        h = h * torch.sigmoid(model.global_branch_gate_logit)
    if knockout == "global":
        h = torch.zeros_like(h)
    branches.append(h)
    if model.use_z3m_branch:
        h = model.z3m_branch(z3m_x)
        if model.z3m_gate_mode == "scalar":
            h = h * model.z3m_gate
        elif model.z3m_gate_mode == "time_sigmoid":
            time_signal = z3m_x[:, -1:].clamp(-5.0, 5.0)
            h = h * torch.sigmoid(model.z3m_time_gate(time_signal))
        if model.use_branch_gates:
            h = h * torch.sigmoid(model.z3m_branch_gate_logit)
        if knockout == "z3M":
            h = torch.zeros_like(h)
        branches.append(h)
    hidden = model.fuse(torch.cat(branches, dim=1))
    return torch.sigmoid(model.hazard_head(hidden).squeeze(1))


def collect_knockout_predictions(model: direct.ThreeBranchHazardNet, dataset: direct.DirectDataset, branch: str | None) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    preds = []
    with torch.no_grad():
        for batch in loader:
            preds.append(
                manual_forward_knockout(
                    model,
                    batch["static"],
                    batch["local"],
                    batch["global"],
                    batch["z3m"],
                    branch,
                )
                .cpu()
                .numpy()
            )
    return np.concatenate(preds)


def branch_knockout(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, direct.TrainConfig]:
    model, datasets, cfg, feature_df = rebuild_model_and_datasets()
    rows = []
    pred_rows = []
    for split, ds in datasets.items():
        y = ds.y.numpy().astype(int)
        full_p = collect_knockout_predictions(model, ds, None)
        for branch in ["full", "static", "local", "global", "z3M"]:
            p = full_p if branch == "full" else collect_knockout_predictions(model, ds, branch)
            m = binary_metrics(y, p, THRESHOLD)
            m["Split"] = split
            m["Variant"] = branch
            rows.append(m)
            pred_rows.append(pd.DataFrame({"Split": split, "Variant": branch, "Y_Relapse": y, "Prob": p}))
    metrics = pd.DataFrame(rows)
    full = metrics[metrics["Variant"].eq("full")][["Split", "AUC", "PR_AUC", "Brier"]].rename(
        columns={"AUC": "Full_AUC", "PR_AUC": "Full_PR_AUC", "Brier": "Full_Brier"}
    )
    metrics = metrics.merge(full, on="Split", how="left")
    metrics["Delta_AUC_vs_Full"] = metrics["AUC"] - metrics["Full_AUC"]
    metrics["Delta_PR_AUC_vs_Full"] = metrics["PR_AUC"] - metrics["Full_PR_AUC"]
    metrics["Delta_Brier_vs_Full"] = metrics["Brier"] - metrics["Full_Brier"]
    metrics.to_csv(TAB / "branch_knockout_importance.csv", index=False)
    pd.concat(pred_rows, ignore_index=True).to_csv(TAB / "branch_knockout_predictions.csv", index=False)

    keys = ["Patient_ID", "Source_Row", "Interval_ID", "Interval_Name", "Start_Time", "Stop_Time", "Y_Relapse", "Split"]
    full_recomputed = []
    for split, ds in datasets.items():
        full_recomputed.append(collect_knockout_predictions(model, ds, None))
    recomputed = pred.copy()
    recomputed["RecomputedProb"] = np.concatenate(full_recomputed)
    recomputed["AbsDiff_FromSaved"] = (recomputed["DirectProb"] - recomputed["RecomputedProb"]).abs()
    recomputed[keys + ["DirectProb", "RecomputedProb", "AbsDiff_FromSaved"]].to_csv(TAB / "checkpoint_prediction_recompute_check.csv", index=False)
    return metrics, feature_df, recomputed, cfg


def training_branch_ablation() -> pd.DataFrame:
    """Read matched training-time branch removal runs."""
    rows = []
    missing = []
    for variant, run_dir in TRAIN_BRANCH_ABLATION_RUNS.items():
        pred_path = run_dir / "DirectThreeBranch_Predictions.csv"
        summary_path = run_dir / "DirectThreeBranch_Summary.csv"
        if not pred_path.exists() or not summary_path.exists():
            missing.append(str(run_dir))
            continue
        pred_df = pd.read_csv(pred_path)
        summary_df = pd.read_csv(summary_path)
        for split in SPLITS:
            sub = pred_df[pred_df["Split"].eq(split)]
            threshold = float(summary_df[summary_df["Split"].eq(split)]["Threshold"].iloc[0])
            metrics = binary_metrics(sub["Y_Relapse"].values, sub["DirectProb"].values, threshold)
            metrics["Split"] = split
            metrics["Variant"] = variant
            metrics["Removed_Branch"] = {
                "full": "none",
                "drop_static_train": "static",
                "drop_local_train": "local",
                "drop_global_train": "global",
                "drop_z3M_train": "z3M",
            }[variant]
            metrics["Best_Epoch"] = int(summary_df[summary_df["Split"].eq(split)]["Best_Epoch"].iloc[0])
            rows.append(metrics)
    if missing:
        raise FileNotFoundError("Missing training-time branch ablation runs: " + "; ".join(missing))
    out = pd.DataFrame(rows)
    full = out[out["Variant"].eq("full")][["Split", "AUC", "PR_AUC", "Brier"]].rename(
        columns={"AUC": "Full_AUC", "PR_AUC": "Full_PR_AUC", "Brier": "Full_Brier"}
    )
    out = out.merge(full, on="Split", how="left")
    out["Delta_AUC_vs_Full"] = out["AUC"] - out["Full_AUC"]
    out["Delta_PR_AUC_vs_Full"] = out["PR_AUC"] - out["Full_PR_AUC"]
    out["Delta_Brier_vs_Full"] = out["Brier"] - out["Full_Brier"]
    out["Ablation_Type"] = "training_time_branch_removal"
    out.to_csv(TAB / "branch_training_ablation_importance.csv", index=False)
    out.to_csv(TAB / "branch_knockout_importance.csv", index=False)
    return out


def copy_source_files() -> None:
    return None


def fig_task_model_schema(summary: pd.DataFrame) -> None:
    row = summary.iloc[0]
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.axis("off")
    boxes = [
        (0.04, 0.70, 0.19, 0.18, "原始纵向随访数据\nbaseline + 0/1/3/6/12/18/24M"),
        (0.29, 0.70, 0.18, 0.18, "3M early-response\n固定锚点 encoder\n12维 z3M 表征"),
        (0.55, 0.70, 0.18, 0.18, "time-sigmoid gate\nStart_Time < 3M 时屏蔽"),
        (0.80, 0.70, 0.16, 0.18, "单一输出\nP(relapse t→t+1)"),
        (0.08, 0.36, 0.17, 0.16, "Static branch\n基线背景 / RAI负荷"),
        (0.31, 0.36, 0.17, 0.16, "Local branch\n当前状态 / 最近变化"),
        (0.54, 0.36, 0.17, 0.16, "Global branch\n0..t 纵向摘要"),
        (0.77, 0.36, 0.17, 0.16, "z3M branch\n3M反应嵌入 + 时间特征"),
        (0.30, 0.09, 0.40, 0.14, "Concat 4 × 48 → LayerNorm/ReLU/Dropout → single hazard head"),
    ]
    for x, y, w, h, text in boxes:
        rect = plt.Rectangle((x, y), w, h, ec="#1F2A44", fc="#F4F7FB", lw=1.4)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color="#1F2A44")
    arrows = [
        ((0.23, 0.79), (0.29, 0.79)),
        ((0.47, 0.79), (0.55, 0.79)),
        ((0.73, 0.79), (0.80, 0.79)),
        ((0.165, 0.36), (0.39, 0.23)),
        ((0.395, 0.36), (0.45, 0.23)),
        ((0.625, 0.36), (0.55, 0.23)),
        ((0.855, 0.36), (0.61, 0.23)),
        ((0.50, 0.70), (0.855, 0.52)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.4, color="#46536B"))
    ax.text(
        0.5,
        0.96,
        f"T5_gate005: Val PR-AUC {row['PR_AUC']:.3f} at best epoch {int(row['Best_Epoch'])}, "
        "temporal test only reported after checkpoint restore",
        ha="center",
        fontsize=11,
        color="#1F2A44",
    )
    save_fig(fig, "Figure_01_Task_Model_Schema.png")


def fig_cohort_flow(core: pd.DataFrame, feature_df: pd.DataFrame) -> None:
    patient_counts = feature_df.groupby("Split")["Patient_ID"].nunique().reindex(SPLITS)
    rows = core.set_index("Split").reindex(SPLITS)
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    ax.axis("off")
    ax.text(0.5, 0.94, "队列切分与 at-risk 动态区间", ha="center", fontsize=14, weight="bold")
    items = [
        (0.05, 0.62, 0.22, 0.20, "全队列\n889 patients\n按患者顺序 temporal split"),
        (0.37, 0.62, 0.22, 0.20, "开发期\n前80% patients\n内部再切 fit / validation"),
        (0.70, 0.62, 0.22, 0.20, "Temporal test\n后20% patients\n不参与选择/阈值/校准"),
    ]
    for x, y, w, h, text in items:
        ax.add_patch(plt.Rectangle((x, y), w, h, fc="#F4F7FB", ec="#1F2A44", lw=1.3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center")
    ax.annotate("", xy=(0.37, 0.72), xytext=(0.27, 0.72), arrowprops=dict(arrowstyle="->", lw=1.4))
    ax.annotate("", xy=(0.70, 0.72), xytext=(0.59, 0.72), arrowprops=dict(arrowstyle="->", lw=1.4))
    table_text = []
    for split in SPLITS:
        table_text.append(
            f"{SPLIT_LABEL[split]}: at-risk patients {int(patient_counts.loc[split])}, "
            f"interval rows {int(rows.loc[split, 'N'])}, events {int(rows.loc[split, 'Events'])}, "
            f"prevalence {rows.loc[split, 'Prevalence']:.1%}"
        )
    ax.text(0.07, 0.36, "\n".join(table_text), fontsize=11, va="top", bbox=dict(boxstyle="round,pad=0.6", fc="white", ec="#CBD3DF"))
    ax.text(
        0.61,
        0.37,
        "At-risk interval 定义：当前 landmark 之前可得信息 → 下一时间窗是否 Normal→Hyper 复发。\n"
        "同一患者可贡献多个 interval，因此不确定性使用 Patient_ID cluster bootstrap。",
        fontsize=10,
        va="top",
        bbox=dict(boxstyle="round,pad=0.6", fc="white", ec="#CBD3DF"),
    )
    save_fig(fig, "Figure_02_Cohort_Landmark_Flow.png")


def fig_metric_bars(core: pd.DataFrame) -> None:
    metrics = ["AUC", "PR_AUC", "F1", "Recall", "Specificity", "PR_AUC_Lift"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    for ax, metric in zip(axes.ravel(), metrics):
        vals = core.set_index("Split").loc[SPLITS, metric]
        ax.bar(range(len(SPLITS)), vals, color=[COLORS[s] for s in SPLITS])
        ax.set_xticks(range(len(SPLITS)))
        ax.set_xticklabels([SPLIT_LABEL[s] for s in SPLITS], rotation=20)
        ax.set_title(metric)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
        if metric != "PR_AUC_Lift":
            ax.set_ylim(0, min(1.0, max(vals) + 0.18))
    save_fig(fig, "Figure_03_Model_Metric_Bars.png")


def fig_pr_roc(pred: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for split in SPLITS:
        sub = pred[pred["Split"].eq(split)]
        y, p = sub["Y_Relapse"].values, sub["DirectProb"].values
        precision, recall, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax.plot(recall, precision, lw=2, color=COLORS[split], label=f"{SPLIT_LABEL[split]} AP={ap:.3f}")
        ax.axhline(y.mean(), color=COLORS[split], ls="--", lw=1, alpha=0.45)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves with split-specific prevalence baselines")
    ax.legend(loc="best")
    save_fig(fig, "Figure_04_PR_Curves.png")

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for split in SPLITS:
        sub = pred[pred["Split"].eq(split)]
        y, p = sub["Y_Relapse"].values, sub["DirectProb"].values
        fpr, tpr, _ = roc_curve(y, p)
        auc_v = roc_auc_score(y, p)
        ax.plot(fpr, tpr, lw=2, color=COLORS[split], label=f"{SPLIT_LABEL[split]} AUC={auc_v:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color="#9AA4B2", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves")
    ax.legend(loc="lower right")
    save_fig(fig, "Figure_05_ROC_Curves.png")


def fig_calibration(calib: pd.DataFrame, core: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    ax.plot([0, 1], [0, 1], ls="--", color="#9AA4B2", label="Perfect calibration")
    for split in SPLITS:
        sub = calib[calib["Split"].eq(split)]
        ax.plot(
            sub["Mean_Predicted_Risk"],
            sub["Observed_Risk"],
            marker="o",
            lw=2,
            color=COLORS[split],
            label=f"{SPLIT_LABEL[split]} Brier={core.set_index('Split').loc[split, 'Brier']:.3f}",
        )
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed event rate")
    ax.set_title("Calibration by risk decile")
    ax.legend(loc="best")
    save_fig(fig, "Figure_06_Calibration_Curves.png")


def fig_confusion(cm: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    for ax, split in zip(axes, SPLITS):
        row = cm[cm["Split"].eq(split)].iloc[0]
        mat = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]], dtype=float)
        ax.imshow(mat, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(mat[i, j])}", ha="center", va="center", fontsize=13, weight="bold")
        ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], ["True 0", "True 1"])
        ax.set_title(f"{SPLIT_LABEL[split]}\nthreshold={THRESHOLD:.3f}")
        ax.grid(False)
    save_fig(fig, "Figure_07_Confusion_Matrices.png")


def fig_threshold_sensitivity(ts: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    metrics = ["F1", "Recall", "Specificity", "PPV"]
    for ax, split in zip(axes, ["Validation", "TemporalTest"]):
        sub = ts[ts["Split"].eq(split)]
        for metric in metrics:
            ax.plot(sub["Threshold"], sub[metric], lw=1.8, label=metric)
        ax.axvline(THRESHOLD, color="#1F2A44", ls="--", lw=1.2, label="locked 0.785")
        ax.set_title(SPLIT_LABEL[split])
        ax.set_xlabel("Threshold")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Metric")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    save_fig(fig, "Figure_08_Threshold_Sensitivity.png")


def fig_topk(topk: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for split in SPLITS:
        sub = topk[topk["Split"].eq(split)]
        axes[0].plot(sub["Top_Fraction"] * 100, sub["Event_Capture"], marker="o", lw=2, label=SPLIT_LABEL[split], color=COLORS[split])
        axes[1].plot(sub["Top_Fraction"] * 100, sub["Lift"], marker="o", lw=2, label=SPLIT_LABEL[split], color=COLORS[split])
    axes[0].set_title("Event capture among top-risk intervals")
    axes[0].set_xlabel("Top-risk interval fraction (%)")
    axes[0].set_ylabel("Captured event fraction")
    axes[1].set_title("Precision lift over prevalence")
    axes[1].set_xlabel("Top-risk interval fraction (%)")
    axes[1].set_ylabel("Lift")
    axes[1].legend(loc="best")
    save_fig(fig, "Figure_09_TopK_Event_Capture.png")


def fig_dca(dca: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.plot(dca["Threshold_Probability"], dca["Model_Net_Benefit"], lw=2.2, color=COLORS["TemporalTest"], label="Model")
    ax.plot(dca["Threshold_Probability"], dca["Treat_All_Net_Benefit"], lw=1.8, color=COLORS["gray"], ls="--", label="Treat all")
    ax.plot(dca["Threshold_Probability"], dca["Treat_None_Net_Benefit"], lw=1.8, color="#000000", ls=":", label="Treat none")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Temporal test decision curve, threshold range 0.05-0.40")
    ax.legend(loc="best")
    save_fig(fig, "Figure_10_Decision_Curve.png")


def fig_training_history(history: pd.DataFrame, summary: pd.DataFrame) -> None:
    best_epoch = int(summary.iloc[0]["Best_Epoch"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    axes[0].plot(history["epoch"], history["train_loss"], color=COLORS["gray"], lw=2, label="train loss")
    axes[0].plot(history["epoch"], history["rank_loss"], color=COLORS["accent"], lw=1.4, label="rank loss")
    axes[0].plot(history["epoch"], history["net_benefit_loss"], color=COLORS["red"], lw=1.4, label="net-benefit loss")
    axes[0].axvline(best_epoch, color="#1F2A44", ls="--", lw=1)
    axes[0].set_title("Loss terms")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(loc="best")
    axes[1].plot(history["epoch"], history["fit_prauc"], color=COLORS["Train"], lw=2, label="Train PR-AUC")
    axes[1].plot(history["epoch"], history["val_prauc"], color=COLORS["Validation"], lw=2, label="Val PR-AUC")
    axes[1].plot(history["epoch"], history["selected_score"], color="#1F2A44", lw=1.8, label="selected score")
    axes[1].axvline(best_epoch, color="#1F2A44", ls="--", lw=1)
    axes[1].set_title("Selection curves")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(loc="best")
    save_fig(fig, "Figure_12_Training_History.png")


def fig_error_casebook(group_df: pd.DataFrame) -> None:
    test = group_df[group_df["Split"].eq("TemporalTest")].copy()
    order = ["TP", "FP", "FN", "TN"]
    test["Error_Group"] = pd.Categorical(test["Error_Group"], order, ordered=True)
    test = test.sort_values("Error_Group")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].bar(test["Error_Group"].astype(str), test["N"], color=[COLORS["TemporalTest"], COLORS["red"], COLORS["accent"], COLORS["gray"]])
    axes[0].set_title("Temporal test error groups")
    axes[0].set_ylabel("Intervals")
    axes[1].bar(test["Error_Group"].astype(str), test["Mean_Probability"], color=[COLORS["TemporalTest"], COLORS["red"], COLORS["accent"], COLORS["gray"]])
    axes[1].axhline(THRESHOLD, color="#1F2A44", ls="--", lw=1, label="threshold")
    axes[1].set_title("Mean predicted risk by group")
    axes[1].set_ylabel("Mean probability")
    axes[1].legend(loc="best")
    save_fig(fig, "Figure_13_Error_Casebook.png")


def fig_patient_warning(patient_top: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for split in SPLITS:
        sub = patient_top[patient_top["Split"].eq(split)]
        axes[0].plot(sub["Top_Fraction"] * 100, sub["Patient_Event_Capture"], marker="o", lw=2, label=SPLIT_LABEL[split], color=COLORS[split])
        axes[1].plot(sub["Top_Fraction"] * 100, sub["Precision"], marker="o", lw=2, label=SPLIT_LABEL[split], color=COLORS[split])
    axes[0].set_title("Patient-level max-risk event capture")
    axes[0].set_xlabel("Top-risk patients (%)")
    axes[0].set_ylabel("Captured event-patient fraction")
    axes[1].set_title("Patient-level warning precision")
    axes[1].set_xlabel("Top-risk patients (%)")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="best")
    save_fig(fig, "Figure_14_Patient_MaxRisk_Warning.png")


def fig_branch_knockout(branch_metrics: pd.DataFrame) -> None:
    sub = branch_metrics[branch_metrics["Split"].eq("TemporalTest") & ~branch_metrics["Variant"].eq("full")].copy()
    sub["PR_AUC_Drop"] = -sub["Delta_PR_AUC_vs_Full"]
    sub["AUC_Drop"] = -sub["Delta_AUC_vs_Full"]
    order = ["static", "local", "global", "z3M"]
    sub["Removed_Branch"] = pd.Categorical(sub["Removed_Branch"], order, ordered=True)
    sub = sub.sort_values("Removed_Branch")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    axes[0].bar(sub["Removed_Branch"].astype(str), sub["PR_AUC_Drop"], color=COLORS["accent"])
    axes[0].axhline(0.0, color="#1F2A44", lw=0.8)
    axes[0].set_title("PR-AUC drop after training-time branch removal")
    axes[0].set_ylabel("Full - branch-removed retrain")
    axes[1].bar(sub["Removed_Branch"].astype(str), sub["AUC_Drop"], color=COLORS["TemporalTest"])
    axes[1].axhline(0.0, color="#1F2A44", lw=0.8)
    axes[1].set_title("AUC drop after training-time branch removal")
    axes[1].set_ylabel("Full - branch-removed retrain")
    save_fig(fig, "Figure_15_Branch_Knockout_Importance.png")


def write_model_config(summary: pd.DataFrame) -> pd.DataFrame:
    row = summary.iloc[0]
    config_cols = [
        c
        for c in summary.columns
        if c
        not in {
            "Split",
            "N",
            "Events",
            "Prevalence",
            "AUC",
            "PR_AUC",
            "F1",
            "Recall",
            "Specificity",
            "RunName",
            "Seed",
        }
    ]
    cfg = pd.DataFrame({"Parameter": config_cols, "Value": [row[c] for c in config_cols]})
    cfg.to_csv(TAB / "model_config.csv", index=False)
    return cfg


def read_landmark_context() -> dict[str, str]:
    ctx: dict[str, str] = {}
    base_dir = ROOT / "results" / "landmark_focus" / "tables"
    path = base_dir / "3m_lr_lgbm_15_metrics.csv"
    if path.exists():
        df = pd.read_csv(path)
        best = df[df["Model"].astype(str).str.contains("LGBM top25", na=False)]
        if not best.empty:
            r = best.iloc[0]
            ctx["3m_lgbm"] = f"3M 二分类 top25 LGBM：Test Acc {r['Accuracy']:.3f}, AUC {r['ROC_AUC']:.3f}, PR-AUC {r['PR_AUC']:.3f}"
    path = base_dir / "binary_temporal_model_summary.csv"
    if path.exists():
        df = pd.read_csv(path)
        ctx["binary_summary"] = md_table(df.head(8), digits=3)
    return ctx


def write_readme(
    summary: pd.DataFrame,
    core: pd.DataFrame,
    ci: pd.DataFrame,
    branch_metrics: pd.DataFrame,
    patient_metrics: pd.DataFrame,
    cfg_table: pd.DataFrame,
    z3m_checks: pd.DataFrame,
    feature_dict: pd.DataFrame,
    landmark_ctx: dict[str, str],
) -> None:
    row_test = core[core["Split"].eq("TemporalTest")].iloc[0]
    row_val = core[core["Split"].eq("Validation")].iloc[0]
    row_train = core[core["Split"].eq("Train")].iloc[0]
    ci_focus = ci[(ci["Split"].eq("TemporalTest")) & (ci["Metric"].isin(["AUC", "PR_AUC", "F1", "Brier"]))].copy()
    branch_test = branch_metrics[branch_metrics["Split"].eq("TemporalTest")][
        [
            "Variant",
            "Removed_Branch",
            "Best_Epoch",
            "Threshold",
            "AUC",
            "PR_AUC",
            "Brier",
            "Accuracy",
            "F1",
            "Recall",
            "Specificity",
            "Delta_AUC_vs_Full",
            "Delta_PR_AUC_vs_Full",
            "Delta_Brier_vs_Full",
        ]
    ]
    patient_short = patient_metrics[["Split", "N", "Events", "AUC", "PR_AUC", "Accuracy", "Recall", "Specificity", "PPV", "F1", "Brier"]]
    z3m_short = z3m_checks.copy()
    feature_short = feature_dict.groupby("Branch").size().reset_index(name="Feature_Count")
    lines = [
        "# 3M 表征增强的四分支动态复发预警模型",
        "",
        "## 摘要",
        "",
        "本报告聚焦一个完全独立于旧 teacher / frozen-fuse 流水线的动态复发模型：在每个随访 landmark `t`，仅使用 `t` 及之前可得信息，预测下一时间窗 `t -> t+1` 内是否复发。当前主模型为 `T5_gate005`，结构为 **static / local / global / z3M 四分支 + 单一 hazard head**。",
        "",
        f"在固定阈值 `{THRESHOLD:.3f}` 下，temporal test 达到 AUC `{row_test['AUC']:.3f}`、PR-AUC `{row_test['PR_AUC']:.3f}`、F1 `{row_test['F1']:.3f}`、Recall `{row_test['Recall']:.3f}`、Specificity `{row_test['Specificity']:.3f}`。该阈值锁定自开发流程中的 train-fit F1 选择，并补充 validation/test threshold sensitivity；temporal test 只用于最终报告。",
        "",
        "### 研究亮点",
        "",
        "- **四分支单头动态风险建模**：static、local、global 与 z3M early-response branch 分别学习背景、当前决策态、累积病程和 3M 早期反应，再由单一 hazard head 输出下一时间窗复发风险。",
        "- **3M 表征不是硬拼特征**：z3M 作为独立分支进入模型，并由 `time_sigmoid` gate 调节；`Start_Time < 3M` 时 z3M 嵌入和 mask 均为 0，避免未来锚点信息提前进入。",
        "- **面向概率校准、稀有事件排序和临床预警效用的训练目标**：主损失保留 patient-weighted BCE 以维持概率风险估计；hard-negative ranking 与 DCA-inspired soft net-benefit 仅作为小权重辅助正则，弱对齐低复发率场景下的高风险排序和临床报警阈值。",
        "- **患者级 cluster bootstrap 性能不确定性估计**：bootstrap CI 按 `Patient_ID` 聚类抽样，而不是把同一患者的多个 interval 当作完全独立样本。",
        "- **interval-level 与 patient-level 同时汇报**：既报告每个动态窗口的复发风险，也报告 patient-level max-risk warning，方便转化为随访预警策略。",
        "- **模型解释、阈值敏感性和误差分析闭环**：报告包含 calibration、DCA、threshold sensitivity、top-k capture、错题本和 matched training-time branch removal。",
        "",
        "## 队列与任务定义",
        "",
        "动态任务定义为：给定患者截至 landmark `t` 的全部可得信息，预测 `t -> t+1` 区间是否出现 Normal→Hyper 复发。由于同一患者可贡献多个 at-risk interval，训练和评估均保留患者级分组意识。",
        "",
        "![队列流程](figures/Figure_02_Cohort_Landmark_Flow.png)",
        "",
        "## 3M early-response encoder 背景",
        "",
        "3M 固定锚点二分类模型提供早期治疗反应证据，但本报告不把 fixed-landmark 结果作为动态模型的最终输出。它只提供一个 z3M 表征，帮助动态模型在 3M 及之后利用早期反应轨道。",
        "",
        f"- {landmark_ctx.get('3m_lgbm', '3M fixed-landmark summary not found.')}",
        "",
        "## 四分支模型结构",
        "",
        "![模型结构](figures/Figure_01_Task_Model_Schema.png)",
        "",
        "模型配置要点：",
        "",
        md_table(cfg_table.head(18), digits=3),
        "",
        "特征数量概览：",
        "",
        md_table(feature_short, digits=0),
        "",
        "## 训练目标：概率校准、稀有事件排序与临床预警效用",
        "",
        "我们设计了一个面向区间级复发预警的临床对齐复合目标函数。该目标保留患者加权二元交叉熵以维持风险概率估计，加入难负样本排序项以改善低复发率场景下的稀有事件排序，并加入小权重、受决策曲线启发的软净获益项，使训练目标弱对齐临床预警阈值。",
        "",
        "`L = L_patient-weighted BCE + lambda_r L_hard-rank + lambda_nb L_soft net benefit`",
        "",
        md_table(
            pd.DataFrame(
                [
                    {
                        "组件": "patient-weighted BCE",
                        "工程含义": "按患者加权的二分类交叉熵",
                        "医学解释": "避免随访更密集的患者在 interval-level 数据中被重复放大，使每位患者对概率学习贡献近似均衡。",
                    },
                    {
                        "组件": "hard-negative ranking",
                        "工程含义": "推动阳性 interval 分数高于高分阴性 interval",
                        "医学解释": "在低复发率下优先优化高风险区排序，减少真正复发窗口被高风险假阳性淹没。",
                    },
                    {
                        "组件": "soft net-benefit",
                        "工程含义": "可导的 DCA-inspired 辅助项",
                        "医学解释": "把临床报警阈值附近的假阳性/假阴性权衡弱写入训练目标；DCA 本身仍作为最终临床效用评价。",
                    },
                ]
            ),
            digits=3,
        ),
        "",
        "训练目标按照任务的临床结构设计。由于长格式数据中同一患者可贡献多个 at-risk interval，我们使用患者加权 BCE，使每位患者对概率估计目标的贡献近似均衡。考虑到区间级复发率较低，我们加入难负样本排序项，惩罚复发 interval 的分数低于高分非复发 interval，从而改善临床高风险区的排序能力。最后，受 DCA 启发，我们在预设报警阈值附近加入小权重软净获益项，使训练目标弱对齐临床预警效用。",
        "",
        "该复合目标不应被理解为替代标准概率风险建模。相反，BCE 被保留以维持概率校准，排序项和软净获益项仅作为小权重辅助正则，分别面向稀有事件预警和临床决策效用。这一设计与动态风险建模中结合概率目标和排序目标的范式一致，也符合临床预测模型中强调校准和决策曲线效用的评价方向。",
        "",
        "## 主结果",
        "",
        "主结果全部从 prediction CSV 重新计算，summary CSV 只作为核对来源。表名使用 `core_binary_metrics`，不硬称 15 项，因为实际包含更多临床性能字段。",
        "",
        md_table(core, digits=3),
        "",
        "Temporal test 关键指标的 Patient_ID cluster bootstrap 95% CI：",
        "",
        md_table(ci_focus, digits=3),
        "",
        "![指标条形图](figures/Figure_03_Model_Metric_Bars.png)",
        "",
        "## PR / ROC / Calibration / DCA",
        "",
        "PR 曲线中每个 split 都使用自己的 prevalence baseline；DCA 仅画临床更相关的 0.05–0.40 阈值范围。",
        "",
        "![PR曲线](figures/Figure_04_PR_Curves.png)",
        "",
        "![ROC曲线](figures/Figure_05_ROC_Curves.png)",
        "",
        "![校准曲线](figures/Figure_06_Calibration_Curves.png)",
        "",
        "![DCA](figures/Figure_10_Decision_Curve.png)",
        "",
        "## 阈值、混淆矩阵与预警能力",
        "",
        f"主阈值 `{THRESHOLD:.3f}` 锁定自开发流程中的 train-fit F1 选择。Temporal test 不参与阈值选择；报告额外给出 validation/test threshold sensitivity，方便观察阈值移动时 sensitivity、specificity、PPV 和 F1 的变化。",
        "",
        "![混淆矩阵](figures/Figure_07_Confusion_Matrices.png)",
        "",
        "![阈值敏感性](figures/Figure_08_Threshold_Sensitivity.png)",
        "",
        "![Top-K捕获](figures/Figure_09_TopK_Event_Capture.png)",
        "",
        "Patient-level max-risk warning：每个患者取所有 interval 里的最大预测风险。",
        "",
        md_table(patient_short, digits=3),
        "",
        "![患者级预警](figures/Figure_14_Patient_MaxRisk_Warning.png)",
        "",
        "## 训练时分支消融与错题本",
        "",
        "分支消融采用 matched training-time branch removal：每个变体都使用同一损失函数和同一选择规则重新训练，只移除一个分支。数值为移除分支重训后相对 full model 的变化，负值代表性能下降。",
        "",
        md_table(branch_test, digits=3),
        "",
        "![分支消融](figures/Figure_15_Branch_Knockout_Importance.png)",
        "",
        "错题本按 TP / FP / FN / TN 复盘。若原始动态特征可完整合并，则输出特征画像；否则仅输出概率和病例轨迹。本次 `feature_merge_check.csv` 记录合并完整性。",
        "",
        "![错题本](figures/Figure_13_Error_Casebook.png)",
        "",
        "![训练历史](figures/Figure_12_Training_History.png)",
        "",
        "## 泄漏控制检查",
        "",
        "- 阈值、模型选择、校准拟合不使用 temporal test。",
        "- `MixedTestWeight = 0` 且 `DisableTestSelection = True`。",
        "- `Start_Time < 3M` 的 interval 中 z3M 嵌入、availability 和 time feature 均应为 0。",
        "- train z3M 使用 OOF / train-only 产物；validation 使用 fit-context；test 使用最终 fit+val encoder 产物。",
        "",
        md_table(z3m_short, digits=3),
        "",
        "## 局限性",
        "",
        "- 动态 interval 同一患者内相关，因此所有 CI 使用 Patient_ID cluster bootstrap，但样本量仍限制了小亚组稳定性。",
        "- z3M branch 当前使用离线 3M 表征；真正 end-to-end pretrain-then-finetune 仍是下一版结构优化方向。",
        "- Temporal test 是严格时间外推，但仍属于单中心回顾性验证，外部验证可进一步增强证据等级。",
        "",
        "## 附录：文件索引",
        "",
        "- `tables/core_binary_metrics.csv`：核心二分类指标。",
        "- `tables/cluster_bootstrap_ci_by_patient.csv`：患者级 cluster bootstrap CI。",
        "- `tables/threshold_sensitivity_validation_selected.csv`：阈值敏感性。",
        "- `tables/topk_interval_capture.csv`：interval-level top-k 预警。",
        "- `tables/patient_level_maxrisk_warning.csv`：patient-level max-risk 预警。",
        "- `tables/decision_curve_005_040.csv`：DCA 曲线。",
        "- `tables/error_casebook.csv`：错题本。",
        "- `tables/branch_knockout_importance.csv`：训练时单分支移除消融。",
        "- `tables/branch_training_ablation_importance.csv`：同上，保留更明确的文件名。",
        "- `tables/feature_dictionary_zh.csv`：特征名中文含义。",
        "",
    ]
    (OUT / "复发.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    set_style()
    summary, pred, history = load_artifacts()
    copy_source_files()
    core, cm = compute_core_tables(pred)
    ts = threshold_sensitivity(pred)
    topk = topk_capture(pred)
    patient_metrics, patient_top = patient_maxrisk(pred)
    dca = dca_curve(pred)
    calib = calibration_bins(pred)
    ci = cluster_bootstrap_ci(pred)
    branch_metrics = training_branch_ablation()
    _, _, cfg, feature_df_from_model = rebuild_model_and_datasets()
    merged, feature_df = merge_feature_frame(pred)
    if len(feature_df_from_model) == len(feature_df):
        feature_df = feature_df_from_model
    error_profile = error_casebook(merged)
    feature_dict = build_feature_dictionary(feature_df)
    z3m_checks = z3m_leakage_checks(feature_df, cfg)
    cfg_table = write_model_config(summary)
    landmark_ctx = read_landmark_context()

    fig_task_model_schema(summary[summary["Split"].eq("Validation")].reset_index(drop=True))
    fig_cohort_flow(core, feature_df)
    fig_metric_bars(core)
    fig_pr_roc(pred)
    fig_calibration(calib, core)
    fig_confusion(cm)
    fig_threshold_sensitivity(ts)
    fig_topk(topk)
    fig_dca(dca)
    fig_training_history(history, summary)
    fig_error_casebook(error_profile)
    fig_patient_warning(patient_top)
    fig_branch_knockout(branch_metrics)

    write_readme(summary, core, ci, branch_metrics, patient_metrics, cfg_table, z3m_checks, feature_dict, landmark_ctx)
    print(f"Saved paper report to {OUT}")
    print(f"Report: {OUT / '复发.md'}")
    print(f"Figures: {FIG}")
    print(f"Tables: {TAB}")
    print("Figure 15 uses matched training-time branch removal ablations.")


if __name__ == "__main__":
    main()
