#!/usr/bin/env python3
"""Add temporal-window PR-AUC comparison artifacts to the dynamic relapse report.

This script is intentionally report-only: it reads already generated prediction
CSVs, computes locked temporal-test PR-AUC by relapse interval, writes tables and
one figure under results/t5_dynamic_paper, and inserts a short section into the
existing Chinese relapse report without regenerating the rest of the report.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "results" / "t5_dynamic_paper"
FIG_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"

T5_PREDICTIONS = ROOT / "results" / "t5_seed2025_paper" / "DirectThreeBranch_Predictions.csv"
SEQ_RUNS_DIR = ROOT / "results" / "sequence_baselines" / "runs"

README_PATH = REPORT_DIR / "复发.md"
FIG_PATH = FIG_DIR / "Figure_22_Window_PR_AUC_Comparison.png"
PRIMARY_TABLE = TABLE_DIR / "window_prauc_comparison.csv"
RAW_TABLE = TABLE_DIR / "window_prauc_comparison_raw_windows.csv"
REPEAT_TABLE = TABLE_DIR / "window_prauc_comparison_repeats.csv"
CAPTURE_TABLE = TABLE_DIR / "window_threshold_capture_comparison.csv"
MSE_FOLD_TABLE = TABLE_DIR / "window_mse_fold_comparison.csv"

WINDOW_ORDER = ["1M->3M", "3M->6M", "6M->12M", "12M->24M"]
RAW_WINDOW_ORDER = ["1M->3M", "3M->6M", "6M->12M", "12M->18M", "18M->24M"]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    group: str
    color: str


MODEL_SPECS = [
    ModelSpec("FourBranch", "4branch", "Proposed", "#c84a3d"),
    ModelSpec("GRU_Composite_z3M", "GRU+z3M", "Sequence baseline", "#4c9a76"),
    ModelSpec("GRU_BCE_noZ", "GRU", "Sequence baseline", "#4c78a8"),
    ModelSpec("LSTM_BCE_noZ", "LSTM", "Sequence baseline", "#8c8c8c"),
]


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _safe_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_true.sum() <= 0:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def _window_group(name: str) -> str:
    if name in {"12M->18M", "18M->24M"}:
        return "12M->24M"
    return name


def _metrics_for(df: pd.DataFrame, window_col: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for window, g in df.groupby(window_col, sort=False):
        y = g["Y_Relapse"].to_numpy(dtype=int)
        p = g["Prob"].to_numpy(dtype=float)
        n = int(len(g))
        events = int(y.sum())
        prevalence = events / n if n else float("nan")
        ap = _safe_ap(y, p)
        rows.append(
            {
                "Window": window,
                "N": n,
                "Events": events,
                "Prevalence": prevalence,
                "AUC": _safe_auc(y, p),
                "PR_AUC": ap,
                "PR_AUC_Lift": ap / prevalence if prevalence > 0 and np.isfinite(ap) else float("nan"),
            }
        )
    return rows


def _load_t5() -> pd.DataFrame:
    if not T5_PREDICTIONS.exists():
        raise FileNotFoundError(f"Missing T5 prediction CSV: {T5_PREDICTIONS}")
    df = pd.read_csv(T5_PREDICTIONS)
    keep = df[df["Split"].eq("TemporalTest")].copy()
    keep["Model"] = "FourBranch"
    keep["Model_Label"] = "4branch"
    keep["Model_Group"] = "Proposed"
    keep["Repeat"] = "locked"
    keep["Prob"] = keep["DirectProb"].astype(float)
    keep["Threshold"] = 0.785
    keep["Window_Group"] = keep["Interval_Name"].map(_window_group)
    return keep


def _iter_sequence_prediction_paths() -> Iterable[tuple[str, str, Path]]:
    for config_dir in sorted(SEQ_RUNS_DIR.iterdir()):
        if not config_dir.is_dir():
            continue
        for repeat_dir in sorted(config_dir.iterdir()):
            pred_path = repeat_dir / "sequence_predictions.csv"
            if pred_path.exists():
                yield config_dir.name, repeat_dir.name, pred_path


def _load_sequence() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    label_map = {spec.key: spec.label for spec in MODEL_SPECS}
    for config, repeat, path in _iter_sequence_prediction_paths():
        if config not in label_map:
            continue
        df = pd.read_csv(path)
        keep = df[df["Split"].eq("TemporalTest")].copy()
        keep["Model"] = config
        keep["Model_Label"] = label_map[config]
        keep["Model_Group"] = "Sequence baseline"
        keep["Repeat"] = repeat
        keep["Prob"] = keep["SeqProb"].astype(float)
        summary_path = path.parent / "sequence_summary.csv"
        if summary_path.exists():
            summary = pd.read_csv(summary_path)
            te = summary[summary["Split"].eq("TemporalTest")]
            threshold = float(te["Threshold"].iloc[0]) if len(te) else float("nan")
        else:
            threshold = float("nan")
        keep["Threshold"] = threshold
        keep["Window_Group"] = keep["Interval_Name"].map(_window_group)
        frames.append(keep)
    if not frames:
        raise FileNotFoundError(f"No sequence prediction CSVs found under {SEQ_RUNS_DIR}")
    return pd.concat(frames, ignore_index=True)


def _per_repeat_metrics(pred: pd.DataFrame, window_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, label, group, repeat), df in pred.groupby(
        ["Model", "Model_Label", "Model_Group", "Repeat"], sort=False
    ):
        for item in _metrics_for(df, window_col):
            item.update({"Model": model, "Model_Label": label, "Model_Group": group, "Repeat": repeat})
            rows.append(item)
    return pd.DataFrame(rows)


def _aggregate(repeat_metrics: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, label, group, window), g in repeat_metrics.groupby(
        ["Model", "Model_Label", "Model_Group", "Window"], sort=False
    ):
        rows.append(
            {
                "Model": model,
                "Model_Label": label,
                "Model_Group": group,
                "Window": window,
                "N_Repeats": int(g["Repeat"].nunique()),
                "N": int(round(g["N"].mean())),
                "Events": int(round(g["Events"].mean())),
                "Prevalence": float(g["Prevalence"].mean()),
                "AUC_Mean": float(g["AUC"].mean()),
                "AUC_SD": float(g["AUC"].std(ddof=0)) if len(g) > 1 else 0.0,
                "PR_AUC_Mean": float(g["PR_AUC"].mean()),
                "PR_AUC_SD": float(g["PR_AUC"].std(ddof=0)) if len(g) > 1 else 0.0,
                "PR_AUC_Lift_Mean": float(g["PR_AUC_Lift"].mean()),
                "Status": "low_events_interpret_cautiously" if g["Events"].mean() < 5 else "ok",
            }
        )
    out = pd.DataFrame(rows)
    out["Window"] = pd.Categorical(out["Window"], categories=order, ordered=True)
    model_order = [spec.key for spec in MODEL_SPECS]
    out["Model"] = pd.Categorical(out["Model"], categories=model_order, ordered=True)
    return out.sort_values(["Window", "Model"]).reset_index(drop=True)


def _threshold_capture_for(pred: pd.DataFrame, window_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, label, group, repeat), df in pred.groupby(
        ["Model", "Model_Label", "Model_Group", "Repeat"], sort=False
    ):
        threshold = float(df["Threshold"].iloc[0])
        for window, g in df.groupby(window_col, sort=False):
            y = g["Y_Relapse"].to_numpy(dtype=int)
            p = g["Prob"].to_numpy(dtype=float)
            pred_pos = p >= threshold
            tp = int(((y == 1) & pred_pos).sum())
            fp = int(((y == 0) & pred_pos).sum())
            tn = int(((y == 0) & ~pred_pos).sum())
            fn = int(((y == 1) & ~pred_pos).sum())
            n = int(len(g))
            events = int(y.sum())
            alerts = int(pred_pos.sum())
            rows.append(
                {
                    "Model": model,
                    "Model_Label": label,
                    "Model_Group": group,
                    "Repeat": repeat,
                    "Window": window,
                    "Threshold": threshold,
                    "N": n,
                    "Events": events,
                    "Alerts": alerts,
                    "TP": tp,
                    "FP": fp,
                    "TN": tn,
                    "FN": fn,
                    "Event_Capture": tp / events if events else float("nan"),
                    "Alert_Rate": alerts / n if n else float("nan"),
                    "Precision": tp / alerts if alerts else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _aggregate_capture(capture: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, label, group, window), g in capture.groupby(
        ["Model", "Model_Label", "Model_Group", "Window"], sort=False
    ):
        rows.append(
            {
                "Model": model,
                "Model_Label": label,
                "Model_Group": group,
                "Window": window,
                "N_Repeats": int(g["Repeat"].nunique()),
                "Threshold_Mean": float(g["Threshold"].mean()),
                "N": int(round(g["N"].mean())),
                "Events": int(round(g["Events"].mean())),
                "Alerts_Mean": float(g["Alerts"].mean()),
                "TP_Mean": float(g["TP"].mean()),
                "FP_Mean": float(g["FP"].mean()),
                "TN_Mean": float(g["TN"].mean()),
                "FN_Mean": float(g["FN"].mean()),
                "Event_Capture_Mean": float(g["Event_Capture"].mean()),
                "Alert_Rate_Mean": float(g["Alert_Rate"].mean()),
                "Precision_Mean": float(g["Precision"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    out["Window"] = pd.Categorical(out["Window"], categories=WINDOW_ORDER, ordered=True)
    out["Model"] = pd.Categorical(
        out["Model"], categories=[spec.key for spec in MODEL_SPECS], ordered=True
    )
    return out.sort_values(["Window", "Model"]).reset_index(drop=True)


def _patient_fold_mse(pred: pd.DataFrame, window_col: str, n_folds: int = 5) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, label, group, repeat), df in pred.groupby(
        ["Model", "Model_Label", "Model_Group", "Repeat"], sort=False
    ):
        patient_ids = pd.Index(sorted(df["Patient_ID"].drop_duplicates()))
        fold_map = {pid: i % n_folds for i, pid in enumerate(patient_ids)}
        fold_df = df.copy()
        fold_df["Fold"] = fold_df["Patient_ID"].map(fold_map).astype(int)
        for (window, fold), g in fold_df.groupby([window_col, "Fold"], sort=False):
            y = g["Y_Relapse"].to_numpy(dtype=float)
            p = g["Prob"].to_numpy(dtype=float)
            rows.append(
                {
                    "Model": model,
                    "Model_Label": label,
                    "Model_Group": group,
                    "Repeat": repeat,
                    "Window": window,
                    "Fold": int(fold),
                    "N": int(len(g)),
                    "Events": int(y.sum()),
                    "MSE": float(np.mean((p - y) ** 2)),
                }
            )
    return pd.DataFrame(rows)


def _aggregate_mse_folds(fold_mse: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, label, group, window), g in fold_mse.groupby(
        ["Model", "Model_Label", "Model_Group", "Window"], sort=False
    ):
        rows.append(
            {
                "Model": model,
                "Model_Label": label,
                "Model_Group": group,
                "Window": window,
                "N_Fold_Estimates": int(len(g)),
                "MSE_Mean": float(g["MSE"].mean()),
                "MSE_Fold_SD": float(g["MSE"].std(ddof=0)) if len(g) > 1 else 0.0,
                "Mean_Fold_N": float(g["N"].mean()),
                "Mean_Fold_Events": float(g["Events"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    out["Window"] = pd.Categorical(out["Window"], categories=WINDOW_ORDER, ordered=True)
    out["Model"] = pd.Categorical(
        out["Model"], categories=[spec.key for spec in MODEL_SPECS], ordered=True
    )
    return out.sort_values(["Window", "Model"]).reset_index(drop=True)


def _plot(primary: pd.DataFrame, capture: pd.DataFrame, mse_fold: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.axisbelow": True,
        }
    )
    fig, (ax_mse, ax_counts) = plt.subplots(
        2,
        1,
        figsize=(12.8, 7.4),
        gridspec_kw={"height_ratios": [4.0, 2.0], "hspace": 0.22},
        sharex=True,
        constrained_layout=True,
    )
    x = np.arange(len(WINDOW_ORDER), dtype=float)
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(MODEL_SPECS))

    for offset, spec in zip(offsets, MODEL_SPECS):
        rows = mse_fold[mse_fold["Model"].astype(str).eq(spec.key)].set_index("Window")
        values = [rows.loc[w, "MSE_Mean"] if w in rows.index else np.nan for w in WINDOW_ORDER]
        errors = [rows.loc[w, "MSE_Fold_SD"] if w in rows.index else 0.0 for w in WINDOW_ORDER]
        ax_mse.bar(
            x + offset,
            values,
            width=width,
            label=spec.label,
            color=spec.color,
            edgecolor="white",
            linewidth=0.8,
            yerr=errors,
            capsize=3,
            ecolor="#222222",
        )
    ax_mse.set_title("Probability MSE by relapse window", fontsize=17, weight="bold", pad=12)
    ax_mse.set_ylabel("MSE / Brier", fontsize=12)
    upper = float(np.nanmax(mse_fold["MSE_Mean"] + mse_fold["MSE_Fold_SD"])) * 1.18
    ax_mse.set_ylim(0, max(0.18, upper))
    ax_mse.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    ax_mse.text(
        0.0,
        -0.15,
        "Whiskers are SD of per-window MSE across five patient-level cross-validation-style folds.",
        transform=ax_mse.transAxes,
        fontsize=10,
        color="#555555",
    )

    t5_support = primary[primary["Model"].astype(str).eq("FourBranch")].set_index("Window")
    seq_support = primary[primary["Model"].astype(str).eq("GRU_Composite_z3M")].set_index("Window")
    t5_capture = capture[capture["Model"].astype(str).eq("FourBranch")].set_index("Window")
    ax_counts.axis("off")
    ax_counts.set_xlim(-1.15, len(WINDOW_ORDER) - 0.5)
    ax_counts.set_ylim(0, 1.25)
    row_y = [1.08, 0.78, 0.50, 0.22]
    row_labels = ["Month window", "4branch events/N", "4branch captured/events", "Seq baselines events/N"]
    for y, label in zip(row_y, row_labels):
        ax_counts.text(-1.08, y, label, ha="left", va="center", fontsize=10.5, color="#333333")
    for i, window in enumerate(WINDOW_ORDER):
        ax_counts.text(i, row_y[0], window, ha="center", va="center", fontsize=10.5, weight="bold")
        if window in t5_support.index:
            ax_counts.text(
                i,
                row_y[1],
                f"{int(t5_support.loc[window, 'Events'])}/{int(t5_support.loc[window, 'N'])}",
                ha="center",
                va="center",
                fontsize=10.5,
            )
        if window in t5_capture.index:
            ax_counts.text(
                i,
                row_y[2],
                f"{int(round(t5_capture.loc[window, 'TP_Mean']))}/{int(t5_capture.loc[window, 'Events'])}",
                ha="center",
                va="center",
                fontsize=10.5,
                color=MODEL_SPECS[0].color,
                weight="bold",
            )
        if window in seq_support.index:
            ax_counts.text(
                i,
                row_y[3],
                f"{int(seq_support.loc[window, 'Events'])}/{int(seq_support.loc[window, 'N'])}",
                ha="center",
                va="center",
                fontsize=10.5,
                color="#555555",
            )
    fig.text(
        0.078,
        0.018,
        "events/N = true relapse intervals / evaluated intervals; captured/events = true positives at fixed 4branch threshold / true relapse intervals.",
        ha="left",
        va="bottom",
        fontsize=9.5,
        color="#666666",
    )
    fig.savefig(FIG_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = [str(row[col]) for col in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _update_readme(primary: pd.DataFrame, capture: pd.DataFrame, mse_fold: pd.DataFrame) -> None:
    if not README_PATH.exists():
        raise FileNotFoundError(f"Missing README: {README_PATH}")
    t5 = primary[primary["Model"].astype(str).eq("FourBranch")][
        ["Window", "N", "Events", "Prevalence", "PR_AUC_Mean", "PR_AUC_Lift_Mean", "Status"]
    ].copy()
    t5_capture = capture[capture["Model"].astype(str).eq("FourBranch")][
        ["Window", "Threshold_Mean", "Alerts_Mean", "TP_Mean", "Event_Capture_Mean", "Precision_Mean"]
    ].copy()
    mse_t5 = mse_fold[mse_fold["Model"].astype(str).eq("FourBranch")][
        ["Window", "MSE_Mean", "MSE_Fold_SD", "N_Fold_Estimates"]
    ].copy()
    t5 = t5.merge(t5_capture, on="Window", how="left")
    t5 = t5.merge(mse_t5, on="Window", how="left")
    t5["Events/N"] = t5.apply(lambda r: f"{int(r['Events'])}/{int(r['N'])}", axis=1)
    t5["TP/Events"] = t5.apply(lambda r: f"{int(round(r['TP_Mean']))}/{int(r['Events'])}", axis=1)
    t5["Alerts/N"] = t5.apply(lambda r: f"{int(round(r['Alerts_Mean']))}/{int(r['N'])}", axis=1)
    t5["Event_Capture_Mean"] = t5["Event_Capture_Mean"].map(lambda x: f"{x:.3f}")
    t5["Precision_Mean"] = t5["Precision_Mean"].map(lambda x: f"{x:.3f}")
    t5["MSE_Mean"] = t5["MSE_Mean"].map(lambda x: f"{x:.3f}")
    t5["MSE_Fold_SD"] = t5["MSE_Fold_SD"].map(lambda x: f"{x:.3f}")
    t5 = t5[
        [
            "Window",
            "Events/N",
            "TP/Events",
            "Alerts/N",
            "Event_Capture_Mean",
            "Precision_Mean",
            "MSE_Mean",
            "MSE_Fold_SD",
        ]
    ]
    table_md = _markdown_table(t5)
    section = f"""## 按随访窗口的复发检出能力

受复发事件数限制，本图按临床随访窗口汇总概率误差和阈值捕获；晚期 `12M->18M` 与 `18M->24M` 合并为 `12M->24M`。该分析只作为窗口级诊断补充，不参与模型选择、阈值选择或校准拟合。

![按窗口MSE与捕获数](figures/Figure_22_Window_PR_AUC_Comparison.png)

柱状图为 MSE / Brier，误差条为按 `Patient_ID` 构造 5 个交叉验证式 folds 后，各 fold 的 per-window MSE 标准差。下方 `events/N` 表示真实复发 interval 数 / 该窗口评估 interval 数；`captured/events` 表示固定阈值 `0.785` 下 4branch 正确报警的复发 interval 数 / 真实复发 interval 数。

4branch 主模型的窗口级结果如下。完整 PR-AUC 对照、阈值捕获和 MSE fold 来源分别见 `tables/window_prauc_comparison.csv`、`tables/window_threshold_capture_comparison.csv`、`tables/window_mse_fold_comparison.csv`。

{table_md}

读图时重点看两件事：MSE 越低表示概率误差越小；`TP/Events` 才是固定阈值下真正抓住了多少复发窗口。

"""
    text = README_PATH.read_text()
    heading = "## 按随访窗口的复发检出能力"
    if heading in text:
        start = text.index(heading)
        next_start = text.find("\n## ", start + len(heading))
        if next_start == -1:
            text = text[:start] + section
        else:
            text = text[:start] + section + text[next_start + 1 :]
    else:
        anchor = "## 阈值、混淆矩阵与预警能力"
        if anchor not in text:
            raise ValueError(f"README anchor not found: {anchor}")
        text = text.replace(anchor, section + anchor, 1)
    index_lines = [
        "- `tables/window_prauc_comparison.csv`：按随访窗口合并后的 temporal-test PR-AUC 对照。",
        "- `tables/window_threshold_capture_comparison.csv`：按随访窗口的固定阈值报警捕获表。",
        "- `tables/window_mse_fold_comparison.csv`：按随访窗口和患者级 folds 计算的 MSE / Brier 误差条来源。",
        "- `tables/window_prauc_comparison_raw_windows.csv`：未合并原始随访窗口 PR-AUC 审计表。",
    ]
    for line in index_lines:
        if line not in text:
            text = text.rstrip() + "\n" + line + "\n"
    README_PATH.write_text(text)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    pred = pd.concat([_load_t5(), _load_sequence()], ignore_index=True)

    merged_repeat = _per_repeat_metrics(pred, "Window_Group")
    raw_repeat = _per_repeat_metrics(pred, "Interval_Name")

    primary = _aggregate(merged_repeat, WINDOW_ORDER)
    raw = _aggregate(raw_repeat, RAW_WINDOW_ORDER)
    capture = _aggregate_capture(_threshold_capture_for(pred, "Window_Group"))
    mse_fold = _aggregate_mse_folds(_patient_fold_mse(pred, "Window_Group"))

    merged_repeat.to_csv(REPEAT_TABLE, index=False)
    primary.to_csv(PRIMARY_TABLE, index=False)
    raw.to_csv(RAW_TABLE, index=False)
    capture.to_csv(CAPTURE_TABLE, index=False)
    mse_fold.to_csv(MSE_FOLD_TABLE, index=False)
    _plot(primary, capture, mse_fold)
    _update_readme(primary, capture, mse_fold)

    print(f"Wrote {FIG_PATH}")
    print(f"Wrote {PRIMARY_TABLE}")
    print(f"Wrote {RAW_TABLE}")
    print(f"Wrote {CAPTURE_TABLE}")
    print(f"Wrote {MSE_FOLD_TABLE}")
    print(f"Updated {README_PATH}")


if __name__ == "__main__":
    main()
