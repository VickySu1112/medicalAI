"""Add README chapter-4-style figures to the dynamic relapse report.

This script is intentionally report-local. It reads the selected four-branch
dynamic relapse predictions and the existing report tables, creates patient-
level clinical reading figures mirroring the main repository README section 4,
and inserts an idempotent Markdown block into results/t5_dynamic_paper/复发.md.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts import relapse_threehead_landmark as base
from utils.config import STATIC_NAMES


REPORT = ROOT / "results" / "t5_dynamic_paper"
FIG = REPORT / "figures"
TAB = REPORT / "tables"
PRED = ROOT / "results" / "t5_seed2025_paper" / "DirectThreeBranch_Predictions.csv"
HISTORY = ROOT / "results" / "t5_seed2025_paper" / "DirectThreeBranch_History.csv"
SUMMARY = ROOT / "results" / "t5_seed2025_paper" / "DirectThreeBranch_Summary.csv"
ERROR_CASEBOOK = TAB / "error_casebook.csv"
README = REPORT / "复发.md"
HTML = REPORT / "复发.html"
THRESHOLD = 0.785
SPLITS = ["Train", "Validation", "TemporalTest"]
SPLIT_LABEL = {"Train": "Train", "Validation": "Validation", "TemporalTest": "Temporal test"}
COLORS = {
    "Train": "#3B6FB6",
    "Validation": "#D98C2B",
    "TemporalTest": "#4B9B6E",
    "event": "#C44E52",
    "nonevent": "#4C78A8",
    "accent": "#765AA6",
    "gray": "#68717D",
}
RISK_CMAP = plt.cm.YlOrRd
RISK_NORM = plt.Normalize(vmin=0.0, vmax=1.0)
RISK_TICKS = np.linspace(0.0, 1.0, 6)
MISSING_COLOR = "#E5E7EB"
START = "<!-- README4_RELAPSE_FIGURES_START -->"
END = "<!-- README4_RELAPSE_FIGURES_END -->"
TOPK_START = "<!-- TOPK_CI_EXPLANATION_START -->"
TOPK_END = "<!-- TOPK_CI_EXPLANATION_END -->"
PATIENT_TOPK_START = "<!-- PATIENT_TOPK_CAPTURE_START -->"
PATIENT_TOPK_END = "<!-- PATIENT_TOPK_CAPTURE_END -->"
HISTORY_START = "<!-- TRAIN_HISTORY_EXPLANATION_START -->"
HISTORY_END = "<!-- TRAIN_HISTORY_EXPLANATION_END -->"
TOPK_FRACS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
AGG_METHODS = [
    ("mean_risk", "Mean_Interval_Risk", "Mean risk"),
]
AGG_LABEL = {key: label for key, _col, label in AGG_METHODS}
AGG_COLUMN = {key: col for key, col, _label in AGG_METHODS}
PRIMARY_AGG = "mean_risk"
ALERT_AGG = PRIMARY_AGG
PRIMARY_RISK_COL = AGG_COLUMN[PRIMARY_AGG]
ALERT_RISK_COL = AGG_COLUMN[ALERT_AGG]


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "Heiti TC", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 140,
            "savefig.dpi": 260,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, name: str, *, tight: bool = True) -> None:
    if tight:
        fig.tight_layout()
    fig.savefig(FIG / name, dpi=260, bbox_inches="tight")
    plt.close(fig)


def wilson_ci(events: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = events / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    return max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom)


def load_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = pd.read_csv(PRED)
    pred["Patient_ID"] = pred["Patient_ID"].astype(str)
    err = pd.read_csv(ERROR_CASEBOOK)
    err["Patient_ID"] = err["Patient_ID"].astype(str)
    return pred, err


def load_history() -> pd.DataFrame:
    hist = pd.read_csv(HISTORY)
    hist = hist.drop(columns=[c for c in hist.columns if hist[c].isna().all()])
    hist.to_csv(TAB / "training_history_curve.csv", index=False)
    return hist


def load_summary() -> pd.DataFrame:
    return pd.read_csv(SUMMARY)


def safe_auc(y: np.ndarray, p: np.ndarray, metric: str) -> float:
    if len(np.unique(y.astype(int))) < 2:
        return float("nan")
    if metric == "roc":
        return float(roc_auc_score(y, p))
    if metric == "pr":
        return float(average_precision_score(y, p))
    raise ValueError(metric)


def binary_at_threshold(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = p >= threshold
    yb = y.astype(bool)
    tp = int((pred & yb).sum())
    fp = int((pred & ~yb).sum())
    tn = int((~pred & ~yb).sum())
    fn = int((~pred & yb).sum())
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    f1 = 2 * ppv * recall / (ppv + recall) if np.isfinite(ppv) and np.isfinite(recall) and (ppv + recall) else 0.0
    return {
        "Threshold": float(threshold),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Accuracy": float((tp + tn) / len(y)) if len(y) else float("nan"),
        "Recall": float(recall),
        "Specificity": float(specificity),
        "PPV": float(ppv),
        "NPV": float(npv),
        "F1": float(f1),
        "Balanced_Accuracy": float(np.nanmean([recall, specificity])),
    }


def select_validation_threshold(patient: pd.DataFrame, risk_col: str) -> float:
    val = patient[patient["Split"].eq("Validation")].copy()
    if val.empty:
        return THRESHOLD
    y = val["Patient_Event"].astype(int).to_numpy()
    p = val[risk_col].astype(float).clip(0.0, 1.0).to_numpy()
    thresholds = np.unique(np.r_[0.0, p, 1.0])
    best_key = (float("-inf"), float("-inf"), float("-inf"))
    best_threshold = THRESHOLD
    for threshold in thresholds:
        m = binary_at_threshold(y, p, float(threshold))
        key = (m["F1"], m["Balanced_Accuracy"], -abs(float(threshold) - THRESHOLD))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def topk_metrics(frame: pd.DataFrame, fracs: list[float] = TOPK_FRACS) -> pd.DataFrame:
    """Compute Top-K capture metrics for an already filtered evaluation frame."""
    sub = frame.sort_values("DirectProb", ascending=False).reset_index(drop=True)
    total_events = float(sub["Y_Relapse"].sum())
    prevalence = float(sub["Y_Relapse"].mean())
    rows = []
    for frac in fracs:
        n_alert = max(1, int(math.ceil(len(sub) * frac)))
        selected = sub.head(n_alert)
        captured = float(selected["Y_Relapse"].sum())
        ppv = captured / n_alert
        rows.append(
            {
                "Top_Fraction": frac,
                "Top_K_Percent": frac * 100.0,
                "Alerted_N": int(n_alert),
                "Captured_Events": int(captured),
                "Total_Events": int(total_events),
                "Recall_at_K": captured / total_events if total_events > 0 else float("nan"),
                "PPV_at_K": ppv,
                "Lift_at_K": ppv / prevalence if prevalence > 0 else float("nan"),
                "Prevalence": prevalence,
                "N": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def cluster_bootstrap_topk(frame: pd.DataFrame, n_boot: int = 1000, seed: int = 20260508) -> pd.DataFrame:
    """Patient-level cluster bootstrap CI for Top-K metrics."""
    point = topk_metrics(frame)
    rng = np.random.default_rng(seed)
    grouped = {pid: part for pid, part in frame.groupby("Patient_ID", sort=False)}
    pids = np.array(list(grouped.keys()))
    boot_rows = []
    for b in range(n_boot):
        sample_pids = rng.choice(pids, size=len(pids), replace=True)
        sample = pd.concat([grouped[pid] for pid in sample_pids], ignore_index=True)
        metrics = topk_metrics(sample)
        metrics["Bootstrap_ID"] = b
        boot_rows.append(metrics)
    boot = pd.concat(boot_rows, ignore_index=True)
    rows = []
    for _, row in point.iterrows():
        frac = float(row["Top_Fraction"])
        part = boot[boot["Top_Fraction"].eq(frac)]
        out = row.to_dict()
        for metric in ["Recall_at_K", "PPV_at_K", "Lift_at_K"]:
            vals = part[metric].astype(float).to_numpy()
            out[f"{metric}_CI_Low"] = float(np.nanpercentile(vals, 2.5))
            out[f"{metric}_CI_High"] = float(np.nanpercentile(vals, 97.5))
        out["Bootstrap_Unit"] = "Patient_ID"
        out["Bootstrap_N"] = n_boot
        rows.append(out)
    return pd.DataFrame(rows)


def topk_ci_tables(pred: pd.DataFrame, patient: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    interval = pred[pred["Split"].eq("TemporalTest")].copy()
    interval_ci = cluster_bootstrap_topk(interval)
    interval_ci.to_csv(TAB / "topk_interval_capture_ci.csv", index=False)

    patient_frame = (
        patient[patient["Split"].eq("TemporalTest")]
        .rename(columns={"Patient_Event": "Y_Relapse", ALERT_RISK_COL: "DirectProb"})
        [["Patient_ID", "Y_Relapse", "DirectProb"]]
        .copy()
    )
    patient_ci = cluster_bootstrap_topk(patient_frame)
    patient_ci.to_csv(TAB / "topk_patient_capture_ci.csv", index=False)

    agg_rows = []
    for method, risk_col, label in AGG_METHODS:
        frame = (
            patient[patient["Split"].eq("TemporalTest")]
            .rename(columns={"Patient_Event": "Y_Relapse", risk_col: "DirectProb"})
            [["Patient_ID", "Y_Relapse", "DirectProb"]]
            .copy()
        )
        ci = cluster_bootstrap_topk(frame)
        ci.insert(0, "Aggregation_Method", method)
        ci.insert(1, "Aggregation_Label", label)
        agg_rows.append(ci)
    all_patient_ci = pd.concat(agg_rows, ignore_index=True)
    all_patient_ci.to_csv(TAB / "patient_level_topk_by_aggregation_ci.csv", index=False)
    return interval_ci, patient_ci, all_patient_ci


def patient_static_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    first = df.sort_values(["Patient_ID", "Start_Time", "Stop_Time"]).groupby("Patient_ID", as_index=False).first()
    target = df.groupby("Patient_ID", as_index=False)["Y_Relapse"].max().rename(columns={"Y_Relapse": "Patient_Event"})
    out = first[["Patient_ID"] + features].merge(target, on="Patient_ID", how="inner")
    out["Patient_ID"] = out["Patient_ID"].astype(str)
    return out


def patient_topk_capture_curve(patient: pd.DataFrame) -> pd.DataFrame:
    """Patient-level top-k capture curve with a train-only static baseline."""
    df_tr, df_te, *_ = base.build_longitudinal_tables()
    static_features = [name for name in STATIC_NAMES if name in df_tr.columns and name in df_te.columns]
    train_static = patient_static_frame(df_tr, static_features)
    test_static = patient_static_frame(df_te, static_features)
    pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=0.5,
                    penalty="l2",
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=20260508,
                    max_iter=1000,
                ),
            ),
        ]
    )
    pipe.fit(train_static[static_features], train_static["Patient_Event"].astype(int))
    test_static["StaticOnlyRisk"] = pipe.predict_proba(test_static[static_features])[:, 1]
    test_static[["Patient_ID", "Patient_Event", "StaticOnlyRisk"]].to_csv(TAB / "patient_static_only_baseline_predictions.csv", index=False)

    model = patient[patient["Split"].eq("TemporalTest")][["Patient_ID", "Patient_Event", PRIMARY_RISK_COL]].copy()
    model["Patient_ID"] = model["Patient_ID"].astype(str)
    merged = model.merge(test_static[["Patient_ID", "StaticOnlyRisk"]], on="Patient_ID", how="inner")
    merged = merged.rename(columns={PRIMARY_RISK_COL: "FourBranchMeanRisk", "Patient_Event": "Y"})
    n = len(merged)
    total_events = max(1, int(merged["Y"].sum()))
    rows = []
    for k in range(5, 101, 5):
        top_n = max(1, int(np.ceil(n * k / 100.0)))
        top_model = merged.sort_values("FourBranchMeanRisk", ascending=False).head(top_n)
        top_static = merged.sort_values("StaticOnlyRisk", ascending=False).head(top_n)
        rows.append(
            {
                "TopK_Percent": k,
                "Alerted_Patients": top_n,
                "Total_Event_Patients": total_events,
                "FourBranch_Captured": int(top_model["Y"].sum()),
                "StaticOnly_Captured": int(top_static["Y"].sum()),
                "FourBranch_Capture_Rate": float(top_model["Y"].sum() / total_events),
                "StaticOnly_Capture_Rate": float(top_static["Y"].sum() / total_events),
                "Random_Baseline": float(k / 100.0),
            }
        )
    curve = pd.DataFrame(rows)
    curve.to_csv(TAB / "topk_patient_capture_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.plot(curve["TopK_Percent"], curve["FourBranch_Capture_Rate"], marker="o", color="#111827", label="4branch mean risk")
    ax.plot(curve["TopK_Percent"], curve["StaticOnly_Capture_Rate"], marker="s", color="#2563EB", label="Static-only baseline")
    ax.plot(curve["TopK_Percent"], curve["Random_Baseline"], ls="--", color="#9CA3AF", label="Random baseline")
    ax.set_xlabel("Top-k highest-risk patients (%)")
    ax.set_ylabel("Captured relapse patients (%)")
    ax.set_title("Patient-level Top-k Capture Curve (Temporal Test)")
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.set_yticklabels([f"{int(v * 100)}%" for v in np.linspace(0, 1.0, 6)])
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    save(fig, "Figure_09B_Patient_TopK_Capture_Curve.png")
    return curve


def patient_interval_frame(pred: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated treatment rows to one patient-window row for patient-level figures."""
    keys = ["Split", "Patient_ID", "Interval_ID", "Interval_Name", "Start_Time", "Stop_Time"]
    out = (
        pred.groupby(keys, as_index=False, sort=False)
        .agg(
            Y_Relapse=("Y_Relapse", "max"),
            DirectProb=("DirectProb", "max"),
            DirectPred=("DirectPred", "max"),
            N_Source_Rows=("Source_Row", "nunique"),
        )
        .sort_values(["Split", "Patient_ID", "Start_Time", "Stop_Time"])
        .reset_index(drop=True)
    )
    out.to_csv(TAB / "readme4_patient_interval_predictions.csv", index=False)
    return out


def patient_aggregation_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, pid), part in pred.sort_values(["Patient_ID", "Start_Time", "Stop_Time"]).groupby(["Split", "Patient_ID"], sort=False):
        probs = part["DirectProb"].astype(float).clip(0.0, 1.0).to_numpy()
        max_risk = float(np.max(probs))
        rows.append(
            {
                "Split": split,
                "Patient_ID": pid,
                "N_Intervals": int(len(part)),
                "Patient_Event": int(part["Y_Relapse"].max()),
                "Max_Interval_Risk": max_risk,
                "Mean_Interval_Risk": float(np.mean(probs)),
                "Ever_Alert": int(float(np.mean(probs)) >= THRESHOLD),
                "First_Start_Time": float(part["Start_Time"].min()),
                "Last_Stop_Time": float(part["Stop_Time"].max()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "readme4_patient_aggregated_risk.csv", index=False)
    return out


def q1q4_summary(patient: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        split_df = patient[patient["Split"].eq(split)].copy()
        for method, risk_col, label in AGG_METHODS:
            if split_df.empty:
                continue
            work = split_df.copy()
            work["Risk_Quartile"] = pd.qcut(work[risk_col], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
            for q, part in work.groupby("Risk_Quartile", observed=False):
                n = int(len(part))
                events = int(part["Patient_Event"].sum())
                low, high = wilson_ci(events, n)
                rows.append(
                    {
                        "Split": split,
                        "Aggregation_Method": method,
                        "Aggregation_Label": label,
                        "Risk_Column": risk_col,
                        "Risk_Quartile": str(q),
                        "N": n,
                        "Events": events,
                        "Observed_Event_Rate": events / n if n else float("nan"),
                        "Mean_Aggregated_Risk": float(part[risk_col].mean()),
                        "Mean_N_Intervals": float(part["N_Intervals"].mean()),
                        "Wilson_CI_Low": low,
                        "Wilson_CI_High": high,
                        "Risk_Min": float(part[risk_col].min()),
                        "Risk_Max": float(part[risk_col].max()),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "patient_level_quartile_by_aggregation.csv", index=False)
    out[out["Aggregation_Method"].eq(PRIMARY_AGG) & out["Split"].eq("TemporalTest")].to_csv(
        TAB / "readme4_patient_q1q4_aggregated.csv", index=False
    )
    return out


def aggregation_benchmark(patient: pd.DataFrame, qsum: pd.DataFrame) -> pd.DataFrame:
    threshold_by_method = {method: select_validation_threshold(patient, risk_col) for method, risk_col, _ in AGG_METHODS}
    rows = []
    for method, risk_col, label in AGG_METHODS:
        threshold = threshold_by_method[method]
        q_method = qsum[qsum["Aggregation_Method"].eq(method)]
        for split in SPLITS:
            sub = patient[patient["Split"].eq(split)].copy()
            if sub.empty:
                continue
            y = sub["Patient_Event"].astype(int).to_numpy()
            p = sub[risk_col].astype(float).clip(0.0, 1.0).to_numpy()
            q_rates = (
                q_method[q_method["Split"].eq(split)]
                .sort_values("Risk_Quartile")["Observed_Event_Rate"]
                .astype(float)
                .to_numpy()
            )
            monotonic_violations = int(np.sum(np.diff(q_rates) < -1e-12)) if len(q_rates) > 1 else 0
            spearman = pd.Series(p).corr(pd.Series(y), method="spearman")
            n_corr = pd.Series(p).corr(sub["N_Intervals"].astype(float), method="spearman")
            row = {
                "Split": split,
                "Aggregation_Method": method,
                "Aggregation_Label": label,
                "Risk_Column": risk_col,
                "N": int(len(sub)),
                "Events": int(y.sum()),
                "Prevalence": float(y.mean()) if len(y) else float("nan"),
                "AUC": safe_auc(y, p, "roc"),
                "PR_AUC": safe_auc(y, p, "pr"),
                "Brier": float(brier_score_loss(y, p)) if len(np.unique(y)) > 1 else float("nan"),
                "Spearman_Event_R": float(spearman) if pd.notna(spearman) else float("nan"),
                "Spearman_NIntervals_R": float(n_corr) if pd.notna(n_corr) else float("nan"),
                "Q1_Q4_Monotonic_Violations": monotonic_violations,
                "Threshold_Selected_From": "Validation",
            }
            row.update(binary_at_threshold(y, p, threshold))
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "patient_level_aggregation_benchmark.csv", index=False)
    return out


def fig_discrimination(pred: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for split in SPLITS:
        sub = pred[pred["Split"].eq(split)]
        y = sub["Y_Relapse"].to_numpy()
        p = sub["DirectProb"].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        axes[0].plot(fpr, tpr, lw=2.2, color=COLORS[split], label=f"{SPLIT_LABEL[split]} AUC={roc_auc_score(y, p):.3f}")
        axes[1].plot(recall, precision, lw=2.2, color=COLORS[split], label=f"{SPLIT_LABEL[split]} AP={average_precision_score(y, p):.3f}")
        axes[1].axhline(float(np.mean(y)), lw=1, ls="--", color=COLORS[split], alpha=0.35)
    axes[0].plot([0, 1], [0, 1], ls="--", color="#A3AAB6", lw=1)
    axes[0].set_title("ROC discrimination")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[1].set_title("PR discrimination with split-specific baselines")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[0].legend(loc="lower right")
    axes[1].legend(loc="upper right")
    save(fig, "Figure_16_Readme4_Discrimination.png")


def fig_q1q4(qsum: pd.DataFrame) -> None:
    test = qsum[qsum["Split"].eq("TemporalTest")].copy()
    primary = test[test["Aggregation_Method"].eq(PRIMARY_AGG)].sort_values("Risk_Quartile")
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    x = np.arange(len(primary))
    y = primary["Observed_Event_Rate"].to_numpy(dtype=float)
    yerr = np.vstack([y - primary["Wilson_CI_Low"].to_numpy(dtype=float), primary["Wilson_CI_High"].to_numpy(dtype=float) - y])
    ax.bar(x, y, yerr=yerr, capsize=4, color="#9CC7E6", edgecolor="#2F5F7F", label="Observed patient event rate")
    ax.plot(x, primary["Mean_Aggregated_Risk"], marker="o", color="#111827", lw=2.2, label="Mean interval risk")
    for i, r in enumerate(primary.itertuples()):
        ax.text(i, min(0.72, y[i] + 0.035), f"{int(r.Events)}/{int(r.N)}", ha="center", fontsize=9)
    ax.set_xticks(x, primary["Risk_Quartile"])
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Patient-level event rate / risk")
    ax.set_title("Temporal test patient strata by mean interval risk")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    save(fig, "Figure_17_Readme4_Patient_Q1Q4_CI.png")


def fig_topk_ci(topk_ci: pd.DataFrame) -> None:
    plot_df = topk_ci.sort_values("Top_Fraction").copy()
    x = plot_df["Top_K_Percent"].to_numpy(dtype=float)
    recall = plot_df["Recall_at_K"].to_numpy(dtype=float)
    recall_yerr = np.vstack(
        [
            recall - plot_df["Recall_at_K_CI_Low"].to_numpy(dtype=float),
            plot_df["Recall_at_K_CI_High"].to_numpy(dtype=float) - recall,
        ]
    )
    lift = plot_df["Lift_at_K"].to_numpy(dtype=float)
    lift_yerr = np.vstack(
        [
            lift - plot_df["Lift_at_K_CI_Low"].to_numpy(dtype=float),
            plot_df["Lift_at_K_CI_High"].to_numpy(dtype=float) - lift,
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    axes[0].errorbar(
        x,
        recall,
        yerr=recall_yerr,
        marker="o",
        lw=2.2,
        capsize=4,
        color=COLORS["TemporalTest"],
        label="Four-branch model",
    )
    axes[0].plot([0, 50], [0, 0.50], ls="--", lw=1.4, color=COLORS["gray"], label="Random baseline")
    axes[0].set_xlim(0, 52)
    axes[0].set_ylim(0, min(1.02, max(0.60, float(np.nanmax(recall_yerr[1] + recall)) + 0.08)))
    axes[0].set_xlabel("Top-K alert rate (%)")
    axes[0].set_ylabel("Captured relapse intervals (Recall@K)")
    axes[0].set_title("Temporal test interval-level Top-K capture")
    axes[0].legend(loc="lower right")

    labels = [f"{int(v)}%" for v in x]
    bars = axes[1].bar(labels, lift, yerr=lift_yerr, capsize=4, color=COLORS["accent"], edgecolor="white", linewidth=0.8)
    for bar, row in zip(bars, plot_df.itertuples()):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.10,
            f"{int(row.Captured_Events)}/{int(row.Total_Events)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[1].axhline(1.0, ls="--", lw=1.2, color=COLORS["gray"], label="No enrichment")
    axes[1].set_xlabel("Top-K alert rate (%)")
    axes[1].set_ylabel("Lift over temporal-test prevalence")
    axes[1].set_title("Risk enrichment in top-risk intervals")
    axes[1].legend(loc="upper right")
    save(fig, "Figure_09_TopK_Event_Capture.png")


def fig_training_history(history: pd.DataFrame, summary: pd.DataFrame) -> None:
    best_epoch = int(summary["Best_Epoch"].dropna().iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))

    axes[0].plot(history["epoch"], history["fit_prauc"], color=COLORS["Train"], lw=2.4, label="Train PR-AUC")
    axes[0].plot(history["epoch"], history["val_prauc"], color=COLORS["Validation"], lw=2.4, label="Validation PR-AUC")
    if "selected_score" in history:
        axes[0].plot(history["epoch"], history["selected_score"], color="#1F2A44", lw=1.2, ls=":", label="Selection score")
    axes[0].axvline(best_epoch, color=COLORS["event"], ls="--", lw=1.2, label=f"Best epoch {best_epoch}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("PR-AUC / selection score")
    axes[0].set_title("Development PR-AUC during training")
    axes[0].legend(loc="lower right")

    axes[1].plot(history["epoch"], history["train_loss"], color=COLORS["gray"], lw=2.4, label="Train loss")
    if "rank_loss" in history:
        axes[1].plot(history["epoch"], history["rank_loss"], color=COLORS["accent"], lw=1.4, alpha=0.7, label="Rank loss")
    axes[1].axvline(best_epoch, color=COLORS["event"], ls="--", lw=1.2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss and learning-rate schedule")
    ax_lr = axes[1].twinx()
    ax_lr.step(history["epoch"], history["lr"], where="post", color=COLORS["TemporalTest"], lw=1.4, alpha=0.75, label="Learning rate")
    ax_lr.set_ylabel("Learning rate")
    lines, labels = axes[1].get_legend_handles_labels()
    lines_lr, labels_lr = ax_lr.get_legend_handles_labels()
    axes[1].legend(lines + lines_lr, labels + labels_lr, loc="upper right")
    save(fig, "Figure_12_Training_History.png")


def fig_waterfall(patient: pd.DataFrame) -> None:
    test = patient[patient["Split"].eq("TemporalTest")].sort_values(PRIMARY_RISK_COL, ascending=False).reset_index(drop=True)
    risks = test[PRIMARY_RISK_COL].astype(float).to_numpy()
    cmap = RISK_CMAP
    norm = RISK_NORM
    fig, ax = plt.subplots(figsize=(12.8, 5.0))
    ax.bar(np.arange(len(test)), risks, color=cmap(norm(risks)), width=0.9, edgecolor="none")
    event_idx = np.flatnonzero(test["Patient_Event"].to_numpy() == 1)
    if len(event_idx):
        ax.scatter(
            event_idx,
            np.minimum(risks[event_idx] + 0.025, 1.01),
            marker="v",
            s=28,
            color=COLORS["event"],
            edgecolor="white",
            linewidth=0.4,
            label="Observed relapse",
            zorder=4,
        )
    qs = np.quantile(np.arange(len(test)), [0.25, 0.50, 0.75])
    for q in qs:
        ax.axvline(q, color="#1F2A44", ls="--", lw=0.8, alpha=0.65)
    ax.set_xlabel("Temporal test patients sorted by mean interval risk")
    ax.set_ylabel("Mean interval relapse risk")
    ax.set_title("Patient-level aggregated risk waterfall")
    ax.set_ylim(0, min(1.02, max(0.2, float(test[PRIMARY_RISK_COL].max()) + 0.08)))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, borderaxespad=0.0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.012, fraction=0.025, ticks=RISK_TICKS)
    cbar.set_label("Mean interval risk")
    save(fig, "Figure_18_Readme4_Patient_Waterfall.png")


def fig_heatmap(pred: pd.DataFrame, patient: pd.DataFrame) -> None:
    test = pred[pred["Split"].eq("TemporalTest")].copy()
    order = test[["Interval_Name", "Start_Time", "Stop_Time"]].drop_duplicates().sort_values(["Start_Time", "Stop_Time"])
    windows = order["Interval_Name"].astype(str).tolist()
    patient_order = patient[patient["Split"].eq("TemporalTest")].sort_values(["Patient_Event", PRIMARY_RISK_COL], ascending=[False, False])
    event_pids = patient_order[patient_order["Patient_Event"].eq(1)]["Patient_ID"].tolist()
    nonevent_pids = patient_order[patient_order["Patient_Event"].eq(0)]["Patient_ID"].tolist()

    def matrix(pids: list[str]) -> np.ndarray:
        piv = test[test["Patient_ID"].isin(pids)].pivot_table(index="Patient_ID", columns="Interval_Name", values="DirectProb", aggfunc="max")
        piv = piv.reindex(index=pids, columns=windows)
        return piv.to_numpy(dtype=float)

    cmap = RISK_CMAP.copy()
    cmap.set_bad(color=MISSING_COLOR)
    fig = plt.figure(figsize=(11.8, 7.8))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 0.028],
        height_ratios=[max(1, len(event_pids)), max(1, len(nonevent_pids))],
        hspace=0.36,
        wspace=0.04,
    )
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]
    cax = fig.add_subplot(gs[:, 1])
    ims = []
    for ax, pids, title in [(axes[0], event_pids, "Observed relapse patients"), (axes[1], nonevent_pids, "No observed relapse patients")]:
        mat = matrix(pids)
        im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", cmap=cmap, norm=RISK_NORM)
        ims.append(im)
        ax.set_title(f"{title} sorted by mean interval risk")
        ax.set_ylabel("Patients")
        ax.set_yticks([])
    axes[0].tick_params(labelbottom=False)
    axes[1].set_xticks(range(len(windows)), windows, rotation=30, ha="right")
    cbar = fig.colorbar(ims[0], cax=cax, ticks=RISK_TICKS)
    cbar.set_label("Interval relapse risk")
    fig.suptitle("Temporal test patient-interval risk heatmap", y=0.965, fontsize=13, weight="bold")
    fig.subplots_adjust(left=0.08, right=0.93, top=0.88, bottom=0.16)
    save(fig, "Figure_19_Readme4_Patient_Risk_Heatmap.png", tight=False)


def choose_typical_patients(patient: pd.DataFrame) -> pd.DataFrame:
    test = patient[patient["Split"].eq("TemporalTest")].copy()
    eligible = test[test["N_Intervals"].ge(4)].copy()
    if len(eligible) < 3:
        eligible = test[test["N_Intervals"].ge(3)].copy()
    if len(eligible) < 3:
        eligible = test
    event = eligible[eligible["Patient_Event"].eq(1)].sort_values([PRIMARY_RISK_COL, "N_Intervals"], ascending=[False, False]).head(1)
    low = eligible[eligible["Patient_Event"].eq(0)].sort_values([PRIMARY_RISK_COL, "N_Intervals"], ascending=[True, False]).head(1)
    used = set(pd.concat([event, low], ignore_index=True)["Patient_ID"].astype(str).tolist())
    adjacent_pool = eligible[~eligible["Patient_ID"].astype(str).isin(used)].copy()
    if adjacent_pool.empty:
        adjacent_pool = eligible
    adjacent = adjacent_pool.iloc[(adjacent_pool[ALERT_RISK_COL] - THRESHOLD).abs().argsort()[:1]]
    out = pd.concat(
        [
            event.assign(Case_Type="High-risk relapse", Case_Label="Patient A"),
            low.assign(Case_Type="Low-risk non-relapse", Case_Label="Patient B"),
            adjacent.assign(Case_Type="Threshold-adjacent", Case_Label="Patient C"),
        ],
        ignore_index=True,
    )
    out.to_csv(TAB / "readme4_typical_patients.csv", index=False)
    return out


def fig_typical(pred: pd.DataFrame, err: pd.DataFrame, chosen: pd.DataFrame) -> None:
    keys = ["Patient_ID", "Split", "Interval_ID"]
    err_agg = (
        err.groupby(keys, as_index=False)
        .agg(FT3_Current=("FT3_Current", "mean"), FT4_Current=("FT4_Current", "mean"), logTSH_Current=("logTSH_Current", "mean"))
    )
    merged = pred.merge(
        err_agg,
        on=keys,
        how="left",
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharey=True)
    for ax, (_, row) in zip(axes, chosen.iterrows()):
        part = merged[(merged["Split"].eq("TemporalTest")) & (merged["Patient_ID"].eq(row["Patient_ID"]))].sort_values(["Start_Time", "Stop_Time"])
        x = np.arange(len(part))
        labels = part["Interval_Name"].astype(str).tolist()
        ax.plot(x, part["DirectProb"], marker="o", lw=2.4, color="#1F2A44", label="Risk")
        ax.axhline(THRESHOLD, color=COLORS["event"], ls="--", lw=1.2, label="threshold")
        for i, y in enumerate(part["Y_Relapse"].astype(int).tolist()):
            if y == 1:
                ax.axvspan(i - 0.45, i + 0.45, color=COLORS["event"], alpha=0.12)
        lab_cols = ["FT3_Current", "FT4_Current", "logTSH_Current"]
        if all(c in part for c in lab_cols):
            lab = part[lab_cols].astype(float)
            lab = (lab - lab.mean()) / lab.std(ddof=0).replace(0, np.nan)
            for col, color in zip(lab_cols, ["#4B9B6E", "#D98C2B", "#765AA6"]):
                ax.plot(x, 0.15 + 0.08 * lab[col].fillna(0), lw=1.2, alpha=0.65, color=color, label=col if ax is axes[0] else None)
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_ylim(0, 1.02)
        ax.set_title(
            f"{row['Case_Label']}: {row['Case_Type']}\n"
            f"mean interval risk={row[PRIMARY_RISK_COL]:.2f}"
        )
        ax.set_ylabel("Predicted relapse risk")
    axes[0].legend(loc="lower left", fontsize=7)
    fig.suptitle("Typical patient longitudinal risk updates", y=1.03, fontsize=13, weight="bold")
    save(fig, "Figure_20_Readme4_Typical_Patients.png")


def update_readme(qsum: pd.DataFrame, bench: pd.DataFrame) -> None:
    primary_q = qsum[(qsum["Split"].eq("TemporalTest")) & (qsum["Aggregation_Method"].eq(PRIMARY_AGG))].sort_values("Risk_Quartile")
    q_brief = ", ".join([f"{r.Risk_Quartile}: {int(r.Events)}/{int(r.N)}" for r in primary_q.itertuples()])
    test_bench = bench[bench["Split"].eq("TemporalTest")].set_index("Aggregation_Method")
    mean_pr = float(test_bench.loc[PRIMARY_AGG, "PR_AUC"])
    mean_corr = float(test_bench.loc[PRIMARY_AGG, "Spearman_NIntervals_R"])
    block = f"""{START}
## 对齐仓库 README 第四章的复发模型补充图

这一节按仓库主 `README.md` 第四章的读图顺序，给当前四分支动态复发模型补齐对应图组。上文已经分别展示了校准、DCA、阈值敏感性和混淆矩阵；这里把它们和患者级风险分层、瀑布图、热图、典型病例放在同一个“主要结果图组”里，方便直接迁移到论文主文。Patient-level 报告统一采用 **mean interval risk** 表示患者随访期间的平均复发风险负担；旧的 product cumulative risk 因容易受 interval 数量影响，仅作为内部诊断，不作为主图排序或分层口径。

### 主要结果图组索引

| 对齐主 README 第四章 | 当前复发报告对应图 |
| --- | --- |
| 4.2 最终模型判别能力 | Figure 16：Train / Validation / Temporal test ROC 与 PR 曲线 |
| 4.3 校准情况 | Figure 06：risk-decile reliability curve + Brier |
| 4.4 决策曲线分析 | Figure 10：temporal test DCA, 0.05-0.40 阈值范围 |
| 4.5 阈值敏感性 | Figure 08：threshold vs F1 / Recall / Specificity / PPV |
| 4.6 混淆矩阵 | Figure 07：固定阈值 `{THRESHOLD:.3f}` 下三 split 混淆矩阵 |
| 4.7 患者级风险分层与 CI | Figure 17：temporal test mean-risk Q1-Q4 风险梯度 |
| 4.8 患者级平均风险瀑布图 | Figure 18：temporal test mean-risk waterfall |
| 4.9 患者级风险热图 | Figure 19：patient-interval risk heatmap |
| 4.10 典型病例纵向风险更新 | Figure 20：三类典型患者风险轨迹 |

![第4章风格判别能力图](figures/Figure_16_Readme4_Discrimination.png)

Figure 16 把 ROC 与 PR 放在同一张判别能力图里。PR 面板为每个 split 单独画 prevalence baseline，避免把 temporal test 的低事件率背景和训练/验证混在一起解释。

![患者级四分位分层](figures/Figure_17_Readme4_Patient_Q1Q4_CI.png)

Figure 17 将 temporal test 患者按 `mean(p_t)` 分为 Q1-Q4。柱为观察到的患者级复发率，误差条为 Wilson 95% CI，黑线为组内平均 interval 风险。这个口径直接描述患者在可观察随访窗口中的平均 next-window 风险负担。当前 mean-risk 四分位事件数为：{q_brief}。

![患者级平均风险瀑布图](figures/Figure_18_Readme4_Patient_Waterfall.png)

Figure 18 按患者级 mean interval risk 从高到低排序。柱体颜色表示平均 interval 风险强度，红色倒三角标记最终观察到复发的患者；这个口径回答的是“患者整体处于高风险状态的负担有多重”。

![患者级风险热图](figures/Figure_19_Readme4_Patient_Risk_Heatmap.png)

Figure 19 将 temporal test 的 patient-interval 风险矩阵化显示，并升级为带边际条图的热图：主矩阵中列是随访窗口、行是患者、颜色是该 interval 的复发风险；顶部边际条同时显示各窗口的平均预测风险和真实复发率；右侧边际条显示同一患者的 mean interval risk。复发患者和未复发患者分面展示，并在各自面板内按 mean risk 排序。

![风险分箱联合热图](figures/Figure_29_Joint_Risk_Window_Marginal_Heatmap.png)

Figure 29 是标准的 joint heatmap with marginal bar plots。横轴为 temporal test interval 的预测复发风险分箱，纵轴为随访窗口；主体热图颜色表示该风险分箱与窗口交叉格内的真实复发率，顶部边际柱/线显示各风险分箱的 interval 数和复发事件数，右侧边际柱显示各窗口的 interval 数。它回答的是：高风险分箱是否真的富集复发，以及这种富集是否集中在某些随访窗口。

![典型病例纵向风险更新](figures/Figure_20_Readme4_Typical_Patients.png)

Figure 20 选择三个典型病例：高 mean-risk 复发、低 mean-risk 未复发、阈值邻近患者。黑线为每个 interval 的复发风险，虚线为固定阈值；浅红背景标记观察到复发的窗口。底部淡色线给出当前 FT3 / FT4 / logTSH 的标准化轨迹，只作临床背景参照。Temporal test patient-level mean-risk PR-AUC 为 `{mean_pr:.3f}`，mean risk 与患者 interval 数的 Spearman 相关为 `{mean_corr:.3f}`。
{END}"""
    text = README.read_text(encoding="utf-8")
    if START in text and END in text:
        before = text.split(START, 1)[0].rstrip()
        after = text.split(END, 1)[1].lstrip()
        new_text = before + "\n\n" + block + "\n\n" + after
    else:
        anchor = "## 训练时分支消融与错题本"
        if anchor not in text:
            anchor = "## 泄漏控制检查"
        new_text = text.replace(anchor, block + "\n\n" + anchor)
    README.write_text(new_text, encoding="utf-8")


def table_topk_summary(topk_ci: pd.DataFrame) -> str:
    focus = topk_ci[topk_ci["Top_Fraction"].isin([0.10, 0.20, 0.30])].copy()
    lines = [
        "| Top-K | Alerted intervals | Captured positives | Recall@K (95% CI) | PPV@K (95% CI) | Lift@K (95% CI) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in focus.itertuples():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"Top {row.Top_K_Percent:.0f}%",
                    f"{int(row.Alerted_N)}",
                    f"{int(row.Captured_Events)}/{int(row.Total_Events)}",
                    f"{row.Recall_at_K:.1%} ({row.Recall_at_K_CI_Low:.1%}-{row.Recall_at_K_CI_High:.1%})",
                    f"{row.PPV_at_K:.1%} ({row.PPV_at_K_CI_Low:.1%}-{row.PPV_at_K_CI_High:.1%})",
                    f"{row.Lift_at_K:.2f}x ({row.Lift_at_K_CI_Low:.2f}-{row.Lift_at_K_CI_High:.2f})",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def insert_or_replace_after(text: str, anchor: str, start: str, end: str, block: str) -> str:
    if start in text and end in text:
        before = text.split(start, 1)[0].rstrip()
        after = text.split(end, 1)[1].lstrip()
        return before + "\n\n" + block + "\n\n" + after
    if anchor not in text:
        return text.rstrip() + "\n\n" + block + "\n"
    return text.replace(anchor, anchor + "\n\n" + block, 1)


def update_topk_history_readme(topk_ci: pd.DataFrame, patient_topk: pd.DataFrame, history: pd.DataFrame, summary: pd.DataFrame) -> None:
    top20 = topk_ci[topk_ci["Top_Fraction"].eq(0.20)].iloc[0]
    best_epoch = int(summary["Best_Epoch"].dropna().iloc[0])
    best_hist = history[history["epoch"].eq(best_epoch)]
    if not best_hist.empty:
        best_train = float(best_hist["fit_prauc"].iloc[0])
        best_val = float(best_hist["val_prauc"].iloc[0])
        best_lr = float(best_hist["lr"].iloc[0])
    else:
        best_train = float("nan")
        best_val = float("nan")
        best_lr = float("nan")
    topk_block = f"""{TOPK_START}
Figure 09 现在只展示 **temporal test interval-level Top-K capture**，对应主任务 `predict relapse in the next interval`。横轴不是固定概率阈值，而是“如果只预警预测风险最高的前 K% interval”；Panel A 显示这些 interval 捕获了多少真实复发窗口，Panel B 显示高风险区阳性率相对总体 temporal-test prevalence 的富集倍数。误差线均为按 `Patient_ID` 聚类抽样的 bootstrap 95% CI。

在 temporal test 中，Top 20% 最高风险 interval 捕获 `{int(top20.Captured_Events)}/{int(top20.Total_Events)}` 个复发 interval，Recall@20% = `{top20.Recall_at_K:.1%}`，PPV@20% = `{top20.PPV_at_K:.1%}`，Lift@20% = `{top20.Lift_at_K:.2f}x`。换句话说，在固定 20% 预警预算下，高风险区复发率相对于总体 prevalence 有明显富集。

{table_topk_summary(topk_ci)}
{TOPK_END}"""
    top20p = patient_topk[patient_topk["TopK_Percent"].eq(20)].iloc[0]
    patient_topk_block = f"""{PATIENT_TOPK_START}
![患者级 Top-K 捕获曲线](figures/Figure_09B_Patient_TopK_Capture_Curve.png)

Figure 09B 将 Top-K capture 改到 patient level：每位患者取所有 interval 预测风险的平均值，再按患者 mean interval risk 从高到低排序。黑线是当前四分支模型，蓝线是只使用治疗前静态基线变量训练的 static-only logistic baseline，灰色虚线是随机选择基线。这个图回答的是临床分诊问题：如果只能重点随访平均风险负担最高的前 K% 患者，能够覆盖多少最终复发患者。

在 temporal test 中，Top 20% 最高风险患者覆盖 `{int(top20p.FourBranch_Captured)}/{int(top20p.Total_Event_Patients)}` 个最终复发患者；static-only baseline 在同一预算下覆盖 `{int(top20p.StaticOnly_Captured)}/{int(top20p.Total_Event_Patients)}` 个。该图是 patient-level 补充，不替代 Figure 09 的 interval-level 主任务口径。
{PATIENT_TOPK_END}"""
    history_block = f"""{HISTORY_START}
Figure 12 用于审计训练收敛与模型选择过程。左图只画 development 侧的 Train / Validation PR-AUC 与 selected score，不画 temporal test；右图画训练损失、ranking loss 和学习率调度。最终报告的 best epoch 为 `{best_epoch}`，该 epoch 的 Train PR-AUC = `{best_train:.3f}`、Validation PR-AUC = `{best_val:.3f}`、learning rate = `{best_lr:.6f}`。Temporal test 仍只在 checkpoint restore 后用于最终报告，不参与训练期选择。
{HISTORY_END}"""
    text = README.read_text(encoding="utf-8")
    text = insert_or_replace_after(text, "![Top-K捕获](figures/Figure_09_TopK_Event_Capture.png)", TOPK_START, TOPK_END, topk_block)
    text = insert_or_replace_after(text, "![患者级预警](figures/Figure_14_Patient_MaxRisk_Warning.png)", PATIENT_TOPK_START, PATIENT_TOPK_END, patient_topk_block)
    text = insert_or_replace_after(text, "![训练历史](figures/Figure_12_Training_History.png)", HISTORY_START, HISTORY_END, history_block)
    additions = [
        ("- `tables/topk_interval_capture_ci.csv`：temporal-test interval-level Top-K capture 及 Patient_ID cluster bootstrap 95% CI。", "- `tables/topk_interval_capture.csv`：interval-level top-k 预警。"),
        ("- `tables/topk_patient_capture_ci.csv`：patient-level mean-risk Top-K capture 及 Patient_ID cluster bootstrap 95% CI。", "- `tables/patient_level_maxrisk_warning.csv`：patient-level max-risk 预警。"),
        ("- `tables/readme4_patient_aggregated_risk.csv`：patient-level mean interval risk 分层来源表。", "- `tables/feature_dictionary_zh.csv`：特征名中文含义。"),
        ("- `tables/readme4_patient_q1q4_aggregated.csv`：patient-level mean-risk Q1-Q4 事件率。", "- `tables/feature_dictionary_zh.csv`：特征名中文含义。"),
        ("- `tables/training_history_curve.csv`：训练期 loss、Train/Validation PR-AUC、selected score 和学习率曲线原始表。", "- `tables/feature_dictionary_zh.csv`：特征名中文含义。"),
    ]
    default_anchor = "- `tables/feature_dictionary_zh.csv`：特征名中文含义。"
    for line, anchor in additions:
        if line not in text and anchor in text:
            text = text.replace(anchor, anchor + "\n" + line, 1)
    README.write_text(text, encoding="utf-8")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    set_style()
    pred, err = load_predictions()
    history = load_history()
    summary = load_summary()
    patient_pred = patient_interval_frame(pred)
    patient = patient_aggregation_table(patient_pred)
    qsum = q1q4_summary(patient)
    bench = aggregation_benchmark(patient, qsum)
    topk_interval_ci, _, _ = topk_ci_tables(pred, patient)
    patient_topk = patient_topk_capture_curve(patient)
    chosen = choose_typical_patients(patient)

    fig_discrimination(pred)
    fig_topk_ci(topk_interval_ci)
    fig_training_history(history, summary)
    fig_q1q4(qsum)
    fig_waterfall(patient)
    fig_heatmap(patient_pred, patient)
    fig_typical(patient_pred, err, chosen)
    update_readme(qsum, bench)
    update_topk_history_readme(topk_interval_ci, patient_topk, history, summary)

    print(f"Updated {README}")
    print(f"New figures written to {FIG}")
    print("Added README4_RELAPSE_FIGURES block.")


if __name__ == "__main__":
    main()
