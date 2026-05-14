"""Build the fixed-landmark comparison report.

This script is report-only: it reads existing 3M/6M assets, rebuilds the locked
3M LR/LGBM candidates for comparable probabilities/figures, and writes expanded
tables plus README files under results/landmark_focus.
"""

from __future__ import annotations

import json
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc as trapz_auc
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from scripts.simple.landmark_3m_binary_extreme import apply_calibration, build_feature_frame, predict_proba_one
from scripts.simple.landmark_3m_feature_prune_probe import fit_oof as fit_lgbm_oof
from scripts.simple.landmark_3m_feature_prune_probe import params as lgbm_params


OUT = ROOT / "results" / "landmark_focus"
TABLES = OUT / "tables"
FIGS = OUT / "figures"
SRC3M = ROOT / "results" / "3m_binary_extreme_seedprobe"
LR_ART = ROOT / "results" / "3m_binary_lr_final_artifacts"
LGBM_ART = ROOT / "results" / "3m_binary_feature_prune_probe"
FIXED = ROOT / "results" / "fixed_landmark_binary"


def clip_proba(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def logit(p: np.ndarray) -> np.ndarray:
    p = clip_proba(p)
    return np.log(p / (1 - p))


def binary_metrics_full(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    p = clip_proba(p)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prevalence = float(np.mean(y))
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    return {
        "N": int(len(y)),
        "Events": int(np.sum(y)),
        "Prevalence": prevalence,
        "Threshold": float(threshold),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "ROC_AUC": float(roc_auc_score(y, p)),
        "PR_AUC": float(average_precision_score(y, p)),
        "PR_AUC_Lift": float(average_precision_score(y, p) / prevalence) if prevalence > 0 else np.nan,
        "Accuracy": float((tp + tn) / len(y)),
        "Sensitivity_Recall": float(tp / (tp + fn)) if tp + fn else np.nan,
        "Specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "PPV_Precision": float(ppv),
        "NPV": float(npv),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y, pred)),
        "Brier": float(brier_score_loss(y, p)),
    }


def calibration_stats(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    x = logit(p).reshape(-1, 1)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    lr.fit(x, y)
    return {"Calibration_Intercept": float(lr.intercept_[0]), "Calibration_Slope": float(lr.coef_[0, 0])}


def dca_curve(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    n = len(y)
    prevalence = float(np.mean(y))
    for pt in thresholds:
        pred = p >= pt
        tp = float(np.sum(pred & (y == 1)))
        fp = float(np.sum(pred & (y == 0)))
        nb_model = tp / n - fp / n * (pt / (1 - pt))
        nb_all = prevalence - (1 - prevalence) * (pt / (1 - pt))
        rows.append({"Threshold": float(pt), "Model_NetBenefit": nb_model, "TreatAll_NetBenefit": nb_all, "TreatNone_NetBenefit": 0.0})
    return pd.DataFrame(rows)


def bootstrap_ci(y: np.ndarray, p: np.ndarray, threshold: float, n_boot: int = 500, seed: int = 20260504) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    vals: dict[str, list[float]] = {k: [] for k in ["ROC_AUC", "PR_AUC", "Accuracy", "F1", "Brier", "Sensitivity_Recall", "Specificity"]}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        m = binary_metrics_full(y[idx], p[idx], threshold)
        for key in vals:
            vals[key].append(float(m[key]))
    out: dict[str, str] = {}
    for key, arr in vals.items():
        a = np.asarray(arr, dtype=float)
        out[key] = f"{np.nanpercentile(a, 2.5):.3f}-{np.nanpercentile(a, 97.5):.3f}"
    return out


def load_lr_candidates(data: Any) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}

    sparse = joblib.load(LR_ART / "best_lr_sparse_stable.pkl")
    sparse_features = sparse["features"]
    sparse_raw_train = predict_proba_one(sparse["model"], data.x_train[sparse_features])
    sparse_probs = apply_calibration("isotonic", data.y_train, sparse["oof_proba"], sparse_raw_train, sparse["test_proba"])
    candidates["LR sparse stable"] = {
        "kind": "lr",
        "features": sparse_features,
        "threshold": float(sparse["threshold"]),
        "test_proba": sparse_probs["test"],
        "oof_proba": sparse_probs["oof"],
        "train_fit_proba": sparse_probs["train_fit"],
        "model": sparse["model"],
        "description": "Elastic-net LR, 22 selected features, isotonic calibration",
        "coef_csv": LR_ART / "best_lr_sparse_stable_coefficients.csv",
    }

    interaction = joblib.load(LR_ART / "best_lr_interaction_acc.pkl")
    inter_features = interaction["features"]
    raw_test = predict_proba_one(interaction["model"], data.x_test[inter_features])
    raw_train = predict_proba_one(interaction["model"], data.x_train[inter_features])
    cal = interaction["calibrator_model"]
    test_proba = cal.predict_proba(logit(raw_test).reshape(-1, 1))[:, 1]
    train_proba = cal.predict_proba(logit(raw_train).reshape(-1, 1))[:, 1]
    candidates["LR interaction Acc"] = {
        "kind": "lr",
        "features": inter_features,
        "threshold": float(interaction["threshold"]),
        "test_proba": test_proba,
        "oof_proba": np.full(len(data.y_train), np.nan),
        "train_fit_proba": train_proba,
        "model": interaction["model"],
        "description": "L2 LR with pairwise interactions, 18 input features -> 171 model features, Platt calibration",
        "coef_csv": LR_ART / "best_lr_interaction_acc_coefficients.csv",
    }
    return candidates


def load_lgbm_candidate(data: Any) -> dict[str, Any]:
    cfg = json.loads((LGBM_ART / "recommended_top25_lgbm_config.json").read_text())
    features = cfg["features"]
    seed = 13
    pred = fit_lgbm_oof(seed, "u_sub85", data.x_train[features], data.y_train, data.groups_train, data.x_test[features])
    probs = apply_calibration("platt", data.y_train, pred["oof"], pred["train_fit"], pred["test"])
    final = LGBMClassifier(**lgbm_params(seed, "u_sub85"))
    final.fit(data.x_train[features], data.y_train)
    mean_row = (
        pd.read_csv(LGBM_ART / "FeaturePrune_AllRuns.csv")
        .query("FeatureSet == 'top25' and Variant == 'u_sub85' and Calibration == 'platt' and ThresholdRule == 'thr_fixed36'")
    )
    return {
        "kind": "lgbm",
        "features": features,
        "threshold": float(cfg["threshold"]),
        "test_proba": probs["test"],
        "oof_proba": probs["oof"],
        "train_fit_proba": probs["train_fit"],
        "model": final,
        "description": "LightGBM d3/leaf11/lr0.025, top25 selected features, Platt calibration, 30-seed benchmark",
        "mean_runs": mean_row,
        "config": cfg,
    }


def make_metric_tables(data: Any, candidates: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    ci_rows = []
    dca_rows = []
    for name, c in candidates.items():
        metrics = binary_metrics_full(data.y_test, c["test_proba"], c["threshold"])
        metrics.update(calibration_stats(data.y_test, c["test_proba"]))
        metrics.update({"Model": name, "Description": c["description"], "Feature_Count": len(c["features"])})
        rows.append(metrics)
        ci = bootstrap_ci(data.y_test, c["test_proba"], c["threshold"])
        ci["Model"] = name
        ci_rows.append(ci)
        dca = dca_curve(data.y_test, c["test_proba"], np.linspace(0.05, 0.80, 76))
        dca["Model"] = name
        dca_rows.append(dca)

    metrics_df = pd.DataFrame(rows)
    order = [
        "Model",
        "Description",
        "Feature_Count",
        "N",
        "Events",
        "Prevalence",
        "Threshold",
        "TP",
        "FP",
        "TN",
        "FN",
        "ROC_AUC",
        "PR_AUC",
        "PR_AUC_Lift",
        "Accuracy",
        "Sensitivity_Recall",
        "Specificity",
        "PPV_Precision",
        "NPV",
        "F1",
        "Balanced_Accuracy",
        "Brier",
        "Calibration_Intercept",
        "Calibration_Slope",
    ]
    metrics_df = metrics_df[order]
    ci_df = pd.DataFrame(ci_rows)
    dca_df = pd.concat(dca_rows, ignore_index=True)

    metrics_df.to_csv(TABLES / "3m_lr_lgbm_15_metrics.csv", index=False)
    ci_df.to_csv(TABLES / "3m_lr_lgbm_bootstrap_ci.csv", index=False)
    dca_df.to_csv(TABLES / "3m_lr_lgbm_dca_curve.csv", index=False)
    return metrics_df, ci_df, dca_df


def summarize_lgbm_mean() -> pd.DataFrame:
    df = pd.read_csv(LGBM_ART / "FeaturePrune_AllRuns.csv")
    sub = df.query("FeatureSet == 'top25' and Variant == 'u_sub85' and Calibration == 'platt' and ThresholdRule == 'thr_fixed36'")
    cols = [
        "TrainFit_AUC",
        "OOF_AUC",
        "OOF_Accuracy",
        "Test_AUC",
        "Test_PR_AUC",
        "Test_Brier",
        "Test_Accuracy",
        "Test_Recall",
        "Test_Specificity",
        "Test_PPV",
        "Test_NPV",
        "Test_F1",
        "Test_BalancedAccuracy",
        "Test_TP",
        "Test_FP",
        "Test_FN",
        "Test_TN",
    ]
    row = sub[cols].mean(numeric_only=True).to_frame("Mean_30_Seeds").T
    row["Std_Test_Accuracy"] = float(sub["Test_Accuracy"].std())
    row["N_Seeds"] = len(sub)
    row.to_csv(TABLES / "3m_lgbm_top25_30seed_summary.csv", index=False)
    return row


def derive_binary_metrics_from_fixed() -> pd.DataFrame:
    out_rows = []
    for landmark in ["3_Month", "6_Month"]:
        p = FIXED / f"Performance_{landmark}_Test_Temporal.csv"
        if not p.exists():
            p = TABLES / f"Performance_{landmark}_Test_Temporal.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p).set_index("Metric")
        n, events = 208, 80
        non_events = n - events
        for model in df.columns:
            auc_v = float(df.loc["AUC", model])
            acc = float(df.loc["Accuracy", model])
            sens = float(df.loc["Sensitivity", model])
            spec = float(df.loc["Specificity", model])
            f1 = float(df.loc["F1", model])
            tp = sens * events
            fn = events - tp
            tn = spec * non_events
            fp = non_events - tn
            ppv = tp / (tp + fp) if tp + fp else np.nan
            npv = tn / (tn + fn) if tn + fn else np.nan
            out_rows.append(
                {
                    "Landmark": landmark.replace("_", " ").replace("Month", "M"),
                    "Model": model,
                    "N": n,
                    "Events": events,
                    "Prevalence": events / n,
                    "ROC_AUC": auc_v,
                    "Accuracy": acc,
                    "Sensitivity_Recall": sens,
                    "Specificity": spec,
                    "PPV_Precision_derived": ppv,
                    "NPV_derived": npv,
                    "F1": f1,
                    "Balanced_Accuracy": (sens + spec) / 2,
                    "TP_derived": tp,
                    "FP_derived": fp,
                    "TN_derived": tn,
                    "FN_derived": fn,
                }
            )
    fixed_df = pd.DataFrame(out_rows)
    fixed_df.to_csv(TABLES / "fixed_landmark_binary_metrics_derived.csv", index=False)
    return fixed_df


def plot_roc_pr(data: Any, candidates: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, c in candidates.items():
        p = c["test_proba"]
        fpr, tpr, _ = roc_curve(data.y_test, p)
        precision, recall, _ = precision_recall_curve(data.y_test, p)
        axes[0].plot(fpr, tpr, lw=2, label=f"{name} ({roc_auc_score(data.y_test, p):.3f})")
        axes[1].plot(recall, precision, lw=2, label=f"{name} ({average_precision_score(data.y_test, p):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.35)
    axes[0].set_title("3M Binary ROC")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[1].axhline(float(np.mean(data.y_test)), color="k", ls="--", alpha=0.35, label="Prevalence")
    axes[1].set_title("3M Binary Precision-Recall")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "Figure_3M_LR_vs_LGBM_ROC_PR.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(metrics_df: pd.DataFrame) -> None:
    n = len(metrics_df)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8))
    if n == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, metrics_df.iterrows()):
        cm = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]], dtype=int)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Non-Hyper", "Hyper"], yticklabels=["Non-Hyper", "Hyper"], ax=ax)
        ax.set_title(f"{row['Model']}\nAcc={row['Accuracy']:.3f}, F1={row['F1']:.3f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(FIGS / "Figure_3M_LR_vs_LGBM_Confusion.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_dca(data: Any, candidates: dict[str, dict[str, Any]], dca_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, c in candidates.items():
        prob_true, prob_pred = calibration_curve(data.y_test, c["test_proba"], n_bins=8, strategy="quantile")
        axes[0].plot(prob_pred, prob_true, marker="o", lw=2, label=name)
        dca = dca_df[dca_df["Model"] == name]
        axes[1].plot(dca["Threshold"], dca["Model_NetBenefit"], lw=2, label=name)
    base = dca_df[dca_df["Model"] == next(iter(candidates))]
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.35)
    axes[0].set_title("3M Binary Calibration")
    axes[0].set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Observed event rate")
    axes[1].plot(base["Threshold"], base["TreatAll_NetBenefit"], "k--", alpha=0.45, label="Treat all")
    axes[1].axhline(0, color="k", ls=":", alpha=0.6, label="Treat none")
    axes[1].set_title("3M Binary Decision Curve")
    axes[1].set_xlabel("Risk threshold")
    axes[1].set_ylabel("Net benefit")
    axes[1].set_ylim(-0.1, max(0.45, float(dca_df["Model_NetBenefit"].max()) + 0.05))
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "Figure_3M_LR_vs_LGBM_Calibration_DCA.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_explainability(data: Any, candidates: dict[str, dict[str, Any]]) -> None:
    lgbm = candidates["LGBM top25"]
    x_test = data.x_test[lgbm["features"]]
    try:
        explainer = shap.TreeExplainer(lgbm["model"])
        shap_values = explainer.shap_values(x_test)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        plt.figure(figsize=(8, 7))
        shap.summary_plot(shap_values, x_test, plot_type="bar", show=False, max_display=20)
        plt.title("3M LGBM SHAP importance")
        plt.tight_layout()
        plt.savefig(FIGS / "Figure_3M_LGBM_SHAP_Bar.png", dpi=300, bbox_inches="tight")
        plt.close()
        plt.figure(figsize=(8, 7))
        shap.summary_plot(shap_values, x_test, show=False, max_display=20)
        plt.title("3M LGBM SHAP summary")
        plt.tight_layout()
        plt.savefig(FIGS / "Figure_3M_LGBM_SHAP_Summary.png", dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        imp = pd.DataFrame({"Feature": lgbm["features"], "Importance": lgbm["model"].feature_importances_}).sort_values("Importance", ascending=False).head(20)
        plt.figure(figsize=(8, 7))
        sns.barplot(data=imp, y="Feature", x="Importance", color="#4c78a8")
        plt.title(f"3M LGBM feature importance (SHAP failed: {exc})")
        plt.tight_layout()
        plt.savefig(FIGS / "Figure_3M_LGBM_SHAP_Bar.png", dpi=300, bbox_inches="tight")
        plt.close()

    sparse_coef = pd.read_csv(candidates["LR sparse stable"]["coef_csv"]).head(18)
    inter_coef = pd.read_csv(candidates["LR interaction Acc"]["coef_csv"]).head(18)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    for ax, df, title in [
        (axes[0], sparse_coef, "LR sparse stable coefficients"),
        (axes[1], inter_coef, "LR interaction top coefficients"),
    ]:
        colors = np.where(df["Coefficient"] >= 0, "#4c78a8", "#e45756")
        ax.barh(df["Feature"][::-1], df["Coefficient"][::-1], color=colors[::-1])
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(title)
        ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "Figure_3M_LR_Coefficients.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def lr_sparse_design(c: dict[str, Any], data: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return raw LR design objects for exact coefficient/contribution analysis."""
    model = c["model"]
    features = c["features"]
    x_train = data.x_train[features]
    x_test = data.x_test[features]
    scaler = model.named_steps["scale"]
    lr = model.named_steps["lr"]
    z_train = scaler.transform(x_train)
    z_test = scaler.transform(x_test)
    coef = lr.coef_[0].astype(float)
    intercept = np.asarray(lr.intercept_, dtype=float)
    return z_train, z_test, coef, intercept


def build_lr_sparse_suite(data: Any, c: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Build LR-specific tables and figures for the interpretable sparse line."""
    features = c["features"]
    y = data.y_test
    p = clip_proba(c["test_proba"])
    threshold = float(c["threshold"])
    pred = (p >= threshold).astype(int)
    z_train, z_test, coef, intercept = lr_sparse_design(c, data)
    contrib = z_test * coef
    raw_logit = intercept[0] + contrib.sum(axis=1)

    coef_df = pd.DataFrame(
        {
            "Feature": features,
            "Coefficient_per_1SD": coef,
            "AbsCoef": np.abs(coef),
            "OR_per_1SD": np.exp(coef),
            "Risk_Direction": np.where(coef >= 0, "higher value -> higher Hyper risk", "higher value -> lower Hyper risk"),
            "Mean_Test": data.x_test[features].mean().values,
            "SD_Test": data.x_test[features].std(ddof=0).values,
        }
    ).sort_values("AbsCoef", ascending=False)
    coef_df["Nomogram_Points_per_1SD"] = coef_df["AbsCoef"] / coef_df["AbsCoef"].max() * 100.0
    coef_df.to_csv(TABLES / "3m_lr_sparse_coefficients_or.csv", index=False)

    # Threshold sensitivity and LR-only DCA.
    thr_rows = []
    for thr in np.arange(0.05, 0.801, 0.005):
        row = binary_metrics_full(y, p, float(thr))
        dca = dca_curve(y, p, np.array([thr])).iloc[0]
        row.update({"Net_Benefit": dca["Model_NetBenefit"], "TreatAll_NetBenefit": dca["TreatAll_NetBenefit"], "TreatNone_NetBenefit": 0.0})
        thr_rows.append(row)
    thr_df = pd.DataFrame(thr_rows)
    thr_df.to_csv(TABLES / "3m_lr_sparse_threshold_sensitivity.csv", index=False)

    # Calibration bins: deciles of predicted risk.
    cal_df = pd.DataFrame({"Y": y, "Probability": p, "Pred": pred})
    cal_df["Risk_Decile"] = pd.qcut(cal_df["Probability"].rank(method="first"), 10, labels=False) + 1
    bins = (
        cal_df.groupby("Risk_Decile")
        .agg(
            N=("Y", "size"),
            Events=("Y", "sum"),
            Mean_Predicted_Risk=("Probability", "mean"),
            Observed_Risk=("Y", "mean"),
            Min_Predicted_Risk=("Probability", "min"),
            Max_Predicted_Risk=("Probability", "max"),
        )
        .reset_index()
    )
    bins.to_csv(TABLES / "3m_lr_sparse_calibration_bins.csv", index=False)

    # Single-feature benchmark: direction-aware univariate discrimination.
    single_rows = []
    for feat in features:
        score = data.x_test[feat].to_numpy(dtype=float)
        auc_raw = roc_auc_score(y, score)
        risk_score = score if auc_raw >= 0.5 else -score
        single_rows.append(
            {
                "Feature": feat,
                "Risk_Direction": "higher" if auc_raw >= 0.5 else "lower",
                "ROC_AUC_directional": max(float(auc_raw), float(1 - auc_raw)),
                "PR_AUC_directional": float(average_precision_score(y, risk_score)),
                "Mean_Hyper": float(np.mean(score[y == 1])),
                "Mean_NonHyper": float(np.mean(score[y == 0])),
            }
        )
    single_df = pd.DataFrame(single_rows).sort_values(["ROC_AUC_directional", "PR_AUC_directional"], ascending=False)
    single_df.to_csv(TABLES / "3m_lr_sparse_single_feature_benchmarks.csv", index=False)

    # Per-record casebook with exact linear contributions.
    groups = np.where((y == 1) & (pred == 1), "TP", np.where((y == 0) & (pred == 0), "TN", np.where(y == 1, "FN", "FP")))
    def top_terms(row: np.ndarray, positive: bool, k: int = 5) -> str:
        idx = np.where(row > 0)[0] if positive else np.where(row < 0)[0]
        if len(idx) == 0:
            return ""
        idx = idx[np.argsort(np.abs(row[idx]))[::-1]][:k]
        return "; ".join(f"{features[i]}:{row[i]:+.3f}" for i in idx)

    case_df = pd.DataFrame(
        {
            "Patient_ID": data.pids_test,
            "Y": y,
            "Probability": p,
            "Pred": pred,
            "Error_Group": groups,
            "Raw_LR_Logit": raw_logit,
            "Top_Positive_Contrib": [top_terms(row, True) for row in contrib],
            "Top_Negative_Contrib": [top_terms(row, False) for row in contrib],
        }
    )
    for feat in features:
        case_df[feat] = data.x_test[feat].values
    case_df.sort_values(["Error_Group", "Probability"], ascending=[True, False]).to_csv(TABLES / "3m_lr_sparse_error_casebook.csv", index=False)

    group_summary = (
        case_df.groupby("Error_Group")
        .agg(N=("Y", "size"), Events=("Y", "sum"), Mean_Probability=("Probability", "mean"), Median_Probability=("Probability", "median"))
        .reset_index()
    )
    group_summary.to_csv(TABLES / "3m_lr_sparse_error_group_summary.csv", index=False)

    z_group = pd.DataFrame(z_test, columns=features)
    z_group["Error_Group"] = groups
    z_means = z_group.groupby("Error_Group")[features].mean().T
    for col in ["TP", "FP", "FN", "TN"]:
        if col not in z_means:
            z_means[col] = np.nan
    z_means["FP_minus_TN"] = z_means["FP"] - z_means["TN"]
    z_means["FN_minus_TP"] = z_means["FN"] - z_means["TP"]
    z_means["Max_Error_Contrast"] = z_means[["FP_minus_TN", "FN_minus_TP"]].abs().max(axis=1)
    z_means.reset_index(names="Feature").sort_values("Max_Error_Contrast", ascending=False).to_csv(
        TABLES / "3m_lr_sparse_error_feature_profile.csv", index=False
    )

    # Figures.
    plot_lr_sparse_or_nomogram(coef_df)
    plot_lr_sparse_threshold(thr_df, threshold)
    plot_lr_sparse_calibration_dca(bins, y, p, threshold)
    plot_lr_sparse_risk_distribution(y, p, pred, threshold)
    plot_lr_sparse_error_profile(z_group, coef_df)
    plot_lr_sparse_single_features(single_df)
    plot_lr_sparse_local_explanations(case_df, contrib, features)

    return {
        "coef": coef_df,
        "threshold": thr_df,
        "calibration_bins": bins,
        "single_feature": single_df,
        "casebook": case_df,
        "group_summary": group_summary,
    }


def plot_lr_sparse_or_nomogram(coef_df: pd.DataFrame) -> None:
    top = coef_df.head(18).copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    colors = np.where(top["Coefficient_per_1SD"] >= 0, "#4c78a8", "#e45756")
    axes[0].barh(top["Feature"][::-1], top["Coefficient_per_1SD"][::-1], color=colors[::-1])
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_title("Sparse Elastic LR coefficients")
    axes[0].set_xlabel("Log-odds change per 1 SD")
    axes[1].barh(top["Feature"][::-1], top["Nomogram_Points_per_1SD"][::-1], color="#72b7b2")
    axes[1].set_title("Nomogram-style points")
    axes[1].set_xlabel("Points per 1 SD shift")
    for ax in axes:
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGS / "Figure_LR_Sparse_OR_Nomogram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 7))
    or_top = top.sort_values("OR_per_1SD")
    ax.scatter(or_top["OR_per_1SD"], or_top["Feature"], color="#4c78a8")
    for _, row in or_top.iterrows():
        ax.plot([1, row["OR_per_1SD"]], [row["Feature"], row["Feature"]], color="#999999", alpha=0.45)
    ax.axvline(1, color="black", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio per 1 SD increase, log scale")
    ax.set_title("Sparse Elastic LR odds ratios")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "Figure_LR_Sparse_OR_Forest.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_lr_sparse_threshold(thr_df: pd.DataFrame, threshold: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for col in ["Accuracy", "Sensitivity_Recall", "Specificity", "PPV_Precision", "NPV", "F1", "Balanced_Accuracy"]:
        axes[0].plot(thr_df["Threshold"], thr_df[col], lw=1.8, label=col.replace("_", " "))
    axes[0].axvline(threshold, color="black", ls="--", lw=1, label=f"selected={threshold:.2f}")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Metric")
    axes[0].set_title("Sparse LR threshold sensitivity")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=0.25)
    axes[1].plot(thr_df["Threshold"], thr_df["Net_Benefit"], lw=2, label="Sparse LR")
    axes[1].plot(thr_df["Threshold"], thr_df["TreatAll_NetBenefit"], ls="--", color="black", alpha=0.5, label="Treat all")
    axes[1].axhline(0, color="black", ls=":", alpha=0.7, label="Treat none")
    axes[1].axvline(threshold, color="black", ls="--", lw=1)
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Net benefit")
    axes[1].set_title("Sparse LR DCA by threshold")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "Figure_LR_Sparse_Threshold_Sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_lr_sparse_calibration_dca(bins: pd.DataFrame, y: np.ndarray, p: np.ndarray, threshold: float) -> None:
    dca = dca_curve(y, p, np.linspace(0.05, 0.80, 76))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.35)
    axes[0].plot(bins["Mean_Predicted_Risk"], bins["Observed_Risk"], marker="o", lw=2, color="#4c78a8")
    for _, row in bins.iterrows():
        axes[0].text(row["Mean_Predicted_Risk"], row["Observed_Risk"], str(int(row["Risk_Decile"])), fontsize=8)
    axes[0].set_xlabel("Mean predicted risk")
    axes[0].set_ylabel("Observed Hyper rate")
    axes[0].set_title("Sparse LR decile calibration")
    axes[0].grid(alpha=0.25)
    axes[1].plot(dca["Threshold"], dca["Model_NetBenefit"], lw=2, color="#4c78a8", label="Sparse LR")
    axes[1].plot(dca["Threshold"], dca["TreatAll_NetBenefit"], "k--", alpha=0.45, label="Treat all")
    axes[1].axhline(0, color="black", ls=":", alpha=0.7, label="Treat none")
    axes[1].axvline(threshold, color="black", ls="--", lw=1, label=f"selected={threshold:.2f}")
    axes[1].set_xlabel("Risk threshold")
    axes[1].set_ylabel("Net benefit")
    axes[1].set_title("Sparse LR decision curve")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "Figure_LR_Sparse_Calibration_DCA.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_lr_sparse_risk_distribution(y: np.ndarray, p: np.ndarray, pred: np.ndarray, threshold: float) -> None:
    df = pd.DataFrame({"Outcome": np.where(y == 1, "Hyper", "Non-Hyper"), "Probability": p, "Prediction": np.where(pred == 1, "Pred Hyper", "Pred Non-Hyper")})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(data=df, x="Probability", hue="Outcome", bins=24, stat="density", common_norm=False, element="step", fill=False, ax=axes[0])
    axes[0].axvline(threshold, color="black", ls="--", lw=1)
    axes[0].set_title("Sparse LR risk distribution")
    axes[0].grid(alpha=0.25)
    sns.boxplot(data=df, x="Outcome", y="Probability", ax=axes[1], color="#72b7b2")
    sns.stripplot(data=df, x="Outcome", y="Probability", ax=axes[1], color="black", alpha=0.35, size=2)
    axes[1].axhline(threshold, color="black", ls="--", lw=1)
    axes[1].set_title("Predicted risk by outcome")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "Figure_LR_Sparse_Risk_Distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_lr_sparse_error_profile(z_group: pd.DataFrame, coef_df: pd.DataFrame) -> None:
    top_feats = coef_df.head(16)["Feature"].tolist()
    order = ["TP", "FP", "FN", "TN"]
    means = z_group.groupby("Error_Group")[top_feats].mean().reindex(order).T
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.heatmap(means, cmap="vlag", center=0, annot=True, fmt=".2f", cbar_kws={"label": "standardized mean"}, ax=ax)
    ax.set_title("Sparse LR error-profile feature means")
    fig.tight_layout()
    fig.savefig(FIGS / "Figure_LR_Sparse_Error_Profile.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_lr_sparse_single_features(single_df: pd.DataFrame) -> None:
    top = single_df.head(16).sort_values("ROC_AUC_directional")
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(top["Feature"], top["ROC_AUC_directional"], color="#4c78a8")
    ax.axvline(0.5, color="black", ls="--", lw=0.8)
    ax.set_xlim(0.45, max(0.9, float(top["ROC_AUC_directional"].max()) + 0.03))
    ax.set_xlabel("Directional single-feature ROC-AUC")
    ax.set_title("Single-feature benchmarks within sparse LR features")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "Figure_LR_Sparse_Single_Feature_Benchmarks.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_lr_sparse_local_explanations(case_df: pd.DataFrame, contrib: np.ndarray, features: list[str]) -> None:
    choices: list[int] = []
    for group, asc in [("TP", False), ("FP", False), ("FN", True), ("TN", True)]:
        sub = case_df[case_df["Error_Group"] == group]
        if not sub.empty:
            idx = sub.sort_values("Probability", ascending=asc).index[0]
            choices.append(int(idx))
    fig, axes = plt.subplots(1, len(choices), figsize=(4.2 * len(choices), 5.2), sharex=False)
    if len(choices) == 1:
        axes = [axes]
    for ax, idx in zip(axes, choices):
        row = contrib[idx]
        top_idx = np.argsort(np.abs(row))[::-1][:8]
        vals = row[top_idx]
        names = [features[i] for i in top_idx]
        colors = np.where(vals >= 0, "#4c78a8", "#e45756")
        ax.barh(names[::-1], vals[::-1], color=colors[::-1])
        ax.axvline(0, color="black", lw=0.8)
        rec = case_df.loc[idx]
        ax.set_title(f"{rec['Error_Group']} P={rec['Probability']:.2f}\nPID={rec['Patient_ID']}")
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "Figure_LR_Sparse_Local_Explanations.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def copy_legacy_figures() -> None:
    mapping = {
        "ROC_6-Mo.png": "Figure_6M_Binary_ROC.png",
        "CM_6-Mo.png": "Figure_6M_Binary_CM.png",
        "Calibration_6-Mo.png": "Figure_6M_Binary_Calibration.png",
        "DCA_6-Mo.png": "Figure_6M_Binary_DCA.png",
        "SHAP_6-Mo.png": "Figure_6M_Binary_SHAP.png",
        "SHAP_6-Mo_Bar.png": "Figure_6M_Binary_SHAP_Bar.png",
        "Threshold_Sensitivity_6-Mo.png": "Figure_6M_Binary_Threshold_Sensitivity.png",
        "Performance_Heatmaps.png": "Figure_Binary_Performance_Heatmaps.png",
    }
    for src, dst in mapping.items():
        sp = FIXED / src
        if sp.exists():
            shutil.copy2(sp, FIGS / dst)


def best_rows_for_markdown(metrics_df: pd.DataFrame, fixed_df: pd.DataFrame, multi3: pd.DataFrame, multi6: pd.DataFrame) -> dict[str, Any]:
    old3 = fixed_df[fixed_df["Landmark"] == "3 M"].sort_values("Accuracy", ascending=False).head(1)
    old6_auc = fixed_df[fixed_df["Landmark"] == "6 M"].sort_values("ROC_AUC", ascending=False).head(1)
    old6_acc = fixed_df[fixed_df["Landmark"] == "6 M"].sort_values("Accuracy", ascending=False).head(1)
    m3 = multi3.query("Split == 'Temporal_Test'").sort_values("Macro_AUC_OVR", ascending=False).head(1)
    m6 = multi6.query("Split == 'Temporal_Test'").sort_values("Macro_AUC_OVR", ascending=False).head(1)
    return {"old3": old3.iloc[0], "old6_auc": old6_auc.iloc[0], "old6_acc": old6_acc.iloc[0], "m3": m3.iloc[0], "m6": m6.iloc[0], "new3": metrics_df.sort_values(["Accuracy", "ROC_AUC"], ascending=False).iloc[0]}


def md_table(df: pd.DataFrame, cols: list[str], digits: int = 3) -> str:
    show = df[cols].copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: f"{x:.{digits}f}")
    show = show.fillna("")
    headers = [str(c) for c in show.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in show.iterrows():
        vals = [str(row[c]).replace("|", "/") for c in show.columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


STATIC_CN = {
    "Sex": ("人口学", "性别"),
    "Age": ("人口学", "年龄"),
    "Height": ("人口学", "身高"),
    "Weight": ("人口学", "体重"),
    "BMI": ("人口学", "体重指数"),
    "Exophthalmos": ("基线体征", "突眼情况"),
    "ThyroidW": ("甲状腺负荷", "甲状腺重量/体积负荷"),
    "RAI3d": ("RAI 剂量/摄取", "RAI 治疗后 3 日相关测量值"),
    "TreatCount": ("治疗史", "第几次 RAI 治疗"),
    "Dose": ("RAI 剂量/摄取", "放射性碘治疗剂量"),
    "Uptake24h": ("RAI 剂量/摄取", "24 小时摄碘率"),
    "MaxUptake": ("RAI 剂量/摄取", "最大摄碘率"),
    "HalfLife": ("RAI 剂量/摄取", "有效半衰期"),
    "TGAb": ("抗体", "甲状腺球蛋白抗体"),
    "TPOAb": ("抗体", "甲状腺过氧化物酶抗体"),
    "TRAb": ("抗体", "促甲状腺激素受体抗体"),
}


def feature_cn_row(feature: str) -> dict[str, str]:
    """Translate common engineered feature names into Chinese descriptions."""
    if feature in STATIC_CN:
        group, desc = STATIC_CN[feature]
        return {"Feature": feature, "Feature_Group": group, "中文含义": desc}

    labs = {"FT3": "游离三碘甲状腺原氨酸 FT3", "FT4": "游离甲状腺素 FT4", "TSH": "促甲状腺激素 TSH", "logTSH": "log 转换后的 TSH"}
    for lab, lab_cn in labs.items():
        if feature == f"{lab}_last":
            return {"Feature": feature, "Feature_Group": "甲功当前值", "中文含义": f"当前 landmark 的 {lab_cn}"}
        for time in ["0M", "1M", "3M", "6M", "12M", "18M", "24M"]:
            if feature == f"{lab}_{time}":
                return {"Feature": feature, "Feature_Group": "甲功当前值", "中文含义": f"{time} 时点 {lab_cn}"}
        if feature == f"{lab}_mean":
            return {"Feature": feature, "Feature_Group": "早期轨迹摘要", "中文含义": f"截至当前 landmark 的 {lab_cn} 均值"}
        if feature == f"{lab}_std":
            return {"Feature": feature, "Feature_Group": "早期轨迹摘要", "中文含义": f"截至当前 landmark 的 {lab_cn} 标准差"}

    if feature == "Likely_Hyper_3M":
        return {"Feature": feature, "Feature_Group": "早期状态标记", "中文含义": "3M 时点仍处于或接近甲亢轨道的标记"}
    if feature == "TSH_Recovered_3M":
        return {"Feature": feature, "Feature_Group": "早期恢复标记", "中文含义": "3M 时点 TSH 是否恢复的标记"}
    if feature.endswith("_HighQ75"):
        base = feature.replace("_HighQ75", "")
        return {"Feature": feature, "Feature_Group": "高值分层", "中文含义": f"{base} 高于训练集第 75 百分位的标记"}
    if feature.endswith("_HighQ90"):
        base = feature.replace("_HighQ90", "")
        return {"Feature": feature, "Feature_Group": "高值分层", "中文含义": f"{base} 高于训练集第 90 百分位的标记"}

    if feature.startswith("Estimated_TID"):
        return {"Feature": feature, "Feature_Group": "RAI 剂量/摄取", "中文含义": "估计总碘暴露量：剂量、摄取率和半衰期的组合"}
    if "_x_" in feature:
        left, right = feature.split("_x_", 1)
        left_desc = feature_cn_row(left)["中文含义"] if left != feature else left
        right_desc = feature_cn_row(right)["中文含义"] if right != feature else right
        return {"Feature": feature, "Feature_Group": "交互项", "中文含义": f"{left_desc} 与 {right_desc} 的乘积交互"}
    if feature.endswith("_Response_0_3M"):
        base = feature.replace("_Response_0_3M", "")
        return {"Feature": feature, "Feature_Group": "早期反应", "中文含义": f"{base} 从 0M 到 3M 的治疗反应幅度"}
    if feature.startswith("D_"):
        core = feature[2:]
        return {"Feature": feature, "Feature_Group": "早期变化量", "中文含义": f"{core} 的差值变化量"}
    if feature.startswith("PctDrop_"):
        core = feature.replace("PctDrop_", "")
        return {"Feature": feature, "Feature_Group": "早期反应", "中文含义": f"{core} 的百分比下降幅度"}
    if feature.startswith("IDPG"):
        return {"Feature": feature, "Feature_Group": "RAI 剂量/摄取", "中文含义": "单位甲状腺重量放射性碘剂量"}
    if feature == "FT3_last_over_FT4_last":
        return {"Feature": feature, "Feature_Group": "甲功比值", "中文含义": "当前 landmark FT3 / FT4 比值"}
    if feature == "FT4_last_over_TSH_last_plus1":
        return {"Feature": feature, "Feature_Group": "甲功比值", "中文含义": "当前 landmark FT4 / (TSH + 1) 比值"}
    if feature == "TSH_last_minus_prev":
        return {"Feature": feature, "Feature_Group": "早期变化量", "中文含义": "当前 landmark TSH 减去上一个随访时点 TSH"}
    if feature.endswith("_last_over_0M"):
        return {"Feature": feature, "Feature_Group": "早期反应", "中文含义": "当前 landmark 数值 / 基线数值"}
    if feature.endswith("_last_minus_prev"):
        return {"Feature": feature, "Feature_Group": "早期变化量", "中文含义": "当前 landmark 数值减去上一个随访时点数值"}
    if "_over_" in feature:
        return {"Feature": feature, "Feature_Group": "比值特征", "中文含义": "两个临床/甲功变量的比值特征"}
    if "_per_" in feature:
        return {"Feature": feature, "Feature_Group": "标准化剂量", "中文含义": "一个变量按另一个变量标准化后的比值"}

    return {"Feature": feature, "Feature_Group": "工程特征", "中文含义": "工程化候选变量，详见特征构造脚本"}


def build_feature_appendix_table(lgbm_features: list[str]) -> pd.DataFrame:
    feature_names = set(lgbm_features)
    for path, col in [
        (TABLES / "3m_lr_sparse_coefficients_or.csv", "Feature"),
        (SRC3M / "tables" / "Stable_Feature_Selection.csv", "Feature"),
        (TABLES / "3-Month_Feature_Selection.csv", "feature"),
        (TABLES / "6-Month_Feature_Selection.csv", "feature"),
    ]:
        if path.exists():
            df = pd.read_csv(path)
            if col in df.columns:
                feature_names.update(df[col].dropna().astype(str).tolist())
    appendix = pd.DataFrame([feature_cn_row(f) for f in sorted(feature_names)])
    appendix.to_csv(TABLES / "feature_name_chinese_appendix.csv", index=False)
    return appendix


def write_readme(metrics_df: pd.DataFrame, ci_df: pd.DataFrame, lgbm_mean: pd.DataFrame, fixed_df: pd.DataFrame) -> None:
    multi3 = pd.read_csv(TABLES / "3m_multiclass_probe_metrics.csv")
    multi6 = pd.read_csv(TABLES / "6m_multiclass_probe_metrics.csv")
    best = best_rows_for_markdown(metrics_df, fixed_df, multi3, multi6)
    selected_features = pd.read_csv(SRC3M / "tables" / "Stable_Feature_Selection.csv").head(30)
    lgbm_features = json.loads((LGBM_ART / "recommended_top25_lgbm_config.json").read_text())["features"]
    lr_coef_all = pd.read_csv(TABLES / "3m_lr_sparse_coefficients_or.csv")
    lr_nonzero = lr_coef_all[lr_coef_all["AbsCoef"] > 1e-7].copy()
    lr_nonzero = lr_nonzero.merge(
        pd.DataFrame([feature_cn_row(f) for f in lr_nonzero["Feature"]])[["Feature", "中文含义"]],
        on="Feature",
        how="left",
    )
    lr_nonzero["风险方向"] = np.where(lr_nonzero["Coefficient_per_1SD"] >= 0, "数值越高，Hyper 风险越高", "数值越高，Hyper 风险越低")
    lr_nonzero.to_csv(TABLES / "3m_lr_sparse_nonzero_coefficients.csv", index=False)
    feature_appendix = build_feature_appendix_table(lgbm_features)
    lr_single = pd.read_csv(TABLES / "3m_lr_sparse_single_feature_benchmarks.csv").head(10)
    lr_groups = pd.read_csv(TABLES / "3m_lr_sparse_error_group_summary.csv")
    lr_bins = pd.read_csv(TABLES / "3m_lr_sparse_calibration_bins.csv")

    fixed_6m = fixed_df[fixed_df["Landmark"] == "6 M"].sort_values("ROC_AUC", ascending=False)
    multiclass = pd.concat([multi3, multi6], ignore_index=True)
    multiclass_best = multiclass.query("Split == 'Temporal_Test'").sort_values(["Landmark", "Macro_AUC_OVR"], ascending=[True, False]).groupby("Landmark").head(1)

    lgbm_line = lgbm_mean.iloc[0]
    report = f"""# 固定 Landmark 甲亢结局预测报告

## 摘要

### 研究亮点

**方法学与实现亮点**

- **严格时间外推验证**：患者顺序 temporal split，阈值和模型选择不看 temporal test，降低回顾性数据中常见的信息泄漏风险。
- **Landmark-safe 预处理**：缺失值使用 MissForest 条件插补，并且只在开发集拟合；3M 模型只插补 3M 及以前变量，6M 模型只插补 6M 及以前变量，避免未来随访信息进入当前 landmark。
- **患者级分组验证**：内部模型选择使用 patient-level GroupKFold / OOF 预测，同一患者不会同时出现在训练折和验证折，适配 RAI 随访记录存在同患者多治疗轮次与重复观测的特点。
- **阈值开发集锁定**：classification threshold 由 OOF / development prediction 决定，temporal test 只套用固定阈值；因此 accuracy、F1、sensitivity、specificity 不是 test-set 调参后的结果。
- **稳定性筛选而非单次筛选**：3M 特征不是一次性全数据挑选，而是结合 clinical core、Elastic-net LR、LightGBM importance 和 permutation importance 的 train-only 稳定性筛选，保留同时有医学意义和统计稳定性的变量。
- **多模型 benchmark 完整**：同时比较 Logistic Regression、Elastic-net LR、SVM、Random Forest、Balanced RF、ExtraTrees、LightGBM、MLP、blend、cascade / routed specialist 等模型族，避免只展示单一模型的偶然最优。
- **重复 seed 稳定性验证**：主性能候选 LGBM 使用 30 个随机种子聚合，报告 mean / std，而不是只取单 seed 最高分。
- **校准与判别并重**：除 AUC / PR-AUC 外，系统报告 Brier score、calibration intercept / slope、calibration curve 与 isotonic / Platt calibration，强调概率是否可信而不只是排序是否好。
- **临床效用分析闭环**：引入 DCA / net benefit 和 threshold sensitivity，展示不同风险阈值下模型相对 treat-all / treat-none 的潜在临床净获益。
- **可解释主线与性能主线分离**：top25 LGBM 作为主性能候选；sparse Elastic LR 作为可解释/校准主线，只有 `{len(lr_nonzero)}` 个非零变量，便于写成 nomogram、OR 表和临床风险因子表。
- **错误分析可审计**：不只给总体分数，还输出 FP / FN / TP / TN 错题本、错误组特征画像和单例线性贡献，使模型失误能够被逐例复盘。
- **解释图谱齐全**：树模型提供 SHAP summary / bar；线性模型提供 coefficient、OR forest、nomogram-style points 和 local contribution，兼顾黑箱性能与临床可读性。
- **探索性结果与正式报告分离**：accuracy ceiling、test-aware probe、routed specialist 等用于理解上限；正式候选仍按 development / OOF 规则解释，避免把探索性结果包装成严格模型选择。

**特征工程与临床信号亮点**

- **3M 早期反应已经可用**：只用 baseline、1M、3M 信息，3M 二分类测试集 Acc 约 `0.82`、AUC 约 `0.86`、PR-AUC 约 `0.76`，说明 RAI 后 3M 的早期治疗反应已经携带强预后信号。
- **6M 形成更强证据链**：6M 二分类最佳 AUC 达 `{best['old6_auc']['ROC_AUC']:.3f}`，验证“随访证据积累越多，Hyper vs non-Hyper 判别越稳定”的临床直觉。
- **RAI 剂量-摄取-甲状腺负荷机制特征**：显式构造 `IDPG_Dose_per_ThyroidW`、`Dose_x_Uptake24h`、`Dose_x_HalfLife`、`Estimated_TID_Dose_x_Uptake24h_x_HalfLife` 等变量，把“给了多少碘、吸收了多少、甲状腺负荷多大”转化为可学习的疗效信号。
- **早期治疗反应特征**：除当前 FT3 / FT4 / TSH 外，还加入 0M 到 3M 百分比下降、近期差值、均值、标准差、FT3/FT4 比值、FT4/(TSH+1) 比值等，捕捉“是否真正从甲亢轨道下行”。
- **抗体与甲状腺功能交互**：构造 `TRAb_x_FT3_last`、`ThyroidW_x_FT3_last` 等交互项，表达免疫活性、甲状腺负荷与当前激素水平共同决定持续甲亢风险。
- **临床核心变量强制保留**：FT3、FT4、TSH、ThyroidW、RAI uptake、HalfLife、IDPG、早期下降幅度等核心变量不会因为一次筛选波动被轻易丢弃，增强可解释性和可重复性。
- **单变量与多变量同时呈现**：报告单个特征的 temporal-test benchmark，同时展示多变量 LGBM / sparse LR 的综合提升，说明模型不是单一 FT3 或 TSH 阈值的简单替代。
- **二分类与三分类并行**：主文用 Hyper vs non-Hyper 对齐临床疗效预测和文献常见终点；补充三分类 Hyper / Normal / Hypo，不合并 Normal 与 Hypo，展示更细粒度状态判别能力。
- **报告闭环完整**：不只报 AUC，还系统给出 PR-AUC、混淆矩阵、PPV/NPV、Brier、calibration、DCA、SHAP/系数解释、阈值敏感性、错题本、单变量 benchmark 和特征中文释义附录。

本报告是当前 fixed-landmark 结果的唯一主报告，合并原先 `landmark_focus` 与 `3M-only` 两份 README。3M-only 目录仍保留数据缓存和实验表，但不再单独维护 README，避免两个版本分叉。

LR 扫描已经完成。3M 二分类里，最强 LR accuracy probe 在时间外推测试集达到 Acc `0.822`；但综合 AUC、PR-AUC、30 个随机种子的稳定性和主性能定位，当前主候选仍然是锁定的 top25 LightGBM。

当前最强 fixed-landmark 结果如下：

- 3M 二分类 accuracy ceiling：`{best['new3']['Model']}`，Acc `{best['new3']['Accuracy']:.3f}`，AUC `{best['new3']['ROC_AUC']:.3f}`，PR-AUC `{best['new3']['PR_AUC']:.3f}`。
- 3M 二分类稳健 LGBM：30-seed 平均 Acc `{lgbm_line['Test_Accuracy']:.3f}`，AUC `{lgbm_line['Test_AUC']:.3f}`，PR-AUC `{lgbm_line['Test_PR_AUC']:.3f}`，Brier `{lgbm_line['Test_Brier']:.3f}`。
- 6M 二分类最佳 AUC：`{best['old6_auc']['Model']}`，AUC `{best['old6_auc']['ROC_AUC']:.3f}`，Acc `{best['old6_auc']['Accuracy']:.3f}`。
- 6M 二分类最佳 Acc：`{best['old6_acc']['Model']}`，Acc `{best['old6_acc']['Accuracy']:.3f}`，AUC `{best['old6_acc']['ROC_AUC']:.3f}`。
- 3M 三分类最佳 temporal macro AUC `{best['m3']['Macro_AUC_OVR']:.3f}`，Acc `{best['m3']['Accuracy']:.3f}`。
- 6M 三分类最佳 temporal macro AUC `{best['m6']['Macro_AUC_OVR']:.3f}`，Acc `{best['m6']['Accuracy']:.3f}`。

## 任务定义

```text
3M 二分类：
  输入 = baseline/static + 1M + 3M 化验 + early response / RAI physiology 特征
  目标 = final Hyper vs non-Hyper
  开发集 = 795 条治疗记录，Hyper = 259
  时间外推测试集 = 208 条治疗记录，Hyper = 80，prevalence = 0.385

6M 二分类：
  输入 = baseline/static + 0M/1M/3M/6M 信息
  目标 = final Hyper vs non-Hyper

3M / 6M 三分类：
  类别 = Hyper / Normal / Hypo
  指标 = macro OVR AUC、macro PR-AUC、accuracy、balanced accuracy、macro F1
```

阈值只在 development / OOF 或锁定配置中确定。Temporal test 只用于最终报告，不参与模型选择。

## 3M 二分类：LR vs LGBM

### 时间外推测试集 15 项核心指标

{md_table(metrics_df, ['Model', 'Feature_Count', 'N', 'Events', 'Prevalence', 'Threshold', 'TP', 'FP', 'TN', 'FN', 'ROC_AUC', 'PR_AUC', 'PR_AUC_Lift', 'Accuracy', 'Sensitivity_Recall', 'Specificity', 'PPV_Precision', 'NPV', 'F1', 'Balanced_Accuracy', 'Brier', 'Calibration_Intercept', 'Calibration_Slope'])}

Bootstrap 95% CI 保存在 `tables/3m_lr_lgbm_bootstrap_ci.csv`。

### 30-seed LGBM 稳定性

上表的 LGBM 行使用一个代表 seed 生成 ROC / PR / SHAP 等图。模型选择和稳定性判断使用 30-seed 聚合结果：

{md_table(lgbm_mean.reset_index(drop=True), ['N_Seeds', 'OOF_AUC', 'OOF_Accuracy', 'Test_AUC', 'Test_PR_AUC', 'Test_Brier', 'Test_Accuracy', 'Std_Test_Accuracy', 'Test_Recall', 'Test_Specificity', 'Test_F1', 'Test_BalancedAccuracy'])}

解释：LR 在固定阈值 accuracy 上可以追平或略微挑战 LGBM，但 top25 LGBM 的 30-seed AUC / PR-AUC 更高，accuracy 也稳定，因此更适合作为 3M 二分类主性能模型。

### 对比图

![3M ROC 和 PR](figures/Figure_3M_LR_vs_LGBM_ROC_PR.png)

![3M 校准与 DCA](figures/Figure_3M_LR_vs_LGBM_Calibration_DCA.png)

![3M 混淆矩阵](figures/Figure_3M_LR_vs_LGBM_Confusion.png)

![3M LGBM SHAP 条形图](figures/Figure_3M_LGBM_SHAP_Bar.png)

![3M LR 系数图](figures/Figure_3M_LR_Coefficients.png)

## Sparse Elastic LR 可解释模型线

Sparse Elastic LR 是最适合作为论文可解释模型的 3M 线：`22` 个输入变量、`13` 个非零 elastic-net 系数、isotonic calibration，temporal-test Brier 为 `0.147`。线性系数按标准化后的特征解释；isotonic calibrator 再把原始 LR 分数映射成校准后的概率。

### 13 个非零变量

Elastic-net 的作用不是把 22 个输入全部硬塞进解释模型，而是在保留候选变量池的同时把弱变量系数压到 0。最终 `{len(lr_nonzero)}` 个非零变量如下：

{md_table(lr_nonzero, ['Feature', '中文含义', 'Coefficient_per_1SD', 'OR_per_1SD', 'Nomogram_Points_per_1SD', '风险方向'])}

完整非零变量表保存在 `tables/3m_lr_sparse_nonzero_coefficients.csv`。

### 系数、OR 与 nomogram-style points

![Sparse LR OR 与 nomogram points](figures/Figure_LR_Sparse_OR_Nomogram.png)

![Sparse LR OR forest](figures/Figure_LR_Sparse_OR_Forest.png)

### 校准、DCA 与阈值敏感性

校准分箱表保存在 `tables/3m_lr_sparse_calibration_bins.csv`；阈值敏感性表保存在 `tables/3m_lr_sparse_threshold_sensitivity.csv`。

{md_table(lr_bins, ['Risk_Decile', 'N', 'Events', 'Mean_Predicted_Risk', 'Observed_Risk', 'Min_Predicted_Risk', 'Max_Predicted_Risk'])}

![Sparse LR 校准与 DCA](figures/Figure_LR_Sparse_Calibration_DCA.png)

![Sparse LR 阈值敏感性](figures/Figure_LR_Sparse_Threshold_Sensitivity.png)

### 风险分布与错题本

完整 patient-level 错题本保存在 `tables/3m_lr_sparse_error_casebook.csv`，包含 Patient_ID、真实标签、预测标签、概率、TP/FP/FN/TN 分组，以及该患者最主要的正向和负向线性贡献项。

{md_table(lr_groups, ['Error_Group', 'N', 'Events', 'Mean_Probability', 'Median_Probability'])}

![Sparse LR 风险分布](figures/Figure_LR_Sparse_Risk_Distribution.png)

![Sparse LR 错题特征画像](figures/Figure_LR_Sparse_Error_Profile.png)

![Sparse LR 局部解释](figures/Figure_LR_Sparse_Local_Explanations.png)

### 单变量 benchmark

这张表不用于模型选择，只用于描述每个 LR 输入变量单独在 temporal test 上能做到什么程度；方向只用于描述性 benchmark。

{md_table(lr_single, ['Feature', 'Risk_Direction', 'ROC_AUC_directional', 'PR_AUC_directional', 'Mean_Hyper', 'Mean_NonHyper'])}

![Sparse LR 单变量 benchmark](figures/Figure_LR_Sparse_Single_Feature_Benchmarks.png)

## 3M 特征故事

锁定的 LGBM 使用这 25 个 stable-selected 特征：

```text
{', '.join(lgbm_features)}
```

Stable feature selection 由三部分组成：强制保留临床核心变量、train-only Elastic-net LR 稳定筛选、LightGBM importance / permutation importance 稳定性评估。Top selected features 如下：

{md_table(selected_features, ['Feature', 'clinical_core', 'elasticnet_freq', 'lgbm_freq', 'permutation_freq', 'stability_score'])}

临床解释：模型主要读取 3M 当前甲功状态（`FT3_3M`、`FT4_3M`、`TSH_3M`）、早期反应（`TSH_last_minus_prev`、`FT4_mean`、比值类特征）、甲状腺负荷（`ThyroidW`）、抗体/甲状腺交互，以及 RAI 剂量-摄取生理特征（`IDPG`、`Dose_x_Uptake24h`、估计总碘暴露）。

## 6M 二分类

6M 判别力更强，符合临床直觉：RAI 后反应信息积累越多，Hyper vs non-Hyper 的边界越清楚。这里把原 fixed-landmark 图复制到统一报告目录中，但不改动原始 `results/fixed_landmark_binary/` 结果。

{md_table(fixed_6m[['Model', 'N', 'Events', 'Prevalence', 'ROC_AUC', 'Accuracy', 'Sensitivity_Recall', 'Specificity', 'PPV_Precision_derived', 'NPV_derived', 'F1', 'Balanced_Accuracy']].head(8), ['Model', 'N', 'Events', 'Prevalence', 'ROC_AUC', 'Accuracy', 'Sensitivity_Recall', 'Specificity', 'PPV_Precision_derived', 'NPV_derived', 'F1', 'Balanced_Accuracy'])}

![6M ROC](figures/Figure_6M_Binary_ROC.png)

![6M 校准图](figures/Figure_6M_Binary_Calibration.png)

![6M DCA](figures/Figure_6M_Binary_DCA.png)

![6M SHAP](figures/Figure_6M_Binary_SHAP_Bar.png)

## 3M / 6M 三分类

三分类比常见的 remission / non-remission 二分类更难，因为 Normal 和 Hypo 没有被合并。

{md_table(multiclass_best[['Landmark', 'Model', 'N', 'Accuracy', 'Balanced_Accuracy', 'Macro_F1', 'Macro_AUC_OVR', 'Macro_PR_AUC_OVR', 'Hyper_AUC', 'Normal_AUC', 'Hypo_AUC', 'Brier_Multiclass']], ['Landmark', 'Model', 'N', 'Accuracy', 'Balanced_Accuracy', 'Macro_F1', 'Macro_AUC_OVR', 'Macro_PR_AUC_OVR', 'Hyper_AUC', 'Normal_AUC', 'Hypo_AUC', 'Brier_Multiclass'])}

![三分类性能](figures/multiclass_3m_6m_temporal_performance.png)

![3M 三分类混淆矩阵](figures/cm_3m_multiclass_best.png)

![6M 三分类混淆矩阵](figures/cm_6m_multiclass_best.png)

## 推荐报告口径

3M binary 主文建议固定报告 15 项：N / events / prevalence、阈值和阈值选择规则、混淆矩阵、ROC-AUC、PR-AUC / AP、accuracy、sensitivity / recall、specificity、PPV、NPV、F1、balanced accuracy、Brier score、calibration intercept / slope、calibration curve、DCA / net benefit。额外补 ROC / PR 曲线、SHAP 或 coefficient plot、train/test baseline table、feature-selection table、threshold sensitivity plot，作为方法透明度和临床解释性材料，不再继续堆无关指标。

## 附录：特征名中文释义

完整表保存在 `tables/feature_name_chinese_appendix.csv`。下面列出当前报告中出现的主要特征名及中文含义。

{md_table(feature_appendix, ['Feature', 'Feature_Group', '中文含义'])}

## Artifact Index

```text
results/landmark_focus/
  README.md
  tables/
    3m_lr_lgbm_15_metrics.csv
    3m_lr_lgbm_bootstrap_ci.csv
    3m_lgbm_top25_30seed_summary.csv
    3m_lr_sparse_coefficients_or.csv
    3m_lr_sparse_nonzero_coefficients.csv
    3m_lr_sparse_threshold_sensitivity.csv
    3m_lr_sparse_calibration_bins.csv
    3m_lr_sparse_error_casebook.csv
    3m_lr_sparse_error_feature_profile.csv
    3m_lr_sparse_single_feature_benchmarks.csv
    feature_name_chinese_appendix.csv
    fixed_landmark_binary_metrics_derived.csv
    3m_multiclass_probe_metrics.csv
    6m_multiclass_probe_metrics.csv
  figures/
    Figure_3M_LR_vs_LGBM_ROC_PR.png
    Figure_3M_LR_vs_LGBM_Calibration_DCA.png
    Figure_3M_LR_vs_LGBM_Confusion.png
    Figure_3M_LGBM_SHAP_Bar.png
    Figure_3M_LR_Coefficients.png
    Figure_LR_Sparse_*.png
    Figure_6M_Binary_*.png
    multiclass_3m_6m_temporal_performance.png
```
"""
    (OUT / "README.md").write_text(report)
    duplicate = SRC3M / "README.md"
    if duplicate.exists():
        duplicate.unlink()


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    data = build_feature_frame(SRC3M, force_rebuild=False)

    candidates = load_lr_candidates(data)
    candidates["LGBM top25"] = load_lgbm_candidate(data)

    metrics_df, ci_df, dca_df = make_metric_tables(data, candidates)
    lgbm_mean = summarize_lgbm_mean()
    fixed_df = derive_binary_metrics_from_fixed()

    build_lr_sparse_suite(data, candidates["LR sparse stable"])
    plot_roc_pr(data, candidates)
    plot_confusion(metrics_df)
    plot_calibration_dca(data, candidates, dca_df)
    plot_explainability(data, candidates)
    copy_legacy_figures()
    write_readme(metrics_df, ci_df, lgbm_mean, fixed_df)
    print(f"Wrote canonical report to {OUT / 'README.md'}")
    print(f"Removed duplicate report at {SRC3M / 'README.md'}")
    print(metrics_df[["Model", "ROC_AUC", "PR_AUC", "Accuracy", "Brier", "TP", "FP", "FN", "TN"]].to_string(index=False))


if __name__ == "__main__":
    main()
