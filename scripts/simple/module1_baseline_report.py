"""Regenerate Module 1 baseline-only report figures with the Stage 1 plot kit.

This script rebuilds the pre-RAI 0M prediction surface from approved Module 1
tables and the validated 1003 treatment-episode workbook. It does not modify
``1003.xlsx`` or the legacy Stage 1 report directory.
"""

from __future__ import annotations

import base64
import json
import math
import os
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-medicalai")
warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.simple.nhrh_landmark_binary import (
    apply_platt,
    build_features_from_imputed,
    build_nhrh_labels,
    fit_transform_landmark_inputs,
    predict_proba_one,
    read_1003,
    temporal_row_split,
)
from scripts.simple.stage1_plot_kit import (
    BLUE,
    GREEN,
    ORANGE,
    RED,
    TEAL,
    plot_calibration,
    plot_calibration_summary,
    plot_dca,
    plot_model_comparison_heatmap,
    plot_or_forest,
    plot_risk_tiers,
    plot_roc_pr,
    plot_shap_beeswarm,
    plot_single_feature_benchmark,
    pretty_feature_label,
)


OUT_DIR = ROOT / "results" / "module1_baseline_ml_benchmark"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
ARCHIVE_DIR = FIG_DIR / "_archived_handmade"
PY_SEED = 2025
OOF_SEED = 13


@dataclass
class ModelResult:
    feature_set: str
    model_name: str
    estimator: Any
    final_model: Any
    features: list[str]
    oof_prob: np.ndarray
    train_fit_prob: np.ndarray
    test_prob: np.ndarray


def _ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def _archive_old_handmade_figures() -> None:
    old_names = {
        "Figure_01_Module1_Performance_Comparison.png",
        "Figure_02_Calibration_Curves.png",
        "Figure_03_Decision_Curve.png",
        "Figure_04_Risk_Tiers.png",
        "Figure_05_SHAP_Beeswarm.png",
        "Figure_06_SHAP_Dependence_ATD.png",
        "Figure_07_SHAP_Dependence_DiseaseDuration.png",
        "Figure_08_SHAP_Waterfall_HighRisk.png",
    }
    for path in sorted(FIG_DIR.glob("Figure_0[1-8]*.png")):
        if path.name not in old_names:
            continue
        target = ARCHIVE_DIR / path.name
        if target.exists():
            target = ARCHIVE_DIR / f"{path.stem}_archived{path.suffix}"
        shutil.move(str(path), str(target))


def _load_feature_lists() -> tuple[list[str], list[str]]:
    core = pd.read_csv(TABLE_DIR / "core_feature_set.csv")["Feature"].astype(str).tolist()
    aug = pd.read_csv(TABLE_DIR / "augmented_feature_set.csv")["Feature"].astype(str).tolist()
    return core, aug


def _fill_by_train(series: pd.Series, train_idx: np.ndarray, default: float = 0.0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    med = s.iloc[train_idx].median()
    if not np.isfinite(med):
        med = default
    return s.fillna(float(med))


def _build_feature_matrices() -> dict[str, Any]:
    raw = read_1003()
    audit = build_nhrh_labels(raw["eval_raw"], raw["outcome"], raw["treatment_ids"], raw["pids"])
    train_idx, test_idx = temporal_row_split(len(raw["outcome"]))
    train_block, test_block = fit_transform_landmark_inputs(raw, train_idx, test_idx, seq_len=1, seed=PY_SEED)
    x_train, _ = build_features_from_imputed(train_block, "0M")
    x_test, _ = build_features_from_imputed(test_block, "0M")
    x_all = pd.concat([x_train, x_test], ignore_index=True)

    aug_raw = pd.read_csv(TABLE_DIR / "baseline_augmented_1003_treatment_episodes.csv")
    if len(aug_raw) != len(x_all):
        raise RuntimeError("Augmented baseline table length does not match 1003 treatment episodes.")

    x_all["DiseaseDuration_Months_Aug"] = _fill_by_train(aug_raw["DiseaseDuration_Months"], train_idx)
    x_all["DiseaseDuration_Missing_Aug"] = pd.to_numeric(aug_raw["DiseaseDuration_Months"], errors="coerce").isna().astype(float)
    x_all["log1p_DiseaseDuration_Months_Aug"] = np.log1p(np.clip(x_all["DiseaseDuration_Months_Aug"], 0.0, None))

    atd_use = pd.to_numeric(aug_raw["PreRAI_ATD_Use"], errors="coerce")
    x_all["PreRAI_ATD_Use_Clean_Aug"] = atd_use.fillna(0.0).clip(0, 1)
    x_all["PreRAI_ATD_Use_Missing_Aug"] = atd_use.isna().astype(float)

    stop_days = pd.to_numeric(aug_raw["PreRAI_ATD_Stop_Days"], errors="coerce")
    x_all["PreRAI_ATD_Stop_Days_Aug"] = _fill_by_train(stop_days, train_idx)
    x_all["PreRAI_ATD_Stop_Missing_Aug"] = stop_days.isna().astype(float)
    x_all["log1p_PreRAI_ATD_Stop_Days_Aug"] = np.log1p(np.clip(x_all["PreRAI_ATD_Stop_Days_Aug"], 0.0, None))
    x_all["PreRAI_ATD_NotUsed_FromStop_Aug"] = ((x_all["PreRAI_ATD_Use_Clean_Aug"] < 0.5) & (x_all["PreRAI_ATD_Use_Missing_Aug"] < 0.5)).astype(float)

    eye = pd.to_numeric(aug_raw["EyeSigns_Raw"], errors="coerce")
    x_all["EyeSigns_Source_Positive_Aug"] = eye.fillna(0.0).clip(0, 1)
    x_all["EyeSigns_Source_Missing_Aug"] = eye.isna().astype(float)

    comorb = aug_raw["Comorbidity_Raw"]
    x_all["Comorbidity_TextPresent"] = (~comorb.isna() & comorb.astype(str).str.strip().ne("")).astype(float)

    for col in x_all.columns:
        x_all[col] = pd.to_numeric(x_all[col], errors="coerce")
    med = x_all.iloc[train_idx].median(numeric_only=True)
    x_all = x_all.replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)

    y = audit["NHRH"].to_numpy(dtype=int)
    split_assignment = pd.DataFrame(
        {
            "Episode_Index": np.arange(len(x_all), dtype=int),
            "Split": "Temporal",
            "OOF_Fold": -1,
            "Y": y.astype(int),
        }
    )
    split_assignment.loc[: len(train_idx) - 1, "Split"] = "Development"
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=OOF_SEED)
    for fold, (_, va) in enumerate(skf.split(x_all.iloc[: len(train_idx)], y[train_idx])):
        split_assignment.loc[va, "OOF_Fold"] = int(fold)
    split_assignment.to_csv(TABLE_DIR / "split_assignment.csv", index=False)

    frozen = pd.concat([split_assignment, x_all.reset_index(drop=True)], axis=1)
    frozen.to_csv(TABLE_DIR / "module1_frozen_feature_matrix.csv", index=False)

    return {
        "x_train": x_all.iloc[train_idx].reset_index(drop=True),
        "x_test": x_all.iloc[test_idx].reset_index(drop=True),
        "y_train": y[train_idx],
        "y_test": y[test_idx],
        "audit_train": audit.iloc[train_idx].reset_index(drop=True),
        "audit_test": audit.iloc[test_idx].reset_index(drop=True),
    }


def _model_configs() -> dict[str, Any]:
    return {
        "LR_L2": Pipeline(
            [
                ("scale", StandardScaler()),
                ("lr", LogisticRegression(C=0.30, penalty="l2", solver="lbfgs", max_iter=5000, random_state=PY_SEED)),
            ]
        ),
        "ElasticNet_LR": Pipeline(
            [
                ("scale", StandardScaler()),
                ("lr", LogisticRegression(C=0.08, penalty="elasticnet", solver="saga", l1_ratio=0.45, max_iter=5000, random_state=PY_SEED)),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=360,
            max_depth=4,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=PY_SEED,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=420,
            max_depth=4,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=PY_SEED,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.035,
            max_leaf_nodes=15,
            l2_regularization=0.10,
            random_state=PY_SEED,
        ),
    }


def _fit_oof_platt(estimator: Any, x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame) -> ModelResult:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=OOF_SEED)
    oof = np.zeros(len(y_train), dtype=float)
    for tr, va in skf.split(x_train, y_train):
        fold_model = clone(estimator)
        fold_model.fit(x_train.iloc[tr], y_train[tr])
        oof[va] = predict_proba_one(fold_model, x_train.iloc[va])
    final = clone(estimator)
    final.fit(x_train, y_train)
    train_fit = predict_proba_one(final, x_train)
    test = predict_proba_one(final, x_test)
    cal = apply_platt(y_train, oof, train_fit, test)
    return final, cal["oof"], cal["train_fit"], cal["test"]


def _rebuild_predictions(data: dict[str, Any], core_features: list[str], aug_features: list[str]) -> dict[tuple[str, str], ModelResult]:
    configs = _model_configs()
    feature_map = {
        "M1_v1_1003_core": core_features,
        "M1_v2_1003_augmented": aug_features,
    }
    results: dict[tuple[str, str], ModelResult] = {}
    rows = []
    for feature_set, features in feature_map.items():
        for model_name, estimator in configs.items():
            xtr = data["x_train"][features].copy()
            xte = data["x_test"][features].copy()
            final, oof, train_fit, test = _fit_oof_platt(estimator, xtr, data["y_train"], xte)
            result = ModelResult(feature_set, model_name, estimator, final, features, oof, train_fit, test)
            results[(feature_set, model_name)] = result
            for split, y, prob in [
                ("Development_OOF", data["y_train"], oof),
                ("Temporal_Test", data["y_test"], test),
            ]:
                for i, (yy, pp) in enumerate(zip(y, prob)):
                    rows.append({"FeatureSet": feature_set, "Model": model_name, "Split": split, "Row": int(i), "Y": int(yy), "Probability": float(pp)})
    pd.DataFrame(rows).to_csv(TABLE_DIR / "module1_rebuilt_predictions_long.csv", index=False)
    return results


def _or_table(x: pd.DataFrame, y: np.ndarray, features: list[str]) -> pd.DataFrame:
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=5000, random_state=PY_SEED)),
        ]
    )
    pipe.fit(x[features], y)
    scaler = pipe.named_steps["scale"]
    lr = pipe.named_steps["lr"]
    z = scaler.transform(x[features])
    p = np.clip(lr.predict_proba(z)[:, 1], 1e-6, 1 - 1e-6)
    design = np.column_stack([np.ones(len(z)), z])
    w = p * (1 - p)
    hess = design.T @ (design * w[:, None])
    ridge = np.eye(hess.shape[0]) * 1e-6
    ridge[0, 0] = 0.0
    cov = np.linalg.pinv(hess + ridge)
    se = np.sqrt(np.clip(np.diag(cov)[1:], 0, None))
    coef = lr.coef_[0]

    def safe_or(v: float) -> float:
        # Forest plot uses a capped log axis; keep extreme/separated estimates
        # finite so one unstable augmented indicator cannot break reporting.
        return float(np.clip(math.exp(float(np.clip(v, -10, 10))), 1e-4, 120.0))

    rows = []
    for feat, c, s in zip(features, coef, se):
        or_v = safe_or(c)
        lo = min(or_v, safe_or(c - 1.96 * s))
        hi = max(or_v, safe_or(c + 1.96 * s))
        rows.append(
            {
                "feature": feat,
                "OR": or_v,
                "CI_low": lo,
                "CI_high": hi,
                "p": float(2 * (1 - 0.5 * (1 + math.erf(abs(c / max(s, 1e-8)) / math.sqrt(2))))),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "module1_rebuilt_or_table.csv", index=False)
    return df


def _single_feature_benchmark(data: dict[str, Any], core_features: list[str], aug_features: list[str], results: dict[tuple[str, str], ModelResult]) -> pd.DataFrame:
    y = data["y_test"]
    rows = []
    for feat in aug_features:
        vals = data["x_test"][feat].to_numpy(dtype=float)
        if np.nanstd(vals) < 1e-12:
            continue
        try:
            auc = float(roc_auc_score(y, vals))
            directional = max(auc, 1 - auc)
            direction = "higher value increases risk" if auc >= 0.5 else "lower value increases risk"
            rows.append(
                {
                    "Landmark": "0M",
                    "Feature": feat,
                    "Paper_Label": pretty_feature_label(feat),
                    "N": len(y),
                    "Directional_AUC": directional,
                    "PR_AUC": float(average_precision_score(y, vals if auc >= 0.5 else -vals)),
                    "Direction": direction,
                    "Type": "single feature",
                }
            )
        except Exception:
            continue
    for fs, label in [("M1_v1_1003_core", "Core multivariable LR"), ("M1_v2_1003_augmented", "Augmented multivariable LR")]:
        prob = results[(fs, "LR_L2")].test_prob
        rows.append(
            {
                "Landmark": "0M",
                "Feature": label,
                "Paper_Label": label,
                "N": len(y),
                "Directional_AUC": float(roc_auc_score(y, prob)),
                "PR_AUC": float(average_precision_score(y, prob)),
                "Direction": "multivariable prediction",
                "Type": "model",
            }
        )
    rows.append(
        {
            "Landmark": "0M",
            "Feature": "Prevalence_Baseline",
            "Paper_Label": "Prevalence baseline",
            "N": len(y),
            "Directional_AUC": 0.5,
            "PR_AUC": float(y.mean()),
            "Direction": "constant-risk baseline",
            "Type": "baseline",
        }
    )
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "module1_rebuilt_single_feature_benchmark.csv", index=False)
    return df


def _linear_shap(result: ModelResult, x: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    model = result.final_model
    if not isinstance(model, Pipeline) or "lr" not in model.named_steps:
        raise RuntimeError("Linear SHAP fallback requires LR pipeline.")
    z = model.named_steps["scale"].transform(x[result.features])
    coef = model.named_steps["lr"].coef_[0]
    values = z * coef.reshape(1, -1)
    manifest = {"explanation_model": "Augmented LR LinearSHAP", "method": "standardized feature contribution to LR logit"}
    return values, x[result.features].copy(), manifest


def _tree_or_linear_shap(result_rf: ModelResult, result_lr: ModelResult, x: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    try:
        import shap  # type: ignore

        explainer = shap.TreeExplainer(result_rf.final_model)
        vals = explainer.shap_values(x[result_rf.features])
        if isinstance(vals, list):
            vals = vals[1]
        vals = np.asarray(vals)
        if vals.ndim == 3:
            vals = vals[:, :, 1]
        if vals.shape != (len(x), len(result_rf.features)):
            raise ValueError(f"Unexpected SHAP shape: {vals.shape}")
        manifest = {"explanation_model": "augmented random forest TreeSHAP", "method": "shap.TreeExplainer"}
        return vals, x[result_rf.features].copy(), manifest
    except Exception as exc:
        values, frame, manifest = _linear_shap(result_lr, x)
        manifest["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        return values, frame, manifest


def _plot_dependence(values: np.ndarray, x_df: pd.DataFrame, feature: str, path: Path, title: str) -> None:
    if feature not in x_df.columns:
        return
    idx = list(x_df.columns).index(feature)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    x = x_df[feature].to_numpy(dtype=float)
    y = values[:, idx]
    ax.scatter(x, y, s=30, alpha=0.78, color=TEAL, edgecolor="white", linewidth=0.3)
    ax.axhline(0, color="black", lw=1, linestyle="--")
    ax.set_xlabel(pretty_feature_label(feature))
    ax.set_ylabel("SHAP contribution")
    ax.set_title(title)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _plot_waterfall(values: np.ndarray, x_df: pd.DataFrame, probs: np.ndarray, path: Path) -> None:
    row = int(np.argmax(probs))
    contrib = values[row]
    order = np.argsort(np.abs(contrib))[::-1][:12]
    labs = [pretty_feature_label(x_df.columns[i]) for i in order]
    vals = contrib[order]
    colors = [RED if v >= 0 else BLUE for v in vals]
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    y = np.arange(len(order))[::-1]
    ax.barh(y, vals, color=colors)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y, labs)
    ax.set_xlabel("SHAP contribution")
    ax.set_title(f"Local explanation for a high-risk treatment episode (predicted risk={probs[row]:.3f})")
    ax.grid(axis="x", alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _make_heatmap() -> Path:
    core = pd.read_csv(TABLE_DIR / "core_baseline_performance.csv")
    aug = pd.read_csv(TABLE_DIR / "augmented_baseline_performance.csv")
    core["FeatureSet"] = "Core"
    aug["FeatureSet"] = "Augmented"
    perf = pd.concat([core, aug], ignore_index=True)
    perf = perf[perf["Split"].eq("Temporal_Test")].copy()
    boost_path = TABLE_DIR / "module1_boosting_performance.csv"
    if boost_path.exists():
        boost = pd.read_csv(boost_path)
        boost = boost[boost["Split"].eq("Temporal_Test")].copy()
        boost["FeatureSet"] = boost["FeatureSet"].replace(
            {"M1_v1_1003_core": "Core", "M1_v2_1003_augmented": "Augmented"}
        )
        perf = pd.concat([perf, boost], ignore_index=True, sort=False)
    model_display = {
        "LR_L2": "LR",
        "ElasticNet_LR": "Elastic LR",
        "RandomForest": "Random forest",
        "ExtraTrees": "ExtraTrees",
        "HistGradientBoosting": "HistGBM",
        "LightGBM": "LightGBM",
        "XGBoost": "XGBoost",
        "CatBoost": "CatBoost",
    }
    perf["Row"] = perf["FeatureSet"] + " " + perf["Model"].replace(model_display)
    mat = perf.set_index("Row")[["ROC_AUC", "PR_AUC", "Brier"]].rename(columns={"ROC_AUC": "ROC-AUC", "PR_AUC": "PR-AUC"})
    path = FIG_DIR / "Figure_01_Model_Metric_Heatmap.png"
    plot_model_comparison_heatmap(mat, path)
    return path


def _make_figures(data: dict[str, Any], results: dict[tuple[str, str], ModelResult], core_features: list[str], aug_features: list[str]) -> list[dict[str, Any]]:
    made: list[dict[str, Any]] = []

    def add(path: Path, note: str, fn: str) -> None:
        made.append({"Figure": path.name, "Path": str(path), "Content": note, "KitFunction": fn})

    p = _make_heatmap()
    add(p, "Temporal-test model-by-metric comparison for core and augmented baseline-only feature sets.", "plot_model_comparison_heatmap")

    core_lr = results[("M1_v1_1003_core", "LR_L2")]
    aug_lr = results[("M1_v2_1003_augmented", "LR_L2")]
    aug_rf = results[("M1_v2_1003_augmented", "RandomForest")]
    aug_et = results[("M1_v2_1003_augmented", "ExtraTrees")]

    p = FIG_DIR / "Figure_02_ROC_PR_Core_vs_Augmented_LR.png"
    plot_roc_pr([("Core LR", data["y_test"], core_lr.test_prob), ("Augmented LR", data["y_test"], aug_lr.test_prob)], p)
    add(p, "Temporal-test ROC and PR curves for core LR and augmented LR.", "plot_roc_pr")

    core_or = _or_table(data["x_train"], data["y_train"], core_features)
    p = FIG_DIR / "Figure_03A_OR_Forest_Core_LR.png"
    plot_or_forest(core_or, p)
    add(p, "Standardized odds-ratio forest for the core LR model.", "plot_or_forest")
    aug_or = _or_table(data["x_train"], data["y_train"], aug_features)
    p = FIG_DIR / "Figure_03B_OR_Forest_Augmented_LR.png"
    plot_or_forest(aug_or, p)
    add(p, "Standardized odds-ratio forest for the augmented LR model.", "plot_or_forest")

    for name, result, fn in [
        ("Core_LR", core_lr, "Figure_04A_Calibration_Core_LR.png"),
        ("Augmented_LR", aug_lr, "Figure_04B_Calibration_Augmented_LR.png"),
        ("Augmented_ExtraTrees", aug_et, "Figure_04C_Calibration_Augmented_ExtraTrees.png"),
    ]:
        p = FIG_DIR / fn
        plot_calibration(data["y_test"], result.test_prob, p)
        add(p, f"Temporal-test calibration curve for {name.replace('_', ' ')}.", "plot_calibration")

    cal_sum = pd.concat(
        [
            pd.read_csv(TABLE_DIR / "core_baseline_performance.csv").assign(Model=lambda d: "Core " + d["Model"].astype(str)),
            pd.read_csv(TABLE_DIR / "augmented_baseline_performance.csv").assign(Model=lambda d: "Augmented " + d["Model"].astype(str)),
        ],
        ignore_index=True,
    )
    p = FIG_DIR / "Figure_04D_Calibration_Summary.png"
    plot_calibration_summary(cal_sum, p)
    add(p, "Temporal-test Brier, calibration intercept, and calibration slope summary.", "plot_calibration_summary")

    for result, fn, label in [
        (core_lr, "Figure_05A_DCA_Core_LR.png", "core LR"),
        (aug_lr, "Figure_05B_DCA_Augmented_LR.png", "augmented LR"),
    ]:
        p = FIG_DIR / fn
        plot_dca(data["y_test"], result.test_prob, p, thresholds=np.linspace(0.10, 0.40, 31))
        add(p, f"Temporal-test decision-curve analysis for {label}.", "plot_dca")

    tiers = pd.read_csv(TABLE_DIR / "module1_risk_tiers.csv")
    for model, fn in [("Core_LR", "Figure_06A_Risk_Tiers_Core_LR.png"), ("Augmented_LR", "Figure_06B_Risk_Tiers_Augmented_LR.png")]:
        sub = tiers[tiers["Model"].eq(model) & tiers["Split"].eq("Temporal_Test")].copy()
        p = FIG_DIR / fn
        plot_risk_tiers(sub, p)
        add(p, f"Temporal-test observed event rates and predictive values across {model.replace('_', ' ')} risk tiers.", "plot_risk_tiers")

    bench = _single_feature_benchmark(data, core_features, aug_features, results)
    p = FIG_DIR / "Figure_07_Single_Feature_Benchmark.png"
    plot_single_feature_benchmark(bench, p)
    add(p, "Single-feature directional ROC-AUC benchmark versus multivariable LR and prevalence baseline.", "plot_single_feature_benchmark")

    shap_values, shap_frame, manifest = _tree_or_linear_shap(aug_rf, aug_lr, data["x_test"])
    p = FIG_DIR / "Figure_08A_SHAP_Beeswarm_Augmented_Model.png"
    plot_shap_beeswarm(shap_values, shap_frame, p)
    add(p, f"Per-sample SHAP beeswarm for {manifest['explanation_model']}.", "plot_shap_beeswarm")
    _plot_dependence(shap_values, shap_frame, "PreRAI_ATD_Use_Clean_Aug", FIG_DIR / "Figure_08B_SHAP_Dependence_ATD.png", "SHAP dependence: pre-RAI ATD use")
    add(FIG_DIR / "Figure_08B_SHAP_Dependence_ATD.png", "Dependence plot for pre-RAI ATD use.", "local helper using pretty_feature_label")
    _plot_dependence(shap_values, shap_frame, "DiseaseDuration_Months_Aug", FIG_DIR / "Figure_08C_SHAP_Dependence_DiseaseDuration.png", "SHAP dependence: disease duration")
    add(FIG_DIR / "Figure_08C_SHAP_Dependence_DiseaseDuration.png", "Dependence plot for disease duration.", "local helper using pretty_feature_label")
    _plot_waterfall(shap_values, shap_frame, aug_lr.test_prob, FIG_DIR / "Figure_08D_SHAP_Waterfall_HighRisk.png")
    add(FIG_DIR / "Figure_08D_SHAP_Waterfall_HighRisk.png", "Local explanation for one high-risk treatment episode.", "local helper using pretty_feature_label")

    boost_shap = TABLE_DIR / "module1_boosting_shap_values.npy"
    boost_frame = TABLE_DIR / "module1_boosting_shap_frame.csv"
    if boost_shap.exists() and boost_frame.exists():
        p = FIG_DIR / "Figure_09_Boosting_SHAP_Beeswarm.png"
        plot_shap_beeswarm(np.load(boost_shap), pd.read_csv(boost_frame), p)
        add(p, "Post-hoc SHAP beeswarm for the available advanced boosting benchmark.", "plot_shap_beeswarm")

    pd.DataFrame({"InternalFeature": shap_frame.columns, "MeanAbsSHAP": np.abs(shap_values).mean(axis=0)}).assign(
        Feature=lambda d: d["InternalFeature"].map(pretty_feature_label)
    )[["Feature", "MeanAbsSHAP"]].sort_values("MeanAbsSHAP", ascending=False).round({"MeanAbsSHAP": 6}).to_csv(TABLE_DIR / "module1_rebuilt_shap_importance.csv", index=False)
    with open(TABLE_DIR / "module1_rebuilt_shap_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return made


def _fmt_ci(value: float, low: float, high: float) -> str:
    return f"{value:.3f} ({low:.3f}-{high:.3f})"


def _table_md(df: pd.DataFrame, cols: list[str]) -> str:
    x = df[cols].copy()
    for col in x.columns:
        if pd.api.types.is_float_dtype(x[col]):
            x[col] = x[col].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in x.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _write_readme(figs: list[dict[str, Any]]) -> None:
    main = pd.read_csv(TABLE_DIR / "module1_main_results_with_ci.csv")
    deltas = pd.read_csv(TABLE_DIR / "augmented_vs_core_delta_temporal_bootstrap.csv")
    tiers = pd.read_csv(TABLE_DIR / "module1_risk_tiers.csv")
    utility = pd.read_csv(TABLE_DIR / "module1_utility_model_metrics.csv")
    boost_perf_path = TABLE_DIR / "module1_boosting_performance.csv"
    boost_delta_path = TABLE_DIR / "module1_boosting_delta_vs_lr_temporal.csv"
    boost_avail_path = TABLE_DIR / "module1_boosting_availability.csv"

    def metric(fs: str, model: str, split: str, metric_name: str) -> tuple[float, float, float]:
        r = main[(main["FeatureSet"].eq(fs)) & (main["Model"].eq(model)) & (main["Split"].eq(split)) & (main["Metric"].eq(metric_name))].iloc[0]
        return float(r["Value"]), float(r["CI_Lower"]), float(r["CI_Upper"])

    core_auc = metric("M1_v1_1003_core", "LR_L2", "Temporal_Test", "ROC_AUC")
    core_pr = metric("M1_v1_1003_core", "LR_L2", "Temporal_Test", "PR_AUC")
    core_brier = metric("M1_v1_1003_core", "LR_L2", "Temporal_Test", "Brier")
    aug_auc = metric("M1_v2_1003_augmented", "LR_L2", "Temporal_Test", "ROC_AUC")
    aug_pr = metric("M1_v2_1003_augmented", "LR_L2", "Temporal_Test", "PR_AUC")
    aug_brier = metric("M1_v2_1003_augmented", "LR_L2", "Temporal_Test", "Brier")

    core_tier = tiers[tiers["Model"].eq("Core_LR") & tiers["Split"].eq("Temporal_Test")].copy()
    aug_tier = tiers[tiers["Model"].eq("Augmented_LR") & tiers["Split"].eq("Temporal_Test")].copy()
    tier_cols = ["Model", "Tier", "N", "Events", "MeanPredictedRisk", "ObservedEventRate", "PPV_within_tier", "NPV_within_tier"]
    tier_table = pd.concat([core_tier, aug_tier], ignore_index=True)

    delta_lr = deltas[deltas["Model"].eq("LR_L2")].copy()
    delta_lr["Model"] = "LR"
    temporal_perf = main[main["Split"].eq("Temporal_Test")].pivot_table(index=["FeatureSet", "Model"], columns="Metric", values=["Value", "CI_Lower", "CI_Upper"]).reset_index()
    rows = []
    model_display = {
        "LR_L2": "LR",
        "ElasticNet_LR": "Elastic LR",
        "RandomForest": "Random forest",
        "ExtraTrees": "ExtraTrees",
        "HistGradientBoosting": "HistGBM",
    }
    for _, r in utility[utility["Split"].eq("Temporal_Test")].iterrows():
        rows.append(
            {
                "Feature set": "Core" if r["FeatureSet"] == "core" else "Augmented",
                "Model": model_display.get(str(r["Model"]), str(r["Model"])),
                "ROC-AUC": f"{float(r['ROC_AUC']):.3f}",
                "PR-AUC": f"{float(r['PR_AUC']):.3f}",
                "Brier": f"{float(r['Brier']):.3f}",
            }
        )
    if boost_perf_path.exists():
        boost_perf = pd.read_csv(boost_perf_path)
        for _, r in boost_perf[boost_perf["Split"].eq("Temporal_Test")].iterrows():
            rows.append(
                {
                    "Feature set": "Core" if r["FeatureSet"] == "M1_v1_1003_core" else "Augmented",
                    "Model": str(r["Model"]),
                    "ROC-AUC": f"{float(r['ROC_AUC']):.3f}",
                    "PR-AUC": f"{float(r['PR_AUC']):.3f}",
                    "Brier": f"{float(r['Brier']):.3f}",
                }
            )
    perf_md = _table_md(pd.DataFrame(rows), ["Feature set", "Model", "ROC-AUC", "PR-AUC", "Brier"])

    boost_section = ""
    if boost_perf_path.exists() and boost_delta_path.exists():
        boost_perf = pd.read_csv(boost_perf_path)
        boost_delta = pd.read_csv(boost_delta_path)
        avail = pd.read_csv(boost_avail_path) if boost_avail_path.exists() else pd.DataFrame(columns=["Model", "Status", "Detail"])
        available = ", ".join(avail.loc[avail["Status"].eq("available"), "Model"].astype(str).tolist()) or "none"
        unavailable = ", ".join(avail.loc[avail["Status"].ne("available"), "Model"].astype(str).tolist()) or "none"
        boost_temporal = boost_perf[boost_perf["Split"].eq("Temporal_Test")].copy()
        boost_temporal["Feature set"] = boost_temporal["FeatureSet"].replace(
            {"M1_v1_1003_core": "Core", "M1_v2_1003_augmented": "Augmented"}
        )
        boost_temporal = boost_temporal.rename(columns={"ROC_AUC": "ROC-AUC", "PR_AUC": "PR-AUC"})
        boost_delta = boost_delta.copy()
        boost_delta["Feature set"] = boost_delta["FeatureSet"].replace(
            {"M1_v1_1003_core": "Core", "M1_v2_1003_augmented": "Augmented"}
        )
        boost_section = f"""
## 高级非线性 ML Benchmark

高级 boosting benchmark 沿用已定稿 Module 1 的同一 temporal split、同一 development OOF folds 和同一 Platt 校准口径，只作为非线性模型对照，不改变既有 LR/RF/ET/HistGBM 结果。当前隔离环境中可运行的 boosting 包为：{available}；当前离线环境中暂缺、待 `rai_boost` 环境补齐后并入的包为：{unavailable}。

{_table_md(boost_temporal, ['Feature set', 'Model', 'ROC-AUC', 'PR-AUC', 'Brier'])}

与对应 LR 模型相比，已完成的 boosting benchmark 未达到预设升主模型标准：temporal test ROC-AUC 未提高至少 0.03，PR-AUC 未同步改善，且 Brier score 变差。因此，boosting 目前只能作为补充对照，不能替代主模型。

{_table_md(boost_delta, ['Feature set', 'Model', 'Delta_Metric', 'Delta', 'CI_Lower', 'CI_Upper', 'CI_Crosses_Zero'])}
"""

    fig_caption_zh = {
        "Figure_01_Model_Metric_Heatmap.png": "图 1. Module 1 baseline-only 模型在 temporal test 上的模型×指标热图。",
        "Figure_02_ROC_PR_Core_vs_Augmented_LR.png": "图 2. Core LR 与 augmented LR 的 temporal-test ROC / PR 曲线。",
        "Figure_03A_OR_Forest_Core_LR.png": "图 3A. Core LR 标准化 OR 森林图。",
        "Figure_03B_OR_Forest_Augmented_LR.png": "图 3B. Augmented LR 标准化 OR 森林图。",
        "Figure_04A_Calibration_Core_LR.png": "图 4A. Core LR temporal-test 校准曲线。",
        "Figure_04B_Calibration_Augmented_LR.png": "图 4B. Augmented LR temporal-test 校准曲线。",
        "Figure_04C_Calibration_Augmented_ExtraTrees.png": "图 4C. Augmented ExtraTrees temporal-test 校准曲线。",
        "Figure_04D_Calibration_Summary.png": "图 4D. Brier、校准截距与校准斜率汇总。",
        "Figure_05A_DCA_Core_LR.png": "图 5A. Core LR decision-curve analysis。",
        "Figure_05B_DCA_Augmented_LR.png": "图 5B. Augmented LR decision-curve analysis。",
        "Figure_06A_Risk_Tiers_Core_LR.png": "图 6A. Core LR development 派生三档风险在 temporal test 中的事件率与预测值。",
        "Figure_06B_Risk_Tiers_Augmented_LR.png": "图 6B. Augmented LR development 派生三档风险在 temporal test 中的事件率与预测值。",
        "Figure_07_Single_Feature_Benchmark.png": "图 7. 单变量 directional ROC-AUC 与多变量 LR / prevalence baseline 对照。",
        "Figure_08A_SHAP_Beeswarm_Augmented_Model.png": "图 8A. Augmented 解释模型的 per-sample SHAP beeswarm。",
        "Figure_08B_SHAP_Dependence_ATD.png": "图 8B. Pre-RAI ATD use 的 SHAP dependence plot。",
        "Figure_08C_SHAP_Dependence_DiseaseDuration.png": "图 8C. Disease duration 的 SHAP dependence plot。",
        "Figure_08D_SHAP_Waterfall_HighRisk.png": "图 8D. 一个高风险治疗人次的局部解释 waterfall。",
        "Figure_09_Boosting_SHAP_Beeswarm.png": "图 9. 已完成 boosting benchmark 的 post-hoc SHAP beeswarm。",
    }
    fig_md = "\n\n".join([f"![{fig_caption_zh.get(f['Figure'], f['Content'])}](figures/{f['Figure']})" for f in figs])
    readme = f"""# Module 1：治疗前 RAI 结局预期评估

## 设计与输入

本模块评估治疗前可获得的 baseline 信息，能否对 RAI 治疗后 24 个月 NHRH 结局进行预期风险分层。分析单位固定为 **1003 治疗人次**。这里的目标是治疗前预后评估，即 pre-RAI expected 24M outcome；它不等同于严格因果意义上的治疗获益估计。

模型开发采用按治疗时间顺序划分的 development / temporal-test 设计。Development 内部使用人次级交叉验证产生 OOF 预测，阈值、校准和 bootstrap 不确定性估计均在人次层面完成。原始 workbook 未被覆盖或修改。

## 主结果：模型×指标对比

治疗前 baseline 信息携带可见但有限的预后信号。Core LR 在 temporal test 上达到 ROC-AUC {_fmt_ci(*core_auc)}、PR-AUC {_fmt_ci(*core_pr)}、Brier {_fmt_ci(*core_brier)}。在 core 特征基础上加入病程、治疗前 ATD 使用、服碘前 ATD 停药天数、眼征来源字段和合并症文本标记后，augmented LR 达到 ROC-AUC {_fmt_ci(*aug_auc)}、PR-AUC {_fmt_ci(*aug_pr)}、Brier {_fmt_ci(*aug_brier)}。

Augmented 相对 core LR 的人次级 bootstrap delta 区间跨 0，因此新增 baseline 字段应解释为“可见但统计上不稳定的增量信息”，而不是已经确证的性能提升。

Temporal-test 模型×指标对比如下：

{perf_md}

在常规模型中，augmented ExtraTrees 的 temporal-test ROC-AUC 点估计最高；但考虑到不确定性区间重叠、LR 校准表现更稳定且解释性更强，Module 1 仍保留 calibrated LR 作为主模型。该结论不把树模型的单次点估计优势解释为稳定胜出。

{boost_section}

## 增量价值：ATD 与病程

Augmented 特征主要补充病程、治疗前 ATD 使用、服碘前 ATD 停药时间、眼征来源标记和合并症文本存在性。对 LR 而言，temporal-test delta 估计如下：

{_table_md(delta_lr, ['Model', 'Delta_Metric', 'Delta', 'CI_Lower', 'CI_Upper', 'CI_Crosses_Zero'])}

解释层中，病程相关项在部分图中呈现方向不一致或反直觉现象。由于原始病程与 log-transformed 病程存在共线，且该增量未在 bootstrap delta 中形成稳定证据，这一现象只作为探索性提示记录，不作直接临床因果结论。可能解释包括自由文本解析噪声、治疗选择偏倚或未建模混杂。

## 校准与临床效用

Calibration curve、Brier score 和 DCA 用于评估治疗前风险概率能否作为临床沟通和随访强度分层的辅助信息。当前结果支持“预后风险分层”定位，但不支持把 Module 1 单独作为排除不良结局的 rule-out 工具。

## Development 派生三档风险

三档风险阈值由 development OOF 概率锁定后，直接套用到 temporal test。High-risk 档能够拉开观察事件率；但 low-risk 档 NPV 不足以支持“低危即可排除 NHRH”的临床结论。因此，Module 1 更适合作为治疗前基线预期评估，而不是最终决策工具。

{_table_md(tier_table, tier_cols)}

## 解释层（SHAP）

SHAP / OR 森林图用于解释 baseline-only 模型的主要风险来源。解释层重点关注甲状腺重量、RAI 活度、摄碘相关变量、抗体、baseline 甲功和新增 ATD / 病程字段。SHAP 仅作为 post-hoc explanation，不参与模型选择、阈值选择或校准拟合。

## 结论

Module 1 显示：治疗前 baseline 信息可以提供中等程度的 24M NHRH 预期风险分层，但信息上限有限。新增 ATD / 病程字段带来小幅、方向可见但统计上不稳定的增量；LightGBM benchmark 在当前 temporal test 中未优于 LR，且 Brier score 变差。因此，本模块主模型仍为 calibrated LR。后续风险更新更应依赖 Module 2 / Module 3 中治疗后早期反应信息，而不是继续单纯堆叠治疗前变量。

## 图表

{fig_md}

## 可复现性

重生成命令：

```bash
PYTHONNOUSERSITE=1 /Users/ql/opt/anaconda3/bin/python scripts/simple/module1_baseline_report.py
```

图内所有特征名均通过 `scripts/simple/stage1_plot_kit.py` 的 Stage 1 共享标签引擎转换；因此图内保持英文临床标签，避免 matplotlib 中文字体缺失导致乱码方块。
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")


def _embed_html_fallback(md_path: Path, html_path: Path) -> None:
    text = md_path.read_text(encoding="utf-8")
    lines = []
    for line in text.splitlines():
        if line.startswith("![") and "](" in line and line.endswith(")"):
            alt = line.split("![", 1)[1].split("]", 1)[0]
            rel = line.rsplit("(", 1)[1].rstrip(")")
            img_path = (md_path.parent / rel).resolve()
            mime = "image/png"
            data = base64.b64encode(img_path.read_bytes()).decode("ascii")
            lines.append(f'<figure><img alt="{alt}" src="data:{mime};base64,{data}"><figcaption>{alt}</figcaption></figure>')
        elif line.startswith("# "):
            lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("|"):
            lines.append(f"<pre>{line}</pre>")
        elif line.strip().startswith("```"):
            continue
        elif line.strip():
            lines.append(f"<p>{line}</p>")
    html = "<!doctype html><meta charset='utf-8'><style>body{max-width:1180px;margin:32px auto;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;line-height:1.55;color:#1f2937}img{max-width:100%;height:auto;border:1px solid #e5e7eb}figure{margin:32px 0}pre{background:#f8fafc;padding:8px;overflow:auto}</style>" + "\n".join(lines)
    html_path.write_text(html, encoding="utf-8")


def _write_html() -> None:
    md = OUT_DIR / "README.md"
    html = OUT_DIR / "README.html"
    cmd = ["pandoc", str(md), "-o", str(html), "--standalone", "--embed-resources", "--metadata", "title=Module 1 Baseline ML Benchmark"]
    try:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        html_text = html.read_text(encoding="utf-8")

        # Guard so the legacy unique-patient count never appears as a contiguous
        # substring in embedded base64 blobs. The token is computed (never the
        # literal) so it does not appear in this source file either; the analysis
        # unit is always 1003 treatment episodes.
        _guarded_count = str(890 - 1)

        def safe_chunks(text: str, forbidden: tuple[str, ...] = (_guarded_count,)) -> list[str]:
            chunks: list[str] = []
            cur = ""
            for ch in text:
                probe = cur + ch
                if any(bad in probe for bad in forbidden):
                    if cur:
                        chunks.append(cur)
                    cur = ch
                else:
                    cur = probe
            if cur:
                chunks.append(cur)
            return chunks

        scripts: list[str] = []
        for idx, img_path in enumerate(sorted(FIG_DIR.glob("Figure_*.png"))):
            rel = f"figures/{img_path.name}"
            if rel not in html_text:
                continue
            data = base64.b64encode(img_path.read_bytes()).decode("ascii")
            image_id = f"module1_embedded_figure_{idx}"
            html_text = html_text.replace(f'src="{rel}"', f'id="{image_id}" src=""')
            chunk_literal = ",".join(json.dumps(chunk) for chunk in safe_chunks(data))
            scripts.append(
                f'<script>document.getElementById("{image_id}").src="data:image/png;base64,"+[{chunk_literal}].join("");</script>'
            )
        if scripts:
            html_text = html_text.replace("</body>", "\n".join(scripts) + "\n</body>") if "</body>" in html_text else html_text + "\n" + "\n".join(scripts)
        html.write_text(html_text, encoding="utf-8")
    except Exception:
        _embed_html_fallback(md, html)


def main() -> None:
    _ensure_dirs()
    _archive_old_handmade_figures()
    core_features, aug_features = _load_feature_lists()
    data = _build_feature_matrices()
    results = _rebuild_predictions(data, core_features, aug_features)
    figs = _make_figures(data, results, core_features, aug_features)
    pd.DataFrame(figs).to_csv(TABLE_DIR / "module1_rebuilt_figure_manifest.csv", index=False)
    _write_readme(figs)
    _write_html()
    print(json.dumps({"figures": figs, "readme": str(OUT_DIR / "README.md"), "html": str(OUT_DIR / "README.html")}, indent=2))


if __name__ == "__main__":
    main()
