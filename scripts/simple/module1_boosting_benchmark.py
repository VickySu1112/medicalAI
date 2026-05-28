"""Advanced boosting benchmark for Module 1 baseline-only prediction.

This script is intended to run in an isolated environment that contains
XGBoost, LightGBM, and/or CatBoost. It never rebuilds the cohort and never
touches ``1003.xlsx``. Instead, it consumes the frozen treatment-episode
feature matrix and split assignment produced by ``module1_baseline_report.py``.

The analysis unit is the treatment episode. Cross-validation folds are read
from disk to guarantee comparability with the finalized Module 1 report.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-medicalai")
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "module1_baseline_ml_benchmark"
TABLE_DIR = OUT_DIR / "tables"
PY_SEED = 2025
EPS = 1e-6


@dataclass
class FittedBoosting:
    feature_set: str
    model: str
    config_id: str
    config: dict[str, Any]
    dev_prob: np.ndarray
    temporal_prob: np.ndarray
    final_model: Any
    features: list[str]
    dev_score: float


def _load_feature_list(path: Path) -> list[str]:
    df = pd.read_csv(path)
    return df["Feature"].astype(str).tolist()


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "ROC_AUC": _safe_auc(y, p),
        "PR_AUC": _safe_ap(y, p),
        "Brier": float(brier_score_loss(y, np.clip(p, EPS, 1 - EPS))),
    }


def _bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    vals: dict[str, list[float]] = {"ROC_AUC": [], "PR_AUC": [], "Brier": []}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        pp = p[idx]
        m = _metrics(yy, pp)
        for key, value in m.items():
            if np.isfinite(value):
                vals[key].append(value)
    out: dict[str, tuple[float, float]] = {}
    for key, arr in vals.items():
        if not arr:
            out[key] = (float("nan"), float("nan"))
        else:
            q = np.percentile(np.asarray(arr, dtype=float), [2.5, 97.5])
            out[key] = (float(q[0]), float(q[1]))
    return out


def _bootstrap_delta_ci(
    y: np.ndarray,
    p_boost: np.ndarray,
    p_ref: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    vals: dict[str, list[float]] = {"ROC_AUC": [], "PR_AUC": [], "Brier": []}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        mb = _metrics(yy, p_boost[idx])
        mr = _metrics(yy, p_ref[idx])
        vals["ROC_AUC"].append(mb["ROC_AUC"] - mr["ROC_AUC"])
        vals["PR_AUC"].append(mb["PR_AUC"] - mr["PR_AUC"])
        vals["Brier"].append(mb["Brier"] - mr["Brier"])
    out: dict[str, tuple[float, float]] = {}
    for key, arr in vals.items():
        if not arr:
            out[key] = (float("nan"), float("nan"))
        else:
            q = np.percentile(np.asarray(arr, dtype=float), [2.5, 97.5])
            out[key] = (float(q[0]), float(q[1]))
    return out


def _platt_fit_transform(y_dev: np.ndarray, dev_raw: np.ndarray, temporal_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, LogisticRegression]:
    dev_raw = np.clip(dev_raw, EPS, 1 - EPS)
    temporal_raw = np.clip(temporal_raw, EPS, 1 - EPS)
    dev_logit = np.log(dev_raw / (1 - dev_raw)).reshape(-1, 1)
    temporal_logit = np.log(temporal_raw / (1 - temporal_raw)).reshape(-1, 1)
    cal = LogisticRegression(solver="lbfgs", max_iter=1000)
    cal.fit(dev_logit, y_dev)
    return cal.predict_proba(dev_logit)[:, 1], cal.predict_proba(temporal_logit)[:, 1], cal


def _predict_prob(model: Any, x: pd.DataFrame) -> np.ndarray:
    p = model.predict_proba(x)
    if isinstance(p, list):
        p = p[0]
    p = np.asarray(p)
    if p.ndim == 1:
        return np.clip(p, EPS, 1 - EPS)
    return np.clip(p[:, 1], EPS, 1 - EPS)


def _fit_oof(
    make_model: Callable[[dict[str, Any]], Any],
    config: dict[str, Any],
    x_dev: pd.DataFrame,
    y_dev: np.ndarray,
    x_temporal: pd.DataFrame,
    folds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Any]:
    raw_oof = np.zeros(len(y_dev), dtype=float)
    for fold in sorted(int(v) for v in np.unique(folds) if int(v) >= 0):
        tr = np.where(folds != fold)[0]
        va = np.where(folds == fold)[0]
        model = make_model(config)
        model.fit(x_dev.iloc[tr], y_dev[tr])
        raw_oof[va] = _predict_prob(model, x_dev.iloc[va])
    final = make_model(config)
    final.fit(x_dev, y_dev)
    raw_temporal = _predict_prob(final, x_temporal)
    dev_prob, temporal_prob, _ = _platt_fit_transform(y_dev, raw_oof, raw_temporal)
    return dev_prob, temporal_prob, final


def _module_available(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def _model_spaces() -> tuple[dict[str, Callable[[dict[str, Any]], Any]], dict[str, list[dict[str, Any]]], list[dict[str, str]]]:
    factories: dict[str, Callable[[dict[str, Any]], Any]] = {}
    grids: dict[str, list[dict[str, Any]]] = {}
    availability: list[dict[str, str]] = []

    if _module_available("xgboost"):
        from xgboost import XGBClassifier  # type: ignore

        def make_xgb(cfg: dict[str, Any]) -> Any:
            return XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=PY_SEED,
                n_jobs=-1,
                **cfg,
            )

        factories["XGBoost"] = make_xgb
        grids["XGBoost"] = [
            {"n_estimators": 160, "learning_rate": 0.035, "max_depth": 2, "min_child_weight": 3, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 3.0, "reg_alpha": 0.0},
            {"n_estimators": 220, "learning_rate": 0.025, "max_depth": 2, "min_child_weight": 5, "subsample": 0.80, "colsample_bytree": 0.80, "reg_lambda": 5.0, "reg_alpha": 0.1},
            {"n_estimators": 180, "learning_rate": 0.035, "max_depth": 3, "min_child_weight": 5, "subsample": 0.85, "colsample_bytree": 0.75, "reg_lambda": 5.0, "reg_alpha": 0.2},
        ]
        availability.append({"Model": "XGBoost", "Status": "available", "Detail": ""})
    else:
        availability.append({"Model": "XGBoost", "Status": "unavailable", "Detail": "Python package xgboost is not installed in the active isolated environment."})

    if _module_available("lightgbm"):
        from lightgbm import LGBMClassifier  # type: ignore

        def make_lgbm(cfg: dict[str, Any]) -> Any:
            return LGBMClassifier(
                objective="binary",
                random_state=PY_SEED,
                n_jobs=-1,
                verbosity=-1,
                **cfg,
            )

        factories["LightGBM"] = make_lgbm
        grids["LightGBM"] = [
            {"n_estimators": 160, "learning_rate": 0.035, "max_depth": 2, "num_leaves": 7, "min_child_samples": 25, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 3.0},
            {"n_estimators": 220, "learning_rate": 0.025, "max_depth": 2, "num_leaves": 7, "min_child_samples": 40, "subsample": 0.80, "colsample_bytree": 0.80, "reg_lambda": 5.0},
            {"n_estimators": 180, "learning_rate": 0.035, "max_depth": 3, "num_leaves": 11, "min_child_samples": 35, "subsample": 0.85, "colsample_bytree": 0.75, "reg_lambda": 5.0},
            {"n_estimators": 260, "learning_rate": 0.020, "max_depth": 3, "num_leaves": 15, "min_child_samples": 50, "subsample": 0.80, "colsample_bytree": 0.80, "reg_lambda": 8.0},
        ]
        availability.append({"Model": "LightGBM", "Status": "available", "Detail": ""})
    else:
        availability.append({"Model": "LightGBM", "Status": "unavailable", "Detail": "Python package lightgbm is not installed in the active isolated environment."})

    if _module_available("catboost"):
        from catboost import CatBoostClassifier  # type: ignore

        def make_cat(cfg: dict[str, Any]) -> Any:
            return CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=PY_SEED,
                verbose=False,
                allow_writing_files=False,
                thread_count=-1,
                **cfg,
            )

        factories["CatBoost"] = make_cat
        grids["CatBoost"] = [
            {"iterations": 180, "learning_rate": 0.035, "depth": 2, "l2_leaf_reg": 5.0},
            {"iterations": 240, "learning_rate": 0.025, "depth": 2, "l2_leaf_reg": 8.0},
            {"iterations": 180, "learning_rate": 0.035, "depth": 3, "l2_leaf_reg": 8.0},
        ]
        availability.append({"Model": "CatBoost", "Status": "available", "Detail": ""})
    else:
        availability.append({"Model": "CatBoost", "Status": "unavailable", "Detail": "Python package catboost is not installed in the active isolated environment."})

    return factories, grids, availability


def _select_and_fit(
    model_name: str,
    make_model: Callable[[dict[str, Any]], Any],
    grid: list[dict[str, Any]],
    feature_set: str,
    features: list[str],
    x_dev_all: pd.DataFrame,
    y_dev: np.ndarray,
    x_temporal_all: pd.DataFrame,
    y_temporal: np.ndarray,
    folds: np.ndarray,
) -> FittedBoosting:
    best: FittedBoosting | None = None
    for idx, cfg in enumerate(grid):
        dev_prob, temporal_prob, final = _fit_oof(
            make_model,
            cfg,
            x_dev_all[features],
            y_dev,
            x_temporal_all[features],
            folds,
        )
        dm = _metrics(y_dev, dev_prob)
        # Dev-only model selection. ROC and PR reward discrimination; Brier
        # penalizes poor calibration after Platt scaling.
        score = float(dm["ROC_AUC"] + dm["PR_AUC"] - dm["Brier"])
        candidate = FittedBoosting(
            feature_set=feature_set,
            model=model_name,
            config_id=f"{model_name}_cfg{idx:02d}",
            config=cfg,
            dev_prob=dev_prob,
            temporal_prob=temporal_prob,
            final_model=final,
            features=features,
            dev_score=score,
        )
        if best is None or candidate.dev_score > best.dev_score:
            best = candidate
    if best is None:
        raise RuntimeError(f"No fitted model produced for {model_name} / {feature_set}")
    return best


def _load_lr_reference(feature_set: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    pred = pd.read_csv(TABLE_DIR / "module1_rebuilt_predictions_long.csv")
    sub = pred[
        pred["FeatureSet"].eq(feature_set)
        & pred["Model"].eq("LR_L2")
        & pred["Split"].eq(split)
    ].sort_values("Row")
    if sub.empty:
        raise RuntimeError(f"Missing LR reference predictions for {feature_set} / {split}")
    return sub["Y"].to_numpy(dtype=int), sub["Probability"].to_numpy(dtype=float)


def _write_predictions(results: list[FittedBoosting], y_dev: np.ndarray, y_temporal: np.ndarray) -> None:
    rows = []
    for r in results:
        for split, y, prob in [
            ("Development_OOF", y_dev, r.dev_prob),
            ("Temporal_Test", y_temporal, r.temporal_prob),
        ]:
            for i, (yy, pp) in enumerate(zip(y, prob)):
                rows.append(
                    {
                        "FeatureSet": r.feature_set,
                        "Model": r.model,
                        "Split": split,
                        "Row": int(i),
                        "Y": int(yy),
                        "Probability": float(pp),
                        "ConfigID": r.config_id,
                    }
                )
    pd.DataFrame(rows).to_csv(TABLE_DIR / "module1_boosting_predictions_long.csv", index=False)


def _write_metrics(results: list[FittedBoosting], y_dev: np.ndarray, y_temporal: np.ndarray, n_boot: int) -> None:
    perf_rows = []
    ci_rows = []
    for r in results:
        for split, y, p in [
            ("Development_OOF", y_dev, r.dev_prob),
            ("Temporal_Test", y_temporal, r.temporal_prob),
        ]:
            m = _metrics(y, p)
            ci = _bootstrap_ci(y, p, n_boot=n_boot, seed=PY_SEED + len(perf_rows))
            perf_rows.append(
                {
                    "FeatureSet": r.feature_set,
                    "Model": r.model,
                    "Split": split,
                    "N": int(len(y)),
                    "Events": int(np.sum(y)),
                    **m,
                    "ConfigID": r.config_id,
                    "DevSelectionScore": r.dev_score,
                }
            )
            for metric, value in m.items():
                lo, hi = ci[metric]
                ci_rows.append(
                    {
                        "FeatureSet": r.feature_set,
                        "Model": r.model,
                        "Split": split,
                        "Metric": metric,
                        "Value": value,
                        "CI_Lower": lo,
                        "CI_Upper": hi,
                        "ConfigID": r.config_id,
                    }
                )
    pd.DataFrame(perf_rows).to_csv(TABLE_DIR / "module1_boosting_performance.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(TABLE_DIR / "module1_boosting_bootstrap_ci.csv", index=False)


def _write_deltas(results: list[FittedBoosting], y_temporal: np.ndarray, n_boot: int) -> None:
    rows = []
    ci_rows = []
    for r in results:
        y_ref, p_ref = _load_lr_reference(r.feature_set, "Temporal_Test")
        if not np.array_equal(y_ref, y_temporal):
            raise RuntimeError(f"LR reference labels do not match for {r.feature_set}")
        mb = _metrics(y_temporal, r.temporal_prob)
        mr = _metrics(y_temporal, p_ref)
        ci = _bootstrap_delta_ci(y_temporal, r.temporal_prob, p_ref, n_boot=n_boot, seed=PY_SEED + 100)
        for metric in ["ROC_AUC", "PR_AUC", "Brier"]:
            delta = mb[metric] - mr[metric]
            lo, hi = ci[metric]
            row = {
                "FeatureSet": r.feature_set,
                "Model": r.model,
                "ReferenceModel": "LR_L2",
                "Split": "Temporal_Test",
                "Delta_Metric": f"Delta_{metric}",
                "Delta": delta,
                "CI_Lower": lo,
                "CI_Upper": hi,
                "CI_Crosses_Zero": bool(lo <= 0 <= hi),
                "BoostingValue": mb[metric],
                "LRValue": mr[metric],
                "ConfigID": r.config_id,
            }
            rows.append(row)
            ci_rows.append(row)
    pd.DataFrame(rows).to_csv(TABLE_DIR / "module1_boosting_delta_vs_lr_temporal.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(TABLE_DIR / "module1_boosting_delta_vs_lr_temporal_bootstrap.csv", index=False)


def _write_configs(results: list[FittedBoosting], availability: list[dict[str, str]]) -> None:
    pd.DataFrame(
        [
            {
                "FeatureSet": r.feature_set,
                "Model": r.model,
                "ConfigID": r.config_id,
                "DevSelectionScore": r.dev_score,
                "ConfigJSON": json.dumps(r.config, sort_keys=True),
            }
            for r in results
        ]
    ).to_csv(TABLE_DIR / "module1_boosting_selected_configs.csv", index=False)
    pd.DataFrame(availability).to_csv(TABLE_DIR / "module1_boosting_availability.csv", index=False)


def _write_shap(best: FittedBoosting, x_temporal_all: pd.DataFrame) -> None:
    import shap  # type: ignore

    x = x_temporal_all[best.features].copy()
    explainer = shap.TreeExplainer(best.final_model)
    vals = explainer.shap_values(x)
    if isinstance(vals, list):
        vals = vals[1]
    vals = np.asarray(vals)
    if vals.ndim == 3:
        vals = vals[:, :, 1]
    if vals.shape != (len(x), len(best.features)):
        raise RuntimeError(f"Unexpected SHAP matrix shape: {vals.shape}, expected {(len(x), len(best.features))}")
    np.save(TABLE_DIR / "module1_boosting_shap_values.npy", vals)
    x.to_csv(TABLE_DIR / "module1_boosting_shap_frame.csv", index=False)
    imp = pd.DataFrame(
        {
            "Feature": best.features,
            "MeanAbsSHAP": np.abs(vals).mean(axis=0),
            "Model": best.model,
            "FeatureSet": best.feature_set,
            "ConfigID": best.config_id,
        }
    ).sort_values("MeanAbsSHAP", ascending=False)
    imp.to_csv(TABLE_DIR / "module1_boosting_shap_importance.csv", index=False)
    manifest = {
        "explanation_model": best.model,
        "feature_set": best.feature_set,
        "config_id": best.config_id,
        "method": "shap.TreeExplainer on temporal-test rows",
        "selection_note": "best available boosting candidate by temporal ROC-AUC for post-hoc explanation only",
    }
    (TABLE_DIR / "module1_boosting_shap_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    global OUT_DIR, TABLE_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    OUT_DIR = Path(args.out)
    TABLE_DIR = OUT_DIR / "tables"
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    split = pd.read_csv(TABLE_DIR / "split_assignment.csv")
    frozen = pd.read_csv(TABLE_DIR / "module1_frozen_feature_matrix.csv")
    if len(split) != 1003 or len(frozen) != 1003:
        raise RuntimeError("Expected exactly 1003 treatment episodes in frozen split and feature matrix.")

    core_features = _load_feature_list(TABLE_DIR / "core_feature_set.csv")
    aug_features = _load_feature_list(TABLE_DIR / "augmented_feature_set.csv")
    feature_sets = {
        "M1_v1_1003_core": core_features,
        "M1_v2_1003_augmented": aug_features,
    }

    dev_mask = split["Split"].eq("Development").to_numpy()
    temporal_mask = split["Split"].eq("Temporal").to_numpy()
    y = split["Y"].to_numpy(dtype=int)
    y_dev = y[dev_mask]
    y_temporal = y[temporal_mask]
    folds = split.loc[dev_mask, "OOF_Fold"].to_numpy(dtype=int)

    x_dev_all = frozen.loc[dev_mask].reset_index(drop=True)
    x_temporal_all = frozen.loc[temporal_mask].reset_index(drop=True)

    factories, grids, availability = _model_spaces()
    results: list[FittedBoosting] = []
    for feature_set, features in feature_sets.items():
        missing = sorted(set(features) - set(frozen.columns))
        if missing:
            raise RuntimeError(f"Frozen matrix is missing features for {feature_set}: {missing[:10]}")
        for model_name, make_model in factories.items():
            fitted = _select_and_fit(
                model_name,
                make_model,
                grids[model_name],
                feature_set,
                features,
                x_dev_all,
                y_dev,
                x_temporal_all,
                y_temporal,
                folds,
            )
            results.append(fitted)

    _write_configs(results, availability)
    if not results:
        print(json.dumps({"status": "no_available_boosting_packages", "availability": availability}, indent=2))
        return

    _write_predictions(results, y_dev, y_temporal)
    _write_metrics(results, y_dev, y_temporal, n_boot=args.n_boot)
    _write_deltas(results, y_temporal, n_boot=args.n_boot)

    best = max(results, key=lambda r: _metrics(y_temporal, r.temporal_prob)["ROC_AUC"])
    try:
        _write_shap(best, x_temporal_all)
        shap_status = "ok"
    except Exception as exc:
        shap_status = f"{type(exc).__name__}: {exc}"
        (TABLE_DIR / "module1_boosting_shap_manifest.json").write_text(
            json.dumps({"status": "failed", "reason": shap_status}, indent=2),
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "status": "ok",
                "available_models": sorted(factories),
                "availability": availability,
                "n_results": len(results),
                "shap_status": shap_status,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
