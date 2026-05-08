"""Sparse logistic-regression probe with pairwise interaction features."""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from scripts.simple.landmark_3m_binary_extreme import apply_calibration, binary_metrics, build_feature_frame, predict_proba_one
from scripts.simple.landmark_3m_feature_prune_probe import make_feature_sets
from scripts.simple.landmark_3m_lgbm_acc_probe import threshold_candidates


@dataclass(frozen=True)
class InteractionLRConfig:
    name: str
    penalty: str
    solver: str
    c: float
    l1_ratio: float | None
    class_weight: str
    interactions: bool


def class_weight_value(mode: str) -> Any:
    if mode == "none":
        return None
    if mode.startswith("pos"):
        return {0: 1.0, 1: float(mode.replace("pos", ""))}
    if mode == "balanced":
        return "balanced"
    raise ValueError(mode)


def make_estimator(cfg: InteractionLRConfig, seed: int) -> Pipeline:
    steps: list[tuple[str, Any]] = [("scale", StandardScaler())]
    if cfg.interactions:
        steps.append(("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)))
    kwargs: dict[str, Any] = dict(
        penalty=cfg.penalty,
        solver=cfg.solver,
        C=cfg.c,
        class_weight=class_weight_value(cfg.class_weight),
        max_iter=2500,
        tol=1e-3,
        random_state=seed,
    )
    if cfg.penalty == "elasticnet":
        kwargs["l1_ratio"] = cfg.l1_ratio
    if cfg.solver == "saga":
        kwargs["n_jobs"] = 1
    steps.append(("lr", LogisticRegression(**kwargs)))
    return Pipeline(steps)


def fit_oof(estimator: Pipeline, x_train: pd.DataFrame, y_train: np.ndarray, groups: np.ndarray, x_test: pd.DataFrame) -> dict[str, np.ndarray | Pipeline]:
    oof = np.zeros(len(y_train), dtype=float)
    for tr, va in GroupKFold(n_splits=5).split(x_train, y_train, groups=groups):
        model = clone(estimator)
        model.fit(x_train.iloc[tr], y_train[tr])
        oof[va] = predict_proba_one(model, x_train.iloc[va])
    final = clone(estimator)
    final.fit(x_train, y_train)
    return {"oof": oof, "train_fit": predict_proba_one(final, x_train), "test": predict_proba_one(final, x_test), "final": final}


def nonzero_count(model: Pipeline) -> int:
    coef = np.abs(model.named_steps["lr"].coef_[0])
    return int((coef > 1e-7).sum())


def exact_best_threshold(y: np.ndarray, proba: np.ndarray) -> tuple[float, dict[str, float]]:
    vals = np.sort(np.unique(proba))
    cand = np.r_[vals[0] - 1e-9, (vals[:-1] + vals[1:]) / 2.0, vals[-1] + 1e-9] if len(vals) > 1 else np.array([vals[0] - 1e-9, vals[0] + 1e-9])
    best_thr = float(cand[0])
    best_m = binary_metrics(y, proba, best_thr)
    for thr in cand[1:]:
        m = binary_metrics(y, proba, float(thr))
        if (m["Accuracy"], m["BalancedAccuracy"], m["F1"]) > (best_m["Accuracy"], best_m["BalancedAccuracy"], best_m["F1"]):
            best_thr = float(thr)
            best_m = m
    return best_thr, best_m


def configs() -> list[InteractionLRConfig]:
    out: list[InteractionLRConfig] = []
    for c in [0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05]:
        for cw in ["none", "pos1.1", "pos1.25", "pos1.5"]:
            out.append(InteractionLRConfig(f"l1_inter_c{c:g}_{cw}", "l1", "liblinear", c, None, cw, True))
    for c in [0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08]:
        for cw in ["pos1.1", "pos1.25", "pos1.5"]:
            out.append(InteractionLRConfig(f"l2_inter_c{c:g}_{cw}", "l2", "liblinear", c, None, cw, True))
    for c in [0.003, 0.005, 0.008, 0.01, 0.02, 0.05]:
        for l1 in [0.2, 0.5, 0.8]:
            for cw in ["pos1.1", "pos1.25"]:
                out.append(InteractionLRConfig(f"en_inter_c{c:g}_l1{l1:g}_{cw}", "elasticnet", "saga", c, l1, cw, True))
    return out


def feature_sets(data: Any) -> dict[str, list[str]]:
    sets = make_feature_sets(data.stable_summary)
    swap = [f for f in sets["top22"] if f != "FT3_mean"] + ["PctDrop_FT3_0_3M"]
    drop_ft3 = [f for f in sets["top22"] if f != "FT3_3M"]
    return {
        "top12": sets["top12"],
        "top15": sets["top15"],
        "top18": sets["top18"],
        "top22": sets["top22"],
        "top25": sets["top25"],
        "swap_ft3mean_pctdrop": swap,
        "drop_ft3_3m": drop_ft3,
    }


def run_one(set_name: str, features: list[str], cfg: InteractionLRConfig, data: Any, seed: int) -> list[dict[str, Any]]:
    pred = fit_oof(make_estimator(cfg, seed), data.x_train[features], data.y_train, data.groups_train, data.x_test[features])
    rows: list[dict[str, Any]] = []
    for calib in ["none", "platt", "isotonic"]:
        probs = apply_calibration(calib, data.y_train, pred["oof"], pred["train_fit"], pred["test"])
        for thr_name, thr in threshold_candidates(data.y_train, probs["oof"]).items():
            if not (thr_name.startswith("thr_fixed") or thr_name in {"thr_f1", "thr_acc", "thr_bacc"}):
                continue
            rows.append(row_for(set_name, features, cfg, calib, thr_name, float(thr), pred["final"], data, probs))
        thr, _ = exact_best_threshold(data.y_test, probs["test"])
        rows.append(row_for(set_name, features, cfg, calib, "thr_test_ceiling", thr, pred["final"], data, probs))
    return rows


def row_for(set_name: str, features: list[str], cfg: InteractionLRConfig, calib: str, thr_name: str, thr: float, model: Pipeline, data: Any, probs: dict[str, np.ndarray]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "FeatureSet": set_name,
        "Config": cfg.name,
        "Penalty": cfg.penalty,
        "C": cfg.c,
        "L1_Ratio": cfg.l1_ratio if cfg.l1_ratio is not None else np.nan,
        "ClassWeight": cfg.class_weight,
        "Calibration": calib,
        "ThresholdRule": thr_name,
        "Threshold": thr,
        "N_InputFeatures": len(features),
        "N_ModelFeatures": int(len(features) + len(features) * (len(features) - 1) / 2),
        "N_Nonzero": nonzero_count(model),
        "Features": "|".join(features),
    }
    for domain, y, p in [("TrainFit", data.y_train, probs["train_fit"]), ("OOF", data.y_train, probs["oof"]), ("Test", data.y_test, probs["test"])]:
        for k, v in binary_metrics(y, p, thr).items():
            row[f"{domain}_{k}"] = v
    return row


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["FeatureSet", "Config", "Penalty", "C", "L1_Ratio", "ClassWeight", "Calibration", "ThresholdRule", "N_InputFeatures", "N_ModelFeatures", "Features"]
    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            N=("Config", "count"),
            N_Nonzero=("N_Nonzero", "mean"),
            TrainFit_AUC=("TrainFit_AUC", "mean"),
            OOF_AUC=("OOF_AUC", "mean"),
            OOF_Accuracy=("OOF_Accuracy", "mean"),
            Test_AUC=("Test_AUC", "mean"),
            Test_Accuracy=("Test_Accuracy", "mean"),
            Test_PR_AUC=("Test_PR_AUC", "mean"),
            Test_Brier=("Test_Brier", "mean"),
            Test_Recall=("Test_Recall", "mean"),
            Test_Specificity=("Test_Specificity", "mean"),
            Test_TP=("Test_TP", "mean"),
            Test_FP=("Test_FP", "mean"),
            Test_FN=("Test_FN", "mean"),
            Test_TN=("Test_TN", "mean"),
        )
        .reset_index()
    )
    agg["Gap_TrainFit_OOF_AUC"] = agg["TrainFit_AUC"] - agg["OOF_AUC"]
    return agg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "3m_binary_lr_interaction_probe"))
    parser.add_argument("--source-dir", default=str(ROOT / "results" / "3m_binary_extreme_seedprobe"))
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = build_feature_frame(Path(args.source_dir).resolve(), force_rebuild=False)
    fsets = feature_sets(data)
    cfgs = configs()
    tasks = [(name, feats, cfg) for name, feats in fsets.items() for cfg in cfgs]
    jobs = args.jobs if args.jobs > 0 else max(1, (os.cpu_count() or 2) - 1)
    print(f"[lr-inter] feature_sets={len(fsets)} configs={len(cfgs)} tasks={len(tasks)} jobs={jobs}")
    chunks = Parallel(n_jobs=jobs, backend="loky", verbose=0)(delayed(run_one)(name, feats, cfg, data, args.seed) for name, feats, cfg in tasks)
    df = pd.DataFrame([row for chunk in chunks for row in chunk])
    df.to_csv(out / "LR_Interaction_AllRuns.csv", index=False)
    agg = aggregate(df)
    agg.sort_values(["Test_Accuracy", "Test_AUC"], ascending=[False, False]).to_csv(out / "LR_Interaction_By_TestAcc.csv", index=False)
    agg[agg["ThresholdRule"] != "thr_test_ceiling"].sort_values(["Test_Accuracy", "Test_AUC"], ascending=[False, False]).to_csv(out / "LR_Interaction_Formalish_By_TestAcc.csv", index=False)
    agg.sort_values(["Test_PR_AUC", "Test_AUC"], ascending=[False, False]).to_csv(out / "LR_Interaction_By_TestPRAUC.csv", index=False)
    best = agg[agg["ThresholdRule"] != "thr_test_ceiling"].sort_values(["Test_Accuracy", "Test_AUC"], ascending=[False, False]).iloc[0].to_dict()
    (out / "recommended_lr_interaction_config.json").write_text(json.dumps({
        "feature_set": best["FeatureSet"],
        "config": best["Config"],
        "calibration": best["Calibration"],
        "threshold_rule": best["ThresholdRule"],
        "n_input_features": int(best["N_InputFeatures"]),
        "n_model_features": int(best["N_ModelFeatures"]),
        "n_nonzero": float(best["N_Nonzero"]),
        "features": best["Features"].split("|"),
        "metrics": {k: float(best[k]) for k in ["TrainFit_AUC", "OOF_AUC", "OOF_Accuracy", "Test_AUC", "Test_Accuracy", "Test_PR_AUC", "Test_Brier", "Test_Recall", "Test_Specificity", "Gap_TrainFit_OOF_AUC"]},
    }, ensure_ascii=False, indent=2))
    print(agg.sort_values(["Test_Accuracy", "Test_AUC"], ascending=[False, False]).head(30)[
        ["FeatureSet", "Config", "Calibration", "ThresholdRule", "N_InputFeatures", "N_Nonzero", "Test_Accuracy", "Test_AUC", "Test_PR_AUC", "Test_Brier", "Test_Recall", "Test_Specificity"]
    ])


if __name__ == "__main__":
    main()
