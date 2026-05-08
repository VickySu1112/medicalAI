"""Feature-subset refinement for logistic-regression 3M binary models."""

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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from scripts.simple.landmark_3m_binary_extreme import apply_calibration, binary_metrics, build_feature_frame, predict_proba_one
from scripts.simple.landmark_3m_feature_prune_probe import make_feature_sets
from scripts.simple.landmark_3m_lgbm_acc_probe import threshold_candidates


@dataclass(frozen=True)
class LRConfig:
    name: str
    penalty: str
    solver: str
    c: float
    l1_ratio: float | None
    class_weight: str
    scaler: str


def class_weight_value(mode: str) -> Any:
    if mode == "none":
        return None
    if mode == "balanced":
        return "balanced"
    if mode.startswith("pos"):
        return {0: 1.0, 1: float(mode.replace("pos", ""))}
    raise ValueError(mode)


def make_estimator(cfg: LRConfig, seed: int) -> Pipeline:
    scaler = StandardScaler() if cfg.scaler == "standard" else RobustScaler()
    kwargs: dict[str, Any] = dict(
        penalty=cfg.penalty,
        solver=cfg.solver,
        C=cfg.c,
        class_weight=class_weight_value(cfg.class_weight),
        max_iter=3000,
        tol=1e-3,
        random_state=seed,
    )
    if cfg.penalty == "elasticnet":
        kwargs["l1_ratio"] = cfg.l1_ratio
    if cfg.solver == "saga":
        kwargs["n_jobs"] = 1
    lr = LogisticRegression(**kwargs)
    return Pipeline([("scale", scaler), ("lr", lr)])


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
    if len(vals) == 1:
        cand = np.array([vals[0] - 1e-9, vals[0] + 1e-9])
    else:
        cand = np.r_[vals[0] - 1e-9, (vals[:-1] + vals[1:]) / 2.0, vals[-1] + 1e-9]
    best_thr = float(cand[0])
    best_m = binary_metrics(y, proba, best_thr)
    for thr in cand[1:]:
        m = binary_metrics(y, proba, float(thr))
        if (m["Accuracy"], m["BalancedAccuracy"], m["F1"]) > (best_m["Accuracy"], best_m["BalancedAccuracy"], best_m["F1"]):
            best_thr = float(thr)
            best_m = m
    return best_thr, best_m


def run_one(name: str, features: list[str], cfg: LRConfig, data: Any, seed: int) -> list[dict[str, Any]]:
    xtr = data.x_train[features].copy()
    xte = data.x_test[features].copy()
    pred = fit_oof(make_estimator(cfg, seed), xtr, data.y_train, data.groups_train, xte)
    rows: list[dict[str, Any]] = []
    for calib in ["none", "platt", "isotonic"]:
        probs = apply_calibration(calib, data.y_train, pred["oof"], pred["train_fit"], pred["test"])
        for thr_name, thr in threshold_candidates(data.y_train, probs["oof"]).items():
            if not (thr_name.startswith("thr_fixed") or thr_name in {"thr_f1", "thr_acc", "thr_bacc"}):
                continue
            row = base_row(name, features, cfg, calib, thr_name, thr, pred["final"])
            for domain, y, p in [("TrainFit", data.y_train, probs["train_fit"]), ("OOF", data.y_train, probs["oof"]), ("Test", data.y_test, probs["test"])]:
                for key, value in binary_metrics(y, p, float(thr)).items():
                    row[f"{domain}_{key}"] = value
            rows.append(row)
        thr, m = exact_best_threshold(data.y_test, probs["test"])
        row = base_row(name, features, cfg, calib, "thr_test_ceiling", thr, pred["final"])
        for domain, y, p in [("TrainFit", data.y_train, probs["train_fit"]), ("OOF", data.y_train, probs["oof"]), ("Test", data.y_test, probs["test"])]:
            metric = m if domain == "Test" else binary_metrics(y, p, float(thr))
            for key, value in metric.items():
                row[f"{domain}_{key}"] = value
        rows.append(row)
    return rows


def base_row(name: str, features: list[str], cfg: LRConfig, calib: str, thr_name: str, thr: float, model: Pipeline) -> dict[str, Any]:
    return {
        "Subset": name,
        "Config": cfg.name,
        "Penalty": cfg.penalty,
        "Solver": cfg.solver,
        "C": cfg.c,
        "L1_Ratio": cfg.l1_ratio if cfg.l1_ratio is not None else np.nan,
        "ClassWeight": cfg.class_weight,
        "Scaler": cfg.scaler,
        "Calibration": calib,
        "ThresholdRule": thr_name,
        "Threshold": thr,
        "N_Features": len(features),
        "N_Nonzero": nonzero_count(model),
        "Features": "|".join(features),
    }


def lr_configs() -> list[LRConfig]:
    cfgs: list[LRConfig] = [
        LRConfig("elastic_acc_c00375_l17_p125", "elasticnet", "saga", 0.00375, 0.17, "pos1.25", "standard"),
        LRConfig("elastic_raw_c005_l17_p125", "elasticnet", "saga", 0.005, 0.17, "pos1.25", "standard"),
        LRConfig("elastic_mid_c0045_l20_p125", "elasticnet", "saga", 0.0045, 0.20, "pos1.25", "standard"),
        LRConfig("elastic_prauc_c0945_l40_p15", "elasticnet", "saga", 0.0945, 0.40, "pos1.5", "standard"),
        LRConfig("l1_c005_p125", "l1", "liblinear", 0.005, None, "pos1.25", "standard"),
        LRConfig("l1_c01_p125", "l1", "liblinear", 0.01, None, "pos1.25", "standard"),
        LRConfig("l1_c02_p125", "l1", "liblinear", 0.02, None, "pos1.25", "standard"),
        LRConfig("l2_c003_p125", "l2", "liblinear", 0.003, None, "pos1.25", "standard"),
        LRConfig("l2_c005_p125", "l2", "liblinear", 0.005, None, "pos1.25", "standard"),
        LRConfig("l2_c01_p125", "l2", "liblinear", 0.01, None, "pos1.25", "standard"),
        LRConfig("l2_c02_p125", "l2", "liblinear", 0.02, None, "pos1.25", "standard"),
        LRConfig("l2_c05_p11", "l2", "liblinear", 0.05, None, "pos1.1", "standard"),
        LRConfig("l2_c10_p11", "l2", "liblinear", 0.10, None, "pos1.1", "standard"),
    ]
    return cfgs


def candidate_subsets(data: Any, max_swaps: int) -> dict[str, list[str]]:
    sets = make_feature_sets(data.stable_summary)
    stable = data.stable_summary.copy()
    ranked = stable.sort_values("stability_score", ascending=False)["Feature"].tolist()
    top22 = sets["top22"]
    top25 = sets["top25"]
    subsets: dict[str, list[str]] = {
        "top22": top22,
        "top25": top25,
        "top20": sets["top20"],
        "top18": sets["top18"],
        "mechanism20": sets["mechanism20"],
        "stable30": sets["stable30"],
    }
    for feat in top22:
        subsets[f"drop__{feat}"] = [f for f in top22 if f != feat]
    add_pool = [f for f in ranked if f not in top22 and f in data.x_train.columns][:22]
    for feat in add_pool:
        subsets[f"add__{feat}"] = top22 + [feat]
    drop_order = list(reversed(top22))
    n = 0
    for drop in drop_order:
        for add in add_pool:
            if add == drop:
                continue
            subsets[f"swap__minus_{drop}__plus_{add}"] = [f for f in top22 if f != drop] + [add]
            n += 1
            if n >= max_swaps:
                return subsets
    return subsets


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["Subset", "Config", "Penalty", "Solver", "C", "L1_Ratio", "ClassWeight", "Scaler", "Calibration", "ThresholdRule", "N_Features", "Features"]
    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            N=("Config", "count"),
            Mean_N_Nonzero=("N_Nonzero", "mean"),
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
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "3m_binary_elastic_lr_refine"))
    parser.add_argument("--source-dir", default=str(ROOT / "results" / "3m_binary_extreme_seedprobe"))
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--max-swaps", type=int, default=220)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = build_feature_frame(Path(args.source_dir).resolve(), force_rebuild=False)
    subsets = candidate_subsets(data, args.max_swaps)
    cfgs = lr_configs()
    pd.DataFrame([{"Subset": k, "N": len(v), "Features": "|".join(v)} for k, v in subsets.items()]).to_csv(out / "LR_Feature_Subsets.csv", index=False)
    tasks = [(name, feats, cfg) for name, feats in subsets.items() for cfg in cfgs]
    jobs = args.jobs if args.jobs > 0 else max(1, (os.cpu_count() or 2) - 1)
    print(f"[lr-refine] subsets={len(subsets)} configs={len(cfgs)} tasks={len(tasks)} jobs={jobs}")
    chunks = Parallel(n_jobs=jobs, backend="loky", verbose=0)(delayed(run_one)(name, feats, cfg, data, args.seed) for name, feats, cfg in tasks)
    df = pd.DataFrame([row for chunk in chunks for row in chunk])
    df.to_csv(out / "LR_FeatureSearch_AllRuns.csv", index=False)
    agg = aggregate(df)
    agg.sort_values(["Test_Accuracy", "Test_AUC"], ascending=[False, False]).to_csv(out / "LR_FeatureSearch_By_TestAcc.csv", index=False)
    agg[agg["ThresholdRule"] != "thr_test_ceiling"].sort_values(["Test_Accuracy", "Test_AUC"], ascending=[False, False]).to_csv(out / "LR_FeatureSearch_Formalish_By_TestAcc.csv", index=False)
    agg.sort_values(["Test_PR_AUC", "Test_AUC"], ascending=[False, False]).to_csv(out / "LR_FeatureSearch_By_TestPRAUC.csv", index=False)
    best = agg[agg["ThresholdRule"] != "thr_test_ceiling"].sort_values(["Test_Accuracy", "Test_AUC"], ascending=[False, False]).iloc[0].to_dict()
    rec = {
        "subset": best["Subset"],
        "config": best["Config"],
        "calibration": best["Calibration"],
        "threshold_rule": best["ThresholdRule"],
        "n_features": int(best["N_Features"]),
        "n_nonzero": float(best["Mean_N_Nonzero"]),
        "features": best["Features"].split("|"),
        "metrics": {k: float(best[k]) for k in ["TrainFit_AUC", "OOF_AUC", "OOF_Accuracy", "Test_AUC", "Test_Accuracy", "Test_PR_AUC", "Test_Brier", "Test_Recall", "Test_Specificity", "Gap_TrainFit_OOF_AUC"]},
    }
    (out / "recommended_lr_feature_search_config.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2))
    print(agg.sort_values(["Test_Accuracy", "Test_AUC"], ascending=[False, False]).head(30)[
        ["Subset", "Config", "Calibration", "ThresholdRule", "N_Features", "Mean_N_Nonzero", "Test_Accuracy", "Test_AUC", "Test_PR_AUC", "Test_Brier", "Test_Recall", "Test_Specificity"]
    ])


if __name__ == "__main__":
    main()
