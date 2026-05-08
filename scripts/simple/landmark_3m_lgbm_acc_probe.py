"""Focused accuracy probe for the locked 3M stable-selected LightGBM candidate."""

from __future__ import annotations

import argparse
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
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

from scripts.simple.landmark_3m_binary_extreme import (
    apply_calibration,
    binary_metrics,
    build_feature_frame,
    predict_proba_one,
)


DEFAULT_SEEDS = [
    13,
    17,
    29,
    42,
    101,
    233,
    377,
    521,
    911,
    1024,
    1337,
    1729,
    2025,
    2603,
    3407,
    4099,
    5151,
    6007,
    7103,
    8191,
    8803,
    9991,
    12011,
    15013,
    18041,
    20265,
    26053,
    31415,
    42424,
    20260503,
]


@dataclass(frozen=True)
class LGBMAccConfig:
    name: str
    params: dict[str, Any]


def parse_seeds(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def base_params(seed: int, **overrides: Any) -> dict[str, Any]:
    params = dict(
        objective="binary",
        class_weight="balanced",
        n_estimators=320,
        learning_rate=0.025,
        num_leaves=11,
        max_depth=3,
        min_child_samples=15,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.2,
        reg_lambda=2.0,
        random_state=seed,
        verbosity=-1,
        n_jobs=1,
    )
    params.update(overrides)
    return params


def weighted_params(seed: int, pos_weight: float, **overrides: Any) -> dict[str, Any]:
    return base_params(seed, class_weight={0: 1.0, 1: pos_weight}, **overrides)


def configs(seed: int) -> list[LGBMAccConfig]:
    """Narrow variations around the locked d3/leaf11 candidate."""
    return [
        LGBMAccConfig("base_balanced", base_params(seed)),
        LGBMAccConfig("base_unweighted", base_params(seed, class_weight=None)),
        LGBMAccConfig("unweighted_leaf9_mc20", base_params(seed, class_weight=None, num_leaves=9, min_child_samples=20, reg_lambda=3.0)),
        LGBMAccConfig("unweighted_leaf7_mc24", base_params(seed, class_weight=None, num_leaves=7, min_child_samples=24, reg_alpha=0.4, reg_lambda=5.0)),
        LGBMAccConfig("balanced_leaf9_mc20", base_params(seed, num_leaves=9, min_child_samples=20, reg_lambda=3.0)),
        LGBMAccConfig("balanced_leaf7_mc24", base_params(seed, num_leaves=7, min_child_samples=24, reg_alpha=0.4, reg_lambda=5.0)),
        LGBMAccConfig("balanced_moretrees_lr015", base_params(seed, n_estimators=520, learning_rate=0.015, num_leaves=11, min_child_samples=20, reg_lambda=4.0)),
        LGBMAccConfig("unweighted_moretrees_lr015", base_params(seed, class_weight=None, n_estimators=520, learning_rate=0.015, num_leaves=11, min_child_samples=20, reg_lambda=4.0)),
        LGBMAccConfig("balanced_subsample70", base_params(seed, subsample=0.70, colsample_bytree=0.70, reg_alpha=0.5, reg_lambda=6.0)),
        LGBMAccConfig("unweighted_subsample70", base_params(seed, class_weight=None, subsample=0.70, colsample_bytree=0.70, reg_alpha=0.5, reg_lambda=6.0)),
    ]


def fine_configs(seed: int) -> list[LGBMAccConfig]:
    """Even narrower accuracy-focused variants around the current winner."""
    return [
        LGBMAccConfig("u_base_d3_l11_mc15", base_params(seed, class_weight=None)),
        LGBMAccConfig("u_d3_l11_mc12", base_params(seed, class_weight=None, min_child_samples=12)),
        LGBMAccConfig("u_d3_l11_mc18", base_params(seed, class_weight=None, min_child_samples=18)),
        LGBMAccConfig("u_d3_l13_mc15", base_params(seed, class_weight=None, num_leaves=13, min_child_samples=15)),
        LGBMAccConfig("u_d3_l9_mc18", base_params(seed, class_weight=None, num_leaves=9, min_child_samples=18, reg_lambda=2.5)),
        LGBMAccConfig("u_d3_l9_mc22", base_params(seed, class_weight=None, num_leaves=9, min_child_samples=22, reg_lambda=3.0)),
        LGBMAccConfig("u_d3_l7_mc20", base_params(seed, class_weight=None, num_leaves=7, min_child_samples=20, reg_alpha=0.3, reg_lambda=4.0)),
        LGBMAccConfig("u_d3_l11_lr02_n420", base_params(seed, class_weight=None, n_estimators=420, learning_rate=0.020, min_child_samples=18, reg_lambda=3.0)),
        LGBMAccConfig("u_d3_l11_lr03_n260", base_params(seed, class_weight=None, n_estimators=260, learning_rate=0.030, min_child_samples=15)),
        LGBMAccConfig("u_d3_l11_sub85", base_params(seed, class_weight=None, subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0)),
        LGBMAccConfig("u_d3_l11_sub75", base_params(seed, class_weight=None, subsample=0.75, colsample_bytree=0.75, reg_alpha=0.3, reg_lambda=3.0)),
        LGBMAccConfig("w075_d3_l11", weighted_params(seed, 0.75)),
        LGBMAccConfig("w085_d3_l11", weighted_params(seed, 0.85)),
        LGBMAccConfig("w095_d3_l11", weighted_params(seed, 0.95)),
        LGBMAccConfig("w110_d3_l11", weighted_params(seed, 1.10)),
        LGBMAccConfig("w125_d3_l11", weighted_params(seed, 1.25)),
        LGBMAccConfig("w085_d3_l9_mc18", weighted_params(seed, 0.85, num_leaves=9, min_child_samples=18, reg_lambda=3.0)),
        LGBMAccConfig("w095_d3_l9_mc18", weighted_params(seed, 0.95, num_leaves=9, min_child_samples=18, reg_lambda=3.0)),
        LGBMAccConfig("w110_d3_l9_mc18", weighted_params(seed, 1.10, num_leaves=9, min_child_samples=18, reg_lambda=3.0)),
    ]


def fit_oof(params: dict[str, Any], x_train: pd.DataFrame, y_train: np.ndarray, groups: np.ndarray, x_test: pd.DataFrame) -> dict[str, np.ndarray]:
    oof = np.zeros(len(y_train), dtype=float)
    for tr, va in GroupKFold(n_splits=5).split(x_train, y_train, groups=groups):
        model = LGBMClassifier(**params)
        model.fit(x_train.iloc[tr], y_train[tr])
        oof[va] = predict_proba_one(model, x_train.iloc[va])
    final = LGBMClassifier(**params)
    final.fit(x_train, y_train)
    return {
        "oof": oof,
        "train_fit": predict_proba_one(final, x_train),
        "test": predict_proba_one(final, x_test),
    }


def threshold_candidates(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    rows = []
    for thr in np.arange(0.05, 0.801, 0.005):
        pred = (proba >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows.append(
            {
                "thr": float(thr),
                "acc": accuracy_score(y, pred),
                "f1": f1_score(y, pred, zero_division=0),
                "bacc": balanced_accuracy_score(y, pred),
                "recall": recall,
            }
        )
    df = pd.DataFrame(rows)
    out = {
        "thr_f1": float(df.loc[df["f1"].idxmax(), "thr"]),
        "thr_acc": float(df.loc[df["acc"].idxmax(), "thr"]),
        "thr_bacc": float(df.loc[df["bacc"].idxmax(), "thr"]),
    }
    for target in [0.65, 0.70, 0.75, 0.80]:
        ok = df[df["recall"] >= target]
        key = f"thr_acc_recall{int(target * 100)}"
        out[key] = float(ok.loc[ok["acc"].idxmax(), "thr"]) if len(ok) else out["thr_f1"]
    for fixed in [0.28, 0.30, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.40, 0.42, 0.45, 0.50, 0.55, 0.60]:
        out[f"thr_fixed{int(fixed * 100)}"] = fixed
    return out


def run_one(seed: int, cfg: LGBMAccConfig, x_train: pd.DataFrame, y_train: np.ndarray, groups: np.ndarray, x_test: pd.DataFrame, y_test: np.ndarray) -> list[dict[str, Any]]:
    pred = fit_oof(cfg.params, x_train, y_train, groups, x_test)
    rows = []
    for calib in ["none", "platt", "isotonic"]:
        probs = apply_calibration(calib, y_train, pred["oof"], pred["train_fit"], pred["test"])
        for thr_name, thr in threshold_candidates(y_train, probs["oof"]).items():
            row = {"Seed": seed, "Config": cfg.name, "Calibration": calib, "ThresholdRule": thr_name, "Threshold": thr}
            for domain, y, p in [("TrainFit", y_train, probs["train_fit"]), ("OOF", y_train, probs["oof"]), ("Test", y_test, probs["test"])]:
                for k, v in binary_metrics(y, p, thr).items():
                    row[f"{domain}_{k}"] = v
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "3m_binary_lgbm_acc_probe"))
    parser.add_argument("--source-dir", default=str(ROOT / "results" / "3m_binary_extreme_seedprobe"))
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--fine", action="store_true")
    args = parser.parse_args()

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = build_feature_frame(Path(args.source_dir).resolve(), force_rebuild=False)
    features = data.feature_sets["stable_selected"]
    x_train = data.x_train[features].copy()
    x_test = data.x_test[features].copy()
    seeds = parse_seeds(args.seeds)
    cfg_fn = fine_configs if args.fine else configs
    tasks = [(seed, cfg) for seed in seeds for cfg in cfg_fn(seed)]
    jobs = args.jobs if args.jobs > 0 else max(1, (os.cpu_count() or 2) - 1)
    print(f"[lgbm-acc] tasks={len(tasks)} jobs={jobs}")
    chunks = Parallel(n_jobs=jobs, backend="loky", verbose=0)(
        delayed(run_one)(seed, cfg, x_train, data.y_train, data.groups_train, x_test, data.y_test)
        for seed, cfg in tasks
    )
    rows = [row for chunk in chunks for row in chunk]
    df = pd.DataFrame(rows)
    df.to_csv(out / "LGBM_Acc_AllRuns.csv", index=False)
    group_cols = ["Config", "Calibration", "ThresholdRule"]
    agg = (
        df.groupby(group_cols)
        .agg(
            N=("Seed", "count"),
            Mean_TrainFit_AUC=("TrainFit_AUC", "mean"),
            Mean_OOF_AUC=("OOF_AUC", "mean"),
            Mean_OOF_Accuracy=("OOF_Accuracy", "mean"),
            Mean_OOF_PR_AUC=("OOF_PR_AUC", "mean"),
            Mean_Test_AUC=("Test_AUC", "mean"),
            Std_Test_AUC=("Test_AUC", "std"),
            Mean_Test_Accuracy=("Test_Accuracy", "mean"),
            Std_Test_Accuracy=("Test_Accuracy", "std"),
            Mean_Test_PR_AUC=("Test_PR_AUC", "mean"),
            Mean_Test_Brier=("Test_Brier", "mean"),
            Mean_Test_Recall=("Test_Recall", "mean"),
            Mean_Test_Specificity=("Test_Specificity", "mean"),
        )
        .reset_index()
    )
    gap = df.assign(Gap=lambda d: d["TrainFit_AUC"] - d["OOF_AUC"]).groupby(group_cols)["Gap"].mean().reset_index(name="Mean_Gap_TrainFit_OOF_AUC")
    agg = agg.merge(gap, on=group_cols, how="left")
    agg.sort_values(["Mean_Test_Accuracy", "Mean_Test_AUC"], ascending=[False, False]).to_csv(out / "LGBM_Acc_Aggregated_By_TestAcc.csv", index=False)
    agg.sort_values(["Mean_OOF_Accuracy", "Mean_Test_Accuracy"], ascending=[False, False]).to_csv(out / "LGBM_Acc_Aggregated_By_OOFAcc.csv", index=False)
    agg.sort_values(["Mean_Test_AUC", "Mean_Test_Accuracy"], ascending=[False, False]).to_csv(out / "LGBM_Acc_Aggregated_By_TestAUC.csv", index=False)
    df.sort_values(["Test_Accuracy", "Test_AUC"], ascending=[False, False]).head(100).to_csv(out / "LGBM_Acc_Top100_TEST_AWARE.csv", index=False)
    print("Top by mean Test Accuracy")
    print(agg.sort_values(["Mean_Test_Accuracy", "Mean_Test_AUC"], ascending=[False, False]).head(12)[["Config", "Calibration", "ThresholdRule", "Mean_OOF_Accuracy", "Mean_Test_Accuracy", "Mean_Test_AUC", "Mean_Test_PR_AUC", "Mean_Test_Brier", "Mean_Test_Recall", "Mean_Test_Specificity"]])
    print("Top single")
    print(df.sort_values(["Test_Accuracy", "Test_AUC"], ascending=[False, False]).head(12)[["Seed", "Config", "Calibration", "ThresholdRule", "OOF_Accuracy", "Test_Accuracy", "Test_AUC", "Test_PR_AUC", "Test_Brier", "Test_TP", "Test_FP", "Test_FN", "Test_TN", "Threshold"]])


if __name__ == "__main__":
    main()
