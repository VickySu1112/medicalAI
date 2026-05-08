"""Prune stable-selected 3M features while preserving LGBM accuracy."""

from __future__ import annotations

import argparse
import os
import sys
import warnings
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
from sklearn.model_selection import GroupKFold

from scripts.simple.landmark_3m_binary_extreme import apply_calibration, binary_metrics, build_feature_frame, predict_proba_one
from scripts.simple.landmark_3m_lgbm_acc_probe import DEFAULT_SEEDS, parse_seeds, threshold_candidates


def params(seed: int, variant: str) -> dict[str, Any]:
    base = dict(
        objective="binary",
        n_estimators=320,
        learning_rate=0.025,
        num_leaves=11,
        max_depth=3,
        min_child_samples=15,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=2.0,
        random_state=seed,
        verbosity=-1,
        n_jobs=1,
    )
    if variant == "u_sub85":
        base["class_weight"] = None
    elif variant == "w125_raw":
        base["class_weight"] = {0: 1.0, 1: 1.25}
        base["subsample"] = 0.80
        base["colsample_bytree"] = 0.80
    else:
        raise ValueError(variant)
    return base


def make_feature_sets(stable: pd.DataFrame) -> dict[str, list[str]]:
    stable = stable[stable["final_keep"].astype(bool)].copy()
    ranked = stable.sort_values(["stability_score", "clinical_core"], ascending=[False, False])
    clinical = stable.loc[stable["clinical_core"].astype(bool), "Feature"].tolist()
    sets: dict[str, list[str]] = {"stable30": ranked["Feature"].tolist()}
    for k in [25, 22, 20, 18, 15, 12]:
        keep = list(dict.fromkeys(clinical + ranked["Feature"].head(k).tolist()))
        if len(keep) > k:
            core_rank = stable[stable["Feature"].isin(keep)].sort_values("stability_score", ascending=False)["Feature"].tolist()
            keep = core_rank[:k]
        else:
            for feat in ranked["Feature"]:
                if len(keep) >= k:
                    break
                if feat not in keep:
                    keep.append(feat)
        sets[f"top{k}"] = keep
    # Mechanism-lite keeps core clinical plus strongest non-core ratios/interactions.
    mechanism_priority = [
        "FT3_3M",
        "FT4_3M",
        "TSH_3M",
        "FT3_last_over_FT4_last",
        "FT4_last_over_TSH_last_plus1",
        "TSH_last_minus_prev",
        "FT4_mean",
        "ThyroidW",
        "Estimated_TID_Dose_x_Uptake24h_x_HalfLife",
        "FT3_1M",
        "Age",
        "Dose_x_Uptake24h",
        "ThyroidW_x_FT3_last",
        "TRAb_x_FT3_last",
        "HalfLife",
        "MaxUptake",
        "IDPG_Dose_per_ThyroidW",
        "PctDrop_FT4_0_3M",
        "PctDrop_FT3_0_3M",
        "Dose",
    ]
    sets["mechanism20"] = [f for f in mechanism_priority if f in stable["Feature"].values][:20]
    sets["mechanism15"] = [f for f in mechanism_priority if f in stable["Feature"].values][:15]
    return sets


def fit_oof(seed: int, variant: str, x_train: pd.DataFrame, y_train: np.ndarray, groups: np.ndarray, x_test: pd.DataFrame) -> dict[str, np.ndarray]:
    oof = np.zeros(len(y_train), dtype=float)
    p = params(seed, variant)
    for tr, va in GroupKFold(n_splits=5).split(x_train, y_train, groups=groups):
        model = LGBMClassifier(**p)
        model.fit(x_train.iloc[tr], y_train[tr])
        oof[va] = predict_proba_one(model, x_train.iloc[va])
    final = LGBMClassifier(**p)
    final.fit(x_train, y_train)
    return {"oof": oof, "train_fit": predict_proba_one(final, x_train), "test": predict_proba_one(final, x_test)}


def run_one(seed: int, feature_set: str, features: list[str], variant: str, data: Any) -> list[dict[str, Any]]:
    xtr = data.x_train[features].copy()
    xte = data.x_test[features].copy()
    pred = fit_oof(seed, variant, xtr, data.y_train, data.groups_train, xte)
    rows = []
    calibs = ["none", "platt"] if variant == "w125_raw" else ["none", "platt"]
    for calib in calibs:
        probs = apply_calibration(calib, data.y_train, pred["oof"], pred["train_fit"], pred["test"])
        for thr_name, thr in threshold_candidates(data.y_train, probs["oof"]).items():
            if not (thr_name.startswith("thr_fixed") or thr_name in {"thr_f1", "thr_acc", "thr_bacc"}):
                continue
            row = {
                "Seed": seed,
                "FeatureSet": feature_set,
                "Variant": variant,
                "Calibration": calib,
                "ThresholdRule": thr_name,
                "Threshold": thr,
                "N_Features": len(features),
                "Features": "|".join(features),
            }
            for domain, y, p in [("TrainFit", data.y_train, probs["train_fit"]), ("OOF", data.y_train, probs["oof"]), ("Test", data.y_test, probs["test"])]:
                for k, v in binary_metrics(y, p, thr).items():
                    row[f"{domain}_{k}"] = v
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "3m_binary_feature_prune_probe"))
    parser.add_argument("--source-dir", default=str(ROOT / "results" / "3m_binary_extreme_seedprobe"))
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--jobs", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = build_feature_frame(Path(args.source_dir).resolve(), force_rebuild=False)
    sets = make_feature_sets(data.stable_summary)
    pd.DataFrame([{"FeatureSet": k, "N": len(v), "Features": "|".join(v)} for k, v in sets.items()]).to_csv(out / "Pruned_Feature_Sets.csv", index=False)
    seeds = parse_seeds(args.seeds)
    variants = ["u_sub85", "w125_raw"]
    tasks = [(seed, fs, feats, variant) for seed in seeds for fs, feats in sets.items() for variant in variants]
    jobs = args.jobs if args.jobs > 0 else max(1, (os.cpu_count() or 2) - 1)
    print(f"[prune] tasks={len(tasks)} jobs={jobs}")
    chunks = Parallel(n_jobs=jobs, backend="loky", verbose=0)(
        delayed(run_one)(seed, fs, feats, variant, data) for seed, fs, feats, variant in tasks
    )
    df = pd.DataFrame([row for chunk in chunks for row in chunk])
    df.to_csv(out / "FeaturePrune_AllRuns.csv", index=False)
    group_cols = ["FeatureSet", "Variant", "Calibration", "ThresholdRule", "N_Features", "Features"]
    agg = (
        df.groupby(group_cols)
        .agg(
            N=("Seed", "count"),
            Mean_TrainFit_AUC=("TrainFit_AUC", "mean"),
            Mean_OOF_AUC=("OOF_AUC", "mean"),
            Mean_OOF_Accuracy=("OOF_Accuracy", "mean"),
            Mean_Test_AUC=("Test_AUC", "mean"),
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
    agg.sort_values(["Mean_Test_Accuracy", "Mean_Test_AUC"], ascending=[False, False]).to_csv(out / "FeaturePrune_Aggregated_By_TestAcc.csv", index=False)
    agg.sort_values(["Mean_OOF_Accuracy", "Mean_Test_Accuracy"], ascending=[False, False]).to_csv(out / "FeaturePrune_Aggregated_By_OOFAcc.csv", index=False)
    print(agg.sort_values(["Mean_Test_Accuracy", "Mean_Test_AUC"], ascending=[False, False]).head(20)[["FeatureSet", "Variant", "Calibration", "ThresholdRule", "N_Features", "Mean_Test_Accuracy", "Mean_Test_AUC", "Mean_Test_PR_AUC", "Mean_Test_Brier", "Mean_Test_Recall", "Mean_Test_Specificity"]])


if __name__ == "__main__":
    main()
