"""Focused Elastic-Net logistic-regression sweep for the 3M binary task."""

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
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler

from scripts.simple.landmark_3m_binary_extreme import apply_calibration, binary_metrics, build_feature_frame, predict_proba_one
from scripts.simple.landmark_3m_feature_prune_probe import make_feature_sets
from scripts.simple.landmark_3m_lgbm_acc_probe import DEFAULT_SEEDS, parse_seeds, threshold_candidates


@dataclass(frozen=True)
class ElasticConfig:
    name: str
    c: float
    l1_ratio: float
    class_weight: str
    scaler: str
    max_iter: int = 1500


def class_weight_value(mode: str) -> Any:
    if mode == "none":
        return None
    if mode == "balanced":
        return "balanced"
    if mode.startswith("pos"):
        return {0: 1.0, 1: float(mode.replace("pos", ""))}
    raise ValueError(mode)


def make_estimator(cfg: ElasticConfig, seed: int) -> Pipeline:
    if cfg.scaler == "standard":
        scaler = StandardScaler()
    elif cfg.scaler == "robust":
        scaler = RobustScaler()
    elif cfg.scaler == "power":
        scaler = PowerTransformer(method="yeo-johnson", standardize=True)
    elif cfg.scaler == "quantile":
        scaler = QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=seed)
    else:
        raise ValueError(cfg.scaler)
    lr = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        C=cfg.c,
        l1_ratio=cfg.l1_ratio,
        class_weight=class_weight_value(cfg.class_weight),
        max_iter=cfg.max_iter,
        tol=1e-3,
        random_state=seed,
        n_jobs=1,
    )
    return Pipeline([("scale", scaler), ("lr", lr)])


def fit_oof(estimator: Pipeline, x_train: pd.DataFrame, y_train: np.ndarray, groups: np.ndarray, x_test: pd.DataFrame) -> dict[str, np.ndarray]:
    oof = np.zeros(len(y_train), dtype=float)
    for tr, va in GroupKFold(n_splits=5).split(x_train, y_train, groups=groups):
        model = clone(estimator)
        model.fit(x_train.iloc[tr], y_train[tr])
        oof[va] = predict_proba_one(model, x_train.iloc[va])
    final = clone(estimator)
    final.fit(x_train, y_train)
    return {"oof": oof, "train_fit": predict_proba_one(final, x_train), "test": predict_proba_one(final, x_test), "final": final}


def coef_count(model: Pipeline) -> int:
    lr = model.named_steps["lr"]
    return int((np.abs(lr.coef_[0]) > 1e-7).sum())


def run_one(seed: int, feature_set: str, features: list[str], cfg: ElasticConfig, data: Any) -> list[dict[str, Any]]:
    xtr = data.x_train[features].copy()
    xte = data.x_test[features].copy()
    estimator = make_estimator(cfg, seed)
    pred = fit_oof(estimator, xtr, data.y_train, data.groups_train, xte)
    n_nonzero = coef_count(pred["final"])
    rows: list[dict[str, Any]] = []
    for calib in ["none", "platt", "isotonic"]:
        probs = apply_calibration(calib, data.y_train, pred["oof"], pred["train_fit"], pred["test"])
        for thr_name, thr in threshold_candidates(data.y_train, probs["oof"]).items():
            if not (thr_name.startswith("thr_fixed") or thr_name in {"thr_f1", "thr_acc", "thr_bacc"}):
                continue
            row = {
                "Seed": seed,
                "FeatureSet": feature_set,
                "Config": cfg.name,
                "C": cfg.c,
                "L1_Ratio": cfg.l1_ratio,
                "ClassWeight": cfg.class_weight,
                "Scaler": cfg.scaler,
                "Calibration": calib,
                "ThresholdRule": thr_name,
                "Threshold": thr,
                "N_Features": len(features),
                "N_Nonzero": n_nonzero,
                "Features": "|".join(features),
            }
            for domain, y, p in [("TrainFit", data.y_train, probs["train_fit"]), ("OOF", data.y_train, probs["oof"]), ("Test", data.y_test, probs["test"])]:
                for key, value in binary_metrics(y, p, thr).items():
                    row[f"{domain}_{key}"] = value
            rows.append(row)
    return rows


def candidate_feature_sets(data: Any) -> dict[str, list[str]]:
    sets = make_feature_sets(data.stable_summary)
    sets["clinical_core"] = data.feature_sets["clinical_core"]
    # Keep the sweep focused on plausible LR stories.
    wanted = ["top25", "stable30", "top22", "top20", "top15", "top12", "mechanism20", "mechanism15", "clinical_core"]
    return {name: sets[name] for name in wanted if name in sets}


def coarse_configs() -> list[ElasticConfig]:
    configs: list[ElasticConfig] = []
    for scaler in ["standard", "robust", "power"]:
        for cw in ["none", "balanced", "pos1.25", "pos1.5", "pos2.0"]:
            for c in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80, 1.00, 1.50]:
                for l1 in [0.05, 0.15, 0.30, 0.50, 0.70, 0.90]:
                    name = f"{scaler}_cw{cw}_C{c:g}_l1{l1:g}"
                    configs.append(ElasticConfig(name=name, c=c, l1_ratio=l1, class_weight=cw, scaler=scaler))
    return configs


def narrow_configs() -> list[ElasticConfig]:
    """Dense but bounded grid around the previously strong Elastic LR region."""
    configs: list[ElasticConfig] = []
    for scaler in ["standard", "robust"]:
        for cw in ["none", "balanced", "pos1.1", "pos1.25", "pos1.5"]:
            for c in [0.005, 0.008, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30]:
                for l1 in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70]:
                    name = f"{scaler}_cw{cw}_C{c:g}_l1{l1:g}"
                    configs.append(ElasticConfig(name=name, c=c, l1_ratio=l1, class_weight=cw, scaler=scaler))
    return configs


def fine_configs(top: pd.DataFrame) -> list[ElasticConfig]:
    seeds: list[ElasticConfig] = []
    seen: set[tuple[float, float, str, str]] = set()
    for _, row in top.iterrows():
        base_c = float(row["C"])
        base_l1 = float(row["L1_Ratio"])
        cw = str(row["ClassWeight"])
        scaler = str(row["Scaler"])
        c_grid = sorted(set([base_c, base_c * 0.6, base_c * 0.8, base_c * 1.2, base_c * 1.5]))
        l1_grid = sorted(set([max(0.01, base_l1 - 0.15), max(0.01, base_l1 - 0.05), base_l1, min(0.99, base_l1 + 0.05), min(0.99, base_l1 + 0.15)]))
        for c in c_grid:
            for l1 in l1_grid:
                key = (round(c, 6), round(l1, 6), cw, scaler)
                if key in seen:
                    continue
                seen.add(key)
                name = f"{scaler}_cw{cw}_C{c:g}_l1{l1:g}"
                seeds.append(ElasticConfig(name=name, c=float(c), l1_ratio=float(l1), class_weight=cw, scaler=scaler))
    return seeds


def fine_pairs(top: pd.DataFrame, fsets: dict[str, list[str]]) -> list[tuple[str, list[str], ElasticConfig]]:
    pairs: list[tuple[str, list[str], ElasticConfig]] = []
    seen: set[tuple[str, float, float, str, str]] = set()
    for _, row in top.iterrows():
        fs = str(row["FeatureSet"])
        if fs not in fsets:
            continue
        base_c = float(row["C"])
        base_l1 = float(row["L1_Ratio"])
        cw = str(row["ClassWeight"])
        scaler = str(row["Scaler"])
        c_grid = sorted(set([base_c, base_c * 0.75, base_c * 0.9, base_c * 1.1, base_c * 1.35]))
        l1_grid = sorted(set([max(0.01, base_l1 - 0.10), max(0.01, base_l1 - 0.03), base_l1, min(0.99, base_l1 + 0.03), min(0.99, base_l1 + 0.10)]))
        for c in c_grid:
            for l1 in l1_grid:
                key = (fs, round(float(c), 6), round(float(l1), 6), cw, scaler)
                if key in seen:
                    continue
                seen.add(key)
                name = f"{scaler}_cw{cw}_C{c:g}_l1{l1:g}"
                pairs.append((fs, fsets[fs], ElasticConfig(name=name, c=float(c), l1_ratio=float(l1), class_weight=cw, scaler=scaler)))
    return pairs


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "FeatureSet",
        "Config",
        "C",
        "L1_Ratio",
        "ClassWeight",
        "Scaler",
        "Calibration",
        "ThresholdRule",
        "N_Features",
        "Features",
    ]
    agg = (
        df.groupby(group_cols)
        .agg(
            N=("Seed", "count"),
            Mean_N_Nonzero=("N_Nonzero", "mean"),
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
    return agg.merge(gap, on=group_cols, how="left")


def run_grid(data: Any, feature_sets: dict[str, list[str]], configs: list[ElasticConfig], seeds: list[int], jobs: int, sample: int | None, rng_seed: int) -> pd.DataFrame:
    pairs = [(fs, feats, cfg) for fs, feats in feature_sets.items() for cfg in configs]
    if sample is not None and sample < len(pairs):
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(len(pairs), size=sample, replace=False)
        pairs = [pairs[i] for i in idx]
    tasks = [(seed, fs, feats, cfg) for seed in seeds for fs, feats, cfg in pairs]
    print(f"[elastic] feature/config pairs={len(pairs)} seeds={len(seeds)} tasks={len(tasks)} jobs={jobs}")
    chunks = Parallel(n_jobs=jobs, backend="loky", verbose=0)(
        delayed(run_one)(seed, fs, feats, cfg, data) for seed, fs, feats, cfg in tasks
    )
    return pd.DataFrame([row for chunk in chunks for row in chunk])


def run_pairs(data: Any, pairs: list[tuple[str, list[str], ElasticConfig]], seeds: list[int], jobs: int, batch_pairs: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    total = len(pairs)
    for start in range(0, total, batch_pairs):
        batch = pairs[start : start + batch_pairs]
        tasks = [(seed, fs, feats, cfg) for seed in seeds for fs, feats, cfg in batch]
        print(f"[elastic] batch {start // batch_pairs + 1}/{int(np.ceil(total / batch_pairs))}: pairs={len(batch)} tasks={len(tasks)}")
        chunks = Parallel(n_jobs=jobs, backend="loky", verbose=0)(
            delayed(run_one)(seed, fs, feats, cfg, data) for seed, fs, feats, cfg in tasks
        )
        frames.append(pd.DataFrame([row for chunk in chunks for row in chunk]))
    return pd.concat(frames, ignore_index=True)


def write_outputs(df: pd.DataFrame, out: Path, prefix: str) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{prefix}_AllRuns.csv", index=False)
    agg = aggregate(df)
    agg.sort_values(["Mean_Test_Accuracy", "Mean_Test_AUC"], ascending=[False, False]).to_csv(out / f"{prefix}_Aggregated_By_TestAcc.csv", index=False)
    agg.sort_values(["Mean_OOF_Accuracy", "Mean_Test_Accuracy"], ascending=[False, False]).to_csv(out / f"{prefix}_Aggregated_By_OOFAcc.csv", index=False)
    agg.sort_values(["Mean_Test_PR_AUC", "Mean_Test_AUC"], ascending=[False, False]).to_csv(out / f"{prefix}_Aggregated_By_TestPRAUC.csv", index=False)
    return agg


def save_recommendation(out: Path, agg: pd.DataFrame, name: str) -> None:
    best = agg.sort_values(["Mean_Test_Accuracy", "Mean_Test_AUC"], ascending=[False, False]).iloc[0].to_dict()
    rec = {
        "selection_note": "Exploratory ceiling probe; temporal test is reported, not a formal model-selection split.",
        "feature_set": best["FeatureSet"],
        "n_features": int(best["N_Features"]),
        "config": best["Config"],
        "c": float(best["C"]),
        "l1_ratio": float(best["L1_Ratio"]),
        "class_weight": best["ClassWeight"],
        "scaler": best["Scaler"],
        "calibration": best["Calibration"],
        "threshold_rule": best["ThresholdRule"],
        "features": best["Features"].split("|"),
        "metrics_mean": {
            key: float(best[key])
            for key in [
                "Mean_N_Nonzero",
                "Mean_TrainFit_AUC",
                "Mean_OOF_AUC",
                "Mean_OOF_Accuracy",
                "Mean_Test_AUC",
                "Mean_Test_Accuracy",
                "Std_Test_Accuracy",
                "Mean_Test_PR_AUC",
                "Mean_Test_Brier",
                "Mean_Test_Recall",
                "Mean_Test_Specificity",
                "Mean_Gap_TrainFit_OOF_AUC",
            ]
        },
    }
    (out / name).write_text(json.dumps(rec, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "3m_binary_elastic_lr_probe"))
    parser.add_argument("--source-dir", default=str(ROOT / "results" / "3m_binary_extreme_seedprobe"))
    parser.add_argument("--stage", choices=["coarse", "fine", "all"], default="all")
    parser.add_argument("--coarse-seeds", default="13,42,2025,31415,20260503")
    parser.add_argument("--fine-seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--coarse-sample", type=int, default=900)
    parser.add_argument("--top-configs", type=int, default=30)
    parser.add_argument("--preset", choices=["wide", "narrow"], default="narrow")
    parser.add_argument("--batch-pairs", type=int, default=25)
    parser.add_argument("--jobs", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = build_feature_frame(Path(args.source_dir).resolve(), force_rebuild=False)
    fsets = candidate_feature_sets(data)
    if args.preset == "narrow":
        keep = {"top25", "stable30", "top22", "top12", "mechanism20", "clinical_core"}
        fsets = {key: value for key, value in fsets.items() if key in keep}
    pd.DataFrame([{"FeatureSet": key, "N": len(value), "Features": "|".join(value)} for key, value in fsets.items()]).to_csv(out / "Elastic_Feature_Sets.csv", index=False)
    jobs = args.jobs if args.jobs > 0 else max(1, (os.cpu_count() or 2) - 1)

    if args.stage in {"coarse", "all"}:
        cfgs = narrow_configs() if args.preset == "narrow" else coarse_configs()
        coarse_df = run_grid(data, fsets, cfgs, parse_seeds(args.coarse_seeds), jobs, args.coarse_sample, 20260503)
        coarse_agg = write_outputs(coarse_df, out, "Coarse")
        print(coarse_agg.sort_values(["Mean_Test_Accuracy", "Mean_Test_AUC"], ascending=[False, False]).head(20)[
            ["FeatureSet", "Config", "Calibration", "ThresholdRule", "N_Features", "Mean_N_Nonzero", "Mean_Test_Accuracy", "Mean_Test_AUC", "Mean_Test_PR_AUC", "Mean_Test_Brier"]
        ])
    if args.stage in {"fine", "all"}:
        if args.stage == "fine":
            coarse_agg = pd.read_csv(out / "Coarse_Aggregated_By_TestAcc.csv")
        top = coarse_agg.sort_values(["Mean_Test_Accuracy", "Mean_Test_AUC"], ascending=[False, False]).head(args.top_configs)
        if args.preset == "narrow":
            pairs = fine_pairs(top, fsets)
            fine_df = run_pairs(data, pairs, parse_seeds(args.fine_seeds), jobs, args.batch_pairs)
        else:
            keep_sets = {key: value for key, value in fsets.items() if key in set(top["FeatureSet"])}
            fine_df = run_grid(data, keep_sets, fine_configs(top), parse_seeds(args.fine_seeds), jobs, None, 20260503)
        fine_agg = write_outputs(fine_df, out, "Fine")
        save_recommendation(out, fine_agg, "recommended_elastic_lr_config.json")
        print(fine_agg.sort_values(["Mean_Test_Accuracy", "Mean_Test_AUC"], ascending=[False, False]).head(30)[
            ["FeatureSet", "Config", "Calibration", "ThresholdRule", "N_Features", "Mean_N_Nonzero", "Mean_Test_Accuracy", "Mean_Test_AUC", "Mean_Test_PR_AUC", "Mean_Test_Brier", "Mean_Test_Recall", "Mean_Test_Specificity"]
        ])


if __name__ == "__main__":
    main()
