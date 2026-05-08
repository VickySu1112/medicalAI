"""Focused probes around the best 3M binary stable-selected LGBM path.

Adds stronger regularization, train-only feature-noise augmentation, and an
error-notebook cascade probe. Outputs are isolated under results/.
"""

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from scripts.simple.landmark_3m_binary_extreme import (
    apply_calibration,
    best_thresholds,
    binary_metrics,
    build_feature_frame,
    df_to_markdown,
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
class ProbeConfig:
    name: str
    params: dict[str, Any]
    noise_ratio: float = 0.0
    noise_std: float = 0.0
    noise_pos_only: bool = False


def parse_seeds(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def lgbm_params(seed: int, **overrides: Any) -> dict[str, Any]:
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


def probe_configs(seed: int) -> list[ProbeConfig]:
    return [
        ProbeConfig("base_d3_leaf11", lgbm_params(seed)),
        ProbeConfig("reg_d2_leaf7_l5", lgbm_params(seed, n_estimators=260, learning_rate=0.025, num_leaves=7, max_depth=2, min_child_samples=24, subsample=0.75, colsample_bytree=0.75, reg_alpha=0.5, reg_lambda=5.0)),
        ProbeConfig("reg_d3_leaf9_l8", lgbm_params(seed, n_estimators=300, learning_rate=0.022, num_leaves=9, max_depth=3, min_child_samples=26, subsample=0.70, colsample_bytree=0.70, reg_alpha=0.8, reg_lambda=8.0)),
        ProbeConfig("reg_d3_leaf7_l12", lgbm_params(seed, n_estimators=280, learning_rate=0.020, num_leaves=7, max_depth=3, min_child_samples=30, subsample=0.65, colsample_bytree=0.70, reg_alpha=1.0, reg_lambda=12.0)),
        ProbeConfig("reg_d4_leaf15_l6", lgbm_params(seed, n_estimators=360, learning_rate=0.020, num_leaves=15, max_depth=4, min_child_samples=24, subsample=0.70, colsample_bytree=0.70, reg_alpha=0.8, reg_lambda=6.0)),
        ProbeConfig("noise_all_r05_s002", lgbm_params(seed), noise_ratio=0.5, noise_std=0.02),
        ProbeConfig("noise_all_r10_s002", lgbm_params(seed), noise_ratio=1.0, noise_std=0.02),
        ProbeConfig("noise_all_r05_s004", lgbm_params(seed, reg_lambda=3.0), noise_ratio=0.5, noise_std=0.04),
        ProbeConfig("noise_pos_r10_s003", lgbm_params(seed), noise_ratio=1.0, noise_std=0.03, noise_pos_only=True),
        ProbeConfig("noise_pos_r15_s004", lgbm_params(seed, reg_lambda=4.0), noise_ratio=1.5, noise_std=0.04, noise_pos_only=True),
    ]


def augment_train(
    x: pd.DataFrame,
    y: np.ndarray,
    cfg: ProbeConfig,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray]:
    if cfg.noise_ratio <= 0 or cfg.noise_std <= 0:
        return x, y
    base_idx = np.where(y == 1)[0] if cfg.noise_pos_only else np.arange(len(y))
    n_aug = max(1, int(round(len(base_idx) * cfg.noise_ratio)))
    pick = rng.choice(base_idx, size=n_aug, replace=True)
    std = x.std(axis=0).replace(0.0, 1.0).fillna(1.0).to_numpy()
    q01 = x.quantile(0.01).to_numpy()
    q99 = x.quantile(0.99).to_numpy()
    arr = x.iloc[pick].to_numpy(dtype=float)
    noise = rng.normal(0.0, cfg.noise_std, size=arr.shape) * std
    aug = np.clip(arr + noise, q01, q99)
    x_aug = pd.concat([x, pd.DataFrame(aug, columns=x.columns)], axis=0, ignore_index=True)
    y_aug = np.concatenate([y, y[pick]])
    return x_aug, y_aug


def fit_oof(cfg: ProbeConfig, seed: int, x_train: pd.DataFrame, y_train: np.ndarray, groups: np.ndarray, x_test: pd.DataFrame) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    oof = np.zeros(len(y_train), dtype=float)
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=5).split(x_train, y_train, groups=groups)):
        model = LGBMClassifier(**cfg.params)
        x_aug, y_aug = augment_train(x_train.iloc[tr].reset_index(drop=True), y_train[tr], cfg, np.random.default_rng(seed + fold * 1009))
        model.fit(x_aug, y_aug)
        oof[va] = predict_proba_one(model, x_train.iloc[va])
    final = LGBMClassifier(**cfg.params)
    x_full_aug, y_full_aug = augment_train(x_train.reset_index(drop=True), y_train, cfg, rng)
    final.fit(x_full_aug, y_full_aug)
    return {
        "oof": oof,
        "train_fit": predict_proba_one(final, x_train),
        "test": predict_proba_one(final, x_test),
        "final_model": final,
    }


def rows_for_probs(
    seed: int,
    cfg_name: str,
    calib: str,
    y_train: np.ndarray,
    y_test: np.ndarray,
    probs: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    thrs = best_thresholds(y_train, probs["oof"])
    rows = []
    for thr_name, thr in thrs.items():
        row = {
            "Seed": seed,
            "Config": cfg_name,
            "Calibration": calib,
            "ThresholdRule": thr_name,
            "Threshold": thr,
        }
        for domain, y, p in [("TrainFit", y_train, probs["train_fit"]), ("OOF", y_train, probs["oof"]), ("Test", y_test, probs["test"])]:
            metrics = binary_metrics(y, p, thr)
            for k, v in metrics.items():
                row[f"{domain}_{k}"] = v
        rows.append(row)
    return rows


def run_one(seed: int, cfg: ProbeConfig, x_train: pd.DataFrame, y_train: np.ndarray, groups: np.ndarray, x_test: pd.DataFrame, y_test: np.ndarray) -> list[dict[str, Any]]:
    pred = fit_oof(cfg, seed, x_train, y_train, groups, x_test)
    rows: list[dict[str, Any]] = []
    for calib in ["none", "platt", "isotonic"]:
        probs = apply_calibration(calib, y_train, pred["oof"], pred["train_fit"], pred["test"])
        rows.extend(rows_for_probs(seed, cfg.name, calib, y_train, y_test, probs))
    return rows


def run_regularization_probe(args: argparse.Namespace, x_train: pd.DataFrame, y_train: np.ndarray, groups: np.ndarray, x_test: pd.DataFrame, y_test: np.ndarray, out: Path) -> pd.DataFrame:
    seeds = parse_seeds(args.seeds)
    tasks = [(seed, cfg) for seed in seeds for cfg in probe_configs(seed)]
    jobs = args.jobs if args.jobs > 0 else max(1, (os.cpu_count() or 2) - 1)
    print(f"[probe] regularization/noise tasks={len(tasks)} jobs={jobs}")
    results = Parallel(n_jobs=jobs, backend="loky", verbose=0)(
        delayed(run_one)(seed, cfg, x_train, y_train, groups, x_test, y_test) for seed, cfg in tasks
    )
    rows = [row for chunk in results for row in chunk]
    df = pd.DataFrame(rows)
    df.to_csv(out / "Regularization_Augmentation_AllRuns.csv", index=False)
    group_cols = ["Config", "Calibration", "ThresholdRule"]
    agg = (
        df.groupby(group_cols)
        .agg(
            N=("Seed", "count"),
            Mean_OOF_AUC=("OOF_AUC", "mean"),
            Mean_OOF_PR_AUC=("OOF_PR_AUC", "mean"),
            Mean_OOF_Accuracy=("OOF_Accuracy", "mean"),
            Mean_Test_AUC=("Test_AUC", "mean"),
            Std_Test_AUC=("Test_AUC", "std"),
            Mean_Test_PR_AUC=("Test_PR_AUC", "mean"),
            Mean_Test_Accuracy=("Test_Accuracy", "mean"),
            Mean_Test_Brier=("Test_Brier", "mean"),
            Mean_TrainFit_AUC=("TrainFit_AUC", "mean"),
        )
        .reset_index()
    )
    agg["Mean_Gap_TrainFit_OOF_AUC"] = (
        df.assign(Gap=lambda d: d["TrainFit_AUC"] - d["OOF_AUC"])
        .groupby(group_cols)["Gap"]
        .mean()
        .values
    )
    agg = agg.sort_values(["Mean_OOF_AUC", "Mean_Test_AUC"], ascending=[False, False])
    agg.to_csv(out / "Regularization_Augmentation_Aggregated.csv", index=False)
    agg.sort_values(["Mean_Test_AUC", "Mean_Test_Accuracy"], ascending=[False, False]).head(30).to_csv(out / "Top_TEST_AWARE_RegAug_By_Test_AUC.csv", index=False)
    agg.sort_values(["Mean_Test_PR_AUC", "Mean_Test_AUC"], ascending=[False, False]).head(30).to_csv(out / "Top_TEST_AWARE_RegAug_By_Test_PR_AUC.csv", index=False)
    return df


def exact_base_predictions(seed: int, x_train: pd.DataFrame, y_train: np.ndarray, groups: np.ndarray, x_test: pd.DataFrame) -> tuple[dict[str, np.ndarray], float]:
    cfg = ProbeConfig("base_d3_leaf11", lgbm_params(seed))
    pred = fit_oof(cfg, seed, x_train, y_train, groups, x_test)
    probs = apply_calibration("isotonic", y_train, pred["oof"], pred["train_fit"], pred["test"])
    thr = best_thresholds(y_train, probs["oof"])["thr_f1"]
    return probs, thr


def save_error_notebook(
    out: Path,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    pids_test: np.ndarray,
    probs: dict[str, np.ndarray],
    thr: float,
) -> pd.DataFrame:
    pred = (probs["test"] >= thr).astype(int)
    err_type = np.where((y_test == 1) & (pred == 1), "TP", np.where((y_test == 0) & (pred == 0), "TN", np.where(pred == 1, "FP", "FN")))
    cases = x_test.copy()
    cases.insert(0, "Patient_ID", pids_test)
    cases.insert(1, "Y", y_test)
    cases.insert(2, "Pred", pred)
    cases.insert(3, "Proba", probs["test"])
    cases.insert(4, "ErrorType", err_type)
    cases.to_csv(out / "Error_Notebook_Test_Cases.csv", index=False)

    std = x_train.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    rows = []
    comparisons = [("FP", "TN"), ("FN", "TP"), ("Errors", "Correct")]
    for a, b in comparisons:
        if a == "Errors":
            mask_a = np.isin(err_type, ["FP", "FN"])
            mask_b = np.isin(err_type, ["TP", "TN"])
        else:
            mask_a = err_type == a
            mask_b = err_type == b
        if mask_a.sum() < 2 or mask_b.sum() < 2:
            continue
        mean_a = x_test.loc[mask_a].mean(axis=0)
        mean_b = x_test.loc[mask_b].mean(axis=0)
        smd = (mean_a - mean_b) / std
        for feat in x_test.columns:
            rows.append(
                {
                    "Comparison": f"{a}_vs_{b}",
                    "Feature": feat,
                    "Mean_A": mean_a[feat],
                    "Mean_B": mean_b[feat],
                    "StdMeanDiff": smd[feat],
                    "AbsStdMeanDiff": abs(smd[feat]),
                    "N_A": int(mask_a.sum()),
                    "N_B": int(mask_b.sum()),
                }
            )
    diff = pd.DataFrame(rows).sort_values(["Comparison", "AbsStdMeanDiff"], ascending=[True, False])
    diff.to_csv(out / "Error_Notebook_Feature_Differences.csv", index=False)

    top = diff.groupby("Comparison").head(12)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.22 * len(top))))
    plot = top.copy()
    plot["Label"] = plot["Comparison"] + " | " + plot["Feature"]
    sns.barplot(data=plot, y="Label", x="StdMeanDiff", hue="Comparison", dodge=False, ax=ax)
    ax.axvline(0, color="black", lw=1)
    ax.set_title("Error Notebook: Feature Shifts In Misclassified Cases")
    ax.set_xlabel("Standardized mean difference")
    ax.set_ylabel("")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "Figure_Error_Notebook_Feature_Shifts.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return cases


def cascade_probe(
    seed: int,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    groups: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    out: Path,
) -> pd.DataFrame:
    base_cfg = ProbeConfig("base_d3_leaf11", lgbm_params(seed))
    windows = [0.08, 0.12, 0.16, 0.20]
    alphas = [0.35, 0.50, 0.65, 0.80]
    rows = []
    for window in windows:
        for alpha in alphas:
            oof = np.zeros(len(y_train), dtype=float)
            train_fit = np.zeros(len(y_train), dtype=float)
            for fold, (tr, va) in enumerate(GroupKFold(n_splits=5).split(x_train, y_train, groups=groups)):
                base = LGBMClassifier(**base_cfg.params)
                base.fit(x_train.iloc[tr], y_train[tr])
                p_tr = predict_proba_one(base, x_train.iloc[tr])
                p_va = predict_proba_one(base, x_train.iloc[va])
                thr_tr = best_thresholds(y_train[tr], p_tr)["thr_f1"]
                hard_tr = (np.abs(p_tr - thr_tr) <= window) | ((p_tr >= thr_tr).astype(int) != y_train[tr])
                if hard_tr.sum() >= 30 and len(np.unique(y_train[tr][hard_tr])) == 2:
                    spec = LGBMClassifier(**lgbm_params(seed + fold + 71, n_estimators=180, learning_rate=0.025, num_leaves=7, max_depth=2, min_child_samples=8, reg_lambda=4.0))
                    spec.fit(x_train.iloc[tr].iloc[hard_tr], y_train[tr][hard_tr])
                    p_spec = predict_proba_one(spec, x_train.iloc[va])
                    gate = np.abs(p_va - thr_tr) <= window
                    p_va = np.where(gate, (1 - alpha) * p_va + alpha * p_spec, p_va)
                oof[va] = p_va
            # final train/test
            base = LGBMClassifier(**base_cfg.params)
            base.fit(x_train, y_train)
            p_train = predict_proba_one(base, x_train)
            p_test = predict_proba_one(base, x_test)
            thr = best_thresholds(y_train, oof)["thr_f1"]
            hard = (np.abs(oof - thr) <= window) | ((oof >= thr).astype(int) != y_train)
            gate_rate = float(np.mean(np.abs(p_test - thr) <= window))
            if hard.sum() >= 30 and len(np.unique(y_train[hard])) == 2:
                spec = LGBMClassifier(**lgbm_params(seed + 1701, n_estimators=200, learning_rate=0.025, num_leaves=7, max_depth=2, min_child_samples=8, reg_lambda=4.0))
                spec.fit(x_train.loc[hard], y_train[hard])
                p_spec_train = predict_proba_one(spec, x_train)
                p_spec_test = predict_proba_one(spec, x_test)
                gate_train = np.abs(p_train - thr) <= window
                gate_test = np.abs(p_test - thr) <= window
                p_train = np.where(gate_train, (1 - alpha) * p_train + alpha * p_spec_train, p_train)
                p_test = np.where(gate_test, (1 - alpha) * p_test + alpha * p_spec_test, p_test)
            probs = {"oof": oof, "train_fit": p_train, "test": p_test}
            for row in rows_for_probs(seed, f"cascade_w{window}_a{alpha}", "none", y_train, y_test, probs):
                row["Window"] = window
                row["Alpha"] = alpha
                row["HardTrainRate"] = float(np.mean(hard))
                row["TestGateRate"] = gate_rate
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "Error_Notebook_Cascade_Probe.csv", index=False)
    return df


def write_summary(out: Path, reg_df: pd.DataFrame, cascade_df: pd.DataFrame) -> None:
    agg = pd.read_csv(out / "Regularization_Augmentation_Aggregated.csv")
    lines = [
        "# 3M Stable-Selected Regularization, Augmentation, And Error Notebook Probe",
        "",
        "## Regularization / Train-Only Noise",
        "",
        "Top by mean temporal-test AUC:",
        "",
        df_to_markdown(agg.sort_values(["Mean_Test_AUC", "Mean_Test_Accuracy"], ascending=[False, False]).head(10)),
        "",
        "Top by mean temporal-test PR-AUC:",
        "",
        df_to_markdown(agg.sort_values(["Mean_Test_PR_AUC", "Mean_Test_AUC"], ascending=[False, False]).head(10)),
        "",
        "## Error-Notebook Cascade",
        "",
        df_to_markdown(cascade_df.sort_values(["Test_AUC", "Test_Accuracy"], ascending=[False, False]).head(10)),
    ]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "3m_binary_extreme_probe2"))
    parser.add_argument("--source-dir", default=str(ROOT / "results" / "3m_binary_extreme_seedprobe"))
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--skip-reg", action="store_true")
    args = parser.parse_args()

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = build_feature_frame(Path(args.source_dir).resolve(), force_rebuild=False)
    features = data.feature_sets["stable_selected"]
    x_train = data.x_train[features].copy()
    x_test = data.x_test[features].copy()

    if args.skip_reg:
        reg_df = pd.DataFrame()
    else:
        reg_df = run_regularization_probe(args, x_train, data.y_train, data.groups_train, x_test, data.y_test, out)
    probs, thr = exact_base_predictions(20265, x_train, data.y_train, data.groups_train, x_test)
    save_error_notebook(out, x_train, data.y_train, x_test, data.y_test, data.pids_test, probs, thr)
    cascade_df = cascade_probe(20265, x_train, data.y_train, data.groups_train, x_test, data.y_test, out)
    if len(reg_df):
        write_summary(out, reg_df, cascade_df)
    print("[probe] done")
    if len(reg_df):
        print(pd.read_csv(out / "Regularization_Augmentation_Aggregated.csv").head(10)[["Config", "Calibration", "ThresholdRule", "Mean_OOF_AUC", "Mean_Test_AUC", "Mean_Test_Accuracy", "Mean_Test_PR_AUC"]])
    print(cascade_df.sort_values(["Test_AUC", "Test_Accuracy"], ascending=[False, False]).head(10)[["Config", "ThresholdRule", "OOF_AUC", "Test_AUC", "Test_Accuracy", "Test_PR_AUC", "TestGateRate"]])


if __name__ == "__main__":
    main()
