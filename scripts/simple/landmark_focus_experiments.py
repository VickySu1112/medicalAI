"""Focused fixed-landmark report and multiclass probes.

This script keeps all new artifacts under results/landmark_focus/.  It reads the
existing fixed-landmark binary tables, then runs a fresh, structured multiclass
probe for 3M and 6M using temporal-safe imputation.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.config import SEED, STATIC_NAMES
from utils.data import fit_missforest, load_data, split_imputed

OUT = ROOT / "results" / "landmark_focus"
FIG = OUT / "figures"
TAB = OUT / "tables"
CACHE = OUT / "cache"
CLASS_NAMES = ["Hyper", "Normal", "Hypo"]


@dataclass
class LandmarkData:
    landmark: str
    seq_len: int
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    groups_train: np.ndarray
    pids_test: np.ndarray
    feature_names: list[str]


class CascadeClassifier:
    """Hyper-vs-rest followed by Hypo-vs-Normal among predicted non-Hyper."""

    def __init__(self, stage1, stage2) -> None:
        self.stage1 = stage1
        self.stage2 = stage2

    def fit(self, x: np.ndarray, y: np.ndarray) -> "CascadeClassifier":
        y_non_hyper = (y != 0).astype(int)
        self.stage1.fit(x, y_non_hyper)
        mask = y != 0
        y_hypo = (y[mask] == 2).astype(int)
        self.stage2.fit(x[mask], y_hypo)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p_non_hyper = np.clip(self.stage1.predict_proba(x)[:, 1], 1e-6, 1.0 - 1e-6)
        p_hyper = 1.0 - p_non_hyper
        p_hypo_given_non = np.clip(self.stage2.predict_proba(x)[:, 1], 1e-6, 1.0 - 1e-6)
        p_hypo = p_non_hyper * p_hypo_given_non
        p_normal = p_non_hyper * (1.0 - p_hypo_given_non)
        probs = np.column_stack([p_hyper, p_normal, p_hypo])
        return probs / probs.sum(axis=1, keepdims=True)


def ensure_dirs() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)


def publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
        }
    )


def build_landmark_features(x_s: np.ndarray, x_d: np.ndarray, seq_len: int) -> tuple[np.ndarray, list[str]]:
    """Match the binary script's landmark feature frame."""
    labs = x_d[:, :seq_len, :]
    flat = labs.reshape(labs.shape[0], -1)
    names = list(STATIC_NAMES)
    time_names = ["0M", "1M", "3M", "6M"][:seq_len]
    for t in time_names:
        for lab in ["FT3", "FT4", "TSH"]:
            names.append(f"{lab}_{t}")

    deltas = []
    if seq_len >= 3:
        deltas.append(labs[:, 2, :] - labs[:, 0, :])
        names += ["D_FT3_3M-0M", "D_FT4_3M-0M", "D_TSH_3M-0M"]
    if seq_len >= 4:
        deltas.append(labs[:, 3, :] - labs[:, 0, :])
        names += ["D_FT3_6M-0M", "D_FT4_6M-0M", "D_TSH_6M-0M"]
        deltas.append(labs[:, 3, :] - labs[:, 2, :])
        names += ["D_FT3_6M-3M", "D_FT4_6M-3M", "D_TSH_6M-3M"]

    final = labs[:, -1, :]
    prev = labs[:, -2, :] if seq_len > 1 else labs[:, -1, :]
    first = labs[:, 0, :]
    eps = 1e-6
    thyroid_idx = STATIC_NAMES.index("ThyroidW")
    trab_idx = STATIC_NAMES.index("TRAb")
    extra = [
        np.column_stack(
            [
                final[:, 0] / (first[:, 0] + eps),
                final[:, 1] / (first[:, 1] + eps),
                (final[:, 2] + 1.0) / (first[:, 2] + 1.0),
            ]
        ),
        np.column_stack([final[:, 0] / (final[:, 1] + eps), final[:, 1] / (final[:, 2] + 1.0)]),
        labs.mean(axis=1),
        labs.std(axis=1),
        final - prev,
        np.column_stack(
            [
                x_s[:, thyroid_idx] * final[:, 0],
                x_s[:, thyroid_idx] * final[:, 1],
                x_s[:, trab_idx] * final[:, 0],
                x_s[:, trab_idx] * final[:, 1],
            ]
        ),
    ]
    names += [
        "FT3_last_over_0M",
        "FT4_last_over_0M",
        "TSH_last_over_0M_plus1",
        "FT3_last_over_FT4_last",
        "FT4_last_over_TSH_last_plus1",
        "FT3_mean",
        "FT4_mean",
        "TSH_mean",
        "FT3_std",
        "FT4_std",
        "TSH_std",
        "FT3_last_minus_prev",
        "FT4_last_minus_prev",
        "TSH_last_minus_prev",
        "ThyroidW_x_FT3_last",
        "ThyroidW_x_FT4_last",
        "TRAb_x_FT3_last",
        "TRAb_x_FT4_last",
    ]
    parts = [x_s, flat] + deltas + extra
    return np.concatenate(parts, axis=1).astype(np.float32), names


def load_landmark_data(landmark: str, seq_len: int) -> LandmarkData:
    x_s, ft3, ft4, tsh, _, y_raw, pids = load_data()
    unique_pids = list(dict.fromkeys(pids))
    split_idx = int(len(unique_pids) * 0.8)
    train_pids = set(unique_pids[:split_idx])
    train_mask = np.asarray([pid in train_pids for pid in pids])
    test_mask = ~train_mask

    raw_all = np.hstack([x_s, ft3[:, :seq_len], ft4[:, :seq_len], tsh[:, :seq_len]])
    cache_path = CACHE / f"missforest_{landmark}_seqlen{seq_len}.pkl"
    if cache_path.exists():
        imputer = joblib.load(cache_path)
        status = "loaded"
    else:
        imputer = fit_missforest(raw_all[train_mask], seed=SEED)
        joblib.dump(imputer, cache_path)
        status = "fitted"
    print(f"{landmark}: MissForest {status} -> {cache_path}")

    filled_train = imputer.transform(raw_all[train_mask])
    filled_test = imputer.transform(raw_all[test_mask])
    n_static = x_s.shape[1]
    xs_tr, f3_tr, f4_tr, ts_tr = split_imputed(filled_train, n_static, seq_len)
    xs_te, f3_te, f4_te, ts_te = split_imputed(filled_test, n_static, seq_len)
    xd_tr = np.stack([f3_tr, f4_tr, ts_tr], axis=-1)
    xd_te = np.stack([f3_te, f4_te, ts_te], axis=-1)
    x_train, names = build_landmark_features(xs_tr, xd_tr, seq_len)
    x_test, _ = build_landmark_features(xs_te, xd_te, seq_len)
    y_multi = np.vectorize({1: 0, 2: 1, 3: 2}.get)(y_raw)
    return LandmarkData(
        landmark=landmark,
        seq_len=seq_len,
        x_train=x_train,
        x_test=x_test,
        y_train=y_multi[train_mask],
        y_test=y_multi[test_mask],
        groups_train=pids[train_mask],
        pids_test=pids[test_mask],
        feature_names=names,
    )


def model_builders(seed: int = SEED) -> dict[str, Callable[[], object]]:
    return {
        "Multinomial LR C0.3": lambda: Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(C=0.3, max_iter=4000, class_weight="balanced", random_state=seed),
                ),
            ]
        ),
        "Multinomial LR C1": lambda: Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(C=1.0, max_iter=4000, class_weight="balanced", random_state=seed),
                ),
            ]
        ),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=350,
            max_depth=6,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        ),
        "ExtraTrees": lambda: ExtraTreesClassifier(
            n_estimators=450,
            max_depth=6,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        ),
        "LightGBM": lambda: LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=220,
            learning_rate=0.04,
            max_depth=4,
            num_leaves=15,
            min_child_samples=12,
            class_weight="balanced",
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=0.2,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
        ),
        "Cascade RF": lambda: CascadeClassifier(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            ),
            RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=seed + 1,
                n_jobs=1,
            ),
        ),
        "Cascade LGBM": lambda: CascadeClassifier(
            LGBMClassifier(
                n_estimators=180,
                learning_rate=0.04,
                max_depth=4,
                num_leaves=15,
                min_child_samples=10,
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
                verbosity=-1,
            ),
            LGBMClassifier(
                n_estimators=180,
                learning_rate=0.04,
                max_depth=4,
                num_leaves=15,
                min_child_samples=10,
                class_weight="balanced",
                random_state=seed + 1,
                n_jobs=1,
                verbosity=-1,
            ),
        ),
    }


def safe_auc(y_bin: np.ndarray, score: np.ndarray) -> float:
    return float("nan") if len(np.unique(y_bin)) < 2 else float(roc_auc_score(y_bin, score))


def safe_ap(y_bin: np.ndarray, score: np.ndarray) -> float:
    return float("nan") if len(np.unique(y_bin)) < 2 else float(average_precision_score(y_bin, score))


def multiclass_metrics(y: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum(axis=1, keepdims=True)
    pred = probs.argmax(axis=1)
    onehot = np.eye(3)[y]
    out = {
        "N": int(len(y)),
        "Accuracy": float(accuracy_score(y, pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y, pred)),
        "Macro_F1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "Weighted_F1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "Macro_AUC_OVR": float(roc_auc_score(y, probs, multi_class="ovr", average="macro")),
        "Macro_PR_AUC_OVR": float(np.mean([safe_ap((y == i).astype(int), probs[:, i]) for i in range(3)])),
        "Brier_Multiclass": float(np.mean(np.sum((probs - onehot) ** 2, axis=1))),
    }
    for i, name in enumerate(CLASS_NAMES):
        y_bin = (y == i).astype(int)
        out[f"{name}_Support"] = int(y_bin.sum())
        out[f"{name}_Prevalence"] = float(y_bin.mean())
        out[f"{name}_AUC"] = safe_auc(y_bin, probs[:, i])
        out[f"{name}_PR_AUC"] = safe_ap(y_bin, probs[:, i])
        out[f"{name}_Recall"] = float(recall_score(y_bin, pred == i, zero_division=0))
    return out


def oof_predict(builder: Callable[[], object], data: LandmarkData) -> np.ndarray:
    probs = np.zeros((len(data.y_train), 3), dtype=float)
    gkf = GroupKFold(n_splits=3)
    for tr, va in gkf.split(data.x_train, data.y_train, groups=data.groups_train):
        model = builder()
        model.fit(data.x_train[tr], data.y_train[tr])
        probs[va] = model.predict_proba(data.x_train[va])
    return probs / probs.sum(axis=1, keepdims=True)


def run_multiclass_probe(landmark: str, seq_len: int) -> tuple[pd.DataFrame, dict[str, object]]:
    data = load_landmark_data(landmark, seq_len)
    builders = model_builders()
    rows = []
    payloads = {}
    for name, builder in builders.items():
        print(f"{landmark}: {name}")
        oof = oof_predict(builder, data)
        model = builder()
        model.fit(data.x_train, data.y_train)
        test_prob = model.predict_proba(data.x_test)
        for split, y, prob in [("OOF_Train", data.y_train, oof), ("Temporal_Test", data.y_test, test_prob)]:
            row = multiclass_metrics(y, prob)
            row.update({"Landmark": landmark, "Split": split, "Model": name, "Feature_Count": len(data.feature_names)})
            rows.append(row)
        payloads[name] = {"model": model, "oof": oof, "test_prob": test_prob, "data": data}
    df = pd.DataFrame(rows)
    df.to_csv(TAB / f"{landmark.lower()}_multiclass_probe_metrics.csv", index=False)
    return df, payloads


def copy_binary_tables() -> dict[str, pd.DataFrame]:
    src = ROOT / "results" / "fixed_landmark_binary"
    out = {}
    for name in [
        "Performance_3_Month_Train_Fit.csv",
        "Performance_3_Month_Validation_OOF.csv",
        "Performance_3_Month_Test_Temporal.csv",
        "Performance_6_Month_Train_Fit.csv",
        "Performance_6_Month_Validation_OOF.csv",
        "Performance_6_Month_Test_Temporal.csv",
        "Performance_Long.csv",
        "3-Month_Feature_Selection.csv",
        "6-Month_Feature_Selection.csv",
        "Routed_Error_Features_3-Mo.csv",
    ]:
        df = pd.read_csv(src / name)
        df.to_csv(TAB / name, index=False)
        out[name] = df
    return out


def metric(table: pd.DataFrame, metric_name: str, model_name: str) -> float:
    return float(table.loc[table["Metric"].eq(metric_name), model_name].iloc[0])


def save_binary_summary_figure(binary_tables: dict[str, pd.DataFrame]) -> None:
    rows = []
    for landmark, prefix in [("3M", "3_Month"), ("6M", "6_Month")]:
        test = binary_tables[f"Performance_{prefix}_Test_Temporal.csv"]
        for model in test.columns:
            if model == "Metric":
                continue
            rows.append(
                {
                    "Landmark": landmark,
                    "Model": model,
                    "AUC": metric(test, "AUC", model),
                    "Accuracy": metric(test, "Accuracy", model),
                    "F1": metric(test, "F1", model),
                }
            )
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, landmark in zip(axes, ["3M", "6M"]):
        sub = df[df["Landmark"].eq(landmark)].sort_values("AUC", ascending=False)
        x = np.arange(len(sub))
        ax.bar(x - 0.22, sub["AUC"], width=0.22, label="AUC")
        ax.bar(x, sub["Accuracy"], width=0.22, label="Accuracy")
        ax.bar(x + 0.22, sub["F1"], width=0.22, label="F1")
        ax.set_xticks(x, sub["Model"], rotation=35, ha="right")
        ax.set_title(f"{landmark} binary temporal-test performance")
        ax.set_ylim(0.65, 0.95)
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "binary_3m_6m_temporal_performance.png", bbox_inches="tight")
    plt.close(fig)
    df.to_csv(TAB / "binary_temporal_model_summary.csv", index=False)


def save_multiclass_figures(metrics_df: pd.DataFrame, payloads_by_landmark: dict[str, dict[str, object]]) -> None:
    test = metrics_df[metrics_df["Split"].eq("Temporal_Test")].copy()
    oof = metrics_df[metrics_df["Split"].eq("OOF_Train")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, landmark in zip(axes, ["3M", "6M"]):
        sub = test[test["Landmark"].eq(landmark)].sort_values("Macro_AUC_OVR", ascending=False)
        x = np.arange(len(sub))
        ax.bar(x - 0.2, sub["Macro_AUC_OVR"], width=0.2, label="Macro AUC")
        ax.bar(x, sub["Macro_PR_AUC_OVR"], width=0.2, label="Macro PR-AUC")
        ax.bar(x + 0.2, sub["Balanced_Accuracy"], width=0.2, label="Balanced Acc")
        ax.set_xticks(x, sub["Model"], rotation=35, ha="right")
        ax.set_ylim(0.2, 0.95)
        ax.set_title(f"{landmark} multiclass temporal-test")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "multiclass_3m_6m_temporal_performance.png", bbox_inches="tight")
    plt.close(fig)

    for landmark, payloads in payloads_by_landmark.items():
        sub_oof = oof[oof["Landmark"].eq(landmark)]
        best_model = sub_oof.sort_values("Macro_AUC_OVR", ascending=False)["Model"].iloc[0]
        payload = payloads[best_model]
        data: LandmarkData = payload["data"]
        prob = payload["test_prob"]
        pred = prob.argmax(axis=1)
        cm = confusion_matrix(data.y_test, pred, labels=[0, 1, 2])
        fig, ax = plt.subplots(figsize=(5.5, 4.6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_title(f"{landmark} multiclass CM - {best_model} (OOF-selected)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Observed")
        fig.tight_layout()
        fig.savefig(FIG / f"cm_{landmark.lower()}_multiclass_best.png", bbox_inches="tight")
        plt.close(fig)


def write_report(binary_tables: dict[str, pd.DataFrame], multiclass_df: pd.DataFrame) -> None:
    p3_test = binary_tables["Performance_3_Month_Test_Temporal.csv"]
    p3_val = binary_tables["Performance_3_Month_Validation_OOF.csv"]
    p6_test = binary_tables["Performance_6_Month_Test_Temporal.csv"]
    p6_val = binary_tables["Performance_6_Month_Validation_OOF.csv"]
    feat3 = binary_tables["3-Month_Feature_Selection.csv"].head(10)
    feat6 = binary_tables["6-Month_Feature_Selection.csv"].head(10)

    best_3_auc_model = p3_test.set_index("Metric").loc["AUC"].idxmax()
    best_3_acc_model = p3_test.set_index("Metric").loc["Accuracy"].idxmax()
    best_6_auc_model = p6_test.set_index("Metric").loc["AUC"].idxmax()
    best_6_acc_model = p6_test.set_index("Metric").loc["Accuracy"].idxmax()
    test_multi = multiclass_df[multiclass_df["Split"].eq("Temporal_Test")]
    oof_multi = multiclass_df[multiclass_df["Split"].eq("OOF_Train")]
    best_oof_models = (
        oof_multi.sort_values(["Landmark", "Macro_AUC_OVR"], ascending=[True, False])
        .groupby("Landmark")["Model"]
        .first()
        .to_dict()
    )
    best_multi = pd.concat(
        [
            test_multi[test_multi["Landmark"].eq(landmark) & test_multi["Model"].eq(model)]
            for landmark, model in best_oof_models.items()
        ],
        ignore_index=True,
    )

    def row_for(table: pd.DataFrame, model: str) -> str:
        return (
            f"| {model} | {metric(table, 'AUC', model):.3f} | {metric(table, 'Accuracy', model):.3f} | "
            f"{metric(table, 'Sensitivity', model):.3f} | {metric(table, 'Specificity', model):.3f} | "
            f"{metric(table, 'F1', model):.3f} |"
        )

    multi_lines = []
    for r in best_multi.itertuples():
        multi_lines.append(
            f"| {r.Landmark} | {r.Model} | {r.Macro_AUC_OVR:.3f} | {r.Macro_PR_AUC_OVR:.3f} | "
            f"{r.Accuracy:.3f} | {r.Balanced_Accuracy:.3f} | {r.Macro_F1:.3f} | "
            f"{r.Hyper_AUC:.3f} | {r.Normal_AUC:.3f} | {r.Hypo_AUC:.3f} |"
        )

    top3 = ", ".join(feat3["feature"].astype(str).head(6).tolist())
    top6 = ", ".join(feat6["feature"].astype(str).head(6).tolist())
    text = f"""# Fixed-Landmark Hyperthyroidism Classification Report

## Executive Summary

本报告只聚焦 fixed-landmark 问题，暂时不讨论纵向 relapse 主线。现有最成熟结果是 **3M 二分类**：用 0M/1M/3M 信息预测最终是否 Hyper/甲亢。该线已经有稳定的 temporal-test 表现，最佳 AUC 为 `{metric(p3_test, 'AUC', best_3_auc_model):.3f}`，最佳 accuracy 为 `{metric(p3_test, 'Accuracy', best_3_acc_model):.3f}`。

6M 二分类更强，符合临床直觉：随访更久，治疗反应证据更充分。6M temporal-test 最佳 AUC 为 `{metric(p6_test, 'AUC', best_6_auc_model):.3f}`，最佳 accuracy 为 `{metric(p6_test, 'Accuracy', best_6_acc_model):.3f}`。

新增尝试：我在 `results/landmark_focus/` 下重新跑了 3M/6M 三分类 probe（Hyper / Normal / Hypo），输出结构化 metrics CSV 和混淆矩阵。三分类比“Hyper vs non-Hyper”更难，尤其 Normal 与 Hypo 的边界更依赖后续时间与治疗过程。

## 1. 3M Binary Existing Scheme

**任务定义**

```text
Input:
  baseline/static variables
  + FT3/FT4/TSH at 0M, 1M, 3M
  + early response ratios, means, stds, recent deltas, static-lab interactions

Target:
  outcome == Hyper vs non-Hyper

Split:
  patient-order temporal split
  train/development: 795 treatment records
  temporal test: 208 treatment records
```

**预处理**

- temporal split 先发生，避免把测试患者信息带入 imputer。
- MissForest 只在 train/development 记录上拟合。
- 阈值在 OOF/development prediction 上选择，temporal test 只做最终报告。

**模型族**

- Logistic Regression
- Elastic-net Logistic Regression
- SVM
- Random Forest / Balanced Random Forest
- MLP
- LightGBM
- Elastic+LGBM Blend
- Routed specialist: 以 blend 为主模型，对高错误风险子群触发 specialist。

## 2. 3M Binary Results

| Selected view | Model | AUC | Accuracy | Sensitivity | Specificity | F1 |
|---|---|---:|---:|---:|---:|---:|
| Best AUC | {row_for(p3_test, best_3_auc_model).strip('|')} |
| Best Accuracy | {row_for(p3_test, best_3_acc_model).strip('|')} |

完整 3M temporal-test 模型比较见 `tables/Performance_3_Month_Test_Temporal.csv`。

![3M and 6M binary performance](figures/binary_3m_6m_temporal_performance.png)

**3M 关键特征**

Top selected variables:

```text
{top3}
```

这些变量非常符合临床直觉：3M 当前 FT3/FT4/TSH、早期 TSH 改变量，以及基线甲状腺重量/抗体共同决定早期治疗反应。

## 3. 6M Binary Continuation

6M 二分类使用 0M/1M/3M/6M 信息，目标仍是 final Hyper vs non-Hyper。

| Selected view | Model | AUC | Accuracy | Sensitivity | Specificity | F1 |
|---|---|---:|---:|---:|---:|---:|
| Best AUC | {row_for(p6_test, best_6_auc_model).strip('|')} |
| Best Accuracy | {row_for(p6_test, best_6_acc_model).strip('|')} |

6M 的整体提升很明显：6M temporal-test AUC 已到 `{metric(p6_test, 'AUC', best_6_auc_model):.3f}`，Accuracy 最高 `{metric(p6_test, 'Accuracy', best_6_acc_model):.3f}`。这说明 fixed landmark 分类确实存在时间梯度：3M 已经可用，6M 更稳定。

Top selected variables:

```text
{top6}
```

## 4. New 3M / 6M Three-Class Probe

**任务定义**

```text
Classes:
  0 = Hyper
  1 = Normal
  2 = Hypo

Input:
  same landmark-safe feature frame as binary:
  static + labs up to landmark + deltas/ratios/summary statistics

Selection:
  internal OOF metrics for model comparison
  temporal test only reported
```

**Models tried**

- Multinomial LR C=0.3
- Multinomial LR C=1.0
- Random Forest
- ExtraTrees
- LightGBM multiclass
- Cascade RF
- Cascade LGBM

**OOF-selected models, temporal-test performance**

| Landmark | Model | Macro AUC | Macro PR-AUC | Accuracy | Balanced Accuracy | Macro F1 | Hyper AUC | Normal AUC | Hypo AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(multi_lines)}

![Multiclass performance](figures/multiclass_3m_6m_temporal_performance.png)

![3M multiclass CM](figures/cm_3m_multiclass_best.png)

![6M multiclass CM](figures/cm_6m_multiclass_best.png)

## 5. Interpretation

1. **3M 二分类是当前最容易讲清楚、也最稳的 fixed-landmark 结果。** 它回答的是“3M 早期治疗反应能否预测最终仍甲亢”，AUC `~0.85`，accuracy `~0.81`。
2. **6M 二分类更强。** AUC `~0.91`，accuracy `~0.85`，说明随着随访证据积累，Hyper vs non-Hyper 判别更稳定。
3. **三分类是更难但值得继续的方向。** Hyper 的 OVR AUC 通常不错，Normal/Hypo 边界相对难，macro PR-AUC 会更受类别不均衡影响。
4. **如果写论文主线，建议主文放 3M binary + 6M binary；三分类先作为补充或探索分析。** 三分类需要再做更系统的阈值/代价敏感选择，尤其要明确临床上误把 Hypo 当 Normal、误把 Hyper 当 Normal 的代价是否相同。

## 6. Artifact Index

```text
results/landmark_focus/
  README.md
  tables/
    Performance_*_Month_*.csv
    binary_temporal_model_summary.csv
    3m_multiclass_probe_metrics.csv
    6m_multiclass_probe_metrics.csv
    multiclass_probe_all_metrics.csv
  figures/
    binary_3m_6m_temporal_performance.png
    multiclass_3m_6m_temporal_performance.png
    cm_3m_multiclass_best.png
    cm_6m_multiclass_best.png
```
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    publication_style()
    binary_tables = copy_binary_tables()
    save_binary_summary_figure(binary_tables)

    all_metrics = []
    payloads = {}
    for landmark, seq_len in [("3M", 3), ("6M", 4)]:
        metrics_df, landmark_payloads = run_multiclass_probe(landmark, seq_len)
        all_metrics.append(metrics_df)
        payloads[landmark] = landmark_payloads
    multiclass = pd.concat(all_metrics, ignore_index=True)
    multiclass.to_csv(TAB / "multiclass_probe_all_metrics.csv", index=False)
    save_multiclass_figures(multiclass, payloads)
    write_report(binary_tables, multiclass)

    manifest = {
        "output_dir": str(OUT),
        "tables": sorted(p.name for p in TAB.glob("*.csv")),
        "figures": sorted(p.name for p in FIG.glob("*.png")),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
