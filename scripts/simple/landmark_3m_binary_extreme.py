"""3M-only fixed-landmark binary benchmark.

This script is intentionally isolated from the existing fixed-landmark outputs.
It focuses on P(final Hyper | baseline + 1M + 3M information), adds clinically
motivated RAI/early-response features, runs repeated seed benchmarks, and keeps
formal model selection separate from test-aware ceiling probes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
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

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from imblearn.ensemble import BalancedRandomForestClassifier
except Exception:  # pragma: no cover - optional dependency guard
    BalancedRandomForestClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency guard
    LGBMClassifier = None

from utils.config import SEED, STATIC_NAMES, TIME_STAMPS
from utils.data import extract_flat_features, fit_missforest, load_data, split_imputed


OUT_DEFAULT = ROOT / "results" / "3m_binary_extreme"
DEFAULT_SEEDS = [13, 17, 29, 42, 101, 2025, 3407, 8803, 12011, 20260503]
LABS = ["FT3", "FT4", "TSH"]
SEQ_LEN = 3


@dataclass(frozen=True)
class Dataset3M:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    groups_train: np.ndarray
    pids_train: np.ndarray
    pids_test: np.ndarray
    feature_sets: dict[str, list[str]]
    stable_summary: pd.DataFrame
    raw_info: dict[str, Any]


def parse_seeds(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.asarray(a, dtype=float) / (np.asarray(b, dtype=float) + eps)


def log1p_pos(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(np.asarray(x, dtype=float), 0.0, None))


def build_original_landmark_features(
    x_s: np.ndarray,
    x_d: np.ndarray,
) -> tuple[pd.DataFrame, list[str]]:
    """Match the existing fixed_landmark_binary.py 3M feature frame."""
    x_base, feat_names = extract_flat_features(x_s, x_d, SEQ_LEN)
    labs = x_d[:, :SEQ_LEN, :]
    final = labs[:, -1, :]
    prev = labs[:, -2, :]
    first = labs[:, 0, :]
    eps = 1e-6
    thyroid_idx = STATIC_NAMES.index("ThyroidW")
    trab_idx = STATIC_NAMES.index("TRAb")

    extra = np.column_stack(
        [
            final[:, 0] / (first[:, 0] + eps),
            final[:, 1] / (first[:, 1] + eps),
            (final[:, 2] + 1.0) / (first[:, 2] + 1.0),
            final[:, 0] / (final[:, 1] + eps),
            final[:, 1] / (final[:, 2] + 1.0),
            labs.mean(axis=1),
            labs.std(axis=1),
            final - prev,
            x_s[:, thyroid_idx] * final[:, 0],
            x_s[:, thyroid_idx] * final[:, 1],
            x_s[:, trab_idx] * final[:, 0],
            x_s[:, trab_idx] * final[:, 1],
        ]
    )
    extra_names = [
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
    x_all = np.concatenate([x_base, extra], axis=1)
    names = feat_names + extra_names
    return pd.DataFrame(x_all, columns=names), names


def add_extended_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Add 3M-safe RAI physiology and early-response features."""
    out = df.copy()
    eps = 1e-6

    for col in ["FT3_0M", "FT3_1M", "FT3_3M", "FT4_0M", "FT4_1M", "FT4_3M", "TSH_0M", "TSH_1M", "TSH_3M"]:
        if col not in out:
            raise ValueError(f"Missing required 3M lab column: {col}")

    static = {name: out[name].to_numpy(dtype=float) for name in STATIC_NAMES if name in out}
    dose = static.get("Dose")
    thyroid_w = static.get("ThyroidW")
    uptake24 = static.get("Uptake24h")
    max_uptake = static.get("MaxUptake")
    half_life = static.get("HalfLife")
    rai3d = static.get("RAI3d")
    trab = static.get("TRAb")
    tgab = static.get("TGAb")
    tpoab = static.get("TPOAb")

    rai_names: list[str] = []
    if dose is not None and thyroid_w is not None:
        out["IDPG_Dose_per_ThyroidW"] = safe_div(dose, thyroid_w)
        out["Dose_x_ThyroidW"] = dose * thyroid_w
        rai_names += ["IDPG_Dose_per_ThyroidW", "Dose_x_ThyroidW"]
    if dose is not None and uptake24 is not None:
        out["Dose_x_Uptake24h"] = dose * uptake24
        out["Dose_per_Uptake24h"] = safe_div(dose, uptake24)
        rai_names += ["Dose_x_Uptake24h", "Dose_per_Uptake24h"]
    if dose is not None and max_uptake is not None:
        out["Dose_x_MaxUptake"] = dose * max_uptake
        rai_names.append("Dose_x_MaxUptake")
    if dose is not None and half_life is not None:
        out["Dose_x_HalfLife"] = dose * half_life
        out["Estimated_TID_Dose_x_Uptake24h_x_HalfLife"] = dose * (uptake24 if uptake24 is not None else 1.0) * half_life
        rai_names += ["Dose_x_HalfLife", "Estimated_TID_Dose_x_Uptake24h_x_HalfLife"]
    if rai3d is not None and dose is not None:
        out["RAI3d_x_Dose"] = rai3d * dose
        rai_names.append("RAI3d_x_Dose")
    if thyroid_w is not None:
        out["ThyroidW_x_logTSH_3M"] = thyroid_w * log1p_pos(out["TSH_3M"].to_numpy())
        out["ThyroidW_x_FT3_Response_0_3M"] = thyroid_w * safe_div(out["FT3_0M"] - out["FT3_3M"], out["FT3_0M"].abs() + eps)
        out["ThyroidW_x_FT4_Response_0_3M"] = thyroid_w * safe_div(out["FT4_0M"] - out["FT4_3M"], out["FT4_0M"].abs() + eps)
        rai_names += ["ThyroidW_x_logTSH_3M", "ThyroidW_x_FT3_Response_0_3M", "ThyroidW_x_FT4_Response_0_3M"]
    if dose is not None and thyroid_w is not None:
        idpg = out["IDPG_Dose_per_ThyroidW"].to_numpy()
        out["IDPG_x_FT3_Response_0_3M"] = idpg * safe_div(out["FT3_0M"] - out["FT3_3M"], out["FT3_0M"].abs() + eps)
        out["IDPG_x_FT4_Response_0_3M"] = idpg * safe_div(out["FT4_0M"] - out["FT4_3M"], out["FT4_0M"].abs() + eps)
        rai_names += ["IDPG_x_FT3_Response_0_3M", "IDPG_x_FT4_Response_0_3M"]

    resp_names: list[str] = []
    for lab in ["FT3", "FT4"]:
        v0 = out[f"{lab}_0M"].to_numpy(dtype=float)
        v1 = out[f"{lab}_1M"].to_numpy(dtype=float)
        v3 = out[f"{lab}_3M"].to_numpy(dtype=float)
        out[f"PctDrop_{lab}_0_1M"] = safe_div(v0 - v1, np.abs(v0) + eps)
        out[f"PctDrop_{lab}_0_3M"] = safe_div(v0 - v3, np.abs(v0) + eps)
        out[f"PctDrop_{lab}_1_3M"] = safe_div(v1 - v3, np.abs(v1) + eps)
        out[f"{lab}_3M_over_0M"] = safe_div(v3, v0)
        out[f"{lab}_3M_minus_0M_abs"] = v3 - v0
        resp_names += [
            f"PctDrop_{lab}_0_1M",
            f"PctDrop_{lab}_0_3M",
            f"PctDrop_{lab}_1_3M",
            f"{lab}_3M_over_0M",
            f"{lab}_3M_minus_0M_abs",
        ]

    t0 = log1p_pos(out["TSH_0M"].to_numpy())
    t1 = log1p_pos(out["TSH_1M"].to_numpy())
    t3 = log1p_pos(out["TSH_3M"].to_numpy())
    out["logTSH_0M"] = t0
    out["logTSH_1M"] = t1
    out["logTSH_3M"] = t3
    out["D_logTSH_3M_0M"] = t3 - t0
    out["D_logTSH_3M_1M"] = t3 - t1
    out["logTSH_slope_0_3M"] = (t3 - t0) / 3.0
    out["TSH_Recovered_3M"] = ((out["TSH_3M"] >= 0.35) & (out["TSH_3M"] <= 4.5)).astype(float)
    out["Likely_Hyper_3M"] = ((out["FT3_3M"] > out["FT3_3M"].median()) & (out["TSH_3M"] < out["TSH_3M"].quantile(0.35))).astype(float)
    resp_names += [
        "logTSH_0M",
        "logTSH_1M",
        "logTSH_3M",
        "D_logTSH_3M_0M",
        "D_logTSH_3M_1M",
        "logTSH_slope_0_3M",
        "TSH_Recovered_3M",
        "Likely_Hyper_3M",
    ]

    antibody_names: list[str] = []
    for name, arr in [("TRAb", trab), ("TGAb", tgab), ("TPOAb", tpoab)]:
        if arr is None:
            continue
        q75 = np.nanpercentile(arr, 75)
        q90 = np.nanpercentile(arr, 90)
        out[f"{name}_HighQ75"] = (arr >= q75).astype(float)
        out[f"{name}_HighQ90"] = (arr >= q90).astype(float)
        antibody_names += [f"{name}_HighQ75", f"{name}_HighQ90"]
    if trab is not None:
        out["TRAb_HighQ75_x_FT3_3M"] = out["TRAb_HighQ75"] * out["FT3_3M"]
        out["TRAb_HighQ75_x_FT4_3M"] = out["TRAb_HighQ75"] * out["FT4_3M"]
        if thyroid_w is not None:
            out["TRAb_HighQ75_x_ThyroidW"] = out["TRAb_HighQ75"] * thyroid_w
            antibody_names.append("TRAb_HighQ75_x_ThyroidW")
        antibody_names += ["TRAb_HighQ75_x_FT3_3M", "TRAb_HighQ75_x_FT4_3M"]

    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(out.median(numeric_only=True))
    return out, {"rai": rai_names, "response": resp_names, "antibody": antibody_names}


def temporal_split_indices(pids: np.ndarray, ratio: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    unique_pids = list(dict.fromkeys(pids))
    split_idx = int(len(unique_pids) * ratio)
    train_pids = set(unique_pids[:split_idx])
    train_idx = np.where(np.array([pid in train_pids for pid in pids]))[0]
    test_idx = np.where(np.array([pid not in train_pids for pid in pids]))[0]
    return train_idx, test_idx


def build_feature_frame(out_dir: Path, force_rebuild: bool = False) -> Dataset3M:
    cache = out_dir / "cache" / "feature_frame_3m.pkl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and not force_rebuild:
        try:
            return joblib.load(cache)
        except Exception:
            # Older runs may have pickled Dataset3M under __main__ when the
            # script was executed directly. Rebuild instead of failing.
            pass

    x_s_raw, ft3_raw, ft4_raw, tsh_raw, _, y_raw, pids = load_data()
    train_idx, test_idx = temporal_split_indices(pids)
    n_static = x_s_raw.shape[1]
    raw_all = np.hstack([x_s_raw, ft3_raw[:, :SEQ_LEN], ft4_raw[:, :SEQ_LEN], tsh_raw[:, :SEQ_LEN]])
    imputer = fit_missforest(raw_all[train_idx], seed=SEED)
    filled_train = imputer.transform(raw_all[train_idx])
    filled_test = imputer.transform(raw_all[test_idx])
    xs_tr, f3_tr, f4_tr, ts_tr = split_imputed(filled_train, n_static, SEQ_LEN)
    xs_te, f3_te, f4_te, ts_te = split_imputed(filled_test, n_static, SEQ_LEN)

    xd_tr = np.stack([f3_tr, f4_tr, ts_tr], axis=-1)
    xd_te = np.stack([f3_te, f4_te, ts_te], axis=-1)
    base_tr, baseline_names = build_original_landmark_features(xs_tr, xd_tr)
    base_te, _ = build_original_landmark_features(xs_te, xd_te)

    ext_all, blocks = add_extended_features(pd.concat([base_tr, base_te], axis=0, ignore_index=True))
    x_train = ext_all.iloc[: len(base_tr)].reset_index(drop=True)
    x_test = ext_all.iloc[len(base_tr) :].reset_index(drop=True)
    y_train = (y_raw[train_idx] == 1).astype(int)
    y_test = (y_raw[test_idx] == 1).astype(int)
    groups_train = np.asarray(pids[train_idx])

    feature_sets = build_feature_sets(x_train, y_train, groups_train, baseline_names, blocks, out_dir)
    stable_summary = pd.read_csv(out_dir / "tables" / "Stable_Feature_Selection.csv")
    data = Dataset3M(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        groups_train=groups_train,
        pids_train=np.asarray(pids[train_idx]),
        pids_test=np.asarray(pids[test_idx]),
        feature_sets=feature_sets,
        stable_summary=stable_summary,
        raw_info={
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_train_pos": int(y_train.sum()),
            "n_test_pos": int(y_test.sum()),
            "baseline_feature_count": len(baseline_names),
            "expanded_feature_count": int(x_train.shape[1]),
        },
    )
    joblib.dump(data, cache)
    return data


def choose_uncorrelated_features(df: pd.DataFrame, cols: list[str], priority: list[str], cutoff: float = 0.95) -> list[str]:
    keep: list[str] = []
    seen = set()
    ordered = [c for c in priority if c in cols] + [c for c in cols if c not in priority]
    corr = df[cols].corr(method="spearman").abs().fillna(0.0)
    for col in ordered:
        if col in seen:
            continue
        keep.append(col)
        high = corr.index[corr[col] >= cutoff].tolist()
        seen.update(high)
    return keep


def build_feature_sets(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    baseline_names: list[str],
    blocks: dict[str, list[str]],
    out_dir: Path,
) -> dict[str, list[str]]:
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    rai = blocks["rai"]
    response = blocks["response"]
    antibody = blocks["antibody"]
    expanded = list(x_train.columns)

    clinical_priority = [
        "FT3_3M",
        "FT4_3M",
        "TSH_3M",
        "logTSH_3M",
        "D_TSH_3M-0M",
        "D_logTSH_3M_0M",
        "ThyroidW",
        "Dose",
        "Uptake24h",
        "MaxUptake",
        "HalfLife",
        "IDPG_Dose_per_ThyroidW",
        "PctDrop_FT3_0_3M",
        "PctDrop_FT4_0_3M",
        "TSH_Recovered_3M",
        "TRAb",
        "TGAb",
        "TPOAb",
    ]
    clinical_core = [c for c in clinical_priority if c in x_train.columns][:14]
    filtered = choose_uncorrelated_features(x_train, expanded, clinical_priority, cutoff=0.96)
    stable_summary = stability_select_features(x_train[filtered], y_train, groups_train, clinical_core)
    stable_summary.to_csv(table_dir / "Stable_Feature_Selection.csv", index=False)
    selected = stable_summary.loc[stable_summary["final_keep"], "Feature"].tolist()
    selected = [c for c in selected if c in x_train.columns]

    sets = {
        "baseline": baseline_names,
        "baseline_rai": list(dict.fromkeys(baseline_names + rai)),
        "baseline_response": list(dict.fromkeys(baseline_names + response + antibody)),
        "expanded": expanded,
        "stable_selected": selected,
        "clinical_core": clinical_core,
    }
    pd.DataFrame(
        [
            {"FeatureSet": k, "N_Features": len(v), "Features": json.dumps(v, ensure_ascii=False)}
            for k, v in sets.items()
        ]
    ).to_csv(table_dir / "Feature_Sets.csv", index=False)
    return sets


def stability_select_features(
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    clinical_core: list[str],
) -> pd.DataFrame:
    gkf = GroupKFold(n_splits=5)
    elastic_counts = pd.Series(0.0, index=x.columns)
    lgbm_counts = pd.Series(0.0, index=x.columns)
    perm_counts = pd.Series(0.0, index=x.columns)
    n_folds = 0

    for fold, (tr, va) in enumerate(gkf.split(x, y, groups=groups)):
        n_folds += 1
        lr = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        C=0.1,
                        l1_ratio=0.5,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=SEED + fold,
                    ),
                ),
            ]
        )
        lr.fit(x.iloc[tr], y[tr])
        coef = np.abs(lr.named_steps["lr"].coef_[0])
        elastic_counts += (coef > 1e-7).astype(float)

        if LGBMClassifier is not None:
            lgbm = LGBMClassifier(
                objective="binary",
                class_weight="balanced",
                random_state=SEED + fold,
                n_estimators=180,
                learning_rate=0.035,
                num_leaves=12,
                max_depth=3,
                min_child_samples=15,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                verbosity=-1,
                n_jobs=1,
            )
            lgbm.fit(x.iloc[tr], y[tr])
            imp = pd.Series(lgbm.feature_importances_, index=x.columns).sort_values(ascending=False)
            lgbm_counts.loc[imp.head(25).index] += 1.0
            try:
                pi = permutation_importance(
                    lgbm,
                    x.iloc[va],
                    y[va],
                    scoring="roc_auc",
                    n_repeats=4,
                    random_state=SEED + fold,
                    n_jobs=1,
                )
                pser = pd.Series(pi.importances_mean, index=x.columns).sort_values(ascending=False)
                perm_counts.loc[pser.head(25).index] += 1.0
            except Exception:
                pass

    summary = pd.DataFrame(
        {
            "Feature": x.columns,
            "clinical_core": [c in clinical_core for c in x.columns],
            "elasticnet_freq": (elastic_counts / max(n_folds, 1)).values,
            "lgbm_freq": (lgbm_counts / max(n_folds, 1)).values,
            "permutation_freq": (perm_counts / max(n_folds, 1)).values,
        }
    )
    summary["stability_score"] = (
        0.35 * summary["elasticnet_freq"] + 0.35 * summary["lgbm_freq"] + 0.30 * summary["permutation_freq"]
    )
    summary["final_keep"] = summary["clinical_core"] | (summary["stability_score"] >= 0.32)
    if int(summary["final_keep"].sum()) < 15:
        top_idx = summary.sort_values("stability_score", ascending=False).head(15).index
        summary.loc[top_idx, "final_keep"] = True
    if int(summary["final_keep"].sum()) > 30:
        keep = set(summary.loc[summary["clinical_core"], "Feature"])
        ranked = summary.loc[~summary["clinical_core"]].sort_values("stability_score", ascending=False)
        for feat in ranked["Feature"]:
            if len(keep) >= 30:
                break
            keep.add(feat)
        summary["final_keep"] = summary["Feature"].isin(keep)
    return summary.sort_values(["final_keep", "stability_score"], ascending=[False, False]).reset_index(drop=True)


def make_model_configs(seed: int, quick: bool = False) -> list[tuple[str, Any]]:
    configs: list[tuple[str, Any]] = []
    lr_grid = [
        ("Elastic_C0.03_l10.3", 0.03, 0.3),
        ("Elastic_C0.10_l10.5", 0.10, 0.5),
        ("Elastic_C0.30_l10.7", 0.30, 0.7),
        ("Elastic_C1.00_l10.5", 1.00, 0.5),
    ]
    for name, c, ratio in lr_grid:
        configs.append(
            (
                name,
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "lr",
                            LogisticRegression(
                                penalty="elasticnet",
                                solver="saga",
                                C=c,
                                l1_ratio=ratio,
                                class_weight="balanced",
                                max_iter=6000,
                                random_state=seed,
                            ),
                        ),
                    ]
                ),
            )
        )
    configs.append(
        (
            "Logistic_L2_C0.10",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "lr",
                        LogisticRegression(
                            penalty="l2",
                            solver="lbfgs",
                            C=0.1,
                            class_weight="balanced",
                            max_iter=4000,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        )
    )
    configs.append(
        (
            "SVM_RBF_C1",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("svc", SVC(C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=seed)),
                ]
            ),
        )
    )
    if not quick:
        configs.append(
            (
                "SVM_RBF_C2_g005",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("svc", SVC(C=2.0, gamma=0.05, probability=True, class_weight="balanced", random_state=seed)),
                    ]
                ),
            )
        )

    configs += [
        (
            "RF_d4_leaf5",
            RandomForestClassifier(
                n_estimators=260,
                max_depth=4,
                min_samples_leaf=5,
                max_features="sqrt",
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            ),
        ),
        (
            "ExtraTrees_d4_leaf5",
            ExtraTreesClassifier(
                n_estimators=320,
                max_depth=4,
                min_samples_leaf=5,
                max_features="sqrt",
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            ),
        ),
        (
            "ExtraTrees_d6_leaf3",
            ExtraTreesClassifier(
                n_estimators=420,
                max_depth=6,
                min_samples_leaf=3,
                max_features=0.6,
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            ),
        ),
    ]
    if BalancedRandomForestClassifier is not None:
        configs += [
            (
                "BalancedRF_d3_leaf3",
                BalancedRandomForestClassifier(
                    n_estimators=320,
                    max_depth=3,
                    min_samples_leaf=3,
                    sampling_strategy="all",
                    replacement=True,
                    random_state=seed,
                    n_jobs=1,
                ),
            ),
            (
                "BalancedRF_d4_leaf5",
                BalancedRandomForestClassifier(
                    n_estimators=420,
                    max_depth=4,
                    min_samples_leaf=5,
                    sampling_strategy="all",
                    replacement=True,
                    random_state=seed,
                    n_jobs=1,
                ),
            ),
        ]
    if LGBMClassifier is not None:
        configs += [
            (
                "LGBM_d2_leaf7_lr03",
                LGBMClassifier(
                    objective="binary",
                    class_weight="balanced",
                    n_estimators=240,
                    learning_rate=0.03,
                    num_leaves=7,
                    max_depth=2,
                    min_child_samples=18,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.1,
                    reg_lambda=1.5,
                    random_state=seed,
                    verbosity=-1,
                    n_jobs=1,
                ),
            ),
            (
                "LGBM_d3_leaf11_lr025",
                LGBMClassifier(
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
                ),
            ),
            (
                "LGBM_d4_leaf15_lr02",
                LGBMClassifier(
                    objective="binary",
                    class_weight="balanced",
                    n_estimators=360,
                    learning_rate=0.02,
                    num_leaves=15,
                    max_depth=4,
                    min_child_samples=20,
                    subsample=0.75,
                    colsample_bytree=0.75,
                    reg_alpha=0.3,
                    reg_lambda=3.0,
                    random_state=seed,
                    verbosity=-1,
                    n_jobs=1,
                ),
            ),
        ]
    if quick:
        keep_names = {"Elastic_C0.10_l10.5", "ExtraTrees_d4_leaf5", "BalancedRF_d3_leaf3", "LGBM_d3_leaf11_lr025"}
        configs = [c for c in configs if c[0] in keep_names]
    return configs


def predict_proba_one(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        score = np.asarray(model.decision_function(x), dtype=float)
        return 1.0 / (1.0 + np.exp(-score))
    raise TypeError(f"Model has no predict_proba/decision_function: {type(model)}")


def best_thresholds(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    rows = []
    for thr in np.arange(0.05, 0.801, 0.005):
        pred = (proba >= thr).astype(int)
        rows.append(
            {
                "thr": float(thr),
                "f1": f1_score(y, pred, zero_division=0),
                "acc": accuracy_score(y, pred),
                "bacc": balanced_accuracy_score(y, pred),
            }
        )
    df = pd.DataFrame(rows)
    return {
        "thr_f1": float(df.loc[df["f1"].idxmax(), "thr"]),
        "thr_acc": float(df.loc[df["acc"].idxmax(), "thr"]),
        "thr_bacc": float(df.loc[df["bacc"].idxmax(), "thr"]),
    }


def binary_metrics(y: np.ndarray, proba: np.ndarray, thr: float) -> dict[str, float]:
    proba = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prevalence = float(np.mean(y))
    return {
        "AUC": roc_auc_score(y, proba),
        "PR_AUC": average_precision_score(y, proba),
        "PR_Lift": average_precision_score(y, proba) / prevalence if prevalence > 0 else np.nan,
        "Brier": brier_score_loss(y, proba),
        "Accuracy": accuracy_score(y, pred),
        "BalancedAccuracy": balanced_accuracy_score(y, pred),
        "F1": f1_score(y, pred, zero_division=0),
        "Recall": tp / (tp + fn) if tp + fn else np.nan,
        "Specificity": tn / (tn + fp) if tn + fp else np.nan,
        "PPV": tp / (tp + fp) if tp + fp else np.nan,
        "NPV": tn / (tn + fn) if tn + fn else np.nan,
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }


def fit_oof_and_test(
    estimator: Any,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    groups: np.ndarray,
    x_test: pd.DataFrame,
    seed: int,
) -> dict[str, Any]:
    splits = GroupKFold(n_splits=5)
    oof = np.zeros(len(y_train), dtype=float)
    for fold, (tr, va) in enumerate(splits.split(x_train, y_train, groups=groups)):
        model = clone(estimator)
        model.fit(x_train.iloc[tr], y_train[tr])
        oof[va] = predict_proba_one(model, x_train.iloc[va])
    final = clone(estimator)
    final.fit(x_train, y_train)
    train_fit = predict_proba_one(final, x_train)
    test = predict_proba_one(final, x_test)
    return {"oof": oof, "train_fit": train_fit, "test": test, "final_model": final}


def apply_calibration(mode: str, y_train: np.ndarray, oof: np.ndarray, train_fit: np.ndarray, test: np.ndarray) -> dict[str, np.ndarray]:
    oof = np.clip(oof, 1e-6, 1 - 1e-6)
    train_fit = np.clip(train_fit, 1e-6, 1 - 1e-6)
    test = np.clip(test, 1e-6, 1 - 1e-6)
    if mode == "none":
        return {"oof": oof, "train_fit": train_fit, "test": test}
    if mode == "platt":
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
        lr.fit(np.log(oof / (1 - oof)).reshape(-1, 1), y_train)
        return {
            "oof": lr.predict_proba(np.log(oof / (1 - oof)).reshape(-1, 1))[:, 1],
            "train_fit": lr.predict_proba(np.log(train_fit / (1 - train_fit)).reshape(-1, 1))[:, 1],
            "test": lr.predict_proba(np.log(test / (1 - test)).reshape(-1, 1))[:, 1],
        }
    if mode == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof, y_train)
        return {"oof": iso.transform(oof), "train_fit": iso.transform(train_fit), "test": iso.transform(test)}
    raise ValueError(mode)


def result_rows_for_prediction(
    run_id: str,
    seed: int,
    feature_set: str,
    model_name: str,
    calib: str,
    n_features: int,
    y_train: np.ndarray,
    y_test: np.ndarray,
    probs: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    thrs = best_thresholds(y_train, probs["oof"])
    rows = []
    for threshold_name, thr in thrs.items():
        train_m = binary_metrics(y_train, probs["train_fit"], thr)
        oof_m = binary_metrics(y_train, probs["oof"], thr)
        test_m = binary_metrics(y_test, probs["test"], thr)
        row = {
            "RunID": f"{run_id}__{threshold_name}",
            "BaseRunID": run_id,
            "Seed": seed,
            "FeatureSet": feature_set,
            "Model": model_name,
            "Calibration": calib,
            "ThresholdRule": threshold_name,
            "Threshold": thr,
            "N_Features": n_features,
            "FormalScore_OOF_AUC": oof_m["AUC"],
            "CeilingScore_Test_AUC": test_m["AUC"],
        }
        for prefix, metrics in [("TrainFit", train_m), ("OOF", oof_m), ("Test", test_m)]:
            for k, v in metrics.items():
                row[f"{prefix}_{k}"] = v
        rows.append(row)
    return rows, thrs


def run_seed_feature_set(
    seed: int,
    feature_set: str,
    feature_names: list[str],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    groups: np.ndarray,
    pids_train: np.ndarray,
    pids_test: np.ndarray,
    quick: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    warnings.simplefilter("ignore")
    xtr = x_train[feature_names].copy()
    xte = x_test[feature_names].copy()
    configs = make_model_configs(seed, quick=quick)
    rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    raw_preds: dict[str, dict[str, np.ndarray]] = {}
    status = {"Seed": seed, "FeatureSet": feature_set, "NConfigs": len(configs), "OK": True, "Error": ""}

    for model_name, estimator in configs:
        try:
            pred = fit_oof_and_test(estimator, xtr, y_train, groups, xte, seed)
            raw_preds[model_name] = {k: pred[k] for k in ["oof", "train_fit", "test"]}
            for calib in ["none", "platt", "isotonic"]:
                probs = apply_calibration(calib, y_train, pred["oof"], pred["train_fit"], pred["test"])
                run_id = f"s{seed}__{feature_set}__{model_name}__{calib}"
                model_rows, _ = result_rows_for_prediction(
                    run_id,
                    seed,
                    feature_set,
                    model_name,
                    calib,
                    len(feature_names),
                    y_train,
                    y_test,
                    probs,
                )
                rows.extend(model_rows)
                pred_rows.extend(make_prediction_rows(run_id, pids_train, y_train, probs["oof"], "OOF"))
                pred_rows.extend(make_prediction_rows(run_id, pids_test, y_test, probs["test"], "Test"))
        except Exception as exc:
            status["OK"] = False
            status["Error"] += f"{model_name}: {exc}; "

    blend_metric_rows, blend_pred_rows = build_blend_rows(
        seed, feature_set, len(feature_names), raw_preds, y_train, y_test, pids_train, pids_test
    )
    rows.extend(blend_metric_rows)
    pred_rows.extend(blend_pred_rows)
    return rows, pred_rows, status


def make_prediction_rows(
    run_id: str,
    pids: np.ndarray,
    y: np.ndarray,
    proba: np.ndarray,
    domain: str,
) -> list[dict[str, Any]]:
    return [
        {"BaseRunID": run_id, "Domain": domain, "Patient_ID": pid, "Y": int(label), "Proba": float(p)}
        for pid, label, p in zip(pids, y, proba)
    ]


def build_blend_rows(
    seed: int,
    feature_set: str,
    n_features: int,
    raw_preds: dict[str, dict[str, np.ndarray]],
    y_train: np.ndarray,
    y_test: np.ndarray,
    pids_train: np.ndarray,
    pids_test: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pred_rows: list[dict[str, Any]] = []
    if len(raw_preds) < 2:
        return [], []

    def family_best(prefixes: tuple[str, ...]) -> str | None:
        candidates = [name for name in raw_preds if name.startswith(prefixes)]
        if not candidates:
            return None
        return max(candidates, key=lambda name: roc_auc_score(y_train, raw_preds[name]["oof"]))

    lr = family_best(("Elastic", "Logistic"))
    lgbm = family_best(("LGBM",))
    forest = family_best(("BalancedRF", "ExtraTrees", "RF"))
    blend_specs = []
    if lr and lgbm:
        blend_specs.append(("Blend_LR_LGBM", [lr, lgbm]))
    if lr and lgbm and forest:
        blend_specs.append(("Blend_LR_LGBM_Forest", [lr, lgbm, forest]))

    out_rows: list[dict[str, Any]] = []
    for blend_name, members in blend_specs:
        best_w = None
        best_auc = -1.0
        if len(members) == 2:
            grid = [(w, 1.0 - w) for w in np.linspace(0.0, 1.0, 21)]
        else:
            grid = []
            for a in np.linspace(0.0, 1.0, 11):
                for b in np.linspace(0.0, 1.0 - a, 11):
                    c = 1.0 - a - b
                    grid.append((a, b, c))
        for weights in grid:
            oof = sum(w * raw_preds[m]["oof"] for w, m in zip(weights, members))
            auc = roc_auc_score(y_train, oof)
            if auc > best_auc:
                best_auc = auc
                best_w = weights
        if best_w is None:
            continue
        probs = {
            "oof": sum(w * raw_preds[m]["oof"] for w, m in zip(best_w, members)),
            "train_fit": sum(w * raw_preds[m]["train_fit"] for w, m in zip(best_w, members)),
            "test": sum(w * raw_preds[m]["test"] for w, m in zip(best_w, members)),
        }
        for calib in ["none", "platt", "isotonic"]:
            cal_probs = apply_calibration(calib, y_train, probs["oof"], probs["train_fit"], probs["test"])
            run_id = f"s{seed}__{feature_set}__{blend_name}__{calib}"
            model_rows, _ = result_rows_for_prediction(
                run_id, seed, feature_set, blend_name, calib, n_features, y_train, y_test, cal_probs
            )
            for row in model_rows:
                row["BlendMembers"] = "|".join(members)
                row["BlendWeights"] = "|".join(f"{w:.2f}" for w in best_w)
            out_rows.extend(model_rows)
            pred_rows.extend(make_prediction_rows(run_id, pids_train, y_train, cal_probs["oof"], "OOF"))
            pred_rows.extend(make_prediction_rows(run_id, pids_test, y_test, cal_probs["test"], "Test"))
    return out_rows, pred_rows


def aggregate_results(all_runs: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["FeatureSet", "Model", "Calibration", "ThresholdRule", "N_Features"]
    agg = (
        all_runs.groupby(group_cols, dropna=False)
        .agg(
            N=("RunID", "count"),
            Mean_OOF_AUC=("OOF_AUC", "mean"),
            Std_OOF_AUC=("OOF_AUC", "std"),
            Mean_OOF_PR_AUC=("OOF_PR_AUC", "mean"),
            Mean_OOF_Accuracy=("OOF_Accuracy", "mean"),
            Mean_OOF_Brier=("OOF_Brier", "mean"),
            Mean_Test_AUC=("Test_AUC", "mean"),
            Std_Test_AUC=("Test_AUC", "std"),
            Mean_Test_PR_AUC=("Test_PR_AUC", "mean"),
            Mean_Test_Accuracy=("Test_Accuracy", "mean"),
            Mean_Test_Brier=("Test_Brier", "mean"),
            Mean_TrainFit_AUC=("TrainFit_AUC", "mean"),
            Mean_Gap_TrainFit_OOF_AUC=("TrainFit_AUC", lambda s: np.nan),
        )
        .reset_index()
    )
    gap = (
        all_runs.assign(Gap=lambda d: d["TrainFit_AUC"] - d["OOF_AUC"])
        .groupby(group_cols, dropna=False)["Gap"]
        .mean()
        .reset_index(name="Mean_Gap_TrainFit_OOF_AUC")
    )
    agg = agg.drop(columns=["Mean_Gap_TrainFit_OOF_AUC"]).merge(gap, on=group_cols, how="left")
    return agg.sort_values(["Mean_OOF_AUC", "Mean_OOF_Accuracy", "Mean_OOF_Brier"], ascending=[False, False, True])


def save_main_figures(out_dir: Path, all_runs: pd.DataFrame, agg: pd.DataFrame, preds: pd.DataFrame, data: Dataset3M) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    top_formal = agg.head(10).copy()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    labels = top_formal["FeatureSet"] + "\n" + top_formal["Model"].str.replace("_", " ", regex=False)
    x = np.arange(len(top_formal))
    ax.bar(x - 0.18, top_formal["Mean_OOF_AUC"], width=0.36, label="OOF AUC", color="#2f6fbb")
    ax.bar(x + 0.18, top_formal["Mean_Test_AUC"], width=0.36, label="Temporal Test AUC", color="#2a9d8f")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0.70, min(1.0, max(top_formal["Mean_OOF_AUC"].max(), top_formal["Mean_Test_AUC"].max()) + 0.05))
    ax.set_ylabel("AUC")
    ax.set_title("Top Formal 3M Binary Configurations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "Figure_Top_Formal_AUC.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    order = ["baseline", "baseline_rai", "baseline_response", "expanded", "stable_selected", "clinical_core"]
    plot_df = all_runs[all_runs["ThresholdRule"] == "thr_bacc"].copy()
    sns.boxplot(data=plot_df, x="FeatureSet", y="Test_AUC", order=[o for o in order if o in plot_df["FeatureSet"].unique()], ax=ax, color="#98c1d9")
    sns.stripplot(data=plot_df, x="FeatureSet", y="Test_AUC", order=[o for o in order if o in plot_df["FeatureSet"].unique()], ax=ax, color="#293241", size=2, alpha=0.35)
    ax.set_title("Temporal-Test AUC Distribution Across Feature Families")
    ax.set_xlabel("")
    ax.set_ylabel("Temporal Test AUC")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(fig_dir / "Figure_Feature_Family_Seed_Stability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    stable = data.stable_summary.head(30)
    fig, ax = plt.subplots(figsize=(8.5, 8))
    heat = stable.set_index("Feature")[["elasticnet_freq", "lgbm_freq", "permutation_freq", "stability_score"]]
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
    ax.set_title("Train-Only Stable Feature Selection")
    fig.tight_layout()
    fig.savefig(fig_dir / "Figure_Stable_Feature_Selection.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    best = agg.iloc[0]
    best_runs = all_runs[
        (all_runs["FeatureSet"] == best["FeatureSet"])
        & (all_runs["Model"] == best["Model"])
        & (all_runs["Calibration"] == best["Calibration"])
        & (all_runs["ThresholdRule"] == best["ThresholdRule"])
    ].sort_values("OOF_AUC", ascending=False)
    best_base = best_runs.iloc[0]["BaseRunID"]
    pred = preds[(preds["BaseRunID"] == best_base) & (preds["Domain"] == "Test")]
    if len(pred) > 0:
        y = pred["Y"].to_numpy()
        p = pred["Proba"].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        axes[0].plot(fpr, tpr, lw=2.2, color="#1d4e89")
        axes[0].plot([0, 1], [0, 1], "--", color="gray", lw=1)
        axes[0].set_title(f"ROC: {best['Model']} ({best['FeatureSet']})")
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].text(0.62, 0.08, f"AUC={roc_auc_score(y, p):.3f}")
        axes[1].plot(rec, prec, lw=2.2, color="#2a9d8f")
        axes[1].axhline(y.mean(), linestyle="--", color="gray", lw=1)
        axes[1].set_title("Precision-Recall")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].text(0.60, min(0.95, max(0.1, y.mean() + 0.08)), f"AP={average_precision_score(y, p):.3f}")
        fig.tight_layout()
        fig.savefig(fig_dir / "Figure_Best_ROC_PR.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        frac, mean_pred = calibration_curve(y, np.clip(p, 1e-6, 1 - 1e-6), n_bins=6, strategy="quantile")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        axes[0].plot([0, 1], [0, 1], "--", color="gray", lw=1)
        axes[0].plot(mean_pred, frac, marker="o", lw=2, color="#7b2cbf")
        axes[0].set_title("Calibration")
        axes[0].set_xlabel("Predicted probability")
        axes[0].set_ylabel("Observed Hyper rate")
        axes[1].hist(p, bins=20, color="#7b2cbf", alpha=0.8, edgecolor="white")
        axes[1].set_title(f"Prediction Distribution (Brier={brier_score_loss(y, p):.3f})")
        axes[1].set_xlabel("Predicted probability")
        fig.tight_layout()
        fig.savefig(fig_dir / "Figure_Best_Calibration.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        threshold = float(best_runs.iloc[0]["Threshold"])
        cm = confusion_matrix(y, (p >= threshold).astype(int), labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5.5, 4.6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Hyper", "Hyper"], yticklabels=["Non-Hyper", "Hyper"], ax=ax)
        ax.set_title(f"Confusion Matrix at OOF {best['ThresholdRule']}={threshold:.3f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fig.tight_layout()
        fig.savefig(fig_dir / "Figure_Best_Confusion_Matrix.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        dca = compute_dca_df(y, p)
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.plot(dca["threshold"], dca["model"], lw=2.2, label="Model", color="#2d6a4f")
        ax.plot(dca["threshold"], dca["treat_all"], "--", lw=1.6, label="Treat all", color="#c1121f")
        ax.plot(dca["threshold"], dca["treat_none"], ":", lw=1.6, label="Treat none", color="black")
        ax.set_title("Decision Curve Analysis")
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "Figure_Best_DCA.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    key_features = [c for c in ["FT3_3M", "FT4_3M", "TSH_3M", "PctDrop_FT3_0_3M", "PctDrop_FT4_0_3M", "IDPG_Dose_per_ThyroidW", "ThyroidW"] if c in data.x_train.columns]
    box_df = data.x_train[key_features].copy()
    box_df["Outcome"] = np.where(data.y_train == 1, "Hyper", "Non-Hyper")
    melted = box_df.melt(id_vars="Outcome", var_name="Feature", value_name="Value")
    g = sns.catplot(data=melted, x="Outcome", y="Value", col="Feature", col_wrap=3, kind="box", sharey=False, height=3.0, aspect=1.1)
    g.fig.suptitle("Key 3M Early-Response Features in Training Set", y=1.02)
    g.savefig(fig_dir / "Figure_Key_Feature_Boxplots.png", dpi=300, bbox_inches="tight")
    plt.close(g.fig)


def compute_dca_df(y: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    rows = []
    prevalence = float(np.mean(y))
    n = len(y)
    for thr in np.arange(0.02, 0.51, 0.01):
        pred = (proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        odds = thr / (1 - thr)
        rows.append(
            {
                "threshold": thr,
                "model": tp / n - fp / n * odds,
                "treat_all": prevalence - (1 - prevalence) * odds,
                "treat_none": 0.0,
            }
        )
    return pd.DataFrame(rows)


def df_to_markdown(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Small markdown table writer without the optional tabulate dependency."""
    if df.empty:
        return "_No rows._"
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda v: "" if pd.isna(v) else format(float(v), floatfmt))
        else:
            formatted[col] = formatted[col].map(lambda v: "" if pd.isna(v) else str(v))
    headers = [str(c) for c in formatted.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in formatted.iterrows():
        values = [str(row[col]).replace("|", "\\|") for col in formatted.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_readme(out_dir: Path, data: Dataset3M, all_runs: pd.DataFrame, agg: pd.DataFrame) -> None:
    top = agg.head(8)
    ceiling = all_runs.sort_values(["Test_AUC", "Test_Accuracy", "OOF_AUC"], ascending=[False, False, False]).head(8)
    stable = data.stable_summary[data.stable_summary["final_keep"]].head(30)
    baseline_row = agg[agg["FeatureSet"].eq("baseline")].head(1)
    best = top.iloc[0]
    lines = [
        "# 3M Binary Extreme Benchmark",
        "",
        "## Cohort And Task",
        "",
        "Task: predict final Hyper vs Non-Hyper using only baseline, 1M, and 3M information.",
        "",
        f"- Train records: {data.raw_info['n_train']} (Hyper={data.raw_info['n_train_pos']})",
        f"- Temporal test records: {data.raw_info['n_test']} (Hyper={data.raw_info['n_test_pos']})",
        f"- Baseline features: {data.raw_info['baseline_feature_count']}",
        f"- Expanded features: {data.raw_info['expanded_feature_count']}",
        "",
        "## Best Formal Candidate",
        "",
        "Formal selection ranks by internal OOF AUC. Temporal test is reported but not used for selection.",
        "",
        f"- Feature set: `{best['FeatureSet']}`",
        f"- Model: `{best['Model']}`",
        f"- Calibration: `{best['Calibration']}`",
        f"- Threshold rule: `{best['ThresholdRule']}`",
        f"- Mean OOF AUC: {best['Mean_OOF_AUC']:.3f}",
        f"- Mean OOF Accuracy: {best['Mean_OOF_Accuracy']:.3f}",
        f"- Mean Test AUC: {best['Mean_Test_AUC']:.3f}",
        f"- Mean Test Accuracy: {best['Mean_Test_Accuracy']:.3f}",
        f"- Mean Test PR-AUC: {best['Mean_Test_PR_AUC']:.3f}",
        f"- Mean Test Brier: {best['Mean_Test_Brier']:.3f}",
        "",
        "## Baseline Reference",
        "",
    ]
    if len(baseline_row):
        b = baseline_row.iloc[0]
        lines += [
            f"- Best baseline mean OOF AUC: {b['Mean_OOF_AUC']:.3f}",
            f"- Best baseline mean Test AUC: {b['Mean_Test_AUC']:.3f}",
            f"- Best baseline mean Test Accuracy: {b['Mean_Test_Accuracy']:.3f}",
        ]
    lines += [
        "",
        "## Top Formal Configurations",
        "",
        df_to_markdown(top[
            [
                "FeatureSet",
                "Model",
                "Calibration",
                "ThresholdRule",
                "Mean_OOF_AUC",
                "Mean_OOF_Accuracy",
                "Mean_Test_AUC",
                "Mean_Test_Accuracy",
                "Mean_Test_PR_AUC",
                "Mean_Test_Brier",
            ]
        ]),
        "",
        "## Test-Aware Ceiling Probe",
        "",
        "The rows below are exploratory. They are useful for understanding the possible upper bound, but should not be used as the paper's model-selection rule.",
        "",
        df_to_markdown(ceiling[
            [
                "Seed",
                "FeatureSet",
                "Model",
                "Calibration",
                "ThresholdRule",
                "OOF_AUC",
                "OOF_Accuracy",
                "Test_AUC",
                "Test_Accuracy",
                "Test_PR_AUC",
                "Test_Brier",
            ]
        ]),
        "",
        "## Stable Selected Features",
        "",
        df_to_markdown(
            stable[["Feature", "clinical_core", "elasticnet_freq", "lgbm_freq", "permutation_freq", "stability_score"]]
        ),
        "",
        "## Figures",
        "",
        "- `figures/Figure_Top_Formal_AUC.png`",
        "- `figures/Figure_Feature_Family_Seed_Stability.png`",
        "- `figures/Figure_Stable_Feature_Selection.png`",
        "- `figures/Figure_Best_ROC_PR.png`",
        "- `figures/Figure_Best_Calibration.png`",
        "- `figures/Figure_Best_Confusion_Matrix.png`",
        "- `figures/Figure_Best_DCA.png`",
        "- `figures/Figure_Key_Feature_Boxplots.png`",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    data = build_feature_frame(out_dir, force_rebuild=args.force_rebuild)
    seeds = parse_seeds(args.seeds)
    if args.quick:
        seeds = seeds[: max(1, args.quick_seeds)]
    feature_sets = data.feature_sets
    if args.feature_sets:
        requested = [x.strip() for x in args.feature_sets.split(",") if x.strip()]
        feature_sets = {k: v for k, v in feature_sets.items() if k in requested}
    tasks = [(seed, fs, names) for seed in seeds for fs, names in feature_sets.items()]
    jobs = args.jobs if args.jobs > 0 else max(1, (os.cpu_count() or 2) - 1)
    print(f"[3M] tasks={len(tasks)} seeds={seeds} feature_sets={list(feature_sets)} jobs={jobs}")
    start = time.time()
    results = Parallel(n_jobs=jobs, backend="loky", verbose=0)(
        delayed(run_seed_feature_set)(
            seed,
            fs,
            names,
            data.x_train,
            data.x_test,
            data.y_train,
            data.y_test,
            data.groups_train,
            data.pids_train,
            data.pids_test,
            args.quick,
        )
        for seed, fs, names in tasks
    )
    rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    for r, p, s in results:
        rows.extend(r)
        pred_rows.extend(p)
        statuses.append(s)
    all_runs = pd.DataFrame(rows)
    preds = pd.DataFrame(pred_rows)
    status_df = pd.DataFrame(statuses)
    agg = aggregate_results(all_runs)

    table_dir = out_dir / "tables"
    all_runs.to_csv(table_dir / "AllRuns.csv", index=False)
    agg.to_csv(table_dir / "Aggregated_ByConfig.csv", index=False)
    agg.head(20).to_csv(table_dir / "Top20_Formal_By_OOF_AUC.csv", index=False)
    agg.sort_values(["Mean_OOF_PR_AUC", "Mean_OOF_AUC"], ascending=[False, False]).head(20).to_csv(
        table_dir / "Top20_Formal_By_OOF_PR_AUC.csv", index=False
    )
    all_runs.sort_values(["Test_AUC", "Test_Accuracy"], ascending=[False, False]).head(50).to_csv(
        table_dir / "Top50_TEST_AWARE_CeilingProbe.csv", index=False
    )
    agg.sort_values(["Mean_Test_PR_AUC", "Mean_Test_AUC"], ascending=[False, False]).head(20).to_csv(
        table_dir / "Top20_TEST_AWARE_By_Mean_Test_PR_AUC.csv", index=False
    )
    status_df.to_csv(table_dir / "Run_Status.csv", index=False)
    if len(preds):
        preds.to_parquet(table_dir / "Predictions_AllRuns.parquet", index=False)
    save_main_figures(out_dir, all_runs, agg, preds, data)
    write_readme(out_dir, data, all_runs, agg)
    elapsed = time.time() - start
    print(f"[3M] finished in {elapsed/60:.1f} min")
    print(agg.head(10)[["FeatureSet", "Model", "Calibration", "ThresholdRule", "Mean_OOF_AUC", "Mean_Test_AUC", "Mean_Test_Accuracy"]])


def main() -> None:
    parser = argparse.ArgumentParser(description="3M binary extreme fixed-landmark benchmark")
    parser.add_argument("--out-dir", default=str(OUT_DEFAULT))
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="Run a small smoke benchmark")
    parser.add_argument("--quick-seeds", type=int, default=2)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--feature-sets", default="", help="Comma-separated subset of feature sets")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
