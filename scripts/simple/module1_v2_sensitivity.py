#!/usr/bin/env python
"""M1·v2 (augmented 10-feature) sensitivity analyses.

Four complementary stress tests on the clinically-curated 10-feature M1·v2a
augmented pool (core 9 + log1p disease duration):

  S1. VIF (multicollinearity) on the 10 standardized features.
  S2. Sex-stratified AUC on dev OOF and temporal test predictions.
  S3. class_weight='balanced' vs default — calibration / discrimination
      sensitivity to the 40.8% prevalence.
  S4. Multi-seed bootstrap CI stability for temporal ROC-AUC.

All analyses live on dev OOF predictions; temporal test predictions are
re-used only where the prompt specifies (S2 / S4 temporal). The forbidden
unique-patient count literal never appears in this file — runtime computed
via str(890 - 1) for any audit checks.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PY_SEED = 2025
OOF_SEED = 13
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 7

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "module1_v2_lasso_clean"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

SOURCE_TABLE_DIR = ROOT / "results" / "module1_baseline_ml_benchmark" / "tables"
FROZEN_MATRIX_PATH = SOURCE_TABLE_DIR / "module1_frozen_feature_matrix.csv"

M1_V2A_CURATED_FEATURES = [
    "Sex",
    "ThyroidW",
    "Uptake24h",
    "HalfLife",
    "TRAb",
    "TGAb",
    "TPOAb",
    "FT4_0M",
    "TSH_0M",
    "log1p_DiseaseDuration_Months_Aug",
]

PRETTY = {
    "Sex": "Sex",
    "ThyroidW": "Thyroid weight",
    "Uptake24h": "24-hour thyroid RAI uptake",
    "HalfLife": "Effective iodine half-life",
    "TRAb": "TSH receptor antibody",
    "TGAb": "Thyroglobulin antibody",
    "TPOAb": "Thyroid peroxidase antibody",
    "FT4_0M": "FT4 at baseline",
    "TSH_0M": "TSH at baseline",
    "log1p_DiseaseDuration_Months_Aug": "Log disease duration (months)",
}


# ---------------------------------------------------------------------------
# I/O + model helpers
# ---------------------------------------------------------------------------


def load_inputs() -> dict:
    frozen = pd.read_csv(FROZEN_MATRIX_PATH)
    needed = {"Episode_Index", "Split", "OOF_Fold", "Y"}
    feature_cols = [c for c in frozen.columns if c not in needed]
    for col in feature_cols:
        frozen[col] = pd.to_numeric(frozen[col], errors="coerce")
    frozen[feature_cols] = (
        frozen[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    expected = 1003
    if len(frozen) != expected:
        raise RuntimeError(f"Frozen matrix has {len(frozen)} rows; expected {expected}")

    dev = frozen["Split"].eq("Development").to_numpy()
    test = frozen["Split"].eq("Temporal").to_numpy()
    return {"frozen": frozen, "dev_mask": dev, "test_mask": test}


def make_l2(class_weight=None) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    C=1.0,
                    max_iter=5000,
                    random_state=PY_SEED,
                    class_weight=class_weight,
                ),
            ),
        ]
    )


def fit_oof_platt(
    X_dev: pd.DataFrame,
    y_dev: np.ndarray,
    *,
    class_weight=None,
) -> tuple[np.ndarray, CalibratedClassifierCV]:
    """5-fold OOF predictions with per-fold Platt calibration; also returns a
    final full-dev model with Platt for temporal scoring.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=OOF_SEED)
    oof = np.zeros(len(y_dev), dtype=float)
    for tr, va in skf.split(X_dev, y_dev):
        cal = CalibratedClassifierCV(
            base_estimator=make_l2(class_weight=class_weight), method="sigmoid", cv=3
        )
        cal.fit(X_dev.iloc[tr], y_dev[tr])
        oof[va] = cal.predict_proba(X_dev.iloc[va])[:, 1]
    final = CalibratedClassifierCV(
        base_estimator=make_l2(class_weight=class_weight), method="sigmoid", cv=3
    )
    final.fit(X_dev, y_dev)
    return oof, final


def bootstrap_auc_ci(
    y: np.ndarray, p: np.ndarray, *, n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    n_samp = len(y)
    if n_samp == 0:
        return float("nan"), float("nan"), float("nan")
    for _ in range(n):
        idx = rng.integers(0, n_samp, size=n_samp)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    if not aucs:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.mean(aucs)),
        float(np.percentile(aucs, 2.5)),
        float(np.percentile(aucs, 97.5)),
    )


def calib_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Logistic recalibration: fit y ~ 1 + logit(p) on (y, p); report intercept
    and slope. Slope = 1 + intercept = 0 means perfectly calibrated.
    """
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs", max_iter=2000)
    lr.fit(z, y)
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


# ---------------------------------------------------------------------------
# S1: VIF (multicollinearity)
# ---------------------------------------------------------------------------


def s1_vif(frozen: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    dev_mask = frozen["Split"].eq("Development")
    X = frozen.loc[dev_mask, list(features)].copy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    rows: list[dict] = []
    for i, f in enumerate(features):
        try:
            v = variance_inflation_factor(Xs, i)
        except Exception:
            v = float("nan")
        rows.append({"Feature": f, "PrettyLabel": PRETTY[f], "VIF": float(v)})
    df = pd.DataFrame(rows).sort_values("VIF", ascending=False)
    return df


def plot_s1(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = df["PrettyLabel"].tolist()
    vifs = df["VIF"].tolist()
    y_pos = np.arange(len(labels))
    bar_colors = ["#a23b3b" if v >= 10 else "#d28b18" if v >= 5 else "#456ea6" for v in vifs]
    ax.barh(y_pos, vifs, color=bar_colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(5, color="#888", linestyle=":", linewidth=1, label="VIF=5")
    ax.axvline(10, color="#a23b3b", linestyle=":", linewidth=1, label="VIF=10")
    ax.set_xlabel("Variance Inflation Factor (VIF)")
    ax.set_title("S1. Multicollinearity on M1·v2a (10 features) — standardized on development cohort")
    ax.legend(loc="lower right", fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# S2: Sex-stratified AUC
# ---------------------------------------------------------------------------


def s2_subgroup_auc(
    frozen: pd.DataFrame, oof_dev: np.ndarray, prob_test: np.ndarray
) -> pd.DataFrame:
    rows: list[dict] = []
    y_all = frozen["Y"].to_numpy()
    dev_mask = frozen["Split"].eq("Development").to_numpy()
    test_mask = frozen["Split"].eq("Temporal").to_numpy()

    for split, mask, prob in (("Dev_OOF", dev_mask, oof_dev), ("Temporal", test_mask, prob_test)):
        y_split = y_all[mask]
        sex_split = frozen.loc[mask, "Sex"].to_numpy()
        # Overall
        mean, lo, hi = bootstrap_auc_ci(y_split, prob)
        rows.append(
            {
                "Split": split,
                "Subgroup": "Overall",
                "N": int(len(y_split)),
                "Events": int(y_split.sum()),
                "ROC_AUC_mean": mean,
                "CI_Low": lo,
                "CI_High": hi,
            }
        )
        # By Sex: 0 = Male, 1 = Female (standard convention in this dataset)
        for sex_val, label in ((0.0, "Male (Sex=0)"), (1.0, "Female (Sex=1)")):
            sub_mask = sex_split == sex_val
            if sub_mask.sum() < 5 or len(np.unique(y_split[sub_mask])) < 2:
                rows.append(
                    {
                        "Split": split,
                        "Subgroup": label,
                        "N": int(sub_mask.sum()),
                        "Events": int(y_split[sub_mask].sum()),
                        "ROC_AUC_mean": float("nan"),
                        "CI_Low": float("nan"),
                        "CI_High": float("nan"),
                    }
                )
                continue
            mean, lo, hi = bootstrap_auc_ci(y_split[sub_mask], prob[sub_mask])
            rows.append(
                {
                    "Split": split,
                    "Subgroup": label,
                    "N": int(sub_mask.sum()),
                    "Events": int(y_split[sub_mask].sum()),
                    "ROC_AUC_mean": mean,
                    "CI_Low": lo,
                    "CI_High": hi,
                }
            )
    return pd.DataFrame(rows)


def plot_s2(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, split in zip(axes, ("Dev_OOF", "Temporal")):
        sub = df[df["Split"] == split].reset_index(drop=True)
        labels = sub["Subgroup"].tolist()
        means = sub["ROC_AUC_mean"].tolist()
        lo = (sub["ROC_AUC_mean"] - sub["CI_Low"]).abs().tolist()
        hi = (sub["CI_High"] - sub["ROC_AUC_mean"]).abs().tolist()
        y_pos = np.arange(len(labels))
        ax.errorbar(means, y_pos, xerr=[lo, hi], fmt="o", color="#1d4e89", capsize=4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.axvline(0.5, color="#888", linestyle=":", linewidth=1)
        ax.set_xlim(0.40, 0.90)
        ax.set_xlabel(f"ROC-AUC on {split}")
        ax.set_title(f"S2. M1·v2a sex-stratified AUC ({split})")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        # Annotate N (events)
        for i, (n, ev) in enumerate(zip(sub["N"], sub["Events"])):
            ax.text(0.405, i + 0.25, f"N={n} (ev={ev})", fontsize=7, color="#444")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# S3: class_weight sensitivity
# ---------------------------------------------------------------------------


def s3_class_weight(
    X_dev: pd.DataFrame,
    y_dev: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict] = []
    for cw_label, cw in (("default(None)", None), ("balanced", "balanced")):
        oof, final = fit_oof_platt(X_dev, y_dev, class_weight=cw)
        prob_test = final.predict_proba(X_test)[:, 1]
        m_oof, lo_oof, hi_oof = bootstrap_auc_ci(y_dev, oof)
        m_tmp, lo_tmp, hi_tmp = bootstrap_auc_ci(y_test, prob_test)
        brier_oof = brier_score_loss(y_dev, oof)
        brier_tmp = brier_score_loss(y_test, prob_test)
        intc_oof, slp_oof = calib_intercept_slope(y_dev, oof)
        intc_tmp, slp_tmp = calib_intercept_slope(y_test, prob_test)
        rows.append(
            {
                "ClassWeight": cw_label,
                "OOF_AUC_mean": m_oof,
                "OOF_AUC_CI_Low": lo_oof,
                "OOF_AUC_CI_High": hi_oof,
                "OOF_Brier": float(brier_oof),
                "OOF_CalibIntercept": intc_oof,
                "OOF_CalibSlope": slp_oof,
                "Temporal_AUC_mean": m_tmp,
                "Temporal_AUC_CI_Low": lo_tmp,
                "Temporal_AUC_CI_High": hi_tmp,
                "Temporal_Brier": float(brier_tmp),
                "Temporal_CalibIntercept": intc_tmp,
                "Temporal_CalibSlope": slp_tmp,
            }
        )
    return pd.DataFrame(rows)


def plot_s3(
    df: pd.DataFrame,
    X_dev: pd.DataFrame,
    y_dev: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    out: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    # Left: ROC on temporal for both class_weight settings
    ax = axes[0]
    for cw_label, cw, color in (
        ("default(None)", None, "#1d4e89"),
        ("balanced", "balanced", "#a23b3b"),
    ):
        _, final = fit_oof_platt(X_dev, y_dev, class_weight=cw)
        prob = final.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc_val = roc_auc_score(y_test, prob)
        ax.plot(fpr, tpr, color=color, label=f"{cw_label}: AUC={auc_val:.3f}")
    ax.plot([0, 1], [0, 1], color="#aaa", linestyle=":", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("S3a. M1·v2a Temporal ROC: class_weight comparison")
    ax.legend(loc="lower right", fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Right: calibration intercept / slope bar
    ax = axes[1]
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["Temporal_CalibIntercept"], width, label="Temporal intercept", color="#456ea6")
    ax.bar(x + width / 2, df["Temporal_CalibSlope"] - 1, width, label="Temporal slope − 1", color="#d28b18")
    ax.axhline(0, color="#888", linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(df["ClassWeight"].tolist())
    ax.set_ylabel("Calibration deviation from ideal (0)")
    ax.set_title("S3b. M1·v2a Temporal calibration")
    ax.legend(loc="best", fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# S4: Multi-seed bootstrap
# ---------------------------------------------------------------------------


def s4_multi_seed(
    y_test: np.ndarray, prob_test: np.ndarray, seeds: Sequence[int]
) -> pd.DataFrame:
    rows: list[dict] = []
    for s in seeds:
        m, lo, hi = bootstrap_auc_ci(y_test, prob_test, seed=int(s))
        rows.append(
            {
                "Seed": int(s),
                "ROC_AUC_mean": m,
                "CI_Low": lo,
                "CI_High": hi,
                "CI_Width": hi - lo,
            }
        )
    return pd.DataFrame(rows)


def plot_s4(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_pos = np.arange(len(df))
    means = df["ROC_AUC_mean"].tolist()
    lo = (df["ROC_AUC_mean"] - df["CI_Low"]).abs().tolist()
    hi = (df["CI_High"] - df["ROC_AUC_mean"]).abs().tolist()
    ax.errorbar(means, y_pos, xerr=[lo, hi], fmt="o", color="#1d4e89", capsize=4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"seed={s}" for s in df["Seed"].tolist()])
    ax.invert_yaxis()
    ax.axvline(np.mean(means), color="#888", linestyle=":", linewidth=1, label=f"mean ROC = {np.mean(means):.3f}")
    ax.set_xlabel("Temporal-test ROC-AUC (bootstrap × 1000)")
    ax.set_title("S4. M1·v2a multi-seed bootstrap stability")
    ax.legend(loc="best", fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for i, (m, lo_i, hi_i) in enumerate(zip(means, df["CI_Low"], df["CI_High"])):
        ax.text(hi_i + 0.003, i, f"[{lo_i:.3f}, {hi_i:.3f}]", fontsize=7, color="#444", va="center")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Forbidden-token runtime check (CLAUDE.md hard constraint).
    forbidden = str(890 - 1)
    if forbidden in Path(__file__).read_text():
        raise RuntimeError("Forbidden unique-patient count literal present in source.")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs()
    frozen = inputs["frozen"]
    dev_mask = inputs["dev_mask"]
    test_mask = inputs["test_mask"]
    y_all = frozen["Y"].to_numpy()
    X_all = frozen[M1_V2A_CURATED_FEATURES].copy()

    X_dev = X_all.loc[dev_mask].reset_index(drop=True)
    y_dev = y_all[dev_mask]
    X_test = X_all.loc[test_mask].reset_index(drop=True)
    y_test = y_all[test_mask]

    # ---- S1 VIF ----
    print("S1. VIF …")
    vif_df = s1_vif(frozen, M1_V2A_CURATED_FEATURES)
    vif_df.to_csv(TABLE_DIR / "sensitivity_vif_augmented.csv", index=False)
    plot_s1(vif_df, FIG_DIR / "Figure_S1_VIF_augmented.png")
    print(f"  VIF range: [{vif_df['VIF'].min():.2f}, {vif_df['VIF'].max():.2f}]")

    # ---- S2 Sex-stratified ----
    print("S2. Sex-stratified AUC …")
    oof_dev, final_default = fit_oof_platt(X_dev, y_dev, class_weight=None)
    prob_test = final_default.predict_proba(X_test)[:, 1]
    s2_df = s2_subgroup_auc(frozen, oof_dev, prob_test)
    s2_df.to_csv(TABLE_DIR / "sensitivity_subgroup_auc_augmented.csv", index=False)
    plot_s2(s2_df, FIG_DIR / "Figure_S2_Subgroup_AUC_augmented.png")
    print(f"  Dev OOF overall: {s2_df.iloc[0]['ROC_AUC_mean']:.3f}")

    # ---- S3 class_weight ----
    print("S3. class_weight sensitivity …")
    s3_df = s3_class_weight(X_dev, y_dev, X_test, y_test)
    s3_df.to_csv(TABLE_DIR / "sensitivity_class_weight_augmented.csv", index=False)
    plot_s3(s3_df, X_dev, y_dev, X_test, y_test, FIG_DIR / "Figure_S3_ClassWeight_augmented.png")
    delta = s3_df.iloc[1]["Temporal_AUC_mean"] - s3_df.iloc[0]["Temporal_AUC_mean"]
    print(f"  Δ(balanced − default) temporal AUC = {delta:+.4f}")

    # ---- S4 Multi-seed bootstrap ----
    print("S4. Multi-seed bootstrap …")
    seeds = [2025, 2026, 2027, 2028, 2029]
    s4_df = s4_multi_seed(y_test, prob_test, seeds)
    s4_df.to_csv(TABLE_DIR / "sensitivity_multi_seed_augmented.csv", index=False)
    plot_s4(s4_df, FIG_DIR / "Figure_S4_MultiSeed_augmented.png")
    print(
        f"  Mean ROC across {len(seeds)} seeds: {s4_df['ROC_AUC_mean'].mean():.4f} "
        f"(±{s4_df['ROC_AUC_mean'].std():.4f}); "
        f"mean CI width {s4_df['CI_Width'].mean():.4f}"
    )

    # Summary JSON
    summary = {
        "S1_VIF_max": float(vif_df["VIF"].max()),
        "S1_VIF_max_feature": str(vif_df.iloc[0]["Feature"]),
        "S2_overall_dev_AUC": float(s2_df.iloc[0]["ROC_AUC_mean"]),
        "S2_overall_temporal_AUC": float(s2_df[s2_df["Split"] == "Temporal"].iloc[0]["ROC_AUC_mean"]),
        "S3_delta_temporal_AUC_balanced_vs_default": float(delta),
        "S4_seed_mean_AUC": float(s4_df["ROC_AUC_mean"].mean()),
        "S4_seed_std_AUC": float(s4_df["ROC_AUC_mean"].std()),
        "S4_mean_CI_width": float(s4_df["CI_Width"].mean()),
    }
    (TABLE_DIR / "sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print("\nDone. Sensitivity tables + figures saved.")


if __name__ == "__main__":
    main()
