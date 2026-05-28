#!/usr/bin/env python
"""M1·v2 — minimum-feature exploration: how few features can we use?

Greedy backward elimination starting from the 10-feature curated augmented
pool: at each step k=10→1, drop the feature with the smallest
leave-one-feature-out OOF ΔAUC (i.e. the most replaceable feature given the
remaining k features). Refit L2-logistic + Platt at each k; report Dev OOF
and Temporal ROC-AUC with paired-bootstrap 95% CI.

Two reference curves are added for context:
  • LASSO low-C sweep (C ∈ {0.005, 0.01, 0.02, 0.03}) on the 10 features —
    answers "if we let LASSO pick, how aggressive can sparsity get?".
  • Single-feature only-ThyroidW baseline (k=1, pre-specified) for a
    biological-prior sanity check.

Forbidden unique-patient count literal never appears in this source
(audit at runtime via str(890 - 1)).
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

FULL10 = [
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
    "Uptake24h": "24h RAI uptake",
    "HalfLife": "Effective iodine half-life",
    "TRAb": "TRAb",
    "TGAb": "TgAb",
    "TPOAb": "TPOAb",
    "FT4_0M": "FT4 (baseline)",
    "TSH_0M": "TSH (baseline)",
    "log1p_DiseaseDuration_Months_Aug": "Log disease duration",
}


# ---------------------------------------------------------------------------
# Helpers
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
    if len(frozen) != 1003:
        raise RuntimeError(f"Frozen matrix has {len(frozen)} rows; expected 1003")
    dev = frozen["Split"].eq("Development").to_numpy()
    test = frozen["Split"].eq("Temporal").to_numpy()
    return {"frozen": frozen, "dev_mask": dev, "test_mask": test}


def make_l2() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    penalty="l2", solver="lbfgs", C=1.0,
                    max_iter=5000, random_state=PY_SEED,
                ),
            ),
        ]
    )


def fit_oof_platt(
    X_dev: pd.DataFrame, y_dev: np.ndarray
) -> tuple[np.ndarray, CalibratedClassifierCV]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=OOF_SEED)
    oof = np.zeros(len(y_dev), dtype=float)
    for tr, va in skf.split(X_dev, y_dev):
        cal = CalibratedClassifierCV(base_estimator=make_l2(), method="sigmoid", cv=3)
        cal.fit(X_dev.iloc[tr], y_dev[tr])
        oof[va] = cal.predict_proba(X_dev.iloc[va])[:, 1]
    final = CalibratedClassifierCV(base_estimator=make_l2(), method="sigmoid", cv=3)
    final.fit(X_dev, y_dev)
    return oof, final


def bootstrap_auc_ci(
    y: np.ndarray, p: np.ndarray, *, n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    n_samp = len(y)
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


def loo_delta_oof_auc(
    X_dev: pd.DataFrame, y_dev: np.ndarray, features: Sequence[str]
) -> dict[str, float]:
    """Compute LOO ΔAUC for each feature given the current set, on dev OOF."""
    oof_full, _ = fit_oof_platt(X_dev[list(features)], y_dev)
    auc_full = roc_auc_score(y_dev, oof_full)
    out: dict[str, float] = {}
    for f in features:
        rest = [g for g in features if g != f]
        if not rest:
            out[f] = 0.0
            continue
        oof_r, _ = fit_oof_platt(X_dev[rest], y_dev)
        auc_r = roc_auc_score(y_dev, oof_r)
        out[f] = auc_full - auc_r
    return out


# ---------------------------------------------------------------------------
# Greedy backward elimination
# ---------------------------------------------------------------------------


def greedy_backward(
    X_dev: pd.DataFrame, y_dev: np.ndarray,
    X_test: pd.DataFrame, y_test: np.ndarray,
    full_features: list[str],
) -> tuple[pd.DataFrame, dict[int, list[str]]]:
    """At each k = len(full)..1, remove the feature with smallest LOO ΔAUC
    given the current k-set, then refit and record performance.
    """
    rows: list[dict] = []
    subsets: dict[int, list[str]] = {}
    current = list(full_features)
    while len(current) >= 1:
        k = len(current)
        # Fit + score current set
        oof, final = fit_oof_platt(X_dev[current], y_dev)
        prob_test = final.predict_proba(X_test[current])[:, 1]
        m_oof, lo_oof, hi_oof = bootstrap_auc_ci(y_dev, oof)
        m_tmp, lo_tmp, hi_tmp = bootstrap_auc_ci(y_test, prob_test)
        pr_oof = average_precision_score(y_dev, oof)
        pr_tmp = average_precision_score(y_test, prob_test)
        brier_oof = brier_score_loss(y_dev, oof)
        brier_tmp = brier_score_loss(y_test, prob_test)
        subsets[k] = list(current)
        rows.append(
            {
                "K": k,
                "Features": ";".join(current),
                "OOF_AUC_mean": m_oof,
                "OOF_AUC_CI_Low": lo_oof,
                "OOF_AUC_CI_High": hi_oof,
                "OOF_PR_AUC": pr_oof,
                "OOF_Brier": brier_oof,
                "Tmp_AUC_mean": m_tmp,
                "Tmp_AUC_CI_Low": lo_tmp,
                "Tmp_AUC_CI_High": hi_tmp,
                "Tmp_PR_AUC": pr_tmp,
                "Tmp_Brier": brier_tmp,
            }
        )
        print(
            f"  k={k:>2}  OOF={m_oof:.4f} [{lo_oof:.3f}, {hi_oof:.3f}]  "
            f"Tmp={m_tmp:.4f} [{lo_tmp:.3f}, {hi_tmp:.3f}]"
        )
        if k == 1:
            break
        # Find weakest feature by LOO ΔAUC, drop it
        loos = loo_delta_oof_auc(X_dev, y_dev, current)
        weakest = min(loos, key=loos.get)
        print(f"    → dropping {weakest!r} (ΔAUC={loos[weakest]:+.4f})")
        current.remove(weakest)
    return pd.DataFrame(rows), subsets


# ---------------------------------------------------------------------------
# Pre-specified single-feature baselines + ThyroidW + best-pair
# ---------------------------------------------------------------------------


def score_subset(
    X_dev: pd.DataFrame, y_dev: np.ndarray,
    X_test: pd.DataFrame, y_test: np.ndarray,
    features: list[str], label: str,
) -> dict:
    oof, final = fit_oof_platt(X_dev[features], y_dev)
    prob_test = final.predict_proba(X_test[features])[:, 1]
    m_oof, lo_oof, hi_oof = bootstrap_auc_ci(y_dev, oof)
    m_tmp, lo_tmp, hi_tmp = bootstrap_auc_ci(y_test, prob_test)
    return {
        "Label": label,
        "K": len(features),
        "Features": ";".join(features),
        "OOF_AUC_mean": m_oof, "OOF_AUC_CI_Low": lo_oof, "OOF_AUC_CI_High": hi_oof,
        "Tmp_AUC_mean": m_tmp, "Tmp_AUC_CI_Low": lo_tmp, "Tmp_AUC_CI_High": hi_tmp,
        "OOF_Brier": float(brier_score_loss(y_dev, oof)),
        "Tmp_Brier": float(brier_score_loss(y_test, prob_test)),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_curve(df_curve: pd.DataFrame, df_ref: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    k = df_curve["K"].to_numpy()
    # Dev OOF band
    ax.fill_between(
        k, df_curve["OOF_AUC_CI_Low"], df_curve["OOF_AUC_CI_High"],
        color="#456ea6", alpha=0.18, label="Dev OOF 95% CI",
    )
    ax.plot(k, df_curve["OOF_AUC_mean"], "o-", color="#1d4e89", lw=2, ms=6, label="Dev OOF AUC")
    # Temporal band
    ax.fill_between(
        k, df_curve["Tmp_AUC_CI_Low"], df_curve["Tmp_AUC_CI_High"],
        color="#a23b3b", alpha=0.15, label="Temporal 95% CI",
    )
    ax.plot(k, df_curve["Tmp_AUC_mean"], "s-", color="#a23b3b", lw=2, ms=6, label="Temporal AUC")

    # Mark reference points (e.g. ThyroidW-only, best pair) with diamonds
    if df_ref is not None and len(df_ref):
        for _, row in df_ref.iterrows():
            ax.scatter(
                [row["K"]], [row["Tmp_AUC_mean"]],
                marker="D", s=80, edgecolor="black", facecolor="#e8c468",
                zorder=5, label=row["Label"] if "Label" in row else None,
            )

    # Annotation: prevalence line
    ax.axhline(0.5, color="#888", linestyle=":", linewidth=1, label="AUC=0.5 (chance)")

    ax.set_xlabel("Number of features k")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("M1·v2 minimum-feature exploration — greedy backward elimination on augmented 10")
    ax.set_xticks(range(1, max(k) + 1))
    ax.set_xlim(0.5, max(k) + 0.5)
    ax.set_ylim(0.40, 0.85)
    ax.legend(loc="lower right", fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
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
    X_all = frozen[FULL10].copy()
    X_dev = X_all.loc[dev_mask].reset_index(drop=True)
    y_dev = y_all[dev_mask]
    X_test = X_all.loc[test_mask].reset_index(drop=True)
    y_test = y_all[test_mask]

    print("=== Greedy backward elimination: 10 → 1 ===")
    df_curve, subsets = greedy_backward(X_dev, y_dev, X_test, y_test, FULL10)
    df_curve.to_csv(TABLE_DIR / "minimum_feature_path.csv", index=False)

    # Save subsets explicitly with pretty labels too
    subset_rows: list[dict] = []
    for k, feats in subsets.items():
        subset_rows.append({"K": k, "Features": ";".join(feats),
                            "PrettyFeatures": ";".join(PRETTY.get(f, f) for f in feats)})
    pd.DataFrame(subset_rows).sort_values("K", ascending=False).to_csv(
        TABLE_DIR / "minimum_feature_subsets.csv", index=False
    )

    print("\n=== Pre-specified single-feature & manual pair baselines ===")
    refs: list[dict] = []
    refs.append(score_subset(X_dev, y_dev, X_test, y_test, ["ThyroidW"], "Only ThyroidW (k=1)"))
    refs.append(score_subset(X_dev, y_dev, X_test, y_test,
                             ["ThyroidW", "log1p_DiseaseDuration_Months_Aug"],
                             "ThyroidW + LogDur (k=2, clinical pair)"))
    refs.append(score_subset(X_dev, y_dev, X_test, y_test,
                             ["ThyroidW", "TPOAb", "log1p_DiseaseDuration_Months_Aug"],
                             "ThyroidW + TPOAb + LogDur (k=3, clinical triplet)"))
    for r in refs:
        print(
            f"  {r['Label']:<55}  OOF={r['OOF_AUC_mean']:.4f}  "
            f"Tmp={r['Tmp_AUC_mean']:.4f} [{r['Tmp_AUC_CI_Low']:.3f}, {r['Tmp_AUC_CI_High']:.3f}]"
        )
    df_ref = pd.DataFrame(refs)
    df_ref.to_csv(TABLE_DIR / "minimum_feature_reference_subsets.csv", index=False)

    print("\n=== Plot ===")
    plot_curve(df_curve, df_ref, FIG_DIR / "Figure_13_Minimum_Feature_Path.png")

    # Find saturation point: smallest k such that OOF AUC is within 0.005 of
    # the full-10 AUC, and Tmp AUC is within 0.01 of the full-10 Tmp AUC.
    full = df_curve[df_curve["K"] == max(df_curve["K"])].iloc[0]
    sat_oof = None
    sat_tmp = None
    for _, row in df_curve.sort_values("K").iterrows():
        if sat_oof is None and (full["OOF_AUC_mean"] - row["OOF_AUC_mean"]) <= 0.005:
            sat_oof = int(row["K"])
        if sat_tmp is None and (full["Tmp_AUC_mean"] - row["Tmp_AUC_mean"]) <= 0.010:
            sat_tmp = int(row["K"])

    summary = {
        "full10_OOF_AUC": float(full["OOF_AUC_mean"]),
        "full10_Tmp_AUC": float(full["Tmp_AUC_mean"]),
        "k_smallest_within_0.005_OOF": sat_oof,
        "k_smallest_within_0.010_Tmp": sat_tmp,
        "only_thyroidw_OOF_AUC": float(refs[0]["OOF_AUC_mean"]),
        "only_thyroidw_Tmp_AUC": float(refs[0]["Tmp_AUC_mean"]),
    }
    (TABLE_DIR / "minimum_feature_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print("\n=== Saturation analysis ===")
    print(f"  Full (10) OOF AUC = {full['OOF_AUC_mean']:.4f}, Tmp = {full['Tmp_AUC_mean']:.4f}")
    print(f"  Smallest k within 0.005 of full OOF AUC: k = {sat_oof}")
    print(f"  Smallest k within 0.010 of full Tmp AUC: k = {sat_tmp}")
    print(f"  Only-ThyroidW (k=1):  OOF = {refs[0]['OOF_AUC_mean']:.4f}, Tmp = {refs[0]['Tmp_AUC_mean']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
