#!/usr/bin/env python
"""M1·v2 — does normalising Thyroid weight by an anthropometric covariate
(Weight / BSA / Height / BMI) beat raw ThyroidW?

Seven single-variable variants in head-to-head on temporal AUC:
  V1. ThyroidW (raw)                                       — baseline (g)
  V2. ThyroidW / Weight                                    — g/kg, body-mass normalised
  V3. ThyroidW / BSA (Mosteller: sqrt(H_cm × W_kg / 3600)) — g/m², surface normalised
  V4. ThyroidW / Height                                    — g/m
  V5. ThyroidW / BMI                                       — g·m²/kg
  V6. log1p(ThyroidW)                                      — log-transformed raw
  V7. log1p(ThyroidW / BSA)                                — log-transformed BSA-normalised

Each variant is fit as a single-feature L2-logistic + Platt + 5-fold OOF on
development; paired bootstrap ΔAUC vs raw ThyroidW for both Dev OOF and
Temporal. Forbidden unique-patient count literal never appears (runtime
audit via str(890 - 1)).
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
from sklearn.metrics import roc_auc_score, brier_score_loss
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


def fit_oof_platt(X_dev, y_dev):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=OOF_SEED)
    oof = np.zeros(len(y_dev), dtype=float)
    for tr, va in skf.split(X_dev, y_dev):
        cal = CalibratedClassifierCV(base_estimator=make_l2(), method="sigmoid", cv=3)
        cal.fit(X_dev.iloc[tr], y_dev[tr])
        oof[va] = cal.predict_proba(X_dev.iloc[va])[:, 1]
    final = CalibratedClassifierCV(base_estimator=make_l2(), method="sigmoid", cv=3)
    final.fit(X_dev, y_dev)
    return oof, final


def bootstrap_auc_ci(y, p, *, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    if not aucs:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def paired_bootstrap_delta(y, p_a, p_b, *, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    if not diffs:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def score(frame_dev_X: pd.DataFrame, y_dev, frame_test_X: pd.DataFrame, y_test, label: str):
    oof, final = fit_oof_platt(frame_dev_X, y_dev)
    prob_test = final.predict_proba(frame_test_X)[:, 1]
    m_oof, lo_oof, hi_oof = bootstrap_auc_ci(y_dev, oof)
    m_tmp, lo_tmp, hi_tmp = bootstrap_auc_ci(y_test, prob_test)
    return {
        "Label": label,
        "OOF_AUC_mean": m_oof, "OOF_AUC_CI_Low": lo_oof, "OOF_AUC_CI_High": hi_oof,
        "Tmp_AUC_mean": m_tmp, "Tmp_AUC_CI_Low": lo_tmp, "Tmp_AUC_CI_High": hi_tmp,
        "OOF_Brier": float(brier_score_loss(y_dev, oof)),
        "Tmp_Brier": float(brier_score_loss(y_test, prob_test)),
        "_oof": oof,
        "_test": prob_test,
    }


def build_variants(frozen: pd.DataFrame) -> pd.DataFrame:
    """Construct all 7 single-variable derivatives of ThyroidW + anthropo
    covariates. Returns a wide frame containing the 7 columns + Episode_Index.
    """
    out = pd.DataFrame(index=frozen.index)
    out["Episode_Index"] = frozen["Episode_Index"]
    # V1 raw
    out["V1_ThyroidW"] = frozen["ThyroidW"]
    # V2 g/kg
    weight = frozen["Weight"].replace(0, np.nan)
    out["V2_ThyroidW_per_Weight"] = frozen["ThyroidW"] / weight
    # V3 g/m² (Mosteller BSA: sqrt(H_cm × W_kg / 3600); H stored in metres)
    h_cm = frozen["Height"] * 100.0
    bsa = np.sqrt((h_cm * frozen["Weight"]).clip(lower=0) / 3600.0).replace(0, np.nan)
    out["V3_ThyroidW_per_BSA"] = frozen["ThyroidW"] / bsa
    # V4 g/m (Height is in metres)
    height = frozen["Height"].replace(0, np.nan)
    out["V4_ThyroidW_per_Height"] = frozen["ThyroidW"] / height
    # V5 g·m²/kg (ThyroidW / BMI)
    bmi = frozen["BMI"].replace(0, np.nan)
    out["V5_ThyroidW_per_BMI"] = frozen["ThyroidW"] / bmi
    # V6 log1p(raw)
    out["V6_log1p_ThyroidW"] = np.log1p(frozen["ThyroidW"].clip(lower=0))
    # V7 log1p(g/m²)
    out["V7_log1p_ThyroidW_per_BSA"] = np.log1p(out["V3_ThyroidW_per_BSA"].clip(lower=0))
    # Fill any leftover NaNs with 0 (already filled upstream, defensive)
    out = out.fillna(0.0)
    return out


def plot_compare(records, deltas, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: temporal AUC + 95% CI for the 7 variants
    ax = axes[0]
    labels = [r["Label"] for r in records]
    means = [r["Tmp_AUC_mean"] for r in records]
    los = [r["Tmp_AUC_mean"] - r["Tmp_AUC_CI_Low"] for r in records]
    his = [r["Tmp_AUC_CI_High"] - r["Tmp_AUC_mean"] for r in records]
    y_pos = np.arange(len(labels))
    colors = ["#1d4e89" if "raw" in lab.lower() else "#456ea6" for lab in labels]
    ax.errorbar(means, y_pos, xerr=[los, his], fmt="o", color="black", capsize=5, zorder=4)
    for i, (m, c) in enumerate(zip(means, colors)):
        ax.scatter([m], [i], s=100, color=c, zorder=5, edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0.5, color="#888", linestyle=":", linewidth=1)
    ax.set_xlim(0.45, 0.80)
    ax.set_xlabel("Temporal-test ROC-AUC (95% CI)")
    ax.set_title("M1·v2 — single-variable ThyroidW normalisation variants")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for i, (m, lo, hi) in enumerate(zip(means, [r["Tmp_AUC_CI_Low"] for r in records],
                                       [r["Tmp_AUC_CI_High"] for r in records])):
        ax.text(hi + 0.005, i, f"{m:.3f}  [{lo:.3f}, {hi:.3f}]",
                fontsize=7, color="#444", va="center")

    # Right: paired ΔAUC vs V1 raw (Temporal)
    ax = axes[1]
    drop_first = deltas[1:]  # skip V1 vs V1 (always 0)
    labels = [d["Label"] for d in drop_first]
    means = [d["Tmp_Delta_mean"] for d in drop_first]
    los = [d["Tmp_Delta_CI_Low"] for d in drop_first]
    his = [d["Tmp_Delta_CI_High"] for d in drop_first]
    y_pos = np.arange(len(labels))
    colors_d = ["#a23b3b" if lo > 0 else ("#1d4e89" if hi < 0 else "#888") for lo, hi in zip(los, his)]
    ax.errorbar(means, y_pos,
                xerr=[[m - lo for m, lo in zip(means, los)],
                      [hi - m for m, hi in zip(means, his)]],
                fmt="o", color="black", capsize=5, zorder=4)
    for i, (m, c) in enumerate(zip(means, colors_d)):
        ax.scatter([m], [i], s=100, color=c, zorder=5, edgecolor="black")
    ax.axvline(0, color="#444", linestyle=":", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(-0.05, 0.05)
    ax.set_xlabel("Paired ΔAUC vs V1 raw ThyroidW (temporal)")
    ax.set_title("Normalisation variants — paired bootstrap ΔAUC (temporal)")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for i, (m, lo, hi) in enumerate(zip(means, los, his)):
        sig = "*" if (lo > 0 or hi < 0) else ""
        ax.text(hi + 0.002, i, f"{m:+.4f}{sig}  [{lo:+.4f}, {hi:+.4f}]",
                fontsize=7, color="#444", va="center")

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


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
    y_dev = y_all[dev_mask]
    y_test = y_all[test_mask]

    variants = build_variants(frozen)
    feature_cols = [c for c in variants.columns if c != "Episode_Index"]

    # Print descriptive stats on dev
    print("=== Variant descriptives (dev N=802) ===")
    desc = variants.loc[dev_mask, feature_cols].describe(percentiles=[0.05, 0.5, 0.95]).T
    desc = desc[["mean", "std", "min", "5%", "50%", "95%", "max"]]
    print(desc.round(3).to_string())
    print()

    # Score each single-variable variant
    print("=== Single-variable head-to-head ===")
    records = []
    for col in feature_cols:
        X_dev = variants.loc[dev_mask, [col]].reset_index(drop=True)
        X_test = variants.loc[test_mask, [col]].reset_index(drop=True)
        r = score(X_dev, y_dev, X_test, y_test, col)
        records.append(r)
        print(
            f"  {col:<35}  "
            f"OOF={r['OOF_AUC_mean']:.4f} [{r['OOF_AUC_CI_Low']:.3f}, {r['OOF_AUC_CI_High']:.3f}]  "
            f"Tmp={r['Tmp_AUC_mean']:.4f} [{r['Tmp_AUC_CI_Low']:.3f}, {r['Tmp_AUC_CI_High']:.3f}]"
        )

    # Paired bootstrap ΔAUC vs V1 raw
    print("\n=== Paired ΔAUC vs V1 raw (temporal) ===")
    base = records[0]  # V1
    deltas = [{
        "Label": records[0]["Label"], "Tmp_Delta_mean": 0.0, "Tmp_Delta_CI_Low": 0.0,
        "Tmp_Delta_CI_High": 0.0, "OOF_Delta_mean": 0.0, "OOF_Delta_CI_Low": 0.0,
        "OOF_Delta_CI_High": 0.0,
    }]
    for r in records[1:]:
        d_oof_m, d_oof_lo, d_oof_hi = paired_bootstrap_delta(y_dev, r["_oof"], base["_oof"])
        d_tmp_m, d_tmp_lo, d_tmp_hi = paired_bootstrap_delta(y_test, r["_test"], base["_test"])
        deltas.append({
            "Label": r["Label"],
            "OOF_Delta_mean": d_oof_m, "OOF_Delta_CI_Low": d_oof_lo, "OOF_Delta_CI_High": d_oof_hi,
            "Tmp_Delta_mean": d_tmp_m, "Tmp_Delta_CI_Low": d_tmp_lo, "Tmp_Delta_CI_High": d_tmp_hi,
        })
        sig_t = "*" if (d_tmp_lo > 0 or d_tmp_hi < 0) else ""
        sig_o = "*" if (d_oof_lo > 0 or d_oof_hi < 0) else ""
        print(
            f"  vs V1: {r['Label']:<35}  "
            f"ΔOOF={d_oof_m:+.4f}{sig_o} [{d_oof_lo:+.4f}, {d_oof_hi:+.4f}]  "
            f"ΔTmp={d_tmp_m:+.4f}{sig_t} [{d_tmp_lo:+.4f}, {d_tmp_hi:+.4f}]"
        )

    # Save
    df_rec = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in records])
    df_rec.to_csv(TABLE_DIR / "normalize_variants.csv", index=False)
    pd.DataFrame(deltas).to_csv(TABLE_DIR / "normalize_variants_delta_vs_raw.csv", index=False)

    summary = {
        "best_tmp_AUC": float(max(records, key=lambda x: x["Tmp_AUC_mean"])["Tmp_AUC_mean"]),
        "best_tmp_variant": max(records, key=lambda x: x["Tmp_AUC_mean"])["Label"],
        "raw_tmp_AUC": float(records[0]["Tmp_AUC_mean"]),
        "any_variant_sig_better_than_raw_tmp": any(
            d["Tmp_Delta_CI_Low"] > 0 for d in deltas[1:]
        ),
        "any_variant_sig_worse_than_raw_tmp": any(
            d["Tmp_Delta_CI_High"] < 0 for d in deltas[1:]
        ),
    }
    (TABLE_DIR / "normalize_variants_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    plot_compare(records, deltas, FIG_DIR / "Figure_15_Normalize_Variants.png")
    print("\nDone. Figure_15 + 2 CSVs + summary JSON written.")


if __name__ == "__main__":
    main()
