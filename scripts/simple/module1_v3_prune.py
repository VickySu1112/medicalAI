#!/usr/bin/env python
"""M1 v3 feature pruning exploration: how many features are really needed?

Compares the full 10-feature baseline against four candidate slim
variants, motivated by the multi-signal ranking (5-fold stability +
permutation importance + bootstrap selection frequency + LOO ΔAUC):

  v10  Full (baseline)            — ThyroidW, Sex, Uptake24h, HalfLife,
                                    TRAb, TgAb, TPOAb, FT4, TSH, LogDur
  v9   Drop TgAb                  — TgAb has 5-fold stability = 0/5,
                                    bootstrap = 0.33, PI ≈ 0
  v8   Drop TgAb + HalfLife       — both 5-fold = 0/5 (LASSO never picked
                                    them naturally in any of the 5 folds)
  v6   Drop the 4 weakest         — drop HalfLife, TgAb, Uptake24h, TRAb;
                                    keep ThyroidW, LogDur, TPOAb, Sex,
                                    TSH, FT4 (the 6 with 5/5 stability
                                    OR explicit clinical familiarity)
  v4   Top-4 only                 — ThyroidW, LogDur, TPOAb, Sex (the 4
                                    with bootstrap freq ≥ 0.78 + LOO
                                    direction-consistent)

For each variant: 5-fold OOF L2-logistic + Platt + temporal predictions;
1000-rep paired bootstrap ΔAUC vs v10. Outputs CSV + JSON + plot.

Forbidden unique-patient count literal never appears (audit at runtime
via str(890 - 1)).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

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

warnings.filterwarnings("ignore")

PY_SEED = 2025
OOF_SEED = 13
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 7

ROOT = Path(__file__).resolve().parents[2]
V3_DIR = ROOT / "results" / "module1_v3"
V3_FIG = V3_DIR / "figures"
V3_TAB = V3_DIR / "tables"
FROZEN = ROOT / "results" / "module1_baseline_ml_benchmark" / "tables" / "module1_frozen_feature_matrix.csv"

V10 = [
    "Sex", "ThyroidW", "Uptake24h", "HalfLife", "TRAb",
    "TGAb", "TPOAb", "FT4_0M", "TSH_0M",
    "log1p_DiseaseDuration_Months_Aug",
]
V9 = [f for f in V10 if f != "TGAb"]
V8 = [f for f in V10 if f not in {"TGAb", "HalfLife"}]
V6 = [f for f in V10 if f not in {"TGAb", "HalfLife", "Uptake24h", "TRAb"}]
V4 = ["ThyroidW", "log1p_DiseaseDuration_Months_Aug", "TPOAb", "Sex"]

VARIANTS = [
    ("v10 (Full)", V10, "v10"),
    ("v9 (− TgAb)", V9, "v9"),
    ("v8 (− TgAb − HalfLife)", V8, "v8"),
    ("v6 (drop 4 weakest)", V6, "v6"),
    ("v4 (Top-4 only)", V4, "v4"),
]

PRETTY = {
    "Sex": "Sex", "ThyroidW": "Thyroid weight",
    "Uptake24h": "24h RAI uptake", "HalfLife": "Effective iodine half-life",
    "TRAb": "TRAb", "TGAb": "TgAb", "TPOAb": "TPOAb",
    "FT4_0M": "FT4 (baseline)", "TSH_0M": "TSH (baseline)",
    "log1p_DiseaseDuration_Months_Aug": "Log disease duration",
}


def load_data():
    frozen = pd.read_csv(FROZEN)
    needed = {"Episode_Index", "Split", "OOF_Fold", "Y"}
    feat = [c for c in frozen.columns if c not in needed]
    for col in feat:
        frozen[col] = pd.to_numeric(frozen[col], errors="coerce")
    frozen[feat] = frozen[feat].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frozen, frozen["Split"].eq("Development").to_numpy(), frozen["Split"].eq("Temporal").to_numpy()


def make_l2():
    return Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(
            penalty="l2", solver="lbfgs", C=1.0, max_iter=5000, random_state=PY_SEED)),
    ])


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


def paired_delta(y, p_a, p_b, *, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
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


def calib_intercept_slope(y, p):
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs", max_iter=2000)
    lr.fit(z, y)
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


def plot_summary(df: pd.DataFrame, df_delta: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))

    # Left: standalone temporal AUC with 95% CI per variant
    ax = axes[0]
    y = np.arange(len(df))
    means = df["Tmp_AUC_mean"].to_numpy()
    los = (df["Tmp_AUC_mean"] - df["Tmp_AUC_CI_Low"]).to_numpy()
    his = (df["Tmp_AUC_CI_High"] - df["Tmp_AUC_mean"]).to_numpy()
    colors = ["#a23b3b" if r["Label"].startswith("v10") else "#1d4e89" for _, r in df.iterrows()]
    ax.errorbar(means, y, xerr=[los, his], fmt="o", color="black", capsize=5, zorder=4)
    for i, (m, c) in enumerate(zip(means, colors)):
        ax.scatter([m], [i], s=120, color=c, edgecolor="black", zorder=5)
    ax.set_yticks(y); ax.set_yticklabels(df["Label"].tolist())
    ax.invert_yaxis()
    ax.axvline(0.5, color="#888", linestyle=":", linewidth=1)
    ax.set_xlim(0.55, 0.80)
    ax.set_xlabel("Temporal-test ROC-AUC (95% CI)")
    ax.set_title("M1 v3 — feature pruning candidates (standalone AUC)")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for i, (m, lo, hi, k) in enumerate(zip(means, df["Tmp_AUC_CI_Low"], df["Tmp_AUC_CI_High"], df["K"])):
        ax.text(hi + 0.003, i, f"k={int(k)}  {m:.3f}  [{lo:.3f}, {hi:.3f}]",
                fontsize=7, color="#444", va="center")

    # Right: paired ΔAUC vs v10 baseline (excluded v10 from this panel)
    ax = axes[1]
    sub = df_delta[df_delta["Variant"] != "v10"].reset_index(drop=True)
    y = np.arange(len(sub))
    means = sub["Tmp_Delta_mean"].to_numpy()
    los = sub["Tmp_Delta_CI_Low"].to_numpy()
    his = sub["Tmp_Delta_CI_High"].to_numpy()
    colors = ["#a23b3b" if hi < 0 else ("#1d4e89" if lo > 0 else "#888")
              for lo, hi in zip(los, his)]
    ax.errorbar(means, y,
                xerr=[[m - lo for m, lo in zip(means, los)],
                      [hi - m for m, hi in zip(means, his)]],
                fmt="o", color="black", capsize=5, zorder=4)
    for i, (m, c) in enumerate(zip(means, colors)):
        ax.scatter([m], [i], s=120, color=c, edgecolor="black", zorder=5)
    ax.axvline(0, color="#444", linestyle=":", linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels(sub["Label"].tolist())
    ax.invert_yaxis()
    ax.set_xlim(-0.05, 0.05)
    ax.set_xlabel("Paired ΔAUC vs v10 baseline (temporal)")
    ax.set_title("Paired bootstrap × 1000 — Δ on the same episodes")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for i, (m, lo, hi) in enumerate(zip(means, los, his)):
        sig = "★" if (lo > 0 or hi < 0) else ""
        ax.text(hi + 0.002, i, f"{m:+.4f}{sig}  [{lo:+.4f}, {hi:+.4f}]",
                fontsize=7, color="#444", va="center")

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    forbidden = str(890 - 1)
    if forbidden in Path(__file__).read_text():
        raise RuntimeError("Forbidden unique-patient count literal present in source.")

    V3_FIG.mkdir(parents=True, exist_ok=True)
    V3_TAB.mkdir(parents=True, exist_ok=True)

    frozen, dev_mask, tst_mask = load_data()
    y_dev = frozen.loc[dev_mask, "Y"].to_numpy()
    y_tst = frozen.loc[tst_mask, "Y"].to_numpy()

    rows = []
    cached = {}

    print("=== Standalone performance ===")
    for label, feats, tag in VARIANTS:
        X_dev = frozen.loc[dev_mask, feats].reset_index(drop=True)
        X_tst = frozen.loc[tst_mask, feats].reset_index(drop=True)
        oof, final = fit_oof_platt(X_dev, y_dev)
        p_tst = final.predict_proba(X_tst)[:, 1]
        m_o, lo_o, hi_o = bootstrap_auc_ci(y_dev, oof)
        m_t, lo_t, hi_t = bootstrap_auc_ci(y_tst, p_tst)
        brier_o = float(brier_score_loss(y_dev, oof))
        brier_t = float(brier_score_loss(y_tst, p_tst))
        intc_t, slp_t = calib_intercept_slope(y_tst, p_tst)
        r = {
            "Variant": tag, "Label": label,
            "K": len(feats),
            "Features": ";".join(feats),
            "OOF_AUC_mean": m_o, "OOF_AUC_CI_Low": lo_o, "OOF_AUC_CI_High": hi_o,
            "Tmp_AUC_mean": m_t, "Tmp_AUC_CI_Low": lo_t, "Tmp_AUC_CI_High": hi_t,
            "OOF_Brier": brier_o, "Tmp_Brier": brier_t,
            "Tmp_CalibIntercept": intc_t, "Tmp_CalibSlope": slp_t,
        }
        rows.append(r)
        cached[tag] = {"oof": oof, "tst": p_tst}
        print(
            f"  {label:<28}  k={len(feats):>2}  "
            f"OOF={m_o:.4f} [{lo_o:.3f}, {hi_o:.3f}]  "
            f"Tmp={m_t:.4f} [{lo_t:.3f}, {hi_t:.3f}]  "
            f"Brier={brier_t:.3f}  slope={slp_t:.2f}"
        )

    # Paired ΔAUC vs v10
    print("\n=== Paired ΔAUC vs v10 (temporal) ===")
    delta_rows = []
    base = cached["v10"]
    for tag, label in [(r["Variant"], r["Label"]) for r in rows]:
        if tag == "v10":
            delta_rows.append({"Variant": tag, "Label": label,
                               "OOF_Delta_mean": 0, "OOF_Delta_CI_Low": 0, "OOF_Delta_CI_High": 0,
                               "Tmp_Delta_mean": 0, "Tmp_Delta_CI_Low": 0, "Tmp_Delta_CI_High": 0})
            continue
        cur = cached[tag]
        d_o, lo_o, hi_o = paired_delta(y_dev, cur["oof"], base["oof"])
        d_t, lo_t, hi_t = paired_delta(y_tst, cur["tst"], base["tst"])
        delta_rows.append({
            "Variant": tag, "Label": label,
            "OOF_Delta_mean": d_o, "OOF_Delta_CI_Low": lo_o, "OOF_Delta_CI_High": hi_o,
            "Tmp_Delta_mean": d_t, "Tmp_Delta_CI_Low": lo_t, "Tmp_Delta_CI_High": hi_t,
        })
        sig_t = "★" if (lo_t > 0 or hi_t < 0) else ""
        print(
            f"  {label:<28}  ΔOOF={d_o:+.4f} [{lo_o:+.4f}, {hi_o:+.4f}]  "
            f"ΔTmp={d_t:+.4f}{sig_t} [{lo_t:+.4f}, {hi_t:+.4f}]"
        )

    # Save
    df = pd.DataFrame(rows)
    df.to_csv(V3_TAB / "prune_variants_performance.csv", index=False)
    df_delta = pd.DataFrame(delta_rows)
    df_delta.to_csv(V3_TAB / "prune_variants_delta_vs_v10.csv", index=False)

    summary = {
        "variants": [{
            "tag": r["Variant"], "k": r["K"], "Features": r["Features"],
            "Tmp_AUC": r["Tmp_AUC_mean"], "Tmp_AUC_CI": [r["Tmp_AUC_CI_Low"], r["Tmp_AUC_CI_High"]],
            "Brier": r["Tmp_Brier"], "Calib_slope": r["Tmp_CalibSlope"],
        } for r in rows],
        "deltas_vs_v10": [{
            "tag": d["Variant"],
            "Tmp_Delta_mean": d["Tmp_Delta_mean"],
            "Tmp_Delta_CI": [d["Tmp_Delta_CI_Low"], d["Tmp_Delta_CI_High"]],
            "CI_above_zero": bool(d["Tmp_Delta_CI_Low"] > 0),
            "CI_below_zero": bool(d["Tmp_Delta_CI_High"] < 0),
        } for d in delta_rows if d["Variant"] != "v10"],
    }
    (V3_TAB / "prune_variants_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    plot_summary(df, df_delta, V3_FIG / "Figure_v3_22_Prune_Comparison.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
