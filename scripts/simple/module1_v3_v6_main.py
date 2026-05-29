#!/usr/bin/env python
"""M1 v3 — v6 (6 features) main analysis pipeline.

Promotes the 6-feature variant to the primary delivery line. The 6
features:
  Sex, Thyroid weight, TPOAb, FT4 baseline, TSH baseline, Log disease duration.

Drops the 4 features that were either never naturally selected by 5-fold
LASSO (HalfLife, TgAb) or only selected in 2/5 folds with PI/LOO ≈ 0
(Uptake24h, TRAb).

Outputs into results/module1_v3/{figures,tables}/:
  Tables
    v6_performance.csv             — dev OOF + temporal ROC/PR/Brier + CI
    v6_or_table.csv                — Wald-CI standardized OR per feature
    v6_calibration.csv             — Brier + intercept + slope
    v6_risk_tiers.csv              — 3/4/5-tier observed NHRH by split
    v6_shap_summary.csv            — mean|SHAP| + signed
    v6_pi.csv                      — permutation importance + CI
    v6_loo_delta_auc.csv           — leave-one-feature-out paired Δ
    v6_sensitivity_summary.json    — VIF / Sex stratified / cw / multi-seed
    v6_subgroup_auc.csv
    v6_class_weight.csv
    v6_multi_seed.csv
    v6_vif.csv
    v6_stability_bootstrap.csv     — 500 bootstraps LASSO@chosen-C
    v6_outcome_dose_correlation.csv

  Figures (v3-namespaced):
    Figure_v3_04_OR_Forest.png
    Figure_v3_05_Calibration.png
    Figure_v3_06_DCA.png
    Figure_v3_07_RiskTiers_paired.png
    Figure_v3_08_SHAP_Beeswarm.png
    Figure_v3_09_SHAP_Bar.png
    Figure_v3_10_PI.png
    Figure_v3_11_LOO_DeltaAUC.png
    Figure_v3_12_FinerTiers.png
    Figure_v3_13_PredictedRisk_vs_Dose.png

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
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
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

try:
    import shap  # type: ignore
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

PY_SEED = 2025
OOF_SEED = 13
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 7

ROOT = Path(__file__).resolve().parents[2]
V3_DIR = ROOT / "results" / "module1_v3"
V3_FIG = V3_DIR / "figures"
V3_TAB = V3_DIR / "tables"
FROZEN = ROOT / "results" / "module1_baseline_ml_benchmark" / "tables" / "module1_frozen_feature_matrix.csv"

V6_FEATURES = [
    "Sex",
    "ThyroidW",
    "TPOAb",
    "FT4_0M",
    "TSH_0M",
    "log1p_DiseaseDuration_Months_Aug",
]

PRETTY = {
    "Sex": "Sex",
    "ThyroidW": "Thyroid weight",
    "TPOAb": "TPOAb",
    "FT4_0M": "FT4 at baseline",
    "TSH_0M": "TSH at baseline",
    "log1p_DiseaseDuration_Months_Aug": "Log disease duration (months)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_inputs() -> dict:
    frozen = pd.read_csv(FROZEN)
    needed = {"Episode_Index", "Split", "OOF_Fold", "Y"}
    feat = [c for c in frozen.columns if c not in needed]
    for col in feat:
        frozen[col] = pd.to_numeric(frozen[col], errors="coerce")
    frozen[feat] = frozen[feat].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(frozen) != 1003:
        raise RuntimeError(f"Frozen matrix has {len(frozen)} rows; expected 1003")
    return {
        "frozen": frozen,
        "dev_mask": frozen["Split"].eq("Development").to_numpy(),
        "test_mask": frozen["Split"].eq("Temporal").to_numpy(),
    }


def make_l2(class_weight=None) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(
            penalty="l2", solver="lbfgs", C=1.0, max_iter=5000,
            random_state=PY_SEED, class_weight=class_weight)),
    ])


def fit_oof_platt(X_dev, y_dev, class_weight=None):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=OOF_SEED)
    oof = np.zeros(len(y_dev), dtype=float)
    for tr, va in skf.split(X_dev, y_dev):
        cal = CalibratedClassifierCV(
            base_estimator=make_l2(class_weight=class_weight),
            method="sigmoid", cv=3)
        cal.fit(X_dev.iloc[tr], y_dev[tr])
        oof[va] = cal.predict_proba(X_dev.iloc[va])[:, 1]
    final = CalibratedClassifierCV(
        base_estimator=make_l2(class_weight=class_weight),
        method="sigmoid", cv=3)
    final.fit(X_dev, y_dev)
    return oof, final


def bootstrap_metric_ci(y, p, metric_fn, *, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            vals.append(metric_fn(y[idx], p[idx]))
        except Exception:
            continue
    if not vals:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


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


def calib_intercept_slope(y, p):
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs", max_iter=2000)
    lr.fit(z, y)
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(max(0, center - half)), float(min(1, center + half))


# ---------------------------------------------------------------------------
# Section runners
# ---------------------------------------------------------------------------


def section_performance(y_dev, oof_dev, y_test, p_test) -> pd.DataFrame:
    rows = []
    for split, y, p in [
        ("Dev_OOF", y_dev, oof_dev),
        ("Temporal_Test", y_test, p_test),
    ]:
        for metric_name, fn in [
            ("ROC_AUC", roc_auc_score),
            ("PR_AUC", average_precision_score),
            ("Brier", brier_score_loss),
        ]:
            m, lo, hi = bootstrap_metric_ci(y, p, fn)
            rows.append({"Split": split, "Metric": metric_name,
                         "Value": m, "CI_Lower": lo, "CI_Upper": hi})
    return pd.DataFrame(rows)


def section_or_table(X_dev_scaled: pd.DataFrame, y_dev: np.ndarray, features) -> pd.DataFrame:
    lr = LogisticRegression(penalty="l2", solver="lbfgs", C=1.0, max_iter=5000, random_state=PY_SEED)
    lr.fit(X_dev_scaled, y_dev)
    # Wald variance via sandwich approximation: use np.linalg.pinv(X'WX)
    p_hat = lr.predict_proba(X_dev_scaled)[:, 1]
    W = p_hat * (1 - p_hat)
    Xa = np.c_[np.ones(len(X_dev_scaled)), X_dev_scaled.to_numpy()]
    XtWX = Xa.T @ (W[:, None] * Xa)
    cov = np.linalg.pinv(XtWX)
    se = np.sqrt(np.maximum(np.diag(cov)[1:], 0))
    coef = lr.coef_[0]
    rows = []
    for i, f in enumerate(features):
        b = coef[i]
        ci_low = b - 1.96 * se[i]
        ci_high = b + 1.96 * se[i]
        z = b / (se[i] if se[i] > 0 else 1e-12)
        p_val = 2 * (1 - stats.norm.cdf(abs(z)))
        rows.append({
            "feature": f, "PrettyLabel": PRETTY.get(f, f),
            "OR": float(np.exp(b)),
            "CI_low": float(np.exp(ci_low)),
            "CI_high": float(np.exp(ci_high)),
            "p": float(p_val),
        })
    return pd.DataFrame(rows)


def plot_or_forest(or_df: pd.DataFrame, out: Path) -> None:
    df = or_df.copy()
    df["log_OR"] = np.log(df["OR"])
    df = df.sort_values("log_OR")
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.5 * len(df) + 1.5)))
    y_pos = np.arange(len(df))
    ax.errorbar(df["OR"], y_pos,
                xerr=[df["OR"] - df["CI_low"], df["CI_high"] - df["OR"]],
                fmt="o", color="#1d4e89", capsize=4, markersize=8, lw=1.2)
    ax.axvline(1.0, color="#888", linestyle=":", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["PrettyLabel"].tolist(), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio (95% CI) — per 1 SD")
    ax.set_title("M1 v6 — standardized OR forest plot (6 features)")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for i, row in df.reset_index(drop=True).iterrows():
        ax.text(max(row["CI_high"], 1.05), i,
                f"  OR={row['OR']:.2f}  [{row['CI_low']:.2f}, {row['CI_high']:.2f}]  p={row['p']:.3g}",
                fontsize=7, va="center", color="#444")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_calibration(y_test, p_test, out: Path) -> None:
    # 10 deciles
    deciles = np.quantile(p_test, np.linspace(0, 1, 11))
    pts = []
    for i in range(10):
        m = (p_test >= deciles[i]) & (p_test <= deciles[i + 1]) if i == 9 else \
            (p_test >= deciles[i]) & (p_test < deciles[i + 1])
        if m.sum() < 3:
            continue
        pts.append((p_test[m].mean(), y_test[m].mean(), m.sum()))
    if not pts:
        return
    xs, ys, ns = zip(*pts)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], color="#888", linestyle="--", linewidth=1, label="Ideal")
    ax.plot(xs, ys, "o-", color="#1d4e89", lw=2, markersize=8, label="Decile binning")
    intc, slp = calib_intercept_slope(y_test, p_test)
    brier = brier_score_loss(y_test, p_test)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability (decile)")
    ax.set_ylabel("Observed NHRH rate")
    ax.set_title(f"M1 v6 — Temporal calibration\nBrier={brier:.3f}  intercept={intc:.3f}  slope={slp:.2f}")
    ax.legend(loc="upper left", fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return intc, slp, brier


def plot_dca(y_test, p_test, out: Path) -> None:
    thresholds = np.linspace(0.01, 0.99, 99)
    prev = y_test.mean()
    n = len(y_test)
    nb_model = []
    nb_all = []
    nb_none = []
    for t in thresholds:
        tp = ((p_test >= t) & (y_test == 1)).sum()
        fp = ((p_test >= t) & (y_test == 0)).sum()
        nb = (tp - fp * (t / (1 - t))) / n
        nb_model.append(nb)
        tp_all = (y_test == 1).sum()
        fp_all = (y_test == 0).sum()
        nb_all.append((tp_all - fp_all * (t / (1 - t))) / n)
        nb_none.append(0.0)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(thresholds, nb_model, color="#1d4e89", lw=2, label="M1 v6 (6 features)")
    ax.plot(thresholds, nb_all, color="#a23b3b", linestyle="--", lw=1.2, label="Treat all")
    ax.plot(thresholds, nb_none, color="#888", linestyle=":", lw=1, label="Treat none")
    ax.set_xlim(0, 1); ax.set_ylim(-0.05, max(0.05, prev + 0.05))
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("M1 v6 — Decision curve (temporal test)")
    ax.legend(loc="upper right", fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def section_risk_tiers(oof_dev, y_dev, p_test, y_test) -> pd.DataFrame:
    rows = []
    for n_tiers, names in [
        (3, ["Low", "Intermediate", "High"]),
        (4, ["Q1", "Q2", "Q3", "Q4"]),
        (5, ["P1", "P2", "P3", "P4", "P5"]),
    ]:
        qs = np.linspace(0, 1, n_tiers + 1)[1:-1]
        cuts = [float(np.quantile(oof_dev, q)) for q in qs]
        for split, prob, y in [("Dev_OOF", oof_dev, y_dev), ("Temporal_Test", p_test, y_test)]:
            last = -np.inf
            for i, c in enumerate(cuts + [np.inf]):
                mask = (prob >= last) & (prob < c) if i < len(cuts) else (prob >= last)
                n = int(mask.sum()); k = int(y[mask].sum())
                lo, hi = wilson_ci(k, n)
                rate = k / n if n > 0 else float("nan")
                rows.append({"N_Tiers": n_tiers, "Split": split, "Tier": names[i],
                             "N": n, "Events": k, "ObservedEventRate": rate,
                             "CI_Low": lo, "CI_High": hi})
                last = c
    return pd.DataFrame(rows)


def plot_paired_tiers_3(tiers_df, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sub = tiers_df[tiers_df["N_Tiers"] == 3].copy()
    splits = ["Dev_OOF", "Temporal_Test"]
    tiers = ["Low", "Intermediate", "High"]
    x = np.arange(len(tiers))
    width = 0.35
    colors = {"Dev_OOF": "#456ea6", "Temporal_Test": "#a23b3b"}
    for j, split in enumerate(splits):
        d = sub[sub["Split"] == split].set_index("Tier").loc[tiers]
        means = d["ObservedEventRate"].to_numpy()
        lo = (d["ObservedEventRate"] - d["CI_Low"]).to_numpy()
        hi = (d["CI_High"] - d["ObservedEventRate"]).to_numpy()
        ax.bar(x + (j - 0.5) * width, means, width, color=colors[split],
               yerr=[lo, hi], capsize=3, label=f"{split.replace('_', ' ')}",
               edgecolor="black", linewidth=0.4)
        for i, n, k in zip(x + (j - 0.5) * width, d["N"], d["Events"]):
            ax.text(i, 0.02, f"N={int(n)}\nev={int(k)}",
                    ha="center", fontsize=7, color="white")
    ax.set_xticks(x); ax.set_xticklabels(tiers, fontsize=10)
    ax.set_ylabel("Observed NHRH rate (Wilson 95% CI)")
    ax.set_title("M1 v6 — Risk tier observed NHRH (dev OOF + temporal paired)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, 0.85)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_finer_tiers(tiers_df, out: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11, 11), sharey=True)
    for row, n_tiers in enumerate((3, 4, 5)):
        for col, split in enumerate(("Dev_OOF", "Temporal_Test")):
            ax = axes[row, col]
            sub = tiers_df[(tiers_df["N_Tiers"] == n_tiers) & (tiers_df["Split"] == split)]
            x = np.arange(len(sub))
            means = sub["ObservedEventRate"].to_numpy()
            lo = (sub["ObservedEventRate"] - sub["CI_Low"]).to_numpy()
            hi = (sub["CI_High"] - sub["ObservedEventRate"]).to_numpy()
            colors = ["#456ea6" if i < len(sub)/3 else ("#d28b18" if i < 2*len(sub)/3 else "#a23b3b") for i in range(len(sub))]
            ax.bar(x, means, color=colors, edgecolor="black", linewidth=0.4,
                   yerr=[lo, hi], capsize=4)
            for i, (n, k) in enumerate(zip(sub["N"], sub["Events"])):
                ax.text(i, means[i] + hi[i] + 0.03, f"N={int(n)}\nev={int(k)}",
                        ha="center", fontsize=7, color="#444")
            ax.set_xticks(x); ax.set_xticklabels(sub["Tier"].tolist(), fontsize=8)
            ax.set_ylim(0, 0.85)
            ax.set_title(f"{n_tiers}-tier {split.replace('_', ' ')}")
            if col == 0:
                ax.set_ylabel("Observed NHRH rate (Wilson 95% CI)")
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
    fig.suptitle("M1 v6 — finer stratification (3 / 4 / 5 tiers)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=160)
    plt.close(fig)


def section_shap(X_dev, y_dev, features):
    """SHAP LinearExplainer on the L2 logistic model."""
    if not HAS_SHAP:
        return None, None
    scaler = StandardScaler().fit(X_dev)
    X_scaled = scaler.transform(X_dev)
    lr = LogisticRegression(penalty="l2", solver="lbfgs", C=1.0, max_iter=5000, random_state=PY_SEED)
    lr.fit(X_scaled, y_dev)
    explainer = shap.LinearExplainer(lr, X_scaled, feature_perturbation="interventional")
    shap_vals = explainer.shap_values(X_scaled)
    return shap_vals, X_scaled


def plot_shap_beeswarm(shap_vals, X_scaled, features, out: Path) -> None:
    if shap_vals is None:
        return
    fig = plt.figure(figsize=(9, 5.5))
    shap.summary_plot(shap_vals, X_scaled, feature_names=[PRETTY.get(f, f) for f in features],
                      show=False, max_display=len(features))
    fig.suptitle("M1 v6 — SHAP beeswarm (6 features, dev OOF)", fontsize=11, y=0.99)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_shap_bar(shap_summary: pd.DataFrame, out: Path) -> None:
    df = shap_summary.sort_values("MeanAbsSHAP", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(df))
    ax.barh(y, df["MeanAbsSHAP"], color="#1d4e89", edgecolor="black", linewidth=0.4)
    ax.set_yticks(y); ax.set_yticklabels(df["PrettyLabel"].tolist(), fontsize=9)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("M1 v6 — SHAP global importance (6 features)")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for i, v in enumerate(df["MeanAbsSHAP"]):
        ax.text(v + 0.005, i, f"{v:.3f}", fontsize=8, va="center", color="#444")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def section_pi(final, X_dev, y_dev, features):
    """Permutation importance with 95% bootstrap CI on the OOF (using final
    re-fit on full dev, scored back on dev)."""
    res = permutation_importance(final, X_dev, y_dev, n_repeats=30,
                                 scoring="roc_auc", random_state=PY_SEED)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for i, f in enumerate(features):
        vals = res.importances[i]
        boot = []
        for _ in range(1000):
            samp = rng.choice(vals, size=len(vals), replace=True)
            boot.append(samp.mean())
        rows.append({"Feature": f, "PrettyLabel": PRETTY.get(f, f),
                     "MeanImportance": float(np.mean(vals)),
                     "StdImportance": float(np.std(vals)),
                     "CI_Low": float(np.percentile(boot, 2.5)),
                     "CI_High": float(np.percentile(boot, 97.5))})
    return pd.DataFrame(rows).sort_values("MeanImportance", ascending=False).reset_index(drop=True)


def plot_pi(pi_df, out: Path) -> None:
    df = pi_df.sort_values("MeanImportance", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    y = np.arange(len(df))
    ax.errorbar(df["MeanImportance"], y,
                xerr=[df["MeanImportance"] - df["CI_Low"], df["CI_High"] - df["MeanImportance"]],
                fmt="o", color="#1d4e89", capsize=4, markersize=8)
    ax.set_yticks(y); ax.set_yticklabels(df["PrettyLabel"].tolist(), fontsize=9)
    ax.set_xlabel("Permutation importance Δ ROC-AUC (30 repeats + bootstrap CI)")
    ax.set_title("M1 v6 — Permutation importance (6 features)")
    ax.axvline(0, color="#888", linestyle=":", linewidth=1)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for i, (m, lo, hi) in enumerate(zip(df["MeanImportance"], df["CI_Low"], df["CI_High"])):
        ax.text(hi + 0.001, i, f"{m:.4f}  [{lo:.4f}, {hi:.4f}]", fontsize=7, va="center", color="#444")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def section_loo(X_dev, y_dev, features):
    """Leave-one-feature-out Δ AUC with paired-bootstrap CI."""
    oof_full, _ = fit_oof_platt(X_dev, y_dev)
    auc_full = roc_auc_score(y_dev, oof_full)
    rows = []
    for f in features:
        rest = [g for g in features if g != f]
        oof_r, _ = fit_oof_platt(X_dev[rest], y_dev)
        d_mean, d_lo, d_hi = paired_bootstrap_delta(y_dev, oof_full, oof_r)
        rows.append({"Feature": f, "PrettyLabel": PRETTY.get(f, f),
                     "FullAUC": float(auc_full),
                     "LOO_AUC": float(roc_auc_score(y_dev, oof_r)),
                     "DeltaAUC": d_mean, "CI_Low": d_lo, "CI_High": d_hi})
    return pd.DataFrame(rows).sort_values("DeltaAUC", ascending=False).reset_index(drop=True)


def plot_loo(loo_df, out: Path) -> None:
    df = loo_df.sort_values("DeltaAUC", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.arange(len(df))
    colors = ["#a23b3b" if lo > 0 else "#1d4e89" for lo in df["CI_Low"]]
    ax.errorbar(df["DeltaAUC"], y,
                xerr=[df["DeltaAUC"] - df["CI_Low"], df["CI_High"] - df["DeltaAUC"]],
                fmt="o", color="black", capsize=4, markersize=8)
    for i, c in enumerate(colors):
        ax.scatter([df["DeltaAUC"].iloc[i]], [i], s=100, color=c, edgecolor="black", zorder=5)
    ax.axvline(0, color="#888", linestyle=":", linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels(df["PrettyLabel"].tolist(), fontsize=9)
    ax.set_xlabel("Leave-one-out Δ OOF AUC (paired bootstrap CI)")
    ax.set_title("M1 v6 — LOO ΔAUC (6 features)")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for i, (m, lo, hi) in enumerate(zip(df["DeltaAUC"], df["CI_Low"], df["CI_High"])):
        sig = "★" if lo > 0 else ""
        ax.text(hi + 0.002, i, f"{m:+.4f}{sig}  [{lo:+.4f}, {hi:+.4f}]", fontsize=7, va="center", color="#444")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def section_vif(X_dev):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_dev)
    rows = []
    for i, f in enumerate(X_dev.columns):
        try:
            v = variance_inflation_factor(Xs, i)
        except Exception:
            v = float("nan")
        rows.append({"Feature": f, "PrettyLabel": PRETTY.get(f, f), "VIF": float(v)})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


def section_subgroup_auc(frozen, dev_mask, test_mask, oof_dev, p_test):
    y_all = frozen["Y"].to_numpy()
    rows = []
    for split, mask, prob in [
        ("Dev_OOF", dev_mask, oof_dev),
        ("Temporal", test_mask, p_test),
    ]:
        y_split = y_all[mask]
        sex_split = frozen.loc[mask, "Sex"].to_numpy()
        m, lo, hi = bootstrap_metric_ci(y_split, prob, roc_auc_score)
        rows.append({"Split": split, "Subgroup": "Overall", "N": int(len(y_split)),
                     "Events": int(y_split.sum()), "ROC_AUC_mean": m,
                     "CI_Low": lo, "CI_High": hi})
        for sex_val, lab in [(0.0, "Male (Sex=0)"), (1.0, "Female (Sex=1)")]:
            sub = sex_split == sex_val
            if sub.sum() < 5 or len(np.unique(y_split[sub])) < 2:
                continue
            mb, lob, hib = bootstrap_metric_ci(y_split[sub], prob[sub], roc_auc_score)
            rows.append({"Split": split, "Subgroup": lab, "N": int(sub.sum()),
                         "Events": int(y_split[sub].sum()), "ROC_AUC_mean": mb,
                         "CI_Low": lob, "CI_High": hib})
    return pd.DataFrame(rows)


def section_class_weight(X_dev, y_dev, X_test, y_test):
    rows = []
    for label, cw in [("default(None)", None), ("balanced", "balanced")]:
        oof, final = fit_oof_platt(X_dev, y_dev, class_weight=cw)
        p_test = final.predict_proba(X_test)[:, 1]
        m_o, lo_o, hi_o = bootstrap_metric_ci(y_dev, oof, roc_auc_score)
        m_t, lo_t, hi_t = bootstrap_metric_ci(y_test, p_test, roc_auc_score)
        intc, slp = calib_intercept_slope(y_test, p_test)
        rows.append({"ClassWeight": label,
                     "OOF_AUC_mean": m_o, "OOF_AUC_CI_Low": lo_o, "OOF_AUC_CI_High": hi_o,
                     "Temporal_AUC_mean": m_t, "Temporal_AUC_CI_Low": lo_t, "Temporal_AUC_CI_High": hi_t,
                     "Temporal_Brier": float(brier_score_loss(y_test, p_test)),
                     "Temporal_CalibIntercept": intc, "Temporal_CalibSlope": slp})
    return pd.DataFrame(rows)


def section_multi_seed(y_test, p_test, seeds=(2025, 2026, 2027, 2028, 2029)):
    rows = []
    for s in seeds:
        m, lo, hi = bootstrap_metric_ci(y_test, p_test, roc_auc_score, seed=int(s))
        rows.append({"Seed": int(s), "ROC_AUC_mean": m,
                     "CI_Low": lo, "CI_High": hi, "CI_Width": hi - lo})
    return pd.DataFrame(rows)


def section_bootstrap_stability(frozen, dev_mask, features, C, reps=500, seed=PY_SEED):
    """500-bootstrap LASSO at chosen C; record per-feature selection freq."""
    X_dev = frozen.loc[dev_mask, features].reset_index(drop=True)
    y_dev = frozen.loc[dev_mask, "Y"].to_numpy()
    counts = {f: 0 for f in features}
    rng = np.random.default_rng(seed)
    for r in range(reps):
        idx = rng.integers(0, len(y_dev), size=len(y_dev))
        if len(np.unique(y_dev[idx])) < 2:
            continue
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(penalty="l1", solver="saga", C=C,
                                       max_iter=10000, random_state=PY_SEED, tol=1e-4)),
        ])
        pipe.fit(X_dev.iloc[idx], y_dev[idx])
        coef = pipe.named_steps["lr"].coef_[0]
        for i, f in enumerate(features):
            if abs(coef[i]) > 1e-8:
                counts[f] += 1
    rows = []
    for f in features:
        rows.append({"Feature": f, "PrettyLabel": PRETTY.get(f, f),
                     "Frequency_k_over_n": counts[f] / reps,
                     "Count": counts[f], "Total": reps})
    return pd.DataFrame(rows).sort_values("Frequency_k_over_n", ascending=False).reset_index(drop=True)


def section_dose_correlation(frozen, dev_mask, test_mask, oof_dev, p_test):
    rows = []
    for split, mask, prob in [("Dev_OOF (N=802)", dev_mask, oof_dev),
                               ("Temporal (N=201)", test_mask, p_test)]:
        dose = frozen.loc[mask, "Dose"].to_numpy()
        dose_per_g = frozen.loc[mask, "IDPG_Dose_per_ThyroidW"].to_numpy()
        for var, vals in [("Dose (mCi)", dose), ("Dose per gram (mCi/g)", dose_per_g)]:
            rho, pval = stats.spearmanr(prob, vals)
            slope, intercept, r, _, _ = stats.linregress(prob, vals)
            rows.append({"Split": split, "Variable": var,
                         "Spearman_rho": float(rho), "Spearman_p": float(pval),
                         "Linear_slope": float(slope), "Linear_R": float(r)})
    return pd.DataFrame(rows)


def plot_dose_vs_risk(frozen, dev_mask, test_mask, oof_dev, p_test, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for col, (split, mask, prob) in enumerate([
        ("Dev OOF (N=802)", dev_mask, oof_dev),
        ("Temporal (N=201)", test_mask, p_test),
    ]):
        dose = frozen.loc[mask, "Dose"].to_numpy()
        dose_per_g = frozen.loc[mask, "IDPG_Dose_per_ThyroidW"].to_numpy()
        ax = axes[0, col]
        ax.scatter(prob, dose, alpha=0.35, s=18, color="#1d4e89")
        rho, pval = stats.spearmanr(prob, dose)
        ax.set_xlabel("M1 v6 predicted NHRH probability")
        ax.set_ylabel("Delivered RAI activity (mCi)")
        ax.set_title(f"OV3a {split}\nSpearman ρ = {rho:.3f} (p = {pval:.2g})")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax = axes[1, col]
        ax.scatter(prob, dose_per_g, alpha=0.35, s=18, color="#1d4e89")
        rho2, pval2 = stats.spearmanr(prob, dose_per_g)
        ax.set_xlabel("M1 v6 predicted NHRH probability")
        ax.set_ylabel("Dose per gram thyroid (mCi/g)")
        ax.set_title(f"OV3b {split}\nSpearman ρ = {rho2:.3f} (p = {pval2:.2g})")
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

    V3_FIG.mkdir(parents=True, exist_ok=True)
    V3_TAB.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs()
    frozen = inputs["frozen"]
    dev_mask = inputs["dev_mask"]
    test_mask = inputs["test_mask"]
    y_dev = frozen.loc[dev_mask, "Y"].to_numpy()
    y_test = frozen.loc[test_mask, "Y"].to_numpy()
    X_dev = frozen.loc[dev_mask, V6_FEATURES].reset_index(drop=True)
    X_test = frozen.loc[test_mask, V6_FEATURES].reset_index(drop=True)

    print("Fitting v6 (6 features) — OOF + final ...")
    oof_dev, final = fit_oof_platt(X_dev, y_dev)
    p_test = final.predict_proba(X_test)[:, 1]
    auc_d = roc_auc_score(y_dev, oof_dev); auc_t = roc_auc_score(y_test, p_test)
    print(f"  Dev OOF AUC = {auc_d:.4f}  Temporal AUC = {auc_t:.4f}")

    # Performance
    perf = section_performance(y_dev, oof_dev, y_test, p_test)
    perf.to_csv(V3_TAB / "v6_performance.csv", index=False)

    # OR Forest
    scaler = StandardScaler().fit(X_dev)
    X_dev_s = pd.DataFrame(scaler.transform(X_dev), columns=V6_FEATURES)
    or_df = section_or_table(X_dev_s, y_dev, V6_FEATURES)
    or_df.to_csv(V3_TAB / "v6_or_table.csv", index=False)
    plot_or_forest(or_df, V3_FIG / "Figure_v3_04_OR_Forest.png")

    # Calibration
    intc, slp, brier = plot_calibration(y_test, p_test, V3_FIG / "Figure_v3_05_Calibration.png")
    pd.DataFrame([{"Split": "Temporal_Test", "Brier": brier,
                   "Calibration_Intercept": intc, "Calibration_Slope": slp}]).to_csv(
        V3_TAB / "v6_calibration.csv", index=False)

    # DCA
    plot_dca(y_test, p_test, V3_FIG / "Figure_v3_06_DCA.png")

    # Risk tiers
    tiers = section_risk_tiers(oof_dev, y_dev, p_test, y_test)
    tiers.to_csv(V3_TAB / "v6_risk_tiers.csv", index=False)
    plot_paired_tiers_3(tiers, V3_FIG / "Figure_v3_07_RiskTiers_paired.png")
    plot_finer_tiers(tiers, V3_FIG / "Figure_v3_12_FinerTiers.png")

    # SHAP
    if HAS_SHAP:
        print("SHAP analysis ...")
        shap_vals, X_scaled = section_shap(X_dev, y_dev, V6_FEATURES)
        plot_shap_beeswarm(shap_vals, X_scaled, V6_FEATURES, V3_FIG / "Figure_v3_08_SHAP_Beeswarm.png")
        shap_summary = pd.DataFrame({
            "Feature": V6_FEATURES,
            "PrettyLabel": [PRETTY.get(f, f) for f in V6_FEATURES],
            "MeanAbsSHAP": np.mean(np.abs(shap_vals), axis=0),
            "MeanSignedSHAP": np.mean(shap_vals, axis=0),
        }).sort_values("MeanAbsSHAP", ascending=False)
        shap_summary.to_csv(V3_TAB / "v6_shap_summary.csv", index=False)
        plot_shap_bar(shap_summary, V3_FIG / "Figure_v3_09_SHAP_Bar.png")
    else:
        print("SHAP not available — skipping.")

    # PI
    print("Permutation importance ...")
    pi_df = section_pi(final, X_dev, y_dev, V6_FEATURES)
    pi_df.to_csv(V3_TAB / "v6_pi.csv", index=False)
    plot_pi(pi_df, V3_FIG / "Figure_v3_10_PI.png")

    # LOO
    print("Leave-one-feature-out ...")
    loo_df = section_loo(X_dev, y_dev, V6_FEATURES)
    loo_df.to_csv(V3_TAB / "v6_loo_delta_auc.csv", index=False)
    plot_loo(loo_df, V3_FIG / "Figure_v3_11_LOO_DeltaAUC.png")

    # VIF
    vif_df = section_vif(X_dev)
    vif_df.to_csv(V3_TAB / "v6_vif.csv", index=False)

    # Subgroup AUC
    sub_df = section_subgroup_auc(frozen, dev_mask, test_mask, oof_dev, p_test)
    sub_df.to_csv(V3_TAB / "v6_subgroup_auc.csv", index=False)

    # class_weight
    cw_df = section_class_weight(X_dev, y_dev, X_test, y_test)
    cw_df.to_csv(V3_TAB / "v6_class_weight.csv", index=False)

    # multi-seed
    ms_df = section_multi_seed(y_test, p_test)
    ms_df.to_csv(V3_TAB / "v6_multi_seed.csv", index=False)

    # Bootstrap stability (use C=0.20 which selected ~10 features in v2 LASSO
    # path; on v6 candidate pool of 6 we use C=0.30 (most relaxed))
    print("LASSO bootstrap stability (500 reps) ...")
    boot_df = section_bootstrap_stability(frozen, dev_mask, V6_FEATURES, C=0.30, reps=500)
    boot_df.to_csv(V3_TAB / "v6_stability_bootstrap.csv", index=False)

    # Dose correlation
    dose_df = section_dose_correlation(frozen, dev_mask, test_mask, oof_dev, p_test)
    dose_df.to_csv(V3_TAB / "v6_outcome_dose_correlation.csv", index=False)
    plot_dose_vs_risk(frozen, dev_mask, test_mask, oof_dev, p_test,
                      V3_FIG / "Figure_v3_13_PredictedRisk_vs_Dose.png")

    # Sensitivity summary
    summary = {
        "V6_features": V6_FEATURES,
        "performance_temporal_ROC_AUC": float(perf[
            (perf["Split"] == "Temporal_Test") & (perf["Metric"] == "ROC_AUC")]["Value"].iloc[0]),
        "performance_temporal_Brier": float(perf[
            (perf["Split"] == "Temporal_Test") & (perf["Metric"] == "Brier")]["Value"].iloc[0]),
        "calibration_intercept_temporal": float(intc),
        "calibration_slope_temporal": float(slp),
        "VIF_max": float(vif_df["VIF"].max()),
        "VIF_max_feature": str(vif_df.iloc[0]["Feature"]),
        "subgroup_dev_AUC_overall": float(sub_df[
            (sub_df["Split"] == "Dev_OOF") & (sub_df["Subgroup"] == "Overall")]["ROC_AUC_mean"].iloc[0]),
        "subgroup_temporal_AUC_overall": float(sub_df[
            (sub_df["Split"] == "Temporal") & (sub_df["Subgroup"] == "Overall")]["ROC_AUC_mean"].iloc[0]),
        "class_weight_delta_temporal_AUC": float(
            cw_df.iloc[1]["Temporal_AUC_mean"] - cw_df.iloc[0]["Temporal_AUC_mean"]),
        "multi_seed_mean_AUC": float(ms_df["ROC_AUC_mean"].mean()),
        "multi_seed_std_AUC": float(ms_df["ROC_AUC_mean"].std()),
        "loo_top_feature": str(loo_df.iloc[0]["Feature"]),
        "loo_top_delta": float(loo_df.iloc[0]["DeltaAUC"]),
        "bootstrap_top_feature_freq": {row["Feature"]: float(row["Frequency_k_over_n"])
                                        for _, row in boot_df.iterrows()},
    }
    (V3_TAB / "v6_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== V6 SUMMARY ===")
    for k, v in summary.items():
        if isinstance(v, (str, int, float, bool)):
            print(f"  {k}: {v}")
    print("\nDone.")


if __name__ == "__main__":
    main()
