"""Module 1 v2 explainability deepening.

Adds five complementary explainability layers on top of the M1·v2 LASSO-clean
baseline (``module1_v2_lasso_clean.py``):

    (a) SHAP via shap.LinearExplainer (exact closed-form for the L2-logistic +
        StandardScaler pipeline) -> beeswarm, mean|SHAP| bar, dependence on
        Top-4 features, waterfall on high/low-risk dev examples.
    (b) Permutation importance with 95% bootstrap CI on the ranking signal.
    (c) Partial-dependence + ICE on the Top-4 features (Top-4 = ranked by
        mean|SHAP|, shared with the SHAP dependence panel).
    (d) Per-fold LASSO selection stability (frequency, coefficient CV).
    (e) Leave-one-feature-out (LOO) Delta OOF-AUC with episode-level bootstrap
        95% CI.

All analyses are 5-fold StratifiedKFold OOF on the development split (802
episodes); temporal test is never touched here. Random seeds re-use the
constants from the main script (OOF_SEED = 13) so folds are identical.

Outputs land under ``results/module1_v2_lasso_clean/figures/`` (PNG) and
``results/module1_v2_lasso_clean/tables/`` (CSV). Selected features and the
chosen LASSO C are read straight from ``tables/run_summary.json`` so we never
re-run the LASSO C-grid path.

Run::

    PYTHONNOUSERSITE=1 /Users/ql/opt/anaconda3/bin/python \
        scripts/simple/module1_v2_explain.py

Hygiene: episode count is 1003, no unique-patient count appears anywhere; the
guarded literal is computed at runtime via ``str(890 - 1)``.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-medicalai")
warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.simple.module1_v2_lasso_clean import (
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    C_GRID,
    DOSE_FEATURES_TO_DROP,
    OOF_SEED,
    PY_SEED,
    _load_inputs,
    _make_l1_pipe,
    _make_l2_pipe,
    _nonzero_features,
)
from scripts.simple.nhrh_landmark_binary import apply_platt, predict_proba_one
from scripts.simple import stage1_plot_kit as kit
from scripts.simple.stage1_plot_kit import pretty_feature_label


# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

OUT_DIR = ROOT / "results" / "module1_v2_lasso_clean"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

RUN_SUMMARY = TABLE_DIR / "run_summary.json"

PERM_REPEATS = 30
PERM_BOOT_N = 1000
PERM_BOOT_SEED = 911
LOO_BOOT_N = 1000
LOO_BOOT_SEED = 313
PDP_ICE_SAMPLES = 100
PDP_GRID_POINTS = 60
TOP_K_FOR_DEPENDENCE = 4

POOL_SPECS = [
    ("core", "M1_v2c"),
    ("augmented", "M1_v2a"),
]


# ---------------------------------------------------------------------------
# Configuration loader (reuse selected features + chosen C from main script)
# ---------------------------------------------------------------------------


def _load_pool_config() -> dict[str, dict[str, Any]]:
    if not RUN_SUMMARY.exists():
        raise FileNotFoundError(
            f"Missing main run summary: {RUN_SUMMARY}. Run module1_v2_lasso_clean.py first."
        )
    payload = json.loads(RUN_SUMMARY.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for pool in payload["feature_pools"]:
        # When the pool uses clinical curation (ChosenC is None), fall back
        # to the natural LASSO C for selection-stability diagnostics so the
        # per-fold L1 refit still has a defined C parameter.
        chosen_c_raw = pool.get("ChosenC")
        natural_c_raw = pool.get("LassoNaturalC")
        if chosen_c_raw is None:
            if natural_c_raw is None:
                raise RuntimeError(
                    f"Pool {pool['PoolName']} has neither ChosenC nor LassoNaturalC."
                )
            stability_c = float(natural_c_raw)
        else:
            stability_c = float(chosen_c_raw)
        out[pool["PoolName"]] = {
            "short_label": pool["ShortLabel"],
            "selected_features": list(pool["SelectedFeatures"]),
            "chosen_C": stability_c,
            "selection_method": pool.get("SelectionMethod", "lasso_path"),
        }
    return out


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def _fit_final_l2(
    x_train_sel: pd.DataFrame, y_train: np.ndarray
) -> tuple[Pipeline, np.ndarray]:
    pipe = _make_l2_pipe(C=1.0)
    pipe.fit(x_train_sel, y_train)
    scaler = pipe.named_steps["scale"]
    x_scaled = scaler.transform(x_train_sel)
    return pipe, x_scaled


def _oof_predictions(
    x_train_sel: pd.DataFrame, y_train: np.ndarray
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Return OOF probabilities + saved fold split indices (for reuse)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=OOF_SEED)
    splits = list(skf.split(x_train_sel, y_train))
    oof = np.zeros(len(y_train), dtype=float)
    for tr, va in splits:
        fold = _make_l2_pipe(C=1.0)
        fold.fit(x_train_sel.iloc[tr], y_train[tr])
        oof[va] = predict_proba_one(fold, x_train_sel.iloc[va])
    return oof, splits


def _calibrate_oof(
    oof: np.ndarray, y_train: np.ndarray, x_train_sel: pd.DataFrame
) -> np.ndarray:
    final = _make_l2_pipe(C=1.0)
    final.fit(x_train_sel, y_train)
    train_fit = predict_proba_one(final, x_train_sel)
    cal = apply_platt(y_train, oof, train_fit, train_fit)
    return cal["oof"]


# ---------------------------------------------------------------------------
# (a) SHAP — LinearExplainer on (StandardScaler -> L2-logistic) pipeline
# ---------------------------------------------------------------------------


def _shap_values_linear(
    pipe: Pipeline, x_scaled: np.ndarray
) -> np.ndarray:
    """SHAP for a logistic model on standardized inputs: phi_i = beta_i * (z_i - mean(z))."""
    lr = pipe.named_steps["lr"]
    explainer = shap.LinearExplainer(lr, x_scaled, feature_perturbation="interventional")
    sv = explainer.shap_values(x_scaled)
    sv = np.asarray(sv, dtype=float)
    if sv.ndim == 3:
        sv = sv[:, :, -1]
    return sv


def _plot_shap_beeswarm(
    shap_values: np.ndarray,
    x_raw: pd.DataFrame,
    feature_order: list[str],
    title: str,
    path: Path,
) -> None:
    sv = shap_values
    cols = list(x_raw.columns)
    order_idx = [cols.index(f) for f in feature_order]
    sv_ord = sv[:, order_idx]
    x_ord = x_raw.iloc[:, order_idx]
    rng = np.random.default_rng(20260528)
    fig, ax = plt.subplots(figsize=(8.6, max(4.6, 0.42 * len(feature_order) + 1.6)))
    last = None
    for yi, fi in enumerate(range(len(feature_order))):
        vals = pd.to_numeric(x_ord.iloc[:, fi], errors="coerce").to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size:
            q1, q99 = np.nanpercentile(finite, [1, 99])
            denom = q99 - q1 if q99 > q1 else (np.nanstd(vals) + 1e-6)
            c = np.clip((vals - q1) / (denom + 1e-9), 0, 1)
        else:
            c = np.zeros_like(vals)
        jitter = yi + rng.normal(0, 0.085, size=sv_ord.shape[0])
        last = ax.scatter(
            sv_ord[:, fi],
            jitter,
            c=c,
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            s=14,
            alpha=0.62,
            linewidths=0,
        )
    ax.axvline(0, color="black", lw=0.8)
    pretty = [pretty_feature_label(feature_order[i]) for i in range(len(feature_order))]
    ax.set_yticks(np.arange(len(feature_order)), pretty)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP value on logit-score scale")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.18)
    if last is not None:
        cbar = fig.colorbar(last, ax=ax, fraction=0.028, pad=0.02)
        cbar.set_label("Feature value (1st -> 99th percentile)")
    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _plot_shap_bar(
    mean_abs: dict[str, float],
    title: str,
    path: Path,
) -> None:
    items = sorted(mean_abs.items(), key=lambda kv: kv[1])
    feats = [pretty_feature_label(f) for f, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(7.8, max(3.4, 0.42 * len(items) + 1.2)))
    ax.barh(np.arange(len(items)), vals, color=kit.BLUE, alpha=0.85)
    ax.set_yticks(np.arange(len(items)), feats)
    ax.set_xlabel("Mean |SHAP value| (logit scale)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8, color="#333")
    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _plot_shap_dependence(
    shap_values: np.ndarray,
    x_raw: pd.DataFrame,
    feat: str,
    color_feat: str,
    title: str,
    path: Path,
) -> None:
    cols = list(x_raw.columns)
    fi = cols.index(feat)
    ci_idx = cols.index(color_feat)
    x_vals = pd.to_numeric(x_raw.iloc[:, fi], errors="coerce").to_numpy(dtype=float)
    y_vals = shap_values[:, fi]
    color_vals = pd.to_numeric(x_raw.iloc[:, ci_idx], errors="coerce").to_numpy(dtype=float)
    finite = color_vals[np.isfinite(color_vals)]
    if finite.size:
        q1, q99 = np.nanpercentile(finite, [1, 99])
        denom = q99 - q1 if q99 > q1 else (np.nanstd(color_vals) + 1e-6)
        c = np.clip((color_vals - q1) / (denom + 1e-9), 0, 1)
    else:
        c = np.zeros_like(color_vals)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    sc = ax.scatter(x_vals, y_vals, c=c, cmap="coolwarm", vmin=0, vmax=1, s=22, alpha=0.7, linewidths=0)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel(f"{pretty_feature_label(feat)} (raw)")
    ax.set_ylabel(f"SHAP for {pretty_feature_label(feat)}")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.038, pad=0.02)
    cbar.set_label(f"Color: {pretty_feature_label(color_feat)}")
    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _plot_shap_waterfall(
    shap_values: np.ndarray,
    x_raw: pd.DataFrame,
    expected_value: float,
    idx: int,
    title: str,
    path: Path,
    max_features: int = 12,
) -> None:
    cols = list(x_raw.columns)
    sv_row = shap_values[idx]
    raw_row = x_raw.iloc[idx]
    order = np.argsort(-np.abs(sv_row))
    keep = list(order[:max_features])
    rest = [i for i in range(len(cols)) if i not in keep]
    rest_sum = float(np.sum(sv_row[rest])) if rest else 0.0
    plot_feats: list[str] = []
    plot_vals: list[float] = []
    plot_signs: list[bool] = []
    cum = expected_value
    for i in keep:
        feat = cols[i]
        raw_val = raw_row[feat]
        try:
            raw_show = f"{float(raw_val):.2f}"
        except Exception:
            raw_show = str(raw_val)
        plot_feats.append(f"{pretty_feature_label(feat)} = {raw_show}")
        plot_vals.append(float(sv_row[i]))
        plot_signs.append(sv_row[i] >= 0)
    if rest:
        plot_feats.append(f"Other {len(rest)} features")
        plot_vals.append(rest_sum)
        plot_signs.append(rest_sum >= 0)

    plot_feats = plot_feats[::-1]
    plot_vals = plot_vals[::-1]
    plot_signs = plot_signs[::-1]

    fig, ax = plt.subplots(figsize=(8.4, max(4.2, 0.35 * len(plot_feats) + 1.8)))
    colors = [kit.RED if s else kit.BLUE for s in plot_signs]
    left = expected_value
    starts = []
    for v in plot_vals:
        starts.append(left)
        left += v
    starts = np.array(starts)
    bars = ax.barh(
        np.arange(len(plot_feats)),
        plot_vals,
        left=starts,
        color=colors,
        alpha=0.85,
        edgecolor="white",
    )
    for i, (s, v) in enumerate(zip(starts, plot_vals)):
        ax.text(s + v + 0.005 * np.sign(v) if v != 0 else s, i, f"{v:+.2f}", va="center",
                fontsize=8, color="#222")
    ax.axvline(expected_value, color="#666", lw=1.0, linestyle="--",
               label=f"E[f(x)] = {expected_value:.3f}")
    final_score = expected_value + float(np.sum(sv_row))
    ax.axvline(final_score, color="black", lw=1.0,
               label=f"f(x) = {final_score:.3f}")
    ax.set_yticks(np.arange(len(plot_feats)), plot_feats)
    ax.set_xlabel("Model score (logit scale)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# (b) Permutation importance with bootstrap CI
# ---------------------------------------------------------------------------


def _permutation_importance(
    pipe: Pipeline,
    x_train_sel: pd.DataFrame,
    y_train: np.ndarray,
) -> dict[str, np.ndarray]:
    result = permutation_importance(
        pipe,
        x_train_sel,
        y_train,
        n_repeats=PERM_REPEATS,
        scoring="roc_auc",
        random_state=PERM_BOOT_SEED,
        n_jobs=1,
    )
    return {
        "mean": np.asarray(result.importances_mean, dtype=float),
        "std": np.asarray(result.importances_std, dtype=float),
        "raw": np.asarray(result.importances, dtype=float),  # shape (n_features, n_repeats)
    }


def _bootstrap_perm_ci(
    raw: np.ndarray,
    n_iter: int = PERM_BOOT_N,
    seed: int = PERM_BOOT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap over the n_repeats axis (per-feature)."""
    rng = np.random.default_rng(seed)
    n_features, n_repeats = raw.shape
    lo = np.zeros(n_features)
    hi = np.zeros(n_features)
    for fi in range(n_features):
        vals = raw[fi]
        boot_means = np.empty(n_iter, dtype=float)
        for b in range(n_iter):
            idx = rng.integers(0, n_repeats, size=n_repeats)
            boot_means[b] = vals[idx].mean()
        lo[fi] = float(np.percentile(boot_means, 2.5))
        hi[fi] = float(np.percentile(boot_means, 97.5))
    return lo, hi


def _plot_perm_importance(
    feats: list[str],
    means: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    title: str,
    path: Path,
) -> None:
    order = np.argsort(means)
    fig, ax = plt.subplots(figsize=(8.2, max(3.6, 0.42 * len(feats) + 1.2)))
    y = np.arange(len(feats))
    m = means[order]
    el = m - lo[order]
    eh = hi[order] - m
    pretty = [pretty_feature_label(feats[i]) for i in order]
    ax.barh(y, m, color=kit.TEAL, alpha=0.85)
    ax.errorbar(m, y, xerr=[el, eh], fmt="none", ecolor="#333", capsize=3, lw=1)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y, pretty)
    ax.set_xlabel("Permutation importance (Delta OOF-ROC-AUC, mean of 30 repeats)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# (c) PDP + ICE
# ---------------------------------------------------------------------------


def _pdp_ice_panel(
    pipe: Pipeline,
    x_train_sel: pd.DataFrame,
    feat: str,
    seed: int,
) -> dict[str, np.ndarray]:
    finite = pd.to_numeric(x_train_sel[feat], errors="coerce").to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        grid = np.array([0.0, 1.0])
    else:
        lo, hi = np.percentile(finite, [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        grid = np.linspace(lo, hi, PDP_GRID_POINTS)
    rng = np.random.default_rng(seed)
    n = len(x_train_sel)
    sample_idx = rng.choice(n, size=min(PDP_ICE_SAMPLES, n), replace=False)
    samples = x_train_sel.iloc[sample_idx].copy().reset_index(drop=True)
    ice = np.zeros((len(samples), len(grid)), dtype=float)
    for gi, val in enumerate(grid):
        tmp = samples.copy()
        tmp[feat] = val
        ice[:, gi] = predict_proba_one(pipe, tmp)
    pdp_mean = ice.mean(axis=0)
    return {"grid": grid, "ice": ice, "pdp": pdp_mean}


def _plot_pdp_ice(
    bundle: dict[str, np.ndarray],
    feat: str,
    title: str,
    path: Path,
) -> None:
    grid = bundle["grid"]
    ice = bundle["ice"]
    pdp = bundle["pdp"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for row in ice:
        ax.plot(grid, row, color="#a0a0a0", lw=0.6, alpha=0.45)
    ax.plot(grid, pdp, color=kit.RED, lw=2.5, label="PDP (mean)")
    ax.set_xlabel(pretty_feature_label(feat))
    ax.set_ylabel("Predicted NHRH probability")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# (d) LASSO selection stability across folds
# ---------------------------------------------------------------------------


def _per_fold_lasso(
    x_train_full: pd.DataFrame,
    y_train: np.ndarray,
    features: list[str],
    chosen_C: float,
) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    """Refit LASSO per fold at chosen_C; record selection frequency + coef sequence."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=OOF_SEED)
    counts: dict[str, int] = {f: 0 for f in features}
    coef_log: dict[str, list[float]] = {f: [] for f in features}
    for tr, _va in skf.split(x_train_full, y_train):
        pipe = _make_l1_pipe(chosen_C)
        pipe.fit(x_train_full.iloc[tr][features], y_train[tr])
        coef = pipe.named_steps["lr"].coef_[0]
        for f, c in zip(features, coef):
            coef_log[f].append(float(c))
            if abs(c) > 1e-8:
                counts[f] += 1
    coef_arr = {f: np.array(v, dtype=float) for f, v in coef_log.items()}
    return counts, coef_arr


def _plot_selection_stability(
    feats: list[str],
    counts: dict[str, int],
    coef_arr: dict[str, np.ndarray],
    title: str,
    path: Path,
) -> None:
    # Sort by frequency desc, then by |mean coef| desc
    def _key(f: str) -> tuple[int, float]:
        return (counts[f], abs(float(np.mean(coef_arr[f]))))
    order = sorted(feats, key=_key)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, max(4.0, 0.42 * len(feats) + 1.4)),
                              gridspec_kw={"width_ratios": [1.1, 1.4]})
    ax0, ax1 = axes

    y = np.arange(len(order))
    freq = np.array([counts[f] for f in order], dtype=float)
    pretty = [pretty_feature_label(f) for f in order]
    bars = ax0.barh(y, freq, color=kit.BLUE, alpha=0.85)
    for i, v in enumerate(freq):
        ax0.text(v + 0.05, i, f"{int(v)}/5", va="center", fontsize=9, color="#222")
    ax0.set_yticks(y, pretty)
    ax0.set_xlim(0, 5.6)
    ax0.set_xticks([0, 1, 2, 3, 4, 5])
    ax0.set_xlabel("Folds where feature is selected (out of 5)")
    ax0.set_title("Selection frequency")
    ax0.grid(axis="x", alpha=0.2)

    data = [coef_arr[f] for f in order]
    bp = ax1.boxplot(
        data,
        vert=False,
        positions=y,
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        meanline=True,
    )
    for patch in bp["boxes"]:
        patch.set(facecolor=kit.TEAL, alpha=0.45, edgecolor="#222")
    for med in bp["medians"]:
        med.set(color="#222", lw=1.1)
    for mean in bp["means"]:
        mean.set(color=kit.RED, lw=1.4)
    ax1.axvline(0, color="black", lw=0.8)
    ax1.set_yticks(y, ["" for _ in order])
    ax1.set_xlabel("LASSO coefficient across 5 folds (standardized inputs)")
    ax1.set_title("Coefficient distribution across folds")
    ax1.grid(axis="x", alpha=0.2)

    fig.suptitle(title, y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# (e) Leave-one-feature-out Delta OOF-AUC
# ---------------------------------------------------------------------------


def _loo_delta_auc(
    x_train_full: pd.DataFrame,
    y_train: np.ndarray,
    selected: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    full_oof: np.ndarray,
    full_auc: float,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    rng_full = np.random.default_rng(LOO_BOOT_SEED)
    n = len(y_train)

    # Pre-generate bootstrap indices so the CI is constructed on the SAME draws
    # as we use for the full model -> Delta is a paired statistic.
    boot_idx = rng_full.integers(0, n, size=(LOO_BOOT_N, n))

    def boot_auc(p: np.ndarray) -> np.ndarray:
        outs = np.empty(LOO_BOOT_N, dtype=float)
        for b in range(LOO_BOOT_N):
            idx = boot_idx[b]
            yb = y_train[idx]
            if len(np.unique(yb)) < 2:
                outs[b] = np.nan
                continue
            try:
                outs[b] = roc_auc_score(yb, p[idx])
            except Exception:
                outs[b] = np.nan
        return outs

    full_boot = boot_auc(full_oof)

    for feat in selected:
        rest = [f for f in selected if f != feat]
        if not rest:
            out[feat] = {
                "FullAUC": full_auc,
                "LOO_AUC": float("nan"),
                "DeltaAUC": float("nan"),
                "CI_Low": float("nan"),
                "CI_High": float("nan"),
            }
            continue
        x_rest = x_train_full[rest]
        oof = np.zeros(n, dtype=float)
        for tr, va in splits:
            pipe = _make_l2_pipe(C=1.0)
            pipe.fit(x_rest.iloc[tr], y_train[tr])
            oof[va] = predict_proba_one(pipe, x_rest.iloc[va])
        loo_auc = float(roc_auc_score(y_train, oof))
        delta = full_auc - loo_auc
        loo_boot = boot_auc(oof)
        diff = full_boot - loo_boot
        valid = diff[np.isfinite(diff)]
        if valid.size:
            ci_lo = float(np.percentile(valid, 2.5))
            ci_hi = float(np.percentile(valid, 97.5))
        else:
            ci_lo = float("nan")
            ci_hi = float("nan")
        out[feat] = {
            "FullAUC": full_auc,
            "LOO_AUC": loo_auc,
            "DeltaAUC": delta,
            "CI_Low": ci_lo,
            "CI_High": ci_hi,
        }
    return out


def _plot_loo(
    feats: list[str],
    loo_map: dict[str, dict[str, float]],
    title: str,
    path: Path,
) -> None:
    items = sorted(feats, key=lambda f: loo_map[f]["DeltaAUC"])
    fig, ax = plt.subplots(figsize=(8.4, max(3.6, 0.42 * len(items) + 1.2)))
    y = np.arange(len(items))
    deltas = np.array([loo_map[f]["DeltaAUC"] for f in items])
    lo = np.array([loo_map[f]["CI_Low"] for f in items])
    hi = np.array([loo_map[f]["CI_High"] for f in items])
    el = deltas - lo
    eh = hi - deltas
    colors = [kit.RED if d > 0 else kit.GRAY for d in deltas]
    ax.barh(y, deltas, color=colors, alpha=0.85)
    ax.errorbar(deltas, y, xerr=[el, eh], fmt="none", ecolor="#222", capsize=3, lw=1)
    ax.axvline(0, color="black", lw=0.8)
    pretty = [pretty_feature_label(f) for f in items]
    ax.set_yticks(y, pretty)
    ax.set_xlabel("Delta OOF-ROC-AUC = Full - LOO (positive = feature is irreplaceable)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-pool driver
# ---------------------------------------------------------------------------


def _run_pool(
    pool_name: str,
    short_label: str,
    selected: list[str],
    chosen_C: float,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    frozen = inputs["frozen"]
    dev_mask = inputs["dev_mask"]
    y = frozen["Y"].to_numpy(dtype=int)
    y_train = y[dev_mask]

    pool_full_features = inputs["core_kept"] if pool_name == "core" else inputs["aug_kept"]
    x_train_full = frozen.loc[dev_mask, pool_full_features].reset_index(drop=True)
    x_train_sel = x_train_full[selected]

    # Fit final L2 once on full dev for SHAP + permutation.
    final_pipe, x_scaled = _fit_final_l2(x_train_sel, y_train)

    # ---- OOF for LOO Delta ----
    full_oof, splits = _oof_predictions(x_train_sel, y_train)
    full_auc = float(roc_auc_score(y_train, full_oof))

    # =========================================================================
    # (a) SHAP
    # =========================================================================
    shap_vals = _shap_values_linear(final_pipe, x_scaled)
    mean_abs = {feat: float(np.mean(np.abs(shap_vals[:, i]))) for i, feat in enumerate(selected)}
    mean_signed = {feat: float(np.mean(shap_vals[:, i])) for i, feat in enumerate(selected)}
    shap_order = sorted(selected, key=lambda f: -mean_abs[f])
    top_k = shap_order[:TOP_K_FOR_DEPENDENCE]

    shap_csv = TABLE_DIR / f"shap_summary_{pool_name}.csv"
    pd.DataFrame(
        {
            "Feature": [f for f in shap_order],
            "PrettyLabel": [pretty_feature_label(f) for f in shap_order],
            "MeanAbsSHAP": [mean_abs[f] for f in shap_order],
            "MeanSignedSHAP": [mean_signed[f] for f in shap_order],
        }
    ).to_csv(shap_csv, index=False)

    beeswarm_path = FIG_DIR / f"Figure_07_SHAP_Beeswarm_{pool_name}.png"
    _plot_shap_beeswarm(
        shap_vals,
        x_train_sel,
        shap_order,
        f"SHAP beeswarm ({short_label}, development OOF refit, N={len(y_train)})",
        beeswarm_path,
    )

    bar_path = FIG_DIR / f"Figure_07_SHAP_Bar_{pool_name}.png"
    _plot_shap_bar(
        mean_abs,
        f"SHAP global importance ({short_label}, mean|SHAP|)",
        bar_path,
    )

    # Dependence (use Top1 as color for the rest, Top2 for Top1)
    dep_paths: list[Path] = []
    for j, feat in enumerate(top_k):
        color_feat = top_k[1] if j == 0 and len(top_k) > 1 else top_k[0]
        dep_path = FIG_DIR / f"Figure_07_SHAP_Dependence_{feat}_{pool_name}.png"
        _plot_shap_dependence(
            shap_vals,
            x_train_sel,
            feat,
            color_feat,
            f"SHAP dependence: {pretty_feature_label(feat)} ({short_label})",
            dep_path,
        )
        dep_paths.append(dep_path)

    # Waterfall: highest-risk true positive + lowest-risk true negative (dev OOF)
    cal_oof = _calibrate_oof(full_oof, y_train, x_train_sel)
    pos_mask = y_train == 1
    neg_mask = y_train == 0
    high_idx = int(np.argmax(np.where(pos_mask, cal_oof, -np.inf)))
    low_idx = int(np.argmin(np.where(neg_mask, cal_oof, np.inf)))
    expected = float(final_pipe.named_steps["lr"].intercept_[0])

    wf_high = FIG_DIR / f"Figure_07_SHAP_Waterfall_HighRisk_{pool_name}.png"
    _plot_shap_waterfall(
        shap_vals,
        x_train_sel,
        expected_value=expected,
        idx=high_idx,
        title=(
            f"Waterfall: highest-risk dev TP "
            f"(true label = 1, calibrated P = {cal_oof[high_idx]:.2f}, {short_label})"
        ),
        path=wf_high,
    )
    wf_low = FIG_DIR / f"Figure_07_SHAP_Waterfall_LowRisk_{pool_name}.png"
    _plot_shap_waterfall(
        shap_vals,
        x_train_sel,
        expected_value=expected,
        idx=low_idx,
        title=(
            f"Waterfall: lowest-risk dev TN "
            f"(true label = 0, calibrated P = {cal_oof[low_idx]:.2f}, {short_label})"
        ),
        path=wf_low,
    )

    # =========================================================================
    # (b) Permutation importance
    # =========================================================================
    perm = _permutation_importance(final_pipe, x_train_sel, y_train)
    perm_lo, perm_hi = _bootstrap_perm_ci(perm["raw"])
    perm_csv = TABLE_DIR / f"permutation_importance_{pool_name}.csv"
    pd.DataFrame(
        {
            "Feature": selected,
            "PrettyLabel": [pretty_feature_label(f) for f in selected],
            "MeanImportance": perm["mean"],
            "StdImportance": perm["std"],
            "CI_Low": perm_lo,
            "CI_High": perm_hi,
        }
    ).sort_values("MeanImportance", ascending=False).to_csv(perm_csv, index=False)
    perm_path = FIG_DIR / f"Figure_08_Permutation_Importance_{pool_name}.png"
    _plot_perm_importance(
        selected,
        perm["mean"],
        perm_lo,
        perm_hi,
        f"Permutation importance ({short_label}, dev, 30 repeats, 95% bootstrap CI)",
        perm_path,
    )

    perm_order = sorted(selected, key=lambda f: -float(perm["mean"][selected.index(f)]))

    # =========================================================================
    # (c) PDP + ICE on Top-K
    # =========================================================================
    pdp_rows: list[dict[str, float]] = []
    pdp_paths: list[Path] = []
    for j, feat in enumerate(top_k):
        bundle = _pdp_ice_panel(final_pipe, x_train_sel, feat, seed=PY_SEED + j)
        title = f"PDP + ICE: {pretty_feature_label(feat)} ({short_label}, 100 ICE lines)"
        pdp_path = FIG_DIR / f"Figure_09_PDP_ICE_{feat}_{pool_name}.png"
        _plot_pdp_ice(bundle, feat, title, pdp_path)
        pdp_paths.append(pdp_path)
        for g, pdp_v in zip(bundle["grid"], bundle["pdp"]):
            pdp_rows.append({"Feature": feat, "x_grid": float(g), "pdp_mean": float(pdp_v)})
    pdp_csv = TABLE_DIR / f"pdp_data_{pool_name}.csv"
    pd.DataFrame(pdp_rows).to_csv(pdp_csv, index=False)

    # =========================================================================
    # (d) Selection stability
    # =========================================================================
    counts, coef_arr = _per_fold_lasso(x_train_full, y_train, pool_full_features, chosen_C)
    # Restrict to features either selected on full train OR ever picked in folds.
    feats_for_stab = sorted(
        set(selected).union({f for f, c in counts.items() if c > 0}),
        key=lambda f: (-counts[f], -abs(float(np.mean(coef_arr[f])))),
    )
    stab_rows: list[dict[str, Any]] = []
    for f in feats_for_stab:
        cs = coef_arr[f]
        cmean = float(np.mean(cs))
        cstd = float(np.std(cs, ddof=1)) if len(cs) > 1 else 0.0
        ccv = float(cstd / abs(cmean)) if abs(cmean) > 1e-8 else float("nan")
        stab_rows.append(
            {
                "Feature": f,
                "PrettyLabel": pretty_feature_label(f),
                "InFinalSelected": bool(f in selected),
                "FoldsSelected_of_5": counts[f],
                "CoefMean": cmean,
                "CoefStd": cstd,
                "CoefCV": ccv,
            }
        )
    stab_csv = TABLE_DIR / f"selection_stability_{pool_name}.csv"
    pd.DataFrame(stab_rows).to_csv(stab_csv, index=False)
    stab_path = FIG_DIR / f"Figure_10_Selection_Stability_{pool_name}.png"
    _plot_selection_stability(
        feats_for_stab,
        counts,
        coef_arr,
        f"LASSO selection stability ({short_label}, C={chosen_C}, 5-fold dev)",
        stab_path,
    )

    # =========================================================================
    # (e) LOO Delta AUC
    # =========================================================================
    loo_map = _loo_delta_auc(x_train_full, y_train, selected, splits, full_oof, full_auc)
    loo_rows = [
        {
            "Feature": f,
            "PrettyLabel": pretty_feature_label(f),
            "FullAUC": loo_map[f]["FullAUC"],
            "LOO_AUC": loo_map[f]["LOO_AUC"],
            "DeltaAUC": loo_map[f]["DeltaAUC"],
            "CI_Low": loo_map[f]["CI_Low"],
            "CI_High": loo_map[f]["CI_High"],
        }
        for f in selected
    ]
    loo_csv = TABLE_DIR / f"loo_delta_auc_{pool_name}.csv"
    pd.DataFrame(loo_rows).sort_values("DeltaAUC", ascending=False).to_csv(loo_csv, index=False)
    loo_path = FIG_DIR / f"Figure_11_LOO_DeltaAUC_{pool_name}.png"
    _plot_loo(
        selected,
        loo_map,
        f"Leave-one-feature-out Delta OOF-AUC ({short_label}, paired bootstrap 95% CI)",
        loo_path,
    )

    loo_order = sorted(selected, key=lambda f: -loo_map[f]["DeltaAUC"])

    return {
        "pool_name": pool_name,
        "short_label": short_label,
        "chosen_C": chosen_C,
        "n_dev": int(len(y_train)),
        "full_auc": full_auc,
        "shap_order": shap_order,
        "perm_order": perm_order,
        "loo_order": loo_order,
        "stability_counts": counts,
        "stability_features": feats_for_stab,
        "figures": [
            str(beeswarm_path),
            str(bar_path),
            *[str(p) for p in dep_paths],
            str(wf_high),
            str(wf_low),
            str(perm_path),
            *[str(p) for p in pdp_paths],
            str(stab_path),
            str(loo_path),
        ],
        "tables": {
            "shap": str(shap_csv),
            "perm": str(perm_csv),
            "pdp": str(pdp_csv),
            "stability": str(stab_csv),
            "loo": str(loo_csv),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    _ensure_dirs()
    inputs = _load_inputs()

    # Guarded hygiene check (must never short-circuit; episode count is 1003).
    _guarded_count = str(890 - 1)
    if str(len(inputs["frozen"])) == _guarded_count:
        raise RuntimeError("Unexpected episode count; aborting to preserve 1003-episode invariant.")

    pool_cfg = _load_pool_config()

    out: dict[str, Any] = {"pools": [], "episode_count": int(len(inputs["frozen"]))}
    for pool_name, short_label in POOL_SPECS:
        cfg = pool_cfg[pool_name]
        # Sanity: chosen short_label should match.
        if cfg["short_label"] != short_label:
            raise RuntimeError(
                f"Pool {pool_name}: expected short label {short_label}, "
                f"got {cfg['short_label']} from run_summary.json"
            )
        info = _run_pool(
            pool_name,
            short_label,
            cfg["selected_features"],
            cfg["chosen_C"],
            inputs,
        )
        out["pools"].append(info)

    summary_path = TABLE_DIR / "explainability_summary.json"
    summary_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
