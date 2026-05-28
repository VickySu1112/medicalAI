#!/usr/bin/env python
"""M1·v2 — LASSO selection stability under resampling and subgrouping.

Three independent stability tests of the LASSO feature-selection step
(beyond the 5-fold in-CV stability already shown in §4b.4):

  ST1. Bootstrap-LASSO selection frequency
       500 episode-level bootstrap replicates of development(802);
       at each bootstrap, fit L1-logistic on the augmented feature pool
       (after dropping Dose family) at the same chosen C used in the
       main run (C=0.08 for augmented natural LASSO; C=0.15 for core).
       Record per-feature selection frequency (k / 500) for both pools.

  ST2. Meinshausen-Bühlmann subsample stability
       100 random 50%-subsample replicates of development;
       same per-fold L1-logistic; per-feature selection frequency
       reported. By M-B definition: "stable" features have selection
       probability > 0.6 across resamples.

  ST3. Subgroup re-selection
       Split development by three clinically relevant axes:
         (a) Sex (Male vs Female)
         (b) Age tertile (Young < T1 < Mid < T2 < Old)
         (c) TRAb tertile (Low < T1 < Mid < T2 < High)
       Re-fit L1-logistic on each subgroup separately at the same
       chosen C; report each feature's per-subgroup selection
       indicator. A feature consistently selected across all 8
       subgroups (2 Sex + 3 Age + 3 TRAb = 8 splits) is "subgroup-
       robust".

Combined visual: per-feature stability heatmap across the three tests.

All in dev-only; temporal test untouched in this script. Forbidden
unique-patient count literal never appears (audit at runtime via
str(890 - 1)).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PY_SEED = 2025
OOF_SEED = 13
BOOTSTRAP_REPS = 500
SUBSAMPLE_REPS = 100
SUBSAMPLE_FRAC = 0.5
STABLE_THRESHOLD = 0.6  # Meinshausen-Bühlmann
CORE_C = 0.15
AUG_C = 0.08

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "module1_v2_lasso_clean"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

SOURCE_TABLE_DIR = ROOT / "results" / "module1_baseline_ml_benchmark" / "tables"
FROZEN_MATRIX_PATH = SOURCE_TABLE_DIR / "module1_frozen_feature_matrix.csv"
CORE_FEATURE_PATH = SOURCE_TABLE_DIR / "core_feature_set.csv"
AUG_FEATURE_PATH = SOURCE_TABLE_DIR / "augmented_feature_set.csv"

DOSE_DROP = ["Dose", "IDPG_Dose_per_ThyroidW"]

# Pretty labels (for any feature that might appear in core or aug pool)
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
    "PreRAI_ATD_Use_Clean_Aug": "Pre-RAI ATD use",
    "PreRAI_ATD_Use_Missing_Aug": "Pre-RAI ATD use (missing)",
    "PreRAI_ATD_Stop_Days_Aug": "Pre-RAI ATD stop (days)",
    "PreRAI_ATD_Stop_Missing_Aug": "Pre-RAI ATD stop (missing)",
    "log1p_PreRAI_ATD_Stop_Days_Aug": "Log pre-RAI ATD stop (days)",
    "PreRAI_ATD_NotUsed_FromStop_Aug": "Pre-RAI ATD not used",
    "EyeSigns_Source_Positive_Aug": "Eye signs positive",
    "EyeSigns_Source_Missing_Aug": "Eye signs (missing)",
    "Comorbidity_TextPresent": "Comorbidity present",
    "DiseaseDuration_Months_Aug": "Disease duration (months)",
    "DiseaseDuration_Missing_Aug": "Disease duration (missing)",
    "Age": "Age",
    "Height": "Height",
    "Weight": "Weight",
    "BMI": "BMI",
    "Exophthalmos": "Exophthalmos",
    "RAI3d": "RAI 3-day",
    "MaxUptake": "Max uptake",
    "FT3_0M": "FT3 (baseline)",
    "logTSH_0M": "log TSH (baseline)",
    "FT3_0M_over_FT4_0M": "FT3/FT4 ratio",
    "FT4_0M_over_TSH_0M_plus1": "FT4/(TSH+1) ratio",
    "TSH_Recovered_0M": "TSH recovered at baseline",
    "Likely_Hyper_0M": "Likely hyper at baseline",
    "FT3_mean_0_0M": "FT3 mean window 0",
    "FT3_std_0_0M": "FT3 std window 0",
    "FT4_mean_0_0M": "FT4 mean window 0",
    "FT4_std_0_0M": "FT4 std window 0",
    "TSH_mean_0_0M": "TSH mean window 0",
    "TSH_std_0_0M": "TSH std window 0",
    "TreatCount": "Treatment count",
}


# ---------------------------------------------------------------------------
# Data + LASSO helpers
# ---------------------------------------------------------------------------


def load_inputs() -> dict:
    frozen = pd.read_csv(FROZEN_MATRIX_PATH)
    needed = {"Episode_Index", "Split", "OOF_Fold", "Y"}
    feat = [c for c in frozen.columns if c not in needed]
    for col in feat:
        frozen[col] = pd.to_numeric(frozen[col], errors="coerce")
    frozen[feat] = frozen[feat].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(frozen) != 1003:
        raise RuntimeError(f"Frozen matrix has {len(frozen)} rows; expected 1003")
    dev = frozen["Split"].eq("Development").to_numpy()
    core_full = pd.read_csv(CORE_FEATURE_PATH)["Feature"].astype(str).tolist()
    aug_full = pd.read_csv(AUG_FEATURE_PATH)["Feature"].astype(str).tolist()
    core_kept = [f for f in core_full if f not in DOSE_DROP]
    aug_kept = [f for f in aug_full if f not in DOSE_DROP]
    return {"frozen": frozen, "dev_mask": dev,
            "core_kept": core_kept, "aug_kept": aug_kept}


def make_l1(C: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    penalty="l1", solver="saga", C=C,
                    max_iter=10000, random_state=PY_SEED, tol=1e-4,
                ),
            ),
        ]
    )


def selected_features(X: pd.DataFrame, y: np.ndarray, features: list[str], C: float):
    """Fit L1 LASSO on X[features], return non-zero feature names."""
    pipe = make_l1(C)
    pipe.fit(X[features], y)
    coef = pipe.named_steps["lr"].coef_[0]
    return [f for f, c in zip(features, coef) if abs(c) > 1e-8]


# ---------------------------------------------------------------------------
# ST1 + ST2: resampling stability
# ---------------------------------------------------------------------------


def bootstrap_stability(
    X_dev: pd.DataFrame, y_dev: np.ndarray,
    features: list[str], C: float, reps: int, seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(y_dev)
    counts = {f: 0 for f in features}
    for r in range(reps):
        idx = rng.integers(0, n, size=n)
        # Require both classes present
        if len(np.unique(y_dev[idx])) < 2:
            continue
        sel = selected_features(X_dev.iloc[idx], y_dev[idx], features, C)
        for f in sel:
            counts[f] += 1
        if (r + 1) % 100 == 0:
            print(f"    bootstrap {r+1}/{reps} done")
    df = pd.DataFrame(
        [{"Feature": f, "PrettyLabel": PRETTY.get(f, f),
          "Frequency_k_over_n": counts[f] / reps, "Count": counts[f], "Total": reps}
         for f in features]
    ).sort_values("Frequency_k_over_n", ascending=False).reset_index(drop=True)
    return df


def subsample_stability(
    X_dev: pd.DataFrame, y_dev: np.ndarray,
    features: list[str], C: float, reps: int, frac: float, seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(y_dev)
    n_sub = int(n * frac)
    counts = {f: 0 for f in features}
    for r in range(reps):
        idx = rng.choice(n, size=n_sub, replace=False)
        if len(np.unique(y_dev[idx])) < 2:
            continue
        sel = selected_features(X_dev.iloc[idx], y_dev[idx], features, C)
        for f in sel:
            counts[f] += 1
        if (r + 1) % 25 == 0:
            print(f"    subsample {r+1}/{reps} done")
    df = pd.DataFrame(
        [{"Feature": f, "PrettyLabel": PRETTY.get(f, f),
          "Frequency_k_over_n": counts[f] / reps, "Count": counts[f], "Total": reps}
         for f in features]
    ).sort_values("Frequency_k_over_n", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# ST3: subgroup re-selection
# ---------------------------------------------------------------------------


def tertile_bins(values: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Return per-row tertile label (0/1/2) and the two cut-points."""
    cuts = [np.quantile(values, 1/3), np.quantile(values, 2/3)]
    labels = np.zeros_like(values, dtype=int)
    labels[(values >= cuts[0]) & (values < cuts[1])] = 1
    labels[values >= cuts[1]] = 2
    return labels, cuts


def subgroup_stability(
    frozen: pd.DataFrame, dev_mask: np.ndarray,
    features: list[str], C: float,
) -> pd.DataFrame:
    """Re-fit L1 on each subgroup split. Returns wide DataFrame: rows =
    features, columns = subgroup labels (1=selected, 0=not).
    """
    X_dev = frozen.loc[dev_mask, features].reset_index(drop=True)
    y_dev = frozen.loc[dev_mask, "Y"].to_numpy()

    cols: dict[str, list[int]] = {}

    # Sex: 0 male, 1 female
    sex_dev = frozen.loc[dev_mask, "Sex"].to_numpy()
    for sex_val, label in ((0.0, "Male"), (1.0, "Female")):
        sub = sex_val == sex_val  # always True placeholder
        mask = (sex_dev == sex_val)
        if mask.sum() < 20 or len(np.unique(y_dev[mask])) < 2:
            cols[f"Sex={label}"] = [np.nan] * len(features)
            continue
        sel = selected_features(X_dev[mask], y_dev[mask], features, C)
        cols[f"Sex={label} (N={int(mask.sum())})"] = [
            1 if f in sel else 0 for f in features
        ]

    # Age tertile
    age_dev = frozen.loc[dev_mask, "Age"].to_numpy()
    age_lab, age_cuts = tertile_bins(age_dev)
    for k, name in enumerate([
        f"Age<{age_cuts[0]:.0f}",
        f"{age_cuts[0]:.0f}≤Age<{age_cuts[1]:.0f}",
        f"Age≥{age_cuts[1]:.0f}",
    ]):
        mask = age_lab == k
        if mask.sum() < 20 or len(np.unique(y_dev[mask])) < 2:
            cols[name] = [np.nan] * len(features)
            continue
        sel = selected_features(X_dev[mask], y_dev[mask], features, C)
        cols[f"{name} (N={int(mask.sum())})"] = [
            1 if f in sel else 0 for f in features
        ]

    # TRAb tertile
    trab_dev = frozen.loc[dev_mask, "TRAb"].to_numpy()
    trab_lab, trab_cuts = tertile_bins(trab_dev)
    for k, name in enumerate([
        f"TRAb<{trab_cuts[0]:.1f}",
        f"{trab_cuts[0]:.1f}≤TRAb<{trab_cuts[1]:.1f}",
        f"TRAb≥{trab_cuts[1]:.1f}",
    ]):
        mask = trab_lab == k
        if mask.sum() < 20 or len(np.unique(y_dev[mask])) < 2:
            cols[name] = [np.nan] * len(features)
            continue
        sel = selected_features(X_dev[mask], y_dev[mask], features, C)
        cols[f"{name} (N={int(mask.sum())})"] = [
            1 if f in sel else 0 for f in features
        ]

    df = pd.DataFrame(cols, index=features)
    df["Subgroup_HitRate"] = df.sum(axis=1, skipna=True) / df.notna().sum(axis=1)
    df = df.assign(Feature=df.index, PrettyLabel=[PRETTY.get(f, f) for f in df.index])
    df = df.reset_index(drop=True).sort_values("Subgroup_HitRate", ascending=False)
    return df


# ---------------------------------------------------------------------------
# Combined plot
# ---------------------------------------------------------------------------


def plot_stability_summary(
    pool_label: str,
    bootstrap_df: pd.DataFrame, subsample_df: pd.DataFrame, subgroup_df: pd.DataFrame,
    out: Path,
) -> None:
    """3-panel: (a) Bootstrap freq bars (b) Subsample freq bars (c) Subgroup heatmap."""
    # Order features by bootstrap frequency, descending
    features_order = bootstrap_df["Feature"].tolist()
    boot_freq = bootstrap_df.set_index("Feature").loc[features_order, "Frequency_k_over_n"].to_numpy()
    sub_freq = subsample_df.set_index("Feature").loc[features_order, "Frequency_k_over_n"].to_numpy()
    pretty = [PRETTY.get(f, f) for f in features_order]

    fig = plt.figure(figsize=(15, 0.45 * len(features_order) + 4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.4], wspace=0.4)

    # (a) Bootstrap
    ax = fig.add_subplot(gs[0])
    y = np.arange(len(features_order))
    colors_b = ["#a23b3b" if v >= 0.9 else ("#d28b18" if v >= STABLE_THRESHOLD else "#456ea6")
                for v in boot_freq]
    ax.barh(y, boot_freq, color=colors_b, edgecolor="black", linewidth=0.4)
    ax.axvline(STABLE_THRESHOLD, color="#444", linestyle=":", linewidth=1, label=f"M-B threshold {STABLE_THRESHOLD}")
    ax.set_yticks(y); ax.set_yticklabels(pretty, fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0, 1.0)
    ax.set_xlabel(f"Bootstrap selection frequency\n({BOOTSTRAP_REPS} reps, n=802)")
    ax.set_title(f"(a) Bootstrap stability — {pool_label}")
    ax.legend(loc="lower right", fontsize=7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for i, v in enumerate(boot_freq):
        ax.text(min(v + 0.02, 0.98), i, f"{v:.2f}", fontsize=7, va="center", color="#333")

    # (b) Subsample
    ax = fig.add_subplot(gs[1])
    colors_s = ["#a23b3b" if v >= 0.9 else ("#d28b18" if v >= STABLE_THRESHOLD else "#456ea6")
                for v in sub_freq]
    ax.barh(y, sub_freq, color=colors_s, edgecolor="black", linewidth=0.4)
    ax.axvline(STABLE_THRESHOLD, color="#444", linestyle=":", linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels([])  # share with left
    ax.invert_yaxis(); ax.set_xlim(0, 1.0)
    ax.set_xlabel(f"Subsample selection frequency\n({SUBSAMPLE_REPS} × {int(SUBSAMPLE_FRAC*100)}%, M-B)")
    ax.set_title(f"(b) Meinshausen-Bühlmann — {pool_label}")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for i, v in enumerate(sub_freq):
        ax.text(min(v + 0.02, 0.98), i, f"{v:.2f}", fontsize=7, va="center", color="#333")

    # (c) Subgroup heatmap
    ax = fig.add_subplot(gs[2])
    subgroup_cols = [c for c in subgroup_df.columns if c not in ("Feature", "PrettyLabel", "Subgroup_HitRate")]
    sub_matrix = subgroup_df.set_index("Feature").loc[features_order, subgroup_cols].to_numpy(dtype=float)
    im = ax.imshow(sub_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(np.arange(len(features_order))); ax.set_yticklabels([])
    ax.set_xticks(np.arange(len(subgroup_cols)))
    ax.set_xticklabels(subgroup_cols, rotation=45, ha="right", fontsize=7)
    ax.set_title(f"(c) Subgroup re-selection — {pool_label}\n(green=selected, red=not, gray=insufficient)")
    # Annotate per-cell value
    for i in range(sub_matrix.shape[0]):
        for j in range(sub_matrix.shape[1]):
            v = sub_matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="#666")
            else:
                ax.text(j, i, "✓" if v == 1 else "·", ha="center", va="center",
                        fontsize=7, color="white" if v == 1 else "black")
    plt.colorbar(im, ax=ax, shrink=0.5, pad=0.02)

    fig.suptitle(f"M1·v2 LASSO stability triad — {pool_label}", fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
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
    core_feats = inputs["core_kept"]
    aug_feats = inputs["aug_kept"]
    y_dev = frozen.loc[dev_mask, "Y"].to_numpy()

    pools = [
        ("core", "M1·v2c (core natural-LASSO @ C=0.15)", core_feats, CORE_C),
        ("aug", "M1·v2a (augmented natural-LASSO @ C=0.08)", aug_feats, AUG_C),
    ]

    summary = {}
    for tag, label, feats, C in pools:
        print(f"\n=== Pool: {label}  (candidates n={len(feats)}, chosen C={C}) ===")
        X_dev = frozen.loc[dev_mask, feats].reset_index(drop=True)
        print(f"  ST1: bootstrap LASSO selection frequency ({BOOTSTRAP_REPS} reps)")
        boot = bootstrap_stability(X_dev, y_dev, feats, C, BOOTSTRAP_REPS, seed=PY_SEED)
        boot.to_csv(TABLE_DIR / f"stability_bootstrap_{tag}.csv", index=False)
        print("  Top 10 by bootstrap frequency:")
        print(boot.head(10).to_string(index=False))

        print(f"  ST2: subsample LASSO (Meinshausen-Bühlmann, {SUBSAMPLE_REPS} × {int(SUBSAMPLE_FRAC*100)}%)")
        sub = subsample_stability(X_dev, y_dev, feats, C, SUBSAMPLE_REPS, SUBSAMPLE_FRAC, seed=PY_SEED + 1)
        sub.to_csv(TABLE_DIR / f"stability_subsample_{tag}.csv", index=False)

        print("  ST3: subgroup re-selection (Sex × Age tertile × TRAb tertile)")
        subgroup = subgroup_stability(frozen, dev_mask, feats, C)
        subgroup.to_csv(TABLE_DIR / f"stability_subgroup_{tag}.csv", index=False)

        # Combined plot
        plot_stability_summary(label, boot, sub, subgroup,
                               FIG_DIR / f"Figure_17_LASSO_Stability_{tag}.png")

        # Stable features by threshold
        stable_boot = boot[boot["Frequency_k_over_n"] >= STABLE_THRESHOLD]["Feature"].tolist()
        stable_sub = sub[sub["Frequency_k_over_n"] >= STABLE_THRESHOLD]["Feature"].tolist()
        stable_subgroup = subgroup[subgroup["Subgroup_HitRate"] >= STABLE_THRESHOLD]["Feature"].tolist()
        triad_stable = sorted(set(stable_boot) & set(stable_sub) & set(stable_subgroup))

        summary[tag] = {
            "pool_size": len(feats),
            "chosen_C": C,
            "stable_threshold": STABLE_THRESHOLD,
            "bootstrap_stable_count": len(stable_boot),
            "subsample_stable_count": len(stable_sub),
            "subgroup_stable_count": len(stable_subgroup),
            "triad_stable_features": triad_stable,
            "triad_stable_count": len(triad_stable),
        }
        print(f"  → Triad-stable (≥{STABLE_THRESHOLD} on all 3 tests): {triad_stable}")

    (TABLE_DIR / "stability_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
