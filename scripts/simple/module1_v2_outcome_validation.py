#!/usr/bin/env python
"""M1·v2 — outcome / treatment-effect validation on top of the LASSO-clean
augmented 10-feature model.

Four complementary outcome-side analyses, all using dev OOF + temporal
predictions from the M1·v2a (10) model:

  OV1. Finer risk stratification (3 → 4 → 5 tier monotonicity)
       Re-split predicted risk into quartiles and quintiles (cuts locked
       on dev OOF, applied to temporal), check observed NHRH rate monotone
       across tiers on both splits. Plot tier event rates with 95% CI.

  OV2. Decile observed-vs-predicted calibration
       Bin predicted probabilities into 10 deciles (dev OOF own bins;
       temporal uses dev OOF cut-points), compute mean predicted vs
       observed NHRH rate per bin with Wilson 95% CI; plot reliability
       diagram.

  OV3. Predicted-risk vs delivered RAI activity (treatment intensity audit)
       Scatter predicted NHRH probability vs delivered Dose (mCi) and
       Dose-per-gram (mCi/g). Annotate Spearman ρ and linear slope.
       This tests whether physicians titrated dose to clinical severity
       — the indirect confounding-by-indication footprint.

  OV4. Observed-NHRH curve along predicted risk percentile
       Sliding window (width = 20% of N) of observed NHRH rate as a
       function of predicted-risk percentile rank, on both dev OOF and
       temporal. Should be monotonically increasing if model is well-
       discriminating.

All outputs live in results/module1_v2_lasso_clean/. Forbidden unique-
patient count literal never appears (audit at runtime via str(890 - 1)).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PY_SEED = 2025
OOF_SEED = 13

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "module1_v2_lasso_clean"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

SOURCE_TABLE_DIR = ROOT / "results" / "module1_baseline_ml_benchmark" / "tables"
FROZEN_MATRIX_PATH = SOURCE_TABLE_DIR / "module1_frozen_feature_matrix.csv"

M1_V2A_FULL = [
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


def load_inputs() -> dict:
    frozen = pd.read_csv(FROZEN_MATRIX_PATH)
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


def make_l2() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(penalty="l2", solver="lbfgs", C=1.0,
                                  max_iter=5000, random_state=PY_SEED)),
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
# OV1: finer tier stratification (quartile + quintile)
# ---------------------------------------------------------------------------


def tier_table(prob: np.ndarray, y: np.ndarray, cuts: list[float], tier_names: list[str]) -> pd.DataFrame:
    rows = []
    last = -np.inf
    for i, c in enumerate(cuts + [np.inf]):
        mask = (prob >= last) & (prob < c) if i < len(cuts) else (prob >= last)
        n = int(mask.sum()); k = int(y[mask].sum())
        mean_p = float(prob[mask].mean()) if n > 0 else float("nan")
        rate = k / n if n > 0 else float("nan")
        lo, hi = wilson_ci(k, n)
        rows.append({"Tier": tier_names[i], "Cut_Upper": c if c != np.inf else None,
                     "N": n, "Events": k, "MeanPredictedRisk": mean_p,
                     "ObservedEventRate": rate, "CI_Low": lo, "CI_High": hi})
        last = c
    return pd.DataFrame(rows)


def finer_tiers(
    prob_dev: np.ndarray, y_dev: np.ndarray,
    prob_test: np.ndarray, y_test: np.ndarray,
) -> dict[int, dict[str, pd.DataFrame]]:
    """For each n_tiers in {3, 4, 5}, define cuts on dev OOF quantiles and
    apply to both splits; return dict tiers -> {dev, temporal} DataFrames.
    """
    out: dict[int, dict[str, pd.DataFrame]] = {}
    for n_tiers in (3, 4, 5):
        qs = np.linspace(0, 1, n_tiers + 1)[1:-1]
        cuts = [float(np.quantile(prob_dev, q)) for q in qs]
        if n_tiers == 3:
            names = ["Low (T1)", "Intermediate (T2)", "High (T3)"]
        elif n_tiers == 4:
            names = ["Q1", "Q2", "Q3", "Q4"]
        else:
            names = ["P1", "P2", "P3", "P4", "P5"]
        dev_df = tier_table(prob_dev, y_dev, cuts, names)
        tmp_df = tier_table(prob_test, y_test, cuts, names)
        out[n_tiers] = {"dev": dev_df, "temporal": tmp_df, "cuts": cuts}
    return out


def plot_finer_tiers(tiers_dict, out: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11, 11), sharey=True)
    for row, n_tiers in enumerate((3, 4, 5)):
        for col, split in enumerate(("dev", "temporal")):
            ax = axes[row, col]
            df = tiers_dict[n_tiers][split]
            x = np.arange(len(df))
            means = df["ObservedEventRate"].to_numpy()
            lo = (df["ObservedEventRate"] - df["CI_Low"]).to_numpy()
            hi = (df["CI_High"] - df["ObservedEventRate"]).to_numpy()
            colors = ["#456ea6" if i < len(df)/3 else ("#d28b18" if i < 2*len(df)/3 else "#a23b3b")
                      for i in range(len(df))]
            ax.bar(x, means, color=colors, edgecolor="black", linewidth=0.4,
                   yerr=[lo, hi], capsize=4, error_kw={"linewidth": 0.8, "ecolor": "#222"})
            for i, (n, k) in enumerate(zip(df["N"], df["Events"])):
                ax.text(i, means[i] + hi[i] + 0.03, f"N={int(n)}\nev={int(k)}",
                        ha="center", fontsize=7, color="#444")
            ax.set_xticks(x); ax.set_xticklabels(df["Tier"].tolist(), fontsize=8, rotation=20)
            ax.axhline(0.408 if split == "temporal" else 0.364,
                       color="#888", linestyle=":", linewidth=1,
                       label=f"{split} prevalence")
            ax.set_ylim(0, 0.85)
            split_lab = "Dev OOF (N=802)" if split == "dev" else "Temporal (N=201)"
            ax.set_title(f"{n_tiers}-tier {split_lab}")
            if col == 0:
                ax.set_ylabel(f"Observed NHRH rate (Wilson 95% CI)")
            ax.legend(loc="upper left", fontsize=7)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
    fig.suptitle("OV1. M1·v2a (10) finer stratification: 3 → 4 → 5 tiers", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# OV2: decile observed vs predicted calibration
# ---------------------------------------------------------------------------


def decile_calibration(prob: np.ndarray, y: np.ndarray, cuts: list[float] | None = None):
    if cuts is None:
        qs = np.linspace(0, 1, 11)[1:-1]
        cuts = [float(np.quantile(prob, q)) for q in qs]
    rows = []
    last = -np.inf
    for i, c in enumerate(cuts + [np.inf]):
        mask = (prob >= last) & (prob < c) if i < len(cuts) else (prob >= last)
        n = int(mask.sum()); k = int(y[mask].sum())
        mean_p = float(prob[mask].mean()) if n > 0 else float("nan")
        rate = k / n if n > 0 else float("nan")
        lo, hi = wilson_ci(k, n)
        rows.append({"Decile": i + 1, "N": n, "Events": k,
                     "MeanPredicted": mean_p, "ObservedRate": rate,
                     "CI_Low": lo, "CI_High": hi})
        last = c
    return pd.DataFrame(rows), cuts


def plot_decile_calibration(dev_df, tmp_df, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, df, split in zip(axes, (dev_df, tmp_df), ("Dev OOF (N=802)", "Temporal (N=201)")):
        ax.plot([0, 1], [0, 1], color="#888", linestyle="--", linewidth=1, label="Ideal")
        lo = (df["ObservedRate"] - df["CI_Low"]).to_numpy()
        hi = (df["CI_High"] - df["ObservedRate"]).to_numpy()
        ax.errorbar(df["MeanPredicted"], df["ObservedRate"],
                    yerr=[lo, hi], fmt="o-", color="#1d4e89", lw=2, capsize=4,
                    label="Observed (Wilson 95% CI)")
        for _, row in df.iterrows():
            ax.text(row["MeanPredicted"] + 0.01, row["ObservedRate"] - 0.02,
                    f"D{int(row['Decile'])} N={int(row['N'])}", fontsize=7, color="#444")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability (decile)")
        ax.set_ylabel("Observed NHRH rate")
        ax.set_title(f"OV2. Decile calibration — {split}")
        ax.legend(loc="upper left", fontsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# OV3: predicted risk vs delivered RAI activity (confounding-by-indication
# footprint)
# ---------------------------------------------------------------------------


def plot_dose_vs_risk(
    prob_dev: np.ndarray, frozen: pd.DataFrame, dev_mask: np.ndarray,
    prob_test: np.ndarray, test_mask: np.ndarray, out: Path,
) -> pd.DataFrame:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    rows_summary = []
    for col, (split, prob, mask, label) in enumerate([
        ("Dev OOF (N=802)", prob_dev, dev_mask, "dev"),
        ("Temporal (N=201)", prob_test, test_mask, "temporal"),
    ]):
        dose = frozen.loc[mask, "Dose"].to_numpy()
        dose_per_g = frozen.loc[mask, "IDPG_Dose_per_ThyroidW"].to_numpy()

        # Row 0: prob vs total dose (mCi)
        ax = axes[0, col]
        ax.scatter(prob, dose, alpha=0.35, s=18, color="#1d4e89", edgecolor="none")
        rho_d, p_d = stats.spearmanr(prob, dose)
        slope_d, intercept_d, r_d, _, _ = stats.linregress(prob, dose)
        xs = np.linspace(prob.min(), prob.max(), 50)
        ax.plot(xs, slope_d * xs + intercept_d, color="#a23b3b", lw=1.6,
                label=f"slope={slope_d:.2f}  ρ={rho_d:.3f} (p={p_d:.2g})")
        ax.set_xlabel("M1·v2a predicted NHRH probability")
        ax.set_ylabel("Delivered RAI activity (mCi)")
        ax.set_title(f"OV3a. Predicted risk vs delivered dose — {split}")
        ax.legend(loc="upper left", fontsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        # Row 1: prob vs dose per gram (mCi/g)
        ax = axes[1, col]
        ax.scatter(prob, dose_per_g, alpha=0.35, s=18, color="#1d4e89", edgecolor="none")
        rho_g, p_g = stats.spearmanr(prob, dose_per_g)
        slope_g, intercept_g, r_g, _, _ = stats.linregress(prob, dose_per_g)
        ax.plot(xs, slope_g * xs + intercept_g, color="#a23b3b", lw=1.6,
                label=f"slope={slope_g:.4f}  ρ={rho_g:.3f} (p={p_g:.2g})")
        ax.set_xlabel("M1·v2a predicted NHRH probability")
        ax.set_ylabel("Dose per gram thyroid (mCi/g)")
        ax.set_title(f"OV3b. Predicted risk vs dose-per-gram — {split}")
        ax.legend(loc="upper left", fontsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        rows_summary.append({
            "Split": split, "Variable": "Dose (mCi)",
            "Spearman_rho": float(rho_d), "Spearman_p": float(p_d),
            "Linear_slope": float(slope_d), "Linear_R": float(r_d),
        })
        rows_summary.append({
            "Split": split, "Variable": "Dose per gram (mCi/g)",
            "Spearman_rho": float(rho_g), "Spearman_p": float(p_g),
            "Linear_slope": float(slope_g), "Linear_R": float(r_g),
        })

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return pd.DataFrame(rows_summary)


# ---------------------------------------------------------------------------
# OV4: sliding-window observed NHRH along predicted-risk percentile
# ---------------------------------------------------------------------------


def sliding_window_curve(prob: np.ndarray, y: np.ndarray, win_frac: float = 0.20):
    order = np.argsort(prob)
    p_sorted = prob[order]
    y_sorted = y[order]
    n = len(p_sorted)
    w = max(20, int(n * win_frac))
    centers, rates, lo_s, hi_s = [], [], [], []
    for start in range(0, n - w + 1):
        end = start + w
        center = (start + end - 1) / 2 / (n - 1)  # percentile rank center
        k = int(y_sorted[start:end].sum())
        rate = k / w
        lo, hi = wilson_ci(k, w)
        centers.append(center)
        rates.append(rate)
        lo_s.append(lo)
        hi_s.append(hi)
    return (np.array(centers), np.array(rates),
            np.array(lo_s), np.array(hi_s))


def plot_sliding(prob_dev, y_dev, prob_test, y_test, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for prob, y, color, label in [
        (prob_dev, y_dev, "#1d4e89", f"Dev OOF (N={len(y_dev)})"),
        (prob_test, y_test, "#a23b3b", f"Temporal (N={len(y_test)})"),
    ]:
        c, r, lo, hi = sliding_window_curve(prob, y, win_frac=0.20)
        ax.fill_between(c, lo, hi, color=color, alpha=0.15)
        ax.plot(c, r, color=color, lw=2, label=label)
    ax.axhline(0.408, color="#888", linestyle=":", linewidth=1, label="Temporal prevalence 0.408")
    ax.axhline(0.364, color="#bbb", linestyle=":", linewidth=1, label="Dev prevalence 0.364")
    ax.set_xlabel("Predicted-risk percentile rank (sliding window width 20%)")
    ax.set_ylabel("Observed NHRH rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.85)
    ax.set_title("OV4. M1·v2a observed NHRH rate along predicted-risk percentile")
    ax.legend(loc="upper left", fontsize=8)
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
    y_dev = frozen.loc[dev_mask, "Y"].to_numpy()
    y_test = frozen.loc[test_mask, "Y"].to_numpy()

    X_dev = frozen.loc[dev_mask, M1_V2A_FULL].reset_index(drop=True)
    X_test = frozen.loc[test_mask, M1_V2A_FULL].reset_index(drop=True)

    print("Fitting M1·v2a (10) on dev with Platt …")
    oof_dev, final = fit_oof_platt(X_dev, y_dev)
    prob_test = final.predict_proba(X_test)[:, 1]

    # OV1
    print("OV1: finer risk stratification 3 / 4 / 5 tiers …")
    tiers = finer_tiers(oof_dev, y_dev, prob_test, y_test)
    rows = []
    for n_tiers, splits_d in tiers.items():
        for split, df in splits_d.items():
            if isinstance(df, list):
                continue
            df2 = df.copy()
            df2["N_Tiers"] = n_tiers
            df2["Split"] = split
            rows.append(df2)
    pd.concat(rows, ignore_index=True).to_csv(
        TABLE_DIR / "outcome_finer_tiers.csv", index=False
    )
    plot_finer_tiers(tiers, FIG_DIR / "Figure_18_Finer_Tiers.png")
    for n_tiers in (3, 4, 5):
        tmp = tiers[n_tiers]["temporal"]
        spread = tmp["ObservedEventRate"].max() - tmp["ObservedEventRate"].min()
        print(f"  {n_tiers}-tier temporal spread = {spread:.3f}  "
              f"(top tier {tmp.iloc[-1]['ObservedEventRate']:.3f} "
              f"vs bottom tier {tmp.iloc[0]['ObservedEventRate']:.3f})")

    # OV2
    print("OV2: decile observed-vs-predicted calibration …")
    dev_dec, cuts = decile_calibration(oof_dev, y_dev)
    tmp_dec, _ = decile_calibration(prob_test, y_test, cuts=cuts)
    dev_dec.to_csv(TABLE_DIR / "outcome_decile_calibration_dev.csv", index=False)
    tmp_dec.to_csv(TABLE_DIR / "outcome_decile_calibration_temporal.csv", index=False)
    plot_decile_calibration(dev_dec, tmp_dec, FIG_DIR / "Figure_19_Decile_Calibration.png")

    # OV3
    print("OV3: predicted risk vs delivered dose (treatment intensity audit) …")
    df_dose = plot_dose_vs_risk(
        oof_dev, frozen, dev_mask, prob_test, test_mask,
        FIG_DIR / "Figure_20_PredictedRisk_vs_Dose.png",
    )
    df_dose.to_csv(TABLE_DIR / "outcome_dose_correlation.csv", index=False)
    print("  Spearman ρ summary:")
    print(df_dose.to_string(index=False))

    # OV4
    print("OV4: sliding-window observed NHRH along predicted-risk percentile …")
    plot_sliding(oof_dev, y_dev, prob_test, y_test,
                 FIG_DIR / "Figure_21_Sliding_Observed_NHRH.png")

    summary = {
        "tier_spreads_temporal": {
            n: float(tiers[n]["temporal"]["ObservedEventRate"].max()
                     - tiers[n]["temporal"]["ObservedEventRate"].min())
            for n in (3, 4, 5)
        },
        "top_tier_temporal_rate_5tiers": float(tiers[5]["temporal"].iloc[-1]["ObservedEventRate"]),
        "bottom_tier_temporal_rate_5tiers": float(tiers[5]["temporal"].iloc[0]["ObservedEventRate"]),
        "spearman_predrisk_vs_dose_dev": float(
            df_dose[(df_dose["Split"].str.startswith("Dev")) & (df_dose["Variable"] == "Dose (mCi)")]["Spearman_rho"].iloc[0]
        ),
        "spearman_predrisk_vs_dose_per_gram_dev": float(
            df_dose[(df_dose["Split"].str.startswith("Dev")) & (df_dose["Variable"] == "Dose per gram (mCi/g)")]["Spearman_rho"].iloc[0]
        ),
        "spearman_predrisk_vs_dose_temporal": float(
            df_dose[(df_dose["Split"].str.startswith("Temporal")) & (df_dose["Variable"] == "Dose (mCi)")]["Spearman_rho"].iloc[0]
        ),
        "spearman_predrisk_vs_dose_per_gram_temporal": float(
            df_dose[(df_dose["Split"].str.startswith("Temporal")) & (df_dose["Variable"] == "Dose per gram (mCi/g)")]["Spearman_rho"].iloc[0]
        ),
    }
    (TABLE_DIR / "outcome_validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
