#!/usr/bin/env python
"""M1 v6 — final 4-tier risk stratification (isotonic-binned + macaron colors).

Produces the canonical 4-tier figure used as the main risk-stratification
display (replaces both the prior 3-tier §6 figure and the 3/4/5 comparison
panel in §6.5). Each tier gets a distinct macaron color, rising from cool
(low risk) to warm (high risk):

  Q1 — pistachio mint    #A8E6CF
  Q2 — butter cream      #FFEAA7
  Q3 — peach blush       #FAB1A0
  Q4 — rose strawberry   #E17055

Outputs:
  results/module1_v3/figures/Figure_v3_11_FourTier_Macaron.png
    Two-panel (Dev OOF / Temporal) bar plot, Wilson 95% CI, N/events
    annotations.
  results/module1_v3/tables/v6_4tier_isotonic_official.csv
    Dev + Temporal 4-tier rows (Tier, N, Events, ObservedEventRate,
    Wilson CI lo/hi).
  results/module1_v3/tables/v6_4tier_isotonic_summary.json
    Cuts, NPV per tier, monotone flag per split.

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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PY_SEED = 2025
OOF_SEED = 13

ROOT = Path(__file__).resolve().parents[2]
V3_DIR = ROOT / "results" / "module1_v3"
V3_FIG = V3_DIR / "figures"
V3_TAB = V3_DIR / "tables"
FROZEN = ROOT / "results" / "module1_baseline_ml_benchmark" / "tables" / "module1_frozen_feature_matrix.csv"

V6 = ["Sex", "ThyroidW", "TPOAb", "FT4_0M", "TSH_0M", "log1p_DiseaseDuration_Months_Aug"]

# Macaron palette — four distinct pastel colors, cool→warm by risk
MACARON = {
    "Q1": "#A8E6CF",  # pistachio mint
    "Q2": "#FFEAA7",  # butter cream
    "Q3": "#FAB1A0",  # peach blush
    "Q4": "#E17055",  # rose strawberry
}
EDGE_COLOR = "#444444"


def load():
    f = pd.read_csv(FROZEN)
    needed = {"Episode_Index", "Split", "OOF_Fold", "Y"}
    feat = [c for c in f.columns if c not in needed]
    for c in feat:
        f[c] = pd.to_numeric(f[c], errors="coerce")
    f[feat] = f[feat].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return f, f.Split.eq("Development").to_numpy(), f.Split.eq("Temporal").to_numpy()


def fit_v6(frozen, dev_mask, test_mask):
    y_dev = frozen.loc[dev_mask, "Y"].to_numpy()
    y_tst = frozen.loc[test_mask, "Y"].to_numpy()
    X_dev = frozen.loc[dev_mask, V6].reset_index(drop=True)
    X_tst = frozen.loc[test_mask, V6].reset_index(drop=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=OOF_SEED)
    oof = np.zeros(len(y_dev), dtype=float)
    for tr, va in skf.split(X_dev, y_dev):
        cal = CalibratedClassifierCV(
            base_estimator=Pipeline([
                ("scale", StandardScaler()),
                ("lr", LogisticRegression(penalty="l2", solver="lbfgs", C=1.0,
                                          max_iter=5000, random_state=PY_SEED)),
            ]),
            method="sigmoid", cv=3)
        cal.fit(X_dev.iloc[tr], y_dev[tr])
        oof[va] = cal.predict_proba(X_dev.iloc[va])[:, 1]
    final = CalibratedClassifierCV(
        base_estimator=Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(penalty="l2", solver="lbfgs", C=1.0,
                                      max_iter=5000, random_state=PY_SEED)),
        ]),
        method="sigmoid", cv=3)
    final.fit(X_dev, y_dev)
    return oof, final.predict_proba(X_tst)[:, 1], y_dev, y_tst


def wilson(k, n, alpha=0.05):
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(max(0, center - half)), float(min(1, center + half))


def cuts_isotonic_binned(oof, y):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof, y)
    y_iso = iso.predict(oof)
    iso_q = [float(np.quantile(y_iso, q)) for q in (0.25, 0.50, 0.75)]
    p_sorted_idx = np.argsort(oof)
    p_sorted = oof[p_sorted_idx]
    y_iso_sorted = y_iso[p_sorted_idx]
    cuts = []
    for q in iso_q:
        idx = int(np.searchsorted(y_iso_sorted, q))
        idx = min(idx, len(p_sorted) - 1)
        cuts.append(float(p_sorted[idx]))
    return sorted(cuts)


def apply_4tier_cuts(prob, y, cuts):
    names = ["Q1", "Q2", "Q3", "Q4"]
    rows = []
    last = -np.inf
    for i, c in enumerate(cuts + [np.inf]):
        m = (prob >= last) & (prob < c) if i < len(cuts) else (prob >= last)
        n = int(m.sum()); k = int(y[m].sum())
        rate = k / n if n > 0 else float("nan")
        lo, hi = wilson(k, n)
        rows.append({"Tier": names[i], "N": n, "Events": k,
                     "ObservedEventRate": rate, "NPV": 1 - rate,
                     "CI_Low": lo, "CI_High": hi})
        last = c
    return pd.DataFrame(rows)


def plot_4tier_macaron(dev_df, tmp_df, cuts, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    for ax, df, title in zip(
        axes,
        (dev_df, tmp_df),
        (f"Dev OOF (N=802) — cuts at {cuts[0]:.3f} / {cuts[1]:.3f} / {cuts[2]:.3f}",
         "Temporal (N=201)"),
    ):
        x = np.arange(len(df))
        means = df["ObservedEventRate"].to_numpy()
        lo = (df["ObservedEventRate"] - df["CI_Low"]).to_numpy()
        hi = (df["CI_High"] - df["ObservedEventRate"]).to_numpy()
        colors = [MACARON[t] for t in df["Tier"]]
        bars = ax.bar(x, means, color=colors, edgecolor=EDGE_COLOR,
                      linewidth=0.9, yerr=[lo, hi], capsize=6,
                      error_kw={"linewidth": 1.1, "ecolor": EDGE_COLOR})
        # Tier labels under bars
        ax.set_xticks(x); ax.set_xticklabels(df["Tier"].tolist(), fontsize=11, weight="bold")
        ax.set_ylim(0, 0.85)
        ax.set_title(title, fontsize=10)
        # N / events annotation
        for i, (n_, k_, m_) in enumerate(zip(df["N"], df["Events"], means)):
            ax.text(i, m_ + hi[i] + 0.03,
                    f"N = {int(n_)}\nev = {int(k_)}\nrate = {m_:.3f}",
                    ha="center", fontsize=8, color="#333")
        # Prevalence reference
        if "Temporal" in title:
            ax.axhline(0.408, color="#888", linestyle=":", linewidth=1.2,
                       label="Temporal prevalence 0.408")
            ax.legend(loc="upper left", fontsize=8)
        else:
            ax.axhline(0.364, color="#888", linestyle=":", linewidth=1.2,
                       label="Dev prevalence 0.364")
            ax.legend(loc="upper left", fontsize=8)
        ax.set_ylabel("Observed NHRH rate" if ax is axes[0] else "")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle(
        "M1 v6 — 4-tier risk stratification (isotonic-binned cuts, macaron palette)",
        fontsize=12, weight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    forbidden = str(890 - 1)
    if forbidden in Path(__file__).read_text():
        raise RuntimeError("Forbidden literal present.")

    V3_FIG.mkdir(parents=True, exist_ok=True)
    V3_TAB.mkdir(parents=True, exist_ok=True)

    frozen, dev_mask, tst_mask = load()
    print("Fitting M1 v6 ...")
    oof, p_tst, y_dev, y_tst = fit_v6(frozen, dev_mask, tst_mask)

    cuts = cuts_isotonic_binned(oof, y_dev)
    print(f"  Isotonic-binned 4-tier cuts: {[round(c, 4) for c in cuts]}")

    dev_df = apply_4tier_cuts(oof, y_dev, cuts)
    tmp_df = apply_4tier_cuts(p_tst, y_tst, cuts)

    # Save merged CSV with Split column
    merged = pd.concat([
        dev_df.assign(Split="Dev_OOF"),
        tmp_df.assign(Split="Temporal_Test"),
    ], ignore_index=True)
    merged.to_csv(V3_TAB / "v6_4tier_isotonic_official.csv", index=False)
    print(f"\nSaved {V3_TAB}/v6_4tier_isotonic_official.csv")

    # Plot
    plot_4tier_macaron(dev_df, tmp_df, cuts, V3_FIG / "Figure_v3_11_FourTier_Macaron.png")
    print(f"Saved {V3_FIG}/Figure_v3_11_FourTier_Macaron.png")

    # Summary
    rates_tmp = tmp_df["ObservedEventRate"].tolist()
    summary = {
        "cuts_isotonic_binned": cuts,
        "macaron_palette": MACARON,
        "dev_OOF_rates_Q1_Q2_Q3_Q4": dev_df["ObservedEventRate"].tolist(),
        "temporal_rates_Q1_Q2_Q3_Q4": rates_tmp,
        "temporal_NPV_Q1": float(tmp_df.iloc[0]["NPV"]),
        "temporal_PPV_Q4": float(tmp_df.iloc[-1]["ObservedEventRate"]),
        "temporal_spread_Q4_minus_Q1": float(rates_tmp[-1] - rates_tmp[0]),
        "dev_monotone": all(dev_df["ObservedEventRate"].iloc[i] <= dev_df["ObservedEventRate"].iloc[i + 1]
                            for i in range(3)),
        "temporal_monotone": all(rates_tmp[i] <= rates_tmp[i + 1] for i in range(3)),
    }
    (V3_TAB / "v6_4tier_isotonic_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n=== Summary ===")
    print(f"  Dev monotone: {summary['dev_monotone']}")
    print(f"  Temporal monotone: {summary['temporal_monotone']}")
    print(f"  Temporal Q1 NPV = {summary['temporal_NPV_Q1']:.3f}")
    print(f"  Temporal Q4 PPV = {summary['temporal_PPV_Q4']:.3f}")
    print(f"  Temporal spread (Q4 − Q1) = {summary['temporal_spread_Q4_minus_Q1']:.3f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
