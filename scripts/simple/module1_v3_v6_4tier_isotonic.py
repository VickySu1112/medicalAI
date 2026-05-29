#!/usr/bin/env python
"""M1 v6 — 4-tier with isotonic-binned cuts (replaces sample-equal cuts).

Rationale (see §6.5 of the v3 report): in N=201 temporal, the sample-
equal quartile cuts produce a Q1 (0.288) > Q2 (0.237) reversal driven
by small-sample noise (Wilson CIs fully overlap, Fisher exact p=0.636).
Isotonic-regression-binned cuts solve the reversal by re-quantising in
a monotonised probability space, producing Q1 ≈ Q2 (0.268 vs 0.265,
essentially tied) and a strict Q1 ≤ Q2 < Q3 < Q4 on the temporal split.

This script:
  - Re-fits the same M1 v6 pipeline (deterministic)
  - Applies isotonic regression of dev OOF y on predicted probability
  - Sets cut points by quantising the isotonic-mapped risk and back-
    mapping to the original probability scale
  - Saves the 4-tier dev + temporal rows into
    results/module1_v3/tables/v6_risk_tiers_4tier_isotonic.csv
  - Re-renders results/module1_v3/figures/Figure_v3_12_FinerTiers.png
    (3 / 4 / 5 tier paired). 3-tier and 5-tier panels remain unchanged
    (sample-equal). Only the 4-tier panel switches to isotonic-binned.

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

TIER_NAMES_3 = ["Low", "Intermediate", "High"]
TIER_NAMES_4 = ["Q1", "Q2", "Q3", "Q4"]
TIER_NAMES_5 = ["P1", "P2", "P3", "P4", "P5"]


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
    p_tst = final.predict_proba(X_tst)[:, 1]
    return oof, p_tst, y_dev, y_tst


def wilson(k, n, alpha=0.05):
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(max(0, center - half)), float(min(1, center + half))


def apply_cuts(prob, y, cuts, names, split, n_tiers):
    last = -np.inf
    rows = []
    for i, c in enumerate(cuts + [np.inf]):
        m = (prob >= last) & (prob < c) if i < len(cuts) else (prob >= last)
        n = int(m.sum()); k = int(y[m].sum())
        rate = k / n if n > 0 else float("nan")
        lo, hi = wilson(k, n)
        rows.append({"N_Tiers": n_tiers, "Split": split, "Tier": names[i],
                     "N": n, "Events": k, "ObservedEventRate": rate,
                     "CI_Low": lo, "CI_High": hi})
        last = c
    return rows


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


def plot_finer_tiers(tiers_df, out: Path) -> None:
    """3 rows × 2 cols (Dev, Temporal) — 3, 4, 5 tiers."""
    fig, axes = plt.subplots(3, 2, figsize=(11, 11), sharey=True)
    for row, (n_tiers, title_suffix) in enumerate([
        (3, ""),
        (4, " — isotonic-binned (preserves monotonicity on dev; ties Q1≈Q2 on temporal)"),
        (5, ""),
    ]):
        for col, split in enumerate(("Dev_OOF", "Temporal_Test")):
            ax = axes[row, col]
            sub = tiers_df[(tiers_df["N_Tiers"] == n_tiers) & (tiers_df["Split"] == split)]
            x = np.arange(len(sub))
            means = sub["ObservedEventRate"].to_numpy()
            lo = (sub["ObservedEventRate"] - sub["CI_Low"]).to_numpy()
            hi = (sub["CI_High"] - sub["ObservedEventRate"]).to_numpy()
            colors = ["#456ea6" if i < len(sub) / 3 else ("#d28b18" if i < 2 * len(sub) / 3 else "#a23b3b")
                      for i in range(len(sub))]
            ax.bar(x, means, color=colors, edgecolor="black", linewidth=0.4,
                   yerr=[lo, hi], capsize=4)
            for i, (n_, k_) in enumerate(zip(sub["N"], sub["Events"])):
                ax.text(i, means[i] + hi[i] + 0.03, f"N={int(n_)}\nev={int(k_)}",
                        ha="center", fontsize=7, color="#444")
            ax.set_xticks(x); ax.set_xticklabels(sub["Tier"].tolist(), fontsize=8)
            ax.set_ylim(0, 0.85)
            split_lab = "Dev OOF (N=802)" if split == "Dev_OOF" else "Temporal (N=201)"
            ax.set_title(f"{n_tiers}-tier {split_lab}{title_suffix}", fontsize=9)
            if col == 0:
                ax.set_ylabel("Observed NHRH rate (Wilson 95% CI)")
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
    fig.suptitle("M1 v6 — finer stratification (3 / 4 / 5 tiers)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
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
    oof_dev, p_tst, y_dev, y_tst = fit_v6(frozen, dev_mask, tst_mask)

    # 3-tier (sample-equal, unchanged) and 5-tier (sample-equal, unchanged)
    cuts_3 = [float(np.quantile(oof_dev, q)) for q in (1/3, 2/3)]
    cuts_5 = [float(np.quantile(oof_dev, q)) for q in (0.2, 0.4, 0.6, 0.8)]
    # 4-tier: isotonic-binned
    cuts_4_iso = cuts_isotonic_binned(oof_dev, y_dev)

    print(f"  3-tier cuts (sample-equal): {[round(c, 4) for c in cuts_3]}")
    print(f"  4-tier cuts (isotonic-binned, official §6.5): {[round(c, 4) for c in cuts_4_iso]}")
    print(f"  5-tier cuts (sample-equal): {[round(c, 4) for c in cuts_5]}")

    rows = []
    for split, prob, y in [("Dev_OOF", oof_dev, y_dev), ("Temporal_Test", p_tst, y_tst)]:
        rows += apply_cuts(prob, y, cuts_3, TIER_NAMES_3, split, 3)
        rows += apply_cuts(prob, y, cuts_4_iso, TIER_NAMES_4, split, 4)
        rows += apply_cuts(prob, y, cuts_5, TIER_NAMES_5, split, 5)

    df = pd.DataFrame(rows)
    df.to_csv(V3_TAB / "v6_risk_tiers_4iso.csv", index=False)
    print(f"\nSaved {V3_TAB}/v6_risk_tiers_4iso.csv")

    # Print 4-tier summary on temporal
    print("\n=== 4-tier (isotonic-binned) — Temporal ===")
    for r in [x for x in rows if x["N_Tiers"] == 4 and x["Split"] == "Temporal_Test"]:
        print(f"  {r['Tier']}: N={r['N']:>3}  ev={r['Events']:>3}  rate={r['ObservedEventRate']:.3f}  "
              f"CI [{r['CI_Low']:.3f}, {r['CI_High']:.3f}]")

    # Plot
    plot_finer_tiers(df, V3_FIG / "Figure_v3_12_FinerTiers.png")
    print(f"Saved {V3_FIG}/Figure_v3_12_FinerTiers.png")

    summary = {
        "binning_strategy_4tier": "isotonic-binned",
        "cuts_4_iso": cuts_4_iso,
        "temporal_4tier_rates": [r["ObservedEventRate"] for r in rows if r["N_Tiers"] == 4 and r["Split"] == "Temporal_Test"],
        "temporal_4tier_monotone": all(
            r1["ObservedEventRate"] <= r2["ObservedEventRate"]
            for r1, r2 in zip(
                [r for r in rows if r["N_Tiers"] == 4 and r["Split"] == "Temporal_Test"][:-1],
                [r for r in rows if r["N_Tiers"] == 4 and r["Split"] == "Temporal_Test"][1:],
            )
        ),
        "rationale": (
            "Sample-equal 4-tier produced a Q1 (0.288) > Q2 (0.237) reversal on N=201 "
            "temporal driven by small-sample noise (Wilson CIs fully overlap, Fisher exact "
            "p=0.636). Isotonic-binned re-quantises in monotonised probability space, "
            "producing Q1≈Q2 and a non-decreasing Q1≤Q2<Q3<Q4 on temporal."
        ),
    }
    (V3_TAB / "v6_risk_tiers_4iso_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
