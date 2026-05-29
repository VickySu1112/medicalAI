#!/usr/bin/env python
"""M1 v6 — alternative 4-tier binning strategies to address temporal Q1/Q2
non-monotonicity.

Compares five strategies on the M1 v6 (6 features) predicted probability:

  S1 Sample-equal (current §6.5 baseline)
     cuts = dev OOF quantiles at 25 / 50 / 75 %; each dev tier has equal N
  S2 Events-equal
     cuts at points where cumulative dev events = 25 / 50 / 75 % of total
  S3 Cum-rate-equal
     cuts at points where cumulative dev observed event rate = 25 / 50 / 75 %
  S4 Isotonic-binned
     fit isotonic regression of dev observed rate vs predicted prob;
     re-quantile in the isotonic-mapped space, then back-map
  S5 Predicted-rate-bands
     fixed cuts on the predicted probability axis at 0.2 / 0.35 / 0.55

Outputs:
  tables/binning_alt_summary.csv  — per-strategy per-tier per-split (N,
    events, rate, Wilson CI)
  tables/binning_alt_monotone.csv — per-strategy monotonicity flags +
    pairwise Fisher exact p for adjacent tiers
  figures/Figure_v3_18_BinningAlternatives.png — paired panel

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

V6 = [
    "Sex", "ThyroidW", "TPOAb", "FT4_0M", "TSH_0M",
    "log1p_DiseaseDuration_Months_Aug",
]


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


def apply_cuts(prob, y, cuts, names):
    last = -np.inf
    out = []
    for i, c in enumerate(cuts + [np.inf]):
        m = (prob >= last) & (prob < c) if i < len(cuts) else (prob >= last)
        n = int(m.sum()); k = int(y[m].sum())
        rate = k / n if n > 0 else float("nan")
        lo, hi = wilson(k, n)
        out.append({"Tier": names[i], "N": n, "Events": k,
                    "ObservedRate": rate, "CI_Low": lo, "CI_High": hi})
        last = c
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Five strategies — each computes 3 cut points on dev OOF, applied to both
# splits.
# ---------------------------------------------------------------------------


def cuts_sample_equal(oof, y):
    """Current baseline: dev OOF prob quantiles 25/50/75."""
    return [float(np.quantile(oof, q)) for q in (0.25, 0.50, 0.75)]


def cuts_events_equal(oof, y):
    """Each tier has equal dev events. Sort by prob, walk cumulative events."""
    order = np.argsort(oof)
    p_sorted = oof[order]
    y_sorted = y[order]
    total_ev = y_sorted.sum()
    targets = [total_ev / 4, total_ev / 2, 3 * total_ev / 4]
    cuts = []
    cum = np.cumsum(y_sorted)
    for t in targets:
        idx = int(np.searchsorted(cum, t))
        idx = min(idx, len(p_sorted) - 1)
        cuts.append(float(p_sorted[idx]))
    return cuts


def cuts_cum_rate_equal(oof, y):
    """Cuts at points where cumulative observed rate = quartile of total
    rate? Closer to events-equal, but uses cumulative density of events
    rather than raw events count."""
    order = np.argsort(oof)
    p_sorted = oof[order]
    y_sorted = y[order]
    # Cumulative observed rate up to position i
    cum_events = np.cumsum(y_sorted)
    cum_n = np.arange(1, len(p_sorted) + 1)
    cum_rate = cum_events / cum_n
    # Target the 3 points that split cumulative rate range into 4 equal parts
    rate_min, rate_max = cum_rate.min(), cum_rate.max()
    targets = np.linspace(rate_min, rate_max, 5)[1:-1]
    cuts = []
    for t in targets:
        idx = int(np.searchsorted(cum_rate, t))
        idx = min(idx, len(p_sorted) - 1)
        cuts.append(float(p_sorted[idx]))
    return cuts


def cuts_isotonic_binned(oof, y):
    """Fit isotonic regression of dev observed rate vs predicted prob, then
    quantile in the isotonic-mapped space, then back-map to original prob.

    The mapping y_iso = iso(prob) is monotone non-decreasing by design;
    quantile of y_iso at 25/50/75 gives cut levels on the (recalibrated)
    risk scale; finding the prob value where iso(prob) crosses those
    levels yields cuts in the original prob space.
    """
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof, y)
    y_iso = iso.predict(oof)
    iso_q = [float(np.quantile(y_iso, q)) for q in (0.25, 0.50, 0.75)]
    # Find smallest prob where iso(prob) >= q
    cuts = []
    p_sorted_idx = np.argsort(oof)
    p_sorted = oof[p_sorted_idx]
    y_iso_sorted = y_iso[p_sorted_idx]
    for q in iso_q:
        idx = int(np.searchsorted(y_iso_sorted, q))
        idx = min(idx, len(p_sorted) - 1)
        cuts.append(float(p_sorted[idx]))
    return cuts


def cuts_pred_rate_bands(oof, y):
    """Fixed cuts on the predicted probability axis at clinically chosen
    levels 0.20 / 0.35 / 0.55 (low/mod-low/mod-high/high)."""
    return [0.20, 0.35, 0.55]


def cuts_bootstrap_stable_monotone(oof, y, n_boot=500, seed=PY_SEED):
    """Bootstrap the dev OOF predictions; for each bootstrap sample, find
    the cut triple (in the candidate prob percentile grid 10..90 step 5)
    that maximises the spread (Q4 rate − Q1 rate) on that bootstrap
    while keeping it monotone Q1<Q2<Q3<Q4 with no-tier-empty constraint.
    Take median of the chosen cuts.

    This is a "monotonicity-aware" cut search that should produce cuts
    where Q1/Q2 are well-separated by design.
    """
    rng = np.random.default_rng(seed)
    candidates = np.arange(10, 95, 5) / 100.0  # 10%, 15%, ..., 90%
    cut_history = []
    for b in range(n_boot):
        idx = rng.integers(0, len(oof), size=len(oof))
        p = oof[idx]; yb = y[idx]
        best_spread = -np.inf
        best_cuts = None
        # Search over triples of quantile positions (cheap because only 17 candidates)
        for i_lo in range(len(candidates) - 2):
            q_lo = float(np.quantile(p, candidates[i_lo]))
            for i_mid in range(i_lo + 1, len(candidates) - 1):
                q_mid = float(np.quantile(p, candidates[i_mid]))
                if q_mid <= q_lo:
                    continue
                for i_hi in range(i_mid + 1, len(candidates)):
                    q_hi = float(np.quantile(p, candidates[i_hi]))
                    if q_hi <= q_mid:
                        continue
                    cuts = [q_lo, q_mid, q_hi]
                    df = apply_cuts(p, yb, cuts, TIER_NAMES)
                    rates = df["ObservedRate"].tolist()
                    if any(np.isnan(r) for r in rates):
                        continue
                    # require minimum tier N
                    if df["N"].min() < 20:
                        continue
                    if is_monotone(rates):
                        spread = rates[-1] - rates[0]
                        if spread > best_spread:
                            best_spread = spread
                            best_cuts = cuts
        if best_cuts is not None:
            cut_history.append(best_cuts)
    if not cut_history:
        # fallback: use sample-equal
        return cuts_sample_equal(oof, y)
    arr = np.array(cut_history)
    return [float(np.median(arr[:, k])) for k in range(3)]


STRATEGIES = [
    ("S1_sample_equal", "S1 Sample-equal (current)", cuts_sample_equal),
    ("S2_events_equal", "S2 Events-equal", cuts_events_equal),
    ("S3_cum_rate_equal", "S3 Cumulative-rate-equal", cuts_cum_rate_equal),
    ("S4_isotonic_binned", "S4 Isotonic-binned", cuts_isotonic_binned),
    ("S5_pred_rate_bands", "S5 Predicted-rate-bands (0.20/0.35/0.55)", cuts_pred_rate_bands),
    ("S6_bootstrap_monotone", "S6 Bootstrap-monotone (500 boots, median cuts)", cuts_bootstrap_stable_monotone),
]

TIER_NAMES = ["Q1", "Q2", "Q3", "Q4"]


# ---------------------------------------------------------------------------
# Per-strategy run + monotonicity report
# ---------------------------------------------------------------------------


def is_monotone(rates):
    return all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))


def pairwise_fisher_p(df):
    """Fisher exact for each adjacent pair (Q1 vs Q2, Q2 vs Q3, Q3 vs Q4)."""
    ps = []
    for i in range(len(df) - 1):
        a = df.iloc[i]
        b = df.iloc[i + 1]
        contingency = [
            [int(a["Events"]), int(a["N"] - a["Events"])],
            [int(b["Events"]), int(b["N"] - b["Events"])],
        ]
        try:
            _, p = stats.fisher_exact(contingency)
        except Exception:
            p = float("nan")
        ps.append(p)
    return ps


def plot_alternatives(by_strategy, out: Path):
    n = len(STRATEGIES)
    fig, axes = plt.subplots(n, 2, figsize=(13, 2.5 * n + 1), sharey=True)
    for row, (tag, name, _) in enumerate(STRATEGIES):
        for col, split in enumerate(("Dev_OOF", "Temporal_Test")):
            ax = axes[row, col]
            df = by_strategy[tag][split]
            x = np.arange(len(df))
            rates = df["ObservedRate"].to_numpy()
            lo = (df["ObservedRate"] - df["CI_Low"]).to_numpy()
            hi = (df["CI_High"] - df["ObservedRate"]).to_numpy()
            mono = is_monotone(rates.tolist())
            colors = ["#a23b3b" if not mono else ("#456ea6" if i < 2 else "#d28b18" if i < 3 else "#a23b3b")
                      for i in range(len(df))]
            ax.bar(x, rates, color=colors, edgecolor="black", linewidth=0.4,
                   yerr=[lo, hi], capsize=4)
            for i, (n_, k_) in enumerate(zip(df["N"], df["Events"])):
                ax.text(i, rates[i] + hi[i] + 0.03,
                        f"N={int(n_)}\nev={int(k_)}",
                        ha="center", fontsize=7, color="#444")
            ax.set_xticks(x); ax.set_xticklabels(TIER_NAMES, fontsize=9)
            ax.set_ylim(0, 0.85)
            label = "Dev OOF (N=802)" if split == "Dev_OOF" else "Temporal (N=201)"
            ax.set_title(f"{name} — {label}  {'✅ monotone' if mono else '❌ Q1>Q2 reversal'}",
                         fontsize=9)
            if col == 0:
                ax.set_ylabel("Observed NHRH rate (Wilson 95% CI)")
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
    fig.suptitle("M1 v6 — 5 alternative 4-tier binning strategies", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    forbidden = str(890 - 1)
    if forbidden in Path(__file__).read_text():
        raise RuntimeError("Forbidden literal present.")

    V3_FIG.mkdir(parents=True, exist_ok=True)
    V3_TAB.mkdir(parents=True, exist_ok=True)

    frozen, dev_mask, test_mask = load()
    print("Fitting M1 v6 ...")
    oof_dev, p_tst, y_dev, y_tst = fit_v6(frozen, dev_mask, test_mask)

    all_rows = []
    monotone_rows = []
    by_strategy = {}

    for tag, name, cuts_fn in STRATEGIES:
        cuts = cuts_fn(oof_dev, y_dev)
        cuts_sorted = sorted(cuts)
        dev_df = apply_cuts(oof_dev, y_dev, cuts_sorted, TIER_NAMES)
        tmp_df = apply_cuts(p_tst, y_tst, cuts_sorted, TIER_NAMES)
        by_strategy[tag] = {"Dev_OOF": dev_df, "Temporal_Test": tmp_df, "cuts": cuts_sorted}

        for split, df in [("Dev_OOF", dev_df), ("Temporal_Test", tmp_df)]:
            for _, r in df.iterrows():
                all_rows.append({
                    "Strategy": tag, "Strategy_Label": name, "Split": split,
                    "Cut_Low": cuts_sorted[0], "Cut_Mid": cuts_sorted[1], "Cut_High": cuts_sorted[2],
                    **{k: r[k] for k in ["Tier", "N", "Events", "ObservedRate", "CI_Low", "CI_High"]},
                })

        dev_mono = is_monotone(dev_df["ObservedRate"].tolist())
        tmp_mono = is_monotone(tmp_df["ObservedRate"].tolist())
        dev_ps = pairwise_fisher_p(dev_df)
        tmp_ps = pairwise_fisher_p(tmp_df)

        monotone_rows.append({
            "Strategy": tag, "Strategy_Label": name,
            "Cuts": ";".join(f"{c:.4f}" for c in cuts_sorted),
            "Dev_Monotone_Q1Q2Q3Q4": dev_mono,
            "Temporal_Monotone_Q1Q2Q3Q4": tmp_mono,
            "Dev_Fisher_Q1vQ2_p": dev_ps[0],
            "Dev_Fisher_Q2vQ3_p": dev_ps[1],
            "Dev_Fisher_Q3vQ4_p": dev_ps[2],
            "Temporal_Fisher_Q1vQ2_p": tmp_ps[0],
            "Temporal_Fisher_Q2vQ3_p": tmp_ps[1],
            "Temporal_Fisher_Q3vQ4_p": tmp_ps[2],
            "Dev_Spread_Top_minus_Bottom": float(dev_df["ObservedRate"].iloc[-1] - dev_df["ObservedRate"].iloc[0]),
            "Temporal_Spread_Top_minus_Bottom": float(tmp_df["ObservedRate"].iloc[-1] - tmp_df["ObservedRate"].iloc[0]),
        })

        print(f"\n=== {name} ===")
        print(f"  cuts (dev OOF): {[round(c, 4) for c in cuts_sorted]}")
        print(f"  Dev   N:{list(dev_df['N'])}  ev:{list(dev_df['Events'])}  rate:{[round(r, 3) for r in dev_df['ObservedRate']]}  monotone={dev_mono}")
        print(f"  Tmp   N:{list(tmp_df['N'])}  ev:{list(tmp_df['Events'])}  rate:{[round(r, 3) for r in tmp_df['ObservedRate']]}  monotone={tmp_mono}")

    pd.DataFrame(all_rows).to_csv(V3_TAB / "binning_alt_summary.csv", index=False)
    pd.DataFrame(monotone_rows).to_csv(V3_TAB / "binning_alt_monotone.csv", index=False)
    plot_alternatives(by_strategy, V3_FIG / "Figure_v3_18_BinningAlternatives.png")
    print("\nDone. Outputs:")
    print(f"  {V3_TAB / 'binning_alt_summary.csv'}")
    print(f"  {V3_TAB / 'binning_alt_monotone.csv'}")
    print(f"  {V3_FIG / 'Figure_v3_18_BinningAlternatives.png'}")


if __name__ == "__main__":
    main()
