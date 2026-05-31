#!/usr/bin/env python
"""M2 · v2 EBM — patient-level prediction decomposition + counterfactual.

Glass-box's killer use: take 3 real episodes (low/mid/high risk at 6M), show
how the predicted log-odds decomposes into per-feature contributions, then run
counterfactual perturbations on the dominant features and re-predict.

Uses 6M EBM (highest AUC time point) on the aggregated-axis feature set.

Output: results/module2_v2_vertical/m2v2_ebm_patient_decomp/
  - figures/F_decomp_{Low,Mid,High}.png       # waterfall per patient
  - figures/F_counterfactual_{Low,Mid,High}.png  # sensitivity to lead feature
  - tables/decomp_table.csv                    # numeric breakdown
  - tables/counterfactual_table.csv            # CF results
  - tables/patient_profiles.csv                # 3 picked patients' raw values
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier

from scripts.simple.module2_v2_shared import load_stacked, PY_SEED
from scripts.simple.module2_v2_b4_trab_patch import add_current_velocity
from scripts.simple.module2_v2_b4_ebm_axes import FEATS, build_feats_at_L

OUT = ROOT / "results" / "module2_v2_vertical" / "m2v2_ebm_patient_decomp"
LANDMARK = 6  # use 6M EBM (best AUC)
DISP = {"ThyroidW": "Thyroid weight (g)", "TRAb": "TRAb", "TGAb": "TGAb", "TPOAb": "TPOAb",
        "Sex": "Sex", "FT4_0M": "FT4 baseline", "TSH_0M": "TSH baseline",
        "log1p_DiseaseDuration_Months_Aug": "log Duration (mo)", "Uptake24h": "24h uptake",
        "HalfLife": "Iodine half-life", "TSH_current": "TSH (current)",
        "TSH_velocity": "TSH velocity", "Hormone_load": "FT3,FT4 level",
        "T3T4_balance": "FT3,FT4 gap", "Velocity_load": "FT3,FT4 velocity",
        "Velocity_balance": "FT3,FT4 vel gap", "intercept": "intercept"}


def _disp(t):
    return " × ".join(DISP.get(p, p) for p in t.split(" & ")) if " & " in t else DISP.get(t, t)


def _decompose(ebm, x_row: pd.Series):
    """Use EBM explain_local to extract per-term log-odds contribution.

    Returns dict {term_name: contribution}. EBM's intercept_ is the bias.
    """
    X1 = pd.DataFrame([x_row], columns=x_row.index)
    expl = ebm.explain_local(X1)
    d = expl.data(0)   # first (only) sample
    names = d["names"]; scores = d["scores"]
    # explain_local also reports the intercept under "extra" usually
    contrib = dict(zip(names, scores))
    contrib["intercept"] = float(ebm.intercept_[0])
    return contrib


def _pick_patients(rows, p_cal, is_dev, lm):
    """Pick 3 dev-set 6M episodes representative of low / mid / high predicted risk."""
    mask = is_dev & (lm == LANDMARK)
    idx_pool = np.where(mask)[0]
    pp = p_cal[idx_pool]
    # quantile pick
    lo = idx_pool[np.argmin(np.abs(pp - np.quantile(pp, 0.10)))]
    md = idx_pool[np.argmin(np.abs(pp - np.median(pp)))]
    hi = idx_pool[np.argmin(np.abs(pp - np.quantile(pp, 0.90)))]
    return {"Low": lo, "Mid": md, "High": hi}


def main() -> None:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    print("Loading stacked + 6M EBM …", flush=True)
    sd = load_stacked(); sd = add_current_velocity(sd, "FT3")
    rows = sd.rows
    y = rows["Y_24M_NHRH"].values
    lm = rows["landmark"].values
    is_dev = (rows["Split"] == "Development").values

    devL = is_dev & (lm == LANDMARK)
    feat_all = build_feats_at_L(rows, devL)
    live = [c for c in FEATS if feat_all.loc[devL, c].std() > 1e-9]
    ebm = ExplainableBoostingClassifier(random_state=PY_SEED, interactions=5, max_interaction_bins=16).fit(
        feat_all.loc[devL, live], y[devL])
    p_dev = ebm.predict_proba(feat_all.loc[devL, live])[:, 1]
    # back to full-length vector for indexing
    p_full = np.zeros(len(rows)); p_full[np.where(devL)[0]] = p_dev

    picks = _pick_patients(rows, p_full, is_dev, lm)
    print(f"  picked episodes (row indices): {picks}", flush=True)

    # 1. patient profiles (raw clinical values)
    profile_cols = ["episode_id", "landmark", "Y_24M_NHRH",
                    "ThyroidW", "TPOAb", "TRAb", "log1p_DiseaseDuration_Months_Aug",
                    "TSH_0M", "FT4_0M", "TSH_current", "FT4_current",
                    "FT3_current", "TSH_velocity", "FT4_velocity", "FT3_velocity"]
    prof_recs = []
    for tag, idx in picks.items():
        rec = {"tag": tag, "predicted_p": round(float(p_full[idx]), 4)}
        for c in profile_cols:
            if c in rows.columns:
                rec[c] = float(rows[c].iloc[idx]) if isinstance(rows[c].iloc[idx], (int, float, np.number)) else rows[c].iloc[idx]
        prof_recs.append(rec)
    prof_df = pd.DataFrame(prof_recs)
    prof_df.to_csv(OUT / "tables" / "patient_profiles.csv", index=False)
    print("\n=== Patient profiles ===")
    print(prof_df.to_string(index=False), flush=True)

    # 2. decomposition tables + waterfall figs
    decomp_recs = []
    for tag, idx in picks.items():
        x_row = feat_all.loc[idx, live]
        contrib = _decompose(ebm, x_row)
        # sort by |value| desc; keep top 8 + intercept
        items = sorted(((k, v) for k, v in contrib.items() if k != "intercept"),
                       key=lambda t: -abs(t[1]))[:8]
        items.append(("intercept", contrib["intercept"]))
        total_logodds = sum(v for _, v in items)
        total_p = 1 / (1 + np.exp(-total_logodds))
        for k, v in items:
            decomp_recs.append({"tag": tag, "term": _disp(k),
                                "log_odds_contribution": round(v, 4),
                                "OR_factor": round(float(np.exp(v)), 3)})
        # waterfall
        fig, a = plt.subplots(figsize=(7, 4))
        labels = [_disp(k) for k, _ in items][::-1]
        vals = [v for _, v in items][::-1]
        colors = ["#cc4444" if v > 0 else "#2a7f5f" for v in vals]
        a.barh(range(len(vals)), vals, color=colors)
        a.set_yticks(range(len(vals))); a.set_yticklabels(labels, fontsize=8)
        a.axvline(0, color="#444", lw=0.8)
        a.set_title(f"{tag} risk patient (predicted P = {p_full[idx]:.3f}, Y = {int(y[idx])})\n"
                    f"per-feature log-odds contribution (red = ↑risk, green = ↓risk)", fontsize=10)
        a.set_xlabel("log-odds contribution", fontsize=8)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT / "figures" / f"F_decomp_{tag}.png", dpi=160)
        plt.close(fig)
    pd.DataFrame(decomp_recs).to_csv(OUT / "tables" / "decomp_table.csv", index=False)
    print(f"\n  decomposition tables + waterfalls saved", flush=True)

    # 3. counterfactual: sweep top-1 feature for each patient
    cf_recs = []
    for tag, idx in picks.items():
        x_row = feat_all.loc[idx, live].copy()
        contrib = _decompose(ebm, x_row)
        top1_univ = max(((k, v) for k, v in contrib.items() if k != "intercept" and " & " not in k),
                        key=lambda t: abs(t[1]))[0]
        # sweep this feature across its dev-set distribution
        col_vals = feat_all.loc[devL, top1_univ].values
        grid = np.linspace(np.percentile(col_vals, 5), np.percentile(col_vals, 95), 25)
        orig_val = float(x_row[top1_univ])
        ps = []
        for v in grid:
            x_cf = x_row.copy(); x_cf[top1_univ] = v
            X1 = pd.DataFrame([x_cf], columns=x_cf.index)
            ps.append(float(ebm.predict_proba(X1)[0, 1]))
        # save
        for v, p in zip(grid, ps):
            cf_recs.append({"tag": tag, "swept_feature": _disp(top1_univ),
                            "value": round(float(v), 4), "predicted_p": round(p, 4)})
        # figure
        fig, a = plt.subplots(figsize=(6.5, 3.8))
        a.plot(grid, ps, "o-", color="#1d4e89", lw=2)
        a.axvline(orig_val, color="#a23b3b", ls="--", label=f"actual = {orig_val:.2f}")
        a.axhline(p_full[idx], color="#888", ls=":", label=f"current P = {p_full[idx]:.3f}")
        a.set_title(f"{tag} risk patient — counterfactual sweep of {_disp(top1_univ)}", fontsize=10)
        a.set_xlabel(f"{_disp(top1_univ)} (z-score)" if top1_univ in ("Hormone_load","Velocity_load","T3T4_balance","Velocity_balance") else _disp(top1_univ),
                     fontsize=8)
        a.set_ylabel("EBM predicted P(24M NHRH)", fontsize=8)
        a.legend(fontsize=8)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT / "figures" / f"F_counterfactual_{tag}.png", dpi=160)
        plt.close(fig)
    pd.DataFrame(cf_recs).to_csv(OUT / "tables" / "counterfactual_table.csv", index=False)
    print("  counterfactual sweeps saved", flush=True)

    summary = {
        "landmark": LANDMARK,
        "n_patients_shown": len(picks),
        "method": "EBM explain_local for additive decomposition + univariate counterfactual sweep on top-1 feature",
        "picks_logic": "10th/50th/90th percentile of predicted P on dev@6M",
        "patient_profiles": prof_df.to_dict(orient="records"),
        "note": "EBM is additive: predicted log-odds = intercept + sum of per-feature contributions (+ pairwise interactions). Counterfactual sweeps one feature while holding the rest fixed."
    }
    (OUT / "tables" / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved → {OUT}", flush=True)


if __name__ == "__main__":
    main()
