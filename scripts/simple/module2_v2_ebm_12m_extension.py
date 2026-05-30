#!/usr/bin/env python
"""M2 · v2 EBM 12M extension — adds a 4th landmark (12M) to the per-landmark
EBM analysis, to test whether the importance drift continues past 6M or
plateaus.

Hypothesis (pre-registered): 6M shows "FT3,FT4 综合变化速度" as the lead;
12M may show one of (a) velocity still dominant (drift plateau), (b) cumulative
hormone level / stability indicators emerge, or (c) sustained TRAb takes over.
**AUC rising at 12M is a landmarking tautology (shorter lead time) and is NOT
sold as the finding**;  the new content is the *importance shift*.

Independent loader (does not mutate `module2_v2_shared.LANDMARKS`):
  - reads stage2_long_table.csv + module1_frozen_feature_matrix.csv
  - extends stacked rows to (1003 × 4 = 4012) for landmarks (1, 3, 6, 12)
  - constructs the same orthogonal hormone axes (Hormone_load / T3T4_balance /
    Velocity_load / Velocity_balance) per landmark

Output: results/module2_v2_vertical/m2v2_ebm_12m/
  - figures/F_perlandmark_importance_top6.png       (4-panel bar chart)
  - figures/F_lead_feature_shapes.png                (top-1 shape per landmark)
  - figures/F_importance_drift_heatmap.png           (group × landmark)
  - figures/F_metrics_over_time.png                  (ROC/PR/Brier)
  - tables/perf_per_landmark.csv                     (incl. 12M N, events)
  - tables/native_importance_top6.csv                (table for paper)
  - tables/summary.json                              (decision-rule inputs)
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
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)

from scripts.simple.module2_v2_shared import (
    BLOCK_A_BURDEN,
    BLOCK_B_EXPOSURE,
    M1_FROZEN,
    STAGE2_LONG,
    PY_SEED,
)

OUT = ROOT / "results" / "module2_v2_vertical" / "m2v2_ebm_12m"
LANDMARKS_EXT = (1, 3, 6, 12)
PREV = {1: 0, 3: 1, 6: 3, 12: 6}  # previous landmark for velocity
SQRT2 = np.sqrt(2.0)
DISP = {"ThyroidW": "Thyroid weight", "TRAb": "TRAb", "TGAb": "TGAb", "TPOAb": "TPOAb",
        "Sex": "Sex", "FT4_0M": "FT4 (0M)", "TSH_0M": "TSH (0M)",
        "log1p_DiseaseDuration_Months_Aug": "Duration (log,mo)", "Uptake24h": "24h uptake",
        "HalfLife": "Iodine half-life", "TSH_current": "TSH (current)", "TSH_velocity": "TSH velocity",
        "Hormone_load": "FT3,FT4 level", "T3T4_balance": "FT3,FT4 gap",
        "Velocity_load": "FT3,FT4 velocity", "Velocity_balance": "FT3,FT4 vel gap"}


def _disp(t):
    return " × ".join(DISP.get(p, p) for p in t.split(" & ")) if " & " in t else DISP.get(t, t)


# ---------------------------------------------------------------------------
# Independent loader (does not touch module2_v2_shared.LANDMARKS)
# ---------------------------------------------------------------------------


def load_stacked_12m() -> pd.DataFrame:
    """Build a stacked dataset over landmarks (1, 3, 6, 12).

    Returns DataFrame with one row per (episode × landmark in {1,3,6,12}).
    Columns: episode_id, landmark, Split, Y_24M_NHRH, baseline burden (Block A),
    RAI exposure (Block B), TSH_current, FT4_current, FT3_current,
    TSH_velocity, FT4_velocity, FT3_velocity, plus audit columns.
    """
    m1 = pd.read_csv(M1_FROZEN, low_memory=False)
    needed = {"Episode_Index", "Split", "Y"}.union(BLOCK_A_BURDEN).union(BLOCK_B_EXPOSURE)
    miss = needed - set(m1.columns)
    if miss:
        raise RuntimeError(f"M1 frozen missing: {miss}")
    if len(m1) != 1003:
        raise RuntimeError(f"M1 frozen rows {len(m1)} != 1003")

    s2 = pd.read_csv(STAGE2_LONG, low_memory=False)
    if "Treatment_Index" not in s2.columns:
        raise RuntimeError("Stage2 long table needs Treatment_Index")
    s2 = s2.drop_duplicates("Treatment_ID", keep="first")
    s2["Episode_Index"] = s2["Treatment_Index"].astype(int)
    # Stage2 0M would collide with M1 frozen's FT4_0M/TSH_0M on merge — skip 0M
    # from s2 (M1 is authoritative for baseline). For FT3_0M (not in M1) take it
    # from s2 separately.
    s2_ft3_0m = s2[["Episode_Index", "FT3_0M"]].rename(columns={"FT3_0M": "FT3_0M_s2"})
    need_cols = []
    for m in ("FT3", "FT4", "TSH"):
        for L in (1, 3, 6, 12):
            c = f"{m}_{L}M"
            if c not in s2.columns:
                raise RuntimeError(f"Stage2 long table missing column: {c}")
            need_cols.append(c)
    s2 = s2[["Episode_Index"] + need_cols]
    combined = (m1.merge(s2, on="Episode_Index", validate="one_to_one")
                  .merge(s2_ft3_0m, on="Episode_Index", validate="one_to_one"))
    combined["FT3_0M"] = combined["FT3_0M_s2"]

    rows = []
    for _, ep in combined.iterrows():
        for L in LANDMARKS_EXT:
            r = {"episode_id": int(ep["Episode_Index"]), "landmark": L,
                 "Split": ep["Split"], "Y_24M_NHRH": int(ep["Y"])}
            for f in BLOCK_A_BURDEN:
                r[f] = ep[f]
            for f in BLOCK_B_EXPOSURE:
                r[f] = ep[f]
            r["TSH_current"] = ep[f"TSH_{L}M"]
            r["FT4_current"] = ep[f"FT4_{L}M"]
            r["FT3_current"] = ep[f"FT3_{L}M"]
            p = PREV[L]
            dt = L - p
            r["TSH_velocity"] = (ep[f"TSH_{L}M"] - ep[f"TSH_{p}M"]) / dt
            r["FT4_velocity"] = (ep[f"FT4_{L}M"] - ep[f"FT4_{p}M"]) / dt
            r["FT3_velocity"] = (ep[f"FT3_{L}M"] - ep[f"FT3_{p}M"]) / dt
            rows.append(r)
    out = pd.DataFrame(rows)
    # Dev-only median impute (consistent with shared)
    dev = (out["Split"] == "Development").values
    num_cols = ["TSH_current", "FT4_current", "FT3_current",
                "TSH_velocity", "FT4_velocity", "FT3_velocity"]
    for c in num_cols:
        out[c] = out[c].replace([np.inf, -np.inf], np.nan)
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].fillna(out.loc[dev, c].median())
    # Block A/B already numeric in M1 frozen
    for c in BLOCK_A_BURDEN + BLOCK_B_EXPOSURE:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(out.loc[dev, c].median() if dev.any() else 0)
    return out


def _zfit(v, m):
    mu = np.nanmean(v[m]); sd = np.nanstd(v[m])
    if not np.isfinite(sd) or sd < 1e-9:
        return np.zeros_like(v)
    return np.nan_to_num((v - mu) / sd, nan=0.0)


def build_feats_at_L(rows: pd.DataFrame, maskL: np.ndarray) -> pd.DataFrame:
    """Same orthogonal axes as the main EBM analysis."""
    ft3 = rows["FT3_current"].values.astype(float)
    ft4 = rows["FT4_current"].values.astype(float)
    dft3 = rows["FT3_velocity"].values.astype(float)
    dft4 = rows["FT4_velocity"].values.astype(float)
    z3, z4 = _zfit(ft3, maskL), _zfit(ft4, maskL)
    dz3, dz4 = _zfit(dft3, maskL), _zfit(dft4, maskL)
    df = pd.DataFrame(index=rows.index)
    for c in BLOCK_A_BURDEN + BLOCK_B_EXPOSURE + ["TSH_current", "TSH_velocity"]:
        df[c] = rows[c].values
    df["Hormone_load"] = (z3 + z4) / SQRT2
    df["T3T4_balance"] = (z3 - z4) / SQRT2
    df["Velocity_load"] = (dz3 + dz4) / SQRT2
    df["Velocity_balance"] = (dz3 - dz4) / SQRT2
    return df


FEATS = (BLOCK_A_BURDEN + BLOCK_B_EXPOSURE +
         ["TSH_current", "TSH_velocity",
          "Hormone_load", "T3T4_balance", "Velocity_load", "Velocity_balance"])


# ---------------------------------------------------------------------------
# Run per-landmark EBM + figures
# ---------------------------------------------------------------------------


def main() -> None:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    print("Loading 12M-extended stacked dataset…", flush=True)
    rows = load_stacked_12m()
    print(f"  rows: {len(rows)} (expect 1003 × 4 = 4012)", flush=True)
    y = rows["Y_24M_NHRH"].values
    lm = rows["landmark"].values
    is_dev = (rows["Split"] == "Development").values

    M = {}
    perf_rows = []
    for L in LANDMARKS_EXT:
        devL = is_dev & (lm == L); tstL = (~is_dev) & (lm == L)
        feat = build_feats_at_L(rows, devL)
        live = [c for c in FEATS if feat.loc[devL, c].std() > 1e-9]
        Xtr, ytr = feat.loc[devL, live], y[devL]
        Xte, yte = feat.loc[tstL, live], y[tstL]
        e = ExplainableBoostingClassifier(random_state=PY_SEED, interactions=5).fit(Xtr, ytr)
        p = e.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, p); pr = average_precision_score(yte, p); br = brier_score_loss(yte, p)
        M[L] = {"ebm": e, "Xte": Xte, "yte": yte, "p": p, "live": live, "auc": auc}
        perf_rows.append({"landmark": f"{L}M", "N_test": int(tstL.sum()),
                          "events_test": int(yte.sum()), "ROC_AUC": round(auc, 4),
                          "PR_AUC": round(pr, 4), "Brier": round(br, 4),
                          "lead_months_to_24M": 24 - L,
                          "FT4_current_coverage_pct": round(rows.loc[devL, "FT4_current"].notna().mean() * 100, 1)})
        print(f"  {L}M EBM AUC={auc:.4f}  N_test={int(tstL.sum())}  events={int(yte.sum())}", flush=True)

    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(OUT / "tables" / "perf_per_landmark.csv", index=False)
    print("\n=== Per-landmark perf ===")
    print(perf_df.to_string(index=False), flush=True)

    # Native top-6 importance per landmark
    top_records = []
    for L in LANDMARKS_EXT:
        g = M[L]["ebm"].explain_global()
        terms = sorted(zip(g.data()["names"], g.data()["scores"]), key=lambda t: -t[1])[:6]
        for rank, (n, s) in enumerate(terms, 1):
            top_records.append({"landmark": f"{L}M", "rank": rank,
                                "feature": _disp(n), "importance": round(float(s), 4)})
    top_df = pd.DataFrame(top_records)
    top_df.to_csv(OUT / "tables" / "native_importance_top6.csv", index=False)
    print("\n=== Top-6 per landmark ===")
    for L in LANDMARKS_EXT:
        sub = top_df[top_df["landmark"] == f"{L}M"]
        print(f"  {L}M: " + " | ".join(f"{r['feature']} {r['importance']:.3f}" for _, r in sub.iterrows()))

    # Fig: per-landmark importance bars (4 panels)
    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    for k, L in enumerate(LANDMARKS_EXT):
        sub = top_df[top_df["landmark"] == f"{L}M"].iloc[::-1]
        a = axs[k // 2][k % 2]
        a.barh(range(len(sub)), sub["importance"].values, color="#2a6f97")
        a.set_yticks(range(len(sub))); a.set_yticklabels(sub["feature"].tolist(), fontsize=8)
        a.set_title(f"{L}M  (AUC {M[L]['auc']:.3f},  lead {24-L}M to 24M endpoint)", fontsize=10)
        a.set_xlabel("EBM native importance (mean |contribution|)", fontsize=8)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("EBM per-landmark importance — does the drift continue past 6M?", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "figures" / "F_perlandmark_importance_top6.png", dpi=160)
    plt.close(fig)

    # Fig: lead-feature shape per landmark
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    for k, L in enumerate(LANDMARKS_EXT):
        g = M[L]["ebm"].explain_global()
        univ = [n for n, s in sorted(zip(g.data()["names"], g.data()["scores"]), key=lambda t: -t[1])
                if " & " not in n]
        if not univ:
            continue
        fn = univ[0]; dd = g.data(M[L]["ebm"].term_names_.index(fn))
        xs = np.asarray(dd.get("names", []), float); ys = np.asarray(dd.get("scores", []), float)
        a = axs[k // 2][k % 2]
        if len(xs) == len(ys) + 1:
            mid = (xs[:-1] + xs[1:]) / 2
            a.step(mid, ys, where="mid", color="#a23b3b", lw=2)
            a.fill_between(mid, ys, step="mid", alpha=0.12, color="#a23b3b")
        elif len(ys):
            a.plot(xs[:len(ys)], ys, "o-", color="#a23b3b")
        a.axhline(0, ls=":", color="#999")
        a.set_title(f"{L}M  lead: {_disp(fn)}", fontsize=10)
        a.set_xlabel("value", fontsize=8); a.set_ylabel("log-odds contribution", fontsize=8)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("EBM lead-feature shape function per landmark (1/3/6/12 M)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "figures" / "F_lead_feature_shapes.png", dpi=160)
    plt.close(fig)

    # Fig: importance drift heatmap (groups × landmark)
    GRP = {"ThyroidW": ["ThyroidW"], "Antibodies": ["TRAb", "TGAb", "TPOAb"],
           "Baseline/duration": ["FT4_0M", "TSH_0M", "log1p_DiseaseDuration_Months_Aug", "Sex"],
           "RAI exposure": ["Uptake24h", "HalfLife"],
           "FT3,FT4 level": ["Hormone_load"], "TSH (current)": ["TSH_current"],
           "FT3,FT4 velocity": ["Velocity_load"], "TSH velocity": ["TSH_velocity"],
           "FT3,FT4 gap": ["T3T4_balance"]}
    mat = np.zeros((len(GRP), len(LANDMARKS_EXT)))
    for j, L in enumerate(LANDMARKS_EXT):
        g = M[L]["ebm"].explain_global()
        imp = {n: s for n, s in zip(g.data()["names"], g.data()["scores"]) if " & " not in n}
        for i, (gn, cols) in enumerate(GRP.items()):
            mat[i, j] = sum(imp.get(c, 0) for c in cols)
    fig, a = plt.subplots(figsize=(7, 4.6))
    im = a.imshow(mat, cmap="YlOrRd", aspect="auto")
    a.set_xticks(range(4)); a.set_xticklabels([f"{L}M" for L in LANDMARKS_EXT])
    a.set_yticks(range(len(GRP))); a.set_yticklabels(list(GRP), fontsize=8)
    for i in range(len(GRP)):
        for j in range(4):
            a.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                   color="white" if mat[i, j] > mat.max() * 0.6 else "#222")
    a.set_title("EBM group importance drift — 1M → 12M", fontsize=10)
    fig.colorbar(im, ax=a, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "F_importance_drift_heatmap.png", dpi=160)
    plt.close(fig)

    # Fig: metrics over time
    fig, a = plt.subplots(figsize=(6, 4))
    a.plot([f"{L}M" for L in LANDMARKS_EXT], perf_df["ROC_AUC"], "o-", label="ROC-AUC")
    a.plot([f"{L}M" for L in LANDMARKS_EXT], perf_df["PR_AUC"], "s-", label="PR-AUC")
    a.plot([f"{L}M" for L in LANDMARKS_EXT], perf_df["Brier"], "^-", label="Brier (lower better)")
    a.legend(fontsize=8); a.set_title("EBM metrics: 1M → 12M", fontsize=10)
    a.set_xlabel("landmark"); a.set_ylim(0, max(0.9, perf_df["ROC_AUC"].max() + 0.05))
    a.annotate("AUC rising = shorter lead-time tautology; see paper §3.7", xy=(0.5, 0.04),
               xycoords="axes fraction", ha="center", fontsize=7, color="#666")
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "F_metrics_over_time.png", dpi=160)
    plt.close(fig)

    summary = {
        "landmarks": list(LANDMARKS_EXT),
        "per_landmark": perf_df.to_dict(orient="records"),
        "top6": top_df.to_dict(orient="records"),
        "drift_interpretation_hint": (
            "Compare top-1 at 6M vs 12M:"
            "  - velocity remains #1 → drift plateau (negative finding still useful)"
            "  - level/cumulative emerges → new chapter for 12M phenotype"
            "  - TRAb-related rises → immune-driven late-phase indicator"
        ),
        "AUC_tautology_note": (
            "Rising AUC at later landmarks reflects shorter lead time to 24M endpoint;"
            " do NOT sell as discrimination finding. Importance drift is the real content."
        ),
    }
    (OUT / "tables" / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved → {OUT}", flush=True)


if __name__ == "__main__":
    main()
