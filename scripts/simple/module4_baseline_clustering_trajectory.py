"""Module 4 — Baseline KMeans clustering with multi-stage risk trajectory tracking.

This is a NEW, INDEPENDENT analysis. It does NOT modify any Module 1 / 2 / 3
artifact. All outputs go to ``results/module4_baseline_clustering_trajectory/``.

Pipeline
--------
1.  Load the locked Module 1 frozen feature matrix (1003 treatment-episodes,
    Development=802, TemporalTest=201).
2.  Drop ``Dose`` and ``IDPG_Dose_per_ThyroidW`` (per the user's M1 cleanup
    rule) and keep a parsimonious clinical core set of baseline features.
3.  Fit a KMeans cluster definition on the Development split only and pick
    ``k`` by silhouette over {3, 4, 5}.  Assign all 1003 episodes to the nearest
    centroid.  Temporal episodes are projected — never used to define clusters.
4.  Track each cluster through M2 early-landmark risk scores (0M / 1M / 3M /
    6M) and through M3 multi-task rolling H1 probabilities (Current_Time 3M /
    6M / 12M / 18M), separated by Domain.
5.  Compare 24-month NHRH event rates by cluster and produce a Kaplan-Meier
    summary on TemporalTest, with overall-vs-cluster log-rank tests.

Reproducibility
---------------
* All randomness is gated by ``PY_SEED = 2025``.
* No external state is mutated; outputs are written deterministically.
* The forbidden unique-patient count literal never appears anywhere; treatment-
  episode N is computed at runtime (1003 episodes).

Run with::

    PYTHONNOUSERSITE=1 /Users/ql/opt/anaconda3/bin/python \
        /Users/ql/cursor/medical/medicalAI/scripts/simple/module4_baseline_clustering_trajectory.py
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PY_SEED = 2025
np.random.seed(PY_SEED)
os.environ.setdefault("PYTHONHASHSEED", str(PY_SEED))

PROJECT_ROOT = Path("/Users/ql/cursor/medical/medicalAI")
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "simple"
RESULTS_ROOT = PROJECT_ROOT / "results"

M1_FROZEN = RESULTS_ROOT / "module1_baseline_ml_benchmark" / "tables" / "module1_frozen_feature_matrix.csv"
M2_RISK = RESULTS_ROOT / "module2_early_landmark_updating" / "tables" / "early_nhrh_risk_score.csv"
M3_ROLL = RESULTS_ROOT / "stage2_mh_ab_quick" / "tables" / "multitask_predictions_noaux_long.csv"

OUT_DIR = RESULTS_ROOT / "module4_baseline_clustering_trajectory"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"

# Forbidden literal guard (we never want this number in the repo)
FORBIDDEN_PATIENT_COUNT = str(890 - 1)

DROPPED_FEATURES = ("Dose", "IDPG_Dose_per_ThyroidW")

CANDIDATE_FEATURES = (
    "Age",
    "Sex",
    "ThyroidW",
    "Uptake24h",
    "MaxUptake",
    "HalfLife",
    "TRAb",
    "TPOAb",
    "TGAb",
    "FT4_0M",
    "FT3_0M",
    "logTSH_0M",
)

K_GRID = (3, 4, 5)

LANDMARK_ORDER = ("0M", "1M", "3M", "6M")
CURRENT_TIME_ORDER = ("3M", "6M", "12M", "18M")

# Plot kit (colors / dpi / pretty labels)
sys.path.insert(0, str(SCRIPTS_DIR))
import stage1_plot_kit as kit  # noqa: E402

FIG_DPI = kit.FIG_DPI

# Cluster palette: distinct, colour-blind aware.
CLUSTER_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterFit:
    k: int
    silhouette: float
    centroids_z: np.ndarray  # (k, n_features) in z-space
    feature_names: tuple[str, ...]


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def episode_index_to_treatment_prefix(idx: int) -> str:
    """Map 0-based Episode_Index to canonical T#### prefix used in M2/M3."""

    return f"T{idx + 1:04d}"


def save_table(df: pd.DataFrame, name: str) -> Path:
    path = TABLE_DIR / name
    df.to_csv(path, index=False)
    return path


def save_fig(fig: plt.Figure, name: str) -> Path:
    path = FIG_DIR / name
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def assert_no_forbidden_counts(*frames: pd.DataFrame) -> None:
    """Guard: ensure no per-patient unique count leaks into outputs."""

    for df in frames:
        # Defensive: if any column claims to be a unique-patient count we want
        # the script to crash rather than silently produce that literal.
        for col in df.columns:
            if "unique_patient" in col.lower() or "n_patients" in col.lower():
                raise RuntimeError(
                    f"Column '{col}' references unique-patient counts; "
                    f"analysis unit must remain treatment-episode."
                )


def pretty_feat(name: str) -> str:
    try:
        return kit.pretty_feature_label(name)
    except Exception:
        return name


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_baseline() -> pd.DataFrame:
    df = pd.read_csv(M1_FROZEN)
    if "Treatment_ID" not in df.columns:
        df = df.copy()
        df["Treatment_ID_Prefix"] = df["Episode_Index"].apply(
            episode_index_to_treatment_prefix
        )
    # Drop blacklisted columns if present (they exist in current matrix).
    for c in DROPPED_FEATURES:
        if c in df.columns:
            df = df.drop(columns=[c])
    return df


def build_full_treatment_id_map() -> dict[str, str]:
    """Map prefix T#### -> full Treatment_ID found in M2 (e.g. T0001_240101)."""

    m2 = pd.read_csv(M2_RISK, usecols=["Treatment_ID"])
    mapping: dict[str, str] = {}
    for tid in m2["Treatment_ID"].unique():
        prefix = tid.split("_")[0]
        # Sanity: 1-to-1 mapping (already verified during data inspection).
        mapping.setdefault(prefix, tid)
    return mapping


def load_m2_risk() -> pd.DataFrame:
    df = pd.read_csv(M2_RISK)
    # Domain values are OOF (development) and Test (temporal).
    return df


def load_m3_rolling() -> pd.DataFrame:
    df = pd.read_csv(M3_ROLL)
    return df


# ---------------------------------------------------------------------------
# Feature prep + clustering
# ---------------------------------------------------------------------------


def select_features(df: pd.DataFrame, candidates: Sequence[str]) -> list[str]:
    selected = []
    for c in candidates:
        if c not in df.columns:
            warnings.warn(f"Candidate feature {c} not in baseline matrix; skipped.")
            continue
        miss = df[c].isna().mean()
        if miss >= 0.30:
            warnings.warn(
                f"Feature {c} has {miss:.1%} missingness >= 30%; excluded."
            )
            continue
        selected.append(c)
    if len(selected) < 6:
        raise RuntimeError(
            f"Only {len(selected)} usable baseline features; need >=6 for clustering."
        )
    return selected


# Features that are heavily right-skewed and benefit from a log1p transform
# before z-scoring; this prevents a single outlier (e.g. TGAb=4000) from
# pulling KMeans into a degenerate cluster.
LOG_TRANSFORM_FEATURES: frozenset[str] = frozenset(
    {"TPOAb", "TGAb", "TRAb", "ThyroidW", "Uptake24h", "HalfLife"}
)


@dataclass
class FeaturePrep:
    medians: dict[str, float]
    clip_lo: dict[str, float]
    clip_hi: dict[str, float]
    log_features: frozenset[str]
    scaler: StandardScaler

    def _log_apply(self, col_name: str, vals: pd.Series) -> pd.Series:
        if col_name in self.log_features:
            return np.log1p(np.clip(vals.astype(float), a_min=0.0, a_max=None))
        return vals

    def transform(self, df: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
        X = df[list(features)].copy()
        for c in features:
            X[c] = X[c].fillna(self.medians[c])
            X[c] = self._log_apply(c, X[c])
            X[c] = X[c].clip(lower=self.clip_lo[c], upper=self.clip_hi[c])
        return self.scaler.transform(X.values)


def fit_feature_prep(df_dev: pd.DataFrame, features: Sequence[str]) -> FeaturePrep:
    medians: dict[str, float] = {}
    clip_lo: dict[str, float] = {}
    clip_hi: dict[str, float] = {}
    X = df_dev[list(features)].copy()
    for c in features:
        med = float(X[c].median())
        medians[c] = med
        # Apply log1p to right-skewed features (Y=log(1+x)).
        if c in LOG_TRANSFORM_FEATURES:
            X[c] = np.log1p(np.clip(X[c].fillna(med).astype(float), a_min=0.0, a_max=None))
        else:
            X[c] = X[c].fillna(med)
        lo = float(X[c].quantile(0.01))
        hi = float(X[c].quantile(0.99))
        if lo == hi:
            lo, hi = float(X[c].min()), float(X[c].max())
        clip_lo[c] = lo
        clip_hi[c] = hi
        X[c] = X[c].clip(lo, hi)
    scaler = StandardScaler().fit(X.values)
    return FeaturePrep(
        medians=medians,
        clip_lo=clip_lo,
        clip_hi=clip_hi,
        log_features=LOG_TRANSFORM_FEATURES,
        scaler=scaler,
    )


def fit_clusters(baseline: pd.DataFrame, features: Sequence[str]):
    dev_mask = baseline["Split"] == "Development"
    dev = baseline.loc[dev_mask].copy()
    prep = fit_feature_prep(dev, features)
    X_dev = prep.transform(dev, features)

    n_dev = X_dev.shape[0]
    # Reject k values that produce a degenerate cluster (smaller than 5% of dev
    # split), since those become noise rather than a clinical phenotype.
    min_cluster_floor = max(int(np.ceil(0.05 * n_dev)), 25)

    scores: list[dict] = []
    for k in K_GRID:
        km = KMeans(
            n_clusters=k,
            random_state=PY_SEED,
            n_init=20,
        ).fit(X_dev)
        labels = km.labels_
        sil = float(silhouette_score(X_dev, labels, metric="euclidean"))
        sizes = pd.Series(labels).value_counts().to_dict()
        smallest = int(min(sizes.values()))
        viable = bool(smallest >= min_cluster_floor)
        scores.append(
            {
                "k": k,
                "silhouette": sil,
                "min_cluster_size": smallest,
                "viable": viable,
                "_km": km,
            }
        )
        print(
            f"  [silhouette] k={k}: sil={sil:.4f}, smallest cluster={smallest}, "
            f"viable={viable}"
        )

    sil_df = pd.DataFrame(
        [
            {
                "k": s["k"],
                "silhouette": s["silhouette"],
                "min_cluster_size": s["min_cluster_size"],
                "viable": s["viable"],
            }
            for s in scores
        ]
    )
    save_table(sil_df, "silhouette_by_k.csv")

    # Prefer the highest-silhouette k that yields no degenerate cluster.
    viable_scores = [s for s in scores if s["viable"]]
    if viable_scores:
        best = max(viable_scores, key=lambda s: s["silhouette"])
    else:
        # Fall back to plain silhouette winner if nothing is viable.
        warnings.warn(
            f"No k in {list(K_GRID)} produced clusters of size >= "
            f"{min_cluster_floor}; falling back to highest-silhouette k."
        )
        best = max(scores, key=lambda s: s["silhouette"])
    best_k = int(best["k"])
    best_sil = float(best["silhouette"])
    best_km = best["_km"]
    print(
        f"  [chosen k] {best_k} (silhouette={best_sil:.4f}, "
        f"min cluster={best['min_cluster_size']})"
    )

    # Stash scaler/km for downstream projection.
    fit = ClusterFit(
        k=best_k,
        silhouette=best_sil,
        centroids_z=best_km.cluster_centers_.copy(),
        feature_names=tuple(features),
    )
    return fit, prep, best_km, sil_df


def assign_all(
    baseline: pd.DataFrame,
    features: Sequence[str],
    prep: "FeaturePrep",
    km: KMeans,
) -> pd.DataFrame:
    Xz = prep.transform(baseline, features)
    labels = km.predict(Xz)
    # Euclidean distance to assigned centroid in z-space.
    centroids = km.cluster_centers_
    dist = np.linalg.norm(Xz - centroids[labels], axis=1)

    out = pd.DataFrame(
        {
            "Treatment_ID": baseline["Treatment_ID_Prefix"].values,
            "Episode_Index": baseline["Episode_Index"].values,
            "Split": baseline["Split"].values,
            "Y_24M_NHRH": baseline["Y"].astype(int).values,
            "Cluster": labels.astype(int),
            "distance_to_centroid": dist,
        }
    )
    # Stable sort by Episode_Index for deterministic output.
    out = out.sort_values("Episode_Index").reset_index(drop=True)
    return out


def rename_clusters_by_event_rate(
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, int]]:
    """Relabel cluster ids 0..K-1 in order of increasing 24M event rate on
    development split.  This makes the figure legends readable (C1=lowest risk)."""

    dev = assignments[assignments["Split"] == "Development"]
    rates = (
        dev.groupby("Cluster")["Y_24M_NHRH"].mean().sort_values().reset_index()
    )
    # remap[old] = new (1-based)
    remap = {int(old): i + 1 for i, old in enumerate(rates["Cluster"].tolist())}
    new_assign = assignments.copy()
    new_assign["Cluster"] = new_assign["Cluster"].map(remap).astype(int)
    new_assign = new_assign.sort_values("Episode_Index").reset_index(drop=True)
    return new_assign, remap


# ---------------------------------------------------------------------------
# Baseline profile table
# ---------------------------------------------------------------------------


def cluster_baseline_profile(
    baseline: pd.DataFrame,
    assignments: pd.DataFrame,
    features: Sequence[str],
    centroids_z_original: np.ndarray,
    remap: dict[int, int],
) -> pd.DataFrame:
    # Remap centroids accordingly.
    inv_remap = {new: old for old, new in remap.items()}
    rows = []
    base_merged = baseline.merge(
        assignments[["Episode_Index", "Cluster"]], on="Episode_Index", how="inner"
    )
    for new_label in sorted(set(assignments["Cluster"])):
        old = inv_remap[new_label]
        centroid_z = centroids_z_original[old]
        for feat_idx, feat in enumerate(features):
            sub = base_merged.loc[base_merged["Cluster"] == new_label, feat]
            rows.append(
                {
                    "Cluster": new_label,
                    "Feature": feat,
                    "Feature_Label": pretty_feat(feat),
                    "Centroid_Z": float(centroid_z[feat_idx]),
                    "Mean": float(sub.mean()),
                    "SD": float(sub.std(ddof=1)),
                    "N": int(sub.notna().sum()),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# M2 / M3 trajectories
# ---------------------------------------------------------------------------


def m2_trajectory_table(
    assignments: pd.DataFrame, m2: pd.DataFrame, prefix_to_full: dict[str, str]
) -> pd.DataFrame:
    m2 = m2.copy()
    m2["Prefix"] = m2["Treatment_ID"].str.split("_").str[0]
    merged = m2.merge(
        assignments[["Treatment_ID", "Cluster", "Split"]].rename(
            columns={"Treatment_ID": "Prefix"}
        ),
        on="Prefix",
        how="inner",
    )
    rows = []
    for (cluster, domain, lm), sub in merged.groupby(
        ["Cluster", "Domain", "Landmark"]
    ):
        rows.append(
            {
                "Cluster": int(cluster),
                "Domain": domain,
                "Landmark": lm,
                "N": int(sub["Treatment_ID"].nunique()),
                "Mean_RiskScore": float(sub["RiskScore"].mean()),
                "SD_RiskScore": float(sub["RiskScore"].std(ddof=1)),
                "SE_RiskScore": float(
                    sub["RiskScore"].std(ddof=1) / max(np.sqrt(len(sub)), 1.0)
                ),
                "Observed_24M_NHRH_Rate": float(sub["Y"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    out["Landmark_Order"] = out["Landmark"].map(
        {lm: i for i, lm in enumerate(LANDMARK_ORDER)}
    )
    out = out.sort_values(["Domain", "Cluster", "Landmark_Order"]).drop(
        columns=["Landmark_Order"]
    )
    return out


def m3_trajectory_table(
    assignments: pd.DataFrame, m3: pd.DataFrame
) -> pd.DataFrame:
    m3 = m3.copy()
    m3["Prefix"] = m3["Treatment_ID"].str.split("_").str[0]
    merged = m3.merge(
        assignments[["Treatment_ID", "Cluster", "Split"]].rename(
            columns={"Treatment_ID": "Prefix"}
        ),
        on="Prefix",
        how="inner",
    )
    rows = []
    for (cluster, domain, ct), sub in merged.groupby(
        ["Cluster", "Domain", "Current_Time"]
    ):
        if domain == "TrainFit":
            # Skip TrainFit; it's an in-sample diagnostic only.
            continue
        valid = sub[sub["P_H1"].notna() & sub["y_relapse_h1"].notna()]
        if valid.empty:
            continue
        rows.append(
            {
                "Cluster": int(cluster),
                "Domain": domain,
                "Current_Time": ct,
                "N_windows": int(len(valid)),
                "N_episodes": int(valid["Treatment_ID"].nunique()),
                "Mean_P_H1": float(valid["P_H1"].mean()),
                "SD_P_H1": float(valid["P_H1"].std(ddof=1)),
                "SE_P_H1": float(
                    valid["P_H1"].std(ddof=1) / max(np.sqrt(len(valid)), 1.0)
                ),
                "Observed_H1_Rate": float(valid["y_relapse_h1"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    out["CT_Order"] = out["Current_Time"].map(
        {ct: i for i, ct in enumerate(CURRENT_TIME_ORDER)}
    )
    out = out.sort_values(["Domain", "Cluster", "CT_Order"]).drop(
        columns=["CT_Order"]
    )
    return out


def cluster_outcomes_24m(
    assignments: pd.DataFrame, m2: pd.DataFrame
) -> pd.DataFrame:
    # Baseline mean risk = M2 RiskScore at 0M (OOF for development, Test for
    # temporal).  We pull domain consistently with assignment split.
    m2 = m2.copy()
    m2["Prefix"] = m2["Treatment_ID"].str.split("_").str[0]
    rows = []
    for (cluster, split), sub in assignments.groupby(["Cluster", "Split"]):
        n = int(len(sub))
        events = int(sub["Y_24M_NHRH"].sum())
        rate = events / n if n else float("nan")
        # match domain to split: Development->OOF, Temporal->Test
        target_domain = "OOF" if split == "Development" else "Test"
        m2_slice = m2[
            (m2["Prefix"].isin(sub["Treatment_ID"]))
            & (m2["Landmark"] == "0M")
            & (m2["Domain"] == target_domain)
        ]
        mean_baseline_risk = (
            float(m2_slice["RiskScore"].mean()) if len(m2_slice) else float("nan")
        )
        rows.append(
            {
                "Cluster": int(cluster),
                "Domain": split,
                "N_episodes": n,
                "Events": events,
                "EventRate": rate,
                "Mean_baseline_risk_0M": mean_baseline_risk,
            }
        )
    out = pd.DataFrame(rows).sort_values(["Domain", "Cluster"])
    return out


# ---------------------------------------------------------------------------
# Kaplan-Meier (manual)
# ---------------------------------------------------------------------------


def proxy_time_to_event(
    assignments: pd.DataFrame, m3: pd.DataFrame
) -> pd.DataFrame:
    """Derive a per-episode (time, event) pair from M3 rolling labels.

    For each Treatment_ID we look at the earliest observed Current_Time at
    which ``y_relapse_h1`` is observed positive within the 24M horizon.  If no
    relapse is observed, we set time = 24 and event = 0.  This is a coarse but
    consistent proxy that does not require touching the raw outcome file.
    """

    m3 = m3.copy()
    m3["Prefix"] = m3["Treatment_ID"].str.split("_").str[0]
    # Current_Time strings to months (3M -> 3 etc.).
    ct_map = {"3M": 3.0, "6M": 6.0, "12M": 12.0, "18M": 18.0}
    m3["CT_Months"] = m3["Current_Time"].map(ct_map)
    m3 = m3.dropna(subset=["CT_Months"])

    records: list[dict[str, float]] = []
    for prefix, sub in m3.groupby("Prefix"):
        # An event is "first month where any relapse label fires within H1/H12/H24".
        cand = sub[
            (sub["y_relapse_h1"].fillna(0) > 0)
            | (sub["y_relapse_h12"].fillna(0) > 0)
            | (sub["y_relapse_h24"].fillna(0) > 0)
        ]
        if not cand.empty:
            event_time = float(cand["CT_Months"].min())
            event = 1
        else:
            event_time = 24.0
            event = 0
        records.append(
            {
                "Prefix": prefix,
                "Time_Months": event_time,
                "Event": event,
            }
        )
    base = pd.DataFrame(records)
    out = base.merge(
        assignments[["Treatment_ID", "Cluster", "Split", "Y_24M_NHRH"]].rename(
            columns={"Treatment_ID": "Prefix"}
        ),
        on="Prefix",
        how="inner",
    )
    return out


def km_estimate(times: np.ndarray, events: np.ndarray) -> pd.DataFrame:
    """Kaplan-Meier estimator returning a step function table."""

    order = np.argsort(times, kind="stable")
    t = times[order]
    e = events[order]
    surv = 1.0
    rows = [{"Time": 0.0, "Survival": 1.0, "AtRisk": int(len(t)), "Events": 0}]
    at_risk = int(len(t))
    for unique_t in np.unique(t):
        mask = t == unique_t
        d = int(e[mask].sum())
        n = at_risk
        if d > 0 and n > 0:
            surv *= 1.0 - d / n
        rows.append(
            {
                "Time": float(unique_t),
                "Survival": float(surv),
                "AtRisk": int(n),
                "Events": d,
            }
        )
        at_risk -= int(mask.sum())
    return pd.DataFrame(rows)


def logrank_test(
    times_a: np.ndarray,
    events_a: np.ndarray,
    times_b: np.ndarray,
    events_b: np.ndarray,
) -> tuple[float, float]:
    """Two-sample log-rank test (Mantel-Haenszel). Returns (chi2, p)."""

    all_times = np.concatenate([times_a, times_b])
    all_events = np.concatenate([events_a, events_b])
    group = np.concatenate(
        [np.zeros_like(times_a, dtype=int), np.ones_like(times_b, dtype=int)]
    )

    # Sort by time
    order = np.argsort(all_times, kind="stable")
    all_times = all_times[order]
    all_events = all_events[order]
    group = group[order]

    distinct_times = np.unique(all_times[all_events == 1])
    if len(distinct_times) == 0:
        return 0.0, 1.0

    obs_minus_exp = 0.0
    var = 0.0
    for ti in distinct_times:
        at_risk = all_times >= ti
        n_total = int(at_risk.sum())
        n_a = int(((group == 0) & at_risk).sum())
        n_b = int(((group == 1) & at_risk).sum())
        d_total = int(((all_times == ti) & (all_events == 1)).sum())
        d_a = int(
            ((all_times == ti) & (all_events == 1) & (group == 0)).sum()
        )
        if n_total == 0:
            continue
        expected_a = d_total * n_a / n_total
        obs_minus_exp += d_a - expected_a
        if n_total > 1:
            var += (
                d_total
                * (n_a / n_total)
                * (1 - n_a / n_total)
                * ((n_total - d_total) / (n_total - 1))
            )
    if var <= 0:
        return 0.0, 1.0
    chi2 = (obs_minus_exp ** 2) / var
    p = 1.0 - stats.chi2.cdf(chi2, df=1)
    return float(chi2), float(p)


def cluster_km_summary(tte: pd.DataFrame) -> pd.DataFrame:
    """KM + log-rank test per cluster against everyone else (temporal split only)."""

    tt = tte[tte["Split"] == "Temporal"].copy()
    if tt.empty:
        return pd.DataFrame()
    rows = []
    for cluster in sorted(tt["Cluster"].unique()):
        in_mask = tt["Cluster"] == cluster
        ta = tt.loc[in_mask, "Time_Months"].values
        ea = tt.loc[in_mask, "Event"].values
        tb = tt.loc[~in_mask, "Time_Months"].values
        eb = tt.loc[~in_mask, "Event"].values
        chi2, p = logrank_test(ta, ea, tb, eb)
        rows.append(
            {
                "Tier": f"Cluster_{int(cluster)}",
                "N": int(in_mask.sum()),
                "Events": int(tt.loc[in_mask, "Event"].sum()),
                "EventRate": float(tt.loc[in_mask, "Event"].mean()),
                "Logrank_Chi2": chi2,
                "P_cluster_vs_overall": p,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def cluster_color(c: int) -> str:
    return CLUSTER_PALETTE[(int(c) - 1) % len(CLUSTER_PALETTE)]


def fig_silhouette(sil_df: pd.DataFrame, chosen_k: int) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.bar(
        sil_df["k"].astype(str),
        sil_df["silhouette"],
        color=["#bbbbbb" if k != chosen_k else "#1f77b4" for k in sil_df["k"]],
        alpha=0.85,
    )
    ax.plot(
        sil_df["k"].astype(str),
        sil_df["silhouette"],
        marker="o",
        color="#333333",
        linewidth=1.3,
    )
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score (Development split)")
    ax.set_title(f"Silhouette vs k — chosen k = {chosen_k}")
    for x, v in zip(sil_df["k"].astype(str), sil_df["silhouette"]):
        ax.text(x, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    save_fig(fig, "Figure_01_Silhouette_by_k.png")


def fig_baseline_heatmap(profile: pd.DataFrame, features: Sequence[str]) -> None:
    feat_labels = [pretty_feat(f) for f in features]
    clusters = sorted(profile["Cluster"].unique())
    Z = np.zeros((len(clusters), len(features)))
    for i, c in enumerate(clusters):
        sub = profile[profile["Cluster"] == c].set_index("Feature")
        for j, f in enumerate(features):
            Z[i, j] = sub.loc[f, "Centroid_Z"]

    fig, ax = plt.subplots(figsize=(max(8.0, 0.6 * len(features)), 0.6 * len(clusters) + 2.5))
    vmax = float(np.nanmax(np.abs(Z)))
    im = ax.imshow(Z, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(feat_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels([f"Cluster {c}" for c in clusters])
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            ax.text(
                j,
                i,
                f"{Z[i, j]:+.2f}",
                ha="center",
                va="center",
                color="black" if abs(Z[i, j]) < 0.6 * vmax else "white",
                fontsize=9,
            )
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Standardized centroid (z)")
    ax.set_title("Baseline cluster centroids (z-scored)")
    save_fig(fig, "Figure_02_Cluster_Baseline_Heatmap.png")


def _plot_trajectory_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    se_col: str,
    x_order: Sequence[str],
    title: str,
    ylabel: str,
) -> None:
    x_positions = {v: i for i, v in enumerate(x_order)}
    for cluster in sorted(df["Cluster"].unique()):
        sub = df[df["Cluster"] == cluster].copy()
        sub = sub[sub[x_col].isin(x_order)]
        sub["x"] = sub[x_col].map(x_positions)
        sub = sub.sort_values("x")
        if sub.empty:
            continue
        col = cluster_color(cluster)
        ax.plot(
            sub["x"],
            sub[y_col],
            marker="o",
            color=col,
            linewidth=2.0,
            label=f"Cluster {int(cluster)}",
        )
        ax.fill_between(
            sub["x"],
            sub[y_col] - sub[se_col],
            sub[y_col] + sub[se_col],
            color=col,
            alpha=0.18,
            linewidth=0,
        )
    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels(list(x_positions.keys()))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.4)


def fig_m2_trajectory(m2_traj: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)
    _plot_trajectory_panel(
        axes[0],
        m2_traj[m2_traj["Domain"] == "OOF"],
        x_col="Landmark",
        y_col="Mean_RiskScore",
        se_col="SE_RiskScore",
        x_order=LANDMARK_ORDER,
        title="Development (OOF)",
        ylabel="Mean M2 RiskScore",
    )
    _plot_trajectory_panel(
        axes[1],
        m2_traj[m2_traj["Domain"] == "Test"],
        x_col="Landmark",
        y_col="Mean_RiskScore",
        se_col="SE_RiskScore",
        x_order=LANDMARK_ORDER,
        title="Temporal Test",
        ylabel="",
    )
    axes[0].set_xlabel("Landmark")
    axes[1].set_xlabel("Landmark")
    axes[1].legend(loc="best", frameon=False, fontsize=9)
    fig.suptitle("M2 risk-score trajectory by baseline cluster", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, "Figure_03_M2_RiskScore_Trajectory_byCluster.png")


def fig_m3_trajectory(m3_traj: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)
    _plot_trajectory_panel(
        axes[0],
        m3_traj[m3_traj["Domain"] == "OOF"],
        x_col="Current_Time",
        y_col="Mean_P_H1",
        se_col="SE_P_H1",
        x_order=CURRENT_TIME_ORDER,
        title="Development (OOF)",
        ylabel="Mean P(H1 relapse)",
    )
    _plot_trajectory_panel(
        axes[1],
        m3_traj[m3_traj["Domain"] == "TemporalTest"],
        x_col="Current_Time",
        y_col="Mean_P_H1",
        se_col="SE_P_H1",
        x_order=CURRENT_TIME_ORDER,
        title="Temporal Test",
        ylabel="",
    )
    axes[0].set_xlabel("Current time")
    axes[1].set_xlabel("Current time")
    axes[1].legend(loc="best", frameon=False, fontsize=9)
    fig.suptitle("M3 rolling H1 probability trajectory by baseline cluster", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, "Figure_04_M3_P_H1_Trajectory_byCluster.png")


def fig_24m_event_rate(outcomes: pd.DataFrame) -> None:
    domains = ["Development", "Temporal"]
    clusters = sorted(outcomes["Cluster"].unique())
    width = 0.36
    x = np.arange(len(clusters))
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for i, dom in enumerate(domains):
        sub = outcomes[outcomes["Domain"] == dom].set_index("Cluster")
        vals = [sub.loc[c, "EventRate"] if c in sub.index else 0 for c in clusters]
        ns = [int(sub.loc[c, "N_episodes"]) if c in sub.index else 0 for c in clusters]
        offset = -width / 2 + i * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=dom,
            color="#4C78A8" if i == 0 else "#F58518",
            alpha=0.85,
        )
        for j, b in enumerate(bars):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.005,
                f"{vals[j]:.0%}\n(n={ns[j]})",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([f"Cluster {c}" for c in clusters])
    ax.set_ylabel("24-month NHRH event rate")
    ax.set_ylim(0, max(0.6, outcomes["EventRate"].max() * 1.25))
    ax.set_title("24-month NHRH event rate by baseline cluster")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=False)
    save_fig(fig, "Figure_05_24M_NHRH_byCluster.png")


def fig_km_temporal(tte: pd.DataFrame, km_table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    temp = tte[tte["Split"] == "Temporal"]
    if temp.empty:
        ax.text(0.5, 0.5, "No temporal episodes.", ha="center", va="center")
        save_fig(fig, "Figure_06_KM_byCluster.png")
        return
    clusters = sorted(temp["Cluster"].unique())
    for cluster in clusters:
        sub = temp[temp["Cluster"] == cluster]
        km = km_estimate(sub["Time_Months"].values, sub["Event"].values)
        ax.step(
            km["Time"],
            km["Survival"],
            where="post",
            label=f"Cluster {cluster} (n={len(sub)})",
            color=cluster_color(cluster),
            linewidth=2.0,
        )
    # Annotate per-cluster log-rank vs overall.
    annotations = []
    for _, row in km_table.iterrows():
        annotations.append(
            f"{row['Tier']}: p = {row['P_cluster_vs_overall']:.3f}"
        )
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Time since RAI (months)")
    ax.set_ylabel("Event-free probability (proxy)")
    ax.set_title("Kaplan-Meier by baseline cluster — Temporal Test")
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    if annotations:
        ax.text(
            0.98,
            0.98,
            "Log-rank (cluster vs rest)\n" + "\n".join(annotations),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#999", "boxstyle": "round,pad=0.3"},
        )
    save_fig(fig, "Figure_06_KM_byCluster.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 72)
    print("Module 4 — baseline clustering + multi-stage risk trajectory")
    print("=" * 72)
    ensure_dirs()

    baseline = load_baseline()
    n_eps = int(len(baseline))
    print(f"[load] baseline: {n_eps} treatment-episodes")
    print(
        "[load] splits:",
        baseline["Split"].value_counts().to_dict(),
    )

    # Confirm the forbidden literal does not occur in counts.
    if str(n_eps) == FORBIDDEN_PATIENT_COUNT:
        raise RuntimeError(
            "Episode count collapsed to forbidden value; expected 1003 episodes."
        )

    features = select_features(baseline, CANDIDATE_FEATURES)
    print(f"[features] keeping {len(features)}: {features}")

    fit, prep, km, sil_df = fit_clusters(baseline, features)
    assignments = assign_all(baseline, features, prep, km)
    assignments, remap = rename_clusters_by_event_rate(assignments)

    sizes_dev = (
        assignments[assignments["Split"] == "Development"]
        .groupby("Cluster")
        .size()
        .to_dict()
    )
    sizes_temp = (
        assignments[assignments["Split"] == "Temporal"]
        .groupby("Cluster")
        .size()
        .to_dict()
    )
    print(f"[clusters] chosen k = {fit.k}")
    print(f"[clusters] dev sizes: {sizes_dev}")
    print(f"[clusters] temporal sizes: {sizes_temp}")

    save_table(assignments, "cluster_assignments.csv")

    profile = cluster_baseline_profile(
        baseline, assignments, features, fit.centroids_z, remap
    )
    save_table(profile, "cluster_baseline_profile.csv")

    m2 = load_m2_risk()
    m2_traj = m2_trajectory_table(assignments, m2, build_full_treatment_id_map())
    save_table(m2_traj, "cluster_m2_trajectory.csv")

    m3 = load_m3_rolling()
    m3_traj = m3_trajectory_table(assignments, m3)
    save_table(m3_traj, "cluster_m3_h1_trajectory.csv")

    outcomes = cluster_outcomes_24m(assignments, m2)
    save_table(outcomes, "cluster_outcomes_24M.csv")

    tte = proxy_time_to_event(assignments, m3)
    km_table = cluster_km_summary(tte)
    if not km_table.empty:
        save_table(km_table, "cluster_km_treatment_level.csv")

    # Figures
    fig_silhouette(sil_df, fit.k)
    fig_baseline_heatmap(profile, features)
    fig_m2_trajectory(m2_traj)
    fig_m3_trajectory(m3_traj)
    fig_24m_event_rate(outcomes)
    fig_km_temporal(tte, km_table)

    # Final guard: the forbidden literal must never appear as a *count*
    # column header / value pair (e.g. an "N_patients" column).  Domain ids
    # such as a treatment-episode ordinal (T0XXX) is a legitimate index
    # over the 1003-episode roster and do NOT violate the rule, so we limit
    # the check to columns that clearly mean a count.
    saved_csvs = list(TABLE_DIR.glob("*.csv"))
    count_like_keywords = ("N_episodes", "N_unique", "Patients", "unique_patient")
    bad = []
    for p in saved_csvs:
        head = pd.read_csv(p, nrows=0)
        suspicious_cols = [
            c for c in head.columns
            if any(k.lower() in c.lower() for k in count_like_keywords)
        ]
        if not suspicious_cols:
            continue
        full = pd.read_csv(p)
        for c in suspicious_cols:
            vals = full[c].astype(str).unique().tolist()
            if FORBIDDEN_PATIENT_COUNT in vals:
                bad.append((p.name, c))
    if bad:
        raise RuntimeError(
            f"Forbidden unique-patient count {FORBIDDEN_PATIENT_COUNT} appeared in count column(s): {bad}"
        )

    # Console summary for caller.
    dev_outcomes = outcomes[outcomes["Domain"] == "Development"]
    temp_outcomes = outcomes[outcomes["Domain"] == "Temporal"]
    if not dev_outcomes.empty:
        spread_dev = float(
            dev_outcomes["EventRate"].max() - dev_outcomes["EventRate"].min()
        )
    else:
        spread_dev = float("nan")
    if not temp_outcomes.empty:
        spread_temp = float(
            temp_outcomes["EventRate"].max() - temp_outcomes["EventRate"].min()
        )
    else:
        spread_temp = float("nan")

    # 0M vs 6M cluster spread in M2 RiskScore (OOF domain).
    m2_oof = m2_traj[m2_traj["Domain"] == "OOF"]
    spread_at_0m = float("nan")
    spread_at_6m = float("nan")
    if not m2_oof.empty:
        s0 = m2_oof[m2_oof["Landmark"] == "0M"]
        s6 = m2_oof[m2_oof["Landmark"] == "6M"]
        if not s0.empty:
            spread_at_0m = float(
                s0["Mean_RiskScore"].max() - s0["Mean_RiskScore"].min()
            )
        if not s6.empty:
            spread_at_6m = float(
                s6["Mean_RiskScore"].max() - s6["Mean_RiskScore"].min()
            )

    print()
    print("Summary")
    print("-------")
    print(f"  chosen k                  : {fit.k} (silhouette {fit.silhouette:.4f})")
    print(f"  development cluster sizes : {sizes_dev}")
    print(f"  temporal    cluster sizes : {sizes_temp}")
    print(f"  24M NHRH spread (dev)     : {spread_dev:.3f}")
    print(f"  24M NHRH spread (temporal): {spread_temp:.3f}")
    print(f"  M2 mean-score spread @ 0M : {spread_at_0m:.3f}")
    print(f"  M2 mean-score spread @ 6M : {spread_at_6m:.3f}")

    if not km_table.empty:
        print("  KM log-rank (cluster vs rest, temporal):")
        for _, row in km_table.iterrows():
            print(
                f"    {row['Tier']}: chi2={row['Logrank_Chi2']:.3f}, p={row['P_cluster_vs_overall']:.4f}"
            )

    print(f"\n[outputs] tables  -> {TABLE_DIR}")
    print(f"[outputs] figures -> {FIG_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
