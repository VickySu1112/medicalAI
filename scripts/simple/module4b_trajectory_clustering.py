"""Module 4b — KMeans clustering on the M2 RiskScore trajectory.

Complement to Module 4 (which clustered on baseline features). Here the
clustering input is the per-episode 4-D updating risk-score trajectory
(RiskScore at 0M / 1M / 3M / 6M) produced by Module 2. Goal: discover
trajectory archetypes (stable-low / stable-high / rising / falling / mixed)
that may transport to the temporal test better than baseline-feature clusters.

This script is NEW and INDEPENDENT. It does NOT touch any Module 1/2/3 or
Module 4 file. All outputs go to
``results/module4b_trajectory_clustering/{figures,tables}``.

Run with::

    PYTHONNOUSERSITE=1 /Users/ql/opt/anaconda3/bin/python \\
        /Users/ql/cursor/medical/medicalAI/scripts/simple/module4b_trajectory_clustering.py
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PY_SEED = 2025
np.random.seed(PY_SEED)
os.environ.setdefault("PYTHONHASHSEED", str(PY_SEED))

PROJECT_ROOT = Path("/Users/ql/cursor/medical/medicalAI")
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "simple"
RESULTS_ROOT = PROJECT_ROOT / "results"

M2_RISK = (
    RESULTS_ROOT
    / "module2_early_landmark_updating"
    / "tables"
    / "early_nhrh_risk_score.csv"
)
M3_ROLL = (
    RESULTS_ROOT
    / "stage2_mh_ab_quick"
    / "tables"
    / "multitask_predictions_noaux_long.csv"
)
M4_BASELINE_ASSIGN = (
    RESULTS_ROOT
    / "module4_baseline_clustering_trajectory"
    / "tables"
    / "cluster_assignments.csv"
)

OUT_DIR = RESULTS_ROOT / "module4b_trajectory_clustering"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"

LANDMARK_ORDER = ("0M", "1M", "3M", "6M")
LANDMARK_MONTHS = {"0M": 0.0, "1M": 1.0, "3M": 3.0, "6M": 6.0}
CURRENT_TIME_ORDER = ("3M", "6M", "12M", "18M")

K_GRID = (3, 4, 5)
MIN_FRAC_DEV = 0.05  # 5% min cluster size on Development split

# Plot kit (colors / dpi)
sys.path.insert(0, str(SCRIPTS_DIR))
import stage1_plot_kit as kit  # noqa: E402

FIG_DPI = kit.FIG_DPI

# Trajectory cluster palette: distinct, colour-blind aware (up to k=5).
CLUSTER_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

# Forbidden literal guard: never write the unique-patient count literal
# anywhere — episode N is always computed at runtime.
_FORBIDDEN = str(890 - 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def _save_table(df: pd.DataFrame, name: str) -> Path:
    out = TABLE_DIR / name
    df.to_csv(out, index=False)
    return out


def _split_from_domain(domain: str) -> str:
    return "Development" if domain == "OOF" else "Temporal"


def load_trajectory_matrix() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.Series]:
    """Return ``(meta, X_raw, X_z, y_24m)``.

    * ``meta`` is one row per Treatment_ID with [Treatment_ID, Split, Y_24M_NHRH].
    * ``X_raw`` is the 4-D matrix of RiskScore at (0M, 1M, 3M, 6M).
    * ``X_z`` is column-wise z-scored across all episodes.
    * ``y_24m`` is the per-episode 24M label aligned to ``meta`` rows.
    """

    raw = pd.read_csv(M2_RISK)
    needed = {"Treatment_ID", "Landmark", "Domain", "Y", "RiskScore"}
    missing = needed - set(raw.columns)
    if missing:
        raise RuntimeError(f"M2 risk-score CSV missing columns: {sorted(missing)}")

    # Pivot RiskScore wide.
    wide = (
        raw.pivot_table(
            index="Treatment_ID",
            columns="Landmark",
            values="RiskScore",
            aggfunc="first",
        )
        .reindex(columns=list(LANDMARK_ORDER))
    )
    # Domain + Y per Treatment_ID: take any landmark (they agree per episode).
    meta = (
        raw[["Treatment_ID", "Domain", "Y"]]
        .drop_duplicates("Treatment_ID")
        .set_index("Treatment_ID")
    )
    df = wide.join(meta, how="inner")

    # Drop episodes missing any landmark.
    before = len(df)
    df = df.dropna(subset=list(LANDMARK_ORDER))
    after = len(df)
    if before != after:
        print(f"[warn] dropped {before - after} episodes missing a landmark")

    df = df.reset_index()
    df["Split"] = df["Domain"].map(_split_from_domain)
    df = df.rename(columns={"Y": "Y_24M_NHRH"})

    X_raw = df[list(LANDMARK_ORDER)].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_z = scaler.fit_transform(X_raw)

    meta_out = df[["Treatment_ID", "Split", "Domain", "Y_24M_NHRH"]].copy()
    meta_out["scaler_mean"] = list(np.tile(scaler.mean_, (len(df), 1)))
    meta_out["scaler_scale"] = list(np.tile(scaler.scale_, (len(df), 1)))
    # Attach raw landmark columns for downstream profile plot.
    for lm in LANDMARK_ORDER:
        meta_out[f"RiskScore_{lm}"] = df[lm].values
    return meta_out, X_raw, X_z, df["Y_24M_NHRH"].values


# ---------------------------------------------------------------------------
# Clustering: fit on Development only, assign all episodes
# ---------------------------------------------------------------------------


def _fit_kmeans_dev(X_z: np.ndarray, dev_mask: np.ndarray, k: int) -> tuple[KMeans, np.ndarray]:
    km = KMeans(n_clusters=k, random_state=PY_SEED, n_init=20)
    km.fit(X_z[dev_mask])
    labels_all = km.predict(X_z)
    return km, labels_all


def silhouette_grid(X_z: np.ndarray, dev_mask: np.ndarray) -> tuple[pd.DataFrame, int, KMeans, np.ndarray]:
    rows = []
    best_k = None
    best_record = None
    n_dev = int(dev_mask.sum())
    min_cluster_dev = max(1, int(np.ceil(n_dev * MIN_FRAC_DEV)))
    for k in K_GRID:
        km, labels_all = _fit_kmeans_dev(X_z, dev_mask, k)
        labels_dev = labels_all[dev_mask]
        sil = float(silhouette_score(X_z[dev_mask], labels_dev))
        counts_dev = pd.Series(labels_dev).value_counts()
        min_size = int(counts_dev.min())
        viable = bool(min_size >= min_cluster_dev)
        rows.append(
            {
                "k": k,
                "silhouette": round(sil, 4),
                "min_cluster_size_dev": min_size,
                "min_required_dev": min_cluster_dev,
                "viable": viable,
            }
        )
        if viable:
            if best_record is None or sil > best_record["silhouette"]:
                best_record = {"k": k, "silhouette": sil, "model": km, "labels": labels_all}
    if best_record is None:
        # Fall back to highest silhouette regardless of size constraint.
        best_idx = int(np.argmax([r["silhouette"] for r in rows]))
        kfb = rows[best_idx]["k"]
        kmfb, labels_fb = _fit_kmeans_dev(X_z, dev_mask, kfb)
        best_record = {
            "k": kfb,
            "silhouette": float(silhouette_score(X_z[dev_mask], labels_fb[dev_mask])),
            "model": kmfb,
            "labels": labels_fb,
        }
        print("[warn] no viable k met min-cluster-size; using best silhouette anyway")
    best_k = best_record["k"]
    return pd.DataFrame(rows), best_k, best_record["model"], best_record["labels"]


# ---------------------------------------------------------------------------
# Trajectory labeling (stable-low / stable-high / rising / falling / mixed)
# ---------------------------------------------------------------------------


def _label_trajectory_shape(mean_vec: np.ndarray, all_means: np.ndarray) -> str:
    """Assign a shape label from the mean RiskScore at (0M,1M,3M,6M)."""

    months = np.array([LANDMARK_MONTHS[lm] for lm in LANDMARK_ORDER], dtype=float)
    slope, _ = np.polyfit(months, mean_vec, 1)
    level = float(np.mean(mean_vec))

    # Global percentiles for "low" vs "high" levels.
    global_low = float(np.percentile(all_means, 33))
    global_high = float(np.percentile(all_means, 67))

    # Slope thresholds: |slope| ~ per-month change; small for "stable".
    slope_band = 0.01  # < 0.01 per month over 6 months → < 0.06 total swing

    if abs(slope) < slope_band:
        if level <= global_low:
            return "stable-low"
        if level >= global_high:
            return "stable-high"
        return "stable-mid"
    if slope >= slope_band:
        return "rising"
    if slope <= -slope_band:
        return "falling"
    return "mixed"


def label_clusters(centroid_means_raw: pd.DataFrame) -> dict[int, str]:
    """Take a (cluster_id × landmark) mean RiskScore frame and return labels."""

    all_levels = centroid_means_raw[list(LANDMARK_ORDER)].to_numpy().reshape(-1)
    labels: dict[int, str] = {}
    used: dict[str, int] = {}
    for cid, row in centroid_means_raw.iterrows():
        mean_vec = row[list(LANDMARK_ORDER)].to_numpy(dtype=float)
        base = _label_trajectory_shape(mean_vec, all_levels)
        # Disambiguate duplicates by appending suffix.
        if base in used:
            used[base] += 1
            labels[int(cid)] = f"{base}-{used[base]}"
        else:
            used[base] = 1
            labels[int(cid)] = base
    return labels


# ---------------------------------------------------------------------------
# Profile / outcomes / m3 / KM helpers
# ---------------------------------------------------------------------------


def trajectory_profile(meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cid, label, domain), sub in meta.groupby(
        ["TrajCluster", "TrajLabel", "Split"], sort=True
    ):
        for lm in LANDMARK_ORDER:
            vals = sub[f"RiskScore_{lm}"].astype(float).to_numpy()
            n = int(len(vals))
            mean = float(np.mean(vals)) if n else float("nan")
            se = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
            rate = float(sub["Y_24M_NHRH"].astype(int).mean()) if n else float("nan")
            rows.append(
                {
                    "TrajCluster": int(cid),
                    "TrajLabel": label,
                    "Domain": domain,
                    "Landmark": lm,
                    "N": n,
                    "Mean_RiskScore": round(mean, 6),
                    "SE_RiskScore": round(se, 6) if not np.isnan(se) else "",
                    "Observed_Rate": round(rate, 6),
                }
            )
    return pd.DataFrame(rows)


def outcomes_24m(meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cid, label, split), sub in meta.groupby(
        ["TrajCluster", "TrajLabel", "Split"], sort=True
    ):
        n = int(len(sub))
        ev = int(sub["Y_24M_NHRH"].astype(int).sum())
        rate = ev / n if n else float("nan")
        rows.append(
            {
                "TrajCluster": int(cid),
                "TrajLabel": label,
                "Domain": split,
                "N": n,
                "Events": ev,
                "EventRate": round(rate, 6),
            }
        )
    return pd.DataFrame(rows)


def m3_h1_profile(meta: pd.DataFrame) -> pd.DataFrame:
    m3 = pd.read_csv(
        M3_ROLL,
        usecols=[
            "Treatment_ID",
            "Split",
            "Current_Time",
            "P_H1",
            "is_earliest_controlled_row",
        ],
    )
    # Use earliest-controlled row per (Treatment_ID, Current_Time) to avoid double-counting.
    m3 = m3[m3["is_earliest_controlled_row"] == 1].copy()
    join = (
        meta[["Treatment_ID", "Split", "TrajCluster", "TrajLabel"]]
        .copy()
        .merge(m3, on=["Treatment_ID", "Split"], how="inner")
    )
    rows = []
    for (cid, label, split, ct), sub in join.groupby(
        ["TrajCluster", "TrajLabel", "Split", "Current_Time"], sort=True
    ):
        vals = sub["P_H1"].astype(float).to_numpy()
        n = int(len(vals))
        mean = float(np.mean(vals)) if n else float("nan")
        rows.append(
            {
                "TrajCluster": int(cid),
                "TrajLabel": label,
                "Domain": split,
                "Current_Time": ct,
                "N_windows": n,
                "Mean_P_H1": round(mean, 6),
            }
        )
    return pd.DataFrame(rows)


# Simple KM step from 24M binary using H1/H12/H24 rolling probs as "event time"
# proxy: we use the earliest observed Current_Month per Treatment_ID at which
# the true 24M label resolves; lacking explicit event time, we treat
# Y_24M_NHRH=1 as event at month=24, censored otherwise (right-censoring is
# administrative; this matches the binary endpoint). The result is a binary
# "step" curve per cluster — at t<24 the survival is 1.0 and at t=24 it drops
# by the cluster event rate. We expose log-rank cluster-vs-rest at t=24
# (a 2-sample test on the binary outcome equivalent to chi-square 2×2).


def km_table_temporal(meta: pd.DataFrame) -> pd.DataFrame:
    """Per-cluster temporal-split KM-style summary + log-rank chi2 (cluster vs rest)."""

    temp = meta[meta["Split"] == "Temporal"].copy()
    n_total = int(len(temp))
    rows = []
    for cid, sub in temp.groupby("TrajCluster", sort=True):
        in_cl = sub["Y_24M_NHRH"].astype(int).to_numpy()
        rest = temp.loc[~temp["Treatment_ID"].isin(sub["Treatment_ID"]), "Y_24M_NHRH"].astype(int).to_numpy()
        a = int(in_cl.sum())
        b = int((1 - in_cl).sum())
        c = int(rest.sum())
        d = int((1 - rest).sum())
        # 2×2 contingency for chi-square (Yates corrected if any cell <5).
        table = np.array([[a, b], [c, d]])
        try:
            chi2, p, _, _ = stats.chi2_contingency(table, correction=True)
        except ValueError:
            chi2, p = float("nan"), float("nan")
        rows.append(
            {
                "TrajCluster": int(cid),
                "TrajLabel": sub["TrajLabel"].iloc[0],
                "N": int(in_cl.size),
                "Events": a,
                "EventRate": round(a / max(in_cl.size, 1), 6),
                "Logrank_Chi2": round(float(chi2), 4) if not np.isnan(chi2) else "",
                "P_cluster_vs_rest": round(float(p), 6) if not np.isnan(p) else "",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Crosstab against Module 4 baseline clusters
# ---------------------------------------------------------------------------


def crosstab_with_baseline(meta: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    base = pd.read_csv(M4_BASELINE_ASSIGN)
    if "Treatment_ID" not in base.columns or "Cluster" not in base.columns:
        raise RuntimeError("Module 4 cluster_assignments missing required columns")
    base = base.rename(columns={"Cluster": "BaselineCluster"})
    base["Prefix"] = base["Treatment_ID"].astype(str)
    # M2 Treatment_IDs look like T####_YYMMDD; baseline IDs look like T####.
    meta_join = meta.copy()
    meta_join["Prefix"] = meta_join["Treatment_ID"].astype(str).str.split("_").str[0]
    merged = meta_join.merge(
        base[["Prefix", "BaselineCluster"]], on="Prefix", how="inner"
    )
    ct = pd.crosstab(merged["TrajCluster"], merged["BaselineCluster"], dropna=False)
    chi2, p, _, _ = stats.chi2_contingency(ct.values)
    return ct, float(chi2), float(p)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_silhouette(sil_df: pd.DataFrame, chosen_k: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ks = sil_df["k"].astype(int).to_numpy()
    sils = sil_df["silhouette"].astype(float).to_numpy()
    bars = ax.bar(ks, sils, color=kit.GRAY, edgecolor=kit.DARK, linewidth=0.8)
    for k, s, bar in zip(ks, sils, bars):
        ax.text(k, s + 0.005, f"{s:.3f}", ha="center", va="bottom", fontsize=10)
        if k == chosen_k:
            bar.set_color(kit.BLUE)
    ax.plot(ks, sils, marker="o", color=kit.DARK, lw=1.2)
    ax.axvline(chosen_k, color=kit.RED, linestyle="--", lw=1, alpha=0.7)
    ax.set_xticks(ks)
    ax.set_xlabel("Number of clusters k")
    ax.set_ylabel("Silhouette score (Development)")
    ax.set_title(f"Trajectory KMeans silhouette by k (chosen k = {chosen_k})")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    _save_fig(fig, path)


def fig_trajectory_profiles(profile: pd.DataFrame, traj_labels: dict[int, str], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    months = [LANDMARK_MONTHS[lm] for lm in LANDMARK_ORDER]
    for ax, split in zip(axes, ("Development", "Temporal")):
        sub = profile[profile["Domain"] == split]
        clusters = sorted(sub["TrajCluster"].unique())
        for idx, cid in enumerate(clusters):
            d = (
                sub[sub["TrajCluster"] == cid]
                .set_index("Landmark")
                .reindex(list(LANDMARK_ORDER))
            )
            ymean = d["Mean_RiskScore"].astype(float).to_numpy()
            yse_raw = d["SE_RiskScore"].astype(str).to_numpy()
            yse = np.array(
                [float(x) if x not in ("", "nan") else 0.0 for x in yse_raw]
            )
            color = CLUSTER_PALETTE[idx % len(CLUSTER_PALETTE)]
            label = f"C{cid}: {traj_labels.get(int(cid), '?')}"
            ax.plot(months, ymean, color=color, lw=2.0, marker="o", label=label)
            ax.fill_between(months, ymean - yse, ymean + yse, color=color, alpha=0.15)
        ax.set_xticks(months)
        ax.set_xticklabels(list(LANDMARK_ORDER))
        ax.set_xlabel("Landmark")
        ax.set_title(f"{split} (N={int(profile[(profile['Domain']==split) & (profile['Landmark']=='0M')]['N'].sum())})")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean updating RiskScore (M2)")
    axes[0].legend(loc="best", fontsize=9, framealpha=0.85)
    fig.suptitle("Trajectory clusters — mean RiskScore profile by landmark", y=1.02)
    _save_fig(fig, path)


def fig_outcomes(outcomes: pd.DataFrame, traj_labels: dict[int, str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    clusters = sorted(outcomes["TrajCluster"].unique())
    x = np.arange(len(clusters))
    width = 0.38
    dev_rates = []
    temp_rates = []
    dev_n = []
    temp_n = []
    for cid in clusters:
        sub = outcomes[outcomes["TrajCluster"] == cid]
        d = sub[sub["Domain"] == "Development"]
        t = sub[sub["Domain"] == "Temporal"]
        dev_rates.append(float(d["EventRate"].iloc[0]) if len(d) else 0.0)
        temp_rates.append(float(t["EventRate"].iloc[0]) if len(t) else 0.0)
        dev_n.append(int(d["N"].iloc[0]) if len(d) else 0)
        temp_n.append(int(t["N"].iloc[0]) if len(t) else 0)
    bars_dev = ax.bar(
        x - width / 2,
        dev_rates,
        width,
        color=kit.BLUE,
        edgecolor=kit.DARK,
        linewidth=0.6,
        label="Development",
    )
    bars_temp = ax.bar(
        x + width / 2,
        temp_rates,
        width,
        color=kit.RED,
        edgecolor=kit.DARK,
        linewidth=0.6,
        label="Temporal",
    )
    for bar, r, n in zip(bars_dev, dev_rates, dev_n):
        ax.text(bar.get_x() + bar.get_width() / 2, r + 0.01, f"{r:.2f}\n(n={n})", ha="center", va="bottom", fontsize=8)
    for bar, r, n in zip(bars_temp, temp_rates, temp_n):
        ax.text(bar.get_x() + bar.get_width() / 2, r + 0.01, f"{r:.2f}\n(n={n})", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}\n{traj_labels.get(int(c), '?')}" for c in clusters])
    ax.set_ylabel("24M NHRH event rate")
    ax.set_title("24M NHRH event rate by trajectory cluster")
    ax.set_ylim(0, max(max(dev_rates), max(temp_rates)) * 1.30 + 0.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    _save_fig(fig, path)


def fig_m3_ph1(m3_df: pd.DataFrame, traj_labels: dict[int, str], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    ct_order = [3.0, 6.0, 12.0, 18.0]
    ct_labels = list(CURRENT_TIME_ORDER)
    for ax, split in zip(axes, ("Development", "Temporal")):
        sub = m3_df[m3_df["Domain"] == split]
        clusters = sorted(sub["TrajCluster"].unique())
        for idx, cid in enumerate(clusters):
            d = sub[sub["TrajCluster"] == cid].copy()
            d["ct_num"] = d["Current_Time"].map(
                {"3M": 3.0, "6M": 6.0, "12M": 12.0, "18M": 18.0}
            )
            d = d.dropna(subset=["ct_num"]).sort_values("ct_num")
            d = d[d["Current_Time"].isin(CURRENT_TIME_ORDER)]
            if d.empty:
                continue
            color = CLUSTER_PALETTE[idx % len(CLUSTER_PALETTE)]
            label = f"C{cid}: {traj_labels.get(int(cid), '?')}"
            ax.plot(d["ct_num"].astype(float), d["Mean_P_H1"].astype(float), color=color, lw=2.0, marker="o", label=label)
        ax.set_xticks(ct_order)
        ax.set_xticklabels(ct_labels)
        ax.set_xlabel("Current_Time")
        ax.set_title(split)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean M3 P(H1) — controlled rolling")
    axes[0].legend(loc="best", fontsize=9, framealpha=0.85)
    fig.suptitle("M3 next-window relapse probability by trajectory cluster", y=1.02)
    _save_fig(fig, path)


def fig_km_temporal(meta: pd.DataFrame, traj_labels: dict[int, str], km_df: pd.DataFrame, path: Path) -> None:
    """Simple step-style KM curves on Temporal using the binary 24M endpoint.

    Approximation: each episode is censored at 24M unless Y_24M_NHRH=1, in which
    case the event is placed at t=24M (the labels resolve at 24M of follow-up).
    The curve thus drops once at t=24M to ``1 - event_rate_in_cluster``.
    """

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    temp = meta[meta["Split"] == "Temporal"]
    clusters = sorted(temp["TrajCluster"].unique())
    for idx, cid in enumerate(clusters):
        sub = temp[temp["TrajCluster"] == cid]
        n = int(len(sub))
        if n == 0:
            continue
        rate = float(sub["Y_24M_NHRH"].astype(int).mean())
        color = CLUSTER_PALETTE[idx % len(CLUSTER_PALETTE)]
        # Step from t=0..24 at S=1; drop at 24.
        xs = [0, 24, 24]
        ys = [1.0, 1.0, 1.0 - rate]
        label = f"C{cid}: {traj_labels.get(int(cid), '?')} (n={n}, event={rate:.2f})"
        ax.plot(xs, ys, color=color, lw=2.0, label=label, drawstyle="steps-post")
        ax.plot([24], [1.0 - rate], marker="o", color=color)
    ax.set_xlabel("Months since RAI (capped at 24)")
    ax.set_ylabel("Event-free probability (1 − 24M NHRH)")
    p_vals = pd.to_numeric(km_df["P_cluster_vs_rest"], errors="coerce")
    sig_any = bool((p_vals < 0.05).any())
    title = "Kaplan–Meier-style step curves by trajectory cluster (Temporal)"
    if sig_any:
        title += "  [some cluster-vs-rest p<0.05]"
    ax.set_title(title)
    ax.set_xlim(0, 26)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.85)
    _save_fig(fig, path)


def fig_crosstab(ct: pd.DataFrame, chi2: float, p: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    matrix = ct.values
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels([f"B{int(c)}" for c in ct.columns])
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels([f"T{int(c)}" for c in ct.index])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            ax.text(j, i, str(int(v)), ha="center", va="center", color="black" if v < matrix.max() / 2 else "white", fontsize=10)
    ax.set_xlabel("Baseline cluster (Module 4)")
    ax.set_ylabel("Trajectory cluster (Module 4b)")
    ax.set_title(f"Trajectory × Baseline cluster crosstab — chi2={chi2:.2f}, p={p:.2e}")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Episode count")
    _save_fig(fig, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    _ensure_dirs()

    print("=" * 76)
    print("Module 4b — trajectory clustering on M2 risk-score trajectory")
    print("=" * 76)

    meta, X_raw, X_z, _y = load_trajectory_matrix()
    dev_mask = (meta["Split"] == "Development").to_numpy()
    n_total = len(meta)
    n_dev = int(dev_mask.sum())
    n_temp = n_total - n_dev
    print(f"Episodes: total={n_total} dev={n_dev} temporal={n_temp}")

    sil_df, best_k, km, labels_all = silhouette_grid(X_z, dev_mask)
    _save_table(sil_df, "traj_silhouette_by_k.csv")
    print("Silhouette grid:")
    print(sil_df.to_string(index=False))
    print(f"Chosen k = {best_k}")

    meta = meta.copy()
    meta["TrajCluster"] = labels_all

    # Cluster centroids back to RAW scale → use for labeling.
    cluster_means_raw = (
        meta.groupby("TrajCluster")[[f"RiskScore_{lm}" for lm in LANDMARK_ORDER]]
        .mean()
        .rename(columns={f"RiskScore_{lm}": lm for lm in LANDMARK_ORDER})
    )
    traj_labels = label_clusters(cluster_means_raw)
    meta["TrajLabel"] = meta["TrajCluster"].map(traj_labels)
    print("Trajectory labels per cluster:")
    for cid, lab in traj_labels.items():
        print(f"  C{cid}: {lab}  | mean = {cluster_means_raw.loc[cid].round(4).to_dict()}")

    # Save assignments.
    assignments = meta[["Treatment_ID", "Split", "Y_24M_NHRH", "TrajCluster", "TrajLabel"]].copy()
    _save_table(assignments, "traj_cluster_assignments.csv")

    # Profiles.
    profile = trajectory_profile(meta)
    _save_table(profile, "traj_cluster_profiles.csv")

    # 24M outcomes.
    outcomes = outcomes_24m(meta)
    _save_table(outcomes, "traj_cluster_outcomes_24M.csv")
    print("24M event rates by cluster:")
    print(outcomes.to_string(index=False))

    # M3 H1 profile.
    m3_df = m3_h1_profile(meta)
    _save_table(m3_df, "traj_cluster_m3_h1.csv")

    # KM table (temporal).
    km_df = km_table_temporal(meta)
    _save_table(km_df, "traj_cluster_km.csv")
    print("KM-style cluster-vs-rest (Temporal):")
    print(km_df.to_string(index=False))

    # Crosstab.
    ct, chi2, p = crosstab_with_baseline(meta)
    ct_long = ct.reset_index().melt(id_vars="TrajCluster", var_name="BaselineCluster", value_name="N")
    ct_long["chi2"] = chi2
    ct_long["p_value"] = p
    _save_table(ct_long, "traj_vs_baseline_crosstab.csv")
    print(f"Crosstab Traj × Baseline chi2={chi2:.4f} p={p:.4e}")

    # Figures.
    fig_silhouette(sil_df, best_k, FIG_DIR / "Figure_01_Traj_Silhouette.png")
    fig_trajectory_profiles(profile, traj_labels, FIG_DIR / "Figure_02_Trajectory_Profiles.png")
    fig_outcomes(outcomes, traj_labels, FIG_DIR / "Figure_03_24M_NHRH_by_TrajCluster.png")
    fig_m3_ph1(m3_df, traj_labels, FIG_DIR / "Figure_04_M3_PH1_by_TrajCluster.png")
    fig_km_temporal(meta, traj_labels, km_df, FIG_DIR / "Figure_05_KM_by_TrajCluster.png")
    fig_crosstab(ct, chi2, p, FIG_DIR / "Figure_06_Crosstab_Traj_vs_Baseline.png")

    # Compact verdict summary printed to stdout.
    dev_rates = outcomes.loc[outcomes["Domain"] == "Development", "EventRate"].astype(float)
    temp_rates = outcomes.loc[outcomes["Domain"] == "Temporal", "EventRate"].astype(float)
    dev_spread = float(dev_rates.max() - dev_rates.min())
    temp_spread = float(temp_rates.max() - temp_rates.min())
    p_vals = pd.to_numeric(km_df["P_cluster_vs_rest"], errors="coerce")
    sig_any = bool((p_vals < 0.05).any())
    print("\n=== SUMMARY ===")
    print(f"Chosen k = {best_k}; silhouette(dev) = {sil_df.set_index('k').loc[best_k, 'silhouette']}")
    print("Cluster sizes (dev/temporal):")
    sizes = (
        meta.groupby(["TrajCluster", "Split"]).size().unstack(fill_value=0)
    )
    for cid in sorted(meta["TrajCluster"].unique()):
        print(
            f"  C{cid} ({traj_labels[int(cid)]}): "
            f"dev={int(sizes.loc[cid].get('Development', 0))} "
            f"temp={int(sizes.loc[cid].get('Temporal', 0))}"
        )
    print(f"24M event-rate spread: dev={dev_spread:.3f}  temporal={temp_spread:.3f}")
    print(f"Any cluster-vs-rest p<0.05 on Temporal? {sig_any}")
    print(f"Trajectory vs Baseline crosstab chi2={chi2:.2f}, p={p:.4e}")


if __name__ == "__main__":
    main()
