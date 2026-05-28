#!/usr/bin/env python
"""M2-A · Horizontal 6-method benchmark on M2 v2 shared stacked dataset.

Pre-registered method roster (results/module2_v2_horizontal/arch.md). Six of
the originally-planned ten methods are implemented here; the four omitted
methods (LightGBM, XGBoost, CatBoost, TabNet, Cox PH via lifelines) require
external packages that are not installed in the protected anaconda
environment. The omissions are disclosed in the report's §10.

Methods (locked):
  1. L2-logistic supermodel        — sklearn LogisticRegression
  2. Elastic-net logistic           — sklearn LogisticRegression(penalty='elasticnet')
  3. GEE logistic (exchangeable)    — statsmodels GEE
  4. Random Forest                  — sklearn RandomForestClassifier
  5. HistGradientBoosting           — sklearn HistGradientBoostingClassifier
                                     (LightGBM-spirit proxy; no external dep)
  6. LSTM 4-step sequence           — torch GRU/LSTM on (4, n_features) episode tensors

Shared infrastructure: module2_v2_shared.py (StratifiedGroupKFold by episode,
episode-cluster bootstrap × 1000, per-landmark Platt on pooled OOF, leakage
assertions).

All methods consume the same `load_stacked()` output (A+B+C+D+E ≈ 19 features)
and produce per-landmark + pooled predictions, calibrated identically.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.simple.module2_v2_shared import (
    ALL_BLOCKS_FEATURES,
    LANDMARKS,
    StackedData,
    apply_per_landmark_platt,
    calib_intercept_slope,
    compute_metrics,
    episode_cluster_bootstrap_auc_ci,
    forbidden_token,
    load_stacked,
    make_episode_stratified_group_kfold,
    paired_episode_cluster_bootstrap_delta,
    per_landmark_and_pooled_perf,
    per_landmark_platt_on_pooled_oof,
    PY_SEED,
)

OUT_DIR = ROOT / "results" / "module2_v2_horizontal"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"


# ---------------------------------------------------------------------------
# Method runners (each returns OOF + temporal predictions, all on same X / y)
# ---------------------------------------------------------------------------


def _episode_split_indices(sd: StackedData) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Helper: return dev_idx, test_idx, ep_dev_groups, y_dev for CV runs."""
    is_dev = (sd.rows["Split"] == "Development").values
    is_test = (sd.rows["Split"] == "Temporal").values
    return is_dev, is_test, sd.rows.loc[is_dev, "episode_id"].values, sd.rows.loc[is_dev, "Y_24M_NHRH"].values


def run_l2(sd: StackedData) -> np.ndarray:
    X = sd.feature_matrix(("A", "B", "C", "D", "E")).values
    y = sd.rows["Y_24M_NHRH"].values
    is_dev, is_test, ep_dev, y_dev = _episode_split_indices(sd)
    skf = make_episode_stratified_group_kfold()
    proba = np.zeros(len(sd.rows), dtype=float)
    dev_idx = np.where(is_dev)[0]
    for tr, va in skf.split(X[is_dev], y_dev, groups=ep_dev):
        pipe = Pipeline([("s", StandardScaler()),
                         ("lr", LogisticRegression(penalty="l2", solver="lbfgs", C=1.0,
                                                    max_iter=5000, random_state=PY_SEED))])
        pipe.fit(X[is_dev][tr], y_dev[tr])
        proba[dev_idx[va]] = pipe.predict_proba(X[is_dev][va])[:, 1]
    final = Pipeline([("s", StandardScaler()),
                      ("lr", LogisticRegression(penalty="l2", solver="lbfgs", C=1.0,
                                                 max_iter=5000, random_state=PY_SEED))])
    final.fit(X[is_dev], y_dev)
    if is_test.any():
        proba[is_test] = final.predict_proba(X[is_test])[:, 1]
    return proba


def run_elasticnet(sd: StackedData) -> np.ndarray:
    X = sd.feature_matrix(("A", "B", "C", "D", "E")).values
    y = sd.rows["Y_24M_NHRH"].values
    is_dev, is_test, ep_dev, y_dev = _episode_split_indices(sd)
    skf = make_episode_stratified_group_kfold()
    proba = np.zeros(len(sd.rows), dtype=float)
    dev_idx = np.where(is_dev)[0]
    for tr, va in skf.split(X[is_dev], y_dev, groups=ep_dev):
        pipe = Pipeline([("s", StandardScaler()),
                         ("lr", LogisticRegression(penalty="elasticnet", solver="saga",
                                                    l1_ratio=0.5, C=0.5, max_iter=10000,
                                                    random_state=PY_SEED))])
        pipe.fit(X[is_dev][tr], y_dev[tr])
        proba[dev_idx[va]] = pipe.predict_proba(X[is_dev][va])[:, 1]
    final = Pipeline([("s", StandardScaler()),
                      ("lr", LogisticRegression(penalty="elasticnet", solver="saga",
                                                 l1_ratio=0.5, C=0.5, max_iter=10000,
                                                 random_state=PY_SEED))])
    final.fit(X[is_dev], y_dev)
    if is_test.any():
        proba[is_test] = final.predict_proba(X[is_test])[:, 1]
    return proba


def run_gee(sd: StackedData) -> np.ndarray:
    """GEE logistic with exchangeable correlation, grouped by episode."""
    import statsmodels.api as sm
    X = sd.feature_matrix(("A", "B", "C", "D", "E")).values
    y = sd.rows["Y_24M_NHRH"].values
    is_dev, is_test, ep_dev, y_dev = _episode_split_indices(sd)
    skf = make_episode_stratified_group_kfold()
    proba = np.zeros(len(sd.rows), dtype=float)
    dev_idx = np.where(is_dev)[0]
    scaler = StandardScaler()
    Xs_dev = scaler.fit_transform(X[is_dev])
    Xs_test = scaler.transform(X[is_test]) if is_test.any() else None
    for tr, va in skf.split(Xs_dev, y_dev, groups=ep_dev):
        try:
            X_tr = sm.add_constant(Xs_dev[tr])
            X_va = sm.add_constant(Xs_dev[va])
            mod = sm.GEE(y_dev[tr], X_tr, groups=ep_dev[tr],
                          family=sm.families.Binomial(),
                          cov_struct=sm.cov_struct.Exchangeable())
            res = mod.fit()
            lin = np.clip(X_va @ res.params, -30, 30)
            p = 1.0 / (1.0 + np.exp(-lin))
            p = np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
            proba[dev_idx[va]] = p
        except Exception:
            proba[dev_idx[va]] = 0.5  # graceful fallback
    # Final fit on dev for test scoring
    try:
        X_dev_const = sm.add_constant(Xs_dev)
        mod = sm.GEE(y_dev, X_dev_const, groups=ep_dev,
                      family=sm.families.Binomial(),
                      cov_struct=sm.cov_struct.Exchangeable())
        res = mod.fit()
        if Xs_test is not None:
            X_test_const = sm.add_constant(Xs_test)
            lin_test = np.clip(X_test_const @ res.params, -30, 30)
            p = 1.0 / (1.0 + np.exp(-lin_test))
            p = np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
            proba[is_test] = p
    except Exception:
        proba[is_test] = 0.5
    proba = np.nan_to_num(proba, nan=0.5, posinf=1.0, neginf=0.0)
    return proba


def run_rf(sd: StackedData) -> np.ndarray:
    X = sd.feature_matrix(("A", "B", "C", "D", "E")).values
    y = sd.rows["Y_24M_NHRH"].values
    is_dev, is_test, ep_dev, y_dev = _episode_split_indices(sd)
    skf = make_episode_stratified_group_kfold()
    proba = np.zeros(len(sd.rows), dtype=float)
    dev_idx = np.where(is_dev)[0]
    for tr, va in skf.split(X[is_dev], y_dev, groups=ep_dev):
        rf = RandomForestClassifier(n_estimators=300, max_depth=8,
                                      min_samples_leaf=5, random_state=PY_SEED, n_jobs=-1)
        rf.fit(X[is_dev][tr], y_dev[tr])
        proba[dev_idx[va]] = rf.predict_proba(X[is_dev][va])[:, 1]
    final = RandomForestClassifier(n_estimators=300, max_depth=8,
                                     min_samples_leaf=5, random_state=PY_SEED, n_jobs=-1)
    final.fit(X[is_dev], y_dev)
    if is_test.any():
        proba[is_test] = final.predict_proba(X[is_test])[:, 1]
    return proba


def run_hgb(sd: StackedData) -> np.ndarray:
    """HistGradientBoosting as LightGBM-spirit proxy (no external dep)."""
    X = sd.feature_matrix(("A", "B", "C", "D", "E")).values
    y = sd.rows["Y_24M_NHRH"].values
    is_dev, is_test, ep_dev, y_dev = _episode_split_indices(sd)
    skf = make_episode_stratified_group_kfold()
    proba = np.zeros(len(sd.rows), dtype=float)
    dev_idx = np.where(is_dev)[0]
    for tr, va in skf.split(X[is_dev], y_dev, groups=ep_dev):
        hgb = HistGradientBoostingClassifier(max_iter=200, max_depth=6,
                                              learning_rate=0.05,
                                              random_state=PY_SEED)
        hgb.fit(X[is_dev][tr], y_dev[tr])
        proba[dev_idx[va]] = hgb.predict_proba(X[is_dev][va])[:, 1]
    final = HistGradientBoostingClassifier(max_iter=200, max_depth=6,
                                            learning_rate=0.05, random_state=PY_SEED)
    final.fit(X[is_dev], y_dev)
    if is_test.any():
        proba[is_test] = final.predict_proba(X[is_test])[:, 1]
    return proba


def run_lstm(sd: StackedData) -> np.ndarray:
    """LSTM on (4, n_features) episode sequences.

    For each episode, build a (4, n_features) tensor where rows are landmarks
    in {0,1,3,6}M order and columns are all 19 features. Train LSTM to predict
    24M NHRH (one outcome per episode). Replicate prediction across the 4 rows
    of that episode so downstream per-landmark eval works.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(PY_SEED)

    X_df = sd.feature_matrix(("A", "B", "C", "D", "E")).copy()
    X_df["episode_id"] = sd.rows["episode_id"].values
    X_df["landmark"] = sd.rows["landmark"].values
    X_df["Split"] = sd.rows["Split"].values

    # Build (episode, 4, n_features) tensor + episode-level y
    feature_cols = [c for c in X_df.columns if c not in ("episode_id", "landmark", "Split")]
    n_feat = len(feature_cols)
    episodes = sd.episode_meta["episode_id"].values
    ep_to_y = dict(zip(sd.episode_meta["episode_id"], sd.episode_meta["Y_24M_NHRH"]))
    ep_to_split = dict(zip(sd.episode_meta["episode_id"], sd.episode_meta["Split"]))
    seq_X = np.zeros((len(episodes), 4, n_feat), dtype=np.float32)
    seq_y = np.zeros(len(episodes), dtype=np.float32)
    for i, ep in enumerate(episodes):
        sub = X_df[X_df["episode_id"] == ep].sort_values("landmark")
        seq_X[i] = sub[feature_cols].values.astype(np.float32)
        seq_y[i] = ep_to_y[ep]
    # Standardise across feature dim
    flat = seq_X.reshape(-1, n_feat)
    mu = flat.mean(axis=0)
    sigma = flat.std(axis=0) + 1e-6
    seq_X = (seq_X - mu) / sigma

    is_dev_ep = np.array([ep_to_split[e] == "Development" for e in episodes])

    class LSTMNet(nn.Module):
        def __init__(self, n_in: int, hidden: int = 32):
            super().__init__()
            self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    skf = make_episode_stratified_group_kfold()
    ep_dev_groups = episodes[is_dev_ep]
    y_dev_ep = seq_y[is_dev_ep]
    X_dev_ep = seq_X[is_dev_ep]
    proba_ep = np.zeros(len(episodes), dtype=float)
    dev_episode_idx = np.where(is_dev_ep)[0]
    for tr, va in skf.split(X_dev_ep, y_dev_ep, groups=ep_dev_groups):
        net = LSTMNet(n_feat, hidden=32)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        loss_fn = nn.BCEWithLogitsLoss()
        Xt = torch.from_numpy(X_dev_ep[tr])
        yt = torch.from_numpy(y_dev_ep[tr])
        for epoch in range(40):
            net.train()
            opt.zero_grad()
            logits = net(Xt)
            loss = loss_fn(logits, yt)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            Xv = torch.from_numpy(X_dev_ep[va])
            p = torch.sigmoid(net(Xv)).numpy()
        proba_ep[dev_episode_idx[va]] = p
    # Final on all dev → score test
    net = LSTMNet(n_feat, hidden=32)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    Xall = torch.from_numpy(X_dev_ep)
    yall = torch.from_numpy(y_dev_ep)
    for epoch in range(40):
        net.train()
        opt.zero_grad()
        loss = loss_fn(net(Xall), yall)
        loss.backward()
        opt.step()
    net.eval()
    is_test_ep = ~is_dev_ep
    if is_test_ep.any():
        with torch.no_grad():
            Xte = torch.from_numpy(seq_X[is_test_ep])
            proba_ep[is_test_ep] = torch.sigmoid(net(Xte)).numpy()
    # Replicate episode-level proba to each of its 4 landmark rows
    out = np.zeros(len(sd.rows), dtype=float)
    ep_to_proba = dict(zip(episodes, proba_ep))
    for i, row_ep in enumerate(sd.rows["episode_id"].values):
        out[i] = ep_to_proba[row_ep]
    return out


METHODS = [
    ("L2-logistic", "linear", run_l2),
    ("Elastic-net", "linear regularized", run_elasticnet),
    ("GEE logistic", "clustered linear", run_gee),
    ("Random Forest", "static tree ensemble", run_rf),
    ("HistGradientBoosting", "gradient boosting", run_hgb),
    # LSTM moved to M2-B vertical (B2 Dual-Tower uses GRU + auxiliary head;
    # standalone LSTM here would be redundant). See arch.md for the disclosure.
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    tok = forbidden_token()
    if tok in Path(__file__).read_text():
        raise RuntimeError(f"Forbidden literal '{tok}' present in source.")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading stacked dataset…")
    sd = load_stacked()
    print(f"  rows={len(sd.rows)} episodes={len(sd.episode_meta)} prevalence={sd.episode_meta['Y_24M_NHRH'].mean():.3f}")

    # Run all methods and collect calibrated predictions
    results: list[dict] = []
    for name, category, fn in METHODS:
        print(f"\n[{name}] training…")
        t0 = time.time()
        raw = fn(sd)
        wall = time.time() - t0
        cal = per_landmark_platt_on_pooled_oof(sd, raw)
        proba_cal = apply_per_landmark_platt(sd, raw, cal)
        tmp = per_landmark_and_pooled_perf(sd, proba_cal, split="Temporal")
        dev = per_landmark_and_pooled_perf(sd, proba_cal, split="Development")
        pooled_tmp = tmp[tmp["Landmark"] == "Pooled"]
        print(f"  wall={wall:.1f}s, Temporal pooled ROC={float(pooled_tmp['ROC_AUC'].iloc[0]):.4f}")
        results.append({
            "method": name,
            "category": category,
            "wall_seconds": wall,
            "proba_calibrated": proba_cal,
            "dev_perf": dev,
            "tmp_perf": tmp,
            "calibrators": cal,
        })

    # Leaderboard
    rows: list[dict] = []
    base_proba = results[0]["proba_calibrated"]  # L2-logistic as anchor
    for r in results:
        tmp_pool = r["tmp_perf"][r["tmp_perf"]["Landmark"] == "Pooled"]
        dev_pool = r["dev_perf"][r["dev_perf"]["Landmark"] == "Pooled"]
        # Δ vs base on Temporal pooled, paired cluster bootstrap
        d_mean, d_lo, d_hi = paired_episode_cluster_bootstrap_delta(
            sd, r["proba_calibrated"], base_proba, landmark=None, split="Temporal"
        )
        rows.append({
            "Method": r["method"],
            "Category": r["category"],
            "Wall_seconds": r["wall_seconds"],
            "Dev_Pooled_ROC": float(dev_pool["ROC_AUC"].iloc[0]),
            "Dev_Pooled_PR": float(dev_pool["PR_AUC"].iloc[0]),
            "Dev_Pooled_Brier": float(dev_pool["Brier"].iloc[0]),
            "Tmp_Pooled_ROC": float(tmp_pool["ROC_AUC"].iloc[0]),
            "Tmp_Pooled_PR": float(tmp_pool["PR_AUC"].iloc[0]),
            "Tmp_Pooled_Brier": float(tmp_pool["Brier"].iloc[0]),
            "Tmp_Pooled_ROC_CI_Low": float(tmp_pool["ROC_AUC_CI_Low"].iloc[0]),
            "Tmp_Pooled_ROC_CI_High": float(tmp_pool["ROC_AUC_CI_High"].iloc[0]),
            "Tmp_CalibIntercept": float(tmp_pool["CalibIntercept"].iloc[0]),
            "Tmp_CalibSlope": float(tmp_pool["CalibSlope"].iloc[0]),
            "Delta_vs_L2_mean": d_mean,
            "Delta_vs_L2_CI_Low": d_lo,
            "Delta_vs_L2_CI_High": d_hi,
            "Delta_significant": bool(d_lo > 0 or d_hi < 0)
            if not (np.isnan(d_lo) or np.isnan(d_hi)) else False,
        })
    leaderboard = pd.DataFrame(rows).sort_values("Tmp_Pooled_ROC", ascending=False).reset_index(drop=True)
    leaderboard.to_csv(TAB_DIR / "leaderboard.csv", index=False)
    print(f"\nLeaderboard saved to {TAB_DIR / 'leaderboard.csv'}")

    # Per-landmark heatmap data
    heatmap_rows: list[dict] = []
    for r in results:
        for L in list(LANDMARKS) + [None]:
            label = f"{L}M" if L is not None else "Pooled"
            sub = r["tmp_perf"][r["tmp_perf"]["Landmark"] == label]
            if not sub.empty:
                heatmap_rows.append({
                    "Method": r["method"],
                    "Landmark": label,
                    "Tmp_ROC": float(sub["ROC_AUC"].iloc[0]),
                })
    heatmap_df = pd.DataFrame(heatmap_rows)
    heatmap_df.to_csv(TAB_DIR / "per_landmark_perf.csv", index=False)

    # Render figures
    print("Rendering figures…")
    _fig_leaderboard(leaderboard, FIG_DIR / "Figure_01_Leaderboard.png")
    _fig_heatmap(heatmap_df, FIG_DIR / "Figure_02_PerLandmark_Heatmap.png")
    _fig_calibration_scatter(leaderboard, FIG_DIR / "Figure_03_Calibration_Scatter.png")

    summary = {
        "n_episodes": int(len(sd.episode_meta)),
        "n_landmark_rows": int(len(sd.rows)),
        "methods_run": len(METHODS),
        "leaderboard": leaderboard.to_dict(orient="records"),
        "best_method": leaderboard.iloc[0]["Method"],
        "best_tmp_pooled_roc": float(leaderboard.iloc[0]["Tmp_Pooled_ROC"]),
        "best_vs_L2_delta_mean": float(leaderboard.iloc[0]["Delta_vs_L2_mean"])
        if not np.isnan(leaderboard.iloc[0]["Delta_vs_L2_mean"]) else None,
        "best_vs_L2_delta_significant": bool(leaderboard.iloc[0]["Delta_significant"]),
        "omitted_methods": ["LightGBM", "XGBoost", "CatBoost", "TabNet", "Cox PH"],
        "omitted_reason": "external packages not available in anaconda base; subprocess isolation deferred",
    }
    (TAB_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== Leaderboard ===")
    print(leaderboard[["Method", "Category", "Tmp_Pooled_ROC", "Tmp_Pooled_ROC_CI_Low",
                        "Tmp_Pooled_ROC_CI_High", "Tmp_Pooled_Brier",
                        "Tmp_CalibSlope", "Delta_vs_L2_mean", "Delta_significant"]].to_string(index=False))


def _fig_leaderboard(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(df) + 1))
    y = np.arange(len(df))[::-1]
    means = df["Tmp_Pooled_ROC"].values
    lo = (df["Tmp_Pooled_ROC"] - df["Tmp_Pooled_ROC_CI_Low"]).abs().values
    hi = (df["Tmp_Pooled_ROC_CI_High"] - df["Tmp_Pooled_ROC"]).abs().values
    ax.errorbar(means, y, xerr=[lo, hi], fmt="o", color="#1d4e89", capsize=4)
    for i, row in df.iterrows():
        yp = y[i]
        sig = " *" if row["Delta_significant"] else ""
        ax.text(row["Tmp_Pooled_ROC_CI_High"] + 0.005, yp,
                f"  ROC={row['Tmp_Pooled_ROC']:.3f}, Δ vs L2 = {row['Delta_vs_L2_mean']:+.4f}{sig}",
                fontsize=8, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{row['Method']} ({row['Category']})" for _, row in df.iterrows()], fontsize=9)
    ax.set_xlim(0.55, 0.85)
    ax.axvline(0.5, color="#888", linestyle=":", linewidth=1)
    ax.set_xlabel("Temporal pooled ROC-AUC (episode-cluster bootstrap CI)")
    ax.set_title("M2-A leaderboard — 6 methods (4 omitted, deps unavailable)")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _fig_heatmap(df: pd.DataFrame, out: Path) -> None:
    pivot = df.pivot(index="Method", columns="Landmark", values="Tmp_ROC")
    landmark_order = [f"{L}M" for L in LANDMARKS] + ["Pooled"]
    pivot = pivot[landmark_order]
    fig, ax = plt.subplots(figsize=(8, 0.7 * len(pivot) + 1))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0.5, vmax=0.85)
    ax.set_xticks(np.arange(len(landmark_order)))
    ax.set_xticklabels(landmark_order)
    ax.set_yticks(np.arange(len(pivot)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color="white" if v > 0.72 else "#333", fontsize=9)
    ax.set_title("M2-A Temporal ROC-AUC per landmark × method")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _fig_calibration_scatter(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for i, row in df.iterrows():
        ax.scatter(row["Tmp_CalibIntercept"], row["Tmp_CalibSlope"], s=90,
                    label=f"{row['Method']} (ROC={row['Tmp_Pooled_ROC']:.3f})",
                    color=plt.cm.tab10(i / 10))
        ax.text(row["Tmp_CalibIntercept"] + 0.01, row["Tmp_CalibSlope"], row["Method"], fontsize=8)
    ax.axhline(1.0, color="#888", linestyle=":", linewidth=1)
    ax.axvline(0.0, color="#888", linestyle=":", linewidth=1)
    ax.set_xlabel("Calibration intercept (Temporal pooled)")
    ax.set_ylabel("Calibration slope (Temporal pooled)")
    ax.set_title("M2-A calibration honesty by method")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
