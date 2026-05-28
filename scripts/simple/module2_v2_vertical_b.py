#!/usr/bin/env python
"""M2-B · Three novel architectures (B1 MDJN, B2 Dual-Tower, B3 CLAN).

For each PyTorch architecture, trains on dev with 5-fold StratifiedGroupKFold
by episode, scores temporal-test, calibrates per-landmark Platt on pooled OOF,
and saves predictions + per-landmark + pooled performance. B3 additionally
extracts attention matrices per episode for risk-group visualization.

Outputs under results/module2_v2_vertical/{b1_mdjn,b2_dualtower,b3_clan}/.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scripts.simple.module2_v2_shared import (
    BLOCKS,
    LANDMARKS,
    StackedData,
    apply_per_landmark_platt,
    forbidden_token,
    load_stacked,
    make_episode_stratified_group_kfold,
    paired_episode_cluster_bootstrap_delta,
    per_landmark_and_pooled_perf,
    per_landmark_platt_on_pooled_oof,
    PY_SEED,
)

OUT_DIR = ROOT / "results" / "module2_v2_vertical"

torch.set_num_threads(1)
torch.manual_seed(PY_SEED)
np.random.seed(PY_SEED)


# ---------------------------------------------------------------------------
# Build episode-tensor representation
# ---------------------------------------------------------------------------


def build_episode_tensors(sd: StackedData):
    """Return seq_X (n_ep, 4, n_feat), seq_y (n_ep,), is_dev (n_ep,), episodes (n_ep,).

    Standardised per feature using dev-only mean/std.
    """
    X_df = sd.feature_matrix(("A", "B", "C", "D", "E")).copy()
    X_df["episode_id"] = sd.rows["episode_id"].values
    X_df["landmark"] = sd.rows["landmark"].values
    feature_cols = [c for c in X_df.columns if c not in ("episode_id", "landmark")]
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
    is_dev = np.array([ep_to_split[e] == "Development" for e in episodes])
    # Standardise per feature using dev-only stats
    dev_flat = seq_X[is_dev].reshape(-1, n_feat)
    mu = dev_flat.mean(axis=0)
    sigma = dev_flat.std(axis=0) + 1e-6
    seq_X = (seq_X - mu) / sigma
    # Block indices: which feature columns map to A/B/C/D/E
    block_idx = {}
    for b in ("A", "B", "C", "D", "E"):
        block_idx[b] = [feature_cols.index(c) for c in BLOCKS[b] if c in feature_cols]
    return seq_X, seq_y, is_dev, episodes, feature_cols, block_idx


# ---------------------------------------------------------------------------
# B1: MDJN  ·  Mechanism-Disentangled Joint Network
# ---------------------------------------------------------------------------


class B1_MDJN(nn.Module):
    def __init__(self, block_idx: dict[str, list[int]]):
        super().__init__()
        self.block_idx = block_idx
        dims = {"A": 16, "B": 8, "C": 16, "D": 8, "E": 8}
        # MDJN operates on landmark-pooled (mean over 4) per-block tensors.
        self.encoders = nn.ModuleDict()
        total = 0
        for b in ("A", "B", "C", "D", "E"):
            n_in = max(1, len(block_idx[b]))
            self.encoders[b] = nn.Sequential(
                nn.Linear(n_in, dims[b]), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(dims[b], dims[b]),
            )
            total += dims[b]
        self.fusion = nn.Sequential(
            nn.Linear(total, 16), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x, return_reprs: bool = False):
        # x: (B, 4, n_feat); average over the 4 landmarks per-block.
        pooled = x.mean(dim=1)  # (B, n_feat)
        reprs = {}
        zs = []
        for b in ("A", "B", "C", "D", "E"):
            cols = self.block_idx[b]
            if len(cols) == 0:
                # Empty block (shouldn't happen with our setup); skip
                continue
            sub = pooled[:, cols]
            r = self.encoders[b](sub)
            reprs[b] = r
            zs.append(r)
        cat = torch.cat(zs, dim=-1)
        out = self.fusion(cat).squeeze(-1)
        if return_reprs:
            return out, reprs
        return out


# ---------------------------------------------------------------------------
# B2: Static-Dynamic Dual-Tower with Aux 6M Multi-Task
# ---------------------------------------------------------------------------


class B2_DualTower(nn.Module):
    def __init__(self, block_idx: dict[str, list[int]]):
        super().__init__()
        self.block_idx = block_idx
        self.static_dim = len(block_idx["A"]) + len(block_idx["B"])
        self.dyn_dim = len(block_idx["C"]) + len(block_idx["D"])
        self.static_tower = nn.Sequential(
            nn.Linear(max(1, self.static_dim), 24), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(24, 24),
        )
        self.trajectory_tower = nn.GRU(
            input_size=max(1, self.dyn_dim), hidden_size=24, batch_first=True,
        )
        self.main_head = nn.Sequential(
            nn.Linear(48, 16), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, 1),
        )
        self.aux_head = nn.Sequential(
            nn.Linear(48, 16), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, 1),
        )

    def forward(self, x):
        # x: (B, 4, n_feat)
        # Static: use landmark-0 row for baseline burden + RAI exposure.
        a_cols = self.block_idx["A"]
        b_cols = self.block_idx["B"]
        static_cols = a_cols + b_cols
        if len(static_cols) == 0:
            static_in = torch.zeros(x.size(0), 1, device=x.device)
        else:
            static_in = x[:, 0, static_cols]  # 0M row
        static_repr = self.static_tower(static_in)
        # Dynamic: 4 timesteps × (C + D)
        c_cols = self.block_idx["C"]
        d_cols = self.block_idx["D"]
        dyn_cols = c_cols + d_cols
        if len(dyn_cols) == 0:
            traj_in = torch.zeros(x.size(0), 4, 1, device=x.device)
        else:
            traj_in = x[:, :, dyn_cols]
        traj_out, _ = self.trajectory_tower(traj_in)
        traj_repr = traj_out[:, -1, :]  # last timestep
        cat = torch.cat([static_repr, traj_repr], dim=-1)
        main = self.main_head(cat).squeeze(-1)
        aux = self.aux_head(cat).squeeze(-1)
        return main, aux


# ---------------------------------------------------------------------------
# B3: CLAN  ·  Cross-Landmark Attention Network
# ---------------------------------------------------------------------------


class B3_CLAN(nn.Module):
    def __init__(self, n_feat: int, embed_dim: int = 16, num_heads: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(n_feat, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(4, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=0.1,
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 8), nn.ReLU(), nn.Linear(8, 1),
        )

    def forward(self, x, return_attn: bool = False):
        # x: (B, 4, n_feat)
        tok = self.input_proj(x) + self.pos_emb.unsqueeze(0)
        attended, attn_weights = self.attn(tok, tok, tok, need_weights=True, average_attn_weights=True)
        pooled = attended.mean(dim=1)
        out = self.head(pooled).squeeze(-1)
        if return_attn:
            return out, attn_weights
        return out


# ---------------------------------------------------------------------------
# Training & evaluation harness
# ---------------------------------------------------------------------------


def train_b1_b3(
    Net,
    seq_X: np.ndarray,
    seq_y: np.ndarray,
    is_dev: np.ndarray,
    episodes: np.ndarray,
    block_idx: dict,
    *,
    n_feat: int,
    epochs: int = 60,
    n_folds: int = 5,
    return_attn: bool = False,
):
    """Generic train + OOF + temporal for B1 or B3 architectures (single head)."""
    skf = make_episode_stratified_group_kfold()
    ep_dev = episodes[is_dev]
    y_dev = seq_y[is_dev]
    X_dev = seq_X[is_dev]
    proba_ep = np.zeros(len(episodes), dtype=float)
    dev_ep_idx = np.where(is_dev)[0]
    attn_list: list[np.ndarray] = []
    fold_test_preds: list[np.ndarray] = []
    for tr, va in skf.split(X_dev, y_dev, groups=ep_dev):
        if Net is B3_CLAN:
            net = Net(n_feat=n_feat)
        else:
            net = Net(block_idx=block_idx)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        loss_fn = nn.BCEWithLogitsLoss()
        Xt = torch.from_numpy(X_dev[tr])
        yt = torch.from_numpy(y_dev[tr])
        for ep in range(epochs):
            net.train()
            opt.zero_grad()
            logits = net(Xt)
            loss = loss_fn(logits, yt)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            Xv = torch.from_numpy(X_dev[va])
            logits_va = net(Xv)
            proba_ep[dev_ep_idx[va]] = torch.sigmoid(logits_va).numpy()
    # Final fit on all dev
    if Net is B3_CLAN:
        net = Net(n_feat=n_feat)
    else:
        net = Net(block_idx=block_idx)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    Xall = torch.from_numpy(X_dev)
    yall = torch.from_numpy(y_dev)
    for ep in range(epochs):
        net.train()
        opt.zero_grad()
        loss = loss_fn(net(Xall), yall)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        Xte = torch.from_numpy(seq_X[~is_dev])
        if return_attn:
            logits_te, attn_te = net(Xte, return_attn=True)
            attn_list.append(attn_te.numpy())
        else:
            logits_te = net(Xte)
        proba_ep[~is_dev] = torch.sigmoid(logits_te).numpy()
    # Also extract attention on dev for visualisation if requested
    dev_attn = None
    if return_attn:
        with torch.no_grad():
            logits_dev_all, attn_dev = net(torch.from_numpy(X_dev), return_attn=True)
            dev_attn = attn_dev.numpy()
    return proba_ep, dev_attn, attn_list[0] if attn_list else None


def train_b2(
    seq_X: np.ndarray,
    seq_y: np.ndarray,
    aux_y: np.ndarray,
    is_dev: np.ndarray,
    episodes: np.ndarray,
    block_idx: dict,
    *,
    epochs: int = 60,
    alpha: float = 0.7,
):
    skf = make_episode_stratified_group_kfold()
    ep_dev = episodes[is_dev]
    y_dev = seq_y[is_dev]
    a_dev = aux_y[is_dev]
    X_dev = seq_X[is_dev]
    proba_ep = np.zeros(len(episodes), dtype=float)
    aux_ep = np.zeros(len(episodes), dtype=float)
    dev_ep_idx = np.where(is_dev)[0]
    for tr, va in skf.split(X_dev, y_dev, groups=ep_dev):
        net = B2_DualTower(block_idx=block_idx)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        loss_fn = nn.BCEWithLogitsLoss()
        Xt = torch.from_numpy(X_dev[tr])
        yt = torch.from_numpy(y_dev[tr])
        at = torch.from_numpy(a_dev[tr])
        for ep in range(epochs):
            net.train()
            opt.zero_grad()
            main, aux = net(Xt)
            loss = alpha * loss_fn(main, yt) + (1 - alpha) * loss_fn(aux, at)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            Xv = torch.from_numpy(X_dev[va])
            main_va, aux_va = net(Xv)
            proba_ep[dev_ep_idx[va]] = torch.sigmoid(main_va).numpy()
            aux_ep[dev_ep_idx[va]] = torch.sigmoid(aux_va).numpy()
    # Final on all dev for test scoring
    net = B2_DualTower(block_idx=block_idx)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    Xall = torch.from_numpy(X_dev)
    yall = torch.from_numpy(y_dev)
    aall = torch.from_numpy(a_dev)
    for ep in range(epochs):
        net.train()
        opt.zero_grad()
        main, aux = net(Xall)
        loss = alpha * loss_fn(main, yall) + (1 - alpha) * loss_fn(aux, aall)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        Xte = torch.from_numpy(seq_X[~is_dev])
        main_te, aux_te = net(Xte)
        proba_ep[~is_dev] = torch.sigmoid(main_te).numpy()
        aux_ep[~is_dev] = torch.sigmoid(aux_te).numpy()
    return proba_ep, aux_ep


# ---------------------------------------------------------------------------
# Replication helper: episode → 4 row predictions
# ---------------------------------------------------------------------------


def replicate_to_rows(sd: StackedData, proba_ep: np.ndarray, episodes: np.ndarray) -> np.ndarray:
    out = np.zeros(len(sd.rows), dtype=float)
    ep_to_p = dict(zip(episodes, proba_ep))
    for i, ep in enumerate(sd.rows["episode_id"].values):
        out[i] = ep_to_p[ep]
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    tok = forbidden_token()
    if tok in Path(__file__).read_text():
        raise RuntimeError(f"Forbidden literal '{tok}' present in source.")

    for sub in ("b1_mdjn", "b2_dualtower", "b3_clan"):
        (OUT_DIR / sub / "figures").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / sub / "tables").mkdir(parents=True, exist_ok=True)

    print("Loading stacked dataset…", flush=True)
    sd = load_stacked()
    seq_X, seq_y, is_dev, episodes, feature_cols, block_idx = build_episode_tensors(sd)
    n_feat = seq_X.shape[2]
    print(f"  tensor shape: {seq_X.shape}, dev episodes: {is_dev.sum()}, n_feat={n_feat}", flush=True)

    # Aux 6M target: derive from M2-Base output (approximate Eval_6M = Hyper proxy)
    # We use Y_24M_NHRH as the aux target's surrogate — admittedly imperfect, but
    # demonstrates the multi-task structure. A true Eval_6M_Hyper feature would
    # plug in here in a future iteration.
    aux_y = seq_y.copy()  # surrogate
    print(f"  aux 6M target: using Y_24M_NHRH as surrogate (no Eval_6M available)")

    summary_rows: list[dict] = []

    # --- B1 MDJN ---
    print("\n[B1 MDJN] training…")
    t0 = time.time()
    proba_ep_b1, _, _ = train_b1_b3(
        B1_MDJN, seq_X, seq_y, is_dev, episodes, block_idx, n_feat=n_feat,
        epochs=40, return_attn=False,
    )
    wall_b1 = time.time() - t0
    proba_b1_rows = replicate_to_rows(sd, proba_ep_b1, episodes)
    cal_b1 = per_landmark_platt_on_pooled_oof(sd, proba_b1_rows)
    proba_b1_cal = apply_per_landmark_platt(sd, proba_b1_rows, cal_b1)
    perf_b1 = per_landmark_and_pooled_perf(sd, proba_b1_cal, split="Temporal")
    pool_b1 = perf_b1[perf_b1["Landmark"] == "Pooled"]
    print(f"  wall={wall_b1:.1f}s, Temporal pooled ROC={float(pool_b1['ROC_AUC'].iloc[0]):.4f}")
    perf_b1.to_csv(OUT_DIR / "b1_mdjn" / "tables" / "perf_temporal.csv", index=False)
    summary_rows.append({"Arch": "B1 MDJN", "TmpROC": float(pool_b1["ROC_AUC"].iloc[0]),
                          "Brier": float(pool_b1["Brier"].iloc[0]),
                          "CalibSlope": float(pool_b1["CalibSlope"].iloc[0]),
                          "Wall_seconds": wall_b1})

    # --- B2 Dual-Tower ---
    print("\n[B2 Dual-Tower] training…")
    t0 = time.time()
    proba_ep_b2, aux_ep_b2 = train_b2(seq_X, seq_y, aux_y, is_dev, episodes, block_idx,
                                       epochs=40, alpha=0.7)
    wall_b2 = time.time() - t0
    proba_b2_rows = replicate_to_rows(sd, proba_ep_b2, episodes)
    cal_b2 = per_landmark_platt_on_pooled_oof(sd, proba_b2_rows)
    proba_b2_cal = apply_per_landmark_platt(sd, proba_b2_rows, cal_b2)
    perf_b2 = per_landmark_and_pooled_perf(sd, proba_b2_cal, split="Temporal")
    pool_b2 = perf_b2[perf_b2["Landmark"] == "Pooled"]
    print(f"  wall={wall_b2:.1f}s, Temporal pooled ROC={float(pool_b2['ROC_AUC'].iloc[0]):.4f}")
    perf_b2.to_csv(OUT_DIR / "b2_dualtower" / "tables" / "perf_temporal.csv", index=False)
    summary_rows.append({"Arch": "B2 Dual-Tower", "TmpROC": float(pool_b2["ROC_AUC"].iloc[0]),
                          "Brier": float(pool_b2["Brier"].iloc[0]),
                          "CalibSlope": float(pool_b2["CalibSlope"].iloc[0]),
                          "Wall_seconds": wall_b2})

    # --- B3 CLAN ---
    print("\n[B3 CLAN] training…")
    t0 = time.time()
    proba_ep_b3, dev_attn_b3, _ = train_b1_b3(
        B3_CLAN, seq_X, seq_y, is_dev, episodes, block_idx, n_feat=n_feat,
        epochs=40, return_attn=True,
    )
    wall_b3 = time.time() - t0
    proba_b3_rows = replicate_to_rows(sd, proba_ep_b3, episodes)
    cal_b3 = per_landmark_platt_on_pooled_oof(sd, proba_b3_rows)
    proba_b3_cal = apply_per_landmark_platt(sd, proba_b3_rows, cal_b3)
    perf_b3 = per_landmark_and_pooled_perf(sd, proba_b3_cal, split="Temporal")
    pool_b3 = perf_b3[perf_b3["Landmark"] == "Pooled"]
    print(f"  wall={wall_b3:.1f}s, Temporal pooled ROC={float(pool_b3['ROC_AUC'].iloc[0]):.4f}")
    perf_b3.to_csv(OUT_DIR / "b3_clan" / "tables" / "perf_temporal.csv", index=False)
    summary_rows.append({"Arch": "B3 CLAN", "TmpROC": float(pool_b3["ROC_AUC"].iloc[0]),
                          "Brier": float(pool_b3["Brier"].iloc[0]),
                          "CalibSlope": float(pool_b3["CalibSlope"].iloc[0]),
                          "Wall_seconds": wall_b3})

    # --- B3 attention visualization ---
    if dev_attn_b3 is not None:
        # dev_attn_b3 shape: (n_dev_ep, 4, 4) — averaged over heads
        print("Generating B3 attention heatmaps…")
        dev_proba_for_grouping = proba_ep_b3[is_dev]
        # Risk groups based on terciles of B3 dev predictions
        t1, t2 = np.percentile(dev_proba_for_grouping, [33, 67])
        groups = np.where(
            dev_proba_for_grouping <= t1, "low",
            np.where(dev_proba_for_grouping <= t2, "mid", "high"),
        )
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
        for ax, g in zip(axes, ("low", "mid", "high")):
            mask = groups == g
            if mask.sum() < 5:
                continue
            mean_attn = dev_attn_b3[mask].mean(axis=0)
            im = ax.imshow(mean_attn, cmap="YlOrRd", vmin=0, vmax=0.5)
            ax.set_xticks(range(4))
            ax.set_xticklabels([f"{L}M" for L in LANDMARKS])
            ax.set_yticks(range(4))
            ax.set_yticklabels([f"{L}M" for L in LANDMARKS])
            ax.set_title(f"{g.title()} risk (n={int(mask.sum())})")
            ax.set_xlabel("Key landmark")
            ax.set_ylabel("Query landmark")
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, f"{mean_attn[i, j]:.2f}", ha="center", va="center",
                            fontsize=8, color="white" if mean_attn[i, j] > 0.25 else "#333")
        fig.suptitle("B3 CLAN: mean attention weights by risk group (Dev OOF)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(OUT_DIR / "b3_clan" / "figures" / "Figure_attention_by_risk_group.png", dpi=160)
        plt.close(fig)

    # --- Comparison figure ---
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUT_DIR / "tables_comparison.csv", index=False)
    # Compare against M2-Base (read from its run_summary)
    base_summary_path = ROOT / "results" / "module2_v2_base" / "tables" / "run_summary.json"
    if base_summary_path.exists():
        with open(base_summary_path) as f:
            base = json.load(f)
        df_summary.loc[len(df_summary)] = {
            "Arch": "M2-Base.4 (L2)", "TmpROC": base["ablation_steps"][-1]["tmp_pooled_roc"],
            "Brier": base["ablation_steps"][-1]["tmp_pooled_brier"],
            "CalibSlope": float("nan"), "Wall_seconds": float("nan"),
        }
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(df_summary))[::-1]
    ax.barh(y, df_summary["TmpROC"], color=["#a23b3b", "#d28b18", "#5a3f8a", "#1d4e89"][:len(df_summary)])
    for i, row in df_summary.iterrows():
        ax.text(row["TmpROC"] + 0.003, y[i], f"{row['TmpROC']:.4f}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(df_summary["Arch"].tolist())
    ax.set_xlim(0.5, 0.85)
    ax.axvline(0.5, color="#888", linestyle=":", linewidth=1)
    ax.set_xlabel("Temporal pooled ROC-AUC")
    ax.set_title("M2-B vertical architectures vs M2-Base anchor")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Figure_comparison.png", dpi=160)
    plt.close(fig)
    # Run summary JSON
    summary = {
        "n_episodes": int(len(sd.episode_meta)),
        "n_landmark_rows": int(len(sd.rows)),
        "architectures": df_summary.to_dict(orient="records"),
        "best_arch": df_summary.iloc[df_summary["TmpROC"].idxmax()]["Arch"],
        "best_tmp_roc": float(df_summary["TmpROC"].max()),
        "note": "Aux 6M target in B2 uses Y_24M_NHRH as surrogate (Eval_6M_Hyper not yet wired in); paper will document limitation.",
    }
    (OUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== Summary ===")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
