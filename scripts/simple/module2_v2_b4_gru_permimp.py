#!/usr/bin/env python
"""M2 · B4 — GRU permutation importance (per-landmark group drift), model-agnostic.

A GRU has no interpretable coefficients, so permutation / drop importance is the
ONLY clean way to rank feature groups for it — which is exactly the collinearity-
robust method we settled on. Here: train the unidirectional GRU on dev, then on
the held-out temporal set permute each clinical group's columns across episodes
(consistently across the 4 steps, preserving within-block trajectory) and measure
the per-landmark (per-step) AUC drop. Compare the drift to the LR group analysis.

Cheap: no retraining per permutation — just re-predict with permuted inputs.
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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from scripts.simple.module2_v2_shared import LANDMARKS, load_stacked, PY_SEED
from scripts.simple.module2_v2_vertical_b4_gru import TargetGRU, build_timeaware_tensors

torch.set_num_threads(1)
OUT = ROOT / "results" / "module2_v2_vertical" / "b4_target_gru"

# clinical groups -> feature names (mapped to tensor columns via feature_cols)
GROUPS = {
    "Goiter (ThyroidW)": ["ThyroidW"],
    "Antibodies": ["TRAb", "TGAb", "TPOAb"],
    "Baseline labs/chronicity": ["FT4_0M", "TSH_0M", "log1p_DiseaseDuration_Months_Aug", "Sex"],
    "RAI exposure": ["Uptake24h", "HalfLife"],
    "Current level (C)": ["TSH_current", "FT4_current"],
    "Momentum (D)": ["TSH_velocity", "FT4_velocity"],
}


def train_gru_on_dev(seq_X, dt, mask, seq_y, is_dev, block_idx, seed, epochs=80):
    torch.manual_seed(seed); np.random.seed(seed)
    net = TargetGRU(block_idx, hidden=12, use_decay=False, zoneout=0.1)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    bce = nn.BCEWithLogitsLoss()
    dev = np.where(is_dev)[0]
    X = torch.from_numpy(seq_X[dev]); D = torch.from_numpy(dt[dev]); M = torch.from_numpy(mask[dev])
    Y = torch.from_numpy(seq_y[dev].astype(np.float32))
    n = X.shape[0]
    for _ in range(epochs):
        net.train(); perm = torch.randperm(n)
        for i in range(0, n, 64):
            b = perm[i:i + 64]
            loss = bce(net(X[b], D[b], M[b])[0], Y[b].unsqueeze(1).expand(-1, 4))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    net.eval()
    return net


def predict_steps(net, X, dt, mask):
    with torch.no_grad():
        return torch.sigmoid(net(torch.from_numpy(X), torch.from_numpy(dt), torch.from_numpy(mask))[0]).numpy()


def main() -> None:
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    sd = load_stacked()
    seq_X, dt, mask, seq_y, is_dev, episodes, feature_cols, block_idx = build_timeaware_tensors(sd)
    col = {f: i for i, f in enumerate(feature_cols)}
    grp_cols = {g: [col[f] for f in fs if f in col] for g, fs in GROUPS.items()}

    print("Training GRU on dev (3-seed ensemble)…", flush=True)
    nets = [train_gru_on_dev(seq_X, dt, mask, seq_y, is_dev, block_idx, s) for s in (2025, 2026, 2027)]

    test = np.where(~is_dev)[0]
    Xt, dtt, mt = seq_X[test], dt[test], mask[test]
    yt = seq_y[test]

    def ens_pred(X):
        return np.mean([predict_steps(nt, X, dtt, mt) for nt in nets], axis=0)  # (n,4)

    base = ens_pred(Xt)
    base_auc = {L: roc_auc_score(yt, base[:, s]) for s, L in enumerate(LANDMARKS)}
    print("GRU baseline per-step temporal AUC:", {f"{L}M": round(v, 4) for L, v in base_auc.items()}, flush=True)

    R = 10
    rng = np.random.default_rng(7)
    drift = {g: {} for g in GROUPS}
    for g, cols in grp_cols.items():
        if not cols:
            continue
        for s, L in enumerate(LANDMARKS):
            drops = []
            for _ in range(R):
                Xp = Xt.copy()
                perm = rng.permutation(Xp.shape[0])
                Xp[:, :, cols] = Xp[perm][:, :, cols]   # shuffle group across episodes (all 4 steps)
                p = ens_pred(Xp)
                drops.append(base_auc[L] - roc_auc_score(yt, p[:, s]))
            drift[g][L] = round(float(np.mean(drops)), 4)

    table = pd.DataFrame({f"{L}M": {g: drift[g][L] for g in GROUPS if g in drift and drift[g]} for L in LANDMARKS})
    table.insert(0, "group", table.index)
    table.to_csv(OUT / "tables" / "gru_perm_importance_drift.csv", index=False)
    print("\n=== GRU permutation importance (ΔAUC on temporal, per landmark) ===", flush=True)
    print(table.to_string(index=False), flush=True)
    print("\n=== GRU per-landmark leaderboard ===", flush=True)
    lead = {}
    for L in LANDMARKS:
        ranked = sorted([g for g in GROUPS if drift[g]], key=lambda g: drift[g][L], reverse=True)
        lead[f"{L}M"] = [(g, drift[g][L]) for g in ranked[:3]]
        print(f"  {L}M: " + "  |  ".join(f"{g} {drift[g][L]:+.3f}" for g in ranked[:3]), flush=True)

    (OUT / "tables" / "gru_permimp_summary.json").write_text(json.dumps({
        "method": "GRU permutation importance (group shuffled across episodes), temporal per-step AUC drop, 3-seed ens, R=10",
        "baseline_per_step_AUC": {f"{L}M": round(v, 4) for L, v in base_auc.items()},
        "drift": {g: drift[g] for g in GROUPS if drift[g]}, "leaderboard": lead,
    }, indent=2, ensure_ascii=False))
    print(f"\nSaved → {OUT}", flush=True)


if __name__ == "__main__":
    main()
