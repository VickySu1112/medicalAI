#!/usr/bin/env python
"""M2 · B5 GRU-ODE(-lite) — continuous-time hidden evolution between landmarks.

Phase 3 of the B4 plan, run in an ISOLATED venv (rai_ode + torchdiffeq) so the
production anaconda env is never touched. Tests whether continuous-time Neural-ODE
evolution of the hidden state between the unequal {0,1,3,6}M landmarks adds
anything beyond the plain unidirectional GRU (B4 V1 base = 0.730).

Architecture (GRU-ODE flavour):
  static A+B -> h0; for each landmark t: evolve h over Δt via odeint (Neural ODE),
  then GRU-update h with the observation (C+D) at t; a shared per-step head ->
  P(24M | <= t). Unidirectional => time-safe by construction; deep supervision.

The full GRU-ODE-Bayes (De Brouwer, NeurIPS 2019) additionally propagates Gaussian
uncertainty (the "Bayes" part); on 4 fixed landmarks the continuity prior has
little to learn, so we use the deterministic GRU-ODE core and document this.

Reuses the B4 data pipeline + episode-cluster CV + per-landmark Platt for a fair
head-to-head. OOF (dev) for selection; temporal only reported.

Run (isolated env):
    PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        /Users/ql/.venvs/rai_ode/bin/python scripts/simple/module2_v2_vertical_b5_gruode.py
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import sys
import time
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
from sklearn.model_selection import StratifiedKFold
from torchdiffeq import odeint

from scripts.simple.module2_v2_shared import (
    apply_per_landmark_platt,
    fit_l2_predict_oof,
    load_stacked,
    make_episode_stratified_group_kfold,
    paired_episode_cluster_bootstrap_delta,
    per_landmark_and_pooled_perf,
    per_landmark_platt_on_pooled_oof,
    run_leakage_assertions,
)
from scripts.simple.module2_v2_vertical_b4_gru import (
    DT_MONTHS,
    build_timeaware_tensors,
    place_step_predictions_into_rows,
)

torch.set_num_threads(1)
OUT = ROOT / "results" / "module2_v2_vertical" / "b5_gruode"
SEEDS = (2025, 2026, 2027)


class ODEFunc(nn.Module):
    """Autonomous Neural ODE: dh/dt = f(h)."""

    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, hidden))

    def forward(self, t, y):  # noqa: ARG002 (autonomous — t unused)
        return self.net(y)


class GRUODENet(nn.Module):
    def __init__(self, block_idx: dict[str, list[int]], hidden: int = 12, head_dropout: float = 0.3):
        super().__init__()
        self.static_cols = list(block_idx["A"]) + list(block_idx["B"])
        self.dyn_cols = list(block_idx["C"]) + list(block_idx["D"])
        self.hidden = hidden
        self.static_mlp = nn.Sequential(nn.Linear(max(1, len(self.static_cols)), hidden), nn.Tanh())
        self.ode = ODEFunc(hidden)
        self.gru = nn.GRUCell(max(1, len(self.dyn_cols)), hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, 8), nn.ReLU(), nn.Dropout(head_dropout), nn.Linear(8, 1),
        )

    def forward(self, x, dt, mask):  # noqa: ARG002 (mask kept for signature parity)
        B = x.size(0)
        static_in = x[:, 0, self.static_cols] if self.static_cols else torch.zeros(B, 1)
        h = self.static_mlp(static_in)
        logits = []
        for t in range(4):
            step_dt = float(DT_MONTHS[t])
            if step_dt > 0.0:
                tt = torch.tensor([0.0, step_dt], dtype=h.dtype)
                h = odeint(self.ode, h, tt, method="rk4", options={"step_size": 0.5})[-1]
            obs = x[:, t, self.dyn_cols] if self.dyn_cols else torch.zeros(B, 1)
            h = self.gru(obs, h)
            logits.append(self.head(h).squeeze(-1))
        return torch.stack(logits, dim=1), None


def _set_seed(s: int) -> None:
    torch.manual_seed(s)
    np.random.seed(s)


def _inner_split(y: np.ndarray, seed: int = 11, n_splits: int = 6):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return next(skf.split(np.zeros(len(y)), y))


def _fit_predict(X_tr, dt_tr, m_tr, y_tr, X_va, dt_va, m_va, y_va,
                 X_tg, dt_tg, m_tg, block_idx, *, seed, hidden, epochs, patience):
    _set_seed(seed)
    net = GRUODENet(block_idx, hidden=hidden)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    bce = nn.BCEWithLogitsLoss()
    Xtr = torch.from_numpy(X_tr); dttr = torch.from_numpy(dt_tr); mtr = torch.from_numpy(m_tr)
    ytr = torch.from_numpy(y_tr.astype(np.float32))
    Xva = torch.from_numpy(X_va); dtva = torch.from_numpy(dt_va); mva = torch.from_numpy(m_va)
    n = Xtr.shape[0]; bs = 64
    best, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            bidx = perm[i:i + bs]
            main, _ = net(Xtr[bidx], dttr[bidx], mtr[bidx])
            loss = bce(main, ytr[bidx].unsqueeze(1).expand(-1, 4))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(Xva, dtva, mva)[0]).numpy()
        yf = np.repeat(y_va, 4)
        auc = roc_auc_score(yf, pv.reshape(-1)) if len(np.unique(yf)) > 1 else 0.5
        if auc > best + 1e-4:
            best, best_state, bad = auc, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pt = torch.sigmoid(net(torch.from_numpy(X_tg), torch.from_numpy(dt_tg),
                               torch.from_numpy(m_tg))[0]).numpy()
    return pt


def train_ode(seq_X, dt, mask, seq_y, is_dev, episodes, block_idx, *,
              hidden=12, epochs=60, patience=12, seeds=SEEDS):
    n_ep = seq_X.shape[0]
    dev = np.where(is_dev)[0]
    test = np.where(~is_dev)[0]
    Xd, dtd, md, yd = seq_X[dev], dt[dev], mask[dev], seq_y[dev]
    ep_d = episodes[dev]
    proba = np.zeros((n_ep, 4))
    for fi, (tr_rel, va_rel) in enumerate(make_episode_stratified_group_kfold().split(Xd, yd, groups=ep_d)):
        i_tr, i_val = _inner_split(yd[tr_rel])
        tr, val = tr_rel[i_tr], tr_rel[i_val]
        preds = [_fit_predict(Xd[tr], dtd[tr], md[tr], yd[tr], Xd[val], dtd[val], md[val], yd[val],
                              Xd[va_rel], dtd[va_rel], md[va_rel], block_idx,
                              seed=s, hidden=hidden, epochs=epochs, patience=patience) for s in seeds]
        proba[dev[va_rel]] = np.mean(preds, axis=0)
        print(f"  [fold {fi + 1}/5] {len(va_rel)} episodes", flush=True)
    if len(test):
        i_tr, i_val = _inner_split(yd)
        preds = [_fit_predict(Xd[i_tr], dtd[i_tr], md[i_tr], yd[i_tr], Xd[i_val], dtd[i_val], md[i_val], yd[i_val],
                              seq_X[test], dt[test], mask[test], block_idx,
                              seed=s, hidden=hidden, epochs=epochs, patience=patience) for s in seeds]
        proba[test] = np.mean(preds, axis=0)
    return proba


def main() -> None:
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    print(f"torch {torch.__version__} + torchdiffeq loaded", flush=True)
    sd = load_stacked()
    run_leakage_assertions(sd, full=True)
    seq_X, dt, mask, seq_y, is_dev, episodes, _, block_idx = build_timeaware_tensors(sd)
    print(f"  tensor {seq_X.shape}; training GRU-ODE (3-seed ensemble)…", flush=True)

    t0 = time.time()
    proba = train_ode(seq_X, dt, mask, seq_y, is_dev, episodes, block_idx)
    print(f"  done in {time.time() - t0:.1f}s", flush=True)

    raw = place_step_predictions_into_rows(sd, proba, episodes)
    cal = apply_per_landmark_platt(sd, raw, per_landmark_platt_on_pooled_oof(sd, raw))
    tmp = per_landmark_and_pooled_perf(sd, cal, split="Temporal")
    tmp.to_csv(OUT / "tables" / "perf_temporal.csv", index=False)
    print("\n=== B5 GRU-ODE temporal per-landmark + pooled ===", flush=True)
    print(tmp.to_string(index=False), flush=True)

    l2_raw, _ = fit_l2_predict_oof(sd, block_subset=("A", "B", "C", "D", "E"))
    l2_cal = apply_per_landmark_platt(sd, l2_raw, per_landmark_platt_on_pooled_oof(sd, l2_raw))
    mu, lo, hi = paired_episode_cluster_bootstrap_delta(sd, cal, l2_cal, split="Temporal")
    pool = float(tmp.set_index("Landmark").loc["Pooled", "ROC_AUC"])
    slope = float(tmp.set_index("Landmark").loc["Pooled", "CalibSlope"])
    summary = {
        "arch": "B5 GRU-ODE(-lite) — Neural-ODE hidden evolution + GRU update",
        "seeds": list(SEEDS),
        "temporal_pooled_ROC": pool,
        "temporal_pooled_CalibSlope": slope,
        "per_landmark_ROC": {L: float(tmp.set_index("Landmark").loc[L, "ROC_AUC"])
                             for L in ["0M", "1M", "3M", "6M", "Pooled"]},
        "delta_vs_L2super_pooled": mu, "delta_CI": [lo, hi],
        "reference": {"B4_V1_base": 0.730, "L2_supermodel": 0.739, "per_LR_Eval": 0.770},
        "note": ("Deterministic GRU-ODE core (no Bayesian uncertainty propagation). "
                 "Isolated venv rai_ode + torchdiffeq; anaconda untouched."),
    }
    (OUT / "tables" / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nGRU-ODE pooled {pool:.4f} | Δ vs L2 {mu:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
          f"ref B4 V1 base 0.730, per-LR+Eval 0.770", flush=True)
    print(f"Saved → {OUT}", flush=True)


if __name__ == "__main__":
    main()
