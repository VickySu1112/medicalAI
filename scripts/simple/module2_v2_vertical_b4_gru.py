#!/usr/bin/env python
"""M2 · B4 TARGET-GRU — Time-Aware Recurrent Gated network with deep supervision.

A single unidirectional GRU-D-style sequence model over the 4 landmarks
(0M/1M/3M/6M) of one treatment episode — the elegant single-model replacement
for the per-landmark 4-LR.

Design:
  * Static block A+B (baseline burden + RAI exposure, read from the 0M row) ->
    small MLP -> initializes the GRU hidden state h0.
  * Dynamic block C+D (current TSH/FT4 + velocity) fed step-by-step with Δt and a
    mask channel, through a GRU-D cell with learnable per-unit time decay.
  * UNIDIRECTIONAL => step-t hidden encodes only landmarks <= t, so the per-step
    prediction at landmark t is time-safe BY CONSTRUCTION (no mask trick).
  * SHARED per-step head on every step's hidden state (deep supervision): trained
    with the same 24M label on all 4 steps; the step-t output IS that landmark's
    time-safe prediction. Replaces the B2/B3 "replicate one episode number to 4
    rows" hack (the reason B2/B3 per-landmark ROCs were all identical).

Phase 1 (single GRU-D) verified the harness. Phase 2 (this file) runs the variant
sweep V1 vanilla / V2 GRU-D / V3 AWD + 5-seed ensemble + train-only augmentation,
selects the OOF-pooled winner, and emits a method×landmark comparison.

Outputs under results/module2_v2_vertical/b4_target_gru/.

Run (anaconda or a torch venv):
    PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python \
        scripts/simple/module2_v2_vertical_b4_gru.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

# MUST be set before importing torch (macOS torch hang guard — see
# module2_v2_vertical_b.py; B1/B2/B3 stalled with 0 stdout without this).
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from scripts.simple.module2_v2_shared import (
    LANDMARKS,
    StackedData,
    apply_per_landmark_platt,
    fit_l2_predict_oof,
    forbidden_token,
    load_stacked,
    make_episode_stratified_group_kfold,
    paired_episode_cluster_bootstrap_delta,
    per_landmark_and_pooled_perf,
    per_landmark_platt_on_pooled_oof,
    run_leakage_assertions,
    PY_SEED,
)
from scripts.simple.module2_v2_vertical_b import build_episode_tensors

torch.set_num_threads(1)

OUT_DIR = ROOT / "results" / "module2_v2_vertical" / "b4_target_gru"

LM_TO_STEP = {0: 0, 1: 1, 3: 2, 6: 3}
DT_MONTHS = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)  # Δt since prev landmark
SEEDS = (2025, 2026, 2027, 2028, 2029)

# Reference temporal ROC of prior baselines (from committed runs on the anaconda
# env; per-landmark numbers used for the method×landmark heatmap context).
REF_BASELINES = {
    "per-LR + Eval (4-LR)": 0.770,
    "L2 supermodel": 0.739,
    "Random Forest": 0.7435,
    "B2 Dual-Tower (GRU)": 0.701,
    "B3 CLAN": 0.719,
}
REF_PER_LANDMARK = {  # temporal ROC at 0/1/3/6M + Pooled
    "per-LR+Eval": {0: 0.686, 1: 0.721, 3: 0.847, 6: 0.788, "Pooled": 0.770},
    "RandomForest": {0: 0.658, 1: 0.697, 3: 0.788, 6: 0.804, "Pooled": 0.743},
}

# Phase-2 variant grid (shared cell, feature-flagged). aug per-config.
VARIANTS = [
    {"key": "V1_vanilla", "label": "V1 vanilla GRU", "use_decay": False, "zoneout": 0.1, "weight_drop": 0.0, "augment": True},
    {"key": "V2_gru_d", "label": "V2 GRU-D", "use_decay": True, "zoneout": 0.1, "weight_drop": 0.0, "augment": True},
    {"key": "V3_awd", "label": "V3 AWD-GRU", "use_decay": True, "zoneout": 0.1, "weight_drop": 0.2, "augment": True},
    {"key": "V2_noaug", "label": "V2 GRU-D (no aug)", "use_decay": True, "zoneout": 0.1, "weight_drop": 0.0, "augment": False},
]


# ---------------------------------------------------------------------------
# Tensors
# ---------------------------------------------------------------------------


def build_timeaware_tensors(sd: StackedData):
    seq_X, seq_y, is_dev, episodes, feature_cols, block_idx = build_episode_tensors(sd)
    n_ep = seq_X.shape[0]
    dt = np.broadcast_to(DT_MONTHS.reshape(1, 4, 1), (n_ep, 4, 1)).astype(np.float32).copy()
    mask = np.ones((n_ep, 4, 1), dtype=np.float32)
    return seq_X, dt, mask, seq_y, is_dev, episodes, feature_cols, block_idx


# ---------------------------------------------------------------------------
# Model: GRU-D cell (manual gates) + TARGET-GRU
# ---------------------------------------------------------------------------


class GRUDCell(nn.Module):
    """Manual-gate GRU cell with GRU-D learnable hidden time-decay.

    * Hidden decays by exp(-softplus(gamma)·Δt) before each update so older
      information fades on the unequal {0,1,3,6}M spacing.
    * zoneout randomly preserves the (decayed) previous hidden state.
    * weight_drop = AWD-LSTM-style DropConnect on the recurrent weight W_hh
      (via F.dropout on the weight matrix → autograd-safe) for the V3 variant.
    """

    def __init__(self, input_size: int, hidden_size: int, *,
                 zoneout: float = 0.0, use_decay: bool = True, weight_drop: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_decay = use_decay
        self.zoneout = zoneout
        self.weight_drop = weight_drop
        self.W_ih = nn.Linear(input_size, 3 * hidden_size, bias=True)
        self.W_hh = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.gamma = nn.Parameter(torch.zeros(hidden_size))  # softplus -> decay >= 0
        self._reset()

    def _reset(self) -> None:
        std = 1.0 / (self.hidden_size ** 0.5)
        for lin in (self.W_ih, self.W_hh):
            nn.init.uniform_(lin.weight, -std, std)
            nn.init.zeros_(lin.bias)
        nn.init.zeros_(self.gamma)

    def forward(self, x_t: torch.Tensor, h: torch.Tensor, dt_t: torch.Tensor) -> torch.Tensor:
        if self.use_decay:
            rate = torch.clamp(nn.functional.softplus(self.gamma), max=5.0).unsqueeze(0)  # (1,H)
            h_prev = h * torch.exp(-rate * dt_t)  # (B,1)*(1,H) -> (B,H)
        else:
            h_prev = h
        gi = self.W_ih(x_t)
        w_hh = self.W_hh.weight
        if self.training and self.weight_drop > 0.0:
            w_hh = nn.functional.dropout(w_hh, p=self.weight_drop, training=True)
        gh = nn.functional.linear(h_prev, w_hh, self.W_hh.bias)
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        h_new = (1.0 - z) * n + z * h_prev
        if self.training and self.zoneout > 0.0:
            zo = (torch.rand_like(h_new) < self.zoneout).float()
            h_new = zo * h_prev + (1.0 - zo) * h_new
        return h_new


class TargetGRU(nn.Module):
    """Unidirectional GRU-D over 4 landmarks with a shared per-step head."""

    def __init__(self, block_idx: dict[str, list[int]], *, hidden: int = 12,
                 head_dropout: float = 0.3, input_dropout: float = 0.2,
                 zoneout: float = 0.0, use_decay: bool = True, weight_drop: float = 0.0,
                 with_aux: bool = False):
        super().__init__()
        self.block_idx = block_idx
        self.static_cols = list(block_idx["A"]) + list(block_idx["B"])
        # Block F (optional): per-landmark Eval immune-state features, folded into
        # the dynamic input when present (time-safe — step t sees only its own Eval).
        self.dyn_cols = list(block_idx["C"]) + list(block_idx["D"]) + list(block_idx.get("F", []))
        self.hidden = hidden
        self.input_dropout = input_dropout
        self.with_aux = with_aux

        self.static_mlp = nn.Sequential(
            nn.Linear(max(1, len(self.static_cols)), hidden), nn.Tanh(),
        )
        gru_in = max(1, len(self.dyn_cols)) + 2  # + Δt + mask channels
        self.cell = GRUDCell(gru_in, hidden, zoneout=zoneout, use_decay=use_decay,
                             weight_drop=weight_drop)
        self.head = nn.Sequential(
            nn.Linear(hidden, 8), nn.ReLU(), nn.Dropout(head_dropout), nn.Linear(8, 1),
        )
        if with_aux:
            self.aux_head = nn.Sequential(
                nn.Linear(hidden, 8), nn.ReLU(), nn.Dropout(head_dropout), nn.Linear(8, 1),
            )

    def forward(self, x: torch.Tensor, dt: torch.Tensor, mask: torch.Tensor):
        B = x.size(0)
        if self.static_cols:
            static_in = x[:, 0, self.static_cols]
        else:
            static_in = torch.zeros(B, 1, device=x.device)
        h = self.static_mlp(static_in)

        if self.dyn_cols:
            dyn = x[:, :, self.dyn_cols]
        else:
            dyn = torch.zeros(B, 4, 1, device=x.device)
        if self.training and self.input_dropout > 0.0:
            keep = (torch.rand(B, 1, dyn.size(-1), device=x.device) > self.input_dropout).float()
            dyn = dyn * keep / (1.0 - self.input_dropout)

        main_logits, aux_logits = [], []
        for t in range(4):
            inp = torch.cat([dyn[:, t, :], dt[:, t, :], mask[:, t, :]], dim=-1)
            h = self.cell(inp, h, dt[:, t, :])
            main_logits.append(self.head(h).squeeze(-1))
            if self.with_aux:
                aux_logits.append(self.aux_head(h).squeeze(-1))
        main = torch.stack(main_logits, dim=1)  # (B,4)
        aux = torch.stack(aux_logits, dim=1) if self.with_aux else None
        return main, aux


# ---------------------------------------------------------------------------
# Augmentation (TRAIN-FOLD ONLY — never on OOF/temporal)
# ---------------------------------------------------------------------------


def apply_augmentation(x: torch.Tensor, dt: torch.Tensor, mask: torch.Tensor,
                       block_idx: dict[str, list[int]], seed: int):
    g = torch.Generator().manual_seed(int(seed) % (2 ** 31 - 1))
    x = x.clone(); dt = dt.clone(); mask = mask.clone()
    B = x.shape[0]
    # (a) assay-noise jitter on current TSH/FT4 (block C), ~7% on standardized scale
    c_cols = list(block_idx["C"])
    if c_cols:
        noise = torch.randn(B, x.shape[1], len(c_cols), generator=g) * 0.07
        x[:, :, c_cols] = x[:, :, c_cols] + noise
    # (b) landmark dropout: ~15% episodes drop one of steps 1..3 (never step 0)
    drop = torch.rand(B, generator=g) < 0.15
    steps = torch.randint(1, 4, (B,), generator=g)
    for i in range(B):
        if bool(drop[i]):
            mask[i, int(steps[i]), 0] = 0.0
    # (c) Δt jitter ±10% multiplicative (order-preserving)
    jit = (1.0 + torch.randn(dt.shape, generator=g) * 0.1).clamp(0.5, 1.5)
    dt = dt * jit
    return x, dt, mask


# ---------------------------------------------------------------------------
# Training: deep-supervision OOF harness
# ---------------------------------------------------------------------------


def _set_seed(s: int) -> None:
    torch.manual_seed(s)
    np.random.seed(s)


def _inner_split(y: np.ndarray, seed: int = 11, n_splits: int = 6):
    """One stratified split of episode-tensor rows for early stopping.

    Each episode-tensor row is one (unique) episode, so a stratified split here
    cannot leak landmarks across the boundary — the whole episode is on one side.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr, val = next(skf.split(np.zeros(len(y)), y))
    return tr, val


def _fit_predict_steps(X_tr, dt_tr, m_tr, y_tr, X_val, dt_val, m_val, y_val,
                       X_tgt, dt_tgt, m_tgt, block_idx, *, seed, arch_kw, train_kw):
    _set_seed(seed)
    net = TargetGRU(block_idx, **arch_kw)
    opt = torch.optim.AdamW(net.parameters(), lr=train_kw["lr"],
                            weight_decay=train_kw["weight_decay"])
    bce = nn.BCEWithLogitsLoss()

    Xtr = torch.from_numpy(X_tr); dttr = torch.from_numpy(dt_tr); mtr = torch.from_numpy(m_tr)
    ytr = torch.from_numpy(y_tr.astype(np.float32))
    Xva = torch.from_numpy(X_val); dtva = torch.from_numpy(dt_val); mva = torch.from_numpy(m_val)
    n = Xtr.shape[0]
    bs = train_kw["batch_size"]

    best_auc, best_state, bad = -1.0, None, 0
    for epoch in range(train_kw["max_epochs"]):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            bidx = perm[i:i + bs]
            xb, dtb, mb, yb = Xtr[bidx], dttr[bidx], mtr[bidx], ytr[bidx]
            if train_kw["augment"]:
                xb, dtb, mb = apply_augmentation(xb, dtb, mb, block_idx, seed * 100003 + epoch)
            main, _ = net(xb, dtb, mb)
            target = yb.unsqueeze(1).expand(-1, 4)      # deep supervision
            loss = bce(main, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(Xva, dtva, mva)[0]).numpy()
        y_flat = np.repeat(y_val, 4)
        auc = roc_auc_score(y_flat, pv.reshape(-1)) if len(np.unique(y_flat)) > 1 else 0.5
        if auc > best_auc + 1e-4:
            best_auc = auc
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= train_kw["patience"]:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pt = torch.sigmoid(net(torch.from_numpy(X_tgt), torch.from_numpy(dt_tgt),
                               torch.from_numpy(m_tgt))[0]).numpy()
    return pt


def train_target_gru(seq_X, dt, mask, seq_y, is_dev, episodes, block_idx, *,
                     hidden=12, max_epochs=120, patience=15, seeds=SEEDS,
                     batch_size=64, lr=1e-3, weight_decay=1e-3, augment=False,
                     head_dropout=0.3, input_dropout=0.2, zoneout=0.0,
                     use_decay=True, weight_drop=0.0, n_folds_override=None):
    """Per-step deep-supervision OOF on dev + final-fit prediction on temporal.

    Returns proba_step (n_ep, 4): OOF step-probs for dev, final-model step-probs
    for temporal, averaged over `seeds`.
    """
    arch_kw = dict(hidden=hidden, head_dropout=head_dropout, input_dropout=input_dropout,
                   zoneout=zoneout, use_decay=use_decay, weight_drop=weight_drop, with_aux=False)
    train_kw = dict(max_epochs=max_epochs, patience=patience, batch_size=batch_size,
                    lr=lr, weight_decay=weight_decay, augment=augment)

    n_ep = seq_X.shape[0]
    dev_pos = np.where(is_dev)[0]
    test_pos = np.where(~is_dev)[0]
    Xd, dtd, md, yd = seq_X[dev_pos], dt[dev_pos], mask[dev_pos], seq_y[dev_pos]
    ep_d = episodes[dev_pos]
    proba_step = np.zeros((n_ep, 4), dtype=float)

    splitter = make_episode_stratified_group_kfold()
    folds = list(splitter.split(Xd, yd, groups=ep_d))
    if n_folds_override is not None:
        folds = folds[:n_folds_override]

    for fi, (tr_rel, va_rel) in enumerate(folds):
        i_tr, i_val = _inner_split(yd[tr_rel], seed=11)
        tr_abs, val_abs = tr_rel[i_tr], tr_rel[i_val]
        seed_preds = [
            _fit_predict_steps(
                Xd[tr_abs], dtd[tr_abs], md[tr_abs], yd[tr_abs],
                Xd[val_abs], dtd[val_abs], md[val_abs], yd[val_abs],
                Xd[va_rel], dtd[va_rel], md[va_rel],
                block_idx, seed=s, arch_kw=arch_kw, train_kw=train_kw,
            )
            for s in seeds
        ]
        proba_step[dev_pos[va_rel]] = np.mean(seed_preds, axis=0)

    if len(test_pos) and n_folds_override is None:
        i_tr, i_val = _inner_split(yd, seed=11)
        Xt, dtt, mt = seq_X[test_pos], dt[test_pos], mask[test_pos]
        seed_preds = [
            _fit_predict_steps(
                Xd[i_tr], dtd[i_tr], md[i_tr], yd[i_tr],
                Xd[i_val], dtd[i_val], md[i_val], yd[i_val],
                Xt, dtt, mt,
                block_idx, seed=s, arch_kw=arch_kw, train_kw=train_kw,
            )
            for s in seeds
        ]
        proba_step[test_pos] = np.mean(seed_preds, axis=0)

    return proba_step


def place_step_predictions_into_rows(sd: StackedData, proba_step: np.ndarray,
                                     episodes: np.ndarray) -> np.ndarray:
    """Row (episode, landmark=L) <- step-L prediction (uses <= L info)."""
    ep_pos = {int(e): i for i, e in enumerate(episodes)}
    out = np.zeros(len(sd.rows), dtype=float)
    eids = sd.rows["episode_id"].values
    lms = sd.rows["landmark"].values
    for r in range(len(sd.rows)):
        out[r] = proba_step[ep_pos[int(eids[r])], LM_TO_STEP[int(lms[r])]]
    return out


def assert_time_safety(block_idx: dict[str, list[int]], n_feat: int) -> None:
    _set_seed(0)
    net = TargetGRU(block_idx, hidden=12, use_decay=True)
    net.eval()
    B = 8
    x = torch.randn(B, 4, n_feat)
    dt = torch.from_numpy(np.broadcast_to(DT_MONTHS.reshape(1, 4, 1), (B, 4, 1)).astype(np.float32).copy())
    mask = torch.ones(B, 4, 1)
    for t in range(3):
        x2 = x.clone()
        x2[:, t + 1:, :] = torch.randn(B, 4 - (t + 1), n_feat)
        with torch.no_grad():
            o1 = net(x, dt, mask)[0][:, t]
            o2 = net(x2, dt, mask)[0][:, t]
        assert torch.allclose(o1, o2, atol=1e-5), f"TIME-SAFETY VIOLATED at step {t}"
    print("  [assert] time-safety OK: step-t output ignores landmarks > t", flush=True)


# ---------------------------------------------------------------------------
# Config evaluation
# ---------------------------------------------------------------------------


def evaluate_config(sd, tensors, *, label, augment, use_decay, zoneout, weight_drop,
                    seeds=SEEDS, max_epochs=120, n_folds_override=None):
    seq_X, dt, mask, seq_y, is_dev, episodes, _, block_idx = tensors
    proba_step = train_target_gru(
        seq_X, dt, mask, seq_y, is_dev, episodes, block_idx,
        seeds=seeds, augment=augment, use_decay=use_decay, zoneout=zoneout,
        weight_drop=weight_drop, max_epochs=max_epochs, n_folds_override=n_folds_override,
    )
    raw = place_step_predictions_into_rows(sd, proba_step, episodes)
    cal = per_landmark_platt_on_pooled_oof(sd, raw)
    proba_cal = apply_per_landmark_platt(sd, raw, cal)
    perf_tmp = per_landmark_and_pooled_perf(sd, proba_cal, split="Temporal")
    perf_dev = per_landmark_and_pooled_perf(sd, proba_cal, split="Development")
    dev_pool = float(perf_dev[perf_dev["Landmark"] == "Pooled"]["ROC_AUC"].iloc[0])
    tmp_pool = float(perf_tmp[perf_tmp["Landmark"] == "Pooled"]["ROC_AUC"].iloc[0])
    return {"label": label, "perf_tmp": perf_tmp, "perf_dev": perf_dev,
            "proba_cal": proba_cal, "dev_pooled_roc": dev_pool, "tmp_pooled_roc": tmp_pool}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_method_landmark_heatmap(rows: list[tuple[str, dict]], path: Path) -> None:
    cols = ["0M", "1M", "3M", "6M", "Pooled"]
    keys = [0, 1, 3, 6, "Pooled"]
    M = np.array([[r[1].get(k, np.nan) for k in keys] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(rows) + 1.5))
    im = ax.imshow(M, cmap="YlGnBu", vmin=0.62, vmax=0.86, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    for i in range(len(rows)):
        for j in range(len(cols)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] > 0.78 else "#222")
    ax.set_title("Temporal ROC by method × landmark (B4 variants vs baselines)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_variant_calibration(results: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = ["#8a8a8a", "#2a7f5f", "#a23b3b", "#1d4e89"]
    for res, c in zip(results, colors):
        sub = res["perf_tmp"][res["perf_tmp"]["Landmark"] != "Pooled"]
        ax.scatter(sub["CalibIntercept"], sub["CalibSlope"], color=c, s=45, label=res["label"])
    ax.axhline(1.0, color="#888", linestyle=":", linewidth=1)
    ax.axvline(0.0, color="#888", linestyle=":", linewidth=1)
    ax.axhspan(0.8, 1.2, color="#2a7f5f", alpha=0.08)
    ax.set_xlabel("Calibration intercept (ideal 0)")
    ax.set_ylabel("Calibration slope (ideal 1)")
    ax.set_title("B4 variants: per-landmark calibration (temporal)")
    ax.legend(fontsize=7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="1 fold, 1 seed, few epochs — verify wiring")
    args = parser.parse_args()

    tok = forbidden_token()
    if tok in Path(__file__).read_text():
        raise RuntimeError(f"Forbidden literal '{tok}' present in source.")

    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)

    print("Loading stacked dataset…", flush=True)
    sd = load_stacked()
    run_leakage_assertions(sd, full=True)
    tensors = build_timeaware_tensors(sd)
    seq_X = tensors[0]
    n_feat = seq_X.shape[2]
    print(f"  tensor {seq_X.shape}, dev={int(tensors[4].sum())}, n_feat={n_feat}", flush=True)

    print("Time-safety unit test…", flush=True)
    assert_time_safety(tensors[7], n_feat)

    if args.smoke:
        print("[SMOKE] V2 1 fold / 1 seed / 5 epochs", flush=True)
        res = evaluate_config(sd, tensors, label="V2 smoke", augment=False, use_decay=True,
                              zoneout=0.0, weight_drop=0.0, seeds=(PY_SEED,), max_epochs=5,
                              n_folds_override=1)
        print(res["perf_tmp"].to_string(index=False), flush=True)
        print("[SMOKE] wiring OK.", flush=True)
        return

    # ---- Phase 2: variant sweep (5-seed ensemble each) ----
    t0 = time.time()
    results = []
    for v in VARIANTS:
        print(f"\n[{v['label']}] 5-seed ensemble (aug={v['augment']})…", flush=True)
        res = evaluate_config(sd, tensors, label=v["label"], augment=v["augment"],
                              use_decay=v["use_decay"], zoneout=v["zoneout"],
                              weight_drop=v["weight_drop"], seeds=SEEDS)
        res["key"] = v["key"]
        results.append(res)
        res["perf_tmp"].to_csv(OUT_DIR / "tables" / f"perf_temporal_{v['key']}.csv", index=False)
        print(f"  dev pooled ROC={res['dev_pooled_roc']:.4f} | temporal pooled ROC={res['tmp_pooled_roc']:.4f}",
              flush=True)
    print(f"\nVariant sweep done in {time.time() - t0:.1f}s", flush=True)

    # Winner by DEV (OOF) pooled ROC — model selection on OOF only
    winner = max(results, key=lambda r: r["dev_pooled_roc"])
    print(f"\n>>> OOF winner: {winner['label']} (dev pooled {winner['dev_pooled_roc']:.4f}, "
          f"temporal pooled {winner['tmp_pooled_roc']:.4f})", flush=True)

    # Variant comparison table
    comp_rows = []
    for res in results:
        pt = res["perf_tmp"].set_index("Landmark")
        row = {"Variant": res["label"], "Dev_Pooled_ROC": res["dev_pooled_roc"]}
        for L in ["0M", "1M", "3M", "6M", "Pooled"]:
            row[f"Tmp_ROC_{L}"] = float(pt.loc[L, "ROC_AUC"]) if L in pt.index else np.nan
        row["Tmp_Pooled_Brier"] = float(pt.loc["Pooled", "Brier"])
        row["Tmp_Pooled_CalibSlope"] = float(pt.loc["Pooled", "CalibSlope"])
        comp_rows.append(row)
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(OUT_DIR / "tables" / "variant_comparison.csv", index=False)
    print("\n=== Variant comparison (temporal ROC) ===", flush=True)
    print(comp_df.to_string(index=False), flush=True)

    # L2 supermodel baseline (live, reproducible) + paired ΔAUC for winner
    print("\nComputing L2 supermodel baseline + paired ΔAUC (winner vs L2)…", flush=True)
    l2_raw, _ = fit_l2_predict_oof(sd, block_subset=("A", "B", "C", "D", "E"))
    l2_cal = apply_per_landmark_platt(sd, l2_raw, per_landmark_platt_on_pooled_oof(sd, l2_raw))
    perf_l2 = per_landmark_and_pooled_perf(sd, l2_cal, split="Temporal")
    delta_rows = []
    for L in [None] + list(LANDMARKS):
        mu, lo, hi = paired_episode_cluster_bootstrap_delta(
            sd, winner["proba_cal"], l2_cal, landmark=L, split="Temporal")
        delta_rows.append({"Landmark": "Pooled" if L is None else f"{L}M",
                           "Delta_ROC_vs_L2super": mu, "CI_Low": lo, "CI_High": hi,
                           "excludes_0": bool((lo > 0) or (hi < 0))})
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(OUT_DIR / "tables" / "delta_winner_vs_L2.csv", index=False)
    print(delta_df.to_string(index=False), flush=True)

    # method × landmark heatmap: B4 variants + live L2 + reference baselines
    heat_rows = []
    for res in results:
        pt = res["perf_tmp"].set_index("Landmark")["ROC_AUC"]
        heat_rows.append((res["label"], {L: float(pt.get(f"{L}M", np.nan)) for L in LANDMARKS}
                          | {"Pooled": float(pt.get("Pooled", np.nan))}))
    pl2 = perf_l2.set_index("Landmark")["ROC_AUC"]
    heat_rows.append(("L2 supermodel", {L: float(pl2.get(f"{L}M", np.nan)) for L in LANDMARKS}
                      | {"Pooled": float(pl2.get("Pooled", np.nan))}))
    for name, d in REF_PER_LANDMARK.items():
        heat_rows.append((name, d))
    fig_method_landmark_heatmap(heat_rows, OUT_DIR / "figures" / "Figure_03_Method_x_Landmark.png")
    fig_variant_calibration(results, OUT_DIR / "figures" / "Figure_02_Variant_Calibration.png")

    # Run summary + decision-rule inputs (for the winner)
    wp = winner["perf_tmp"].set_index("Landmark")
    pool_delta = delta_df[delta_df["Landmark"] == "Pooled"].iloc[0]
    slopes = winner["perf_tmp"][winner["perf_tmp"]["Landmark"] != "Pooled"]["CalibSlope"].astype(float)
    summary = {
        "phase": 2,
        "arch": "B4 TARGET-GRU (unidirectional GRU-D, deep supervision, 5-seed ensemble)",
        "n_episodes": int(len(sd.episode_meta)),
        "n_landmark_rows": int(len(sd.rows)),
        "seeds": list(SEEDS),
        "variants": comp_df.to_dict(orient="records"),
        "oof_winner": winner["label"],
        "winner_temporal_pooled_ROC": float(wp.loc["Pooled", "ROC_AUC"]),
        "winner_per_landmark_ROC": {L: float(wp.loc[L, "ROC_AUC"]) for L in
                                     ["0M", "1M", "3M", "6M", "Pooled"]},
        "winner_temporal_pooled_CalibSlope": float(wp.loc["Pooled", "CalibSlope"]),
        "winner_delta_vs_L2super_pooled": float(pool_delta["Delta_ROC_vs_L2super"]),
        "winner_delta_CI": [float(pool_delta["CI_Low"]), float(pool_delta["CI_High"])],
        "calibration_stable": bool(((slopes >= 0.8) & (slopes <= 1.2)).all()),
        "ref_baselines": REF_BASELINES,
        "phase1_reference": {"V2_single_seed_no_aug_pooled_ROC": 0.727},
    }
    (OUT_DIR / "tables" / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
