"""Standalone direct three-branch relapse hazard model.

This script is intentionally separate from relapse_teacher_frozen_fuse.py:
it trains static/local/global encoders plus one final hazard head directly on
Y_Relapse, with optional training-only augmentation. It does not read
Teacher035 targets, does not fit a frozen LR readout, and writes only to the
user-provided output directory.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import pandas as pd
import scipy.integrate
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

if not hasattr(scipy.integrate, "trapz"):
    scipy.integrate.trapz = scipy.integrate.trapezoid

from scripts import relapse_threehead_landmark as base
from utils.config import SEED
from utils.evaluation import compute_binary_metrics


Z3M_BRANCH_MODES = {"branch", "add_branch"}
Z3M_COMPONENT_COLS = [f"selected_z3m_{j:02d}" for j in range(12)]


@dataclass
class TrainConfig:
    output_dir: Path
    seed: int = SEED
    run_name: str = "direct_threebranch"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 128
    max_epochs: int = 80
    patience: int = 12
    lr: float = 8e-4
    embed_dim: int = 48
    static_embed_dim: int = 0
    dropout: float = 0.15
    weight_decay: float = 1e-4
    select_metric: str = "val_prauc"
    mixed_train_weight: float = 0.3
    mixed_val_weight: float = 1.0
    mixed_test_weight: float = 0.0
    disable_test_selection: bool = True
    train_prauc_cap: float = 0.0
    z3m_mode: str = "none"
    z3m_path: Path = ROOT / "results" / "simple" / "z_3m_features.csv"
    z3m_branch_lr_scale: float = 1.0
    z3m_gate_init: float = 1.0
    z3m_gate_mode: str = "scalar"
    z3m_time_features: bool = False
    pretrain_z3m_branch: bool = False
    z3m_pretrain_epochs: int = 40
    z3m_pretrain_lr: float = 1e-3
    z3m_pretrain_weight_decay: float = 1e-4
    z3m_pretrain_patience: int = 8
    z3m_aux_loss_weight: float = 0.0
    z3m_pretrain_target: str = "z3m"
    drop_static_branch: bool = False
    drop_local_branch: bool = False
    drop_global_branch: bool = False
    feature_selection_json: Path | None = None
    balanced_sampler: bool = False
    loss: str = "bce"
    focal_gamma: float = 2.0
    pos_weight_scale: float = 1.0
    label_smoothing: float = 0.0
    patient_weighted_bce: bool = False
    rank_loss_weight: float = 0.0
    rank_top_neg_frac: float = 0.3
    rank_tau: float = 1.0
    net_benefit_loss_weight: float = 0.0
    clinical_threshold: float = 0.20
    net_benefit_temperature: float = 0.7
    ap_loss_weight: float = 0.0
    ap_tau: float = 1.0
    ap_warmup_epochs: int = 0
    ap_select_after_warmup: bool = False
    lr_scheduler: str = "none"
    lr_scheduler_metric: str = "selected_score"
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    min_lr: float = 1e-6
    libauc_ap_loss_weight: float = 0.0
    libauc_margin: float = 1.0
    libauc_gamma: float = 0.9
    libauc_soap: bool = False
    branch_drop_prob: float = 0.0
    augment: bool = False
    bootstrap_ratio: float = 0.10
    feature_noise_ratio: float = 0.0
    feature_noise_std: float = 0.04
    mixup_ratio: float = 0.15
    mixup_alpha: float = 8.0
    mixup_beta: float = 2.0
    consistency_weight: float = 0.05
    mask_prob: float = 0.15
    input_gate_l1: float = 0.0
    input_gate_init: float = 0.95
    branch_gate_l1: float = 0.0
    branch_gate_init: float = 0.95
    static_branch_gate_l1: float = 0.0
    static_branch_gate_init: float = 0.95


class DirectDataset(Dataset):
    def __init__(
        self,
        tensors: dict[str, np.ndarray],
        y: np.ndarray,
        interval_stage: np.ndarray,
        prev_state: np.ndarray,
        risk_decile: np.ndarray,
    ) -> None:
        self.static = torch.tensor(tensors["static"], dtype=torch.float32)
        self.local = torch.tensor(tensors["local"], dtype=torch.float32)
        self.global_x = torch.tensor(tensors["global"], dtype=torch.float32)
        z3m = tensors.get("z3m", np.zeros((len(y), 0), dtype=np.float32))
        self.z3m = torch.tensor(z3m, dtype=torch.float32)
        self.z3m_target = torch.tensor(tensors.get("z3m_target", np.zeros(len(y), dtype=np.float32)), dtype=torch.float32)
        self.z3m_target_mask = torch.tensor(tensors.get("z3m_target_mask", np.zeros(len(y), dtype=np.float32)), dtype=torch.float32)
        self.patient_weight = torch.tensor(tensors.get("patient_weight", np.ones(len(y), dtype=np.float32)), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.sample_index = torch.arange(len(y), dtype=torch.long)
        self.interval_stage = torch.tensor(interval_stage, dtype=torch.long)
        self.prev_state = torch.tensor(prev_state, dtype=torch.long)
        self.risk_decile = torch.tensor(risk_decile, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "static": self.static[idx],
            "local": self.local[idx],
            "global": self.global_x[idx],
            "z3m": self.z3m[idx],
            "z3m_target": self.z3m_target[idx],
            "z3m_target_mask": self.z3m_target_mask[idx],
            "patient_weight": self.patient_weight[idx],
            "y": self.y[idx],
            "sample_index": self.sample_index[idx],
            "interval_stage": self.interval_stage[idx],
            "prev_state": self.prev_state[idx],
            "risk_decile": self.risk_decile[idx],
        }


class ThreeBranchHazardNet(nn.Module):
    def __init__(
        self,
        static_dim: int,
        local_dim: int,
        global_dim: int,
        embed_dim: int,
        dropout: float,
        static_embed_dim: int | None = None,
        use_static_branch: bool = True,
        use_local_branch: bool = True,
        use_global_branch: bool = True,
        z3m_dim: int = 0,
        use_z3m_branch: bool = False,
        z3m_gate_init: float = 1.0,
        z3m_gate_mode: str = "scalar",
        input_gate_init: float = 0.95,
        input_gates: bool = False,
        branch_gate_init: float = 0.95,
        branch_gates: bool = False,
        static_branch_gate_init: float = 0.95,
        static_branch_gate: bool = False,
    ) -> None:
        super().__init__()
        self.use_static_branch = use_static_branch
        self.use_local_branch = use_local_branch
        self.use_global_branch = use_global_branch
        self.use_z3m_branch = use_z3m_branch and z3m_dim > 0
        self.use_input_gates = input_gates
        self.use_branch_gates = branch_gates
        self.z3m_gate_mode = z3m_gate_mode
        static_out_dim = int(static_embed_dim or embed_dim)
        self.static_embed_dim = static_out_dim
        self.embed_dim = embed_dim
        self.static_branch = base.MLPBranch(static_dim, static_out_dim, dropout) if use_static_branch else None
        self.local_branch = base.MLPBranch(local_dim, embed_dim, dropout) if use_local_branch else None
        self.global_branch = base.LinearBranch(global_dim, embed_dim) if use_global_branch else None
        self.z3m_branch = base.MLPBranch(z3m_dim, embed_dim, dropout) if self.use_z3m_branch else None
        self.z3m_aux_head = nn.Linear(embed_dim, 1) if self.use_z3m_branch else None
        if self.use_z3m_branch and z3m_gate_mode == "scalar":
            self.z3m_gate = nn.Parameter(torch.tensor(float(z3m_gate_init)))
            self.z3m_time_gate = None
        elif self.use_z3m_branch and z3m_gate_mode == "time_sigmoid":
            self.z3m_gate = None
            self.z3m_time_gate = nn.Linear(1, 1)
            init = float(np.clip(z3m_gate_init, 1e-4, 1.0 - 1e-4))
            nn.init.zeros_(self.z3m_time_gate.weight)
            nn.init.constant_(self.z3m_time_gate.bias, float(np.log(init / (1.0 - init))))
        elif self.use_z3m_branch:
            raise ValueError(f"Unsupported z3m_gate_mode={z3m_gate_mode!r}")
        else:
            self.z3m_gate = None
            self.z3m_time_gate = None
        if self.use_input_gates:
            gate_init = float(np.clip(input_gate_init, 1e-4, 1.0 - 1e-4))
            init_logit = float(np.log(gate_init / (1.0 - gate_init)))
            self.static_input_gate_logit = nn.Parameter(torch.full((static_dim,), init_logit)) if use_static_branch else None
            self.local_input_gate_logit = nn.Parameter(torch.full((local_dim,), init_logit)) if use_local_branch else None
            self.global_input_gate_logit = nn.Parameter(torch.full((global_dim,), init_logit)) if use_global_branch else None
            self.z3m_input_gate_logit = nn.Parameter(torch.full((z3m_dim,), init_logit)) if self.use_z3m_branch else None
        else:
            self.static_input_gate_logit = None
            self.local_input_gate_logit = None
            self.global_input_gate_logit = None
            self.z3m_input_gate_logit = None
        if self.use_branch_gates:
            branch_init = float(np.clip(branch_gate_init, 1e-4, 1.0 - 1e-4))
            branch_logit = float(np.log(branch_init / (1.0 - branch_init)))
            self.static_branch_gate_logit = nn.Parameter(torch.tensor(branch_logit)) if use_static_branch else None
            self.local_branch_gate_logit = nn.Parameter(torch.tensor(branch_logit)) if use_local_branch else None
            self.global_branch_gate_logit = nn.Parameter(torch.tensor(branch_logit)) if use_global_branch else None
            self.z3m_branch_gate_logit = nn.Parameter(torch.tensor(branch_logit)) if self.use_z3m_branch else None
        else:
            self.static_branch_gate_logit = None
            self.local_branch_gate_logit = None
            self.global_branch_gate_logit = None
            self.z3m_branch_gate_logit = None
        if static_branch_gate and use_static_branch and self.static_branch_gate_logit is None:
            static_init = float(np.clip(static_branch_gate_init, 1e-4, 1.0 - 1e-4))
            self.static_branch_gate_logit = nn.Parameter(torch.tensor(float(np.log(static_init / (1.0 - static_init)))))
        branch_count = int(use_static_branch) + int(use_local_branch) + int(use_global_branch) + int(self.use_z3m_branch)
        if branch_count <= 0:
            raise ValueError("At least one branch must be enabled")
        fuse_in_dim = (
            (static_out_dim if use_static_branch else 0)
            + (embed_dim if use_local_branch else 0)
            + (embed_dim if use_global_branch else 0)
            + (embed_dim if self.use_z3m_branch else 0)
        )
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.hazard_head = nn.Linear(embed_dim, 1)

    def forward(
        self,
        static_x: torch.Tensor,
        local_x: torch.Tensor,
        global_x: torch.Tensor,
        z3m_x: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        branches = []
        if self.use_input_gates:
            if self.use_static_branch:
                assert self.static_input_gate_logit is not None
                static_x = static_x * torch.sigmoid(self.static_input_gate_logit)
            if self.use_local_branch:
                assert self.local_input_gate_logit is not None
                local_x = local_x * torch.sigmoid(self.local_input_gate_logit)
            if self.use_global_branch:
                assert self.global_input_gate_logit is not None
                global_x = global_x * torch.sigmoid(self.global_input_gate_logit)
        if self.use_static_branch:
            assert self.static_branch is not None
            h_static = self.static_branch(static_x)
            if self.static_branch_gate_logit is not None:
                assert self.static_branch_gate_logit is not None
                h_static = h_static * torch.sigmoid(self.static_branch_gate_logit)
            branches.append(h_static)
        if self.use_local_branch:
            assert self.local_branch is not None
            h_local = self.local_branch(local_x)
            if self.use_branch_gates:
                assert self.local_branch_gate_logit is not None
                h_local = h_local * torch.sigmoid(self.local_branch_gate_logit)
            branches.append(h_local)
        if self.use_global_branch:
            assert self.global_branch is not None
            h_global = self.global_branch(global_x)
            if self.use_branch_gates:
                assert self.global_branch_gate_logit is not None
                h_global = h_global * torch.sigmoid(self.global_branch_gate_logit)
            branches.append(h_global)
        if self.use_z3m_branch:
            assert self.z3m_branch is not None
            if z3m_x is None:
                raise ValueError("z3m_x is required when --z3m-mode branch/add_branch is used")
            if self.use_input_gates:
                assert self.z3m_input_gate_logit is not None
                z3m_x = z3m_x * torch.sigmoid(self.z3m_input_gate_logit)
            h_z3m = self.z3m_branch(z3m_x)
            if self.z3m_gate_mode == "scalar":
                assert self.z3m_gate is not None
                h_z3m = h_z3m * self.z3m_gate
            elif self.z3m_gate_mode == "time_sigmoid":
                assert self.z3m_time_gate is not None
                time_signal = z3m_x[:, -1:].clamp(-5.0, 5.0)
                h_z3m = h_z3m * torch.sigmoid(self.z3m_time_gate(time_signal))
            if self.use_branch_gates:
                assert self.z3m_branch_gate_logit is not None
                h_z3m = h_z3m * torch.sigmoid(self.z3m_branch_gate_logit)
            branches.append(h_z3m)
        else:
            h_z3m = None
        fused_input = torch.cat(branches, dim=1)
        hidden = self.fuse(fused_input)
        logit = self.hazard_head(hidden).squeeze(1)
        out = {"logit": logit, "prob": torch.sigmoid(logit), "hidden": hidden}
        if h_z3m is not None:
            assert self.z3m_aux_head is not None
            out["z3m_aux_logit"] = self.z3m_aux_head(h_z3m).squeeze(1)
        return out

    def regularization_loss(self, input_weight: float = 0.0, branch_weight: float = 0.0, static_branch_weight: float = 0.0) -> torch.Tensor:
        ref = self.hazard_head.weight.sum() * 0.0
        terms = []
        if input_weight > 0 and self.use_input_gates:
            for gate in [
                self.static_input_gate_logit,
                self.local_input_gate_logit,
                self.global_input_gate_logit,
                self.z3m_input_gate_logit,
            ]:
                if gate is not None:
                    terms.append(input_weight * torch.sigmoid(gate).mean())
        if branch_weight > 0 and self.use_branch_gates:
            for gate in [
                self.static_branch_gate_logit,
                self.local_branch_gate_logit,
                self.global_branch_gate_logit,
                self.z3m_branch_gate_logit,
            ]:
                if gate is not None:
                    terms.append(branch_weight * torch.sigmoid(gate))
        if static_branch_weight > 0 and self.static_branch_gate_logit is not None:
            terms.append(static_branch_weight * torch.sigmoid(self.static_branch_gate_logit))
        if not terms:
            return ref
        return torch.stack([term.reshape(()) for term in terms]).sum()


class Z3MBranchPretrainNet(nn.Module):
    def __init__(self, z3m_dim: int, embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.branch = base.MLPBranch(z3m_dim, embed_dim, dropout)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, z3m_x: torch.Tensor) -> torch.Tensor:
        return self.head(self.branch(z3m_x)).squeeze(1)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def risk_proxy_deciles(df_fit: pd.DataFrame, *frames: pd.DataFrame) -> list[np.ndarray]:
    cols = [
        "FT3_Current",
        "FT4_Current",
        "logTSH_Current",
        "Delta_FT4_1step",
        "Delta_TSH_1step",
        "Prior_Relapse_Count",
        "Ever_Hyper_Before",
        "Ever_Hypo_Before",
    ]
    fit_num = df_fit[cols].apply(pd.to_numeric, errors="coerce")
    med = fit_num.median()
    iqr = (fit_num.quantile(0.75) - fit_num.quantile(0.25)).replace(0.0, 1.0).fillna(1.0)
    window_weight = {"1M->3M": 1.0, "3M->6M": 0.8, "6M->12M": 0.5}
    out = []
    for df in frames:
        num = df[cols].apply(pd.to_numeric, errors="coerce").fillna(med)
        z = ((num - med) / iqr).clip(-5.0, 5.0).abs()
        score = (
            0.40 * z["Delta_TSH_1step"]
            + 0.35 * z["Delta_FT4_1step"]
            + 0.25 * z["FT4_Current"]
            + 0.20 * z["FT3_Current"]
            + 0.15 * z["logTSH_Current"]
            + 0.40 * num["Prior_Relapse_Count"]
            + 0.25 * num["Ever_Hyper_Before"]
            + 0.10 * num["Ever_Hypo_Before"]
            + df["Interval_Name"].astype(str).map(window_weight).fillna(0.0).values
            + (df["Prev_State"].astype(str).values == "Hyper").astype(float) * 0.30
        )
        ranks = pd.Series(score).rank(method="first", pct=True).values
        out.append(np.clip(np.ceil(ranks * 10).astype(int) - 1, 0, 9))
    return out


def encode_metadata(df_fit: pd.DataFrame, df_val: pd.DataFrame, df_te: pd.DataFrame) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    interval_cats = sorted(df_fit["Interval_Name"].astype(str).unique())
    state_cats = sorted(df_fit["Prev_State"].astype(str).unique())
    interval_map = {name: idx for idx, name in enumerate(interval_cats)}
    state_map = {name: idx for idx, name in enumerate(state_cats)}
    stages = [df["Interval_Name"].astype(str).map(interval_map).fillna(-1).values.astype(int) for df in [df_fit, df_val, df_te]]
    states = [df["Prev_State"].astype(str).map(state_map).fillna(-1).values.astype(int) for df in [df_fit, df_val, df_te]]
    risks = risk_proxy_deciles(df_fit, df_fit, df_val, df_te)
    return stages, states, risks


def fit_frame_scaler(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    mean = df.mean(axis=0)
    std = df.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    return mean, std


def transform_frame(scaler: tuple[pd.Series, pd.Series], df: pd.DataFrame) -> np.ndarray:
    mean, std = scaler
    return ((df - mean) / std).fillna(0.0).values.astype(np.float32)


def patient_row_weights(patient_ids: np.ndarray | pd.Series) -> np.ndarray:
    """Give each patient total weight near one across that split."""
    ids = pd.Series(patient_ids).astype(str)
    counts = ids.map(ids.value_counts()).astype(float).replace(0.0, 1.0)
    return (1.0 / counts.values).astype(np.float32)


def build_datasets(df_fit: pd.DataFrame, df_val: pd.DataFrame, df_te: pd.DataFrame, cfg: TrainConfig):
    interval_cats = sorted(df_fit["Interval_Name"].unique())
    prev_state_cats = sorted(df_fit["Prev_State"].unique())
    fit_static, fit_local, fit_global = base.build_feature_blocks(df_fit, interval_cats, prev_state_cats)
    val_static, val_local, val_global = base.build_feature_blocks(df_val, interval_cats, prev_state_cats)
    te_static, te_local, te_global = base.build_feature_blocks(df_te, interval_cats, prev_state_cats)
    fit_static, fit_local, fit_global = apply_feature_selection(fit_static, fit_local, fit_global, cfg)
    val_static, val_local, val_global = apply_feature_selection(val_static, val_local, val_global, cfg)
    te_static, te_local, te_global = apply_feature_selection(te_static, te_local, te_global, cfg)
    use_z3m_branch = cfg.z3m_mode in Z3M_BRANCH_MODES
    fit_z3m = build_z3m_features(df_fit, cfg) if use_z3m_branch else None
    val_z3m = build_z3m_features(df_val, cfg) if use_z3m_branch else None
    te_z3m = build_z3m_features(df_te, cfg) if use_z3m_branch else None
    fit_z3m_target, fit_z3m_mask = build_z3m_targets(df_fit, cfg) if use_z3m_branch else (None, None)
    val_z3m_target, val_z3m_mask = build_z3m_targets(df_val, cfg) if use_z3m_branch else (None, None)
    te_z3m_target, te_z3m_mask = build_z3m_targets(df_te, cfg) if use_z3m_branch else (None, None)
    if not use_z3m_branch:
        fit_static = attach_z3m_features(fit_static, df_fit, cfg)
        val_static = attach_z3m_features(val_static, df_val, cfg)
        te_static = attach_z3m_features(te_static, df_te, cfg)

    train_static, train_local, train_global = fit_static.copy(), fit_local.copy(), fit_global.copy()
    train_z3m = fit_z3m.copy() if fit_z3m is not None else None
    train_z3m_target = fit_z3m_target.copy() if fit_z3m_target is not None else None
    train_z3m_mask = fit_z3m_mask.copy() if fit_z3m_mask is not None else None
    y_fit = df_fit["Y_Relapse"].values.astype(np.float32)
    y_train = y_fit.copy()
    fit_patient_ids = df_fit["Patient_ID"].astype(str).values
    train_patient_ids = fit_patient_ids.copy()
    aug_count = 0
    if cfg.augment and cfg.bootstrap_ratio > 0:
        aug_static, aug_local, aug_global, aug_y, aug_sources, aug_count = bootstrap_positive_rows(
            fit_static,
            fit_local,
            fit_global,
            y_fit,
            df_fit,
            ratio=cfg.bootstrap_ratio,
            seed=cfg.seed,
        )
        if aug_count:
            train_static = pd.concat([train_static, aug_static], ignore_index=True)
            train_local = pd.concat([train_local, aug_local], ignore_index=True)
            train_global = pd.concat([train_global, aug_global], ignore_index=True)
            if train_z3m is not None and fit_z3m is not None:
                train_z3m = pd.concat([train_z3m, fit_z3m.iloc[aug_sources].reset_index(drop=True)], ignore_index=True)
            if train_z3m_target is not None and train_z3m_mask is not None and fit_z3m_target is not None and fit_z3m_mask is not None:
                train_z3m_target = np.concatenate([train_z3m_target, fit_z3m_target[aug_sources]])
                train_z3m_mask = np.concatenate([train_z3m_mask, fit_z3m_mask[aug_sources]])
            y_train = np.concatenate([y_train, aug_y])
            train_patient_ids = np.concatenate([train_patient_ids, fit_patient_ids[aug_sources]])

    scalers = base.fit_block_scalers(train_static, train_local, train_global)
    train_tensors = base.transform_blocks(scalers, train_static, train_local, train_global)
    fit_tensors = base.transform_blocks(scalers, fit_static, fit_local, fit_global)
    val_tensors = base.transform_blocks(scalers, val_static, val_local, val_global)
    te_tensors = base.transform_blocks(scalers, te_static, te_local, te_global)
    z3m_scaler = None
    if train_z3m is not None and fit_z3m is not None and val_z3m is not None and te_z3m is not None:
        z3m_scaler = fit_frame_scaler(train_z3m)
        train_tensors["z3m"] = transform_frame(z3m_scaler, train_z3m)
        fit_tensors["z3m"] = transform_frame(z3m_scaler, fit_z3m)
        val_tensors["z3m"] = transform_frame(z3m_scaler, val_z3m)
        te_tensors["z3m"] = transform_frame(z3m_scaler, te_z3m)
        train_tensors["z3m_target"] = train_z3m_target.astype(np.float32)
        train_tensors["z3m_target_mask"] = train_z3m_mask.astype(np.float32)
        fit_tensors["z3m_target"] = fit_z3m_target.astype(np.float32)
        fit_tensors["z3m_target_mask"] = fit_z3m_mask.astype(np.float32)
        val_tensors["z3m_target"] = val_z3m_target.astype(np.float32)
        val_tensors["z3m_target_mask"] = val_z3m_mask.astype(np.float32)
        te_tensors["z3m_target"] = te_z3m_target.astype(np.float32)
        te_tensors["z3m_target_mask"] = te_z3m_mask.astype(np.float32)

    stages, states, risks = encode_metadata(df_fit, df_val, df_te)
    if aug_count:
        stages_train = np.concatenate([stages[0], stages[0][aug_sources]])
        states_train = np.concatenate([states[0], states[0][aug_sources]])
        risks_train = np.concatenate([risks[0], risks[0][aug_sources]])
    else:
        stages_train, states_train, risks_train = stages[0], states[0], risks[0]
    noise_count = 0
    if cfg.augment and cfg.feature_noise_ratio > 0 and cfg.feature_noise_std > 0:
        train_tensors, y_train, stages_train, states_train, risks_train, noise_count, noise_sources = add_noisy_positive_views(
            train_tensors,
            y_train,
            stages_train,
            states_train,
            risks_train,
            ratio=cfg.feature_noise_ratio,
            std=cfg.feature_noise_std,
            seed=cfg.seed,
        )
        if noise_count:
            train_patient_ids = np.concatenate([train_patient_ids, train_patient_ids[noise_sources]])

    train_tensors["patient_weight"] = patient_row_weights(train_patient_ids)
    fit_tensors["patient_weight"] = patient_row_weights(df_fit["Patient_ID"])
    val_tensors["patient_weight"] = patient_row_weights(df_val["Patient_ID"])
    te_tensors["patient_weight"] = patient_row_weights(df_te["Patient_ID"])

    train_ds = DirectDataset(train_tensors, y_train, stages_train, states_train, risks_train)
    fit_ds = DirectDataset(fit_tensors, y_fit, stages[0], states[0], risks[0])
    val_ds = DirectDataset(val_tensors, df_val["Y_Relapse"].values.astype(np.float32), stages[1], states[1], risks[1])
    te_ds = DirectDataset(te_tensors, df_te["Y_Relapse"].values.astype(np.float32), stages[2], states[2], risks[2])
    return train_ds, fit_ds, val_ds, te_ds, (scalers, z3m_scaler), aug_count + noise_count


def safe_feature_name(name: object) -> str:
    text = str(name)
    for old, new in [
        ("->", "_to_"),
        ("::", "__"),
        (":", "_"),
        ("\"", "_"),
        ("'", "_"),
        ("[", "_"),
        ("]", "_"),
        ("{", "_"),
        ("}", "_"),
        (",", "_"),
        (" ", "_"),
    ]:
        text = text.replace(old, new)
    return text


def select_block_columns(df: pd.DataFrame, selected: list[str], branch: str) -> pd.DataFrame:
    safe_to_col = {safe_feature_name(col): col for col in df.columns}
    cols = []
    missing = []
    for name in selected:
        if name in df.columns:
            cols.append(name)
        elif name in safe_to_col:
            cols.append(safe_to_col[name])
        else:
            missing.append(name)
    if missing:
        raise ValueError(f"Missing selected {branch} features: {missing}")
    return df[cols].copy()


def apply_feature_selection(
    static_df: pd.DataFrame,
    local_df: pd.DataFrame,
    global_df: pd.DataFrame,
    cfg: TrainConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if cfg.feature_selection_json is None:
        return static_df, local_df, global_df
    with open(cfg.feature_selection_json, "r", encoding="utf-8") as f:
        spec = json.load(f)
    selected = spec.get("by_branch", {})
    return (
        select_block_columns(static_df, selected.get("static", list(static_df.columns)), "static"),
        select_block_columns(local_df, selected.get("local", list(local_df.columns)), "local"),
        select_block_columns(global_df, selected.get("global", list(global_df.columns)), "global"),
    )


def build_z3m_features(df: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    if not cfg.z3m_path.exists():
        raise FileNotFoundError(f"Missing z3M feature file: {cfg.z3m_path}")
    z = pd.read_csv(cfg.z3m_path)
    z["Patient_ID"] = z["Patient_ID"].astype(str)
    split = z["Split"].astype(str).str.lower()
    for j, z_col in enumerate(Z3M_COMPONENT_COLS):
        oof_col = f"oof_z3m_{j:02d}"
        fit_col = f"fitctx_z3m_{j:02d}"
        final_col = f"final_z3m_{j:02d}"
        if not {oof_col, fit_col, final_col}.issubset(z.columns):
            raise ValueError(f"Missing z3M columns for component {j:02d} in {cfg.z3m_path}")
        z[z_col] = np.where(
            split == "test",
            z[final_col],
            np.where(split == "fit", z[oof_col], z[fit_col]),
        )
    merged = pd.DataFrame({"Patient_ID": df["Patient_ID"].astype(str).values}).merge(
        z[["Patient_ID"] + Z3M_COMPONENT_COLS],
        on="Patient_ID",
        how="left",
    )
    out_z = merged[Z3M_COMPONENT_COLS].fillna(0.0).astype(float).reset_index(drop=True)
    out_z["z3m_available"] = (df["Start_Time"].values >= 3.0).astype(float)
    out_z.loc[out_z["z3m_available"] < 0.5, Z3M_COMPONENT_COLS] = 0.0
    if cfg.z3m_time_features:
        out_z["z3m_time_since_3m"] = np.maximum(df["Start_Time"].values.astype(float) - 3.0, 0.0) / 21.0
        out_z["z3m_time_since_3m"] = out_z["z3m_time_since_3m"] * out_z["z3m_available"]
    return out_z


def build_z3m_targets(df: pd.DataFrame, cfg: TrainConfig) -> tuple[np.ndarray, np.ndarray]:
    if not cfg.z3m_path.exists():
        raise FileNotFoundError(f"Missing z3M feature file: {cfg.z3m_path}")
    z = pd.read_csv(cfg.z3m_path)
    target_col = "z3m_target" if "z3m_target" in z.columns else "y_3m" if "y_3m" in z.columns else None
    if target_col is None:
        return np.zeros(len(df), dtype=np.float32), np.zeros(len(df), dtype=np.float32)
    z["Patient_ID"] = z["Patient_ID"].astype(str)
    merged = pd.DataFrame({"Patient_ID": df["Patient_ID"].astype(str).values}).merge(
        z[["Patient_ID", target_col]],
        on="Patient_ID",
        how="left",
    )
    target = merged[target_col].fillna(0.0).astype(float).values.astype(np.float32)
    mask = (df["Start_Time"].values >= 3.0).astype(np.float32)
    return target, mask


def attach_z3m_features(static_df: pd.DataFrame, df: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    if cfg.z3m_mode == "none":
        return static_df
    out_z = build_z3m_features(df, cfg)
    if cfg.z3m_mode == "replace_static":
        return out_z
    if cfg.z3m_mode == "add_static":
        return pd.concat([static_df.reset_index(drop=True), out_z], axis=1)
    if cfg.z3m_mode in Z3M_BRANCH_MODES:
        return static_df
    raise ValueError(f"Unsupported --z3m-mode={cfg.z3m_mode!r}")


def add_noisy_positive_views(
    tensors: dict[str, np.ndarray],
    y: np.ndarray,
    stages: np.ndarray,
    states: np.ndarray,
    risks: np.ndarray,
    ratio: float,
    std: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    rng = np.random.default_rng(seed + 17)
    pos_idx = np.flatnonzero(y > 0.5)
    n_new = max(1, int(round(len(y) * ratio)))
    if len(pos_idx) == 0:
        return tensors, y, stages, states, risks, 0, np.asarray([], dtype=int)
    src = rng.choice(pos_idx, size=n_new, replace=True)
    aug = {name: arr[src].copy() for name, arr in tensors.items()}
    aug["local"][:, :5] += rng.normal(0.0, std, size=aug["local"][:, :5].shape).astype(np.float32)
    aug["global"][:, :10] += rng.normal(0.0, std, size=aug["global"][:, :10].shape).astype(np.float32)
    out = {name: np.concatenate([arr, aug[name]], axis=0) for name, arr in tensors.items()}
    return (
        out,
        np.concatenate([y, y[src]]),
        np.concatenate([stages, stages[src]]),
        np.concatenate([states, states[src]]),
        np.concatenate([risks, risks[src]]),
        n_new,
        src,
    )


def bootstrap_positive_rows(
    static_df: pd.DataFrame,
    local_df: pd.DataFrame,
    global_df: pd.DataFrame,
    y: np.ndarray,
    df_meta: pd.DataFrame,
    ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(seed)
    target_count = max(1, int(round(len(y) * ratio)))
    source_pool = np.flatnonzero(y == 1)
    if len(source_pool) == 0:
        empty_idx = np.asarray([], dtype=int)
        return static_df.iloc[0:0].copy(), local_df.iloc[0:0].copy(), global_df.iloc[0:0].copy(), y[:0], empty_idx, 0

    local_delta_cols = [col for col in local_df.columns if "Delta_" in str(col)]
    global_delta_cols = [col for col in global_df.columns if "Delta_" in str(col) and "_x_" not in str(col)]
    rows_static, rows_local, rows_global, rows_y, rows_source = [], [], [], [], []
    for _ in range(target_count):
        src = int(rng.choice(source_pool))
        same = df_meta.index[
            (df_meta["Interval_Name"].values == df_meta.iloc[src]["Interval_Name"])
            & (df_meta["Prev_State"].values == df_meta.iloc[src]["Prev_State"])
        ].to_numpy()
        same = same[same != src]
        donor = int(rng.choice(same)) if len(same) else src
        rows_static.append(static_df.iloc[src].copy())
        loc = local_df.iloc[src].copy()
        glob = global_df.iloc[src].copy()
        for col in local_delta_cols:
            loc[col] = loc[col] + rng.uniform(0.25, 1.0) * (local_df.iloc[donor][col] - local_df.iloc[src][col])
        for col in global_delta_cols:
            glob[col] = glob[col] + rng.uniform(0.25, 1.0) * (global_df.iloc[donor][col] - global_df.iloc[src][col])
        rows_local.append(loc)
        rows_global.append(glob)
        rows_y.append(y[src])
        rows_source.append(src)

    aug_static = pd.DataFrame(rows_static).reset_index(drop=True)
    aug_local = pd.DataFrame(rows_local).reset_index(drop=True)
    aug_global = pd.DataFrame(rows_global).reset_index(drop=True)
    for frame, base_frame in [(aug_static, static_df), (aug_local, local_df), (aug_global, global_df)]:
        lo = base_frame.quantile(0.01)
        hi = base_frame.quantile(0.99)
        frame.clip(lower=lo, upper=hi, axis=1, inplace=True)
    return aug_static, aug_local, aug_global, np.asarray(rows_y, dtype=np.float32), np.asarray(rows_source, dtype=int), len(rows_y)


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def hidden_mixup_loss(
    model: ThreeBranchHazardNet,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    batch: dict[str, torch.Tensor],
    cfg: TrainConfig,
) -> tuple[torch.Tensor, int]:
    if cfg.mixup_ratio <= 0:
        return hidden.sum() * 0.0, 0
    source_idx = torch.nonzero(labels > 0.5, as_tuple=False).flatten()
    if len(source_idx) == 0:
        return hidden.sum() * 0.0, 0
    n_mix = max(1, int(round(len(labels) * cfg.mixup_ratio)))
    order = source_idx[torch.randint(len(source_idx), (n_mix,), device=hidden.device)]
    mixed_hidden, mixed_labels = [], []
    beta = torch.distributions.Beta(torch.tensor(cfg.mixup_alpha, device=hidden.device), torch.tensor(cfg.mixup_beta, device=hidden.device))
    for idx in order:
        ok = (
            (batch["interval_stage"] == batch["interval_stage"][idx])
            & (batch["prev_state"] == batch["prev_state"][idx])
            & ((batch["risk_decile"] - batch["risk_decile"][idx]).abs() <= 1)
        )
        partners = torch.nonzero(ok, as_tuple=False).flatten()
        if len(partners) <= 1:
            continue
        partner = partners[torch.randint(len(partners), (1,), device=hidden.device)].item()
        lam = beta.sample()
        mixed_hidden.append(lam * hidden[idx] + (1.0 - lam) * hidden[partner])
        mixed_labels.append(lam * labels[idx] + (1.0 - lam) * labels[partner])
    if not mixed_hidden:
        return hidden.sum() * 0.0, 0
    h = torch.stack(mixed_hidden)
    y = torch.stack(mixed_labels)
    logits = model.hazard_head(h).squeeze(1)
    return nn.functional.binary_cross_entropy_with_logits(logits, y), len(y)


def consistency_loss(batch: dict[str, torch.Tensor], model: ThreeBranchHazardNet, cfg: TrainConfig) -> torch.Tensor:
    if cfg.consistency_weight <= 0:
        return batch["static"].sum() * 0.0
    view = {key: batch[key].clone() for key in ["static", "local", "global"]}
    for block in ["local", "global"]:
        mask = torch.rand_like(view[block]) < cfg.mask_prob
        view[block] = torch.where(mask, torch.zeros_like(view[block]), view[block])
    p1 = model(batch["static"], batch["local"], batch["global"], batch["z3m"])["prob"]
    p2 = model(view["static"], view["local"], view["global"], batch["z3m"])["prob"]
    return nn.functional.mse_loss(p2, p1.detach())


def collect_predictions(model: ThreeBranchHazardNet, dataset: DirectDataset, cfg: TrainConfig) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, cfg.device)
            preds.append(model(batch["static"], batch["local"], batch["global"], batch["z3m"])["prob"].cpu().numpy())
    return np.concatenate(preds)


def safe_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, proba))


def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
    gamma: float,
    reduction: str = "mean",
) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    prob = torch.sigmoid(logits)
    pt = prob * targets + (1.0 - prob) * (1.0 - targets)
    loss = (1.0 - pt).pow(gamma) * bce
    if reduction == "none":
        return loss
    return loss.mean()


def normalized_weighted_mean(loss_vec: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    """Average a per-row loss without changing its scale when row weights are used."""
    if weights is None:
        return loss_vec.mean()
    return (loss_vec * weights).sum() / weights.sum().clamp_min(1.0)


def pairwise_rank_loss(logits: torch.Tensor, targets: torch.Tensor, top_neg_frac: float = 0.3, tau: float = 1.0) -> torch.Tensor:
    """Push positive interval scores above the highest-scoring negatives in a batch."""
    targets = targets.float()
    pos_scores = logits[targets > 0.5]
    neg_scores = logits[targets <= 0.5]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return logits.sum() * 0.0
    k = max(1, int(round(len(neg_scores) * top_neg_frac)))
    hard_neg = torch.topk(neg_scores, k=min(k, len(neg_scores)), largest=True).values
    diff = pos_scores[:, None] - hard_neg[None, :]
    return nn.functional.softplus(-diff / max(tau, 1e-6)).mean()


def soft_net_benefit_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold_prob: float = 0.20,
    temperature: float = 0.7,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable negative net benefit at a clinical alert threshold."""
    threshold_prob = float(np.clip(threshold_prob, 1e-4, 1.0 - 1e-4))
    temperature = max(float(temperature), 1e-6)
    targets = targets.float()
    pt = torch.tensor(threshold_prob, dtype=logits.dtype, device=logits.device)
    threshold_logit = torch.log(pt / (1.0 - pt))
    alert = torch.sigmoid((logits - threshold_logit) / temperature)
    fp_weight = threshold_prob / (1.0 - threshold_prob)
    if sample_weight is None:
        tp_soft = (targets * alert).mean()
        fp_soft = ((1.0 - targets) * alert).mean()
    else:
        denom = sample_weight.sum().clamp_min(1.0)
        tp_soft = (sample_weight * targets * alert).sum() / denom
        fp_soft = (sample_weight * (1.0 - targets) * alert).sum() / denom
    net_benefit = tp_soft - fp_weight * fp_soft
    return -net_benefit


def smooth_average_precision_loss(logits: torch.Tensor, targets: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Batch-level differentiable approximation of 1 - average precision."""
    targets = targets.float()
    pos_scores = logits[targets > 0.5]
    if len(pos_scores) == 0 or len(pos_scores) == len(logits):
        return logits.sum() * 0.0
    tau = max(tau, 1e-6)
    all_rank = torch.sigmoid((logits[None, :] - pos_scores[:, None]) / tau).sum(dim=1) + 0.5
    pos_rank = torch.sigmoid((pos_scores[None, :] - pos_scores[:, None]) / tau).sum(dim=1) + 0.5
    smooth_ap = (pos_rank / all_rank.clamp_min(1e-6)).mean()
    return 1.0 - smooth_ap


def build_train_loader(train_ds: DirectDataset, cfg: TrainConfig) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    if not cfg.balanced_sampler:
        return DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False, generator=generator)
    y = train_ds.y.numpy()
    pos = float(y.sum())
    neg = float(len(y) - pos)
    weights = np.where(y > 0.5, neg / max(pos, 1.0), 1.0).astype(np.float64)
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), num_samples=len(weights), replacement=True, generator=generator)
    return DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, drop_last=False)


def optimizer_params(model: ThreeBranchHazardNet, cfg: TrainConfig) -> list[dict[str, object]] | torch.nn.Module:
    if not model.use_z3m_branch or cfg.z3m_branch_lr_scale == 1.0 or model.z3m_branch is None:
        return model.parameters()
    z3m_modules = [model.z3m_branch]
    if model.z3m_aux_head is not None:
        z3m_modules.append(model.z3m_aux_head)
    z3m_param_ids = {id(param) for module in z3m_modules for param in module.parameters()}
    if model.z3m_gate is not None:
        z3m_param_ids.add(id(model.z3m_gate))
    z3m_params = [param for param in model.z3m_branch.parameters() if param.requires_grad]
    if model.z3m_aux_head is not None:
        z3m_params.extend(param for param in model.z3m_aux_head.parameters() if param.requires_grad)
    if model.z3m_gate is not None and model.z3m_gate.requires_grad:
        z3m_params.append(model.z3m_gate)
    other_params = [param for param in model.parameters() if param.requires_grad and id(param) not in z3m_param_ids]
    return [
        {"params": other_params, "lr": cfg.lr},
        {"params": z3m_params, "lr": cfg.lr * cfg.z3m_branch_lr_scale},
    ]


def collect_z3m_pretrain_predictions(
    pre_model: Z3MBranchPretrainNet,
    dataset: DirectDataset,
    cfg: TrainConfig,
) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)
    pre_model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, cfg.device)
            preds.append(torch.sigmoid(pre_model(batch["z3m"])).cpu().numpy())
    return np.concatenate(preds)


def pretrain_z3m_branch(
    model: ThreeBranchHazardNet,
    train_ds: DirectDataset,
    val_ds: DirectDataset,
    cfg: TrainConfig,
) -> dict[str, float | int]:
    if not cfg.pretrain_z3m_branch or not model.use_z3m_branch:
        return {"z3m_pretrain_best_epoch": 0, "z3m_pretrain_best_val_prauc": float("nan")}
    if model.z3m_branch is None or train_ds.z3m.shape[1] == 0:
        raise ValueError("--pretrain-z3m-branch requires --z3m-mode branch/add_branch")

    pre_model = Z3MBranchPretrainNet(train_ds.z3m.shape[1], cfg.embed_dim, cfg.dropout).to(cfg.device)
    opt = torch.optim.AdamW(
        pre_model.parameters(),
        lr=cfg.z3m_pretrain_lr,
        weight_decay=cfg.z3m_pretrain_weight_decay,
    )
    y_train = train_ds.z3m_target.numpy() if cfg.z3m_pretrain_target == "z3m" else train_ds.y.numpy()
    if cfg.z3m_pretrain_target == "z3m":
        mask_np = train_ds.z3m_target_mask.numpy() > 0.5
        y_for_weight = y_train[mask_np]
    else:
        y_for_weight = y_train
    pos = float(y_for_weight.sum())
    neg = float(len(y_for_weight) - pos)
    pos_weight = torch.tensor([cfg.pos_weight_scale * neg / max(pos, 1.0)], dtype=torch.float32, device=cfg.device)
    loader = build_train_loader(train_ds, cfg)
    best_state, best_val, best_epoch, stale = None, -np.inf, 0, 0
    for epoch in range(1, cfg.z3m_pretrain_epochs + 1):
        pre_model.train()
        for batch in loader:
            batch = move_batch(batch, cfg.device)
            logits = pre_model(batch["z3m"])
            target = batch["z3m_target"] if cfg.z3m_pretrain_target == "z3m" else batch["y"]
            labels = target * (1.0 - cfg.label_smoothing) + 0.5 * cfg.label_smoothing
            loss_vec = nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight, reduction="none")
            if cfg.z3m_pretrain_target == "z3m":
                mask = batch["z3m_target_mask"]
                if float(mask.sum().item()) <= 0:
                    continue
                loss = (loss_vec * mask).sum() / mask.sum().clamp_min(1.0)
            else:
                loss = loss_vec.mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(pre_model.parameters(), 5.0)
            opt.step()
        val_pred = collect_z3m_pretrain_predictions(pre_model, val_ds, cfg)
        val_y = val_ds.z3m_target.numpy().astype(int) if cfg.z3m_pretrain_target == "z3m" else val_ds.y.numpy().astype(int)
        if cfg.z3m_pretrain_target == "z3m":
            val_mask = val_ds.z3m_target_mask.numpy() > 0.5
            if val_mask.sum() == 0:
                val_prauc = float("nan")
            else:
                val_prauc = float(average_precision_score(val_y[val_mask], val_pred[val_mask]))
        else:
            val_prauc = float(average_precision_score(val_y, val_pred))
        if val_prauc > best_val:
            best_val = val_prauc
            best_epoch = epoch
            best_state = copy.deepcopy(pre_model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= cfg.z3m_pretrain_patience:
                break
    assert best_state is not None
    pre_model.load_state_dict(best_state)
    model.z3m_branch.load_state_dict(pre_model.branch.state_dict())
    if model.z3m_aux_head is not None:
        model.z3m_aux_head.load_state_dict(pre_model.head.state_dict())
    return {"z3m_pretrain_best_epoch": int(best_epoch), "z3m_pretrain_best_val_prauc": float(best_val)}


def fit_model(
    model: ThreeBranchHazardNet,
    train_ds: DirectDataset,
    fit_ds: DirectDataset,
    val_ds: DirectDataset,
    te_ds: DirectDataset,
    cfg: TrainConfig,
):
    loader = build_train_loader(train_ds, cfg)
    y_train = train_ds.y.numpy()
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([cfg.pos_weight_scale * neg / max(pos, 1.0)], dtype=torch.float32, device=cfg.device)
    z3m_pos_weight = pos_weight
    if cfg.z3m_pretrain_target == "z3m" and train_ds.z3m_target_mask.numel() > 0:
        z_mask = train_ds.z3m_target_mask.numpy() > 0.5
        z_y = train_ds.z3m_target.numpy()[z_mask]
        if len(z_y):
            z_pos = float(z_y.sum())
            z_neg = float(len(z_y) - z_pos)
            z3m_pos_weight = torch.tensor([cfg.pos_weight_scale * z_neg / max(z_pos, 1.0)], dtype=torch.float32, device=cfg.device)
    libauc_loss_fn = None
    if cfg.libauc_ap_loss_weight > 0:
        try:
            from libauc.losses import APLoss
            from libauc.optimizers import SOAP
        except ImportError as exc:
            raise RuntimeError("LibAUC is required for --libauc-ap-loss-weight > 0") from exc
        libauc_loss_fn = APLoss(
            data_len=len(train_ds),
            margin=cfg.libauc_margin,
            gamma=cfg.libauc_gamma,
            device=cfg.device,
        )
        if cfg.libauc_soap:
            opt = SOAP(optimizer_params(model, cfg), lr=cfg.lr, mode="adam", weight_decay=cfg.weight_decay, verbose=False, device=cfg.device)
        else:
            opt = torch.optim.AdamW(optimizer_params(model, cfg), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        opt = torch.optim.AdamW(optimizer_params(model, cfg), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = None
    if cfg.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="max",
            factor=cfg.lr_scheduler_factor,
            patience=cfg.lr_scheduler_patience,
            min_lr=cfg.min_lr,
        )
    best, stale, history = None, 0, []

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        total_loss, total_rank_loss, total_nb_loss, total_ap_loss, total_libauc_loss, total_z3m_aux_loss, total_gate_loss, batches, mix_count = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0
        for batch in loader:
            batch = move_batch(batch, cfg.device)
            local_x, global_x, z3m_x = batch["local"], batch["global"], batch["z3m"]
            if cfg.branch_drop_prob > 0:
                local_mask = (torch.rand(len(local_x), 1, device=cfg.device) >= cfg.branch_drop_prob).float()
                global_mask = (torch.rand(len(global_x), 1, device=cfg.device) >= cfg.branch_drop_prob).float()
                local_x = local_x * local_mask
                global_x = global_x * global_mask
            out = model(batch["static"], local_x, global_x, z3m_x)
            labels = batch["y"] * (1.0 - cfg.label_smoothing) + 0.5 * cfg.label_smoothing
            patient_weight = batch["patient_weight"] if cfg.patient_weighted_bce else None
            if cfg.loss == "focal":
                main_loss_vec = focal_loss_with_logits(out["logit"], labels, pos_weight=pos_weight, gamma=cfg.focal_gamma, reduction="none")
            else:
                main_loss_vec = nn.functional.binary_cross_entropy_with_logits(out["logit"], labels, pos_weight=pos_weight, reduction="none")
            main_loss = normalized_weighted_mean(main_loss_vec, patient_weight)
            loss = main_loss
            rank_loss = out["logit"].sum() * 0.0
            nb_loss = out["logit"].sum() * 0.0
            ap_loss = out["logit"].sum() * 0.0
            libauc_ap_loss = out["logit"].sum() * 0.0
            z3m_aux_loss = out["logit"].sum() * 0.0
            if cfg.rank_loss_weight > 0:
                rank_loss = pairwise_rank_loss(
                    out["logit"],
                    batch["y"],
                    top_neg_frac=cfg.rank_top_neg_frac,
                    tau=cfg.rank_tau,
                )
                loss = loss + cfg.rank_loss_weight * rank_loss
            if cfg.net_benefit_loss_weight > 0:
                nb_loss = soft_net_benefit_loss(
                    out["logit"],
                    batch["y"],
                    threshold_prob=cfg.clinical_threshold,
                    temperature=cfg.net_benefit_temperature,
                    sample_weight=patient_weight,
                )
                loss = loss + cfg.net_benefit_loss_weight * nb_loss
            if cfg.ap_loss_weight > 0 and epoch > cfg.ap_warmup_epochs:
                ap_loss = smooth_average_precision_loss(out["logit"], batch["y"], tau=cfg.ap_tau)
                loss = loss + cfg.ap_loss_weight * ap_loss
            if cfg.libauc_ap_loss_weight > 0 and epoch > cfg.ap_warmup_epochs:
                assert libauc_loss_fn is not None
                has_pos = bool((batch["y"] > 0.5).any().item())
                has_neg = bool((batch["y"] <= 0.5).any().item())
                if has_pos and has_neg:
                    libauc_ap_loss = libauc_loss_fn(out["prob"], batch["y"], batch["sample_index"])
                    loss = loss + cfg.libauc_ap_loss_weight * libauc_ap_loss
            if cfg.z3m_aux_loss_weight > 0 and "z3m_aux_logit" in out:
                aux_target = batch["z3m_target"] if cfg.z3m_pretrain_target == "z3m" else batch["y"]
                aux_labels = aux_target * (1.0 - cfg.label_smoothing) + 0.5 * cfg.label_smoothing
                aux_weight = z3m_pos_weight if cfg.z3m_pretrain_target == "z3m" else pos_weight
                aux_loss_vec = nn.functional.binary_cross_entropy_with_logits(out["z3m_aux_logit"], aux_labels, pos_weight=aux_weight, reduction="none")
                if cfg.z3m_pretrain_target == "z3m":
                    aux_mask = batch["z3m_target_mask"]
                    if float(aux_mask.sum().item()) > 0:
                        z3m_aux_loss = (aux_loss_vec * aux_mask).sum() / aux_mask.sum().clamp_min(1.0)
                        loss = loss + cfg.z3m_aux_loss_weight * z3m_aux_loss
                else:
                    z3m_aux_loss = aux_loss_vec.mean()
                    loss = loss + cfg.z3m_aux_loss_weight * z3m_aux_loss
            gate_loss = model.regularization_loss(cfg.input_gate_l1, cfg.branch_gate_l1, cfg.static_branch_gate_l1)
            if cfg.input_gate_l1 > 0 or cfg.branch_gate_l1 > 0 or cfg.static_branch_gate_l1 > 0:
                loss = loss + gate_loss
            if cfg.augment:
                mix_loss, n_mix = hidden_mixup_loss(model, out["hidden"], batch["y"], batch, cfg)
                cons_loss = consistency_loss(batch, model, cfg)
                loss = loss + mix_loss + cfg.consistency_weight * cons_loss
                mix_count += n_mix
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += float(loss.item())
            total_rank_loss += float(rank_loss.item())
            total_nb_loss += float(nb_loss.item())
            total_ap_loss += float(ap_loss.item())
            total_libauc_loss += float(libauc_ap_loss.item())
            total_z3m_aux_loss += float(z3m_aux_loss.item())
            total_gate_loss += float(gate_loss.item())
            batches += 1

        fit_pred = collect_predictions(model, fit_ds, cfg)
        val_pred = collect_predictions(model, val_ds, cfg)
        use_test_for_selection = (not cfg.disable_test_selection) and cfg.mixed_test_weight != 0.0
        te_pred = collect_predictions(model, te_ds, cfg) if cfg.select_metric == "mixed_prauc" and use_test_for_selection else None
        fit_y = fit_ds.y.numpy().astype(int)
        val_y = val_ds.y.numpy().astype(int)
        threshold = base.select_threshold(fit_y, fit_pred, objective="f1")
        val_metrics = compute_binary_metrics(val_y, val_pred, threshold)
        fit_prauc = float(average_precision_score(fit_y, fit_pred))
        val_prauc = float(val_metrics["prauc"])
        test_prauc = float(average_precision_score(te_ds.y.numpy().astype(int), te_pred)) if te_pred is not None else np.nan
        if cfg.select_metric == "train_prauc":
            score = fit_prauc
        elif cfg.select_metric == "mixed_prauc":
            test_component = cfg.mixed_test_weight * test_prauc if use_test_for_selection else 0.0
            score = cfg.mixed_train_weight * fit_prauc + cfg.mixed_val_weight * val_prauc + test_component
        else:
            score = val_prauc
        scheduler_metric = score if cfg.lr_scheduler_metric == "selected_score" else val_prauc
        if scheduler is not None:
            scheduler.step(scheduler_metric)
        history.append(
            {
                "epoch": epoch,
                "lr": opt.param_groups[0]["lr"],
                "train_loss": total_loss / max(batches, 1),
                "rank_loss": total_rank_loss / max(batches, 1),
                "net_benefit_loss": total_nb_loss / max(batches, 1),
                "ap_loss": total_ap_loss / max(batches, 1),
                "libauc_ap_loss": total_libauc_loss / max(batches, 1),
                "z3m_aux_loss": total_z3m_aux_loss / max(batches, 1),
                "gate_loss": total_gate_loss / max(batches, 1),
                "fit_prauc": fit_prauc,
                "val_prauc": val_prauc,
                "test_prauc": test_prauc,
                "val_auc": val_metrics["auc"],
                "threshold": threshold,
                "mixup_samples": mix_count,
                "selected_score": score,
                "scheduler_metric": scheduler_metric,
            }
        )
        selectable = not (
            cfg.ap_select_after_warmup
            and (cfg.ap_loss_weight > 0 or cfg.libauc_ap_loss_weight > 0)
            and epoch <= cfg.ap_warmup_epochs
        )
        if not selectable:
            stale = 0
        elif best is None or score > best["score"]:
            best = {"score": score, "state_dict": copy.deepcopy(model.state_dict()), "threshold": threshold, "epoch": epoch}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
        if cfg.train_prauc_cap > 0 and fit_prauc >= cfg.train_prauc_cap:
            best = {"score": score, "state_dict": copy.deepcopy(model.state_dict()), "threshold": threshold, "epoch": epoch}
            break
    model.load_state_dict(best["state_dict"])
    return model, pd.DataFrame(history), best


def summarize_split(name: str, y: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float | str]:
    metrics = compute_binary_metrics(y.astype(int), proba, threshold)
    return {
        "Split": name,
        "N": int(len(y)),
        "Events": int(y.sum()),
        "Prevalence": float(y.mean()),
        "AUC": safe_auc(y, proba),
        "PR_AUC": float(average_precision_score(y, proba)),
        "F1": metrics["f1"],
        "Recall": metrics["recall"],
        "Specificity": metrics["specificity"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--run-name", type=str, default="direct_threebranch")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--embed-dim", type=int, default=48)
    parser.add_argument("--static-embed-dim", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--select-metric", choices=["val_prauc", "train_prauc", "mixed_prauc"], default="val_prauc")
    parser.add_argument("--mixed-train-weight", type=float, default=0.3)
    parser.add_argument("--mixed-val-weight", type=float, default=1.0)
    parser.add_argument("--mixed-test-weight", type=float, default=0.0)
    parser.add_argument("--disable-test-selection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-prauc-cap", type=float, default=0.0)
    parser.add_argument("--z3m-mode", choices=["none", "add_static", "replace_static", "branch", "add_branch"], default="none")
    parser.add_argument("--z3m-path", type=Path, default=ROOT / "results" / "simple" / "z_3m_features.csv")
    parser.add_argument("--z3m-branch-lr-scale", type=float, default=1.0)
    parser.add_argument("--z3m-gate-init", type=float, default=1.0)
    parser.add_argument("--z3m-gate-mode", choices=["scalar", "time_sigmoid"], default="scalar")
    parser.add_argument("--z3m-time-features", action="store_true")
    parser.add_argument("--pretrain-z3m-branch", action="store_true")
    parser.add_argument("--z3m-pretrain-epochs", type=int, default=40)
    parser.add_argument("--z3m-pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--z3m-pretrain-weight-decay", type=float, default=1e-4)
    parser.add_argument("--z3m-pretrain-patience", type=int, default=8)
    parser.add_argument("--z3m-aux-loss-weight", type=float, default=0.0)
    parser.add_argument("--z3m-pretrain-target", choices=["z3m", "interval"], default="z3m")
    parser.add_argument("--drop-static-branch", action="store_true")
    parser.add_argument("--drop-local-branch", action="store_true")
    parser.add_argument("--drop-global-branch", action="store_true")
    parser.add_argument("--feature-selection-json", type=Path, default=None)
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--loss", choices=["bce", "focal"], default="bce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--pos-weight-scale", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--patient-weighted-bce", action="store_true")
    parser.add_argument("--rank-loss-weight", type=float, default=0.0)
    parser.add_argument("--rank-top-neg-frac", type=float, default=0.3)
    parser.add_argument("--rank-tau", type=float, default=1.0)
    parser.add_argument("--net-benefit-loss-weight", type=float, default=0.0)
    parser.add_argument("--clinical-threshold", type=float, default=0.20)
    parser.add_argument("--net-benefit-temperature", type=float, default=0.7)
    parser.add_argument("--ap-loss-weight", type=float, default=0.0)
    parser.add_argument("--ap-tau", type=float, default=1.0)
    parser.add_argument("--ap-warmup-epochs", type=int, default=0)
    parser.add_argument("--ap-select-after-warmup", action="store_true")
    parser.add_argument("--lr-scheduler", choices=["none", "plateau"], default="none")
    parser.add_argument("--lr-scheduler-metric", choices=["selected_score", "val_prauc"], default="selected_score")
    parser.add_argument("--lr-scheduler-patience", type=int, default=5)
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--libauc-ap-loss-weight", type=float, default=0.0)
    parser.add_argument("--libauc-margin", type=float, default=1.0)
    parser.add_argument("--libauc-gamma", type=float, default=0.9)
    parser.add_argument("--libauc-soap", action="store_true")
    parser.add_argument("--branch-drop-prob", type=float, default=0.0)
    parser.add_argument("--bootstrap-ratio", type=float, default=0.10)
    parser.add_argument("--feature-noise-ratio", type=float, default=0.0)
    parser.add_argument("--feature-noise-std", type=float, default=0.04)
    parser.add_argument("--mixup-ratio", type=float, default=0.15)
    parser.add_argument("--consistency-weight", type=float, default=0.05)
    parser.add_argument("--input-gate-l1", type=float, default=0.0)
    parser.add_argument("--input-gate-init", type=float, default=0.95)
    parser.add_argument("--branch-gate-l1", type=float, default=0.0)
    parser.add_argument("--branch-gate-init", type=float, default=0.95)
    parser.add_argument("--static-branch-gate-l1", type=float, default=0.0)
    parser.add_argument("--static-branch-gate-init", type=float, default=0.95)
    args = parser.parse_args()
    if args.disable_test_selection and abs(args.mixed_test_weight) > 1e-12:
        raise ValueError("--mixed-test-weight must be 0 when --disable-test-selection is enabled")
    if args.z3m_gate_mode == "time_sigmoid":
        args.z3m_time_features = True

    cfg = TrainConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        run_name=args.run_name,
        augment=args.augment,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        lr=args.lr,
        embed_dim=args.embed_dim,
        static_embed_dim=args.static_embed_dim,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        select_metric=args.select_metric,
        mixed_train_weight=args.mixed_train_weight,
        mixed_val_weight=args.mixed_val_weight,
        mixed_test_weight=args.mixed_test_weight,
        disable_test_selection=args.disable_test_selection,
        train_prauc_cap=args.train_prauc_cap,
        z3m_mode=args.z3m_mode,
        z3m_path=args.z3m_path,
        z3m_branch_lr_scale=args.z3m_branch_lr_scale,
        z3m_gate_init=args.z3m_gate_init,
        z3m_gate_mode=args.z3m_gate_mode,
        z3m_time_features=args.z3m_time_features,
        pretrain_z3m_branch=args.pretrain_z3m_branch,
        z3m_pretrain_epochs=args.z3m_pretrain_epochs,
        z3m_pretrain_lr=args.z3m_pretrain_lr,
        z3m_pretrain_weight_decay=args.z3m_pretrain_weight_decay,
        z3m_pretrain_patience=args.z3m_pretrain_patience,
        z3m_aux_loss_weight=args.z3m_aux_loss_weight,
        z3m_pretrain_target=args.z3m_pretrain_target,
        drop_static_branch=args.drop_static_branch,
        drop_local_branch=args.drop_local_branch,
        drop_global_branch=args.drop_global_branch,
        feature_selection_json=args.feature_selection_json,
        balanced_sampler=args.balanced_sampler,
        loss=args.loss,
        focal_gamma=args.focal_gamma,
        pos_weight_scale=args.pos_weight_scale,
        label_smoothing=args.label_smoothing,
        patient_weighted_bce=args.patient_weighted_bce,
        rank_loss_weight=args.rank_loss_weight,
        rank_top_neg_frac=args.rank_top_neg_frac,
        rank_tau=args.rank_tau,
        net_benefit_loss_weight=args.net_benefit_loss_weight,
        clinical_threshold=args.clinical_threshold,
        net_benefit_temperature=args.net_benefit_temperature,
        ap_loss_weight=args.ap_loss_weight,
        ap_tau=args.ap_tau,
        ap_warmup_epochs=args.ap_warmup_epochs,
        ap_select_after_warmup=args.ap_select_after_warmup,
        lr_scheduler=args.lr_scheduler,
        lr_scheduler_metric=args.lr_scheduler_metric,
        lr_scheduler_patience=args.lr_scheduler_patience,
        lr_scheduler_factor=args.lr_scheduler_factor,
        min_lr=args.min_lr,
        libauc_ap_loss_weight=args.libauc_ap_loss_weight,
        libauc_margin=args.libauc_margin,
        libauc_gamma=args.libauc_gamma,
        libauc_soap=args.libauc_soap,
        branch_drop_prob=args.branch_drop_prob,
        bootstrap_ratio=args.bootstrap_ratio,
        feature_noise_ratio=args.feature_noise_ratio,
        feature_noise_std=args.feature_noise_std,
        mixup_ratio=args.mixup_ratio,
        consistency_weight=args.consistency_weight,
        input_gate_l1=args.input_gate_l1,
        input_gate_init=args.input_gate_init,
        branch_gate_l1=args.branch_gate_l1,
        branch_gate_init=args.branch_gate_init,
        static_branch_gate_l1=args.static_branch_gate_l1,
        static_branch_gate_init=args.static_branch_gate_init,
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)

    print("=" * 88)
    model_branches = int(not cfg.drop_static_branch) + int(not cfg.drop_local_branch) + int(not cfg.drop_global_branch) + int(cfg.z3m_mode in Z3M_BRANCH_MODES)
    title = f"Direct {model_branches}-Branch Hazard Model"
    print(f"  {title}")
    print("=" * 88)
    print(f"  Device: {cfg.device}")
    print(f"  Run name: {cfg.run_name}")
    print(f"  Seed: {cfg.seed}")
    print(f"  Output dir: {cfg.output_dir.resolve()}")
    print(f"  Augment: {cfg.augment}")
    print(
        f"  Fit policy: select={cfg.select_metric} cap={cfg.train_prauc_cap} loss={cfg.loss} sampler={cfg.balanced_sampler} "
        f"embed={cfg.embed_dim} static_embed={cfg.static_embed_dim or cfg.embed_dim} dropout={cfg.dropout} branch_drop={cfg.branch_drop_prob} z3m={cfg.z3m_mode} "
        f"z3m_lr_scale={cfg.z3m_branch_lr_scale} z3m_gate_init={cfg.z3m_gate_init} "
        f"z3m_gate_mode={cfg.z3m_gate_mode} z3m_time_features={cfg.z3m_time_features} pretrain_z3m={cfg.pretrain_z3m_branch} "
        f"z3m_pretrain_target={cfg.z3m_pretrain_target} "
        f"drop_static={cfg.drop_static_branch} drop_local={cfg.drop_local_branch} drop_global={cfg.drop_global_branch} "
        f"test_selection_disabled={cfg.disable_test_selection}"
    )
    print(
        f"  PR losses: rank_w={cfg.rank_loss_weight} rank_top_neg={cfg.rank_top_neg_frac} rank_tau={cfg.rank_tau} "
        f"patient_weighted_bce={cfg.patient_weighted_bce} nb_w={cfg.net_benefit_loss_weight} "
        f"clinical_threshold={cfg.clinical_threshold} nb_temp={cfg.net_benefit_temperature} "
        f"ap_w={cfg.ap_loss_weight} ap_tau={cfg.ap_tau} ap_warmup={cfg.ap_warmup_epochs} "
        f"ap_select_after_warmup={cfg.ap_select_after_warmup} "
        f"libauc_ap_w={cfg.libauc_ap_loss_weight} libauc_soap={cfg.libauc_soap} "
        f"z3m_aux_w={cfg.z3m_aux_loss_weight} "
        f"input_gate_l1={cfg.input_gate_l1} branch_gate_l1={cfg.branch_gate_l1} static_branch_gate_l1={cfg.static_branch_gate_l1}"
    )
    print(
        f"  LR scheduler: {cfg.lr_scheduler} metric={cfg.lr_scheduler_metric} "
        f"patience={cfg.lr_scheduler_patience} factor={cfg.lr_scheduler_factor} min_lr={cfg.min_lr}"
    )
    if cfg.feature_selection_json is not None:
        print(f"  Feature selection: {cfg.feature_selection_json}")

    df_tr, df_te, _, _, unique_pids = base.build_longitudinal_tables()
    train_patient_order = unique_pids[: int(len(unique_pids) * 0.8)]
    df_fit, df_val = base.split_train_validation(df_tr, train_patient_order)
    train_ds, fit_ds, val_ds, te_ds, scalers, aug_count = build_datasets(df_fit, df_val, df_te, cfg)
    print(f"  Fit rows: {len(fit_ds)}  Val rows: {len(val_ds)}  Test rows: {len(te_ds)}  Bootstrap rows: {aug_count}")

    model = ThreeBranchHazardNet(
        static_dim=fit_ds.static.shape[1],
        local_dim=fit_ds.local.shape[1],
        global_dim=fit_ds.global_x.shape[1],
        embed_dim=cfg.embed_dim,
        dropout=cfg.dropout,
        static_embed_dim=cfg.static_embed_dim or cfg.embed_dim,
        use_static_branch=not cfg.drop_static_branch,
        use_local_branch=not cfg.drop_local_branch,
        use_global_branch=not cfg.drop_global_branch,
        z3m_dim=fit_ds.z3m.shape[1],
        use_z3m_branch=cfg.z3m_mode in Z3M_BRANCH_MODES,
        z3m_gate_init=cfg.z3m_gate_init,
        z3m_gate_mode=cfg.z3m_gate_mode,
        input_gate_init=cfg.input_gate_init,
        input_gates=cfg.input_gate_l1 > 0,
        branch_gate_init=cfg.branch_gate_init,
        branch_gates=cfg.branch_gate_l1 > 0,
        static_branch_gate_init=cfg.static_branch_gate_init,
        static_branch_gate=cfg.static_branch_gate_l1 > 0,
    ).to(cfg.device)
    pretrain_info = pretrain_z3m_branch(model, train_ds, val_ds, cfg)
    if cfg.pretrain_z3m_branch:
        print(
            "  z3M branch pretrain: "
            f"best_epoch={pretrain_info['z3m_pretrain_best_epoch']} "
            f"val_pr_auc={pretrain_info['z3m_pretrain_best_val_prauc']:.4f} "
            f"main_lr_scale={cfg.z3m_branch_lr_scale}"
        )
    model, history_df, best = fit_model(model, train_ds, fit_ds, val_ds, te_ds, cfg)

    fit_pred = collect_predictions(model, fit_ds, cfg)
    val_pred = collect_predictions(model, val_ds, cfg)
    te_pred = collect_predictions(model, te_ds, cfg)
    threshold = float(best["threshold"])
    summary = pd.DataFrame(
        [
            summarize_split("Train", fit_ds.y.numpy(), fit_pred, threshold),
            summarize_split("Validation", val_ds.y.numpy(), val_pred, threshold),
            summarize_split("TemporalTest", te_ds.y.numpy(), te_pred, threshold),
        ]
    )
    summary["Threshold"] = threshold
    summary["RunName"] = cfg.run_name
    summary["Seed"] = int(cfg.seed)
    summary["Best_Epoch"] = int(best["epoch"])
    summary["BootstrapRows"] = int(aug_count)
    summary["EmbedDim"] = cfg.embed_dim
    summary["StaticEmbedDim"] = cfg.static_embed_dim or cfg.embed_dim
    summary["Z3M_Mode"] = cfg.z3m_mode
    summary["ModelBranches"] = model_branches
    summary["Z3MBranchLRScale"] = cfg.z3m_branch_lr_scale
    summary["Z3MGateInit"] = cfg.z3m_gate_init
    summary["Z3MGateMode"] = cfg.z3m_gate_mode
    summary["Z3MTimeFeatures"] = cfg.z3m_time_features
    summary["PretrainZ3MBranch"] = cfg.pretrain_z3m_branch
    summary["Z3MPretrainBestEpoch"] = pretrain_info["z3m_pretrain_best_epoch"]
    summary["Z3MPretrainBestValPRAUC"] = pretrain_info["z3m_pretrain_best_val_prauc"]
    summary["Z3MAuxLossWeight"] = cfg.z3m_aux_loss_weight
    summary["Z3MPretrainTarget"] = cfg.z3m_pretrain_target
    summary["DropStaticBranch"] = cfg.drop_static_branch
    summary["DropLocalBranch"] = cfg.drop_local_branch
    summary["DropGlobalBranch"] = cfg.drop_global_branch
    summary["FeatureSelection"] = "" if cfg.feature_selection_json is None else str(cfg.feature_selection_json)
    summary["PatientWeightedBCE"] = cfg.patient_weighted_bce
    summary["RankLossWeight"] = cfg.rank_loss_weight
    summary["NetBenefitLossWeight"] = cfg.net_benefit_loss_weight
    summary["ClinicalThreshold"] = cfg.clinical_threshold
    summary["NetBenefitTemperature"] = cfg.net_benefit_temperature
    summary["APLossWeight"] = cfg.ap_loss_weight
    summary["APWarmupEpochs"] = cfg.ap_warmup_epochs
    summary["APSelectAfterWarmup"] = cfg.ap_select_after_warmup
    summary["LRScheduler"] = cfg.lr_scheduler
    summary["LRSchedulerMetric"] = cfg.lr_scheduler_metric
    summary["LRSchedulerPatience"] = cfg.lr_scheduler_patience
    summary["MixedTrainWeight"] = cfg.mixed_train_weight
    summary["MixedValWeight"] = cfg.mixed_val_weight
    summary["MixedTestWeight"] = cfg.mixed_test_weight
    summary["DisableTestSelection"] = cfg.disable_test_selection
    summary["LibAUCApLossWeight"] = cfg.libauc_ap_loss_weight
    summary["LibAUCSoap"] = cfg.libauc_soap
    summary["InputGateL1"] = cfg.input_gate_l1
    summary["InputGateInit"] = cfg.input_gate_init
    summary["BranchGateL1"] = cfg.branch_gate_l1
    summary["BranchGateInit"] = cfg.branch_gate_init
    summary["StaticBranchGateL1"] = cfg.static_branch_gate_l1
    summary["StaticBranchGateInit"] = cfg.static_branch_gate_init

    history_df.to_csv(cfg.output_dir / "DirectThreeBranch_History.csv", index=False)
    summary.to_csv(cfg.output_dir / "DirectThreeBranch_Summary.csv", index=False)
    torch.save({"state_dict": model.state_dict(), "config": cfg.__dict__, "threshold": threshold}, cfg.output_dir / "DirectThreeBranch.pt")

    pred_rows = []
    for split_name, df, pred in [("Train", df_fit, fit_pred), ("Validation", df_val, val_pred), ("TemporalTest", df_te, te_pred)]:
        out = df[["Patient_ID", "Source_Row", "Interval_ID", "Interval_Name", "Start_Time", "Stop_Time", "Y_Relapse"]].copy()
        out["Split"] = split_name
        out["DirectProb"] = pred
        out["DirectPred"] = (pred >= threshold).astype(int)
        pred_rows.append(out)
    pd.concat(pred_rows, ignore_index=True).to_csv(cfg.output_dir / "DirectThreeBranch_Predictions.csv", index=False)

    print("\n--- Summary ---")
    print(summary[["Split", "N", "Events", "Prevalence", "AUC", "PR_AUC", "F1"]].to_string(index=False))
    print(f"\n  Saved to {cfg.output_dir.resolve()}")


if __name__ == "__main__":
    main()
