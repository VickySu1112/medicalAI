"""Add chapter-5 explainability figures to the dynamic relapse report.

The script is report-local: it reads the selected four-branch checkpoint and
prediction artifacts, rebuilds the transformed tensors, computes development
SHAP explanations with a temporal-test consistency check, upgrades Figure 19
to a marginal heatmap, and refreshes the public Markdown/HTML report.
"""

from __future__ import annotations

import math
import subprocess
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch.utils.data import DataLoader

from scripts import relapse_direct_threebranch_aug as direct
from scripts import relapse_threehead_landmark as base


REPORT = ROOT / "results" / "t5_dynamic_paper"
FIG = REPORT / "figures"
TAB = REPORT / "tables"
README = REPORT / "复发.md"
HTML = REPORT / "复发.html"
CKPT = ROOT / "results" / "t5_seed2025_paper" / "DirectThreeBranch.pt"
PRED = ROOT / "results" / "t5_seed2025_paper" / "DirectThreeBranch_Predictions.csv"
SUMMARY = ROOT / "results" / "t5_seed2025_paper" / "DirectThreeBranch_Summary.csv"
THRESHOLD = 0.785
SPLITS = ["Train", "Validation", "TemporalTest"]
SPLIT_LABEL = {"Train": "Train", "Validation": "Validation", "TemporalTest": "Temporal test"}
RISK_CMAP = plt.cm.YlOrRd
RISK_NORM = plt.Normalize(vmin=0.0, vmax=1.0)
RISK_TICKS = np.linspace(0.0, 1.0, 6)
MISSING_COLOR = "#E5E7EB"
CH5_START = "<!-- CH5_EXPLAINABILITY_START -->"
CH5_END = "<!-- CH5_EXPLAINABILITY_END -->"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "Heiti TC", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 140,
            "savefig.dpi": 260,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.25,
        }
    )


def save(fig: plt.Figure, name: str, *, tight: bool = True) -> None:
    if tight:
        fig.tight_layout()
    fig.savefig(FIG / name, dpi=260, bbox_inches="tight")
    plt.close(fig)


def to_path(value: object) -> object:
    if value in (None, ""):
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return Path(value)


def load_cfg_and_checkpoint() -> tuple[direct.TrainConfig, dict[str, torch.Tensor]]:
    checkpoint = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg_dict = dict(checkpoint["config"])
    cfg_dict["output_dir"] = to_path(cfg_dict.get("output_dir"))
    cfg_dict["z3m_path"] = to_path(cfg_dict.get("z3m_path"))
    cfg_dict["feature_selection_json"] = to_path(cfg_dict.get("feature_selection_json"))
    cfg = direct.TrainConfig(**cfg_dict)
    cfg.device = "cpu"
    cfg.batch_size = 512
    cfg.augment = False
    return cfg, checkpoint["state_dict"]


def build_split_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    df_tr, df_te, _, _, unique_pids = base.build_longitudinal_tables()
    train_patient_order = unique_pids[: int(len(unique_pids) * 0.8)]
    df_fit, df_val = base.split_train_validation(df_tr, train_patient_order)
    return df_fit, df_val, df_te, unique_pids


def rebuild_model_data() -> tuple[direct.ThreeBranchHazardNet, dict[str, direct.DirectDataset], direct.TrainConfig, pd.DataFrame, dict[str, list[str]]]:
    cfg, state_dict = load_cfg_and_checkpoint()
    direct.seed_everything(cfg.seed)
    df_fit, df_val, df_te, _ = build_split_frames()
    _, fit_ds, val_ds, te_ds, _, _ = direct.build_datasets(df_fit, df_val, df_te, cfg)
    model = direct.ThreeBranchHazardNet(
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
        use_z3m_branch=cfg.z3m_mode in direct.Z3M_BRANCH_MODES,
        z3m_gate_init=cfg.z3m_gate_init,
        z3m_gate_mode=cfg.z3m_gate_mode,
        input_gate_init=cfg.input_gate_init,
        input_gates=cfg.input_gate_l1 > 0,
        branch_gate_init=cfg.branch_gate_init,
        branch_gates=cfg.branch_gate_l1 > 0,
        static_branch_gate_init=cfg.static_branch_gate_init,
        static_branch_gate=cfg.static_branch_gate_l1 > 0,
    )
    model.load_state_dict(state_dict)
    model.eval()
    frames = []
    for split, df in [("Train", df_fit), ("Validation", df_val), ("TemporalTest", df_te)]:
        tmp = df.copy()
        tmp["Split"] = split
        frames.append(tmp)
    feature_df = pd.concat(frames, ignore_index=True)
    feature_df["Patient_ID"] = feature_df["Patient_ID"].astype(str)
    names = feature_names(df_fit, df_val, df_te, cfg)
    return model, {"Train": fit_ds, "Validation": val_ds, "TemporalTest": te_ds}, cfg, feature_df, names


def feature_names(df_fit: pd.DataFrame, df_val: pd.DataFrame, df_te: pd.DataFrame, cfg: direct.TrainConfig) -> dict[str, list[str]]:
    interval_cats = sorted(df_fit["Interval_Name"].unique())
    prev_state_cats = sorted(df_fit["Prev_State"].unique())
    static_df, local_df, global_df = base.build_feature_blocks(df_fit, interval_cats, prev_state_cats)
    static_df, local_df, global_df = direct.apply_feature_selection(static_df, local_df, global_df, cfg)
    z3m_cols: list[str] = []
    if cfg.z3m_mode in direct.Z3M_BRANCH_MODES:
        z3m_cols = list(direct.build_z3m_features(df_fit, cfg).columns)
    elif cfg.z3m_mode != "none":
        static_df = direct.attach_z3m_features(static_df, df_fit, cfg)
    return {
        "static": list(static_df.columns),
        "local": list(local_df.columns),
        "global": list(global_df.columns),
        "z3M": z3m_cols,
    }


def all_feature_names(names: dict[str, list[str]]) -> tuple[list[str], list[str], list[str]]:
    branches, raw, labels = [], [], []
    for branch in ["static", "local", "global", "z3M"]:
        for name in names.get(branch, []):
            branches.append(branch)
            raw.append(name)
            if name.startswith("selected_z3m_"):
                suffix = name.split("_")[-1]
                labels.append(f"z3M latent {suffix}")
            elif name == "z3m_available":
                labels.append("z3M available")
            elif name == "z3m_time_since_3m":
                labels.append("z3M time since 3M")
            else:
                labels.append(f"{branch}: {name}")
    return branches, raw, labels


def dataset_inputs(ds: direct.DirectDataset) -> list[torch.Tensor]:
    return [ds.static, ds.local, ds.global_x, ds.z3m]


def concat_inputs(inputs: list[torch.Tensor]) -> np.ndarray:
    return np.concatenate([x.detach().cpu().numpy() for x in inputs], axis=1)


class LogitWrapper(torch.nn.Module):
    def __init__(self, model: direct.ThreeBranchHazardNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, static_x: torch.Tensor, local_x: torch.Tensor, global_x: torch.Tensor, z3m_x: torch.Tensor) -> torch.Tensor:
        return self.model(static_x, local_x, global_x, z3m_x)["logit"].unsqueeze(1)


def collect_predictions(model: direct.ThreeBranchHazardNet, ds: direct.DirectDataset) -> np.ndarray:
    loader = DataLoader(ds, batch_size=512, shuffle=False)
    out = []
    with torch.no_grad():
        for batch in loader:
            out.append(model(batch["static"], batch["local"], batch["global"], batch["z3m"])["prob"].detach().cpu().numpy())
    return np.concatenate(out)


def prediction_recompute_check(model: direct.ThreeBranchHazardNet, datasets: dict[str, direct.DirectDataset]) -> pd.DataFrame:
    saved = pd.read_csv(PRED)
    preds = []
    for split in SPLITS:
        preds.append(collect_predictions(model, datasets[split]))
    recomputed = np.concatenate(preds)
    check = saved[["Split", "Patient_ID", "Interval_ID", "DirectProb"]].copy()
    check["RecomputedProb"] = recomputed
    check["AbsDiff"] = (check["DirectProb"] - check["RecomputedProb"]).abs()
    check.to_csv(TAB / "shap_prediction_recompute_check.csv", index=False)
    max_abs = float(check["AbsDiff"].max())
    if max_abs > 1e-5:
        raise RuntimeError(f"Recomputed predictions differ from saved DirectProb: max_abs={max_abs:.6g}")
    return check


def sample_indices(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(y) <= n:
        return np.arange(len(y))
    n_pos = min(len(pos), max(16, int(round(n * max(float(y.mean()), 0.08)))))
    n_neg = max(0, n - n_pos)
    pos_sel = rng.choice(pos, size=n_pos, replace=False) if len(pos) else np.array([], dtype=int)
    neg_sel = rng.choice(neg, size=min(n_neg, len(neg)), replace=False) if len(neg) else np.array([], dtype=int)
    idx = np.concatenate([pos_sel, neg_sel])
    if len(idx) < n:
        rem = np.setdiff1d(np.arange(len(y)), idx, assume_unique=False)
        idx = np.concatenate([idx, rng.choice(rem, size=min(n - len(idx), len(rem)), replace=False)])
    return np.sort(idx)


def list_take(inputs: list[torch.Tensor], idx: np.ndarray) -> list[torch.Tensor]:
    return [x[idx].detach().clone() for x in inputs]


def squeeze_shap(values: list[np.ndarray]) -> np.ndarray:
    arrs = []
    for value in values:
        arr = np.asarray(value)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[:, :, 0]
        arrs.append(arr)
    return np.concatenate(arrs, axis=1)


def compute_gradient_shap(
    model: direct.ThreeBranchHazardNet,
    datasets: dict[str, direct.DirectDataset],
    names: dict[str, list[str]],
    cfg: direct.TrainConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    try:
        import shap
    except Exception as exc:
        raise RuntimeError("shap is unavailable") from exc

    branches, raw_names, labels = all_feature_names(names)
    expected_dim = len(labels)
    actual_dim = sum(x.shape[1] for x in dataset_inputs(datasets["Train"]))
    if expected_dim != actual_dim:
        raise ValueError(f"Feature-name count mismatch: names={expected_dim} tensors={actual_dim}")

    train_inputs = dataset_inputs(datasets["Train"])
    dev_inputs = [torch.cat([datasets["Train"].static, datasets["Validation"].static], dim=0),
                  torch.cat([datasets["Train"].local, datasets["Validation"].local], dim=0),
                  torch.cat([datasets["Train"].global_x, datasets["Validation"].global_x], dim=0),
                  torch.cat([datasets["Train"].z3m, datasets["Validation"].z3m], dim=0)]
    dev_y = torch.cat([datasets["Train"].y, datasets["Validation"].y], dim=0).numpy().astype(int)
    test_inputs = dataset_inputs(datasets["TemporalTest"])
    test_y = datasets["TemporalTest"].y.numpy().astype(int)

    bg_idx = sample_indices(datasets["Train"].y.numpy(), 64, cfg.seed + 101)
    dev_idx = sample_indices(dev_y, 384, cfg.seed + 202)
    test_idx = sample_indices(test_y, 256, cfg.seed + 303)
    wrapper = LogitWrapper(model).eval()
    background_inputs = list_take(train_inputs, bg_idx)
    with torch.no_grad():
        background_logits = wrapper(*background_inputs).squeeze(1).detach().cpu().numpy()
    expected_logit = float(np.mean(background_logits))
    expected_prob = float(1.0 / (1.0 + np.exp(-expected_logit)))
    explainer = shap.GradientExplainer(wrapper, background_inputs)
    dev_values = squeeze_shap(explainer.shap_values(list_take(dev_inputs, dev_idx), nsamples=96))
    test_values = squeeze_shap(explainer.shap_values(list_take(test_inputs, test_idx), nsamples=96))
    dev_x = concat_inputs(list_take(dev_inputs, dev_idx))
    test_x = concat_inputs(list_take(test_inputs, test_idx))
    if not np.isfinite(dev_values).all() or not np.isfinite(test_values).all():
        raise RuntimeError("SHAP values contain non-finite entries")

    dev_imp = importance_table(dev_values, dev_x, labels, raw_names, branches, split="Development")
    test_imp = importance_table(test_values, test_x, labels, raw_names, branches, split="TemporalTest")
    dev_imp.to_csv(TAB / "shap_feature_importance_dev.csv", index=False)
    test_imp.to_csv(TAB / "shap_feature_importance_temporal_test.csv", index=False)

    manifest = {
        "method": "GradientExplainer",
        "scale": "logit",
        "background_split": "Train development subset only",
        "background_n": len(bg_idx),
        "development_explain_n": len(dev_idx),
        "temporal_test_explain_n": len(test_idx),
        "feature_count": expected_dim,
        "expected_logit_background_mean": expected_logit,
        "expected_prob_background_mean": expected_prob,
        "temporal_test_used_for_background": False,
        "temporal_test_used_for_selection": False,
    }
    pd.DataFrame([manifest]).to_csv(TAB / "shap_background_manifest.csv", index=False)
    payload = {
        "dev_values": dev_values,
        "test_values": test_values,
        "dev_x": dev_x,
        "test_x": test_x,
        "dev_y": dev_y[dev_idx],
        "test_y": test_y[test_idx],
        "dev_idx": dev_idx,
        "test_idx": test_idx,
        "labels": labels,
        "raw_names": raw_names,
        "branches": branches,
        "explainer": explainer,
        "expected_logit": expected_logit,
        "expected_prob": expected_prob,
        "manifest": manifest,
    }
    return dev_imp, test_imp, pd.DataFrame([manifest]), payload


def importance_table(values: np.ndarray, x: np.ndarray, labels: list[str], raw_names: list[str], branches: list[str], split: str) -> pd.DataFrame:
    mean_abs = np.abs(values).mean(axis=0)
    signed_mean = values.mean(axis=0)
    corr = []
    for j in range(values.shape[1]):
        if np.std(x[:, j]) < 1e-12 or np.std(values[:, j]) < 1e-12:
            corr.append(0.0)
        else:
            corr.append(float(np.corrcoef(x[:, j], values[:, j])[0, 1]))
    out = pd.DataFrame(
        {
            "Split": split,
            "Rank": np.argsort(-mean_abs).argsort() + 1,
            "Branch": branches,
            "Feature": raw_names,
            "Display_Feature": labels,
            "MeanAbsSHAP_Logit": mean_abs,
            "MeanSignedSHAP_Logit": signed_mean,
            "Feature_SHAP_Correlation": corr,
        }
    ).sort_values("MeanAbsSHAP_Logit", ascending=False)
    out["Rank"] = np.arange(1, len(out) + 1)
    return out


def shap_branch_tables(dev_imp: pd.DataFrame, test_imp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    branch = (
        pd.concat([dev_imp, test_imp], ignore_index=True)
        .groupby(["Split", "Branch"], as_index=False)
        .agg(MeanAbsSHAP_Logit=("MeanAbsSHAP_Logit", "sum"))
    )
    branch["Share"] = branch.groupby("Split")["MeanAbsSHAP_Logit"].transform(lambda s: s / s.sum())
    branch.to_csv(TAB / "shap_branch_importance.csv", index=False)

    merged = dev_imp[["Branch", "Feature", "Display_Feature", "Rank", "MeanAbsSHAP_Logit"]].merge(
        test_imp[["Branch", "Feature", "Display_Feature", "Rank", "MeanAbsSHAP_Logit"]],
        on=["Branch", "Feature", "Display_Feature"],
        how="inner",
        suffixes=("_Dev", "_TemporalTest"),
    )
    merged["Rank_Delta_TestMinusDev"] = merged["Rank_TemporalTest"] - merged["Rank_Dev"]
    merged.to_csv(TAB / "shap_dev_test_rank_concordance.csv", index=False)
    return branch, merged


def gate_values(model: direct.ThreeBranchHazardNet, datasets: dict[str, direct.DirectDataset], names: dict[str, list[str]]) -> pd.DataFrame:
    rows = []

    def sig(x: torch.Tensor) -> np.ndarray:
        return torch.sigmoid(x.detach().cpu()).numpy()

    for branch, attr, cols in [
        ("static", "static_input_gate_logit", names.get("static", [])),
        ("local", "local_input_gate_logit", names.get("local", [])),
        ("global", "global_input_gate_logit", names.get("global", [])),
        ("z3M", "z3m_input_gate_logit", names.get("z3M", [])),
    ]:
        param = getattr(model, attr, None)
        if param is None:
            rows.append({"Gate_Type": "input", "Branch": branch, "Feature": "__mean__", "Gate_Value": 1.0, "Note": "input gate disabled"})
            continue
        vals = sig(param)
        for name, value in zip(cols, vals):
            rows.append({"Gate_Type": "input", "Branch": branch, "Feature": name, "Gate_Value": float(value), "Note": "learned input gate"})
        rows.append({"Gate_Type": "input", "Branch": branch, "Feature": "__mean__", "Gate_Value": float(vals.mean()), "Note": "mean learned input gate"})

    for branch, attr in [
        ("static", "static_branch_gate_logit"),
        ("local", "local_branch_gate_logit"),
        ("global", "global_branch_gate_logit"),
        ("z3M", "z3m_branch_gate_logit"),
    ]:
        param = getattr(model, attr, None)
        value = float(torch.sigmoid(param.detach()).cpu()) if param is not None else 1.0
        rows.append({"Gate_Type": "branch", "Branch": branch, "Feature": "__branch__", "Gate_Value": value, "Note": "learned branch gate" if param is not None else "branch gate disabled"})

    if model.use_z3m_branch and model.z3m_gate_mode == "time_sigmoid":
        z = datasets["TemporalTest"].z3m
        with torch.no_grad():
            time_signal = z[:, -1:].clamp(-5.0, 5.0)
            val = torch.sigmoid(model.z3m_time_gate(time_signal)).squeeze(1).cpu().numpy()
        rows.append({"Gate_Type": "z3m_time_sigmoid", "Branch": "z3M", "Feature": "__temporal_test_mean__", "Gate_Value": float(np.mean(val)), "Note": "mean temporal-test z3M time gate"})
        for q, value in zip(["q05", "q50", "q95"], np.quantile(val, [0.05, 0.50, 0.95])):
            rows.append({"Gate_Type": "z3m_time_sigmoid", "Branch": "z3M", "Feature": q, "Gate_Value": float(value), "Note": "temporal-test z3M time gate quantile"})
    elif model.z3m_gate is not None:
        rows.append({"Gate_Type": "z3m_scalar", "Branch": "z3M", "Feature": "__scalar__", "Gate_Value": float(model.z3m_gate.detach().cpu()), "Note": "learned scalar multiplier"})

    out = pd.DataFrame(rows)
    out.to_csv(TAB / "gate_values.csv", index=False)
    return out


def fig_gate_branch(gates: pd.DataFrame, branch_ablation: pd.DataFrame, branch_imp: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    branch_order = ["static", "local", "global", "z3M"]
    rows = []
    for branch in branch_order:
        if branch == "z3M":
            z3_time = gates[(gates["Gate_Type"].eq("z3m_time_sigmoid")) & (gates["Feature"].eq("__temporal_test_mean__"))]
            if not z3_time.empty:
                rows.append({"Branch": "z3M\n(time)", "Gate_Value": float(z3_time.iloc[0]["Gate_Value"])})
                continue
        gate_row = gates[
            (gates["Gate_Type"].eq("branch"))
            & (gates["Feature"].eq("__branch__"))
            & (gates["Branch"].eq(branch))
        ]
        if not gate_row.empty:
            rows.append({"Branch": branch, "Gate_Value": float(gate_row.iloc[0]["Gate_Value"])})
    gate_rows = pd.DataFrame(rows)
    if not gate_rows.empty:
        axes[0].bar(gate_rows["Branch"], gate_rows["Gate_Value"], color=["#7BAED6", "#9CCB86", "#F3B572", "#B99AD6"][: len(gate_rows)])
        axes[0].set_ylim(0, 1.05)
        axes[0].set_title("Effective branch/time gates")
        axes[0].set_ylabel("Gate value")
        for i, value in enumerate(gate_rows["Gate_Value"]):
            axes[0].text(i, value + 0.03, f"{value:.2f}", ha="center", fontsize=8)

    test = branch_ablation[branch_ablation["Split"].eq("TemporalTest") & ~branch_ablation["Variant"].eq("full")].copy()
    test["Removed"] = test["Removed_Branch"]
    test["PR_AUC_Drop"] = -test["Delta_PR_AUC_vs_Full"]
    test["Brier_Increase"] = test["Delta_Brier_vs_Full"]
    axes[1].bar(test["Removed"], test["PR_AUC_Drop"], color="#765AA6", alpha=0.9)
    axes[1].axhline(0, color="#68717D", lw=1.0)
    axes[1].set_title("Training-time branch removal")
    axes[1].set_ylabel("Full - removed PR-AUC")
    for i, value in enumerate(test["PR_AUC_Drop"]):
        axes[1].text(i, value + (0.004 if value >= 0 else -0.008), f"{value:+.3f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)

    dev_branch = branch_imp[branch_imp["Split"].eq("Development")].sort_values("Share", ascending=False)
    axes[2].bar(dev_branch["Branch"], dev_branch["Share"], color="#4B9B6E", alpha=0.9)
    axes[2].set_ylim(0, max(0.05, float(dev_branch["Share"].max()) + 0.08))
    axes[2].set_title("SHAP share by branch")
    axes[2].set_ylabel("Share of mean |SHAP|")
    for i, value in enumerate(dev_branch["Share"]):
        axes[2].text(i, value + 0.01, f"{value:.1%}", ha="center", fontsize=8)
    save(fig, "Figure_23_Gate_Branch_Contribution.png")


def fig_shap_beeswarm(payload: dict[str, object]) -> None:
    import shap

    values = payload["dev_values"]
    x = payload["dev_x"]
    labels = payload["labels"]
    plt.figure(figsize=(10.2, 7.2))
    shap.summary_plot(values, x, feature_names=labels, max_display=20, show=False, plot_size=None)
    plt.title("Development feature-level SHAP on logit scale", fontsize=12, weight="bold")
    plt.tight_layout()
    plt.savefig(FIG / "Figure_24_SHAP_Global_Beeswarm_Dev.png", dpi=260, bbox_inches="tight")
    plt.close()


def fig_shap_bar(dev_imp: pd.DataFrame, branch_imp: pd.DataFrame) -> None:
    top = dev_imp.head(20).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.6), gridspec_kw={"width_ratios": [2.2, 1.0]})
    colors = top["Branch"].map({"static": "#7BAED6", "local": "#9CCB86", "global": "#F3B572", "z3M": "#B99AD6"}).fillna("#8A8F99")
    axes[0].barh(top["Display_Feature"], top["MeanAbsSHAP_Logit"], color=colors)
    axes[0].set_xlabel("Mean |SHAP| on logit scale")
    axes[0].set_title("Top development features")
    for y, value in enumerate(top["MeanAbsSHAP_Logit"]):
        axes[0].text(value + 0.002, y, f"{value:.3f}", va="center", fontsize=7)

    branch = branch_imp[branch_imp["Split"].eq("Development")].sort_values("MeanAbsSHAP_Logit", ascending=False)
    axes[1].bar(branch["Branch"], branch["MeanAbsSHAP_Logit"], color="#765AA6", alpha=0.9)
    axes[1].set_title("Branch-level SHAP sum")
    axes[1].set_ylabel("Sum mean |SHAP|")
    save(fig, "Figure_25_SHAP_Feature_Bar_Dev.png")


def fig_shap_consistency(concord: pd.DataFrame) -> None:
    top = concord.sort_values("Rank_Dev").head(25).copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))
    axes[0].scatter(concord["MeanAbsSHAP_Logit_Dev"], concord["MeanAbsSHAP_Logit_TemporalTest"], s=28, color="#4B9B6E", alpha=0.75)
    lim = max(float(concord["MeanAbsSHAP_Logit_Dev"].max()), float(concord["MeanAbsSHAP_Logit_TemporalTest"].max())) * 1.08
    axes[0].plot([0, lim], [0, lim], ls="--", color="#68717D", lw=1)
    axes[0].set_xlabel("Development mean |SHAP|")
    axes[0].set_ylabel("Temporal-test mean |SHAP|")
    axes[0].set_title("Importance magnitude consistency")
    for _, row in concord.sort_values("MeanAbsSHAP_Logit_Dev", ascending=False).head(6).iterrows():
        axes[0].text(row["MeanAbsSHAP_Logit_Dev"], row["MeanAbsSHAP_Logit_TemporalTest"], str(row["Display_Feature"])[:22], fontsize=6)

    axes[1].barh(top["Display_Feature"].iloc[::-1], top["Rank_Delta_TestMinusDev"].iloc[::-1], color="#D98C2B")
    axes[1].axvline(0, color="#68717D", lw=1)
    axes[1].set_xlabel("Rank shift: temporal test - development")
    axes[1].set_title("Top-feature rank stability")
    save(fig, "Figure_26_SHAP_Test_Consistency.png")


def fig_local(payload: dict[str, object], model: direct.ThreeBranchHazardNet, datasets: dict[str, direct.DirectDataset]) -> pd.DataFrame:
    explainer = payload["explainer"]
    labels = payload["labels"]
    expected_logit = float(payload["expected_logit"])
    expected_prob = float(payload["expected_prob"])
    test_ds = datasets["TemporalTest"]
    pred = collect_predictions(model, test_ds)
    y = test_ds.y.numpy().astype(int)
    high_candidates = np.flatnonzero((pred >= np.quantile(pred, 0.90)) & (y == 1))
    low_candidates = np.flatnonzero((pred <= np.quantile(pred, 0.10)) & (y == 0))
    if len(high_candidates) == 0:
        high_candidates = np.array([int(np.argmax(pred))])
    if len(low_candidates) == 0:
        low_candidates = np.array([int(np.argmin(pred))])
    case_indices = [int(high_candidates[np.argmax(pred[high_candidates])]), int(low_candidates[np.argmin(pred[low_candidates])])]
    case_names = ["Patient A", "Patient B"]
    case_rows = []
    contrib_rows = []
    for case_name, idx, filename in zip(case_names, case_indices, ["Figure_27_SHAP_Local_HighRisk.png", "Figure_28_SHAP_Local_LowRisk.png"]):
        xs = list_take(dataset_inputs(test_ds), np.array([idx]))
        vals = squeeze_shap(explainer.shap_values(xs, nsamples=128))[0]
        xvals = concat_inputs(xs)[0]
        with torch.no_grad():
            actual_logit = float(model(*xs)["logit"].detach().cpu().numpy()[0])
        actual_prob = float(pred[idx])
        top_n = min(11, len(vals))
        order = np.argsort(np.abs(vals))[-top_n:]
        order = order[np.argsort(np.abs(vals[order]))]
        other = actual_logit - expected_logit - float(vals[order].sum())
        entries = []
        other_count = len(vals) - len(order)
        if abs(other) > 1e-6:
            entries.append((f"{other_count} other features + residual", other, ""))
        for j in order:
            value = float(xvals[j])
            if abs(value) >= 100:
                value_text = f"{value:.1f}"
            elif abs(value) >= 10:
                value_text = f"{value:.2f}"
            else:
                value_text = f"{value:.3f}"
            entries.append((str(labels[j]), float(vals[j]), value_text))

        fig, ax = plt.subplots(figsize=(10.8, 6.5))
        cumulative = expected_logit
        lefts, widths, colors, ylabels = [], [], [], []
        for label, contribution, value_text in entries:
            new_value = cumulative + contribution
            lefts.append(min(cumulative, new_value))
            widths.append(abs(contribution))
            colors.append("#F03B6D" if contribution >= 0 else "#4C78A8")
            ylabels.append(f"{value_text} = {label}" if value_text else label)
            cumulative = new_value

        y_pos = np.arange(len(entries))
        ax.barh(y_pos, widths, left=lefts, color=colors, height=0.64)
        running = expected_logit
        for yy, (_, contribution, _) in zip(y_pos, entries):
            new_value = running + contribution
            ax.plot([new_value, new_value], [yy - 0.32, yy + 0.32], color="#9CA3AF", ls="--", lw=0.8)
            xpos = new_value + (0.03 if contribution >= 0 else -0.03)
            ha = "left" if contribution >= 0 else "right"
            ax.text(xpos, yy, f"{contribution:+.2f}", va="center", ha=ha, fontsize=9, color=colors[yy])
            running = new_value
        ax.axvline(expected_logit, color="#9CA3AF", ls="--", lw=1.0)
        ax.axvline(actual_logit, color="#111827", ls="--", lw=1.0)
        ax.text(expected_logit, -0.95, f"E[f(x)]={expected_prob:.3f}", ha="center", va="top", color="#6B7280", fontsize=10)
        ax.text(actual_logit, len(entries) - 0.2, f"f(x)={actual_prob:.3f}", ha="center", va="bottom", color="#111827", fontsize=11)
        ax.set_yticks(y_pos, ylabels)
        ax.set_xlabel("Cumulative contribution on relapse logit scale")
        ax.set_title(f"{case_name}: SHAP waterfall, predicted risk={actual_prob:.3f}, observed={int(y[idx])}")
        ax.grid(axis="x", alpha=0.18)
        xmin = min([expected_logit, actual_logit] + lefts) - 0.25
        xmax = max([expected_logit, actual_logit] + [l + w for l, w in zip(lefts, widths)]) + 0.25
        ax.set_xlim(xmin, xmax)
        save(fig, filename)
        case_rows.append(
            {
                "Case_Label": case_name,
                "TemporalTest_Row_Index": idx,
                "Expected_Prob": expected_prob,
                "Predicted_Risk": actual_prob,
                "Observed_Relapse": int(y[idx]),
                "Figure": filename,
            }
        )
        for rank, (label, contribution, value_text) in enumerate(reversed(entries), start=1):
            contrib_rows.append(
                {
                    "Case_Label": case_name,
                    "Rank_From_Top": rank,
                    "Feature_Label": label,
                    "Feature_Value_Transformed": value_text,
                    "SHAP_Logit_Contribution": contribution,
                }
            )
    out = pd.DataFrame(case_rows)
    out.to_csv(TAB / "shap_local_cases.csv", index=False)
    pd.DataFrame(contrib_rows).to_csv(TAB / "shap_local_waterfall_contributions.csv", index=False)
    return out


def heatmap_marginals() -> None:
    pred_path = TAB / "readme4_patient_interval_predictions.csv"
    patient_path = TAB / "readme4_patient_aggregated_risk.csv"
    if pred_path.exists() and patient_path.exists():
        pred = pd.read_csv(pred_path)
        patient = pd.read_csv(patient_path)
        pred["Patient_ID"] = pred["Patient_ID"].astype(str)
        patient["Patient_ID"] = patient["Patient_ID"].astype(str)
    else:
        pred = pd.read_csv(PRED)
        pred["Patient_ID"] = pred["Patient_ID"].astype(str)
        patient = (
            pred[pred["Split"].eq("TemporalTest")]
            .groupby("Patient_ID", as_index=False)
            .agg(Patient_Event=("Y_Relapse", "max"), Mean_Interval_Risk=("DirectProb", "mean"), Max_Interval_Risk=("DirectProb", "max"), N_Intervals=("Interval_ID", "count"))
        )
        pred = pred[pred["Split"].eq("TemporalTest")].copy()
    test = pred[pred["Split"].eq("TemporalTest")].copy() if "Split" in pred.columns else pred.copy()
    order = test[["Interval_Name", "Start_Time", "Stop_Time"]].drop_duplicates().sort_values(["Start_Time", "Stop_Time"])
    windows = order["Interval_Name"].astype(str).tolist()
    patient_order = patient[patient["Split"].eq("TemporalTest")].copy() if "Split" in patient.columns else patient.copy()
    risk_col = "Mean_Interval_Risk" if "Mean_Interval_Risk" in patient_order.columns else "Max_Interval_Risk"
    patient_order = patient_order.sort_values(["Patient_Event", risk_col], ascending=[False, False])
    event_pids = patient_order[patient_order["Patient_Event"].eq(1)]["Patient_ID"].astype(str).tolist()
    nonevent_pids = patient_order[patient_order["Patient_Event"].eq(0)]["Patient_ID"].astype(str).tolist()

    def matrix(pids: list[str]) -> np.ndarray:
        piv = test[test["Patient_ID"].astype(str).isin(pids)].pivot_table(index="Patient_ID", columns="Interval_Name", values="DirectProb", aggfunc="max")
        return piv.reindex(index=pids, columns=windows).to_numpy(dtype=float)

    marg = []
    for window in windows:
        part = test[test["Interval_Name"].astype(str).eq(window)]
        marg.append(
            {
                "Interval_Name": window,
                "N": int(len(part)),
                "Events": int(part["Y_Relapse"].sum()),
                "Observed_Event_Rate": float(part["Y_Relapse"].mean()) if len(part) else np.nan,
                "Mean_Predicted_Risk": float(part["DirectProb"].mean()) if len(part) else np.nan,
            }
        )
    marg_df = pd.DataFrame(marg)
    marg_df.to_csv(TAB / "readme4_heatmap_marginals.csv", index=False)

    cmap = RISK_CMAP.copy()
    cmap.set_bad(color=MISSING_COLOR)
    fig = plt.figure(figsize=(12.8, 8.6))
    gs = fig.add_gridspec(
        3,
        3,
        width_ratios=[1.0, 0.13, 0.032],
        height_ratios=[0.85, max(1, len(event_pids)), max(1, len(nonevent_pids))],
        hspace=0.33,
        wspace=0.06,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[2, 0])]
    row_axes = [fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, 1])]
    cax = fig.add_subplot(gs[1:, 2])
    x = np.arange(len(windows))
    ax_top.bar(x, marg_df["Mean_Predicted_Risk"], width=0.58, color="#D98C2B", alpha=0.55, label="Mean predicted risk")
    ax_top.plot(x, marg_df["Observed_Event_Rate"], marker="o", lw=2.0, color="#C44E52", label="Observed event rate")
    ax_top.set_ylim(0, 1.0)
    ax_top.set_yticks([])
    ax_top.set_xticks(x, [])
    ax_top.set_ylabel("0-1 risk", fontsize=8)
    ax_top.set_title("Window-level marginals", fontsize=10)
    ax_top.legend(loc="upper right", ncols=2, frameon=False, fontsize=8)

    ims = []
    for ax, rax, pids, title in [
        (axes[0], row_axes[0], event_pids, "Observed relapse patients"),
        (axes[1], row_axes[1], nonevent_pids, "No observed relapse patients"),
    ]:
        mat = matrix(pids)
        im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", cmap=cmap, norm=RISK_NORM)
        ims.append(im)
        ax.set_title(f"{title} sorted by mean interval risk")
        ax.set_ylabel("Patients")
        ax.set_yticks([])
        sub = patient_order.set_index("Patient_ID").reindex(pids)
        risks = sub[risk_col].to_numpy(dtype=float)
        ypos = np.arange(len(pids))
        rax.barh(ypos, risks, color=RISK_CMAP(RISK_NORM(risks)), height=0.85)
        rax.invert_yaxis()
        rax.set_xlim(0, 1)
        rax.set_xticks([0, 0.5, 1.0])
        rax.set_yticks([])
        rax.set_title("Patient\nmean risk", fontsize=8)
        rax.grid(axis="x", alpha=0.2)
    axes[0].tick_params(labelbottom=False)
    axes[1].set_xticks(range(len(windows)), windows, rotation=30, ha="right")
    cbar = fig.colorbar(ims[0], cax=cax, ticks=RISK_TICKS)
    cbar.set_label("Interval relapse risk")
    fig.suptitle("Temporal test patient-interval risk heatmap with marginal bars", y=0.975, fontsize=13, weight="bold")
    fig.subplots_adjust(left=0.08, right=0.94, top=0.90, bottom=0.15)
    save(fig, "Figure_19_Readme4_Patient_Risk_Heatmap.png", tight=False)


def joint_risk_window_heatmap() -> None:
    """Draw a standard joint heatmap with marginal bar plots for temporal-test risk bins."""

    pred_path = TAB / "readme4_patient_interval_predictions.csv"
    pred = pd.read_csv(pred_path if pred_path.exists() else PRED)
    if "Split" in pred.columns:
        pred = pred[pred["Split"].eq("TemporalTest")].copy()
    pred["DirectProb"] = pd.to_numeric(pred["DirectProb"], errors="coerce").clip(0.0, 1.0)
    pred["Y_Relapse"] = pd.to_numeric(pred["Y_Relapse"], errors="coerce").fillna(0).astype(int)

    windows = (
        pred[["Interval_Name", "Start_Time", "Stop_Time"]]
        .drop_duplicates()
        .sort_values(["Start_Time", "Stop_Time"])["Interval_Name"]
        .astype(str)
        .tolist()
    )
    bins = np.linspace(0.0, 1.0, 11)
    bin_labels = [f"{bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(len(bins) - 1)]
    pred["Risk_Bin"] = pd.cut(pred["DirectProb"], bins=bins, labels=bin_labels, include_lowest=True, right=True)

    grouped = (
        pred.groupby(["Interval_Name", "Risk_Bin"], observed=False)
        .agg(N=("Y_Relapse", "size"), Events=("Y_Relapse", "sum"), Mean_Predicted_Risk=("DirectProb", "mean"))
        .reset_index()
    )
    grouped["Observed_Event_Rate"] = np.where(grouped["N"] > 0, grouped["Events"] / grouped["N"], np.nan)
    grouped["Window_Order"] = grouped["Interval_Name"].astype(str).map({w: i for i, w in enumerate(windows)})
    grouped.to_csv(TAB / "joint_risk_window_heatmap.csv", index=False)

    rate = (
        grouped.pivot(index="Interval_Name", columns="Risk_Bin", values="Observed_Event_Rate")
        .reindex(index=windows, columns=bin_labels)
        .to_numpy(dtype=float)
    )
    count = (
        grouped.pivot(index="Interval_Name", columns="Risk_Bin", values="N")
        .reindex(index=windows, columns=bin_labels)
        .fillna(0)
        .to_numpy(dtype=float)
    )
    total_by_bin = count.sum(axis=0)
    total_by_window = count.sum(axis=1)
    event_by_bin = (
        grouped.pivot(index="Interval_Name", columns="Risk_Bin", values="Events")
        .reindex(index=windows, columns=bin_labels)
        .fillna(0)
        .to_numpy(dtype=float)
        .sum(axis=0)
    )
    cmap = RISK_CMAP.copy()
    cmap.set_bad(color=MISSING_COLOR)
    fig = plt.figure(figsize=(12.6, 7.0))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[0.9, 4.2],
        width_ratios=[5.2, 1.1, 0.16],
        hspace=0.22,
        wspace=0.08,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])
    cax = fig.add_subplot(gs[1, 2])

    x = np.arange(len(bin_labels))
    ax_top.bar(x, total_by_bin, color="#4C78A8", alpha=0.78, label="Intervals")
    ax_top.plot(x, event_by_bin, color="#C44E52", marker="o", lw=1.8, label="Relapse events")
    ax_top.set_xlim(-0.5, len(bin_labels) - 0.5)
    ax_top.set_ylabel("Count")
    ax_top.set_xticks([])
    ax_top.set_title("Predicted-risk marginal distribution")
    ax_top.legend(loc="upper left", ncols=2, frameon=False)

    im = ax_heat.imshow(np.ma.masked_where(count == 0, rate), aspect="auto", cmap=cmap, norm=RISK_NORM)
    ax_heat.set_xticks(x, bin_labels, rotation=35, ha="right")
    ax_heat.set_yticks(np.arange(len(windows)), windows)
    ax_heat.set_xlabel("Predicted next-window relapse risk bin")
    ax_heat.set_ylabel("Follow-up window")
    ax_heat.set_title("Cell color = observed relapse rate", pad=10)
    for i in range(len(windows)):
        for j in range(len(bin_labels)):
            n = int(count[i, j])
            if n >= 8:
                color = "#111827" if np.nan_to_num(rate[i, j]) < 0.55 else "white"
                ax_heat.text(j, i, f"{int(round(rate[i, j] * 100))}%", ha="center", va="center", fontsize=6, color=color)

    y = np.arange(len(windows))
    ax_right.barh(y, total_by_window, color="#4C78A8", alpha=0.78, label="Intervals")
    ax_right.set_yticks(y, [])
    ax_right.invert_yaxis()
    ax_right.set_xlabel("Count")
    ax_right.set_title("Window\nmarginal", fontsize=9)
    ax_right.grid(axis="x", alpha=0.2)

    cbar = fig.colorbar(im, cax=cax, ticks=RISK_TICKS)
    cbar.set_label("Observed relapse rate")
    fig.suptitle("Joint risk-bin heatmap with marginal bar plots on temporal test intervals", fontsize=13, weight="bold", y=0.98)
    save(fig, "Figure_29_Joint_Risk_Window_Marginal_Heatmap.png", tight=False)


def update_heatmap_caption(text: str) -> str:
    old = (
        "Figure 19 将 temporal test 的 patient-interval 风险矩阵化显示：列是随访窗口，行是患者，颜色是该 interval 的复发风险。"
        "复发患者和未复发患者分面展示，并在各自面板内按 mean risk 排序。"
    )
    new = (
        "Figure 19 将 temporal test 的 patient-interval 风险矩阵化显示，并升级为带边际条图的热图：主矩阵中列是随访窗口、行是患者、颜色是该 interval 的复发风险；"
        "顶部边际条同时显示各窗口的平均预测风险和真实复发率；右侧边际条显示同一患者的 mean interval risk。复发患者和未复发患者分面展示，并在各自面板内按 mean risk 排序。"
    )
    return text.replace(old, new)


def top_items(df: pd.DataFrame, n: int = 8) -> str:
    return "、".join(str(x) for x in df.head(n)["Display_Feature"].tolist())


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def read_branch_ablation() -> pd.DataFrame:
    path = TAB / "branch_training_ablation_importance.csv"
    if not path.exists():
        path = TAB / "branch_knockout_importance.csv"
    return pd.read_csv(path)


def explainability_block(dev_imp: pd.DataFrame, test_imp: pd.DataFrame, branch_imp: pd.DataFrame, manifest: pd.DataFrame, concord: pd.DataFrame) -> str:
    top_dev = top_items(dev_imp, 8)
    top_test = top_items(test_imp, 6)
    method = str(manifest.loc[0, "method"])
    bg_n = int(manifest.loc[0, "background_n"])
    dev_n = int(manifest.loc[0, "development_explain_n"])
    test_n = int(manifest.loc[0, "temporal_test_explain_n"])
    branch_table = branch_imp[branch_imp["Split"].eq("Development")][["Branch", "Share"]].sort_values("Share", ascending=False).copy()
    branch_table["Share"] = branch_table["Share"].map(lambda x: f"{x:.1%}")
    branch_md = md_table(branch_table)
    return f"""
{CH5_START}
## 模型解释：门控、特征层 SHAP 与局部解释

本节对齐仓库主 `README.md` 第五章的解释层，但口径改为当前四分支单头动态复发模型。解释分析是 **post-hoc explanation**：不用于特征筛选、阈值选择、模型选择或校准拟合。主解释层使用 Train/Validation development rows；temporal test 只作为外推一致性检查和局部病例说明。SHAP 在 logit scale 上解释，正值表示提高下一时间窗复发风险。z3M embedding 维度只解释为 `3M early-response latent component`，不把单个 embedding 维度硬翻译成具体临床变量。

### 5.1 门控与分支贡献

![门控与分支贡献](figures/Figure_23_Gate_Branch_Contribution.png)

Figure 23 把三层解释放在一起：左侧是模型内显式 gate 或默认分支 multiplier，中间是 matched training-time branch removal 对 temporal-test PR-AUC 的影响，右侧是 development SHAP 在四个分支上的贡献占比。分支 SHAP 占比如下：

{branch_md}

### 5.2 Development feature-level SHAP

![Development SHAP beeswarm](figures/Figure_24_SHAP_Global_Beeswarm_Dev.png)

![Development SHAP feature bar](figures/Figure_25_SHAP_Feature_Bar_Dev.png)

Figure 24-25 使用 `{method}`，background 仅来自 development train subset（`n={bg_n}`），解释 development 抽样 rows（`n={dev_n}`）。当前 development 解释层的前列特征包括：{top_dev}。这些图用于说明模型在工程特征层如何分配风险贡献，不改变模型本身。

### 5.3 Temporal-test SHAP consistency

![SHAP consistency](figures/Figure_26_SHAP_Test_Consistency.png)

Figure 26 将 development 与 temporal test（解释抽样 `n={test_n}`）的 mean(|SHAP|) 和 top-feature rank 做一致性对照。temporal test 的前列解释信号包括：{top_test}。该分析只用于描述外推解释是否大体稳定，不用于选择模型。

### 5.4 个体级局部解释

![高风险局部解释](figures/Figure_27_SHAP_Local_HighRisk.png)

![低风险局部解释](figures/Figure_28_SHAP_Local_LowRisk.png)

Figure 27-28 改为 waterfall-style 局部解释，不暴露真实 PID。每张图从 development background mean risk `E[f(x)]` 出发，逐项累加主要 SHAP contribution 到该患者预测风险 `f(x)`；红色表示提高复发风险，蓝色表示降低复发风险，`other features + residual` 汇总未展示的小贡献和近似残差。局部图的作用是审计单个预警或低风险判断由哪些输入驱动，而不是用个案替代总体性能评价。
{CH5_END}
"""


def update_file_index(text: str) -> str:
    anchor = "- `tables/feature_dictionary_zh.csv`：特征名中文含义。"
    additions = [
        "- `tables/shap_feature_importance_dev.csv`：development feature-level SHAP 重要性。",
        "- `tables/shap_feature_importance_temporal_test.csv`：temporal-test SHAP 一致性检查。",
        "- `tables/shap_branch_importance.csv`：按分支聚合的 SHAP 贡献。",
        "- `tables/shap_dev_test_rank_concordance.csv`：development 与 temporal-test SHAP 排名一致性。",
        "- `tables/shap_local_cases.csv`：Patient A/B 局部解释索引。",
        "- `tables/gate_values.csv`：显式 gate 与默认 branch multiplier。",
        "- `tables/shap_background_manifest.csv`：SHAP background / sample / scale 审计记录。",
        "- `tables/readme4_heatmap_marginals.csv`：Figure 19 的窗口边际统计。",
        "- `tables/joint_risk_window_heatmap.csv`：Figure 29 的风险分箱 x 随访窗口联合热图来源表。",
    ]
    for line in additions:
        if line not in text:
            text = text.replace(anchor, anchor + "\n" + line)
    return text


def update_readme(dev_imp: pd.DataFrame, test_imp: pd.DataFrame, branch_imp: pd.DataFrame, manifest: pd.DataFrame, concord: pd.DataFrame) -> None:
    text = README.read_text(encoding="utf-8")
    text = update_heatmap_caption(text)
    joint_block = (
        "![风险分箱联合热图](figures/Figure_29_Joint_Risk_Window_Marginal_Heatmap.png)\n\n"
        "Figure 29 是标准的 joint heatmap with marginal bar plots。横轴为 temporal test interval 的预测复发风险分箱，纵轴为随访窗口；"
        "主体热图颜色表示该风险分箱与窗口交叉格内的真实复发率，顶部边际柱/线显示各风险分箱的 interval 数和复发事件数，右侧边际柱显示各窗口的 interval 数。"
        "它回答的是：高风险分箱是否真的富集复发，以及这种富集是否集中在某些随访窗口。"
    )
    if "Figure 29 是标准的 joint heatmap" not in text:
        text = text.replace(
            "Figure 19 将 temporal test 的 patient-interval 风险矩阵化显示，并升级为带边际条图的热图：主矩阵中列是随访窗口、行是患者、颜色是该 interval 的复发风险；顶部边际条同时显示各窗口的平均预测风险和真实复发率；右侧边际条显示同一患者的 mean interval risk。复发患者和未复发患者分面展示，并在各自面板内按 mean risk 排序。",
            "Figure 19 将 temporal test 的 patient-interval 风险矩阵化显示，并升级为带边际条图的热图：主矩阵中列是随访窗口、行是患者、颜色是该 interval 的复发风险；顶部边际条同时显示各窗口的平均预测风险和真实复发率；右侧边际条显示同一患者的 mean interval risk。复发患者和未复发患者分面展示，并在各自面板内按 mean risk 排序。\n\n" + joint_block,
        )
    block = explainability_block(dev_imp, test_imp, branch_imp, manifest, concord)
    if CH5_START in text and CH5_END in text:
        start = text.index(CH5_START)
        end = text.index(CH5_END, start) + len(CH5_END)
        text = text[:start] + block.strip() + text[end:]
    else:
        anchor = "## 泄漏控制检查"
        text = text.replace(anchor, block.strip() + "\n\n" + anchor)
    text = update_file_index(text)
    README.write_text(text, encoding="utf-8")


def export_html() -> None:
    subprocess.run(
        [
            "pandoc",
            "复发.md",
            "--standalone",
            "--embed-resources",
            "--metadata",
            "title=3M 表征增强的四分支动态复发预警模型",
            "-o",
            "复发.html",
        ],
        cwd=REPORT,
        check=True,
    )


def validate_public_text() -> None:
    text = README.read_text(encoding="utf-8") + "\n" + HTML.read_text(encoding="utf-8")
    banned = ["seed=2025", "Figure 21", "Figure_21", "lead-time", "提前预警"]
    found = [word for word in banned if word in text]
    if found:
        raise RuntimeError(f"Public report still contains banned text: {found}")
    local_cases = pd.read_csv(TAB / "shap_local_cases.csv")
    if local_cases["Case_Label"].str.contains("Patient_ID|PID|210", regex=True).any():
        raise RuntimeError("Local case labels expose patient identifiers")


def main() -> None:
    set_style()
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    model, datasets, cfg, _feature_df, names = rebuild_model_data()
    prediction_recompute_check(model, datasets)
    branch_ablation = read_branch_ablation()
    gates = gate_values(model, datasets, names)
    heatmap_marginals()
    joint_risk_window_heatmap()
    try:
        dev_imp, test_imp, manifest, payload = compute_gradient_shap(model, datasets, names, cfg)
    except Exception as exc:
        pd.DataFrame(
            [
                {
                    "method": "failed",
                    "scale": "logit",
                    "failure": repr(exc),
                    "fallback": "not_generated",
                    "temporal_test_used_for_background": False,
                    "temporal_test_used_for_selection": False,
                }
            ]
        ).to_csv(TAB / "shap_background_manifest.csv", index=False)
        raise
    branch_imp, concord = shap_branch_tables(dev_imp, test_imp)
    fig_gate_branch(gates, branch_ablation, branch_imp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig_shap_beeswarm(payload)
    fig_shap_bar(dev_imp, branch_imp)
    fig_shap_consistency(concord)
    fig_local(payload, model, datasets)
    update_readme(dev_imp, test_imp, branch_imp, manifest, concord)
    export_html()
    validate_public_text()
    print(f"Updated explainability report: {README}")
    print(f"Updated HTML: {HTML}")


if __name__ == "__main__":
    main()
