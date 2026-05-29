#!/usr/bin/env python
"""M2 · B4 diagnostics — bias/variance, learning curve, confusion + hard cases.

Turns "is it under/over-fitting or feature-limited?" from alchemy into an explicit
answer:

A1. Bias-variance: train (in-sample) ROC vs OOF ROC vs temporal ROC, for the L2
    supermodel (low capacity), RandomForest (high capacity), and the B4 V1 GRU.
    train≈OOF → low variance → feature/Bayes ceiling; train≫OOF → overfitting.
A2. Learning curve vs sample size: OOF pooled ROC of the L2 supermodel at
    25/50/75/100% of dev episodes. Still rising at 100% → data-limited; flat →
    signal/feature-limited.
A3. Epoch learning curve: GRU train-ROC vs inner-val-ROC per epoch (overfit shape).

C.  Confusion matrix (per-landmark + pooled, Youden threshold) with sens/spec/
    PPV/NPV, and a hard-case profile: are the confidently-wrong episodes a
    structured subgroup (fixable) or diffuse noise (irreducible ceiling)?

Reuses the M2 shared pipeline. OOF for everything diagnostic; temporal only read
out, never tuned on.
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
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.simple.module2_v2_shared import (
    ALL_BLOCKS_FEATURES,
    BLOCKS,
    LANDMARKS,
    PRETTY,
    apply_per_landmark_platt,
    fit_l2_predict_oof,
    load_stacked,
    per_landmark_platt_on_pooled_oof,
    PY_SEED,
)
from scripts.simple.module2_v2_vertical_b4_gru import (
    TargetGRU,
    build_timeaware_tensors,
)

torch.set_num_threads(1)
OUT = ROOT / "results" / "module2_v2_vertical" / "b4_target_gru"
ABCDE = ("A", "B", "C", "D", "E")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pooled_roc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")


def _lr_pipe():
    return Pipeline([("s", StandardScaler()),
                     ("lr", LogisticRegression(penalty="l2", solver="lbfgs", C=1.0,
                                               max_iter=5000, random_state=PY_SEED))])


def oof_pooled_roc_lr(X, y, ep, *, seed=13):
    """5-fold StratifiedGroupKFold OOF L2-logistic pooled ROC on given arrays."""
    oof = np.zeros(len(y))
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y, groups=ep):
        pipe = _lr_pipe()
        pipe.fit(X[tr], y[tr])
        oof[va] = pipe.predict_proba(X[va])[:, 1]
    return _pooled_roc(y, oof)


# ---------------------------------------------------------------------------
# A1. Bias-variance: train vs OOF vs temporal
# ---------------------------------------------------------------------------


def diag_bias_variance(sd):
    X = sd.feature_matrix(ABCDE).values
    y = sd.rows["Y_24M_NHRH"].values
    ep = sd.rows["episode_id"].values
    is_dev = (sd.rows["Split"] == "Development").values
    is_test = ~is_dev
    Xd, yd, epd = X[is_dev], y[is_dev], ep[is_dev]

    rows = []

    # --- L2 supermodel ---
    oof_all, final = fit_l2_predict_oof(sd, block_subset=ABCDE)
    train_p = final.predict_proba(Xd)[:, 1]
    rows.append({"model": "L2 supermodel",
                 "train_ROC": _pooled_roc(yd, train_p),
                 "oof_ROC": _pooled_roc(yd, oof_all[is_dev]),
                 "temporal_ROC": _pooled_roc(y[is_test], oof_all[is_test])})

    # --- RandomForest (high capacity) ---
    rf = RandomForestClassifier(n_estimators=400, random_state=PY_SEED, n_jobs=1)
    rf.fit(Xd, yd)
    rf_train = rf.predict_proba(Xd)[:, 1]
    rf_oof = np.zeros(len(yd))
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=13)
    for tr, va in skf.split(Xd, yd, groups=epd):
        m = RandomForestClassifier(n_estimators=400, random_state=PY_SEED, n_jobs=1)
        m.fit(Xd[tr], yd[tr])
        rf_oof[va] = m.predict_proba(Xd[va])[:, 1]
    rf_tmp = rf.predict_proba(X[is_test])[:, 1]
    rows.append({"model": "RandomForest",
                 "train_ROC": _pooled_roc(yd, rf_train),
                 "oof_ROC": _pooled_roc(yd, rf_oof),
                 "temporal_ROC": _pooled_roc(y[is_test], rf_tmp)})

    # --- GRU V1 (one model on a train split, train vs held-out val) ---
    seq_X, dt, mask, seq_y, isdev_e, episodes, _, block_idx = build_timeaware_tensors(sd)
    dev_pos = np.where(isdev_e)[0]
    Xe, dte, me, ye = seq_X[dev_pos], dt[dev_pos], mask[dev_pos], seq_y[dev_pos]
    epe = episodes[dev_pos]
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=13)
    tr_e, va_e = next(skf.split(Xe, ye, groups=epe))
    gru_train_roc, gru_val_roc, _ = _train_gru_logging(Xe, dte, me, ye, block_idx, tr_e, va_e,
                                                        epochs=120, patience=15, log_curve=False)
    rows.append({"model": "B4 V1 GRU", "train_ROC": gru_train_roc, "oof_ROC": gru_val_roc,
                 "temporal_ROC": float("nan")})  # val on held-out fold = OOF proxy

    df = pd.DataFrame(rows)
    df["train_minus_oof"] = df["train_ROC"] - df["oof_ROC"]
    return df


def _train_gru_logging(Xe, dte, me, ye, block_idx, tr_e, va_e, *, epochs, patience, log_curve):
    """Train V1 GRU on tr_e, return (final train ROC, best val ROC, per-epoch curve)."""
    torch.manual_seed(PY_SEED); np.random.seed(PY_SEED)
    net = TargetGRU(block_idx, hidden=12, use_decay=False, zoneout=0.1)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    bce = nn.BCEWithLogitsLoss()
    Xt = torch.from_numpy(Xe[tr_e]); dtt = torch.from_numpy(dte[tr_e]); mt = torch.from_numpy(me[tr_e])
    yt = torch.from_numpy(ye[tr_e].astype(np.float32))
    Xv = torch.from_numpy(Xe[va_e]); dtv = torch.from_numpy(dte[va_e]); mv = torch.from_numpy(me[va_e])
    n = Xt.shape[0]
    curve = []
    best_val, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n, 64):
            b = perm[i:i + 64]
            loss = bce(net(Xt[b], dtt[b], mt[b])[0], yt[b].unsqueeze(1).expand(-1, 4))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        net.eval()
        with torch.no_grad():
            ptr = torch.sigmoid(net(Xt, dtt, mt)[0]).numpy()
            pva = torch.sigmoid(net(Xv, dtv, mv)[0]).numpy()
        tr_roc = _pooled_roc(np.repeat(ye[tr_e], 4), ptr.reshape(-1))
        va_roc = _pooled_roc(np.repeat(ye[va_e], 4), pva.reshape(-1))
        curve.append({"epoch": ep, "train_ROC": tr_roc, "val_ROC": va_roc})
        if va_roc > best_val + 1e-4:
            best_val, best_state, bad = va_roc, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience and not log_curve:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        ptr = torch.sigmoid(net(Xt, dtt, mt)[0]).numpy()
    final_train = _pooled_roc(np.repeat(ye[tr_e], 4), ptr.reshape(-1))
    return final_train, best_val, curve


# ---------------------------------------------------------------------------
# A2. Learning curve vs sample size (L2 supermodel)
# ---------------------------------------------------------------------------


def diag_learning_curve(sd):
    X = sd.feature_matrix(ABCDE).values
    y = sd.rows["Y_24M_NHRH"].values
    ep = sd.rows["episode_id"].values
    is_dev = (sd.rows["Split"] == "Development").values
    Xd, yd, epd = X[is_dev], y[is_dev], ep[is_dev]
    dev_eps = np.unique(epd)
    # episode-level outcome for stratified subsample
    ep_y = {e: yd[epd == e][0] for e in dev_eps}
    rows = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        rocs = []
        for seed in (1, 2, 3):
            rng = np.random.default_rng(seed)
            pos = np.array([e for e in dev_eps if ep_y[e] == 1])
            neg = np.array([e for e in dev_eps if ep_y[e] == 0])
            k_pos = max(2, int(len(pos) * frac)); k_neg = max(2, int(len(neg) * frac))
            sel = set(rng.choice(pos, k_pos, replace=False)) | set(rng.choice(neg, k_neg, replace=False))
            m = np.array([e in sel for e in epd])
            if frac >= 1.0:
                m = np.ones(len(epd), dtype=bool)
            rocs.append(oof_pooled_roc_lr(Xd[m], yd[m], epd[m], seed=13 + seed))
            if frac >= 1.0:
                break
        rows.append({"frac": frac, "n_episodes": int(round(len(dev_eps) * frac)),
                     "oof_ROC_mean": float(np.mean(rocs)), "oof_ROC_std": float(np.std(rocs))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# C. Confusion matrix + hard-case profile (L2 supermodel calibrated OOF)
# ---------------------------------------------------------------------------


def _confusion_at(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    npv = tn / (tn + fn) if tn + fn else float("nan")
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "Sens": sens, "Spec": spec, "PPV": ppv, "NPV": npv}


def diag_confusion_and_hardcases(sd):
    oof_all, _ = fit_l2_predict_oof(sd, block_subset=ABCDE)
    cal = apply_per_landmark_platt(sd, oof_all, per_landmark_platt_on_pooled_oof(sd, oof_all))
    y = sd.rows["Y_24M_NHRH"].values
    is_dev = (sd.rows["Split"] == "Development").values
    lm = sd.rows["landmark"].values

    # Youden threshold on dev OOF (pooled)
    yd, pd_ = y[is_dev], cal[is_dev]
    ths = np.linspace(0.05, 0.95, 181)
    j = [(_confusion_at(yd, pd_, t)["Sens"] + _confusion_at(yd, pd_, t)["Spec"] - 1) for t in ths]
    thr = float(ths[int(np.argmax(j))])

    conf_rows = []
    for label, mask in [(f"{L}M", is_dev & (lm == L)) for L in LANDMARKS] + [("Pooled", is_dev)]:
        c = _confusion_at(y[mask], cal[mask], thr)
        conf_rows.append({"split": "Dev(OOF)", "Landmark": label, "thr": round(thr, 3), **c})
    is_test = ~is_dev
    for label, mask in [(f"{L}M", is_test & (lm == L)) for L in LANDMARKS] + [("Pooled", is_test)]:
        c = _confusion_at(y[mask], cal[mask], thr)
        conf_rows.append({"split": "Temporal", "Landmark": label, "thr": round(thr, 3), **c})
    conf_df = pd.DataFrame(conf_rows)

    # Hard-case profile on dev OOF: confidently-wrong rows
    feats = list(ALL_BLOCKS_FEATURES)
    Xdf = sd.feature_matrix(ABCDE)
    dev_idx = np.where(is_dev)[0]
    p = cal[dev_idx]; yy = y[dev_idx]
    err = np.abs(yy - p)
    K = 40
    hard_fn = dev_idx[(yy == 1)][np.argsort(-(1 - p[yy == 1]))[:K]]   # relapsed, predicted lowest
    hard_fp = dev_idx[(yy == 0)][np.argsort(-(p[yy == 0]))[:K]]       # non-relapse, predicted highest
    correct = dev_idx[err < 0.25]
    prof = []
    key_feats = ["TSH_current", "FT4_current", "TSH_velocity", "FT4_velocity",
                 "TRAb", "TGAb", "TPOAb", "TSH_0M", "FT4_0M", "ThyroidW",
                 "log1p_DiseaseDuration_Months_Aug", "Uptake24h", "HalfLife"]
    for f in key_feats:
        if f not in sd.rows.columns:
            continue
        col = sd.rows[f].values
        prof.append({"feature": PRETTY.get(f, f),
                     "hard_FN_mean": float(np.nanmean(col[hard_fn])),
                     "hard_FP_mean": float(np.nanmean(col[hard_fp])),
                     "correct_mean": float(np.nanmean(col[correct]))})
    prof_df = pd.DataFrame(prof)
    # landmark distribution of hard cases (structured-in-time?)
    lm_dist = {
        "hard_FN": {f"{L}M": int((lm[hard_fn] == L).sum()) for L in LANDMARKS},
        "hard_FP": {f"{L}M": int((lm[hard_fp] == L).sum()) for L in LANDMARKS},
    }
    return conf_df, prof_df, lm_dist, thr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    print("Loading stacked dataset…", flush=True)
    sd = load_stacked()

    print("\n[A1] Bias-variance (train vs OOF vs temporal)…", flush=True)
    bv = diag_bias_variance(sd)
    bv.to_csv(OUT / "tables" / "diag_bias_variance.csv", index=False)
    print(bv.to_string(index=False), flush=True)

    print("\n[A2] Learning curve vs sample size (L2 supermodel)…", flush=True)
    lc = diag_learning_curve(sd)
    lc.to_csv(OUT / "tables" / "diag_learning_curve.csv", index=False)
    print(lc.to_string(index=False), flush=True)

    print("\n[A3] GRU epoch learning curve…", flush=True)
    seq_X, dt, mask, seq_y, isdev_e, episodes, _, block_idx = build_timeaware_tensors(sd)
    dev_pos = np.where(isdev_e)[0]
    Xe, dte, me, ye, epe = (seq_X[dev_pos], dt[dev_pos], mask[dev_pos], seq_y[dev_pos], episodes[dev_pos])
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=13)
    tr_e, va_e = next(skf.split(Xe, ye, groups=epe))
    _, _, curve = _train_gru_logging(Xe, dte, me, ye, block_idx, tr_e, va_e,
                                     epochs=120, patience=999, log_curve=True)
    cdf = pd.DataFrame(curve)
    cdf.to_csv(OUT / "tables" / "diag_gru_epoch_curve.csv", index=False)
    print(f"  final epoch train={cdf['train_ROC'].iloc[-1]:.3f} val={cdf['val_ROC'].iloc[-1]:.3f}; "
          f"best val={cdf['val_ROC'].max():.3f} @epoch {int(cdf['val_ROC'].idxmax())}", flush=True)

    print("\n[C] Confusion matrix + hard-case profile…", flush=True)
    conf_df, prof_df, lm_dist, thr = diag_confusion_and_hardcases(sd)
    conf_df.to_csv(OUT / "tables" / "diag_confusion.csv", index=False)
    prof_df.to_csv(OUT / "tables" / "diag_hardcase_profile.csv", index=False)
    print(f"  Youden threshold (dev OOF) = {thr:.3f}", flush=True)
    print(conf_df[conf_df["Landmark"] == "Pooled"].to_string(index=False), flush=True)
    print("\n  hard-case landmark distribution:", json.dumps(lm_dist), flush=True)
    print("\n  hard-case feature profile (FN=missed relapse, FP=false alarm):", flush=True)
    print(prof_df.to_string(index=False), flush=True)

    # Figures
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(lc["n_episodes"], lc["oof_ROC_mean"], yerr=lc["oof_ROC_std"], marker="o", color="#1d4e89")
    ax.set_xlabel("dev episodes used"); ax.set_ylabel("OOF pooled ROC (L2 supermodel)")
    ax.set_title("A2 learning curve — flat ⇒ feature/signal-limited")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "figures" / "Figure_04_LearningCurve.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cdf["epoch"], cdf["train_ROC"], label="train", color="#a23b3b")
    ax.plot(cdf["epoch"], cdf["val_ROC"], label="val (held-out fold)", color="#2a7f5f")
    ax.set_xlabel("epoch"); ax.set_ylabel("pooled ROC"); ax.legend()
    ax.set_title("A3 GRU epoch curve — gap ⇒ variance/overfit")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "figures" / "Figure_05_GRU_EpochCurve.png", dpi=160); plt.close(fig)

    summary = {
        "bias_variance": bv.to_dict(orient="records"),
        "learning_curve": lc.to_dict(orient="records"),
        "gru_curve_final": {"train": float(cdf["train_ROC"].iloc[-1]), "val": float(cdf["val_ROC"].iloc[-1]),
                            "best_val": float(cdf["val_ROC"].max())},
        "youden_threshold": thr,
        "confusion_pooled_dev": conf_df[(conf_df["split"] == "Dev(OOF)") & (conf_df["Landmark"] == "Pooled")].to_dict(orient="records"),
        "hardcase_landmark_dist": lm_dist,
    }
    (OUT / "tables" / "diagnostics_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved diagnostics → {OUT}", flush=True)


if __name__ == "__main__":
    main()
