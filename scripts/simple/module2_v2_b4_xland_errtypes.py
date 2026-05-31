#!/usr/bin/env python
"""M2 · B4 — per-landmark confusion + error-subtype mining with an HONEST
"can it be characterized?" verdict (Phase 4 task1, script 3).

For every landmark (1/3/6/12), on the corrected+LOCF EBM at its Youden@OOF cut:

  1. Confusion matrix (dev OOF + temporal read-out) with Sens/Spec/PPV/NPV.
  2. FN / FP feature profiles (confidently-wrong cases): mean feature value of
     missed-relapse (FN) vs false-alarm (FP) vs correct, per landmark.
  3. Error SUBTYPES — rule-first clinical labels on the FN/FP pool:
        FN-persistent-hyper : missed relapse, still Hyper at L (signal present,
                              model under-weights it)
        FN-silent           : missed relapse, euthyroid/Normal at L (no current
                              signal — the genuinely hard, look-normal relapse)
        FN-other            : missed relapse, Hypo/Unknown
        FP-big-goiter       : false alarm with large baseline goiter (ThyroidW
                              high) — anatomy-driven over-call
        FP-early-hyper      : false alarm still Hyper at L (slow normaliser that
                              does NOT relapse)
        FP-other            : remaining false alarms
     + KMeans(k=3) on standardized FN/FP features as an unsupervised CHECK.
  4. Subtype proportion DRIFT across landmarks (do the subtypes shift in mix?).
  5. **诚实判定:能否概括** — three pre-registered gates:
        G1 silhouette(KMeans on errors, dev OOF pooled) ≥ 0.25
        G2 rule↔cluster agreement ARI ≥ 0.10 (rules align with unsupervised)
        G3 ≥1 subtype holds ≥15% share at ≥3 of the 4 landmarks (cross-landmark
           persistence)
     ALL gates → "errors ARE characterizable (coherent subtypes)".
     ANY gate fails → spelled out plainly: "errors are diffuse → feature ceiling"
     (a negative result is reported honestly, not hidden).

口径 = corrected+LOCF, EBM-OOF dev / temporal read-out, naive+persistence,
time-safe. Functional state at 6M/12M derived from corrected hormones.

Outputs (results/module2_v2_vertical/b4_target_gru/):
    tables/xland_confusion.csv
    tables/xland_fn_fp_profile.csv
    tables/xland_error_subtypes.csv
    tables/xland_subtype_drift.csv
    tables/xland_characterizable_verdict.json
    tables/xland_errtypes_summary.json
    figures/xland_Figure_04_SubtypeDrift.png
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
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
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from scripts.simple.module2_v2_b4_xland_shared import (
    PROFILE_FEATS,
    TBL,
    FIG,
    build_tracking_long,
    ensure_dirs,
    feature_rows,
)

# pre-registered gates (declared BEFORE looking at results)
GATE_SILHOUETTE = 0.25
GATE_ARI = 0.10
GATE_SHARE = 0.15
GATE_NLM = 3  # subtype must clear GATE_SHARE at ≥3 landmarks

SUBTYPE_FN = ("FN-persistent-hyper", "FN-silent", "FN-other")
SUBTYPE_FP = ("FP-big-goiter", "FP-early-hyper", "FP-other")
ALL_SUBTYPES = SUBTYPE_FN + SUBTYPE_FP

PRETTY = {
    "TSH_current": "TSH(current)", "FT3_current": "FT3(current)", "FT4_current": "FT4(current)",
    "TSH_velocity": "TSH velocity", "FT3_velocity": "FT3 velocity", "FT4_velocity": "FT4 velocity",
    "TRAb": "TRAb", "TGAb": "TGAb", "TPOAb": "TPOAb", "TSH_0M": "TSH(0M)", "FT4_0M": "FT4(0M)",
    "ThyroidW": "甲状腺重量", "log1p_DiseaseDuration_Months_Aug": "病程(log)",
    "Uptake24h": "24h摄取率", "HalfLife": "碘半衰期", "Sex": "性别",
}


# ---------------------------------------------------------------------------
# confusion
# ---------------------------------------------------------------------------


def _confusion(y, pred_label):
    tp = int(((pred_label == 1) & (y == 1)).sum())
    fp = int(((pred_label == 1) & (y == 0)).sum())
    tn = int(((pred_label == 0) & (y == 0)).sum())
    fn = int(((pred_label == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "Sens": round(sens, 4), "Spec": round(spec, 4),
            "PPV": round(ppv, 4), "NPV": round(npv, 4)}


def confusion_table(long_df, landmarks):
    recs = []
    for split in ("Development", "Temporal"):
        sub = long_df[long_df["Split"] == split]
        for L in sorted(landmarks):
            g = sub[sub["landmark"] == L]
            if g.empty:
                continue
            c = _confusion(g["Y"].values, g["pred_label"].values)
            recs.append({"split": "Dev(OOF)" if split == "Development" else "Temporal",
                         "Landmark": f"{L}M", "thr": round(float(g["thr"].iloc[0]), 3), **c})
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------
# FN / FP confidently-wrong pool (dev OOF) + profiles
# ---------------------------------------------------------------------------


def _merge_feats(long_df, feat_df):
    """Attach PROFILE_FEATS to the long tracking frame by (episode, landmark)."""
    return long_df.merge(feat_df, on=["episode_id", "landmark"], how="left",
                         suffixes=("", "_f"))


def fn_fp_pool(merged, L, split="Development", k_conf=40):
    """Confidently-wrong FN/FP at landmark L on the given split.

    FN = relapsed (Y=1) predicted lowest p; FP = non-relapse (Y=0) predicted
    highest p. Returns (fn_df, fp_df, correct_df) feature sub-frames.
    """
    g = merged[(merged["landmark"] == L) & (merged["Split"] == split)].copy()
    g_pos = g[g["Y"] == 1].sort_values("p")           # lowest p first → most-missed
    g_neg = g[g["Y"] == 0].sort_values("p", ascending=False)  # highest p → worst alarm
    fn = g_pos.head(min(k_conf, len(g_pos)))
    fp = g_neg.head(min(k_conf, len(g_neg)))
    corr = g[(g["p"] - g["Y"]).abs() < 0.25]
    return fn, fp, corr


def profile_table(merged, landmarks):
    recs = []
    for L in sorted(landmarks):
        fn, fp, corr = fn_fp_pool(merged, L)
        for f in PROFILE_FEATS:
            recs.append({
                "Landmark": f"{L}M", "feature": PRETTY.get(f, f),
                "FN_mean": round(float(np.nanmean(fn[f])) if len(fn) else float("nan"), 3),
                "FP_mean": round(float(np.nanmean(fp[f])) if len(fp) else float("nan"), 3),
                "correct_mean": round(float(np.nanmean(corr[f])) if len(corr) else float("nan"), 3),
                "n_FN": int(len(fn)), "n_FP": int(len(fp)),
            })
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------
# rule-first subtypes
# ---------------------------------------------------------------------------


def _rule_label(row, kind):
    """Assign a clinical error-subtype label to one FN or FP row."""
    state = row["state"]
    if kind == "FN":
        if state == "Hyper":
            return "FN-persistent-hyper"
        if state == "Normal":
            return "FN-silent"
        return "FN-other"
    # FP
    # big goiter = ThyroidW in the upper tertile is judged at pool level; here use
    # a fixed clinically-large cut as a fallback, refined by caller via threshold.
    if row.get("_big_goiter", False):
        return "FP-big-goiter"
    if state == "Hyper":
        return "FP-early-hyper"
    return "FP-other"


def subtype_assignments(merged, landmarks):
    """Per-landmark rule subtype counts on the dev-OOF confidently-wrong pool."""
    big_cut = np.nanpercentile(merged["ThyroidW"].values, 66.7)  # global upper-tertile goiter
    rows = []
    detail = {}
    for L in sorted(landmarks):
        fn, fp, _ = fn_fp_pool(merged, L)
        fp = fp.copy()
        fp["_big_goiter"] = fp["ThyroidW"].values >= big_cut
        labels = []
        for _, r in fn.iterrows():
            labels.append(_rule_label(r, "FN"))
        for _, r in fp.iterrows():
            labels.append(_rule_label(r, "FP"))
        detail[L] = {"labels": labels,
                     "fn_idx": fn.index.tolist(), "fp_idx": fp.index.tolist(),
                     "fn": fn, "fp": fp}
        from collections import Counter
        cnt = Counter(labels)
        n_tot = max(len(labels), 1)
        for st in ALL_SUBTYPES:
            rows.append({"Landmark": f"{L}M", "_L": L, "subtype": st,
                         "n": int(cnt.get(st, 0)),
                         "share": round(cnt.get(st, 0) / n_tot, 4)})
    return pd.DataFrame(rows), detail, big_cut


# ---------------------------------------------------------------------------
# KMeans unsupervised check + honest verdict
# ---------------------------------------------------------------------------


def kmeans_check(merged, landmarks, detail):
    """Pooled KMeans(k=3) on standardized errors → silhouette + ARI vs rules.

    Pool ALL confidently-wrong FN/FP rows across landmarks (dev OOF). Cluster the
    standardized PROFILE_FEATS; silhouette measures whether the errors form
    separable clusters at all; ARI compares the unsupervised partition to the
    rule labels (do clinical rules recover an actual structure?).
    """
    frames = []
    rule_labels = []
    for L in sorted(landmarks):
        d = detail[L]
        pool = pd.concat([d["fn"], d["fp"]], axis=0)
        if pool.empty:
            continue
        frames.append(pool[PROFILE_FEATS])
        rule_labels.extend(d["labels"])
    if not frames:
        return {"silhouette": float("nan"), "ari": float("nan"), "n_pool": 0,
                "k": 3, "cluster_sizes": []}
    X = pd.concat(frames, axis=0).values
    X = np.nan_to_num(X, nan=np.nanmean(X))
    Xs = StandardScaler().fit_transform(X)
    k = 3 if len(Xs) >= 6 else 2
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)
    sil = float(silhouette_score(Xs, km.labels_)) if len(np.unique(km.labels_)) > 1 else float("nan")
    # ARI between rule labels and clusters (encode rule labels to ints)
    rl = pd.Series(rule_labels).astype("category").cat.codes.values
    ari = float(adjusted_rand_score(rl, km.labels_)) if len(rl) == len(km.labels_) else float("nan")
    from collections import Counter
    sizes = sorted(Counter(km.labels_.tolist()).values(), reverse=True)
    return {"silhouette": round(sil, 4), "ari": round(ari, 4),
            "n_pool": int(len(Xs)), "k": k, "cluster_sizes": [int(s) for s in sizes]}


def cross_landmark_persistence(sub_df, landmarks):
    """Does any subtype hold ≥GATE_SHARE at ≥GATE_NLM landmarks?"""
    persist = {}
    n_lm = len(landmarks)
    for st in ALL_SUBTYPES:
        s = sub_df[sub_df["subtype"] == st]
        n_clear = int((s["share"] >= GATE_SHARE).sum())
        persist[st] = {"n_landmarks_ge_share": n_clear, "of": n_lm,
                       "passes": n_clear >= min(GATE_NLM, n_lm)}
    any_pass = any(v["passes"] for v in persist.values())
    return persist, any_pass


def verdict(km, persist_any, n_landmarks):
    g1 = (km["silhouette"] >= GATE_SILHOUETTE) if np.isfinite(km["silhouette"]) else False
    g2 = (km["ari"] >= GATE_ARI) if np.isfinite(km["ari"]) else False
    # G3 needs ≥3 landmarks to be meaningful; if fewer (quick), mark indeterminate
    g3 = bool(persist_any) if n_landmarks >= GATE_NLM else False
    passed = bool(g1 and g2 and g3)
    if passed:
        statement = (
            "错例可概括:三门槛全过 —— KMeans 轮廓系数达标(误例成簇)、规则↔聚类 ARI 达标"
            "(临床规则与无监督结构一致)、且至少一个亚型在≥3个地标维持≥15%占比(跨地标稳定)。"
            "可按亚型分诊。"
        )
    else:
        fails = []
        if not g1:
            fails.append(f"轮廓系数 {km['silhouette']} < {GATE_SILHOUETTE}(误例不成簇)")
        if not g2:
            fails.append(f"规则-聚类 ARI {km['ari']} < {GATE_ARI}(规则与无监督结构不一致)")
        if not g3:
            if n_landmarks < GATE_NLM:
                fails.append(f"地标数 {n_landmarks} < {GATE_NLM}(跨地标占比门槛不可判,需全跑)")
            else:
                fails.append(f"无亚型在≥{GATE_NLM}个地标维持≥{int(GATE_SHARE*100)}%占比(亚型不跨地标稳定)")
        statement = (
            "错例弥散 → 特征天花板:未过门槛(" + "；".join(fails) + ")。"
            "误例不形成可分、跨地标稳定的临床亚型,残余错误更像现有特征集的内在上限"
            "(看似正常却复发的静默型 + 不可约噪声),而非某个可命名、可补特征修复的子群。"
            "(negative result,据实报告)"
        )
    return {
        "gates": {
            "G1_silhouette": {"value": km["silhouette"], "threshold": GATE_SILHOUETTE, "pass": bool(g1)},
            "G2_rule_cluster_ARI": {"value": km["ari"], "threshold": GATE_ARI, "pass": bool(g2)},
            "G3_cross_landmark_share": {"any_subtype_persists": bool(persist_any),
                                        "min_share": GATE_SHARE, "min_landmarks": GATE_NLM,
                                        "pass": bool(g3)},
        },
        "all_pass": passed,
        "verdict": "characterizable" if passed else "diffuse_feature_ceiling",
        "statement": statement,
    }


# ---------------------------------------------------------------------------
# drift figure
# ---------------------------------------------------------------------------


def _plot_drift(sub_df, landmarks):
    Ls = sorted(landmarks)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    colors = {"FN-persistent-hyper": "#c1121f", "FN-silent": "#6a040f", "FN-other": "#e85d75",
              "FP-big-goiter": "#1d4e89", "FP-early-hyper": "#168aad", "FP-other": "#90c2e7"}
    for st in ALL_SUBTYPES:
        s = sub_df[sub_df["subtype"] == st].sort_values("_L")
        ax.plot(s["_L"], s["share"], marker="o", label=st, color=colors.get(st))
    ax.axhline(GATE_SHARE, ls=":", color="#555", lw=1)
    ax.text(Ls[0], GATE_SHARE + 0.01, f"占比门槛 {int(GATE_SHARE*100)}%", fontsize=8, color="#555")
    ax.set_xticks(Ls); ax.set_xticklabels([f"{L}M" for L in Ls])
    ax.set_xlabel("地标 (landmark)"); ax.set_ylabel("亚型占错例池比例")
    ax.set_title("错例亚型比例漂移(规则标注,dev OOF 置信错例池)", fontsize=10)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "xland_Figure_04_SubtypeDrift.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke: 6M only")
    args = ap.parse_args()
    ensure_dirs()

    long_df, meta = build_tracking_long(quick=args.quick)
    landmarks = meta["landmarks"]
    feat_df = feature_rows(quick=args.quick)
    merged = _merge_feats(long_df, feat_df)

    conf = confusion_table(long_df, landmarks)
    conf.to_csv(TBL / "xland_confusion.csv", index=False)

    prof = profile_table(merged, landmarks)
    prof.to_csv(TBL / "xland_fn_fp_profile.csv", index=False)

    sub_df, detail, big_cut = subtype_assignments(merged, landmarks)
    sub_df.drop(columns=["_L"]).to_csv(TBL / "xland_error_subtypes.csv", index=False)

    # drift = the same sub_df pivoted (share by subtype × landmark)
    drift = sub_df.pivot(index="subtype", columns="Landmark", values="share").reset_index()
    drift.to_csv(TBL / "xland_subtype_drift.csv", index=False)
    if len(landmarks) > 1:
        _plot_drift(sub_df, landmarks)

    km = kmeans_check(merged, landmarks, detail)
    persist, persist_any = cross_landmark_persistence(sub_df, landmarks)
    vd = verdict(km, persist_any, len(landmarks))

    # ---- report ----
    print("=== (1) 逐地标混淆矩阵 (EBM, Youden@OOF) ===", flush=True)
    print(conf.to_string(index=False), flush=True)

    print("\n=== (2) FN(漏报复发)/FP(误报)特征画像 — 节选 ===", flush=True)
    show = prof[prof["feature"].isin(["TSH(current)", "FT4(current)", "FT3 velocity",
                                      "TRAb", "甲状腺重量", "病程(log)"])]
    print(show.to_string(index=False), flush=True)

    print("\n=== (3) 错例亚型(规则优先,dev OOF 置信错例池)===", flush=True)
    print(sub_df.drop(columns=["_L"]).to_string(index=False), flush=True)
    print(f"\n  KMeans 无监督校验: silhouette={km['silhouette']} | rule↔cluster ARI={km['ari']} "
          f"| n_pool={km['n_pool']} k={km['k']} sizes={km['cluster_sizes']}", flush=True)

    print("\n=== (4) 亚型比例漂移 (share by subtype × landmark) ===", flush=True)
    print(drift.to_string(index=False), flush=True)

    print("\n=== (5) 【诚实判定:错例能否概括?】三门槛(预注册) ===", flush=True)
    for gk, gv in vd["gates"].items():
        print(f"  {gk}: {gv}", flush=True)
    print(f"  跨地标占比持久性: {json.dumps(persist, ensure_ascii=False)}", flush=True)
    print(f"\n  >>> 判定: {vd['verdict'].upper()} (all_pass={vd['all_pass']})", flush=True)
    print(f"  >>> {vd['statement']}", flush=True)

    (TBL / "xland_characterizable_verdict.json").write_text(
        json.dumps({**vd, "kmeans": km, "cross_landmark_persistence": persist,
                    "big_goiter_cut_ThyroidW": round(float(big_cut), 3)},
                   indent=2, ensure_ascii=False)
    )
    (TBL / "xland_errtypes_summary.json").write_text(
        json.dumps({
            "口径": "corrected+LOCF, EBM-OOF dev / temporal read-out, time-safe",
            "landmarks": [f"{L}M" for L in landmarks],
            "confusion": conf.to_dict(orient="records"),
            "subtypes": sub_df.drop(columns=["_L"]).to_dict(orient="records"),
            "kmeans_check": km,
            "verdict": vd,
            "note": "FN/FP pools = top-40 confidently-wrong per class per landmark (dev OOF). "
            "State at 6M/12M derived from corrected hormones.",
        }, indent=2, ensure_ascii=False)
    )
    print(f"\nSaved → {TBL}  &  {FIG}", flush=True)


if __name__ == "__main__":
    main()
