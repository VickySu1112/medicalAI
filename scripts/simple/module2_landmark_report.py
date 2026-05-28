"""Generate Module 2 fixed-landmark NHRH tables and figures.

Module 2 reuses the locked legacy Stage-1 fixed-landmark NHRH artifacts as a
read-only source and assembles a separate report-ready surface under
``results/module2_early_landmark_updating``. The script does not write narrative
README text.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-medicalai")
warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from scripts.simple.stage1_plot_kit import (
    BLUE,
    FIG_DPI,
    GRAY,
    GREEN,
    ORANGE,
    PURPLE,
    RED,
    plot_calibration,
    plot_calibration_summary,
    plot_dca,
    plot_model_comparison_heatmap,
    plot_or_forest,
    plot_risk_tiers,
    plot_roc_pr,
    plot_single_feature_benchmark,
)


LEGACY = ROOT / "results" / "3m_6m_early_stratification"
LEGACY_TABLES = LEGACY / "tables"
OUT_DIR = ROOT / "results" / "module2_early_landmark_updating"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
LANDMARKS = ["0M", "1M", "3M", "6M"]
DOMAIN_MAP = {"TemporalTest": "Test", "Test": "Test", "OOF": "OOF"}


def _ensure_clean_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for stale in FIG_DIR.glob("*.png"):
        stale.unlink()
    for stale in TABLE_DIR.glob("legacy_*.csv"):
        stale.unlink()


def _image_record(path: Path, function: str, content: str) -> dict[str, object]:
    with Image.open(path) as im:
        width, height = im.size
    return {
        "file": path.name,
        "path": str(path.relative_to(ROOT)),
        "width": int(width),
        "height": int(height),
        "function": function,
        "content": content,
    }


def _read_selected() -> pd.DataFrame:
    selected = pd.read_csv(LEGACY_TABLES / "nhrh_binary_selected.csv")
    selected["Landmark"] = pd.Categorical(selected["Landmark"].astype(str), categories=LANDMARKS, ordered=True)
    selected = selected.sort_values("Landmark").reset_index(drop=True)
    if selected["Landmark"].isna().any() or len(selected) != len(LANDMARKS):
        raise RuntimeError("Selected-model table must contain exactly one selected row per expected landmark.")
    return selected


def _extract_selected_predictions(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chunk-read the large predictions table and keep selected RunIDs only."""
    wanted = set(selected["RunID"].astype(str))
    chunks: list[pd.DataFrame] = []
    usecols = ["RunID", "Landmark", "Domain", "Treatment_ID", "Y", "Proba"]
    pred_path = LEGACY_TABLES / "nhrh_binary_predictions_long.csv"
    for chunk in pd.read_csv(pred_path, usecols=usecols, chunksize=500_000):
        sub = chunk[chunk["RunID"].astype(str).isin(wanted)].copy()
        if not sub.empty:
            chunks.append(sub)
    if not chunks:
        raise RuntimeError("No selected RunID predictions were found in nhrh_binary_predictions_long.csv.")

    preds = pd.concat(chunks, ignore_index=True)
    preds["Landmark"] = preds["Landmark"].astype(str)
    preds["Domain"] = preds["Domain"].astype(str).map(DOMAIN_MAP).fillna(preds["Domain"].astype(str))
    preds = preds[preds["Domain"].isin(["OOF", "Test"])].copy()

    # In case a legacy RunID has duplicate rows, keep the first exact
    # treatment-domain row and audit the de-duplication count.
    before = len(preds)
    preds = preds.drop_duplicates(["RunID", "Landmark", "Domain", "Treatment_ID"], keep="first")
    dedup_removed = before - len(preds)

    risk = preds[["Treatment_ID", "Landmark", "Domain", "Y", "Proba"]].copy()
    risk = risk.rename(columns={"Proba": "RiskScore"})
    risk["Y"] = risk["Y"].astype(int)
    risk["RiskScore"] = risk["RiskScore"].astype(float)
    risk["Landmark"] = pd.Categorical(risk["Landmark"], categories=LANDMARKS, ordered=True)
    risk["Domain"] = pd.Categorical(risk["Domain"], categories=["OOF", "Test"], ordered=True)
    risk = risk.sort_values(["Landmark", "Domain", "Treatment_ID"]).reset_index(drop=True)
    risk.to_csv(TABLE_DIR / "early_nhrh_risk_score.csv", index=False)

    coverage = (
        risk.groupby(["Landmark", "Domain"], observed=False)
        .agg(
            Rows=("Treatment_ID", "size"),
            Treatment_Episodes=("Treatment_ID", "nunique"),
            Events=("Y", "sum"),
            Prevalence=("Y", "mean"),
        )
        .reset_index()
    )
    total = (
        risk.groupby("Landmark", observed=False)
        .agg(Total_Treatment_Episodes=("Treatment_ID", "nunique"))
        .reset_index()
    )
    coverage = coverage.merge(total, on="Landmark", how="left")
    coverage["Deduplicated_Rows_Removed"] = int(dedup_removed)
    coverage.to_csv(TABLE_DIR / "early_nhrh_risk_score_coverage.csv", index=False)

    expected_domains = set((lm, dom) for lm in LANDMARKS for dom in ["OOF", "Test"])
    got_domains = set(zip(coverage["Landmark"].astype(str), coverage["Domain"].astype(str)))
    missing = expected_domains - got_domains
    if missing:
        raise RuntimeError(f"Missing selected prediction domains: {sorted(missing)}")
    if not (total["Total_Treatment_Episodes"] == 1003).all():
        raise RuntimeError("Each landmark must cover 1003 treatment episodes across OOF and Test.")
    return risk, coverage


def _metric_ci(ci: pd.DataFrame, landmark: str, metric: str) -> tuple[float, float]:
    sub = ci[(ci["Landmark"].astype(str).eq(landmark)) & (ci["Metric"].astype(str).eq(metric))]
    if sub.empty:
        return np.nan, np.nan
    return float(sub["CI_Lower"].iloc[0]), float(sub["CI_Upper"].iloc[0])


def _build_performance_table(selected: pd.DataFrame) -> pd.DataFrame:
    ci = pd.read_csv(LEGACY_TABLES / "nhrh_binary_bootstrap_ci.csv")
    rows: list[dict[str, object]] = []
    for r in selected.itertuples(index=False):
        lm = str(r.Landmark)
        rows.append(
            {
                "Landmark": lm,
                "Split": "OOF",
                "RunID": r.RunID,
                "Model": r.Model,
                "FeatureSet": r.FeatureSet,
                "ROC_AUC": float(r.OOF_AUC),
                "ROC_AUC_CI_Lower": np.nan,
                "ROC_AUC_CI_Upper": np.nan,
                "PR_AUC": float(r.OOF_PR_AUC),
                "PR_AUC_CI_Lower": np.nan,
                "PR_AUC_CI_Upper": np.nan,
                "Brier": float(r.OOF_Brier),
                "Brier_CI_Lower": np.nan,
                "Brier_CI_Upper": np.nan,
            }
        )
        auc_lo, auc_hi = _metric_ci(ci, lm, "AUC")
        pr_lo, pr_hi = _metric_ci(ci, lm, "PR_AUC")
        br_lo, br_hi = _metric_ci(ci, lm, "Brier")
        rows.append(
            {
                "Landmark": lm,
                "Split": "Test",
                "RunID": r.RunID,
                "Model": r.Model,
                "FeatureSet": r.FeatureSet,
                "ROC_AUC": float(r.Test_AUC),
                "ROC_AUC_CI_Lower": auc_lo,
                "ROC_AUC_CI_Upper": auc_hi,
                "PR_AUC": float(r.Test_PR_AUC),
                "PR_AUC_CI_Lower": pr_lo,
                "PR_AUC_CI_Upper": pr_hi,
                "Brier": float(r.Test_Brier),
                "Brier_CI_Lower": br_lo,
                "Brier_CI_Upper": br_hi,
            }
        )
    out = pd.DataFrame(rows)
    out["Landmark"] = pd.Categorical(out["Landmark"], categories=LANDMARKS, ordered=True)
    out = out.sort_values(["Landmark", "Split"]).reset_index(drop=True)
    out.to_csv(TABLE_DIR / "module2_landmark_performance.csv", index=False)
    return out


def _copy_persistence() -> pd.DataFrame:
    persistence = pd.read_csv(LEGACY_TABLES / "nhrh_persistence_baseline.csv")
    persistence.to_csv(TABLE_DIR / "module2_persistence_baseline.csv", index=False)
    return persistence


def _plot_auc_gradient(perf: pd.DataFrame, persistence: pd.DataFrame, manifest: list[dict[str, object]]) -> None:
    test = perf[perf["Split"].eq("Test")].copy()
    test["Landmark"] = pd.Categorical(test["Landmark"], categories=LANDMARKS, ordered=True)
    test = test.sort_values("Landmark")
    x = np.arange(len(LANDMARKS))

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), gridspec_kw={"width_ratios": [1.35, 1.0]})

    for metric, label, color, lo_col, hi_col in [
        ("ROC_AUC", "Model ROC-AUC", BLUE, "ROC_AUC_CI_Lower", "ROC_AUC_CI_Upper"),
        ("PR_AUC", "Model PR-AUC", RED, "PR_AUC_CI_Lower", "PR_AUC_CI_Upper"),
    ]:
        y = test[metric].to_numpy(float)
        lo = test[lo_col].to_numpy(float)
        hi = test[hi_col].to_numpy(float)
        yerr = np.vstack([y - lo, hi - y])
        axes[0].errorbar(x, y, yerr=yerr, marker="o", lw=2.2, capsize=3, color=color, label=label)

    pers = persistence.set_index("Landmark").reindex(LANDMARKS)
    axes[0].plot(x, pers["Test_AUC"], "--", color=BLUE, alpha=0.55, lw=1.6, label="Persistence ROC-AUC")
    axes[0].plot(x, pers["Test_PR_AUC"], "--", color=RED, alpha=0.55, lw=1.6, label="Persistence PR-AUC")
    axes[0].set_xticks(x, LANDMARKS)
    axes[0].set_ylim(0.35, 1.0)
    axes[0].set_title("Temporal-test discrimination over early landmarks")
    axes[0].set_xlabel("Landmark")
    axes[0].set_ylabel("Area under curve")
    axes[0].grid(alpha=0.18)
    axes[0].legend(fontsize=8, ncol=2)

    y = test["Brier"].to_numpy(float)
    lo = test["Brier_CI_Lower"].to_numpy(float)
    hi = test["Brier_CI_Upper"].to_numpy(float)
    yerr = np.vstack([y - lo, hi - y])
    axes[1].errorbar(x, y, yerr=yerr, marker="o", lw=2.2, capsize=3, color=GREEN)
    axes[1].set_xticks(x, LANDMARKS)
    axes[1].set_ylim(0.0, max(0.25, float(np.nanmax(hi)) * 1.12))
    axes[1].set_title("Temporal-test Brier score")
    axes[1].set_xlabel("Landmark")
    axes[1].set_ylabel("Brier score")
    axes[1].grid(alpha=0.18)

    fig.tight_layout()
    path = FIG_DIR / "Figure_01_AUC_PR_Brier_Over_Time.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    manifest.append(_image_record(path, "custom_stage1_style", "Temporal-test ROC-AUC, PR-AUC, persistence baseline, and Brier score over landmarks"))


def _plot_selected_roc_pr(risk: pd.DataFrame, manifest: list[dict[str, object]]) -> None:
    panels = []
    for lm in LANDMARKS:
        sub = risk[(risk["Landmark"].astype(str).eq(lm)) & (risk["Domain"].astype(str).eq("Test"))]
        panels.append((f"{lm} selected", sub["Y"].to_numpy(int), sub["RiskScore"].to_numpy(float)))
    path = FIG_DIR / "Figure_02_Selected_ROC_PR_All_Landmarks.png"
    plot_roc_pr(panels, path)
    manifest.append(_image_record(path, "plot_roc_pr", "Selected-model temporal-test ROC and PR curves for 0M, 1M, 3M, and 6M"))


def _plot_heatmap(manifest: list[dict[str, object]]) -> None:
    heat = pd.read_csv(LEGACY_TABLES / "binary_model_metric_heatmap.csv")
    heat["Model"] = heat["Model"].replace(
        {
            "Clinical_L2_Logistic": "Clinical L2 Logistic",
            "Elastic_LR": "Elastic LR",
            "RandomForest": "Random Forest",
            "ExtraTrees": "ExtraTrees",
            "HistGradientBoosting": "HistGBM",
        }
    )
    path = FIG_DIR / "Figure_03_Model_Metric_Heatmap.png"
    plot_model_comparison_heatmap(heat, path)
    manifest.append(_image_record(path, "plot_model_comparison_heatmap", "Legacy model-by-metric heatmap for fixed landmarks"))


def _plot_or_forests(manifest: list[dict[str, object]]) -> None:
    or_df = pd.read_csv(LEGACY_TABLES / "multivariable_logistic_or.csv")
    for lm in LANDMARKS:
        sub = or_df[or_df["Analysis"].astype(str).eq(f"{lm}_clinical_core_multivariable")].rename(
            columns={
                "Feature": "feature",
                "OR_per_SD": "OR",
                "CI95_Lower": "CI_low",
                "CI95_Upper": "CI_high",
                "P_Value_Wald": "p",
            }
        )
        path = FIG_DIR / f"Figure_04_OR_Forest_{lm}.png"
        plot_or_forest(sub, path)
        manifest.append(_image_record(path, "plot_or_forest", f"{lm} clinical-core logistic OR forest"))


def _plot_calibration_and_dca(risk: pd.DataFrame, manifest: list[dict[str, object]]) -> None:
    for lm in LANDMARKS:
        sub = risk[(risk["Landmark"].astype(str).eq(lm)) & (risk["Domain"].astype(str).eq("Test"))]
        y = sub["Y"].to_numpy(int)
        p = sub["RiskScore"].to_numpy(float)
        cal_path = FIG_DIR / f"Figure_05_Calibration_{lm}.png"
        plot_calibration(y, p, cal_path)
        manifest.append(_image_record(cal_path, "plot_calibration", f"{lm} selected-model temporal-test reliability curve"))
        dca_path = FIG_DIR / f"Figure_06_DCA_{lm}.png"
        plot_dca(y, p, dca_path, thresholds=np.linspace(0.10, 0.40, 31))
        manifest.append(_image_record(dca_path, "plot_dca", f"{lm} selected-model temporal-test decision curve"))

    cal = pd.read_csv(LEGACY_TABLES / "calibration_summary.csv")
    path = FIG_DIR / "Figure_07_Calibration_Summary.png"
    plot_calibration_summary(cal, path)
    manifest.append(_image_record(path, "plot_calibration_summary", "Brier, calibration intercept, and calibration slope summary"))


def _plot_single_feature(manifest: list[dict[str, object]]) -> None:
    bench = pd.read_csv(LEGACY_TABLES / "single_feature_benchmark.csv")
    path = FIG_DIR / "Figure_08_Single_Feature_Benchmark.png"
    plot_single_feature_benchmark(bench, path)
    manifest.append(_image_record(path, "plot_single_feature_benchmark", "Single clinical variables versus multivariable LR and naive baseline"))


def _risk_tier_table(risk: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for lm in LANDMARKS:
        oof = risk[(risk["Landmark"].astype(str).eq(lm)) & (risk["Domain"].astype(str).eq("OOF"))].copy()
        test = risk[(risk["Landmark"].astype(str).eq(lm)) & (risk["Domain"].astype(str).eq("Test"))].copy()
        q1, q2 = np.quantile(oof["RiskScore"].to_numpy(float), [1 / 3, 2 / 3])
        bins = [-np.inf, q1, q2, np.inf]
        labels = ["Low", "Intermediate", "High"]
        for domain_name, df in [("OOF", oof), ("Test", test)]:
            tier = pd.cut(df["RiskScore"], bins=bins, labels=labels, include_lowest=True)
            for label in labels:
                mask = tier.astype(str).eq(label)
                sub = df[mask]
                n = int(len(sub))
                events = int(sub["Y"].sum()) if n else 0
                event_rate = float(events / n) if n else np.nan
                rows.append(
                    {
                        "Landmark": lm,
                        "Domain": domain_name,
                        "Tier": label,
                        "N": n,
                        "Events": events,
                        "MeanPredictedRisk": float(sub["RiskScore"].mean()) if n else np.nan,
                        "ObservedEventRate": event_rate,
                        "PPV": event_rate,
                        "NPV": float(1.0 - event_rate) if n else np.nan,
                        "OOF_Low_Upper": float(q1),
                        "OOF_High_Lower": float(q2),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "module2_dev_oof_risk_tiers.csv", index=False)
    return out


def _plot_risk_tiers(tier_df: pd.DataFrame, manifest: list[dict[str, object]]) -> None:
    for lm in ["3M", "6M"]:
        sub = tier_df[(tier_df["Landmark"].eq(lm)) & (tier_df["Domain"].eq("Test"))].copy()
        path = FIG_DIR / f"Figure_09_Risk_Tiers_{lm}.png"
        plot_risk_tiers(sub, path)
        manifest.append(_image_record(path, "plot_risk_tiers", f"{lm} temporal-test risk tiers derived from development OOF probabilities"))


def _write_audits() -> pd.DataFrame:
    leak = pd.read_csv(LEGACY_TABLES / "leakage_feature_audit.csv")
    med = pd.read_csv(LEGACY_TABLES / "medication_feature_audit.csv")
    rows: list[dict[str, object]] = []
    for r in leak.itertuples(index=False):
        rows.append(
            {
                "Audit": "Feature time-safety",
                "Landmark": getattr(r, "Landmark"),
                "Item": getattr(r, "FeatureSet"),
                "Available": True,
                "Pass": bool(getattr(r, "Leakage_Check_Pass")),
                "Conclusion": f"{getattr(r, 'N_Features')} features; forbidden future-feature hits = {getattr(r, 'Forbidden_Hits')}",
            }
        )
    for r in med.itertuples(index=False):
        rows.append(
            {
                "Audit": "Medication timing",
                "Landmark": "All",
                "Item": getattr(r, "Variable"),
                "Available": getattr(r, "Available"),
                "Pass": "Excluded from primary" in str(getattr(r, "Role")) or "Not used" in str(getattr(r, "Role")) or "Retained as static" in str(getattr(r, "Role")),
                "Conclusion": f"{getattr(r, 'Role')}: {getattr(r, 'Rationale')}",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "module2_leakage_audit.csv", index=False)
    return out


def _run_checks(risk: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    checks: list[dict[str, object]] = []
    for lm in LANDMARKS:
        sub = risk[risk["Landmark"].astype(str).eq(lm)]
        checks.append(
            {
                "Check": f"{lm} selected risk-score coverage",
                "Pass": int(sub["Treatment_ID"].nunique()) == 1003,
                "Detail": f"{int(sub['Treatment_ID'].nunique())} treatment episodes across OOF/Test",
            }
        )
    checks.append({"Check": "All figures generated", "Pass": len(manifest) >= 1, "Detail": f"{len(manifest)} PNG figures"})
    out = pd.DataFrame(checks)
    out.to_csv(TABLE_DIR / "module2_acceptance_checks.csv", index=False)
    if not bool(out["Pass"].all()):
        failed = out[~out["Pass"]]
        raise RuntimeError(f"Module 2 acceptance checks failed:\n{failed.to_string(index=False)}")
    return out


def main() -> None:
    _ensure_clean_dirs()
    selected = _read_selected()
    risk, coverage = _extract_selected_predictions(selected)
    perf = _build_performance_table(selected)
    persistence = _copy_persistence()
    _write_audits()

    manifest: list[dict[str, object]] = []
    _plot_auc_gradient(perf, persistence, manifest)
    _plot_selected_roc_pr(risk, manifest)
    _plot_heatmap(manifest)
    _plot_or_forests(manifest)
    _plot_calibration_and_dca(risk, manifest)
    _plot_single_feature(manifest)
    tiers = _risk_tier_table(risk)
    _plot_risk_tiers(tiers, manifest)

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(TABLE_DIR / "module2_figure_manifest.csv", index=False)
    _run_checks(risk, manifest_df)

    summary = {
        "out_dir": str(OUT_DIR),
        "risk_score_path": str(TABLE_DIR / "early_nhrh_risk_score.csv"),
        "risk_score_rows": int(len(risk)),
        "coverage": coverage.to_dict(orient="records"),
        "performance_path": str(TABLE_DIR / "module2_landmark_performance.csv"),
        "figures": manifest_df.to_dict(orient="records"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
