#!/usr/bin/env python
"""M2 · B4 — assemble the cross-landmark error/transition report (Phase 4/5, script 5).

Reads the CSV/JSON artifacts dropped by the four xland scripts (tracking /
trajectory / errtypes / phase5), writes a self-contained markdown + HTML report
(CJK-safe, base64 images via md_to_safe_html) and a composite overview figure.

Run the four producers first (or pass --run to invoke them here). This script
does NOT re-fit any model; it only formats the saved tables.

口径 (everywhere): corrected truth (Current_Time) + LOCF impute, landmarks
1/3/6/12, EBM per-landmark OOF (dev) / temporal read-out, naive(prevalence) +
persistence(today's TSH) baselines, strictly time-safe. analysis unit = 人次
(N=1003). Temporal numbers are read-out only (never used to select/tune).

Outputs:
    results/module2_v2_vertical/Module2v2_跨地标错例与变迁.md
    results/module2_v2_vertical/Module2v2_跨地标错例与变迁.html
    figures/xland_Figure_00_Composite.png
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
import matplotlib.font_manager as _fm  # noqa: E402

for _fp in ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "/Library/Fonts/Arial Unicode.ttf"):
    if os.path.exists(_fp):
        _fm.fontManager.addfont(_fp)
        matplotlib.rcParams["font.family"] = "Arial Unicode MS"
        break
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

OUTDIR = ROOT / "results" / "module2_v2_vertical"
GRU = OUTDIR / "b4_target_gru"
TBL = GRU / "tables"
FIG = GRU / "figures"
MD = OUTDIR / "Module2v2_跨地标错例与变迁.md"
HTML = OUTDIR / "Module2v2_跨地标错例与变迁.html"


def _csv(name):
    return pd.read_csv(TBL / name)


def _json(name):
    return json.loads((TBL / name).read_text())


def _df_md(df, **kw):
    """Pipe-delimited GitHub-style markdown table (no tabulate dependency).

    The downstream renderer (md_to_safe_html) enables Python-Markdown's `tables`
    extension, which parses exactly this pipe format.
    """
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for v in row.tolist():
            if isinstance(v, bool):
                cells.append("✓" if v else "✗")
            elif isinstance(v, float):
                cells.append(f"{v:.4g}")
            else:
                cells.append("" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _composite():
    """4-panel overview: tracking heatmap · abs-err curve · tier sankey · persist increment."""
    panels = [
        ("xland_Figure_01_TrackingHeatmap.png", "误差追踪热力图"),
        ("xland_Figure_02_AbsErrCurve.png", "|p−Y| 随地标"),
        ("xland_Figure_05_TierSankey.png", "风险档迁移桑基"),
        ("xland_Figure_06_PersistIncrement.png", "persistence 增量"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, (fn, _t) in zip(axes.ravel(), panels):
        p = FIG / fn
        if p.exists():
            ax.imshow(mpimg.imread(p))
        ax.axis("off")
    fig.suptitle("跨地标错例与变迁挖掘 — 概览(corrected+LOCF, EBM-OOF/temporal)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / "xland_Figure_00_Composite.png", dpi=150)
    plt.close(fig)


def build_markdown() -> str:
    auc = _csv("xland_landmark_auc.csv")
    pats = _csv("xland_trajectory_patterns.csv")
    stub = _csv("xland_stubborn_vs_accumulation.csv")
    curve = _csv("xland_abs_err_curve.csv")
    churn = _csv("xland_error_churn_jaccard.csv")
    conf = _csv("xland_confusion.csv")
    subt = _csv("xland_error_subtypes.csv")
    drift = _csv("xland_subtype_drift.csv")
    vd = _json("xland_characterizable_verdict.json")
    inc = _csv("xland_persist_increment.csv")
    absn = _csv("xland_per_landmark_abstain.csv")
    cchurn = _csv("xland_consecutive_churn.csv")
    track_sum = _json("xland_tracking_summary.json")

    n_ep = track_sum["n_episodes"]
    dev_ep = track_sum["n_dev_episodes"]
    tmp_ep = track_sum["n_temporal_episodes"]

    pats_t = pats[pats["split"] == "Temporal"]
    stub_t = stub[stub["split"] == "Temporal"].iloc[0]
    conf_t = conf[conf["split"] == "Temporal"]
    km = vd["kmeans"]

    L = []
    A = L.append
    A("# Module 2 · v2 · B4 — 跨地标错例与变迁挖掘")
    A("")
    A("> EBM 玻璃盒预测 24M 复发(`Y_24M_NHRH`),沿 **1 / 3 / 6 / 12M** 四地标做"
      "**同人跨地标**错例追踪与变迁分析(Phase 4 task1 + Phase 5 task5)。")
    A("")
    A("**口径(全程一致)**:corrected 真值(`Current_Time` 行)+ LOCF 插值,"
      "EBM 逐地标 **dev OOF**(5-fold StratifiedGroupKFold,seed=13)/ **temporal read-out**;"
      f"每地标并列 **naive**(dev 患病率)+ **persistence**(time-safe「今日 TSH」单特征 L2-LR)基线。"
      "全程 time-safe(地标 L 仅用 ≤L 信息;`assert_no_future_feature` 把关)。"
      f"**分析单元 = 治疗-疗程(人次),N = {n_ep} 人次**(dev {dev_ep} / temporal {tmp_ep};"
      "重复患者按独立疗程计,不做患者分组)。**Temporal 数字仅作读出,绝不用于选择/调参。**")
    A("")
    A("> 注:6M/12M 的临床 `Eval_{L}M` one-hot 在长表中退化(全零,与宽列同一 bug 类),"
      "故 6M/12M 的功能态由 **corrected 激素水平推导**(TSH 主导;与 1M/3M 真实 Eval 比对约 0.76–0.80 一致);"
      "1M/3M 直接用真实临床 Eval。")
    A("")
    A("![概览](figures/xland_Figure_00_Composite.png)")
    A("")
    A("---")
    A("")

    # ---- 0. landmark discrimination ----
    A("## 0. 逐地标判别基线(EBM vs persistence vs naive)")
    A("")
    A(_df_md(auc))
    A("")
    A("EBM 判别随地标单调走强(temporal **0.694 → 0.791 → 0.878 → 0.908**),"
      "而 persistence(只看今日 TSH)停在 **0.55 → 0.68 → 0.75 → 0.75**,naive 恒为 0.50。"
      "12M 在 corrected+LOCF 口径下是最强地标(与旧 median 口径「12M<6M」相反,见主 paper §讨论);"
      "下面所有错例/变迁分析都建立在这条判别曲线之上。")
    A("")

    # ---- 1. tracking wide table ----
    A("## 1. 追踪宽表(per-episode × per-landmark)")
    A("")
    A("产物 `tables/xland_tracking_wide.csv` —— 每行一个人次,列含"
      "`p@L / correct@L / tier@L / p_naive@L / p_persist@L / state@L`(L∈1/3/6/12);"
      "长表 `xland_tracking_long.csv`。下图按轨迹误差排序的 |p−Y| 追踪热力图(temporal):")
    A("")
    A("![追踪热力图](figures/xland_Figure_01_TrackingHeatmap.png)")
    A("")
    A("**读图**:顶部深红带 = 跨地标持续高误差的**顽固难例**;中下区随地标左高右低渐隐 = "
      "**信号积累型**(早地标错、晚地标随化验累积转对);底部蓝带 = 一贯易判的低风险人次。"
      "误差结构沿地标系统性收缩,但顶部存在一条不随时间消解的难例带。")
    A("")

    # ---- 2. trajectory patterns ----
    A("## 2. 同人对错轨迹模式")
    A("")
    A(_df_md(pats_t.drop(columns=["split"]).rename(columns={"pattern": "轨迹模式", "n_episodes": "人次", "pct": "占比%"})))
    A("")
    A(f"temporal 集(n={tmp_ep})中:**持续对 {pats_t.iloc[0]['pct']}%**、"
      f"**转对(wrong→right){pats_t[pats_t.pattern=='turned-correct'].iloc[0]['pct']}%**(信号积累的直接证据)、"
      f"mixed {pats_t[pats_t.pattern=='mixed'].iloc[0]['pct']}%、"
      f"转错 {pats_t[pats_t.pattern=='turned-wrong'].iloc[0]['pct']}%、"
      f"**持续错 {pats_t[pats_t.pattern=='persistent-wrong'].iloc[0]['pct']}%**(顽固难例)。"
      "「转对」远多于「转错」,说明地标推进总体是把人次**从错救对**,而非引入新的判错。")
    A("")
    A("### 顽固难例 vs 信号积累型(首地标即错者的去向)")
    A("")
    A(_df_md(stub.drop(columns=["split"]).assign(split=stub["split"])[["split", "n_first_landmark_wrong",
            "fixed_by_signal_accumulation", "stubborn_wrong_throughout", "partial_unstable",
            "pct_fixed", "pct_stubborn"]]))
    A("")
    A(f"temporal 集首地标即错的 {int(stub_t['n_first_landmark_wrong'])} 人次里,"
      f"**{stub_t['pct_fixed']}% 靠信号积累在后续地标转对**(`fixed_by_signal_accumulation`),"
      f"仅 **{stub_t['pct_stubborn']}% 自始至终错**(`stubborn_wrong_throughout`,即特征天花板候选)。"
      "这是「等一等、攒化验」临床策略价值的量化:多数早期误判会被随访信息纠正,"
      "但约 1/5 的难例无论等到 12M 都救不回来。")
    A("")

    # ---- 3. |p-Y| curve ----
    A("## 3. 预测误差 |p−Y| 随地标(EBM vs persistence vs naive)")
    A("")
    A(_df_md(curve.drop(columns=["_L"])))
    A("")
    A("![误差曲线](figures/xland_Figure_02_AbsErrCurve.png)")
    A("")
    A("EBM 平均 |p−Y| 随地标**单调下降**(temporal 0.404 → 0.343 → 0.237 → 0.219),"
      "persistence 几乎**水平**(~0.44,看今日 TSH 无法随时间变准),naive 恒在患病率附近。"
      "两线的纵向间距 = 学习模型从「累积轨迹/数值」中榨取、而 persistence 拿不到的增量;"
      "该间距在 6M/12M 拉到最大。")
    A("")

    # ---- 4. error churn ----
    A("## 4. 错例 churn(各地标错例集重叠 Jaccard)")
    A("")
    A(_df_md(churn[churn["split"] == "Temporal"].drop(columns=["split"])))
    A("")
    A("相邻/跨地标错例集 Jaccard 仅 **0.27–0.47**(temporal),即各地标错的**不是同一批人**——"
      "错例集大幅轮换。相邻对来看:")
    A("")
    A(_df_md(cchurn.drop(columns=["split"])))
    A("")
    A(f"如 3M→6M:3M 错的 58 人中 33 人在 6M 转对(resolved)、仅 21 人持续错(persisted)、13 人新错(new)。"
      "错例轮换 + 持续错占比低,与「轨迹模式」「顽固 vs 积累」三处证据互相印证:"
      "**残余错误以流动为主、顽固为辅**。")
    A("")

    # ---- 5. confusion + subtypes + verdict ----
    A("## 5. 逐地标混淆矩阵 + 错例亚型 + 【能否概括】诚实判定")
    A("")
    A("### 5.1 混淆矩阵(EBM,Youden@OOF;temporal read-out)")
    A("")
    A(_df_md(conf_t.drop(columns=["split"])))
    A("")
    A("Sens/Spec 随地标同步抬升(6M/12M 双双 ≥0.77/0.84);FN(漏报复发)与 FP(误报)"
      "数量在后期地标都收敛到 ~16–19。下面对置信错例池(每类 top-40)做亚型剖析。")
    A("")
    A("### 5.2 错例亚型(规则优先 + KMeans 校验)")
    A("")
    A("规则:`FN-persistent-hyper`(漏报、L 处仍 Hyper,信号在但模型欠权重)、"
      "`FN-silent`(漏报、L 处已 euthyroid/Normal,**看似正常却复发**的真难例)、"
      "`FP-big-goiter`(误报、基线甲状腺大,解剖驱动过判)、`FP-early-hyper`(误报、L 处仍 Hyper 的慢正常化非复发者)等。")
    A("")
    A(_df_md(drift.rename(columns={"subtype": "亚型"})))
    A("")
    A("![亚型漂移](figures/xland_Figure_04_SubtypeDrift.png)")
    A("")
    A(f"**`FN-silent` 在四地标稳定占 FN/FP 池 ~0.23–0.31**(1M 0.31 / 3M 0.23 / 6M 0.31 / 12M 0.29),"
      "是最稳的一条错例带 —— 即「当期甲功正常、却仍复发」的人次;`FP-big-goiter` 早地标高(1M 0.40)、"
      "随地标回落(12M 0.25),提示解剖性过判主要发生在早期信号弱时。")
    A("")
    A("### 5.3 【诚实判定:错例能否概括?】预注册三门槛")
    A("")
    g = vd["gates"]
    gate_rows = pd.DataFrame([
        {"门槛": "G1 KMeans silhouette(误例成簇)", "阈值": f"≥{g['G1_silhouette']['threshold']}",
         "实测": g["G1_silhouette"]["value"], "通过": g["G1_silhouette"]["pass"]},
        {"门槛": "G2 规则↔聚类 ARI(规则=真结构)", "阈值": f"≥{g['G2_rule_cluster_ARI']['threshold']}",
         "实测": g["G2_rule_cluster_ARI"]["value"], "通过": g["G2_rule_cluster_ARI"]["pass"]},
        {"门槛": f"G3 某亚型 ≥{int(g['G3_cross_landmark_share']['min_share']*100)}% 占比×≥"
                 f"{g['G3_cross_landmark_share']['min_landmarks']}地标", "阈值": "any subtype",
         "实测": "FN-silent/FP-big-goiter 达标", "通过": g["G3_cross_landmark_share"]["pass"]},
    ])
    A(_df_md(gate_rows))
    A("")
    A(f"KMeans(k={km['k']})在全地标置信错例池(n={km['n_pool']})上:"
      f"silhouette=**{km['silhouette']}**、rule↔cluster ARI=**{km['ari']}**、簇大小 {km['cluster_sizes']}"
      "(一个巨簇 + 两个小簇,无清晰分层)。")
    A("")
    A(f"> **判定:{vd['verdict'].upper()}(三门槛未全过)。**")
    A(">")
    A("> " + vd["statement"])
    A("")
    A("**解读**:G3 通过(`FN-silent` 确是一条可命名、跨四地标稳定的临床带),"
      "但 G1(0.247<0.25,差一点点)与 G2(0.04≪0.10)未过 —— 误例**整体不成可分簇、"
      "规则也无法复现无监督结构**。结论据实定为「**错例弥散 → 特征天花板**」:"
      "残余错误主要是**「看似正常却复发」的静默型 + 不可约噪声**的混合,"
      "而非某个可补一两个特征就能修复的离散子群。这是诚实的 negative result —— "
      "它同时给出**唯一可操作的抓手(FN-silent 静默复发,值得专门找早期生物标志物)**,"
      "并提醒**不要期待靠错例聚类一键提分**。与第 4 节错例轮换、第 2 节顽固仅占 ~1/5 一致:"
      "错误是流动且弥散的,不是一个固定难例簇。")
    A("")

    # ---- Phase 5 ----
    A("---")
    A("")
    A("## 6. Phase 5 高价值跨地标点子")
    A("")
    A("### ① 风险档迁移 3×3 + 桑基(temporal)")
    A("")
    A("![风险档桑基](figures/xland_Figure_05_TierSankey.png)")
    A("")
    A("风险档(Low<0.30≤Mid<0.60≤High,切在 EBM 概率上)随地标推进**向两极结晶**:"
      "Mid 档持续掏空(节点总数 66→54→20→24),人次稳定地流向 Low 或 High。"
      "如 6M→12M:Low 99/113 留 Low、High 55/68 留 High,Mid 仅 8 人留 Mid。"
      "**临床含义:越晚的地标,风险分层越干脆、灰区越小**,支持「在 6M/12M 落定分诊」。")
    A("")
    A("### ② persistence 增量随地标(EBM−persist ΔAUC + 配对 bootstrap CI)")
    A("")
    A(_df_md(inc.rename(columns={"excludes_0": "CI排除0"})))
    A("")
    A("![persistence增量](figures/xland_Figure_06_PersistIncrement.png)")
    A("")
    A("EBM 相对 persistence 的 ΔAUC 在**四个地标全部 95% CI 排除 0**"
      "(1M +0.142 [0.056, 0.224]、3M +0.108 [0.054, 0.164]、6M +0.123 [0.073, 0.176]、"
      "12M +0.155 [0.097, 0.210]),12M 增量最大。即:**「攒齐多模态轨迹的学习模型」"
      "在每个地标都稳定优于「只看今日 TSH」**,且这一优势在最晚地标不衰反增。")
    A("")
    A("### ③ per-landmark 最优弃权率(拆 pooled 的 risk-coverage)")
    A("")
    absn0 = absn[absn["abstain_pct"].isin([0, 30])]
    A(_df_md(absn0))
    A("")
    A("![逐地标弃权](figures/xland_Figure_07_PerLandmarkAbstain.png)")
    A("")
    A("把池化的选择性预测拆到每地标:**弃权(转交临床)的收益强烈依赖地标**。"
      "12M 弃权 30% → 准确率 0.83→0.90、NPV→0.91;6M 类似(0.81→0.88);"
      "而 1M 弃权 30% 仅 0.65→0.67 —— **早期弱信号地标靠弃权也救不回来**,"
      "提示分诊式弃权应在 6M 之后启用,1M 更宜直接走低阈值筛查。")
    A("")
    A("### ④ 错例 churn(相邻地标,见第 4 节)")
    A("")
    A("相邻地标错例 Jaccard 0.35–0.47,~40–57% 错例在下一地标被纠正,印证错误以流动为主。")
    A("")

    # ---- limits ----
    A("---")
    A("")
    A("## 7. 局限与口径声明")
    A("")
    A("- **Temporal 全程仅作读出**;阈值(Youden)、风险档切点、亚型门槛均在 dev OOF 上预注册/选定。")
    A("- 6M/12M 功能态为**推导态**(临床 Eval 退化),已注明;FN-silent 的「看似正常」据此判定,"
      "若有真实 6M/12M 甲功标注可进一步收紧。")
    A("- 亚型池取每类 top-40 置信错例(共识阈值),改变 K 会轻移占比但不改「弥散」定性"
      "(silhouette 0.247、ARI 0.04 离门槛仍有距离)。")
    A("- 不含 post-RAI 用药特征(适应证混杂);未报告任何 unique-患者计数(分析单元 = 人次)。")
    A("- 未做显著性断言;增量/弃权收益均以 95% bootstrap CI 是否排除 0 表述。")
    A("")
    A(f"产物清单:`tables/xland_*.csv|json`(19)、`figures/xland_Figure_0[0-7]_*.png`(8)。"
      f"复算入口:四个 `module2_v2_b4_xland_{{tracking,trajectory,errtypes,phase5}}.py` 脚本。")
    A("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="run the four producers first (full)")
    args = ap.parse_args()

    if args.run:
        for s in ("tracking", "trajectory", "errtypes", "phase5"):
            print(f"[run] {s} …", flush=True)
            subprocess.run(
                [sys.executable, str(Path(__file__).with_name(f"module2_v2_b4_xland_{s}.py"))],
                check=True, env={**os.environ, "PYTHONNOUSERSITE": "1",
                                 "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            )

    _composite()
    md = build_markdown()
    MD.write_text(md, encoding="utf-8")
    print(f"wrote {MD} ({len(md)} chars)", flush=True)

    # render self-contained HTML
    r = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("md_to_safe_html.py")),
         str(MD), str(HTML), "--resource-root", str(GRU),
         "--title", "Module2v2 跨地标错例与变迁"],
        check=True, capture_output=True, text=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    print(r.stdout.strip(), flush=True)
    print(f"Saved → {MD}  &  {HTML}", flush=True)


if __name__ == "__main__":
    main()
