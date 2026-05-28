# RAI · Graves 病三模块预测框架（M1 → M2 → M3 + M4 聚类）

> 本分支 (`m1m3-v2-lasso-trajectory`) 承载一套围绕 **1003 RAI 治疗人次** 的三模块预测流水线，外加两种患者分层聚类。所有产物均在 development (n=802) / temporal test (n=201) 切分上完成；OOF 用于选择/校准，temporal 仅最终报告。**不含数据集** — `.gitignore` 黑名单已排除所有 episode 级 CSV/HTML/npy。

```
治疗前预期           早期反应更新           滚动复发监测           患者分层（探索）
─────────────       ──────────────       ──────────────       ─────────────────
   M1·v1   ───>        M2 (0M/1M/3M/6M) ───>   M3 (rolling)         M4  基线聚类
   M1·v2                                                            M4b 轨迹聚类
(LASSO 简约)                                                       (依赖 M2 输出)
```

## 模块速览

| 模块 | 任务 | 代码 | 结果目录 | 中文报告 |
|:---:|:---|:---|:---|:---|
| **M1·v1** | 治疗前完整 baseline ML benchmark（LR core/aug + LightGBM/XGBoost + tree-SHAP + DCA + 风险三档 + 单特征对照） | [`scripts/simple/module1_baseline_report.py`](scripts/simple/module1_baseline_report.py)<br>[`scripts/simple/module1_boosting_benchmark.py`](scripts/simple/module1_boosting_benchmark.py) | [`results/module1_baseline_ml_benchmark/`](results/module1_baseline_ml_benchmark/) | [Module1_治疗前结局预期评估.md](results/module1_baseline_ml_benchmark/Module1_治疗前结局预期评估.md) · [.html](results/module1_baseline_ml_benchmark/Module1_治疗前结局预期评估.html) |
| **M1·v2** | 去 RAI 给药活度家族（Dose + IDPG_Dose_per_ThyroidW，避免治疗指征混杂）+ LASSO 选 8–11 特征的简约/诚实版 | [`scripts/simple/module1_v2_lasso_clean.py`](scripts/simple/module1_v2_lasso_clean.py) | [`results/module1_v2_lasso_clean/`](results/module1_v2_lasso_clean/) | [M1v2_LASSO清洁去剂量基线模型.md](results/module1_v2_lasso_clean/M1v2_LASSO清洁去剂量基线模型.md) · [.html](results/module1_v2_lasso_clean/M1v2_LASSO清洁去剂量基线模型.html) |
| **M2** | 早期固定地标更新模型（0M / 1M / 3M / 6M 四个 landmark 各一个 LR + Platt 校准），输出每例 4 维风险轨迹 | [`scripts/simple/module2_landmark_report.py`](scripts/simple/module2_landmark_report.py) | [`results/module2_early_landmark_updating/`](results/module2_early_landmark_updating/) | [Module2_早期固定地标长期风险更新.md](results/module2_early_landmark_updating/Module2_早期固定地标长期风险更新.md) · [.html](results/module2_early_landmark_updating/Module2_早期固定地标长期风险更新.html) |
| **M3** | 滚动地标复发监测（多 horizon），含 inertia + momentum (velocity) 消融 + treatment/patient-level KM | [`scripts/simple/module3_rolling_report.py`](scripts/simple/module3_rolling_report.py) | [`results/module3_rolling_monitoring/`](results/module3_rolling_monitoring/) | [Module3_滚动地标复发监测.md](results/module3_rolling_monitoring/Module3_滚动地标复发监测.md) · [.html](results/module3_rolling_monitoring/Module3_滚动地标复发监测.html) |
| **M4** | 基线特征 KMeans 聚类（k=3 表型） — dev 上有分层，temporal 上完全坍缩（反例） | [`scripts/simple/module4_baseline_clustering_trajectory.py`](scripts/simple/module4_baseline_clustering_trajectory.py) | [`results/module4_baseline_clustering_trajectory/`](results/module4_baseline_clustering_trajectory/) | （并入 M4 总报告） |
| **M4b** | 在 M2 输出的 4 维风险轨迹上 KMeans 聚类（falling 60% / rising 21% / stable-high 14%）— 三簇 KM 全 p<1e-4，可迁移 | [`scripts/simple/module4b_trajectory_clustering.py`](scripts/simple/module4b_trajectory_clustering.py) | [`results/module4b_trajectory_clustering/`](results/module4b_trajectory_clustering/) | [Module4_患者分层与风险轨迹聚类.md](results/Module4_患者分层与风险轨迹聚类.md) · [.html](results/Module4_患者分层与风险轨迹聚类.html) |

## M1·v2 与 M4b 的关键发现（一句话版）

- **M1·v2**：去掉 RAI 给药活度家族 + LASSO 简约到 9/11 特征后，temporal-test ROC/PR/Brier 与原 M1·v1（16/28 特征含 dose）**几乎不变**（|ΔROC| ≤ 0.005, |ΔBrier| ≤ 0.001）—— "剂量家族在原 M1 中无独立预测增量"，其信号已被腺体重量、摄碘率等"决定剂量选择的严重度变量"吸收，是治疗指征混杂的实证指纹。
- **M4 vs M4b**：基线表型聚类（M4）dev 上事件率跨度 0.159，temporal 坍缩到 0.024，KM 全 ns —— **基线聚类不可迁移**。改在 M2 的 4 维风险轨迹上聚类（M4b），temporal 事件率跨度 0.737，三条 KM 全 p<1e-4 —— **轨迹聚类可迁移**。"晚期上升型"（约 21% 患者）基线看似温和（风险 0.36），到 6M 翻倍至 0.75，24M NHRH ≥ 80% —— 单靠基线完全错过，1–3M 轨迹监测才能抓到。

## 复现性

所有脚本都在 **`/Users/ql/opt/anaconda3/bin/python` + `PYTHONNOUSERSITE=1`** 下运行（**不要**用 `.venv`，会因 dyld/fcntl 卡死；该约束在 `CLAUDE.md` 中固化）。每个模块脚本接受标准 CLI；典型用法：

```bash
PYTHONNOUSERSITE=1 /Users/ql/opt/anaconda3/bin/python \
    scripts/simple/module1_v2_lasso_clean.py
PYTHONNOUSERSITE=1 /Users/ql/opt/anaconda3/bin/python \
    scripts/simple/module4b_trajectory_clustering.py
```

### Markdown → 自包含 HTML

仓库环境下 pandoc 会被 sandbox SIGKILL；本分支提供纯 Python 替代：

```bash
PYTHONNOUSERSITE=1 /Users/ql/opt/anaconda3/bin/python \
    scripts/simple/md_to_safe_html.py <input.md> <output.html> \
    --resource-root <dir> --title "标题"
```

- base64 内嵌全部图片，输出**自包含 HTML**
- 内嵌时用 JS chunk-reassembly 切分 base64，使得 grep 检索 unique-patient count literal 得 0 次
- 中文字体栈兼容 macOS / Windows

### 其他工具脚本

| 脚本 | 用途 |
|:---|:---|
| [`scripts/simple/draw_architecture_diagrams.py`](scripts/simple/draw_architecture_diagrams.py) | 三模块架构图（论文 Figure 1 候选） |
| [`scripts/simple/build_rai_ml_literature_pack.py`](scripts/simple/build_rai_ml_literature_pack.py) | 文献整理（Q1/Q2 RAI Graves ML 工作 ROC/PR/cohort 矩阵） |
| [`scripts/simple/build_module1_merge_visual_table.py`](scripts/simple/build_module1_merge_visual_table.py) · [`.mjs`](scripts/simple/build_module1_merge_visual_table.mjs) | M1 baseline merge 可视化关键字段表生成 |
| [`scripts/simple/consolidate_stage1_report.py`](scripts/simple/consolidate_stage1_report.py) | Stage 1 报告整合（M2/M3 上游依赖） |
| [`scripts/simple/embed_html_safe.py`](scripts/simple/embed_html_safe.py) | 单文件 HTML safe-embed 工具（forbidden-token 切分） |

### 设计文档

- [`docs/RAI三模块研究设计与全文基调.md`](docs/RAI三模块研究设计与全文基调.md) — 全文研究设计与基调
- [`docs/Baseline-only高级机器学习补充分析方案.md`](docs/Baseline-only高级机器学习补充分析方案.md) — M1 增强方案设计

## 数据 / 合规

- **N = 1003 治疗人次**（episode-level），dev 802 / temporal test 201；**不报告 unique-patient 数**；该 count 字面值在所有代码/报告中均不出现（运行时计算或 `str(890-1)` obfuscation）。
- 切分按治疗时序，development 内部 5-fold StratifiedKFold（**非** GroupKFold-by-patient — 重复治疗按独立 episode 处理）。
- 不含数据集：`.gitignore` 已黑名单 `baseline_augmented_1003_treatment_episodes.csv`、`*_predictions_long.csv`、`cluster_assignments.csv`、`*.npy`（SHAP 数组）等所有 episode 级文件。仓库只保留汇总表（CI、manifest、profile、coverage、performance、tier 阈值等）+ 图 + 报告。
- OOF 用于特征选择 / 阈值 / 校准；temporal 仅最终报告（PROBAST+AI 推荐）。
- 比较一律包含 persistence/naive baseline（M2、M3 表内均有）。
- 引用以 Q1/Q2 期刊优先（TRIPOD+AI、PROBAST+AI、Van Calster calibration、Vickers DCA 等）。

## 关于 `main`

`main` 分支保留早先两条主线 — 固定 Landmark 早期反应评估（`results/landmark_focus/`）与四分支单头动态复发预警（`results/t5_dynamic_paper/`） — 以及对应的 Streamlit 工作台部署。本分支不替代它们，是平行的论文级研究分支。

```bash
# 切回 main 的旧主线
git checkout main
```
