# M2 · RAI 治疗后早期复发 EBM 玻璃盒（corrected+LOCF · 1/3/6/12M）

> 本分支精简为**纯 M2**：RAI 治疗后早期固定地标的长期复发判定。
> 围绕 **1003 RAI 治疗人次**（development 802 / temporal test 201，分析单位为人次，重复治疗按独立 episode 处理），在 **1/3/6/12M** 四个固定地标各训练一个 **EBM 玻璃盒**（逐地标可解释 GAM），预测 **24M 复发（NHRH）**。所有产物（图、汇总表、报告）自包含可独立阅读，不含原始数据集。

---

## 一小时关键变化（项目总结）

### 背景

M2 回答的是治疗后早期的临床问题——「3 / 6 个月时还要不要担心 24 个月以后」。范式为**固定终点 24M NHRH** 上的多地标 landmark 更新：在 **1M / 3M / 6M / 12M** 四个固定地标各拟合一个独立的 EBM 玻璃盒，全程 time-safe（地标 L 只用 ≤L 的特征），并在每个地标并列 **naive（患病率）** 与 **persistence（time-safe 当期 TSH 逻辑回归）** 两条朴素基线以量化学习模型的增量。0M（治疗前）不由 M2 承担，归 M1 模块。分析口径为 **N=1003 人次**，CV/bootstrap 按 episode 处理，不报告 unique-patient 数。

### 最大发现：两处退化读列 bug

复盘主线数据 loader 时发现：它读取的是几乎全空的**宽列**——`FT4_6M` 在 1003 人次里只有 **13** 个真值、`FT4_12M` **全为 0**、`Eval` one-hot 列**全零**。当期甲功的真值其实一直躺在 long 表的 `*_Current` 列（按 `Current_Time=="{L}M"` 取行）里，从未被正确读出。换言之，旧主线的「当期化验」特征长期处于退化状态，旧 6M 的 0.817 主要是靠**激素变化速度 + 静态基线**撑起来的，而非当期水平本身。

### 口径修正 + 叙事反转

切换到 corrected 真值（`*_Current` 列）后，6M EBM 从 **0.817 → 0.878**、并新增 **12M = 0.908**。更关键的是叙事反转：**当期甲功水平在后期 6M / 12M 占主导**，而旧报告里「激素动量（速度）主导 6M」的结论，其实是 current 列退化后的人为产物——主 paper 与图解教程已用删除线就此更正。动量并未消失，而是以「**水平 × 速度**」12M 交互项的形式提供增量（momentum-beyond-inertia：在「现在是什么样」之上量化「正在往哪走」）。

### 其余成果

- **LOCF（结转患者自身轨迹）是最优插值**：在缺失高发的 12M 上把同一 episode 上一次真值结转，优于 missForest / KNN / median，也优于「只告诉模型这里缺了」（informative-missingness 对照≈median）——增益来自结转的**数值/轨迹本身**。
- **12M 入主 paper 且不降反升**：旧 median 口径下 12M(0.803) < 6M(0.855)（「越晚越难」），LOCF 口径下 **12M(0.908) > 6M(0.878)**，全程单调、多 seed 稳定。
- **跨地标错例：弥散 → 特征天花板**。错误不再集中于某个可命名子群，而是弥散分布——可命名的系统性错误只剩一类 **FN-silent「看似正常却复发」**；轨迹分析（temporal）显示持续对 50.2% / 转对 23.4%（信号积累）/ 转错 5.5% / 持续错 7.5%，首地标即错的 70 人里 **67% 靠后续信号积累转对、21% 顽固到底**，印证「半量特征即饱和」的天花板。
- **交互项可大幅 prune**：自动发现的两两交互项判别增益在 4/4 地标的 95% CI 均跨 0，且 GA2M（含交互）→ GAM（纯加性）的退化为非显著——交互更多是可解释性装饰而非判别来源，可透明保留供审计。配套选择性预测：弃权约 50% 后准确率升到 0.78 / NPV 0.84。
- **M1 复查无同类读列 bug**，数据干净，结论不受影响。

### 产物

loader 扩展（1/3/6/12M + corrected 真值）+ EBM-OOF 评估 + 插值对比实验 + **atlas_locf（47 图）** + 主 paper 重做（12M 入主图文版）+ prune 诊断 / 跨地标错例与变迁 / 原理教程 / 图解讲解 报告 + 可玩交互版（plotly）；全程落盘于工作日志 [`docs/worklog_m2_ebm_xland.md`](docs/worklog_m2_ebm_xland.md)。

---

### 图 1 · 重要性时间迁移热图（叙事反转核心）

各地标 EBM 分组重要性随时间迁移：解剖负荷 → 当期甲功水平在后期占主导，动量退居交互增量。

![重要性时间迁移热图](results/module2_v2_vertical/m2v2_ebm_full_locf/figures/F41_GroupImportance_heatmap.png)

### 图 2 · EBM vs persistence / naive（学习增量）

逐地标 EBM 时间外 AUC 与 persistence / naive 基线对比；EBM−persistence 的 ΔAUC 在四地标全 95% CI 排除 0（1M +0.142 / 3M +0.108 / 6M +0.123 / 12M +0.155，配对 episode bootstrap），12M 增量最大。

![EBM vs persistence](results/module2_v2_vertical/m2v2_ebm_full_locf/figures/F38_EBM_vs_persistence.png)

### 图 3 · 12M「FT3,FT4 综合水平」阈值形状

12M 地标当期甲功综合水平的 1D 形状函数：过零处急升，局部 OR ≈ 30——后期当期水平对终点的强直接预测性。

![12M 综合水平阈值形状](results/module2_v2_vertical/m2v2_ebm_full_locf/figures/F32_Shape_12M_r1.png)

### 表 1 · 插值方法对比（temporal AUC，按地标）

> 数据取自 `docs/worklog_m2_ebm_xland.md`。同一正交轴 EBM × 每地标 OOF + temporal read-out + 配对 episode-cluster bootstrap（n=1000）。

| 插值方法 | 1M | 3M | 6M | 12M |
|:---|:---:|:---:|:---:|:---:|
| median（基线） | 0.704 | 0.799 | 0.855 | 0.803 |
| **LOCF（激素延续，推荐）** | 0.694 | 0.791 | **0.878** | **0.908** |
| missForest | 0.696 | 0.796 | 0.856 | 0.789 |
| KNN | 0.699 | 0.802 | 0.835 | 0.781 |
| ebm_native（NaN 原生 bin） | 0.707 | 0.802 | 0.853 | 0.801 |
| median+missing-indicator | 0.707 | 0.797 | 0.855 | 0.800 |

**ΔAUC vs median（配对 bootstrap）**：**LOCF @12M = +0.105，CI [0.059, 0.154] → 排除 0**（全矩阵唯一排除 0 的提升）；其余方法在任一地标 CI 均跨 0。低缺失地标（1/3M）各法在 bootstrap 噪声内不可区分，LOCF 的微降不可分辨；增益集中在缺失高发的 6M/12M。

### 表 2 · 退化 bug 的真值缺失对照

> 主线 loader 读的「宽列」几乎全空；真值实际在 long 表 `*_Current` 列，三激素人次充足。

| 列来源 | FT4_6M | FT4_12M | 当期真值人次（`*_Current`） |
|:---|:---:|:---:|:---|
| **宽列（主线误读）** | 13 / 1003 | 0 / 1003 | — |
| **`*_Current`（corrected 真值）** | — | — | 3M 820 / 6M 746 / 12M 605 |

---

## M2 报告导航

| 报告 | 类型 | 入口 |
|:---|:---|:---|
| **EBM 玻璃盒论文 · 图文版（47 图 + 逐图评论，★ 推荐入口）** | 主交付 | [Module2v2_EBM_paper.md](results/module2_v2_vertical/Module2v2_EBM_paper.md) |
| EBM 交互版（plotly，可玩 drill-down / 2D 交互查表） | 交互 HTML | [Module2v2_EBM_interactive.html](results/module2_v2_vertical/Module2v2_EBM_interactive.html) |
| EBM 原理图解教程 / 讲解（人话→数学→1D→2D→正交辨析） | 入门讲解 | [Module2v2_EBM_讲解.md](results/module2_v2_vertical/Module2v2_EBM_讲解.md) |
| EBM 交互项诊断与 prune | 诊断 | [Module2v2_EBM交互项诊断与prune.md](results/module2_v2_vertical/Module2v2_EBM交互项诊断与prune.md) |
| 跨地标错例与变迁（弥散 → 特征天花板 / FN-silent） | 错例分析 | [Module2v2_跨地标错例与变迁.md](results/module2_v2_vertical/Module2v2_跨地标错例与变迁.md) |
| 扩展四项发现 | 续作 | [Module2v2_扩展四项发现.md](results/module2_v2_vertical/Module2v2_扩展四项发现.md) |
| probe 可解释性与诊断（偏差-方差 / 学习曲线 / 选择性预测） | 诊断 | [Module2v2_probe_可解释性与诊断.md](results/module2_v2_vertical/Module2v2_probe_可解释性与诊断.md) |
| 12 方法对比（L2 / Elastic-net / GEE / RF / HGB / GRU…） | 横向对比 | [Module2v2_compare.md](results/module2_v2_vertical/Module2v2_compare.md) |
| B4 时间感知 GRU / GRU-ODE | 架构探索 | [Module2v2_B4_时间感知GRU.md](results/module2_v2_vertical/Module2v2_B4_时间感知GRU.md) |
| 三轨综合 + 论文推荐 | 综合 | [Module2v2_三轨综合与论文推荐.md](results/module2_v2_synthesis/Module2v2_三轨综合与论文推荐.md) |
| M2 · v1 fallback（per-landmark 4-LR + Eval state，早期固定地标更新） | v1 fallback | [Module2_早期固定地标长期风险更新.md](results/module2_early_landmark_updating/Module2_早期固定地标长期风险更新.md) |

结果目录：[`results/module2_v2_vertical/`](results/module2_v2_vertical/)（含 `m2v2_ebm_full_locf/` 47 图主产物）· [`results/module2_v2_synthesis/`](results/module2_v2_synthesis/) · [`results/module2_early_landmark_updating/`](results/module2_early_landmark_updating/) · [`results/module2_v2_base/`](results/module2_v2_base/) · [`results/module2_v2_horizontal/`](results/module2_v2_horizontal/)

---

## 更新日志 / Roadmap

> 每次*有洞见*的 merge/push 在此留一行（日期 · 一句洞见 · PR/commit）；倒序，新条目加最上面。

- **2026-05-31** · **统一 corrected 真值口径 + LOCF（叙事反转）** — 发现主线「当期化验」读错列（`FT4_6M` 仅 13/1003 真值、`FT4_12M` 全 0；真值其实在 long 表 `*_Current` 列）：切真值后 6M EBM **0.817→0.878**、新增 **12M 0.908**；LOCF 为最优插值（12M ΔAUC +0.105 CI 排除 0）；12M 入主图文版 paper（不降反升）；旧「激素动量主导 6M」被确认为 current 退化产物（已删除线更正），动量改以「水平×速度」12M 交互增量呈现。loader 扩到 1/3/6/12，三激素真值人次 3M 820 / 6M 746 / 12M 605。
- **2026-05-30** · 12M EBM 扩展 + 患者级分解（waterfall / 反事实）+ GREAT 对标。
- **2026-05-30** · EBM 论文 §3.6：自动发现的两两交互项「医学解读 + 可信度分级」（可信项 vs 弱交互透明保留供审计）。
- **2026-05-30** · M2 不做 0M 性能分析、从 1M 起（0M 治疗前归 M1 模块）；跨地标叙事统一为 1M/3M/6M/12M。
- **2026-05-30** · EBM 交互版升级 drill-down — 下拉选特征大图 / x 轴滑块缩放 / 交互项 2D 查表悬停看每格人次+事件率；新增 EBM 原理图解教程（人话→数学→1D 形状→2D 交互→正交/共线辨析）。
- **2026-05-29** · EBM 玻璃盒论文图文版 — 47 图内嵌 + 逐图评论 + §2.1 数学定位（GA2M / LR 局部 OR = exp(Δ)）；EBM vs 12 类 ML 方法对比（印证特征天花板）。
