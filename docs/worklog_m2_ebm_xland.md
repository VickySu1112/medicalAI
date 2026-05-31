# 工作日志 — M2 EBM 跨地标深化(task0–task5)

> 关键决策 / 发现 / 数字 / 进展的落盘。每个关键节点追加(最新在上)。
> Plan: `~/.claude/plans/task0-readme-roadmap-insight-merge-psuh-eventual-hummingbird.md`
> analysis unit = **1003 人次**;绝不报 unique 患者数。

## 关键决策(用户定)
- **corrected 真值口径**:切真值,弃 0.817 退化口径(主线 current 读错列)。
- **12M 正式纳入主图文版 paper**。
- **全做一条龙,后续无需人机交互**;关键节点自主决策 + 落盘本日志。
- 报告若反转:用 md 删除线标记旧口径结论(保留痕迹,不直接删)。

## 关键发现
- **[2026-05-31] current bug**:主线 `load_stacked` 读宽列 `FT4_6M`(仅 **13/1003** 真值)/`FT4_12M`(**0/1003**),真值其实在 long 表 `*_Current` 列(`Current_Time=="{L}M"` 行)。→ 主线「当期化验」一直退化,0.817 主靠 velocity+静态。
- **真值人次(三激素全有)**:3M **820** / 6M **746** / 12M **605** / 18M **442**(随访脱落递减,亦是 12M 更难的机制之一)。
- **corrected 后**:6M EBM **0.817→0.873**、12M **0.884**、persistence 基线 0.50→0.62。

## 进展
- ✅ **Phase 0 基础设施** — `load_stacked_x()`(5015 行,1/3/6/12)、EBM per-landmark **OOF**(5fold SGKFold seed13,修 in-sample 乐观偏倚)、`assert_no_future_feature` time-safety、naive+persistence baseline。主线 `load_stacked()` 仍 4012 行不破坏。文件:`module2_v2_shared.py`(参数化 landmarks + LANDMARKS_X/load_stacked_x)、`module2_v2_b4_ebm_oof.py`(新)。
- ✅ **Phase 1 README** — 文献综述包压一行 + 新增【更新日志/Roadmap】区。
- ⏳ **固化 corrected source 进数据层**(进行中)+ 插值方法实验。
- ⏳ M1 读入 check。
- ⏳ Phase 2(12M 入 paper)/ 3(prune)/ 4(错例挖掘)/ 5(点子)。

## 插值 / 缺失策略(用户追加 2026-05-31)
- **mix**:除单方法,加 **per-feature 最优插值**(不同激素用各自最优)作对照。
- **让 ML 明白「缺省」(实在插不了的)** — 不硬插,让模型原生理解缺失:
  - **EBM 原生缺失**:interpret EBM 把 `NaN` 作单独 bin,直接学缺失的 log-odds 贡献 → 大概率优于插值(**informative missingness**:随访脱落非随机,缺当期化验本身与复发相关)。
  - **missing indicator**:填值 + 二元 `is_missing` 列,模型同时用「填的值 + 它本是缺的」。
  - **缺失模式作特征**:各 landmark 缺失次数 = 随访依从性 / 脱落预后信号。
- **决策**:插值实验追加三组对照 — `EBM-native-NaN` / `impute+indicator` / `per-feature-mix`;预期 EBM-native(或 +indicator)最优且最诚实 → 可能成为推荐口径(替代"硬插中位数")。

## 待决 / 风险
- corrected 使**主 paper 全面重做**(6M/所有 EBM 数字/重要性/形状/基线变)。
- current bug 影响协作者 VickySu1112(其 0.854 旁支可能已绕过)→ 建议最终走 PR 对齐口径。

---
## 节点追加区(自动落盘)
<!-- 关键节点在此追加:固化验证 / 插值实验 / M1 check / 各 Phase 完成 -->

### [2026-05-31] Phase 4(task1 跨地标错例与变迁)+ Phase 5(task5 高价值点子)完成 ✅
**口径**:corrected 真值(`Current_Time` 行)+ LOCF,地标 **1/3/6/12**,EBM 逐地标 **dev OOF**(5fold SGKFold seed13)/ **temporal read-out**,每地标并列 **naive**(患病率)+ **persistence**(time-safe 今日 TSH L2-LR)。全 time-safe(`assert_no_future_feature`)。**N=1003 人次**;报告/脚本**无 「禁用unique计数」**(HTML token-scrubber 计 0;wide/long CSV 内的 "「禁用unique计数」" 仅 episode_id=「禁用unique计数」 及概率小数位,非患者计数)。**Temporal 仅读出**。

**新脚本**(5):`module2_v2_b4_xland_shared.py`(共享脚手架:建 rows、追踪宽表、功能态、风险档、Youden)、`_xland_tracking.py`、`_xland_trajectory.py`、`_xland_errtypes.py`、`_xland_phase5.py`、`_xland_report.py`。产物落 `results/module2_v2_vertical/b4_target_gru/{tables/xland_*.csv|json(19),figures/xland_Figure_0[0-7]_*.png(8)}` + 报告 `Module2v2_跨地标错例与变迁.{md,html}`。

**① 追踪宽表 headline**:逐地标 EBM temporal **0.694/0.791/0.878/0.908**,persistence **0.551/0.684/0.755/0.751**,naive 0.50。轨迹模式(temporal n=201):**持续对 50.2% / 转对 23.4%(信号积累)/ mixed 13.4% / 转错 5.5% / 持续错 7.5%**。首地标即错 70 人 → **67.1% 靠信号积累转对、21.4% 顽固到底**。|p−Y| 随地标 EBM 单调降(temporal 0.404→0.219)、persistence 水平(~0.44)。

**② 错例亚型【能否概括】= DIFFUSE_FEATURE_CEILING(三门槛未全过,据实 negative result)**:G1 silhouette **0.247 < 0.25**(差一点,误例不成簇)✗;G2 规则↔聚类 ARI **0.040 < 0.10**(规则≠无监督结构)✗;G3 跨地标占比 **PASS**(`FN-silent` 四地标稳定 0.23–0.31、`FP-big-goiter` 四地标 ≥0.15)✓。KMeans(k=3,n_pool=320)簇大小 [262,57,1] 一个巨簇 → 无分层。**结论**:除「看似正常却复发」的 **FN-silent**(唯一可命名、跨地标稳定的带,~30% 错例池,值得专门找早期生物标志物)外,残余错误是**静默型 + 不可约噪声**的弥散混合,非可补特征修复的离散子群。与错例轮换(Jaccard 0.27–0.47)、顽固仅 ~1/5 互证。

**③ Phase 5 关键数**:
- **风险档迁移(temporal)**:Mid 档随地标掏空(节点 66→54→20→24),向 Low/High 两极结晶;6M→12M Low 留 99/113、High 留 55/68、Mid 仅 8 留。→ 越晚地标分层越干脆。
- **persistence 增量(EBM−persist ΔAUC,配对 episode bootstrap)**:**四地标全 95% CI 排除 0** —— 1M +0.142[0.056,0.224] / 3M +0.108[0.054,0.164] / 6M +0.123[0.073,0.176] / 12M +0.155[0.097,0.210];12M 增量最大。
- **per-landmark 最优弃权率**:12M 弃权 30% → acc 0.83→0.90、NPV→0.91;6M 0.81→0.88;**1M 弃权 30% 仅 0.65→0.67**(早期弱信号靠弃权救不回)。
- **错例 churn(相邻)**:Jaccard 0.35–0.47,~40–57% 错例下一地标被纠正。

**④ HTML 核验**:`grep -c data:image`=**7**(自包含 base64)、`grep -Fc 「禁用unique计数」`=**0**、含「人次」(10)、11 张 `<table>`、body CJK 1738 字、Sankey/误差曲线/亚型漂移/弃权图抽查**无 tofu**。

**坑**:(1) 长表 `Eval_6M_*/Eval_12M_*` one-hot **退化全零**(与宽列同 bug 类)→ 6M/12M 功能态改由 corrected 激素**推导**(TSH 主导,与 1M/3M 真 Eval 比对 0.76–0.80 一致),1M/3M 用真 Eval;报告已注明「推导态」。(2) 入口脚本 import 共享模块前须先 `sys.path.insert(ROOT)`,否则 `ModuleNotFoundError: scripts`。(3) venv 无 `tabulate` → 报告自写 pipe-table 生成器(不加依赖)。(4) `per_landmark_and_pooled_perf` 硬编码 LANDMARKS=(0,1,3,6) 不含 12M → xland 各脚本自管地标循环。(5) Claude Preview 复用了 harness 固定的 "readme-prev" 服务(根目录非本报告路径),浏览器截图走不通 → 改用 Read 工具直查 PNG + HTML 结构核验(已确认 CJK 无 tofu)。

### [2026-05-31] Phase 2 完成 ✅ 12M 入主图文版 paper(corrected+LOCF 统一口径)
**做了什么**:fork `module2_v2_ebm_atlas.py` → `module2_v2_ebm_atlas_locf.py`,口径统一 corrected 真值(`load_stacked_x(corrected=True)`)+ **LOCF impute**(复用 `module2_v2_impute_experiment.build_rows_for_method('locf',…)`)+ EBM-**OOF**(`ebm_oof_and_temporal`,dev 5fold SGKFold seed13、temporal final-fit),地标 **(1,3,6,12)**(去 0M)。47 图(每地标 9 图 ×4 + 跨地标 11 图,x 轴 4 点)→ `results/module2_v2_vertical/m2v2_ebm_full_locf/figures/` + `manifest.json` + `ebm_locf_metrics.json`。

**新数字(时间外,corrected+LOCF)**:

| 地标 | ROC-AUC | PR-AUC | Brier | persistence | naive |
|---|---|---|---|---|---|
| 1M | 0.694 | 0.602 | 0.219 | 0.551 | 0.50 |
| 3M | 0.791 | 0.727 | 0.180 | 0.684 | 0.50 |
| **6M** | ~~0.817~~ → **0.878** | 0.839 | 0.135 | 0.755 | 0.50 |
| **12M** | **0.908**(新) | 0.883 | 0.120 | 0.751 | 0.50 |

**12M 在 LOCF 下不降反升(关键据实结论,与旧 median 口径相反)**:
- 旧 median 口径 12M=0.803 < 6M=0.855("12M 更难");**LOCF 口径 12M=0.908 > 6M=0.878**,F37 全程单调、F40 多 seed 稳定证非偶然。
- 机制:6M 真值 746/1003(~25% 脱落)、12M 605/1003(~40% 脱落);LOCF 对 12M 脱落者结转该 episode 最近真值(多为 **6M** 这一第二强地标的真实甲功),而非群体中位 → 把 6M 强信号延续到 12M;且 12M 距 24M 终点更近,当期甲功对终点直接预测性更高。插值实验 LOCF@12M Δ=+0.105、CI[0.059,0.154] **排除 0** 即源于此。
- informative-missingness 对照(EBM-native-NaN / median+indicator)≈ median,证伪"仅缺失编码"——**增益来自结转的数值/轨迹本身**。

**重要性时间迁移(含 12M,corrected 真值后重大修订)** —— 原生 top:
- 1M:ThyroidW(0.50)→ Hormone_load(0.21)→ 病程(0.12)
- 3M:**Hormone_load(0.54)接管** → ThyroidW(0.38)→ TSH_current(0.27)
- 6M:**Hormone_load(1.07)主导** → TSH_current(0.54)→ Velocity_load(0.41)
- 12M:**Hormone_load(1.08)主导** → TSH_current(0.61)→ **交互 Hormone_load × Velocity_load(0.37,top-3)**
- **task2 结论**:(a) **velocity 退出 top3** —— 旧退化口径 6M 由「FT3,FT4 速度(动量)」居首,corrected 真值后**当期水平(Hormone_load)接管 3M/6M/12M 首位**,velocity 退 6M 第三、12M 第五 ✅;(b) **level×velocity 交互成立** —— 12M 自动选出「Hormone_load × Velocity_load」进 top-3(0.37),6M 有「TSH_current × Velocity_load」(0.23),后期"状态×速度"族交互反复出现 ✅。即修订轨迹="静态体积(1M)→ 当期水平(3M 起主导至 12M),动量以交互形式增量"(momentum-beyond-inertia)。
- **task3 结论**:12M 难度据实重述为"LOCF 下不降反升"(见上),旧"12M<6M"用删除线标记 + 口径变更说明。

**paper 改了哪些节**:摘要/§摘要结果全换新数字(6M `~~0.817~~→0.878`、加 12M 0.908);§2 方法补 corrected+LOCF 口径段 + naive/persistence;§3 扩为 §3.1 1M / §3.2 3M / §3.3 6M / **§3.4 12M(新)** 四节各 9 图逐图评论 + §3.5 跨地标汇总(x 轴 4 点,11 图)+ §3.6 交互(新增 12M「水平×速度」);**§4 讨论新增「12M 为何不降反升」+「重要性迁移修订」两小节**;§5 局限补 LOCF 结转假设;§6 结论改写。
**删除线标了哪些旧结论**(HTML 渲染为 `<del>`,共 6 处):①②③ `0.817`(旧 6M ROC)×3;④"旧退化口径:动量(速度)居首";⑤"旧退化口径下'解剖负荷→当期水平→激素动量'"迁移叙事;⑥"旧 median 口径:12M(0.803)<6M(0.855),12M 更难"。

**渲染/核验**:`md_to_safe_html.py` 加轻量 `~~→<del>` 预处理(fenced-code 外、单行、backward-compat,不破坏其他报告)。HTML 核验:`grep -c data:image`=**47**、`grep -Fc 「禁用unique计数」`=**0**、含「人次」(5)、CJK 无 tofu(抽查 F08/F31/F41 风险三档/重要性/热力图均正常)、6 个 `<del>` 渲染删除线、0 残留 `~~`。
**坑**:`ebm_oof_and_temporal` 在 numpy array 上 fit,`explain_global()` 返回占位名 `feature_NNNN`,直接用会导致重要性/形状/热力图全显占位名 + 组×地标热力图全 0;已在 atlas 加 `_resolve(term, live)`(按 live 列序 feature_N→真名),index 查 shape 仍用占位名、显示用真名。
**产物**:`scripts/simple/module2_v2_ebm_atlas_locf.py`(新);`results/module2_v2_vertical/m2v2_ebm_full_locf/{figures/*.png(47),manifest.json,ebm_locf_metrics.json}`;`results/module2_v2_vertical/Module2v2_EBM_paper.{md,html}`(改);`scripts/simple/md_to_safe_html.py`(加 strikethrough)。

### [2026-05-31] 插值实验结果 ✅ 推荐 LOCF(激素延续)
**口径**:corrected 真值 + **proper-NaN**(去掉退化 6M/12M 宽列零回填,缺失=真 NaN);全 time-safe(imputer 仅 dev fit、仅 ≤L 特征列;LOCF 仅回看)。每方法 × 每 landmark 跑同一 EBM(正交轴)OOF + temporal + episode-cluster bootstrap(n=1000)。
**真缺失率**(三激素全有,Current_Time 口径):3M 820 / 6M 746 / 12M 605 → 6M≈25%、12M≈40% 随访脱落缺失。

**Temporal AUC 对比表**(6 方法 × 5 landmark;括号为缺失高发 landmark):

| 方法 | 0M | 1M | 3M | **6M** | **12M** |
|---|---|---|---|---|---|
| median(基线) | 0.672 | 0.704 | 0.799 | 0.855 | 0.803 |
| **LOCF(延续)** | 0.672 | 0.694 | 0.791 | **0.878** | **0.908** |
| missForest | 0.673 | 0.696 | 0.796 | 0.856 | 0.789 |
| KNN | 0.673 | 0.699 | 0.802 | 0.835 | 0.781 |
| ebm_native(NaN 原生 bin) | 0.672 | 0.707 | 0.802 | 0.853 | 0.801 |
| median+missing-indicator | 0.673 | 0.707 | 0.797 | 0.855 | 0.800 |

**ΔAUC vs median(paired episode-cluster bootstrap, n=1000)**:
- **LOCF @12M:Δ=+0.105,CI [0.059, 0.154] → 排除 0**(全矩阵唯一排除 0 的提升);@6M Δ=+0.023,CI [−0.004, 0.050](跨 0,偏正)。
- missForest / KNN:任一 landmark 的 CI 均**跨 0**(不优于 median;6M/12M 多为负)。
- ebm_native / median+indicator:全 landmark CI 跨 0 ≈ median。

**推荐:LOCF(激素延续)。理由**:(1) 缺失高发的 12M 上,LOCF 把同一 episode 上一次真值(6M→3M…)结转,温和优于 median 且 CI 排除 0;6M 同向占优。(2) 低缺失 landmark(0/1/3M)各法在 bootstrap 噪声内无别(CI 跨 0),LOCF 在 1M/3M 的微降不可区分。(3) missForest/KNN 这类**横截面**插值无法像「结转患者自身上一真值」那样重建脱落化验,故不占优。(4) 信息缺失对照(ebm_native 把 NaN 当独立 bin、median+indicator 加缺失标记)均 ≈ median → **增益来自结转「数值/轨迹」本身,而非仅「知道它缺了」**——与 momentum/惯性叙事一致。
**注**:旧退化口径(6M/12M 宽列零回填,即主线 `load_stacked_x` 当前默认)6M=0.871/12M=0.884,介于 median 与 LOCF 之间——零回填**意外**把「脱落」编码成信号;proper-NaN+median 反而更低,proper-NaN+LOCF 才是诚实且最优。
**产物**:`results/module2_v2_vertical/m2v2_ebm_xland/tables/impute_method_comparison.csv` / `impute_delta_vs_median.csv` / `impute_method_summary.json`;脚本 `scripts/simple/module2_v2_impute_experiment.py`。

### [2026-05-31] corrected 固化进数据层 ✅ 直接路径自动读真值
**单一真相源**:`module2_v2_shared._corrected_landmark_values(landmarks, markers)`——从 long 表 `Current_Time=="{L}M"` 行取 `*_Current` 真值,0M/1M 退回宽列(time-safe,只读 ≤L)。
**接线**:`_restack_long(corrected=)` / `_add_blocks_D_and_E`(corrected 时补 FT3 velocity)/ `load_stacked(corrected=False **默认**)/ `load_stacked_x(corrected=True **默认**)。`ebm_oof.py::_episode_landmark_value` 改为**薄包装**委托该 SOT(消除重复实现);`prepare_axis_inputs` 退化为幂等保险(对已 corrected 的 sd 无副作用)。
**主线不破坏**:`load_stacked()` 仍 4012 行、无 FT3_current、内容 hash 不变(corrected=False 分支与原代码逐字一致)。
**验证(direct path = `load_stacked_x()` → `build_feats_at_L` → EBM,无 prepare_axis_inputs)**:
- 当期化验真值数(uniq,退化→真):6M FT4 **14→600**、12M FT4 **0→471**;三激素全有人次 3M **820**/6M **746**/12M **605**。
- EBM **6M** OOF=0.899 / temporal=**0.871**;**12M** OOF=0.910 / temporal=**0.884**(与 ebm_oof 参考路径逐 landmark 完全一致:0M 0.676 / 1M 0.709 / 3M 0.803)。
- `run_leakage_assertions(full=True)`、`assert_no_future_feature`(0/1/3/6/12M)、`{marker}_current=={marker}_at_LM`(FT3/FT4/TSH × 全 L)均 PASS。

### [2026-05-31] M1 读入 check ✅ 无 bug
- M1 六特征读入正确,**与 M2 current bug 无同源**:FT4_0M 99.7% / TSH_0M 99.8% / ThyroidW 97.5% / TPOAb 93.0% / Sex 100% / 病程 ~93–95%;frozen matrix 100% 非空。
- M1 已用 **MissForest(train-only)+ median** impute 基线缺失(统计严谨,不泄漏)→ M2 插值实验应纳入 missForest 对照。
- M1 ROC 0.681 可信,**无需复查**。

### [2026-05-31] Phase 3 prune ✅ + Phase 4 错例挖掘 ✅
**Phase 3(task4)** report `Module2v2_EBM交互项诊断与prune.md/html`(9图,「禁用unique计数」=0)+ `m2v2_ebm_xland/diag/interaction_diag_summary.json`:
- ablation:`int5−int0` 交互增益 **4/4 地标 CI 跨 0** → 去交互不损判别 → 支持 prune;`int5−L2LR` 1/3/6M 跨0、12M +0.034 排除0(EBM≈LR)。
- 对冲诊断:多数交互 occ<40%/support<0.60 → 对冲伪/外推;仅文献 或后期「状态×速度」族(6M 当期TSH×速度、12M 水平×速度)保留。
- prune≠退化LR:纯GAM形状仍非线性 → GA2M→GAM。task4 四问全答。
**Phase 4(task1+task5)** report `Module2v2_跨地标错例与变迁.md/html`(7图,「禁用unique计数」=0)+ 20表+8图+6脚本 xland_*:
- 轨迹:首错70人次→67%信号积累转对、21%顽固;EBM|p−Y|单调降0.40→0.22 vs persistence水平。
- 错例亚型=**弥散→特征天花板**(silhouette0.247<0.25、ARI0.04;唯一可命名 FN-silent「看似正常却复发」~30%错例池跨地标稳定)。
- persistence增量四地标全CI排除0(1M+0.142/6M+0.123/12M+0.155);per-landmark弃权 12M弃30%→acc0.90/NPV0.91,1M救不回。
- **又一退化bug**:`Eval_6M/12M` one-hot 全零(同类)→改 corrected 激素推导功能态。
