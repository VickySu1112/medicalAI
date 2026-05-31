# Module 2 · v2 · B4 — 跨地标错例与变迁挖掘

> EBM 玻璃盒预测 24M 复发(`Y_24M_NHRH`),沿 **1 / 3 / 6 / 12M** 四地标做**同人跨地标**错例追踪与变迁分析(Phase 4 task1 + Phase 5 task5)。

**口径(全程一致)**:corrected 真值(`Current_Time` 行)+ LOCF 插值,EBM 逐地标 **dev OOF**(5-fold StratifiedGroupKFold,seed=13)/ **temporal read-out**;每地标并列 **naive**(dev 患病率)+ **persistence**(time-safe「今日 TSH」单特征 L2-LR)基线。全程 time-safe(地标 L 仅用 ≤L 信息;`assert_no_future_feature` 把关)。**分析单元 = 治疗-疗程(人次),N = 1003 人次**(dev 802 / temporal 201;重复患者按独立疗程计,不做患者分组)。**Temporal 数字仅作读出,绝不用于选择/调参。**

> 注:6M/12M 的临床 `Eval_{L}M` one-hot 在长表中退化(全零,与宽列同一 bug 类),故 6M/12M 的功能态由 **corrected 激素水平推导**(TSH 主导;与 1M/3M 真实 Eval 比对约 0.76–0.80 一致);1M/3M 直接用真实临床 Eval。

![概览](figures/xland_Figure_00_Composite.png)

---

## 0. 逐地标判别基线(EBM vs persistence vs naive)

| Landmark | N_dev | N_temporal | Prevalence_dev | EBM_OOF | EBM_Temporal | Persist_OOF | Persist_Temporal | Naive_OOF | Naive_Temporal | state_source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1M | 802 | 201 | 0.3641 | 0.7385 | 0.6939 | 0.5169 | 0.5509 | 0.5 | 0.5 | clinical Eval |
| 3M | 802 | 201 | 0.3641 | 0.8267 | 0.7914 | 0.6915 | 0.6837 | 0.5 | 0.5 | clinical Eval |
| 6M | 802 | 201 | 0.3641 | 0.902 | 0.878 | 0.7377 | 0.7545 | 0.5 | 0.5 | derived(corrected hormones) |
| 12M | 802 | 201 | 0.3641 | 0.9187 | 0.9075 | 0.7732 | 0.7512 | 0.5 | 0.5 | derived(corrected hormones) |

EBM 判别随地标单调走强(temporal **0.694 → 0.791 → 0.878 → 0.908**),而 persistence(只看今日 TSH)停在 **0.55 → 0.68 → 0.75 → 0.75**,naive 恒为 0.50。12M 在 corrected+LOCF 口径下是最强地标(与旧 median 口径「12M<6M」相反,见主 paper §讨论);下面所有错例/变迁分析都建立在这条判别曲线之上。

## 1. 追踪宽表(per-episode × per-landmark)

产物 `tables/xland_tracking_wide.csv` —— 每行一个人次,列含`p@L / correct@L / tier@L / p_naive@L / p_persist@L / state@L`(L∈1/3/6/12);长表 `xland_tracking_long.csv`。下图按轨迹误差排序的 |p−Y| 追踪热力图(temporal):

![追踪热力图](figures/xland_Figure_01_TrackingHeatmap.png)

**读图**:顶部深红带 = 跨地标持续高误差的**顽固难例**;中下区随地标左高右低渐隐 = **信号积累型**(早地标错、晚地标随化验累积转对);底部蓝带 = 一贯易判的低风险人次。误差结构沿地标系统性收缩,但顶部存在一条不随时间消解的难例带。

## 2. 同人对错轨迹模式

| 轨迹模式 | 人次 | 占比% |
| --- | --- | --- |
| persistent-correct | 101 | 50.2 |
| turned-correct | 47 | 23.4 |
| mixed | 27 | 13.4 |
| turned-wrong | 11 | 5.5 |
| persistent-wrong | 15 | 7.5 |

temporal 集(n=201)中:**持续对 50.2%**、**转对(wrong→right)23.4%**(信号积累的直接证据)、mixed 13.4%、转错 5.5%、**持续错 7.5%**(顽固难例)。「转对」远多于「转错」,说明地标推进总体是把人次**从错救对**,而非引入新的判错。

### 顽固难例 vs 信号积累型(首地标即错者的去向)

| split | n_first_landmark_wrong | fixed_by_signal_accumulation | stubborn_wrong_throughout | partial_unstable | pct_fixed | pct_stubborn |
| --- | --- | --- | --- | --- | --- | --- |
| Temporal | 70 | 47 | 15 | 8 | 67.1 | 21.4 |
| Development | 246 | 185 | 23 | 38 | 75.2 | 9.3 |

temporal 集首地标即错的 70 人次里,**67.1% 靠信号积累在后续地标转对**(`fixed_by_signal_accumulation`),仅 **21.4% 自始至终错**(`stubborn_wrong_throughout`,即特征天花板候选)。这是「等一等、攒化验」临床策略价值的量化:多数早期误判会被随访信息纠正,但约 1/5 的难例无论等到 12M 都救不回来。

## 3. 预测误差 |p−Y| 随地标(EBM vs persistence vs naive)

| split | landmark | n | EBM_mean_abs_err | Persist_mean_abs_err | Naive_mean_abs_err |
| --- | --- | --- | --- | --- | --- |
| Development | 1M | 802 | 0.38 | 0.4609 | 0.4631 |
| Development | 3M | 802 | 0.3119 | 0.4208 | 0.4631 |
| Development | 6M | 802 | 0.2229 | 0.4346 | 0.4631 |
| Development | 12M | 802 | 0.1955 | 0.4351 | 0.4631 |
| Temporal | 1M | 201 | 0.404 | 0.4619 | 0.475 |
| Temporal | 3M | 201 | 0.3433 | 0.4336 | 0.475 |
| Temporal | 6M | 201 | 0.2366 | 0.4355 | 0.475 |
| Temporal | 12M | 201 | 0.2185 | 0.4451 | 0.475 |

![误差曲线](figures/xland_Figure_02_AbsErrCurve.png)

EBM 平均 |p−Y| 随地标**单调下降**(temporal 0.404 → 0.343 → 0.237 → 0.219),persistence 几乎**水平**(~0.44,看今日 TSH 无法随时间变准),naive 恒在患病率附近。两线的纵向间距 = 学习模型从「累积轨迹/数值」中榨取、而 persistence 拿不到的增量;该间距在 6M/12M 拉到最大。

## 4. 错例 churn(各地标错例集重叠 Jaccard)

| landmark_a | landmark_b | n_err_a | n_err_b | n_overlap | n_union | jaccard |
| --- | --- | --- | --- | --- | --- | --- |
| 1M | 3M | 70 | 58 | 41 | 87 | 0.4713 |
| 1M | 6M | 70 | 38 | 23 | 85 | 0.2706 |
| 1M | 12M | 70 | 34 | 23 | 81 | 0.284 |
| 3M | 6M | 58 | 38 | 25 | 71 | 0.3521 |
| 3M | 12M | 58 | 34 | 25 | 67 | 0.3731 |
| 6M | 12M | 38 | 34 | 21 | 51 | 0.4118 |

相邻/跨地标错例集 Jaccard 仅 **0.27–0.47**(temporal),即各地标错的**不是同一批人**——错例集大幅轮换。相邻对来看:

| pair | n_err_a | n_err_b | n_persisted | n_resolved | n_new | jaccard |
| --- | --- | --- | --- | --- | --- | --- |
| 1M→3M | 70 | 58 | 41 | 29 | 17 | 0.4713 |
| 3M→6M | 58 | 38 | 25 | 33 | 13 | 0.3521 |
| 6M→12M | 38 | 34 | 21 | 17 | 13 | 0.4118 |

如 3M→6M:3M 错的 58 人中 33 人在 6M 转对(resolved)、仅 21 人持续错(persisted)、13 人新错(new)。错例轮换 + 持续错占比低,与「轨迹模式」「顽固 vs 积累」三处证据互相印证:**残余错误以流动为主、顽固为辅**。

## 5. 逐地标混淆矩阵 + 错例亚型 + 【能否概括】诚实判定

### 5.1 混淆矩阵(EBM,Youden@OOF;temporal read-out)

| Landmark | thr | TP | FP | TN | FN | Sens | Spec | PPV | NPV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1M | 0.355 | 52 | 40 | 79 | 30 | 0.6341 | 0.6639 | 0.5652 | 0.7248 |
| 3M | 0.335 | 62 | 38 | 81 | 20 | 0.7561 | 0.6807 | 0.62 | 0.802 |
| 6M | 0.365 | 63 | 19 | 100 | 19 | 0.7683 | 0.8403 | 0.7683 | 0.8403 |
| 12M | 0.34 | 64 | 16 | 103 | 18 | 0.7805 | 0.8655 | 0.8 | 0.8512 |

Sens/Spec 随地标同步抬升(6M/12M 双双 ≥0.77/0.84);FN(漏报复发)与 FP(误报)数量在后期地标都收敛到 ~16–19。下面对置信错例池(每类 top-40)做亚型剖析。

### 5.2 错例亚型(规则优先 + KMeans 校验)

规则:`FN-persistent-hyper`(漏报、L 处仍 Hyper,信号在但模型欠权重)、`FN-silent`(漏报、L 处已 euthyroid/Normal,**看似正常却复发**的真难例)、`FP-big-goiter`(误报、基线甲状腺大,解剖驱动过判)、`FP-early-hyper`(误报、L 处仍 Hyper 的慢正常化非复发者)等。

| 亚型 | 12M | 1M | 3M | 6M |
| --- | --- | --- | --- | --- |
| FN-other | 0.2125 | 0.025 | 0.2375 | 0.1625 |
| FN-persistent-hyper | 0 | 0.1625 | 0.0375 | 0.025 |
| FN-silent | 0.2875 | 0.3125 | 0.225 | 0.3125 |
| FP-big-goiter | 0.25 | 0.4 | 0.275 | 0.175 |
| FP-early-hyper | 0.0375 | 0.0875 | 0.175 | 0.0625 |
| FP-other | 0.2125 | 0.0125 | 0.05 | 0.2625 |

![亚型漂移](figures/xland_Figure_04_SubtypeDrift.png)

**`FN-silent` 在四地标稳定占 FN/FP 池 ~0.23–0.31**(1M 0.31 / 3M 0.23 / 6M 0.31 / 12M 0.29),是最稳的一条错例带 —— 即「当期甲功正常、却仍复发」的人次;`FP-big-goiter` 早地标高(1M 0.40)、随地标回落(12M 0.25),提示解剖性过判主要发生在早期信号弱时。

### 5.3 【诚实判定:错例能否概括?】预注册三门槛

| 门槛 | 阈值 | 实测 | 通过 |
| --- | --- | --- | --- |
| G1 KMeans silhouette(误例成簇) | ≥0.25 | 0.2466 | ✗ |
| G2 规则↔聚类 ARI(规则=真结构) | ≥0.1 | 0.0397 | ✗ |
| G3 某亚型 ≥15% 占比×≥3地标 | any subtype | FN-silent/FP-big-goiter 达标 | ✓ |

KMeans(k=3)在全地标置信错例池(n=320)上:silhouette=**0.2466**、rule↔cluster ARI=**0.0397**、簇大小 [262, 57, 1](一个巨簇 + 两个小簇,无清晰分层)。

> **判定:DIFFUSE_FEATURE_CEILING(三门槛未全过)。**
>
> 错例弥散 → 特征天花板:未过门槛(轮廓系数 0.2466 < 0.25(误例不成簇)；规则-聚类 ARI 0.0397 < 0.1(规则与无监督结构不一致))。误例不形成可分、跨地标稳定的临床亚型,残余错误更像现有特征集的内在上限(看似正常却复发的静默型 + 不可约噪声),而非某个可命名、可补特征修复的子群。(negative result,据实报告)

**解读**:G3 通过(`FN-silent` 确是一条可命名、跨四地标稳定的临床带),但 G1(0.247<0.25,差一点点)与 G2(0.04≪0.10)未过 —— 误例**整体不成可分簇、规则也无法复现无监督结构**。结论据实定为「**错例弥散 → 特征天花板**」:残余错误主要是**「看似正常却复发」的静默型 + 不可约噪声**的混合,而非某个可补一两个特征就能修复的离散子群。这是诚实的 negative result —— 它同时给出**唯一可操作的抓手(FN-silent 静默复发,值得专门找早期生物标志物)**,并提醒**不要期待靠错例聚类一键提分**。与第 4 节错例轮换、第 2 节顽固仅占 ~1/5 一致:错误是流动且弥散的,不是一个固定难例簇。

---

## 6. Phase 5 高价值跨地标点子

### ① 风险档迁移 3×3 + 桑基(temporal)

![风险档桑基](figures/xland_Figure_05_TierSankey.png)

风险档(Low<0.30≤Mid<0.60≤High,切在 EBM 概率上)随地标推进**向两极结晶**:Mid 档持续掏空(节点总数 66→54→20→24),人次稳定地流向 Low 或 High。如 6M→12M:Low 99/113 留 Low、High 55/68 留 High,Mid 仅 8 人留 Mid。**临床含义:越晚的地标,风险分层越干脆、灰区越小**,支持「在 6M/12M 落定分诊」。

### ② persistence 增量随地标(EBM−persist ΔAUC + 配对 bootstrap CI)

| Landmark | EBM_temporal | Persist_temporal | dAUC_EBM_minus_Persist | CI_low | CI_high | CI排除0 | dAUC_EBM_minus_Naive |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1M | 0.6939 | 0.5509 | 0.1424 | 0.0556 | 0.2236 | ✓ | 0.1949 |
| 3M | 0.7914 | 0.6837 | 0.1081 | 0.054 | 0.1635 | ✓ | 0.2922 |
| 6M | 0.878 | 0.7545 | 0.1227 | 0.0734 | 0.1755 | ✓ | 0.3779 |
| 12M | 0.9075 | 0.7512 | 0.1552 | 0.0965 | 0.2103 | ✓ | 0.4079 |

![persistence增量](figures/xland_Figure_06_PersistIncrement.png)

EBM 相对 persistence 的 ΔAUC 在**四个地标全部 95% CI 排除 0**(1M +0.142 [0.056, 0.224]、3M +0.108 [0.054, 0.164]、6M +0.123 [0.073, 0.176]、12M +0.155 [0.097, 0.210]),12M 增量最大。即:**「攒齐多模态轨迹的学习模型」在每个地标都稳定优于「只看今日 TSH」**,且这一优势在最晚地标不衰反增。

### ③ per-landmark 最优弃权率(拆 pooled 的 risk-coverage)

| Landmark | abstain_pct | coverage_pct | n_retained | accuracy | NPV | PPV |
| --- | --- | --- | --- | --- | --- | --- |
| 1M | 0 | 100 | 201 | 0.6517 | 0.7248 | 0.5652 |
| 1M | 30 | 70 | 141 | 0.6667 | 0.7429 | 0.5915 |
| 3M | 0 | 100 | 201 | 0.7114 | 0.802 | 0.62 |
| 3M | 30 | 70 | 141 | 0.773 | 0.8462 | 0.7105 |
| 6M | 0 | 100 | 201 | 0.8109 | 0.8403 | 0.7683 |
| 6M | 30 | 70 | 141 | 0.8794 | 0.9067 | 0.8485 |
| 12M | 0 | 100 | 201 | 0.8308 | 0.8512 | 0.8 |
| 12M | 30 | 70 | 141 | 0.9007 | 0.9114 | 0.8871 |

![逐地标弃权](figures/xland_Figure_07_PerLandmarkAbstain.png)

把池化的选择性预测拆到每地标:**弃权(转交临床)的收益强烈依赖地标**。12M 弃权 30% → 准确率 0.83→0.90、NPV→0.91;6M 类似(0.81→0.88);而 1M 弃权 30% 仅 0.65→0.67 —— **早期弱信号地标靠弃权也救不回来**,提示分诊式弃权应在 6M 之后启用,1M 更宜直接走低阈值筛查。

### ④ 错例 churn(相邻地标,见第 4 节)

相邻地标错例 Jaccard 0.35–0.47,~40–57% 错例在下一地标被纠正,印证错误以流动为主。

---

## 7. 局限与口径声明

- **Temporal 全程仅作读出**;阈值(Youden)、风险档切点、亚型门槛均在 dev OOF 上预注册/选定。
- 6M/12M 功能态为**推导态**(临床 Eval 退化),已注明;FN-silent 的「看似正常」据此判定,若有真实 6M/12M 甲功标注可进一步收紧。
- 亚型池取每类 top-40 置信错例(共识阈值),改变 K 会轻移占比但不改「弥散」定性(silhouette 0.247、ARI 0.04 离门槛仍有距离)。
- 不含 post-RAI 用药特征(适应证混杂);未报告任何 unique-患者计数(分析单元 = 人次)。
- 未做显著性断言;增量/弃权收益均以 95% bootstrap CI 是否排除 0 表述。

产物清单:`tables/xland_*.csv|json`(19)、`figures/xland_Figure_0[0-7]_*.png`(8)。复算入口:四个 `module2_v2_b4_xland_{tracking,trajectory,errtypes,phase5}.py` 脚本。
