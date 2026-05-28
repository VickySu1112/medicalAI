# 30 篇文献主题综合

## 1. 与本研究最直接相关的 RAI / Graves 文献

P01-P13 共同支持一个结论：RAI 后治疗失败、未缓解、非完全缓解、难治和 NHRH 虽术语不同，但都指向“治疗未能带来稳定控制”。甲状腺负荷、RAI 摄取/剂量学、TRAb、ATD 史以及早期 FT3/FT4 下降是反复出现的变量。本研究的优势不是发现单个全新变量，而是把这些变量按时间可见性放回 0M、1M/3M/6M 和 rolling landmark 三个临床时刻。

## 2. Landmark 和纵向 biomarker 方法学

P14-P17 支持动态预测的核心设计：风险不应只在 baseline 固定一次，而应在新 biomarker 到来时更新。P15 尤其支撑“轨迹摘要”语言，即 current value、delta、slope、AUC、rebound、history count 等特征是对个体纵向信息的压缩。对应到本项目，Module 2 是长期 NHRH updater，Module 3 是下一窗口 rolling monitor。

## 3. 评价与报告规范

P18-P26 解释为什么本研究不能只报 AUC。校准决定概率能不能用于风险沟通，DCA 决定阈值下是否有临床净获益，PR-AUC 对低事件率 H1 任务更敏感。TRIPOD+AI 和 PROBAST+AI 支持当前报告中的 temporal validation、缺失处理、泄漏控制、校准、DCA 和适用性边界。

## 4. 相似医学数据分析 / ML 文章

P27-P30 提供外部方法类比：临床预测模型要考虑中心间异质性、EHR 纵向信息和动态风险更新。它们支持使用 longitudinal histories，但也提醒我们在 1003 治疗人次规模下不应盲目追求深度模型；可解释、校准良好的 logistic landmark pipeline 更符合当前证据强度。

## 5. 可直接写入论文的定位句

- 治疗前 baseline 模型的中等表现不是失败，而是治疗前信息上限的诚实估计。
- 早期治疗反应是长期 NHRH 风险更新的主要信息来源，这一点与 RAI nomogram 和 TRAb/FT3/FT4 变化文献一致。
- rolling H1 是低事件率预警任务，因此 PR-AUC、DCA、NPV 和治疗级风险层级比 accuracy 更有临床意义。
- 所有文献对照都应避免直接横比 AUC，因为 endpoint、验证方式和可见信息时点不同。
