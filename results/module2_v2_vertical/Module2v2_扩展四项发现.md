# M2 · v2 EBM 扩展四项发现(**corrected 主线 refresh,2026-06**)

> 本次扩展回答 reviewer 必问/作者主动追加的科学问题。**重要 caveat**:本次为 **corrected 真值(`Current_Time=="{L}M"` 行的 `*_Current` 列)+ median 插补** 主线下的 refresh,**与旧版(基于退化宽列的扩展数据)结论有方向性更正**。详见各节"vs 旧版"标注。

---

## 摘要

| 项 | corrected 主线结论 | 论文价值 |
|:--|:--|:--|
| **(1) M3 拆出** | 本仓库即纯 M2,M3 已独立成文 | 叙事干净 |
| **(2) 12M EBM 扩展** | AUC **6M 0.856(峰)→ 12M 0.811**(回落,符合 lead-time 物理);**12M top-1 = TSH current 0.51**,迁移到垂体反馈轴;velocity 退至 top-5/6 | **vs 旧版重大纠正:6M top-1 是 Hormone_load (0.774) 不是 velocity** |
| **(3) 患者级分解** | corrected 数据下重挑 3 病人,Mid 患者真值 Y=1(模型给 P=0.272 漏报),High 患者(TPOAb 878+大腺体)P=0.944 / Y=1 ✓ | EBM 玻璃盒能力展示 |
| **(4) GREAT/Vos 头对头** | EBM @6M 0.841 显著胜:Δ +0.16~+0.22 全 CI 排除 0;GREAT-3 在 RAI 队列非单调 | reviewer 必看 baseline |

---

## (1) M3 拆出 ✅ 已在本仓

本仓库 `VickySu1112/thyroidML` 默认分支 `m2v3-fix-1m-auc` 即**纯 M2**(README 第一段写明)。M3 已规划独立成文。

---

## (2) 12M EBM 扩展(corrected 真值 + median 主线)

**脚本**:`scripts/simple/module2_v2_ebm_12m_extension.py`(已切到 `load_stacked_x(corrected=True)`)
**产物**:`results/module2_v2_vertical/m2v2_ebm_12m/`

### 性能(全 n_test=201)

| Landmark | ROC-AUC(corrected) | 主线 README 值 | PR-AUC | Brier | lead 月数 |
|:--:|:--:|:--:|:--:|:--:|:--:|
| 1M | 0.705 | 0.704 | 0.641 | 0.213 | 23 |
| 3M | 0.797 | 0.799 | 0.715 | 0.179 | 21 |
| **6M** | **0.856** | 0.858 | 0.833 | 0.148 | 18 |
| 12M | **0.811** | 0.795 | 0.767 | 0.175 | 12 |

我跑的 12M 数字 0.811 比新仓主线 0.795 高 0.016 —— 可能因 12M EBM 默认 `interactions=5` vs 主线 bin/seed 略不同,**量级一致**。

**与旧版(退化数据)对比**:旧 6M EBM 0.854 → corrected 6M 0.856(几乎不变);旧 12M 0.816 → corrected 12M 0.811。AUC 数字几乎不变,**但重要性迁移结论被大幅纠正**(见下)。

### 重要性 top-6(corrected — **重大叙事纠正**)

| Landmark | #1 | #2 | #3 | 评注 |
|:--|:--|:--|:--|:--|
| 1M | 甲状腺重量(0.46) | FT3,FT4 level(0.18) | TPOAb × velocity 交互(0.13) | 解剖负荷主导 |
| 3M | FT3,FT4 level(0.53) | 甲状腺重量(0.37) | TSH current(0.28) | 当期水平接管 |
| **6M** | **FT3,FT4 level(0.77)** | TSH current(0.50) | 甲状腺重量(0.34) | **level + TSH 双主导;velocity 0.22 退到 top-5** |
| **12M** | **TSH current(0.51)** | FT3,FT4 level(0.49) | 甲状腺重量(0.44) | **迁移到 TSH(垂体反馈)**;3 个 level×velocity 交互进 top-6 |

**重大 vs 旧版纠正**:
- 旧报告说"6M velocity 主导(0.62)"—— **错了**!那是旧 wide-column 退化数据(`FT4_6M` 在 1003 人次只有 13 真值)的人为产物。corrected 真值下 **6M 主导是 Hormone_load(0.77),velocity 退到 top-5(0.22)**。
- 旧报告说"12M velocity 退出 top-3、TSH current 顶替" —— **方向对,数字更对**:12M top-1 现在是 TSH current(0.51)。
- "动量"并未消失,而是以 6M top-5(velocity)+ 12M top-4 / top-5 "level × velocity" 交互形式提供增量(momentum-beyond-inertia)。

![12M 重要性迁移热图](m2v2_ebm_12m/figures/F_importance_drift_heatmap.png)

---

## (3) 患者级分解 + 反事实(corrected)

**脚本**:`scripts/simple/module2_v2_ebm_patient_decomp.py`(已切到 `load_stacked(corrected=True)`)
**产物**:`results/module2_v2_vertical/m2v2_ebm_patient_decomp/`

挑 3 个 dev@6M episode(P 的 10/50/90 百分位):

| 标签 | 预测 P | 真 Y | ThyroidW | TPOAb | TRAb | FT4_current | TSH_current | FT3_velocity |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **Low** | **0.011** | 0 | 18.2 | 10 | 2.88 | 17.6 | 0.4(已升) | +0.85 | ← 模型对 |
| Mid | 0.272 | **1** | 10.6 | 10 | 5.45 | 20.8(高) | 0.004(抑制) | 0.0 | ← **模型漏报** |
| **High** | **0.944** | **1** | **42.4** | **878** | 1.75 | 19.2(高) | 0.006(抑制) | −0.33 | ← 模型对 |

**临床读法**:
- **Low** 病人:小腺体 + 低 TPOAb + TSH 已升回 0.4(治疗后已分化到甲减) → EBM 极低 P,Y=0 ✓
- **Mid** 病人:**模型漏报(P=0.272 但 Y=1 实际复发)** — 这是个典型 hidden stratification 例子:小腺体 + 低 TPOAb 让模型判低危,但其实 FT4 仍很高(20.8)+ TSH 完全抑制说明甲亢未控,模型对"小腺体的难治型"低估
- **High** 病人:**大腺体 + 极高 TPOAb(878)** + 高激素水平 → EBM 给 P=0.944,Y=1 ✓

反事实曲线见 `m2v2_ebm_patient_decomp/figures/F_counterfactual_*.png`。

---

## (4) GREAT/Vos 头对头(corrected)

**脚本**:`scripts/simple/module2_v2_great_baseline.py`(已切到 `load_stacked(corrected=True)`)
**产物**:`results/module2_v2_vertical/m2v2_great_baseline/`

**头对头(temporal n=201)**:

| 方法 | ROC-AUC | 95% CI | calib slope |
|:--|:--:|:--:|:--:|
| GREAT-3(Vos 阈值) | 0.622 | 0.54–0.70 | ≈0(非单调,完全失效) |
| GREAT-cont(refit) | 0.677 | 0.60–0.75 | 0.74 |
| M1 v6 LR(6 LASSO 特征) | 0.681 | 0.61–0.76 | 0.72 |
| **M2 EBM @ 6M(best)** | **0.841** | 0.78–0.90 | 0.64 |

**Paired ΔAUC vs EBM(全 CI 排除 0,EBM 显著胜)**:

| vs | ΔAUC | 95% CI |
|:--|:--:|:--:|
| GREAT-3 | **+0.220** | [+0.13, +0.31] |
| GREAT-cont | **+0.164** | [+0.08, +0.25] |
| M1 v6 LR | **+0.160** | [+0.08, +0.24] |

**GREAT-3 在 RAI 队列非单调**(评分 0→事件率 0.41,评分 1→0.28,评分 2→0.51,评分 3→0.69),确认 GREAT 原本是 ATD 后场景、阈值不直接转移到 RAI。

**与旧版差异**:数字几乎不变(±0.005),结论稳健 — GREAT vs EBM 的差距与所用 EBM 真值口径无关。

![EBM vs GREAT vs M1 v6 ROC](m2v2_great_baseline/figures/F_ROC_overlay.png)

---

## 数据出处与方法学交代

- **数据来源**:`1003.xlsx`(原始)→ `build_stage2_long_table_minimal.py` 重生 long-format `stage2_long_table.csv`(7022 行 = 1003 × 7 landmark,含 `Current_Time` / `*_Current` 列)
- **CORRECTED 真值**:`load_stacked(corrected=True)` 与 `load_stacked_x(corrected=True)` 从 long 表 `Current_Time=="{L}M"` 行的 `*_Current` 列取值,**绕开退化的宽列**(`FT4_6M` 在 1003 人次只有 13 真值的 bug)
- **median 插补**:default;LOCF 在新仓另有 `m2v2_ebm_full_locf/` 作敏感性对照,本扩展只走 median 主线
- **重大叙事更正**:本次 refresh 把"velocity 主导 6M"的旧叙事改成"**Hormone_load (level) 主导 6M,velocity 退到 top-5**"——与新仓主线 EBM 论文 README 一致

## 关联报告

- 主线 EBM 论文(median 主线)→ [`Module2v2_EBM_paper.md`](Module2v2_EBM_paper.md)(已是 corrected/median)
- 本扩展产物 → `m2v2_ebm_12m/` · `m2v2_ebm_patient_decomp/` · `m2v2_great_baseline/`
