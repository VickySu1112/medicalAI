# RAI 随访风险工作台与论文报告

在线应用：https://rai-tmu.streamlit.app/

本仓库当前只保留两条主线：

1. **固定 Landmark 早期反应评估**
   以 3M 为主、6M 为补充，评估 RAI 后 Graves 病患者是否仍处于 Hyper / non-Hyper 高风险轨道。主报告见：
   - [固定 Landmark 甲亢结局预测报告](results/landmark_focus/README.md)
   - [HTML 版](results/landmark_focus/固定Landmark甲亢结局预测报告.html)

2. **四分支单头动态复发预警模型**
   在每个 landmark `t`，只使用 `t` 及之前信息，预测下一时间窗 `t -> t+1` 是否发生 Normal -> Hyper 复发。主报告见：
   - [复发.md](results/t5_dynamic_paper/复发.md)
   - [HTML 版](results/t5_dynamic_paper/复发.html)

## 当前方法概览

- `3M` 主线：fixed-landmark early-response classifier，使用 baseline、1M、3M 的时间安全特征，报告 AUC、PR-AUC、accuracy、calibration、DCA、SHAP、阈值敏感性和错题分析。
- `复发` 主线：static / local / global / z3M 四分支 encoder，经 time-aware gated fusion 后接单一 hazard head，输出下一时间窗复发风险。
- 验证边界：先按患者时间顺序切分，再做 train-only 插补、阈值选择和模型选择；temporal test 只用于最终报告。
- Streamlit app：保留 `streamlit_app.py`、`thyroid_app/` 与 `artifacts/`，用于部署 3M 评估与动态复发预警。

## 主要入口

```bash
streamlit run streamlit_app.py
```

导出 / 校验部署 bundle：

```bash
python scripts/export_streamlit_artifacts.py
python scripts/verify_streamlit_artifacts.py
```

## 目录说明

- `streamlit_app.py`：中文 Streamlit 工作台入口。
- `thyroid_app/`：部署特征构建、推理、页面内容与 artifact 导出逻辑。
- `artifacts/`：Streamlit Cloud 使用的固定模型 bundle。
- `scripts/relapse_direct_threebranch_aug.py`：当前动态复发主模型训练 / sweep 入口。
- `scripts/simple/`：3M 报告、复发报告和补充图表生成脚本。
- `results/landmark_focus/`：固定 landmark 主报告。
- `results/t5_dynamic_paper/`：最新动态复发主报告。

历史 teacher / frozen-fuse / two-stage 结果不再作为当前主线报告入口。
