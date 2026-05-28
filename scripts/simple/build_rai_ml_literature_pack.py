#!/usr/bin/env python
"""Build a 30-paper literature pack for the RAI three-module manuscript.

The pack is intentionally file-based and reviewable:
- one markdown summary per included paper
- one master CSV table
- search/screening logs
- raw OpenAlex metadata when available

Network metadata enriches the hand-curated paper list, but the summaries are
kept explicit and conservative so they can be reviewed by clinicians.
"""
from __future__ import annotations

import csv
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/literature/rai_ml_q2plus_30"
SUM = OUT / "summaries"
TAB = OUT / "tables"
SRC = OUT / "sources"


JOURNAL_EVIDENCE: dict[str, dict[str, str]] = {
    "Frontiers in Endocrinology": {
        "status": "Q1/Q2 supported",
        "evidence": "Endocrinology journal with public SJR/JCR-style quartile evidence; use institutional JCR/CAS for final partition wording.",
        "url": "https://www.frontiersin.org/journals/endocrinology",
    },
    "Health Care Science": {
        "status": "Q2 supported",
        "evidence": "Wiley clinical science journal with public quartile evidence; current institutional verification still recommended.",
        "url": "https://onlinelibrary.wiley.com/journal/27706760",
    },
    "European Journal of Endocrinology": {
        "status": "Q1 supported",
        "evidence": "Leading European endocrine journal; public metrics commonly place it above Q2.",
        "url": "https://academic.oup.com/ejendo",
    },
    "European Journal of Nuclear Medicine and Molecular Imaging": {
        "status": "Q1 supported",
        "evidence": "Leading nuclear medicine journal; public SJR/JCR-style metrics commonly report Q1.",
        "url": "https://link.springer.com/journal/259",
    },
    "Endocrine Connections": {
        "status": "Q2+ plausible",
        "evidence": "Peer-reviewed endocrine journal indexed in major databases; manual JCR/CAS check recommended for exact year.",
        "url": "https://ec.bioscientifica.com/",
    },
    "International Journal of Endocrinology": {
        "status": "Q2 supported",
        "evidence": "Endocrinology-focused journal with public quartile evidence; manual current quartile check recommended.",
        "url": "https://www.hindawi.com/journals/ije/",
    },
    "Thyroid": {
        "status": "Q1 supported",
        "evidence": "Leading thyroid/endocrinology journal and guideline venue.",
        "url": "https://home.liebertpub.com/publications/thyroid/55",
    },
    "BMJ": {
        "status": "Top-tier supported",
        "evidence": "Top-tier general medical journal; accepted as above Q2 for reporting/risk-of-bias statements.",
        "url": "https://www.bmj.com/",
    },
    "Annals of Internal Medicine": {
        "status": "Top-tier supported",
        "evidence": "Top-tier general/internal medicine journal; accepted as above Q2 for reporting and risk-of-bias tools.",
        "url": "https://www.acpjournals.org/journal/aim",
    },
    "BMC Medicine": {
        "status": "Q1/top medical supported",
        "evidence": "High-impact BMC medical journal; manual current JCR/CAS check can be added if required.",
        "url": "https://bmcmedicine.biomedcentral.com/",
    },
    "BMC Medical Research Methodology": {
        "status": "Q1/Q2 methods supported",
        "evidence": "Recognized medical methodology journal used for prediction-method articles.",
        "url": "https://bmcmedresmethodol.biomedcentral.com/",
    },
    "Statistics in Medicine": {
        "status": "Q1/Q2 methods supported",
        "evidence": "Leading biostatistics journal used for prediction methodology.",
        "url": "https://onlinelibrary.wiley.com/journal/10970258",
    },
    "Biometrics": {
        "status": "Q1/Q2 methods supported",
        "evidence": "Leading biostatistics journal used for dynamic prediction and joint-model methodology.",
        "url": "https://academic.oup.com/biometrics",
    },
    "Scandinavian Journal of Statistics": {
        "status": "Q1/Q2 methods supported",
        "evidence": "Recognized statistics journal; landmarking foundation article included for method relevance.",
        "url": "https://onlinelibrary.wiley.com/journal/14679469",
    },
    "Medical Decision Making": {
        "status": "Q1/Q2 methods supported",
        "evidence": "Core decision-science journal; foundational DCA article included for method relevance.",
        "url": "https://journals.sagepub.com/home/mdm",
    },
    "Epidemiology": {
        "status": "Q1/Q2 methods supported",
        "evidence": "High-quality epidemiology journal; dynamic prediction and performance framework papers included for methodological analogy.",
        "url": "https://journals.lww.com/epidem/",
    },
    "PLOS ONE": {
        "status": "Q1/Q2 broad journal supported",
        "evidence": "Peer-reviewed broad journal; PR-AUC paper is highly cited and methodologically central.",
        "url": "https://journals.plos.org/plosone/",
    },
    "The American Journal of Surgery": {
        "status": "Q1/Q2 clinical journal supported",
        "evidence": "Clinical surgery journal; RAI failure meta-analysis included for feature evidence. Manual current quartile check recommended.",
        "url": "https://www.sciencedirect.com/journal/the-american-journal-of-surgery",
    },
    "npj Digital Medicine": {
        "status": "Q1/top digital medicine supported",
        "evidence": "High-impact Nature Portfolio digital medicine journal.",
        "url": "https://www.nature.com/npjdigitalmed/",
    },
    "Scientific Reports": {
        "status": "Q1/Q2 broad journal supported",
        "evidence": "Nature Portfolio broad peer-reviewed journal; included for classic EHR representation learning.",
        "url": "https://www.nature.com/srep/",
    },
}


KEY_FINDING_ZH: dict[str, str] = {
    "P01": "NHRH 可作为 RAI 后未愈合或复发的临床复合不良结局；影像、Ki-67 与常规临床变量可共同提供风险信息。",
    "P02": "NHRH 被明确作为治疗失败或复发复合终点处理，并提示不同性别的风险因素可能不同。",
    "P03": "RAIU、有效半衰期、总碘剂量和单位甲状腺剂量等 RAI 生理变量与 non-remission 风险相关。",
    "P04": "治疗后早期 FT3/FT4 下降及 RAI 生理变量有助于识别 non-complete remission。",
    "P05": "常规临床和 RAI 相关变量可分层难治性 Graves 风险，但 endpoint 与 NHRH 并不完全相同。",
    "P06": "TRAb 的动态变化模式与 RAI 后疗效相关，支持把 TRAb 视为纵向 biomarker 而非单点变量。",
    "P07": "非线性机器学习和可解释性工具可用于 RAI 结局预测，但验证方式和 endpoint 需要谨慎比较。",
    "P08": "RAI 前基线临床特征已包含一定的术后甲功状态预测信号。",
    "P09": "甲状腺负荷和剂量密度类变量是 RAI 疗效预测中反复出现的临床合理因素。",
    "P10": "RAI 治疗决策需要综合疾病背景、剂量、准备过程和随访管理。",
    "P11": "Graves 病管理需要联合实验室指标、临床状态、治疗选择和随访信息。",
    "P12": "简单临床风险评分可进行复发风险分层，但需要外部验证。",
    "P13": "RAI 失败研究中反复出现基线严重程度、甲状腺负荷和剂量相关因素，但研究间异质性明显。",
    "P14": "Landmarking 可在新 biomarker 到来时更新风险预测，同时避免完整联合模型的复杂分布假设。",
    "P15": "个体纵向 biomarker 轨迹可通过摘要特征进入动态预测模型。",
    "P16": "风险预测应随新的纵向测量更新，并用时间敏感方式评价预测性能。",
    "P17": "Landmarking 是一种透明的动态预测框架，核心是只使用 landmark 时点以前的信息。",
    "P18": "判别能力好不等于概率可用；校准曲线、校准截距/斜率和 Brier 分数是必要评价。",
    "P19": "DCA 可在不同阈值概率下比较模型、treat-all 和 treat-none 的净获益。",
    "P20": "在类别不平衡任务中，PR 曲线通常比 ROC 曲线更能反映阳性预测的实际价值。",
    "P21": "TRIPOD+AI 要求清楚报告数据来源、候选预测因子、缺失处理、验证、性能、校准和预期用途。",
    "P22": "PROBAST+AI 强调从研究对象、预测因子、结局、分析、验证和适用性评估偏倚风险。",
    "P23": "临床预测模型需要透明报告，便于复现和独立评价。",
    "P24": "预测模型偏倚可来自对象选择、预测因子测量、结局定义和分析流程。",
    "P25": "预测模型开发需要考虑过拟合、事件数、收缩和预期性能，避免模型复杂度超过数据支持。",
    "P26": "临床预测模型不能只依赖单一指标，需要同时报告判别、校准和临床实用性。",
    "P27": "个体参与者数据预测研究需要关注异质性和模型可迁移性。",
    "P28": "纵向 EHR 信息可提升临床预测，但复杂模型必须配套严格验证和解释。",
    "P29": "EHR 历史可学习到与未来疾病风险相关的患者表征。",
    "P30": "重复临床测量可用于随时间更新患者特异性风险，是动态预测在真实注册队列中的临床例子。",
}


POSSIBLE_USE_ZH: dict[str, str] = {
    "P01": "可能用于支撑本文 NHRH 复合终点定义，并说明为何需要同时报告校准和临床效用。",
    "P02": "可能用于补充 NHRH 术语来源、临床含义以及亚组异质性讨论。",
    "P03": "可能用于解释 Module 1 中 RAI 摄取、剂量密度和甲状腺负荷变量的临床合理性。",
    "P04": "可能用于支撑 Module 2 的核心论点：早期治疗反应应更新长期 NHRH 风险，而不是只看 baseline。",
    "P05": "可能用于 related work 中梳理 non-remission、refractory 和 NHRH 等相邻 endpoint 的边界。",
    "P06": "可能用于支持 Module 2/3 中 TRAb 轨迹特征和免疫机制解释，但不直接作为模型优越性证据。",
    "P07": "可能用于说明为何需要加入高级 ML benchmark 和 SHAP，同时强调 temporal validation 更严格。",
    "P08": "可能作为同病种 ML 建模参照，用于说明 baseline feature engineering 和可解释性做法。",
    "P09": "可能用于支持甲状腺重量、剂量/甲状腺负荷等变量的医学解释。",
    "P10": "可能用于临床背景和限制性表述，避免把模型写成治疗剂量因果推荐器。",
    "P11": "可能用于定义 Graves 管理边界，说明模型只能辅助风险分层，不能替代临床决策。",
    "P12": "可能用于支撑本文采用可解释风险层级和强调外部验证必要性的写法。",
    "P13": "可能用于说明 core feature 的临床可解释性，同时提醒不要过度解读单个预测因子。",
    "P14": "可能用于支撑 Module 2/3 的 landmark updater 叙事，而不是把多个地标模型写成彼此割裂的模型。",
    "P15": "可能用于支撑 trajectory summary / momentum 特征的写法，尤其是 current、delta、slope、AUC、rebound 等摘要。",
    "P16": "可能作为动态预测和联合模型背景，说明本文采用更务实的 landmark 近似而非完整 joint model。",
    "P17": "可能用于支撑时间安全设计和“不使用未来信息”的方法学原则。",
    "P18": "可能用于解释为何三模块均报告 Brier、calibration curve、intercept 和 slope。",
    "P19": "可能用于解释 DCA 图和阈值范围下净获益的临床意义。",
    "P20": "可能用于支撑 Module 3 低事件率 H1 任务中 PR-AUC、PPV 和风险富集比 accuracy 更关键。",
    "P21": "可能用于规范全文报告结构、claim-evidence map 和模型用途边界。",
    "P22": "可能用于审稿风险自查，尤其是泄漏、适用性和偏倚风险讨论。",
    "P23": "可能用于补充 TRIPOD+AI 的传统预测模型报告基础。",
    "P24": "可能用于说明本文为何单列 leakage audit、缺失处理、temporal test 和适用性限制。",
    "P25": "可能用于支撑本文不盲目升级高容量模型、优先保留校准良好 LR 的保守策略。",
    "P26": "可能用于支撑多指标评价框架：ROC-AUC、PR-AUC、Brier、校准、DCA 和风险层级共同解释。",
    "P27": "可能用于讨论外部验证和中心间迁移风险，避免把单中心 temporal test 写成已充分泛化。",
    "P28": "可能用于把本文放入 longitudinal EHR / clinical ML 背景，同时解释为何当前数据规模下不强推深度模型。",
    "P29": "可能用于表示纵向历史可以形成风险表征，但本文采用更透明的 landmark feature 而非黑箱表示学习。",
    "P30": "可能作为非内分泌领域动态预测类比，支撑 M2/M3 随访中持续更新风险的临床逻辑。",
}


def slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return re.sub(r"_+", "_", text)[:90]


def reconstruct_abstract(inv: dict[str, list[int]] | None) -> str:
    if not inv:
        return ""
    pos: dict[int, str] = {}
    for word, indices in inv.items():
        for idx in indices:
            pos[int(idx)] = word
    return " ".join(pos[i] for i in sorted(pos))


def fetch_json(url: str, timeout: int = 25) -> dict[str, Any] | None:
    req = urllib.request.Request(url, headers={"User-Agent": "medicalAI-litpack/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"_fetch_error": str(exc), "_url": url}


def openalex_lookup(entry: dict[str, Any]) -> dict[str, Any]:
    base = "https://api.openalex.org/works"
    if entry.get("doi"):
        doi = entry["doi"].replace("https://doi.org/", "").strip()
        url = f"{base}/doi:{urllib.parse.quote(doi, safe='')}"
    else:
        q = urllib.parse.quote(entry["title"])
        url = f"{base}?search={q}&per_page=1&mailto=ql@example.com"
    data = fetch_json(url)
    time.sleep(0.12)
    if not data:
        return {}
    if "results" in data:
        return data["results"][0] if data["results"] else {"_fetch_error": "no_results", "_url": url}
    return data


PAPERS: list[dict[str, Any]] = [
    {
        "id": "P01",
        "short_key": "Wang_EC_NHRH_Ki67_2024",
        "title": "Ultrasound combined with Ki-67 to construct the prognostic model for radioactive iodine therapy outcomes in Graves' disease patients",
        "year": 2024,
        "journal": "Endocrine Connections",
        "doi": "10.1530/EC-23-0429",
        "url": "https://doi.org/10.1530/EC-23-0429",
        "theme": "Direct RAI/NHRH endpoint",
        "quartile_evidence": "Endocrinology journal; likely Q2 or above, verify manually",
        "reading_basis": "local PDF + metadata",
        "local_file": "ref_paper/pdfs/2024_Wang_EndocrineConnections_NHRH_Ki67_ultrasound_EC-23-0429.pdf",
        "summary": "This is one of the closest endpoint precedents for the current project because it explicitly uses nonhealing or recurrence of hyperthyroidism after RAI as an adverse outcome. It combines ultrasound and Ki-67 information with clinical predictors, and reports discrimination, calibration-type information and clinical usefulness. The key value for our paper is endpoint framing: persistence and recurrence can be treated as a clinically meaningful composite because both imply inadequate durable control after RAI.",
        "methods": "Retrospective prognostic modeling for RAI outcomes in Graves disease, including ultrasound/Ki-67 predictors and multivariable risk modeling.",
        "findings": "NHRH is operationalized as a clinically relevant adverse endpoint; imaging/proliferation features can add risk information beyond routine variables.",
        "relevance": "Supports our NHRH endpoint definition and the need to report calibration/clinical utility rather than AUC alone.",
        "limitations": "Smaller single-center cohort and imaging/Ki-67 features are not directly available in our baseline table.",
        "module_link": "M1/M2 endpoint justification",
    },
    {
        "id": "P02",
        "short_key": "Shen_HCS_NHRH_Sex_2025",
        "title": "Sex-related differences in risk factors associated with nonhealing or recurrence of hyperthyroidism in patients with Graves' disease treated with radioactive iodine",
        "year": 2025,
        "journal": "Health Care Science",
        "doi": "10.1002/hcs2.70021",
        "url": "https://doi.org/10.1002/hcs2.70021",
        "theme": "Direct RAI/NHRH endpoint",
        "quartile_evidence": "Wiley clinical science journal; verify current JCR/SJR manually",
        "reading_basis": "local HTML + metadata",
        "local_file": "ref_paper/html/2025_Shen_HealthCareScience_NHRH_sex_related_DOAJ.html",
        "summary": "This paper is useful because it uses the same NHRH terminology and explicitly contrasts NHRH with non-NHRH outcomes such as hypothyroidism or euthyroidism. The sex-stratified analysis also reminds us that clinical predictors can differ across subgroups, which is relevant when interpreting a single global model.",
        "methods": "Retrospective risk-factor analysis after RAI with sex-stratified modeling.",
        "findings": "NHRH is treated as a failure/recurrence composite, and risk factors can differ by sex.",
        "relevance": "Directly supports the terminology and clinical meaning of the composite endpoint.",
        "limitations": "Not a dynamic landmark model; subgroup findings should not be imported into our dataset without testing.",
        "module_link": "M1/M2 endpoint and subgroup discussion",
    },
    {
        "id": "P03",
        "short_key": "Yu_Frontiers_NonRemission_2024",
        "title": "Nomogram construction and evaluation for predicting non-remission after a single radioactive iodine therapy for Graves' hyperthyroidism",
        "year": 2024,
        "journal": "Frontiers in Endocrinology",
        "doi": "10.3389/fendo.2024.1391014",
        "url": "https://www.frontiersin.org/journals/endocrinology/articles/10.3389/fendo.2024.1391014/full",
        "theme": "RAI nomogram / non-remission",
        "quartile_evidence": "Frontiers in Endocrinology is commonly indexed in endocrine Q2/Q1 contexts; verify current local quartile manually",
        "reading_basis": "local PDF + full-text webpage",
        "local_file": "ref_paper/pdfs/2024_Yu_Frontiers_non_remission_single_RAI_1391014.pdf",
        "summary": "The study builds a nomogram for non-remission after single RAI therapy. It is important for our Module 1 because it uses treatment-time variables such as RAI uptake and dose-related indices to predict a long-term treatment outcome. It also provides an external benchmark for conventional nomogram reporting.",
        "methods": "Retrospective nomogram with internal validation, ROC, calibration and DCA.",
        "findings": "RAIU, effective half-life, total iodine dose and iodine dose per gram are clinically meaningful predictors in RAI outcome prediction.",
        "relevance": "Supports inclusion and cautious interpretation of RAI physiology and dose-density variables.",
        "limitations": "Validation design and endpoint timing differ from our temporal split and 24M NHRH framing.",
        "module_link": "M1 variable rationale and benchmark",
    },
    {
        "id": "P04",
        "short_key": "Wang_Frontiers_NCR_2025",
        "title": "A prognostic nomogram model for non-complete remission following initial radioiodine therapy in Graves' hyperthyroidism",
        "year": 2025,
        "journal": "Frontiers in Endocrinology",
        "doi": "10.3389/fendo.2025.1692702",
        "url": "https://www.frontiersin.org/journals/endocrinology/articles/10.3389/fendo.2025.1692702/full",
        "theme": "Early response / non-complete remission",
        "quartile_evidence": "Frontiers in Endocrinology; verify current local quartile manually",
        "reading_basis": "local PDF + full-text webpage",
        "local_file": "ref_paper/pdfs/2025_Wang_Frontiers_non_complete_remission_nomogram_1692702.pdf",
        "summary": "This paper is central for Module 2 because it demonstrates that early post-RAI biochemical change, especially early FT3/FT4 response, improves prediction of non-complete remission. It provides an external clinical rationale for our landmark updater from 0M to 1M/3M/6M.",
        "methods": "Nomogram using treatment-time and early follow-up variables; reports discrimination, calibration and DCA.",
        "findings": "Early FT3/FT4 decline and RAI physiology variables help identify non-complete remission.",
        "relevance": "Supports the claim that early treatment response should update long-term risk instead of relying only on baseline predictors.",
        "limitations": "Uses a single early window and random/internal validation rather than a full temporal landmark trajectory.",
        "module_link": "M2 landmark updater",
    },
    {
        "id": "P05",
        "short_key": "Liao_Frontiers_Refractory_2025",
        "title": "Development and validation of a nomogram prediction model for factors influencing 131I-refractory Graves' hyperthyroidism",
        "year": 2025,
        "journal": "Frontiers in Endocrinology",
        "doi": "10.3389/fendo.2025.1628226",
        "url": "https://www.frontiersin.org/journals/endocrinology/articles/10.3389/fendo.2025.1628226/full",
        "theme": "Refractory Graves after RAI",
        "quartile_evidence": "Frontiers in Endocrinology; verify current local quartile manually",
        "reading_basis": "local PDF + full-text webpage",
        "local_file": "ref_paper/pdfs/2025_Liao_Frontiers_131I_refractory_nomogram_1628226.pdf",
        "summary": "This paper addresses refractory Graves hyperthyroidism after RAI, a concept close to treatment failure. It helps position our NHRH outcome within a broader family of non-remission, refractory and persistent/recurrent endpoints.",
        "methods": "LASSO/logistic nomogram with validation and clinical utility reporting.",
        "findings": "Routine clinical and RAI-related variables can stratify refractory risk, but definitions differ from NHRH.",
        "relevance": "Useful for related-work discussion and endpoint taxonomy.",
        "limitations": "Endpoint is refractory disease, not exactly our composite NHRH or rolling H1 relapse.",
        "module_link": "Related endpoint family",
    },
    {
        "id": "P06",
        "short_key": "Ma_Frontiers_TRAb_2025",
        "title": "Impact of serum TRAb level changes on the efficacy of 131I therapy in Graves' disease: a decision tree prediction model",
        "year": 2025,
        "journal": "Frontiers in Endocrinology",
        "doi": "10.3389/fendo.2025.1581353",
        "url": "https://www.frontiersin.org/journals/endocrinology/articles/10.3389/fendo.2025.1581353/full",
        "theme": "TRAb trajectory / RAI efficacy",
        "quartile_evidence": "Frontiers in Endocrinology; verify current local quartile manually",
        "reading_basis": "local PDF + full-text webpage",
        "local_file": "ref_paper/pdfs/2025_Ma_Frontiers_TRAb_change_decision_tree_1581353.pdf",
        "summary": "This study directly supports treating TRAb as a longitudinal biomarker rather than only as a baseline feature. It examines serum TRAb changes after RAI and uses a decision-tree model to relate antibody dynamics to treatment efficacy.",
        "methods": "Decision-tree prediction based on TRAb level changes and clinical outcome after RAI.",
        "findings": "TRAb change patterns carry prognostic information for post-RAI response.",
        "relevance": "Supports the biomarker-trajectory language used in our Module 2/3 discussion.",
        "limitations": "The model form and endpoint are not the same as our temporal landmark and H1 monitoring tasks.",
        "module_link": "M2/M3 biomarker trajectory interpretation",
    },
    {
        "id": "P07",
        "short_key": "Lu_Frontiers_ML_RAI_2025",
        "title": "From data to decision: an interpretable machine learning model for optimizing RAI therapy in Graves' hyperthyroidism",
        "year": 2025,
        "journal": "Frontiers in Endocrinology",
        "doi": "10.3389/fendo.2025.1711029",
        "url": "https://www.frontiersin.org/journals/endocrinology/articles/10.3389/fendo.2025.1711029/full",
        "theme": "Interpretable ML RAI benchmark",
        "quartile_evidence": "Frontiers in Endocrinology; verify current local quartile manually",
        "reading_basis": "local PDF + full-text webpage",
        "local_file": "ref_paper/pdfs/2025_Lu_Frontiers_interpretable_ML_RAI_1711029.pdf",
        "summary": "This is the closest machine-learning comparator. It uses multiple ML algorithms and interpretability tools for RAI therapy outcome prediction. It is useful as a benchmark, but the validation design and endpoint are different, so our report should compare cautiously rather than directly claim superiority.",
        "methods": "ML benchmark including tree models and interpretation; reports discrimination and clinical evaluation.",
        "findings": "Nonlinear models can perform well in internal/random validation settings for RAI outcome prediction.",
        "relevance": "Justifies our inclusion of nonlinear ML benchmarks and SHAP, while motivating temporal validation.",
        "limitations": "Likely less stringent temporal validation than our Module 1/2/3 design; endpoint definitions differ.",
        "module_link": "M1 advanced ML benchmark and discussion",
    },
    {
        "id": "P08",
        "short_key": "Duan_EC_EarlyHypothyroid_2022",
        "title": "Machine learning identifies baseline clinical features that predict early hypothyroidism in patients with Graves' disease after radioiodine therapy",
        "year": 2022,
        "journal": "Endocrine Connections",
        "doi": "10.1530/EC-22-0119",
        "url": "https://doi.org/10.1530/EC-22-0119",
        "theme": "RAI ML / opposite endpoint",
        "quartile_evidence": "Endocrine Connections; likely Q2 in endocrine contexts, verify manually",
        "reading_basis": "local PDF + metadata",
        "local_file": "ref_paper/pdfs/2022_Duan_EndocrineConnections_ML_early_hypothyroidism_EC-22-0119.pdf",
        "summary": "Although the endpoint is early hypothyroidism rather than persistent/recurrent hyperthyroidism, the paper is useful for machine-learning reporting in the same disease-treatment setting. It demonstrates how baseline clinical variables can be mapped to post-RAI outcomes with interpretable ML.",
        "methods": "Baseline-feature ML models for early hypothyroidism after RAI.",
        "findings": "Routine baseline features contain signal for early post-RAI thyroid state.",
        "relevance": "Methodological comparator for feature engineering, ML benchmark and interpretability.",
        "limitations": "Endpoint direction is opposite to NHRH; cannot be used as direct efficacy comparator.",
        "module_link": "M1 ML benchmark context",
    },
    {
        "id": "P09",
        "short_key": "Feng_IJE_PredictiveFactors_2024",
        "title": "Predictive Factors for the Efficacy of Radioactive Iodine Treatment of Graves' Disease",
        "year": 2024,
        "journal": "International Journal of Endocrinology",
        "doi": "10.1155/2024/7535093",
        "url": "https://doi.org/10.1155/2024/7535093",
        "theme": "RAI clinical predictors",
        "quartile_evidence": "International Journal of Endocrinology; verify current quartile manually",
        "reading_basis": "metadata / open access article",
        "local_file": "",
        "summary": "This study provides a clinical-predictor benchmark for RAI efficacy in Graves disease. It is useful because it highlights thyroid volume/weight and iodine dose per gram as predictors, which aligns with our OR/SHAP interpretation of thyroid burden and dose-density features.",
        "methods": "Clinical predictor analysis and nomogram-style evaluation for RAI efficacy.",
        "findings": "Thyroid burden and dose-density physiology are relevant to RAI response.",
        "relevance": "Supports our interpretation that thyroid weight and dose per thyroid tissue are clinically grounded predictors.",
        "limitations": "Endpoint and validation may differ; not a dynamic landmark framework.",
        "module_link": "M1/M2 variable rationale",
    },
    {
        "id": "P10",
        "short_key": "EANM_Guideline_RAI_2023",
        "title": "The EANM guideline on radioiodine therapy of benign thyroid disease",
        "year": 2023,
        "journal": "European Journal of Nuclear Medicine and Molecular Imaging",
        "doi": "10.1007/s00259-023-06274-5",
        "url": "https://doi.org/10.1007/s00259-023-06274-5",
        "theme": "Guideline / RAI clinical context",
        "quartile_evidence": "EJNMMI is a high-impact nuclear medicine journal; Q1/Q2, verify manually",
        "reading_basis": "local PDF + local HTML",
        "local_file": "ref_paper/pdfs/2023_EANM_guideline_radioiodine_benign_thyroid_s00259-023-06274-5.pdf",
        "summary": "The guideline is not a prediction-model paper, but it is essential clinical background. It frames RAI preparation, dosing considerations, follow-up and the clinical meaning of persistent or recurrent hyperthyroidism.",
        "methods": "Clinical guideline from nuclear medicine experts.",
        "findings": "RAI management depends on thyroid disease context, dosing, preparation and follow-up.",
        "relevance": "Supports clinical background and helps avoid overclaiming model use as treatment recommendation.",
        "limitations": "Guideline does not validate our prediction model.",
        "module_link": "Clinical context and limitations",
    },
    {
        "id": "P11",
        "short_key": "ATA_Hyperthyroidism_Guideline_2016",
        "title": "2016 American Thyroid Association Guidelines for Diagnosis and Management of Hyperthyroidism and Other Causes of Thyrotoxicosis",
        "year": 2016,
        "journal": "Thyroid",
        "doi": "10.1089/thy.2016.0229",
        "url": "https://pubmed.ncbi.nlm.nih.gov/27521067/",
        "theme": "Guideline / Graves management",
        "quartile_evidence": "Thyroid is a leading endocrine journal; Q1/Q2, verify manually",
        "reading_basis": "PubMed/guideline",
        "local_file": "",
        "summary": "The ATA guideline anchors the clinical management of hyperthyroidism and Graves disease. For our paper, its role is to constrain the claim: prediction models can support risk communication and follow-up intensity, but should not be written as automated treatment decision systems.",
        "methods": "Clinical guideline and evidence synthesis.",
        "findings": "Management of Graves disease requires integration of labs, clinical state, therapy choice and follow-up.",
        "relevance": "Supports our conservative clinical-use boundary.",
        "limitations": "Not a ML/prediction paper.",
        "module_link": "Clinical boundary and discussion",
    },
    {
        "id": "P12",
        "short_key": "GREAT_ExternalValidation_2017",
        "title": "External validation of the GREAT score to predict relapse risk in Graves' disease",
        "year": 2017,
        "journal": "European Journal of Endocrinology",
        "doi": "10.1530/EJE-16-0986",
        "url": "https://doi.org/10.1530/EJE-16-0986",
        "theme": "Risk score / Graves relapse",
        "quartile_evidence": "European Journal of Endocrinology is high-impact endocrine journal; Q1/Q2, verify manually",
        "reading_basis": "metadata + known external-validation article",
        "local_file": "",
        "summary": "Although the GREAT score concerns relapse risk in Graves disease rather than specifically post-RAI NHRH, it is valuable because it shows the tradition of interpretable clinical risk scoring and external validation in Graves disease.",
        "methods": "External validation of a clinical relapse risk score.",
        "findings": "Simple clinical risk scores can stratify relapse risk but require external validation.",
        "relevance": "Supports our emphasis on interpretable risk tiers and external-validation limitations.",
        "limitations": "Therapy setting differs from our RAI cohort.",
        "module_link": "Risk-score precedent",
    },
    {
        "id": "P13",
        "short_key": "RAI_Failure_MetaAnalysis_2021",
        "title": "Predictive factors of radioiodine therapy failure in Graves' Disease: A meta-analysis",
        "year": 2022,
        "journal": "The American Journal of Surgery",
        "doi": "10.1016/j.amjsurg.2021.03.068",
        "url": "https://www.sciencedirect.com/science/article/pii/S0002961021002294",
        "theme": "RAI failure meta-analysis",
        "quartile_evidence": "Clinical specialty journal; verify current quartile manually",
        "reading_basis": "title/abstract metadata",
        "local_file": "",
        "summary": "This meta-analysis summarizes predictors of RAI treatment failure in Graves disease. Its main value is supporting the recurrent finding that thyroid size/volume, uptake, FT4, ATD history and dose-related factors matter, while also showing heterogeneity across studies.",
        "methods": "Meta-analysis of clinical predictors for RAI failure.",
        "findings": "Several baseline severity and thyroid-burden variables recur across RAI failure studies.",
        "relevance": "Supports clinical plausibility of our core features and cautions against overinterpreting a single predictor.",
        "limitations": "Definitions and study designs are heterogeneous; it does not test our three-module framework.",
        "module_link": "Related work and feature plausibility",
        "abstract_override": "OpenAlex did not expose the abstract. The article is used here as a meta-analytic source on predictors of RAI therapy failure in Graves disease, mainly to support feature plausibility and between-study heterogeneity rather than a direct benchmark against our temporal framework.",
    },
    {
        "id": "P14",
        "short_key": "Putter_Landmarking2_2022",
        "title": "Landmarking 2.0: Bridging the gap between joint models and landmarking",
        "year": 2022,
        "journal": "Statistics in Medicine",
        "doi": "10.1002/sim.9336",
        "url": "https://pubmed.ncbi.nlm.nih.gov/35098578/",
        "theme": "Landmark dynamic prediction methodology",
        "quartile_evidence": "Statistics in Medicine is a leading biostatistics journal; Q1/Q2, verify manually",
        "reading_basis": "PubMed abstract + methods article",
        "local_file": "",
        "summary": "This paper is one of the methodological anchors for our dynamic landmark framing. It argues for landmarking as a practical bridge between full joint modeling and simpler dynamic prediction strategies, especially when longitudinal biomarkers are available.",
        "methods": "Statistical methodology for landmark dynamic prediction with longitudinal information.",
        "findings": "Landmarking can update risk predictions using time-dependent biomarker summaries without requiring a fully specified joint model.",
        "relevance": "Directly supports Module 2/3 as landmark updaters rather than separate unrelated classifiers.",
        "limitations": "Methodological paper; our implementation uses pragmatic ML/logistic approximations rather than the full formal framework.",
        "module_link": "M2/M3 methodology",
    },
    {
        "id": "P15",
        "short_key": "Devaux_BMC_Landmark_ML_2022",
        "title": "Individual dynamic predictions of clinical endpoint from large dimensional longitudinal biomarker history: a landmark approach",
        "year": 2022,
        "journal": "BMC Medical Research Methodology",
        "doi": "10.1186/s12874-022-01660-3",
        "url": "https://doi.org/10.1186/s12874-022-01660-3",
        "theme": "Landmark + longitudinal biomarker summaries",
        "quartile_evidence": "BMC Medical Research Methodology is a recognized methods journal; verify quartile manually",
        "reading_basis": "open access methods article",
        "local_file": "",
        "summary": "This paper is highly relevant because it explicitly combines landmark prediction with longitudinal biomarker histories and machine-learning-style summaries. It supports our use of current values, slopes, cumulative histories and trajectory summaries instead of treating each lab as a static covariate.",
        "methods": "Landmark approach for dynamic prediction using high-dimensional longitudinal biomarker summaries.",
        "findings": "Summaries of individual biomarker trajectories can be incorporated into dynamic prediction models.",
        "relevance": "Supports our trajectory momentum design and biomarker family ablation language.",
        "limitations": "Not specific to Graves disease or RAI.",
        "module_link": "M2/M3 biomarker trajectory methods",
    },
    {
        "id": "P16",
        "short_key": "Rizopoulos_DynamicPrediction_JointModels_2011",
        "title": "Dynamic predictions and prospective accuracy in joint models for longitudinal and time-to-event data",
        "year": 2011,
        "journal": "Biometrics",
        "doi": "10.1111/j.1541-0420.2010.01546.x",
        "url": "https://doi.org/10.1111/j.1541-0420.2010.01546.x",
        "theme": "Dynamic prediction / joint models",
        "quartile_evidence": "Biometrics is a leading statistics journal; Q1/Q2, verify manually",
        "reading_basis": "metadata/abstract",
        "local_file": "",
        "summary": "This paper provides a rigorous dynamic prediction reference from the joint-model literature. It is useful in our discussion for explaining why dynamic risk estimates should be evaluated prospectively and why longitudinal marker history can update risk.",
        "methods": "Joint modeling of longitudinal and time-to-event data with dynamic prediction and prospective accuracy.",
        "findings": "Risk prediction can be updated as new longitudinal measurements arrive, and accuracy should be assessed in a time-aware manner.",
        "relevance": "Supports the overall dynamic-risk concept, while our approach remains landmark-based rather than full joint modeling.",
        "limitations": "Our endpoints are binary/rolling landmark tasks rather than a fully censored survival joint model.",
        "module_link": "Dynamic prediction background",
    },
    {
        "id": "P17",
        "short_key": "vanHouwelingen_Landmarking_2007",
        "title": "Dynamic prediction by landmarking in event history analysis",
        "year": 2007,
        "journal": "Scandinavian Journal of Statistics",
        "doi": "10.1111/j.1467-9469.2006.00529.x",
        "url": "https://doi.org/10.1111/j.1467-9469.2006.00529.x",
        "theme": "Landmarking foundation",
        "quartile_evidence": "Scandinavian Journal of Statistics is a recognized statistics journal; verify quartile manually",
        "reading_basis": "metadata/abstract",
        "local_file": "",
        "summary": "This is a foundational landmarking reference. It helps justify using a sequence of clinically meaningful landmark times to update predictions based only on information available at those times.",
        "methods": "Landmark-based dynamic prediction in event-history analysis.",
        "findings": "Landmarking provides a transparent framework for dynamic prediction without using future information.",
        "relevance": "Supports our time-safe 0M/1M/3M/6M and rolling landmark structure.",
        "limitations": "Classical survival/event-history framing differs from our pragmatic binary and rolling classification outputs.",
        "module_link": "Landmark method foundation",
    },
    {
        "id": "P18",
        "short_key": "VanCalster_Calibration_2019",
        "title": "Calibration: the Achilles heel of predictive analytics",
        "year": 2019,
        "journal": "BMC Medicine",
        "doi": "10.1186/s12916-019-1466-7",
        "url": "https://doi.org/10.1186/s12916-019-1466-7",
        "theme": "Calibration methodology",
        "quartile_evidence": "BMC Medicine is high-impact; Q1/Q2, verify manually",
        "reading_basis": "open access review",
        "local_file": "",
        "summary": "This article is central for explaining why AUC alone is insufficient. It frames calibration as a core requirement when predicted probabilities are used for clinical communication or decision thresholds.",
        "methods": "Methodological review and commentary on calibration in predictive analytics.",
        "findings": "Discrimination can be good while probabilities are poorly calibrated; calibration curves, intercepts, slopes and Brier scores matter.",
        "relevance": "Supports our decision to report Brier/calibration in every module.",
        "limitations": "General methodology, not thyroid-specific.",
        "module_link": "M1/M2/M3 calibration reporting",
    },
    {
        "id": "P19",
        "short_key": "Vickers_DCA_2006",
        "title": "Decision curve analysis: a novel method for evaluating prediction models",
        "year": 2006,
        "journal": "Medical Decision Making",
        "doi": "10.1177/0272989X06295361",
        "url": "https://doi.org/10.1177/0272989X06295361",
        "theme": "Decision curve analysis",
        "quartile_evidence": "Medical Decision Making is a recognized decision-science journal; Q1/Q2, verify manually",
        "reading_basis": "methods article",
        "local_file": "",
        "summary": "This is the foundational DCA paper. It supports our clinical-utility framing: a prediction model should be judged by net benefit over clinically relevant threshold probabilities, not just by statistical discrimination.",
        "methods": "Introduces net benefit and decision curve analysis.",
        "findings": "DCA compares model use against treat-all and treat-none strategies across threshold probabilities.",
        "relevance": "Supports our DCA panels and the claim that clinical threshold utility is part of model assessment.",
        "limitations": "DCA requires clinically meaningful threshold ranges and does not prove causal treatment benefit.",
        "module_link": "M1/M2/M3 DCA",
    },
    {
        "id": "P20",
        "short_key": "Saito_PR_AUC_2015",
        "title": "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets",
        "year": 2015,
        "journal": "PLOS ONE",
        "doi": "10.1371/journal.pone.0118432",
        "url": "https://doi.org/10.1371/journal.pone.0118432",
        "theme": "Imbalanced evaluation / PR-AUC",
        "quartile_evidence": "PLOS ONE is broad peer-reviewed journal; verify current quartile manually",
        "reading_basis": "open access article",
        "local_file": "",
        "summary": "This paper justifies emphasizing PR-AUC in our low-event-rate H1 rolling monitoring task. ROC-AUC can look acceptable even when positive predictive value in the high-risk region is clinically limited.",
        "methods": "Methodological comparison of ROC and precision-recall plots under class imbalance.",
        "findings": "Precision-recall curves can be more informative than ROC curves for imbalanced classification.",
        "relevance": "Supports our Module 3 interpretation that PR-AUC and risk-tier enrichment are more clinically relevant than accuracy alone.",
        "limitations": "General classification methodology, not medical-specific.",
        "module_link": "M3 low-event-rate evaluation",
    },
    {
        "id": "P21",
        "short_key": "TRIPOD_AI_2024",
        "title": "TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods",
        "year": 2024,
        "journal": "BMJ",
        "doi": "10.1136/bmj-2023-078378",
        "url": "https://www.bmj.com/content/385/bmj-2023-078378",
        "theme": "Reporting guideline",
        "quartile_evidence": "BMJ is a top-tier medical journal",
        "reading_basis": "guideline/reporting statement",
        "local_file": "",
        "summary": "TRIPOD+AI is the reporting standard that underpins our insistence on transparent data sources, temporal validation, calibration, missing-data handling, and clear model-use boundaries.",
        "methods": "Consensus reporting guideline for regression and machine-learning prediction models.",
        "findings": "Clinical prediction studies should report data sources, candidate predictors, missing data, validation, performance, calibration and intended use clearly.",
        "relevance": "Supports the report structure and the claim-evidence map.",
        "limitations": "Guideline, not empirical evidence for our model.",
        "module_link": "Reporting standard",
    },
    {
        "id": "P22",
        "short_key": "PROBAST_AI_2025",
        "title": "PROBAST+AI: an updated quality, risk of bias, and applicability assessment tool for prediction models using regression or artificial intelligence methods",
        "year": 2025,
        "journal": "BMJ",
        "doi": "10.1136/bmj-2024-082505",
        "url": "https://www.bmj.com/content/388/bmj-2024-082505",
        "theme": "Risk of bias / applicability",
        "quartile_evidence": "BMJ is a top-tier medical journal",
        "reading_basis": "quality/risk-of-bias tool",
        "local_file": "",
        "summary": "PROBAST+AI is relevant because the main threats to our study are not just model accuracy but leakage, validation design, missing-data handling, calibration and applicability to other centers.",
        "methods": "Risk-of-bias and applicability assessment framework for prediction models.",
        "findings": "Prediction model studies should be judged by participants, predictors, outcome, analysis, validation and applicability.",
        "relevance": "Supports our bias-control language and limitations.",
        "limitations": "Does not validate our results; it guides reporting quality.",
        "module_link": "Bias and applicability checklist",
    },
    {
        "id": "P23",
        "short_key": "TRIPOD_2015",
        "title": "Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis (TRIPOD): the TRIPOD statement",
        "year": 2015,
        "journal": "Annals of Internal Medicine",
        "doi": "10.7326/M14-0697",
        "url": "https://doi.org/10.7326/M14-0697",
        "theme": "Prediction model reporting",
        "quartile_evidence": "Annals of Internal Medicine is top-tier",
        "reading_basis": "reporting guideline",
        "local_file": "",
        "summary": "The original TRIPOD statement remains useful for structuring prediction-model reports: participants, predictors, outcomes, sample size, model development, performance and interpretation should be transparent.",
        "methods": "Reporting guideline for multivariable prediction models.",
        "findings": "Transparent reporting is required for reproducibility and appraisal.",
        "relevance": "Supports our table/figure manifest and methods reporting.",
        "limitations": "Pre-AI guideline; supplemented by TRIPOD+AI.",
        "module_link": "Reporting standard",
    },
    {
        "id": "P24",
        "short_key": "PROBAST_2019",
        "title": "PROBAST: A Tool to Assess the Risk of Bias and Applicability of Prediction Model Studies",
        "year": 2019,
        "journal": "Annals of Internal Medicine",
        "doi": "10.7326/M18-1376",
        "url": "https://doi.org/10.7326/M18-1376",
        "theme": "Risk of bias / prediction models",
        "quartile_evidence": "Annals of Internal Medicine is top-tier",
        "reading_basis": "risk-of-bias tool",
        "local_file": "",
        "summary": "PROBAST provides the pre-AI risk-of-bias framework for prediction model studies. It is particularly relevant to our handling of temporal validation, missingness, participant selection and outcome definition.",
        "methods": "Structured tool for prediction-model bias and applicability.",
        "findings": "Bias can enter through participant selection, predictor measurement, outcome definition and analysis.",
        "relevance": "Supports our leakage, missing-data and temporal-test checks.",
        "limitations": "Original version not AI-specific; use with PROBAST+AI.",
        "module_link": "Risk-of-bias framing",
    },
    {
        "id": "P25",
        "short_key": "Riley_SampleSize_2020",
        "title": "Minimum sample size for developing a multivariable prediction model: PART II - binary and time-to-event outcomes",
        "year": 2020,
        "journal": "Statistics in Medicine",
        "doi": "10.1002/sim.7992",
        "url": "https://doi.org/10.1002/sim.7992",
        "theme": "Prediction model sample size",
        "quartile_evidence": "Statistics in Medicine is a leading biostatistics journal; Q1/Q2, verify manually",
        "reading_basis": "methods article",
        "local_file": "",
        "summary": "This paper is relevant for judging whether complex models are appropriate for the available event counts. It supports our conservative preference for calibrated LR when treatment-pre baseline information and event counts limit complex ML gains.",
        "methods": "Sample-size framework for developing prediction models with binary/time-to-event outcomes.",
        "findings": "Prediction model development requires attention to overfitting, shrinkage, event fraction and anticipated performance.",
        "relevance": "Supports our argument that not all tasks justify high-capacity nonlinear models.",
        "limitations": "Generic methods paper; exact calculations would need our final event counts and predictor degrees of freedom.",
        "module_link": "Model complexity and limitations",
    },
    {
        "id": "P26",
        "short_key": "Steyerberg_PerformancePrediction_2010",
        "title": "Assessing the performance of prediction models: a framework for traditional and novel measures",
        "year": 2010,
        "journal": "Epidemiology",
        "doi": "10.1097/EDE.0b013e3181c30fb2",
        "url": "https://doi.org/10.1097/EDE.0b013e3181c30fb2",
        "theme": "Prediction performance framework",
        "quartile_evidence": "Epidemiology is a high-quality methods journal; verify quartile manually",
        "reading_basis": "methods article",
        "local_file": "",
        "summary": "This paper supports a multi-metric view of prediction performance. It helps explain why discrimination, calibration, classification, reclassification and clinical usefulness answer different questions.",
        "methods": "Framework article for prediction-model performance assessment.",
        "findings": "No single metric is sufficient for clinical prediction models.",
        "relevance": "Supports our use of ROC-AUC, PR-AUC, Brier, calibration, DCA and tiers together.",
        "limitations": "General framework, not disease-specific.",
        "module_link": "Evaluation completeness",
    },
    {
        "id": "P27",
        "short_key": "Debray_IndividualParticipantPrediction_2017",
        "title": "Individual participant data meta-analysis for a binary outcome: one-stage or two-stage?",
        "year": 2017,
        "journal": "Statistics in Medicine",
        "doi": "10.1002/sim.7582",
        "url": "https://doi.org/10.1002/sim.7582",
        "theme": "Prediction validation / heterogeneity",
        "quartile_evidence": "Statistics in Medicine; Q1/Q2, verify manually",
        "reading_basis": "methods metadata",
        "local_file": "",
        "summary": "Although not directly used in our modeling, this paper is useful for discussing why external validation and between-center heterogeneity matter. Our temporal validation is stronger than random split but still not a multi-center validation.",
        "methods": "Methodological comparison of one-stage and two-stage IPD meta-analysis for binary outcomes.",
        "findings": "Heterogeneity and transportability matter when estimating and validating prediction effects.",
        "relevance": "Supports the limitation that our thresholds and calibration require external validation.",
        "limitations": "Not specific to landmark or thyroid disease.",
        "module_link": "External-validation limitation",
    },
    {
        "id": "P28",
        "short_key": "Rajkomar_NPJ_EHR_2018",
        "title": "Scalable and accurate deep learning with electronic health records",
        "year": 2018,
        "journal": "npj Digital Medicine",
        "doi": "10.1038/s41746-018-0029-1",
        "url": "https://doi.org/10.1038/s41746-018-0029-1",
        "theme": "Clinical ML with longitudinal EHR",
        "quartile_evidence": "npj Digital Medicine is a high-impact digital medicine journal; Q1/Q2, verify manually",
        "reading_basis": "open access clinical ML article",
        "local_file": "",
        "summary": "This article is a broad clinical-ML benchmark showing that longitudinal EHR data can support multiple prediction tasks. For our paper, it is useful as contrast: we choose interpretable landmark LR rather than high-capacity deep learning because sample size, event rate and clinical transparency differ.",
        "methods": "Large-scale EHR deep learning for multiple clinical prediction endpoints.",
        "findings": "Longitudinal EHR information can improve predictions, but model complexity requires careful validation and interpretation.",
        "relevance": "Supports the general idea of longitudinal medical data prediction while justifying our simpler model choice.",
        "limitations": "Different scale, data type and clinical setting.",
        "module_link": "Clinical ML context",
    },
    {
        "id": "P29",
        "short_key": "Miotto_DeepPatient_2016",
        "title": "Deep Patient: An Unsupervised Representation to Predict the Future of Patients from the Electronic Health Records",
        "year": 2016,
        "journal": "Scientific Reports",
        "doi": "10.1038/srep26094",
        "url": "https://doi.org/10.1038/srep26094",
        "theme": "Longitudinal EHR representation learning",
        "quartile_evidence": "Scientific Reports is broad peer-reviewed journal; verify current quartile manually",
        "reading_basis": "open access clinical ML article",
        "local_file": "",
        "summary": "Deep Patient is a classic EHR representation-learning paper. It helps frame why longitudinal histories can be transformed into patient-level risk representations, while also highlighting that our current project deliberately uses simpler, auditable landmark features.",
        "methods": "Unsupervised representation learning from EHR followed by predictive modeling.",
        "findings": "Temporal EHR histories can encode future disease risk.",
        "relevance": "Background for representation learning and why we avoid unreviewable black-box complexity in this dataset.",
        "limitations": "Not thyroid-specific and much larger EHR scale than our cohort.",
        "module_link": "Longitudinal ML background",
    },
    {
        "id": "P30",
        "short_key": "Keogh_CF_DynamicPrediction_2019",
        "title": "Dynamic prediction of survival in cystic fibrosis: a landmarking analysis using UK patient registry data",
        "year": 2019,
        "journal": "Epidemiology",
        "doi": "10.1097/EDE.0000000000000920",
        "url": "https://journals.lww.com/epidem/fulltext/2019/01000/dynamic_prediction_of_survival_in_cystic_fibrosis_.5.aspx",
        "theme": "Clinical dynamic prediction example",
        "quartile_evidence": "Epidemiology is a leading epidemiology journal; Q1/Q2, verify current local quartile manually",
        "reading_basis": "journal full text + repository metadata",
        "local_file": "",
        "summary": "This clinical landmarking example is useful because it shows dynamic prediction in a longitudinal registry setting outside endocrinology. It supports the idea that risk estimates should be updated when new measurements arrive rather than fixed at baseline.",
        "methods": "Landmarking analysis with longitudinal registry measurements for dynamic survival prediction.",
        "findings": "Repeated clinical measurements can update patient-specific risk over time.",
        "relevance": "External analogy for our M2/M3 landmark updating strategy.",
        "limitations": "Different disease, endpoint and survival model; used only as methodological analogy.",
        "module_link": "Dynamic prediction clinical example",
    },
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_supporting_outputs(table_rows: list[dict[str, Any]], search_rows: list[dict[str, Any]]) -> None:
    journals = sorted({str(row["Journal"]) for row in table_rows})
    q_rows = []
    for journal in journals:
        info = JOURNAL_EVIDENCE.get(
            journal,
            {
                "status": "manual check required",
                "evidence": "No local journal evidence mapping yet; verify exact JCR/CAS partition before final submission.",
                "url": "",
            },
        )
        q_rows.append(
            {
                "Journal": journal,
                "Q2PlusStatus": info["status"],
                "Evidence": info["evidence"],
                "EvidenceURL": info["url"],
            }
        )
    write_csv(TAB / "journal_quartile_evidence.csv", q_rows)
    write_csv(
        TAB / "open_access_sources.csv",
        [
            {
                "PaperID": row["PaperID"],
                "ShortKey": row["ShortKey"],
                "DOI": row["DOI"],
                "OpenAccessPDF": row.get("OpenAccessPDF", ""),
                "OpenAccessLanding": row.get("OpenAccessLanding", ""),
                "LocalFile": row.get("LocalFile", ""),
            }
            for row in table_rows
        ],
    )
    status_by_journal = {row["Journal"]: row["Q2PlusStatus"] for row in q_rows}

    md = [
        "# 30 篇文献汇总总表",
        "",
        "> Q2PlusStatus 来自 `tables/journal_quartile_evidence.csv`；严格投稿前建议用机构 JCR/CAS 再核一次。",
        "",
        "| PaperID | ShortKey | Year | Journal | Theme | Q2PlusStatus | DOI | ModuleLink | 关键发现 | 在本文中的可能用法 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in table_rows:
        paper_id = row["PaperID"]
        md.append(
            "| {PaperID} | {ShortKey} | {Year} | {Journal} | {Theme} | {Q2PlusStatus} | {DOI} | {ModuleLink} | {KeyFindingZH} | {PossibleUseZH} |".format(
                **row,
                Q2PlusStatus=status_by_journal.get(row["Journal"], "manual check required"),
                KeyFindingZH=KEY_FINDING_ZH.get(paper_id, row["KeyFinding"]),
                PossibleUseZH=POSSIBLE_USE_ZH.get(paper_id, row["UseInOurPaper"]),
            )
        )
    (OUT / "30_paper_summary_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    synthesis = """# 30 篇文献主题综合

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
"""
    (OUT / "thematic_synthesis.md").write_text(synthesis, encoding="utf-8")

    ok_count = sum(1 for row in search_rows if row["ResultStatus"] == "ok")
    summary_count = len(list(SUM.glob("P*.md")))
    audit = f"""# Literature pack completion audit

- Required papers: 30
- Summary files found: {summary_count}
- Master CSV rows: {len(table_rows)}
- OpenAlex metadata status: {ok_count}/30 ok
- Journal quartile evidence table: `tables/journal_quartile_evidence.csv`
- Open-access source links: `tables/open_access_sources.csv`
- Markdown master table: `30_paper_summary_table.md`
- Thematic synthesis: `thematic_synthesis.md`

## Remaining caution

Most journals are Q1/Q2 or top-tier by public web evidence / journal standing. For strict institutional “中科院/JCR 二区以上” wording, final submission should re-check the exact journal-year partition in the institution's subscribed JCR/CAS database. This pack marks the evidence source and keeps this limitation explicit.
"""
    (OUT / "completion_audit.md").write_text(audit, encoding="utf-8")


def main() -> None:
    for d in [OUT, SUM, TAB, SRC]:
        d.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {}
    table_rows: list[dict[str, Any]] = []
    screening_rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []

    for entry in PAPERS:
        oa = openalex_lookup(entry)
        metadata[entry["id"]] = oa
        abstract = reconstruct_abstract(oa.get("abstract_inverted_index") if isinstance(oa, dict) else None)
        if not abstract and entry.get("abstract_override"):
            abstract = entry["abstract_override"]
        oa_journal = ""
        oa_cites = ""
        oa_year = ""
        oa_doi = ""
        oa_pdf = ""
        oa_landing = ""
        if isinstance(oa, dict) and not oa.get("_fetch_error"):
            oa_journal = (((oa.get("primary_location") or {}).get("source") or {}).get("display_name") or "")
            oa_cites = oa.get("cited_by_count", "")
            oa_year = oa.get("publication_year", "")
            oa_doi = (oa.get("doi") or "").replace("https://doi.org/", "")
            best_oa = oa.get("best_oa_location") or {}
            oa_pdf = best_oa.get("pdf_url") or ""
            oa_landing = best_oa.get("landing_page_url") or ""
        search_rows.append(
            {
                "PaperID": entry["id"],
                "Query": entry.get("doi") or entry["title"],
                "Backend": "OpenAlex works API",
                "ResultStatus": "ok" if isinstance(oa, dict) and not oa.get("_fetch_error") else "needs_manual_check",
                "Error": oa.get("_fetch_error", "") if isinstance(oa, dict) else "",
            }
        )
        screening_rows.append(
            {
                "PaperID": entry["id"],
                "Decision": "include",
                "Reason": entry["theme"],
                "Q2PlusEvidence": entry["quartile_evidence"],
                "NeedsManualQuartileCheck": "yes" if "verify" in entry["quartile_evidence"].lower() or "manual" in entry["quartile_evidence"].lower() else "no",
            }
        )
        row = {
            "PaperID": entry["id"],
            "ShortKey": entry["short_key"],
            "Year": entry["year"],
            "Title": entry["title"],
            "Journal": entry["journal"],
            "OpenAlexJournal": oa_journal,
            "DOI": entry["doi"] or oa_doi,
            "URL": entry["url"],
            "Theme": entry["theme"],
            "QuartileEvidence": entry["quartile_evidence"],
            "ReadingBasis": entry["reading_basis"],
            "LocalFile": entry["local_file"],
            "ModuleLink": entry["module_link"],
            "KeyFinding": entry["findings"],
            "UseInOurPaper": entry["relevance"],
            "LimitationsForUse": entry["limitations"],
            "OpenAlexYear": oa_year,
            "OpenAlexCitedBy": oa_cites,
            "AbstractAvailable": "yes" if abstract else "no",
            "OpenAccessPDF": oa_pdf,
            "OpenAccessLanding": oa_landing,
            "关键发现": KEY_FINDING_ZH.get(entry["id"], entry["findings"]),
            "在本文中的可能用法": POSSIBLE_USE_ZH.get(entry["id"], entry["relevance"]),
        }
        table_rows.append(row)

        md = f"""# {entry['id']}. {entry['title']}

## Citation

- **Short key**: `{entry['short_key']}`
- **Year / journal**: {entry['year']}, *{entry['journal']}*
- **DOI**: {entry['doi'] or 'not confirmed'}
- **URL**: {entry['url']}
- **Theme**: {entry['theme']}
- **Q2+ evidence**: {entry['quartile_evidence']}
- **Reading basis**: {entry['reading_basis']}
- **Local file**: {entry['local_file'] or 'not downloaded yet'}

## One-paper independent summary

{entry['summary']}

## Methods readout

{entry['methods']}

## Main findings relevant to this project

{entry['findings']}

## How it supports the RAI three-module manuscript

{entry['relevance']}

## Caution / boundary for citation

{entry['limitations']}

## Module mapping

{entry['module_link']}

## Abstract / metadata notes

{abstract if abstract else 'OpenAlex abstract was unavailable or not recovered in this run; use DOI/URL/local full text for follow-up reading.'}
"""
        (SUM / f"{entry['id']}_{slugify(entry['short_key'])}.md").write_text(md, encoding="utf-8")

    (SRC / "openalex_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(TAB / "paper_summary_table.csv", table_rows)
    write_csv(TAB / "search_log.csv", search_rows)
    write_csv(TAB / "screening_decisions.csv", screening_rows)
    write_supporting_outputs(table_rows, search_rows)

    index = ["# 30-paper literature pack index", ""]
    for row in table_rows:
        index.append(f"- [{row['PaperID']} {row['ShortKey']}](summaries/{row['PaperID']}_{slugify(row['ShortKey'])}.md): {row['Theme']}")
    (OUT / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    print(f"Wrote {len(PAPERS)} paper summaries to {SUM}")
    print(f"Wrote master table to {TAB / 'paper_summary_table.csv'}")


if __name__ == "__main__":
    main()
