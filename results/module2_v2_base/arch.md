# M2-Base · Pre-registered architecture

**Locked decisions** (decided before any temporal-test evaluation):

## Data
- Stacked long format: 1003 episodes × 4 landmarks = **4012 landmark-rows**
- `episode_id`, `landmark ∈ {0,1,3,6}`, `landmark_time_centered = landmark − 1.5`, `is_baseline_landmark`, `Y_24M_NHRH` (constant within episode)

## Five mechanism blocks
- **A** Baseline burden: Sex, ThyroidW, TRAb, TGAb, TPOAb, FT4_0M, TSH_0M, log1p_DiseaseDuration_Months_Aug
- **B** RAI exposure: Uptake24h, HalfLife
- **C** Current dynamic: TSH_current, FT4_current (landmark-conditional lookup; audit columns retained)
- **D** Momentum: TSH_velocity = (TSH_current − TSH_prev) / Δt, FT4_velocity same; L=0M → 0 + `velocity_observed=0`
- **E** Time interactions: landmark_time_centered, is_baseline_landmark, velocity_observed, landmark_time × {TSH_current, FT4_current, TSH_velocity, FT4_velocity}

## Model
- Single L2-logistic + Platt calibration; primary architecture
- LightGBM as non-linear benchmark (subprocess-isolated if needed)
- Original 4 independent LR retained as fallback for head-to-head comparison

## CV / Bootstrap / Calibration
- `StratifiedGroupKFold(n_splits=5)` by `episode_id`, stratified on `Y_24M_NHRH`
- Episode-level cluster bootstrap × 1000 for all ΔAUC / 95% CI
- Per-landmark Platt scalers fit on pooled outer-OOF (not per-fold)
- Nested CV: outer 5-fold evaluation, inner 3-fold hyperparam tuning
- L2 penalty grid: C ∈ {0.001, 0.01, 0.1, 1, 10}

## Nested ablation (fixed-sequence)
```
M2-Base.0 = A
M2-Base.1 = A + B
M2-Base.2 = A + B + C
M2-Base.3 = A + B + C + D
M2-Base.4 = A + B + C + D + E (full supermodel)
```
Sensitivity: Shapley 5-block decomposition over 5! = 120 orderings.

## Inference
- Coefficient CIs from refit unpenalized LR with **cluster-robust (sandwich) variance** clustered by episode
- L2 point estimates used for prediction only

## Hold-out discipline
- Temporal split locked globally: 802 dev / 201 temporal (episode-level)
- Temporal test evaluated **only once**, at end, after all decisions frozen here

## Missing data
- MICE m=10 on long format conditional on landmark + outcome
- Missingness indicators included as features
- Complete-case sensitivity analysis

## Competing events
- Death / second RAI / thyroidectomy before 24M: **excluded** from primary analysis
- Sensitivity: recode as event=0 and event=1

## Multiple testing
- Nested ablation: fixed-sequence hierarchical (no correction needed)
- Per-landmark p-values within each step: Benjamini-Hochberg FDR
- OR forest: report CIs only, no p-values

This file is the methodological pre-registration. Any deviation from this in the implementation must be flagged in the report's §10 Sensitivity section with justification.
