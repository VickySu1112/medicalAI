# M2 v2 · Synthesis & Pre-registered paper recommendation rule

This synthesis directory holds the cross-track leaderboard and the **pre-registered decision rule** for which M2 variant becomes the paper's primary M2 narrative.

## Pre-registered decision rule (frozen here, applied later)

Let `Δ_best_vs_base` = (best non-Base method's temporal pooled ROC-AUC) − (M2-Base's temporal pooled ROC-AUC), with episode-cluster paired bootstrap 95% CI.

Let `M2_B_yields_insight` = boolean: did any of B1/B2/B3's representation / attention visualizations reveal a clinically-actionable pattern that the linear M2-Base cannot show? (Specifically: t-SNE block clusters that separate prognostic subgroups; aux 6M head learning curve showing dynamic encoder utility; attention weights showing landmark-specific importance varying across risk groups.)

Let `calibration_stable` = boolean: best non-Base method's per-landmark calibration intercept ∈ [−0.2, 0.2] AND slope ∈ [0.8, 1.2].

| Outcome | Decision |
|:---|:---|
| `Δ_best_vs_base` CI crosses 0 AND `M2_B_yields_insight` = False | **Recommend M2-Base** as paper primary. ML alternatives reported as supplementary. "Parsimony preferred when complexity does not pay off." |
| `Δ_best_vs_base` CI > 0 AND `calibration_stable` = True AND `M2_B_yields_insight` = False | **Recommend best M2-A method + M2-Base anchor**. "ML SOTA confirmed; LR provides interpretable anchor." |
| `Δ_best_vs_base` CI crosses 0 AND `M2_B_yields_insight` = True | **Recommend best M2-B architecture + M2-Base anchor**. "Architectural innovation yields clinical insight not visible in linear models." |
| `Δ_best_vs_base` CI > 0 AND `calibration_stable` = True AND `M2_B_yields_insight` = True | **Recommend M2-A best + M2-B best + M2-Base** three-pronged paper. Full methodological + applied + innovative narrative. |
| `Δ_best_vs_base` CI > 0 AND `calibration_stable` = False | Demote to **M2-Base + caveat**: "non-linear methods discriminate marginally better but miscalibrate; LR retained for clinical use." |

## Leaderboard output

`tables/m2_v2_leaderboard.csv`: each method × per-landmark + pooled ROC / PR / Brier / calibration intercept / slope / cluster bootstrap CI.

`tables/m2_v2_recommendation.json`: applied decision rule outcome + recommended primary + secondary methods + rationale.

`figures/m2_v2_synthesis_dashboard.png`: 3-panel — (a) leaderboard horizontal bars with CI; (b) calibration scatter (intercept vs slope, per landmark, color by method); (c) recommendation flow diagram per decision rule.

## Reporting commitment

The recommendation is **data-driven** but the **rule is pre-registered here**. The synthesis report (`Module2v2_三轨综合与论文推荐.md`) will:

1. State this rule verbatim in §1.
2. Show the leaderboard outcomes in §2-§3.
3. Apply the rule in §4, naming the recommended primary.
4. Explicitly disclose all alternative outcomes that did NOT occur (counterfactual transparency).
5. Document any rule deviation with explicit justification (none expected).

This ensures reviewer 2 cannot accuse cherry-picking the "best" architecture post-hoc.
