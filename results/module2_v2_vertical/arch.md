# M2-B · Pre-registered 3 novel architectures (vertical innovations) — DEFERRED

> **Implementation status (2026-05-29)**: B1 / B2 / B3 PyTorch architectures
> are pre-registered here but **not yet implemented in this commit**. The
> first implementation attempt (`scripts/simple/module2_v2_vertical_b.py`,
> retained in git history) consistently produced no stdout despite a live
> process — likely a torch+pandas import-order interaction with the
> protected anaconda environment. Rather than ship half-working code, the
> M2-B track is deferred to a subsequent commit; the synthesis
> recommendation in `../module2_v2_synthesis/` therefore applies the
> pre-registered decision rule using only M2-Base + M2-A results and
> documents B1/B2/B3 as "future work". Decision rule outcomes will be
> revisited once B1/B2/B3 land.



**Shared infrastructure**: identical to M2-Base and M2-A — same stacking, same blocks, same CV, same cluster bootstrap.

## Three architectures (locked)

### B1. MDJN — Mechanism-Disentangled Joint Network

Five mechanism blocks → five independent sub-encoders → concat → fusion head:

```
Block A burden   → encoder_A (MLP 16→16) → repr_A (dim 16)
Block B exposure → encoder_B (MLP 2→8)   → repr_B (dim 8)
Block C dynamic  → encoder_C (MLP 2→16)  → repr_C (dim 16)
Block D momentum → encoder_D (MLP 2→8)   → repr_D (dim 8)
Block E time int → encoder_E (MLP 5→8)   → repr_E (dim 8)
                                ↓
[repr_A ⊕ repr_B ⊕ repr_C ⊕ repr_D ⊕ repr_E] → fusion_MLP (56→16→1) → P(24M NHRH)
```

Loss: BCE on 24M NHRH; Adam lr=1e-3; dropout=0.2; early stopping on inner-fold OOF AUC.

**Analyses**: per-block representation ablation (zero out repr_X, re-eval); t-SNE on each block representation; per-block contribution to final logit via integrated gradients.

### B2. Static-Dynamic Dual-Tower with Aux 6M Multi-Task

```
[Block A + Block B (static, ~10 features)] → static_tower (MLP 10→24) → static_repr (24)
[Block C + Block D over 4 landmarks (4×4 tensor)] → trajectory_tower (GRU hidden=24) → trajectory_repr (24)
                                              ↓
[static_repr ⊕ trajectory_repr] → main_head (48→16→1) → P(24M NHRH)
                                ↘
                                  aux_head (48→16→1) → P(Eval_6M = Hyper)
loss = 0.7 × BCE(24M) + 0.3 × BCE(6M)
```

**Analyses**: aux head learning curve (does trajectory_repr actually learn 6M signal?); ablation of aux loss (set α=1.0); static-only vs dynamic-only halves to confirm both contribute.

### B3. CLAN — Cross-Landmark Attention Network

```
For each episode, build 4 tokens (one per landmark):
  token_L = concat(Block_A, Block_B, Block_C@L, Block_D@L) + positional_embed(L)
  → tokens shape: (4 timesteps, feature_dim)

Multi-head self-attention (num_heads=2, embed_dim=16, dropout=0.1)
  → attended_tokens (4, 16) + attention_weights (4, 4)

mean-pool over attended tokens → final_repr (16) → MLP(16→8→1) → P(24M NHRH)
```

Loss: BCE on 24M NHRH; Adam lr=1e-3.

**Analyses**: extract attention_weights for every episode; aggregate by predicted risk group (low/mid/high tertile from M2-Base); plot mean attention heatmap per group → "which landmark decides which patients"; per-patient attention visualization for representative cases.

## Shared training/eval

- PyTorch (CPU-only; conda env protected)
- Adam optimizer + ReduceLROnPlateau; max 200 epochs; early stop patience=15 on inner-fold AUC
- Same `StratifiedGroupKFold(5)` and episode-cluster bootstrap CIs
- Calibration: per-landmark Platt on pooled outer-OOF predictions (same as M2-Base)

## Hyperparameters (locked, no architecture-search)

These are *pre-registered* — no post-hoc tuning to make architecture look better:

- B1: dropout=0.2, hidden multipliers as above
- B2: GRU hidden=24, α=0.7
- B3: num_heads=2, embed_dim=16, dropout=0.1

If validation AUC < 0.50 on inner fold, architecture flagged as failed but reported honestly.

## Pre-registered hypotheses

H_B1: per-block representations carry distinct, separable information (t-SNE clusters differ).
H_B2: aux 6M head trained jointly improves main 24M head vs static-only baseline (ΔAUC > 0, CI excludes 0).
H_B3: attention weights for high-risk patients concentrate on later landmarks (6M); for low-risk, weights more uniform.

Each hypothesis tested in §4 of the M2-B report. Failed hypotheses reported honestly.

This file locks M2-B's architectural choices. Substantial deviation requires a §10 disclosure.
