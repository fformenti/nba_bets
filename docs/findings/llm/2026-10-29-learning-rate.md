# E9: is 2e-4 the right learning rate? (2026-10-29)

*E11 (`e11-llama32-1b-compact`, lr 2e-4) against `e9-llama32-1b-lr1e4` and `e9-llama32-1b-lr4e4`
(configs `qlora/llama32_1b/lr1e4.yaml` and `lr4e4.yaml`, each `compact.yaml` with only
`training.learning_rate` changed). Same Hub dataset (`nba-bets-compact`), same games, same seeds.
Scored on all 2,949 validation games, Platt cross-fit (`outputs/comparisons/e9_lr/`).*

| Run | LR | Calibrated log-loss | Accuracy | AUC | vs E11, paired (95% CI) |
|---|---|---|---|---|---|
| `e9-llama32-1b-lr1e4` | 1e-4 | 0.6301 | 64.6% | 0.692 | +0.0015 [−0.0005, +0.0036] |
| `e11-llama32-1b-compact` | 2e-4 | 0.6286 | 64.5% | 0.694 | — |
| `e9-llama32-1b-lr4e4` | 4e-4 | 0.6282 | 64.4% | 0.695 | −0.0004 [−0.0018, +0.0011] |

## Findings
- **Neither side wins, so 2e-4 stays.** Both intervals include zero and both gaps are under the
  0.002 bar. 4e-4's gap is the size of the seed noise (E7: 0.0004). 1e-4 leans worse but isn't
  significantly worse. The planned follow-up step (5e-5 or 8e-4) isn't worth a run.
- **The learning rate changes the route, not the destination.** All three finish at eval loss
  1.278–1.279 on the same 1,000-row sample (`evaluation.seed` 42 for all three, so the curves are
  comparable).

  | | 1e-4 | 2e-4 | 4e-4 |
  |---|---|---|---|
  | Train loss, steps 0–50 (format learning) | 2.70 | 2.34 | 2.08 |
  | Grad-norm clip rate, steps 0–50 / 50–200 | 100% / 40% | 80% / 13% | 80% / 13% |
  | Median grad norm, steps 400–727 | 0.62 | 0.39 | 0.27 |
  | Eval loss at step 727 | 1.2789 | 1.2787 | 1.2780 |

  - 1e-4 learns the answer format slowest and is still clipped on 40% of steps 50–200. Its grad norm
    stays highest to the end: it is slightly less settled, which fits its small lean worse.
  - 4e-4 learns the format fastest and wobbles once (first-token accuracy 0.627 at step 300, against
    0.67 either side) before recovering. Its clip rate matches 2e-4's, so it tested the step size,
    not the clip.
- **The predictions barely move.** `p_home` correlates with E11's at 0.994 for 4e-4 (the same as
  E4 vs E11) and 0.988 for 1e-4 (the same as the E7 seed repeat). The two E9 runs correlate at
  0.981 with each other.
- **Against XGBoost** (0.6268): 4e-4 +0.0015 [−0.0020, +0.0051], 1e-4 +0.0033 [−0.0009, +0.0078].
  Both ties.

## What it means
A 4× range of learning rate moves log-loss by about as much as a new seed. This is the fourth knob,
after model size (E4–E6), prompt format (E11, E12) and seed (E7), that lands on the same
0.628–0.630 ceiling. Optimization isn't the bottleneck; the information in the features is.
Capacity knobs (E10: LoRA rank, target modules) are even less likely to help.
