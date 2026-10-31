# E7: how much does a result move by chance? (2026-10-21)

*E4 (`e4-llama32-1b`, `training.seed: 42`) against E7 (`e7-llama32-1b-seed43`, config
`qlora/llama32_1b/seed43.yaml`). Nothing else changed. Both were scored on all 2,949
validation games, Platt cross-fit.*

| Run | Seed | Calibrated log-loss | Accuracy | AUC | Picked home (calibrated) |
|---|---|---|---|---|---|
| `e4-llama32-1b` | 42 | 0.6286 | 64.2% | 0.694 | 61% |
| `e7-llama32-1b-seed43` | 43 | 0.6283 | 64.3% | 0.694 | 60% |

## Findings
- **The seed barely matters.** Game by game, the two runs' `p_home` correlate at 0.993. They differ
  by 0.016 on average and pick the same winner in 96% of games.
- **The seed-to-seed gap is 0.0004 log-loss.** A paired bootstrap on the same games, with 5,000
  resamples, puts the gap's 95% interval at [−0.0019, +0.0011], with an SD of 0.0008.
- **Two other sources of noise are smaller than they look:**
  - **Calibration:** the Platt cross-fit splits games in two by row order. Re-sorting the rows moves
    log-loss by about 0.0002 (E4 scores 0.6284 or 0.6286).
  - **Game sampling:** one model's log-loss has an SD of 0.0064 across resamples of the 2,949 games.
    That spread doesn't matter here: every run is scored on the same games, so differences are
    paired and most of it cancels.
- **Averaging the two seeds gains nothing.** The average scores 0.628, the same as either run, which
  fits a 0.993 correlation. Seed ensembles aren't worth GPU time.
- **The W&B eval curves seemed to disagree, but it was a sampling artefact.** E7's in-training eval
  loss was 1.300 against E4's 1.279, and its first-token accuracy 0.658 against 0.676. The
  1,000-row eval sample was drawn with `training.seed`, so E7 was evaluated on different games.
  `finetune.py::load_eval_dataset` now draws it with `evaluation.seed` (42), so later seed repeats
  get comparable curves and earlier runs keep the same sample.

## The decision rule this sets
A change counts as a win over E4 only if both hold:
1. **Calibrated validation log-loss drops by at least 0.002.** That's about 2.5× the seed SD, and
   five times the observed seed gap.
2. **A paired bootstrap of the per-game log-loss difference has a 95% interval that excludes
   zero.** Run it against E4 on the same game_ids.

The second test is the stricter one when the two models differ a lot, as XGBoost and the LLM do:
their per-game errors are less alike than two seeds', so the interval comes out wider. "Not worse"
means the 95% interval includes zero and the point estimate is within 0.002. Use it for E11
(compact), which is a speed change.
