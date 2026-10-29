# Should validation grow beyond three seasons? Not yet (2026-10-28)

*Question: training data looks sufficient, so would a bigger validation set give steadier
decisions and help training? Today's split is train ≤ 2020/21, validation 2021/22–2023/24
(2,949 games), and test from 2024/25 (`configs/train/_defaults.yaml`).*

## Paired gaps on validation (Platt for the LLMs, sigmoid for XGBoost)

Computed with `diagnostics.py::paired_log_loss_gap`: 5,000 resamples, `default_rng(0)`, and a negative
gap means the first model is better.

| Comparison | Games | Gap | 95% interval | Per-game SD |
|---|---|---|---|---|
| E11 − E4 | 2,949 | −0.0000 | [−0.0014, +0.0014] | 0.038 |
| E7 − E4 | 2,949 | −0.0003 | [−0.0018, +0.0012] | 0.043 |
| E17 − E11 | 2,938 | −0.0017 | [−0.0041, +0.0006] | 0.065 |
| E11 − XGBoost | 2,949 | +0.0018 | [−0.0019, +0.0057] | 0.108 |
| E17 − XGBoost | 2,938 | −0.0000 | [−0.0037, +0.0036] | 0.101 |

The E7 − E4 row reproduces `seed_noise.md`'s throwaway script ([−0.0019, +0.0011]).

## Findings
- **Validation doesn't steer training.** `load_best_model_at_end` is off and no config sets early
  stopping. The in-training eval is only logged. Validation does two jobs: it ranks experiments and
  it fits Platt. A model learns about recent seasons only by training on them, so a bigger
  validation set takes information *away* from training.
- **A bigger set narrows the interval only by √n.**
  - Two more seasons (2019/20 and 2020/21, about 1,550 games) shrink every interval above by about 19%.
  - E17's gap would still straddle zero.
  - Halving the interval would take four times the games.
- **Pairing already does most of the work.** One model's own log-loss carries a ±0.012 interval.
  The paired intervals are 3–8× tighter.
- **The seasons that would move are the most valuable training data.**
  - Train would end in 2018/19, seven seasons before the season being predicted.
  - Home teams won 58–59% of games in 2015–19, but 53–55% from 2019/20 on.
  - So the post-2019 seasons are the scarcest look at today's game.
- **"Enough training data" is still untested.** Eval loss flattens by step 200–300, which shows one
  pass over the data is enough. It doesn't show that more data wouldn't help. E8, the learning
  curve, is the test.

## Decision
Keep `val_seasons: 3`, and judge every comparison with `paired_log_loss_gap`. Revisit only if E8
shows the curve flat by 25–50% of train. At that point, moving to 5 seasons becomes a reasonable
one-change experiment, with the scoreboard rebuilt on the new split.

## E17, settled
E17 − E11 = −0.0017 [−0.0041, +0.0006]. The gap is below the 0.002 bar and its interval includes
zero, so E17 doesn't pass the win rule. It is "not worse", and it also ties XGBoost.
