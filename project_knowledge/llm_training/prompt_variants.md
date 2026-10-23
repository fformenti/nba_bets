# E11 and E12: prompt variants on the 1B (2026-10-23)

*Each run changes one thing from E4 (`qlora/llama32_1b/v2`, seed 42). They were scored on all
2,949 validation games, Platt cross-fit, and compared with E4 game by game, using a paired bootstrap
with 5,000 resamples. The win rule comes from `seed_noise.md`: at least 0.002 better, with a 95%
interval that excludes zero.*

| Run | Change | Log-loss | vs E4 (95% CI) | Accuracy | AUC | s/step | Peak | Wall |
|---|---|---|---|---|---|---|---|---|
| `e4-llama32-1b` | — | 0.6284 | — | 64.2% | 0.694 | 2.21 | 7.9 GB | 31 min |
| `e7-llama32-1b-seed43` | seed 43 | 0.6280 | −0.0004 [−0.0019, +0.0011] | 64.4% | 0.694 | 2.25 | 7.9 GB | 32 min |
| `e12-llama32-1b-venue20` | venue win% line kept to 20 games | 0.6282 | −0.0002 [−0.0009, +0.0004] | 64.4% | 0.694 | 2.28 | 7.9 GB | 32 min |
| `e11-llama32-1b-compact` | alignment whitespace removed | 0.6285 | +0.0001 [−0.0013, +0.0015] | 64.2% | 0.694 | **1.48** | **5.8 GB** | **21 min** |

(This table's log-loss comes from a script that sorts rows by game_id before the Platt cross-fit,
so it differs from the CLI by about 0.0002.)

## E12, venue20: no effect
- The "Adj last-season venue win%" line appears in 39.5% of validation games, up from 0.7%. Even on
  those 1,165 games, E12 matches E4: −0.0001, with a 95% interval of [−0.0012, +0.0011].
- **The model gets nothing from the line.** E12's `p_home` correlates with E4's at 0.9987, higher
  than the seed repeat's 0.993. Both runs share seed 42 and the same data order, and 60% of their
  prompts are identical. For the rest, the extra line barely moves the prediction.
- **Likely reason:** last season's home or away record adds little once the prompt already has the
  current season's form, rest and travel. Its information is either already there or weak.
- **Not adopted:** it costs tokens and buys nothing. Its dataset `fformenti/nba-bets-venue20` and
  model `fformenti/e12-llama32-1b-venue20` can be deleted.

## E11, compact: same quality, a third faster
- **Not worse by the rule:** +0.0001, and the interval includes zero. Its `p_home` correlate with
  E4's at 0.994, about what a seed change gives.
- **Faster and lighter:** 1.48 s/step against 2.21 (−33%), matching the 34% token cut (425 → 282).
  Peak memory is 5.8 GB against 7.9, and the run takes 21 min instead of 31.
- **Adopted:** `qlora/llama32_1b/compact` (dataset `fformenti/nba-bets-compact`) is the working
  config from here on. New 1B experiments should `_include` `compact.yaml`, not `v2.yaml`.

## What it adds up to
Five models (1B, 3B, 8B, two 1B prompt variants) and XGBoost all land at 0.628–0.630. Neither
moving the existing information around nor adding one more view of it changes anything. The next
gain has to come from information the features don't hold yet, such as injuries and availability,
or from combining model families (N2).
