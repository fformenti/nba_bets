# E16: does the margin's magnitude help the sign? (2026-10-30)

*E11 (`e11-llama32-1b-compact`, completes `+7`) against `e16-llama32-1b-compact-classification`
(config `qlora/llama32_1b/compact_classification.yaml`, completes ` won` / ` lost`). Every prompt
line before the cue is byte-identical on all 27,054 games the two datasets share. E16 trains on 139
fewer games (23,108 vs 23,247) because the build changed between pushes. Scored on the 2,938
validation games both runs have, Platt cross-fit (`outputs/comparisons/e16_vs_e11/`).*

| Run | Target | Calibrated log-loss | Accuracy | AUC | vs E11, paired (95% CI) |
|---|---|---|---|---|---|
| `e11-llama32-1b-compact` | signed margin | 0.6288 | 64.5% | 0.694 | — |
| `e16-llama32-1b-compact-classification` | won / lost | 0.6300 | 64.1% | 0.692 | +0.0011 [−0.0017, +0.0039] |

## Findings
- **It's a tie, so E11 stays.** The interval includes zero and the gap is under the 0.002 bar, which
  makes E16 "not worse" under the win rule. Its best case, −0.0017, also falls short of a win, so
  rerunning E11 on identical games wouldn't change the decision. With a tie, regression is the better
  target: it gives the same probability and a margin estimate too.
- **The magnitude dominates the loss but doesn't move the sign.** E16's eval loss averages over two
  tokens, so its total is 0.3138 × 2 = 0.63 nats, nearly all of it the answer because EOS is easy.
  E11's averages over three, for a total of 1.2787 × 3 = 3.84. E11's calibrated log-loss puts its
  sign at about the same 0.63, which leaves about 3.2 nats, or 84% of its loss, on the magnitude.
  The worry was that this would dilute the gradient on the sign. Removing it changed nothing we can
  measure, in either direction.
- **The target changes the model more than a seed does, without improving it.** `p_home` correlates
  with E11's at 0.978 (a seed repeat: 0.993) and picks agree on 93% of games (a seed repeat: 96%).
  That difference is not useful information: averaging the two in logit space gives 0.6287,
  −0.0002 [−0.0016, +0.0012] against E11.
- **Raw outputs lean home less.** E16 picks home 67.6% of the time before Platt, against E11's 71.2%
  and a true rate of 55%. Platt fixes both, and gains 0.0008 here against 0.0032 on E11. The ranking
  doesn't improve, so this doesn't change the result.
- **Against XGBoost** (same games): +0.0029 [−0.0010, +0.0067], a tie.

## Two measurement lessons
- **W&B `first_token_accuracy` isn't comparable between runs on different datasets.** W&B shows E16
  at 64.9% and E11 at 67.2%, which looks like a loss. The 1,000-row eval sample depends on the
  dataset's length, so the two runs share only 628 rows. E11 alone scores 67.6% on its own sample and
  65.2% on E16's (greedy, from the inspection CSV). On all shared games the gap is −0.7 points
  [−1.6, +0.1]. Runs on the same dataset, like E9's, do share the sample. Across datasets, compare
  the inspection CSVs on shared games.
- **Greedy answers overstate the home lean in every run.** About 5.5% of games tie exactly between
  the two answers in 16-bit logits, and greedy resolves all of those to home, because the home token
  has the lower id in both modes (`+` 10 vs `-` 12, ` won` 2834 vs ` lost` 5675). This is why greedy
  picks home 2–3 points more often than `p_home > 0.5` (E16: 69.9% vs 67.6%; E11: 73.4% vs 71.2%).
  `p_home` is computed in fp32 and doesn't have this problem, so the scoreboard is unaffected.

## What it means
This is the fifth knob, after model size (E4–E6), prompt format (E11, E12), seed (E7) and learning
rate (E9), that lands on the same 0.628–0.630 ceiling. What the model is asked to say doesn't
matter either. The limit is the information in the features.
