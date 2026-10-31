# E4–E6: does model size help? (v2 recipe, 1 epoch, full train split, Colab L4)

*Scored on all 2,949 validation games, Platt cross-fit. The command is in
`docs/plans/llm-finetune-handoff.md` §4.*

**v2 recipe:** LoRA r16 on q/k/v/o, 4-bit NF4, effective batch 32, lr 2e-4 cosine, loss on the
completion only, 1 epoch = 727 steps. Gradient checkpointing is off (see
`2026-10-20-e3-speed-benchmark.md`).

| Run | Model | Calibrated log-loss | Accuracy | AUC | s/step | Peak memory | Wall time |
|---|---|---|---|---|---|---|---|
| test_run2 (reference) | Llama-3.1-8B, LLM.int8, 2 epochs, old recipe | 0.628 | 64.0% | 0.694 | — | — | 9.7 h |
| `e4-llama32-1b` | Llama-3.2-1B | **0.629** | 64.2% | 0.695 | 2.21 | 7.9 GB | 31 min |
| `e5-llama32-3b` | Llama-3.2-3B | 0.630 | 63.8% | 0.690 | 6.19 | 9.95 GB | 86 min |
| `e6-llama31-8b` | Llama-3.1-8B | *skipped (see below)* | | | ~13.8 (E3) | 12.3 GB (E3) | ~3 h est. |

Floors: always picking home scores 0.688. The untuned 8B with 5 shots scores 0.659.
XGBoost on the same features scores 0.628.

## E4 findings
- **The 1B matches the old 8B run** within noise, in 1/19th of the wall time.
- **It's skewed toward home before calibration.** Raw, it picks home 71% of the time against a
  true rate of 55%, and calibrated log-loss is 0.629 against 0.632 raw. Platt brings the home
  share to 61% without changing the ranking.
- **Learning saturates early.** `eval/first_token_accuracy` peaked at step 200 of 727 (0.683)
  and eval loss was flat from about step 300. E8's learning curve should confirm it.
- **Margin:** the RMSE of the probability-weighted expected margin is 13.9 points; the greedy
  answer's is 14.3.
- **The memory estimate held:** about 0.7 GB of activations per row with checkpointing off,
  so 8 rows per micro-batch peaked at 7.9 GB.

## E5 findings
- **The 3B is no better than the 1B:** 0.630 against 0.629 calibrated log-loss, and 63.8%
  against 64.2% accuracy. It's 2.8× slower (86 min).
- **Same shape as the 1B:** raw it picks home 70%, eval loss is flat from about step 200, and
  the expected-margin RMSE is 13.9.
- **Memory:** 4×8 with checkpointing off peaked at 9.95 GB; the estimate was about 10 GB.

## Conclusion (2026-10-21): the features set the ceiling, not the model
- Four very different models land on the same validation log-loss: 1B 0.629, 3B 0.630,
  test_run2's 8B 0.628, and XGBoost on the same features 0.628. E0 had already found
  XGBoost's learning curve nearly flat.
- **E6 (the 8B on the v2 recipe) was skipped** by the user's decision. It would cost about
  3 h of L4 to confirm a pattern three points already show. Revisit it only if new features
  change the picture.
- **Llama-3.2-1B (`qlora/llama32_1b/v2`) is the working model.** Each experiment costs
  about 30 min of L4.
- **What follows:** gains have to come from new information in the prompt (venue20,
  injuries and availability) or from combining models. Tuning capacity (learning rate, LoRA
  rank, model family) is unlikely to move a ceiling that model size didn't.
