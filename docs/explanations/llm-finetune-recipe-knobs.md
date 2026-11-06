# What the LLM fine-tune does, and which knobs can move AUC

## Context
The LLM fine-tune began as boilerplate code, run with mostly default settings. This page explains
what is implemented, sorts every config knob by the job it does, and ranks what is left to try for a
better AUC and accuracy. The numbers come from the scoreboard in
`docs/plans/llm-finetune-handoff.md` §2 and from `docs/findings/llm/`. Validation figures are after
Platt recalibration.

## What is implemented

1. **The task.** Each game becomes a text table of about 280 tokens: streaks, rest days, adjusted
   win% over the last 3–21 games, venue form and travel. The prompt ends with `Home margin =`, and
   Llama-3.2-1B, adapted with LoRA, is trained to complete it with the signed margin, e.g. `+3`.
   The working config is `configs/train_llm/qlora/llama32_1b/compact.yaml` (E11).
2. **Only the first token is scored.** `p_home` = P(`+`) / (P(`+`) + P(`-`)) at the first answer
   token. For game `22100230`, E11 gave `+` 0.622 and `-` 0.377, so `p_home` = 0.625. The home team
   won by 3. So the LLM is a binary classifier dressed as a text generator.
3. **Two settings make that work:**
   - `train_on_completion_only: true` computes the loss on the answer tokens only. Without it, the
     sign would be about 1 token in 280 and would get almost none of the gradient.
   - **Platt recalibration** fits a logistic curve on validation (cross-fit). Before it, the
     fine-tunes pick home about 70% of the time against a true rate of 55%. Platt removes that lean.
4. **AUC measures ranking, not calibration.** AUC asks whether, for a game the home team won and one
   it lost, the win gets the higher `p_home`. Platt is a monotonic rescaling: it keeps every game's
   rank, so AUC stays where it was. Platt improves log-loss and accuracy. To raise AUC, the model
   has to rank games better.

## Every knob does one of three jobs

| Job | Knobs in this repo | Can it raise AUC? |
|---|---|---|
| **Cost**: how much speed and memory the same training uses | 4-bit NF4 quantization, gradient checkpointing, micro-batch 8 × accumulation 4, bf16, fused AdamW | Barely. E3 showed these mostly change seconds per step and GB used. 4-bit adds a small rounding error to the frozen weights. A bf16 run of the 1B is untested. |
| **Learning**: how the weights move toward the answer | LoRA `r`/`alpha`/`target_modules`/dropout, learning rate, cosine schedule, warmup, epochs, effective batch 32, gradient clip, completion-only loss, seed, model size, the target format | Only when learning is the bottleneck, and here it isn't. 1B, 3B and 8B tie (E4, E5, test_run2). r=32 tied r=16. A second epoch bought nothing. lr 1e-4 and 4e-4 tie 2e-4 (E9). won/lost ties the margin (E16). |
| **Information**: what the model can know about the game | Which stats go in the prompt, how many games train, which games count (min-4 vs min-15), new data sources | Yes. It is the only job that can lift a ceiling all the models share. So far only the existing stats have been rearranged (compact, venue20), and those tied. |

"Defaults" undersells the current recipe. PEFT's defaults are `r=8`, `alpha=8`. The repo uses r16 /
α32, and it changed three settings because of measurements:
- the effective batch went from 8 to 32, because 78% of steps were hitting the gradient clip;
- the clip went from 0.3 to 1.0, because 0.3 clipped most steps;
- training went from two epochs to one, because the second added nothing.

## Why the evidence points at the inputs

Four very different models score between 0.628 and 0.630 log-loss on the same inputs: LLMs at 1B,
3B and 8B, and XGBoost, which works nothing like an LLM. When models this different agree, capacity
and the training method are not what limits them. The inputs are. The noise floor is ±0.0004 from a
seed change (E7), so these ties are real, not bad luck.

TabLLM, the main study on fine-tuning LLMs on table rows written out as text, saw the same pattern.
The LLM helps most when labeled examples are scarce, and with many rows gradient-boosted trees catch
up. This dataset has 23,000 training games.

## What else can be done, ranked

1. **Add new information.** Injuries and player availability are allowed as inputs, and no current
   feature covers them. A team missing its star can look identical to a full-strength team in every
   stat the prompt shows. This is the only lever the evidence points to. It is a data project, not a
   training change (`docs/findings/pipeline/2026-11-02-nba-data-sources.md` compares sources).
2. **Blend the LLM with XGBoost (N2, CPU only, free).** Averaging helps only when two models make
   *different* mistakes. Two LLM seeds correlate at 0.993 and averaging them gained nothing. E16's
   won/lost model correlates with E11 at 0.978, and averaging those two gained nothing either. Check
   the LLM–XGBoost correlation first.
3. **One cheap learning check: LoRA on all layers.** `target_modules` lists only the attention
   projections (`q/k/v/o`). *LoRA Without Regret* found that attention-only LoRA underperforms LoRA
   that includes the MLP layers, even when the attention-only version gets a higher rank to match
   the parameter count. The run takes 21 minutes on Colab. With every other learning knob tied,
   expect a tie here too, but it is the one setting where outside evidence goes against the current
   choice. The handoff deprioritizes it as E10.
4. **E8, the learning curve.** Train on 10%, 25% and 50% of the data. If 25% already scores like
   100%, more seasons of the *same* stats won't help, which strengthens the case for item 1.

## Benchmarks
- Always pick home: 54.9%, 0.688 log-loss, AUC 0.50.
- Base 8B, 5-shot: 61.2%, 0.659.
- Every fine-tune and XGBoost: about 64.5%, 0.628, AUC 0.695.
- Polymarket on *test*: 70.0%, 0.579. Test is scored once, in E15.

## Learn more
- Lesson and knob-by-knob cheat sheet: `learning/llm-finetuning/lessons/0001-three-kinds-of-knobs.html`,
  `learning/llm-finetuning/reference/recipe-knobs.html`.
- [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) (Thinking Machines, 2025): which
  layers to target, learning rate, α/r scaling.
- [TRL: LoRA Without Regret](https://huggingface.co/docs/trl/main/en/lora_without_regret): the same
  findings reproduced in the library this repo trains with.
- [TabLLM](https://proceedings.mlr.press/v206/hegselmann23a.html) (Hegselmann et al., AISTATS 2023):
  LLMs on serialized tables versus gradient-boosted trees.
- [QLoRA](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023): NF4 and double quantization.
- [PEFT LoraConfig](https://huggingface.co/docs/peft/package_reference/lora): field definitions and
  defaults.
- [scikit-learn: probability calibration](https://scikit-learn.org/stable/modules/calibration.html):
  Platt (sigmoid) scaling.
