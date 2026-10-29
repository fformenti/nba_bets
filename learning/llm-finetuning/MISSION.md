# Mission: Fine-tuning LLMs for NBA game prediction

## Why
Felipe fine-tunes Llama with LoRA to predict NBA winners in this repo, starting from boilerplate settings. They want to understand what each part of the recipe does, so they can decide what to try next to improve AUC and accuracy rather than tweaking defaults blindly.

## Success looks like
- Can explain every knob in `configs/train_llm/qlora/_defaults.yaml` and say whether it affects cost, learning, or information.
- Can read a scoreboard plus the noise floor and say whether a run is a real win.
- Can choose the next experiment and justify it with evidence (their own runs plus primary sources).
- Stretch: get close to Polymarket on test (70.0%, 0.579 log-loss).

## Constraints
- Compute: Colab L4, about 21 min per 1B run. Experiments change one thing at a time, cheapest first.
- Learning comes before winning (from `plans/llm_finetune_handoff.md`).
- Polymarket prices are a benchmark only, never a model input.

## Out of scope
- Pretraining and full fine-tuning theory beyond what explains the LoRA recipe.
- The 2025/26 test-season gap, until injury data exists.
