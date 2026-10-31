# Finding and loading a specific LLM checkpoint after a training disconnect

## Context
A LoRA fine-tuning run (`test_run2`) got disconnected mid-training. The goal was
to run evaluation/prediction against a specific pushed checkpoint. Weights & Biases
(Forge Corewave) showed 32 "versions" of the model artifact, but the Hugging Face
model card for `fformenti/test_run2` showed no equivalent version list — this
explains why, and how `--revision` is used.

## How the pieces fit together

1. **`push_to_hub` is what pushes checkpoints.** `configs/train_llm/*.yaml` sets
   `hub.push_checkpoints: true`, which `src/ml/llm/finetune.py:300-304`
   (`build_sft_config`) maps onto `SFTConfig(push_to_hub=..., hub_strategy="every_save",
   hub_model_id=build_hub_model_id(config, run_name))`. `hub_model_id` resolves to
   `"{hub.user}/{run_name}"` i.e. `fformenti/test_run2` — one Hub repo per run.

2. **`hub_strategy="every_save"` (hardcoded, not config-driven) pushes a new commit
   to the repo's `main` branch every `save_steps`**, overwriting the adapter files
   at the repo root each time (per HF's official docs) — it does *not* create
   separate `checkpoint-N/` folders on the Hub. A final explicit push also happens
   at the very end of training (`finetune.py:463`,
   `trainer.model.push_to_hub(...)`) — this one never runs if training is
   interrupted, so `main` HEAD ends up being the last completed `every_save` push.

3. **`revision` (in `src/ml/llm/evaluate.py:45`,
   `PeftModel.from_pretrained(base_model, hub_model_id, revision=revision)`) is a
   standard git revision (branch/tag/commit SHA) on that one Hub repo.** It's the
   mechanism to pin evaluation/inspection to a specific pushed state. Since `main`
   HEAD already is the last checkpoint pushed before a disconnect, an explicit SHA
   is only needed to reach an *older* checkpoint than the most recent one.

4. **W&B's N artifact versions are unrelated to HF Hub commits.**
   `src/ml/llm/train.py:29-36` sets `WANDB_LOG_MODEL="checkpoint"`, which makes HF's
   `WandbCallback` log each saved checkpoint as a new W&B artifact version
   (`model-test_run2:v0..vN`) — a separate, parallel logging system with no link to
   HF Hub git history. HF doesn't show a flat "versions" list on the model card at
   all; its version history is the git commit log, under the "Files and versions"
   tab's history icon, or directly at
   `https://huggingface.co/fformenti/test_run2/commits/main`. The `Adapter (1353)`
   shown in HF's "Model tree" is an unrelated internal HF linking ID between the
   adapter repo and base model, not a checkpoint step.

5. **No production LLM prediction path exists yet.** `src/ml/prediction/pipeline.py`
   (used by `make daily-cycle` / `predict-upcoming`) has zero references to the LLM
   track. The LLM work on `feature/train_LLM` is evaluation/inspection-only; there
   is no `predict-llm` Makefile target. The two usable entry points are
   `src/cli/evaluate_llm.py` (scored accuracy against a labeled split) and
   `src/cli/inspect_llm.py` (raw per-example decoded outputs, no scoring) — both
   already accept `--revision`.

## What to do

1. Open `https://huggingface.co/fformenti/test_run2/commits/main` to see the commit
   log (each commit ≈ one checkpoint push). Grab the SHA of the checkpoint wanted,
   or skip this and use `main` HEAD (the default when `--revision` is omitted),
   which should already be the last checkpoint pushed before a disconnect.
2. Run one of:
   ```bash
   uv run python -m src.cli.evaluate_llm --run-name test_run2 --revision <sha>
   uv run python -m src.cli.inspect_llm --run-name test_run2 --revision <sha>
   ```
   (omit `--revision` entirely to use `main` HEAD).

## Verification
Confirm the intended commit was actually loaded by checking the logged
`hub_model_id`/accuracy output after running `evaluate_llm.py`/`inspect_llm.py`
(and, if MLflow tracking is enabled for the eval run, the `revision` param logged
at `evaluate.py:179`).
