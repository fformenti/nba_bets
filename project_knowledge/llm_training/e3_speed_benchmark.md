# E3: training speed on a Colab L4 (Llama-3.1-8B)

*Runs 2026-10-20, W&B project `felipeformenti/nba-bets`, runs `e3-*`.
Configs: `configs/train_llm/qlora/llama31_8b/bench*.yaml`.*

## Setup
- **Base:** `bench.yaml` is `v2.yaml` (LoRA r16 on q/k/v/o, 4-bit NF4, gradient checkpointing,
  batch 8×4 = 32) cut to 50 optimizer steps on the 1,600 oldest train rows. There's no eval and
  no checkpoint push.
- **Variants:** each one changes a single setting.
- **Speed measure:** s/step is the median interval between logged steps, read from W&B history.
  Peak memory is `torch.cuda.max_memory_reserved`, logged as `train/peak_gpu_reserved_gb`.
- **Sequence length:** prompts are about 426 tokens and completions 2, so a step is about
  32 × 430 ≈ 14k tokens.

## Results

| Run | Change from `bench` | s/step | samples/s | Peak reserved |
|---|---|---|---|---|
| `e3-4bit` | — (4-bit, gc, 8×4) | 18.4 | 1.73 | 8.2 GB |
| `e3-bf16` | no quantization | 18.3 | 1.73 | 18.7 GB |
| `e3-8bit` | LLM.int8 | 19.4 | 1.64 | 18.6 GB |
| `e3-b2x16` | micro-batch 2×16 | 18.3 | 1.74 | 6.6 GB |
| `e3-nogc` | gradient checkpointing off | — | — | OOM (22 GB) |
| `e3-nogc-b2x16` | gc off, 2×16 | *pending* | | |
| `e3-nogc-b4x8` | gc off, 4×8 | *pending* | | |

## What it means
- **Precision and micro-batch size don't change speed** on an L4: all within about 6%. The card
  is compute-bound on the frozen 8B's matmuls, which cost the same whether the weights are
  stored in 4, 8 or 16 bits.
  - The benchmark's goal, 1.5× the samples/s of 8-bit, isn't reachable through those settings.
- **What does change speed:**
  - **Fewer FLOPs per token.** Turning gradient checkpointing off removes one of the roughly
    three forward-pass-equivalents per step, which should save up to about 25%.
  - **Fewer tokens.** The compact prompts of E11/E12 would help.
  - **A smaller model.** See E4/E5.
  - **A faster GPU.**
- **Memory without gradient checkpointing.** The 8B keeps about 2.9 GB of activations per row
  (8 rows overflowed 22 GB).
  - At 2 rows that's about 6 GB on the 5.5 GB 4-bit base, which fits.
  - At 4 rows it's about 12 GB, which is borderline.
  - bf16's 16 GB base leaves no room, so only 4-bit can turn checkpointing off.
- **Epoch time** (727 steps on the full train split): 8B ≈ **3.7 h** at the current settings.

## The double-forward fix, measured (Qwen3-1.7B, 4-bit, 8×4)
- `resume-drill-2` (before) against `e3-drill-fixed` (after), same config:
  - Training went from **8.8 to 6.7 s/step** (−24%).
  - A 200-row eval went from **27 to 12.5 s** (−54%).
  - The loss and `first_token_accuracy` curves match within noise.
- **Cause:** asking trl 1.16 for `return_outputs=True` re-ran a full-logits forward pass on
  every step. See `plans/llm_finetune_handoff.md` §3.

## Decision rule for E6 (the 8B with `v2.yaml`)
- If a 4-bit run with gradient checkpointing off fits and is **at least 15% faster**, use it.
- Otherwise use **bf16 with gradient checkpointing at 8×4**. It's the same speed as 4-bit and
  has no quantization error.
- The 8-bit details are in `quantization_8bit.md`.
