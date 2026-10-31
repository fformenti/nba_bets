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
| `e3-nogc-b2x16` | gc off, 2×16 | **13.8** | 2.30 | 12.3 GB |
| `e3-nogc-b4x8` | gc off, 4×8 | **13.4** | 2.37 | 18.6 GB |

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
  - **Measured:** 2 rows peaked at 12.3 GB and 4 rows at 18.6 GB. That's about 3 GB per row on
    a ~6 GB base, close to the estimate.
- **Turning gradient checkpointing off saved 25–27%**: 18.4 → 13.8 s/step at 2×16 and 13.4 at
  4×8. That's the expected size, since it drops one recomputed forward pass per step. Losses
  after 50 steps match the other runs.
- **Epoch time** (727 steps on the full train split): 8B ≈ **2.8 h** with checkpointing off at
  2×16, against 3.7 h before.

## The double-forward fix, measured (Qwen3-1.7B, 4-bit, 8×4)
- `resume-drill-2` (before) against `e3-drill-fixed` (after), same config:
  - Training went from **8.8 to 6.7 s/step** (−24%).
  - A 200-row eval went from **27 to 12.5 s** (−54%).
  - The loss and `first_token_accuracy` curves match within noise.
- **Cause:** asking trl 1.16 for `return_outputs=True` re-ran a full-logits forward pass on
  every step. See `docs/plans/llm-finetune-handoff.md` §3.

## Decision for E6 (the 8B with `v2.yaml`)
- **The rule set in advance:** if 4-bit with gradient checkpointing off fits and is at least 15%
  faster, use it. Otherwise use bf16 with checkpointing at 8×4.
- **Adopted (2026-10-21): 4-bit, gradient checkpointing off, 2×16**, in
  `qlora/llama31_8b/v2.yaml`.
  - 4×8 was 3% faster but peaked at 18.6 of 22 GB. The benchmark used the 1,600 oldest train
    rows; the full split, and the eval batches during training, may run longer, so the extra
    10 GB of headroom is worth about 5 minutes an epoch.
  - The trade-off: 4-bit's NF4 weights carry a small quantization error that bf16 doesn't. bf16
    can't turn checkpointing off on 22 GB (its weights alone are 16 GB), so it would be about
    33% slower per epoch.
- `bench.yaml` pins the old reference (checkpointing on, 8×4), so every E3 run still
  reproduces.
- **Applied, unmeasured, to the smaller models:**
  - `llama32_1b/v2` turns checkpointing off at 8×4. At about a quarter of the 8B's activation
    memory, the estimate is around 8 GB.
  - `llama32_3b/v2` turns it off at 4×8, estimated around 10 GB.
  - E4 and E5 report their real peaks (`train/peak_gpu_reserved_gb`).
- The 8-bit details are in `2026-10-20-quantization-8bit.md`.
