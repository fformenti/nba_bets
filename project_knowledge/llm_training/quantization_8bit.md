# 8-bit (LLM.int8) quantization: what it costs and why it warns

*Measured 2026-10-20 on a Colab L4 (E3). Code: `src/ml/llm/finetune.py::build_quant_config`.*

## What `quantization.mode: "8bit"` does
- bitsandbytes' LLM.int8 stores each frozen linear layer's weights in int8.
- On every forward pass, it multiplies most input columns with int8 weights. A few outlier
  columns, the ones with large activations, are multiplied separately in fp16.
- **The matmul kernels run only in fp16.** `bnb_8bit_compute_dtype` doesn't change that.

## The warning
```
bitsandbytes.autograd._functions - WARNING - MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization
```
- **Why it fires:**
  - We train with `bf16: true`, so activations reaching each 8-bit layer are bf16.
  - bitsandbytes casts them to fp16 before the int8 matmul and logs a warning on **every call**
    (`bitsandbytes/autograd/_functions.py`, `MatMul8bitLt.forward`). That's hundreds of
    lines per step.
- **It is harmless.**
  - The cast is how LLM.int8 always works under bf16 training. test_run2, the first full run
    (64.2% val / 66.9% test), trained this way.
  - bf16 → fp16 loses range, not precision: fp16 has more mantissa bits. Llama activations
    entering the attention and MLP projections stay well within fp16's ±65,504.
- **How it's silenced:** `build_quant_config`'s 8-bit branch attaches
  `_drop_int8_cast_notice`, a logging filter, to the `bitsandbytes.autograd._functions` logger.
  - The filter drops only messages starting with `MatMul8bitLt: inputs will be cast`. Every
    other bitsandbytes warning still shows.
  - Training and inspection both load through `build_base_model`, so both are covered.
  - `tests/test_llm_training_config.py::test_8bit_load_silences_only_the_fp16_cast_notice`
    checks it.
- **The alternative we didn't take:** run fully in fp16 (`compute_dtype: float16`,
  `fp16: true`), so no cast happens.
  - That's what `configs/train_llm/qlora/llama31_8b/baseline_t4.yaml` does. A T4 has no bf16
    support, so fp16 is the right choice there anyway.
  - On an L4/A100 it would trade bf16's range for fp16 loss scaling across the whole model,
    just to avoid a cast that costs nothing measurable.

## E3: 8-bit against the alternatives (Llama-3.1-8B, LoRA r16, batch 8×4, gradient checkpointing)

| Mode | s/step | samples/s | Peak GPU memory (reserved) |
|---|---|---|---|
| 4-bit NF4 | 18.4 | 1.73 | 8.2 GB |
| bf16, no quantization | 18.3 | 1.73 | 18.7 GB |
| **8-bit LLM.int8** | **19.4** | **1.64** | **18.6 GB** |

- On this card 8-bit is the slowest mode, by about 6%. A rerun (`e3-8bit-replay`) reproduced
  this.
- It saves almost no memory compared with bf16. An 8B's int8 weights are about 8.5 GB, but the
  activations, the outlier handling and the fp16 workspaces fill the rest.
- Losses after 50 steps were the same within noise for all three modes.

## When 8-bit is still the right choice
- **The model doesn't fit in bf16 but 4-bit hurts quality.** int8 approximates the weights more
  closely than NF4. If a larger model loses accuracy under 4-bit, 8-bit is the middle ground.
  Compare on validation before adopting it.
- **Reproducing test_run2.** `qlora/llama31_8b/baseline.yaml` pins 8-bit for that reason.
- **Otherwise prefer bf16 when it fits, and 4-bit when it doesn't.** On an L4 all three train at
  the same speed, so memory headroom decides. 4-bit's headroom is what allows turning
  gradient checkpointing off; see `e3_speed_benchmark.md`.
