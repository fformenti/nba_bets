# LLM fine-tuning: findings so far and the experiment plan (handoff, updated 2026-10-21)

Start a new session from this file. It summarises the review of the first fine-tune, the code
changes made so far, the results of experiments E0–E3, and what to run next. **E3 has one step
left: two runs with gradient checkpointing off at smaller micro-batches (§6). See §5 for exactly
what the user runs next.** Durable write-ups of finished findings live in
`project_knowledge/llm_training/`.

## 1. Context and goals

- **Task:** a LoRA fine-tune of a causal LM (TRL `SFTTrainer`) that reads a pre-game stats prompt
  and answers with the signed NBA home margin, e.g. `+15`. The prompt ends with `Home margin =`,
  and the completion has no leading space.
- **Metrics:** sign accuracy is primary and RMSE in points is secondary. Experiments are judged
  on **validation log-loss after Platt recalibration** (§4).
- **The user is learning fine-tuning.** The priority is learning, and beating Polymarket is a
  stretch goal. Explain the reasoning behind each choice, and run experiments one change at a
  time, cheapest first.
- **Hard rule:** Polymarket prices are **never** a model input; they are a benchmark only.
  Injury and availability data are allowed later.
- **Out of scope for now:** the 2025/26 test-season gap. That data hasn't been gathered yet.
- **Platform:** Colab Pro (L4, 24 GB) driven through the repo CLI. Budget ≤ $50/month, no local
  GPU. Hub user `fformenti`. W&B is used for tracking (entity `felipeformenti`, project
  `nba-bets`).
- **Data:** HF dataset `fformenti/nba-bets` with 23,247 train, 2,949 validation (three seasons)
  and 1,018 test games. The prompt config is `configs/llm_prompt/per_team.yaml`.

## 2. Code changes (all uncommitted on `feature/train_LLM`)

542 tests pass (`uv run --extra gpu pytest`). Nothing is committed yet; the user hasn't asked for
a commit.

### Session 3 (2026-10-21)
- **E3 results pulled from W&B** and written up in
  `project_knowledge/llm_training/e3_speed_benchmark.md` (summary in §6).
- **8-bit cast warning silenced.** In 8-bit mode, `build_quant_config` attaches a logging filter
  (`_drop_int8_cast_notice`) to `bitsandbytes.autograd._functions`. It drops only
  `MatMul8bitLt: inputs will be cast ... to float16`, which LLM.int8 logs on every matmul under
  bf16 training.
  - The cast itself is expected and harmless. The user wants to keep 8-bit for future models.
  - See `project_knowledge/llm_training/quantization_8bit.md`. A test in
    `tests/test_llm_training_config.py` checks that other bitsandbytes warnings still show.
- **E3 step 4 configs:** `bench_nogc_b2x16.yaml` and `bench_nogc_b4x8.yaml`.
- **Colab runner:**
  - Drive is no longer unmounted after each run, only on release or in the new "End of session"
    cell (§5).
  - The setup cell uninstalls torchao.
  - The user's duplicate config cell is removed.

### Session 2 (2026-10-20)
- **The GPU stack is pinned to trl 1.x.**
  - The `gpu` extra now requires `trl>=1.16,<2` and `peft>=0.21`.
  - `uv.lock` was refreshed to trl 1.16.0 and peft 0.21.2. trl needs `datasets>=4.7`, so uv also
    moved datasets 4.4→5.1, pyarrow 21→25 and mlflow 3.8→3.17. The lock now matches what Colab's
    unlocked `pip install .[gpu]` gets, and all tests pass on it.
  - `uv sync --extra gpu` works on the Mac, because bitsandbytes now ships a macOS arm64 wheel.
    That makes the trl tests runnable locally.
- **trl 0.x code removed** (`src/ml/llm/finetune.py`): the logits-based hit counter, the liger
  branch and the virtual-token slice. `first_completion_token_hits(is_top1, label_mask)` is the
  only counter left.
- **Fixed a double forward pass in training.** See the finding in §3. `compute_loss` now catches
  trl's fused output with a forward hook instead of asking for `return_outputs=True`.
  - `tests/test_llm_trainer_trl.py` trains a tiny random Llama with LoRA on CPU through the real
    trl. It asserts exactly one forward per micro-batch, and that `first_token_accuracy` is
    logged in train and in eval.
  - It's skipped when the gpu extra isn't installed.
- **Peak GPU memory is logged.** `peak_gpu_allocated_gb` and `peak_gpu_reserved_gb` go to the
  console, W&B and MLflow at the end of every training run (`finetune.py::peak_gpu_memory`).
- **Inspection does one prompt pass instead of two** (`src/ml/llm/diagnostics.py`).
  - `score_prompts` runs:
    1. `prompt_pass`;
    2. `sign_and_margin`'s two sign steps, on copies of the cache;
    3. `greedy_decode` on the original cache.
  - It replaces `model.generate` followed by a second prompt pass. Expect inspection to take
    roughly half as long.
  - It's checked against `model.generate` + the old two-pass `sign_and_margin` on a tiny random
    Llama: a left-padded batch, one row stopping early on EOS, and every step's logits compared.
    A deliberate off-by-one in decode positions fails that test.
- **`tokenization_check` follows trl 1.x:** the completion starts at the common prefix of the
  prompt and prompt+completion tokenizations.
- **E3 configs:** `configs/train_llm/qlora/llama31_8b/bench.yaml` plus
  `bench_{8bit,bf16,nogc,b2x16}.yaml`, which change one thing each.
- **Colab runner** (local `notebooks/colab_runner.ipynb`, gitignored): the copy-to-Drive step
  now creates `outputs/llm/eval/<RUN_NAME>` first. Before, it failed on runs that inspect nothing,
  such as bench runs.
- **macOS test gotcha:** lightgbm's OpenMP and torch's segfault together in a multithreaded
  forward pass. The two tests that run real forwards call `torch.set_num_threads(1)`. Colab is
  unaffected.

### Session 1
- **Resume works** (`src/ml/llm/finetune.py`):
  - `hub_strategy="checkpoint"` pushes `last-checkpoint/` (adapter, optimizer, scheduler, RNG
    and trainer state).
  - `find_resumable_checkpoint` downloads that folder. Rerunning with the same `--run-name`
    resumes.
- **W&B** (`src/ml/llm/train.py::_configure_wandb_env`):
  - `WANDB_RUN_ID=<run_name>` and `WANDB_RESUME=allow`, so a resumed run continues the same
    W&B run.
  - `log_model` and `watch` both default to `"false"`. test_run2 uploaded 13.8 GB of checkpoints
    to W&B.
- **New config knobs** (`src/ml/config/schema.py`):
  - `training.tf32`, `load_best_model_at_end`, `metric_for_best_model` and
    `early_stopping_patience`, with validators.
  - `data.train_sample_strategy: first|random`.
  - `wandb.log_model`.
- **Diagnostics** (`src/ml/llm/diagnostics.py`, `src/cli/inspect_llm.py`):
  - `p_home` is computed in fp32 from the last hidden state. With bf16, 4.8% of test_run2's
    validation games had exactly 0.5, and ties went greedily to "+".
  - Expected-value margin: E[margin] = Σ P(sign)·Σ P(m|sign)·m. This needs single-token
    magnitudes; Qwen3 tokenizes digit by digit, so its RMSE is skipped.
  - The summary adds log-loss, Brier, AUC, a 10-bin calibration table, the share predicted home
    and three RMSEs: greedy, expected, and predicting the mean.
  - Flags: `--base-only`, `--few-shot K`, and `--calibrate-with SPLIT` (cross-fitted Platt
    recalibration, CPU only, used together with `--summarize-only`).
- **Prompt variants** (`src/ml/llm/serialization.py`):
  - `configs/llm_prompt/per_team_compact.yaml` (`layout: compact`) cuts prompts from 425 to 282
    tokens (−34%).
  - `per_team_venue20.yaml` keeps the "Adj last-season venue win%" line until a team's 20th
    venue game; the line appears in 40% of rows instead of 0.8%.
  - Both need datasets built and pushed (`fformenti/nba-bets-compact`,
    `fformenti/nba-bets-venue20`) before E11/E12. **Not built yet.**
- **Configs** (`configs/train_llm/`):
  - `qlora/_defaults.yaml` is the v2 recipe: 4-bit NF4, r16/α32/dropout 0.05, 1 epoch, batch
    8×4 = 32, lr 2e-4, warmup 0.05, `adamw_torch_fused`, max_grad_norm 1.0, eval/save every
    100 steps, `max_eval_samples` 1000.
  - `qlora/llama31_8b/baseline.yaml` pins test_run2 exactly.
  - `qlora/<model>/v2.yaml` exists for `llama31_8b`, `llama32_1b`, `llama32_3b`, `qwen3_1p7b`
    and `qwen3_4b`.
  - Extra variants: `llama32_1b/compact.yaml`, `llama32_1b/venue20.yaml` and
    `qwen3_1p7b/resume_drill.yaml`.
- **Colab runner:** `notebooks/colab_runner.ipynb`. It is gitignored, and the user keeps a copy
  on Drive. Its config cell has `LLM_CONFIG`, `RUN_NAME`, `INSPECT_SPLITS`, `TRAIN`,
  `TRAIN_ARGS`, `INSPECT_ARGS` and `RELEASE_RUNTIME_WHEN_DONE`. The run cell uses a `run()`
  subprocess helper that:
  - fails loudly when a command fails;
  - kills the child process when the cell is stopped;
  - copies `outputs/llm/eval/<RUN_NAME>` to Drive, then calls `drive.flush_and_unmount()`.

## 3. Findings

### Every trl 1.x run so far paid for a second forward pass (found and fixed in session 2)
- **What happened:** our `compute_loss` override called
  `super().compute_loss(..., return_outputs=True)` to read the outputs for
  `first_token_accuracy`. trl 1.16 computes the loss with a *fused* LM head that never builds the
  `(batch, seq, vocab)` logits. Asked for outputs, it runs **a second, full-logits forward pass**
  on every train and eval step, with a FutureWarning. On the tiny test model this showed as 4
  forwards for 2 steps.
- **Why it matters:** with gradient checkpointing a step is about four forward passes' worth of
  compute, so one extra forward is roughly +25% time. On top of that come Llama's 128k-vocab
  logits: about 0.9 GB per micro-batch of 8.
- **Before-fix number:** resume-drill-2 (Qwen3-1.7B, 4-bit, 8×4) trained at **8.7 s per
  optimizer step**, or 3.7 samples/s. That's from W&B history, counting only intervals without
  an eval. Each eval of 200 rows plus checkpoint push added about 27 s.
  - At that pace one epoch (727 steps) is about 1.8 h for a 1.7B model, so the earlier size
    ladder estimates were optimistic.
- **After the fix (E3 step 0, `e3-drill-fixed`, same config):** **6.7 s per step**, 24% faster.
  An eval of 200 rows dropped from 27 s to 12.5 s, 54% faster, because eval no longer computes
  full logits twice. The loss and first-token-accuracy curves match resume-drill-2 to within
  noise, so the metric is unchanged.
  - One epoch (727 steps) of the 1.7B is now about 1.35 h of pure training.
- **Affected runs:** E2 (resume-drill-1/-2). test_run2 ran from a copied notebook trainer and was
  not affected.
- **Lesson:** when subclassing a library trainer, check what the hook point costs. "Return the
  outputs" was free in trl 0.x and a full extra pass in 1.x.

### Phase 2 diagnosis of test_run2 (Llama-3.1-8B, LLM.int8, 2 epochs, 9.7 h on L4)
- **Validation:** 64.2% accuracy and log-loss 0.630. **Test:** 66.9%. It ties XGBoost (64.8 /
  67.2) on the same features and trails Polymarket (70.0%, log-loss 0.579).
- **The data sets the ceiling, not the model.** The model and XGBoost hit the same ceiling with
  the same features. An XGBoost learning curve (E0) is nearly flat: 0.6330 with 10% of the data
  and 0.6276 with 100%. The best XGBoost depth is 2.
- **Training was healthy but wasteful:**
  - It converged within one epoch; the second epoch bought nothing.
  - 8-bit is slow.
  - With an effective batch of 8, 78% of steps hit the gradient clip.
  - Each evaluation took 1.1 h.
  - 35% of prompt tokens were whitespace.
- **It favours the home team about 9.5 points too often.** Most of the rest of the apparent
  lean came from bf16 ties.
  - Greedy decoding collapses the margin to ±5/±7. Greedy RMSE is 15.0, against 16.4 for
    predicting the mean.
- **E0, Platt recalibration of test_run2:**
  - Validation log-loss (cross-fit) goes from 0.6304 to 0.6282.
  - Test log-loss, fitted on validation, goes from 0.6101 to 0.6060.
  - The share of test games predicted home falls from 62.2% to 57.2% (actual 53.2%).

### E2: the resume drill passed (Qwen3-1.7B, 60 steps, 1,920 rows)
- The run was stopped at step 30 and the same cell rerun. Training continued from the Hub's
  `last-checkpoint/`, and W&B continued the same run.
- `first_token_accuracy` is logged.
- The model learned the answer format completely: sign mass 0.998.

### E1: untuned baselines (Llama-3.1-8B, 4-bit, all 2,949 validation games)
- **Zero-shot fails on the format.** P(bare `+`/`-`) is 6e-5. Because the prompt ends with `=`,
  the base model wants a space token (`" +"`/`" -"`) and writes `0.5 * …`.
  - Lesson: much of what fine-tuning teaches here is the answer format.
- **5-shot learns the format** (sign mass 0.987) but leans away: only 37% of answers are "+".
  The likely cause is recency bias, since the five shots in order are `+6, +11, +6, -7, -12`.
  This is untested. Platt recalibration removes the lean.
- **The fine-tune mostly sharpens what the base model already reads from the stats.** Game by
  game, `p_home` from 5-shot and from test_run2 correlate at 0.68.
- **Cost:** 5-shot took about 4.5 h (2,470-token prompts) and zero-shot about 30 min. Don't
  repeat 5-shot for other models unless one is a serious candidate.

## 4. Scoreboard (validation, Platt cross-fit, 2,949 games)

The command for any run is
`uv run python -m src.cli.inspect_llm --config <any llm config> --run-name <run> --split validation --summarize-only --calibrate-with validation`.
CSVs live in `outputs/llm/eval/<run>/inspection_validation.csv`. When the user downloads from
Drive, they land in `outputs/eval/` and have to be copied over.

| Run | Accuracy | Log-loss | AUC | Share picked home |
|---|---|---|---|---|
| Always pick home | 54.9% | 0.688 | 0.50 | 100% |
| base-8b-0shot | 55.4% | 0.684 | 0.561 | 83% |
| **base-8b-5shot (the floor)** | 61.2% | **0.659** | 0.640 | 65% |
| resume-drill-2 (Qwen3-1.7B, 60 steps) | 62.2% | 0.648 | 0.660 | 62% |
| **test_run2 (the bar to beat)** | 64.0% | **0.628** | 0.694 | 62% |
| XGBoost (E0, for reference) | — | 0.628 | — | — |

The decision rule: a fine-tune must beat **0.659** to show training does anything, and
**0.628** to beat test_run2. Report accuracy, AUC, calibration and RMSE alongside log-loss.

**Reading speed and memory from W&B** (Claude can do this locally through the API with the
`WANDB_API_KEY` in `~/.secrets/env`):
- The run summary has `train_runtime`, `train_samples_per_second`, `peak_gpu_allocated_gb` and
  `peak_gpu_reserved_gb`.
- `train_runtime` includes in-training evals. For pure step time, use the history's
  `_timestamp` deltas between `train/global_step` logs.

## 5. Before the next GPU run

### Closed in session 2
1. Inspection's second prompt pass is gone (§2).
2. HF access to `meta-llama/Llama-3.2-1B`, `-3B` and `Meta-Llama-3.1-8B` was verified by
   downloading each one's `config.json` with the project token.
3. Version drift is fixed: trl and peft are pinned to 1.x and the lock refreshed (§2).

### Still true
- **Colab gotchas:**
  - Choose an L4 runtime; the setup cell asserts that CUDA is available.
  - Colab Pro has no background execution, so set `RELEASE_RUNTIME_WHEN_DONE=True` for long
    runs.
  - Every run needs a new `RUN_NAME`, which names the Hub repo, the W&B run and the eval folder.
    Reusing a name resumes training and overwrites the CSVs.
  - `--base-only` runs don't push anything to the Hub or W&B.
  - Colab preinstalls **torchao 0.10**. peft ≥ 0.21 raises `ImportError: Found an incompatible
    version of torchao` whenever it wraps an *unquantized* model, as in bf16 runs. With
    bitsandbytes layers the bnb dispatcher matches first, so 4-bit and 8-bit never reach the
    check. The setup cell now runs `pip uninstall -y torchao`; nothing in the repo uses it. This
    first hit `e3-bf16` on 2026-10-20.
  - **Never unmount Drive mid-session.** The run cell used to end with
    `drive.flush_and_unmount()`, and the next run in the session couldn't remount, even with
    `force_remount=True`. The only way out was a new runtime and a full reinstall.
    - Now Drive stays mounted from the setup cell. The run cell flushes and unmounts only when
      `RELEASE_RUNTIME_WHEN_DONE=True`, right before releasing.
    - With release off, run the last cell, "End of session", before disconnecting, so the final
      copies reach Drive.
  - Colab's secret `HF_llm_training_token` must be the same token verified above. If it's a
    fine-grained token, it needs "Read access to contents of all public gated repos you can
    access".
- **Don't touch `src/ml/llm/make_dataset.ipynb`.** It has the user's own uncommitted edits.

### Actions for the user, in order
Steps 1–3 of the 2026-10-20 list ran: E3 steps 0–3 are done (§6). What's left:

1. **Sync the repo to Drive** (`MyDrive/nba_bets`), including `notebooks/colab_runner.ipynb`.
   Then **start a fresh runtime** (*Runtime → Disconnect and delete runtime*, then reconnect).
   - *Why:* the new notebook keeps Drive mounted between runs, and the new bench configs only
     reach Colab once Drive has them.
2. **Run E3 step 4 in one L4 session**, about 35 min.
   1. The config cell is already set to `LLM_CONFIG="qlora/llama31_8b/bench_nogc_b2x16"`,
      `RUN_NAME="e3-nogc-b2x16"`, `INSPECT_SPLITS=[]` and `RELEASE_RUNTIME_WHEN_DONE=False`.
      *Run all*.
   2. Then set `LLM_CONFIG="qlora/llama31_8b/bench_nogc_b4x8"` and `RUN_NAME="e3-nogc-b4x8"`,
      and rerun **only the config cell and the run cell**.
   3. A `CUDA out of memory` is a result (most likely for b4x8).
   4. Run the last cell ("End of session"), then disconnect and delete the runtime.
   - *Why:* gradient checkpointing off is the only setting left that should speed up the 8B
     (§6). These two runs show whether it fits at a smaller micro-batch.
3. **Tell Claude which of the two finished.** Claude pulls the numbers from W&B, sets
   `qlora/llama31_8b/v2.yaml` for E6 by the decision rule in §6, and updates
   `project_knowledge/llm_training/e3_speed_benchmark.md`.
4. **Delete the benchmark Hub repos:** on huggingface.co, open each `fformenti/e3-*` model repo
   (`e3-drill-fixed`, `e3-4bit`, `e3-8bit`, `e3-8bit-replay`, `e3-bf16`, `e3-b2x16`, plus the two
   step-4 runs and `e3-nogc` if it exists), then *Settings → Delete this model*.
   - *Why:* each holds a 50- or 60-step adapter that no experiment will use.
5. **Then E4 and E5** (§6), the same way with `INSPECT_SPLITS=["validation"]` and
   `RELEASE_RUNTIME_WHEN_DONE=True`.
   - Afterwards, download `MyDrive/nba_bets/outputs/llm/eval/<RUN_NAME>/` into the repo's
     `outputs/llm/eval/`, so Claude can add the runs to §4.

## 6. Experiment plan (cheapest first, one change each)

Decisions use full validation. Test is scored **once**, in E15.

### E3: speed benchmark (about 1.5 h total, one Colab session)
- **Goal:** find the fastest 8B setup that fits in 24 GB. Success means at least 1.5× the
  samples/s of 8-bit. Secondary goal: measure the double-pass fix.
- **Step 0, the fix (`e3-drill-fixed`):** rerun `qwen3_1p7b/resume_drill` unchanged and compare
  pure s/step against **8.7 s** before the fix.
- **Step 1, precision:** `bench` (4-bit NF4), `bench_8bit` (LLM.int8, as test_run2) and
  `bench_bf16` (no quantization; about 16 GB of weights).
  - Each run is 50 optimizer steps on the same 1,600 oldest train rows. There's no eval and no
    checkpoint push.
  - The default load dtype is `"auto"`, so bf16 really loads bf16 rather than fp32.
- **Step 2, gradient checkpointing:** `bench_nogc` turns it off, trading memory for about 25%
  less compute.
- **Step 3, micro-batch:** `bench_b2x16` uses 2×16 instead of 8×4, keeping an effective batch
  of 32.
- Steps 2 and 3 build on 4-bit. **If bf16 wins step 1**, change their `_include` to
  `bench_bf16.yaml` and rerun both.
- **Record:** `train_samples_per_second` and `train/peak_gpu_reserved_gb` from the W&B summary.
  An OOM disqualifies a variant.
- **Results** (L4; s/step is the median interval between logged steps; full write-up in
  `project_knowledge/llm_training/e3_speed_benchmark.md`):

  | Run | s/step | samples/s | Peak reserved | Notes |
  |---|---|---|---|---|
  | `e3-drill-fixed` (Qwen3-1.7B) | 6.7 | 3.9* | 2.7 GB | was 8.7–8.8 s/step before the fix |
  | `e3-4bit` (8B) | 18.4 | 1.73 | 8.2 GB | |
  | `e3-bf16` | 18.3 | 1.73 | 18.7 GB | |
  | `e3-8bit` | 19.4 | 1.64 | 18.6 GB | `e3-8bit-replay` reproduced it |
  | `e3-b2x16` | 18.3 | 1.74 | 6.6 GB | |
  | `e3-nogc` | — | — | OOM | 8 rows × ~2.9 GB of activations overflow 22 GB |

  \* The drill's samples/s includes its six evals.
  - **Precision and micro-batch don't change speed** (all within about 6%). The L4 is
    compute-bound on the 8B's matmuls. **The 1.5×-over-8-bit goal can't be reached through
    precision.** Final losses are equal within noise.
  - **8-bit is dominated:** it's the slowest and has bf16's memory. It's kept for future models
    (see `project_knowledge/llm_training/quantization_8bit.md`).
- **Step 4, gradient checkpointing off at smaller micro-batches** (pending): `bench_nogc_b2x16`
  (about 12 GB expected, fits) and `bench_nogc_b4x8` (about 17 GB, borderline). Only 4-bit
  leaves room for this.
- **Decision rule for E6:**
  - If a 4-bit run with gradient checkpointing off fits and is **at least 15% faster** than 18.4
    s/step, put 4-bit, gc off and that micro-batch into `qlora/llama31_8b/v2.yaml`.
  - Otherwise use **bf16, gc on, 8×4** (`quantization.mode: none`). It's the same speed as 4-bit
    with no quantization error, and fits at 18.7 GB.

### E4–E6: size ladder with the v2 config, 1 epoch, full train set
- **Runs:**
  - E4: Llama-3.2-1B (`qlora/llama32_1b/v2`).
  - E5: Llama-3.2-3B (`qlora/llama32_3b/v2`).
  - E6: Llama-3.1-8B (`qlora/llama31_8b/v2`, with E3's winner).
- **Durations:**
  - 8B ≈ **3.7 h** per epoch (727 steps × 18.3 s), plus evals. Less if step 4 wins.
  - The 1B and 3B aren't measured yet. E4 gives the 1B speed. Interpolating between the
    Qwen3-1.7B drill (6.7 s/step, about 1.35 h per epoch) and the 8B, a guess is about 1 h for
    the 1B and about 2 h for the 3B.
- **Success criteria:**
  - 8B-v2 at least matches test_run2 (0.628 calibrated / 64%) in at most a third of the time.
  - **Adopt 1B as the working model** if it's within 0.005 log-loss and 1 point of 8B. That
    makes E7–E12 nearly free.
- **After each run,** score validation with the recalibrated summary and add it to the
  scoreboard in §4. Watch `eval_loss` and `eval_first_token_accuracy` in W&B for the point
  where improvement flattens.
- **If 1B becomes the working model,** rerun E3's precision step on it before E7. Its bf16
  weights are only about 2.5 GB, so 4-bit's dequantization overhead probably buys nothing
  there. That needs a `qlora/llama32_1b/bench*.yaml` pair modelled on the 8B ones.

### E7: seed repeat
- Repeat the working model with `training.seed: 43`, one change only.
- The difference between the two seeds is the noise floor. **A later "win" must exceed twice
  that difference.**

### E8–E15 (later, all on the working model)
| # | Experiment | Success criterion |
|---|---|---|
| E8 | Learning curve: 10 / 25 / 50 / 100% of train with `train_sample_strategy: random` | Find the plateau → subset size for E9+ |
| E9 | LR 1e-4 / 2e-4 / 4e-4 | Change only if > 2× noise |
| E10 | LoRA r 16 vs 64; attention only vs all linear layers | Same rule |
| E11 | Compact prompt (`llama32_1b/compact.yaml`) | Not worse than noise → adopt (−34% tokens) |
| E12 | Venue line kept until the 20th venue game (`llama32_1b/venue20.yaml`) | Better log-loss on early-season games |
| E13 | Cross-family: Qwen3-1.7B / 4B | Compare; note digit tokenization (no expected-margin RMSE) |
| E14 | Gemma 4 / Phi-4-mini, **after a license check** | Same |
| E15 | Final: best config, full train, **score test once** | Report vs XGBoost 67.2% and Polymarket 69.9% / 0.579 |

- **E11/E12 need their datasets first.** Build and push them locally (CPU, a few minutes each).
  Claude can run these, but pushing to the Hub needs the user's go-ahead:
  `make build-llm-dataset LLM_CONFIG=qlora/llama32_1b/compact ARGS=--push`, and the same with
  `venue20`. If the working model isn't 1B, copy the two configs to that model's folder first.
- **Dropped:** recency weighting of training games. The sklearn backtest showed it hurt
  log-loss.

## 7. How to run an experiment on Colab
1. Sync the repo to Drive at `MyDrive/nba_bets`; `outputs/` is skipped.
2. Open `notebooks/colab_runner.ipynb` on an L4 runtime and set the config cell:
   - **To train and inspect:** `LLM_CONFIG="qlora/<model>/<variant>"`, a new `RUN_NAME`,
     `TRAIN=True`, `INSPECT_SPLITS=["validation"]`.
   - **To train only** (benchmarks): the same with `INSPECT_SPLITS=[]`.
   - **For an untuned baseline:** `TRAIN=False` and
     `INSPECT_ARGS="--base-only [--few-shot 5 --batch-size 8]"`.
3. Run all cells. To resume after a disconnect, rerun with the same `RUN_NAME`. Within one
   session, a further run only needs the config cell and the run cell.
4. Download `outputs/llm/eval/<RUN_NAME>/` from Drive into the repo, then summarize locally
   with the command in §4.
