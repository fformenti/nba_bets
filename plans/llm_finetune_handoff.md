# LLM fine-tuning handoff (updated 2026-10-23)

## 0. Start here

**Where things stand, in five lines:**
1. A LoRA fine-tune reads a pre-game stats prompt and answers with the signed home margin (`+7`). It is judged on validation log-loss after Platt recalibration.
2. **Every model lands at 0.628–0.630:** Llama-3.2-1B, Llama-3.2-3B, the first 8B run (test_run2) and XGBoost on the same features. The features set the ceiling, not model size.
3. **The working config is Llama-3.2-1B with the compact prompt** (`configs/train_llm/qlora/llama32_1b/compact.yaml`, E11). One run takes 21 min on a Colab L4. New 1B experiments `_include` it.
4. **The noise floor is known (E7):** a different seed moves log-loss by 0.0004. A change wins only if it beats E4 by ≥ 0.002 *and* a paired bootstrap excludes zero (§2).
5. **Prompt tweaks don't move it either (E11, E12, 2026-10-23).** **Next, Claude does N2**, a CPU-only blend of the LLM and XGBoost. Then the user decides between E8 (learning curve) and scoping new data (injuries).

**Read in this order:** this section, then §3 (next steps), §2 (scoreboard), then whatever §4 or `project_knowledge/llm_training/` the step needs. `plans/llm_findings_2026-10-23.md` summarizes the latest session (E7, E11, E12).

**How to work with this user:**
- **Explaining:** the user is learning fine-tuning. Explain the reasoning behind each choice, and give a recommendation rather than a menu.
- **Experiments:** change one thing per experiment, cheapest first. Judge every result against the noise floor E7 measured (§2).
- **Actions for the user:** when a step needs them (Colab runs, Hub deletions, syncing Drive), write the exact steps, the values to set and *why* the step matters.
- **End of session:** update this file. Write finished findings up durably in `project_knowledge/llm_training/` (one topic per file); the user asked for that folder.
- **Ask first:**
  - before pushing anything to the Hugging Face Hub;
  - before committing (nothing is committed unless the user asks);
  - before deleting the user's files.
- **Project rules** (`CLAUDE.md`):
  - `uv` only, never `pip` or `python`.
  - Secrets go through `src/config/secrets.py::require_env`, never at import.
  - CLI modules hold no logic.
  - Update `.claude/docs/PROJECT_STRUCTURE.md` when files under `src/` or `configs/` change.
- **Hard rules:**
  - Polymarket prices are never a model input; they are a benchmark only.
  - Don't touch `src/ml/llm/make_dataset.ipynb`; it holds the user's own uncommitted edits.
  - Test is scored **once**, in E15.

**Tools that work locally** (the gpu extra is installed: `uv sync --extra gpu`):
- **Score a run:**
  `uv run python -m src.cli.inspect_llm --config <any llm config> --run-name <run> --split validation --summarize-only --calibrate-with validation`.
  - It reads `outputs/runs/<run>/llm/inspection_validation.csv` and writes nothing.
- **Scoreboard without a GPU:** `make plot-compare NAME=<name> SERIES="<run> <run> ..."` scores each run's Platt series on the games they share (`outputs/comparisons/<name>/tables/scoreboard.csv`). Its numbers equal the second summary above. Layout: `docs/OUTPUTS.md`.
  - Read `log_loss`, `accuracy_p_home` and `auc_p_home` from the *second* summary, the one after Platt recalibration.
- **W&B API** (entity `felipeformenti`, project `nba-bets`):
  - Connect with `wandb.Api()`, after setting `WANDB_API_KEY` from `require_env("WANDB_API_KEY", "", secrets_file=GLOBAL_SECRETS_FILE)`.
  - Summary keys: `train_runtime`, `train_samples_per_second`, `train/peak_gpu_reserved_gb`.
  - For pure s/step, take the median `_timestamp` delta between `train/loss` rows. `train_runtime` includes evals.
- **HF API:** connect with `HfApi(token=require_env("HF_llm_training_token", ""))`. The Hub user is `fformenti`.
- **Compare two runs, paired:** align both CSVs on `game_id` and run `diagnostics.py::recalibrate` on each. Then bootstrap the per-game log-loss difference. `seed_noise.md` describes the method; the script was a throwaway, so N2 should add it as a library function.
- **Tests:** `MLFLOW_DISABLE_AGENT_HINT=1 uv run --extra gpu pytest -q`. 626 pass.

## 1. Goals and setup

- **Task:** the model reads a pre-game stats prompt that ends `Home margin =` and completes with the signed margin, with no leading space.
  - It's trained with TRL `SFTTrainer` and LoRA, with loss on the completion only.
  - Sign accuracy is primary; RMSE in points is secondary.
  - Experiments are compared on **validation log-loss after Platt recalibration**.
- **The user's goals:** learning comes first. Beating Polymarket is a stretch goal; on test, Polymarket scores 70.0% and 0.579.
  - Injury and availability data are allowed as future inputs.
  - The 2025/26 test-season gap is out of scope until that data exists.
- **Platform:**
  - Colab Pro with an L4 (22 GB usable), driven through the repo CLI by `colab_notebooks/colab_runner.ipynb`.
  - Budget ≤ $50/month, no local GPU.
  - Tracking is W&B. Adapters push to the Hub as `fformenti/<RUN_NAME>`.
- **Data:** HF dataset `fformenti/nba-bets`: 23,247 train, 2,949 validation (three seasons) and 1,018 test games.
  - Splits come from `src/ml/datasets/splits.py::build_splits`, the same function the XGBoost pipeline uses.
  - The prompt config is `configs/llm_prompt/per_team.yaml`, about 426 tokens per prompt.
- **The v2 recipe** (`configs/train_llm/qlora/_defaults.yaml` plus each model's `v2.yaml`):
  - 4-bit NF4, LoRA r16/α32/dropout 0.05 on q/k/v/o.
  - 1 epoch (727 steps), effective batch 32, lr 2e-4 cosine, warmup 0.05, `adamw_torch_fused`, max_grad_norm 1.0.
  - Eval every 100 steps on 1,000 validation rows.
  - **Gradient checkpointing off.** The micro-batch is 8×4 for the 1B, 4×8 for the 3B and 2×16 for the 8B.

## 2. Scoreboard (validation, Platt cross-fit, all 2,949 games)

| Run | Accuracy | Log-loss | AUC | Picked home | Notes |
|---|---|---|---|---|---|
| Always pick home | 54.9% | 0.688 | 0.50 | 100% | |
| base-8b-0shot | 55.4% | 0.684 | 0.561 | 83% | wrong answer format (E1) |
| **base-8b-5shot (the floor)** | 61.2% | **0.659** | 0.640 | 65% | beating it shows training does something |
| resume-drill-2 | 62.2% | 0.648 | 0.660 | 62% | Qwen3-1.7B, only 60 steps |
| **test_run2 (the old bar)** | 64.0% | **0.628** | 0.694 | 62% | 8B, LLM.int8, 2 epochs, 9.7 h |
| **e4-llama32-1b (the comparison base)** | 64.2% | **0.629** | 0.695 | 61% | 1 epoch, 31 min |
| e5-llama32-3b | 63.8% | 0.630 | 0.690 | 62% | 86 min |
| e7-llama32-1b-seed43 | 64.3% | 0.628 | 0.694 | 60% | E4 with seed 43 |
| e12-llama32-1b-venue20 | 64.4% | 0.628 | 0.694 | 61% | venue line in 40% of prompts; a tie, not adopted |
| **e11-llama32-1b-compact (working config)** | 64.5% | 0.629 | 0.694 | 61% | prompt −34% tokens; a tie; 21 min, 5.8 GB |
| XGBoost (E0) | — | 0.628 | — | — | same features; learning curve nearly flat |

- **Noise floor (E7):**
  - E4 and E7 differ by 0.0004 log-loss. The paired SD is 0.0008, and their `p_home` correlate at 0.993.
  - **A change wins only if** it beats E4's 0.6286 by ≥ 0.002 **and** a paired bootstrap of the per-game difference has a 95% interval excluding zero.
  - "Not worse" means the interval includes zero and the point estimate is within 0.002.
  - → `project_knowledge/llm_training/seed_noise.md`
- **Raw outputs lean home.** Before calibration the fine-tunes pick home 70–71% of the time against a true rate of 55%. Platt removes the lean without changing the ranking.
- Write-ups: `project_knowledge/llm_training/size_ladder.md` (E4–E6), `seed_noise.md` (E7), `prompt_variants.md` (E11, E12), `e3_speed_benchmark.md` and `quantization_8bit.md`.

## 3. Next steps, in priority order

*Done:*
- *N1, E7 (the seed repeat).*
- *N3, both datasets on the Hub (2026-10-23).*
- *N4, E12 venue20 (a tie, not adopted) and E11 compact (a tie, 33% faster, adopted).*

*See §2, `seed_noise.md` and `prompt_variants.md`. Before any Colab run, set a new `LLM_CONFIG` and `RUN_NAME`: reusing a `RUN_NAME` resumes that run and overwrites its CSVs.*

### N2. Blend the LLM and XGBoost (Claude, CPU only)
- **Question:** do the two models carry different information? If their validation `p_home` correlate well below 1, an average can beat both. It costs no GPU time.
- **Inputs:**
  - E4's per-game predictions: `outputs/runs/e4-llama32-1b/predictions/validation.csv` (shared schema: `game_id`, `p_home`, `y_true`, `calibration` none/platt). The raw inspection CSV is at `outputs/runs/e4-llama32-1b/llm/inspection_validation.csv`.
  - XGBoost's validation `p_home` per gameId, trained on the same train split. **These exist now (2026-10-23):** every `make train` writes them.
    - `outputs/runs/xgboost-20261023-165536/predictions/validation.csv`, series `xgboost` with `calibration` `none` and `sigmoid` (out-of-fold on validation). Same 2,949 gameIds as the LLM runs.
    - On those games: xgboost [sigmoid] log-loss 0.6268, accuracy 65.1%, AUC 0.698; e11 [platt] 0.6286; e4 [platt] 0.6286 (`outputs/comparisons/llm_vs_xgboost/`).
    - E0's 0.628 came from an ad-hoc script in session 1, and its predictions weren't saved.
    - `outputs/archive/2026-08-12_season_decay_tuning/backtest_predictions.csv` is a walk-forward backtest from August with different training windows. It covers only 984 of the 2,949 validation games, so it can't be used.
- **To do:**
  1. ~~Find out how the classifier pipeline trains on `build_splits`.~~
  2. ~~Add a library function that returns validation `p_home` by gameId.~~ Done: `src.ml.results.read_predictions(run_id, ["validation"])`.
  3. Report the correlation, each model's calibrated log-loss on the same games, and the log-loss of the average and of a logistic stack. Cross-fit the stack the way the Platt step does.
  4. Add the paired comparison as a library function, for example `diagnostics.py::paired_log_loss_gap`: align on `game_id`, recalibrate each model, then bootstrap. Later experiments reuse it. The throwaway scripts used for E7/E11/E12 sorted by `game_id`, recalibrated each run and bootstrapped 5,000 times with `default_rng(0)`.
- **LLM side:** use E11 (`e11-llama32-1b-compact`, the working config) and E4 as a cross-check. Their `p_home` correlate at 0.994.
- **Success:** the blend passes the §2 win rule against E4.
- **For contrast:** two LLM seeds correlate at 0.993, and averaging them gained nothing (E7). A blend helps only if XGBoost's correlation with the LLM is clearly lower.

### E16. Classification target (user, Colab, about 21 min; code ready 2026-10-27)
- **Question:** does the margin's magnitude token help as an auxiliary signal, or does it dilute the gradient on the sign? The prompt ends `Home team` and the model completes ` won` / ` lost`; every line before it is byte-identical to compact's.
- **Code:** `data.target_mode: regression | classification` (`src/ml/llm/target.py`). `p_home` is still the two-way softmax over the first completion token, so the scoreboard, Platt and `plot-compare` need no change. The mode is recorded in the W&B config and `run.json`. A config pointed at the other mode's dataset fails at load. Dataset names keep the modes apart: a classification `dataset_name` must end in `-classification` and a regression one must not (`LLMDataConfig` checks this when the config loads, so a push can't overwrite the other mode's dataset). Regression keeps its original names.
- **Steps:**
  1. Push the dataset (ask first): `make build-llm-dataset LLM_CONFIG=qlora/llama32_1b/compact_classification ARGS=--push` → `fformenti/nba-bets-compact-classification`.
  2. Colab: `LLM_CONFIG=qlora/llama32_1b/compact_classification`, `RUN_NAME=e16-llama32-1b-compact-classification`, `INSPECT_SPLITS=["validation"]`.
  3. Compare with E11 using `make plot-compare` and the paired bootstrap, under the §2 win rule.
- **Caveat, the game set has changed:** today's build has 23,108 / 2,938 / 1,008 games against the Hub's 23,247 / 2,949 / 1,018. All 160 missing games are absent from the current build, and every shared row is byte-identical to the Hub's. That points to the HEAD commit (`03b77a0`, neutral-court fix and filters). So E16 trains on 139 fewer games than E11; `plot-compare` scores both on the 2,938 shared validation games. For a strict one-change test, rebuild `nba-bets-compact` as well and rerun E11.
- **Don't compare** `eval_loss` or `mean_token_accuracy` across modes, because regression's loss includes the magnitude token. Read `first_token_accuracy` instead.

### E17. Early-season games, minimum games played 4 (user, Colab, about 25 min; config ready 2026-10-28)
- **Question:** does training on games 5–15 of each season help? At 15, "Adj Last-season win%" (hidden from 13 games on) never reached any split.
- **Checked first:** the season split puts every gated prompt line in all three splits at near-equal rates (last-season win% 11/11/12%, Playoffs 37/40/37%), so early-only and late-only lines can't pile up in one split.
- **Config:** `configs/train_llm/qlora/llama32_1b/compact_min4.yaml` → `configs/train/llm_features_min4.yaml`. Scoped to the LLM: `_defaults.yaml` stays at 15, so XGBoost, E11 and E16 keep their game set. Splits are 27,268 / 3,440 / 1,210 games (current build at 15: 23,108 / 2,938 / 1,008).
- **Steps:**
  1. Push the dataset (ask first): `make build-llm-dataset LLM_CONFIG=qlora/llama32_1b/compact_min4 ARGS=--push` → `fformenti/nba-bets-compact-min4`.
  2. Colab: `LLM_CONFIG=qlora/llama32_1b/compact_min4`, `RUN_NAME=e17-llama32-1b-compact-min4`, `INSPECT_SPLITS=["validation"]`.
  3. Compare with E11 on the validation games they share (`make plot-compare`, paired bootstrap). The new early-season games are harder, so the full 3,440-game log-loss is not comparable with §2. Score them separately as well.

### N5. E8, learning curve (user, Colab, about 15 min in total)
- **Why:** eval loss and first-token accuracy plateau by about step 200–300 of 727 (E4, E5, E11). This tests whether more data helps, which decides whether gathering more seasons is worth it.
- **Runs:** 10 / 25 / 50% of train with `data.train_sample_strategy: random` and `data.max_train_samples` set accordingly.
  - Create configs `llama32_1b/lc10.yaml` etc. that `_include` **`compact.yaml`**.
  - 100% is E11.
- **Success:** the size where log-loss stops improving within the noise.
- **Rank against the data project:** if the curve is already flat at 25%, more seasons won't help, and only new kinds of information will.

### Deprioritized, with reasons
- **E9 (learning rate 1e-4/4e-4) and E10 (LoRA r64, all-linear targets):** model size didn't move the ceiling, so capacity tweaks are unlikely to.
- **E13/E14 (Qwen3, Gemma, Phi):** the same argument. Qwen3 also tokenizes digit by digit, so it has no expected-margin RMSE.
- **E6 (the 8B on v2):** skipped by the user's decision (2026-10-21), about 3 h for a likely tie. Its config `qlora/llama31_8b/v2.yaml` is ready. Revisit only if new features change the picture.
- **1B in bf16 instead of 4-bit:** an optional quality check. It removes NF4 quantization error, and the 1B fits easily. If tried, it's one change: `quantization.mode: none`.

### The real lever, for the user to scope
**New information.** Every model hits the same ceiling with the same features. Injuries and availability (allowed) are the obvious candidate. It's a data project:
- find a source;
- add it to the ETL and the prompt config;
- build the dataset.

N1–N4 are done, and E12 showed that one more view of existing stats adds nothing. **Raise this with the user now**, alongside N2's result.

### E15, last
The best config, trained on the full train split. **Score test once.** Report it against XGBoost (67.2%) and Polymarket (69.9% / 0.579). Scoring test needs `SCORE_TEST=True` in `colab_notebooks/test_llm.ipynb`.

## 4. Findings, condensed

- **Prompt variants (E11/E12, 2026-10-23):**
  - **venue20:** keeping last-season venue win% in 40% of prompts changed nothing (−0.0002 [−0.0009, +0.0004]), even on the games that show it. Its `p_home` correlate with E4's at 0.9987. Not adopted.
  - **compact:** a tie (+0.0001 [−0.0013, +0.0015]) at 1.48 s/step against 2.21, and 5.8 GB against 7.9. Adopted.
  - → `project_knowledge/llm_training/prompt_variants.md`
- **Seed noise (E7, 2026-10-21):**
  - A new seed moves log-loss by 0.0004 (paired SD 0.0008). Picks agree in 96% of games, and averaging seeds gains nothing.
  - The in-training eval sample used to follow `training.seed`; it now follows `evaluation.seed`.
  - → `project_knowledge/llm_training/seed_noise.md`
- **Size ladder (E4/E5, 2026-10-21):** the 1B ties test_run2 in 1/19th of the time, and the 3B is no better. The features set the ceiling. → `project_knowledge/llm_training/size_ladder.md`
- **Speed on an L4 (E3, 2026-10-20):**
  - Precision (4-bit, 8-bit, bf16) and micro-batch size barely change the 8B's speed (18.3–19.4 s/step); the card is compute-bound.
  - Turning gradient checkpointing off is 25% faster (13.8 s/step at 2×16, 12.3 GB). It needs 4-bit's memory headroom.
  - 8-bit is the slowest mode and uses bf16's memory.
  - → `project_knowledge/llm_training/e3_speed_benchmark.md`, `quantization_8bit.md`
- **A double forward pass (fixed, 2026-10-20):**
  - Our `compute_loss` override asked trl 1.16 for `return_outputs=True`, which silently ran a second full-logits forward pass on every step.
  - Fixed with a forward hook. Training is 24% faster and eval 54% faster.
  - Lesson: when subclassing a library trainer, measure what the hook point costs.
- **test_run2 diagnosis:**
  - Validation 64.2% / 0.630 raw. Test 66.9%, tying XGBoost (67.2%).
  - It converged within one epoch; the second bought nothing.
  - With an effective batch of 8, 78% of steps hit the gradient clip.
  - 35% of prompt tokens were whitespace (E11 removes them).
  - Greedy decoding collapses margins to ±5/±7. Use the probability-weighted expected margin; its RMSE is 13.9 on the 1B.
- **E1, untuned 8B:**
  - Zero-shot fails on the format: the prompt ends in `=` and the base model wants a space.
  - 5-shot learns the format but leans away from home, probably recency bias from the shot order.
  - Fine-tuning mostly teaches the format and sharpens what the base model already reads: game by game, p_home from 5-shot and from test_run2 correlate at 0.68.
  - 5-shot costs 4.5 h; don't repeat it casually.
- **E2:** resume from the Hub's `last-checkpoint/` works, and W&B continues the same run.
- **Dropped:** recency weighting of training games, which hurt log-loss in the sklearn backtest.

## 5. Running on Colab

1. **Sync the repo to Drive** at `MyDrive/nba_bets`. The setup cell rsyncs it to local disk and skips `outputs/`.
2. **Open `colab_notebooks/colab_runner.ipynb` on an L4.** In the config cell:
   - **Train and inspect:** set `LLM_CONFIG`, a new `RUN_NAME`, `TRAIN=True` and `INSPECT_SPLITS=["validation"]`.
   - **Benchmark, train only:** the same with `INSPECT_SPLITS=[]`.
   - **Untuned baseline:** `TRAIN=False` with `INSPECT_ARGS="--base-only [--few-shot 5 --batch-size 8]"`.
3. **Run all.** For a second run in the same session, rerun only the config cell and the run cell.
4. **Download** `MyDrive/nba_bets/outputs/runs/<RUN_NAME>/` into the repo's `outputs/runs/<RUN_NAME>/`.
   - That's the only run location; code reads and writes nothing else. `make plot RUN=<RUN_NAME>` draws it.
   - **Drive still holds the old layout** (`outputs/llm/eval/`) until someone runs `uv run python -m src.cli.migrate_outputs_layout --outputs-dir <Drive mount>/nba_bets/outputs` once. Until then the notebooks' reuse cell finds nothing and rescores.
   - Keep the folder name equal to `RUN_NAME`.
5. **To score adapters already on the Hub without training,** use `colab_notebooks/test_llm.ipynb`. The user wrote it.
   - It reuses CSVs already on Drive and prints the Platt-calibrated scoreboard.
   - It has a playground cell. Scoring test is guarded by `SCORE_TEST`.

**Gotchas:**
- **`RUN_NAME`:** it names the Hub repo, the W&B run and the eval folder. Reusing a name *resumes* training and overwrites its CSVs.
- **Releasing the runtime:** Colab Pro has no background execution. Use `RELEASE_RUNTIME_WHEN_DONE=True` for single runs; the call is `google.colab.runtime.unassign()`.
- **Drive:** never unmount it mid-session; a second run in the same session couldn't remount it. The run cell flushes and unmounts only when releasing. With release off, run the "End of session" cell before disconnecting.
- **torchao:** Colab preinstalls torchao 0.10, and peft ≥ 0.21 raises on it when wrapping an unquantized (bf16) model. The setup cell uninstalls it.
- **`.gdoc` files:** Drive's Google Docs shortcuts can't be rsynced; the setup cell excludes `*.gdoc`.
- **HF token:** the Colab secret `HF_llm_training_token` needs read access to the gated Meta Llama repos. That access is already granted for 3.2-1B, 3.2-3B and 3.1-8B.

## 6. Artifact inventory (what to keep where)

| Where | Keep | Delete / done |
|---|---|---|
| **HF models** | `test_run2` (the reference), `e4-llama32-1b` (the comparison base), `e11-llama32-1b-compact` (working config) | deleted: all `e3-*`. **To delete:** `e5-llama32-3b`, `e7-llama32-1b-seed43` (its job was the noise number), `e12-llama32-1b-venue20` (not adopted), `resume-drill-1`, `resume-drill-2`, `test_run`; and `nba-bets-2026-07-16_18.03.32` / `nba-bets-2026-07-23_11.34.23` if the user confirms they're the July notebook experiments |
| **HF datasets** | `nba-bets`, `nba-bets-compact` (working config); `games` until the user says it's obsolete | optional: `nba-bets-venue20` (rebuildable with one command) |
| **W&B** | everything: small, free, and cited by the write-ups | optional: `e3-nogc` (OOM, empty) |
| **Drive `outputs/llm/eval/`** (old layout; migrate to `outputs/runs/`, see §5) | `base-8b-0shot`, `base-8b-5shot`, `test_run2`, `resume-drill-2`, `e4-llama32-1b`, `e5-llama32-3b`, `e7-llama32-1b-seed43`, `e11-llama32-1b-compact`, `e12-llama32-1b-venue20` | the `e3-*` folders (each holds only a session `mlflow.db`) |
| **Repo `outputs/runs/`** | the same nine runs, migrated 2026-10-23 from `outputs/llm/eval/` (docs/OUTPUTS.md) | — |

Each run folder's `tracking/mlflow.db` is a copy of that Colab session's whole MLflow database. It isn't needed for scoring.

## 7. Code change log (uncommitted unless the user has committed since)

**Target modes (2026-10-27):**
- `src/ml/llm/target.py`: `TargetSpec` (cue, home/away first token, `has_magnitude`, `completion`, `parse`) and `SPECS` for regression and classification. `LLMDataConfig.target_mode` selects one; the default `regression` leaves every existing config and dataset unchanged. A validator requires classification dataset names to end in `-classification` and forbids it for regression.
- The cue is threaded through `serialize_frame` → `_as_per_team`. `dataset.py` writes `spec.completion` (`_format_completion` is gone; the regression spec reproduces it, which a test pins). `load_hf_split` refuses a dataset built in the other mode.
- `diagnostics.py`: `label_token_ids` replaces `sign_token_ids`; `p_home` is read at index 0 (the home answer). `_home_won` reads `margin` when present. The summary keys `raw_starts_with_*` became `raw_answered_{home,away,other}`. Classification reports no RMSE.
- The greedy tester parses via `spec.parse`; a won/lost answer reports no `pred_margin`.
- Tests: `tests/test_llm_target.py` (it uses the cached Llama tokenizer when present), plus additions to the split-parity, per_team, diagnostics and trl tests.

**Outputs restructure (2026-10-23):**
- `outputs/` is now `runs/<run_id>/`, `comparisons/`, `eda/`, `checkpoints/`, `archive/`; every path lives in `src/config/paths.py`. Full layout and schema: `docs/OUTPUTS.md`.
- Training and evaluation write result files only (`predictions/<split>.csv`, `metrics/<split>.json`, `run.json`); figures come from `make plot` / `make plot-compare` (`src/ml/plotting/`).
- `inspect_llm` writes the shared predictions (raw + Platt) beside the inspection CSV; LLM checkpoints moved to `outputs/checkpoints/<run>/` (`paths.output_dir` removed from every LLM config).
- Both Colab notebooks read and write `outputs/runs/`; the scoreboard cell calls `build_comparison`.
- The nine eval folders were migrated (`make migrate-outputs`); their converted metrics reproduce §2's accuracy and log-loss exactly. §2's AUC column mixes raw and Platt-cross-fitted AUC, so it differs from `metrics/validation.json` by ≤ 0.003.

**Session 3 (2026-10-21):**
- **8-bit cast warning:** `finetune.py::build_quant_config` filters bitsandbytes' per-matmul `MatMul8bitLt: inputs will be cast ... to float16` notice in 8-bit mode. The filter is `_drop_int8_cast_notice`, and a test checks it.
- **Speed configs:** the E3 configs `bench_nogc_b2x16` and `bench_nogc_b4x8`. `bench.yaml` pins the old reference.
- **v2 configs:** gradient checkpointing is off in the 8B, 1B and 3B `v2.yaml`, at 2×16, 8×4 and 4×8. The 1B change reaches `compact` and `venue20`.
- **E7 config:** `llama32_1b/seed43.yaml`, with a test.
- **Eval sample seed:** `finetune.py::load_eval_dataset` draws the 1,000-row in-training eval sample with `evaluation.seed` instead of `training.seed`. Every run so far used 42 for both, so their samples are unchanged. Tested by `test_eval_sample_ignores_the_training_seed`.
- **Colab runner:**
  - Drive stays mounted between runs.
  - The setup cell uninstalls torchao and excludes `.gdoc` files.
  - `runtime.unassign()` replaces a call that didn't exist.
  - The duplicate config cell is gone.

**Session 2 (2026-10-20):**
- **Dependencies:** pinned `trl>=1.16,<2` and `peft>=0.21`, and refreshed `uv.lock`.
  - datasets 5.1, pyarrow 25 and mlflow 3.17 came along.
  - `uv sync --extra gpu` works on macOS.
- **trl 0.x code removed.**
- **Double forward fixed:** the hook in `FirstTokenAccuracySFTTrainer.compute_loss`. `tests/test_llm_trainer_trl.py` trains a tiny Llama through the real trl and asserts one forward per micro-batch.
- **Peak memory:** `peak_gpu_memory()` logs it to W&B and MLflow.
- **Inspection:** one prompt pass instead of two (`diagnostics.py::score_prompts`). It's tested against `model.generate` step by step.
- **`tokenization_check`:** follows trl 1.x's common-prefix rule.
- **macOS:** lightgbm's and torch's OpenMP segfault together, so the tests that run real forwards call `torch.set_num_threads(1)`.

**Session 1:**
- **Resume from the Hub:** `hub_strategy="checkpoint"` plus `find_resumable_checkpoint`.
- **W&B:** the run id is the run name, and resume is allowed. `log_model` is off.
- **Config knobs:** `tf32`, best-model and early-stopping options, `train_sample_strategy` and `wandb.log_model`.
- **Diagnostics:**
  - `p_home` computed in fp32.
  - The expected margin.
  - Log-loss, Brier, AUC, a calibration table and three RMSEs.
  - The flags `--base-only`, `--few-shot` and `--calibrate-with`.
- **Prompt variants:** `per_team_compact.yaml` and `per_team_venue20.yaml`. Their datasets were built and pushed on 2026-10-23.
- **Configs:** `qlora/_defaults.yaml` (v2), `llama31_8b/baseline.yaml` (pins test_run2 exactly, 8-bit) and `baseline_t4.yaml` (fp16 for a T4).
