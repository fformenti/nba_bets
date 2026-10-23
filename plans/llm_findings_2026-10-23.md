# LLM fine-tuning: findings from E7, E11 and E12 (2026-10-21 → 2026-10-23)

A session summary for picking the work up later. The living plan is
`plans/llm_finetune_handoff.md`; start a new session from its §0. The detailed write-ups are
in `project_knowledge/llm_training/`.

## 1. The headline
Every model scores the same on validation: Llama 1B, 3B and 8B, two prompt variants of the 1B, and
XGBoost on the same features all land at **0.628–0.630 calibrated log-loss**. Model size, the random
seed and the prompt layout don't change it, and neither does adding one more view of the existing
stats. **The features set the ceiling.** Gains must come from new information, such as injuries and
availability, or from combining different model families.

## 2. Results (validation, 2,949 games, Platt cross-fit, paired with E4)

| Run | What changed from E4 | Log-loss | vs E4 (95% CI) | Accuracy | s/step | Peak | Wall |
|---|---|---|---|---|---|---|---|
| `e4-llama32-1b` | — (base) | 0.6284 | — | 64.2% | 2.21 | 7.9 GB | 31 min |
| `e7-llama32-1b-seed43` | training seed 42 → 43 | 0.6280 | −0.0004 [−0.0019, +0.0011] | 64.4% | 2.25 | 7.9 GB | 32 min |
| `e12-llama32-1b-venue20` | last-season venue line kept longer | 0.6282 | −0.0002 [−0.0009, +0.0004] | 64.4% | 2.28 | 7.9 GB | 32 min |
| `e11-llama32-1b-compact` | whitespace and header removed | 0.6285 | +0.0001 [−0.0013, +0.0015] | 64.2% | **1.48** | **5.8 GB** | **21 min** |

## 3. Findings

### E7: the noise floor
- Changing only the seed moved log-loss by 0.0004 (paired SD 0.0008). The two runs' `p_home`
  correlate at 0.993 and pick the same winner in 96% of games.
- Averaging the two seeds scored 0.628, the same as either one. Seed ensembles aren't worth GPU time.
- **The win rule from here on:** a change beats E4 only if log-loss drops by at least 0.002 *and*
  a paired bootstrap of the per-game difference has a 95% interval excluding zero. "Not worse"
  means the interval includes zero and the point estimate is within 0.002.
- **The E7 eval curve in W&B looked worse, but it was a sampling artefact.** The 1,000-row
  in-training eval sample was drawn with `training.seed`, so E7 was evaluated on different games.
  - **Fixed:** `src/ml/llm/finetune.py::load_eval_dataset` now uses `evaluation.seed`.
  - **Tested by** `test_eval_sample_ignores_the_training_seed`.

### E12: venue20 adds nothing
- **Why it was tried:** the base prompt shows each team's *current-season* home or road record
  ("Adj Venue win%", L5/L10). Early in a season those windows rest on a handful of games, which is
  noisy. Last season's home or road record ("Adj Last-season venue win%") is a stable prior for that
  stretch.
  - The base prompt hides the line once both teams have **7** venue games.
  - The splits already drop games before 15 games played, by which point each team has about 7
    home or road games. So in the base dataset the line almost never shows: **0.8%** of rows.
  - venue20 raises the threshold to **20** venue games, so the line shows through roughly the first
    half of each season: **40%** of rows.
  - **Question tested:** does a long-run venue prior help while this season's venue sample is
    small?
- **Result:** no. It's a tie overall, and also on the 1,165 games where the line shows (−0.0001
  [−0.0012, +0.0011]). Its `p_home` correlate with E4's at 0.9987, so the model barely reacts to the
  line.
- **Probable reason:** the prompt already carries enough form, margin and venue information. The
  prior is either redundant or too weak to matter.
- **Not adopted.**

### E11: compact is as good and a third faster
- It removes alignment whitespace and the definitions header, which is identical in every row.
  Prompts drop from 425 to 282 tokens (−34%).
- **Same quality:** +0.0001 against E4, with an interval that includes zero.
- **Cheaper:** 1.48 s/step against 2.21 (−33%), 5.8 GB against 7.9 GB, and 21 min against 31 min.
- **Adopted:** `configs/train_llm/qlora/llama32_1b/compact.yaml` (dataset
  `fformenti/nba-bets-compact`) is the working config. New 1B experiments should `_include` it.

## 4. How the three datasets relate
All three hold **the same games in the same splits**: 23,247 train, 2,949 validation and 1,018
test, from `build_splits`. Only the prompt text differs, so any score difference comes from the
prompt alone.

```
configs/llm_prompt/per_team.yaml         → fformenti/nba-bets          (base: aligned layout, venue line threshold 7)
├── per_team_venue20.yaml  (_include base) → fformenti/nba-bets-venue20  (base + threshold 20)
└── per_team_compact.yaml  (_include base) → fformenti/nba-bets-compact  (base + layout: compact)
```

**compact forks from the base, not from venue20.** It keeps the base threshold of 7, so it doesn't
carry the extended venue line. Each variant changes exactly one thing from the base.

Example: the same validation game (`22100230`) early in 2021/22.
- **Base:** the venue line is hidden.
- **venue20:** the base prompt plus `Adj Last-season venue win%  50  58` in the team table.
- **compact:** the base prompt's content with single spaces instead of aligned columns, and
  without the two-line definitions header.

## 5. State of the code (uncommitted)
- **`src/ml/llm/finetune.py`:** the eval sample is drawn with `evaluation.seed`, and a test covers it.
- **`configs/train_llm/qlora/llama32_1b/seed43.yaml`:** the E7 config. Its comment now gives the
  measured noise floor.
- **Tests:** 544 pass (`MLFLOW_DISABLE_AGENT_HINT=1 uv run --extra gpu pytest -q`).
- **Write-ups:**
  - `project_knowledge/llm_training/seed_noise.md`
  - `project_knowledge/llm_training/prompt_variants.md`
  - `plans/llm_finetune_handoff.md`, updated to these results.

## 6. Next steps
1. **N2, blending the LLM with XGBoost** (Claude, CPU, no GPU).
   - Regenerate XGBoost's validation predictions by gameId; they were never saved.
   - Blend them with E11's `p_home` and score the blend with the win rule.
   - Turn the paired-bootstrap comparison into a library function.
   - A blend helps only if XGBoost's predictions correlate with the LLM's clearly below 0.99.
2. **Then the user chooses:**
   - **E8, the learning curve:** 10/25/50% of train on the compact config, about 15 min of L4 in
     total. It tells whether more historical seasons would help.
   - **New information, such as injuries and availability:** a data project to scope (source, ETL,
     prompt line, dataset). Given these results, this is the likely real lever.
3. **E15, last:** the best config on test, scored once.

## 7. Optional cleanup on the Hub
- **Keep:**
  - models `test_run2`, `e4-llama32-1b` and `e11-llama32-1b-compact`;
  - datasets `nba-bets` and `nba-bets-compact`.
- **Can delete:**
  - models `e5-llama32-3b`, `e7-llama32-1b-seed43`, `e12-llama32-1b-venue20`, `resume-drill-1`,
    `resume-drill-2` and `test_run`;
  - dataset `nba-bets-venue20`, which one command rebuilds.
- **Keep every eval folder** in `outputs/llm/eval/`; the scoreboard comes from them.
- **Before any Colab run,** set a new `RUN_NAME`. Reusing a name resumes that run and overwrites its
  CSVs.
