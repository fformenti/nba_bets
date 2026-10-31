# Project documentation

The index of everything written about this project. Paths quoted inside the docs start at the repo
root. Architecture and layering rules live in [CLAUDE.md](../CLAUDE.md), and the file tree in
[PROJECT_STRUCTURE.md](../.claude/docs/PROJECT_STRUCTURE.md).

## Where a document goes

| You are writing | Under `docs/` | Name | Reader |
|---|---|---|---|
| Work ahead: a plan, a handoff | `plans/` | `<topic>.md` | assistants |
| A plan that is finished or superseded | `plans/archive/` | `YYYY-MM-DD-<topic>.md` | assistants |
| What was measured: an experiment, audit or EDA | `findings/<area>/` | `YYYY-MM-DD-<topic>.md` | assistants |
| A contract or runbook, kept true as the code changes | `reference/` | `<topic>.md` | both |
| An explanation of how something works | `explanations/` | `<topic>.md` | the human reader |
| Something to fix later | `todo.md` | a new entry under "Open" | both |

- **Trust.** `reference/` is current. `findings/` is true as of its date, and a later finding
  overrules an earlier one. `plans/` is intent.
- **Names.** Lowercase kebab-case. The date is when the event happened, not the last edit.
- **Areas.** `findings/llm/` covers fine-tuning. `findings/pipeline/` covers ETL, features, data
  quality and the sklearn models.
- **Keeping this current.** Add a line below for every new doc.

## Plans

- [llm-finetune-handoff.md](plans/llm-finetune-handoff.md): the living LLM plan, scoreboard and
  next steps. Start every LLM session at its §0, and update it at the end.
- [archive/2026-03-23-sample-weighting-redesign.md](plans/archive/2026-03-23-sample-weighting-redesign.md):
  cross-season decay weighting. Superseded.

## Findings: LLM

- [2026-10-20-e3-speed-benchmark.md](findings/llm/2026-10-20-e3-speed-benchmark.md): E3, training
  speed on a Colab L4.
- [2026-10-20-quantization-8bit.md](findings/llm/2026-10-20-quantization-8bit.md): what 8-bit
  quantization costs, and why it warns.
- [2026-10-21-size-ladder.md](findings/llm/2026-10-21-size-ladder.md): E4–E6, does model size help?
- [2026-10-21-seed-noise.md](findings/llm/2026-10-21-seed-noise.md): E7, the noise floor and the
  win rule.
- [2026-10-23-prompt-variants.md](findings/llm/2026-10-23-prompt-variants.md): E11 and E12, prompt
  variants on the 1B.
- [2026-10-23-e7-e11-e12-session-summary.md](findings/llm/2026-10-23-e7-e11-e12-session-summary.md):
  session summary, including how the three Hub datasets relate.
- [2026-10-28-validation-size.md](findings/llm/2026-10-28-validation-size.md): should validation
  grow beyond three seasons?
- [2026-10-29-learning-rate.md](findings/llm/2026-10-29-learning-rate.md): E9, is 2e-4 the right
  learning rate?
- [2026-10-30-target-mode.md](findings/llm/2026-10-30-target-mode.md): E16, does the margin's
  magnitude help the sign?

## Findings: pipeline

- [2026-03-23-games-features-eda.md](findings/pipeline/2026-03-23-games-features-eda.md):
  pre-training EDA of `games_features.csv`.
- [2026-08-11-pipeline-audit.md](findings/pipeline/2026-08-11-pipeline-audit.md): cold-start audit,
  data provenance and why the reference inputs are versioned.
- [2026-08-11-rebuild-audit.md](findings/pipeline/2026-08-11-rebuild-audit.md): full-rebuild
  reproducibility. Its model metrics are superseded.
- [2026-08-12-conference-split.md](findings/pipeline/2026-08-12-conference-split.md): why three
  conference models became one.

## Reference

- [outputs.md](reference/outputs.md): the `outputs/` layout, the result schema and plotting.
- [runpod-training.md](reference/runpod-training.md): runbook for training the LLM on a RunPod GPU box.

## Explanations

- [llm-hub-revisions-vs-wandb-versions.md](explanations/llm-hub-revisions-vs-wandb-versions.md):
  finding and loading a specific LLM checkpoint after a training disconnect.

## To do

- [todo.md](todo.md): the backlog.
