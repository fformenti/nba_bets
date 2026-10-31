# To do

## Open

Nothing filed here yet. The audits keep their own open items: see "Still open" in
`docs/findings/pipeline/2026-08-11-pipeline-audit.md` and
`docs/findings/pipeline/2026-08-11-rebuild-audit.md`.

## Closed

Not fixed (pre-existing, unrelated)

src/ml/llm/evaluate.py:86 passes output_dir= to ClassificationTester, whose __init__ (testers.py:239) takes no such argument, and line 94 reads metrics['accuracy'] from a run() that returns None. make evaluate-llm will TypeError on the first call. Nothing to do with the prose removal — worth a separate pass before your first evaluation run.
(2026-10-23: obsolete. evaluate.py uses BatchedClassificationTester, whose run() returns the metrics; it now writes outputs/runs/<run>/ instead of HTML charts.)
