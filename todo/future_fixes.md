Not fixed (pre-existing, unrelated)

src/ml/llm/evaluate.py:86 passes output_dir= to ClassificationTester, whose __init__ (testers.py:239) takes no such argument, and line 94 reads metrics['accuracy'] from a run() that returns None. make evaluate-llm will TypeError on the first call. Nothing to do with the prose removal — worth a separate pass before your first evaluation run.
