# Notes

- Workspace lives at `learning/llm-finetuning/` inside the nba_bets repo, not the repo root, so lessons don't clutter the project.
- The user describes their recipe as "boilerplate with default values", but it was already tuned from evidence (batch, clip, epochs). Lessons should help them see the reasoning behind choices they already made.
- Handoff doc preference: explain the reasoning, give a recommendation rather than a menu, change one thing per experiment. Apply the same to lessons.
- Ground every lesson in their real numbers (scoreboard in `docs/plans/llm-finetune-handoff.md` §2).
- Lesson 1 (2026-10-29): three kinds of knobs (cost / learning / information), AUC is ranking and Platt leaves it alone, ranked next steps.
- Candidate next lessons: what LoRA's r and ΔW = BA physically are; why completion-only loss matters; log-loss vs AUC vs accuracy with their own CSV; how a blend/stack works (N2); reading a learning curve (E8).
