---
name: simplify
description: >
  Reduction-focused review of existing Python code. Trigger on /simplify,
  "clean up", "reduce complexity", "make simpler", "remove duplication",
  "simplify this". Does NOT cover initial code creation standards (software-dev
  persona) or ML-pipeline review (ml-pipeline skill).
---

# Simplify

## Scope

Reduction-only pass on existing code. Not a style or standards audit. The goal
is to remove what is unnecessary — not to rewrite what is already working.

## What to look for

- **Dead code**: unused imports, variables assigned but never read, feature
  columns computed but never passed to the model
- **Duplicate logic**: the same transformation or pattern implemented in more
  than one place across modules
- **Unnecessary indirection**: helper functions called exactly once, wrappers
  that add no logic, intermediate variables used only to be immediately passed
  to the next call
- **Over-abstraction**: custom classes or transformers that replicate what a
  standard sklearn primitive (`StandardScaler`, `SimpleImputer`,
  `OrdinalEncoder`) already does out of the box
- **Manual path construction**: string concatenation or `os.path.join` where an
  import from `src/config/paths.py` already provides the canonical path
- **Collapsed DataFrames**: a DataFrame created, never examined, and immediately
  passed into the next call — collapse it into a single expression

## What NOT to touch

Type hints, docstrings, naming conventions, and logging patterns are
`software-dev.md` territory. Do not re-audit them here. If you notice a
violation, mention it briefly but do not treat it as the focus of this pass.

## Reference

Load `references/pythonic-style.md` for the Zen of Python review lens and the
project-specific checklist before producing suggestions.

## Response pattern

1. **Read** the file(s) in scope
2. **Identify** each reduction opportunity with a one-line rationale
3. **Propose** concrete changes — show the before/after for each
4. **Apply** only what the user agrees to
