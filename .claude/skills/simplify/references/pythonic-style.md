# Pythonic Style Reference

When to read: performing a simplification pass on existing Python code.

## Zen of Python — five load-bearing aphorisms for this codebase

| Aphorism | Review lens |
|----------|-------------|
| **Simple is better than complex** | Primary reduction lens. If the simpler version is equally correct, prefer it unconditionally. |
| **Flat is better than nested** | Applies to ETL and feature engineering chains. More than two levels of nesting is a signal to extract or flatten. |
| **If the implementation is hard to explain, it's a bad idea** | Flags over-abstracted transformers or pipelines where the data flow requires a diagram to follow. |
| **There should be one obvious way to do it** | Catches duplicated utility functions across modules. If two modules both define a lag helper, one is redundant. |
| **Sparse is better than dense** | Flags one-liners that compress multiple operations into a single expression and hide logic during feature construction. |

## Project-specific checklist

Run through this list before finalizing suggestions:

- [ ] Is a path being constructed with string concatenation or `os.path.join`
      instead of being imported from `src/config/paths.py`?
- [ ] Are multiple feature functions repeating the same lag calculation pattern
      that could be a single shared call?
- [ ] Is a custom sklearn transformer replicating what `StandardScaler`,
      `SimpleImputer`, or `OrdinalEncoder` already does natively?
- [ ] Is a function living in `training/` that belongs in `features/` — i.e.,
      pure transformation logic with no orchestration or I/O?
- [ ] Is there a DataFrame created, never inspected, and immediately passed into
      the next call (collapse into one expression)?
- [ ] Are there unused imports at the top of the file?
- [ ] Are there feature columns computed in a feature function but never
      included in the final merged feature set?
