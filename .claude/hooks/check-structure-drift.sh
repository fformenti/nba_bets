#!/bin/bash
# Detect drift between actual file tree and PROJECT_STRUCTURE.md
# Called by Stop hook — outputs reminder only when drift exists

set -euo pipefail

INPUT=$(cat)

# Prevent infinite loop: if this is a re-trigger after Claude acted on our reminder, stay silent
STOP_HOOK_ACTIVE=$(echo "$INPUT" | jq -r '.stop_hook_active // false')
if [ "$STOP_HOOK_ACTIVE" = "true" ]; then
  exit 0
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STRUCTURE_FILE="$PROJECT_ROOT/.claude/docs/PROJECT_STRUCTURE.md"

if [ ! -f "$STRUCTURE_FILE" ]; then
  exit 0
fi

# Get actual files under src/ and configs/ (excluding __pycache__, __init__.py, .pyc, .DS_Store)
ACTUAL_BASENAMES=$(cd "$PROJECT_ROOT" && find src configs -type f \
  ! -path '*/__pycache__/*' \
  ! -name '*.pyc' \
  ! -name '.DS_Store' \
  ! -name '__init__.py' \
  2>/dev/null | xargs -I{} basename {} | sort -u)

# Extract documented file paths from PROJECT_STRUCTURE.md
# Matches lines like "│   ├── filename.py" or "│   └── filename.ext" and strips tree chars + descriptions
DOCUMENTED_FILES=$(grep -oE '(├──|└──)\s+[^ /]+\.[a-zA-Z0-9]+' "$STRUCTURE_FILE" 2>/dev/null \
  | sed 's/.*── //' | sort -u)

# Remove root-level files from documented list (they're not under src/ or configs/)
# These are files like CLAUDE.md, pyproject.toml, mlflow.db, Makefile that live at project root
ROOT_FILES="CLAUDE.md Makefile pyproject.toml mlflow.db"
for f in $ROOT_FILES; do
  DOCUMENTED_FILES=$(echo "$DOCUMENTED_FILES" | grep -v "^${f}$" 2>/dev/null || true)
done

# Find new files not mentioned in PROJECT_STRUCTURE.md
NEW_FILES=$(comm -23 <(echo "$ACTUAL_BASENAMES") <(echo "$DOCUMENTED_FILES"))

# Find documented files that no longer exist
MISSING_FILES=$(comm -13 <(echo "$ACTUAL_BASENAMES") <(echo "$DOCUMENTED_FILES"))

if [ -n "$NEW_FILES" ] || [ -n "$MISSING_FILES" ]; then
  REMINDER="PROJECT_STRUCTURE.md is out of sync with the codebase."

  if [ -n "$NEW_FILES" ]; then
    REMINDER="$REMINDER New files not documented: $(echo "$NEW_FILES" | head -10 | tr '\n' ', ' | sed 's/,$//')."
  fi

  if [ -n "$MISSING_FILES" ]; then
    REMINDER="$REMINDER Documented files no longer exist: $(echo "$MISSING_FILES" | head -10 | tr '\n' ', ' | sed 's/,$//')."
  fi

  REMINDER="$REMINDER Please update .claude/docs/PROJECT_STRUCTURE.md to reflect the current file structure."

  echo "$REMINDER"
fi

exit 0
