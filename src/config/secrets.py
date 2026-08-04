"""Secret lookup for the project.

Resolution order:

1. the real process environment — direnv locally, ``export`` on a GPU box
2. a repo-local ``.env``, loaded non-overriding, for machines without direnv
3. ``RuntimeError`` naming the variable and saying where to put it

Values never live in the repo. Locally they come from ``~/.secrets/nba_bets.env``
(mode 600), wired up by the repo's ``.envrc``. ``override=False`` is what makes
the remote path work: on a rented GPU box you can ``scp`` a ``.env`` next to the
repo, and if that box exports the variables itself, its own values still win.

Importing this module must never raise — ``tests/test_imports_smoke.py`` imports
every module under ``src/``, including on machines with no credentials at all.
That is why callers use :func:`require_env` at call time rather than building
API clients at module scope.
"""

import os

from dotenv import load_dotenv

# Runs once, on first import, for the non-direnv case. A real environment
# variable always beats the file.
load_dotenv(override=False)

SECRETS_FILE = "~/.secrets/nba_bets.env"

# A few keys are shared with other projects and so live in the global file that
# ~/.zshrc sources, not the per-project one.
GLOBAL_SECRETS_FILE = "~/.secrets/env"

# Named here rather than in each consumer so the token has one spelling.
HF_TOKEN_ENV_VAR = "HF_llm_training_token"
WANDB_TOKEN_ENV_VAR = "WANDB_API_KEY"
OPENAI_ENV_VAR = "OPENAI_API_KEY"
SERPER_ENV_VAR = "SERPER_API_KEY"


def require_env(name: str, hint: str = "", secrets_file: str = SECRETS_FILE) -> str:
    """Return the variable's value, or raise a message that says how to fix it."""
    value = os.environ.get(name)
    if not value:
        message = (
            f"{name} is not set. Add it to {secrets_file} and run `direnv reload`, "
            "or export it directly when running on a remote box."
        )
        if hint:
            message = f"{message} {hint}"
        raise RuntimeError(message)
    return value
