"""Secret lookup for the project.

Resolution order:

1. the real process environment — direnv locally, ``export`` on a GPU box
2. a repo-local ``.env``, loaded non-overriding, for machines without direnv
3. Colab's Secrets panel, which never populates ``os.environ`` on its own
4. ``RuntimeError`` naming the variable and saying where to put it

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

# Google Drive OAuth (src/gdrive/). Files, not values: the client secret is
# downloaded from Google Cloud, the token is written by `make drive-auth`. Each
# default can be overridden by pointing its env var at another path.
DRIVE_CLIENT_SECRETS_FILE = "~/.secrets/nba_bets_drive/client_secret.json"
DRIVE_TOKEN_FILE = "~/.secrets/nba_bets_drive/token.json"  # noqa: S105 — a path
DRIVE_CLIENT_SECRETS_ENV_VAR = "NBA_BETS_DRIVE_CLIENT_SECRETS"
DRIVE_TOKEN_ENV_VAR = "NBA_BETS_DRIVE_TOKEN"  # noqa: S105 — a variable name


def load_env_file(secrets_file: str = SECRETS_FILE) -> None:
    """Explicitly load the secrets file, for kernels direnv never reaches
    (VS Code's Jupyter extension, a notebook server started before `cd`).

    Safe to call from a shell that already has the real vars exported —
    override=False means an existing process env value always wins.
    """
    load_dotenv(os.path.expanduser(secrets_file), override=False)


def _colab_secret(name: str) -> str | None:
    """Colab's Secrets panel doesn't populate os.environ; ask it directly."""
    try:
        from google.colab import userdata
    except ImportError:
        return None
    try:
        return userdata.get(name)
    except Exception:  # SecretNotFoundError, or notebook access not granted
        return None


def require_env(name: str, hint: str = "", secrets_file: str = SECRETS_FILE) -> str:
    """Return the variable's value, or raise a message that says how to fix it."""
    value = os.environ.get(name)
    if not value:
        value = _colab_secret(name)
        if value:
            # Libraries that read the environment themselves (wandb) see it too.
            os.environ[name] = value
    if not value:
        message = (
            f"{name} is not set. Add it to {secrets_file} and run `direnv reload`, "
            "export it directly when running on a remote box, or add it to Colab "
            "Secrets with notebook access enabled."
        )
        if hint:
            message = f"{message} {hint}"
        raise RuntimeError(message)
    return value
