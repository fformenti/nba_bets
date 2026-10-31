# nba-bets

NBA game prediction (sklearn and fine-tuned LLMs) and Polymarket betting. Architecture,
data flow and layering rules: [CLAUDE.md](CLAUDE.md). Everything else written about the project: [docs/](docs/README.md).

## Google Drive sync

Colab trains from `My Drive/nba_bets/`: the notebooks in `colab_notebooks/` rsync that
folder onto the runtime and `pip install` it. `make drive-sync` keeps it in step with the
repo through the Drive API (`src/gdrive/`).

| Repo path | Direction | Deletes? |
|---|---|---|
| `src/`, `configs/`, `colab_notebooks/`, `pyproject.toml` | push (repo is the truth) | Drive-only files go to Drive's trash (recoverable 30 days) |
| `outputs/runs/`, `comparisons/` | pull (Drive → repo); on a content clash a newer Drive copy wins, a newer local copy is kept | never, on either side |

Nothing else under `My Drive/nba_bets/` (`data/`, `mlruns/`, `wandb/`, …) is read or written. Files are
compared by md5, so unchanged files cost nothing. `.DS_Store`, `__pycache__`, `*.pyc` and
`.ipynb_checkpoints` are ignored. Pull also skips figures and images (`*/figures/`,
`*.png`, …): they are redrawn locally from the tables. `outputs/eda/` and
`outputs/archive/` are local-only and never synced.

### One-time setup

1. In [Google Cloud Console](https://console.cloud.google.com), create a project (e.g.
   `nba-bets-drive`).
2. **APIs & Services → Library → Google Drive API → Enable.**
3. **Google Auth Platform → Branding / Audience:** user type *External*, then add your
   Google account under *Test users*.
4. **Clients → Create client → Desktop app**, download the JSON, and store it outside the
   repo:
   ```bash
   mkdir -p ~/.secrets/nba_bets_drive
   mv ~/Downloads/client_secret_*.json ~/.secrets/nba_bets_drive/client_secret.json
   chmod 600 ~/.secrets/nba_bets_drive/client_secret.json
   ```
5. `make drive-auth`: a browser opens; accept the "unverified app" warning (it is your own
   app) and allow Drive access. The token is saved to `~/.secrets/nba_bets_drive/token.json`.

While the app's publishing status is *Testing*, Google expires the token after 7 days;
re-run `make drive-auth`, or click **Audience → Publish app** to stop the expiry. The
paths can be overridden with `NBA_BETS_DRIVE_CLIENT_SECRETS` and `NBA_BETS_DRIVE_TOKEN`.

### Usage

```bash
make drive-sync DRY_RUN=1          # print what would be uploaded, downloaded, trashed
make drive-sync                    # do it
make drive-sync ONLY=code          # push src/, configs/, colab_notebooks/, pyproject.toml
make drive-sync ONLY=outputs       # pull outputs/ only, e.g. after a Colab run
make drive-ls DRIVE_PATH=outputs/runs RECURSIVE=1   # inspect Drive
```

Push before starting a Colab session; sync `outputs` after it finishes.
`outputs/checkpoints/` is never synced: adapters are large and live on the training box.
