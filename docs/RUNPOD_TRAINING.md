# Training the LLM on a RunPod GPU box

Step-by-step runbook for QLoRA fine-tuning `meta-llama/Meta-Llama-3.1-8B` on the
`fformenti/nba-bets` dataset via `make train-llm` (`src/cli/train_llm.py`).

This can't run on the laptop: `bitsandbytes` has no macOS wheel, which is why the fine-tuning stack
lives in the `gpu` extra of `pyproject.toml` rather than the core dependencies.

The training data is pulled from the Hugging Face Hub, not from the repo, so **the pod needs no data
files** — only a HF token.

---

## 0. Before you rent a GPU

Do all of this from your laptop. Every minute spent fixing credentials on a running pod is billed.

1. **Accept the Llama 3.1 license.** Visit
   [meta-llama/Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B) while logged
   in as the account that owns your token, and request access. It is a gated repo — without this,
   `from_pretrained` fails with a `403` *after* the pod is already billing and after `preflight()`
   has passed (preflight only checks the dataset, not the model).

2. **Check your HF token has `write` scope.** The run pushes checkpoints *and* the final adapter to
   `fformenti/<run-name>`. A read token gets you through preflight and then fails at the first
   `save_steps` boundary. Check at https://huggingface.co/settings/tokens.

3. **Confirm the dataset is current.** The pod loads `fformenti/nba-bets` with `train` / `validation`
   / `test` splits. It's built and uploaded by:

   ```bash
   make build-llm-dataset ARGS=--push
   ```

   Run this from the laptop — the pod has no `data/processed/` files. Without `--push` the dataset is
   only built and summarised locally, and the pod keeps loading whatever is already on the Hub.

   The Hub repo still holds the **pre-refactor** dataset until you push once. That version predates
   the current serializer and uses the old `text` schema, so training against it fails as described
   in the troubleshooting table below. Push before the first pod run, and again whenever the features
   or the serialization format change.

   The schema is load-bearing: each split must have exactly `game_id`, `prompt` and `completion`, and
   **no `text` column**. trl picks its collator by inspecting column names, so a stray `text` column
   silently switches training to language modeling over the prompt — where the answer never appears.
   `completion` values are signed integers with no leading space (`+7`, `-24`), which the Llama
   tokenizer splits into exactly two tokens: the sign and the magnitude.

4. **Push your branch.** You'll `git clone` on the pod, so anything uncommitted locally won't be
   there.

---

## 1. Pick the pod

| Setting | Value | Why |
|---|---|---|
| GPU | **RTX 4090 / A5000 (24 GB)** minimum, **L40S / A6000 (48 GB)** comfortable | 8-bit 8B weights are ~9 GB; the rest is activations at `max_sequence_length: 1024` |
| Architecture | Ampere or newer | `training.bf16: true` and `optim: paged_adamw_32bit` require it. **T4 and V100 will not work** — bf16 is unsupported |
| Template | RunPod official **PyTorch 2.x / CUDA 12.x** | CUDA driver + build tools already present |
| Container disk | ≥ 60 GB | 16 GB of base-model shards plus `outputs/llm/<run>/checkpoint-*` with `save_total_limit: 10` |
| Network volume | **Yes**, mounted at `/workspace` | See below |
| Pricing | **Spot is fine** | See below |

**Attach a network volume.** The container disk is destroyed when you terminate the pod. Putting the
HF cache and the repo on a `/workspace` volume means a restart doesn't re-download 16 GB of Llama
weights — that download is the single biggest avoidable cost in this workflow.

**Spot instances are safe here.** The trainer runs with `hub_strategy="every_save"`, so every
`save_steps: 500` a checkpoint goes to the Hub. `find_resumable_checkpoint()`
(`src/ml/training/llm_finetune.py:54`) finds the newest one and resumes. A preemption costs you at
most 500 steps, and spot is roughly half price. This is designed for interruption.

---

## 2. Set up the box

Connect via the RunPod web terminal, or **Connect → SSH over exposed TCP** for a real terminal.

### 2.1 Point the caches at the volume — do this first

```bash
cat >> ~/.bashrc <<'EOF'
export HF_HOME=/workspace/.cache/huggingface
export WANDB_DIR=/workspace/wandb
EOF
source ~/.bashrc
```

If you skip this, the model cache lands on the ephemeral container disk and you re-download it on
every pod restart.

### 2.2 Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version
```

This project uses `uv` for everything. Never `pip install` or call `python` directly.

### 2.3 Clone and install

```bash
cd /workspace
git clone https://github.com/<your-org>/nba_bets.git
cd nba_bets
git checkout feature/train_LLM

uv sync --extra gpu
```

`--extra gpu` is what pulls in `peft`, `trl`, `accelerate`, `bitsandbytes`, `wandb` and
`sentencepiece`. A plain `uv sync` will install fine and then fail at import time.

### 2.4 Verify CUDA

```bash
nvidia-smi
uv run python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

If `torch.cuda.get_device_name(0)` throws, stop here — `preflight()` will refuse to run anyway.

---

## 3. Secrets

`.env` is gitignored, so it does not come with the clone. Two variables matter for training:

| Variable | Used by | Required? |
|---|---|---|
| `HF_llm_training_token` | `get_hf_token()` — model/dataset access + Hub pushes | Always |
| `WANDB_API_KEY` | `preflight()` when `tracking.wandb.enabled: true` | Unless you disable wandb |

Note the unusual name: it is `HF_llm_training_token`, *not* `HF_TOKEN`.

On your laptop these come from `~/.secrets/` via direnv, not from a `.env` — see
the Secrets section of `CLAUDE.md`. The pod is the case the `.env` fallback in
`src/config/secrets.py` exists for.

**Preferred — RunPod environment variables.** Set them in the pod's template/env config before
starting the pod. They survive restarts and never touch the volume's filesystem. `load_dotenv()` does
not override real environment variables, so this takes precedence cleanly.

**Alternative — a `.env` on the pod:**

```bash
cat > /workspace/nba_bets/.env <<'EOF'
HF_llm_training_token=hf_xxxxxxxxxxxxxxxxxxxx
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
EOF
chmod 600 /workspace/nba_bets/.env
```

Remember this file lives on a persistent volume that other pods can mount.

---

## 4. Smoke test first

The config on this branch already has `data.max_train_samples: 100` — that's a deliberate smoke-test
cap. Run it as-is before committing to a multi-hour job:

```bash
cd /workspace/nba_bets
uv run python -m src.cli.train_llm \
  --config configs/train_llm/llama31_8b_qlora.yaml \
  --run-name smoke-test \
  --max-train-samples 100
```

> `make train-llm` only forwards `--config` and `--run-name`. To use `--max-train-samples` you must
> call the module directly, as above.

Watch for these three lines from `preflight()`:

```
GPU: NVIDIA GeForce RTX 4090 | free 23.5 GB of 25.4 GB
Dataset fformenti/nba-bets is reachable.
Memory footprint: 9084.6 MB (quantization=8bit)
```

Then confirm `fformenti/smoke-test` shows up at https://huggingface.co/fformenti. If it's there, the
whole path — auth, gated model, dataset, GPU, Hub push — works. Delete the repo afterwards.

---

## 5. The real run

### 5.1 Adjust the config

Edit `configs/train_llm/llama31_8b_qlora.yaml`:

```yaml
data:
  max_train_samples: null    # was 100 — use the full training split

tracking:
  mlflow:
    enabled: false           # recommended, see below
```

**On MLflow:** `tracking_uri` is `sqlite:///mlflow.db`, a file on the pod that dies with it. wandb
already captures everything you need for a remote run. Either turn MLflow off, or point it at the
volume (`sqlite:////workspace/mlflow.db` — note four slashes for an absolute path) and remember to
copy the file off before terminating.

### 5.2 Launch it detached

Use `tmux` so an SSH drop doesn't kill training, and so you can re-attach and watch:

```bash
tmux new -s train
cd /workspace/nba_bets
make train-llm LLM_RUN=nba-bets-2026-07-30
```

Detach with `Ctrl-b d`. Re-attach later with `tmux attach -t train`.

The `nohup` form in the `train_llm.py` docstring works too, but you lose the ability to watch:

```bash
nohup make train-llm LLM_RUN=nba-bets-2026-07-30 > train.log 2>&1 &
tail -f train.log
```

### 5.3 Always pass an explicit `LLM_RUN`

With `run_name: null` in the config, `resolve_run_name()` generates a fresh timestamp on every launch
(`src/ml/training/llm_finetune.py:46`). The run name *is* the Hub repo name, so a generated one means
a relaunch writes to a different repo and **cannot resume**. Pin it.

---

## 6. Resuming after an interruption

Re-run the **identical** command with the **same** `LLM_RUN`:

```bash
make train-llm LLM_RUN=nba-bets-2026-07-30
```

What happens: `find_resumable_checkpoint()` lists the files in `fformenti/nba-bets-2026-07-30`, picks
the highest `checkpoint-N`, `snapshot_download`s it into `outputs/llm/<run>/checkpoint-N`, and hands
it to `trainer.train(resume_from_checkpoint=...)`. You'll see:

```
Found checkpoint-1500 in fformenti/nba-bets-2026-07-30; downloading to resume.
```

To deliberately start over into the same repo name:

```bash
uv run python -m src.cli.train_llm \
  --config configs/train_llm/llama31_8b_qlora.yaml \
  --run-name nba-bets-2026-07-30 \
  --resume never
```

---

## 7. Evaluate and collect results

```bash
make evaluate-llm LLM_RUN=nba-bets-2026-07-30
```

Runs on the `test` split with `evaluation.size: 200` rows. Charts are written as standalone HTML to
`outputs/llm/eval/<run_name>/` so they can be pulled off a headless box:

```bash
# from your laptop
scp -r -P <port> root@<pod-ip>:/workspace/nba_bets/outputs/llm/eval/nba-bets-2026-07-30 ./
```

**The adapter is already on the Hub** at `fformenti/<run-name>` — that's the deliverable. The only
other thing worth retrieving is `mlflow.db`, if you left MLflow enabled.

### Shut the pod down

- **Stop** keeps the network volume (still billed, ~$0.07/GB/month) and the container disk contents
  are lost.
- **Terminate** releases everything except the network volume.

Don't leave a 4090 idling overnight because evaluation finished at 2am.

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `RuntimeError: HF_llm_training_token is not set` | No `.env` on the pod, or the variable name is wrong (it is *not* `HF_TOKEN`) |
| `RuntimeError: WANDB_API_KEY is not set but tracking.wandb.enabled is true` | Add the key, or set `tracking.wandb.enabled: false` |
| `403` / "gated repo" when loading the model | Llama 3.1 license not accepted for the account owning your token. Preflight won't catch this — it only checks the dataset |
| `No CUDA device visible` | Wrong template, or `uv sync` pulled a CPU-only torch. Check `uv run python -c "import torch; print(torch.version.cuda)"` |
| `ModuleNotFoundError: peft` / `trl` / `bitsandbytes` | You ran `uv sync` without `--extra gpu` |
| CUDA OOM | Set `per_device_train_batch_size: 1` and `gradient_accumulation_steps: 8` to hold the effective batch size at 8. Failing that, lower `data.max_sequence_length` from 1024 |
| `element 0 of tensors does not require grad` | Gradient checkpointing interacting with the k-bit model. As a fallback set `training.gradient_checkpointing: false` (costs memory) |
| First step takes forever | 16 GB base-model download. Check `echo $HF_HOME` points at `/workspace` |
| `ValueError: completion_only_loss ... not supported for language modeling datasets` | The dataset has a `text` column, so trl took the language-modeling path. Rebuild it with `make build-llm-dataset` — the schema must be `game_id` / `prompt` / `completion` |
| `KeyError: Unexpected input keys in examples` | Same root cause, opposite symptom: neither a text field nor both of `prompt`/`completion` were found |

---

## Quick reference

```bash
# one-time setup on a fresh pod
export HF_HOME=/workspace/.cache/huggingface
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
cd /workspace && git clone <repo> && cd nba_bets && git checkout feature/train_LLM
uv sync --extra gpu

# smoke test
uv run python -m src.cli.train_llm --config configs/train_llm/llama31_8b_qlora.yaml \
  --run-name smoke-test --max-train-samples 100

# real run (max_train_samples: null in the config first)
tmux new -s train
make train-llm LLM_RUN=nba-bets-2026-07-30

# resume — identical command
make train-llm LLM_RUN=nba-bets-2026-07-30

# evaluate
make evaluate-llm LLM_RUN=nba-bets-2026-07-30
```
