# dpo-train — MLX DPO training sidecar (local-only)

The training engine for the Conduct code-gen fine-tuning flywheel
([gotoplanb/conduct#45](https://github.com/gotoplanb/conduct/issues/45)).

**Native macOS daemon on the M5, like ComfyUI / ACE-Step** — not a Docker
service (MLX needs Metal GPU access, which Docker-on-Mac doesn't expose). The
Conduct worker reaches it at `host.docker.internal:8077`. The **heavy ML stack
lives here, never in Conduct**: Conduct's `dpo_fine_tune` task type is a thin
HTTP client that POSTs preference pairs; this daemon runs the DPO training.

Wraps **[mlx-tune](https://github.com/ARahim3/mlx-tune)** (DPOTrainer + LoRA +
merged-safetensors export). Conduct owns the contract; this daemon implements it.

## Contract

```
POST /train
  {"base_model": "<HF repo id>",            # FastLanguageModel loads from HF
   "pairs": [{"prompt","chosen","rejected", ...}, ...],
   "training": {epochs, lora_rank, lora_alpha, beta, learning_rate, max_seq_length}?}
->
  {"tag": "<base>-dpo-<sha8>",              # registered as an Ollama tag
   "artifact_path": "<merged safetensors dir>",
   "pairs_consumed": N, "training_time_s": T, "dataset_sha": "..."}

409 if a run is already in flight (one per host — training saturates memory)
400 bad input · 500 training failure · GET /health -> 200 "ok"
```

`base_model` is a **HuggingFace repo id** (e.g. `mlx-community/gemma-3-4b-it`),
not an Ollama tag — mlx-tune loads from HF. Conduct is model-source-agnostic;
the client picks it.

## Why safetensors → Ollama, not GGUF

mlx-tune's `export_to_gguf` (via `mlx_lm.fuse`) is **only supported for
Llama/Mistral/Mixtral** — and the flywheel target is **Gemma**. So after DPO we
save merged 16-bit safetensors and `ollama create <tag> -f Modelfile` with
`FROM <merged dir>`; Ollama imports safetensors directly for Gemma/Llama/Qwen/…,
which is architecture-agnostic where GGUF is not. The resulting tag is then
servable by Conduct's normal routing/swap path.

## Run / test

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python mlx-tune
.venv/bin/python -m unittest test_app        # HTTP + contract (no MLX/training)
PORT=8077 .venv/bin/python app.py            # the daemon
curl localhost:8077/health                    # -> ok
```

## Status / TODO

- ✅ Daemon + contract + HTTP layer (unit-tested, trainer stubbed).
- ✅ DPO pipeline written against the real mlx-tune API (from_pretrained →
  get_peft_model → prepare_preference_dataset → DPOTrainer → save merged →
  ollama create).
- ⏳ **Not yet run end-to-end** against a real base model (a real DPO run is GB
  download + minutes of compute). First live smoke: a small Gemma on a synthetic
  preference set, confirming the merged-safetensors → `ollama create` path works
  for Gemma. Pick the base model deliberately (don't guess the HF id).
- Open (per conduct#45): does this become its own git repo? concurrency beyond
  one-per-host? cost schema in `metadata.training`?

## Threat model

Trains arbitrary preference data into a model on owned hardware — local-only,
never cloud. One run at a time (memory). `ollama create` runs locally.

## Run as a managed service (launchd)

The sidecar should run as a user LaunchAgent (auto-start at login, auto-restart
on crash — Metal OOMs hard-abort the process, so KeepAlive matters). A user
agent (not a root daemon) is required: MLX needs the GUI user session for Metal.

```bash
cp launchd/com.gotoplanb.dpo-train.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.gotoplanb.dpo-train.plist
launchctl enable  gui/$(id -u)/com.gotoplanb.dpo-train
curl localhost:8077/health   # -> ok
```

Manage it:
```bash
launchctl print    gui/$(id -u)/com.gotoplanb.dpo-train   # status + pid
launchctl kickstart -k gui/$(id -u)/com.gotoplanb.dpo-train  # restart
launchctl bootout  gui/$(id -u)/com.gotoplanb.dpo-train   # stop/unload
```
Logs: `logs/sidecar.out.log`, `logs/sidecar.err.log` (gitignored).
