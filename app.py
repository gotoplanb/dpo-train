#!/usr/bin/env python3
"""dpo-train: a native-macOS DPO training sidecar wrapping mlx-tune (#45).

The heavy ML stack lives here, NOT in Conduct. Conduct's dpo_fine_tune task type
is a thin HTTP client (codegen/train_client.py) that POSTs preference pairs to
this daemon; the daemon runs the mlx-tune DPO pipeline on Apple-Silicon unified
memory, exports a GGUF, registers it as an Ollama tag, and returns the tag.

Same pattern as ComfyUI / ACE-Step: a native Mac service Conduct reaches at
host.docker.internal:<port>. Native (not Docker) because MLX needs Metal.

Stdlib HTTP only (no web framework). The mlx-tune import is lazy — inside the
worker — so the module imports (and /health responds) without MLX present, and
so the HTTP layer is unit-testable with the trainer stubbed.

Contract (owned by Conduct, conduct#45):
  POST /train
    {"base_model": "<HF repo id>",            # from_pretrained loads from HF
     "pairs": [{"prompt","chosen","rejected", ...}, ...],
     "training": {epochs, lora_rank, lora_alpha, beta, learning_rate}?}
  -> {"tag", "gguf_path", "pairs_consumed", "training_time_s", "dataset_sha"}
  -> 409 if a training run is already in flight (one per host)
  -> 400 on bad input, 500 on training failure
  GET /health -> 200 "ok"
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess  # noqa: S404 - registers the GGUF as an Ollama tag (trusted argv)
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

OUTPUT_DIR = Path(os.environ.get("DPO_OUTPUT_DIR", str(Path.home() / "dpo-train" / "out")))
MAX_PAIRS = 50_000
# One training run at a time per host — training saturates unified memory, so a
# second concurrent run would thrash. Excess /train calls get 409.
_train_lock = threading.Lock()


class TrainError(ValueError):
    """Bad request — returned as 400, never crashes the daemon."""


def _dataset_sha(pairs: list[dict]) -> str:
    h = hashlib.sha256()
    for p in pairs:
        h.update(json.dumps([p.get("prompt", ""), p.get("chosen", ""), p.get("rejected", "")],
                            sort_keys=True).encode())
    return h.hexdigest()


def _tag_for(base_model: str, sha: str) -> str:
    """A servable Ollama tag derived from the base + dataset hash. Drop the HF
    org prefix first (last path segment), then slugify."""
    short = re.sub(r"[^a-z0-9]+", "-", base_model.split("/")[-1].lower()).strip("-")
    return f"{short}-dpo-{sha[:8]}"


def _validate(payload: dict) -> tuple[str, list[dict], dict]:
    base_model = payload.get("base_model")
    if not isinstance(base_model, str) or not base_model:
        raise TrainError("base_model (a HF repo id) is required")
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise TrainError("pairs must be a non-empty list")
    if len(pairs) > MAX_PAIRS:
        raise TrainError(f"too many pairs: {len(pairs)} > {MAX_PAIRS}")
    for p in pairs:
        if not (isinstance(p, dict) and p.get("chosen") and p.get("rejected")):
            raise TrainError("each pair needs non-empty 'chosen' and 'rejected'")
    return base_model, pairs, payload.get("training") or {}


def _ollama_register(tag: str, model_path: str) -> None:
    """Make the trained model servable under `tag` via Ollama (the open-question
    answer in conduct#45: the sidecar owns artifact -> servable tag).

    We register the **merged safetensors** directory, not GGUF: mlx_lm's GGUF
    export only supports Llama/Mistral/Mixtral, and the flywheel target is Gemma.
    Ollama imports safetensors directly (`FROM <dir>`) for Gemma/Llama/Qwen/…,
    so this path is architecture-agnostic where GGUF is not."""
    with tempfile.TemporaryDirectory() as td:
        modelfile = Path(td) / "Modelfile"
        modelfile.write_text(f"FROM {model_path}\n")
        subprocess.run(  # noqa: S603 - argv is constructed, not shell
            ["ollama", "create", tag, "-f", str(modelfile)],  # noqa: S607
            check=True, capture_output=True, text=True, timeout=900,
        )


def run_dpo(base_model: str, pairs: list[dict], training: dict) -> dict:
    """The real mlx-tune DPO pipeline. Heavy import is local so the module loads
    without MLX. Returns the result dict matching Conduct's contract."""
    import mlx_tune as mt  # noqa: PLC0415 - lazy: keeps /health + tests MLX-free

    sha = _dataset_sha(pairs)
    tag = _tag_for(base_model, sha)
    started = time.monotonic()

    model, tokenizer = mt.FastLanguageModel.from_pretrained(
        base_model, max_seq_length=int(training.get("max_seq_length", 2048)),
    )
    model = mt.FastLanguageModel.get_peft_model(
        model, r=int(training.get("lora_rank", 16)),
        lora_alpha=int(training.get("lora_alpha", 32)),
    )
    train_ds = mt.prepare_preference_dataset(pairs, tokenizer, "dpo")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged_dir = OUTPUT_DIR / f"{tag}-merged"
    cfg = mt.DPOConfig(
        beta=float(training.get("beta", 0.1)),
        learning_rate=float(training.get("learning_rate", 1e-5)),
        num_train_epochs=int(training.get("epochs", 1)),
        output_dir=str(OUTPUT_DIR / f"{tag}-run"),
    )
    mt.DPOTrainer(model, train_ds, tokenizer=tokenizer, args=cfg).train()
    # Merged safetensors (not GGUF — see _ollama_register), then register.
    mt.save_model_hf_format(model, tokenizer, str(merged_dir), save_method="merged_16bit")
    _ollama_register(tag, str(merged_dir))

    return {
        "tag": tag, "artifact_path": str(merged_dir), "pairs_consumed": len(train_ds),
        "training_time_s": round(time.monotonic() - started, 1), "dataset_sha": sha,
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self._send(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/train":
            self._send(404, {"error": "not_found"})
            return
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        try:
            base_model, pairs, training = _validate(json.loads(raw))
        except (ValueError, TypeError) as e:
            self._send(400, {"error": "bad_request", "detail": str(e)[:300]})
            return
        if not _train_lock.acquire(blocking=False):
            self._send(409, {"error": "busy", "detail": "a training run is already in flight"})
            return
        try:
            self._send(200, run_dpo(base_model, pairs, training))
        except subprocess.CalledProcessError as e:  # ollama create failed
            self._send(500, {"error": "ollama_register_failed", "detail": (e.stderr or "")[:300]})
        except Exception as e:  # noqa: BLE001 - one bad run must not kill the daemon
            self._send(500, {"error": "training_failed", "detail": str(e)[:300]})
        finally:
            _train_lock.release()

    def log_message(self, *_args) -> None:  # quiet per-request logging
        pass


def main() -> None:
    port = int(os.environ.get("PORT", "8077"))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()  # noqa: S104


if __name__ == "__main__":
    main()
