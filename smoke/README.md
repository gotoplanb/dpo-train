# Smoke harnesses

Throwaway end-to-end validators for the sidecar (drive `app.run_dpo` directly).

- `adventure_smoke.py` — DPO `mlx-community/Llama-3.2-3B-Instruct-4bit` on synthetic
  adventure-narration preference pairs. **Verified 2026-06-22**: trains → merges →
  `ollama create` → serves **clean** (no UNK_BYTE / `▁` artifacts). The base for the
  adventure-game-engine DPO.
- `code_smoke.py` — earlier code-flavored variant. Note: `google/gemma-4-E4B` is
  multimodal (can't text-DPO); Gemma merges also mangle the SentencePiece vocab on
  Ollama import — prefer Llama/BPE bases.

Run: `.venv/bin/python smoke/adventure_smoke.py` (needs `hf` auth + a few minutes).
