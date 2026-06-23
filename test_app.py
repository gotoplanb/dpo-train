"""Stdlib unittest for the dpo-train sidecar HTTP + contract layer (#45).

No MLX / no real training — run_dpo is the heavy part (lazy mlx_tune import) and
is stubbed here. The real DPO pipeline is validated by a live training run.

Run: python3 -m unittest test_app
"""

from __future__ import annotations

import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import app  # noqa: E402


class ContractTests(unittest.TestCase):
    def test_dataset_sha_is_stable_and_content_sensitive(self):
        a = [{"prompt": "p", "chosen": "c", "rejected": "r"}]
        self.assertEqual(app._dataset_sha(a), app._dataset_sha(list(a)))
        self.assertNotEqual(app._dataset_sha(a), app._dataset_sha(
            [{"prompt": "p", "chosen": "c2", "rejected": "r"}]))

    def test_tag_derivation(self):
        tag = app._tag_for("mlx-community/gemma-3-4b-it", "deadbeef0000")
        self.assertTrue(tag.startswith("gemma-3-4b-it-dpo-"))
        self.assertTrue(tag.endswith("deadbeef"))

    def test_ollama_bin_resolves_without_path(self):
        # conduct#46: under launchd the PATH is minimal so a bare 'ollama' isn't
        # found. The resolver must still return an absolute, existing binary.
        import os
        from unittest import mock
        with mock.patch.dict(os.environ, {"PATH": "/nonexistent"}, clear=False):
            os.environ.pop("OLLAMA_BIN", None)
            resolved = app._ollama_bin()
        # either a real install location, or the documented last-resort bare name
        self.assertTrue(resolved == "ollama" or Path(resolved).is_absolute())

    def test_ollama_bin_honors_override(self):
        import os
        from unittest import mock
        with mock.patch.dict(os.environ, {"OLLAMA_BIN": "/custom/ollama"}, clear=False):
            self.assertEqual(app._ollama_bin(), "/custom/ollama")

    def test_validate_rejects_missing_base_model(self):
        with self.assertRaises(app.TrainError):
            app._validate({"pairs": [{"chosen": "c", "rejected": "r"}]})

    def test_validate_rejects_empty_pairs(self):
        with self.assertRaises(app.TrainError):
            app._validate({"base_model": "m", "pairs": []})

    def test_validate_rejects_pair_without_chosen_rejected(self):
        with self.assertRaises(app.TrainError):
            app._validate({"base_model": "m", "pairs": [{"prompt": "p"}]})

    def test_validate_passes_good_payload(self):
        bm, pairs, tr = app._validate({
            "base_model": "m", "pairs": [{"chosen": "c", "rejected": "r"}],
            "training": {"epochs": 2},
        })
        self.assertEqual(bm, "m")
        self.assertEqual(len(pairs), 1)
        self.assertEqual(tr["epochs"], 2)


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def _post(self, body):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/train",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(req, timeout=10)

    def test_health(self):
        r = urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=5)
        self.assertEqual(r.read(), b"ok")

    def test_bad_request_is_400(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self._post({"pairs": []})  # missing base_model
        self.assertEqual(cm.exception.code, 400)

    def test_train_success_path_stubbed(self):
        app.run_dpo = lambda base_model, pairs, training: {  # type: ignore[assignment]
            "tag": "m-dpo-abc", "artifact_path": "/out/x.gguf",
            "pairs_consumed": len(pairs), "training_time_s": 1.0, "dataset_sha": "abc",
        }
        r = self._post({"base_model": "m", "pairs": [{"chosen": "c", "rejected": "r"}]})
        out = json.loads(r.read())
        self.assertEqual(out["tag"], "m-dpo-abc")
        self.assertEqual(out["pairs_consumed"], 1)

    def test_busy_returns_409(self):
        # Hold the lock to simulate an in-flight run; a second /train -> 409.
        app._train_lock.acquire()
        try:
            with self.assertRaises(urllib.error.HTTPError) as cm:
                self._post({"base_model": "m", "pairs": [{"chosen": "c", "rejected": "r"}]})
            self.assertEqual(cm.exception.code, 409)
        finally:
            app._train_lock.release()


if __name__ == "__main__":
    unittest.main()
