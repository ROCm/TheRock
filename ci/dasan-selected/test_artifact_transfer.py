"""Focused transfer checks; no GitHub access or large fixtures required."""
import hashlib
import http.client
import io
import json
import sys
import tempfile
import unittest
import urllib.error
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import consume_dasan_artifacts as consumer


def make_zip(contents):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("payload.txt", contents)
    return buffer.getvalue()


class Response:
    def __init__(self, chunks, status=200, headers=None):
        self.chunks = iter(chunks)
        self.status = status
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def getcode(self):
        return self.status

    def read(self, _):
        chunk = next(self.chunks, b"")
        if isinstance(chunk, Exception):
            raise chunk
        return chunk


class TransferTests(unittest.TestCase):
    def run_transfer(self, expected, responses):
        with tempfile.TemporaryDirectory() as folder:
            partial = Path(folder) / "artifact.partial"
            evidence = Path(folder) / "attempts.json"
            with patch.object(consumer, "artifact_response", side_effect=responses) as request, \
                 patch.object(consumer.time, "sleep"):
                try:
                    consumer.transfer_archive("https://api.github.com/artifact", "token", partial,
                                              expected, 123, evidence)
                    result = "verified"
                except (ValueError, RuntimeError) as exc:
                    result = str(exc)
                attempts = json.loads(evidence.read_text())["attempts"]
                data = partial.read_bytes() if partial.exists() else None
            return result, attempts, data, request.call_args_list

    def test_interrupted_transfer_resumes_and_verifies_full_digest(self):
        archive = make_zip(b"payload")
        cut = len(archive) // 2
        expected = hashlib.sha256(archive).hexdigest()
        responses = [Response([archive[:cut], http.client.IncompleteRead(b"", 1)]),
                     Response([archive[cut:]], 206,
                              {"Content-Range": f"bytes {cut}-{len(archive)-1}/{len(archive)}",
                               "Content-Length": str(len(archive)-cut)})]
        result, attempts, data, calls = self.run_transfer(expected, responses)
        self.assertEqual(result, "verified")
        self.assertEqual(data, archive)
        self.assertEqual([a["result"] for a in attempts], ["transfer_error", "verified"])
        self.assertEqual(calls[1].args[-1], cut)

    def test_complete_zip_with_wrong_digest_fails_without_repeating_download(self):
        archive = make_zip(b"valid ZIP, different from pin")
        expected = hashlib.sha256(make_zip(b"pinned ZIP")).hexdigest()
        result, attempts, data, calls = self.run_transfer(expected, [Response([archive])])
        self.assertIn("digest mismatch", result)
        self.assertIsNone(data)
        self.assertEqual(len(calls), 1)
        self.assertEqual(attempts[0]["result"], "complete_zip_digest_mismatch")
        self.assertEqual(attempts[0]["sha256"], hashlib.sha256(archive).hexdigest())

    def test_range_ignored_restarts_instead_of_appending(self):
        archive = make_zip(b"payload")
        cut = len(archive) // 2
        expected = hashlib.sha256(archive).hexdigest()
        responses = [Response([archive[:cut], http.client.IncompleteRead(b"", 1)]),
                     Response([archive], 200, {"Content-Length": str(len(archive))})]
        result, attempts, data, _ = self.run_transfer(expected, responses)
        self.assertEqual(result, "verified")
        self.assertEqual(data, archive)
        self.assertFalse(attempts[1]["resumed"])

    def test_corrupt_zip_retries_fresh_then_verifies(self):
        archive = make_zip(b"payload")
        expected = hashlib.sha256(archive).hexdigest()
        result, attempts, data, calls = self.run_transfer(
            expected, [Response([b"corrupt response"]), Response([archive])])
        self.assertEqual(result, "verified")
        self.assertEqual(data, archive)
        self.assertEqual([a["result"] for a in attempts],
                         ["corrupt_zip_digest_mismatch", "verified"])
        self.assertEqual(calls[1].args[-1], 0)

    def test_terminal_network_error_does_not_record_signed_url(self):
        signed_url = "https://storage.example/artifact.zip?secret=sensitive-token"
        failures = [urllib.error.URLError(signed_url) for _ in range(3)]
        result, attempts, data, calls = self.run_transfer("0" * 64, failures)
        self.assertEqual(result, "Artifact 123 download failed (URLError)")
        self.assertEqual(len(calls), 3)
        self.assertIsNone(data)
        self.assertNotIn(signed_url, json.dumps(attempts))
        self.assertEqual([a["error"] for a in attempts], ["URLError"] * 3)


if __name__ == "__main__":
    unittest.main()
