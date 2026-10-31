# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Exercise real Git checkout conversion and the unchanged raw-byte input gate."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "normal_identity", ROOT / "build_tools/normal_ci_input_identity.py"
)
identity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(identity)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DiagnosticCheckoutTest(unittest.TestCase):
    def test_autocrlf_checkout_preserves_only_declared_payloads_and_raw_hash_gate(self):
        manifest_path = ROOT / "build_tools/normal_ci_declared_inputs.json"
        manifest = json.loads(manifest_path.read_text())
        entries = manifest["patches"]
        self.assertEqual(len(entries), 4)
        with tempfile.TemporaryDirectory() as raw:
            source, checkout = Path(raw) / "source", Path(raw) / "checkout"
            source.mkdir()
            subprocess.run(["git", "init", "--quiet", str(source)], check=True)
            for entry in entries:
                path = source / entry["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                # Verify the actual raw payload before adding it to the fixture.
                # Never normalize working bytes to manufacture the declared hash.
                blob = (ROOT / entry["path"]).read_bytes()
                self.assertNotIn(b"\r\n", blob)
                path.write_bytes(blob)
                self.assertEqual(digest(path), entry["sha256"])
            (source / ".gitattributes").write_bytes(
                (ROOT / ".gitattributes").read_bytes()
            )
            control = source / "unselected-control.txt"
            control.write_bytes(b"ordinary text\nsecond line\n")
            subprocess.run(
                ["git", "-C", str(source), "-c", "core.autocrlf=false", "add", "."],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(source),
                    "-c",
                    "user.name=Fixture",
                    "-c",
                    "user.email=fixture@example.invalid",
                    "commit",
                    "--quiet",
                    "-m",
                    "Inert checkout fixture",
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    "git",
                    "-c",
                    "core.autocrlf=true",
                    "clone",
                    "--quiet",
                    "--no-hardlinks",
                    str(source),
                    str(checkout),
                ],
                check=True,
                capture_output=True,
            )
            for entry in entries:
                path = checkout / entry["path"]
                self.assertNotIn(b"\r\n", path.read_bytes())
                self.assertEqual(digest(path), entry["sha256"])
            self.assertEqual(
                (checkout / control.name).read_bytes(),
                b"ordinary text\r\nsecond line\r\n",
            )
            receipt = identity.declared_sources(
                checkout,
                manifest_path,
                "baseline",
                digest,
                "wsl-rocdxg",
                "",
                "leading-build-stack",
            )
            self.assertTrue(receipt["valid"])
            self.assertEqual(
                sum(
                    row.get("matches_declared_sha256") is True
                    for row in receipt["files"]
                ),
                4,
            )
            # The fix controls checkout; it never normalizes away a wrong raw payload.
            altered = checkout / entries[0]["path"]
            altered.write_bytes(altered.read_bytes().replace(b"\n", b"\r\n"))
            self.assertFalse(
                identity.declared_sources(
                    checkout,
                    manifest_path,
                    "baseline",
                    digest,
                    "wsl-rocdxg",
                    "",
                    "leading-build-stack",
                )["valid"]
            )


if __name__ == "__main__":
    unittest.main()
