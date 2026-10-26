# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Object-store compatibility and managed local cold/warm cache tests.

The managed tests run a miniature CMake/Ninja fixture, not a ROCm build. Each
test uses an explicitly initialized private relational database.
"""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from build_tools.github_actions.artifact_cache.inventory import scan_build
from build_tools.github_actions.artifact_cache.local_store import LocalObjectStore, sha256_file
from build_tools.github_actions.tests import _artifact_cache_fixtures as fixtures


class LocalObjectStoreTest(unittest.TestCase):
    def test_upload_exists_digest_download_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "artifact.tar"
            source.write_bytes(b"hello artifact")
            store = LocalObjectStore(root / "store")
            obj = store.upload_file(source)
            self.assertTrue(store.exists(obj.uri))
            self.assertEqual(obj.digest, sha256_file(source))
            self.assertEqual(store.fetch_digest(obj.uri), obj.digest)
            restored = store.download_file(obj.uri, root / "download.tar")
            self.assertEqual(restored.read_bytes(), source.read_bytes())

    def test_missing_object(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LocalObjectStore(Path(directory) / "store")
            self.assertFalse(store.exists("local://missing.tar"))
            self.assertIsNone(store.fetch_digest("local://missing.tar"))


class LocalCacheScanTest(fixtures.RelationalFixture):
    def test_missing_fingerprint_is_uncacheable_without_guessing_its_cause(self):
        self.outputs(("base_lib_generic", fixtures.FPRINT), ("miopen_lib_gfx94X-dcgpu", None))
        (self.artifacts / "miopen_lib_gfx94X-dcgpu").mkdir()
        records = {record["artifact_identity"]: (record, key)
                   for record, key in scan_build(self.artifacts.parent, self.manifest, granularity="bundle")}
        self.assertIsNotNone(records["base/bundle/generic"][1])
        record, key = records["miopen/bundle/gfx94X-dcgpu"]
        self.assertIsNone(key)
        self.assertEqual(record["fingerprint_state"], "missing")


@unittest.skipUnless(all(shutil.which(tool) for tool in ("cmake", "ninja", "git")),
                     "requires CMake, Ninja and Git")
class LocalCacheReuseTest(fixtures.RelationalFixture):
    def command(self, *args):
        result = subprocess.run(
            [sys.executable, "-m", "build_tools.github_actions.artifact_cache.local_cache", *args],
            cwd=REPO, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(result.stdout)

    def synth(self, change=None):
        args = ["synth", "--out", str(self.root / "fixture-build")]
        if change:
            args += ["--change", change]
        self.command(*args)

    def cache(self):
        return self.command("cache", "--build", str(self.root / "fixture-build"),
                            "--db", str(self.database), "--cache", str(self.root / "objects"),
                            "--variant", "release", "--granularity", "bundle")

    def test_build_once_stores_then_verifies_without_building_again(self):
        self.synth()
        cold, warm = self.cache(), self.cache()
        self.assertTrue(cold["build_executed"])
        self.assertEqual(cold["summary"], {"stored": 5, "verified_cached": 0, "uncacheable": 1})
        self.assertFalse(warm["build_executed"])
        self.assertEqual(warm["summary"], {"stored": 0, "verified_cached": 5, "uncacheable": 1})
        self.assertTrue(all(row["payloads_verified"] for row in warm["artifacts"]
                            if row["outcome"] == "verified_cached"))

    def test_source_change_stores_only_changed_unit_after_managed_build(self):
        self.synth()
        self.cache()
        self.synth(change="blas")
        changed = self.cache()
        self.assertTrue(changed["build_executed"])
        self.assertEqual(changed["summary"], {"stored": 1, "verified_cached": 4, "uncacheable": 1})
        self.assertEqual([row["artifact_identity"] for row in changed["artifacts"]
                          if row["outcome"] == "stored"], ["blas/bundle/gfx94X-dcgpu"])
        warm = self.cache()
        self.assertFalse(warm["build_executed"])
        self.assertEqual(warm["summary"]["verified_cached"], 5)


if __name__ == "__main__":
    unittest.main()
