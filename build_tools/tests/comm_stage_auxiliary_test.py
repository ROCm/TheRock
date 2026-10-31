# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import comm_stage_auxiliary as auxiliary
import comm_stage_evidence as evidence
from _therock_utils.archive_util import open_archive_for_read


class AuxiliaryEvidenceTests(unittest.TestCase):
    def fixture(self, root):
        build = root / "build"
        output = root / "evidence"
        for index, relative in enumerate(auxiliary.ARCHIVES):
            path = build / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"!<arch>\n" + bytes([index]) * 257)
            os.utime(path, ns=(1_700_000_000_123456789, 1_700_000_000_123456789))
        resource = (
            build
            / "comm-libs/rccl/build/src/CMakeFiles/rccl_device_gfx900.dir/__/hipify/gensrc"
            / "specialized/specialized_allreduce_tree_simple_sum_f8e4m3_0_0_4.resources.json"
        )
        resource.parent.mkdir(parents=True)
        resource.write_bytes(b'{"vgpr_count": 16, "sgpr_count": 32}\n')
        return build, output, resource

    def test_actual_compressed_payload_metadata_and_inputs_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            build, output, resource = self.fixture(Path(temporary))
            # Actual retained RCCL compile -o path, not an invented sidecar
            # directory. The driver writes splitext(output)[0]+'.resources.json'.
            self.assertFalse((build / "comm-libs/rccl/build/device_build").exists())
            paths = [build / n for n in auxiliary.ARCHIVES] + [resource]
            before = {p: (p.read_bytes(), auxiliary._identity(p)) for p in paths}
            summary = auxiliary.retain(build, output, time.monotonic() + 30)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["dependency_archives"], 6)
            self.assertEqual(summary["rccl_resource_sidecars"], 1)
            self.assertEqual(summary["verified_payload_members"], 7)
            manifest = {
                row["path"]: row
                for row in map(
                    json.loads,
                    (output / "dependency-archives-and-rccl-resources.jsonl")
                    .read_text()
                    .splitlines(),
                )
            }
            with open_archive_for_read(
                output / "dependency-archives-and-rccl-resources.tar.zst"
            ) as archive:
                for member in archive:
                    with archive.extractfile(member) as stream:
                        data = stream.read()
                    source = build / member.name
                    self.assertEqual(data, before[source][0])
                    self.assertEqual(
                        hashlib.sha256(data).hexdigest(),
                        manifest[member.name]["sha256"],
                    )
                    ns = manifest[member.name]["mtime_ns"]
                    self.assertEqual(
                        member.pax_headers["mtime"],
                        f"{ns // 10**9}.{ns % 10**9:09d}",
                    )
            self.assertIn(resource.relative_to(build).as_posix(), manifest)
            for path in paths:
                self.assertEqual(
                    before[path], (path.read_bytes(), auxiliary._identity(path))
                )
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                auxiliary.retain(build, output, time.monotonic() + 30)

    def test_source_mutation_invalidates_and_preserves_partial_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            build, output, _ = self.fixture(Path(temporary))
            original = evidence.sha256

            def mutate_after_hash(path, deadline):
                result = original(path, deadline)
                if path == build / auxiliary.ARCHIVES[0]:
                    path.write_bytes(path.read_bytes() + b"changed")
                return result

            with patch.object(evidence, "sha256", side_effect=mutate_after_hash):
                with self.assertRaisesRegex(RuntimeError, "changed during capture"):
                    auxiliary.retain(build, output, time.monotonic() + 30)
            summary = json.loads(
                (
                    output / "dependency-archives-and-rccl-resources-summary.json"
                ).read_text()
            )
            self.assertFalse(summary["complete"])
            self.assertTrue(
                (
                    output / "dependency-archives-and-rccl-resources.partial.tar.zst"
                ).is_file()
            )
            self.assertFalse(
                (output / "dependency-archives-and-rccl-resources.tar.zst").exists()
            )

    def test_unchanged_reserve_rejects_before_archive_allocation(self):
        with tempfile.TemporaryDirectory() as temporary:
            build, output, _ = self.fixture(Path(temporary))
            usage = shutil._ntuple_diskusage(100 * 1024**3, 93 * 1024**3, 7 * 1024**3)
            with patch.object(auxiliary.shutil, "disk_usage", return_value=usage):
                with self.assertRaisesRegex(RuntimeError, "disk reserve"):
                    auxiliary.retain(build, output, time.monotonic() + 30)
            self.assertFalse(
                (
                    output / "dependency-archives-and-rccl-resources.partial.tar.zst"
                ).exists()
            )

    def test_missing_archive_cannot_produce_complete_case(self):
        with tempfile.TemporaryDirectory() as temporary:
            build, output, _ = self.fixture(Path(temporary))
            (build / auxiliary.ARCHIVES[0]).unlink()
            with self.assertRaises(FileNotFoundError):
                auxiliary.retain(build, output, time.monotonic() + 30)
            self.assertFalse(
                json.loads(
                    (
                        output / "dependency-archives-and-rccl-resources-summary.json"
                    ).read_text()
                )["complete"]
            )

    def test_missing_sidecars_cannot_produce_complete_case(self):
        with tempfile.TemporaryDirectory() as temporary:
            build, output, resource = self.fixture(Path(temporary))
            resource.unlink()
            with self.assertRaisesRegex(RuntimeError, "No RCCL resource"):
                auxiliary.retain(build, output, time.monotonic() + 30)

    @unittest.skipUnless(sys.platform == "linux", "Linux CI symlink semantics")
    def test_symlink_sidecar_cannot_escape_owned_build(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build, output, resource = self.fixture(root)
            outside = root / "outside.json"
            outside.write_bytes(resource.read_bytes())
            resource.unlink()
            resource.symlink_to(outside)
            with self.assertRaisesRegex(RuntimeError, "aliases or escapes"):
                auxiliary.retain(build, output, time.monotonic() + 30)


if __name__ == "__main__":
    unittest.main()
