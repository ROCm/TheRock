# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import io
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import package_dasan_stack as package


class InventoryTest(unittest.TestCase):
    def test_only_current_run_from_therock_is_accepted(self):
        with patch.dict("os.environ", {"GITHUB_RUN_ID": "123"}):
            package.validate_run_identity("123", "ROCm/TheRock")
            with self.assertRaisesRegex(RuntimeError, "own workflow run"):
                package.validate_run_identity("122", "ROCm/TheRock")
            with self.assertRaisesRegex(RuntimeError, "ROCm/TheRock"):
                package.validate_run_identity("123", "ROCm/rockrel")
            with self.assertRaisesRegex(RuntimeError, "numeric"):
                package.validate_run_identity("latest", "ROCm/TheRock")

    def test_requires_every_name_and_rejects_another_gpu(self):
        available = [
            "amd-llvm_run_generic.tar.zst",
            "rccl_lib_gfx942.tar.zst",
            "rccl_lib_gfx950.tar.zst.sha256sum",
            "blas_lib_gfx950-dcgpu.tar.zst",
        ]
        with self.assertRaisesRegex(RuntimeError, "rccl"):
            package.check_inventory(
                available, {"amd-llvm", "rccl", "blas"}, "gfx950-dcgpu"
            )
        available.append("rccl_lib_gfx950.tar.zst")
        available.append("rccl_lib_gfx950.tar.xz")
        selected = package.check_inventory(
            available, {"amd-llvm", "rccl", "blas"}, "gfx950-dcgpu"
        )
        self.assertEqual(len(selected), 3)
        self.assertNotIn("rccl_lib_gfx942.tar.zst", selected)
        self.assertNotIn("rccl_lib_gfx950.tar.xz", selected)

    def test_full_platform_inventory_includes_wsl_and_runtime_tests(self):
        required = package.required_artifacts(package.get_topology())
        self.assertTrue(
            {"wsl-rocdxg", "core-hiptests", "rocrtst", "rocjitsu", "rccl"} <= required
        )
        topology = SimpleNamespace(
            artifacts={"linux": object(), "windows": object(), "arm": object()},
        )
        linux, windows, arm = topology.artifacts.values()
        topology.is_artifact_disabled_on_platform = (
            lambda artifact, platform: artifact is windows
        )
        topology.is_artifact_disabled_on_processor = (
            lambda artifact, processor: artifact is arm
        )
        self.assertEqual(package.required_artifacts(topology), {"linux"})


class IntegrityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_checksum_proves_downloaded_bytes_and_accepts_standard_sidecar(self):
        archive = self.root / "payload.tar.zst"
        archive.write_bytes(b"complete payload")
        sidecar = self.root / "payload.tar.zst.sha256sum"
        digest = package.sha256(archive)
        for text in (digest, f"{digest}  {archive.name}\n"):
            sidecar.write_text(text)
            self.assertEqual(package.check_digest(archive, sidecar), digest)
        archive.write_bytes(b"truncated")
        with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            package.check_digest(archive, sidecar)
        sidecar.write_text("not a digest")
        with self.assertRaisesRegex(RuntimeError, "Invalid SHA256"):
            package.check_digest(archive, sidecar)

    def symlink(self, target, path, directory=False):
        try:
            path.symlink_to(target, target_is_directory=directory)
        except (OSError, NotImplementedError):
            self.skipTest("Creating symbolic links is unavailable")

    def sdk_aliases(self):
        (self.root / "lib/llvm/amdgcn").mkdir(parents=True)
        self.symlink("lib/llvm", self.root / "llvm", True)
        self.symlink("lib/llvm/amdgcn", self.root / "amdgcn", True)

    def test_sdk_aliases_cannot_escape_or_depend_on_build_tree(self):
        self.sdk_aliases()
        self.assertEqual(package.check_portable_links(self.root), 2)
        self.symlink("../../build-tree/librccl.so", self.root / "lib/librccl.so")
        with self.assertRaisesRegex(RuntimeError, "Nonportable"):
            package.check_portable_links(self.root)

    def test_missing_alias_and_dangling_link_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Missing portable SDK alias"):
            package.check_portable_links(self.root)
        self.sdk_aliases()
        self.symlink("missing.so", self.root / "lib/libmissing.so")
        with self.assertRaisesRegex(RuntimeError, "Dangling"):
            package.check_portable_links(self.root)

    def test_real_archive_roundtrip_has_flat_layout_and_executable_mode(self):
        archive = self.root / "sdk.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            entry = tarfile.TarInfo("bin/hipcc")
            entry.mode = 0o755
            content = b"#!/bin/sh\nexit 0\n"
            entry.size = len(content)
            stream.addfile(entry, io.BytesIO(content))
        output = self.root / "extracted"
        package.extract_sdk(archive, output)
        self.assertEqual((output / "bin/hipcc").read_bytes(), content)
        if sys.platform != "win32":
            self.assertEqual((output / "bin/hipcc").stat().st_mode & 0o777, 0o755)
        with self.assertRaisesRegex(RuntimeError, "clean directory"):
            package.extract_sdk(archive, output)

    def test_archive_cannot_write_outside_installation(self):
        archive = self.root / "malformed.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            entry = tarfile.TarInfo("../outside")
            entry.size = 1
            stream.addfile(entry, io.BytesIO(b"x"))
        with self.assertRaises(tarfile.OutsideDestinationError):
            package.extract_sdk(archive, self.root / "sdk")
        self.assertFalse((self.root / "outside").exists())


if __name__ == "__main__":
    unittest.main()
