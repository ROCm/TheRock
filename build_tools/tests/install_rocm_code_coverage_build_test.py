#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for install_rocm_code_coverage_build.py.

Focuses on the generic-vs-instrumented repo split: the generic install is keyed
on --run-github-repo while the instrumented replacement fetch must be keyed on
--code-coverage-run-github-repo (defaulting to $GITHUB_REPOSITORY).
"""

import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import install_rocm_code_coverage_build as mod
from kpack_archive_test import HIPRAND, ROCRAND, read_kernels, write_reference_kpack


class _FakeBackend:
    """Minimal ArtifactBackend stand-in sufficient for the replacement fetch."""

    def __init__(self):
        self.base_uri = "s3://fake-bucket/fake"

    def list_artifacts(self, name_filter=None):
        # One instrumented blas tar for the requested family.
        return ["blas_lib_generic.tar.xz"]


class TestCodeCoverageRepoSplit(unittest.TestCase):
    def _run_main(self, extra_argv, env=None):
        """Run main() with all network boundaries mocked, capturing kwargs.

        Returns the github_repository kwarg passed to create_backend_from_env
        for the instrumented replacement fetch.
        """
        captured = {}

        def fake_create_backend(**kwargs):
            captured["github_repository"] = kwargs.get("github_repository")
            captured["run_id"] = kwargs.get("run_id")
            return _FakeBackend()

        def fake_find_available(artifact_names, target_families, available):
            # Pretend the single listed artifact matches.
            return list(available)

        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                "--code-coverage-run-id",
                "222",
                "--replace-rocblas",
                "--artifact-group",
                "gfx94X-dcgpu",
                "--output-dir",
                tmp,
                "--run-id",
                "111",
                "--run-github-repo",
                "ROCm/rocm-libraries",
            ] + extra_argv

            patched_env = env or {}
            with (
                mock.patch.object(mod, "install_from_artifacts_main"),
                mock.patch.object(
                    mod, "create_backend_from_env", side_effect=fake_create_backend
                ),
                mock.patch.object(
                    mod, "find_available_artifacts", side_effect=fake_find_available
                ),
                mock.patch.object(
                    mod, "download_artifact", return_value=Path(tmp) / "x"
                ),
                # No archives to extract -> replace step is a no-op.
                mock.patch.object(mod, "replace_instrumented_libraries"),
                mock.patch.dict(os.environ, patched_env, clear=False),
            ):
                mod.main(argv)
        return captured

    def test_replacement_uses_code_coverage_repo_not_generic_repo(self):
        captured = self._run_main(["--code-coverage-run-github-repo", "ROCm/TheRock"])
        self.assertEqual(captured["run_id"], "222")
        self.assertEqual(captured["github_repository"], "ROCm/TheRock")
        self.assertNotEqual(captured["github_repository"], "ROCm/rocm-libraries")

    def test_replacement_repo_defaults_to_github_repository_env(self):
        # main() must be re-imported so the argparse default picks up the env,
        # since the default is evaluated at parse time inside main().
        captured = self._run_main([], env={"GITHUB_REPOSITORY": "ROCm/TheRock"})
        self.assertEqual(captured["github_repository"], "ROCm/TheRock")

    def test_instrumented_binaries_are_normalized_after_the_swap(self):
        with mock.patch.object(
            mod, "normalize_instrumented_binaries", return_value=[]
        ) as normalize:
            self._run_main([])

        (roots,), _ = normalize.call_args
        self.assertEqual([root.name for root in roots], ["lib", "bin"])

    def test_device_code_is_swapped_only_when_asked(self):
        for extra, expected in (([], False), (["--replace-device-code"], True)):
            with self.subTest(extra=extra):
                with mock.patch.object(
                    mod, "overlay_instrumented_device_code"
                ) as overlay:
                    self._run_main(extra)
                self.assertEqual(overlay.called, expected)


class TestDeviceCodeOverlay(unittest.TestCase):
    KPACK = ".kpack/rand_lib_gfx942.kpack"

    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.root = Path(self._temp_dir.name)
        self.dest_dir = self.root / "code-coverage-replacements"
        self.output_dir = self.root / "build"
        self.dest_dir.mkdir()
        (self.output_dir / ".kpack").mkdir(parents=True)

    def _write_instrumented_artifact(self, kernels):
        """Writes rand_lib_gfx942 as the split leaves it: only the kpack."""
        prefix = "math-libs/rocRAND/stage"
        staging = self.root / "staging"
        (staging / ".kpack").mkdir(parents=True)
        write_reference_kpack(staging / self.KPACK, kernels)
        (staging / "artifact_manifest.txt").write_text(prefix + "\n")
        with tarfile.open(self.dest_dir / "rand_lib_gfx942.tar.xz", "w:xz") as tf:
            tf.add(staging / "artifact_manifest.txt", arcname="artifact_manifest.txt")
            tf.add(staging / self.KPACK, arcname=f"{prefix}/{self.KPACK}")

    def test_only_the_projects_code_objects_are_swapped(self):
        installed = self.output_dir / self.KPACK
        write_reference_kpack(
            installed,
            {
                (f"{ROCRAND}#0", "gfx942"): b"base-roc0",
                (f"{HIPRAND}#0", "gfx942"): b"base-hip0",
            },
        )
        self._write_instrumented_artifact(
            {
                (f"{ROCRAND}#0", "gfx942"): b"inst-roc0",
                (f"{HIPRAND}#0", "gfx942"): b"inst-hip0",
            }
        )

        mod.overlay_instrumented_device_code(
            {"rand": ["rocRAND"]}, self.dest_dir, self.output_dir
        )

        # hipRAND shares the archive but was not asked for, so it keeps running
        # the baseline's kernels.
        self.assertEqual(
            read_kernels(installed),
            {
                (f"{HIPRAND}#0", "gfx942"): b"base-hip0",
                (f"{ROCRAND}#0", "gfx942"): b"inst-roc0",
            },
        )

    def test_no_staged_copy_is_left_where_the_report_looks(self):
        # The report globs the install tree for .kpack files, so a leftover
        # staged copy would hand llvm-cov every code object twice.
        write_reference_kpack(
            self.output_dir / self.KPACK, {(f"{ROCRAND}#0", "gfx942"): b"base"}
        )
        self._write_instrumented_artifact({(f"{ROCRAND}#0", "gfx942"): b"inst"})

        mod.overlay_instrumented_device_code(
            {"rand": ["rocRAND"]}, self.dest_dir, self.output_dir
        )

        left = [
            p.relative_to(self.root)
            for d in (self.dest_dir, self.output_dir)
            for p in d.rglob("*.kpack")
            if p.is_file()
        ]
        self.assertEqual(left, [Path("build") / self.KPACK])

    def test_a_kpack_the_baseline_lacks_is_installed_whole(self):
        kernels = {(f"{ROCRAND}#0", "gfx942"): b"inst-roc0"}
        self._write_instrumented_artifact(kernels)

        mod.overlay_instrumented_device_code(
            {"rand": ["rocRAND"]}, self.dest_dir, self.output_dir
        )

        self.assertEqual(read_kernels(self.output_dir / self.KPACK), kernels)

    def test_siblings_sharing_an_archive_are_not_claimed(self):
        # sparse_lib and solver_lib hold the hip* wrappers' code objects too.
        cases = [
            (
                "math-libs/BLAS/rocSPARSE/stage/lib/librocsparse.so.1.0#3",
                "rocSPARSE",
                True,
            ),
            (
                "math-libs/BLAS/hipSPARSE/stage/lib/libhipsparse.so.4#0",
                "rocSPARSE",
                False,
            ),
            (
                "math-libs/BLAS/hipSPARSELt/stage/lib/libhipsparselt.so.0#1",
                "rocSPARSE",
                False,
            ),
            (
                "math-libs/BLAS/rocSOLVER/stage/lib/librocsolver.so.0#7",
                "rocSOLVER",
                True,
            ),
            (
                "math-libs/BLAS/hipSOLVER/stage/lib/libhipsolver.so.1#0",
                "rocSOLVER",
                False,
            ),
        ]
        for key, folder, expected in cases:
            with self.subTest(key=key, folder=folder):
                self.assertIs(mod._matches_folder(key, folder), expected)

    def test_artifacts_not_being_replaced_are_left_alone(self):
        installed = self.output_dir / self.KPACK
        baseline = {(f"{ROCRAND}#0", "gfx942"): b"base-roc0"}
        write_reference_kpack(installed, baseline)
        self._write_instrumented_artifact({(f"{ROCRAND}#0", "gfx942"): b"inst-roc0"})

        mod.overlay_instrumented_device_code(
            {"blas": ["rocBLAS"]}, self.dest_dir, self.output_dir
        )

        self.assertEqual(read_kernels(installed), baseline)


if __name__ == "__main__":
    unittest.main()
