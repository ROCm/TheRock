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
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import install_rocm_code_coverage_build as mod


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


if __name__ == "__main__":
    unittest.main()
