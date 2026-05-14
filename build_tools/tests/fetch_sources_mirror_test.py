#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for git mirror/reference integration in fetch_sources.py."""

import os
import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from fetch_sources import (
    get_enabled_sources,
    parse_source_set_args,
    resolve_reference_dir,
    _resolve_mirror_path,
    _fetch_one_external_git_source,
    _update_one_submodule,
    _update_submodules_with_reference,
)
from _therock_utils.build_topology import ExternalGitSource
from _therock_utils.git_mirrors import MIRROR_DIR_ENV


def _make_args(**kwargs) -> types.SimpleNamespace:
    """Build a minimal args namespace for testing."""
    defaults = {"reference_dir": None}
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


def _make_fetch_args(**kwargs) -> types.SimpleNamespace:
    """Build args namespace for source selection tests."""
    defaults = {
        "stage": None,
        "source_sets": [],
        "skip_submodules": [],
        "include_system_projects": False,
        "system_projects": [],
        "include_compilers": False,
        "compiler_projects": [],
        "include_debug_tools": False,
        "debug_tools": [],
        "include_rocm_libraries": False,
        "include_rocm_systems": False,
        "include_ml_frameworks": False,
        "ml_framework_projects": [],
        "include_media_libs": False,
        "media_libs_projects": [],
        "include_iree_libs": False,
        "iree_libs_projects": [],
        "include_math_libraries": False,
        "math_library_projects": [],
    }
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


class ResolveReferenceDirTest(unittest.TestCase):
    """Tests for resolve_reference_dir."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_returns_path_from_args(self):
        args = _make_args(reference_dir=self.temp_dir)
        result = resolve_reference_dir(args)
        self.assertEqual(result, self.temp_dir.resolve())

    @mock.patch.dict(os.environ, {MIRROR_DIR_ENV: ""}, clear=False)
    def test_returns_none_when_env_empty(self):
        args = _make_args()
        result = resolve_reference_dir(args)
        self.assertIsNone(result)

    def test_returns_none_when_nothing_set(self):
        args = _make_args()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(MIRROR_DIR_ENV, None)
            result = resolve_reference_dir(args)
        self.assertIsNone(result)

    def test_env_var_fallback(self):
        args = _make_args()
        with mock.patch.dict(os.environ, {MIRROR_DIR_ENV: str(self.temp_dir)}):
            result = resolve_reference_dir(args)
        self.assertEqual(result, self.temp_dir.resolve())

    def test_args_takes_precedence_over_env(self):
        other_dir = self.temp_dir / "other"
        other_dir.mkdir()
        args = _make_args(reference_dir=other_dir)
        with mock.patch.dict(os.environ, {MIRROR_DIR_ENV: str(self.temp_dir)}):
            result = resolve_reference_dir(args)
        self.assertEqual(result, other_dir.resolve())

    def test_warns_on_nonexistent_dir(self):
        args = _make_args(reference_dir=Path("/nonexistent/mirror/dir"))
        result = resolve_reference_dir(args)
        self.assertIsNone(result)


class ResolveMirrorPathTest(unittest.TestCase):
    """Tests for _resolve_mirror_path."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_found(self):
        mirror = self.temp_dir / "ROCm" / "llvm-project.git"
        mirror.mkdir(parents=True)
        result = _resolve_mirror_path(
            self.temp_dir, "https://github.com/ROCm/llvm-project.git"
        )
        self.assertEqual(result, mirror)

    def test_missing(self):
        result = _resolve_mirror_path(
            self.temp_dir, "https://github.com/ROCm/llvm-project.git"
        )
        self.assertIsNone(result)


class SourceSetSelectionTest(unittest.TestCase):
    """Tests for source set selection helpers."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.topology_path = self.temp_dir / "BUILD_TOPOLOGY.toml"
        self.branch_config_path = self.temp_dir / "BRANCH_CONFIG.json"
        self.topology_path.write_text(
            textwrap.dedent(
                """
                [source_sets.optional-hrx]
                description = "Optional HRX"
                external_git_sources = [
                  { name = "hrx", origin = "https://github.com/ROCm/hrx.git", commit = "e642a13425f46bcf909078459dd4e07df0723a0d", path = "optional-sources/hrx" },
                ]

                [build_stages.compiler-runtime]
                description = "Compiler runtime"
                artifact_groups = ["hip-runtime"]

                [artifact_groups.hip-runtime]
                description = "HIP runtime"
                type = "generic"
                source_sets = []
                """
            )
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_parse_source_set_args(self):
        self.assertEqual(
            parse_source_set_args(["base,optional-hrx", "optional-hrx"]),
            ["base", "optional-hrx"],
        )

    def test_explicit_source_sets_add_external_sources(self):
        args = _make_fetch_args(source_sets=["optional-hrx"])
        with mock.patch("fetch_sources.TOPOLOGY_PATH", self.topology_path), mock.patch(
            "fetch_sources.BRANCH_CONFIG_PATH", self.branch_config_path
        ):
            projects, external_sources = get_enabled_sources(args)

        self.assertEqual(projects, [])
        self.assertEqual(len(external_sources), 1)
        self.assertEqual(external_sources[0].name, "hrx")

    def test_stage_uses_branch_config_artifact_group_source_sets(self):
        self.branch_config_path.write_text(
            json.dumps(
                {"artifact_groups": {"hip-runtime": {"source_sets": ["optional-hrx"]}}}
            )
        )
        args = _make_fetch_args(stage="compiler-runtime")
        with mock.patch("fetch_sources.TOPOLOGY_PATH", self.topology_path), mock.patch(
            "fetch_sources.BRANCH_CONFIG_PATH", self.branch_config_path
        ):
            projects, external_sources = get_enabled_sources(args)

        self.assertEqual(projects, [])
        self.assertEqual(len(external_sources), 1)
        self.assertEqual(external_sources[0].name, "hrx")


class UpdateOneSubmoduleTest(unittest.TestCase):
    """Tests for _update_one_submodule."""

    @mock.patch("fetch_sources.run_command")
    def test_with_mirror(self, mock_run):
        mirror = Path("/mirrors/ROCm/llvm-project.git")
        _update_one_submodule("compiler/amd-llvm", [], mirror)
        cmd = mock_run.call_args[0][0]
        self.assertIn("--reference", cmd)
        self.assertIn(str(mirror), cmd)

    @mock.patch("fetch_sources.run_command")
    def test_without_mirror(self, mock_run):
        _update_one_submodule("compiler/amd-llvm", [], None)
        cmd = mock_run.call_args[0][0]
        self.assertNotIn("--reference", cmd)

    @mock.patch("fetch_sources.run_command")
    def test_passes_update_args(self, mock_run):
        _update_one_submodule("compiler/amd-llvm", ["--depth", "1"], None)
        cmd = mock_run.call_args[0][0]
        self.assertIn("--depth", cmd)
        self.assertIn("1", cmd)

    @mock.patch("fetch_sources.run_command")
    def test_submodule_path_after_separator(self, mock_run):
        _update_one_submodule("compiler/amd-llvm", [], None)
        cmd = mock_run.call_args[0][0]
        separator_idx = cmd.index("--")
        self.assertEqual(cmd[separator_idx + 1], "compiler/amd-llvm")

    @mock.patch("fetch_sources.run_command")
    def test_fallback_on_reference_failure(self, mock_run):
        mock_run.side_effect = [
            subprocess.CalledProcessError(1, "git"),
            None,
        ]
        mirror = Path("/mirrors/ROCm/llvm-project.git")
        _update_one_submodule("compiler/amd-llvm", [], mirror)

        self.assertEqual(mock_run.call_count, 2)
        first_cmd = mock_run.call_args_list[0][0][0]
        second_cmd = mock_run.call_args_list[1][0][0]
        self.assertIn("--reference", first_cmd)
        self.assertNotIn("--reference", second_cmd)

    @mock.patch("fetch_sources.run_command")
    def test_no_fallback_without_mirror(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(1, "git")
        with self.assertRaises(subprocess.CalledProcessError):
            _update_one_submodule("compiler/amd-llvm", [], None)


class UpdateSubmodulesWithReferenceTest(unittest.TestCase):
    """Tests for _update_submodules_with_reference."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.reference_dir = self.temp_dir / "mirrors"
        self.reference_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @mock.patch("fetch_sources._update_one_submodule")
    @mock.patch("fetch_sources.run_command")
    @mock.patch("fetch_sources._submodule_is_initialized", return_value=False)
    @mock.patch(
        "fetch_sources._get_submodule_url_map",
        return_value={"compiler/amd-llvm": "https://github.com/ROCm/llvm-project.git"},
    )
    def test_two_phase_init(
        self, _mock_urls, _mock_is_init, mock_run_cmd, mock_update_one
    ):
        _update_submodules_with_reference(
            ["compiler/amd-llvm"], [], self.reference_dir, jobs=1
        )

        init_call = mock_run_cmd.call_args
        init_cmd = init_call[0][0]
        self.assertIn("init", init_cmd)
        self.assertIn("compiler/amd-llvm", init_cmd)

        mock_update_one.assert_called_once()

    @mock.patch("fetch_sources.run_command")
    @mock.patch("fetch_sources._submodule_is_initialized", return_value=True)
    @mock.patch("fetch_sources._get_submodule_url_map", return_value={})
    def test_already_initialized_uses_batch(
        self, _mock_urls, _mock_is_init, mock_run_cmd
    ):
        _update_submodules_with_reference(
            ["compiler/amd-llvm"], [], self.reference_dir, jobs=1
        )

        cmd = mock_run_cmd.call_args[0][0]
        self.assertIn("update", cmd)
        self.assertIn("--init", cmd)
        self.assertIn("compiler/amd-llvm", cmd)

    @mock.patch("fetch_sources._update_one_submodule")
    @mock.patch("fetch_sources.run_command")
    @mock.patch("fetch_sources._submodule_is_initialized", side_effect=[False, True])
    @mock.patch(
        "fetch_sources._get_submodule_url_map",
        return_value={
            "compiler/amd-llvm": "https://github.com/ROCm/llvm-project.git",
            "base/rocm-cmake": "https://github.com/ROCm/rocm-cmake.git",
        },
    )
    def test_mixed_init_and_already_init(
        self, _mock_urls, _mock_is_init, mock_run_cmd, mock_update_one
    ):
        _update_submodules_with_reference(
            ["compiler/amd-llvm", "base/rocm-cmake"],
            [],
            self.reference_dir,
            jobs=1,
        )

        mock_update_one.assert_called_once()
        update_one_path = mock_update_one.call_args[0][0]
        self.assertEqual(update_one_path, "compiler/amd-llvm")

        batch_call = mock_run_cmd.call_args_list[-1]
        batch_cmd = batch_call[0][0]
        self.assertIn("base/rocm-cmake", batch_cmd)

    @mock.patch("fetch_sources._update_one_submodule")
    @mock.patch("fetch_sources.run_command")
    @mock.patch("fetch_sources._submodule_is_initialized", return_value=False)
    @mock.patch(
        "fetch_sources._get_submodule_url_map",
        return_value={
            "a": "https://github.com/ROCm/a.git",
            "b": "https://github.com/ROCm/b.git",
        },
    )
    def test_parallel_jobs_gt_one(
        self, _mock_urls, _mock_is_init, mock_run_cmd, mock_update_one
    ):
        _update_submodules_with_reference(["a", "b"], [], self.reference_dir, jobs=4)

        self.assertEqual(mock_update_one.call_count, 2)


class FetchExternalGitSourceTest(unittest.TestCase):
    """Tests for external git source fetching."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.source = ExternalGitSource(
            name="hrx",
            origin="https://github.com/ROCm/hrx.git",
            commit="e642a13425f46bcf909078459dd4e07df0723a0d",
            path="optional-sources/hrx",
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @mock.patch("fetch_sources.run_command")
    def test_clones_and_checks_out_pinned_commit(self, mock_run):
        args = _make_args(depth=None, progress=False)
        with mock.patch("fetch_sources.THEROCK_DIR", self.temp_dir):
            _fetch_one_external_git_source(args, self.source, None)

        commands = [call.args[0] for call in mock_run.call_args_list]
        self.assertEqual(commands[0][0:3], ["git", "clone", "--no-checkout"])
        self.assertEqual(
            commands[1], ["git", "checkout", "--detach", self.source.commit]
        )
        self.assertEqual(commands[2], ["git", "reset", "--hard", self.source.commit])

    @mock.patch("fetch_sources.run_command")
    def test_updates_existing_checkout(self, mock_run):
        source_dir = self.temp_dir / self.source.path
        (source_dir / ".git").mkdir(parents=True)
        args = _make_args(depth=None, progress=False)
        with mock.patch("fetch_sources.THEROCK_DIR", self.temp_dir):
            _fetch_one_external_git_source(args, self.source, None)

        commands = [call.args[0] for call in mock_run.call_args_list]
        self.assertEqual(
            commands[0],
            ["git", "remote", "set-url", "origin", self.source.origin],
        )
        self.assertEqual(commands[1], ["git", "fetch", "origin"])
        self.assertEqual(
            commands[2], ["git", "checkout", "--detach", self.source.commit]
        )


if __name__ == "__main__":
    unittest.main()
