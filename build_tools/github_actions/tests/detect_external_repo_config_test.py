#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for detect_external_repo_config.py"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _therock_utils.build_topology import get_topology

from detect_external_repo_config import (
    get_repo_config,
    get_external_repo_path,
    get_source_stages,
    build_stage_configs,
    import_external_repo_module,
    get_skip_patterns,
    get_test_list,
    main as detect_external_repo_config_main,
    output_github_actions_vars,
    REPO_CONFIGS,
)


class TestExternalRepoJsonCasing(unittest.TestCase):
    """Tests that repo name extraction from external_repo JSON is case-insensitive."""

    def setUp(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            self.temp_file = f.name
        os.environ["GITHUB_OUTPUT"] = self.temp_file

    def tearDown(self):
        if "GITHUB_OUTPUT" in os.environ:
            del os.environ["GITHUB_OUTPUT"]
        if hasattr(self, "temp_file") and os.path.exists(self.temp_file):
            os.unlink(self.temp_file)

    def _run_with_json(self, repository: str) -> int:
        return detect_external_repo_config_main(
            [
                "--external-repo-json",
                f'{{"repository": "{repository}", "ref": "abc123"}}',
            ]
        )

    def test_mixed_case_repo_name(self):
        """ROCm/Rocm-Libraries (mixed case) should resolve to rocm-libraries config."""
        rc = self._run_with_json("ROCm/Rocm-Libraries")
        self.assertEqual(rc, 0)

    def test_uppercase_repo_name(self):
        """ROCm/ROCM-LIBRARIES (all caps) should still resolve to rocm-libraries config."""
        rc = self._run_with_json("ROCm/ROCM-LIBRARIES")
        self.assertEqual(rc, 0)

    def test_lowercase_repo_name(self):
        """ROCm/rocm-libraries (already lowercase) should resolve to rocm-libraries config."""
        rc = self._run_with_json("ROCm/rocm-libraries")
        self.assertEqual(rc, 0)

    def test_unknown_repo_returns_nonzero(self):
        """An unregistered repo should return a non-zero exit code."""
        rc = self._run_with_json("ROCm/SomeUnknownRepo")
        self.assertNotEqual(rc, 0)


class TestGetRepoConfig(unittest.TestCase):
    """Tests for get_repo_config function"""

    def test_rocm_libraries_config(self):
        """Test rocm-libraries configuration"""
        config = get_repo_config("rocm-libraries")
        self.assertEqual(
            config["cmake_source_var"], "THEROCK_ROCM_LIBRARIES_SOURCE_DIR"
        )
        self.assertEqual(config["submodule_path"], "rocm-libraries")
        self.assertEqual(config["skip_submodules"], ["rocm-libraries"])

    def test_rocm_systems_config(self):
        """Test rocm-systems configuration"""
        config = get_repo_config("rocm-systems")
        self.assertEqual(config["cmake_source_var"], "THEROCK_ROCM_SYSTEMS_SOURCE_DIR")
        self.assertEqual(config["submodule_path"], "rocm-systems")
        self.assertEqual(config["skip_submodules"], ["rocm-systems"])

    def test_rocgdb_config(self):
        """Test rocgdb configuration"""
        config = get_repo_config("rocgdb")
        self.assertEqual(config["cmake_source_var"], "THEROCK_ROCGDB_SOURCE_DIR")
        self.assertEqual(config["submodule_path"], "debug-tools/rocgdb/source")
        self.assertEqual(config["skip_submodules"], ["rocgdb"])

    def test_unknown_repo_raises_error(self):
        """Test that unknown repository raises ValueError"""
        with self.assertRaises(ValueError) as context:
            get_repo_config("unknown-repo")
        self.assertIn("Unknown external repository", str(context.exception))
        self.assertIn("unknown-repo", str(context.exception))

    def test_all_repos_have_required_keys(self):
        """Test that all repo configs have required keys"""
        required_keys = {
            "cmake_source_var",
            "submodule_path",
            "skip_submodules",
        }
        for repo_name, config in REPO_CONFIGS.items():
            with self.subTest(repo=repo_name):
                self.assertTrue(
                    required_keys.issubset(config.keys()),
                    f"Repo {repo_name} missing required keys: {required_keys - config.keys()}",
                )


class TestOutputGithubActionsVars(unittest.TestCase):
    """Tests for output_github_actions_vars function"""

    def setUp(self):
        """Set up test fixtures"""
        # Create temporary file for GITHUB_OUTPUT
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            self.temp_file = f.name
        os.environ["GITHUB_OUTPUT"] = self.temp_file

    def tearDown(self):
        """Clean up test fixtures"""
        # Remove GITHUB_OUTPUT env var
        if "GITHUB_OUTPUT" in os.environ:
            del os.environ["GITHUB_OUTPUT"]
        # Delete temp file
        if hasattr(self, "temp_file") and os.path.exists(self.temp_file):
            os.unlink(self.temp_file)

    def test_output_to_file(self):
        """Test output to GITHUB_OUTPUT file"""
        config = {
            "cmake_source_var": "TEST_VAR",
            "submodule_path": "test-dir",
            "skip_submodules": ["test-submodule"],
        }

        output_github_actions_vars(config)

        # Read the output file
        with open(self.temp_file, "r") as f:
            output = f.read()

        # Verify output format
        self.assertIn("cmake_source_var=TEST_VAR", output)
        self.assertIn("submodule_path=test-dir", output)
        self.assertIn("skip_submodules=['test-submodule']", output)

    def test_boolean_conversion(self):
        """Test that booleans are converted to lowercase strings"""
        config = {
            "bool_true": True,
            "bool_false": False,
        }

        output_github_actions_vars(config)

        with open(self.temp_file, "r") as f:
            output = f.read()

        # Verify lowercase (important for bash conditionals)
        self.assertIn("bool_true=true", output)
        self.assertIn("bool_false=false", output)
        self.assertNotIn("True", output)
        self.assertNotIn("False", output)

    def test_config_json_generated(self):
        """Test that config_json is generated by main()."""
        rc = detect_external_repo_config_main(
            [
                "--repository",
                "rocm-libraries",
            ]
        )
        self.assertEqual(rc, 0)

        with open(self.temp_file, "r") as f:
            output = f.read()

        # Verify config_json is included with correct checkout_path (relative with external- prefix)
        self.assertIn("config_json=", output)
        self.assertIn('"checkout_path": "external-rocm-libraries"', output)

    def test_external_repo_json_mixed_case_name(self):
        """Test that mixed-case repo names (e.g. ROCm/ROCgdb) are lowercased."""
        rc = detect_external_repo_config_main(
            [
                "--external-repo-json",
                '{"repository": "ROCm/ROCgdb", "ref": "abc123"}',
            ]
        )
        self.assertEqual(rc, 0)

        with open(self.temp_file, "r") as f:
            output = f.read()

        self.assertIn("config_json=", output)
        self.assertIn('"checkout_path": "external-rocgdb"', output)
        self.assertIn("THEROCK_ROCGDB_SOURCE_DIR", output)

    def test_extra_cmake_options_applies_to_every_stage(self):
        """extra_cmake_options configures the build as a whole, not one stage.

        It is folded into each stage's cmake_options, including the stages that
        never read the external sources (rocgdb is built by debug-tools only).
        """
        for extra in [
            "-DTHEROCK_USE_EXTERNAL_ROCGDB=ON",
            # Multiple flags, and embedded quotes surviving the JSON round trip.
            "-DTHEROCK_USE_EXTERNAL_ROCGDB=ON -DSOME_OTHER_FLAG=value",
            '-DFOO="bar"',
            "",
        ]:
            with self.subTest(extra_cmake_options=extra):
                open(self.temp_file, "w").close()
                rc = detect_external_repo_config_main(
                    [
                        "--external-repo-json",
                        json.dumps(
                            {
                                "repository": "ROCm/ROCgdb",
                                "ref": "abc123",
                                "extra_cmake_options": extra,
                            }
                        ),
                    ]
                )
                self.assertEqual(rc, 0)

                with open(self.temp_file, "r") as f:
                    line = [
                        l for l in f.read().splitlines() if l.startswith("config_json=")
                    ][0]
                stages = json.loads(line.split("=", 1)[1])["stages"]

                self.assertEqual(stages["compiler-runtime"]["cmake_options"], extra)
                self.assertEqual(
                    stages["debug-tools"]["cmake_options"],
                    " ".join(
                        filter(
                            None, ["-DTHEROCK_ROCGDB_SOURCE_DIR=external-rocgdb", extra]
                        )
                    ),
                )


class TestGetExternalRepoPath(unittest.TestCase):
    """Tests for get_external_repo_path function"""

    @patch.dict(
        os.environ,
        {"EXTERNAL_SOURCE_PATH": "rocm-libraries", "GITHUB_WORKSPACE": "/workspace"},
    )
    @patch("detect_external_repo_config.Path")
    @patch("detect_external_repo_config._is_valid_repo_path")
    def test_external_source_path_priority(self, mock_is_valid, mock_path_cls):
        """Test that EXTERNAL_SOURCE_PATH has highest priority"""
        # Mock Path behavior
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.name = "rocm-libraries"
        mock_path_cls.return_value = mock_path

        # Mock validation
        mock_is_valid.return_value = True

        result = get_external_repo_path("rocm-libraries")
        self.assertIsNotNone(result)

    @patch.dict(os.environ, {}, clear=True)
    @patch("detect_external_repo_config.Path")
    @patch("detect_external_repo_config._is_valid_repo_path")
    def test_cwd_fallback(self, mock_is_valid, mock_path_cls):
        """Test that current working directory is used as fallback"""
        mock_cwd = MagicMock()
        mock_cwd.exists.return_value = True
        mock_path_cls.cwd.return_value = mock_cwd
        mock_is_valid.return_value = True

        result = get_external_repo_path("rocm-libraries")
        self.assertEqual(result, mock_cwd)

    @patch.dict(os.environ, {}, clear=True)
    @patch("detect_external_repo_config.Path")
    @patch("detect_external_repo_config._is_valid_repo_path")
    def test_no_valid_path_raises_error(self, mock_is_valid, mock_path_cls):
        """Test that ValueError is raised when no valid path is found"""
        # Clear the cache to ensure this test runs fresh
        get_external_repo_path.cache_clear()

        mock_cwd = MagicMock()
        mock_path_cls.cwd.return_value = mock_cwd
        mock_is_valid.return_value = False

        with self.assertRaises(ValueError) as context:
            get_external_repo_path("rocm-libraries")
        self.assertIn("Could not find external repo", str(context.exception))


class TestImportExternalRepoModule(unittest.TestCase):
    """Tests for import_external_repo_module function"""

    @patch("detect_external_repo_config.get_external_repo_path")
    @patch("importlib.util.spec_from_file_location")
    @patch("importlib.util.module_from_spec")
    def test_successful_import(
        self, mock_module_from_spec, mock_spec_from_file, mock_get_path
    ):
        """Test successful module import"""
        # Mock repo path
        mock_repo_path = MagicMock()
        mock_script_path = MagicMock()
        mock_script_path.exists.return_value = True
        mock_repo_path.__truediv__ = MagicMock(return_value=mock_script_path)
        mock_get_path.return_value = mock_repo_path

        # Mock importlib
        mock_spec = MagicMock()
        mock_loader = MagicMock()
        mock_spec.loader = mock_loader
        mock_module = MagicMock()
        mock_spec_from_file.return_value = mock_spec
        mock_module_from_spec.return_value = mock_module

        result = import_external_repo_module("rocm-libraries", "test_module")
        self.assertEqual(result, mock_module)
        mock_loader.exec_module.assert_called_once_with(mock_module)

    @patch("detect_external_repo_config.get_external_repo_path")
    def test_module_not_found(self, mock_get_path):
        """Test handling when module file doesn't exist"""
        mock_repo_path = MagicMock()
        mock_script_path = MagicMock()
        mock_script_path.exists.return_value = False
        mock_repo_path.__truediv__ = MagicMock(return_value=mock_script_path)
        mock_get_path.return_value = mock_repo_path

        result = import_external_repo_module("rocm-libraries", "missing_module")
        self.assertIsNone(result)

    @patch("detect_external_repo_config.get_external_repo_path")
    @patch("importlib.util.spec_from_file_location")
    @patch("importlib.util.module_from_spec")
    def test_import_error_handling(
        self, mock_module_from_spec, mock_spec_from_file, mock_get_path
    ):
        """Test handling of ImportError during module loading"""
        mock_repo_path = MagicMock()
        mock_script_path = MagicMock()
        mock_script_path.exists.return_value = True
        mock_repo_path.__truediv__ = MagicMock(return_value=mock_script_path)
        mock_get_path.return_value = mock_repo_path

        # Mock import failure
        mock_spec = MagicMock()
        mock_loader = MagicMock()
        mock_spec.loader = mock_loader
        mock_loader.exec_module.side_effect = ImportError("Import failed")
        mock_spec_from_file.return_value = mock_spec
        mock_module_from_spec.return_value = MagicMock()

        result = import_external_repo_module("rocm-libraries", "test_module")
        self.assertIsNone(result)


class TestGetSkipPatterns(unittest.TestCase):
    """Tests for get_skip_patterns function"""

    @patch("detect_external_repo_config.import_external_repo_module")
    def test_get_skip_patterns_success(self, mock_import):
        """Test successful retrieval of skip patterns"""
        mock_module = MagicMock()
        mock_module.SKIPPABLE_PATH_PATTERNS = ["pattern1/*", "pattern2/*"]
        mock_import.return_value = mock_module

        result = get_skip_patterns("rocm-libraries")
        self.assertEqual(result, ["pattern1/*", "pattern2/*"])

    @patch("detect_external_repo_config.import_external_repo_module")
    def test_get_skip_patterns_no_module(self, mock_import):
        """Test when module cannot be imported"""
        mock_import.return_value = None

        result = get_skip_patterns("rocm-libraries")
        self.assertEqual(result, [])

    @patch("detect_external_repo_config.import_external_repo_module")
    def test_get_skip_patterns_no_attribute(self, mock_import):
        """Test when module doesn't have SKIPPABLE_PATH_PATTERNS attribute"""
        mock_module = MagicMock(spec=[])
        del mock_module.SKIPPABLE_PATH_PATTERNS  # Ensure attribute doesn't exist
        mock_import.return_value = mock_module

        result = get_skip_patterns("rocm-libraries")
        self.assertEqual(result, [])


class TestGetTestList(unittest.TestCase):
    """Tests for get_test_list function"""

    @patch("detect_external_repo_config.import_external_repo_module")
    def test_get_test_list_success(self, mock_import):
        """Test successful retrieval of test list"""
        mock_module = MagicMock()
        mock_module.project_map = {
            "project1": {"project_to_test": ["test1", "test2"]},
            "project2": {"project_to_test": "test3"},
        }
        mock_import.return_value = mock_module

        result = get_test_list("rocm-libraries")
        # Result is a sorted list from a set, so order may vary
        self.assertEqual(set(result), {"test1", "test2", "test3"})

    @patch("detect_external_repo_config.import_external_repo_module")
    def test_get_test_list_no_module(self, mock_import):
        """Test when module cannot be imported"""
        mock_import.return_value = None

        result = get_test_list("rocm-libraries")
        self.assertEqual(result, [])

    @patch("detect_external_repo_config.import_external_repo_module")
    def test_get_test_list_no_attribute(self, mock_import):
        """Test when module doesn't have project_map attribute"""
        mock_module = MagicMock(spec=[])
        del mock_module.project_map  # Ensure attribute doesn't exist
        mock_import.return_value = mock_module

        result = get_test_list("rocm-libraries")
        self.assertEqual(result, [])


class TestGetSourceStages(unittest.TestCase):
    """Tests for mapping external repos to the build stages that need them."""

    def test_source_stages_per_repo(self):
        """Derived from BUILD_TOPOLOGY.toml, so these follow the stage layout."""
        for repo, expected in {
            "rocm-libraries": ["cv-libs", "math-libs"],
            "rocgdb": ["debug-tools"],
        }.items():
            with self.subTest(repo=repo):
                self.assertEqual(get_source_stages(repo), expected)

        # rocm-systems is used nearly everywhere, but not by cv-libs.
        rocm_systems = get_source_stages("rocm-systems")
        self.assertIn("compiler-runtime", rocm_systems)
        self.assertNotIn("cv-libs", rocm_systems)


class TestStageConfigs(unittest.TestCase):
    """Tests for the per-stage view the build workflows index into."""

    CONFIG = dict(
        repo_name="rocm-libraries",
        checkout_path="external-rocm-libraries",
        source_package="ROCM_LIBRARIES",
        source_stages=["cv-libs", "math-libs"],
        extra_cmake_options="-DFOO=BAR",
    )

    # stage -> (needs_source, cmake_options, --dvc-projects value)
    EXPECTED = {
        # Builds from the external sources: check it out, point CMake at it,
        # and pull the DVC project rooted in that checkout.
        "math-libs": (
            True,
            "-DTHEROCK_ROCM_LIBRARIES_SOURCE_DIR=external-rocm-libraries -DFOO=BAR",
            "external-rocm-libraries rocm-systems",
        ),
        "cv-libs": (
            True,
            "-DTHEROCK_ROCM_LIBRARIES_SOURCE_DIR=external-rocm-libraries -DFOO=BAR",
            "external-rocm-libraries rocm-systems",
        ),
        # Does not: no checkout, no source dir, and the DVC project rooted in
        # the absent checkout is dropped. extra_cmake_options still applies.
        "compiler-runtime": (False, "-DFOO=BAR", "rocm-systems"),
        "comm-libs": (False, "-DFOO=BAR", "rocm-systems"),
    }

    def test_stage_examples(self):
        stages = build_stage_configs(**self.CONFIG)
        for stage, (needs_source, cmake_options, dvc_projects) in self.EXPECTED.items():
            with self.subTest(stage=stage):
                self.assertEqual(stages[stage]["needs_source"], needs_source)
                self.assertEqual(stages[stage]["cmake_options"], cmake_options)
                self.assertEqual(
                    stages[stage]["fetch_sources_args"],
                    f"--skip-submodules rocm-libraries --dvc-projects {dvc_projects}",
                )

    def test_every_stage_has_an_entry(self):
        """A missing key would silently skip the checkout, so cover all stages."""
        self.assertEqual(
            sorted(build_stage_configs(**self.CONFIG)),
            sorted(get_topology().get_all_stage_names()),
        )

    def test_main_emits_stages_map(self):
        """config_json carries the map, keyed as the workflows index it."""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            temp_file = f.name
        os.environ["GITHUB_OUTPUT"] = temp_file
        try:
            rc = detect_external_repo_config_main(["--repository", "rocm-libraries"])
            self.assertEqual(rc, 0)
            with open(temp_file, "r") as f:
                output = f.read()
        finally:
            del os.environ["GITHUB_OUTPUT"]
            os.unlink(temp_file)

        line = [l for l in output.splitlines() if l.startswith("config_json=")][0]
        stages = json.loads(line.split("=", 1)[1])["stages"]
        self.assertTrue(stages["math-libs"]["needs_source"])
        self.assertFalse(stages["compiler-runtime"]["needs_source"])


if __name__ == "__main__":
    unittest.main()
