#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for detect_external_repo_config.py"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from detect_external_repo_config import (
    compute_all_stage_sparse_checkouts,
    compute_stage_sparse_checkout,
    extract_source_path_from_project,
    get_repo_config,
    get_external_repo_path,
    get_skip_patterns,
    get_stage_source_paths,
    get_test_list,
    import_external_repo_module,
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

    def test_extra_cmake_options_forwarded(self):
        """Test that extra_cmake_options from external_repo JSON is forwarded to config_json."""
        rc = detect_external_repo_config_main(
            [
                "--external-repo-json",
                '{"repository": "ROCm/ROCgdb", "ref": "abc123", "extra_cmake_options": "-DTHEROCK_USE_EXTERNAL_ROCGDB=ON"}',
            ]
        )
        self.assertEqual(rc, 0)

        with open(self.temp_file, "r") as f:
            output = f.read()

        self.assertIn(
            '"extra_cmake_options": "-DTHEROCK_USE_EXTERNAL_ROCGDB=ON"', output
        )

    def test_extra_cmake_options_empty_by_default(self):
        """Test that extra_cmake_options defaults to empty string when not provided."""
        rc = detect_external_repo_config_main(
            [
                "--repository",
                "rocm-libraries",
            ]
        )
        self.assertEqual(rc, 0)

        with open(self.temp_file, "r") as f:
            output = f.read()

        self.assertIn('"extra_cmake_options": ""', output)

    def test_extra_cmake_options_multiple_flags(self):
        """Test that multiple space-separated cmake flags are forwarded intact."""
        rc = detect_external_repo_config_main(
            [
                "--external-repo-json",
                '{"repository": "ROCm/ROCgdb", "ref": "abc123",'
                ' "extra_cmake_options": "-DTHEROCK_USE_EXTERNAL_ROCGDB=ON -DSOME_OTHER_FLAG=value"}',
            ]
        )
        self.assertEqual(rc, 0)

        with open(self.temp_file, "r") as f:
            output = f.read()

        self.assertIn(
            '"extra_cmake_options": "-DTHEROCK_USE_EXTERNAL_ROCGDB=ON -DSOME_OTHER_FLAG=value"',
            output,
        )

    def test_extra_cmake_options_embedded_quotes(self):
        """Test that embedded quotes survive the JSON parse/serialize round-trip."""
        # JSON input: extra_cmake_options value contains escaped double quotes.
        # json.loads produces the Python string:  -DFOO="bar"
        # json.dumps then re-escapes it back to:  "-DFOO=\"bar\""
        rc = detect_external_repo_config_main(
            [
                "--external-repo-json",
                '{"repository": "ROCm/ROCgdb", "ref": "abc123",'
                ' "extra_cmake_options": "-DFOO=\\"bar\\""}',
            ]
        )
        self.assertEqual(rc, 0)

        with open(self.temp_file, "r") as f:
            output = f.read()

        # After round-trip the quotes are re-escaped in the serialized JSON
        self.assertIn('"extra_cmake_options": "-DFOO=\\"bar\\""', output)


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


class TestNormalizeChangedProjects(unittest.TestCase):
    """Tests for normalize_changed_projects function."""

    def test_empty_changed_projects_returns_empty(self):
        """Empty changed_projects should return empty string."""
        from detect_external_repo_config import normalize_changed_projects

        self.assertEqual(normalize_changed_projects(""), "")
        self.assertEqual(normalize_changed_projects("  "), "")

    def test_single_project(self):
        """Single project path should be normalized."""
        from detect_external_repo_config import normalize_changed_projects

        result = normalize_changed_projects("projects/rocprim")
        self.assertEqual(result, "projects/rocprim")

    def test_multiple_projects(self):
        """Multiple projects should all be included."""
        from detect_external_repo_config import normalize_changed_projects

        result = normalize_changed_projects(
            "projects/rocprim,shared/rocroller,dnn-providers/miopen-provider"
        )
        paths = result.split(",")

        self.assertIn("projects/rocprim", paths)
        self.assertIn("shared/rocroller", paths)
        self.assertIn("dnn-providers/miopen-provider", paths)

    def test_paths_are_sorted(self):
        """Paths should be sorted alphabetically."""
        from detect_external_repo_config import normalize_changed_projects

        result = normalize_changed_projects("projects/zebra,projects/alpha")
        paths = result.split(",")

        # Verify sorted order
        self.assertEqual(paths, sorted(paths))

    def test_whitespace_handling(self):
        """Whitespace around paths should be trimmed."""
        from detect_external_repo_config import normalize_changed_projects

        result = normalize_changed_projects("  projects/rocprim , shared/rocroller  ")
        paths = result.split(",")

        self.assertIn("projects/rocprim", paths)
        self.assertIn("shared/rocroller", paths)


class TestExtractSourcePathFromProject(unittest.TestCase):
    """Tests for extract_source_path_from_project function."""

    def test_projects_path(self):
        """Test extraction from projects/ path."""
        self.assertEqual(
            extract_source_path_from_project("projects/rocprim"), "rocprim"
        )

    def test_shared_path(self):
        """Test extraction from shared/ path."""
        self.assertEqual(
            extract_source_path_from_project("shared/rocroller"), "rocroller"
        )

    def test_dnn_providers_path(self):
        """Test extraction from dnn-providers/ path."""
        result = extract_source_path_from_project("dnn-providers/miopen-provider")
        self.assertEqual(result, "miopen-provider")

    def test_simple_path(self):
        """Test extraction from simple path without prefix."""
        self.assertEqual(extract_source_path_from_project("rocprim"), "rocprim")

    def test_empty_path(self):
        """Test extraction from empty path."""
        self.assertIsNone(extract_source_path_from_project(""))
        self.assertIsNone(extract_source_path_from_project("  "))


class TestGetStageSourcePaths(unittest.TestCase):
    """Tests for get_stage_source_paths function."""

    def test_math_libs_includes_expected(self):
        """Test math-libs stage includes expected source paths."""
        source_paths = get_stage_source_paths("math-libs")
        self.assertIn("rocprim", source_paths)
        self.assertIn("rocblas", source_paths)
        self.assertIn("rocroller", source_paths)

    def test_cv_libs_includes_rpp(self):
        """Test cv-libs stage includes rpp."""
        source_paths = get_stage_source_paths("cv-libs")
        self.assertIn("rpp", source_paths)

    def test_unknown_stage_returns_empty(self):
        """Test unknown stage returns empty set."""
        source_paths = get_stage_source_paths("unknown-stage")
        self.assertEqual(source_paths, set())


class TestComputeStageSparseCheckout(unittest.TestCase):
    """Tests for compute_stage_sparse_checkout function."""

    def test_affected_stage(self):
        """Test stage that is affected by changed projects."""
        paths = compute_stage_sparse_checkout(
            "math-libs", "projects/rocprim,shared/rocroller"
        )
        self.assertIn("projects/rocprim", paths)
        self.assertIn("shared/rocroller", paths)

    def test_unaffected_stage(self):
        """Test stage that is NOT affected by changed projects."""
        paths = compute_stage_sparse_checkout("cv-libs", "projects/rocprim")
        self.assertEqual(paths, [])

    def test_empty_changed_projects(self):
        """Test with empty changed_projects."""
        paths = compute_stage_sparse_checkout("math-libs", "")
        self.assertEqual(paths, [])

    def test_partial_affect(self):
        """Test when only some changed projects affect the stage."""
        paths = compute_stage_sparse_checkout(
            "math-libs", "projects/rocprim,projects/rpp"
        )
        # rocprim affects math-libs, rpp does not
        self.assertIn("projects/rocprim", paths)
        self.assertNotIn("projects/rpp", paths)

    def test_paths_are_sorted(self):
        """Test that returned paths are sorted."""
        paths = compute_stage_sparse_checkout(
            "math-libs", "shared/rocroller,projects/rocprim"
        )
        self.assertEqual(paths, sorted(paths))

    def test_includes_build_dependency_paths(self):
        """Test that build dependencies from same source_set are included.

        When rocprim changes, we need its artifact siblings (hipcub, rocthrust)
        AND its build dependencies that are in the same source_set (rand).
        The 'prim' artifact has artifact_deps=['rand', 'core-hip', ...].
        'rand' is in rocm-libraries (source_paths: rocrand, hiprand).
        """
        paths = compute_stage_sparse_checkout("math-libs", "projects/rocprim")

        # Should include siblings from 'prim' artifact
        self.assertIn("projects/rocprim", paths)
        self.assertIn("projects/hipcub", paths)
        self.assertIn("projects/rocthrust", paths)

        # Should include dependencies from same source_set (rocm-libraries)
        # 'prim' depends on 'rand' which has source_paths [rocrand, hiprand]
        self.assertIn("projects/rocrand", paths)
        self.assertIn("projects/hiprand", paths)

    def test_miopen_includes_blas_and_ck_dependencies(self):
        """Test that miopen checkout includes blas and composable-kernel deps.

        miopen has artifact_deps=['blas', 'composable-kernel', 'rand', ...].
        All of these have source_paths in rocm-libraries.
        """
        paths = compute_stage_sparse_checkout("math-libs", "projects/miopen")

        # Should include miopen itself
        self.assertIn("projects/miopen", paths)

        # Should include composable-kernel dependency
        self.assertIn("projects/composablekernel", paths)

        # Should include blas dependencies (rocblas, hipblas, etc.)
        self.assertIn("projects/rocblas", paths)
        self.assertIn("projects/hipblas", paths)

        # Should include rand dependency
        self.assertIn("projects/rocrand", paths)

    def test_excludes_dependencies_from_other_source_sets(self):
        """Test that dependencies from different source_sets are NOT included.

        Many artifacts depend on core-hip, core-runtime, amd-llvm, etc.
        These are in different source_sets (rocm-systems, compilers) and
        should NOT be included in rocm-libraries sparse checkout.
        """
        paths = compute_stage_sparse_checkout("math-libs", "projects/rocprim")

        # core-hip, clr, hip are in rocm-systems, not rocm-libraries
        # They should NOT appear in the checkout paths
        self.assertNotIn("projects/clr", paths)
        self.assertNotIn("projects/hip", paths)
        self.assertNotIn("projects/rocr-runtime", paths)


class TestComputeAllStageSparseCheckouts(unittest.TestCase):
    """Tests for compute_all_stage_sparse_checkouts function."""

    def test_returns_dict_for_all_stages(self):
        """Test that result contains entries for all stages in topology."""
        result = compute_all_stage_sparse_checkouts("projects/rocprim")
        # Should have entries for math-libs since rocprim affects it
        self.assertIn("math-libs", result)
        # The result should be a newline-separated string for affected stages
        self.assertIn("projects/rocprim", result["math-libs"])

    def test_empty_changed_projects_returns_empty_dict(self):
        """Test that empty changed_projects returns empty dict."""
        result = compute_all_stage_sparse_checkouts("")
        self.assertEqual(result, {})

    def test_unaffected_stages_have_empty_string(self):
        """Test that unaffected stages have empty string values."""
        result = compute_all_stage_sparse_checkouts("projects/rocprim")
        # cv-libs should not be affected by rocprim
        self.assertIn("cv-libs", result)
        self.assertEqual(result["cv-libs"], "")

    def test_affected_stages_have_newline_separated_paths(self):
        """Test that affected stages have newline-separated paths."""
        result = compute_all_stage_sparse_checkouts("projects/rocprim,shared/rocroller")
        # math-libs should be affected by both
        paths = result["math-libs"]
        self.assertIn("projects/rocprim", paths)
        self.assertIn("shared/rocroller", paths)
        # Paths should be newline-separated
        self.assertIn("\n", paths)


class TestConfigJsonSparseCheckout(unittest.TestCase):
    """Tests that sparse_checkout_by_stage is included in config_json output."""

    def setUp(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            self.temp_file = f.name
        os.environ["GITHUB_OUTPUT"] = self.temp_file

    def tearDown(self):
        if "GITHUB_OUTPUT" in os.environ:
            del os.environ["GITHUB_OUTPUT"]
        if hasattr(self, "temp_file") and os.path.exists(self.temp_file):
            os.unlink(self.temp_file)

    def test_sparse_checkout_by_stage_in_config_json(self):
        """Test that sparse_checkout_by_stage is included in config_json."""
        rc = detect_external_repo_config_main(
            [
                "--repository",
                "rocm-libraries",
                "--changed-projects",
                "projects/rocprim",
            ]
        )
        self.assertEqual(rc, 0)

        with open(self.temp_file, "r") as f:
            output = f.read()

        self.assertIn("config_json=", output)
        self.assertIn("sparse_checkout_by_stage", output)

    def test_sparse_checkout_empty_when_no_changed_projects(self):
        """Test that sparse_checkout_by_stage is empty when no changed_projects."""
        rc = detect_external_repo_config_main(
            [
                "--repository",
                "rocm-libraries",
            ]
        )
        self.assertEqual(rc, 0)

        with open(self.temp_file, "r") as f:
            output = f.read()

        # Parse the config_json from output
        import json
        import re

        match = re.search(r"config_json=(.+)", output)
        self.assertIsNotNone(match)
        config_json = json.loads(match.group(1))
        self.assertEqual(config_json["sparse_checkout_by_stage"], {})


if __name__ == "__main__":
    unittest.main()
