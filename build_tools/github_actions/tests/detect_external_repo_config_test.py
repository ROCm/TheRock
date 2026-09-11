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
    build_source_path_prefix_map,
    compute_all_stage_sparse_checkouts,
    compute_stage_sparse_checkout,
    extract_source_path_from_project,
    get_all_topology_source_paths,
    get_repo_config,
    get_external_repo_path,
    get_stage_source_paths,
    get_skip_patterns,
    get_test_list,
    import_external_repo_module,
    main as detect_external_repo_config_main,
    normalize_changed_projects,
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
        self.assertEqual(normalize_changed_projects(""), "")
        self.assertEqual(normalize_changed_projects("  "), "")

    def test_single_project(self):
        """Single project path should be normalized."""
        result = normalize_changed_projects("projects/rocprim")
        self.assertEqual(result, "projects/rocprim")

    def test_dedupes_and_sorts(self):
        """Multiple projects should be deduplicated and sorted."""
        result = normalize_changed_projects("projects/b,projects/a,projects/b")
        self.assertEqual(result, "projects/a,projects/b")


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

    def test_empty_path(self):
        """Test extraction from empty path."""
        self.assertIsNone(extract_source_path_from_project(""))

    def test_nested_project_maps_to_parent(self):
        """Nested projects like tensilelite should map to parent hipblaslt.

        The build topology defines hipblaslt as a source_path for the blas artifact,
        but tensilelite is a nested subtree inside hipblaslt in rocm-libraries.
        This should correctly map to hipblaslt.
        """
        result = extract_source_path_from_project("projects/hipblaslt/tensilelite")
        self.assertEqual(result, "hipblaslt")

    def test_single_component(self):
        """Single component paths should return as-is."""
        result = extract_source_path_from_project("rocblas")
        self.assertEqual(result, "rocblas")

    def test_unknown_nested_project_returns_last(self):
        """Unknown nested paths should return the last component as fallback."""
        result = extract_source_path_from_project("projects/unknown/nested")
        self.assertEqual(result, "nested")


class TestGetStageSourcePaths(unittest.TestCase):
    """Tests for get_stage_source_paths function."""

    def test_math_libs_includes_expected(self):
        """Test math-libs stage includes expected source paths."""
        source_paths = get_stage_source_paths("math-libs")
        self.assertIn("rocprim", source_paths)
        self.assertIn("rocblas", source_paths)

    def test_unknown_stage_returns_empty(self):
        """Test unknown stage returns empty set."""
        self.assertEqual(get_stage_source_paths("unknown-stage"), set())


class TestComputeStageSparseCheckout(unittest.TestCase):
    """Tests for compute_stage_sparse_checkout function."""

    def test_affected_stage(self):
        """Test stage that is affected by changed projects."""
        paths, unmapped = compute_stage_sparse_checkout(
            "math-libs", "projects/rocprim,shared/rocroller"
        )
        self.assertIn("projects/rocprim", paths)
        self.assertIn("shared/rocroller", paths)
        self.assertEqual(unmapped, [])

    def test_unaffected_stage(self):
        """Test stage that is NOT affected by changed projects."""
        paths, unmapped = compute_stage_sparse_checkout("cv-libs", "projects/rocprim")
        self.assertEqual(paths, [])

    def test_empty_changed_projects(self):
        """Test with empty changed_projects."""
        paths, unmapped = compute_stage_sparse_checkout("math-libs", "")
        self.assertEqual(paths, [])
        self.assertEqual(unmapped, [])

    def test_includes_artifact_siblings(self):
        """rocprim should include hipcub, rocthrust (same artifact)."""
        paths, unmapped = compute_stage_sparse_checkout("math-libs", "projects/rocprim")
        self.assertIn("projects/rocprim", paths)
        self.assertIn("projects/hipcub", paths)
        self.assertIn("projects/rocthrust", paths)
        self.assertEqual(unmapped, [])

    def test_includes_build_deps_from_same_source_set(self):
        """rocprim's 'prim' artifact depends on 'rand' (rocrand, hiprand)."""
        paths, unmapped = compute_stage_sparse_checkout("math-libs", "projects/rocprim")
        self.assertIn("projects/rocrand", paths)
        self.assertIn("projects/hiprand", paths)

    def test_excludes_deps_from_other_source_sets(self):
        """Dependencies in rocm-systems (clr, hip) should NOT appear."""
        paths, unmapped = compute_stage_sparse_checkout("math-libs", "projects/rocprim")
        self.assertNotIn("projects/clr", paths)
        self.assertNotIn("projects/hip", paths)

    def test_hipblaslt_tensilelite_maps_correctly(self):
        """Nested tensilelite path should map to hipblaslt and include blas siblings."""
        paths, unmapped = compute_stage_sparse_checkout(
            "math-libs", "projects/hipblaslt/tensilelite"
        )
        # tensilelite -> hipblaslt -> blas artifact
        self.assertIn("projects/hipblaslt", paths)
        self.assertIn("projects/rocblas", paths)
        self.assertIn("projects/hipblas", paths)
        self.assertEqual(unmapped, [])

    def test_unmapped_project_is_tracked(self):
        """Projects that can't be mapped to topology should be in unmapped list."""
        paths, unmapped = compute_stage_sparse_checkout(
            "math-libs", "projects/totally_unknown_project"
        )
        self.assertIn("projects/totally_unknown_project", unmapped)

    def test_mixed_mapped_and_unmapped(self):
        """Mix of mapped and unmapped projects should work correctly."""
        paths, unmapped = compute_stage_sparse_checkout(
            "math-libs", "projects/rocprim,projects/unknown_project"
        )
        self.assertIn("projects/rocprim", paths)
        self.assertIn("projects/unknown_project", unmapped)


class TestComputeAllStageSparseCheckouts(unittest.TestCase):
    """Tests for compute_all_stage_sparse_checkouts function."""

    def test_returns_dict_for_all_stages(self):
        """Test that result contains entries for affected stages."""
        result, has_unmapped = compute_all_stage_sparse_checkouts("projects/rocprim")
        self.assertIn("math-libs", result)
        self.assertIn("projects/rocprim", result["math-libs"])
        self.assertFalse(has_unmapped)

    def test_empty_changed_projects_returns_empty_dict(self):
        """Test that empty changed_projects returns empty dict."""
        result, has_unmapped = compute_all_stage_sparse_checkouts("")
        self.assertEqual(result, {})
        self.assertFalse(has_unmapped)

    def test_unaffected_stages_have_empty_string(self):
        """Test that unaffected stages have empty string values."""
        result, has_unmapped = compute_all_stage_sparse_checkouts("projects/rocprim")
        self.assertEqual(result["cv-libs"], "")

    def test_unmapped_projects_signal_fallback(self):
        """Test that unmapped projects set has_unmapped to True."""
        result, has_unmapped = compute_all_stage_sparse_checkouts(
            "projects/rocprim,projects/totally_unknown"
        )
        self.assertTrue(has_unmapped)


class TestBuildSourcePathPrefixMap(unittest.TestCase):
    """Tests for build_source_path_prefix_map function."""

    def test_projects_prefix(self):
        """Standard project paths should have 'projects/' prefix."""
        result = build_source_path_prefix_map("projects/rocblas,projects/hipblas")
        self.assertEqual(result.get("rocblas"), "projects/")
        self.assertEqual(result.get("hipblas"), "projects/")

    def test_shared_prefix(self):
        """Shared paths should have 'shared/' prefix."""
        result = build_source_path_prefix_map("shared/rocroller")
        self.assertEqual(result.get("rocroller"), "shared/")

    def test_mixed_prefixes(self):
        """Mixed prefixes should be correctly extracted."""
        result = build_source_path_prefix_map(
            "projects/rocblas,shared/rocroller,projects/hipblaslt"
        )
        self.assertEqual(result.get("rocblas"), "projects/")
        self.assertEqual(result.get("rocroller"), "shared/")
        self.assertEqual(result.get("hipblaslt"), "projects/")

    def test_nested_project_extracts_parent_prefix(self):
        """Nested projects should extract the prefix to the parent."""
        result = build_source_path_prefix_map("projects/hipblaslt/tensilelite")
        self.assertEqual(result.get("hipblaslt"), "projects/")

    def test_empty_string(self):
        """Empty string should return empty dict."""
        result = build_source_path_prefix_map("")
        self.assertEqual(result, {})


class TestGetAllTopologySourcePaths(unittest.TestCase):
    """Tests for get_all_topology_source_paths function."""

    def test_returns_known_source_paths(self):
        """Should return known source_paths from topology."""
        result = get_all_topology_source_paths()
        self.assertIn("rocblas", result)
        self.assertIn("hipblaslt", result)
        self.assertIn("rocprim", result)
        self.assertIn("miopen", result)

    def test_returns_set(self):
        """Should return a set."""
        result = get_all_topology_source_paths()
        self.assertIsInstance(result, set)


if __name__ == "__main__":
    unittest.main()
