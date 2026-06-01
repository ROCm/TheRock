# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import os
import sys
import tempfile
import unittest
import yaml
from unittest.mock import patch, MagicMock

# Add test_executable_scripts to PYTHONPATH
sys.path.insert(0, os.fspath(Path(__file__).parent.parent / "test_executable_scripts"))

import pytest_runner


class PytestRunnerTest(unittest.TestCase):
    def setUp(self):
        """Save environment so tests don't leak state"""
        self._orig_env = os.environ.copy()

        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_dir = Path(self.temp_dir) / "Tensile" / "Tests"
        self.test_dir.mkdir(parents=True)

        # Create a minimal test_categories.yaml
        self.categories_yaml = Path(self.temp_dir) / "test_categories.yaml"
        self.categories_data = {
            "test_categories": {
                "quick": {
                    "description": "Fast unit tests",
                    "pytest_markers": ["unit"],
                    "exclude_markers": ["disabled"],
                    "labels": ["quick"],
                },
                "standard": {
                    "description": "Pre-checkin validation",
                    "pytest_markers": ["pre_checkin"],
                    "exclude_markers": ["disabled"],
                    "gtest_binaries": [
                        {
                            "name": "TensileTests",
                            "path": "bin/TensileTests",
                            "filter": "--gtest_filter=-*Extended*",
                        }
                    ],
                    "labels": ["standard"],
                },
            },
            "execution_settings": {
                "default_timeout": 300,
                "timeout_multiplier": 1,
                "parallel_workers": 4,
                "category_timeouts": {
                    "quick": 300,
                    "standard": 1800,
                },
                "environment": {
                    "TENSILE_ROCM_ASSEMBLER_PATH": "{ROCM_PATH}/bin/clang++",
                },
            },
        }

        with open(self.categories_yaml, "w") as f:
            yaml.dump(self.categories_data, f)

        # Set required environment variables
        os.environ["TEST_COMPONENT"] = "tensile"
        os.environ["TEST_TYPE"] = "quick"
        os.environ["THEROCK_BIN_DIR"] = str(self.temp_dir)
        os.environ["AMDGPU_FAMILIES"] = "gfx94X"
        os.environ["SHARD_INDEX"] = "1"
        os.environ["TOTAL_SHARDS"] = "1"
        os.environ["ROCM_PATH"] = "/opt/rocm"

    def tearDown(self):
        """Restore environment"""
        os.environ.clear()
        os.environ.update(self._orig_env)

        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -----------------------
    # Environment validation tests
    # -----------------------

    def test_missing_test_component_fails(self):
        """TEST_COMPONENT is required"""
        del os.environ["TEST_COMPONENT"]

        with self.assertRaises(SystemExit):
            pytest_runner.main()

    def test_missing_therock_bin_dir_fails(self):
        """THEROCK_BIN_DIR is required for installed components"""
        del os.environ["THEROCK_BIN_DIR"]

        with self.assertRaises(SystemExit):
            pytest_runner.main()

    def test_invalid_test_type_fails(self):
        """Invalid TEST_TYPE should fail"""
        os.environ["TEST_TYPE"] = "invalid_category"

        with self.assertRaises(SystemExit):
            pytest_runner.main()

    # -----------------------
    # test_categories.yaml parsing tests
    # -----------------------

    def test_loads_categories_yaml(self):
        """Should successfully load and parse test_categories.yaml"""
        # Mock component_path to point to our temp directory
        with patch("pytest_runner.Path") as mock_path:
            mock_component_path = MagicMock()
            mock_component_path.exists.return_value = True
            mock_component_path.__truediv__ = lambda self, other: (
                self.categories_yaml if other == "test_categories.yaml"
                else Path(self.temp_dir) / other
            )
            mock_path.return_value = mock_component_path

            # Should not raise
            config = pytest_runner.load_test_categories(mock_component_path)
            self.assertIn("test_categories", config)
            self.assertIn("quick", config["test_categories"])
            self.assertIn("standard", config["test_categories"])

    def test_missing_categories_yaml_fails(self):
        """Missing test_categories.yaml should fail gracefully"""
        self.categories_yaml.unlink()

        component_path = Path(self.temp_dir)
        with self.assertRaises(FileNotFoundError):
            pytest_runner.load_test_categories(component_path)

    # -----------------------
    # Pytest marker building tests
    # -----------------------

    def test_builds_marker_expression_for_quick(self):
        """Should build correct pytest -m expression for quick category"""
        category_config = self.categories_data["test_categories"]["quick"]

        marker_expr = pytest_runner.build_marker_expression(
            category_config, gpu_arch="gfx94X"
        )

        # Should have unit marker
        self.assertIn("unit", marker_expr)
        # Should exclude disabled
        self.assertIn("not disabled", marker_expr)
        # Should have GPU skip markers
        self.assertIn("not skip-gfx94X", marker_expr)

    def test_builds_marker_expression_with_multiple_markers(self):
        """Should combine multiple pytest markers with 'or'"""
        category_config = {
            "pytest_markers": ["extended", "nightly"],
            "exclude_markers": ["disabled", "weekly"],
        }

        marker_expr = pytest_runner.build_marker_expression(
            category_config, gpu_arch="gfx942"
        )

        # Should OR the include markers
        self.assertIn("extended or nightly", marker_expr)
        # Should AND exclude each marker
        self.assertIn("not disabled", marker_expr)
        self.assertIn("not weekly", marker_expr)
        # Should have GPU skip markers
        self.assertIn("not skip-gfx942", marker_expr)

    # -----------------------
    # GPU architecture filtering tests
    # -----------------------

    def test_extracts_gpu_arch_from_families(self):
        """Should extract base GPU arch from AMDGPU_FAMILIES"""
        test_cases = [
            ("gfx94X", "gfx94X"),
            ("gfx94X-dcgpu", "gfx94X"),
            ("gfx1151", "gfx1151"),
            ("gfx942", "gfx942"),
        ]

        for family, expected_arch in test_cases:
            arch = pytest_runner.extract_gpu_arch(family)
            self.assertEqual(arch, expected_arch)

    def test_generates_gpu_skip_markers(self):
        """Should generate hierarchical GPU skip markers"""
        skip_markers = pytest_runner.generate_gpu_skip_markers("gfx1151")

        # Should have all hierarchy levels
        self.assertIn("skip-gfx1151", skip_markers)
        self.assertIn("skip-gfx115X", skip_markers)
        self.assertIn("skip-gfx11X", skip_markers)

    # -----------------------
    # Pytest sharding tests
    # -----------------------

    def test_shards_tests_correctly(self):
        """Should shard tests using modulo arithmetic like GTest"""
        test_ids = [
            "test_1.py::test_a",
            "test_1.py::test_b",
            "test_2.py::test_c",
            "test_2.py::test_d",
            "test_3.py::test_e",
        ]

        # Shard 1 of 2: indices 0, 2, 4
        shard1 = pytest_runner.shard_tests(test_ids, shard_index=1, total_shards=2)
        self.assertEqual(len(shard1), 3)
        self.assertEqual(shard1, [test_ids[0], test_ids[2], test_ids[4]])

        # Shard 2 of 2: indices 1, 3
        shard2 = pytest_runner.shard_tests(test_ids, shard_index=2, total_shards=2)
        self.assertEqual(len(shard2), 2)
        self.assertEqual(shard2, [test_ids[1], test_ids[3]])

    def test_single_shard_returns_all_tests(self):
        """Single shard should return all tests"""
        test_ids = ["test_1", "test_2", "test_3"]

        result = pytest_runner.shard_tests(test_ids, shard_index=1, total_shards=1)
        self.assertEqual(result, test_ids)

    # -----------------------
    # GTest binary execution tests
    # -----------------------

    def test_detects_gtest_binaries_in_category(self):
        """Should detect gtest_binaries field in standard category"""
        category_config = self.categories_data["test_categories"]["standard"]

        self.assertIn("gtest_binaries", category_config)
        binaries = category_config["gtest_binaries"]
        self.assertEqual(len(binaries), 1)
        self.assertEqual(binaries[0]["name"], "TensileTests")
        self.assertEqual(binaries[0]["path"], "bin/TensileTests")
        self.assertIn("--gtest_filter", binaries[0]["filter"])

    @patch("subprocess.run")
    def test_executes_gtest_binary(self, mock_run):
        """Should execute GTest binary with correct arguments"""
        mock_run.return_value = MagicMock(returncode=0)

        binary_config = {
            "name": "TensileTests",
            "path": "bin/TensileTests",
            "filter": "--gtest_filter=-*Extended*",
        }

        binary_path = Path(self.temp_dir) / binary_config["path"]
        binary_path.parent.mkdir(parents=True, exist_ok=True)
        binary_path.touch()
        binary_path.chmod(0o755)

        pytest_runner.run_gtest_binary(binary_config, Path(self.temp_dir))

        # Verify subprocess.run was called with correct arguments
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        self.assertIn("TensileTests", str(call_args[0]))
        self.assertIn("--gtest_filter=-*Extended*", call_args)
        self.assertIn("--gtest_color=yes", call_args)

    @patch("subprocess.run")
    def test_gtest_failure_exits(self, mock_run):
        """GTest binary failure should exit with non-zero"""
        mock_run.return_value = MagicMock(returncode=1)

        binary_config = {
            "name": "TensileTests",
            "path": "bin/TensileTests",
            "filter": "",
        }

        binary_path = Path(self.temp_dir) / binary_config["path"]
        binary_path.parent.mkdir(parents=True, exist_ok=True)
        binary_path.touch()
        binary_path.chmod(0o755)

        with self.assertRaises(SystemExit) as cm:
            pytest_runner.run_gtest_binary(binary_config, Path(self.temp_dir))

        self.assertEqual(cm.exception.code, 1)

    # -----------------------
    # Environment variable expansion tests
    # -----------------------

    def test_expands_rocm_path_in_environment(self):
        """Should expand {ROCM_PATH} in environment variables"""
        os.environ["ROCM_PATH"] = "/opt/rocm"

        env_settings = {
            "TENSILE_ROCM_ASSEMBLER_PATH": "{ROCM_PATH}/bin/clang++",
            "SOME_OTHER_VAR": "static_value",
        }

        expanded = pytest_runner.expand_environment_vars(env_settings)

        self.assertEqual(
            expanded["TENSILE_ROCM_ASSEMBLER_PATH"], "/opt/rocm/bin/clang++"
        )
        self.assertEqual(expanded["SOME_OTHER_VAR"], "static_value")


if __name__ == "__main__":
    unittest.main()
