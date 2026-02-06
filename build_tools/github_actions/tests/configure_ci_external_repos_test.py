#!/usr/bin/env python3

import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from configure_ci import (
    parse_projects_input,
    cross_product_projects_with_gpu_variants,
)


class TestParseProjectsInput(unittest.TestCase):
    """Test parse_projects_input() - pure logic, no mocking needed."""

    def test_empty_input_returns_empty_list(self):
        """Test that empty or whitespace input returns empty list."""
        self.assertEqual(parse_projects_input(""), [])
        self.assertEqual(parse_projects_input("   "), [])
        self.assertEqual(parse_projects_input("all"), [])

    def test_single_project(self):
        """Test parsing single project."""
        result = parse_projects_input("rocprim")
        self.assertEqual(result, ["rocprim"])

    def test_multiple_projects(self):
        """Test parsing multiple comma-separated projects."""
        result = parse_projects_input("rocprim,rocblas,rocfft")
        self.assertEqual(result, ["rocprim", "rocblas", "rocfft"])

    def test_strips_projects_prefix(self):
        """Test that 'projects/' prefix is stripped."""
        result = parse_projects_input("projects/rocprim,projects/rocblas")
        self.assertEqual(result, ["rocprim", "rocblas"])

    def test_handles_whitespace(self):
        """Test that whitespace around commas is handled."""
        result = parse_projects_input("rocprim , rocblas , rocfft")
        self.assertEqual(result, ["rocprim", "rocblas", "rocfft"])

    def test_mixed_prefixes(self):
        """Test handling of mixed prefixed and non-prefixed projects."""
        result = parse_projects_input("rocprim,projects/rocblas")
        self.assertEqual(result, ["rocprim", "rocblas"])


class TestCrossProductProjectsWithGpuVariants(unittest.TestCase):
    """Test cross_product_projects_with_gpu_variants() - pure logic, no mocking needed."""

    def test_single_project_single_variant(self):
        """Test cross-product with one project and one GPU variant."""
        project_configs = [{"projects_to_test": "rocprim,rocblas"}]
        gpu_variants = [{"family": "gfx94x", "platform": "linux"}]

        result = cross_product_projects_with_gpu_variants(project_configs, gpu_variants)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["family"], "gfx94x")
        self.assertEqual(result[0]["platform"], "linux")
        self.assertEqual(result[0]["projects_to_test"], "rocprim,rocblas")
        self.assertNotIn("cmake_options", result[0])

    def test_single_project_multiple_variants(self):
        """Test cross-product with one project and multiple GPU variants."""
        project_configs = [{"projects_to_test": "rocprim"}]
        gpu_variants = [
            {"family": "gfx94x", "platform": "linux"},
            {"family": "gfx110x", "platform": "linux"},
            {"family": "gfx94x", "platform": "windows"},
        ]

        result = cross_product_projects_with_gpu_variants(project_configs, gpu_variants)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["projects_to_test"], "rocprim")
        self.assertEqual(result[0]["family"], "gfx94x")
        self.assertEqual(result[0]["platform"], "linux")
        self.assertEqual(result[1]["projects_to_test"], "rocprim")
        self.assertEqual(result[1]["family"], "gfx110x")
        self.assertEqual(result[1]["platform"], "linux")
        self.assertEqual(result[2]["projects_to_test"], "rocprim")
        self.assertEqual(result[2]["family"], "gfx94x")
        self.assertEqual(result[2]["platform"], "windows")

    def test_multiple_projects_single_variant(self):
        """Test cross-product with multiple projects and one GPU variant."""
        project_configs = [
            {"projects_to_test": "rocprim"},
            {"projects_to_test": "rocblas"},
            {"projects_to_test": "rocfft"},
        ]
        gpu_variants = [{"family": "gfx94x", "platform": "linux"}]

        result = cross_product_projects_with_gpu_variants(project_configs, gpu_variants)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["projects_to_test"], "rocprim")
        self.assertEqual(result[0]["family"], "gfx94x")
        self.assertEqual(result[1]["projects_to_test"], "rocblas")
        self.assertEqual(result[1]["family"], "gfx94x")
        self.assertEqual(result[2]["projects_to_test"], "rocfft")
        self.assertEqual(result[2]["family"], "gfx94x")

    def test_multiple_projects_multiple_variants(self):
        """Test cross-product with multiple projects and multiple GPU variants."""
        project_configs = [
            {"projects_to_test": "rocprim"},
            {"projects_to_test": "rocblas"},
        ]
        gpu_variants = [
            {"family": "gfx94x", "platform": "linux"},
            {"family": "gfx110x", "platform": "linux"},
        ]

        result = cross_product_projects_with_gpu_variants(project_configs, gpu_variants)

        self.assertEqual(len(result), 4)  # 2 projects * 2 variants = 4
        # Verify all combinations exist
        combinations = [
            (r["projects_to_test"], r["family"], r["platform"]) for r in result
        ]
        self.assertIn(("rocprim", "gfx94x", "linux"), combinations)
        self.assertIn(("rocprim", "gfx110x", "linux"), combinations)
        self.assertIn(("rocblas", "gfx94x", "linux"), combinations)
        self.assertIn(("rocblas", "gfx110x", "linux"), combinations)

    def test_empty_project_configs(self):
        """Test cross-product with empty project configs."""
        project_configs = []
        gpu_variants = [{"family": "gfx94x", "platform": "linux"}]

        result = cross_product_projects_with_gpu_variants(project_configs, gpu_variants)

        self.assertEqual(len(result), 0)

    def test_empty_gpu_variants(self):
        """Test cross-product with empty GPU variants."""
        project_configs = [{"projects_to_test": "rocprim"}]
        gpu_variants = []

        result = cross_product_projects_with_gpu_variants(project_configs, gpu_variants)

        self.assertEqual(len(result), 0)

    def test_preserves_gpu_variant_fields(self):
        """Test that all GPU variant fields are preserved in result."""
        project_configs = [{"projects_to_test": "rocprim"}]
        gpu_variants = [
            {
                "family": "gfx94x",
                "platform": "linux",
                "build_variant": "release",
                "extra_field": "value",
            }
        ]

        result = cross_product_projects_with_gpu_variants(project_configs, gpu_variants)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["family"], "gfx94x")
        self.assertEqual(result[0]["platform"], "linux")
        self.assertEqual(result[0]["build_variant"], "release")
        self.assertEqual(result[0]["extra_field"], "value")
        self.assertEqual(result[0]["projects_to_test"], "rocprim")

    def test_no_cmake_options_in_result(self):
        """Test that cmake_options are NOT included in result (full builds only)."""
        project_configs = [
            {"projects_to_test": "rocprim", "cmake_options": "-DROCBLAS=ON"}
        ]
        gpu_variants = [{"family": "gfx94x", "platform": "linux"}]

        result = cross_product_projects_with_gpu_variants(project_configs, gpu_variants)

        self.assertEqual(len(result), 1)
        self.assertNotIn("cmake_options", result[0])
        self.assertEqual(result[0]["projects_to_test"], "rocprim")

    def test_complex_gpu_variant_structure(self):
        """Test with complex GPU variant structure."""
        project_configs = [
            {"projects_to_test": "rocprim,rocblas"},
            {"projects_to_test": "rocfft"},
        ]
        gpu_variants = [
            {
                "family": "gfx94x",
                "platform": "linux",
                "build_variant": "release",
                "test_labels": ["smoke"],
            },
            {
                "family": "gfx110x",
                "platform": "windows",
                "build_variant": "debug",
                "test_labels": ["full"],
            },
        ]

        result = cross_product_projects_with_gpu_variants(project_configs, gpu_variants)

        self.assertEqual(len(result), 4)  # 2 projects * 2 variants
        # Verify structure is preserved
        for r in result:
            self.assertIn("family", r)
            self.assertIn("platform", r)
            self.assertIn("build_variant", r)
            self.assertIn("test_labels", r)
            self.assertIn("projects_to_test", r)
            self.assertNotIn("cmake_options", r)


if __name__ == "__main__":
    unittest.main()
