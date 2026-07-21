# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for compute_affected_stages.py.

Tests the selective-build logic including:
- Empty input handling
- Project path normalization (projects/x -> x)
- TEST_SUBPROJECTS expansion
- Stage mapping
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import compute_affected_stages as cas


class TestComputeAffected(unittest.TestCase):
    """Tests for compute_affected() function."""

    def test_empty_input_returns_all(self):
        """Empty changed_projects should build all stages."""
        stages, projects = cas.compute_affected("")
        self.assertEqual(stages, "all")
        self.assertEqual(projects, "")

    def test_whitespace_only_returns_all(self):
        """Whitespace-only input should build all stages."""
        stages, projects = cas.compute_affected("   ")
        self.assertEqual(stages, "all")
        self.assertEqual(projects, "")

    def test_none_input_returns_all(self):
        """None input should build all stages."""
        # The function checks `not changed_projects` which handles None
        stages, projects = cas.compute_affected(None)
        self.assertEqual(stages, "all")
        self.assertEqual(projects, "")

    def test_normalizes_project_paths(self):
        """Projects with path prefix should be normalized."""
        with patch.object(cas, "get_subprojects_to_test") as mock_expand:
            with patch.object(cas, "get_topology") as mock_topo:
                mock_expand.return_value = {"hip"}
                mock_topo_instance = MagicMock()
                mock_topo_instance.get_stages_for_projects.return_value = {"core"}
                mock_topo.return_value = mock_topo_instance

                stages, projects = cas.compute_affected("projects/hip")

                # Should normalize "projects/hip" to "hip"
                mock_expand.assert_called_once()
                call_args = mock_expand.call_args[0][0]
                self.assertEqual(call_args, ["hip"])

    def test_expands_test_subprojects(self):
        """Changed projects should expand to include TEST_SUBPROJECTS dependencies."""
        with patch.object(cas, "get_subprojects_to_test") as mock_expand:
            with patch.object(cas, "get_topology") as mock_topo:
                # rocprim triggers rocsparse, rocthrust, etc. via TEST_SUBPROJECTS
                mock_expand.return_value = {"rocprim", "rocsparse", "rocthrust"}
                mock_topo_instance = MagicMock()
                mock_topo_instance.get_stages_for_projects.return_value = {"math-libs"}
                mock_topo.return_value = mock_topo_instance

                stages, projects = cas.compute_affected("rocprim")

                # Verify expansion was called with normalized projects
                mock_expand.assert_called_once()
                # Verify expanded projects are in output
                self.assertIn("rocprim", projects)
                self.assertIn("rocsparse", projects)
                self.assertIn("rocthrust", projects)

    def test_maps_projects_to_stages(self):
        """Projects should be mapped to their build stages."""
        with patch.object(cas, "get_subprojects_to_test") as mock_expand:
            with patch.object(cas, "get_topology") as mock_topo:
                mock_expand.return_value = {"hip", "clr"}
                mock_topo_instance = MagicMock()
                mock_topo_instance.get_stages_for_projects.return_value = {
                    "core",
                    "compiler-runtime",
                }
                mock_topo.return_value = mock_topo_instance

                stages, projects = cas.compute_affected("hip,clr")

                # Stages should be comma-separated and sorted
                self.assertEqual(stages, "compiler-runtime,core")

    def test_no_stages_found_returns_all(self):
        """If no stages found for projects, should fall back to all."""
        with patch.object(cas, "get_subprojects_to_test") as mock_expand:
            with patch.object(cas, "get_topology") as mock_topo:
                mock_expand.return_value = {"unknown_project"}
                mock_topo_instance = MagicMock()
                mock_topo_instance.get_stages_for_projects.return_value = set()
                mock_topo.return_value = mock_topo_instance

                stages, projects = cas.compute_affected("unknown_project")

                self.assertEqual(stages, "all")
                self.assertEqual(projects, "")


if __name__ == "__main__":
    unittest.main()
