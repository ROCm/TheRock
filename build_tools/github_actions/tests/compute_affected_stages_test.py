# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for compute_affected_stages.py.

Tests the selective-build logic including:
- Empty input handling
- Project path normalization (projects/x -> x)
- TEST_SUBPROJECTS expansion
- Stage mapping
- Integration with plan_stage_reuse
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import compute_affected_stages as cas
from stage_reuse_decision import StageReusePlan


class TestComputeAffected(unittest.TestCase):
    """Tests for compute_affected() function.

    compute_affected returns (affected_stages, prebuilt_stages, expanded_projects):
    - affected_stages: comma-separated stages to build, or "all"
    - prebuilt_stages: comma-separated stages to skip (prebuilt), or ""
    - expanded_projects: space-separated project names, or ""
    """

    def test_empty_input_returns_all(self):
        """Empty changed_projects should build all stages."""
        affected, prebuilt, projects = cas.compute_affected("")
        self.assertEqual(affected, "all")
        self.assertEqual(prebuilt, "")
        self.assertEqual(projects, "")

    def test_whitespace_only_returns_all(self):
        """Whitespace-only input should build all stages."""
        affected, prebuilt, projects = cas.compute_affected("   ")
        self.assertEqual(affected, "all")
        self.assertEqual(prebuilt, "")
        self.assertEqual(projects, "")

    def test_none_input_returns_all(self):
        """None input should build all stages."""
        # The function checks `not changed_projects` which handles None
        affected, prebuilt, projects = cas.compute_affected(None)
        self.assertEqual(affected, "all")
        self.assertEqual(prebuilt, "")
        self.assertEqual(projects, "")

    def test_normalizes_project_paths(self):
        """Projects with path prefix should be normalized."""
        with patch.object(cas, "get_subprojects_to_test") as mock_expand:
            with patch.object(cas, "plan_stage_reuse") as mock_plan:
                mock_expand.return_value = {"hip"}
                mock_plan.return_value = StageReusePlan(
                    candidate_stages=("other-stage",),
                    rebuild_stages=("core",),
                    full_rebuild_required=False,
                    reasons=(),
                )

                affected, prebuilt, projects = cas.compute_affected("projects/hip")

                # Should normalize "projects/hip" to "hip"
                mock_expand.assert_called_once()
                call_args = mock_expand.call_args[0][0]
                self.assertEqual(call_args, ["hip"])

    def test_expands_test_subprojects(self):
        """Changed projects should expand to include TEST_SUBPROJECTS dependencies."""
        with patch.object(cas, "get_subprojects_to_test") as mock_expand:
            with patch.object(cas, "plan_stage_reuse") as mock_plan:
                # rocprim triggers rocsparse, rocthrust, etc. via TEST_SUBPROJECTS
                mock_expand.return_value = {"rocprim", "rocsparse", "rocthrust"}
                mock_plan.return_value = StageReusePlan(
                    candidate_stages=("other-stage",),
                    rebuild_stages=("math-libs",),
                    full_rebuild_required=False,
                    reasons=(),
                )

                affected, prebuilt, projects = cas.compute_affected("rocprim")

                # Verify expansion was called with normalized projects
                mock_expand.assert_called_once()
                # Verify expanded projects are in output (space-separated)
                self.assertIn("rocprim", projects)
                self.assertIn("rocsparse", projects)
                self.assertIn("rocthrust", projects)

    def test_maps_projects_to_stages(self):
        """Projects should be mapped to their build stages."""
        with patch.object(cas, "get_subprojects_to_test") as mock_expand:
            with patch.object(cas, "plan_stage_reuse") as mock_plan:
                mock_expand.return_value = {"hip", "clr"}
                mock_plan.return_value = StageReusePlan(
                    candidate_stages=("math-libs",),
                    rebuild_stages=("compiler-runtime", "core"),
                    full_rebuild_required=False,
                    reasons=(),
                )

                affected, prebuilt, projects = cas.compute_affected("hip,clr")

                # Stages should be comma-separated and sorted
                self.assertEqual(affected, "compiler-runtime,core")
                self.assertEqual(prebuilt, "math-libs")

    def test_no_stages_found_returns_all(self):
        """If no stages found for projects, should fall back to all."""
        with patch.object(cas, "get_subprojects_to_test") as mock_expand:
            with patch.object(cas, "plan_stage_reuse") as mock_plan:
                mock_expand.return_value = {"unknown_project"}
                mock_plan.return_value = StageReusePlan(
                    candidate_stages=(),
                    rebuild_stages=(),
                    full_rebuild_required=True,
                    reasons=("no stages found",),
                )

                affected, prebuilt, projects = cas.compute_affected("unknown_project")

                self.assertEqual(affected, "all")
                self.assertEqual(prebuilt, "")
                self.assertEqual(projects, "")


if __name__ == "__main__":
    unittest.main()
