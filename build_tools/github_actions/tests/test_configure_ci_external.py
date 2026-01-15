#!/usr/bin/env python3

"""Unit tests for external-repo behavior in configure_ci.py.

These tests should exercise real logic in `configure_ci.py`, not just assert
string properties.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Add parent directory to path to import configure_ci as a module.
sys.path.insert(0, str(Path(__file__).parent.parent))

import configure_ci


class TestDetectExternalRepo(unittest.TestCase):
    def test_detect_external_repo_from_override(self):
        is_external, repo_name = configure_ci._detect_external_repo(
            Path("C:/workspace/TheRock"),
            repo_override="ROCm/rocm-libraries",
        )
        self.assertTrue(is_external)
        self.assertEqual(repo_name, "rocm-libraries")

    def test_detect_external_repo_from_cwd(self):
        is_external, repo_name = configure_ci._detect_external_repo(
            Path("C:/workspace/source-repo/rocm-systems/subdir"),
            repo_override="",
        )
        self.assertTrue(is_external)
        self.assertEqual(repo_name, "rocm-systems")

    def test_detect_external_repo_not_external(self):
        is_external, repo_name = configure_ci._detect_external_repo(
            Path("C:/workspace/TheRock"),
            repo_override="",
        )
        self.assertFalse(is_external)
        self.assertIsNone(repo_name)


class TestDetectExternalProjectsOrExit(unittest.TestCase):
    def test_no_projects_exits_and_sets_empty_outputs(self):
        base_args = {"base_ref": "HEAD^1", "github_event_name": "pull_request"}

        with (
            patch(
                "external_repo_project_maps.detect_projects_from_changes",
                return_value={"linux_projects": [], "windows_projects": []},
            ),
            patch.object(configure_ci, "gha_set_output") as mock_set_output,
            self.assertRaises(SystemExit) as cm,
        ):
            configure_ci._detect_external_projects_or_exit(base_args, "rocm-libraries")

        self.assertEqual(cm.exception.code, 0)
        mock_set_output.assert_called_once()
        out = mock_set_output.call_args.args[0]
        self.assertEqual(out["linux_variants"], "[]")
        self.assertEqual(out["windows_variants"], "[]")
        self.assertEqual(out["enable_build_jobs"], "false")

    def test_projects_populate_base_args(self):
        base_args = {"base_ref": "HEAD^1", "github_event_name": "pull_request"}

        fake_detection = {
            "linux_projects": [
                {"project_to_test": "rocprim", "cmake_options": "-D..."}
            ],
            "windows_projects": [{"project_to_test": "clr", "cmake_options": "-D..."}],
        }

        with (
            patch(
                "external_repo_project_maps.detect_projects_from_changes",
                return_value=fake_detection,
            ) as mock_detect,
            patch.dict(os.environ, {"PROJECTS": "projects/rocprim"}),
        ):
            configure_ci._detect_external_projects_or_exit(base_args, "rocm-libraries")

        mock_detect.assert_called_once()
        self.assertIn("linux_external_project_configs", base_args)
        self.assertIn("windows_external_project_configs", base_args)
        self.assertEqual(
            base_args["linux_external_project_configs"],
            fake_detection["linux_projects"],
        )
        self.assertEqual(
            base_args["windows_external_project_configs"],
            fake_detection["windows_projects"],
        )


if __name__ == "__main__":
    unittest.main()
