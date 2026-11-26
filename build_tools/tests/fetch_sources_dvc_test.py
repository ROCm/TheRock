#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for DVC source selection and pulls in fetch_sources.py."""

import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch


# Result returned by mocked DVC pulls.
@dataclass
class MockPullResult:
    fetched: int = 0
    cached: int = 0
    skipped: int = 0


# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

import fetch_sources


class TestPullLargeFiles(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.therock_dir = Path(self.temp_dir.name)
        self.project_dir = self.therock_dir / "base" / "rocm-systems"
        dvc_dir = self.project_dir / ".dvc"
        dvc_dir.mkdir(parents=True)
        (dvc_dir / "config").write_text("[core]\nremote = origin\n")
        self.enterContext(patch("fetch_sources.THEROCK_DIR", self.therock_dir))
        self.get_submodule_path = self.enterContext(
            patch("fetch_sources.get_submodule_path", return_value="base/rocm-systems")
        )
        self.pull = self.enterContext(
            patch(
                "fetch_sources.fetch_dvc_artifacts.pull", return_value=MockPullResult()
            )
        )

    def test_submodule_path_and_jobs(self):
        fetch_sources.pull_large_files(["rocm-systems"], jobs=4)

        self.get_submodule_path.assert_called_once_with("rocm-systems")
        self.pull.assert_called_once_with(self.project_dir, jobs=4)

    def test_none_jobs_uses_default(self):
        fetch_sources.pull_large_files(["rocm-systems"])

        self.pull.assert_called_once_with(
            self.project_dir, jobs=fetch_sources.fetch_dvc_artifacts.DEFAULT_JOBS
        )

    def test_missing_dvc_config_warning(self):
        (self.project_dir / ".dvc" / "config").unlink()

        with patch("fetch_sources.log") as log:
            fetch_sources.pull_large_files(["rocm-systems"])

        log.assert_called_once_with(
            f"WARNING: dvc config not found in {self.project_dir}, when expected."
        )
        self.pull.assert_not_called()

    def test_empty_dvc_projects(self):
        fetch_sources.pull_large_files([])

        self.pull.assert_not_called()
        self.get_submodule_path.assert_not_called()


class GetDvcProjectsTest(unittest.TestCase):
    def test_defaults_exclude_unselected_sources(self):
        # Selecting only systems must exclude libraries DVC, even on Windows.
        for is_windows in (False, True):
            with self.subTest(is_windows=is_windows):
                self.assertNotIn(
                    "rocm-libraries",
                    fetch_sources.get_dvc_projects(
                        ["rocm-systems"], ["auto"], is_windows=is_windows
                    ),
                )

    def test_selected_libraries_use_dvc_on_both_platforms(self):
        for is_windows in (False, True):
            with self.subTest(is_windows=is_windows):
                self.assertEqual(
                    fetch_sources.get_dvc_projects(
                        ["rocgdb", "rocm-libraries"], ["auto"], is_windows=is_windows
                    ),
                    ["rocm-libraries"],
                )

    def test_no_selected_dvc_sources(self):
        # rocGDB has no DVC files on either platform.
        for is_windows in (False, True):
            with self.subTest(is_windows=is_windows):
                self.assertEqual(
                    fetch_sources.get_dvc_projects(
                        ["rocgdb"], ["auto"], is_windows=is_windows
                    ),
                    [],
                )

    def test_requested_projects_override_platform_defaults(self):
        for is_windows in (False, True):
            with self.subTest(is_windows=is_windows):
                self.assertEqual(
                    fetch_sources.get_dvc_projects(
                        ["rocm-libraries", "rocm-systems"],
                        ["rocm-systems"],
                        is_windows=is_windows,
                    ),
                    ["rocm-systems"],
                )

    def test_requested_projects_exclude_unselected_sources(self):
        for is_windows in (False, True):
            with self.subTest(is_windows=is_windows):
                self.assertEqual(
                    fetch_sources.get_dvc_projects(
                        ["rocm-systems"], ["rocm-libraries"], is_windows=is_windows
                    ),
                    [],
                )

    def test_empty_request_disables_default_pulls(self):
        for is_windows in (False, True):
            with self.subTest(is_windows=is_windows):
                self.assertEqual(
                    fetch_sources.get_dvc_projects(
                        ["rocm-libraries"], [], is_windows=is_windows
                    ),
                    [],
                )
                self.assertEqual(
                    fetch_sources.get_dvc_projects(
                        ["rocm-libraries"], [""], is_windows=is_windows
                    ),
                    [],
                )

    def test_empty_cli_value_disables_pulls(self):
        with patch("fetch_sources.run") as run:
            fetch_sources.main(["--dvc-projects="])

        self.assertEqual(run.call_args.args[0].dvc_projects, [""])

    def test_freeform_cli_value_is_rejected(self):
        with patch("fetch_sources.run") as run:
            with self.assertRaises(SystemExit) as error:
                fetch_sources.main(["--dvc-projects", "custom-dvc-source"])

        self.assertEqual(error.exception.code, 2)
        run.assert_not_called()

    def test_auto_cannot_be_combined_with_projects(self):
        for project in ("rocm-libraries", "rocm-systems"):
            with self.subTest(project=project):
                with patch("fetch_sources.run") as run:
                    with self.assertRaises(SystemExit) as error:
                        fetch_sources.main(["--dvc-projects", "auto", project])

                self.assertEqual(error.exception.code, 2)
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
