from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import therock_configure_ci


class ConfigureCITest(unittest.TestCase):
    @patch("subprocess.run")
    def test_pull_request(self, mock_run):
        args = {
            "is_pull_request": True,
        }

        mock_process = MagicMock()
        mock_process.stdout = "projects/rocprim/src/main.cpp\nprojects/hipcub/src/main.cpp\nprojects/rocwmma/src/main.cpp"
        mock_run.return_value = mock_process

        project_to_run, test_type = therock_configure_ci.retrieve_projects(args)
        self.assertIn("rocprim", str(project_to_run))
        self.assertIn("hipcub", str(project_to_run))
        self.assertIn("rocwmma", str(project_to_run))
        self.assertEqual(test_type, "full")

    @patch("subprocess.run")
    def test_pull_request_empty(self, mock_run):
        args = {"is_pull_request": True, "input_subtrees": ""}

        mock_process = MagicMock()
        mock_process.stdout = ""
        mock_run.return_value = mock_process

        project_to_run, test_type = therock_configure_ci.retrieve_projects(args)
        self.assertEqual(len(project_to_run), 0)

    @patch("subprocess.run")
    def test_workflow_dispatch(self, mock_run):
        args = {
            "is_workflow_dispatch": True,
            "input_projects": "projects/rocprim projects/hipcub",
        }

        mock_process = MagicMock()
        mock_process.stdout = ""
        mock_run.return_value = mock_process

        project_to_run, test_type = therock_configure_ci.retrieve_projects(args)
        self.assertIn("rocprim", str(project_to_run))
        self.assertIn("hipcub", str(project_to_run))
        self.assertEqual(test_type, "full")

    @patch("subprocess.run")
    def test_workflow_dispatch_bad_input(self, mock_run):
        args = {
            "is_workflow_dispatch": True,
            "input_projects": "projects/rocprim$$projects/hipcub",
        }

        mock_process = MagicMock()
        mock_process.stdout = ""
        mock_run.return_value = mock_process

        project_to_run, test_type = therock_configure_ci.retrieve_projects(args)
        self.assertEqual(len(project_to_run), 0)

    @patch("subprocess.run")
    def test_workflow_dispatch_all(self, mock_run):
        args = {"is_workflow_dispatch": True, "input_projects": "all"}

        mock_process = MagicMock()
        mock_process.stdout = ""
        mock_run.return_value = mock_process

        project_to_run, test_type = therock_configure_ci.retrieve_projects(args)
        self.assertGreaterEqual(len(project_to_run), 5)
        self.assertEqual(test_type, "full")

    @patch("subprocess.run")
    def test_workflow_dispatch_empty(self, mock_run):
        args = {"is_workflow_dispatch": True, "input_projects": ""}

        mock_process = MagicMock()
        mock_process.stdout = ""
        mock_run.return_value = mock_process

        project_to_run, test_type = therock_configure_ci.retrieve_projects(args)
        self.assertEqual(len(project_to_run), 0)

    @patch("subprocess.run")
    def test_is_push(self, mock_run):
        args = {
            "is_push": True,
        }

        mock_process = MagicMock()
        mock_process.stdout = "projects/rocprim/src/main.cpp"
        mock_run.return_value = mock_process

        project_to_run, test_type = therock_configure_ci.retrieve_projects(args)
        self.assertIn("rocprim", str(project_to_run))
        self.assertEqual(test_type, "full")

    def test_is_path_workflow_file_related_to_ci(self):
        workflow_path = ".github/workflows/therocktest.yml"
        self.assertTrue(
            therock_configure_ci.is_path_workflow_file_related_to_ci(workflow_path)
        )
        script_path = ".github/scripts/therocktest.py"
        self.assertTrue(
            therock_configure_ci.is_path_workflow_file_related_to_ci(script_path)
        )
        bad_path = ".github/workflows/test.yml"
        self.assertFalse(
            therock_configure_ci.is_path_workflow_file_related_to_ci(bad_path)
        )

    def test_is_path_skippable(self):
        # Skippable paths
        self.assertTrue(therock_configure_ci.is_path_skippable("README.md"))
        self.assertTrue(therock_configure_ci.is_path_skippable("docs/guide.rst"))
        self.assertTrue(
            therock_configure_ci.is_path_skippable("projects/rocprim/.gitignore")
        )
        self.assertTrue(
            therock_configure_ci.is_path_skippable("projects/hipcub/CHANGELOG.md")
        )
        self.assertTrue(
            therock_configure_ci.is_path_skippable(
                "projects/rocwmma/docs/sphinx/requirements.in"
            )
        )
        self.assertTrue(
            therock_configure_ci.is_path_skippable(
                "shared/tensile/docs/sphinx/requirements.in"
            )
        )
        # Non-skippable paths
        self.assertFalse(
            therock_configure_ci.is_path_skippable("projects/rocprim/src/main.cpp")
        )
        self.assertFalse(therock_configure_ci.is_path_skippable("CMakeLists.txt"))

    def test_check_for_non_skippable_path(self):
        # All skippable
        self.assertFalse(
            therock_configure_ci.check_for_non_skippable_path(
                ["README.md", "docs/guide.rst", ".gitignore"]
            )
        )
        # Contains non-skippable
        self.assertTrue(
            therock_configure_ci.check_for_non_skippable_path(
                ["README.md", "projects/rocprim/src/main.cpp"]
            )
        )
        # None and empty
        self.assertFalse(therock_configure_ci.check_for_non_skippable_path(None))
        self.assertFalse(therock_configure_ci.check_for_non_skippable_path([]))

    @patch("therock_configure_ci.get_modified_paths")
    def test_retrieve_projects_skips_ci_for_skippable_paths(self, mock_get_modified):
        mock_get_modified.return_value = [
            "README.md",
            "docs/guide.rst",
            "projects/rocprim/.gitignore",
        ]

        projects, test_type = therock_configure_ci.retrieve_projects(
            {"is_pull_request": True, "base_ref": "HEAD^"}
        )

        self.assertEqual(projects, [])
        self.assertEqual(test_type, "full")

    @patch("therock_configure_ci.get_modified_paths")
    def test_retrieve_projects_runs_ci_for_non_skippable_paths(self, mock_get_modified):
        mock_get_modified.return_value = ["README.md", "projects/rocprim/src/main.cpp"]

        projects, test_type = therock_configure_ci.retrieve_projects(
            {"is_pull_request": True, "base_ref": "HEAD^"}
        )

        self.assertIn("rocprim", str(projects))
        self.assertEqual(test_type, "full")

    @patch("therock_configure_ci.get_modified_paths")
    def test_retrieve_projects_runs_ci_for_two_projects(self, mock_get_modified):
        mock_get_modified.return_value = [
            "README.md",
            "projects/rocprim/src/main.cpp",
            "projects/hipcub/src/main.cpp",
        ]

        projects, test_type = therock_configure_ci.retrieve_projects(
            {"is_pull_request": True, "base_ref": "HEAD^"}
        )

        self.assertIn("rocprim", str(projects))
        self.assertIn("hipcub", str(projects))
        self.assertEqual(test_type, "full")

    @patch("therock_configure_ci.get_modified_paths")
    def test_retrieve_projects_runs_ci_for_workflow_paths(self, mock_get_modified):
        mock_get_modified.return_value = [".github/workflows/therock-ci.yml"]

        projects, test_type = therock_configure_ci.retrieve_projects(
            {"is_pull_request": True, "base_ref": "HEAD^"}
        )

        # All projects should be tested with smoke tests.. make sure we get at least 4 projects
        self.assertGreaterEqual(len(projects), 5)
        self.assertEqual(test_type, "smoke")


if __name__ == "__main__":
    unittest.main()
