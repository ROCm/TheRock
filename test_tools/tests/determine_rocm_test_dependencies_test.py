# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

THEROCK_DIR = Path(__file__).parent.parent.parent
SCRIPT = Path(__file__).parent.parent / "determine_rocm_test_dependencies.py"

sys.path.insert(0, str(THEROCK_DIR / "test_tools"))

from determine_rocm_test_dependencies import (
    parse_cmake_test_subprojects,
    get_subprojects_to_test,
    list_subprojects,
)


class TestDetermineRocmTestDependencies(unittest.TestCase):
    def test_parse_cmake_test_subprojects(self):
        """Parse CMakeLists.txt and verify key dependencies."""
        test_deps = parse_cmake_test_subprojects(THEROCK_DIR)

        # Verify rocBLAS test dependencies
        self.assertIn("rocblas", test_deps)
        self.assertEqual(set(test_deps["rocblas"]), {"hipblas", "rocsolver"})

        # Verify rocSPARSE test dependencies
        self.assertIn("rocsparse", test_deps)
        self.assertEqual(
            set(test_deps["rocsparse"]), {"hipsparse", "rocsolver", "hipsolver"}
        )

        # Verify rocPRIM test dependencies
        self.assertIn("rocprim", test_deps)
        self.assertEqual(
            set(test_deps["rocprim"]), {"hipcub", "rocthrust", "rocsparse"}
        )

        # Verify rocFFT test dependencies
        self.assertIn("rocfft", test_deps)
        self.assertEqual(set(test_deps["rocfft"]), {"hipfft"})

        # Verify rocWMMA has empty TEST_SUBPROJECTS (tests only itself)
        self.assertIn("rocwmma", test_deps)
        self.assertEqual(test_deps["rocwmma"], [])

    def test_parse_hyphenated_subproject_names(self):
        """Subproject names and TEST_SUBPROJECTS deps may contain hyphens."""
        cmake = """\
therock_cmake_subproject_declare(amd-dbgapi
  RUNTIME_DEPS
    amd-comgr
  TEST_SUBPROJECTS
    rocgdb
    rocr-debug-agent-tests
)

therock_cmake_subproject_declare(rocr-debug-agent
  RUNTIME_DEPS
    amd-dbgapi
  TEST_SUBPROJECTS
    rocr-debug-agent-tests
)

therock_cmake_subproject_declare(rocr-debug-agent-tests
  RUNTIME_DEPS
    rocr-debug-agent
  TEST_SUBPROJECTS
)
"""
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "CMakeLists.txt").write_text(cmake)
            test_deps = parse_cmake_test_subprojects(tmp)

        self.assertEqual(
            set(test_deps["amd-dbgapi"]), {"rocgdb", "rocr-debug-agent-tests"}
        )
        self.assertEqual(test_deps["rocr-debug-agent"], ["rocr-debug-agent-tests"])
        # Empty TEST_SUBPROJECTS placed last: tests only itself.
        self.assertEqual(test_deps["rocr-debug-agent-tests"], [])

    def test_get_subprojects_to_test(self):
        """Test dependency resolution with case-insensitive input."""
        result = get_subprojects_to_test(["rocBLAS"], THEROCK_DIR)
        self.assertEqual(result, {"rocblas", "hipblas", "rocsolver"})

        # Path format normalization (projects/rocblas -> rocblas)
        result = get_subprojects_to_test([Path("projects/rocblas").name], THEROCK_DIR)
        self.assertEqual(result, {"rocblas", "hipblas", "rocsolver"})

    def test_empty_changed_projects_outputs_wildcard(self):
        """Empty --changed-projects outputs '*' for all tests."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT)], capture_output=True, text=True
        )
        self.assertEqual(result.stdout.strip(), "*")

    def test_empty_changed_projects_flag_outputs_wildcard(self):
        """Empty --changed-projects flag outputs '*' for all tests."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--changed-projects"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), "*")

    def test_comma_separated_input(self):
        """Comma-separated projects are parsed correctly."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--changed-projects", "rocblas,hipblas"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        # Output is JSON list
        import json

        projects = json.loads(result.stdout.strip())
        # rocblas -> hipblas, rocsolver; hipblas -> hipblas
        self.assertIn("rocblas", projects)
        self.assertIn("hipblas", projects)
        self.assertIn("rocsolver", projects)

    def test_gha_output_format_comma_separated(self):
        """GHA output produces comma-separated projects_to_test."""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            output_file = f.name

        try:
            env = os.environ.copy()
            env["GITHUB_OUTPUT"] = output_file
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--changed-projects",
                    "rocblas",
                    "--gha-output",
                ],
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(result.returncode, 0)

            with open(output_file) as f:
                content = f.read()
            # Format: projects_to_test=hipblas,rocblas,rocsolver
            self.assertIn("projects_to_test=", content)
            # Verify comma-separated format (not space-separated)
            self.assertIn(",", content)
            self.assertIn("rocblas", content)
        finally:
            os.unlink(output_file)

    def test_projects_prefix_normalization(self):
        """Projects with 'projects/' prefix are normalized."""
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--changed-projects",
                "projects/rocblas",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        import json

        projects = json.loads(result.stdout.strip())
        self.assertIn("rocblas", projects)
        self.assertIn("hipblas", projects)
        self.assertIn("rocsolver", projects)

    def test_list_subprojects(self):
        """list_subprojects returns names, and deps/'empty' when show_deps=True."""
        names = list_subprojects(THEROCK_DIR, show_deps=False)
        self.assertIn("rocblas", names)
        self.assertIn("rocwmma", names)

        deps = list_subprojects(THEROCK_DIR, show_deps=True)
        self.assertEqual(set(deps["rocblas"]), {"hipblas", "rocsolver"})
        self.assertEqual(deps["rocwmma"], "empty")

    def test_unknown_project_warning(self):
        """Unknown project names emit a warning to stderr."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--changed-projects", "rocblass"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("Warning: unrecognized project", result.stderr)
        self.assertIn("rocblass", result.stderr)


if __name__ == "__main__":
    unittest.main()
