# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for exe_stub_gen module."""

import os
import platform
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.exe_stub_gen import (
    generate_exe_link_stub,
    POSIX_EXE_STUB_TEMPLATE,
)


class PosixExeStubTemplateTest(unittest.TestCase):
    """Tests for the POSIX_EXE_STUB_TEMPLATE constant."""

    def test_template_contains_placeholder(self):
        """Template should contain the @EXEC_RELPATH@ placeholder."""
        self.assertIn("@EXEC_RELPATH@", POSIX_EXE_STUB_TEMPLATE)

    def test_template_is_valid_c_code_structure(self):
        """Template should have basic C code structure."""
        self.assertIn("#include", POSIX_EXE_STUB_TEMPLATE)
        self.assertIn("int main(", POSIX_EXE_STUB_TEMPLATE)
        self.assertIn("execv(", POSIX_EXE_STUB_TEMPLATE)
        self.assertIn("dladdr(", POSIX_EXE_STUB_TEMPLATE)

    def test_template_placeholder_replacement(self):
        """Placeholder replacement should produce valid C string literal."""
        result = POSIX_EXE_STUB_TEMPLATE.replace("@EXEC_RELPATH@", "../bin/ls")
        self.assertIn('../bin/ls"', result)
        self.assertNotIn("@EXEC_RELPATH@", result)


@unittest.skipIf(platform.system() == "Windows", "POSIX-only test")
class GenerateExeLinkStubPosixTest(unittest.TestCase):
    """Tests for generate_exe_link_stub on POSIX systems."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _has_cc(self) -> bool:
        """Check if a C compiler is available."""
        try:
            subprocess.run(
                ["cc", "--version"],
                capture_output=True,
                timeout=5,
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @unittest.skipUnless(
        # Check at class load time whether cc is available
        os.system("cc --version > /dev/null 2>&1") == 0,
        "C compiler not available",
    )
    def test_generates_executable(self):
        """Test that generate_exe_link_stub creates an executable file."""
        output_file = Path(self.temp_dir) / "stub"
        generate_exe_link_stub(output_file, "../bin/ls")
        self.assertTrue(output_file.exists())
        self.assertTrue(os.access(output_file, os.X_OK))

    @unittest.skipUnless(
        os.system("cc --version > /dev/null 2>&1") == 0,
        "C compiler not available",
    )
    def test_stub_executes_target(self):
        """Test that the generated stub correctly executes a target."""
        # Create a simple script to be the target
        target_dir = Path(self.temp_dir) / "bin"
        target_dir.mkdir()
        target = target_dir / "echo_test"
        target.write_text("#!/bin/sh\necho STUB_WORKS\n")
        target.chmod(0o755)

        # Generate stub in parent directory pointing to bin/echo_test
        stub = Path(self.temp_dir) / "my_stub"
        generate_exe_link_stub(stub, "bin/echo_test")

        result = subprocess.run(
            [str(stub)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.stdout.strip(), "STUB_WORKS")

    def test_uses_cc_env_variable(self):
        """Test that CC environment variable is respected."""
        with patch.dict(os.environ, {"CC": "gcc"}):
            with patch("subprocess.check_call") as mock_check_call:
                output_file = Path(self.temp_dir) / "stub"
                try:
                    generate_exe_link_stub(output_file, "../bin/tool")
                except Exception:
                    pass  # May fail if gcc isn't installed
                if mock_check_call.called:
                    call_args = mock_check_call.call_args[0][0]
                    self.assertEqual(call_args[0], "gcc")

    def test_default_compiler_is_cc(self):
        """Test that default compiler is 'cc' when CC is not set."""
        env = os.environ.copy()
        env.pop("CC", None)
        with patch.dict(os.environ, env, clear=True):
            with patch("subprocess.check_call") as mock_check_call:
                output_file = Path(self.temp_dir) / "stub"
                try:
                    generate_exe_link_stub(output_file, "../bin/tool")
                except Exception:
                    pass
                if mock_check_call.called:
                    call_args = mock_check_call.call_args[0][0]
                    self.assertEqual(call_args[0], "cc")


@unittest.skipUnless(platform.system() == "Windows", "Windows-only test")
class GenerateExeLinkStubWindowsTest(unittest.TestCase):
    """Tests for generate_exe_link_stub on Windows."""

    def test_raises_not_implemented_on_windows(self):
        """Test that Windows raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            generate_exe_link_stub(Path("output"), "../bin/tool")


if __name__ == "__main__":
    unittest.main()
