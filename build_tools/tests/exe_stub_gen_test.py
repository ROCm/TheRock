# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for exe_stub_gen — platform-specific stub executable generation."""

import contextlib
import os
import platform
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from _therock_utils.exe_stub_gen import generate_exe_link_stub

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
_CC_AVAILABLE = shutil.which(shlex.split(os.getenv("CC", "cc"))[0]) is not None


class GenerateExeLinkStubTest(unittest.TestCase):
    def _capture_generated_source(
        self, relative_link_to: str, *, system: str | None = None
    ) -> str:
        """Invoke generate_exe_link_stub with a no-op compiler and return the
        C source that would have been compiled."""
        captured: dict[str, str] = {}

        def _fake_check_call(args, **kwargs):
            captured["source"] = Path(args[-1]).read_text()

        patches = [
            mock.patch(
                "_therock_utils.exe_stub_gen.subprocess.check_call",
                side_effect=_fake_check_call,
            )
        ]
        if system is not None:
            patches.append(
                mock.patch(
                    "_therock_utils.exe_stub_gen.platform.system",
                    return_value=system,
                )
            )

        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            with tempfile.TemporaryDirectory() as tmp:
                generate_exe_link_stub(Path(tmp) / "stub", relative_link_to)

        return captured["source"]

    def test_windows_raises_not_implemented(self):
        """NotImplementedError is raised on Windows."""
        with mock.patch(
            "_therock_utils.exe_stub_gen.platform.system", return_value="Windows"
        ):
            with self.assertRaises(NotImplementedError):
                generate_exe_link_stub(Path("/tmp/stub"), "target")

    @unittest.skipIf(IS_WINDOWS, "POSIX only")
    def test_invalid_relative_link_to_raises_value_error(self):
        """Characters invalid in a C string literal are rejected before compilation."""
        for bad in ('has"quote', "has\\backslash", "has\nnewline", "has\x00null", ""):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    generate_exe_link_stub(Path("/tmp/stub"), bad)

    @unittest.skipIf(IS_WINDOWS, "POSIX only")
    def test_exec_relpath_placeholder_substituted(self):
        """@EXEC_RELPATH@ is replaced with the supplied relative path in the
        generated C source."""
        source = self._capture_generated_source("my_target_binary")
        self.assertIn("my_target_binary", source)
        self.assertNotIn("@EXEC_RELPATH@", source)

    def test_linux_template_uses_proc_self_exe(self):
        """On Linux the generated source uses /proc/self/exe for path resolution."""
        source = self._capture_generated_source("target", system="Linux")
        self.assertIn("/proc/self/exe", source)
        self.assertNotIn("#include <dlfcn.h>", source)

    def test_posix_template_uses_dladdr(self):
        """On non-Linux POSIX (e.g. macOS) the generated source uses dladdr()."""
        source = self._capture_generated_source("target", system="Darwin")
        self.assertIn("#include <dlfcn.h>", source)
        self.assertNotIn("/proc/self/exe", source)

    @unittest.skipUnless(IS_LINUX, "Linux /proc/self/exe path only")
    @unittest.skipUnless(_CC_AVAILABLE, "C compiler (cc) required")
    def test_stub_works_with_bare_argv0(self):
        """Compiled stub resolves and execs its target when invoked with a bare
        argv[0] (no '/' separator) — the failure mode triggered by MLIR's ROCDL
        target which passes "ld.lld" as argv[0] to llvm::sys::ExecuteAndWait."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # Minimal target script that exits 0.
            target = tmp_path / "target"
            target.write_text("#!/bin/sh\nexit 0\n")
            target.chmod(0o755)

            stub = tmp_path / "stub"
            generate_exe_link_stub(stub, "target")

            # argv[0] = "stub" (bare name, no '/'), while the OS executes the
            # stub via its absolute path. This mirrors how MLIR's ROCDL target
            # invokes ld.lld: absolute executable path, bare string as argv[0].
            result = subprocess.run(
                ["stub"],
                executable=str(stub),
                capture_output=True,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"Stub failed when invoked with bare argv[0]: "
                f"{result.stderr.decode()!r}",
            )


if __name__ == "__main__":
    unittest.main()
