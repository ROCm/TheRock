#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import platform
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from _therock_utils.exe_stub_gen import (
    POSIX_EXE_STUB_TEMPLATE,
    generate_exe_link_stub,
)


IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"


def _require_cc():
    cc = os.getenv("CC", "cc")
    if not shutil.which(cc):
        raise unittest.SkipTest(f"C compiler not found: {cc}")
    return cc


def _write_target(target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("#!/bin/sh\nprintf 'target ran:%s\\n' \"$1\"\n")
    target.chmod(0o755)


def _compile_stub_source(output_file: Path, relative_link_to: str, source: str):
    cc = _require_cc()
    source_file = output_file.with_suffix(".c")
    source_file.write_text(source.replace("@EXEC_RELPATH@", relative_link_to))
    subprocess.check_call(
        [cc, "-fPIE", "-o", str(output_file), str(source_file), "-ldl"]
    )


@unittest.skipIf(IS_WINDOWS, "exe stubs are not implemented on Windows")
class ExeStubGenTest(unittest.TestCase):
    def test_invokes_target_by_absolute_path(self):
        _require_cc()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            stub = tmp / "pkg" / "wrappers" / "tool"
            target = tmp / "pkg" / "bin" / "target"
            stub.parent.mkdir(parents=True)
            _write_target(target)

            generate_exe_link_stub(stub, "../bin/target")

            result = subprocess.run(
                [str(stub), "absolute"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "target ran:absolute\n")
            self.assertEqual(result.stderr, "")

    @unittest.skipUnless(IS_LINUX, "PATH lookup regression is Linux-specific")
    def test_invokes_system_target_by_path_lookup(self):
        _require_cc()
        echo = Path("/bin/echo")
        if not echo.exists():
            raise unittest.SkipTest("/bin/echo not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            stub = tmp / "tool"

            generate_exe_link_stub(stub, os.path.relpath(echo, stub.parent))

            env = os.environ.copy()
            env["PATH"] = str(stub.parent)
            result = subprocess.run(
                ["tool", "path"],
                env=env,
                cwd=tmp,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "path\n")
            self.assertEqual(result.stderr, "")

    def test_invocation_through_symlink_uses_real_stub_location(self):
        _require_cc()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            stub = tmp / "pkg" / "wrappers" / "tool"
            target = tmp / "pkg" / "bin" / "target"
            link = tmp / "path" / "tool"
            stub.parent.mkdir(parents=True)
            link.parent.mkdir()
            _write_target(target)

            generate_exe_link_stub(stub, "../bin/target")
            link.symlink_to(stub)

            result = subprocess.check_output([str(link), "symlink"], cwd=tmp, text=True)
            self.assertEqual(result, "target ran:symlink\n")

    def test_dladdr_fallback_is_available_without_proc_self_exe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            stub = tmp / "pkg" / "wrappers" / "tool"
            target = tmp / "pkg" / "bin" / "target"
            stub.parent.mkdir(parents=True)
            _write_target(target)

            source = POSIX_EXE_STUB_TEMPLATE.replace(
                "#include <unistd.h>\n",
                "#include <unistd.h>\n#undef __linux__\n#undef __CYGWIN__\n",
                1,
            )
            _compile_stub_source(stub, "../bin/target", source)

            result = subprocess.check_output(
                [str(stub), "fallback"], cwd=tmp, text=True
            )
            self.assertEqual(result, "target ran:fallback\n")


if __name__ == "__main__":
    unittest.main()
