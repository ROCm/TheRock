# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for the packaged hipFile ASan test launcher."""

import os
import platform
import runpy
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = (
    Path(__file__).resolve().parent.parent / "test_executable_scripts/test_hipfile.py"
)


class HipFileTestRunnerTests(unittest.TestCase):
    def test_asan_runtime_is_preloaded_for_ctest(self):
        # LLVM_ENABLE_PER_TARGET_RUNTIME_DIR changes the runtime filename.
        for runtime_source in ("environment", "per-target", "legacy"):
            with self.subTest(
                runtime_source=runtime_source
            ), tempfile.TemporaryDirectory() as tmp:
                install = Path(tmp)
                (install / "share/hipfile/test").mkdir(parents=True)
                runtime_name = (
                    "libclang_rt.asan-x86_64.so"
                    if runtime_source == "legacy"
                    else "libclang_rt.asan.so"
                )
                runtime = install / runtime_name
                runtime.touch()
                env = {
                    "THEROCK_BIN_DIR": str(install / "bin"),
                    "LD_PRELOAD": "existing.so",
                }
                if runtime_source == "environment":
                    env["ASAN_RUNTIME_PATH"] = str(runtime)
                ctest_envs = []

                def run(cmd, **kwargs):
                    if cmd[0] == "ctest":
                        ctest_envs.append(kwargs["env"])
                        self.assertIn("--no-tests=error", cmd)
                        return subprocess.CompletedProcess(cmd, 0)
                    requested = cmd[1].split("=", 1)[1]
                    # Clang echoes the requested name when it is unavailable.
                    resolved = (
                        str(runtime)
                        if runtime_source != "environment" and requested == runtime_name
                        else requested
                    )
                    return subprocess.CompletedProcess(cmd, 0, stdout=resolved)

                matrix = types.ModuleType("amdgpu_family_matrix")
                matrix.is_asan = lambda: True
                with (
                    patch.dict(os.environ, env, clear=True),
                    patch.dict(sys.modules, {"amdgpu_family_matrix": matrix}),
                    patch.object(platform, "machine", return_value="x86_64"),
                    patch.object(subprocess, "run", side_effect=run),
                    patch.object(sys, "path", list(sys.path)),
                ):
                    runpy.run_path(str(SCRIPT), run_name="__main__")
                self.assertEqual(len(ctest_envs), 1)
                self.assertEqual(
                    ctest_envs[0]["LD_PRELOAD"], f"existing.so:{runtime.resolve()}"
                )


if __name__ == "__main__":
    unittest.main()
