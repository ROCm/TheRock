# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise relocated installed launchers using real CTest child processes."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "test_executable_scripts")
)
import relocate_csan_ctest


@unittest.skipUnless(
    shutil.which("cmake") and shutil.which("ctest"), "CMake/CTest required"
)
class RelocateCsanCtestTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="csan ctest ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.metadata = self.root / "CTestTestfile.cmake"
        self.child = self.root / "child.py"
        self.result = self.root / "result.json"
        self.child.write_text(
            "import json, os, pathlib, sys\n"
            "pathlib.Path(os.environ['RESULT']).write_text(json.dumps({\n"
            " 'preload': os.environ.get('LD_PRELOAD'),\n"
            " 'retained': os.environ.get('RETAINED'), 'args': sys.argv[1:]}))\n"
            "sys.exit(int(os.environ.get('CHILD_EXIT', '0')))\n"
        )
        self.env = os.environ.copy()
        self.env.update(
            CSAN_REPORT_PREFIX=str(self.root / "report"),
            CSAN_RUNTIME_PATH="/missing/runtime.so",
            CSAN_HSA_RUNTIME_PATH="/missing/hsa.so",
            CSAN_REPORT_PRELOAD_PATH="/missing/report.so",
            LD_PRELOAD="/missing/runtime.so:/missing/hsa.so:/missing/report.so:/missing/caller.so",
            RESULT=str(self.result),
        )
        # A definitely stale path also exercises this on build images that do
        # happen to install /usr/local/therock-tools/bin/cmake themselves.
        self.stale_cmake = str(self.root / "old-image" / "cmake")
        self.stale = patch.object(relocate_csan_ctest, "_BUILD_CMAKE", self.stale_cmake)
        self.stale.start()
        self.addCleanup(self.stale.stop)

    def write_launcher(self, exit_code=0):
        q = relocate_csan_ctest._quoted
        command = [
            self.stale_cmake,
            "-E",
            "env",
            "RETAINED=keep this",
            "LD_PRELOAD=/old/build/csan.so",
            f"CHILD_EXIT={exit_code}",
            "--",
            sys.executable,
            str(self.child),
            self.stale_cmake,
            "two words",
        ]
        self.metadata.write_text(
            'add_test(setup "'
            + shutil.which("cmake").replace("\\", "/")
            + '" "-E" "true")\n'
            "set_tests_properties(setup PROPERTIES FIXTURES_SETUP prepared)\n"
            "add_test([=[child]=] " + " ".join(q(value) for value in command) + ")\n"
            "set_tests_properties(child PROPERTIES FIXTURES_REQUIRED prepared LABELS comprehensive)\n"
            "# Preserve an unrelated absolute argument: " + self.stale_cmake + "\n"
        )

    def run_ctest(self):
        # Do not preload fake paths into CTest itself; the relocated CMake-env
        # child must set the exact prepared chain independently.
        env = self.env.copy()
        env.pop("LD_PRELOAD")
        return subprocess.run(
            [
                shutil.which("ctest"),
                "--test-dir",
                str(self.root),
                "-L",
                "comprehensive",
                "--output-on-failure",
            ],
            env=env,
            capture_output=True,
            text=True,
        )

    def test_relocated_child_preserves_preload_arguments_and_fixtures(self):
        self.write_launcher()
        original = self.metadata.read_text()
        self.assertEqual(
            relocate_csan_ctest.relocate_csan_compute_launchers(self.root, self.env), 1
        )
        result = self.run_ctest()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("2/2", result.stdout)  # CTest still adds the required fixture.
        record = json.loads(self.result.read_text())
        self.assertEqual(record["preload"], self.env["LD_PRELOAD"])
        self.assertEqual(record["retained"], "keep this")
        self.assertEqual(record["args"], [self.stale_cmake, "two words"])
        self.assertIn(original.splitlines()[3], self.metadata.read_text())
        self.assertIn(original.splitlines()[4], self.metadata.read_text())
        self.assertEqual(
            relocate_csan_ctest.relocate_csan_compute_launchers(self.root, self.env), 0
        )

    def test_child_failure_remains_failure(self):
        self.write_launcher(exit_code=23)
        relocate_csan_ctest.relocate_csan_compute_launchers(self.root, self.env)
        result = self.run_ctest()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Failed", result.stdout)
        self.assertTrue(self.result.is_file())

    def test_ordinary_environment_is_byte_identical(self):
        self.write_launcher()
        original = self.metadata.read_bytes()
        env = self.env.copy()
        env.pop("CSAN_REPORT_PREFIX")
        self.assertEqual(
            relocate_csan_ctest.relocate_csan_compute_launchers(self.root, env), 0
        )
        self.assertEqual(self.metadata.read_bytes(), original)

    def test_incomplete_preload_fails_without_modifying_metadata(self):
        self.write_launcher()
        original = self.metadata.read_bytes()
        self.env["LD_PRELOAD"] = self.env["CSAN_RUNTIME_PATH"]
        with self.assertRaisesRegex(ValueError, "prepared runtime:HSA:report"):
            relocate_csan_ctest.relocate_csan_compute_launchers(self.root, self.env)
        self.assertEqual(self.metadata.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
