# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise the artifact runner with small, real CMake/CTest projects."""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "test_executable_scripts"))
import test_rocwmma_install as runner

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _therock_utils.artifact_builder import ArtifactDescriptor, ComponentScanner


@unittest.skipUnless(
    shutil.which("cmake") and shutil.which("ninja"), "Needs CMake and Ninja"
)
class RocwmmaInstallTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sdk = self.root / "SDK with spaces"
        self.build = self.root / "build"
        self.source = self.sdk / "share/rocwmma/tests/package_consumer"
        for relative in [
            "lib/llvm/bin/clang++",
            "lib/llvm/bin/clang++.exe",
            "lib/cmake/rocwmma/rocwmma-config.cmake",
        ]:
            path = self.sdk / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        for source in [self.source, self.source / "openmp"]:
            source.mkdir(parents=True, exist_ok=True)
            (source / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.21)\n"
                "project(runner_fixture NONE)\n"
                "enable_testing()\n"
                'add_test(NAME smoke COMMAND "${CMAKE_COMMAND}" -E true)\n'
            )

    @patch.object(runner.platform, "system", return_value="Linux")
    def test_linux_runs_both_consumers_and_rejects_cached_builds(self, _system):
        runner.run_tests(self.sdk, self.build, "gfx1100,gfx1101")
        for case in ["no_openmp", "openmp"]:
            self.assertIn(
                "100% tests passed", (self.build / case / "test.log").read_text()
            )
        with self.assertRaises(FileExistsError):
            runner.run_tests(self.sdk, self.build, "gfx1100")

    @patch.object(runner.platform, "system", return_value="Linux")
    def test_failures_propagate(self, _system):
        for phase, cmake in [
            ("configure", 'message(FATAL_ERROR "broken export")\n'),
            (
                "build",
                'add_custom_target(fail ALL COMMAND "${CMAKE_COMMAND}" -E false)\n',
            ),
            ("test", 'add_test(NAME fail COMMAND "${CMAKE_COMMAND}" -E false)\n'),
            ("empty", ""),
        ]:
            with self.subTest(phase=phase):
                (self.source / "openmp/CMakeLists.txt").write_text(
                    "cmake_minimum_required(VERSION 3.21)\n"
                    "project(failure_fixture NONE)\nenable_testing()\n" + cmake
                )
                build = self.root / phase
                with self.assertRaises(subprocess.CalledProcessError):
                    runner.run_tests(self.sdk, build, "gfx1100")
                self.assertTrue(
                    (
                        build
                        / "openmp"
                        / f"{'test' if phase == 'empty' else phase}.log"
                    ).is_file()
                )

    @patch.object(runner.platform, "system", return_value="Windows")
    def test_windows_uses_sdk_compiler_and_no_openmp_case(self, _system):
        with patch.object(
            runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)
        ) as run:
            runner.run_tests(self.sdk, self.build, "gfx1100;gfx1101")
        self.assertEqual(run.call_count, 3)
        configure = run.call_args_list[0]
        command = configure.args[0]
        self.assertIn(
            f"-DCMAKE_HIP_COMPILER={self.sdk / 'lib/llvm/bin/clang++.exe'}", command
        )
        self.assertIn("-DCMAKE_HIP_ARCHITECTURES=gfx1100", command)
        self.assertIn("-DCMAKE_RC_COMPILER=rc.exe", command)
        self.assertTrue(
            configure.kwargs["env"]["PATH"].startswith(str(self.sdk / "bin"))
        )
        self.assertFalse((self.build / "openmp").exists())

    def test_artifact_owns_consumer_sources(self):
        relative = Path("math-libs/rocWMMA/stage")
        stage = self.root / "artifact-input" / relative
        shutil.copytree(self.sdk, stage)
        descriptor = ArtifactDescriptor.load_toml_file(
            Path(__file__).resolve().parents[3] / "math-libs/artifact-rocwmma.toml",
            artifact_name="rocwmma",
        )
        scanner = ComponentScanner(self.root / "artifact-input", descriptor)
        for component in ["dev", "test"]:
            output = self.root / component
            output.mkdir()
            scanner.components[component].write_artifact(output)
            consumer = output / relative / "share/rocwmma/tests/package_consumer"
            for project in [consumer, consumer / "openmp"]:
                self.assertEqual(
                    (project / "CMakeLists.txt").is_file(), component == "test"
                )
        self.assertTrue(
            (
                self.root / "dev" / relative / "lib/cmake/rocwmma/rocwmma-config.cmake"
            ).is_file()
        )

    def test_missing_sources_and_target_fail_before_building(self):
        with self.assertRaises(ValueError):
            runner.run_tests(self.sdk, self.build, "")
        (self.source / "CMakeLists.txt").unlink()
        with self.assertRaises(FileNotFoundError):
            runner.run_tests(self.sdk, self.build, "gfx1100")
        self.assertFalse(self.build.exists())


if __name__ == "__main__":
    unittest.main()
