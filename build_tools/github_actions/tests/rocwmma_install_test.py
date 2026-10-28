# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Test the runner contract with synthetic CMake/CTest consumers.

Fixture names and layouts are independent of the cases registered by rocWMMA.
"""

import json
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
        # Case names intentionally differ from source directories: only the
        # manifest establishes the mapping between them.
        for source in [
            self.source / "fixtures/hip_project",
            self.source / "fixtures/platform_project",
        ]:
            source.mkdir(parents=True, exist_ok=True)
            (source / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.21)\n"
                "project(runner_fixture NONE)\n"
                "enable_testing()\n"
                'add_test(NAME smoke COMMAND "${CMAKE_COMMAND}" -E true)\n'
            )

        self.manifest = {
            "version": 1,
            "tests": [
                {"name": "hip_consumer", "source": "fixtures/hip_project", "hip": True},
                {
                    "name": "linux_only_consumer",
                    "source": "fixtures/platform_project",
                    "platforms": ["linux"],
                },
            ],
        }
        self.write_manifest()

    def write_manifest(self):
        (self.source / "tests.json").write_text(json.dumps(self.manifest))

    @patch.object(runner.platform, "system", return_value="Linux")
    def test_linux_runs_registered_consumers_and_rejects_cached_builds(self, _system):
        runner.run_tests(self.sdk, self.build, "gfx1100,gfx1101")
        for case in ["hip_consumer", "linux_only_consumer"]:
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
                (self.source / "fixtures/platform_project/CMakeLists.txt").write_text(
                    "cmake_minimum_required(VERSION 3.21)\n"
                    "project(failure_fixture NONE)\nenable_testing()\n" + cmake
                )
                build = self.root / phase
                with self.assertRaises(subprocess.CalledProcessError):
                    runner.run_tests(self.sdk, build, "gfx1100")
                self.assertTrue(
                    (
                        build
                        / "linux_only_consumer"
                        / f"{'test' if phase == 'empty' else phase}.log"
                    ).is_file()
                )

    @patch.object(runner.platform, "system", return_value="Windows")
    def test_windows_uses_sdk_compiler_and_filters_platforms(self, _system):
        # HIP setup follows metadata, even when the case name changes.
        self.manifest["tests"][0]["name"] = "renamed_hip"
        self.write_manifest()
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
        self.assertEqual(
            command[command.index("-S") + 1], str(self.source / "fixtures/hip_project")
        )
        self.assertIn("-DCMAKE_HIP_ARCHITECTURES=gfx1100", command)
        self.assertIn("-DCMAKE_RC_COMPILER=rc.exe", command)
        self.assertTrue(
            configure.kwargs["env"]["PATH"].startswith(str(self.sdk / "bin"))
        )
        self.assertFalse((self.build / "linux_only_consumer").exists())

    @patch.object(runner.platform, "system", return_value="Linux")
    def test_new_cases_and_phase_selection(self, _system):
        # Registration alone adds cases to an unchanged runner. A shared project
        # also exercises different configurations in independent build trees.
        extra = self.source / "extra"
        extra.mkdir()
        (extra / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.21)\n"
            "project(extra NONE)\n"
            'if(NOT MESSAGE STREQUAL "value with spaces;and a list")\n'
            '  message(FATAL_ERROR "missing case-specific option")\n'
            "endif()\n"
            "if(FAIL_BUILD)\n"
            '  add_custom_target(fail ALL COMMAND "${CMAKE_COMMAND}" -E false)\n'
            "endif()\n"
            "enable_testing()\n"
            'add_test(NAME smoke COMMAND "${CMAKE_COMMAND}" -E true)\n'
            "if(FAIL_TEST)\n"
            '  add_test(NAME fail COMMAND "${CMAKE_COMMAND}" -E false)\n'
            "endif()\n"
        )
        for phase in ["configure", "build", "test"]:
            self.manifest["tests"].append(
                {
                    "name": "extra_" + phase,
                    "source": "extra",
                    "phase": phase,
                    "cmake_options": {
                        "MESSAGE": "value with spaces;and a list",
                        "FAIL_BUILD": "ON" if phase == "configure" else "OFF",
                        "FAIL_TEST": "OFF" if phase == "test" else "ON",
                    },
                }
            )
        self.write_manifest()
        runner.run_tests(self.sdk, self.build, "gfx1100")
        for phase in ["configure", "build", "test"]:
            case_build = self.build / ("extra_" + phase)
            self.assertTrue((case_build / "configure.log").is_file())
            self.assertEqual((case_build / "build.log").is_file(), phase != "configure")
            self.assertEqual((case_build / "test.log").is_file(), phase == "test")

    @patch.object(runner.platform, "system", return_value="Linux")
    def test_invalid_manifests_fail_before_building(self, _system):
        invalid_entries = [
            {"name": "hip_consumer", "source": "fixtures/hip_project"},  # duplicate
            {"name": "../escape", "source": "fixtures/hip_project"},
            {"name": "CON", "source": "fixtures/hip_project"},
            {"name": "nul", "source": "fixtures/hip_project"},
            {"name": "new", "source": ".."},
            {"name": "new", "source": str(self.source)},
            {"name": "new", "source": "C:/outside"},
            {"name": "new", "source": "fixtures/hip_project", "phase": "install"},
            {"name": "new", "source": "fixtures/hip_project", "hip": "true"},
            {"name": "new", "source": "fixtures/hip_project", "platforms": []},
            {"name": "new", "source": "fixtures/hip_project", "platforms": "linux"},
            {"name": "new", "source": "fixtures/hip_project", "platforms": ["linx"]},
            {
                "name": "new",
                "source": "fixtures/hip_project",
                "cmake_options": {"FLAG": True},
            },
            {
                "name": "new",
                "source": "fixtures/hip_project",
                "cmake_options": {"GPU_TARGETS": "gfx942"},
            },
            {
                "name": "new",
                "source": "fixtures/hip_project",
                "cmake_options": {"CMAKE_CXX_COMPILER:FILEPATH": "other"},
            },
            {"name": "new", "source": "fixtures/hip_project", "unexpected": True},
        ]
        for entry in invalid_entries:
            with self.subTest(entry=entry):
                self.manifest["tests"] = [
                    {"name": "hip_consumer", "source": "fixtures/hip_project"},
                    entry,
                ]
                self.write_manifest()
                with self.assertRaises(ValueError):
                    runner.run_tests(self.sdk, self.build, "gfx1100")
                self.assertFalse(self.build.exists())
        for manifest in [
            [],
            {},
            {"version": 2, "tests": []},
            {"version": True, "tests": []},
            {"version": 1, "tests": []},
            {"version": 1, "tests": "wrong"},
        ]:
            with self.subTest(manifest=manifest):
                self.manifest = manifest
                self.write_manifest()
                with self.assertRaises(ValueError):
                    runner.run_tests(self.sdk, self.build, "gfx1100")
                self.assertFalse(self.build.exists())
        (self.source / "tests.json").write_text("{broken json")
        with self.assertRaises(ValueError):
            runner.run_tests(self.sdk, self.build, "gfx1100")
        (self.source / "tests.json").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "manifest is missing"):
            runner.run_tests(self.sdk, self.build, "gfx1100")
        self.assertFalse(self.build.exists())

    def test_platform_filter_logs_and_rejects_empty_selection(self):
        self.manifest["tests"] = [self.manifest["tests"][1]]
        self.write_manifest()
        with self.assertLogs(level="INFO") as logs:
            with self.assertRaisesRegex(ValueError, "no applicable cases"):
                runner.load_cases(self.source, "windows")
        self.assertIn("Skipping linux_only_consumer", "\n".join(logs.output))

    def test_sources_cannot_escape_through_symlinks(self):
        outside = self.root / "outside"
        outside.mkdir()
        shutil.copyfile(
            self.source / "fixtures/hip_project/CMakeLists.txt",
            outside / "CMakeLists.txt",
        )
        try:
            (self.source / "escape").symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest("Directory symlinks unavailable")
        self.manifest["tests"] = [{"name": "escape", "source": "escape"}]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "inside"):
            runner.load_cases(self.source, "linux")

    def test_artifact_owns_consumer_sources(self):
        relative = Path("math-libs/rocWMMA/stage")
        stage = self.root / "artifact-input" / relative
        shutil.copytree(self.sdk, stage)
        for supporting in ["helper.cmake", "helper.h", "helper.hpp", "input.in"]:
            (stage / "share/rocwmma/tests/package_consumer" / supporting).touch()
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
            for project in [
                consumer / "fixtures/hip_project",
                consumer / "fixtures/platform_project",
            ]:
                self.assertEqual(
                    (project / "CMakeLists.txt").is_file(), component == "test"
                )
            for supporting in [
                "tests.json",
                "helper.cmake",
                "helper.h",
                "helper.hpp",
                "input.in",
            ]:
                self.assertEqual((consumer / supporting).is_file(), component == "test")
        self.assertTrue(
            (
                self.root / "dev" / relative / "lib/cmake/rocwmma/rocwmma-config.cmake"
            ).is_file()
        )

    def test_missing_sources_and_target_fail_before_building(self):
        with self.assertRaises(ValueError):
            runner.run_tests(self.sdk, self.build, "")
        (self.source / "fixtures/hip_project/CMakeLists.txt").unlink()
        with self.assertRaises(FileNotFoundError):
            runner.run_tests(self.sdk, self.build, "gfx1100")
        self.assertFalse(self.build.exists())


if __name__ == "__main__":
    unittest.main()
