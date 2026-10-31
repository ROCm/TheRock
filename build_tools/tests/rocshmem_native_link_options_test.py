# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Actual pinned target CMake generation; no ROCm compilation or linking."""

import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).with_name("fixtures") / "rocshmem-native-link"
PATCH = ROOT / "patches/amd-mainline/rocm-systems/0002-rocshmem-native-link-jobs.patch"


def source_text(name):
    with zipfile.ZipFile(HERE / "pinned-source.zip") as archive:
        return archive.read("source/" + name).decode("utf-8")


@unittest.skipUnless(
    sys.platform == "linux" and shutil.which("cmake") and shutil.which("ninja"),
    "Linux CMake/Ninja host generation fixture",
)
class NativeLinkOptionsTest(unittest.TestCase):
    def test_exact_pinned_source_and_normal_mail_patch_application(self):
        with zipfile.ZipFile(HERE / "pinned-source.zip") as archive:
            for row in json.loads((HERE / "source-index.json").read_text()):
                self.assertEqual(
                    hashlib.sha256(archive.read(row["file"])).hexdigest(), row["sha256"]
                )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in [
                "CMakeLists.txt",
                "tools/CMakeLists.txt",
                "tests/functional_tests/CMakeLists.txt",
            ]:
                destination = root / "projects/rocshmem" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    source_text(("projects/rocshmem/" + name).replace("/", "__"))
                )
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            options = [
                "git",
                "-C",
                str(root),
                "-c",
                "user.name=fixture",
                "-c",
                "user.email=fixture@example.invalid",
            ]
            subprocess.run(
                [*options, "commit", "-q", "--no-gpg-sign", "-m", "pinned fixture"],
                check=True,
            )
            subprocess.run(
                [*options, "am", "--no-gpg-sign", str(PATCH)],
                check=True,
                capture_output=True,
            )
            files = subprocess.check_output(
                ["git", "-C", str(root), "diff", "--name-only", "HEAD~1", "HEAD"],
                text=True,
            ).splitlines()
            self.assertEqual(
                set(files),
                {
                    "projects/rocshmem/CMakeLists.txt",
                    "projects/rocshmem/tools/CMakeLists.txt",
                    "projects/rocshmem/tests/functional_tests/CMakeLists.txt",
                },
            )

    def fixture(self, root):
        original = root / "patched-source"
        for relative in [
            "CMakeLists.txt",
            "tools/CMakeLists.txt",
            "tests/functional_tests/CMakeLists.txt",
        ]:
            dest = original / "projects/rocshmem" / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(
                source_text(("projects/rocshmem/" + relative).replace("/", "__"))
            )
        subprocess.run(["git", "-C", str(original), "apply", str(PATCH)], check=True)
        candidate = original / "projects/rocshmem"
        cmake = (candidate / "CMakeLists.txt").read_text()
        start = cmake.index("# Limit this control")
        block = cmake[
            start : cmake.index("endfunction()", start) + len("endfunction()")
        ]
        (root / "native-options.cmake").write_text(block + "\n")
        for relative in ["tools", "tests/functional_tests"]:
            destination = root / relative
            destination.mkdir(parents=True)
            text = (candidate / relative / "CMakeLists.txt").read_text()
            (destination / "CMakeLists.txt").write_text(text)
            for filename in set(re.findall(r"[A-Za-z0-9_]+\.(?:cpp|hpp)", text)):
                (destination / filename).write_text("int fixture();\n")
        (root / "tests/functional_tests/CMakeDeviceBitcodeTester.cmake").write_text(
            source_text(
                "projects__rocshmem__tests__functional_tests__CMakeDeviceBitcodeTester.cmake"
            )
        )
        for relative in [
            "cmake/setup_project.cmake",
            "scripts/functional_tests/driver.sh",
            "tests/functional_tests/test_wrapper.sh",
            "tests/functional_tests/ctest_with_retry.sh",
        ]:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# fixture-only non-performance scaffolding\n")
        architectures = json.loads((HERE / "architectures.json").read_text())[
            "historical_architectures"
        ]
        self.assertEqual(len(architectures), 31)
        flags = " ".join("--offload-arch=" + arch for arch in architectures)
        (root / "CMakeLists.txt").write_text(
            """cmake_minimum_required(VERSION 3.16)
    project(native_link_fixture LANGUAGES CXX)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    set(CMAKE_INSTALL_BINDIR bin)
    set(CMAKE_INSTALL_DATADIR share)
    set(BUILD_TESTS_ONLY OFF)
    set(BUILD_CTESTS OFF)
    set(USE_GDA ON)
    set(USE_SDMA ON)
    include(native-options.cmake)
    function(rocm_install)
    endfunction()
    add_library(fixture_rocshmem INTERFACE)
    add_library(roc::rocshmem ALIAS fixture_rocshmem)
    target_compile_options(fixture_rocshmem INTERFACE -fgpu-rdc """
            + flags
            + """)
    target_link_options(fixture_rocshmem INTERFACE --hip-link -fgpu-rdc """
            + flags
            + """)
    add_subdirectory(tools)
    add_subdirectory(tests/functional_tests)
    """
        )

    def configure(self, root, *options, succeeds=True):
        result = subprocess.run(
            [
                "cmake",
                "-S",
                str(root),
                "-B",
                str(root / "build"),
                "-G",
                "Ninja",
                "-DHIP_CLANG_SUPPORTS_PARALLEL_JOBS=TRUE",
                "-DHIP_CLANG_NUM_PARALLEL_JOBS=1",
                *options,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if succeeds:
            self.assertEqual(result.returncode, 0, result.stdout)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return result.stdout

    def capture(self, root):
        build = root / "build"
        commands = json.loads(
            subprocess.check_output(
                ["ninja", "-C", str(build), "-t", "compdb"], text=True
            )
        )
        compile_bytes = (build / "compile_commands.json").read_bytes()
        links = {
            row["output"]: shlex.split(row["command"])
            for row in commands
            if row.get("output")
            in [
                "tools/rocshmem_info",
                "tests/functional_tests/rocshmem_functional_tests",
            ]
        }
        self.assertEqual(len(links), 2)
        return compile_bytes, links, (build / "build.ninja").read_text()

    def test_actual_target_files_change_only_two_final_link_options(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            self.configure(root)
            baseline, links_a, graph_a = self.capture(root)
            self.configure(root, "-DROCSHMEM_NATIVE_LINK_JOBS=2")
            candidate, links_b, graph_b = self.capture(root)
            self.assertEqual(baseline, candidate)
            compile_rows = json.loads(baseline)
            self.assertEqual(len(compile_rows), 37)
            for target in links_a:
                self.assertNotIn("--offload-jobs=2", links_a[target])
                self.assertEqual(links_b[target].count("--offload-jobs=2"), 1)
                self.assertEqual(
                    [arg for arg in links_b[target] if arg != "--offload-jobs=2"],
                    links_a[target],
                )
                self.assertEqual(
                    sum(arg.startswith("--offload-arch=") for arg in links_b[target]),
                    31,
                )
                self.assertIn("--offload-arch=gfx1250-strict", links_b[target])
            proof = root / "proof"
            proof.mkdir(exist_ok=True)
            for name, data in [
                ("baseline-compile_commands.json", baseline),
                ("candidate-compile_commands.json", candidate),
            ]:
                (proof / name).write_bytes(data)
            (proof / "baseline-build.ninja").write_text(graph_a)
            (proof / "candidate-build.ninja").write_text(graph_b)
            (proof / "final-link-argv.json").write_text(
                json.dumps(dict(A=links_a, B=links_b), indent=2) + "\n"
            )
            (proof / "summary.json").write_text(
                json.dumps(
                    dict(
                        scope="CMake generation fixture using actual patched executable files; no compilation/performance run",
                        compile_edges=len(compile_rows),
                        compile_commands_byte_identical=True,
                        compile_commands_sha256=hashlib.sha256(baseline).hexdigest(),
                        final_links=2,
                        architecture_ids_per_link=31,
                        sole_added_link_argument="--offload-jobs=2",
                    ),
                    indent=2,
                )
                + "\n"
            )

    def test_invalid_unsupported_and_conflicting_controls_fail(self):
        for options, expected in [
            (["-DROCSHMEM_NATIVE_LINK_JOBS=0"], "positive integer"),
            (
                [
                    "-DROCSHMEM_NATIVE_LINK_JOBS=2",
                    "-DHIP_CLANG_SUPPORTS_PARALLEL_JOBS=FALSE",
                ],
                "requires HIP",
            ),
            (
                ["-DROCSHMEM_NATIVE_LINK_JOBS=2", "-DHIP_CLANG_NUM_PARALLEL_JOBS=2"],
                "HIP_CLANG_NUM_PARALLEL_JOBS=1",
            ),
        ]:
            with self.subTest(
                options=options
            ), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.fixture(root)
                self.assertIn(expected, self.configure(root, *options, succeeds=False))


if __name__ == "__main__":
    unittest.main()
