#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check runtime input collection and retained subproject stamps without compilers."""

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

THEROCK_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_INPUTS = THEROCK_ROOT / "core/rocm-runtimes/RuntimeInputs.cmake"
CMAKE = shutil.which("cmake")
NINJA = shutil.which("ninja")
INPUT_NAMES = ("CARGO", "RUSTC", "LLD", "CARGO_HOME", "CARGO_CONFIG", "CARGO_JOBS")


def write_file(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(contents), encoding="utf-8")


@unittest.skipUnless(CMAKE and NINJA, "Requires CMake and Ninja")
class RuntimeInputsTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory(prefix="therock runtime inputs ")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.source = self.root / "source"
        self.build = self.root / "build"
        self.runtime = self.source / "systems/runtimes"
        self.environment = os.environ.copy()
        for name in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS"):
            self.environment.pop(name, None)
        self.project = f"""
            cmake_minimum_required(VERSION 3.25)
            set(THEROCK_ROCM_SYSTEMS_SOURCE_DIR "${{CMAKE_SOURCE_DIR}}/systems")
            include("{RUNTIME_INPUTS.as_posix()}")
            therock_prepare_runtime_inputs(args state depends sources)
            foreach(name args state depends sources)
              file(WRITE "${{CMAKE_BINARY_DIR}}/${{name}}.txt" "${{${{name}}}}")
            endforeach()
            # Capture inputs before restoring PATH for generator tool launchers.
            if(DEFINED FIXTURE_TOOL_PATH)
              set(ENV{{PATH}} "${{FIXTURE_TOOL_PATH}}")
            endif()
            project(runtime_inputs_fixture NONE)
        """
        write_file(self.source / "CMakeLists.txt", self.project)
        self.inputs = (
            "CMakeLists.txt",
            "Cargo.toml",
            "Cargo.lock",
            "rust-toolchain.toml",
            ".cargo/config",
            "build.rs",
            "ddi/src/lib.rs",
            "native/kernel.cpp",
            "include/api.h",
            "cmake/configure.py",
            "native/kernel.S",
            "native/kernel.map",
            "native/shader.bin",
            "tests/fixtures/input",
        )
        for relative in self.inputs:
            write_file(self.runtime / relative, "# input\n")
        for relative in (
            "target/build/generated.rs",
            "cargo-home/registry/Cargo.toml",
            "native/target/generated.cmake",
            ".git/config",
            "README.md",
        ):
            write_file(self.runtime / relative, "output or documentation\n")

    def run_command(self, *arguments: str, cwd: Path | None = None) -> None:
        result = subprocess.run(
            arguments,
            cwd=cwd or self.root,
            env=self.environment,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            " ".join(arguments) + "\n" + result.stdout + result.stderr,
        )

    def configure(self, *arguments: str, cwd: Path | None = None) -> None:
        self.run_command(
            CMAKE,
            "-S",
            str(self.source),
            "-B",
            str(self.build),
            "-GNinja",
            f"-DCMAKE_MAKE_PROGRAM={Path(NINJA).as_posix()}",
            *arguments,
            cwd=cwd,
        )

    def output(self, name: str) -> list[str]:
        return [
            value
            for value in (self.build / f"{name}.txt").read_text().split(";")
            if value
        ]

    def state(self) -> Path:
        return Path((self.build / "state.txt").read_text())

    def test_sources_dependencies_and_default_resets(self) -> None:
        self.configure()
        self.assertCountEqual(
            self.output("args"), [f"-UROCM_RUNTIMES_{name}" for name in INPUT_NAMES]
        )
        expected_sources = {
            (self.runtime / relative).as_posix() for relative in self.inputs
        }
        expected_sources.add(RUNTIME_INPUTS.as_posix())
        self.assertEqual(set(self.output("sources")), expected_sources)
        configure_inputs = (
            "CMakeLists.txt",
            "Cargo.toml",
            "Cargo.lock",
            "rust-toolchain.toml",
            ".cargo/config",
            "include/api.h",
            "cmake/configure.py",
        )
        self.assertEqual(
            set(self.output("depends")),
            {self.state().as_posix(), RUNTIME_INPUTS.as_posix()}
            | {(self.runtime / relative).as_posix() for relative in configure_inputs},
        )
        cache = (self.build / "CMakeCache.txt").read_text()
        for name in INPUT_NAMES:
            kind = (
                "STRING"
                if name == "CARGO_JOBS"
                else "PATH" if name == "CARGO_HOME" else "FILEPATH"
            )
            self.assertIn(f"ROCM_RUNTIMES_{name}:{kind}=\n", cache)

    def test_membership_changes_and_output_pruning(self) -> None:
        self.configure()
        original = self.state().read_bytes()
        added = self.runtime / "new-crate/src/added.rs"
        write_file(added, "// new source\n")
        self.run_command(CMAKE, "--build", str(self.build))
        self.assertIn(added.as_posix(), self.output("sources"))
        self.assertNotEqual(self.state().read_bytes(), original)
        added.unlink()
        self.run_command(CMAKE, "--build", str(self.build))
        self.assertNotIn(added.as_posix(), self.output("sources"))
        self.assertEqual(self.state().read_bytes(), original)
        for relative in ("target/new/generated.rs", "cargo-home/new/Cargo.toml"):
            write_file(self.runtime / relative, "generated\n")
        self.run_command(CMAKE, "--build", str(self.build))
        self.assertEqual(self.state().read_bytes(), original)

    def test_noop_configure_preserves_state_file(self) -> None:
        self.configure()
        contents = self.state().read_bytes()
        # An old timestamp makes an accidental rewrite observable without a sleep.
        os.utime(self.state(), (1, 1))
        timestamp = self.state().stat().st_mtime_ns
        self.configure()
        self.assertEqual(self.state().read_bytes(), contents)
        self.assertEqual(self.state().stat().st_mtime_ns, timestamp)

    def test_explicit_inputs_set_change_clear_and_normalize_paths(self) -> None:
        invocation = self.root / "invocation"
        invocation.mkdir()
        previous = None
        for index, directory in enumerate(("first inputs", "second inputs")):
            values = {
                name: (invocation / directory / name).as_posix() for name in INPUT_NAMES
            }
            values["RUSTC"] = (self.source / directory / "RUSTC").as_posix()
            values["CARGO_JOBS"] = str(index + 2)
            for name, value in values.items():
                if name not in ("CARGO_HOME", "CARGO_JOBS"):
                    write_file(Path(value), "explicit input\n")
            # Untyped paths follow the invocation cwd; typed paths follow -S.
            arguments = [
                f"-DROCM_RUNTIMES_{name}={directory}/{name}"
                for name in INPUT_NAMES
                if name not in ("RUSTC", "CARGO_JOBS")
            ]
            arguments += [
                f"-DROCM_RUNTIMES_CARGO_JOBS={values['CARGO_JOBS']}",
                f"-DROCM_RUNTIMES_RUSTC:FILEPATH={directory}/RUSTC",
            ]
            if index:
                arguments = [
                    f"-DROCM_RUNTIMES_{name}={value}" for name, value in values.items()
                ]
            self.configure(*arguments, cwd=invocation)
            current = self.state().read_bytes()
            if previous is not None:
                self.assertNotEqual(current, previous)
            previous = current
            self.assertCountEqual(
                self.output("args"),
                [f"-DROCM_RUNTIMES_{name}={value}" for name, value in values.items()],
            )
            dependencies = self.output("depends")
            for name in ("CARGO", "RUSTC", "LLD", "CARGO_CONFIG"):
                self.assertIn(values[name], dependencies)
            self.assertNotIn(values["CARGO_HOME"], dependencies)
        self.configure(
            *[
                f"-UROCM_RUNTIMES_{name}" if index % 2 else f"-DROCM_RUNTIMES_{name}="
                for index, name in enumerate(INPUT_NAMES)
            ]
        )
        self.assertCountEqual(
            self.output("args"), [f"-UROCM_RUNTIMES_{name}" for name in INPUT_NAMES]
        )
        self.assertNotEqual(self.state().read_bytes(), previous)

    def test_captured_environment_changes_and_definedness(self) -> None:
        self.configure()
        tool_path = self.environment.get("PATH", os.defpath)
        for name in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "PATH"):
            original = self.environment.get(name)
            for value in (
                (
                    "--cfg changed\x1fflag"
                    if name == "CARGO_ENCODED_RUSTFLAGS"
                    else "changed"
                ),
                "",
                None,
            ):
                with self.subTest(name=name, value=value):
                    previous = self.state().read_bytes()
                    if value is None:
                        self.environment.pop(name, None)
                    else:
                        self.environment[name] = value
                    self.configure(f"-DFIXTURE_TOOL_PATH={tool_path}")
                    state = self.state().read_text()
                    self.assertNotEqual(self.state().read_bytes(), previous)
                    self.assertIn(
                        f"ENV_{name}_DEFINED={'OFF' if value is None else 'ON'}\n",
                        state,
                    )
                    if value is not None:
                        self.assertIn(f"ENV_{name}={value}\n", state)
            if original is not None:
                self.environment[name] = original

    def test_retained_subproject_build_and_test_stamps(self) -> None:
        write_file(
            self.source / "CMakeLists.txt",
            self.project
            + f"""
            set(THEROCK_SOURCE_DIR "{THEROCK_ROOT.as_posix()}")
            set(THEROCK_BINARY_DIR "${{CMAKE_BINARY_DIR}}")
            set(Python3_EXECUTABLE "{Path(sys.executable).as_posix()}")
            set(THEROCK_BUILD_TESTING ON)
            list(APPEND CMAKE_MODULE_PATH "${{THEROCK_SOURCE_DIR}}/cmake")
            set(ROCM_BUILD_FLAGS_STATE_FILE "${{CMAKE_BINARY_DIR}}/flags.cmake")
            file(CONFIGURE OUTPUT "${{ROCM_BUILD_FLAGS_STATE_FILE}}" CONTENT "# no flags\n")
            include(therock_globals)
            include(therock_sanitizers)
            include(therock_flag_utils)
            include(therock_default_targets)
            include(therock_subproject)
            therock_cmake_subproject_declare(fixture
              DISABLE_AMDGPU_TARGETS
              EXTERNAL_SOURCE_DIR "${{THEROCK_ROCM_SYSTEMS_SOURCE_DIR}}/runtimes"
              BINARY_DIR "${{CMAKE_BINARY_DIR}}/fixture"
              FPRINT_FILE_GLOBS "${{state}}"
              EXTRA_DEPENDS ${{depends}}
              CMAKE_ARGS ${{args}})
            target_sources(fixture PRIVATE ${{sources}})
            therock_cmake_subproject_activate(fixture)
            therock_cmake_subproject_build_test(fixture
              COMMAND "${{CMAKE_COMMAND}}" -DMODE=test
                -P "${{THEROCK_ROCM_SYSTEMS_SOURCE_DIR}}/runtimes/record.cmake")
        """,
        )
        write_file(
            self.runtime / "CMakeLists.txt",
            """
            cmake_minimum_required(VERSION 3.25)
            project(runtime_child NONE)
            file(APPEND "${CMAKE_BINARY_DIR}/configure.txt" "run\n")
            add_custom_target(record ALL COMMAND "${CMAKE_COMMAND}" -DMODE=build
              -P "${CMAKE_CURRENT_SOURCE_DIR}/record.cmake")
        """,
        )
        write_file(
            self.runtime / "record.cmake",
            'file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/${MODE}.txt" "run\n")\n',
        )
        self.configure()
        child = self.build / "fixture/build"

        def build_and_count() -> list[int]:
            self.run_command(
                CMAKE,
                "--build",
                str(self.build),
                "--target",
                "fixture+build",
                "build-test-fixture",
            )
            return [
                len((child / f"{name}.txt").read_text().splitlines())
                for name in ("configure", "build", "test")
            ]

        self.assertEqual(build_and_count(), [1, 1, 1])
        self.assertEqual(build_and_count(), [1, 1, 1])
        write_file(self.runtime / "ddi/src/lib.rs", "// changed source\n")
        self.assertEqual(build_and_count(), [1, 2, 2])
        write_file(self.runtime / "Cargo.toml", "# changed configuration\n")
        self.assertEqual(build_and_count(), [2, 3, 3])
        self.assertEqual(build_and_count(), [2, 3, 3])


@unittest.skipUnless(CMAKE and NINJA, "Requires CMake and Ninja")
class RuntimeOptionTest(unittest.TestCase):
    def test_production_option_defaults_prerequisites_and_reset(self) -> None:
        root_cmake = (THEROCK_ROOT / "CMakeLists.txt").read_text()
        production_block = root_cmake.split(
            "# Additional manual features that don't map to artifacts\n", 1
        )[1].split("# Finalize all feature flags.", 1)[0]
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            write_file(
                source / "CMakeLists.txt",
                "cmake_minimum_required(VERSION 3.25)\nproject(runtime_option NONE)\n"
                f'include("{THEROCK_ROOT.as_posix()}/cmake/therock_features.cmake")\n'
                'option(THEROCK_ENABLE_ALL "All" ON)\n'
                'option(THEROCK_ENABLE_CORE "Core" "${THEROCK_ENABLE_ALL}")\n'
                'therock_add_feature(SYSDEPS GROUP CORE DESCRIPTION "System dependencies")\n'
                'therock_add_feature(BASE GROUP CORE DESCRIPTION "Base")\n'
                'therock_add_feature(COMPILER GROUP ALL REQUIRES SYSDEPS DESCRIPTION "Compiler")\n'
                + production_block
                + "therock_finalize_features()\ntherock_report_features()\n"
                'file(WRITE "${CMAKE_BINARY_DIR}/selected.txt" '
                '"${THEROCK_ENABLE_ROCDDI_RUNTIMES};${THEROCK_ENABLE_BASE};'
                '${THEROCK_ENABLE_COMPILER};${THEROCK_ENABLE_SYSDEPS};${THEROCK_RESET_FEATURES}")\n',
            )
            for directory, options, expected in (
                ("default", [], "OFF;ON;ON;ON;"),
                ("minimal", ["-DTHEROCK_ENABLE_ALL=OFF"], "OFF;OFF;OFF;OFF;"),
                ("minimal", ["-DTHEROCK_ENABLE_ROCDDI_RUNTIMES=ON"], "ON;ON;ON;ON;"),
                ("minimal", [], "ON;ON;ON;ON;"),
                (
                    "minimal",
                    ["-DTHEROCK_ENABLE_ROCDDI_RUNTIMES=OFF"],
                    "OFF;OFF;OFF;OFF;",
                ),
                ("minimal", ["-DTHEROCK_ENABLE_ROCDDI_RUNTIMES=ON"], "ON;ON;ON;ON;"),
                ("minimal", ["-DTHEROCK_RESET_FEATURES=ON"], "OFF;OFF;OFF;OFF;OFF"),
            ):
                with self.subTest(directory=directory, options=options):
                    build = source / directory
                    result = subprocess.run(
                        [
                            CMAKE,
                            "-S",
                            str(source),
                            "-B",
                            str(build),
                            "-GNinja",
                            f"-DCMAKE_MAKE_PROGRAM={Path(NINJA).as_posix()}",
                            *options,
                        ],
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    self.assertEqual((build / "selected.txt").read_text(), expected)


if __name__ == "__main__":
    unittest.main()
