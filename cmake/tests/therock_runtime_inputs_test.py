#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise the runtime producer declaration with compiler-free dependencies.

The entire production core/CMakeLists.txt is copied without modification and
entered through add_subdirectory. Only the child runtime and LLVM dependency
are fixtures; the subproject constructor, commands, and stamps are production.
"""

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

THEROCK_ROOT = Path(__file__).resolve().parents[2]
BASELINE_SUPPORTED = platform.system() == "Linux" and platform.machine().lower() in (
    "x86_64",
    "amd64",
)


def write_file(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(contents), encoding="utf-8")


@unittest.skipUnless(platform.system() == "Linux", "Runtime producer requires Linux")
class RuntimeInputsTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory(prefix="therock runtime inputs ")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / "source"
        self.build = self.root / "build"
        self.runtime = self.source / "systems" / "runtimes"
        self.component = self.source / "core" / "rocm-runtimes"
        self.child = self.build / "core" / "rocm-runtimes" / "build"
        self.stage = self.build / "core" / "rocm-runtimes" / "stage"
        self.tools = self.root / "tools"
        self.environment = os.environ.copy()
        self.environment.pop("CARGO_HOME", None)
        self.environment.pop("CARGO_ENCODED_RUSTFLAGS", None)
        self.environment["RUSTFLAGS"] = "--cfg user_runtime_flag"
        self.make_tools(self.tools)
        self.environment["PATH"] = str(self.tools) + os.pathsep + os.environ["PATH"]
        write_file(
            self.source / "CMakeLists.txt",
            f"""
            cmake_minimum_required(VERSION 3.25)
            project(runtime_inputs_fixture NONE)
            set(THEROCK_SOURCE_DIR "{THEROCK_ROOT.as_posix()}")
            set(THEROCK_BINARY_DIR "${{CMAKE_BINARY_DIR}}")
            set(THEROCK_ROCM_SYSTEMS_SOURCE_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}/systems")
            set(Python3_EXECUTABLE "{Path(sys.executable).as_posix()}")
            option(THEROCK_BUILD_TESTING "Build tests" ON)
            set(THEROCK_ENABLE_ROCDDI_RUNTIMES ON)
            list(APPEND CMAKE_MODULE_PATH "${{THEROCK_SOURCE_DIR}}/cmake")
            set(ROCM_BUILD_FLAGS_STATE_FILE "${{CMAKE_BINARY_DIR}}/flags.cmake")
            file(CONFIGURE OUTPUT "${{ROCM_BUILD_FLAGS_STATE_FILE}}" CONTENT "# no flags\\n")
            include(therock_globals)
            include(therock_sanitizers)
            include(therock_flag_utils)
            include(therock_default_targets)
            include(therock_subproject)
            therock_cmake_subproject_declare(amd-llvm
              EXTERNAL_SOURCE_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}/llvm"
              BINARY_DIR "${{CMAKE_BINARY_DIR}}/llvm")
            therock_cmake_subproject_activate(amd-llvm)
            add_subdirectory(core)
            """,
        )
        self.component.mkdir(parents=True)
        shutil.copyfile(
            THEROCK_ROOT / "core/rocm-runtimes/RuntimeInputs.cmake",
            self.component / "RuntimeInputs.cmake",
        )
        shutil.copyfile(
            THEROCK_ROOT / "core" / "CMakeLists.txt",
            self.source / "core" / "CMakeLists.txt",
        )
        write_file(
            self.source / "llvm" / "CMakeLists.txt",
            """
            cmake_minimum_required(VERSION 3.25)
            project(llvm_fixture NONE)
            install(FILES resource.txt DESTINATION lib/llvm/lib/clang/fixture)
            """,
        )
        write_file(
            self.source / "llvm" / "resource.txt", "compiler-free resource directory\n"
        )
        for relative in (
            "CMakeLists.txt",
            "Cargo.toml",
            "Cargo.lock",
            "rust-toolchain.toml",
            ".cargo/config",
            "build.rs",
            "ddi/src/lib.rs",
            "image/vendor/native.cpp",
            "api-headers/include/api.h",
            "cmake/configure.py",
        ):
            write_file(self.runtime / relative, f"initial {relative}\n")
        # Existing Cargo output directories are deliberately outside the input
        # walk, including Rust and CMake files emitted by dependency build.rs.
        write_file(self.runtime / "target" / "build" / "generated.rs", "output\n")
        write_file(self.runtime / "cargo-home" / "registry" / "Cargo.toml", "output\n")
        write_file(
            self.runtime / "ConfigureFixture.cmake", 'set(FIXTURE_PREFIX "initial")\n'
        )
        write_file(
            self.runtime / "fixture.txt.in", "@FIXTURE_PREFIX@ @PUBLIC_HEADER@\n"
        )
        write_file(
            self.runtime / "CMakeLists.txt",
            """
            cmake_minimum_required(VERSION 3.25)
            project(runtime_child_fixture NONE)
            file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/configure-runs.txt" "configure\\n")
            include(CTest)
            find_program(ROCM_RUNTIMES_CARGO NAMES cargo REQUIRED)
            find_program(ROCM_RUNTIMES_RUSTC NAMES rustc REQUIRED)
            find_program(ROCM_RUNTIMES_LLD NAMES ld.lld REQUIRED)
            set(ROCM_RUNTIMES_CARGO_HOME "${CMAKE_CURRENT_BINARY_DIR}/cargo-home" CACHE PATH "")
            set(ROCM_RUNTIMES_CARGO_CONFIG "" CACHE FILEPATH "")
            set(ROCM_RUNTIMES_CARGO_JOBS "" CACHE STRING "")
            include(ConfigureFixture.cmake)
            file(READ "${CMAKE_CURRENT_SOURCE_DIR}/api-headers/include/api.h" PUBLIC_HEADER)
            configure_file(fixture.txt.in fixture.txt @ONLY)
            # Model upstream's configure-time environment capture.
            set(_captured_environment)
            foreach(_name RUSTFLAGS CARGO_ENCODED_RUSTFLAGS PATH)
              if(DEFINED ENV{${_name}})
                list(APPEND _captured_environment "${_name}=$ENV{${_name}}")
              else()
                list(APPEND _captured_environment "--unset=${_name}")
              endif()
            endforeach()
            set(_cargo_command "${CMAKE_COMMAND}" -E env ${_captured_environment}
              "CARGO_HOME=${ROCM_RUNTIMES_CARGO_HOME}" "CARGO_NET_OFFLINE=true"
              "RUSTUP_TOOLCHAIN=fixture" "RUSTUP_AUTO_INSTALL=0"
              "RUSTC=${ROCM_RUNTIMES_RUSTC}"
              "${ROCM_RUNTIMES_CARGO}")
            set(_arguments --frozen
              --source "${CMAKE_CURRENT_SOURCE_DIR}"
              --output "${CMAKE_CURRENT_BINARY_DIR}"
              "--config=${ROCM_RUNTIMES_CARGO_CONFIG}"
              "--jobs=${ROCM_RUNTIMES_CARGO_JOBS}")
            add_custom_target(payload ALL
              COMMAND ${_cargo_command} build ${_arguments} VERBATIM)
            if(BUILD_TESTING)
              add_test(NAME runtime_host COMMAND ${_cargo_command} test ${_arguments})
              set_tests_properties(runtime_host PROPERTIES LABELS host)
              add_test(NAME runtime_device COMMAND "${CMAKE_COMMAND}" -E false)
              set_tests_properties(runtime_device PROPERTIES LABELS device)
            endif()
            """,
        )

    def make_tools(self, directory: Path) -> None:
        for name in ("cargo", "rustc", "ld.lld"):
            write_file(
                directory / name,
                f"""\
                #!{sys.executable}
                import argparse
                import json
                import os
                from pathlib import Path

                parser = argparse.ArgumentParser()
                parser.add_argument("mode")
                parser.add_argument("--frozen", action="store_true")
                for argument in ("source", "output", "config", "jobs"):
                    parser.add_argument("--" + argument, required=True)
                arguments = parser.parse_args()
                output = Path(arguments.output)
                payload = {{}}
                root = Path(arguments.source)
                for path in sorted(root.rglob("*")):
                    relative = path.relative_to(root)
                    if path.is_file() and not {{"target", "cargo-home"}}.intersection(relative.parts):
                        payload["source/" + relative.as_posix()] = path.read_text()
                payload["generated.txt"] = (output / "fixture.txt").read_text()
                if arguments.mode == "build":
                    (output / "payload.json").write_text(json.dumps(payload, sort_keys=True))
                else:
                    assert payload == json.loads((output / "payload.json").read_text())
                record = vars(arguments) | {{
                    "executable": __file__,
                    "environment": {{name: os.environ.get(name) for name in (
                        "CARGO_HOME", "CARGO_NET_OFFLINE", "RUSTUP_TOOLCHAIN",
                        "RUSTUP_AUTO_INSTALL", "RUSTC", "RUSTFLAGS",
                        "CARGO_ENCODED_RUSTFLAGS", "PATH")}},
                }}
                with (output / "executions.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\\n")
                """,
            )
            (directory / name).chmod(0o755)

    def run_command(self, *arguments: str, cwd: Path | None = None) -> str:
        result = subprocess.run(
            arguments,
            cwd=cwd or self.root,
            env=self.environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.assertEqual(
            result.returncode, 0, " ".join(arguments) + "\n" + result.stdout
        )
        return result.stdout

    def configure(self, *arguments: str, cwd: Path | None = None) -> str:
        return self.run_command(
            "cmake",
            "-S",
            str(self.source),
            "-B",
            str(self.build),
            "-GNinja",
            *arguments,
            cwd=cwd,
        )

    def build_and_test(self) -> list[dict]:
        self.run_command(
            "cmake",
            "--build",
            str(self.build),
            "--target",
            "rocm-runtimes",
            "build-test-rocm-runtimes",
        )
        return [
            json.loads(line)
            for line in (self.child / "executions.jsonl").read_text().splitlines()
        ]

    def child_cache(self) -> dict[str, str]:
        return {
            line.split(":", 1)[0]: line.split("=", 1)[1]
            for line in (self.child / "CMakeCache.txt").read_text().splitlines()
            if line.startswith("ROCM_RUNTIMES_") and "=" in line
        }

    def test_retained_build_tracks_runtime_and_component_inputs(self) -> None:
        self.configure()
        executions = self.build_and_test()
        self.assertEqual([item["mode"] for item in executions], ["build", "test"])
        self.assertEqual(self.build_and_test(), executions)
        self.assertTrue((self.build / "core/rocm-runtimes/stamp/stage.stamp").exists())
        self.assertEqual(list(self.stage.iterdir()), [])
        edits = (
            (self.runtime / "ddi/src/lib.rs", "changed Rust\n"),
            (self.runtime / "image/vendor/native.cpp", "changed native\n"),
            (self.runtime / "build.rs", "changed build script\n"),
            (self.runtime / "Cargo.toml", "changed manifest\n"),
            (self.runtime / "Cargo.lock", "changed lock\n"),
            (self.runtime / ".cargo/config", "changed workspace configuration\n"),
            (
                self.runtime / "CMakeLists.txt",
                (self.runtime / "CMakeLists.txt").read_text()
                + "# changed upstream configure\n",
            ),
            (self.runtime / "cmake/configure.py", "changed configure helper\n"),
            (self.runtime / "api-headers/include/api.h", "changed public header\n"),
            (
                self.component / "RuntimeInputs.cmake",
                (self.component / "RuntimeInputs.cmake").read_text()
                + "# changed component helper\n",
            ),
            (
                self.runtime / "ConfigureFixture.cmake",
                'set(FIXTURE_PREFIX "changed")\n',
            ),
            (
                self.runtime / "fixture.txt.in",
                "template @FIXTURE_PREFIX@ @PUBLIC_HEADER@\n",
            ),
            (self.runtime / "new-crate/src/added.rs", "new crate\n"),
        )
        for path, contents in edits:
            with self.subTest(path=path):
                configure_runs = (self.child / "configure-runs.txt").read_text()
                write_file(path, contents)
                current = self.build_and_test()
                self.assertEqual(len(current), len(executions) + 2)
                self.assertEqual(
                    [item["mode"] for item in current[-2:]], ["build", "test"]
                )
                self.assertIn(
                    (self.runtime / "api-headers/include/api.h").read_text(),
                    (self.child / "fixture.txt").read_text(),
                )
                if (
                    path.suffix in (".toml", ".lock", ".py", ".h", ".cmake", ".in")
                    or path.name == "CMakeLists.txt"
                    or path.name == "config"
                ):
                    self.assertGreater(
                        len((self.child / "configure-runs.txt").read_text()),
                        len(configure_runs),
                    )
                executions = current
        self.assertEqual(
            (self.child / "fixture.txt").read_text(),
            "template changed changed public header\n\n",
        )
        (self.runtime / "new-crate/src/added.rs").unlink()
        current = self.build_and_test()
        self.assertEqual(len(current), len(executions) + 2)
        self.assertNotIn(
            "source/new-crate/src/added.rs",
            json.loads((self.child / "payload.json").read_text()),
        )
        write_file(self.runtime / "target/build/new-output.rs", "generated\n")
        write_file(
            self.runtime / "cargo-home/registry/new-dep/Cargo.toml", "generated\n"
        )
        write_file(self.component / "README.md", "documentation only\n")
        self.assertEqual(self.build_and_test(), current)

    def test_retained_cache_set_change_clear_and_relative_paths(self) -> None:
        invocation = self.root / "invocation"
        invocation.mkdir()
        actual_config = self.root / "actual configuration"
        write_file(actual_config / "cargo.toml", "[net]\noffline=true\n")
        visible_config = self.root / "visible configuration"
        visible_config.symlink_to(actual_config, target_is_directory=True)
        config = visible_config / "cargo.toml"
        self.configure(
            "-DROCM_RUNTIMES_CARGO_CONFIG=../visible configuration/cargo.toml",
            "-DROCM_RUNTIMES_CARGO_JOBS=2",
            "-DROCM_RUNTIMES_CARGO_HOME=../explicit home",
            f"-DROCM_RUNTIMES_LLD={self.tools / 'ld.lld'}",
            cwd=invocation,
        )
        executions = self.build_and_test()
        self.assertEqual(executions[-1]["config"], str(config))
        self.assertEqual(executions[-1]["jobs"], "2")
        self.assertEqual(
            executions[-1]["environment"]["CARGO_HOME"],
            str(self.root / "explicit home"),
        )
        state = (self.build / "core/rocm-runtimes-inputs.txt").read_bytes()
        self.configure(cwd=self.build)
        self.assertEqual(
            (self.build / "core/rocm-runtimes-inputs.txt").read_bytes(), state
        )
        self.assertEqual(self.build_and_test(), executions)
        other = self.root / "other tools"
        self.make_tools(other)
        other_config = self.root / "other cargo.toml"
        write_file(other_config, "[build]\njobs=3\n")
        self.configure(
            f"-DROCM_RUNTIMES_CARGO_CONFIG={other_config}",
            "-DROCM_RUNTIMES_CARGO_JOBS=3",
            f"-DROCM_RUNTIMES_LLD={other / 'ld.lld'}",
        )
        executions = self.build_and_test()
        cache = self.child_cache()
        self.assertEqual(cache["ROCM_RUNTIMES_LLD"], str(other / "ld.lld"))
        self.assertEqual(cache["ROCM_RUNTIMES_CARGO"], str(self.tools / "cargo"))
        self.assertEqual(executions[-1]["config"], str(other_config))
        self.assertEqual(executions[-1]["jobs"], "3")
        self.environment["CARGO_HOME"] = str(self.root / "environment home")
        self.configure(
            "-DROCM_RUNTIMES_CARGO_CONFIG=",
            "-DROCM_RUNTIMES_CARGO_JOBS=",
            "-UROCM_RUNTIMES_CARGO_HOME",
            "-UROCM_RUNTIMES_LLD",
        )
        executions = self.build_and_test()
        self.assertEqual(executions[-1]["config"], "")
        self.assertEqual(executions[-1]["jobs"], "")
        self.assertEqual(
            self.child_cache()["ROCM_RUNTIMES_LLD"], str(self.tools / "ld.lld")
        )
        self.assertEqual(
            executions[-1]["environment"]["CARGO_HOME"], str(self.child / "cargo-home")
        )
        self.configure("-DROCM_RUNTIMES_CARGO_HOME=")
        executions = self.build_and_test()
        self.assertEqual(
            executions[-1]["environment"]["CARGO_HOME"], str(self.child / "cargo-home")
        )
        self.environment.pop("CARGO_HOME")
        self.environment["PATH"] = str(other) + os.pathsep + os.environ["PATH"]
        self.configure(
            "-DROCM_RUNTIMES_CARGO=",
            f"-DROCM_RUNTIMES_RUSTC={self.tools / 'rustc'}",
        )
        executions = self.build_and_test()
        self.assertEqual(executions[-1]["executable"], str(other / "cargo"))
        self.assertEqual(
            executions[-1]["environment"]["RUSTC"], str(self.tools / "rustc")
        )
        self.configure("-DROCM_RUNTIMES_RUSTC=")
        executions = self.build_and_test()
        self.assertEqual(executions[-1]["executable"], str(other / "cargo"))
        self.assertEqual(executions[-1]["environment"]["RUSTC"], str(other / "rustc"))
        self.configure(
            f"-DROCM_RUNTIMES_CARGO={self.tools / 'cargo'}",
            f"-DROCM_RUNTIMES_RUSTC={self.tools / 'rustc'}",
        )
        executions = self.build_and_test()
        self.assertEqual(executions[-1]["executable"], str(self.tools / "cargo"))
        self.assertEqual(
            executions[-1]["environment"]["RUSTC"], str(self.tools / "rustc")
        )
        self.configure(
            "-UROCM_RUNTIMES_CARGO",
            "-UROCM_RUNTIMES_RUSTC",
            "-UROCM_RUNTIMES_CARGO_HOME",
        )
        executions = self.build_and_test()
        self.assertEqual(executions[-1]["executable"], str(other / "cargo"))
        self.assertEqual(executions[-1]["environment"]["RUSTC"], str(other / "rustc"))
        self.assertEqual(self.child_cache()["ROCM_RUNTIMES_LLD"], str(other / "ld.lld"))
        for execution in executions:
            self.assertTrue(execution["frozen"])
            self.assertEqual(execution["environment"]["CARGO_NET_OFFLINE"], "true")
            self.assertEqual(execution["environment"]["RUSTUP_AUTO_INSTALL"], "0")
            self.assertEqual(
                execution["environment"]["RUSTFLAGS"], self.environment["RUSTFLAGS"]
            )

    def test_tools_and_config_content_invalidate_retained_build(self) -> None:
        config = self.root / "cargo.toml"
        write_file(config, "initial\n")
        self.configure(
            f"-DROCM_RUNTIMES_CARGO_CONFIG={config}",
            f"-DROCM_RUNTIMES_LLD={self.tools / 'ld.lld'}",
        )
        executions = self.build_and_test()
        for path in (
            config,
            *(self.tools / name for name in ("cargo", "rustc", "ld.lld")),
        ):
            with self.subTest(path=path):
                path.write_text(path.read_text() + "\n# changed\n")
                current = self.build_and_test()
                self.assertEqual(len(current), len(executions) + 2)
                executions = current

    def test_configure_captured_environment_changes_and_clearing(self) -> None:
        self.configure()
        executions = self.build_and_test()
        changes = (
            ("RUSTFLAGS", "--cfg changed_runtime_flag"),
            ("RUSTFLAGS", ""),
            ("RUSTFLAGS", None),
            ("RUSTFLAGS", "--cfg restored_runtime_flag"),
            ("CARGO_ENCODED_RUSTFLAGS", "--cfg\x1fencoded_runtime_flag"),
            ("CARGO_ENCODED_RUSTFLAGS", ""),
            ("CARGO_ENCODED_RUSTFLAGS", None),
            ("PATH", self.environment["PATH"] + os.pathsep + str(self.root)),
        )
        for name, value in changes:
            with self.subTest(name=name, value=value):
                if value is None:
                    self.environment.pop(name)
                else:
                    self.environment[name] = value
                previous = (self.child / "configure-runs.txt").read_text()
                self.configure()
                current = self.build_and_test()
                self.assertEqual(len(current), len(executions) + 2)
                self.assertGreater(
                    len((self.child / "configure-runs.txt").read_text()), len(previous)
                )
                for captured in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS"):
                    self.assertEqual(
                        current[-1]["environment"][captured],
                        self.environment.get(captured),
                    )
                self.assertTrue(
                    current[-1]["environment"]["PATH"].endswith(
                        self.environment["PATH"]
                    )
                )
                self.configure()
                self.assertEqual(self.build_and_test(), current)
                executions = current

    def test_testing_off_removes_outer_test_target(self) -> None:
        self.configure()
        self.build_and_test()
        self.configure("-DTHEROCK_BUILD_TESTING=OFF")
        self.assertNotIn(
            "build-test-rocm-runtimes",
            self.run_command("cmake", "--build", str(self.build), "--target", "help"),
        )
        self.run_command(
            "cmake", "--build", str(self.build), "--target", "rocm-runtimes"
        )
        self.assertIn(
            "BUILD_TESTING:BOOL=OFF", (self.child / "CMakeCache.txt").read_text()
        )

    def test_source_build_requires_runtime_source(self) -> None:
        (self.runtime / "CMakeLists.txt").unlink()
        result = subprocess.run(
            ["cmake", "-S", str(self.source), "-B", str(self.build), "-GNinja"],
            env=self.environment,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "requires a rocm-systems source revision containing runtimes/",
            " ".join((result.stdout + result.stderr).split()),
        )

    def test_source_build_rejects_invalid_explicit_inputs(self) -> None:
        for name in ("CARGO", "RUSTC", "LLD", "CARGO_CONFIG"):
            with self.subTest(name=name):
                result = subprocess.run(
                    [
                        "cmake",
                        "-S",
                        str(self.source),
                        "-B",
                        str(self.root / name),
                        "-GNinja",
                        f"-DROCM_RUNTIMES_{name}={self.root / 'missing'}",
                    ],
                    env=self.environment,
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    f"ROCM_RUNTIMES_{name} must be an existing absolute file path",
                    result.stdout + result.stderr,
                )

    @unittest.skipUnless(BASELINE_SUPPORTED, "BASELINE requires Linux x86-64")
    def test_production_hardening_coverage(self) -> None:
        diagnostic = (
            "Hardening coverage for subproject rocm-runtimes "
            "(runtime Cargo build including native AddrLib): "
            "deferred (no TheRock policy injection)"
        )
        for policy in ("BASELINE", "OFF"):
            output = self.configure(f"-DTHEROCK_HARDENING_POLICY={policy}")
            if policy == "BASELINE":
                self.assertIn(diagnostic, output)
                init = (self.build / "core/rocm-runtimes/_init.cmake").read_text()
                self.assertIn("set(THEROCK_HARDENING_EXCLUDED FALSE)", init)
                self.assertIn('set(THEROCK_HARDENING_POLICY "BASELINE"', init)
            else:
                self.assertNotIn("Hardening coverage", output)
        self.assertFalse((self.child / "CMakeCache.txt").exists())


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
                        ["cmake", "-S", str(source), "-B", str(build), *options],
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    self.assertEqual((build / "selected.txt").read_text(), expected)


if __name__ == "__main__":
    unittest.main()
