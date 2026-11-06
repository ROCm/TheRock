#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for build-directory-independent subproject fingerprints.

These tests use the real therock_subproject.cmake with miniature superprojects,
following the other tests in cmake/tests. No ROCm submodules, GPU, or compiler
are needed. CMake and Ninja must be on PATH. Fetch tests use a local archive.

The fingerprints come from THEROCK_FPRINT after declaration and activation;
the tests do not reimplement hashing or depend on a particular digest value.
"""

import hashlib
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import unittest
from pathlib import Path

THEROCK_ROOT = Path(__file__).resolve().parents[2]
HARNESS_INCLUDES = (
    "therock_globals",
    "therock_sanitizers",
    "therock_flag_utils",
    "therock_default_targets",
    "therock_subproject",
)


def write_file(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(contents), encoding="utf-8")


def run(*args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise AssertionError(" ".join(args) + "\n" + result.stdout)
    return result


def write_harness(source_dir: Path, declarations: str) -> None:
    includes = "\n".join(f"include({name})" for name in HARNESS_INCLUDES)
    write_file(
        source_dir / "CMakeLists.txt",
        f"""
        cmake_minimum_required(VERSION 3.25)
        project(therock_subproject_fingerprint_harness NONE)

        set(THEROCK_SOURCE_DIR "{THEROCK_ROOT.as_posix()}")
        set(THEROCK_BINARY_DIR "${{CMAKE_BINARY_DIR}}")
        list(APPEND CMAKE_MODULE_PATH "${{THEROCK_SOURCE_DIR}}/cmake")
        find_package(Python3 COMPONENTS Interpreter REQUIRED)

        set(ROCM_BUILD_FLAGS_STATE_FILE "${{CMAKE_BINARY_DIR}}/rocm_build_flags_state.cmake")
        file(WRITE "${{ROCM_BUILD_FLAGS_STATE_FILE}}" "# no flags\\n")

        {includes}

        # Export the actual target property, not a second implementation of it.
        function(record_fingerprint target_name)
          get_target_property(fingerprint "${{target_name}}" THEROCK_FPRINT)
          file(WRITE "${{CMAKE_BINARY_DIR}}/${{target_name}}.fingerprint" "${{fingerprint}}\\n")
        endfunction()

        {textwrap.dedent(declarations)}
        """,
    )


def write_projects(source_dir: Path) -> None:
    """A dependency and consumer exercise the generated path bindings."""
    for name in ("dep", "main"):
        write_file(source_dir / name / "payload.txt", f"{name}\n")
        write_file(
            source_dir / name / "CMakeLists.txt",
            f"""
            cmake_minimum_required(VERSION 3.25)
            project({name}_project NONE)
            install(FILES "${{CMAKE_CURRENT_SOURCE_DIR}}/payload.txt"
              DESTINATION lib RENAME {name}.txt)
            """,
        )
    write_harness(
        source_dir,
        """
        therock_cmake_subproject_declare(dep_project
          EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dep"
          BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/dep"
          FPRINT_SOURCE_HASH
          DISABLE_AMDGPU_TARGETS
          INTERFACE_INCLUDE_DIRS include include/extra ${TEST_EXTERNAL_INCLUDE}
          INTERFACE_LINK_DIRS lib lib/extra
          INTERFACE_PROGRAM_DIRS bin
          INTERFACE_PKG_CONFIG_DIRS lib/pkgconfig
        )
        therock_cmake_subproject_provide_package(dep_project
          FingerprintDep lib/cmake/FingerprintDep)
        therock_cmake_subproject_activate(dep_project)
        record_fingerprint(dep_project)

        # These are configuration-only stand-ins. Nothing builds LLVM or HIP.
        if(TEST_TOOLCHAIN)
          foreach(name IN ITEMS amd-llvm hip-clr)
            therock_cmake_subproject_declare(${name}
              EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dep"
              BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/${name}"
              FPRINT_SOURCE_HASH
              DISABLE_AMDGPU_TARGETS
            )
            therock_cmake_subproject_activate(${name})
            record_fingerprint(${name})
          endforeach()
        endif()

        therock_cmake_subproject_declare(main_project
          EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/main"
          BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/main"
          FPRINT_SOURCE_HASH
          DISABLE_AMDGPU_TARGETS
          RUNTIME_DEPS dep_project
          COMPILER_TOOLCHAIN "${TEST_TOOLCHAIN}"
        )
        therock_cmake_subproject_activate(main_project)
        record_fingerprint(main_project)
        """,
    )


def write_fetch_project(source_dir: Path) -> None:
    """Exercise ExternalProject metadata without downloading from a server."""
    write_file(
        source_dir / "payload" / "CMakeLists.txt",
        """
        cmake_minimum_required(VERSION 3.25)
        project(fetched_project NONE)
        """,
    )
    archive = source_dir / "payload.tar"
    with tarfile.open(archive, "w") as output:
        output.add(source_dir / "payload" / "CMakeLists.txt", arcname="CMakeLists.txt")
    write_harness(
        source_dir,
        """
        set(_fetched_source "${CMAKE_CURRENT_BINARY_DIR}/downloaded")
        file(MAKE_DIRECTORY "${_fetched_source}")
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/touch")
        file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/payload.tar" _archive_hash)
        therock_subproject_fetch(fetched_sources
          URL "${CMAKE_CURRENT_SOURCE_DIR}/payload.tar"
          URL_HASH "SHA256=${_archive_hash}"
          PATCH_COMMAND "${CMAKE_COMMAND}" -E echo "${TEST_FETCH_OPTION}"
          SOURCE_DIR "${_fetched_source}"
          PREFIX "${CMAKE_CURRENT_BINARY_DIR}/fetch"
          CMAKE_PROJECT
          TOUCH "${CMAKE_CURRENT_BINARY_DIR}/touch/completed"
        )
        therock_cmake_subproject_declare(fetched_project
          EXTERNAL_SOURCE_DIR "${_fetched_source}"
          BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/fetched"
          DISABLE_AMDGPU_TARGETS
        )
        therock_cmake_subproject_activate(fetched_project)
        record_fingerprint(fetched_project)
        """,
    )


class SubprojectFingerprintTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="therock-fingerprint-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "source"

    def configure(self, build_dir: Path, *args: str) -> dict[str, str]:
        run(
            "cmake",
            "-S",
            str(self.source),
            "-B",
            str(build_dir),
            "-GNinja",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DPython3_EXECUTABLE={Path(sys.executable).as_posix()}",
            *args,
        )
        result = {}
        for path in sorted(build_dir.glob("*.fingerprint")):
            value = path.read_text(encoding="utf-8").strip()
            self.assertRegex(value, r"\A[0-9a-f]{64}\Z", str(path))
            result[path.stem] = value
        self.assertTrue(result, "The fixture did not produce any fingerprints")
        return result

    def test_different_build_directories_keep_fingerprints_and_real_paths(self):
        write_projects(self.source)
        first = self.root / "build a"
        second = self.root / "build b"
        fingerprints = self.configure(first)
        self.assertEqual(set(fingerprints), {"dep_project", "main_project"})
        self.assertEqual(fingerprints, self.configure(second))
        self.assertEqual(fingerprints, self.configure(first))

        for build_dir in (first, second):
            with self.subTest(build_dir=build_dir):
                init = (build_dir / "main" / "_init.cmake").read_text(encoding="utf-8")
                # These values must remain executable absolute paths on disk.
                for relative in (
                    "main/stamp/build.stamp",
                    "main/stamp/stage.stamp",
                    "rocm_build_flags_state.cmake",
                    "main/prefix",
                    "dep/stage/include",
                    "dep/stage/include/extra",
                    "dep/stage/lib",
                    "dep/stage/lib/extra",
                    "dep/dist/bin",
                    "dep/stage/lib/pkgconfig",
                    "dep/dist/lib/cmake/FingerprintDep",
                ):
                    self.assertIn(f'"{(build_dir / relative).as_posix()}"', init)
                self.assertNotIn("BUILD:", init)

    def test_build_type_still_changes_fingerprints(self):
        write_projects(self.source)
        build_dir = self.root / "build"
        release = self.configure(build_dir)
        debug = self.configure(build_dir, "-DCMAKE_BUILD_TYPE=Debug")
        for name in release:
            with self.subTest(name=name):
                self.assertNotEqual(release[name], debug[name])

    def test_opaque_flags_are_not_path_normalized(self):
        write_projects(self.source)
        first = self.root / "build-a"
        second = self.root / "build-b"
        results = []
        for build_dir in (first, second):
            # This flag embeds a build-local path, but is not a typed path input.
            flags = f"-I{(build_dir / 'opaque-include').as_posix()}"
            results.append(self.configure(build_dir, f"-DCMAKE_CXX_FLAGS={flags}"))
            toolchain = (build_dir / "main" / "_toolchain.cmake").read_text(
                encoding="utf-8"
            )
            self.assertIn(flags, toolchain)
        self.assertNotEqual(results[0]["main_project"], results[1]["main_project"])

    def test_external_include_path_still_changes_consumer_fingerprint(self):
        write_projects(self.source)
        build_dir = self.root / "build"
        results = []
        for name in ("build-sibling-a", "build-sibling-b"):
            external = (self.root / name / "include").as_posix()
            results.append(
                self.configure(build_dir, f"-DTEST_EXTERNAL_INCLUDE={external}")
            )
            init = (build_dir / "main" / "_init.cmake").read_text(encoding="utf-8")
            self.assertIn(f'"{external}"', init)
        self.assertNotEqual(results[0]["main_project"], results[1]["main_project"])

    def test_build_produced_toolchain_paths_are_build_independent(self):
        write_projects(self.source)
        for toolchain, provider in (("amd-llvm", "amd-llvm"), ("amd-hip", "hip-clr")):
            with self.subTest(toolchain=toolchain):
                first = self.root / toolchain / "build-a"
                second = self.root / toolchain / "build-b"
                args = (f"-DTEST_TOOLCHAIN={toolchain}",)
                self.assertEqual(
                    self.configure(first, *args), self.configure(second, *args)
                )
                for build_dir in (first, second):
                    text = (build_dir / "main" / "_toolchain.cmake").read_text(
                        encoding="utf-8"
                    )
                    dist = (build_dir / provider / "dist").as_posix()
                    for executable in ("clang", "clang++", "lld"):
                        self.assertIn(f"{dist}/lib/llvm/bin/{executable}", text)
                    if toolchain == "amd-hip":
                        self.assertIn(f"--hip-path={dist}", text)
                        self.assertIn(
                            f"--hip-device-lib-path={dist}/lib/llvm/amdgcn/bitcode",
                            text,
                        )
                    self.assertNotIn("BUILD:", text)

    def test_fetch_metadata_and_source_location_are_build_independent(self):
        write_fetch_project(self.source)
        first = self.root / "build-a"
        second = self.root / "build-b"
        self.assertEqual(self.configure(first), self.configure(second))
        metadata = (first / ".downloaded.smrev").read_text(encoding="utf-8")
        self.assertEqual(
            metadata, (second / ".downloaded.smrev").read_text(encoding="utf-8")
        )
        self.assertIn((self.source / "payload.tar").as_posix(), metadata)
        self.assertIn(
            hashlib.sha256((self.source / "payload.tar").read_bytes()).hexdigest(),
            metadata,
        )

        # Exercise the real touch commands: logical identities must never be used
        # as filesystem paths. The only fetch source is the local fixture archive.
        for build_dir in (first, second):
            run("cmake", "--build", str(build_dir), "--target", "fetched_sources")
            self.assertTrue((build_dir / "touch" / "completed").is_file())
            self.assertEqual(
                (build_dir / "downloaded" / "CMakeLists.txt").read_bytes(),
                (self.source / "payload" / "CMakeLists.txt").read_bytes(),
            )

    def test_caller_supplied_fetch_options_are_preserved(self):
        write_fetch_project(self.source)
        fingerprints = []
        for name in ("build-a", "build-b"):
            build_dir = self.root / name
            option = (build_dir / "caller-option").as_posix()
            fingerprints.append(
                self.configure(build_dir, f"-DTEST_FETCH_OPTION={option}")
            )
            metadata = (build_dir / ".downloaded.smrev").read_text(encoding="utf-8")
            self.assertIn(option, metadata)
        self.assertNotEqual(
            fingerprints[0]["fetched_project"], fingerprints[1]["fetched_project"]
        )


if __name__ == "__main__":
    unittest.main()
