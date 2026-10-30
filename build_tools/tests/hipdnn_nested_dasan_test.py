# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build the hipDNN ExternalProject flag hand-off with an instrumented dependency.

Set DASAN_TEST_CXX to the pinned compiler to exercise real DASan. Otherwise a
GCC adapter maps the sanitizer spelling to ASan for a portable Linux regression.
The generated child build consumes the forwarding block from the source patch.
"""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
PATCH = (
    REPO
    / "patches/amd-mainline/rocm-libraries/0001-hipdnn-preserve-DASan-flags-in-example-provider.patch"
)


@unittest.skipUnless(
    sys.platform.startswith("linux")
    and all(shutil.which(tool) for tool in ("cmake", "ninja", "g++")),
    "Requires Linux CMake/Ninja and a C++ compiler",
)
class HipdnnNestedDasanTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="hipdnn-nested-dasan-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.compiler = os.environ.get("DASAN_TEST_CXX")
        if not self.compiler:
            adapter = self.root / "compiler"
            adapter.write_text(
                "#!/usr/bin/env python3\n"
                "import os,sys\n"
                "args=[a.replace('-fsanitize=daddress','-fsanitize=address') "
                "for a in sys.argv[1:] if a != '-shared-libsan']\n"
                "os.execvp('g++', ['g++', *args])\n"
            )
            adapter.chmod(0o755)
            self.compiler = str(adapter)
        self.extra_flags = os.environ.get("DASAN_TEST_CXX_FLAGS", "")

    def run_command(self, command):
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def build_nested(self, sanitizer="DASAN", shared=False):
        added = [
            line[1:]
            for line in PATCH.read_text().splitlines()
            if line.startswith("+") and not line.startswith("+++")
        ]
        handoff = "        ${EXAMPLE_ENGINE_PLUGIN_DASAN_CMAKE_ARGS}"
        self.assertEqual(added.count(handoff), 1)
        forwarding = "\n".join(line for line in added if line != handoff)
        child = self.root / "child"
        child.mkdir()
        (child / "main.cpp").write_text(
            "extern int dependency(int *);\n"
            "#ifdef KEEP_PARENT_TEXT\n"
            'static_assert(sizeof(KEEP_PARENT_TEXT) == sizeof("two words"));\n'
            "#endif\n"
            "int main() { int value = 7; return dependency(&value) != 7; }\n"
        )
        (child / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.25)\n"
            "project(child LANGUAGES CXX)\n"
            "set(CMAKE_EXPORT_COMPILE_COMMANDS ON)\n"
            "add_executable(nested main.cpp)\n"
            f'target_link_libraries(nested PRIVATE "{self.root}/dependency.a")\n'
        )
        dependency = self.root / "dependency.cpp"
        dependency.write_text("int dependency(int *p) { return *p; }\n")
        flags = shlex.split(self.extra_flags)
        if sanitizer == "DASAN":
            flags += ["-fsanitize=daddress", "-fno-omit-frame-pointer"]
        self.run_command(
            [
                self.compiler,
                *flags,
                "-c",
                str(dependency),
                "-o",
                str(self.root / "dependency.o"),
            ]
        )
        self.run_command(
            [
                "ar",
                "rcs",
                str(self.root / "dependency.a"),
                str(self.root / "dependency.o"),
            ]
        )
        parent_flags = shlex.join(
            flags + ["-DKEEP_PARENT_FLAG=1", '-DKEEP_PARENT_TEXT="two words"']
        )
        if shared:
            parent_flags += " -shared-libsan"
        (self.root / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.25)\n"
            "project(parent LANGUAGES CXX)\n"
            "include(ExternalProject)\n"
            f'set(THEROCK_SANITIZER "{sanitizer}")\n'
            f"set(CMAKE_CXX_FLAGS [==[{parent_flags}]==])\n"
            + forwarding
            + "\nExternalProject_Add(example_engine_plugin\n"
            f' SOURCE_DIR "{child}"\n'
            ' INSTALL_COMMAND ""\n'
            " CMAKE_CACHE_ARGS\n"
            f'  "-DCMAKE_CXX_COMPILER:FILEPATH={self.compiler}"\n'
            '  "-DCMAKE_BUILD_TYPE:STRING=Release"\n'
            '  "-DCMAKE_CXX_FLAGS_RELEASE:STRING=-O1 -DNDEBUG"\n'
            " CMAKE_ARGS\n  ${EXAMPLE_ENGINE_PLUGIN_DASAN_CMAKE_ARGS}\n)\n"
        )
        build = self.root / "build"
        self.run_command(
            ["cmake", "-G", "Ninja", "-S", str(self.root), "-B", str(build)]
        )
        self.run_command(["cmake", "--build", str(build)])
        nested = build / "example_engine_plugin-prefix/src/example_engine_plugin-build"
        commands = json.loads((nested / "compile_commands.json").read_text())
        cache = (nested / "CMakeCache.txt").read_text()
        self.assertIn("CMAKE_CXX_FLAGS_RELEASE:STRING=-O1 -DNDEBUG", cache)
        return commands[0]["command"], cache, nested / "nested"

    def test_nested_instrumented_dependency_links_and_runs(self):
        command, cache, executable = self.build_nested()
        self.assertIn("-fsanitize=daddress", command)
        self.assertIn("-DKEEP_PARENT_FLAG=1", command)
        self.assertIn("-shared-libsan", command)
        self.run_command([str(executable)])

    def test_existing_shared_flag_is_not_duplicated(self):
        command, cache, executable = self.build_nested(shared=True)
        self.assertEqual(command.count("-shared-libsan"), 1)
        self.run_command([str(executable)])

    def test_non_dasan_configuration_keeps_standalone_flags(self):
        command, cache, executable = self.build_nested(sanitizer="OFF")
        self.assertNotIn("-fsanitize=", command)
        self.assertNotIn("-shared-libsan", command)
        self.assertNotIn("KEEP_PARENT_FLAG", command)
        self.run_command([str(executable)])


if __name__ == "__main__":
    unittest.main()
