#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Load modules whose sanitizer dependency is invisible to CMake's link graph.

The small runtime is deliberately linked with raw linker flags, as compiler
sanitizer runtimes are. Successful dlopen before installation and after moving
the install tree tests both the build RPATH and relocatable install RPATH.
"""

import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


THEROCK_ROOT = Path(__file__).resolve().parents[2]
GLOBAL_POST_HOOK = THEROCK_ROOT / "cmake/therock_global_post_subproject.cmake"
RESOURCE_LIB_DIR = "lib/llvm/lib/clang/24/lib/x86_64-unknown-linux-gnu"
NESTED_MODULE_DIR = "lib/hipblaslt/rocisa"
COMPILER = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


def run_command(*args: str) -> str:
    # Neither the developer's compiler flags nor the agent's preloaded sanitizer
    # may provide the dependency this fixture is supposed to resolve itself.
    environment = {
        name: value
        for name, value in os.environ.items()
        if name
        not in (
            "CFLAGS",
            "CXXFLAGS",
            "CPPFLAGS",
            "LDFLAGS",
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
            "LD_AUDIT",
        )
    }
    result = subprocess.run(
        args,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode:
        raise AssertionError(f"Command failed: {' '.join(args)}\n{result.stdout}")
    return result.stdout


def write_fixture(source_dir: Path) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "runtime.c").write_text(
        "int sanitizer_runtime_value(void) { return 37; }\n", encoding="utf-8"
    )
    (source_dir / "module.c").write_text(
        "extern int sanitizer_runtime_value(void);\n"
        "int module_value(void) { return sanitizer_runtime_value(); }\n",
        encoding="utf-8",
    )
    (source_dir / "CMakeLists.txt").write_text(
        textwrap.dedent(
            f"""
            cmake_minimum_required(VERSION 3.25)
            project(therock_module_rpath_fixture C)
            set(THEROCK_SOURCE_DIR "{THEROCK_ROOT.as_posix()}")
            set(THEROCK_BINARY_DIR "${{CMAKE_BINARY_DIR}}")
            set(THEROCK_SUBPROJECT_TARGET fixture)
            set(THEROCK_STAGE_STAMP_FILE "${{CMAKE_BINARY_DIR}}/stage.stamp")
            set(THEROCK_BUILD_STAMP_FILE "${{CMAKE_BINARY_DIR}}/build.stamp")
            set(THEROCK_HARDENING_POLICY OFF)
            set(THEROCK_SPLIT_DEBUG_INFO OFF)
            set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
            set(THEROCK_INSTALL_RPATH_EXECUTABLE_DIR bin)
            set(THEROCK_INSTALL_RPATH_LIBRARY_DIR lib)
            set(THEROCK_PRIVATE_BUILD_RPATH_DIRS "${{CMAKE_BINARY_DIR}}/runtime-only")
            set(THEROCK_PRIVATE_INSTALL_RPATH_DIRS "{RESOURCE_LIB_DIR}")
            set(THEROCK_INCLUDE_CLANG_RESOURCE_DIR_RPATH "${{SANITIZER_MODULE_RPATH}}")

            add_library(fake_runtime SHARED runtime.c)
            set_target_properties(fake_runtime PROPERTIES
              OUTPUT_NAME clang_rt.csan
              LIBRARY_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}/runtime-only")

            foreach(_name nested default)
              add_library(module_${{_name}} MODULE module.c)
              add_dependencies(module_${{_name}} fake_runtime)
              # A raw -l flag does not teach CMake the runtime's build directory.
              # --no-as-needed keeps its DT_NEEDED even though link options
              # precede the module's object file on the linker command line.
              target_link_options(module_${{_name}} PRIVATE
                -Wl,--no-as-needed "-L${{CMAKE_BINARY_DIR}}/runtime-only"
                -lclang_rt.csan -Wl,--as-needed)
              set_target_properties(module_${{_name}} PROPERTIES
                PREFIX ""
                LIBRARY_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}/modules")
              if(SANITIZER_MODULE_RPATH)
                set_target_properties(module_${{_name}} PROPERTIES
                  BUILD_RPATH "$ORIGIN/custom-build"
                  INSTALL_RPATH "$ORIGIN/custom-install")
              else()
                set_target_properties(module_${{_name}} PROPERTIES
                  BUILD_RPATH "${{CMAKE_BINARY_DIR}}/runtime-only")
              endif()
            endforeach()

            set_property(TARGET module_nested PROPERTY
              THEROCK_INSTALL_RPATH_ORIGIN "{NESTED_MODULE_DIR}")
            if(SANITIZER_MODULE_RPATH)
              install(TARGETS fake_runtime LIBRARY DESTINATION "{RESOURCE_LIB_DIR}")
            else()
              # These modules own a different RPATH convention, which the
              # sanitizer-only fix must leave alone for ordinary builds.
              set_property(TARGET module_nested PROPERTY INSTALL_RPATH "$ORIGIN/private")
              set_property(TARGET module_default PROPERTY INSTALL_RPATH
                "$ORIGIN/hipblaslt/rocisa/private")
              install(TARGETS fake_runtime LIBRARY DESTINATION "{NESTED_MODULE_DIR}/private")
            endif()
            install(TARGETS module_nested LIBRARY DESTINATION "{NESTED_MODULE_DIR}")
            install(TARGETS module_default LIBRARY DESTINATION lib)

            include("{GLOBAL_POST_HOOK.as_posix()}")
            file(GENERATE OUTPUT "${{CMAKE_BINARY_DIR}}/module-rpaths.txt" CONTENT
              "nested_build=$<TARGET_PROPERTY:module_nested,BUILD_RPATH>\nnested_install=$<TARGET_PROPERTY:module_nested,INSTALL_RPATH>\ndefault_build=$<TARGET_PROPERTY:module_default,BUILD_RPATH>\ndefault_install=$<TARGET_PROPERTY:module_default,INSTALL_RPATH>\n")
            """
        ),
        encoding="utf-8",
    )


@unittest.skipUnless(
    platform.system() == "Linux"
    and COMPILER
    and shutil.which("cmake")
    and shutil.which("ninja")
    and shutil.which("readelf"),
    "ELF module RPATH integration requires Linux, a C compiler, CMake, Ninja, and readelf",
)
class ModuleRpathTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.build = self.root / "build"
        self.stage = self.root / "stage"
        write_fixture(self.source)

    def configure_and_build(self, sanitizer: bool) -> dict[str, str]:
        run_command(
            "cmake",
            "-S",
            str(self.source),
            "-B",
            str(self.build),
            "-GNinja",
            f"-DCMAKE_C_COMPILER={COMPILER}",
            f"-DCMAKE_INSTALL_PREFIX={self.stage}",
            f"-DSANITIZER_MODULE_RPATH={'ON' if sanitizer else 'OFF'}",
        )
        run_command("cmake", "--build", str(self.build))
        return dict(
            line.split("=", 1)
            for line in (self.build / "module-rpaths.txt").read_text().splitlines()
        )

    def assert_module_loads(self, path: Path) -> None:
        run_command(
            sys.executable,
            "-c",
            "import ctypes, sys; module = ctypes.CDLL(sys.argv[1]); "
            "module.module_value.restype = ctypes.c_int; "
            "assert module.module_value() == 37",
            str(path),
        )

    def install_and_relocate(self) -> Path:
        run_command("cmake", "--install", str(self.build))
        relocated = self.root / "relocated"
        self.stage.rename(relocated)
        # An accidental absolute build-tree RPATH must not make the installed
        # probes pass after relocation.
        (self.build / "runtime-only").rename(self.root / "hidden-build-runtime")
        return relocated

    def test_sanitizer_modules_load_in_build_and_relocated_install_trees(self):
        properties = self.configure_and_build(sanitizer=True)
        for name in ("nested", "default"):
            with self.subTest(module=name):
                self.assert_module_loads(self.build / "modules" / f"module_{name}.so")
                self.assertIn(
                    str(self.build / "runtime-only"),
                    properties[f"{name}_build"].split(";"),
                )
        self.assertEqual(
            properties["nested_install"],
            "$ORIGIN/../../llvm/lib/clang/24/lib/x86_64-unknown-linux-gnu",
        )
        self.assertEqual(
            properties["default_install"],
            "$ORIGIN/llvm/lib/clang/24/lib/x86_64-unknown-linux-gnu",
        )
        relocated = self.install_and_relocate()
        for directory, name in ((NESTED_MODULE_DIR, "nested"), ("lib", "default")):
            with self.subTest(installed_module=name):
                module = relocated / directory / f"module_{name}.so"
                dynamic = run_command("readelf", "-d", str(module))
                self.assertIn("libclang_rt.csan.so", dynamic)
                self.assertIn("$ORIGIN/", dynamic)
                self.assertNotIn(str(self.build), dynamic)
                self.assert_module_loads(module)

    def test_ordinary_modules_keep_custom_build_and_install_rpaths(self):
        properties = self.configure_and_build(sanitizer=False)
        self.assertEqual(properties["nested_build"], str(self.build / "runtime-only"))
        self.assertEqual(properties["default_build"], str(self.build / "runtime-only"))
        self.assertEqual(properties["nested_install"], "$ORIGIN/private")
        self.assertEqual(
            properties["default_install"], "$ORIGIN/hipblaslt/rocisa/private"
        )
        for name in ("nested", "default"):
            self.assert_module_loads(self.build / "modules" / f"module_{name}.so")
        relocated = self.install_and_relocate()
        self.assert_module_loads(relocated / NESTED_MODULE_DIR / "module_nested.so")
        self.assert_module_loads(relocated / "lib/module_default.so")


if __name__ == "__main__":
    unittest.main()
