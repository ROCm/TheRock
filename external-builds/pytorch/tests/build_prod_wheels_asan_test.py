# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))

from build_prod_wheels import (
    _setup_common_build_env,
    add_env_compiler_flags,
    validate_build_args,
)
from setup_pytorch_asan import append_env_text, suppress_benchmark_c2y_warning


class AsanFlagSpacingTest(unittest.TestCase):
    def test_asan_flags_stay_separate_from_later_appends(self):
        env = {
            "CXXFLAGS": " -Wno-error=restrict ",
            "LDFLAGS": "",
        }
        append_env_text(env, "CXXFLAGS", "-fno-omit-frame-pointer")
        append_env_text(env, "LDFLAGS", "-shared-libasan")
        add_env_compiler_flags(
            env, "CXXFLAGS", "-I/opt/rocm/include", "-I/opt/rocm/include/roctracer"
        )
        add_env_compiler_flags(env, "LDFLAGS", "-L/opt/rocm/lib")

        self.assertIn(
            "-fno-omit-frame-pointer -I/opt/rocm/include -I/opt/rocm/include/roctracer",
            env["CXXFLAGS"],
        )
        self.assertIn("-shared-libasan -L/opt/rocm/lib", env["LDFLAGS"])
        self.assertNotIn("-fno-omit-frame-pointer-I", env["CXXFLAGS"])
        self.assertNotIn("-shared-libasan-L", env["LDFLAGS"])


class AsanCompilerFlagsTest(unittest.TestCase):
    def test_asan_does_not_inherit_gcc_warning_workarounds(self):
        env = _setup_common_build_env(
            Path("/tmp/cmake"),
            Path("/tmp/bin"),
            Path("/tmp/rocm"),
            "gfx942",
            None,
            False,
            asan=True,
        )

        for name in ("CXXFLAGS", "CPPFLAGS"):
            self.assertNotIn("maybe-uninitialized", env.get(name, ""))
            self.assertNotIn("restrict", env.get(name, ""))
        self.assertTrue(
            env["PATH"].startswith("/tmp/rocm/lib/llvm/bin" + os.path.pathsep)
        )

    def test_normal_linux_build_keeps_gcc_warning_workarounds(self):
        env = _setup_common_build_env(
            Path("/tmp/cmake"),
            Path("/tmp/bin"),
            Path("/tmp/rocm"),
            "gfx942",
            None,
            False,
        )

        self.assertIn("-Wno-error=maybe-uninitialized", env["CXXFLAGS"])
        self.assertIn("-Wno-error=restrict", env["CPPFLAGS"])
        self.assertFalse(
            "/tmp/rocm/lib/llvm/bin" in env["PATH"].split(os.path.pathsep)[0]
        )


class AsanBenchmarkWarningTest(unittest.TestCase):
    _OLDER_RELEASE = """\
  if(NOT USE_SYSTEM_BENCHMARK)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/benchmark)
  else()
    add_library(benchmark SHARED IMPORTED)
  endif()
"""

    def test_older_release_gains_the_2_14_suppression(self):
        with tempfile.TemporaryDirectory() as tmp:
            pytorch_dir = Path(tmp)
            path = pytorch_dir / "cmake" / "Dependencies.cmake"
            path.parent.mkdir()
            path.write_text(self._OLDER_RELEASE)

            suppress_benchmark_c2y_warning(pytorch_dir)
            patched = path.read_text()
            suppress_benchmark_c2y_warning(pytorch_dir)
            self.assertEqual(path.read_text(), patched)

        self.assertIn(
            "target_compile_options(benchmark PRIVATE -Wno-c2y-extensions)", patched
        )

    def test_release_2_14_is_left_unchanged(self):
        original = self._OLDER_RELEASE.replace(
            "    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/benchmark)\n",
            "    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/benchmark)\n"
            "    target_compile_options(benchmark PRIVATE -Wno-c2y-extensions)\n",
        )
        with tempfile.TemporaryDirectory() as tmp:
            pytorch_dir = Path(tmp)
            path = pytorch_dir / "cmake" / "Dependencies.cmake"
            path.parent.mkdir()
            path.write_text(original)

            suppress_benchmark_c2y_warning(pytorch_dir)
            self.assertEqual(path.read_text(), original)


class AsanBuildSelectionTest(unittest.TestCase):
    def _args(self, **overrides):
        source = Path(__file__).resolve().parent
        args = argparse.Namespace(
            asan=False,
            build_triton=None,
            build_pytorch_audio=None,
            build_pytorch_vision=None,
            build_apex=None,
            triton_dir=source,
            pytorch_dir=source,
            pytorch_audio_dir=source,
            pytorch_vision_dir=source,
            apex_dir=source,
            enable_pytorch_flash_attention=None,
        )
        for name, value in overrides.items():
            setattr(args, name, value)
        return args

    def test_without_asan_existing_sources_still_enable_companions(self):
        parser = argparse.ArgumentParser()
        args = self._args()

        validate_build_args(parser, args)

        self.assertTrue(args.build_triton)
        self.assertTrue(args.build_pytorch_audio)
        self.assertTrue(args.build_pytorch_vision)
        self.assertTrue(args.build_apex)

    def test_asan_leaves_companions_off_when_sources_exist(self):
        parser = argparse.ArgumentParser()
        args = self._args(asan=True)

        validate_build_args(parser, args)

        self.assertFalse(args.build_triton)
        self.assertFalse(args.build_pytorch_audio)
        self.assertFalse(args.build_pytorch_vision)
        self.assertFalse(args.build_apex)

    def test_asan_rejects_an_explicit_companion_build(self):
        parser = argparse.ArgumentParser()
        args = self._args(asan=True, build_pytorch_audio=True)

        with self.assertRaises(SystemExit):
            validate_build_args(parser, args)


if __name__ == "__main__":
    unittest.main()
