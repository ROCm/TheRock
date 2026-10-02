# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import unittest
from pathlib import Path

from build_prod_wheels import (
    _append_env_text,
    _setup_common_build_env,
    add_env_compiler_flags,
    validate_build_args,
    with_asan_local_version,
)


class AsanVersionSuffixTest(unittest.TestCase):
    def test_appends_asan_to_the_incoming_rocm_suffix(self):
        self.assertEqual(
            with_asan_local_version("+rocm10.1.0rc3"), "+rocm10.1.0rc3.asan"
        )

    def test_keeps_a_dev_suffix_and_adds_asan(self):
        self.assertEqual(
            with_asan_local_version("+devrocm10.2.0.dev0-abcdef"),
            "+devrocm10.2.0.dev0-abcdef.asan",
        )

    def test_is_idempotent(self):
        self.assertEqual(
            with_asan_local_version("+rocm10.1.0rc3.asan"), "+rocm10.1.0rc3.asan"
        )

    def test_rejects_a_suffix_that_is_not_a_local_version(self):
        with self.assertRaises(RuntimeError):
            with_asan_local_version("10.1.0rc3")


class AsanFlagSpacingTest(unittest.TestCase):
    def test_asan_flags_stay_separate_from_later_appends(self):
        env = {
            "CXXFLAGS": " -Wno-error=restrict ",
            "LDFLAGS": "",
        }
        _append_env_text(env, "CXXFLAGS", "-fno-omit-frame-pointer")
        _append_env_text(env, "LDFLAGS", "-shared-libasan")
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

    def test_normal_linux_build_keeps_gcc_warning_workarounds(self):
        env = _setup_common_build_env(
            Path("/tmp/cmake"),
            Path("/tmp/bin"),
            Path("/tmp/rocm"),
            "gfx942",
            None,
            False,
            asan=False,
        )

        self.assertIn("-Wno-error=maybe-uninitialized", env["CXXFLAGS"])
        self.assertIn("-Wno-error=restrict", env["CPPFLAGS"])


class AsanBuildSelectionTest(unittest.TestCase):
    def test_asan_leaves_companion_projects_off_when_sources_exist(self):
        parser = argparse.ArgumentParser()
        args = argparse.Namespace(
            asan=True,
            build_triton=None,
            build_pytorch_audio=None,
            build_pytorch_vision=None,
            build_apex=None,
            triton_dir=Path("/tmp/triton"),
            pytorch_dir=Path("/tmp/pytorch"),
            pytorch_audio_dir=Path("/tmp/audio"),
            pytorch_vision_dir=Path("/tmp/vision"),
            apex_dir=Path("/tmp/apex"),
            enable_pytorch_flash_attention=None,
        )

        validate_build_args(parser, args)

        self.assertFalse(args.build_triton)
        self.assertFalse(args.build_pytorch_audio)
        self.assertFalse(args.build_pytorch_vision)
        self.assertFalse(args.build_apex)

    def test_without_asan_existing_sources_still_enable_companions(self):
        parser = argparse.ArgumentParser()
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

        validate_build_args(parser, args)

        self.assertTrue(args.build_triton)
        self.assertTrue(args.build_pytorch_audio)
        self.assertTrue(args.build_pytorch_vision)
        self.assertTrue(args.build_apex)


if __name__ == "__main__":
    unittest.main()
