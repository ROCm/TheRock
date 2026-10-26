# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))

from build_prod_wheels import validate_build_args
from setup_pytorch_sanitizer import SANITIZERS, import_sanity_env


class ImportSanityEnvTest(unittest.TestCase):
    """The import check preloads the ASan runtime, then ROCr.

    The directories below are test fixtures. They are not an install prefix.
    """

    def _fixture_env(self, root: Path, *, include_hsa: bool) -> dict[str, str]:
        env = {
            "ASAN_RUNTIME_PATH": os.fspath(root / "clang" / "libclang_rt.asan.so"),
            "LD_LIBRARY_PATH": os.fspath(root / "lib"),
        }
        if include_hsa:
            env["HSA_RUNTIME_PATH"] = os.fspath(root / "lib" / "libhsa-runtime64.so.1")
        return env

    def test_preloads_asan_runtime_then_hsa(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = self._fixture_env(Path(tmp), include_hsa=True)

            sanity = import_sanity_env(env, SANITIZERS["asan"])

        self.assertEqual(
            sanity["LD_PRELOAD"],
            os.pathsep.join((env["ASAN_RUNTIME_PATH"], env["HSA_RUNTIME_PATH"])),
        )
        self.assertIn("detect_leaks=0", sanity["ASAN_OPTIONS"])
        self.assertEqual(sanity["LD_LIBRARY_PATH"], env["LD_LIBRARY_PATH"])

    def test_missing_hsa_runtime_fails_the_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = self._fixture_env(Path(tmp), include_hsa=False)

            with self.assertRaises(RuntimeError):
                import_sanity_env(env, SANITIZERS["asan"])


class AsanBuildSelectionTest(unittest.TestCase):
    def _args(self, **overrides):
        source = Path(__file__).resolve().parent
        args = argparse.Namespace(
            build_variant="release",
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
        args = self._args(build_variant="asan")

        validate_build_args(parser, args)

        self.assertFalse(args.build_triton)
        self.assertFalse(args.build_pytorch_audio)
        self.assertFalse(args.build_pytorch_vision)
        self.assertFalse(args.build_apex)

    def test_asan_rejects_an_explicit_companion_build(self):
        parser = argparse.ArgumentParser()
        args = self._args(build_variant="asan", build_pytorch_audio=True)

        with self.assertRaises(SystemExit):
            validate_build_args(parser, args)


if __name__ == "__main__":
    unittest.main()
