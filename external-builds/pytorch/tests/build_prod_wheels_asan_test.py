# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))

from build_prod_wheels import (
    _setup_common_build_env,
    add_env_compiler_flags,
    validate_build_args,
)
from setup_pytorch_asan import (
    append_env_text,
    configure_triton_asan,
    enable_apex_asan,
    enable_torchaudio_asan,
    enable_torchvision_asan,
)


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

    def test_asan_allows_an_explicit_audio_build(self):
        parser = argparse.ArgumentParser()
        args = self._args(asan=True, build_pytorch_audio=True)

        validate_build_args(parser, args)

        self.assertTrue(args.build_pytorch_audio)
        self.assertFalse(args.build_triton)
        self.assertFalse(args.build_pytorch_vision)
        self.assertFalse(args.build_apex)

    def test_asan_keeps_an_explicit_audio_opt_out(self):
        parser = argparse.ArgumentParser()
        args = self._args(asan=True, build_pytorch_audio=False)

        validate_build_args(parser, args)

        self.assertFalse(args.build_pytorch_audio)

    def test_asan_allows_an_explicit_triton_build(self):
        parser = argparse.ArgumentParser()
        args = self._args(asan=True, build_triton=True)

        validate_build_args(parser, args)

        self.assertTrue(args.build_triton)
        self.assertFalse(args.build_pytorch_audio)
        self.assertFalse(args.build_pytorch_vision)
        self.assertFalse(args.build_apex)

    def test_asan_configures_triton_shared_runtime_once(self):
        import shlex

        env = {
            "TRITON_APPEND_CMAKE_ARGS": (
                "-DCMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH=FALSE"
            )
        }

        configure_triton_asan(env)
        configured = dict(env)
        configure_triton_asan(env)

        self.assertEqual(configured, env)
        self.assertEqual(env["TRITON_BUILD_WITH_ASAN"], "1")
        cmake_args = shlex.split(env["TRITON_APPEND_CMAKE_ARGS"])
        self.assertIn(
            "-DCMAKE_C_FLAGS=-fsanitize=address -fno-omit-frame-pointer",
            cmake_args,
        )
        self.assertIn(
            "-DCMAKE_CXX_FLAGS=-fsanitize=address -fno-omit-frame-pointer",
            cmake_args,
        )
        self.assertIn("-DCMAKE_MODULE_LINKER_FLAGS=-shared-libasan", cmake_args)

    def test_asan_patches_torchaudio_extensions_once(self):
        import tempfile

        helper = """\
import os

def get_ext_modules():
    extra_compile_args = {"cxx": []}
    modules = [
        extension(
            extra_compile_args=extra_compile_args,
        ),
    ]
    return modules
"""
        setup = "import os\nimport torch\n"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "tools" / "setup_helpers" / "extension.py"
            path.parent.mkdir(parents=True)
            path.write_text(helper)
            setup_path = root / "setup.py"
            setup_path.write_text(setup)

            enable_torchaudio_asan(root)
            patched = path.read_text()
            setup_patched = setup_path.read_text()
            enable_torchaudio_asan(root)
            self.assertEqual(patched, path.read_text())
            self.assertEqual(setup_patched, setup_path.read_text())

        self.assertIn("-shared-libasan", patched)
        self.assertIn("extra_link_args=extra_link_args", patched)
        self.assertEqual(patched.count("extra_link_args=extra_link_args"), 1)
        self.assertIn('os.environ.pop("LD_PRELOAD", None)', setup_patched)

    def test_asan_allows_an_explicit_vision_build(self):
        parser = argparse.ArgumentParser()
        args = self._args(asan=True, build_pytorch_vision=True)

        validate_build_args(parser, args)

        self.assertTrue(args.build_pytorch_vision)
        self.assertFalse(args.build_triton)
        self.assertFalse(args.build_pytorch_audio)
        self.assertFalse(args.build_apex)

    def test_asan_patches_torchvision_extensions_once(self):
        import tempfile

        setup = """\
import os
import torch
BUILD_CUDA_SOURCES = (torch.cuda.is_available() and ((CUDA_HOME is not None) or IS_ROCM)) or FORCE_CUDA

def get_macros_and_flags():
    extra_compile_args = {"cxx": []}
    return define_macros, extra_compile_args

def make_extension():
    return Extension(
        extra_compile_args=extra_compile_args,
    )
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_path = root / "setup.py"
            setup_path.write_text(setup)

            enable_torchvision_asan(root)
            patched = setup_path.read_text()
            enable_torchvision_asan(root)
            self.assertEqual(patched, setup_path.read_text())

        self.assertIn("-shared-libasan", patched)
        self.assertIn("extra_link_args=_asan_link_args()", patched)
        self.assertEqual(patched.count("return define_macros, extra_compile_args"), 1)
        self.assertIn('os.environ.pop("LD_PRELOAD", None)', patched)
        self.assertIn("BUILD_CUDA_SOURCES = FORCE_CUDA or", patched)
        self.assertNotIn(
            "is_available() and ((CUDA_HOME is not None) or IS_ROCM)) or FORCE_CUDA",
            patched,
        )

    def test_asan_allows_an_explicit_apex_build(self):
        parser = argparse.ArgumentParser()
        args = self._args(asan=True, build_apex=True)

        validate_build_args(parser, args)

        self.assertTrue(args.build_apex)
        self.assertFalse(args.build_triton)
        self.assertFalse(args.build_pytorch_audio)
        self.assertFalse(args.build_pytorch_vision)

    def test_asan_patches_apex_extensions_once(self):
        import tempfile

        setup = """\
import os
import torch
if not torch.cuda.is_available() and not IS_ROCM_PYTORCH:
    pass
elif not torch.cuda.is_available() and IS_ROCM_PYTORCH:
    pass
"""
        builder = """\
WARNING = f"{YELLOW} [WARNING] {END}"

def jit_load(self):
        nvcc_args = self.strip_empty_entries(self.nvcc_args())
        cxx_args = self.strip_empty_entries(self.cxx_args())
        op_module = load(name=self.name,
                         extra_ldflags=self.strip_empty_entries(self.extra_ldflags()),
                         )

def builder(self):
        return CppExtension(name=self.absolute_name(),
                            extra_compile_args={'cxx': self.strip_empty_entries(self.cxx_args())},
                            extra_link_args=self.strip_empty_entries(self.extra_ldflags()))

        cuda_ext = ExtensionBuilder(name=self.absolute_name(),
                                    sources=self.strip_empty_entries(self.sources()),
                                    include_dirs=include_dirs,
                                    libraries=self.strip_empty_entries(self.libraries_args()),
                                    extra_compile_args=compile_args,
                                    extra_link_args=self.strip_empty_entries(self.extra_ldflags()))
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_path = root / "setup.py"
            setup_path.write_text(setup)
            builder_path = root / "op_builder" / "builder.py"
            builder_path.parent.mkdir(parents=True)
            builder_path.write_text(builder)

            enable_apex_asan(root)
            setup_patched = setup_path.read_text()
            builder_patched = builder_path.read_text()
            enable_apex_asan(root)
            self.assertEqual(setup_patched, setup_path.read_text())
            self.assertEqual(builder_patched, builder_path.read_text())

        self.assertIn("-shared-libasan", builder_patched)
        self.assertIn(
            "cxx_args = self.strip_empty_entries(self.cxx_args()) + _asan_flags()",
            builder_patched,
        )
        self.assertIn(
            "extra_ldflags=self.strip_empty_entries(self.extra_ldflags()) + _asan_flags()",
            builder_patched,
        )
        self.assertEqual(builder_patched.count("def _asan_flags():"), 1)
        self.assertIn('os.environ.pop("LD_PRELOAD", None)', setup_patched)
        self.assertIn(
            'os.environ.get("USE_ASAN") != "1" and not torch.cuda.is_available()',
            setup_patched,
        )
        self.assertNotIn(
            "if not torch.cuda.is_available() and not IS_ROCM_PYTORCH:",
            setup_patched,
        )
        self.assertNotIn(
            "elif not torch.cuda.is_available() and IS_ROCM_PYTORCH:",
            setup_patched,
        )

    def test_asan_allows_all_explicit_companions(self):
        parser = argparse.ArgumentParser()
        args = self._args(
            asan=True,
            build_triton=True,
            build_pytorch_audio=True,
            build_pytorch_vision=True,
            build_apex=True,
        )

        validate_build_args(parser, args)

        self.assertTrue(args.build_triton)
        self.assertTrue(args.build_pytorch_audio)
        self.assertTrue(args.build_pytorch_vision)
        self.assertTrue(args.build_apex)


if __name__ == "__main__":
    unittest.main()
