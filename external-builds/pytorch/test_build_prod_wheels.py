# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent))

import build_prod_wheels as bpw


class AsanVersionTest(unittest.TestCase):
    def test_rocm_10_1_asan_suffix_is_unique(self):
        self.assertEqual(
            bpw.get_asan_version_suffix("10.1.0+asan.20260807"),
            "+rocm10.1.asan.20260807",
        )

    def test_release_sdk_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "uniquely labelled"):
            bpw.validate_asan_rocm_version("10.1.0")

    def test_old_branch_version_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "ROCm 10.1"):
            bpw.validate_asan_rocm_version("7.15.0+asan.20260807")

    def test_conflicting_explicit_suffix_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "collide"):
            bpw.resolve_asan_version_suffix(
                "10.1.0+asan.20260807", "+rocm10.1"
            )


class LocalAsanIndexTest(unittest.TestCase):
    def _write_manifest(self, root: Path, *, version="10.1.0+asan.20260807"):
        index = root / "whl-asan" / "gfx942-all"
        index.mkdir(parents=True)
        packages = []
        for project in sorted(bpw.ASAN_REQUIRED_LOCAL_PACKAGES):
            filename = f"{project}-{version}.pkg"
            (index / filename).touch()
            packages.append(
                {
                    "normalized_project": project,
                    "version": version,
                    "filename": filename,
                    "size": 0,
                }
            )
        (index / "index-manifest.json").write_text(
            json.dumps(
                {
                    "index_kind": "local-only",
                    "relative_path": "whl-asan/gfx942-all",
                    "packages": packages,
                }
            )
        )
        (index / "index.html").touch()
        return index

    def test_accepts_phase1_directory_or_index_page(self):
        with tempfile.TemporaryDirectory() as td:
            index = self._write_manifest(Path(td))
            expected = "10.1.0+asan.20260807"
            self.assertEqual(bpw.validate_local_asan_index(str(index)), expected)
            self.assertEqual(
                bpw.validate_local_asan_index(str(index / "index.html")), expected
            )

    def test_rejects_incomplete_package_set(self):
        with tempfile.TemporaryDirectory() as td:
            index = self._write_manifest(Path(td))
            manifest_path = index / "index-manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["packages"].pop()
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "missing required packages"):
                bpw.validate_local_asan_index(str(index))

    def test_build_validation_pins_local_version_and_defaults_arch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            index = self._write_manifest(root)
            pytorch_dir = root / "pytorch"
            pytorch_dir.mkdir()
            args = self._build_args(pytorch_dir, index)
            parser = argparse.ArgumentParser()

            bpw.validate_build_args(parser, args)

            self.assertEqual(args.pytorch_rocm_arch, "gfx942:xnack+")
            self.assertEqual(args.rocm_sdk_version, "==10.1.0+asan.20260807")
            self.assertEqual(args.asan_index_version, "10.1.0+asan.20260807")

    def test_build_validation_rejects_remote_index(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            index = self._write_manifest(root)
            pytorch_dir = root / "pytorch"
            pytorch_dir.mkdir()
            args = self._build_args(pytorch_dir, index)
            args.index_url = "https://example.invalid/simple"
            with self.assertRaises(SystemExit):
                bpw.validate_build_args(argparse.ArgumentParser(), args)

    @staticmethod
    def _build_args(pytorch_dir: Path, index: Path):
        return argparse.Namespace(
            asan=True,
            pytorch_dir=pytorch_dir,
            triton_dir=None,
            pytorch_audio_dir=None,
            pytorch_vision_dir=None,
            apex_dir=None,
            build_triton=False,
            build_pytorch_audio=False,
            build_pytorch_vision=False,
            build_apex=False,
            enable_pytorch_flash_attention=None,
            rocm_extras="",
            pytorch_rocm_arch=None,
            index_url=None,
            install_rocm=True,
            find_links=str(index),
            rocm_sdk_version=">1.0",
        )


class AsanEnvironmentTest(unittest.TestCase):
    def _make_sdk(self, root: Path):
        llvm_bin = root / "lib" / "llvm" / "bin"
        llvm_bin.mkdir(parents=True)
        for name in ("clang", "clang++"):
            compiler = llvm_bin / name
            compiler.touch()
            compiler.chmod(0o755)
        bitcode = root / "lib" / "llvm" / "amdgcn" / "bitcode"
        bitcode.mkdir(parents=True)
        (bitcode / "ocml.bc").touch()
        runtime = (
            root
            / "lib"
            / "llvm"
            / "lib"
            / "clang"
            / "23"
            / "lib"
            / "linux"
            / "libclang_rt.asan-x86_64.so"
        )
        runtime.parent.mkdir(parents=True)
        runtime.touch()
        return runtime

    def test_asan_env_uses_rocm_clang_and_shared_runtime(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            runtime = self._make_sdk(root)
            with mock.patch.object(bpw, "capture", return_value=str(runtime)):
                env = bpw._setup_asan_build_env(root, "gfx942:xnack+")

            self.assertEqual(env["USE_ASAN"], "1")
            self.assertEqual(env["USE_ROCM"], "1")
            self.assertEqual(env["USE_CUDA"], "0")
            self.assertEqual(env["CC"], str(root / "lib/llvm/bin/clang"))
            self.assertEqual(env["CXX"], str(root / "lib/llvm/bin/clang++"))
            self.assertEqual(env["CMAKE_C_COMPILER"], env["CC"])
            self.assertEqual(env["CMAKE_CXX_COMPILER"], env["CXX"])
            self.assertEqual(env["PYTORCH_ROCM_ARCH"], "gfx942:xnack+")
            self.assertIn("-fno-omit-frame-pointer", env["CXXFLAGS"])
            self.assertNotIn("maybe-uninitialized", env["CXXFLAGS"])
            self.assertIn("-shared-libasan", env["LDFLAGS"])
            self.assertTrue(env["LD_LIBRARY_PATH"].startswith(str(runtime.parent)))
            self.assertEqual(
                env["CMAKE_ARGS"],
                "-DCMAKE_CXX_SCAN_FOR_MODULES=OFF",
            )

    def test_asan_cmake_args_preserve_caller_arguments(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            runtime = self._make_sdk(root)
            caller_args = "-DFOO=ON -DQUOTED='value with spaces'"
            with mock.patch.object(
                bpw, "capture", return_value=str(runtime)
            ), mock.patch.dict(os.environ, {"CMAKE_ARGS": caller_args}):
                env = bpw._setup_asan_build_env(root, "gfx942:xnack+")

            self.assertEqual(
                env["CMAKE_ARGS"],
                caller_args + " -DCMAKE_CXX_SCAN_FOR_MODULES=OFF",
            )

    def test_runtime_outside_sdk_is_rejected(self):
        with tempfile.TemporaryDirectory() as sdk_td, tempfile.TemporaryDirectory(
        ) as rt_td:
            root = Path(sdk_td)
            self._make_sdk(root)
            runtime = Path(rt_td) / "libclang_rt.asan-x86_64.so"
            runtime.touch()
            with mock.patch.object(bpw, "capture", return_value=str(runtime)):
                with self.assertRaisesRegex(RuntimeError, "outside"):
                    bpw._setup_asan_build_env(root, "gfx942:xnack+")

    def test_common_asan_env_omits_gcc_warning_flags(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env = bpw._setup_common_build_env(
                root / "cmake",
                root / "bin",
                root,
                "gfx942:xnack+",
                None,
                False,
                asan=True,
            )
            self.assertNotIn("CXXFLAGS", env)
            self.assertNotIn("CPPFLAGS", env)

    def test_common_release_env_preserves_gcc_warning_flags(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env = bpw._setup_common_build_env(
                root / "cmake", root / "bin", root, "gfx942", None, False
            )
            self.assertIn("maybe-uninitialized", env["CXXFLAGS"])
            self.assertIn("maybe-uninitialized", env["CPPFLAGS"])
            self.assertNotIn("CMAKE_ARGS", env)


class AsanInstallAndFeatureTest(unittest.TestCase):
    def test_bootstrap_requires_modern_setuptools_and_wheel(self):
        versions = {"setuptools": "69.0", "wheel": "0.45.1"}
        with mock.patch.object(
            bpw.metadata, "version", side_effect=lambda name: versions[name]
        ):
            with self.assertRaisesRegex(RuntimeError, "setuptools>=70.2"):
                bpw.validate_asan_bootstrap_requirements()

    def test_bootstrap_accepts_preinstalled_build_dependencies(self):
        versions = {"setuptools": "75.0", "wheel": "0.45.1"}
        with mock.patch.object(
            bpw.metadata, "version", side_effect=lambda name: versions[name]
        ):
            bpw.validate_asan_bootstrap_requirements()

    def test_install_is_offline_and_includes_single_target_device(self):
        args = argparse.Namespace(
            asan=True,
            pip_cache_dir=None,
            pre=True,
            index_url=None,
            find_links="/local/whl-asan/gfx942-all/index.html",
            rocm_sdk_version="==10.1.0+asan.20260807",
            rocm_extras="device",
            no_index=True,
        )
        with mock.patch.object(bpw, "run_command") as run, mock.patch.object(
            bpw, "get_rocm_sdk_version", return_value="10.1.0+asan.20260807"
        ), mock.patch.object(bpw, "validate_asan_bootstrap_requirements"):
            bpw.do_install_rocm(args)
        install_command = next(
            call.args[0] for call in run.call_args_list if "install" in call.args[0]
        )
        self.assertIn("--no-index", install_command)
        self.assertIn("--no-build-isolation", install_command)
        self.assertIn("--find-links", install_command)
        self.assertIn(
            "rocm[libraries,devel,device]==10.1.0+asan.20260807", install_command
        )

    def test_release_install_preserves_index_and_extras_behavior(self):
        args = argparse.Namespace(
            asan=False,
            no_index=False,
            pip_cache_dir=None,
            pre=True,
            index_url="https://example.invalid/simple",
            find_links=None,
            rocm_sdk_version=">1.0",
            rocm_extras="",
        )
        with mock.patch.object(bpw, "run_command") as run, mock.patch.object(
            bpw, "get_rocm_sdk_version", return_value="10.1.0"
        ):
            bpw.do_install_rocm(args)
        install_command = next(
            call.args[0] for call in run.call_args_list if "install" in call.args[0]
        )
        self.assertNotIn("--no-index", install_command)
        self.assertNotIn("--no-build-isolation", install_command)
        self.assertIn("--index-url", install_command)
        self.assertIn("rocm[libraries,devel]>1.0", install_command)

    def test_asan_defaults_to_prebuilt_aotriton_without_triton_wheel(self):
        args = argparse.Namespace(asan=True, enable_pytorch_flash_attention=None)
        self.assertTrue(
            bpw.resolve_pytorch_flash_attention(
                args, {"PYTORCH_ROCM_ARCH": "gfx942:xnack+"}, None
            )
        )
        args.enable_pytorch_flash_attention = False
        self.assertFalse(
            bpw.resolve_pytorch_flash_attention(
                args, {"PYTORCH_ROCM_ARCH": "gfx942:xnack+"}, None
            )
        )

    def test_asan_sanity_preloads_validated_runtime_only_for_capture(self):
        with tempfile.TemporaryDirectory() as td:
            runtime = Path(td) / "libclang_rt.asan-x86_64.so"
            runtime.touch()
            args = argparse.Namespace(asan=True, asan_runtime_path=runtime)
            build_env = {
                "ASAN_OPTIONS": "detect_leaks=0:abort_on_error=1",
                "LD_LIBRARY_PATH": "/rocm/lib",
            }
            with mock.patch.dict(
                os.environ, {"LD_PRELOAD": "/existing/preload.so"}
            ), mock.patch.object(bpw, "capture", return_value="False") as capture:
                bpw.sanity_check_installed_pytorch(args, build_env)

            sanity_env = capture.call_args.kwargs["env"]
            self.assertEqual(
                sanity_env["LD_PRELOAD"],
                f"{runtime}{os.path.pathsep}/existing/preload.so",
            )
            self.assertEqual(sanity_env["ASAN_OPTIONS"], build_env["ASAN_OPTIONS"])
            self.assertEqual(
                sanity_env["LD_LIBRARY_PATH"], build_env["LD_LIBRARY_PATH"]
            )

    def test_release_sanity_does_not_set_preload(self):
        args = argparse.Namespace(asan=False)
        with mock.patch.object(bpw, "capture", return_value="False") as capture:
            bpw.sanity_check_installed_pytorch(args, {})
        self.assertIsNone(capture.call_args.kwargs["env"])


if __name__ == "__main__":
    unittest.main()
