# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent.parent))

from dasan_math_build import SDK_STAGES, bootstrap_sdk


class BootstrapSdkTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sdk = self.root / "rocm"
        self.output = self.root / "output"
        self.output.mkdir()
        for _, required in SDK_STAGES.values():
            path = self.sdk / required
            if "/cmake/" in required:
                path = path / "package-config.cmake"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n", encoding="utf-8")
        for relative in (
            "include/rocm_kpack/kpack.h",
            "lib/librocm_kpack.so.0",
            "include/amd_comgr/amd_comgr.h",
            "lib/libamd_comgr.so.3",
            "include/hip/hip_runtime.h",
            "lib/libamdhip64.so.8",
            "lib/librocblas.so",
        ):
            path = self.sdk / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(relative, encoding="utf-8")

    def test_imported_stages_preserve_sdk_ownership_and_resume(self):
        bootstrap_sdk(self.sdk, self.output)
        build = self.output / "build"
        self.assertTrue(
            (build / "core/rocm-kpack/stage/include/rocm_kpack/kpack.h").exists()
        )
        self.assertTrue(
            (build / "compiler/amd-comgr/stage/include/amd_comgr/amd_comgr.h").exists()
        )
        self.assertTrue((build / "core/clr/stage/lib/libamdhip64.so.8").exists())
        self.assertFalse(list(build.rglob("librocblas.so")))
        self.assertFalse(
            (build / "compiler/amd-llvm/stage/lib/libamdhip64.so.8").exists()
        )
        marker = build / "compiler/amd-llvm/stage.prebuilt"
        before = marker.stat().st_mtime_ns
        bootstrap_sdk(self.sdk, self.output)
        self.assertEqual(before, marker.stat().st_mtime_ns)

    def test_changed_sdk_requires_new_build_directory(self):
        bootstrap_sdk(self.sdk, self.output)
        (self.sdk / "lib/llvm/bin/clang").write_text(
            "different compiler", encoding="utf-8"
        )
        with self.assertRaisesRegex(RuntimeError, "SDK changed"):
            bootstrap_sdk(self.sdk, self.output)

    def test_existing_non_prebuilt_stage_is_not_overwritten(self):
        stage = self.output / "build/compiler/amd-llvm/stage"
        stage.mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "Refusing to overwrite"):
            bootstrap_sdk(self.sdk, self.output)

    def test_missing_runtime_is_not_marked_prebuilt(self):
        (self.sdk / "lib/cmake/rocm-kpack/package-config.cmake").unlink()
        (self.sdk / "lib/cmake/rocm-kpack").rmdir()
        with self.assertRaisesRegex(RuntimeError, "Incomplete compiler/runtime SDK"):
            bootstrap_sdk(self.sdk, self.output)
        self.assertFalse(
            (self.output / "build/core/rocm-kpack/stage.prebuilt").exists()
        )


if __name__ == "__main__":
    unittest.main()
