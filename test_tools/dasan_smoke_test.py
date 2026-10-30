# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Guard against false success when lit drops or skips device tests."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_tools.dasan_bundle import SUPPORT

from test_tools.dasan_smoke import (
    HIP_TESTS,
    HOST_TESTS,
    OPENMP_TESTS,
    check_lit_results,
    hip_link_runtime_flags,
    write_lit_overlay,
)


class SharedRuntimeConfigurationTest(unittest.TestCase):
    def test_bundle_shared_mode_resolves_runtime_for_lit_subprocesses(self):
        namespace = {}
        exec(SUPPORT, namespace)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bindir = root / "lib/llvm/bin"
            bindir.mkdir(parents=True)
            for name in (
                "clang",
                "FileCheck",
                "not",
                "llvm-symbolizer",
                "llvm-readelf",
                "ld.lld",
                "clang-linker-wrapper",
                "clang-offload-bundler",
            ):
                (bindir / name).write_text("fixture")
            resource = root / "lib/llvm/lib/clang/24"
            runtime = resource / "lib/x86_64-unknown-linux-gnu"
            runtime.mkdir(parents=True)
            for name in (
                "libclang_rt.dasan.a",
                "libclang_rt.dasan-preinit.a",
                "libclang_rt.dasan.so",
            ):
                (runtime / name).write_text("fixture")
            device = resource / "lib/amdgpu-amd-amdhsa/libclang_rt.dasan.a"
            device.parent.mkdir()
            device.write_text("fixture")
            for shared in (False, True):
                config = SimpleNamespace(environment={})

                def fatal(message):
                    raise RuntimeError(message)

                params = {"rocm_root": str(root)}
                if shared:
                    params["dasan_shared_runtime"] = "1"
                lit = SimpleNamespace(params=params, fatal=fatal, note=lambda _: None)
                with patch.dict(os.environ), patch(
                    "subprocess.check_output", side_effect=[str(resource), str(runtime)]
                ):
                    namespace["configure_paths"](
                        config, lit, root, "x86_64-unknown-linux-gnu"
                    )
                    loader_dirs = config.environment["LD_LIBRARY_PATH"].split(
                        os.pathsep
                    )
                    self.assertEqual(str(runtime) in loader_dirs, shared)
                    self.assertEqual(
                        os.environ["LD_LIBRARY_PATH"],
                        config.environment["LD_LIBRARY_PATH"],
                    )

    def execute_overlay(self, path, substitutions):
        config = SimpleNamespace(substitutions=[])
        loaded = []

        def load_config(current, source):
            loaded.append(source)
            current.substitutions = list(substitutions)

        def fatal(message):
            raise RuntimeError(message)

        lit_config = SimpleNamespace(load_config=load_config, fatal=fatal)
        exec(
            (path / "lit.site.cfg.py").read_text(encoding="utf-8"),
            {"config": config, "lit_config": lit_config},
        )
        return config, loaded

    def test_shared_gpu_run_covers_module_host_helper_without_changing_host_tests(self):
        original = [
            ("%clang ", "clang "),
            ("%clang_dasan ", "clang -fsanitize=daddress "),
            ("%clang_hip_dasan ", "clang -xhip -fsanitize=daddress "),
            ("%clang_omp_offload_dasan ", "clang -fopenmp -fsanitize=daddress "),
            ("%run", ""),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            site = root / "original.site.py"
            site.write_text("# original test configuration\n")
            for shared in (False, True):
                overlay = write_lit_overlay(
                    site,
                    root / str(shared),
                    openmp=True,
                    gpu_device_count=8,
                    shared_runtime=shared,
                )
                config, loaded = self.execute_overlay(overlay, original)
                self.assertEqual(loaded, [str(site)])
                self.assertEqual(config.gpu_device_count, 8)
                self.assertTrue(config.can_run_openmp_offload)
                actual = dict(config.substitutions)
                if shared:
                    # module-unload.hip links its HOST executable with this
                    # generic sanitizer substitution and the real HIP library.
                    for pattern in (
                        "%clang_dasan ",
                        "%clang_hip_dasan ",
                        "%clang_omp_offload_dasan ",
                    ):
                        self.assertIn("-shared-libsan", actual[pattern])
                    self.assertEqual(actual["%clang "], "clang ")
                    self.assertEqual(actual["%run"], "")
                else:
                    # Host hsa-missing.c explicitly covers default static and
                    # shared linkage; its invocation must retain both modes.
                    self.assertEqual(config.substitutions, original)
            self.assertEqual(site.read_text(), "# original test configuration\n")

    def test_missing_gpu_substitution_cannot_silently_keep_static_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            overlay = write_lit_overlay(
                root / "original.site.py",
                root,
                openmp=True,
                gpu_device_count=None,
                shared_runtime=True,
            )
            for substitutions in (
                [("%clang_dasan ", "clang -fsanitize=daddress ")],
                [
                    ("%clang_dasan ", "clang -fsanitize=daddress "),
                    ("%clang_hip_dasan ", "clang -xhip -fsanitize=daddress "),
                ],
            ):
                with self.assertRaisesRegex(RuntimeError, "Missing DASan compiler"):
                    self.execute_overlay(overlay, substitutions)

    def test_only_opt_in_hip_linking_uses_the_shared_runtime_and_test_rpath(self):
        runtime = Path("/sdk/lib/llvm/lib/clang/24/lib/x86_64-unknown-linux-gnu")
        self.assertEqual(hip_link_runtime_flags(False, runtime), [])
        self.assertEqual(
            hip_link_runtime_flags(True, runtime),
            ["-shared-libsan", "-Wl,-rpath," + str(runtime)],
        )


class LitCoverageTest(unittest.TestCase):
    def check_results(self, results: dict[str, str], expected: set[str], allowed=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lit.json"
            path.write_text(
                json.dumps(
                    {
                        "tests": [
                            {"name": "DASAN-x86_64-linux :: " + name, "code": code}
                            for name, code in results.items()
                        ]
                    }
                ),
                encoding="utf-8",
            )
            return check_lit_results(path, expected, allowed or set())

    def test_host_only_success_cannot_satisfy_device_scope(self):
        with self.assertRaisesRegex(RuntimeError, "inventory mismatch"):
            self.check_results({name: "PASS" for name in HOST_TESTS}, HIP_TESTS)

    def test_missing_device_test_is_rejected(self):
        results = {name: "PASS" for name in HIP_TESTS}
        results.pop("AMDGPU/heap-overflow.hip")
        with self.assertRaisesRegex(RuntimeError, "inventory mismatch"):
            self.check_results(results, HIP_TESTS)

    def test_only_explicit_multi_device_skip_is_allowed(self):
        results = {name: "PASS" for name in HIP_TESTS}
        results["AMDGPU/multi-device.hip"] = "UNSUPPORTED"
        allowed = {"AMDGPU/multi-device.hip"}
        self.assertEqual(self.check_results(results, HIP_TESTS, allowed), results)
        with self.assertRaisesRegex(RuntimeError, "did not pass"):
            self.check_results(results, HIP_TESTS)
        results["AMDGPU/heap-overflow.hip"] = "UNSUPPORTED"
        with self.assertRaisesRegex(RuntimeError, "did not pass"):
            self.check_results(results, HIP_TESTS, allowed)

    def test_expected_failure_does_not_count_as_pass(self):
        results = {name: "PASS" for name in HOST_TESTS}
        results["initialize.c"] = "XFAIL"
        with self.assertRaisesRegex(RuntimeError, "did not pass"):
            self.check_results(results, HOST_TESTS)

    def test_all_27_source_tests_still_require_pass(self):
        inventory = HOST_TESTS | HIP_TESTS | OPENMP_TESTS
        self.assertEqual(len(inventory), 27)
        self.check_results({name: "PASS" for name in inventory}, inventory)
        for bad_code in ("FAIL", "UNSUPPORTED", "XFAIL"):
            results = {name: "PASS" for name in inventory}
            results["AMDGPU/module-unload.hip"] = bad_code
            with self.assertRaisesRegex(RuntimeError, "did not pass"):
                self.check_results(results, inventory)


if __name__ == "__main__":
    unittest.main()
