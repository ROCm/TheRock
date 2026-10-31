# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Inert source/SDK fixtures: never execute a compiler, loader or cache."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "normal_identity", ROOT / "build_tools/normal_ci_input_identity.py"
)
IDENTITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IDENTITY)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class InputIdentityTest(unittest.TestCase):
    def setUp(self):
        owned = ROOT / ".normal-unit-fixtures"
        owned.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="input-", dir=owned)
        self.root = Path(self.temp.name).resolve()
        self.assertIn(owned.resolve(), self.root.parents)
        self.addCleanup(self.temp.cleanup)

    def file(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
        return path

    def test_generated_compiler_wins_over_different_ambient_path(self):
        selected = self.file(
            "build/core/clr/dist/lib/llvm/bin/clang-24", "active compiler"
        )
        ambient = self.file("ambient/clang", "different PATH compiler")
        self.file(
            "build/math-libs/rocFFT/build/CMakeFiles/3.31/CMakeCXXCompiler.cmake",
            f'set(CMAKE_CXX_COMPILER "{selected.as_posix()}")\n',
        )
        self.file(
            "build/math-libs/rocFFT/build/CMakeCache.txt",
            "CMAKE_HIP_COMPILER:FILEPATH=unresolved-clang\nROCFFT_RUNTIME_COMPILE_DEFAULT:BOOL=OFF\n",
        )
        with patch.dict(os.environ, {"PATH": str(ambient.parent)}):
            result = IDENTITY.generated_inputs(self.root / "build", digest)
        active = next(
            row for row in result["fields"] if row["key"] == "CMAKE_CXX_COMPILER"
        )
        self.assertEqual(active["identity"]["sha256"], digest(selected))
        self.assertNotEqual(active["identity"]["sha256"], digest(ambient))
        hip = next(
            row for row in result["fields"] if row["key"] == "CMAKE_HIP_COMPILER"
        )
        self.assertIn("PATH is not substituted", hip["identity"]["status"])

    def test_only_comgr_or_hiprtc_mutation_changes_sdk_receipt(self):
        clang = self.file("build/core/clr/dist/lib/llvm/bin/clang-24", "same clang")
        comgr = self.file(
            "build/compiler/amd-comgr/dist/lib/libamd_comgr.so", "comgr one"
        )
        hiprtc = self.file("build/core/clr/dist/lib/libhiprtc.so", "hiprtc one")
        self.file("build/core/ROCR-Runtime/dist/lib/libhsa-runtime64.so", "hsa")
        self.file(
            "build/core/clr/dist/lib/llvm/amdgcn/bitcode/used.bc", "device library"
        )

        def rows():
            return {
                row["resolved_path"]: row["sha256"]
                for row in IDENTITY.sdk_inputs(self.root / "build", digest)["files"]
            }

        before = rows()
        comgr.write_text("comgr two")
        after = rows()
        self.assertEqual(before[str(clang)], after[str(clang)])
        self.assertNotEqual(before[str(comgr)], after[str(comgr)])
        self.assertEqual(before[str(hiprtc)], after[str(hiprtc)])
        hiprtc.write_text("hiprtc two")
        later = rows()
        self.assertNotEqual(after[str(hiprtc)], later[str(hiprtc)])

    def test_loader_provider_resolution_and_missing_library_are_explicit(self):
        library = self.file("sdk/libamd_comgr.so", "provider")

        def inspect(binary):
            return subprocess.CompletedProcess(
                [],
                0,
                f"libamd_comgr.so => {library.as_posix()} (0x1234)\nlibmissing.so => not found\n",
                "",
            )

        receipt = IDENTITY.loader_dependencies(
            self.root / "rtc-helper", digest, inspect
        )
        self.assertEqual(receipt["providers"][0]["sha256"], digest(library))
        self.assertEqual(receipt["unresolved"], ["libmissing.so => not found"])

    def test_generated_aot_command_and_distinct_cache_controls_are_recorded(self):
        cwd = self.root / "build/math-libs/rocFFT/build/library/src"
        helper = cwd / "rocfft_aot_helper"
        rtc = cwd / "rocfft_rtc_helper"
        self.file(
            "build/math-libs/rocFFT/build/build.ninja",
            f'  COMMAND = cd "{cwd.as_posix()}" && cmake -E env AMD_COMGR_CACHE=1 ROCFFT_RTC_PROCESS=1 "{helper.as_posix()}" "" out.db "{rtc.as_posix()}" gfx1030\n',
        )
        result = IDENTITY.generated_inputs(self.root / "build", digest)
        row = result["rtc_commands"][0]
        self.assertEqual(row["command_knob_assignments"]["AMD_COMGR_CACHE"], "1")
        self.assertEqual(
            row["aot_source_defined_overrides"]["ROCFFT_RTC_SYS_CACHE_PATH"], ":memory:"
        )
        self.assertEqual(
            row["aot_source_defined_overrides"]["ROCFFT_RTC_CACHE_PATH"],
            str(cwd / "rocfft_temp_cache.db"),
        )
        env = {
            "HOME": str(self.root / "home"),
            "AMD_COMGR_CACHE": "1",
            "ROCFFT_RTC_PROCESS": "0",
        }
        before = IDENTITY.runtime_cache_state(self.root, env, "linux")
        self.assertEqual(before["caches"][0]["inventory"]["status"], "absent")
        cache = self.file("home/.cache/comgr/entry", "not executed")
        after = IDENTITY.runtime_cache_state(self.root, env, "linux")
        self.assertEqual(len(after["caches"][0]["inventory"]["entries"]), 1)
        self.assertEqual(after["ambient_knobs"]["ROCFFT_RTC_PROCESS"], "0")
        self.assertTrue(cache.exists())
        self.assertIn("ccache regime does not control RTC/COMGR", after["scope"])

    def test_declared_untracked_header_change_detected_without_git_diff(self):
        header = self.file(
            "rocm-libraries/projects/rocfft/shared/diagnostic.h", "header one"
        )
        manifest = self.file(
            "declared.json",
            json.dumps(
                {
                    "patches": [],
                    "copied_inputs": [
                        {
                            "path": header.relative_to(self.root).as_posix(),
                            "sha256": digest(header),
                            "cases": ["candidate"],
                            "stages": ["math-libs"],
                        }
                    ],
                }
            ),
        )
        before = IDENTITY.declared_sources(
            self.root, manifest, "candidate", digest, "math-libs"
        )
        self.assertTrue(before["valid"])
        header.write_text("header two")
        after = IDENTITY.declared_sources(
            self.root, manifest, "candidate", digest, "math-libs"
        )
        self.assertFalse(after["valid"])
        self.assertNotEqual(before["files"][0]["sha256"], after["files"][0]["sha256"])
        out_of_scope = IDENTITY.declared_sources(
            self.root, manifest, "candidate", digest, "compiler-runtime"
        )
        self.assertTrue(out_of_scope["valid"])
        self.assertIn("outside declared stage", out_of_scope["files"][0]["status"])

    def test_family_specific_header_is_required_only_for_selected_math_job(self):
        header = self.file("rocm-libraries/projects/rocfft/private.h", "owned header")
        manifest = self.file(
            "declared.json",
            json.dumps(
                {
                    "copied_inputs": [
                        {
                            "path": header.relative_to(self.root).as_posix(),
                            "sha256": digest(header),
                            "stages": ["math-libs"],
                            "families": ["gfx103X-all"],
                        }
                    ]
                }
            ),
        )
        selected = IDENTITY.declared_sources(
            self.root, manifest, "baseline", digest, "math-libs", "gfx103x-all"
        )
        self.assertTrue(selected["valid"])
        header.unlink()
        other_family = IDENTITY.declared_sources(
            self.root, manifest, "baseline", digest, "math-libs", "gfx94X-dcgpu"
        )
        self.assertTrue(other_family["valid"])
        self.assertEqual(
            other_family["files"][0]["status"], "outside declared family scope"
        )
        missing_selected = IDENTITY.declared_sources(
            self.root, manifest, "baseline", digest, "math-libs", "gfx103X-all"
        )
        self.assertFalse(missing_selected["valid"])

    def test_workspace_device_and_visible_io_limits_have_explicit_unavailable_state(
        self,
    ):
        sysfs = self.root / "sysfs"
        device = sysfs / "240:9"
        device.mkdir(parents=True)
        with patch.object(
            IDENTITY.os, "major", return_value=240, create=True
        ), patch.object(IDENTITY.os, "minor", return_value=9, create=True):
            row = IDENTITY.workspace_storage_identity(self.root, sysfs)
            self.assertEqual(row["st_dev"], self.root.stat().st_dev)
            self.assertEqual(row["major"], 240)
            self.assertEqual(row["resolved_sysfs_path"], str(device.resolve()))
            device.rmdir()
            missing = IDENTITY.workspace_storage_identity(self.root, sysfs)
            self.assertIn("backing storage unresolved", missing["status"])
        group = self.file("cgroup/visible/io.max", "240:9 rbps=1000 wbps=max\n").parent
        limits = IDENTITY.io_limit_files([group, group / "absent"])
        self.assertEqual(limits["files"][0]["text"], "240:9 rbps=1000 wbps=max\n")
        self.assertEqual(limits["files"][1]["status"], "unavailable")
        self.assertFalse((group / "absent").exists())

    def test_product_selector_does_not_require_header_for_none_or_other_family(self):
        header = self.file("rocm-libraries/projects/rocfft/private.h", "owned header")
        manifest = self.file(
            "declared.json",
            json.dumps(
                {
                    "copied_inputs": [
                        {
                            "path": header.relative_to(self.root).as_posix(),
                            "sha256": digest(header),
                            "cases": ["baseline", "candidate"],
                            "stages": ["math-libs"],
                            "families": ["gfx103X-all"],
                            "products": ["rocfft-private-batch"],
                        }
                    ]
                }
            ),
        )
        selected = IDENTITY.declared_sources(
            self.root,
            manifest,
            "baseline",
            digest,
            "math-libs",
            "gfx103x-all",
            "rocfft-private-batch",
        )
        self.assertTrue(selected["valid"])
        self.assertEqual(selected["product"], "rocfft-private-batch")
        header.unlink()
        for product, family, expected in [
            ("none", "gfx103X-all", "product"),
            ("rocfft-private-batch", "gfx94X-dcgpu", "family"),
        ]:
            receipt = IDENTITY.declared_sources(
                self.root, manifest, "candidate", digest, "math-libs", family, product
            )
            self.assertTrue(receipt["valid"])
            self.assertIn(expected, receipt["files"][0]["status"])
        missing = IDENTITY.declared_sources(
            self.root,
            manifest,
            "candidate",
            digest,
            "math-libs",
            "gfx103X-all",
            "rocfft-private-batch",
        )
        self.assertFalse(missing["valid"])

    def test_declared_escape_rejected_and_cache_inventory_is_read_only(self):
        manifest = self.file(
            "declared.json",
            json.dumps(
                {"copied_inputs": [{"path": "../foreign.h", "sha256": "0" * 64}]}
            ),
        )
        with self.assertRaises(ValueError):
            IDENTITY.declared_sources(self.root, manifest, "baseline", digest)
        missing = self.root / "missing/comgr"
        self.assertEqual(IDENTITY.cache_inventory(missing)["status"], "absent")
        self.assertFalse(missing.exists())

    def test_live_receipt_filters_secrets_and_hashes_providers_afterward(self):
        library = self.file("sdk/libhiprtc.so", "provider one")
        process = SimpleNamespace(
            environ=lambda: {
                "AMD_COMGR_CACHE": "1",
                "ROCFFT_RTC_CACHE_PATH": "private.db",
                "SECRET_TOKEN": "must-never-appear",
            },
            exe=lambda: "/sdk/rocfft_rtc_helper",
            memory_maps=lambda grouped: [SimpleNamespace(path=library.as_posix())],
        )
        self.assertIsNone(IDENTITY.observed_rtc_process(process, ["/sdk/clang"]))
        receipt = IDENTITY.observed_rtc_process(process, ["/sdk/rocfft_rtc_helper"])
        self.assertNotIn("must-never-appear", json.dumps(receipt))
        self.assertNotIn("SECRET_TOKEN", json.dumps(receipt))
        self.assertEqual(receipt["knobs"]["ROCFFT_RTC_CACHE_PATH"], "private.db")
        profile = self.file(
            "samples.jsonl",
            json.dumps(dict(event="rtc_process_inputs", receipt=receipt)) + "\n",
        )
        resolved = IDENTITY.observed_provider_identities(profile, digest)
        self.assertEqual(resolved["files"][0]["sha256"], digest(library))
        self.assertTrue(resolved["files"][0]["postbuild_stat_matches_all_observations"])
        library.write_text("provider two, changed")
        changed = IDENTITY.observed_provider_identities(profile, digest)
        self.assertFalse(changed["files"][0]["postbuild_stat_matches_all_observations"])


if __name__ == "__main__":
    unittest.main()
