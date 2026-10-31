# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Real stdlib-only capture and inert installed SDK regression fixtures."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "capture_identity", ROOT / "build_tools/normal_ci_input_identity.py"
)
IDENTITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IDENTITY)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CaptureTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="normal-capture-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def file(self, relative, body="inert fixture; never executed"):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        return path

    def test_real_capture_without_site_packages_preserves_receipt_schema(self):
        inputs = dict(
            profile_normal_validation="true",
            profile_normal_case="baseline",
            profile_normal_product="none",
            profile_normal_cache_regime="ci-readonly",
            linux_amdgpu_families="gfx94x,gfx103x",
            windows_amdgpu_families="",
            build_stages="",
            prebuilt_stages="",
            baseline_run_id="",
            stage_reuse_mode="off",
            changed_projects="",
            pr_labels="test_filter:comprehensive",
            build_python_packages="true",
            build_native_linux="true",
            build_native_windows="false",
            build_pytorch="false",
            build_jax="false",
        )
        event = self.file("event.json", json.dumps(dict(inputs=inputs)))
        output = self.root / "normal-validation/wheel-test.json"
        if sys.platform == "linux":
            target = self.file("build/lib/libhiprtc.so.7")
            target.with_name("libhiprtc.so").symlink_to(target.name)
        env = dict(os.environ)
        for key in IDENTITY.RTC_KNOBS:
            env.pop(key, None)
        env.update(
            GITHUB_ACTIONS="true",
            GITHUB_REPOSITORY="ROCm/TheRock",
            GITHUB_EVENT_NAME="workflow_dispatch",
            GITHUB_EVENT_PATH=str(event),
            GITHUB_WORKSPACE=str(self.root),
            RELEASE_TYPE="ci",
            NORMAL_IMAGE_PIN="native-fixture",
            STAGE_NAME="wheel-tests",
            AMDGPU_FAMILIES="gfx94X-dcgpu",
            HOME=str(self.root / "home"),
            USERPROFILE=str(self.root / "home"),
            XDG_CACHE_HOME=str(self.root / "cache"),
            LOCALAPPDATA=str(self.root / "cache"),
            AMD_COMGR_CACHE_DIR=str(self.root / "comgr"),
        )
        probe = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-c",
                "import importlib.util; assert importlib.util.find_spec('psutil') is None",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(probe.returncode, 0, probe.stderr)
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                str(ROOT / "build_tools/normal_ci_validation.py"),
                "capture",
                "--output",
                str(output),
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        receipt = json.loads(output.read_text())
        self.assertIn("cgroup_resolution", receipt["allocation"]["io_limits"])
        self.assertIn("workspace_storage", receipt["allocation"])
        sdk = receipt["sdk_runtime_inputs"]
        if sys.platform == "linux":
            self.assertEqual(len(sdk["files"]), 2)
            self.assertEqual(sdk["new_unique_fingerprint_inputs"], 1)
            self.assertEqual(len(receipt["resolved_loader_providers"]), 1)
            self.assertEqual(
                {row["entry_kind"] for row in sdk["files"]}, {"file", "symlink"}
            )
            self.assertEqual(len({row["sha256"] for row in sdk["files"]}), 1)
        else:
            self.assertEqual(sdk["files"], [])
        self.assertEqual(receipt["sdk_runtime_inputs"]["tool_presence"]["rocgdb"], [])
        self.assertTrue(receipt["sdk_runtime_inputs"]["complete"])
        self.assertGreaterEqual(receipt["sdk_runtime_inputs"]["capture_wall_s"], 0)
        self.assertTrue(receipt["declared_source_inputs"]["valid"])
        self.assertEqual(receipt["python_packages"], [])

    def test_installed_candidates_exclude_unrelated_math_payloads(self):
        # GDB is observed in make argv. Other roots are source-declared
        # candidates; this fixture is not an observed provider inventory.
        selected = json.loads(
            (
                ROOT / "build_tools/tests/fixtures/normal_component_sdk_paths.json"
            ).read_text()
        )["inert_paths"]
        paths = [self.file(path) for path in selected]
        unrelated = self.file("build/lib/librocblas.so.5")
        self.file("build/lib/rocblas/nested/libLLVM-decoy.so")
        visited = []

        def fingerprint(path):
            visited.append(Path(path))
            return digest(path)

        result = IDENTITY.sdk_inputs(self.root / "build", fingerprint)
        self.assertTrue(result["complete"], result)
        self.assertEqual(
            {row["path"] for row in result["files"]}, {str(p) for p in paths}
        )
        self.assertNotIn(unrelated, visited)
        self.assertEqual(result["tool_presence"]["rocgdb"], [str(paths[0])])
        self.assertTrue(
            all(row["sha256"] == digest(row["path"]) for row in result["files"])
        )
        before = next(row for row in result["files"] if "libamd_comgr" in row["path"])[
            "sha256"
        ]
        paths[4].write_text("changed runtime provider")
        after = IDENTITY.sdk_inputs(self.root / "build", digest)
        self.assertNotEqual(
            before,
            next(row for row in after["files"] if "libamd_comgr" in row["path"])[
                "sha256"
            ],
        )

    def test_nested_sdk_missing_and_bounded_inputs_remain_explicit(self):
        selected = self.file("build/core/clr/dist/lib/llvm/bin/clang-24")
        self.file("build/compiler/amd-comgr/dist/lib/libamd_comgr.so")
        self.file("build/math-libs/rocFFT/build/library/src/rocfft_rtc_helper")
        result = IDENTITY.sdk_inputs(self.root / "build", digest)
        self.assertTrue(result["complete"])
        self.assertIn(str(selected), [row["path"] for row in result["files"]])
        self.assertEqual(len(result["files"]), 3)
        self.assertEqual(result["tool_presence"]["rocgdb"], [])
        self.assertFalse(
            IDENTITY.sdk_inputs(self.root / "build", digest, max_entries=1)["complete"]
        )
        bounded = IDENTITY.sdk_inputs(self.root / "build", digest, max_bytes=1)
        self.assertFalse(bounded["complete"])
        self.assertEqual(bounded["files"], [])
        missing = IDENTITY.sdk_inputs(self.root / "absent", digest)
        self.assertEqual(missing["files"], [])
        self.assertFalse(any(row["exists"] for row in missing["roots"]))
        self.assertFalse((self.root / "absent").exists())

    def test_cgroup_resolution_keeps_visible_hierarchy_schema(self):
        values = {
            "/proc/self/cgroup": "0::/docker/job\n",
            "/proc/self/mountinfo": "21 20 0:1 /docker /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
        }

        def read(path, *args, **kwargs):
            return values[path.as_posix()]

        with patch.object(Path, "read_text", read):
            paths, scope = IDENTITY.visible_cgroups()
        self.assertEqual(paths, [Path("/sys/fs/cgroup/job"), Path("/sys/fs/cgroup")])
        self.assertEqual(scope["status"], "v2_visible_hierarchy")
        self.assertEqual(scope["membership"], "/docker/job")
        self.assertEqual(scope["mount_root"], str(Path("/docker")))
        with patch.object(Path, "read_text", side_effect=OSError("unavailable")):
            self.assertEqual(
                IDENTITY.visible_cgroups(), ([], {"status": "unavailable_or_non_linux"})
            )

    @unittest.skipUnless(sys.platform == "linux", "POSIX symlink fixture")
    def test_unavailable_sdk_alias_does_not_claim_complete_inventory(self):
        directory = self.root / "build/lib"
        directory.mkdir(parents=True)
        (directory / "libhiprtc.so").symlink_to("missing.so.7")
        result = IDENTITY.sdk_inputs(self.root / "build", digest)
        self.assertFalse(result["complete"])
        self.assertEqual(result["files"][0]["status"], "unavailable")
        self.assertEqual(result["files"][0]["link_target"], "missing.so.7")
        self.assertEqual(len(result["errors"]), 1)


if __name__ == "__main__":
    unittest.main()
