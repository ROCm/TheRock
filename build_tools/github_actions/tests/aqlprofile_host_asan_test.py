# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

GITHUB_ACTIONS_DIR = Path(__file__).parent.parent
SCRIPT_DIR = GITHUB_ACTIONS_DIR / "test_executable_scripts"
sys.path.insert(0, os.fspath(SCRIPT_DIR))

import test_aqlprofile_host_asan

SOURCE_MANIFEST = (
    Path(__file__).parents[3]
    / "rocm-systems"
    / "projects"
    / "aqlprofile"
    / "test"
    / "host_asan_tests.json"
)


def _completed(stdout: str = "", stderr: str = "", returncode: int = 0) -> Mock:
    return Mock(stdout=stdout, stderr=stderr, returncode=returncode)


def _gtest_listing(names: list[str]) -> str:
    suites: dict[str, list[str]] = {}
    for name in names:
        suite, test = name.rsplit(".", 1)
        suites.setdefault(suite, []).append(test)
    return "".join(
        f"{suite}.\n" + "".join(f"  {test}\n" for test in tests)
        for suite, tests in suites.items()
    )


class AqlProfileHostAsanTest(unittest.TestCase):
    def _install_manifest(self, prefix: Path) -> Path:
        root = (
            prefix
            / "share"
            / "hsa-amd-aqlprofile"
            / "tests"
            / "host-asan"
        )
        root.mkdir(parents=True)
        shutil.copyfile(SOURCE_MANIFEST, root / "host_asan_tests.json")
        return root

    def test_manifest_locks_78_exact_cases_and_all_exclusions(self):
        entries = test_aqlprofile_host_asan._load_manifest(SOURCE_MANIFEST)
        self.assertEqual(len(entries), 15)
        self.assertEqual(sum(len(entry["tests"]) for entry in entries), 78)
        self.assertEqual(
            {entry["name"] for entry in entries},
            test_aqlprofile_host_asan.EXPECTED_EXECUTABLES,
        )

    def test_missing_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "manifest is missing"):
                test_aqlprofile_host_asan._load_manifest(Path(tmp) / "missing.json")

    def test_count_or_digest_mismatch_is_rejected(self):
        mutations = {
            "digest": lambda manifest: manifest["executables"][0]["tests"].__setitem__(
                -1, "Unexpected.HostCase"
            ),
            "count": lambda manifest: manifest["executables"][0]["tests"].pop(),
        }
        for mismatch, mutate in mutations.items():
            with self.subTest(mismatch=mismatch):
                manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
                mutate(manifest)
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "manifest.json"
                    path.write_text(json.dumps(manifest), encoding="utf-8")
                    with self.assertRaisesRegex(
                        RuntimeError, "positive inventory changed"
                    ):
                        test_aqlprofile_host_asan._load_manifest(path)

    def test_missing_binary_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp)
            self._install_manifest(prefix)
            with (
                patch.object(
                    test_aqlprofile_host_asan,
                    "require_direct_clang_asan",
                    side_effect=RuntimeError("required host-ASAN executable is missing"),
                ),
                patch.object(test_aqlprofile_host_asan, "_execute") as execute,
            ):
                with self.assertRaisesRegex(RuntimeError, "executable is missing"):
                    test_aqlprofile_host_asan.run(prefix, {})
                execute.assert_not_called()

    def test_zero_match_is_rejected_before_execution(self):
        with patch.object(
            test_aqlprofile_host_asan,
            "_execute",
            return_value=_completed(),
        ) as execute:
            with self.assertRaisesRegex(RuntimeError, "selector inventory mismatch"):
                test_aqlprofile_host_asan._run_gtest(
                    Path("test"), ["Suite.case"], {}, trace_mode=False
                )
            execute.assert_called_once()

    def test_name_mismatch_is_rejected_before_execution(self):
        listed = _completed(stdout=_gtest_listing(["Suite.unexpected"]))
        with patch.object(
            test_aqlprofile_host_asan, "_execute", return_value=listed
        ) as execute:
            with self.assertRaisesRegex(RuntimeError, "selector inventory mismatch"):
                test_aqlprofile_host_asan._run_gtest(
                    Path("test"), ["Suite.expected"], {}, trace_mode=False
                )
            execute.assert_called_once()

    def test_child_failure_is_propagated(self):
        listed = _completed(stdout=_gtest_listing(["Suite.case"]))
        failed = _completed(stderr="deterministic failure\n", returncode=17)
        with patch.object(
            test_aqlprofile_host_asan,
            "_execute",
            side_effect=[listed, failed],
        ):
            with self.assertRaises(subprocess.CalledProcessError) as caught:
                test_aqlprofile_host_asan._run_gtest(
                    Path("test"), ["Suite.case"], {}, trace_mode=False
                )
            self.assertEqual(caught.exception.returncode, 17)

    def test_environment_removes_preload_and_switches_lsan_for_trace(self):
        with patch.dict(
            os.environ,
            {
                "LD_PRELOAD": "/tmp/not-instrumentation.so",
                "ASAN_OPTIONS": "existing=1",
            },
            clear=False,
        ):
            env = test_aqlprofile_host_asan._test_environment(Path("/opt/rocm"))
        self.assertNotIn("LD_PRELOAD", env)
        self.assertIn("detect_leaks=1", env["ASAN_OPTIONS"])
        self.assertIn("halt_on_error=1", env["ASAN_OPTIONS"])
        self.assertIn("exitcode=23", env["LSAN_OPTIONS"])

        with patch.dict(
            os.environ,
            {"THEROCK_HOST_ASAN_DEVICE_TRACE": "1"},
            clear=True,
        ):
            trace_env = test_aqlprofile_host_asan._test_environment(Path("/opt/rocm"))
        self.assertIn("detect_leaks=0", trace_env["ASAN_OPTIONS"])

    def test_trace_rejects_any_gpu_device_open(self):
        test_aqlprofile_host_asan._check_trace(
            'openat(AT_FDCWD, "/tmp/data", O_RDONLY) = 3\n', ["test"]
        )
        for node in ('/dev/kfd', '/dev/dri/renderD128'):
            with self.subTest(node=node):
                with self.assertRaisesRegex(RuntimeError, "GPU device-node access"):
                    test_aqlprofile_host_asan._check_trace(
                        f'openat(AT_FDCWD, "{node}", O_RDWR) = -1 ENOENT\n',
                        ["test"],
                    )

    def test_all_15_packaged_binaries_require_direct_shared_asan(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp)
            root = self._install_manifest(prefix)
            (root / "bin").mkdir()
            with (
                patch.object(
                    test_aqlprofile_host_asan, "require_direct_clang_asan"
                ) as require,
                patch.object(test_aqlprofile_host_asan, "_run_gtest"),
                patch.object(
                    test_aqlprofile_host_asan,
                    "_execute",
                    return_value=_completed(),
                ),
            ):
                test_aqlprofile_host_asan.run(prefix, {})
            self.assertEqual(require.call_count, 15)
            self.assertEqual(
                {call.args[0].name for call in require.call_args_list},
                test_aqlprofile_host_asan.EXPECTED_EXECUTABLES,
            )


if __name__ == "__main__":
    unittest.main()
