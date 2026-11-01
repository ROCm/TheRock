# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "capacity_probe", ROOT / "build_tools/profile_rocshmem_runtime.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
FIXTURES = ROOT / "build_tools/tests/fixtures/rocshmem-runtime"


class HeapCapacityTest(unittest.TestCase):
    def test_exact_original_graph_only_ten_env_fields_change(self):
        original = (FIXTURES / "actual-40-functional-CTestTestfile.cmake").read_bytes()
        changed = probe.capacity_text(original)
        before, after = original.splitlines(), changed.splitlines()
        differences = [(x, y) for x, y in zip(before, after) if x != y]
        self.assertEqual(len(before), len(after))
        self.assertEqual(len(differences), 10)
        for x, y in differences:
            self.assertEqual(
                y,
                x.replace(
                    b"ROCSHMEM_HEAP_SIZE=6442450944", b"ROCSHMEM_HEAP_SIZE=25769803776"
                ),
            )
        with self.assertRaises(ValueError):
            probe.capacity_text(changed)
        with self.assertRaises(ValueError):
            probe.capacity_text(original + b"# unexpected\n")

    def test_file_receipt_tamper_and_no_prior_adoption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / probe.CTEST_PATH
            path.parent.mkdir(parents=True)
            path.write_bytes(
                (FIXTURES / "actual-40-functional-CTestTestfile.cmake").read_bytes()
            )
            output = root / "runtime-probe"
            output.mkdir()
            with patch.object(probe, "ROOT", root), patch.object(probe, "OUT", output):
                probe.prepare_capacity()
                probe.verify_capacity()
                with self.assertRaises(ValueError):
                    probe.prepare_capacity()
                path.write_bytes(path.read_bytes() + b"# tamper\n")
                with self.assertRaises(ValueError):
                    probe.verify_capacity()

    def test_linked_graph_is_rejected_without_modifying_other_link(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / probe.CTEST_PATH
            path.parent.mkdir(parents=True)
            original = (
                FIXTURES / "actual-40-functional-CTestTestfile.cmake"
            ).read_bytes()
            shared = root / "shared-input.cmake"
            shared.write_bytes(original)
            try:
                os.link(shared, path)
            except OSError as error:
                self.skipTest(
                    "Host does not permit inert hardlink fixture: " + str(error)
                )
            output = root / "runtime-probe"
            output.mkdir()
            with patch.object(probe, "ROOT", root), patch.object(probe, "OUT", output):
                with self.assertRaises(ValueError):
                    probe.prepare_capacity()
            self.assertEqual(shared.read_bytes(), original)

    def test_every_original_executed_case_must_stay_executed(self):
        rows = json.loads((FIXTURES / "actual-45-outcomes.json").read_text())["rows"]

        def stdout(values):
            return "\n".join(
                f"{i}/296 Test #{r['id']}: {r['name']} ....{r['status']} 1.00 sec"
                for i, r in enumerate(values, 1)
            )

        corrected = copy.deepcopy(rows)
        for row in corrected:
            if row["status"] == "***Failed":
                row["status"] = "Passed"
        with patch.object(probe, "ROOT", ROOT):
            probe.require_original_case_coverage(stdout(corrected))
            corrected[0]["status"] = "***Skipped"
            with self.assertRaises(ValueError):
                probe.require_original_case_coverage(stdout(corrected))
            with self.assertRaises(ValueError):
                probe.require_original_case_coverage(stdout(rows))
            with self.assertRaises(ValueError):
                probe.require_original_case_coverage(stdout(corrected[1:]))

    def test_budget_requires_eight_successful_current_measurements(self):
        with patch.dict(os.environ, {"ROCSHMEM_HEAP_ALLOCATOR_TYPE": "coarsegrained"}):
            with self.assertRaises(ValueError):
                probe.prepare({})
        good = dict(
            count_returncode=0,
            count=8,
            devices=[
                dict(
                    device=i,
                    set_returncode=0,
                    memory_returncode=0,
                    free_bytes=probe.MIN_FREE_BYTES,
                    total_bytes=192 * 1024**3,
                )
                for i in range(8)
            ],
        )
        probe.require_memory_budget(good)
        for field, value in [
            ("free_bytes", probe.MIN_FREE_BYTES - 1),
            ("memory_returncode", 1),
            ("set_returncode", 1),
            ("total_bytes", probe.MIN_FREE_BYTES - 1),
            ("free_bytes", None),
        ]:
            broken = copy.deepcopy(good)
            broken["devices"][4][field] = value
            with self.assertRaises(ValueError):
                probe.require_memory_budget(broken)
        broken = copy.deepcopy(good)
        broken["devices"].pop()
        with self.assertRaises(ValueError):
            probe.require_memory_budget(broken)

    def test_memory_gate_precedes_unchanged_original_driver(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "before.json").write_text("{}")
            events = []

            def launched(argv, **kwargs):
                events.append("launch")
                self.assertEqual(events, ["graph", "budget", "launch"])
                self.assertEqual(
                    argv,
                    [
                        probe.sys.executable,
                        "build_tools/github_actions/test_executable_scripts/test_runner.py",
                    ],
                )
                return SimpleNamespace(
                    stdout=io.StringIO("inert original driver\n"), wait=lambda: 0
                )

            with patch.object(probe, "OUT", output), patch.object(
                probe, "verify_capacity", side_effect=lambda: events.append("graph")
            ), patch.object(
                probe, "memory_budget", side_effect=lambda name: events.append("budget")
            ), patch.object(
                probe.subprocess, "Popen", side_effect=launched
            ):
                probe.run_test()
            with patch.object(probe, "OUT", output), patch.object(
                probe, "verify_capacity"
            ), patch.object(
                probe, "memory_budget", side_effect=ValueError("no headroom")
            ), patch.object(
                probe.subprocess, "Popen"
            ) as launch:
                with self.assertRaises(ValueError):
                    probe.run_test()
                launch.assert_not_called()

    @unittest.skipUnless(
        os.name == "posix" and shutil.which("cc"),
        "Linux host C fixture requires cc; no ROCm code is built",
    )
    def test_actual_ctypes_child_with_inert_host_library(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = root / "build/lib/libamdhip64.so"
            library.parent.mkdir(parents=True)
            output = root / "runtime-probe"
            output.mkdir()
            source = root / "fake.c"
            source.write_text(
                "#include <stddef.h>\nstatic int device;\nint hipGetDeviceCount(int*n){*n=8;return 0;}\nint hipSetDevice(int n){device=n;return 0;}\nint hipMemGetInfo(size_t*f,size_t*t){*f=((size_t)40+device)<<30;*t=(size_t)192<<30;return 0;}\n"
            )
            subprocess.run(
                [
                    shutil.which("cc"),
                    "-shared",
                    "-fPIC",
                    str(source),
                    "-o",
                    str(library),
                ],
                check=True,
                capture_output=True,
            )
            with patch.object(probe, "ROOT", root), patch.object(probe, "OUT", output):
                result = probe.memory_budget("inert-host-query")
            self.assertEqual(
                [r["free_bytes"] for r in result["observation"]["devices"]],
                [(40 + i) * 1024**3 for i in range(8)],
            )
            self.assertEqual(result["command"]["returncode"], 0)


if __name__ == "__main__":
    unittest.main()
