# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import importlib.util
from contextlib import closing
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "profile_rocfft_aot", ROOT / "build_tools/profile_rocfft_aot.py"
)
trial = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trial)


class ControllerTests(unittest.TestCase):
    def test_wait_channel_reader_is_bounded_and_reports_missing_data(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for name in ["1", "2", "3"]:
                (root / name).mkdir()
            (root / "1/wchan").write_text("futex_wait_queue")
            (root / "2/wchan").write_text("0")
            data = trial.thread_wait_channels(root, limit=2)
            self.assertEqual(data["wait_channels"], {"futex_wait_queue": 1, "0": 1})
            self.assertTrue(data["truncated"])
            self.assertEqual(data["sampled_task_count"], 2)
            full = trial.thread_wait_channels(root)
            self.assertTrue(full["read_errors"])
            self.assertIn("error", trial.explicit_read(root / "missing"))
            with self.assertRaises(ValueError):
                trial.WaitObserver(root / "other-process", root)

    def test_ci_and_readonly_cache_boundary(self):
        env = dict(
            GITHUB_ACTIONS="true",
            GITHUB_REPOSITORY="ROCm/TheRock",
            RELEASE_TYPE="ci",
            CCACHE_READONLY="1",
        )
        with patch.object(trial.sys, "platform", "linux"):
            trial.require_ci(env)
            for key, bad in [
                ("GITHUB_REPOSITORY", "ROCm/rockrel"),
                ("GITHUB_ACTIONS", "false"),
                ("RELEASE_TYPE", "nightly"),
                ("CCACHE_READONLY", "0"),
                ("ROCFFT_LAYER", "1"),
            ]:
                with self.assertRaises(ValueError):
                    trial.require_ci({**env, key: bad})

    def test_reviewed_patch_cannot_modify_generator_or_architecture_commands(self):
        text = (
            ROOT
            / "patches/build-profile/rocm-libraries/0001-rocfft-aot-phase-diagnostics.patch"
        ).read_text()
        self.assertEqual(trial.patch_paths(text), trial.PATCH_FILES)
        with self.assertRaises(ValueError):
            trial.patch_paths(
                text
                + "\ndiff --git a/projects/rocfft/library/src/rtc_compile.cpp b/projects/rocfft/library/src/rtc_compile.cpp\n"
            )
        for forbidden in [
            "FORCE_OUT_PROCESS =",
            "ROCFFT_RTC_PROCESS=",
            "enable_write_mostly(false)",
            "kgen_logic_files",
        ]:
            self.assertNotIn("+" + forbidden, text)

    def test_generator_input_fingerprints_detect_content_changes(self):
        with tempfile.TemporaryDirectory() as raw:
            source = Path(raw)
            (source / "library/src").mkdir(parents=True)
            (source / "shared").mkdir()
            (source / "library/src/CMakeLists.txt").write_text(
                "set( kgen_embed_files\n ${CMAKE_SOURCE_DIR}/shared/a.h\n)\n"
                "set( kgen_logic_files\n ${CMAKE_SOURCE_DIR}/shared/b.h\n)\n"
            )
            for name in ["a.h", "b.h", "embed-to-header-rtc.py"]:
                (source / "shared" / name).write_text(name)
            before = trial.generator_inputs(source)
            self.assertEqual(len(before), 3)
            (source / "shared/a.h").write_text("changed")
            self.assertNotEqual(before, trial.generator_inputs(source))

    def test_semantic_inventory_is_independent_of_database_page_order(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            rows = [
                ("a", "gfx1030", 123, b"g" * 32, b"first", 0),
                ("b", "gfx1030", 123, b"g" * 32, b"second", 0),
            ]
            results = []
            for i, data in enumerate([rows, rows[::-1]]):
                db = root / f"{i}.db"
                with closing(sqlite3.connect(db)) as handle, handle:
                    handle.execute(
                        "CREATE TABLE cache_v1(kernel_name,arch,hip_version,generator_sum,code,timestamp)"
                    )
                    handle.executemany("INSERT INTO cache_v1 VALUES(?,?,?,?,?,?)", data)
                results.append(trial.inventory_database(db, root / f"{i}.jsonl"))
            self.assertEqual(
                results[0]["semantic_sha256"], results[1]["semantic_sha256"]
            )
            self.assertNotEqual(results[0]["file_sha256"], results[1]["file_sha256"])
            self.assertEqual(results[0]["entries"], 2)
            with closing(sqlite3.connect(root / "1.db")) as handle, handle:
                handle.execute(
                    "UPDATE cache_v1 SET code=? WHERE kernel_name='a'", (b"different",)
                )
            changed = trial.inventory_database(root / "1.db", root / "changed.jsonl")
            self.assertNotEqual(
                changed["semantic_sha256"], results[0]["semantic_sha256"]
            )

    def test_workflow_is_default_off_and_cannot_publish(self):
        text = (ROOT / ".github/workflows/multi_arch_ci.yml").read_text()
        self.assertEqual(
            text.count(
                "profile_rocfft_aot:\n        type: boolean\n        default: false"
            ),
            2,
        )
        job = text.split("  profile_rocfft_aot:\n    name:")[1]
        self.assertIn("github.repository == 'ROCm/TheRock'", job)
        self.assertIn("runs-on: aws-linux-scale-rocm-prod", job)
        self.assertIn("CCACHE_READONLY: 1", job)
        self.assertNotIn("id-token: write", job)
        self.assertNotIn("secrets:", job)
        self.assertIn("if: always()", job)


class HostAggregationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        explicit = os.environ.get("ROCFFT_PROFILE_TEST_CXX")
        compiler = json.loads(explicit) if explicit else None
        if not compiler:
            tool = shutil.which("g++") or shutil.which("clang++")
            if not tool:
                if os.environ.get("GITHUB_ACTIONS") == "true":
                    raise RuntimeError(
                        "A host C++ compiler is required for the diagnostic fixture"
                    )
                raise unittest.SkipTest(
                    "No host C++ compiler; set ROCFFT_PROFILE_TEST_CXX to a JSON argv"
                )
            compiler = [tool]
        cls.temp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temp.name)
        cls.binary = cls.folder / ("fixture.exe" if os.name == "nt" else "fixture")
        subprocess.run(
            [
                *compiler,
                "-std=c++17",
                "-pthread",
                "-O2",
                "-I",
                str(ROOT / "build_tools/diagnostics"),
                str(ROOT / "build_tools/tests/rocfft_profile_fixture.cpp"),
                str(ROOT / "build_tools/tests/rocfft_profile_fixture_other.cpp"),
                "-o",
                str(cls.binary),
            ],
            check=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_real_lock_wait_occupancy_counts_exception_and_prompt_stop(self):
        directory = self.folder / "enabled"
        env = {**os.environ, "ROCFFT_BUILD_PROFILE_DIR": str(directory)}
        result = json.loads(
            subprocess.check_output([str(self.binary)], env=env, text=True)
        )
        traces = list(directory.glob("*.jsonl"))
        self.assertEqual(len(traces), 1)
        rows = [json.loads(line) for line in traces[0].read_text().splitlines()]
        waiting = [
            r
            for r in rows
            if r["event"] == "sample" and r["phases"]["get_mutex_wait"]["active"] == 1
        ]
        self.assertTrue(
            waiting, "Real blocked mutex must appear in a periodic snapshot"
        )
        self.assertEqual(waiting[0]["phases"]["future_wait"]["active"], 1)
        final = rows[-1]
        self.assertEqual(final["event"], "complete")
        self.assertEqual(final["phases"]["store_query"]["completed"], 50000)
        self.assertEqual(final["phases"]["get_mutex_wait"]["completed"], 1)
        self.assertGreater(final["phases"]["get_mutex_wait"]["sum_ns"], 1_000_000_000)
        self.assertEqual(final["phases"]["inprocess_compile"]["completed"], 1)
        self.assertTrue(all(p["active"] == 0 for p in final["phases"].values()))
        self.assertEqual(final["counters"]["subprocess_failures"], 10)
        self.assertEqual(final["counters"]["spawn_errors"], 1)
        self.assertEqual(final["counters"]["last_spawn_errno"], 11)
        self.assertEqual(final["counters"]["source_bytes"], 1234)
        for group in ["self", "waited_children"]:
            usage = final["getrusage"][group]
            if os.name == "posix":
                self.assertTrue(usage["available"])
                self.assertGreaterEqual(usage["user_delta_s"], 0)
                self.assertGreaterEqual(usage["system_delta_s"], 0)
                self.assertGreaterEqual(usage["current_maxrss_kib"], 0)
                if group == "waited_children":
                    self.assertGreater(
                        usage["user_delta_s"] + usage["system_delta_s"],
                        0,
                        "Real waited fixture child CPU must be retained",
                    )
            else:
                self.assertFalse(usage["available"])
                self.assertNotIn("user_delta_s", usage)
        self.assertEqual(len(final["bounded_subprocess_errors"]), 8)
        self.assertTrue(all(len(e) == 160 for e in final["bounded_subprocess_errors"]))
        self.assertEqual(final["properties"]["fixture_property"], 'a"b\nc')
        self.assertLess(
            result["stop_ns"],
            500_000_000,
            "Stop notification must avoid waiting for the next one-second tick",
        )
        print("Host fixture diagnostic overhead observation:", json.dumps(result))

    def test_disabled_session_does_not_create_files(self):
        env = dict(os.environ)
        env.pop("ROCFFT_BUILD_PROFILE_DIR", None)
        result = json.loads(
            subprocess.check_output([str(self.binary)], env=env, text=True)
        )
        self.assertLess(result["stop_ns"], 500_000_000)
        print("Host fixture disabled observation:", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
