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
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "profile_rocfft_aot", ROOT / "build_tools/profile_rocfft_aot.py"
)
trial = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trial)


class ControllerTests(unittest.TestCase):
    def test_helper_guard_rejects_lost_work_fallback_missing_commit_and_policy_change(
        self,
    ):
        import copy

        row = dict(
            event="complete",
            arch="gfx1030",
            bounded_subprocess_errors=[],
            properties={"sqlite_journal_mode": "wal", "sqlite_synchronous": "2"},
            counters=dict.fromkeys(
                [
                    "store_errors",
                    "subprocess_failures",
                    "fallbacks",
                    "pipe_errors",
                    "spawn_errors",
                    "wait_errors",
                    "nonzero_exits",
                    "child_signals",
                ],
                0,
            ),
            phases={
                name: {"completed": count, "active": 0}
                for name, count in [
                    ("generate", 17204),
                    ("store_query", 17204),
                    ("store_mutex_hold", 17204),
                    ("subprocess_compile", 16000),
                    ("inprocess_compile", 1204),
                    ("aot_begin", 1),
                    ("aot_commit", 1),
                ]
            },
        )
        row["counters"]["cache_misses"] = 17204
        trial.validate_helper_completion(row, "1")
        for group, name, value in [
            ("counters", "spawn_errors", 1),
            ("counters", "fallbacks", 1),
            ("properties", "sqlite_synchronous", "1"),
        ]:
            bad = copy.deepcopy(row)
            bad[group][name] = value
            with self.assertRaisesRegex(ValueError, "Changed helper work"):
                trial.validate_helper_completion(bad, "1")
        for name in [
            "store_query",
            "store_mutex_hold",
            "subprocess_compile",
            "aot_commit",
        ]:
            bad = copy.deepcopy(row)
            bad["phases"][name]["completed"] -= 1
            with self.assertRaisesRegex(ValueError, "Changed helper work"):
                trial.validate_helper_completion(bad, "1")
        bad = copy.deepcopy(row)
        bad["phases"]["aot_commit"]["active"] = 1
        with self.assertRaises(ValueError):
            trial.validate_helper_completion(bad, "1")
        # Compile mode may legitimately shift; total work must remain identical.
        row["phases"]["subprocess_compile"]["completed"] = 15000
        row["phases"]["inprocess_compile"]["completed"] = 2204
        trial.validate_helper_completion(row, "1")
        with self.assertRaises(ValueError):
            trial.validate_helper_completion(row, "0")

    def test_owned_modes_are_explicit_and_legacy_switch_stays_unchanged(self):
        self.assertEqual(trial.transaction_setting("legacy", False), "0")
        self.assertEqual(trial.transaction_setting("legacy", True), "1")
        for mode, setting in [
            ("owned-autocommit", "0"),
            ("owned-first", "1"),
            ("owned-all", "all"),
        ]:
            self.assertEqual(trial.transaction_setting(mode, False), setting)
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                trial.transaction_setting(mode, True)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            trial.transaction_setting("surprise", False)

    def test_owned_chain_is_new_same_filesystem_and_never_resets_previous_chain(self):
        with tempfile.TemporaryDirectory() as raw:
            nested = Path(raw)
            env = {
                "ROCFFT_AOT_PRIVATE_CACHE_PATH": "inherited-other-run",
                "KEEP": "same",
            }
            first = trial.prepare_owned_cache(nested, env, "fixture-run")
            path = Path(first["path"])
            self.assertFalse(path.exists())
            self.assertEqual(first["device"], first["original_parent_device"])
            self.assertEqual(path.parent.parent, nested.resolve() / "library/src")
            self.assertEqual(env["ROCFFT_AOT_PRIVATE_CACHE_ID"], first["owner_id"])
            self.assertEqual(
                Path(first["marker"]).read_text(), first["owner_id"] + "\n"
            )
            path.write_bytes(b"first chain must survive")
            second = trial.prepare_owned_cache(nested, env, "fixture-run")
            self.assertNotEqual(first["path"], second["path"])
            self.assertNotEqual(first["owner_id"], second["owner_id"])
            self.assertFalse(Path(second["path"]).exists())
            self.assertEqual(path.read_bytes(), b"first chain must survive")
            self.assertEqual(env["KEEP"], "same")
            shipped = nested / "library/src/rocfft_kernel_cache_gfx1030.db"
            shipped.write_bytes(b"already generated")
            with self.assertRaisesRegex(ValueError, "all seven shipped outputs"):
                trial.prepare_owned_cache(nested, env, "fixture-run")
            self.assertEqual(shipped.read_bytes(), b"already generated")

    def test_owned_chain_requires_all_seven_worksets_and_unchanged_marker(self):
        with tempfile.TemporaryDirectory() as raw:
            info = trial.prepare_owned_cache(Path(raw), {}, "fixture")
            with closing(sqlite3.connect(info["path"])) as db, db:
                db.execute("CREATE TABLE cache_v1(arch TEXT)")
                db.executemany(
                    "INSERT INTO cache_v1 VALUES(?)", [(a,) for a in trial.ARCHES]
                )
            result = trial.verify_owned_cache(info, entries_per_arch=1)
            self.assertEqual(result["entries"], 7)
            with closing(sqlite3.connect(info["path"])) as db, db:
                db.execute("DELETE FROM cache_v1 WHERE arch='gfx1036'")
            with self.assertRaisesRegex(ValueError, "all seven architecture worksets"):
                trial.verify_owned_cache(info, entries_per_arch=1)
            Path(info["marker"]).write_text("another-run\n")
            with self.assertRaisesRegex(ValueError, "identity changed"):
                trial.verify_owned_cache(info, entries_per_arch=1)

    def test_private_comgr_cache_replaces_inherited_path_and_refuses_reuse(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            shared = root / "shared-cache"
            shared.mkdir()
            sentinel = shared / "keep"
            sentinel.write_bytes(b"shared state")
            env = {
                "AMD_COMGR_CACHE": "0",
                "AMD_COMGR_CACHE_DIR": str(shared),
                "AMD_COMGR_CACHE_POLICY": "arbitrary",
            }
            path = trial.prepare_private_compiler_cache(root, env)
            self.assertEqual(env["AMD_COMGR_CACHE"], "1")
            self.assertEqual(Path(env["AMD_COMGR_CACHE_DIR"]), path.resolve())
            self.assertNotIn("AMD_COMGR_CACHE_POLICY", env)
            self.assertEqual(trial.compiler_cache_inventory(path)["entry_count"], 0)
            (path / "nested").mkdir()
            (path / "nested/code").write_bytes(b"code bytes")
            inventory = trial.compiler_cache_inventory(path)
            self.assertEqual(
                (inventory["entry_count"], inventory["file_count"], inventory["bytes"]),
                (2, 1, 10),
            )
            with self.assertRaises(FileExistsError):
                trial.prepare_private_compiler_cache(root, env)
            self.assertEqual(sentinel.read_bytes(), b"shared state")

    def test_trial_refuses_existing_private_database_without_removing_it(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            trial.require_fresh_private_cache(root)
            cache = root / "library/src/rocfft_temp_cache.db"
            cache.parent.mkdir(parents=True)
            cache.write_bytes(b"preexisting private database")
            with self.assertRaisesRegex(ValueError, "fresh, helper-owned"):
                trial.require_fresh_private_cache(root)
            self.assertEqual(cache.read_bytes(), b"preexisting private database")

    def test_compiler_normal_variable_is_resolved_and_conflicts_fail(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            compiler = root / "SDK with spaces/clang++"
            compiler.parent.mkdir()
            compiler.write_bytes(b"fixture compiler")
            (root / "CMakeCache.txt").write_text("GPU_TARGETS:STRING=gfx1030\n")
            metadata = root / "CMakeFiles/3.31.6/CMakeCXXCompiler.cmake"
            metadata.parent.mkdir(parents=True)
            metadata.write_text(f'set(CMAKE_CXX_COMPILER "{compiler.as_posix()}")\n')
            self.assertEqual(trial.resolve_compiler(root), compiler.resolve())
            (root / "CMakeCache.txt").write_text(
                f"CMAKE_CXX_COMPILER:FILEPATH={root / 'another-compiler'}\n"
            )
            with self.assertRaisesRegex(ValueError, "one actual generated"):
                trial.resolve_compiler(root)

    def test_failed_metadata_does_not_discard_semantic_output_checks(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            db = root / "cache.db"
            with closing(sqlite3.connect(db)) as handle, handle:
                handle.execute(
                    "CREATE TABLE cache_v1(kernel_name,arch,hip_version,generator_sum,code,timestamp)"
                )
                handle.execute(
                    "INSERT INTO cache_v1 VALUES(?,?,?,?,?,?)",
                    ("kernel", "gfx1030", 123, b"g" * 32, b"code", 0),
                )

            def fail_metadata():
                raise KeyError("CMAKE_CXX_COMPILER")

            with self.assertRaisesRegex(RuntimeError, "sdk_identity"):
                trial.run_postchecks(
                    [
                        ("sdk_identity", fail_metadata),
                        (
                            "cache_outputs",
                            lambda: trial.inventory_database(
                                db, root / "inventory.jsonl"
                            ),
                        ),
                    ],
                    root,
                )
            rows = (root / "inventory.jsonl").read_text().splitlines()
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                json.loads(rows[0])["code_sha256"],
                trial.hashlib.sha256(b"code").hexdigest(),
            )
            checks = json.loads((root / "postchecks.json").read_text())
            self.assertEqual([c["result"] for c in checks], ["failed", "passed"])

    def test_observer_counts_probe_and_serialized_io_once_and_excludes_wait(self):
        clock = SimpleNamespace(now=0.0)
        rows = []

        class SlowStream:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def write(self, text):
                rows.append(json.loads(text))
                clock.now += 2.0

            def flush(self):
                clock.now += 3.0

        class StopAfterTwo:
            stopped, waits = False, 0

            def is_set(self):
                return self.stopped

            def wait(self, seconds):
                clock.now += seconds
                self.waits += 1
                self.stopped = self.waits == 2

        def probe_timestamp():
            clock.now += 1.0
            return "fixture"

        observer = trial.WaitObserver(Path("rocfft_aot_helper"), Path("unused"))
        observer.stop = StopAfterTwo()
        fake_psutil = SimpleNamespace(
            Process=lambda: SimpleNamespace(children=lambda **_: []), Error=RuntimeError
        )
        with patch.dict(
            sys.modules,
            {
                "psutil": fake_psutil,
                "profile_build": SimpleNamespace(cgroups=lambda: ([], {})),
            },
        ), patch.object(trial, "write_json"), patch.object(
            trial.Path, "open", return_value=SlowStream()
        ), patch.object(
            trial, "utc", side_effect=probe_timestamp
        ), patch.object(
            trial.time, "monotonic", side_effect=lambda: clock.now
        ), patch.object(
            trial.time, "thread_time", return_value=0.0
        ):
            observer._run()
        # Each body costs 1s probe + 2s serialization/write + 3s flush.
        self.assertNotIn("error", observer.summary)
        self.assertEqual(observer.summary["samples"], 2)
        self.assertEqual(observer.summary["active_wall_s"], 12.0)
        self.assertEqual(observer.summary["lifetime_wall_s"], 14.0)
        self.assertEqual([r["observer_active_wall_s"] for r in rows], [0.0, 6.0])
        self.assertTrue(
            all(
                r["observer_active_wall_scope"]
                == "completed_previous_samples_including_write_flush"
                for r in rows
            )
        )

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
                ("AMD_COMGR_DRIVER_OPTIONS_APPEND", "-O0"),
                ("HIPRTC_COMPILE_OPTIONS_APPEND", "-O0"),
                ("HIPRTC_LINK_OPTIONS_APPEND", "-v"),
                ("AMD_COMGR_EMIT_VERBOSE_LOGS", "1"),
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

    def test_transaction_patch_is_limited_to_reviewed_host_files(self):
        text = (
            ROOT
            / "patches/build-profile/rocm-libraries/0002-rocfft-private-aot-transaction.patch"
        ).read_text()
        self.assertEqual(
            trial.patch_paths(text, trial.BATCH_PATCH_FILES), trial.BATCH_PATCH_FILES
        )
        additions = "\n".join(
            line[1:]
            for line in text.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )
        self.assertIn("rbp::private_aot_batch_requested", additions)
        self.assertIn("rbp::Scope phase(rbp::aot_commit)", additions)
        self.assertNotIn("synchronous", additions)
        self.assertNotIn("ROCFFT_RTC_MAX_SUBPROCESSES", additions)
        self.assertNotIn("generator_sum()", additions)
        owned = (
            ROOT
            / "patches/build-profile/rocm-libraries/0003-rocfft-owned-aot-chain.patch"
        ).read_text()
        self.assertEqual(
            trial.patch_paths(
                owned, {"projects/rocfft/library/src/rocfft_aot_helper.cpp"}
            ),
            {"projects/rocfft/library/src/rocfft_aot_helper.cpp"},
        )
        self.assertNotIn("rtc_compile.cpp", owned)

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
        self.assertIn("default: legacy", text)
        self.assertIn("shell: bash", job)
        self.assertIn('--cache-mode "$ROCFFT_AOT_CACHE_MODE"', job)


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
