# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build_tools"))
import profile_profiler_lifecycle as controller
import profile_profiler_lifecycle_trial as trial


class LifecycleTrialTest(unittest.TestCase):
    def setUp(self):
        self.contract = json.loads(controller.CONTRACT.read_text())

    def actual_tests(self):
        path = (
            ROOT / "build_tools/tests/fixtures/profiler_lifecycle_actual_2264.json.gz"
        )
        origin = json.loads(
            path.with_name("profiler_lifecycle_actual_2264_origin.json").read_text()
        )
        compressed = path.read_bytes()
        self.assertEqual(hashlib.sha256(compressed).hexdigest(), origin["gzip_sha256"])
        data = gzip.decompress(compressed)
        self.assertEqual(len(data), origin["uncompressed_bytes"])
        self.assertEqual(
            hashlib.sha256(data).hexdigest(), origin["uncompressed_sha256"]
        )
        tests = json.loads(data)["tests"]
        self.assertEqual(len(tests), 2264)
        self.assertEqual(len({t["name"] for t in tests}), 2262)
        return tests

    def test_complete_actual2264_fixture_selected_namespace_and_all_validators(self):
        tests = self.actual_tests()
        name = self.contract["original_execute_name"]
        original = next(t for t in tests if t["name"] == name)
        trial.verify_execute(original, self.contract)
        self.assertIn("signal-handler-test", trial.properties(original)["DEPENDS"])
        with self.assertRaises(ValueError):
            trial.fixture_closure(tests, name)
        closure = trial.fixture_closure(tests, name, frozenset({"signal-handler-test"}))
        prefix = "tests.integration."
        case = "app-without-signal-handling-spawn"
        expected = [
            prefix + "cleanup." + case,
            prefix + "execute." + case,
            *[
                prefix + "validate." + case + "." + suffix
                for suffix in [
                    "test_clean_exit",
                    "test_flushed_markers",
                    "test_json_is_valid",
                    "test_output_files_exist",
                ]
            ],
        ]
        self.assertEqual(closure, sorted(expected))
        # Exact selected regex matches six records despite unrelated doctor duplicates.
        regex = "^(" + "|".join(re.escape(n) for n in closure) + ")$"
        self.assertEqual(
            sorted(t["name"] for t in tests if re.fullmatch(regex, t["name"])),
            sorted(expected),
        )

    def test_complete_actual_fixture_rejects_selected_duplicates_and_unknown_dependency(
        self,
    ):
        tests = self.actual_tests()
        name = self.contract["original_execute_name"]
        for selected in [
            name,
            "tests.integration.cleanup.app-without-signal-handling-spawn",
            "tests.integration.validate.app-without-signal-handling-spawn.test_clean_exit",
        ]:
            changed = [
                *tests,
                copy.deepcopy(next(t for t in tests if t["name"] == selected)),
            ]
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                trial.fixture_closure(changed, name, frozenset({"signal-handler-test"}))
        changed = copy.deepcopy(tests)
        original = next(t for t in changed if t["name"] == name)
        next(p for p in original["properties"] if p["name"] == "DEPENDS")[
            "value"
        ].append("__unknown_build_dependency__")
        with self.assertRaises(ValueError):
            trial.fixture_closure(changed, name, frozenset({"signal-handler-test"}))
        with self.assertRaises(ValueError):
            trial.fixture_closure(
                tests,
                name,
                frozenset({"signal-handler-test", "__unknown_build_dependency__"}),
            )

    def test_exact_consumer_alias_app_proof_and_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp)
            app = build / "bin/signal-handler/signal-handler-test"
            app.parent.mkdir(parents=True)
            app.write_bytes(b"\x7fELFfixture")
            query = "ninja: Entering directory\nsignal-handler-test:\n  input: phony\n    bin/signal-handler/signal-handler-test\n  outputs:\n"
            proof = trial.verify_build_target_alias(
                build, app, "signal-handler-test", query
            )
            self.assertEqual(
                proof["sha256"], hashlib.sha256(app.read_bytes()).hexdigest()
            )
            for changed in [
                query.replace(
                    "bin/signal-handler/signal-handler-test", "bin/wrong-app"
                ),
                query.replace("signal-handler-test:", "wrong-target:"),
                query.replace("input: phony", "input: unexpected"),
            ]:
                with self.assertRaises(ValueError):
                    trial.verify_build_target_alias(
                        build, app, "signal-handler-test", changed
                    )
            with self.assertRaises(ValueError):
                trial.verify_build_target_alias(
                    build, app.with_name("wrong-app"), "signal-handler-test", query
                )

    @unittest.skipUnless(shutil.which("ninja"), "Ninja source fixture unavailable")
    def test_real_ninja_alias_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build = root / "build"
            build.mkdir()
            (build / "build.ninja").write_text(
                "build signal-handler-test: phony bin/signal-handler/signal-handler-test\n"
            )
            app = build / "bin/signal-handler/signal-handler-test"
            app.parent.mkdir(parents=True)
            app.write_bytes(b"\x7fELFfixture")
            query = subprocess.check_output(
                ["ninja", "-C", str(build), "-t", "query", "signal-handler-test"],
                text=True,
            )
            self.assertEqual(
                trial.verify_build_target_alias(
                    build, app, "signal-handler-test", query
                )["path"],
                str(app),
            )

    def test_original_context_component_and_action_pins(self):
        import yaml

        fixture = json.loads(
            (
                ROOT
                / "build_tools/tests/fixtures/profiler_lifecycle_original_component.json"
            ).read_text()
        )
        original = fixture["component"]
        self.assertEqual(original, self.contract["original_component"])
        changed = {
            key for key in original if original[key] != self.contract["component"][key]
        }
        self.assertEqual(changed, {"test_script"})
        self.assertIn("--cap-add=SYS_PTRACE", original["container_options"])
        self.assertIn("64f8d6e3", original["container_image"])
        workflow = yaml.safe_load(
            (
                ROOT / ".github/workflows/build_profile_profiler_lifecycle.yml"
            ).read_text()
        )
        self.assertEqual(set(workflow["jobs"]), {"context", "lifecycle"})
        job = workflow["jobs"]["lifecycle"]
        self.assertEqual(
            json.loads(job["with"]["component"]), self.contract["component"]
        )
        self.assertEqual(job["with"]["artifact_run_id"], "41271198901")
        self.assertNotIn("secrets", job)
        primary = yaml.safe_load(
            (ROOT / ".github/workflows/multi_arch_ci.yml").read_text()
        )
        trigger = primary.get("on", primary.get(True))
        self.assertFalse(
            trigger["workflow_dispatch"]["inputs"]["profile_profiler_lifecycle"][
                "default"
            ]
        )
        command = ["git", "-C", str(ROOT)]
        if os.environ.get("RECHECK_FIXTURE_GIT_DIR"):
            command = ["git", "--git-dir", os.environ["RECHECK_FIXTURE_GIT_DIR"]]
        raw = subprocess.check_output(
            command
            + [
                "grep",
                "-h",
                "uses:",
                "60faab8d3d433d009f154c3e30f9fd8ac5cfda59",
                "--",
                ".github/workflows",
            ]
        )
        stock = set(re.findall(r"uses:\s*([^\s#]+)", raw.decode()))
        for path in [
            ROOT / ".github/workflows/build_profile_profiler_lifecycle.yml",
            ROOT / ".github/workflows/test_component.yml",
            ROOT / ".github/workflows/multi_arch_ci.yml",
        ]:
            for ref in re.findall(r"uses:\s*([^\s#]+)", path.read_text()):
                if not ref.startswith("./"):
                    self.assertIn(ref, stock)

    def test_exact_actual_argv_deadline_and_expectation_guard(self):
        original = dict(
            name=self.contract["original_execute_name"],
            command=self.contract["original_execute_argv"],
            properties=[
                dict(name="TIMEOUT", value=30),
                dict(name="PASS_REGULAR_EXPRESSION", value=["tool finalization"]),
            ],
        )
        trial.verify_execute(original, self.contract)
        for mutate in [
            lambda t: t["command"].append("--different"),
            lambda t: t["properties"][0].update(value=31),
            lambda t: t["properties"][1].update(value=[".*"]),
        ]:
            changed = copy.deepcopy(original)
            mutate(changed)
            with self.assertRaises(ValueError):
                trial.verify_execute(changed, self.contract)

    def test_fixture_closure_includes_named_and_recursive_dependencies(self):
        def t(name, **props):
            return dict(
                name=name, properties=[dict(name=k, value=v) for k, v in props.items()]
            )

        tests = [
            t("execute", FIXTURES_SETUP=["result"], FIXTURES_REQUIRED=["clean"]),
            t("prepare", FIXTURES_SETUP=["clean"], DEPENDS=["other-required"]),
            t("validate-a", FIXTURES_REQUIRED=["result"]),
            t("validate-b", FIXTURES_REQUIRED=["result"]),
            t("final-cleanup", FIXTURES_CLEANUP=["result"]),
            t("other-required"),
            t("unrelated"),
        ]
        self.assertEqual(
            trial.fixture_closure(tests, "execute"),
            [
                "execute",
                "final-cleanup",
                "other-required",
                "prepare",
                "validate-a",
                "validate-b",
            ],
        )
        with self.assertRaises(ValueError):
            trial.fixture_closure(
                [t for t in tests if t["name"] != "prepare"], "execute"
            )
        with self.assertRaises(ValueError):
            trial.fixture_closure([*tests, tests[0]], "execute")

    def test_new_child_parent_proof_precedes_exe_maps_and_retained_identity_can_reparent(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fields = ["0"] * 20
            fields[0], fields[1], fields[19] = "S", "1", "111"
            child = root / "123"
            (child / "task/123").mkdir(parents=True)
            (child / "stat").write_text("123 (child) " + " ".join(fields))
            for name in ["maps", "wchan", "io", "task/123/children"]:
                (child / name).write_text("")
            with patch.object(trial.os, "readlink") as read_exe:
                row = trial.read_proc(123, True, root, expected_parent=(999, 77))
            self.assertEqual(row["status"], "new-child-parent-rejected")
            read_exe.assert_not_called()
            parent = root / "1"
            parent.mkdir()
            fields[19] = "77"
            (parent / "stat").write_text("1 (parent) " + " ".join(fields))
            with patch.object(trial.os, "readlink") as read_exe:
                row = trial.read_proc(123, True, root, expected_parent=(1, 78))
            self.assertEqual(row["status"], "new-child-parent-identity-rejected")
            read_exe.assert_not_called()
            with patch.object(trial.os, "readlink", return_value="/fixture/python"):
                row = trial.read_proc(123, True, root, expected_parent=(1, 77))
            self.assertEqual(row["status"], "observed")
            with patch.object(trial.os, "readlink", return_value="/fixture/python"):
                row = trial.read_proc(
                    123, True, root, expected_parent=(999, 0), expected_start_ticks=111
                )
            self.assertEqual(row["status"], "observed")
            self.assertEqual(row["ppid"], 1)

    @unittest.skipUnless(
        shutil.which("cmake") and shutil.which("ctest"),
        "CMake/CTest source fixture unavailable",
    )
    def test_real_ctest_generated_fixture_closure_and_original_properties(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CMakeLists.txt").write_text(
                """cmake_minimum_required(VERSION 3.20)
project(LifecycleFixture NONE)
enable_testing()
add_test(NAME setup COMMAND "${CMAKE_COMMAND}" -E echo setup)
set_tests_properties(setup PROPERTIES FIXTURES_SETUP clean)
add_test(NAME execute COMMAND "${CMAKE_COMMAND}" -E echo "tool finalization")
set_tests_properties(execute PROPERTIES FIXTURES_REQUIRED clean FIXTURES_SETUP result TIMEOUT 30 PASS_REGULAR_EXPRESSION "tool finalization")
foreach(name alpha beta gamma delta)
  add_test(NAME "validate-${name}" COMMAND "${CMAKE_COMMAND}" -E echo "${name}")
  set_tests_properties("validate-${name}" PROPERTIES FIXTURES_REQUIRED result)
endforeach()
add_test(NAME unrelated COMMAND "${CMAKE_COMMAND}" -E false)
"""
            )
            subprocess.run(
                ["cmake", "-S", str(root), "-B", str(root / "build")],
                capture_output=True,
                check=True,
            )
            base = ["ctest", "--test-dir", str(root / "build"), "--parallel", "8"]
            data = json.loads(subprocess.check_output([*base, "--show-only=json-v1"]))
            closure = trial.fixture_closure(data["tests"], "execute")
            self.assertEqual(
                closure,
                [
                    "execute",
                    "setup",
                    "validate-alpha",
                    "validate-beta",
                    "validate-delta",
                    "validate-gamma",
                ],
            )
            regex = "^(" + "|".join(re.escape(n) for n in closure) + ")$"
            selected = json.loads(
                subprocess.check_output([*base, "-R", regex, "--show-only=json-v1"])
            )
            self.assertEqual(sorted(t["name"] for t in selected["tests"]), closure)
            original = next(t for t in selected["tests"] if t["name"] == "execute")
            self.assertEqual(trial.properties(original)["TIMEOUT"], 30)
            self.assertEqual(
                trial.properties(original)["PASS_REGULAR_EXPRESSION"],
                ["tool finalization"],
            )
            result = subprocess.run(
                [*base, "-R", regex, "--no-tests=error"], capture_output=True
            )
            self.assertEqual(result.returncode, 0, result.stdout.decode())

    def test_sdk_inputs_ignore_new_consumer_build_but_detect_producer_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in [
                "bin/rocprofv3",
                "lib/librocprofiler-sdk.so",
                "share/rocprofiler-sdk/tests/source.cpp",
                ".kpack/core.kpack",
            ]:
                path = root / "build" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"\x7fELF" + name.encode())
            with patch.object(
                controller,
                "query_loader",
                return_value=subprocess.CompletedProcess([], 0, "", ""),
            ):
                before = controller.capture_inputs(root, "rocprofiler-sdk")
                self.assertEqual(len(before["files"]), 4)
                app = (
                    root
                    / "build/share/rocprofiler-sdk/tests/build/bin/signal-handler/signal-handler-test"
                )
                app.parent.mkdir(parents=True)
                app.write_bytes(b"new consumer build")
                self.assertEqual(
                    before, controller.capture_inputs(root, "rocprofiler-sdk", before)
                )
                (root / "build/lib/librocprofiler-sdk.so").write_bytes(b"changed")
                self.assertNotEqual(
                    before, controller.capture_inputs(root, "rocprofiler-sdk", before)
                )
                (root / "build/bin/rocprofv3").unlink()
                with self.assertRaises(ValueError):
                    controller.capture_inputs(root, "rocprofiler-sdk", before)

    def test_output_cap_preserves_input_and_retains_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input"
            source.mkdir()
            original = source / "output.json"
            original.write_bytes(b"original output")
            digest = hashlib.sha256(original.read_bytes()).hexdigest()
            rows = trial.preserve_files(source, root / "retained", max_bytes=0)
            self.assertFalse(rows[0]["retained"])
            self.assertEqual(rows[0]["sha256"], digest)
            self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(), digest)

    def test_proc_permission_maps_vanish_and_identity_races_are_typed(self):
        def stat(ticks):
            fields = ["0"] * 20
            fields[0], fields[1], fields[19] = "S", "1", str(ticks)
            return "123 (fixture worker) " + " ".join(fields)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "123"
            (target / "task/123").mkdir(parents=True)
            for name, body in [
                ("stat", stat(111)),
                ("maps", "address permissions /fixture/lib.so"),
                ("wchan", "do_wait"),
                ("io", "wchar: 0"),
                ("task/123/children", ""),
            ]:
                (target / name).write_text(body)
            original_read = Path.read_text

            def maps_denied(path, *args, **kwargs):
                if path.name == "maps":
                    raise PermissionError("fixture")
                return original_read(path, *args, **kwargs)

            with patch.object(
                trial.os, "readlink", return_value="/fixture/python"
            ), patch.object(Path, "read_text", maps_denied):
                row = trial.read_proc(123, True, root)
            self.assertEqual(row["status"], "observed")
            self.assertEqual(row["maps_error"], "PermissionError")
            self.assertEqual(row["exe"], "/fixture/python")
            self.assertEqual(row["start_ticks"], 111)
            for final in [stat(222), FileNotFoundError("vanished")]:
                calls = 0

                def changed_stat(path, *args, **kwargs):
                    nonlocal calls
                    if path.name == "stat":
                        calls += 1
                        if calls == 2:
                            if isinstance(final, Exception):
                                raise final
                            return final
                    return original_read(path, *args, **kwargs)

                with patch.object(
                    trial.os, "readlink", return_value="/fixture/python"
                ), patch.object(Path, "read_text", changed_stat):
                    row = trial.read_proc(123, True, root)
                self.assertIn(
                    row["status"], {"identity-race-rejected", "identity-unverified"}
                )
                self.assertNotIn("maps", row)
                self.assertNotIn("exe", row)
            with patch.object(
                Path, "read_text", side_effect=PermissionError("fixture")
            ):
                row = trial.read_proc(123, True, root)
            self.assertEqual(
                row,
                dict(
                    pid=123,
                    status="unavailable",
                    stage="initial-stat",
                    error="PermissionError",
                ),
            )
            with patch.object(
                trial.os, "readlink", side_effect=["/fixture/before", "/fixture/after"]
            ):
                row = trial.read_proc(123, True, root)
            self.assertEqual(row["status"], "exec-race-rejected")
            self.assertNotIn("maps", row)

    @unittest.skipUnless(
        sys.platform == "linux" and Path("/proc/self/stat").exists(),
        "Linux proc source fixture",
    )
    def test_real_owned_child_tree_observed_and_plain_case_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = [
                sys.executable,
                "-c",
                "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(0.3)']); p.wait()",
            ]
            observed = trial.launch(
                command, root, os.environ.copy(), root / "observed", True, 0.02, 0.1
            )
            plain = trial.launch(
                command, root, os.environ.copy(), root / "plain", False, 0.02, 0.1
            )
            rows = [
                json.loads(s)
                for s in (root / "observed/process.jsonl").read_text().splitlines()
            ]
            observed_rows = [r for r in rows if r["status"] == "observed"]
            self.assertGreaterEqual(len({r["pid"] for r in observed_rows}), 2)
            self.assertTrue(
                all("start_ticks" in r and "environ" not in r for r in observed_rows)
            )
            self.assertEqual(observed["returncode"], plain["returncode"])
            self.assertEqual(plain["process_log_bytes"], 0)
            self.assertIsNotNone(observed["waited_child_cpu"]["user_s"])
            self.assertIn("controller_cpu_s", observed)

    def test_actual_replay_fails_outside_therock_ci(self):
        with patch.dict(
            os.environ, {"GITHUB_ACTIONS": "false"}, clear=False
        ), self.assertRaises(ValueError):
            trial.run_trial(ROOT, self.contract)


if __name__ == "__main__":
    unittest.main()
