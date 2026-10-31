# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import profile_rocprofiler_attachment as probe
from rocprofiler_attachment_observer import TargetObserver, proc_identity, roctx_markers
from rocprofiler_attachment_sink import LoopbackSink
from rocprofiler_attachment_entry import run_original
from build_tools.tests.component_cpu_runner_routing_test import evaluate


class ScopeTests(unittest.TestCase):
    def test_matrix_is_pinned_before_containers_are_scheduled(self):
        with mock.patch.object(probe, "git", side_effect=self.git):
            components = [copy.deepcopy(self.contract["component"])]
            sanity = copy.deepcopy(self.contract["sanity_component"])
            self.assertTrue(
                probe.guard_matrix(self.inputs, self.env, components, sanity)[
                    "matrix_validated_before_container_scheduling"
                ]
            )
            with self.assertRaises(ValueError):
                probe.guard_matrix(self.inputs, self.env, components * 2, sanity)
            for role in ["profiler", "sanity"]:
                changed = copy.deepcopy(components)
                changed_sanity = copy.deepcopy(sanity)
                (changed[0] if role == "profiler" else changed_sanity)[
                    "container_options"
                ] += " --privileged"
                with self.subTest(role=role), self.assertRaises(ValueError):
                    probe.guard_matrix(self.inputs, self.env, changed, changed_sanity)

    def test_existing_roctx_marker_is_read_only_and_path_bounded(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            trigger = root / "selected-trigger"
            command = "target\0selected\0" + str(trigger) + "\0"
            self.assertFalse(
                roctx_markers(command, root)["markers"]["-ready"]["exists"]
            )
            marker = Path(str(trigger) + "-ready")
            marker.write_text("existing application marker")
            data = marker.read_bytes()
            self.assertTrue(roctx_markers(command, root)["markers"]["-ready"]["exists"])
            self.assertEqual(marker.read_bytes(), data)
            with self.assertRaises(ValueError):
                roctx_markers(
                    "target\0selected\0" + str(root.parent / "outside") + "\0", root
                )
            if os.name == "posix":
                marker.unlink()
                marker.symlink_to(root / "other")
                with self.assertRaises(ValueError):
                    roctx_markers(command, root)

    def test_actual_flat_inputs_require_attach_provider_and_original_script_bytes(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            build = root / "build"
            (build / "lib").mkdir(parents=True)
            (build / "bin").mkdir()
            (build / "bin/rocprofv3").write_bytes(b"inert entry")
            (build / "llvm/bin").mkdir(parents=True)
            for name in ["amdclang", "amdclang++"]:
                (build / "llvm/bin" / name).write_bytes(
                    b"inert compiler; never executed"
                )
            names = [
                "libamdhip64.so",
                "libhsa-runtime64.so",
                "librocprofiler-sdk.so",
                "librocprofiler-sdk-attach.so",
                "librocprofiler-register.so",
            ]
            for name in names:
                (build / "lib" / name).write_bytes(
                    b"inert; never executed " + name.encode()
                )
            original = root / probe.TEST_ROOT / "rocprofv3/attachment/fixture.sh"
            original.parent.mkdir(parents=True)
            data = b"inert source; never executed"
            original.write_bytes(data)
            contract = {
                "source_script_records": [
                    {
                        "path": "tests/rocprofv3/attachment/fixture.sh",
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                ]
            }
            first = probe.sdk_snapshot(root, root / "before.json", contract)
            self.assertTrue(first["complete"])
            (build / "bin/rocprofv3").write_bytes(b"inert entrY")
            launcher_changed = probe.sdk_snapshot(
                root, root / "launcher-changed.json", contract
            )
            self.assertNotEqual(
                probe.selected_input_bindings(first),
                probe.selected_input_bindings(launcher_changed),
            )
            (build / "bin/rocprofv3").write_bytes(b"inert entry")
            original.write_bytes(data + b"changed")
            self.assertFalse(
                probe.sdk_snapshot(root, root / "changed.json", contract)["complete"]
            )
            original.write_bytes(data)
            (build / "lib/librocprofiler-sdk-attach.so").unlink()
            missing = probe.sdk_snapshot(root, root / "missing.json", contract)
            self.assertFalse(missing["complete"])
            self.assertFalse(
                missing["required_provider_presence"]["librocprofiler-sdk-attach.so"]
            )

    def setUp(self):
        self.contract = json.loads(probe.CONTRACT.read_text())
        self.sha = "a" * 40
        branch = self.contract["branch"]
        self.env = dict(
            GITHUB_REPOSITORY="ROCm/TheRock",
            GITHUB_ACTOR="sa-faizal",
            GITHUB_EVENT_NAME="workflow_dispatch",
            GITHUB_REF="refs/heads/" + branch,
            GITHUB_WORKFLOW_REF="ROCm/TheRock/.github/workflows/test_artifacts.yml@refs/heads/"
            + branch,
            GITHUB_SHA=self.sha,
            GITHUB_RUN_ID="99999999999",
            GITHUB_RUN_ATTEMPT="1",
        )
        self.inputs = dict(
            profile_attachment_diagnostic=True,
            repository="ROCm/TheRock",
            ref=self.sha,
            artifact_run_id="41230700832",
            artifact_group="multi-arch-release",
            amdgpu_families="gfx94X-dcgpu",
            amdgpu_targets="gfx942",
            test_runs_on=self.contract["runner"],
            test_runs_on_cpu="",
            release_type="ci",
            build_variant="release",
            quartz_tracking_id="",
            tests_enabled=True,
            sanity_check_only_for_family=False,
            test_type="comprehensive",
            changed_projects="",
            skip_stages="",
            test_labels='["test:rocprofiler-sdk"]',
        )

    def git(self, *args):
        if args == ("rev-parse", "HEAD"):
            return self.sha
        return (
            "160000 commit "
            + self.contract["source_rocm_systems_sha"]
            + "\trocm-systems"
        )

    def test_exact_fixed_scope_and_component(self):
        with mock.patch.object(probe, "git", side_effect=self.git):
            probe.guard(self.inputs, self.env)
            nested = {
                key: value
                for key, value in self.inputs.items()
                if key
                not in {
                    "test_runs_on_cpu",
                    "tests_enabled",
                    "sanity_check_only_for_family",
                    "test_type",
                    "changed_projects",
                    "skip_stages",
                    "test_labels",
                }
            }
            nested.update(
                platform="linux", component=json.dumps(self.contract["component"])
            )
            self.env["NORMAL_IMAGE_PIN"] = self.contract["component"]["container_image"]
            probe.guard(nested, self.env, component=True)
            for key, value in [
                ("container_options", "--privileged"),
                ("additional_requirements_files", ["other.txt"]),
                ("platform", ["windows"]),
                ("expect_failure", True),
                ("gpu_enable_pal", "1"),
                ("rocm_kpack_debug", "0"),
            ]:
                component = {**self.contract["component"], key: value}
                nested["component"] = json.dumps(component)
                with self.subTest(component_field=key), self.assertRaises(ValueError):
                    probe.guard(nested, self.env, component=True)

    def test_actual_dispatch_payload_omits_only_expected_empty_strings(self):
        fixture = (
            Path(__file__).parent
            / "fixtures/rocprofiler_attachment_dispatch_inputs.json"
        )
        actual = json.loads(fixture.read_text())
        # The retained payload's source is fixed1f2c; mock only its SHA binding.
        actual["ref"] = self.sha
        missing = set(self.inputs) - set(actual)
        self.assertEqual(
            missing,
            {
                "quartz_tracking_id",
                "test_runs_on_cpu",
                "changed_projects",
                "skip_stages",
            },
        )
        with mock.patch.object(probe, "git", side_effect=self.git):
            probe.guard(actual, self.env)
            for key in missing:
                for value in [None, "unexpected", False, 0]:
                    with self.subTest(key=key, value=value), self.assertRaises(
                        ValueError
                    ):
                        probe.guard({**actual, key: value}, self.env)
            for key in [
                "artifact_run_id",
                "tests_enabled",
                "sanity_check_only_for_family",
            ]:
                changed = dict(actual)
                changed.pop(key)
                with self.subTest(missing_required=key), self.assertRaises(ValueError):
                    probe.guard(changed, self.env)

    def test_reject_other_repository_source_run_and_scope(self):
        with mock.patch.object(probe, "git", side_effect=self.git):
            for key, value in [
                ("profile_attachment_diagnostic", False),
                ("artifact_run_id", "41147801272"),
                ("release_type", "dev"),
                ("test_runs_on", "aws-linux-scale-rocm-prod"),
                ("test_labels", '["test:rocgdb-cpu"]'),
                ("ref", "main"),
                ("skip_stages", "math-libs"),
            ]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    probe.guard({**self.inputs, key: value}, self.env)
            for key, value in [
                ("GITHUB_REPOSITORY", "ROCm/rockrel"),
                ("GITHUB_RUN_ID", "41230700832"),
                ("GITHUB_ACTOR", "other"),
                ("GITHUB_REF", "refs/heads/main"),
            ]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    probe.guard(self.inputs, {**self.env, key: value})

    def test_declared_flags_and_all_quartz_sites_guarded(self):
        parent = yaml.load(
            (probe.ROOT / ".github/workflows/test_artifacts.yml").read_text(),
            Loader=yaml.BaseLoader,
        )
        component = yaml.load(
            (probe.ROOT / ".github/workflows/test_component.yml").read_text(),
            Loader=yaml.BaseLoader,
        )
        for trigger in ["workflow_dispatch", "workflow_call"]:
            self.assertEqual(
                parent["on"][trigger]["inputs"]["profile_attachment_diagnostic"][
                    "default"
                ],
                "false",
            )
        self.assertEqual(
            component["on"]["workflow_call"]["inputs"]["profile_attachment_diagnostic"][
                "default"
            ],
            "false",
        )
        sites = []
        for document in [parent, component]:
            for job in document["jobs"].values():
                if "notify_quartz" in job.get("uses", ""):
                    sites.append(job.get("if", ""))
                sites += [
                    step.get("if") or job.get("if", "")
                    for step in job.get("steps", [])
                    if "notify_quartz" in step.get("uses", "")
                ]
        self.assertEqual(len(sites), 4)
        self.assertTrue(
            all("profile_attachment_diagnostic != 'true'" in site for site in sites)
        )
        for site in sites:
            expression = site.replace(
                "fromJSON(toJSON(github.event.inputs))", "github.event.inputs"
            ).replace("always()", "true")
            for typed, inherited in [
                (False, ""),
                (True, ""),
                (False, "true"),
                (True, "true"),
            ]:
                context = {
                    "inputs": {"profile_attachment_diagnostic": typed},
                    "github": {
                        "event": {
                            "inputs": {
                                "profile_attachment_diagnostic": inherited,
                                "profile_normal_validation": "",
                            }
                        }
                    },
                    "vars": {"NOTIFY_QUARTZ_ENABLED": "true"},
                }
                self.assertEqual(
                    bool(evaluate(expression, context)),
                    not (typed or inherited == "true"),
                )


@unittest.skipUnless(
    sys.platform == "linux" and shutil.which("bash"), "Real Linux proc/Bash fixture"
)
class PredicateTests(unittest.TestCase):
    def test_owned_live_target_is_sampled_and_reused_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            (out / "poll-errors").mkdir()
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-c",
                    "import time;print('ready');time.sleep(5)",
                ],
                stdout=subprocess.PIPE,
                text=True,
            )
            try:
                self.assertEqual(child.stdout.readline().strip(), "ready")
                error = out / f"poll-errors/{os.getpid()}-{child.pid}-1.txt"
                error.write_text("")
                (out / "poll-events.tsv").write_text(
                    f"1\t2\t{os.getpid()}\t{child.pid}\t1\t{error}\n"
                )
                observer = TargetObserver(
                    out, os.getpid(), {Path(sys.executable)}, lambda path: "fixture"
                )
                observer.consume()
                observer.snapshot()
                self.assertTrue(observer.summary()["complete"], observer.summary())
                identity = proc_identity(child.pid)
                observer.groups = []
                with mock.patch(
                    "rocprofiler_attachment_observer.proc_identity",
                    return_value={
                        **identity,
                        "start_ticks": identity["start_ticks"] + 1,
                    },
                ):
                    observer.snapshot()
                self.assertFalse(observer.summary()["complete"])
            finally:
                child.terminate()
                child.wait()
                child.stdout.close()

    def script(self, text: str, output: Path, hook: bool = True):
        env = os.environ.copy()
        env["ATTACHMENT_DIAG_ROOT"] = str(output)
        if hook:
            env["BASH_ENV"] = str(probe.HOOK)
        else:
            env.pop("BASH_ENV", None)
        return subprocess.run(
            ["bash", "--noprofile", "--norc", "-c", text],
            env=env,
            capture_output=True,
            text=True,
        )

    def test_actual_expanded_multithread_paths_match_and_status_zero(self):
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            (out / "poll-errors").mkdir()
            app = out / "threads.py"
            app.write_text(
                "import ctypes,threading,time\nctypes.CDLL(None).prctl(15,b'rocp-bg-attach',0,0,0)\n[threading.Thread(target=lambda:time.sleep(5),daemon=True).start() for _ in range(2)]\nprint('ready',flush=True)\ntime.sleep(5)\n"
            )
            body = f"""set -e
"{sys.executable}" "{app}" > "{out}/ready" &
pid=$!
trap 'kill "$pid" 2>/dev/null || :; wait "$pid" 2>/dev/null || :' EXIT
while [[ ! -s "{out}/ready" ]]; do sleep 0.02; done
paths=(/proc/$pid/task/*/comm)
[[ ${{#paths[@]}} -ge 3 ]]
if grep -ql rocp-bg-attach "${{paths[@]}}" 2>/dev/null; then echo ready; else exit 91; fi
"""
            result = self.script(body, out)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "ready\n")
            row = (out / "poll-events.tsv").read_text().strip().split("\t")
            self.assertEqual(row[4], "0")

    def test_nomatch_read_error_negation_and_capture_write_failure_keep_outcomes(self):
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            (out / "poll-errors").mkdir()
            body = """set -e
if grep -ql rocp-bg-attach /proc/$BASHPID/task/*/comm 2>/dev/null; then exit 91; else echo "nomatch:$?"; fi
if ! grep -ql rocp-bg-attach /proc/2147483647/task/*/comm 2>/dev/null; then echo negated; fi
if grep -ql rocp-bg-attach /proc/2147483647/task/*/comm 2>/dev/null; then exit 92; else echo "missing:$?"; fi
"""
            baseline, candidate = self.script(body, out, False), self.script(body, out)
            self.assertEqual(
                (candidate.returncode, candidate.stdout, candidate.stderr),
                (baseline.returncode, baseline.stdout, baseline.stderr),
            )
            rows = [
                line.split("\t")
                for line in (out / "poll-events.tsv").read_text().splitlines()
            ]
            self.assertEqual([row[4] for row in rows], ["1", "2", "2"])
            self.assertIn("No such file", Path(rows[-1][-1]).read_text())
            failed_capture = self.script(body, out / "no-capture-directory")
            self.assertEqual(
                (
                    failed_capture.returncode,
                    failed_capture.stdout,
                    failed_capture.stderr,
                ),
                (baseline.returncode, baseline.stdout, baseline.stderr),
            )

    def test_unrelated_grep_is_unmodified(self):
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            (out / "poll-errors").mkdir()
            source = out / "input.txt"
            source.write_text("one\nneedle\n")
            body = f"set -e; grep -n needle '{source}'; if ! grep absent '{source}'; then echo absent; fi"
            a, b = self.script(body, out, False), self.script(body, out)
            self.assertEqual(
                (a.returncode, a.stdout, a.stderr), (b.returncode, b.stdout, b.stderr)
            )
            self.assertFalse((out / "poll-events.tsv").exists())

    def test_unattributed_pid_is_not_clean_and_no_maps_sample_is_taken(self):
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            (out / "poll-errors").mkdir()
            error = out / f"poll-errors/{os.getpid()}-1-1.txt"
            error.write_text("fixture read error")
            (out / "poll-events.tsv").write_text(
                f"1\t2\t{os.getpid()}\t1\t2\t{error}\n"
            )
            observer = TargetObserver(
                out, os.getpid(), {Path(sys.executable)}, lambda path: "fixture"
            )
            observer.consume()
            observer.snapshot()
            self.assertFalse(observer.summary()["complete"])
            self.assertEqual(len(observer.rejected), 1)
            self.assertEqual(observer.sample_count, 0)
            self.assertEqual(proc_identity(os.getpid())["pid"], os.getpid())


@unittest.skipUnless(
    shutil.which("ctest") and shutil.which("cmake"),
    "Real tiny CTest submission fixture",
)
class LocalSubmissionTests(unittest.TestCase):
    def test_stalled_headers_are_bounded_before_body_handler(self):
        for partial in [b"", b"PUT /partial HTTP/1.0\r\nX-Incomplete:"]:
            with self.subTest(partial=partial), tempfile.TemporaryDirectory() as raw:
                sink = LoopbackSink(Path(raw) / "sink")
                sink.start()
                with socket.create_connection(
                    ("127.0.0.1", sink.server.server_port)
                ) as client:
                    if partial:
                        client.sendall(partial)
                    time.sleep(0.1)
                    began = time.monotonic()
                    receipt = sink.close()
                self.assertLess(time.monotonic() - began, 5)
                self.assertFalse(receipt["complete"])
                self.assertTrue(receipt["errors"])

    def test_bad_url_is_blocked_before_ctest_is_called(self):
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            dashboard, original = out / "dashboard.cmake", out / "original.py"
            dashboard.write_text(
                'set(CTEST_SUBMIT_URL "https://my.cdash.org/submit.php")\n'
            )
            argv = [
                "ctest",
                "-S",
                str(dashboard),
                "--verbose",
                "--output-on-failure",
                "--no-tests=error",
            ]
            original.write_text(
                "import subprocess\nsubprocess.run(" + repr(argv) + ")\n"
            )
            with mock.patch("subprocess.run") as actual_ctest:
                with self.assertRaises(ValueError):
                    run_original(
                        original,
                        ["--enable-cdash"],
                        dashboard,
                        "http://127.0.0.1:12345/nonce",
                        out / "entry.json",
                    )
                actual_ctest.assert_not_called()

    def test_original_python_exit_and_flags_are_preserved(self):
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            original = out / "original.py"
            original.write_text(
                "import sys\nassert sys.argv[1:] == ['--enable-cdash']\nraise SystemExit(17)\n"
            )
            saved_run = subprocess.run
            with self.assertRaises(SystemExit) as result:
                run_original(
                    original,
                    ["--enable-cdash"],
                    out / "dashboard.cmake",
                    "http://127.0.0.1:12345/nonce",
                    out / "entry.json",
                )
            self.assertEqual(result.exception.code, 17)
            self.assertIs(subprocess.run, saved_run)

    def test_generated_url_rejects_external_or_other_loopback_target(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "dashboard.cmake"
            url = "http://127.0.0.1:12345/nonce?project=rocprofiler-sdk-alt"
            path.write_text(f'set(CTEST_SUBMIT_URL "{url}")\n')
            self.assertTrue(probe.validate_dashboard(path, url)["loopback_only"])
            for other in [
                "https://my.cdash.org/submit.php",
                "http://127.0.0.1:12346/other",
            ]:
                path.write_text(f'set(CTEST_SUBMIT_URL "{other}")\n')
                with self.assertRaises(ValueError):
                    probe.validate_dashboard(path, url)

    def test_stalled_partial_upload_has_bounded_close_and_failure_index(self):
        with tempfile.TemporaryDirectory() as raw:
            sink = LoopbackSink(Path(raw) / "sink")
            sink.start()
            with socket.create_connection(
                ("127.0.0.1", sink.server.server_port)
            ) as client:
                client.sendall(
                    f"PUT /{sink.nonce} HTTP/1.0\r\nContent-Length: 99\r\n\r\nx".encode()
                )
                time.sleep(0.1)
                began = time.monotonic()
                receipt = sink.close()
            self.assertLess(time.monotonic() - began, 5)
            self.assertFalse(receipt["complete"])
            self.assertTrue(receipt["errors"])
            self.assertTrue((Path(raw) / "sink/index.json").is_file())

    def test_real_ctest_submission_stays_on_loopback(self):
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            source, build = out / "source", out / "build"
            source.mkdir()
            (source / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.21)\nproject(tiny NONE)\nenable_testing()\nadd_test(NAME tiny COMMAND ${CMAKE_COMMAND} -E true)\n"
            )
            subprocess.run(
                ["cmake", "-S", str(source), "-B", str(build)],
                check=True,
                capture_output=True,
            )
            sink = LoopbackSink(out / "sink")
            sink.start()
            proxy = LoopbackSink(out / "dummy-proxy")
            proxy.start()
            try:
                script = out / "submit.cmake"
                script.write_text(
                    f"""set(CTEST_PROJECT_NAME "attachment-fixture")
set(CTEST_SITE "localhost")
set(CTEST_BUILD_NAME "tiny")
set(CTEST_SOURCE_DIRECTORY "{source.as_posix()}")
set(CTEST_BINARY_DIRECTORY "{build.as_posix()}")
set(CTEST_SUBMIT_URL "{sink.url}")
set(CTEST_DROP_METHOD "https")
ctest_start(Experimental)
ctest_test(RETURN_VALUE tested PARALLEL_LEVEL 8)
ctest_submit(PARTS Test RETURN_VALUE submitted)
if(NOT tested EQUAL 0 OR NOT submitted EQUAL 0)
message(FATAL_ERROR "local submission fixture failed")
endif()
"""
                )
                driver = out / "original.py"
                argv = [
                    "ctest",
                    "-S",
                    str(script),
                    "--verbose",
                    "--output-on-failure",
                    "--no-tests=error",
                ]
                driver.write_text(
                    "import subprocess\nsubprocess.run("
                    + repr(argv)
                    + ", check=True, capture_output=True, text=True)\n"
                )
                controlled = {
                    **os.environ,
                    "http_proxy": f"http://127.0.0.1:{proxy.server.server_port}",
                    "HTTP_PROXY": f"http://127.0.0.1:{proxy.server.server_port}",
                    "no_proxy": "",
                    "NO_PROXY": "",
                }
                with mock.patch.dict(
                    os.environ, probe.loopback_bypass(controlled), clear=True
                ):
                    run_original(
                        driver, ["--enable-cdash"], script, sink.url, out / "entry.json"
                    )
                entry = json.loads((out / "entry.json").read_text())
                self.assertTrue(entry["before_ctest_launch"])
                self.assertEqual(entry["original_command"], argv)
            finally:
                receipt = sink.close()
                proxy_receipt = proxy.close()
            self.assertTrue(receipt["complete"], receipt)
            self.assertFalse(receipt["forwarded"])
            self.assertGreater(receipt["total_bytes"], 100)
            self.assertFalse(proxy_receipt["records"])
            self.assertFalse(proxy_receipt["errors"])


if __name__ == "__main__":
    unittest.main()
