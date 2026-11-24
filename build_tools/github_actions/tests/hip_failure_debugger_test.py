# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise supplementary diagnostics with real host-only CTest/GDB subjects."""

import json
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from hiptests_runner_test import installed_ctest_prefix, run_hip_runner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "test_executable_scripts"))
import hip_failure_debugger as debugger

pytestmark = pytest.mark.skipif(
    platform.system() != "Linux", reason="Host CTest/GDB diagnostics require Linux"
)


@pytest.fixture
def crash_prefix(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    prefix = tmp_path / "sdk"
    (prefix / "bin").mkdir(parents=True)
    tests = prefix / "share/hip/catch_tests"
    (source / "probe.c").write_text(
        r"""
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
extern char **environ;
void ignore_signal(int signal_number) {}
int main(int argc, char **argv) {
  FILE *counter = fopen(argv[2], "a");
  fprintf(counter, "%s\n", argv[1]);
  fclose(counter);
  char pid_path[4096];
  snprintf(pid_path, sizeof(pid_path), "%s.%s.pid", argv[2], argv[1]);
  FILE *pid_file = fopen(pid_path, "a");
  fprintf(pid_file, "%d\n", getpid());
  fclose(pid_file);
  char environment_path[4096];
  snprintf(environment_path, sizeof(environment_path), "%s.%s.env", argv[2], argv[1]);
  FILE *environment = fopen(environment_path, "a");
  char directory[4096];
  fprintf(environment, "CWD=%s\n", getcwd(directory, sizeof(directory)));
  for (char **entry = environ; *entry; ++entry) fprintf(environment, "%s\n", *entry);
  fprintf(environment, "END\n");
  fclose(environment);
  if (getchar() != EOF) return 14;
  printf("prepared=%s property=%s changes=%s\n",
    getenv("THEROCK_REQUIRE_HIP_YAML_ENTRIES"), getenv("PROPERTY"), getenv("CHANGES"));
  fflush(stdout);
  if (!strcmp(argv[1], "pass")) return 0;
  if (!strcmp(argv[1], "skip")) return 77;
  if (!strcmp(argv[1], "fail")) return 12;
  if (!strcmp(argv[1], "sleep")) { sleep(30); return 0; }
  signal(SIGUSR1, ignore_signal);
  raise(SIGUSR1);
  raise(SIGSEGV);
  return 0;
}
"""
    )
    (source / "CMakeLists.txt").write_text(
        f"""cmake_minimum_required(VERSION 3.25)
project(host_probe C)
enable_testing()
add_executable(probe probe.c)
foreach(mode pass crash skip disabled fail)
  add_test(NAME ${{mode}} COMMAND probe ${{mode}} "{tmp_path}/counter")
endforeach()
set_tests_properties(disabled PROPERTIES DISABLED TRUE)
set_tests_properties(skip PROPERTIES SKIP_RETURN_CODE 77)
set_tests_properties(crash PROPERTIES
  ENVIRONMENT "PROPERTY=literal $() \\\"quote\\\";CHANGES=initial"
  ENVIRONMENT_MODIFICATION "CHANGES=string_append:ignored;CHANGES=reset:;CHANGES=string_append::final")
"""
    )
    for command in (
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(tests),
            "-DCMAKE_C_COMPILER=/usr/bin/cc",
        ],
        ["cmake", "--build", str(tests)],
    ):
        subprocess.run(command, check=True, capture_output=True, text=True)
    (tests / ".gdbinit").write_text(f"shell touch {tmp_path}/unsafe-init\n")
    (tests / "probe-gdb.py").write_text(
        f"open({str(tmp_path / 'unsafe-autoload')!r}, 'w').close()\n"
    )
    return prefix


def read_report(tmp_path):
    return json.loads((tmp_path / "original-gdb/report.json").read_text())


def test_only_original_failed_executed_cases_are_debugged(crash_prefix, tmp_path):
    report = tmp_path / "original.xml"
    run = run_hip_runner(crash_prefix, "--junit", str(report), "--gdb-on-failure")
    assert run.returncode == 1, run.stdout + run.stderr
    cases = {case.get("name"): case for case in ET.parse(report).iter("testcase")}
    assert cases["pass"].get("status") == "run"
    assert cases["disabled"].get("status") == "disabled"
    assert cases["skip"].find("skipped") is not None
    assert cases["crash"].find("failure") is not None
    observations = read_report(tmp_path)
    assert [case["name"] for case in observations["cases"]] == ["crash", "fail"]
    assert observations["cases"][0]["status"] == "completed", observations
    crash_log = (tmp_path / "original-gdb/case-001.txt").read_text()
    assert "SIGSEGV" in crash_log
    assert "#0" in crash_log and "main" in crash_log
    assert "prepared=1 property=literal $()" in crash_log
    assert "changes=initial:final" in crash_log
    assert "Start Addr" in crash_log
    assert "rip" in crash_log or "pc" in crash_log
    assert not (tmp_path / "unsafe-init").exists()
    assert not (tmp_path / "unsafe-autoload").exists()
    counts = (tmp_path / "counter").read_text().splitlines()
    assert counts.count("crash") == counts.count("fail") == 2
    assert counts.count("pass") == counts.count("skip") == 1
    assert "disabled" not in counts
    original_log = (
        crash_prefix / "share/hip/catch_tests/Testing/Temporary/LastTest.log"
    ).read_text()
    assert "prepared=1" in original_log and "crash" in original_log
    snapshots = (tmp_path / "counter.crash.env").read_text().split("END\n")
    assert sorted(snapshots[0].splitlines()) == sorted(snapshots[1].splitlines())


def test_default_has_no_debugger_rerun(crash_prefix, tmp_path):
    run = run_hip_runner(crash_prefix, "--junit", str(tmp_path / "original.xml"))
    assert run.returncode == 1
    assert not (tmp_path / "original-gdb").exists()
    assert (tmp_path / "counter").read_text().splitlines().count("crash") == 1


def test_fixture_semantics_are_reported_without_a_rerun(crash_prefix, tmp_path):
    tests = crash_prefix / "share/hip/catch_tests/CTestTestfile.cmake"
    with tests.open("a") as output:
        output.write("set_tests_properties(crash PROPERTIES FIXTURES_SETUP setup)\n")
    run = run_hip_runner(
        crash_prefix,
        "--test-regex",
        "^crash$",
        "--junit",
        str(tmp_path / "original.xml"),
        "--gdb-on-failure",
    )
    assert run.returncode == 1
    entry = read_report(tmp_path)["cases"][0]
    assert entry["status"] == "unavailable" and "FIXTURES_SETUP" in entry["error"]
    assert (tmp_path / "counter").read_text().splitlines() == ["crash"]


def test_disabled_skip_and_pass_do_not_invoke_debugger(crash_prefix, tmp_path):
    run = run_hip_runner(
        crash_prefix,
        "--test-regex",
        "^(disabled|skip|pass)$",
        "--junit",
        str(tmp_path / "original.xml"),
        "--gdb-on-failure",
    )
    assert run.returncode == 0
    assert not (tmp_path / "original-gdb").exists()


def test_gdb_requires_a_junit_report(crash_prefix):
    run = run_hip_runner(crash_prefix, "--gdb-on-failure")
    assert run.returncode == 2
    assert "requires Linux and --junit" in run.stderr


def test_external_driver_commands_are_unavailable(installed_ctest_prefix, tmp_path):
    run = run_hip_runner(
        installed_ctest_prefix,
        "--test-regex",
        "^Unit_Fail$",
        "--junit",
        str(tmp_path / "original.xml"),
        "--gdb-on-failure",
    )
    assert run.returncode == 1
    entry = read_report(tmp_path)["cases"][0]
    assert entry["status"] == "unavailable"
    assert "direct executables" in entry["error"]


def test_stripped_crashes_retain_addresses_mappings_and_registers(
    crash_prefix, tmp_path
):
    subprocess.run(
        ["/usr/bin/strip", str(crash_prefix / "share/hip/catch_tests/probe")],
        check=True,
    )
    run = run_hip_runner(
        crash_prefix,
        "--test-regex",
        "^crash$",
        "--junit",
        str(tmp_path / "original.xml"),
        "--gdb-on-failure",
    )
    assert run.returncode == 1
    entry = read_report(tmp_path)["cases"][0]
    assert entry["status"] == "completed", entry
    log = (tmp_path / "original-gdb" / entry["log"]).read_text()
    assert "SIGSEGV" in log and "#0" in log and "0x" in log
    assert "Start Addr" in log
    assert "rip" in log or "pc" in log


def test_ctest_and_debugger_timeouts_keep_the_original_failure(crash_prefix, tmp_path):
    tests = crash_prefix / "share/hip/catch_tests"
    with (tests / "CTestTestfile.cmake").open("a") as output:
        output.write(
            f'add_test(sleep "{tests}/probe" sleep "{tmp_path}/counter")\n'
            "set_tests_properties(sleep PROPERTIES TIMEOUT 0.5)\n"
        )
    run = run_hip_runner(
        crash_prefix,
        "--test-regex",
        "^sleep$",
        "--junit",
        str(tmp_path / "original.xml"),
        "--gdb-on-failure",
    )
    assert run.returncode == 1
    original = next(ET.parse(tmp_path / "original.xml").iter("testcase"))
    assert original.find("failure").get("message") == "Timeout"
    entry = read_report(tmp_path)["cases"][0]
    assert entry["status"] == "timeout", entry
    for pid in (tmp_path / "counter.sleep.pid").read_text().splitlines():
        stat = Path(f"/proc/{pid}/stat")
        assert (
            not stat.exists() or stat.read_text().rpartition(")")[2].split()[0] == "Z"
        )


def test_bounded_commands_preserve_timeouts_and_output_limits(tmp_path):
    for script, expected in (
        ("import time; time.sleep(30)", "timeout"),
        ("print('x' * 10000)", "output_limit"),
    ):
        log, result = debugger._run_bounded(
            [sys.executable, "-B", "-c", script],
            cwd=tmp_path,
            env=os.environ.copy(),
            timeout=0.2,
            limit=256,
        )
        assert result["status"] == expected
        assert result["returncode"] < 0
        assert len(log) <= 256


def test_environment_modification_order_and_separators():
    base = {"BASE": "base", "REMOVE": "remove"}
    properties = {
        "ENVIRONMENT": ["BASE=property", "RESET=property"],
        "ENVIRONMENT_MODIFICATION": [
            "BASE=unset:",
            "BASE=reset:",
            "RESET=set:modified",
            "RESET=reset:",
            "REMOVE=unset:",
            "ABSENT=set:value",
            "ABSENT=reset:",
            "STRING=set:center",
            "STRING=string_prepend:left",
            "STRING=string_append:right",
            "PATH=path_list_append:a",
            "PATH=path_list_append:b",
            "PATH=path_list_prepend:c",
            "LIST=cmake_list_append:a",
            "LIST=cmake_list_append:b",
            "LIST=cmake_list_prepend:c",
        ],
    }
    assert debugger._test_environment(base, properties) == {
        "BASE": "property",
        "RESET": "property",
        "STRING": "leftcenterright",
        "PATH": f"c{os.pathsep}a{os.pathsep}b",
        "LIST": "c;a;b",
    }


def test_exhausted_case_budget_does_not_execute_failures(
    crash_prefix, tmp_path, monkeypatch
):
    junit = tmp_path / "original.xml"
    tests = crash_prefix / "share/hip/catch_tests"
    inventory, inventory_status = debugger.collect_inventory(
        ["ctest", "--test-dir", str(tests), "--show-only=json-v1"],
        cwd=tmp_path,
        env=os.environ.copy(),
    )
    assert run_hip_runner(crash_prefix, "--junit", str(junit)).returncode == 1
    monkeypatch.setattr(debugger, "MAX_CASES", 0)
    debugger.capture_failures(
        junit=junit,
        inventory=inventory,
        inventory_status=inventory_status,
        test_root=tests,
        env=os.environ.copy(),
    )
    entries = read_report(tmp_path)["cases"]
    assert [entry["name"] for entry in entries] == ["crash", "fail"]
    assert all(
        entry["status"] == "unavailable" and "budget" in entry["error"]
        for entry in entries
    )
    assert (tmp_path / "counter").read_text().splitlines().count("crash") == 1


def test_invalid_original_report_is_a_diagnostic_error(tmp_path):
    junit = tmp_path / "original.xml"
    junit.write_text("invalid xml")
    debugger.capture_failures(
        junit=junit,
        inventory=b"",
        inventory_status={},
        test_root=tmp_path,
        env={},
    )
    assert read_report(tmp_path)["error"]
    assert not read_report(tmp_path)["cases"]
    assert junit.read_text() == "invalid xml"
