# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise installed HIP test filtering and failure reporting with inert CTest cases."""

import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


@pytest.fixture
def installed_ctest_prefix(tmp_path):
    compiler = shutil.which("cmake")
    assert compiler
    tests = tmp_path / "sdk/share/hip/catch_tests"
    tests.mkdir(parents=True)
    (tmp_path / "sdk/bin").mkdir()
    (tests / "CTestTestfile.cmake").write_text(
        f'add_test(Unit_Fail "{compiler}" -E false)\n'
        f'add_test(Unit_Pass "{compiler}" -E true)\n'
        f'add_test(Unit_Full "{compiler}" -E false)\n'
        "set_tests_properties(Unit_Pass Unit_Fail PROPERTIES LABELS smoke)\n"
        "set_tests_properties(Unit_Full PROPERTIES LABELS full)\n"
    )
    return tmp_path / "sdk"


def run_hip_runner(prefix, *args):
    runner = (
        Path(__file__).resolve().parents[1] / "test_executable_scripts/test_hiptests.py"
    )
    return subprocess.run(
        [sys.executable, "-B", str(runner), *args],
        env=os.environ
        | {"THEROCK_BIN_DIR": str(prefix / "bin"), "AMDGPU_FAMILIES": "gfx90a"},
        text=True,
        capture_output=True,
    )


def test_existing_hip_runner_filters_and_retains_real_ctest_results(
    installed_ctest_prefix, tmp_path
):
    report = tmp_path / "junit.xml"
    run = run_hip_runner(
        installed_ctest_prefix,
        "--label-regex",
        "smoke",
        "--test-regex",
        "Pass",
        "--junit",
        str(report),
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert [case.attrib["name"] for case in ET.parse(report).iter("testcase")] == [
        "Unit_Pass"
    ]


@pytest.mark.parametrize(
    "repeat, failure_run, expected_count",
    [(None, 0, 1), ("1", 0, 1), ("3", 0, 3), ("5", 2, 2), ("10000", 1, 1)],
)
def test_repeat_counts_selected_cases_and_stops_on_failure(
    installed_ctest_prefix, tmp_path, repeat, failure_run, expected_count
):
    counter = tmp_path / "counter.txt"
    script = tmp_path / "count.py"
    script.write_text(
        "import os, sys\n"
        "from pathlib import Path\n"
        "assert os.environ['THEROCK_REQUIRE_HIP_YAML_ENTRIES'] == '1'\n"
        "counter = Path(sys.argv[1])\n"
        "count = int(counter.read_text()) + 1 if counter.exists() else 1\n"
        "counter.write_text(str(count))\n"
        "raise SystemExit(count == int(sys.argv[2]))\n"
    )
    tests = installed_ctest_prefix / "share/hip/catch_tests/CTestTestfile.cmake"
    with tests.open("a") as output:
        output.write(
            f'add_test(Unit_Repeat "{sys.executable}" -B "{script}" "{counter}" {failure_run})\n'
            "set_tests_properties(Unit_Repeat PROPERTIES LABELS smoke)\n"
        )
    report = tmp_path / "junit.xml"
    args = [] if repeat is None else ["--repeat", repeat]
    run = run_hip_runner(
        installed_ctest_prefix,
        "--label-regex",
        "smoke",
        "--test-regex",
        "^Unit_Repeat$",
        "--junit",
        str(report),
        *args,
    )
    assert run.returncode == (1 if failure_run else 0), run.stdout + run.stderr
    assert int(counter.read_text()) == expected_count
    assert [case.get("name") for case in ET.parse(report).iter("testcase")] == [
        "Unit_Repeat"
    ]
    if repeat is None or repeat == "1":
        assert "--repeat" not in run.stderr
    else:
        assert f"--repeat until-fail:{repeat}" in run.stderr


@pytest.mark.parametrize("repeat", ["0", "-1", "10001", "1.5", "nan", "abc", ""])
def test_invalid_repeat_fails_before_running_ctest(installed_ctest_prefix, repeat):
    run = run_hip_runner(installed_ctest_prefix, f"--repeat={repeat}")
    assert run.returncode == 2
    assert "--repeat: must be an integer from 1 to 10000" in run.stderr
    assert "++ Exec" not in run.stderr


def test_hip_runner_continues_after_a_failed_case_and_returns_failure(
    installed_ctest_prefix, tmp_path
):
    report = tmp_path / "junit.xml"
    run = run_hip_runner(
        installed_ctest_prefix, "--label-regex", "smoke", "--junit", str(report)
    )
    assert run.returncode != 0
    assert len(list(ET.parse(report).iter("testcase"))) == 2
    assert len(list(ET.parse(report).iter("failure"))) == 1


def test_empty_hip_selection_cannot_report_success(installed_ctest_prefix):
    run = run_hip_runner(installed_ctest_prefix, "--test-regex", "DoesNotExist")
    assert run.returncode != 0
    assert "No tests were found" in run.stdout + run.stderr


def test_empty_label_explicitly_selects_all_labels(installed_ctest_prefix, tmp_path):
    report = tmp_path / "junit.xml"
    run = run_hip_runner(
        installed_ctest_prefix, "--label-regex", "", "--junit", str(report)
    )
    assert run.returncode != 0
    assert len(list(ET.parse(report).iter("testcase"))) == 3


def test_all_disabled_hip_selection_cannot_report_success(
    installed_ctest_prefix, tmp_path
):
    tests = installed_ctest_prefix / "share/hip/catch_tests/CTestTestfile.cmake"
    with tests.open("a") as output:
        output.write(
            'add_test(Unit_Disabled "cmake" -E true)\n'
            "set_tests_properties(Unit_Disabled PROPERTIES DISABLED TRUE)\n"
        )
    report = tmp_path / "junit.xml"
    run = run_hip_runner(
        installed_ctest_prefix,
        "--label-regex",
        "",
        "--test-regex",
        "Unit_Disabled",
        "--junit",
        str(report),
    )
    assert run.returncode != 0
    assert [case.get("status") for case in ET.parse(report).iter("testcase")] == [
        "disabled"
    ]


def test_all_runtime_skipped_hip_selection_cannot_report_success(
    installed_ctest_prefix, tmp_path
):
    tests = installed_ctest_prefix / "share/hip/catch_tests/CTestTestfile.cmake"
    with tests.open("a") as output:
        output.write(
            f'add_test(Unit_Skipped "{sys.executable}" -c "raise SystemExit(77)")\n'
            "set_tests_properties(Unit_Skipped PROPERTIES SKIP_RETURN_CODE 77)\n"
        )
    report = tmp_path / "junit.xml"
    run = run_hip_runner(
        installed_ctest_prefix,
        "--label-regex",
        "",
        "--test-regex",
        "Unit_Skipped",
        "--junit",
        str(report),
    )
    assert run.returncode != 0
    assert "No selected HIP test executed" in run.stderr
    assert len(list(ET.parse(report).iter("skipped"))) == 1
