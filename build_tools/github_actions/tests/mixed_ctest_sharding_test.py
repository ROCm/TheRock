# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise mixed scheduling through real CTest selection and child processes."""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "test_executable_scripts" / "test_runner.py"
)


def load_runner(monkeypatch, tmp_path, shard, total):
    monkeypatch.setenv("THEROCK_BIN_DIR", str(tmp_path))
    monkeypatch.setenv("TEST_COMPONENT", "hipkernelprovider")
    monkeypatch.setenv("TEST_TYPE", "standard")
    monkeypatch.setenv("SHARD_INDEX", str(shard))
    monkeypatch.setenv("TOTAL_SHARDS", str(total))
    monkeypatch.delenv("AMDGPU_FAMILIES", raising=False)
    spec = importlib.util.spec_from_file_location("mixed_runner", SCRIPT)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


@pytest.fixture
def ctest_payload(tmp_path):
    if not shutil.which("ctest"):
        pytest.skip("CTest is required")
    root = tmp_path / "hip_kernel_provider"
    root.mkdir()
    probe = root / "probe.py"
    probe.write_text(
        "import json, os, sys\n"
        "name = sys.argv[1]\n"
        "if os.environ.get('MIXED_FIXTURE_FILE') and name in ('gtest_suite', 'pytest_suite'):\n"
        "    assert os.path.isfile(os.environ['MIXED_FIXTURE_FILE'])\n"
        "sharded = '--gtest_filter=*' in sys.argv\n"
        "if not sharded:\n"
        "    assert 'GTEST_TOTAL_SHARDS' not in os.environ\n"
        "    assert 'GTEST_SHARD_INDEX' not in os.environ\n"
        "    assert 'GTEST_SHARD_STATUS_FILE' not in os.environ\n"
        "cases = list(range(7)) if 'gtest' in name else [None]\n"
        "if sharded:\n"
        "    cases = cases[int(os.environ['GTEST_SHARD_INDEX'])::int(os.environ['GTEST_TOTAL_SHARDS'])]\n"
        "with open(os.environ['MIXED_TEST_LOG'], 'a') as out:\n"
        "    for case in cases: out.write(json.dumps([name, case]) + '\\n')\n"
        "sys.exit(1 if name == os.environ.get('MIXED_FAIL') else 0)\n"
    )
    names = ("gtest_suite", "unfiltered_gtest", "host.a[0]", "host_a0", "pytest_suite")
    entries = []
    for name in (*names, "excluded"):
        args = ' "--gtest_filter=*"' if name == "gtest_suite" else ""
        entries.extend(
            [
                f'add_test("{name}" "{sys.executable}" "{probe}" "{name}"{args})',
                f'set_tests_properties("{name}" PROPERTIES LABELS "'
                + ("integration" if name == "excluded" else "standard")
                + '")',
            ]
        )
    (root / "CTestTestfile.cmake").write_text("\n".join(entries) + "\n")
    return names


@pytest.mark.parametrize("total", [1, 2, 3, 10])
def test_mixed_shards_preserve_every_case_once(
    monkeypatch, tmp_path, ctest_payload, total
):
    log = tmp_path / "visits.jsonl"
    monkeypatch.setenv("MIXED_TEST_LOG", str(log))
    monkeypatch.setenv("GTEST_SHARD_STATUS_FILE", str(tmp_path / "status"))
    for shard in range(1, total + 1):
        runner = load_runner(monkeypatch, tmp_path, shard, total)
        assert runner.main() == 0
    visits = [tuple(json.loads(line)) for line in log.read_text().splitlines()]
    expected = {
        (name, case)
        for name in ctest_payload
        for case in (range(7) if "gtest" in name else [None])
    }
    assert len(visits) == len(set(visits))
    assert set(visits) == expected


@pytest.mark.parametrize("failing_entry", ["gtest_suite", "pytest_suite"])
def test_test_failure_is_not_hidden(
    monkeypatch, tmp_path, ctest_payload, failing_entry
):
    monkeypatch.setenv("MIXED_TEST_LOG", str(tmp_path / "visits.jsonl"))
    monkeypatch.setenv("MIXED_FAIL", failing_entry)
    runner = load_runner(monkeypatch, tmp_path, 1, 1)
    assert runner.main() == 8
    visits = (tmp_path / "visits.jsonl").read_text()
    assert "gtest_suite" in visits and "pytest_suite" in visits


def test_ctest_keeps_required_fixtures(monkeypatch, tmp_path, ctest_payload):
    cmake = shutil.which("cmake")
    if not cmake:
        pytest.skip("CMake is required")
    root = tmp_path / "hip_kernel_provider"
    marker = root / "fixture-ready"
    monkeypatch.setenv("MIXED_TEST_LOG", str(tmp_path / "visits.jsonl"))
    monkeypatch.setenv("MIXED_FIXTURE_FILE", str(marker))
    with (root / "CTestTestfile.cmake").open("a") as manifest:
        manifest.write(
            f'add_test(setup "{cmake}" -E touch "{marker}")\n'
            f'add_test(cleanup "{cmake}" -E rm -f "{marker}")\n'
            "set_tests_properties(setup PROPERTIES FIXTURES_SETUP ready)\n"
            "set_tests_properties(cleanup PROPERTIES FIXTURES_CLEANUP ready)\n"
            "set_tests_properties(gtest_suite pytest_suite PROPERTIES FIXTURES_REQUIRED ready)\n"
        )
    for shard in (1, 2):
        runner = load_runner(monkeypatch, tmp_path, shard, 2)
        assert runner.main() == 0
        assert not marker.exists()


@pytest.mark.parametrize("shard,total", [(0, 2), (3, 2), (1, 0)])
def test_invalid_shard_is_rejected(monkeypatch, tmp_path, shard, total):
    runner = load_runner(monkeypatch, tmp_path, shard, total)
    with pytest.raises(ValueError, match="SHARD_INDEX"):
        runner.mixed_ctest_plan(["ctest"], shard, total, {})


def test_compounded_stride_is_rejected(monkeypatch, tmp_path):
    runner = load_runner(monkeypatch, tmp_path, 1, 2)
    with pytest.raises(ValueError, match="stride"):
        runner.mixed_ctest_plan(["ctest", "--tests-information", "1,,2"], 1, 2, {})


def test_discovery_failure_is_not_hidden(monkeypatch, tmp_path):
    runner = load_runner(monkeypatch, tmp_path, 1, 2)
    with pytest.raises(subprocess.CalledProcessError):
        runner.mixed_ctest_plan(
            ["ctest", "--test-dir", str(tmp_path / "missing")], 1, 2, {}
        )
