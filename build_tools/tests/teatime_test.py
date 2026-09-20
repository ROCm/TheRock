# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Focused tests for privacy-safe teatime component spans."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import teatime


SCRIPT = Path(__file__).parent.parent / "teatime.py"
REPO_ROOT = SCRIPT.parent.parent


def _run(tmp_path: Path, *, component: str = "rocblas") -> subprocess.CompletedProcess:
    span_path = tmp_path / "spans.jsonl"
    environment = os.environ.copy()
    environment["THEROCK_RESOURCE_SPANS_FILE"] = str(span_path)
    environment["PRIVATE_BUILD_VALUE"] = "must-not-appear"
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--telemetry-component",
            component,
            "--telemetry-operation",
            "build",
            str(tmp_path / "build.log"),
            "--",
            sys.executable,
            "-c",
            "print('child output must not enter span records')",
        ],
        capture_output=True,
        env=environment,
        text=True,
    )


def test_component_span_contains_only_declared_metadata(tmp_path):
    completed = _run(tmp_path)
    assert completed.returncode == 0

    records = [
        json.loads(line) for line in (tmp_path / "spans.jsonl").read_text().splitlines()
    ]
    assert [record["event"] for record in records] == ["begin", "end"]
    assert {record["component"] for record in records} == {"rocblas"}
    assert {record["operation"] for record in records} == {"build"}
    assert records[-1]["exit_code"] == 0
    assert records[-1]["duration_ns"] >= 0

    serialized = (tmp_path / "spans.jsonl").read_text()
    assert "PRIVATE_BUILD_VALUE" not in serialized
    assert "must-not-appear" not in serialized
    assert "child output" not in serialized
    assert str(tmp_path) not in serialized


def test_component_label_rejects_untrusted_text(tmp_path):
    completed = _run(tmp_path, component="rocblas;print-secret")
    assert completed.returncode == 2
    assert not (tmp_path / "spans.jsonl").exists()


def test_no_span_file_without_explicit_environment_path(tmp_path):
    environment = os.environ.copy()
    environment.pop("THEROCK_RESOURCE_SPANS_FILE", None)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--telemetry-component",
            "rocblas",
            "--telemetry-operation",
            "build",
            str(tmp_path / "build.log"),
            "--",
            sys.executable,
            "-c",
            "pass",
        ],
        capture_output=True,
        env=environment,
        text=True,
    )
    assert completed.returncode == 0
    assert list(tmp_path.glob("*.jsonl")) == []


def test_source_declared_cmake_component_labels_are_allowlisted():
    declaration = re.compile(r"therock_cmake_subproject_declare\(\s*([^\s\)]+)")
    labels = []
    for path in REPO_ROOT.rglob("CMakeLists.txt"):
        labels.extend(declaration.findall(path.read_text(encoding="utf-8")))

    assert labels
    assert [teatime._telemetry_label(label) for label in labels] == labels
