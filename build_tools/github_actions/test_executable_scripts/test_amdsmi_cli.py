#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
validation of the `amd-smi` CLI output.

This test expects `THEROCK_BIN_DIR` to point to the TheRock `bin/` directory
containing the `amd-smi` binary (CI sets this via the setup action).
"""

import os
import re
import json
import csv
import subprocess
from pathlib import Path

import pytest


def _amd_smi_path() -> Path:
    """Return the path to the `amd-smi` binary from `THEROCK_BIN_DIR`.

    Skips the test via pytest if `THEROCK_BIN_DIR` is not set. Asserts that
    the expected `amd-smi` binary exists at the resolved path.

    Args:
        None

    Returns:
        pathlib.Path: Path to the `amd-smi` binary.
    """
    th = os.getenv("THEROCK_BIN_DIR")
    if not th:
        pytest.skip("THEROCK_BIN_DIR not set; skipping amdsmi tests")
    p = Path(th) / "amd-smi"
    assert p.exists(), f"amd-smi not found at {p}"
    return p


def _run_amd_smi(amd_smi: Path, args: list[str]) -> tuple[int, str, str]:
    """Run `amd-smi list` with the given `args` and return (rc, stdout, stderr).

    The function invokes the binary via subprocess.run and captures text
    output for assertions in the tests.

    Args:
        amd_smi (pathlib.Path): Path to the `amd-smi` binary.
        args (list[str]): Arguments to pass after `amd-smi list`.

    Returns:
        tuple[int, str, str]: Return code, stdout text, stderr text.
    """
    cmd = [str(amd_smi), "list"] + args
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def _parse_gpu_blocks(output: str) -> list[str]:
    """Parse human-readable `amd-smi list` output into GPU text blocks.

    Returns a list where each element is the multiline block describing a
    single GPU. The parser looks for lines that start GPU markers like
    "GPU: <n>" or "GPU <n>:" and groups subsequent lines until the next
    GPU marker.

    Args:
        output (str): The human-readable stdout from `amd-smi list`.

    Returns:
        list[str]: List of multiline GPU description blocks.
    """
    blocks = []
    current = None
    for line in output.splitlines():
        if re.search(r"GPU:\s+(\d+)", line) or re.search(r"GPU\s+(\d+):", line):
            if current is not None:
                blocks.append("\n".join(current))
            current = [line]
            continue
        if current is not None:
            current.append(line)
    if current is not None:
        blocks.append("\n".join(current))
    return blocks


def _validate_human_block(block_text: str) -> list[str]:
    """Validate a single human-readable GPU block.

    Returns a list of missing field names (empty if all required fields
    appear). The function checks for BDF, UUID, KFD_ID, NODE_ID and
    PARTITION_ID in the block_text.

    Args:
        block_text (str): Multiline text block describing a single GPU.

    Returns:
        list[str]: Missing field names (empty if validation passes).
    """
    missing = []
    if not re.search(r"\s*BDF:\s*.+", block_text):
        missing.append("BDF")
    if not re.search(r"\s*UUID:\s*.+", block_text):
        missing.append("UUID")
    if not re.search(r"\s*KFD_ID:\s*\d+", block_text):
        missing.append("KFD_ID")
    if not re.search(r"\s*NODE_ID:\s*\d+", block_text):
        missing.append("NODE_ID")
    if not re.search(r"\s*PARTITION_ID:\s*\d+", block_text):
        missing.append("PARTITION_ID")
    return missing


def _validate_json(obj: dict) -> list[str]:
    """Validate a JSON GPU entry from `amd-smi --json`.

    Returns a list of missing or incorrectly-typed fields. Expected fields
    include `gpu` (int), `bdf` (str), `uuid` (str), `kfd_id` (int),
    `node_id` (int) and `partition_id` (int).

    Args:
        obj (dict): Parsed JSON object representing a GPU entry.

    Returns:
        list[str]: Missing or invalid field names.
    """
    missing = []
    # required keys mapping
    if "gpu" not in obj or not isinstance(obj.get("gpu"), int):
        missing.append("gpu")
    if "bdf" not in obj or not isinstance(obj.get("bdf"), str):
        missing.append("bdf")
    if "uuid" not in obj or not isinstance(obj.get("uuid"), str):
        missing.append("uuid")
    if "kfd_id" not in obj or not isinstance(obj.get("kfd_id"), int):
        missing.append("kfd_id")
    if "node_id" not in obj or not isinstance(obj.get("node_id"), int):
        missing.append("node_id")
    if "partition_id" not in obj or not isinstance(obj.get("partition_id"), int):
        missing.append("partition_id")
    return missing


def _validate_csv_row(row: dict) -> list[str]:
    """Validate a CSV row parsed from `amd-smi --csv` output.

    Expected header names are: `gpu,gpu_bdf,gpu_uuid,kfd_id,node_id,partition_id`.
    Returns a list of missing or invalid fields.

    Args:
        row (dict): Mapping of CSV headers to values as returned by
            `csv.DictReader`.

    Returns:
        list[str]: Missing or invalid field names.
    """
    # expected header names: gpu,gpu_bdf,gpu_uuid,kfd_id,node_id,partition_id
    missing = []
    try:
        if "gpu" not in row or int(row.get("gpu", "")) < 0:
            missing.append("gpu")
    except Exception:
        missing.append("gpu")
    if not row.get("gpu_bdf"):
        missing.append("gpu_bdf")
    if not row.get("gpu_uuid"):
        missing.append("gpu_uuid")
    try:
        if "kfd_id" not in row or int(row.get("kfd_id", "")) < 0:
            missing.append("kfd_id")
    except Exception:
        missing.append("kfd_id")
    try:
        if "node_id" not in row or int(row.get("node_id", "")) < 0:
            missing.append("node_id")
    except Exception:
        missing.append("node_id")
    try:
        if "partition_id" not in row or int(row.get("partition_id", "")) < 0:
            missing.append("partition_id")
    except Exception:
        missing.append("partition_id")
    return missing


@pytest.mark.parametrize(
    "mod_args",
    [
        ([], None),  # human readable on stdout
        (["--json"], None),
        (["--csv"], None),
        (["--file"], "human"),
        (["--json", "--file"], "json"),
        (["--csv", "--file"], "csv"),
    ],
)
def test_amd_smi_list(mod_args, tmp_path):
    """End-to-end test of `amd-smi list` covering output modes.

    The test runs `amd-smi list` with multiple modifier combinations (human,
    JSON, CSV, and file-output variants), parses the output and validates
    required fields for each GPU entry.

    Args:
        mod_args (tuple[list[str], Optional[str]]): Parameterized tuple where
            the first element is a list of modifier args and the second
            element indicates the expected parsed mode when `--file` is
            used.
        tmp_path (pathlib.Path): pytest temporary directory fixture.

    Returns:
        None
    """
    args, expected_mode = mod_args
    amd_smi = _amd_smi_path()

    file_path = None
    run_args = list(args)
    if "--file" in run_args:
        # supply output file
        file_path = tmp_path / "amdsmi_out.txt"
        run_args = [a for a in run_args if a != "--file"]
        run_args.extend(["--file", str(file_path)])

    rc, out, err = _run_amd_smi(amd_smi, run_args)
    assert rc == 0, f"amd-smi failed rc={rc} stderr={err} stdout={out}"

    # If file was requested, stdout should be empty
    if file_path is not None:
        assert out.strip() == "", f"Expected no stdout when using --file, got: {out}"
        assert file_path.exists(), "Expected output file to be created"
        content = file_path.read_text(encoding="utf-8", errors="replace")
    else:
        content = out

    # Validate based on mode
    if expected_mode == "json" or ("--json" in args and expected_mode is None):
        # JSON array expected
        try:
            data = json.loads(content)
        except Exception as e:
            pytest.fail(f"Failed to parse JSON output: {e}\nContent:\n{content}")
        assert isinstance(data, list) and data, "Expected non-empty JSON array"
        for idx, obj in enumerate(data):
            missing = _validate_json(obj)
            assert not missing, f"JSON GPU entry {idx} missing fields: {missing}"

    elif expected_mode == "csv" or ("--csv" in args and expected_mode is None):
        # CSV expected
        try:
            reader = csv.DictReader(content.splitlines())
            rows = list(reader)
        except Exception as e:
            pytest.fail(f"Failed to parse CSV output: {e}\nContent:\n{content}")
        assert rows, "Expected at least one CSV row"
        for idx, row in enumerate(rows):
            missing = _validate_csv_row(row)
            assert not missing, f"CSV row {idx} missing fields: {missing}"

    else:
        # human readable output
        blocks = _parse_gpu_blocks(content)
        assert blocks, "No GPU blocks found in amd-smi human output"
        for idx, block_text in enumerate(blocks):
            missing = _validate_human_block(block_text)
            assert not missing, f"Human GPU block {idx} missing fields: {missing}\nBlock:\n{block_text}"
