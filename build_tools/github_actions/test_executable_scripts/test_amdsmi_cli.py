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
import logging

import pytest


logger = logging.getLogger(__name__)


def _amd_smi_path() -> Path:
    """Return the path to the `amd-smi` binary from `THEROCK_BIN_DIR`.

    Skips the test via pytest if `THEROCK_BIN_DIR` is not set. Asserts that
    the expected `amd-smi` binary exists at the resolved path.

    Args:
        None

    Returns:
        pathlib.Path: Path to the `amd-smi` binary.
    """
    therock_bin_dir_env = os.getenv("THEROCK_BIN_DIR")
    if not therock_bin_dir_env:
        pytest.skip("THEROCK_BIN_DIR not set; skipping amdsmi tests")

    # Resolve the path to an absolute canonical path to avoid cwd-dependent
    # failures (e.g., if a prior step changes directory). Also check that the
    # binary exists and is executable.
    amd_smi_bin_path = (Path(therock_bin_dir_env).expanduser().resolve()) / "amd-smi"
    assert amd_smi_bin_path.exists(), f"amd-smi not found at {amd_smi_bin_path}"
    assert os.access(amd_smi_bin_path, os.X_OK), f"amd-smi is not executable: {amd_smi_bin_path}"
    return amd_smi_bin_path


def _run_amd_smi(amd_smi_path: Path, modifiers: list[str]) -> tuple[int, str, str]:
    """Run `amd-smi list` with the given `modifiers` and return (rc, stdout, stderr).

    The function invokes the binary via subprocess.run and captures text
    output for assertions in the tests.

    Args:
        amd_smi_path (pathlib.Path): Path to the `amd-smi` binary.
        modifiers (list[str]): Arguments to pass after `amd-smi list`.

    Returns:
        tuple[int, str, str]: Return code, stdout text, stderr text.
    """
    cmd = [str(amd_smi_path), "list"] + modifiers
    logger.debug("Running amd-smi: %s", cmd)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    logger.debug("amd-smi returncode=%s", proc.returncode)
    logger.debug("amd-smi stdout:\n%s", proc.stdout)
    logger.debug("amd-smi stderr:\n%s", proc.stderr)
    return proc.returncode, proc.stdout, proc.stderr


def _parse_gpu_blocks(text_output: str) -> list[str]:
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
    gpu_blocks = []
    current_block_lines = None
    for line in text_output.splitlines():
        if re.search(r"GPU:\s+(\d+)", line) or re.search(r"GPU\s+(\d+):", line):
            if current_block_lines is not None:
                gpu_blocks.append("\n".join(current_block_lines))
            current_block_lines = [line]
            continue
        if current_block_lines is not None:
            current_block_lines.append(line)
    if current_block_lines is not None:
        gpu_blocks.append("\n".join(current_block_lines))
    return gpu_blocks


def _validate_human_readable_gpu_block(human_readable_gpu_block_text: str) -> list[str]:
    """Validate a single human-readable GPU block.

    Returns a list of missing field names (empty if all required fields
    appear). The function checks for BDF, UUID, KFD_ID, NODE_ID and
    PARTITION_ID in the block_text.

    Args:
        human_readable_gpu_block_text (str): Multiline text block describing a single GPU.

    Returns:
        list[str]: Missing field names (empty if validation passes).
    """
    missing_fields = []
    if not re.search(r"\s*BDF:\s*.+", human_readable_gpu_block_text):
        missing_fields.append("BDF")
    if not re.search(r"\s*UUID:\s*.+", human_readable_gpu_block_text):
        missing_fields.append("UUID")
    if not re.search(r"\s*KFD_ID:\s*\d+", human_readable_gpu_block_text):
        missing_fields.append("KFD_ID")
    if not re.search(r"\s*NODE_ID:\s*\d+", human_readable_gpu_block_text):
        missing_fields.append("NODE_ID")
    if not re.search(r"\s*PARTITION_ID:\s*\d+", human_readable_gpu_block_text):
        missing_fields.append("PARTITION_ID")
    return missing_fields


def _validate_json(gpu_obj: dict) -> list[str]:
    """Validate a JSON GPU entry from `amd-smi --json`.

    Returns a list of missing or incorrectly-typed fields. Expected fields
    include `gpu` (int), `bdf` (str), `uuid` (str), `kfd_id` (int),
    `node_id` (int) and `partition_id` (int).

    Args:
        obj (dict): Parsed JSON object representing a GPU entry.

    Returns:
        list[str]: Missing or invalid field names.
    """
    missing_fields = []
    # required keys mapping
    if "gpu" not in gpu_obj or not isinstance(gpu_obj.get("gpu"), int):
        missing_fields.append("gpu")
    if "bdf" not in gpu_obj or not isinstance(gpu_obj.get("bdf"), str):
        missing_fields.append("bdf")
    if "uuid" not in gpu_obj or not isinstance(gpu_obj.get("uuid"), str):
        missing_fields.append("uuid")
    if "kfd_id" not in gpu_obj or not isinstance(gpu_obj.get("kfd_id"), int):
        missing_fields.append("kfd_id")
    if "node_id" not in gpu_obj or not isinstance(gpu_obj.get("node_id"), int):
        missing_fields.append("node_id")
    if "partition_id" not in gpu_obj or not isinstance(gpu_obj.get("partition_id"), int):
        missing_fields.append("partition_id")
    return missing_fields


def _validate_csv_row(csv_row: dict) -> list[str]:
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
    missing_fields = []
    try:
        if "gpu" not in csv_row or int(csv_row.get("gpu", "")) < 0:
            missing_fields.append("gpu")
    except Exception:
        missing_fields.append("gpu")
    if not csv_row.get("gpu_bdf"):
        missing_fields.append("gpu_bdf")
    if not csv_row.get("gpu_uuid"):
        missing_fields.append("gpu_uuid")
    try:
        if "kfd_id" not in csv_row or int(csv_row.get("kfd_id", "")) < 0:
            missing_fields.append("kfd_id")
    except Exception:
        missing_fields.append("kfd_id")
    try:
        if "node_id" not in csv_row or int(csv_row.get("node_id", "")) < 0:
            missing_fields.append("node_id")
    except Exception:
        missing_fields.append("node_id")
    try:
        if "partition_id" not in csv_row or int(csv_row.get("partition_id", "")) < 0:
            missing_fields.append("partition_id")
    except Exception:
        missing_fields.append("partition_id")
    return missing_fields


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
    ids=[
        "human-stdout",
        "json-stdout",
        "csv-stdout",
        "human-file",
        "json-file",
        "csv-file",
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
    modifiers, expected_output_mode = mod_args
    amd_smi_bin = _amd_smi_path()

    output_file_path = None
    invocation_args = list(modifiers)
    if "--file" in invocation_args:
        # supply output file
        output_file_path = tmp_path / "amdsmi_out.txt"
        invocation_args = [a for a in invocation_args if a != "--file"]
        invocation_args.extend(["--file", str(output_file_path)])

    return_code, stdout_text, stderr_text = _run_amd_smi(amd_smi_bin, invocation_args)
    assert return_code == 0, f"amd-smi failed rc={return_code} stderr={stderr_text} stdout={stdout_text}"

    # If file was requested, stdout should be empty
    if output_file_path is not None:
        assert stdout_text.strip() == "", f"Expected no stdout when using --file, got: {stdout_text}"
        assert output_file_path.exists(), "Expected output file to be created"
        content_text = output_file_path.read_text(encoding="utf-8", errors="replace")
    else:
        content_text = stdout_text

    # Validate based on mode
    if expected_output_mode == "json" or ("--json" in modifiers and expected_output_mode is None):
        # JSON array expected
        try:
            json_data = json.loads(content_text)
        except Exception as e:
            pytest.fail(f"Failed to parse JSON output: {e}\nContent:\n{content_text}")
        assert isinstance(json_data, list) and json_data, "Expected non-empty JSON array"
        for index, gpu_obj in enumerate(json_data):
            missing_fields = _validate_json(gpu_obj)
            assert not missing_fields, f"JSON GPU entry {index} missing fields: {missing_fields}"

    elif expected_output_mode == "csv" or ("--csv" in modifiers and expected_output_mode is None):
        # CSV expected
        try:
            csv_reader = csv.DictReader(content_text.splitlines())
            csv_rows = list(csv_reader)
        except Exception as e:
            pytest.fail(f"Failed to parse CSV output: {e}\nContent:\n{content_text}")
        assert csv_rows, "Expected at least one CSV row"
        for index, csv_row in enumerate(csv_rows):
            missing_fields = _validate_csv_row(csv_row)
            assert not missing_fields, f"CSV row {index} missing fields: {missing_fields}"

    else:
        # human readable output
        gpu_blocks = _parse_gpu_blocks(content_text)
        assert gpu_blocks, "No GPU blocks found in amd-smi human output"
        for index, human_readable_gpu_block in enumerate(gpu_blocks):
            missing_fields = _validate_human_readable_gpu_block(human_readable_gpu_block)
            assert not missing_fields, f"Human-readable GPU block {index} missing fields: {missing_fields}\nBlock:\n{human_readable_gpu_block}"
