# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""amd-smi CLI tests."""

import csv
import json
import logging
import os
import platform
import re
import subprocess
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)


def is_windows() -> bool:
    return platform.system().lower() == "windows"


def _amd_smi_path() -> Path:
    therock_bin_dir_env = os.getenv("THEROCK_BIN_DIR")
    if not therock_bin_dir_env:
        pytest.fail("THEROCK_BIN_DIR not set; failing amd-smi CLI tests")

    amd_smi_bin_path = (Path(therock_bin_dir_env).expanduser().resolve()) / "amd-smi"
    if not amd_smi_bin_path.exists():
        pytest.fail(f"amd-smi not found at {amd_smi_bin_path}")
    if not os.access(amd_smi_bin_path, os.X_OK):
        pytest.fail(f"amd-smi is not executable: {amd_smi_bin_path}")
    return amd_smi_bin_path


def _run_amd_smi(subcommands: list[str]) -> tuple[int, str, str]:
    amd_smi_bin = _amd_smi_path()
    cmd = [str(amd_smi_bin)] + list(subcommands)
    logger.info("Running amd-smi: %s", cmd)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def _parse_gpu_blocks(text_output: str) -> list[str]:
    gpu_blocks: list[str] = []
    current_block_lines: list[str] | None = None
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
    missing_fields: list[str] = []
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
    missing_fields: list[str] = []
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
    if "partition_id" not in gpu_obj or not isinstance(
        gpu_obj.get("partition_id"), int
    ):
        missing_fields.append("partition_id")
    return missing_fields


def _validate_csv_row(csv_row: dict) -> list[str]:
    missing_fields: list[str] = []
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


AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")


# Module-wide: these are amd-smi CLI tests.
pytestmark = [pytest.mark.amd_smi, pytest.mark.amd_smi_cli]


@pytest.mark.skipif(is_windows(), reason="amd-smi CLI not supported on Windows")
@pytest.mark.skipif(
    AMDGPU_FAMILIES == "gfx1151", reason="Linux gfx1151 does not support amdsmi yet"
)
def test_amd_smi_blocks() -> None:
    """Sanity-gating check: amd-smi list succeeds and reports at least one GPU."""
    return_code, stdout_text, stderr_text = _run_amd_smi(["list"])
    assert (
        return_code == 0
    ), f"amd-smi failed rc={return_code} stderr={stderr_text} stdout={stdout_text}"

    gpu_blocks = _parse_gpu_blocks(stdout_text)
    assert gpu_blocks, "No GPU blocks found in amd-smi output"


@pytest.mark.not_sanity
@pytest.mark.skipif(is_windows(), reason="amd-smi CLI not supported on Windows")
@pytest.mark.skipif(
    AMDGPU_FAMILIES == "gfx1151", reason="Linux gfx1151 does not support amdsmi yet"
)
@pytest.mark.parametrize(
    "mod_args",
    [
        ([], None),
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
def test_amd_smi_list(mod_args, tmp_path: Path) -> None:
    modifiers, expected_output_mode = mod_args

    output_file_path: Path | None = None
    invocation_args = list(modifiers)
    if "--file" in invocation_args:
        output_file_path = tmp_path / "amdsmi_out.txt"
        invocation_args = [a for a in invocation_args if a != "--file"]
        invocation_args.extend(["--file", str(output_file_path)])

    return_code, stdout_text, stderr_text = _run_amd_smi(["list"] + invocation_args)
    assert (
        return_code == 0
    ), f"amd-smi failed rc={return_code} stderr={stderr_text} stdout={stdout_text}"

    if output_file_path is not None:
        assert (
            stdout_text.strip() == ""
        ), f"Expected no stdout with --file, got: {stdout_text}"
        assert output_file_path.exists(), "Expected output file to be created"
        content_text = output_file_path.read_text(encoding="utf-8", errors="replace")
    else:
        content_text = stdout_text

    if expected_output_mode == "json" or (
        "--json" in modifiers and expected_output_mode is None
    ):
        try:
            json_data = json.loads(content_text)
        except Exception as e:
            pytest.fail(f"Failed to parse JSON output: {e}\nContent:\n{content_text}")
        assert (
            isinstance(json_data, list) and json_data
        ), "Expected non-empty JSON array"
        for index, gpu_obj in enumerate(json_data):
            missing_fields = _validate_json(gpu_obj)
            assert (
                not missing_fields
            ), f"JSON GPU entry {index} missing fields: {missing_fields}"

    elif expected_output_mode == "csv" or (
        "--csv" in modifiers and expected_output_mode is None
    ):
        try:
            csv_reader = csv.DictReader(content_text.splitlines())
            csv_rows = list(csv_reader)
        except Exception as e:
            pytest.fail(f"Failed to parse CSV output: {e}\nContent:\n{content_text}")
        assert csv_rows, "Expected at least one CSV row"
        for index, csv_row in enumerate(csv_rows):
            missing_fields = _validate_csv_row(csv_row)
            assert (
                not missing_fields
            ), f"CSV row {index} missing fields: {missing_fields}"

    else:
        gpu_blocks = _parse_gpu_blocks(content_text)
        assert gpu_blocks, "No GPU blocks found in amd-smi human output"
        for index, human_readable_gpu_block in enumerate(gpu_blocks):
            missing_fields = _validate_human_readable_gpu_block(
                human_readable_gpu_block
            )
            assert (
                not missing_fields
            ), f"Human-readable GPU block {index} missing fields: {missing_fields}\nBlock:\n{human_readable_gpu_block}"
