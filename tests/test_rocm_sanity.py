# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import csv
import json
from pathlib import Path
from pytest_check import check
import logging
import os
import platform
import pytest
import re
import shlex
import subprocess
import sys

THIS_DIR = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)

THEROCK_BIN_DIR = Path(os.getenv("THEROCK_BIN_DIR")).resolve()

AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")

# Importing is_asan from github_actions_utils.py
sys.path.append(str(THIS_DIR.parent / "build_tools" / "github_actions"))
from github_actions_utils import is_asan


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
    logger.info("amd-smi returncode=%s", proc.returncode)
    if proc.returncode != 0:
        if proc.stdout:
            logger.error("amd-smi stdout:\n%s", proc.stdout)
        if proc.stderr:
            logger.error("amd-smi stderr:\n%s", proc.stderr)
    else:
        if proc.stdout:
            logger.info("amd-smi stdout:\n%s", proc.stdout)
        if proc.stderr:
            logger.error("amd-smi stderr (unexpected on success):\n%s", proc.stderr)
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


def _gpu_count_from_amd_smi_list_json() -> int:
    return_code, stdout_text, stderr_text = _run_amd_smi(["list", "--json"])
    assert (
        return_code == 0
    ), f"amd-smi list --json failed rc={return_code} stderr={stderr_text} stdout={stdout_text}"
    try:
        data = json.loads(stdout_text)
    except Exception as e:
        pytest.fail(
            f"Failed to parse amd-smi list --json output as JSON: {e}\n{stdout_text}"
        )
    assert isinstance(
        data, list
    ), f"Expected JSON array from amd-smi list --json, got: {type(data)}"
    assert data, "Expected at least one GPU entry in amd-smi list --json"
    return len(data)


def _assert_blocks_have_fields(stdout_text: str, required_fields: list[str]) -> None:
    gpu_blocks = _parse_gpu_blocks(stdout_text)
    assert gpu_blocks, f"No GPU blocks found in output:\n{stdout_text}"
    for index, block in enumerate(gpu_blocks):
        missing: list[str] = []
        for field in required_fields:
            if not re.search(rf"^\s*{re.escape(field)}:\s*.+", block, flags=re.M):
                missing.append(field)
        assert (
            not missing
        ), f"GPU block {index} missing fields {missing}. Block:\n{block}"


def is_windows():
    return "windows" == platform.system().lower()


def run_command(command: list[str], cwd=None):
    logger.info(f"++ Run [{cwd}]$ {shlex.join(command)}")
    process = subprocess.run(
        command, capture_output=True, cwd=cwd, shell=is_windows(), text=True
    )
    if process.returncode != 0:
        logger.error(f"Command failed!")
        logger.error("command stdout:")
        for line in process.stdout.splitlines():
            logger.error(line)
        logger.error("command stderr:")
        for line in process.stderr.splitlines():
            logger.error(line)
        raise Exception(f"Command failed: `{shlex.join(command)}`, see output above")
    return process


@pytest.fixture(scope="session")
def rocm_info_output():
    try:
        return str(run_command([f"{THEROCK_BIN_DIR}/rocminfo"]).stdout)
    except Exception as e:
        logger.info(str(e))
        return None


class TestROCmSanity:
    @pytest.mark.skipif(is_windows(), reason="rocminfo is not supported on Windows")
    # TODO(#3312): Re-enable once rocminfo test is fixed for ASAN builds
    @pytest.mark.skipif(
        is_asan(), reason="rocminfo test fails with ASAN build, see TheRock#3312"
    )
    @pytest.mark.parametrize(
        "to_search",
        [
            (r"Device\s*Type:\s*GPU"),
            (r"Name:\s*gfx"),
            (r"Vendor\s*Name:\s*AMD"),
        ],
        ids=[
            "rocminfo - GPU Device Type Search",
            "rocminfo - GFX Name Search",
            "rocminfo - AMD Vendor Name Search",
        ],
    )
    def test_rocm_output(self, rocm_info_output, to_search):
        if not rocm_info_output:
            pytest.fail("Command rocminfo failed to run")
        check.is_not_none(
            re.search(to_search, rocm_info_output),
            f"Failed to search for {to_search} in rocminfo output",
        )

    # TODO(#3313): Re-enable once hipcc test is fixed for ASAN builds
    @pytest.mark.skipif(
        is_asan(), reason="hipcc test fails with ASAN build, see TheRock#3313"
    )
    def test_hip_printf(self):
        platform_executable_suffix = ".exe" if is_windows() else ""

        # Look up offload arch, e.g. gfx1100, for explicit `--offload-arch`.
        # See https://github.com/ROCm/llvm-project/issues/302:
        #   * If this is omitted on Linux, hipcc uses rocm_agent_enumerator.
        #   * If this is omitted on Windows, hipcc uses a default (e.g. gfx906).
        # We include it on both platforms for consistency.
        offload_arch_executable_file = f"offload-arch{platform_executable_suffix}"
        offload_arch_path = (
            THEROCK_BIN_DIR
            / ".."
            / "lib"
            / "llvm"
            / "bin"
            / offload_arch_executable_file
        ).resolve()
        process = run_command([str(offload_arch_path)])

        # Extract the arch from the command output, working around
        # https://github.com/ROCm/TheRock/issues/1118. We only expect the output
        # to contain 'gfx####` text but some ROCm releases contained stray
        # "HIP Library Path" logging first.
        # **Note**: this partly defaults the purpose of the sanity check, since
        # that should really be a test failure. However, per discussion on
        # https://github.com/ROCm/TheRock/pull/3257 we found that system
        # installs of ROCm (DLLs in system32) take precedence over user
        # installs (PATH env var) under certain conditions. Hopefully a
        # different unit test elsewhere in ROCm catches that more directly.
        offload_arch = None
        for line in process.stdout.splitlines():
            if "gfx" in line:
                offload_arch = line
                break
        assert (
            offload_arch is not None
        ), f"Expected offload-arch to return gfx####, got:\n{process.stdout}"

        # Compiling .cpp file using hipcc
        hipcc_check_executable_file = f"hipcc_check{platform_executable_suffix}"
        run_command(
            [
                f"{THEROCK_BIN_DIR}/hipcc",
                str(THIS_DIR / "hipcc_check.cpp"),
                "-Xlinker",
                f"-rpath={THEROCK_BIN_DIR}/../lib/",
                f"--offload-arch={offload_arch}",
                "-o",
                hipcc_check_executable_file,
            ],
            cwd=str(THEROCK_BIN_DIR),
        )

        # Running and checking the executable
        platform_executable_prefix = "./" if not is_windows() else ""
        hipcc_check_executable = f"{platform_executable_prefix}hipcc_check"
        process = run_command([hipcc_check_executable], cwd=str(THEROCK_BIN_DIR))
        check.equal(process.returncode, 0)
        check.greater(
            os.path.getsize(str(THEROCK_BIN_DIR / hipcc_check_executable_file)), 0
        )

    @pytest.mark.skipif(
        is_windows(),
        reason="rocm_agent_enumerator is not supported on Windows",
    )
    def test_rocm_agent_enumerator(self):
        process = run_command([f"{THEROCK_BIN_DIR}/rocm_agent_enumerator"])
        output = process.stdout
        return_code = process.returncode
        check.equal(return_code, 0)
        check.is_true(output)

    @pytest.mark.skipif(is_windows(), reason="amdsmitst is not supported on Windows")
    # TODO(#2789): Remove skip once amdsmi supports gfx1151
    @pytest.mark.skipif(
        AMDGPU_FAMILIES == "gfx1151", reason="Linux gfx1151 does not support amdsmi yet"
    )
    def test_amdsmi_suite(self):
        amdsmi_test_bin = (
            THEROCK_BIN_DIR.parent / "share" / "amd_smi" / "tests" / "amdsmitst"
        ).resolve()

        assert (
            amdsmi_test_bin.exists()
        ), f"amdsmitst not found at expected location: {amdsmi_test_bin}"
        assert os.access(
            amdsmi_test_bin, os.X_OK
        ), f"amdsmitst is not executable: {amdsmi_test_bin}"

        include_tests = [
            "amdsmitstReadOnly.*",
            "amdsmitstReadWrite.FanReadWrite",
            "amdsmitstReadWrite.TestOverdriveReadWrite",
            "amdsmitstReadWrite.TestPciReadWrite",
            "amdsmitstReadWrite.TestPowerReadWrite",
            "amdsmitstReadWrite.TestPerfCntrReadWrite",
            "amdsmitstReadWrite.TestEvtNotifReadWrite",
            "AmdSmiDynamicMetricTest.*",
        ]

        exclude_tests = [
            "amdsmitstReadOnly.TempRead",
            "amdsmitstReadOnly.TestFrequenciesRead",
            "amdsmitstReadWrite.TestPowerReadWrite",
        ]

        TESTS_TO_IGNORE = {
            "gfx90a": {
                # TODO(#2963): Re-enable once amdsmi tests are fixed for gfx90X-dcgpu
                "linux": [
                    "amdsmitstReadOnly.TestSysInfoRead",
                    "amdsmitstReadOnly.TestIdInfoRead",
                    "amdsmitstReadWrite.TestPciReadWrite",
                ]
            },
            "gfx110X-all": {
                # TODO(#2963): Re-enable once amdsmi tests are fixed for gfx110X-all
                "linux": [
                    "amdsmitstReadWrite.FanReadWrite",
                ]
            },
        }

        platform_key = "windows" if is_windows() else "linux"
        if (
            AMDGPU_FAMILIES in TESTS_TO_IGNORE
            and platform_key in TESTS_TO_IGNORE[AMDGPU_FAMILIES]
        ):
            ignored_tests = TESTS_TO_IGNORE[AMDGPU_FAMILIES][platform_key]
            exclude_tests.extend(ignored_tests)

        gtest_filter = f"{':'.join(include_tests)}:-{':'.join(exclude_tests)}"
        cmd = [str(amdsmi_test_bin), f"--gtest_filter={gtest_filter}"]

        process = run_command(cmd, cwd=str(amdsmi_test_bin.parent))

        combined = (process.stdout or "") + "\n" + (process.stderr or "")
        for line in combined.splitlines():
            if "[==========]" in line:
                print(f"[amdsmitst-summary] {line}")

        check.equal(process.returncode, 0)

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
    def test_amd_smi_list(self, mod_args, tmp_path: Path) -> None:
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
            content_text = output_file_path.read_text(
                encoding="utf-8", errors="replace"
            )
        else:
            content_text = stdout_text

        if expected_output_mode == "json" or (
            "--json" in modifiers and expected_output_mode is None
        ):
            try:
                json_data = json.loads(content_text)
            except Exception as e:
                pytest.fail(
                    f"Failed to parse JSON output: {e}\nContent:\n{content_text}"
                )
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
                pytest.fail(
                    f"Failed to parse CSV output: {e}\nContent:\n{content_text}"
                )
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

    @pytest.mark.skipif(is_windows(), reason="amd-smi CLI not supported on Windows")
    @pytest.mark.skipif(
        AMDGPU_FAMILIES == "gfx1151", reason="Linux gfx1151 does not support amdsmi yet"
    )
    def test_amd_smi_version(self) -> None:
        return_code, stdout_text, stderr_text = _run_amd_smi(["version"])
        assert (
            return_code == 0
        ), f"amd-smi version failed rc={return_code} stderr={stderr_text} stdout={stdout_text}"
        assert re.search(
            r"AMDSMI Tool:\s\d+\.\d+\.\d+.*\sAMDSMI Library version:\s\d+\.\d+\.\d+.*",
            stdout_text,
        ), f"Unexpected amd-smi version output:\n{stdout_text}"

    @pytest.mark.skipif(is_windows(), reason="amd-smi CLI not supported on Windows")
    @pytest.mark.skipif(
        AMDGPU_FAMILIES == "gfx1151", reason="Linux gfx1151 does not support amdsmi yet"
    )
    def test_amd_smi_static(self) -> None:
        return_code, stdout_text, stderr_text = _run_amd_smi(["static"])
        assert (
            return_code == 0
        ), f"amd-smi static failed rc={return_code} stderr={stderr_text} stdout={stdout_text}"
        _assert_blocks_have_fields(stdout_text, ["ASIC", "BUS", "IFWI"])

    @pytest.mark.skipif(is_windows(), reason="amd-smi CLI not supported on Windows")
    @pytest.mark.skipif(
        AMDGPU_FAMILIES == "gfx1151", reason="Linux gfx1151 does not support amdsmi yet"
    )
    @pytest.mark.parametrize(
        "mode,subcommands",
        [
            ("plain", ["metric"]),
            ("watch_time", ["metric", "--watch=1", "--watch_time=3"]),
            ("iterations", ["metric", "--watch=1", "--iterations=3"]),
        ],
        ids=["plain", "watch_time", "iterations"],
    )
    def test_amd_smi_metric(self, mode: str, subcommands: list[str]) -> None:
        return_code, stdout_text, stderr_text = _run_amd_smi(subcommands)
        assert (
            return_code == 0
        ), f"amd-smi {' '.join(subcommands)} failed rc={return_code} stderr={stderr_text} stdout={stdout_text}"

        ngpus = _gpu_count_from_amd_smi_list_json()
        if mode == "watch_time":
            # Expect more than one timestamp for watch_time mode.
            assert (
                len(re.findall(r"TIMESTAMP", stdout_text, flags=re.I)) > 1
            ), f"Expected multiple TIMESTAMP entries in output. stdout:\n{stdout_text}"
        elif mode == "iterations":
            # Expect exactly 3 iterations per GPU in Linux output.
            assert (
                len(re.findall(r"TIMESTAMP", stdout_text, flags=re.I)) == 3 * ngpus
            ), f"Expected {3 * ngpus} TIMESTAMP entries in output. stdout:\n{stdout_text}"

        # Validate that each GPU reports TEMPERATURE and POWER at least once.
        seen: dict[int, dict[str, bool]] = {
            gpu: {"TEMPERATURE": False, "POWER": False} for gpu in range(ngpus)
        }
        current_gpu: int | None = None
        for line in stdout_text.splitlines():
            m1 = re.search(r"GPU:\s+(\d+)", line)
            m2 = re.search(r"GPU\s+(\d+):", line)
            if m1:
                current_gpu = int(m1.group(1))
            elif m2:
                current_gpu = int(m2.group(1))

            if current_gpu is None or current_gpu not in seen:
                continue

            if re.search(r"\s*TEMPERATURE:", line):
                seen[current_gpu]["TEMPERATURE"] = True
            if re.search(r"\s*POWER:", line):
                seen[current_gpu]["POWER"] = True

        missing = {
            gpu: [k for k, v in fields.items() if not v]
            for gpu, fields in seen.items()
            if not all(fields.values())
        }
        assert (
            not missing
        ), f"Missing TEMPERATURE/POWER fields for GPUs: {missing}. stdout:\n{stdout_text}"

    @pytest.mark.skipif(is_windows(), reason="amd-smi CLI not supported on Windows")
    @pytest.mark.skipif(
        AMDGPU_FAMILIES == "gfx1151", reason="Linux gfx1151 does not support amdsmi yet"
    )
    def test_amd_smi_firmware(self) -> None:
        return_code, stdout_text, stderr_text = _run_amd_smi(["firmware"])
        assert (
            return_code == 0
        ), f"amd-smi firmware failed rc={return_code} stderr={stderr_text} stdout={stdout_text}"

        gpu_blocks = _parse_gpu_blocks(stdout_text)
        assert gpu_blocks, f"No GPU blocks found in firmware output:\n{stdout_text}"
        for index, block in enumerate(gpu_blocks):
            assert re.search(
                r"^\s*FW_VERSION:\s+\d+",
                block,
                flags=re.M,
            ), f"GPU block {index} missing FW_VERSION. Block:\n{block}"

    @pytest.mark.skipif(is_windows(), reason="amd-smi CLI not supported on Windows")
    @pytest.mark.skipif(
        AMDGPU_FAMILIES == "gfx1151", reason="Linux gfx1151 does not support amdsmi yet"
    )
    @pytest.mark.parametrize(
        "mode,subcommands",
        [
            ("plain", ["process"]),
            ("watch_time", ["process", "--watch=1", "--watch_time=3"]),
            ("iterations", ["process", "--watch=1", "--iterations=3"]),
        ],
        ids=["plain", "watch_time", "iterations"],
    )
    def test_amd_smi_process(self, mode: str, subcommands: list[str]) -> None:
        return_code, stdout_text, stderr_text = _run_amd_smi(subcommands)
        assert (
            return_code == 0
        ), f"amd-smi {' '.join(subcommands)} failed rc={return_code} stderr={stderr_text} stdout={stdout_text}"

        ngpus = _gpu_count_from_amd_smi_list_json()
        if mode == "watch_time":
            assert (
                len(re.findall(r"TIMESTAMP:\s+\d+", stdout_text)) > 1
            ), f"Expected multiple TIMESTAMP entries in output. stdout:\n{stdout_text}"
        elif mode == "iterations":
            assert (
                len(re.findall(r"TIMESTAMP:\s+\d+", stdout_text)) == 3 * ngpus
            ), f"Expected {3 * ngpus} TIMESTAMP entries in output. stdout:\n{stdout_text}"

        # Validate that each GPU reports the no-running-processes message at least once.
        seen: dict[int, bool] = {gpu: False for gpu in range(ngpus)}
        current_gpu: int | None = None
        for line in stdout_text.splitlines():
            m1 = re.search(r"GPU:\s+(\d+)", line)
            m2 = re.search(r"GPU\s+(\d+):", line)
            if m1:
                current_gpu = int(m1.group(1))
            elif m2:
                current_gpu = int(m2.group(1))

            if current_gpu is None or current_gpu not in seen:
                continue

            if re.search(r"\s*PROCESS_INFO:\s*No running processes detected\s*", line):
                seen[current_gpu] = True

        missing = [gpu for gpu, ok in seen.items() if not ok]
        assert (
            not missing
        ), f"Missing expected PROCESS_INFO message for GPUs: {missing}. stdout:\n{stdout_text}"

    @pytest.mark.skipif(is_windows(), reason="amd-smi CLI not supported on Windows")
    @pytest.mark.skipif(
        AMDGPU_FAMILIES == "gfx1151", reason="Linux gfx1151 does not support amdsmi yet"
    )
    def test_amd_smi_monitor_qt(self) -> None:
        return_code, stdout_text, stderr_text = _run_amd_smi(["monitor", "-qt"])
        assert (
            return_code == 0
        ), f"amd-smi monitor -qt failed rc={return_code} stderr={stderr_text} stdout={stdout_text}"

        output_lines = stdout_text.splitlines()
        header_index: int | None = None
        for idx, line in enumerate(output_lines):
            if re.search(
                r"GPU\s+NAME\s+PID\s+GTT_MEM\s+CPU_MEM\s+VRAM_MEM\s+MEM_USG\s+CU%\s+EVICT",
                line,
            ):
                header_index = idx
                break
        assert (
            header_index is not None
        ), f"Expected header line not found. stdout:\n{stdout_text}"

        ngpus = _gpu_count_from_amd_smi_list_json()
        assert (
            len(output_lines) >= header_index + 1 + ngpus
        ), f"Expected at least {ngpus} data lines after header. stdout:\n{stdout_text}"

        # Expected format:
        # - either all N/A line, or numeric fields and CU% either N/A or '<num> %'.
        cu_percentage = r"(?:N/A|\d+\.\d+\s+%)"
        for gpu in range(ngpus):
            line = output_lines[header_index + 1 + gpu]
            all_na = re.search(
                r"\d+\s+N/A\s+N/A\s+N/A\s+N/A\s+N/A\s+N/A\s+N/A\s+N/A", line
            )
            numeric = re.search(
                rf"\d+\s+.*\s+\d+\s+\d+\.\d+\s+[GKMB]+\s+\d+\.\d+\s+[GKMB]+\s+\d+\.\d+\s+[GKMB]+\s+\d+\.\d+\s+[GKMB]+\s+{cu_percentage}\s+\d+\s+ms",
                line,
            )
            assert (
                all_na or numeric
            ), f"Unexpected monitor -qt line for gpu_index={gpu}: {line!r}\nstdout:\n{stdout_text}"

    @pytest.mark.skipif(is_windows(), reason="amd-smi CLI not supported on Windows")
    @pytest.mark.skipif(
        AMDGPU_FAMILIES == "gfx1151", reason="Linux gfx1151 does not support amdsmi yet"
    )
    def test_amd_smi_gfx_version_matches_rocminfo(self, rocm_info_output) -> None:
        if not rocm_info_output:
            pytest.fail("Command rocminfo failed to run")

        return_code, stdout_text, stderr_text = _run_amd_smi(["static", "--asic"])
        assert (
            return_code == 0
        ), f"amd-smi static --asic failed rc={return_code} stderr={stderr_text} stdout={stdout_text}"

        amdsmi_pattern = r"TARGET_GRAPHICS_VERSION:\s*(\S+)"
        rocminfo_pattern = r"Name:\s*(\S+)"

        versions = re.findall(amdsmi_pattern, stdout_text)
        # Filter to gfx names only, excluding 'amdgcn'.
        names = [
            n
            for n in re.findall(rocminfo_pattern, rocm_info_output)
            if re.search(r"gfx", n, flags=re.IGNORECASE) and "amdgcn" not in n
        ]

        assert versions == names, (
            "TARGET_GRAPHICS_VERSION list did not match rocminfo gfx names.\n"
            f"amd-smi versions={versions}\nrocminfo names={names}\n"
            f"amd-smi stdout:\n{stdout_text}\n"
        )

    @pytest.mark.skipif(is_windows(), reason="amd-smi CLI not supported on Windows")
    @pytest.mark.skipif(
        AMDGPU_FAMILIES == "gfx1151", reason="Linux gfx1151 does not support amdsmi yet"
    )
    def test_amd_smi_gpu_driver_version(self) -> None:
        regex_match = (
            r"AMDSMI Tool:\s\d+\.\d+\.\d+.*\sAMDSMI Library version:\s\d+\.\d+\.\d+.*"
            r"\sROCm version:\s\d+\.\d+\.\d+\s.*\samdgpu version:\s\d+\.\d+\.\d+.*"
        )

        rc1, out1, err1 = _run_amd_smi(["version"])
        assert rc1 == 0, f"amd-smi version failed rc={rc1} stderr={err1} stdout={out1}"

        rc2, out2, err2 = _run_amd_smi(["version", "-g"])
        assert (
            rc2 == 0
        ), f"amd-smi version -g failed rc={rc2} stderr={err2} stdout={out2}"

        assert re.search(
            regex_match, out1
        ), f"Unexpected amd-smi version output:\n{out1}"
        assert re.search(
            regex_match, out2
        ), f"Unexpected amd-smi version -g output:\n{out2}"
