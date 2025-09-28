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

    def test_hip_printf(self):
        platform_executable_suffix = ".exe" if is_windows() else ""

        # Look up amdgpu arch, e.g. gfx1100, for explicit `--offload-arch`.
        # See https://github.com/ROCm/llvm-project/issues/302:
        #   * If this is omitted on Linux, hipcc uses rocm_agent_enumerator.
        #   * If this is omitted on Windows, hipcc uses a default (e.g. gfx906).
        # We include it on both platforms for consistency.
        amdgpu_arch_executable_file = f"amdgpu-arch{platform_executable_suffix}"
        amdgpu_arch_path = (
            THEROCK_BIN_DIR
            / ".."
            / "lib"
            / "llvm"
            / "bin"
            / amdgpu_arch_executable_file
        ).resolve()
        process = run_command([str(amdgpu_arch_path)])
        amdgpu_arch = process.stdout.splitlines()[0]

        # Compiling .cpp file using hipcc
        hipcc_check_executable_file = f"hipcc_check{platform_executable_suffix}"
        run_command(
            [
                f"{THEROCK_BIN_DIR}/hipcc",
                str(THIS_DIR / "hipcc_check.cpp"),
                "-Xlinker",
                f"-rpath={THEROCK_BIN_DIR}/../lib/",
                f"--offload-arch={amdgpu_arch}",
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
