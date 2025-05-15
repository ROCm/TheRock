import pytest
import subprocess
import re
from pathlib import Path
import platform
from pytest_check import check
import logging
import os

THIS_DIR = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)

THEROCK_BIN_DIR = Path(os.getenv("THEROCK_BIN_DIR")).resolve()


def is_windows():
    return "windows" == platform.system().lower()


def run_command(command, cwd=None):
    process = subprocess.run(command, capture_output=True, cwd=cwd, shell=is_windows())
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
        # Compiling .cpp file using hipcc
        platform_executable_suffix = ".exe" if is_windows() else ""
        hipcc_check_executable = f"hipcc_check{platform_executable_suffix}"
        run_command(
            [
                "hipcc",
                str(THIS_DIR / "hipcc_check.cpp"),
                "-o",
                hipcc_check_executable,
            ],
            cwd=str(THEROCK_BIN_DIR),
        )

        # Running and checking the executable
        process = run_command([hipcc_check_executable], cwd=str(THEROCK_BIN_DIR))
        # TODO(geomin12): Fix Windows 3221225477 error code when running exe.
        if not is_windows():
            check.equal(process.returncode, 0)
        check.greater(os.path.getsize(str(THEROCK_BIN_DIR / hipcc_check_executable)), 0)

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
