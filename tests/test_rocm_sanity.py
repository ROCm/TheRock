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
PLATFORM = platform.system().lower()


def run_command(command, cwd=None):
    process = subprocess.run(command, capture_output=True, cwd=cwd)
    return process


@pytest.fixture(scope="session")
def rocm_info_output():
    try:
        return str(run_command([f"{THEROCK_BIN_DIR}/rocminfo"]).stdout)
    except Exception as e:
        logger.info(str(e))
        return None


class TestROCmSanity:
    @pytest.mark.skipif(
        PLATFORM == "windows", reason="rocminfo is not supported on Windows"
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

    @pytest.mark.xfail  # geomin12 is fixing right now, xfail so we can see other tests running
    def test_hip_printf(self):
        # Compiling .cpp file using hipcc
        hipcc_executable = "./hipcc" if PLATFORM == "linux" else "hipcc.exe"
        hipcc_check_executable = "./hipcc_check" if PLATFORM == "linux" else "hipcc_check.exe"
        run_command(
            [
                hipcc_executable,
                str(THIS_DIR / "hipcc_check.cpp"),
                "-o",
                str(THIS_DIR / "hipcc_check"),
            ],
            cwd=str(THEROCK_BIN_DIR),
        )

        # Running and checking the executable
        process = run_command([hipcc_check_executable], cwd=str(THIS_DIR))
        check.equal(process.returncode, 0)
        check.greater(os.path.getsize(str(THIS_DIR / hipcc_check_executable)), 0)

    @pytest.mark.skipif(
        PLATFORM == "windows",
        reason="rocm_agent_enumerator is not supported on Windows",
    )
    def test_rocm_agent_enumerator(self):
        process = run_command([f"{THEROCK_BIN_DIR}/rocm_agent_enumerator"])
        output = process.stdout
        return_code = process.returncode
        check.equal(return_code, 0)
        check.is_true(output)
