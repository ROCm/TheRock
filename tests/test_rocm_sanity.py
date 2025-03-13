import pytest
import subprocess
import re
from pathlib import Path
from pytest_check import check
import logging

THIS_DIR = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)


def run_command(command):
    process = subprocess.run(command, capture_output=True)
    return str(process.stdout)


@pytest.fixture(scope="session")
def rocm_info_output():
    try:
        return run_command(["rocminfo"])
    except Exception as e:
        logger.info(str(e))
        return None


@pytest.fixture(scope="session")
def clinfo_info_output():
    try:
        return run_command(["clinfo"])
    except Exception as e:
        logger.info(str(e))
        return None


class TestROCmSanity:
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

    @pytest.mark.parametrize(
        "to_search",
        [
            (r"Device(\s|\\t)*Type:(\s|\\t)*CL_DEVICE_TYPE_GPU"),
            (r"Name:(\s|\\t)*gfx"),
            (r"Vendor:(\s|\\t)*Advanced Micro Devices, Inc."),
        ],
        ids=[
            "clinfo - GPU Device Type Search",
            "clinfo - GFX Name Search",
            "clinfo - AMD Vendor Name Search",
        ],
    )
    def test_clinfo_output(self, clinfo_info_output, to_search):
        if not clinfo_info_output:
            pytest.fail("Command clinfo failed to run")
        check.is_not_none(
            re.search(to_search, clinfo_info_output),
            f"Failed to search for {to_search} in clinfo output",
        )

    def test_hip_printf(self):
        # Compiling .cpp file using hipcc
        run_command(
            [
                "hipcc",
                str(THIS_DIR / "hip_printf.cpp"),
                "-o",
                str(THIS_DIR / "hip_printf"),
            ]
        )

        # Running the executable
        output = run_command([str(THIS_DIR / "hip_printf")])
        check.is_not_none(re.search(r"Thread.*is\swriting", output))
