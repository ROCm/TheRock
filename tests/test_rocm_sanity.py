import pytest
import subprocess
import re
from pathlib import Path
from pytest_check import check
import logging
import os
import glob

THIS_DIR = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)

BIN_DIR = os.getenv("BIN_DIR")

def run_command(command, cwd=None):
    process = subprocess.run(command, capture_output=True, cwd=cwd)
    logger.info(str(process))
    return str(process.stdout)


@pytest.fixture(scope="session")
def rocm_info_output():
    try:
        return run_command([f"{BIN_DIR}/rocminfo"])
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

    def test_hip_printf(self):
        # Compiling .cpp file using hipcc
        run_command(
            [
                "./hipcc",
                str(THIS_DIR / "hip_printf.cpp"),
                "-o",
                str(THIS_DIR / "hip_printf"),
            ],
            cwd=str(BIN_DIR)
        )
        
        files = glob.glob(str(THIS_DIR) + "/*")
        for file in files:
            if os.path.isfile(file):
                logger.info(f"{file}: {os.path.getsize(file)} bytes")

        # Running the executable
        output = run_command(["./hip_printf"], cwd=str(THIS_DIR))
        check.is_not_none(re.search(r"Thread.*is\swriting", output))
