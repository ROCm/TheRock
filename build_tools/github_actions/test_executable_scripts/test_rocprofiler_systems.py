import logging
import os
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

logging.basicConfig(level=logging.INFO)


class TestRocprofsys:
    @staticmethod
    def configure_paths():
        """Prepend TheRock's bin dir to PATH so that the correct executables are found"""

        # Everything that we need should be in here
        rocm_base = Path(THEROCK_BIN_DIR).resolve().parent

        # PATH
        existing_path = os.environ.get("PATH", "")
        if existing_path:
            os.environ["PATH"] = f"{THEROCK_BIN_DIR}:{existing_path}"
        else:
            os.environ["PATH"] = THEROCK_BIN_DIR

        # ROCM_PATH
        os.environ["ROCM_PATH"] = str(rocm_base)

        # LD_LIBRARY_PATH - prepend ROCm paths while preserving existing
        ld_paths = [
            # For libopenmp-target-lib.so
            rocm_base
            / "share"
            / "rocprofiler-systems"
            / "examples"
            / "lib",
        ]
        new_ld_path = ":".join(str(p) for p in ld_paths)
        existing_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if existing_ld_path:
            os.environ["LD_LIBRARY_PATH"] = f"{new_ld_path}:{existing_ld_path}"
        else:
            os.environ["LD_LIBRARY_PATH"] = new_ld_path

    @staticmethod
    def run_pytest_package():
        TestRocprofsys.configure_paths()

        pytest_package_exec = (
            Path(THEROCK_BIN_DIR).resolve().parent
            / "share"
            / "rocprofiler-systems"
            / "tests"
            / "rocprofsys-tests.pyz"
        )

        cmd = [
            "python3",
            str(pytest_package_exec),
            "--junit-xml=junit.xml",
            "--ci-mode",
            "--log-cli-level=info",
        ]

        logging.info(f"++ Exec: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=THEROCK_DIR, check=True)


if __name__ == "__main__":
    TestRocprofsys.run_pytest_package()
