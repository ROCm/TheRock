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
        rocm_base = Path(THEROCK_BIN_DIR).parent

        # PATH
        existing_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{THEROCK_BIN_DIR}:{existing_path}"

        # ROCM_PATH
        os.environ["ROCM_PATH"] = str(rocm_base)

        # LD_LIBRARY_PATH
        rocm_lib_base = rocm_base / "lib"
        ld_paths = [
            rocm_lib_base,
            rocm_lib_base / "rocprofiler-systems",
            rocm_base / "share" / "rocprofiler-systems" / "examples" / "lib",
        ]
        os.environ["LD_LIBRARY_PATH"] = ":".join(str(p) for p in ld_paths)

    @staticmethod
    def run_pytest_tests():
        TestRocprofsys.configure_paths()

        # Required to force rocprofsys pytest into install mode
        os.environ["ROCPROFSYS_INSTALL_DIR"] = str(Path(THEROCK_BIN_DIR).parent)
        pytest_package_exec = (
            Path(THEROCK_BIN_DIR).parent
            / "share"
            / "rocprofiler-systems"
            / "rocprofsys-tests.pyz"
        )

        cmd = [
            "python3",
            str(pytest_package_exec),
            "--junit-xml=junit.xml",
            # RCCL runtime-instrument is broken
            # RCCL sampling perfetto validation is also broken
            "-k",
            "not TestRCCL",
            # Custom flags -------
            # "--no-output", # Supresses all output
            "--show-output-on-subtest-fail",  # Shows runner output on subtest fail
            # "--show-output", # Shows runner output even on success (REQUIRES -s flag)
            # --------------------
            "-v",
            "-rs",
            "--log-cli-level=info",
        ]

        logging.info(f"++ Exec: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=THEROCK_DIR, check=True)


if __name__ == "__main__":
    TestRocprofsys.run_pytest_tests()
