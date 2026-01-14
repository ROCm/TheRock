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
        rocm_lib_base = rocm_base / "lib"
        ld_paths = [
            rocm_lib_base,
            rocm_lib_base / "rocprofiler-systems",
            rocm_base / "share" / "rocprofiler-systems" / "examples" / "lib",
        ]
        new_ld_path = ":".join(str(p) for p in ld_paths)
        existing_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if existing_ld_path:
            os.environ["LD_LIBRARY_PATH"] = f"{new_ld_path}:{existing_ld_path}"
        else:
            os.environ["LD_LIBRARY_PATH"] = new_ld_path

        # ROCPD_SCHEMA_PATH - rocpd SQL schema files location
        # These are in share/rocprofiler-sdk-rocpd/ (from rocprofiler-sdk)
        # rocpd_schema_path = rocm_base / "share" / "rocprofiler-sdk-rocpd"
        # os.environ["ROCPD_SCHEMA_PATH"] = str(rocpd_schema_path)

    @staticmethod
    def run_pytest_package():
        TestRocprofsys.configure_paths()

        # Required to force rocprofsys pytest into install mode
        os.environ["ROCPROFSYS_INSTALL_DIR"] = str(
            Path(THEROCK_BIN_DIR).resolve().parent
        )
        pytest_package_exec = (
            Path(THEROCK_BIN_DIR).resolve().parent
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
            "-m",
            "not rccl",
            "--show-config",
            "--print-env",
            "--show-output-on-subtest-fail",
            "-v",
            "-rs",
            "--log-cli-level=info",
        ]

        logging.info(f"++ Exec: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=THEROCK_DIR, check=True)


if __name__ == "__main__":
    TestRocprofsys.run_pytest_package()
