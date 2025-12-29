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
    def configure_path():
        """Prepend TheRock's bin dir to PATH so rocminfo, rocm_agent_enumerator, etc. are found first."""
        THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
        existing_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{THEROCK_BIN_DIR}:{existing_path}"

    @staticmethod
    def configure_ld_library_path():
        """Setup LD_LIBRARY_PATH for the tests."""
        THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")

        rocm_lib_base = Path(THEROCK_BIN_DIR).parent / "lib"
        rocm_base = Path(THEROCK_BIN_DIR).parent

        # Some libraries (libamd_comgr_loader, rocm_sysdeps) are in the rocprofiler-systems
        # component dist rather than the final dist/rocm
        build_base = rocm_base.parent.parent  # Go up from dist/rocm to build/
        rocprofsys_dist_lib = (
            build_base / "profiler" / "rocprofiler-systems" / "dist" / "lib"
        )
        os.environ["ROCM_PATH"] = str(rocprofsys_dist_lib.parent)

        ld_paths = [
            rocprofsys_dist_lib,
            rocprofsys_dist_lib / "rocm_sysdeps" / "lib",
            rocm_lib_base,
            rocm_lib_base / "rocprofiler-systems",
            rocm_base / "share" / "rocprofiler-systems" / "examples" / "lib",
        ]
        os.environ["LD_LIBRARY_PATH"] = ":".join(str(p) for p in ld_paths)

    @staticmethod
    def configure_trace_processor_path():
        """
        Setup ROCPROFSYS_TRACE_PROCESSOR_PATH for the tests.

        This is required for perfetto validation tests to work correctly.
        """
        tp_path = Path("/tmp/trace_processor_shell")

        if not tp_path.exists():
            logging.info(f"Downloading trace processor shell to {tp_path}")
            subprocess.run(
                [
                    "curl",
                    "-L",
                    "https://commondatastorage.googleapis.com/perfetto-luci-artifacts/v47.0/linux-amd64/trace_processor_shell",
                    "-o",
                    str(tp_path),
                ],
                check=True,
            )
            subprocess.run(["chmod", "+x", str(tp_path)], check=True)
            logging.info(f"Trace processor shell created at {tp_path}")

        os.environ["ROCPROFSYS_TRACE_PROCESSOR_PATH"] = str(tp_path)

    @staticmethod
    def run_pytest_tests():
        TestRocprofsys.configure_path()
        TestRocprofsys.configure_ld_library_path()
        TestRocprofsys.configure_trace_processor_path()

        # Required to force rocprofsys pytest into install mode
        os.environ["ROCPROFSYS_INSTALL_DIR"] = str(Path(THEROCK_BIN_DIR).parent)
        pytest_test_dir = (
            Path(THEROCK_BIN_DIR).parent
            / "share"
            / "rocprofiler-systems"
            / "tests"
            / "pytest"
        )

        cmd = [
            "pytest",
            str(pytest_test_dir),
            "--junit-xml=junit.xml",
            # "--no-output",
            "-v",
            "-rs",  # Show skip reasons
            "--log-cli-level=info",
        ]

        logging.info(f"++ Exec: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=THEROCK_DIR, check=True)


if __name__ == "__main__":
    TestRocprofsys.run_pytest_tests()
