import logging
import os
import subprocess
import sys
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

logging.basicConfig(level=logging.INFO)
rocm_base = Path(THEROCK_BIN_DIR).resolve().parent

# Environment variables
environ_vars = os.environ.copy()
ld_paths = [
    # Libraries used by examples
    rocm_base
    / "share"
    / "rocprofiler-systems"
    / "examples"
    / "lib",
]

environ_vars["PATH"] = THEROCK_BIN_DIR
environ_vars["ROCM_PATH"] = str(rocm_base)
environ_vars["LD_LIBRARY_PATH"] = ":".join(str(p) for p in ld_paths)
# Required to force the pytest package to use install mode
environ_vars["ROCPROFSYS_INSTALL_DIR"] = str(rocm_base)

# Execute tests
pytest_package_exec = (
    rocm_base / "share" / "rocprofiler-systems" / "tests" / "rocprofsys-tests.pyz"
)

cmd = [
    sys.executable,
    str(pytest_package_exec),
    "--junit-xml=junit.xml",
    "--ci-mode",
    "--log-cli-level=info",
]

logging.info(f"++ Exec: {' '.join(cmd)}")
subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=environ_vars)
