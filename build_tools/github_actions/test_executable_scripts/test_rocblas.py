import os
import subprocess

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")

subprocess.run([f"{THEROCK_BIN_DIR}/rocblas-test", "--yaml", f"{THEROCK_BIN_DIR}/rocblas_smoke.yaml"], cwd=THEROCK_BIN_DIR, check=True)
