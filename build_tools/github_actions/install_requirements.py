import logging
import shlex
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent
THEROCK_OUTPUT_DIR = str(THEROCK_DIR / "build")


def install_requirements(job_name: str):
    match job_name:
        case "rocprofiler_compute":
            requirements_dir = f"{THEROCK_OUTPUT_DIR}/libexec/rocprofiler-compute"
            cmd = [
                "uv",
                "pip",
                "install",
                "-r",
                f"{requirements_dir}/requirements.txt",
                "-r",
                f"{requirements_dir}/requirements-test.txt",
            ]
            logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
            subprocess.run(
                cmd,
                cwd=THEROCK_DIR,
                check=True,
            )
        case _:
            print(f"No extra setup found for {job_name}.")
