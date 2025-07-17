"""
This script determines what test configurations to run

Required environment variables:
  - PLATFORM
"""

import json
import logging
import os
from pathlib import Path

from github_actions_utils import *

logging.basicConfig(level=logging.INFO)

SCRIPT_DIR = Path("./build_tools/github_actions/test_executable_scripts")

test_matrix = {
    # BLAS tests
    "rocblas": {
        "job_name": "rocblas",
        "fetch_artifact_args": "--blas --tests",
        "timeout_minutes": 5,
        "test_script": f"python {SCRIPT_DIR / 'test_rocblas.py'}",
        "platform": ["linux", "windows"],
    },
    "hipblaslt": {
        "job_name": "hipblaslt",
        "fetch_artifact_args": "--blas --tests",
        "timeout_minutes": 30,
        "test_script": f"python {SCRIPT_DIR / 'test_hipblaslt.py'}",
        "platform": ["linux", "windows"],
    },
    # PRIM tests
    "rocprim": {
        "job_name": "rocprim",
        "fetch_artifact_args": "--prim --tests",
        "timeout_minutes": 60,
        "test_script": f"python {SCRIPT_DIR / 'test_rocprim.py'}",
        "platform": ["linux", "windows"],
    },
    "hipcub": {
        "job_name": "hipcub",
        "fetch_artifact_args": "--prim --tests",
        "timeout_minutes": 15,
        "test_script": f"python {SCRIPT_DIR / 'test_hipcub.py'}",
        "platform": ["linux", "windows"],
    },
    "rocthrust": {
        "job_name": "rocthrust",
        "fetch_artifact_args": "--prim --tests",
        "timeout_minutes": 15,
        "test_script": f"python {SCRIPT_DIR / 'test_rocthrust.py'}",
        "platform": ["linux"],
    },
    # SPARSE tests
    "hipsparse": {
        "job_name": "hipsparse",
        "fetch_artifact_args": "--blas --tests",
        "timeout_minutes": 30,
        "test_script": f"python {SCRIPT_DIR / 'test_hipsparse.py'}",
        "platform": ["linux"],
    },
    "rocsparse": {
        "job_name": "rocsparse",
        "fetch_artifact_args": "--blas --tests",
        "timeout_minutes": 120,
        "test_script": f"python {SCRIPT_DIR / 'test_rocsparse.py'}",
        "platform": ["linux", "windows"],
    },
    # RAND tests
    "rocrand": {
        "job_name": "rocrand",
        "fetch_artifact_args": "--rand --tests",
        "timeout_minutes": 60,
        "test_script": f"python {SCRIPT_DIR / 'test_rocrand.py'}",
        "platform": ["linux", "windows"],
    },
    "hiprand": {
        "job_name": "hiprand",
        "fetch_artifact_args": "--rand --tests",
        "timeout_minutes": 5,
        "test_script": f"python {SCRIPT_DIR / 'test_hiprand.py'}",
        "platform": ["linux", "windows"],
    },
    # MIOpen tests
    "miopen": {
        "job_name": "miopen",
        "fetch_artifact_args": "--blas --miopen --tests",
        "timeout_minutes": 5,
        "test_script": f"python {SCRIPT_DIR / 'test_miopen.py'}",
        "platform": ["linux"],
    },
    # RCCL tests
    "rccl": {
        "job_name": "rccl",
        "fetch_artifact_args": "--rccl --tests",
        "timeout_minutes": 15,
        "test_script": f"pytest {SCRIPT_DIR / 'test_rccl.py'} -v -s --log-cli-level=info",
        "platform": ["linux"],
    },
}


def run():
    platform = os.getenv("PLATFORM")
    project_to_test = os.getenv("project_to_test", "*")

    logging.info(f"Selecting projects: {project_to_test}")

    output_matrix = []
    for key in test_matrix:
        # If the test is enabled for a particular platform and a particular (or all) projects are selected
        if platform in test_matrix[key]["platform"] and (
            key in project_to_test or project_to_test == "*"
        ):
            job_name = test_matrix[key]["job_name"]
            logging.info(f"Including job {job_name}")
            output_matrix.append(test_matrix[key])

    gha_set_output({"components": json.dumps(output_matrix)})


if __name__ == "__main__":
    run()
