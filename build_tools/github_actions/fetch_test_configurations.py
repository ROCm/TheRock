"""
This script determines what test configurations to run

Required environment variables:
  - RUNNER_OS (https://docs.github.com/en/actions/how-tos/writing-workflows/choosing-what-your-workflow-does/store-information-in-variables#detecting-the-operating-system)
"""

import json
import logging
import os
from pathlib import Path

from github_actions_utils import *

logging.basicConfig(level=logging.INFO)

# Note: these paths are relative to the repository root. We could make that
# more explicit, or use absolute paths.
SCRIPT_DIR = Path("./build_tools/github_actions/test_executable_scripts")


def _get_script_path(script_name: str) -> str:
    platform_path = SCRIPT_DIR / script_name
    # Convert to posix (using `/` instead of `\\`) so test workflows can use
    # 'bash' as the shell on Linux and Windows.
    posix_path = platform_path.as_posix()
    return str(posix_path)


test_matrix = {
    # BLAS tests
    "rocblas": {
        "job_name": "rocblas",
        "fetch_artifact_args": "--blas --tests",
        "timeout_minutes": 15,
        "test_script": f"python {_get_script_path('test_rocblas.py')}",
        "platform": ["linux", "windows"],
    },
    "hipblaslt": {
        "job_name": "hipblaslt",
        "fetch_artifact_args": "--blas --tests",
        "timeout_minutes": 30,
        "test_script": f"python {_get_script_path('test_hipblaslt.py')}",
        "platform": ["linux", "windows"],
    },
    # PRIM tests
    "rocprim": {
        "job_name": "rocprim",
        "fetch_artifact_args": "--prim --tests",
        "timeout_minutes": 60,
        "test_script": f"python {_get_script_path('test_rocprim.py')}",
        "platform": ["linux", "windows"],
    },
    "hipcub": {
        "job_name": "hipcub",
        "fetch_artifact_args": "--prim --tests",
        "timeout_minutes": 15,
        "test_script": f"python {_get_script_path('test_hipcub.py')}",
        "platform": ["linux", "windows"],
    },
    "rocthrust": {
        "job_name": "rocthrust",
        "fetch_artifact_args": "--prim --tests",
        "timeout_minutes": 15,
        "test_script": f"python {_get_script_path('test_rocthrust.py')}",
        "platform": ["linux"],
    },
    # SPARSE tests
    "hipsparse": {
        "job_name": "hipsparse",
        "fetch_artifact_args": "--blas --tests",
        "timeout_minutes": 30,
        "test_script": f"python {_get_script_path('test_hipsparse.py')}",
        "platform": ["linux"],
    },
    "rocsparse": {
        "job_name": "rocsparse",
        "fetch_artifact_args": "--blas --tests",
        "timeout_minutes": 120,
        "test_script": f"python {_get_script_path('test_rocsparse.py')}",
        "platform": ["linux", "windows"],
    },
    # RAND tests
    "rocrand": {
        "job_name": "rocrand",
        "fetch_artifact_args": "--rand --tests",
        "timeout_minutes": 60,
        "test_script": f"python {_get_script_path('test_rocrand.py')}",
        "platform": ["linux", "windows"],
    },
    "hiprand": {
        "job_name": "hiprand",
        "fetch_artifact_args": "--rand --tests",
        "timeout_minutes": 5,
        "test_script": f"python {_get_script_path('test_hiprand.py')}",
        "platform": ["linux", "windows"],
    },
    # MIOpen tests
    "miopen": {
        "job_name": "miopen",
        "fetch_artifact_args": "--blas --miopen --tests",
        "timeout_minutes": 120,
        "test_script": f"python {_get_script_path('test_miopen.py')}",
        "platform": ["linux"],
    },
    # RCCL tests
    "rccl": {
        "job_name": "rccl",
        "fetch_artifact_args": "--rccl --tests",
        "timeout_minutes": 15,
        "test_script": f"pytest {_get_script_path('test_rccl.py')} -v -s --log-cli-level=info",
        "platform": ["linux"],
    },
    # SOLVER tests
    "hipsolver": {
        "job_name": "hipsolver",
        "fetch_artifact_args": "--blas --tests",
        "timeout_minutes": 5,
        "test_script": f"python {_get_script_path('test_hipsolver.py')}",
        "platform": ["linux", "windows"],
    },
}


def run():
    platform = os.getenv("RUNNER_OS").lower()
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

    gha_set_output({"components": json.dumps(output_matrix), "platform": platform})


if __name__ == "__main__":
    run()
