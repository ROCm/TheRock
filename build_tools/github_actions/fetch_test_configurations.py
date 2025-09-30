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
        "total_shards": 1,
    },
    "hipblaslt": {
        "job_name": "hipblaslt",
        "fetch_artifact_args": "--blas --tests",
        "timeout_minutes": 60,
        "test_script": f"python {_get_script_path('test_hipblaslt.py')}",
        "platform": ["linux", "windows"],
        "total_shards": 4,
    },
    # PRIM tests
    "rocprim": {
        "job_name": "rocprim",
        "fetch_artifact_args": "--prim --tests",
        "timeout_minutes": 60,
        "test_script": f"python {_get_script_path('test_rocprim.py')}",
        "platform": ["linux", "windows"],
        "total_shards": 1,
    },
    "hipcub": {
        "job_name": "hipcub",
        "fetch_artifact_args": "--prim --tests",
        "timeout_minutes": 15,
        "test_script": f"python {_get_script_path('test_hipcub.py')}",
        "platform": ["linux", "windows"],
        "total_shards": 1,
    },
    "rocthrust": {
        "job_name": "rocthrust",
        "fetch_artifact_args": "--prim --tests",
        "timeout_minutes": 15,
        "test_script": f"python {_get_script_path('test_rocthrust.py')}",
        "platform": ["linux"],
        "total_shards": 1,
    },
    # SPARSE tests
    "hipsparse": {
        "job_name": "hipsparse",
        "fetch_artifact_args": "--blas --tests",
        "timeout_minutes": 30,
        "test_script": f"python {_get_script_path('test_hipsparse.py')}",
        "platform": ["linux"],
        "total_shards": 2,
    },
    "rocsparse": {
        "job_name": "rocsparse",
        "fetch_artifact_args": "--blas --tests",
        "timeout_minutes": 120,
        "test_script": f"python {_get_script_path('test_rocsparse.py')}",
        "platform": ["linux", "windows"],
        "total_shards": 4,
        "exclude_family": {
            "windows": ["gfx1151"]  # issue: https://github.com/ROCm/TheRock/issues/1640
        },
    },
    # RAND tests
    "rocrand": {
        "job_name": "rocrand",
        "fetch_artifact_args": "--rand --tests",
        "timeout_minutes": 60,
        "test_script": f"python {_get_script_path('test_rocrand.py')}",
        "platform": ["linux", "windows"],
        "total_shards": 1,
    },
    "hiprand": {
        "job_name": "hiprand",
        "fetch_artifact_args": "--rand --tests",
        "timeout_minutes": 5,
        "test_script": f"python {_get_script_path('test_hiprand.py')}",
        "platform": ["linux", "windows"],
        "total_shards": 1,
    },
    # MIOpen tests
    "miopen": {
        "job_name": "miopen",
        "fetch_artifact_args": "--blas --miopen --tests",
        "timeout_minutes": 120,
        "test_script": f"python {_get_script_path('test_miopen.py')}",
        "platform": ["linux"],
        "total_shards": 4,
    },
    # RCCL tests
    "rccl": {
        "job_name": "rccl",
        "fetch_artifact_args": "--rccl --tests",
        "timeout_minutes": 15,
        "test_script": f"pytest {_get_script_path('test_rccl.py')} -v -s --log-cli-level=info",
        "platform": ["linux"],
        "total_shards": 1,
    },
}


def run():
    platform = os.getenv("RUNNER_OS").lower()
    project_to_test = os.getenv("project_to_test", "*")
    amdgpu_families = os.getenv("AMDGPU_FAMILIES")

    logging.info(f"Selecting projects: {project_to_test}")

    output_matrix = []
    for key in test_matrix:
        # If the test is disabled for a particular platform, skip the test
        if (
            "exclude_family" in test_matrix[key]
            and platform in test_matrix[key]["exclude_family"]
            and amdgpu_families in test_matrix[key]["exclude_family"][platform]
        ):
            logging.info(
                f"Excluding job {job_name} for platform {platform} and family {amdgpu_families}"
            )
            continue

        # If the test is enabled for a particular platform and a particular (or all) projects are selected
        if platform in test_matrix[key]["platform"] and (
            key in project_to_test or project_to_test == "*"
        ):
            job_name = test_matrix[key]["job_name"]
            logging.info(f"Including job {job_name}")
            job_config_data = test_matrix[key]
            # For CI testing, we construct a shard array based on "total_shards" from "fetch_test_configurations.py"
            # This way, the test jobs will be split up into X shards. (ex: [1, 2, 3, 4] = 4 test shards)
            # For display purposes, we add "i + 1" for the job name (ex: 1 of 4). During the actual test sharding in the test executable, this array will become 0th index
            job_config_data["shard_arr"] = [
                i + 1 for i in range(job_config_data["total_shards"])
            ]
            output_matrix.append(job_config_data)

    gha_set_output(
        {
            "components": json.dumps(output_matrix),
            "platform": platform,
        }
    )


if __name__ == "__main__":
    run()
