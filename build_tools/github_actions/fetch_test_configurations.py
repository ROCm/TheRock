"""
This script determines what test configurations to run

Required environment variables:
  - THEROCK_BIN_DIR
  - PLATFORM
"""

import json
import os
from configure_ci import set_github_output

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")

test_matrix = {
    # BLAS tests
    "rocblas": {
        "name": "rocblas",
        "artifact_flags": "--blas --tests",
        "timeout": 5,
        "executable_command": f"{THEROCK_BIN_DIR}/rocblas-test --yaml {THEROCK_BIN_DIR}/rocblas_smoke.yaml",
        "platform": ["linux", "windows"],
    },
    "hipblaslt": {
        "name": "hipblaslt",
        "artifact_flags": "--blas --tests",
        "timeout": 30,
        "executable_command": f"{THEROCK_BIN_DIR}/hipblaslt-test --gtest_filter=*pre_checkin*",
        "platform": ["linux"],
    },
    # PRIM tests
    "rocprim": {
        "name": "rocprim",
        "artifact_flags": "--prim --tests",
        "timeout": 60,
        "executable_command": f"""
            ctest --test-dir {THEROCK_BIN_DIR}/rocprim \\
                --output-on-failure \\
                --parallel 8 \\
                --exclude-regex 'rocprim.lookback_reproducibility|rocprim.linking|rocprim.device_merge_inplace|rocprim.device_merge_sort|rocprim.device_partition|rocprim.device_radix_sort|rocprim.device_select' \\
                --timeout 900 \\
                --repeat until-pass:3
        """,
        "platform": ["linux", "windows"],
    },
    "hipcub": {
        "name": "hipcub",
        "artifact_flags": "--prim --tests",
        "timeout": 15,
        "executable_command": f"""
            ctest \\
                --test-dir {THEROCK_BIN_DIR}/hipcub \\
                --output-on-failure \\
                --parallel 8 \\
                --timeout 300 \\
                --repeat until-pass:3
        """,
        "platform": ["linux", "windows"],
    },
    "rocthrust": {
        "name": "rocthrust",
        "artifact_flags": "--prim --tests",
        "timeout": 5,
        "executable_command": f"""
            ctest \\
                --test-dir {THEROCK_BIN_DIR}/rocthrust \\
                --output-on-failure \\
                --parallel 8 \\
                --exclude-regex "^copy.hip$|scan.hip" \\
                --timeout 60 \\
                --repeat until-pass:3
        """,
        "platform": ["linux"],
    },
}


def run():
    platform = os.getenv("PLATFORM")
    project_to_test = os.getenv("project_to_test", "*")

    output_matrix = []
    for key in test_matrix:
        # If the test is enabled for a particular platform and a particular (or all) projects are selected
        if platform in test_matrix[key]["platform"] and (
            key in project_to_test or project_to_test == "*"
        ):
            output_matrix.append(test_matrix[key])

    set_github_output({"components": json.dumps(output_matrix)})


if __name__ == "__main__":
    run()
