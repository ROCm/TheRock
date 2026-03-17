# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generate a GitHub Actions matrix JSON for PyTorch test sharding.

Outputs a ``matrix`` variable via $GITHUB_OUTPUT suitable for consumption by
``fromJSON()`` in a workflow strategy block.

Usage (in a workflow step)::

    python external-builds/pytorch/generate_test_sharding_matrix.py \\
        --test-configs 'default distributed inductor' \\
        --default-runner 'linux-rocm-docker-mi300-1gpu-ossci' \\
        --multi-gpu-runner 'linux-rocm-docker-mi300-8gpu-ossci'

Example output (written to $GITHUB_OUTPUT as ``matrix=<json>``)::

    {"include":[
      {"test_config":"default","shard":1,"num_shards":6,"runs_on":"linux-rocm-docker-mi300-1gpu-ossci"},
      {"test_config":"default","shard":2,"num_shards":6,"runs_on":"linux-rocm-docker-mi300-1gpu-ossci"},
      ...
      {"test_config":"distributed","shard":1,"num_shards":3,"runs_on":"linux-rocm-docker-mi300-8gpu-ossci"},
      ...
      {"test_config":"inductor","shard":1,"num_shards":2,"runs_on":"linux-rocm-docker-mi300-1gpu-ossci"},
      {"test_config":"inductor","shard":2,"num_shards":2,"runs_on":"linux-rocm-docker-mi300-1gpu-ossci"}
    ]}
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Shard counts mirror the parallelism used by upstream PyTorch CI for the
# corresponding ROCm test configurations (rocm-mi300.yml and
# inductor-rocm-mi300.yml as of March 2026).  Chosen to keep each shard
# under ~3 h on MI300 1-GPU runners (default/inductor) and MI300 8-GPU
# runners (distributed).
SHARDS_PER_CONFIG: dict[str, int] = {
    "default": 6,
    "distributed": 3,
    "inductor": 2,
}
DEFAULT_SHARDS = 4

# Configs that require a multi-GPU runner.
MULTI_GPU_CONFIGS = {"distributed"}


def build_matrix(
    test_configs: list[str],
    default_runner: str,
    multi_gpu_runner: str,
) -> dict:
    includes = []
    for config in test_configs:
        num_shards = SHARDS_PER_CONFIG.get(config, DEFAULT_SHARDS)
        runner = multi_gpu_runner if config in MULTI_GPU_CONFIGS else default_runner
        for shard in range(1, num_shards + 1):
            includes.append(
                {
                    "test_config": config,
                    "shard": shard,
                    "num_shards": num_shards,
                    "runs_on": runner,
                }
            )
    return {"include": includes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-configs",
        required=True,
        help="Space-separated test configurations (e.g. 'default distributed inductor')",
    )
    parser.add_argument(
        "--default-runner",
        required=True,
        help=(
            "Runner label for single-GPU configs. Corresponds to "
            "'test-runs-on' in amdgpu_family_matrix.py "
            "(e.g. 'linux-rocm-docker-mi300-1gpu-ossci')"
        ),
    )
    parser.add_argument(
        "--multi-gpu-runner",
        required=True,
        help=(
            "Runner label for multi-GPU configs (e.g. distributed). Corresponds to "
            "'test-runs-on-multi-gpu' in amdgpu_family_matrix.py "
            "(e.g. 'linux-rocm-docker-mi300-8gpu-ossci')"
        ),
    )
    args = parser.parse_args()

    configs = args.test_configs.split()
    if not configs:
        print("Error: --test-configs must not be empty", file=sys.stderr)
        sys.exit(1)

    matrix = build_matrix(configs, args.default_runner, args.multi_gpu_runner)
    matrix_json = json.dumps(matrix, separators=(",", ":"))

    print(f"Generated matrix with {len(matrix['include'])} jobs:")
    for entry in matrix["include"]:
        print(
            f"  {entry['test_config']} shard {entry['shard']}/{entry['num_shards']}"
            f" -> {entry['runs_on']}"
        )

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"matrix={matrix_json}\n")
    else:
        print(f"\nmatrix={matrix_json}")


if __name__ == "__main__":
    main()
