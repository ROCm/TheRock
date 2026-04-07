# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generate a GitHub Actions matrix JSON for PyTorch test sharding.

Outputs a ``matrix`` variable via $GITHUB_OUTPUT suitable for consumption by
``fromJSON()`` in a workflow strategy block.

Usage (in a workflow step)::

    python external-builds/pytorch/generate_test_sharding_matrix.py \\
        --test-configs 'default distributed inductor' \\
        --default-runner 'linux-gfx942-1gpu-ossci-rocm' \\
        --multi-gpu-runner 'linux-gfx942-8gpu-ossci-rocm'

Example output (written to $GITHUB_OUTPUT as ``matrix=<json>``)::

    {"include":[
      {"test_config":"default","shard":1,"num_shards":6,"runs_on":"linux-gfx942-1gpu-ossci-rocm"},
      {"test_config":"default","shard":2,"num_shards":6,"runs_on":"linux-gfx942-1gpu-ossci-rocm"},
      ...
      {"test_config":"distributed","shard":1,"num_shards":3,"runs_on":"linux-gfx942-8gpu-ossci-rocm"},
      ...
      {"test_config":"inductor","shard":1,"num_shards":2,"runs_on":"linux-gfx942-1gpu-ossci-rocm"},
      {"test_config":"inductor","shard":2,"num_shards":2,"runs_on":"linux-gfx942-1gpu-ossci-rocm"}
    ]}
"""

from __future__ import annotations

import argparse
import json
import os

# Shard counts match upstream PyTorch CI:
#   default (6) & distributed (3):
#     https://github.com/pytorch/pytorch/blob/1ace6e9e198f0221122a81efe39c11eef90b5d80/.github/workflows/trunk.yml#L283-L291
#   inductor (2):
#     https://github.com/pytorch/pytorch/blob/1ace6e9e198f0221122a81efe39c11eef90b5d80/.github/workflows/inductor-rocm-mi300.yml#L51-L52
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
            "(e.g. 'linux-gfx942-1gpu-ossci-rocm')"
        ),
    )
    parser.add_argument(
        "--multi-gpu-runner",
        required=True,
        help=(
            "Runner label for multi-GPU configs (e.g. distributed). Corresponds to "
            "'test-runs-on-multi-gpu' in amdgpu_family_matrix.py "
            "(e.g. 'linux-gfx942-8gpu-ossci-rocm')"
        ),
    )
    args = parser.parse_args()

    configs = args.test_configs.split()
    if not configs:
        parser.error("--test-configs must not be empty")

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
