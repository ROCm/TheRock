# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generate a GitHub Actions matrix JSON for PyTorch test sharding.

Outputs a ``matrix`` variable via $GITHUB_OUTPUT suitable for consumption by
``fromJSON()`` in a workflow strategy block.

Usage (in a workflow step)::

    python external-builds/pytorch/generate_matrix.py \
        --test-configs 'default distributed inductor' \
        --default-runner 'linux-rocm-docker-mi300-1gpu-ossci'
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

# Shard counts are chosen to keep each shard under ~6 h on current runners.
# They mirror the parallelism used by upstream PyTorch CI for the
# corresponding ROCm test configurations (rocm-mi300.yml and
# inductor-rocm-mi300.yml).
SHARDS_PER_CONFIG: dict[str, int] = {
    "default": 6,
    "distributed": 3,
    "inductor": 2,
}
DEFAULT_SHARDS = 4


def derive_runner(default_runner: str, config: str) -> str:
    """Return the runner label for *config*.

    Distributed tests need multiple GPUs, so we derive the multi-GPU runner
    label by replacing "1gpu" with "8gpu" in the default runner label.
    """
    if config == "distributed":
        return re.sub(r"\d+gpu", "8gpu", default_runner)
    return default_runner


def build_matrix(test_configs: list[str], default_runner: str) -> dict:
    includes = []
    for config in test_configs:
        num_shards = SHARDS_PER_CONFIG.get(config, DEFAULT_SHARDS)
        runner = derive_runner(default_runner, config)
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
        help="Default runner label (e.g. 'linux-rocm-docker-mi300-1gpu-ossci')",
    )
    args = parser.parse_args()

    configs = args.test_configs.split()
    if not configs:
        print("Error: --test-configs must not be empty", file=sys.stderr)
        sys.exit(1)

    matrix = build_matrix(configs, args.default_runner)
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
