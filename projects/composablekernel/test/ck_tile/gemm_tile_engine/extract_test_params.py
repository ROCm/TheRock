#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT


import json
import argparse
import os
from pathlib import Path


def extract_test_params(config_file, output_file):
    """Extract test parameters from config JSON and write to output file"""

    # Read config file
    with open(config_file, "r") as f:
        config = json.load(f)

    # Extract test parameters
    test_params = []
    if "test_params" in config and "problem_sizes" in config["test_params"]:
        test_params = config["test_params"]["problem_sizes"]
    else:
        # Default test parameters if none specified
        test_params = [
            {"m": 256, "n": 256, "k": 128, "split_k": 1},
            {"m": 256, "n": 256, "k": 1024, "split_k": 1},
            {"m": 256, "n": 512, "k": 512, "split_k": 1},
            {"m": 512, "n": 256, "k": 512, "split_k": 1},
        ]

    # Write to output file in C++ format
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        f.write("// Generated test parameters for this configuration\n")
        f.write("// This file is auto-generated during CMake configuration\n\n")
        f.write("static const std::vector<GemmTestParams> CONFIG_TEST_PARAMS = {\n")

        for i, params in enumerate(test_params):
            comma = "," if i < len(test_params) - 1 else ""
            f.write(
                f"    {{{params['m']}, {params['n']}, {params['k']}, {params['split_k']}}}{comma}\n"
            )

        f.write("};\n")

    print(
        f"Extracted {len(test_params)} test parameters from {config_file} -> {output_file}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract test parameters from config JSON"
    )
    parser.add_argument("--config_file", required=True, help="Input config JSON file")
    parser.add_argument(
        "--output_file", required=True, help="Output test parameters file"
    )

    args = parser.parse_args()

    if not os.path.exists(args.config_file):
        print(f"Error: Config file not found: {args.config_file}")
        return 1

    extract_test_params(args.config_file, args.output_file)
    return 0


if __name__ == "__main__":
    exit(main())
