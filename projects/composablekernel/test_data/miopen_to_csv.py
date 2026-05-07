#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Convert MIOpen Driver Commands to CSV Test Cases

Parses MIOpen driver commands from log files and converts them to CSV format
for CK convolution testing.

Usage:
    python3 miopen_to_csv.py --input miopen_commands.txt --output conv_cases.csv
    python3 miopen_to_csv.py --input miopen_log.txt --output-2d conv_2d.csv --output-3d conv_3d.csv
"""

import argparse
import csv
import re
import os


def parse_miopen_command(command_line):
    """
    Parse MIOpen driver command line into parameter dictionary

    Example input:
    ./bin/MIOpenDriver conv -n 4 -c 3 -H 224 -W 224 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1

    Returns dict with parsed parameters or None if parsing fails
    """
    if not command_line.strip().startswith("./bin/MIOpenDriver conv"):
        return None

    # Extract parameters using regex
    params = {}

    # Parameter mapping: flag -> description
    # Support both short (-D) and long (--in_d) parameter formats
    param_patterns = {
        "n": r"-n\s+(\d+)",  # batch size
        "c": r"-c\s+(\d+)",  # input channels
        "k": r"-k\s+(\d+)",  # output channels
        "H": r"-H\s+(\d+)",  # input height
        "W": r"-W\s+(\d+)",  # input width
        "D": r"(?:-D|--in_d)\s+(\d+)",  # input depth (3D only) - supports both -D and --in_d
        "y": r"-y\s+(\d+)",  # kernel height
        "x": r"-x\s+(\d+)",  # kernel width
        "z": r"(?:-z|--fil_d)\s+(\d+)",  # kernel depth (3D only) - supports both -z and --fil_d
        "u": r"-u\s+(\d+)",  # stride height
        "v": r"-v\s+(\d+)",  # stride width
        "w": r"(?:-w|--conv_stride_d)\s+(\d+)",  # stride depth (3D only) - supports both -w and --conv_stride_d
        "p": r"-p\s+(\d+)",  # pad height
        "q": r"-q\s+(\d+)",  # pad width
        "s": r"(?:-s|--pad_d)\s+(\d+)",  # pad depth (3D only) - supports both -s and --pad_d
        "l": r"-l\s+(\d+)",  # dilation height
        "j": r"-j\s+(\d+)",  # dilation width
        "r": r"(?:-r|--dilation_d)\s+(\d+)",  # dilation depth (3D only) - supports both -r and --dilation_d
        "g": r"-g\s+(\d+)",  # groups
        "F": r"-F\s+(\d+)",  # direction (1=fwd, 2=bwd_weight, 4=bwd_data)
    }

    for param, pattern in param_patterns.items():
        match = re.search(pattern, command_line)
        if match:
            params[param] = int(match.group(1))

    return params if params else None


def miopen_to_conv_param(miopen_params):
    """
    Convert MIOpen parameters to CK ConvParam format

    Returns dictionary in CSV format or None if conversion fails
    """
    if not miopen_params:
        return None

    # Determine if 2D or 3D convolution
    is_3d = (
        "D" in miopen_params
        or "z" in miopen_params
        or "w" in miopen_params
        or "r" in miopen_params
        or "s" in miopen_params
    )

    # Extract basic parameters with defaults
    ndim = 3 if is_3d else 2
    groups = miopen_params.get("g", 1)
    batch_size = miopen_params.get("n", 1)
    # MIOpen uses total channels (C*G), CK uses channels per group
    out_channels_total = miopen_params.get("k", 64)
    in_channels_total = miopen_params.get("c", 3)
    out_channels = out_channels_total // groups  # CK format: channels per group
    in_channels = in_channels_total // groups  # CK format: channels per group

    if is_3d:
        # 3D convolution
        kernel_d = miopen_params.get("z", 3)
        kernel_h = miopen_params.get("y", 3)
        kernel_w = miopen_params.get("x", 3)

        input_d = miopen_params.get("D", 16)
        input_h = miopen_params.get("H", 32)
        input_w = miopen_params.get("W", 32)

        stride_d = miopen_params.get("w", 1)
        stride_h = miopen_params.get("u", 1)
        stride_w = miopen_params.get("v", 1)

        dilation_d = miopen_params.get("r", 1)
        dilation_h = miopen_params.get("l", 1)
        dilation_w = miopen_params.get("j", 1)

        pad_d = miopen_params.get("s", 0)
        pad_h = miopen_params.get("p", 0)
        pad_w = miopen_params.get("q", 0)

        # Calculate output dimensions
        output_d = (
            input_d + 2 * pad_d - dilation_d * (kernel_d - 1) - 1
        ) // stride_d + 1
        output_h = (
            input_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1
        ) // stride_h + 1
        output_w = (
            input_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1
        ) // stride_w + 1

        # Skip invalid configurations
        if output_d <= 0 or output_h <= 0 or output_w <= 0:
            return None

        direction = miopen_params.get("F", 1)  # 1=fwd, 2=bwd_weight, 4=bwd_data
        direction_name = {1: "fwd", 2: "bwd_weight", 4: "bwd_data"}.get(
            direction, "fwd"
        )

        return {
            "NDim": ndim,
            "Groups": groups,
            "BatchSize": batch_size,
            "OutChannels": out_channels,
            "InChannels": in_channels,
            "KernelD": kernel_d,
            "KernelH": kernel_h,
            "KernelW": kernel_w,
            "InputD": input_d,
            "InputH": input_h,
            "InputW": input_w,
            "OutputD": output_d,
            "OutputH": output_h,
            "OutputW": output_w,
            "StrideD": stride_d,
            "StrideH": stride_h,
            "StrideW": stride_w,
            "DilationD": dilation_d,
            "DilationH": dilation_h,
            "DilationW": dilation_w,
            "LeftPadD": pad_d,
            "LeftPadH": pad_h,
            "LeftPadW": pad_w,
            "RightPadD": pad_d,
            "RightPadH": pad_h,
            "RightPadW": pad_w,
            "TestName": f"MIOpen_3D_{direction_name}",
        }

    else:
        # 2D convolution
        kernel_h = miopen_params.get("y", 3)
        kernel_w = miopen_params.get("x", 3)

        input_h = miopen_params.get("H", 32)
        input_w = miopen_params.get("W", 32)

        stride_h = miopen_params.get("u", 1)
        stride_w = miopen_params.get("v", 1)

        dilation_h = miopen_params.get("l", 1)
        dilation_w = miopen_params.get("j", 1)

        pad_h = miopen_params.get("p", 0)
        pad_w = miopen_params.get("q", 0)

        # Calculate output dimensions
        output_h = (
            input_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1
        ) // stride_h + 1
        output_w = (
            input_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1
        ) // stride_w + 1

        # Skip invalid configurations
        if output_h <= 0 or output_w <= 0:
            return None

        direction = miopen_params.get("F", 1)
        direction_name = {1: "fwd", 2: "bwd_weight", 4: "bwd_data"}.get(
            direction, "fwd"
        )

        return {
            "NDim": ndim,
            "Groups": groups,
            "BatchSize": batch_size,
            "OutChannels": out_channels,
            "InChannels": in_channels,
            "KernelH": kernel_h,
            "KernelW": kernel_w,
            "InputH": input_h,
            "InputW": input_w,
            "OutputH": output_h,
            "OutputW": output_w,
            "StrideH": stride_h,
            "StrideW": stride_w,
            "DilationH": dilation_h,
            "DilationW": dilation_w,
            "LeftPadH": pad_h,
            "LeftPadW": pad_w,
            "RightPadH": pad_h,
            "RightPadW": pad_w,
            "TestName": f"MIOpen_2D_{direction_name}",
        }


def write_csv_cases(test_cases, output_file, ndim):
    """Write test cases to CSV file"""
    if not test_cases:
        print(f"No {ndim}D test cases to write")
        return

    print(f"Writing {len(test_cases)} {ndim}D test cases to {output_file}")

    # Define CSV headers based on dimension
    if ndim == 2:
        headers = [
            "NDim",
            "Groups",
            "BatchSize",
            "OutChannels",
            "InChannels",
            "KernelH",
            "KernelW",
            "InputH",
            "InputW",
            "OutputH",
            "OutputW",
            "StrideH",
            "StrideW",
            "DilationH",
            "DilationW",
            "LeftPadH",
            "LeftPadW",
            "RightPadH",
            "RightPadW",
            "TestName",
        ]
    else:  # 3D
        headers = [
            "NDim",
            "Groups",
            "BatchSize",
            "OutChannels",
            "InChannels",
            "KernelD",
            "KernelH",
            "KernelW",
            "InputD",
            "InputH",
            "InputW",
            "OutputD",
            "OutputH",
            "OutputW",
            "StrideD",
            "StrideH",
            "StrideW",
            "DilationD",
            "DilationH",
            "DilationW",
            "LeftPadD",
            "LeftPadH",
            "LeftPadW",
            "RightPadD",
            "RightPadH",
            "RightPadW",
            "TestName",
        ]

    with open(output_file, "w", newline="") as csvfile:
        # Write header comment
        csvfile.write(f"# {ndim}D Convolution Test Cases from MIOpen Commands\n")
        csvfile.write(f"# Generated {len(test_cases)} test cases\n")

        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        for test_case in test_cases:
            # Only write fields that exist in headers
            filtered_case = {k: v for k, v in test_case.items() if k in headers}
            writer.writerow(filtered_case)


def main():
    parser = argparse.ArgumentParser(
        description="Convert MIOpen commands to CSV test cases"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file with MIOpen driver commands",
    )
    parser.add_argument(
        "--output", type=str, help="Output CSV file (for mixed 2D/3D cases)"
    )
    parser.add_argument(
        "--output-2d",
        type=str,
        default="miopen_conv_2d.csv",
        help="Output CSV file for 2D cases",
    )
    parser.add_argument(
        "--output-3d",
        type=str,
        default="miopen_conv_3d.csv",
        help="Output CSV file for 3D cases",
    )
    parser.add_argument(
        "--filter-duplicates", action="store_true", help="Remove duplicate test cases"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="MIOpen",
        help="Model name to use in test case names (default: MIOpen)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        return 1

    print(f"Parsing MIOpen commands from {args.input}...")

    test_cases_2d = []
    test_cases_3d = []
    total_lines = 0
    parsed_lines = 0

    with open(args.input, "r") as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            line = line.strip()

            # Skip empty lines and non-MIOpen commands
            # Handle both direct commands and logged commands with MIOpen prefix
            if not line:
                continue

            # Extract the actual MIOpenDriver command from logged format
            if "MIOpenDriver conv" in line:
                # Extract command after finding MIOpenDriver
                command_start = line.find("./bin/MIOpenDriver conv")
                if command_start != -1:
                    line = line[command_start:]
                else:
                    # Handle cases where path might be different - create standard format
                    driver_start = line.find("MIOpenDriver conv")
                    if driver_start != -1:
                        line = "./bin/" + line[driver_start:]
                    else:
                        continue
            elif not line.startswith("./bin/MIOpenDriver conv"):
                continue

            try:
                # Parse MIOpen command
                miopen_params = parse_miopen_command(line)
                if not miopen_params:
                    continue

                # Convert to ConvParam format
                conv_param = miopen_to_conv_param(miopen_params)
                if not conv_param:
                    continue

                # Add model name to test name
                conv_param["TestName"] = f"{args.model_name}_{conv_param['NDim']}D_fwd"

                # Separate 2D and 3D cases
                if conv_param["NDim"] == 2:
                    test_cases_2d.append(conv_param)
                else:
                    test_cases_3d.append(conv_param)

                parsed_lines += 1

            except Exception as e:
                print(f"WARNING: Failed to parse line {line_num}: {e}")
                continue

    print(f"Processed {total_lines} lines, parsed {parsed_lines} commands")
    print(f"Found {len(test_cases_2d)} 2D cases, {len(test_cases_3d)} 3D cases")

    # Remove duplicates if requested
    if args.filter_duplicates:
        # Simple duplicate removal based on key parameters
        def make_key(case):
            if case["NDim"] == 2:
                return (
                    case["Groups"],
                    case["BatchSize"],
                    case["OutChannels"],
                    case["InChannels"],
                    case["KernelH"],
                    case["KernelW"],
                    case["InputH"],
                    case["InputW"],
                    case["StrideH"],
                    case["StrideW"],
                )
            else:
                return (
                    case["Groups"],
                    case["BatchSize"],
                    case["OutChannels"],
                    case["InChannels"],
                    case["KernelD"],
                    case["KernelH"],
                    case["KernelW"],
                    case["InputD"],
                    case["InputH"],
                    case["InputW"],
                    case["StrideD"],
                    case["StrideH"],
                    case["StrideW"],
                )

        seen_2d = set()
        unique_2d = []
        for case in test_cases_2d:
            key = make_key(case)
            if key not in seen_2d:
                seen_2d.add(key)
                unique_2d.append(case)

        seen_3d = set()
        unique_3d = []
        for case in test_cases_3d:
            key = make_key(case)
            if key not in seen_3d:
                seen_3d.add(key)
                unique_3d.append(case)

        print(
            f"After deduplication: {len(unique_2d)} 2D cases, {len(unique_3d)} 3D cases"
        )
        test_cases_2d = unique_2d
        test_cases_3d = unique_3d

    # Write output files
    if args.output:
        # Write mixed cases to single file
        all_cases = test_cases_2d + test_cases_3d
        if all_cases:
            print(f"Writing {len(all_cases)} total cases to {args.output}")
            # Use 2D headers for mixed file, extend as needed
            mixed_headers = [
                "NDim",
                "Groups",
                "BatchSize",
                "OutChannels",
                "InChannels",
                "KernelH",
                "KernelW",
                "InputH",
                "InputW",
                "OutputH",
                "OutputW",
                "StrideH",
                "StrideW",
                "DilationH",
                "DilationW",
                "LeftPadH",
                "LeftPadW",
                "RightPadH",
                "RightPadW",
                "TestName",
            ]

            with open(args.output, "w", newline="") as csvfile:
                csvfile.write(
                    "# Mixed 2D/3D Convolution Test Cases from MIOpen Commands\n"
                )
                writer = csv.DictWriter(
                    csvfile, fieldnames=mixed_headers, extrasaction="ignore"
                )
                writer.writeheader()
                for case in all_cases:
                    writer.writerow(case)
    else:
        # Write separate files for 2D and 3D
        if test_cases_2d:
            write_csv_cases(test_cases_2d, args.output_2d, 2)

        if test_cases_3d:
            write_csv_cases(test_cases_3d, args.output_3d, 3)

    print("Conversion completed!")
    return 0


if __name__ == "__main__":
    exit(main())
