#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# -*- coding: utf-8 -*-

from enum import Enum


def parse_args():
    """
    Parse command-line arguments
    -   --shapes_csv : input csv file with M, N, K integer columns
    -   --best       : if set, store only the result reported by the best instance.
                       if not set, store results from all instances
    -   -o           : output csv file
    -   --build_dir  : path to directory where CMake stores all the build artifacts.
                       The profiler binary is bin/ckProfiler relative to this directory.
    -   --op_name    : operator name
    -   --layout     : inputs and output layout
                       r ~ row-major
                       c ~ col-major
                       p ~ preshuffled for mfma
    -   --dtype      : inputs and output dtype
    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--shapes_csv",
        required=True,
        help="Input csv file with M, N, K integer columns",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="If set, store only the result reported by the best instance. If not set, store results from all instances",
    )
    parser.add_argument("-o", default="out.csv", help="Output csv file")
    parser.add_argument(
        "--build_dir",
        default=".",
        help="Path to directory where CMake stores all the build artifacts. The profiler binary is bin/ckProfiler relative to this directory.",
    )
    parser.add_argument(
        "--op_name",
        default="gemm_multiply_multiply_weight_preshuffle",
        help="Operator name",
    )
    parser.add_argument(
        "--layout",
        default="rpr",
        help="Inputs and output layout. r ~ row-major, c ~ col-major, p ~ preshuffled for mfma.",
    )
    parser.add_argument("--dtype", default="f8f8bf16", help="Inputs and output dtype.")

    return vars(parser.parse_args())


def tuples(filename):
    """
    Parse M, N, K integers from the input csv file
    """
    lines = []
    with open(filename, "r", newline="") as f:
        import csv

        reader = csv.reader(f)
        for line in reader:
            try:
                m, n, k = map(int, line)
                lines.append((m, n, k))
            except Exception:
                pass
    return lines


def parse_result(line):
    """
    Parse the ckProfiler stdout line.
    Result: a dict with the instance metadata and performance results
    """
    words = line.split()
    fields = dict()
    if "Perf:" in words or "Perf" in words:
        for key in ("ms", "TFlops", "GB/s"):
            fields[key] = words[words.index(key + ",") - 1]
    for key in (
        "BlkSize:",
        "BlkTile:",
        "WaveTile:",
        "WaveMap:",
        "VmemReadVec:",
        "BlkGemmPipelineScheduler:",
        "BlkGemmPipelineVersion:",
        "BlkGemmPipelinePrefetchStages:",
    ):
        fields[key.strip(":")] = words[words.index(key) + 1].strip(",")
    if "KBatch" in words:
        key = "KBatch"
        fields[key] = words[words.index(key) + 1]

    return fields


class GemmMulMulWP:
    """
    Wrapper for ckProfiler CLI parameters specific to gemm_multiply_multiply_weight_preshuffle
    """

    dtype = Enum("dtype", [("f8f8f16", 0), ("f8f8bf16", 1)])
    layout = Enum("layout", [("rpr", 0)])


class GemmMulMul:
    """
    Wrapper for ckProfiler CLI parameters specific to gemm_multiply_multiply
    """

    dtype = Enum(
        "dtype",
        [
            ("f32f32f32", 0),
            ("f16f16f16", 1),
            ("bf16bf16bf16", 2),
            ("i8i8i8", 3),
            ("f8f16f16", 4),
            ("f16f8f16", 5),
            ("f16f16f8", 6),
            ("f8f8bf16", 7),
            ("i8i8bf16", 8),
            ("i8i8f16", 9),
            ("f8f8f16", 10),
        ],
    )
    layout = Enum(
        "layout",
        [
            ("rrr", 0),
            ("rcr", 1),
            ("crr", 2),
            ("ccr", 3),
        ],
    )


OPs = Enum(
    "ops",
    [
        ("gemm_multiply_multiply_weight_preshuffle", GemmMulMulWP),
        ("gemm_multiply_multiply", GemmMulMul),
    ],
)


def run_shape(shape, profiler_bin, op_name, dtype, layout):
    """
    Launch ckProfiler in subprocess and collect its stdout
    """
    import subprocess

    m, n, k = shape
    try:
        op = OPs[op_name]
    except KeyError:
        raise AssertionError(f"Invalid operator {op_name}")
    name_arg = op.name
    op_wrapper = op.value()

    try:
        dtype_arg = str(op_wrapper.dtype[dtype].value)
    except KeyError:
        raise AssertionError(f"Invalid dtype for {op_name}: {dtype}")

    try:
        layout_wrapper = op_wrapper.layout[layout]
    except KeyError:
        raise AssertionError(f"Invalid layout for {op_name}: {layout}")
    layout_arg = str(layout_wrapper.value)
    # verification: no, initialization: decimal, print tensor: no, time kernel: yes
    meta_args = map(str, [0, 2, 0, 1])

    layout_a = layout_wrapper.name[0]
    if layout_a == "r":
        stride_a = k
    elif layout_a == "c":
        stride_a = n
    else:
        raise AssertionError(
            f"Couldn't decide StrideA from layout {layout_wrapper.name}"
        )

    layout_b = layout_wrapper.name[1]
    if layout_b == "r":
        stride_b = n
    elif layout_b in ("c", "p"):
        stride_b = k
    else:
        raise AssertionError(
            f"Couldn't decide StrideB from layout {layout_wrapper.name}"
        )

    # M, N, K, StrideA, StrideB, StrideD0, StrideD1, StrideE
    shape_args = map(str, [m, n, k, stride_a, stride_b, 0, 0, n])
    # kBatch, number of warm-up cycles, number of iterations, rotating buffer size in MB
    control_args = map(str, [1, 50, 10, 4096])

    cmd = [
        profiler_bin,
        name_arg,
        dtype_arg,
        layout_arg,
        *meta_args,
        *shape_args,
        *control_args,
    ]
    print(" ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    ).stdout

    return result.splitlines()


def filter_output_line(result_line, best_only):
    """
    Filter out ckProfiler output lines which don't report performance results
    """
    if "DeviceGemmXdlUniversal" in result_line:
        if best_only:
            if "Best Perf" in result_line:
                return True
        else:
            if "Best Perf" not in result_line:
                return True
    return False


def write_results(filename, results):
    """
    Write out the performance results to a csv file
    """
    if not results:
        return
    with open(filename, "w", newline="") as f:
        import csv

        fields = list(results[0].keys())
        writer = csv.DictWriter(f, dialect="unix", fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def add_shape_to_metadata(shape, metadata):
    """
    Adds M, N, K to the parsed profiler results
    """
    m, n, k = shape
    return metadata | {"M": m, "N": n, "K": k}


def main():
    """
    Main driver:
    - parses command line arguments
    - parses input shapes to run ckProfiler with
    - for each shape,
       - runs ckProfiler
       - parses the ckProfiler output
    - writes out the results for all shapes
    """
    args = parse_args()
    filename = args["shapes_csv"]
    shapes = tuples(filename)

    all_results = []
    from functools import partial
    from os import path

    profiler_bin = path.join(args["build_dir"], "bin", "ckProfiler")

    try:
        from tqdm import tqdm as iterate
    except ImportError:

        def iterate(x):
            return x

    for s in iterate(shapes):
        run_shape_stdout_lines = run_shape(
            s, profiler_bin, args["op_name"], args["dtype"], args["layout"]
        )
        results_single_shape = map(
            lambda r: add_shape_to_metadata(s, r),
            map(
                parse_result,
                filter(
                    partial(filter_output_line, best_only=args["best"]),
                    run_shape_stdout_lines,
                ),
            ),
        )
        all_results.extend(list(results_single_shape))

    write_results(args["o"], all_results)


if __name__ == "__main__":
    main()
