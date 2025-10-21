#!/usr/bin/env python3
"""Checks for long path lengths under a given root directory.

Usage example:
    python ./build_tools/hack/check_path_lengths.py rocm-libraries --limit 220

Example output:
    ```
    These paths are longer than the limit of 200 characters:
    201, projects/composablekernel/library/src/tensor_operation_instance/gpu/gemm_ab_scale/device_gemm_ab_scale_xdl_f8_f8_bf16/device_gemm_ab_scale_xdl_f8_f8_bf16_km_kn_mn_128_128_128_comp_kpadding_instance.cpp
    201, projects/composablekernel/library/src/tensor_operation_instance/gpu/gemm_ab_scale/device_gemm_ab_scale_xdl_f8_f8_bf16/device_gemm_ab_scale_xdl_f8_f8_bf16_mk_kn_mn_128_128_128_comp_kpadding_instance.cpp
    201, projects/composablekernel/library/src/tensor_operation_instance/gpu/gemm_ab_scale/device_gemm_ab_scale_xdl_f8_f8_bf16/device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_comp_kpadding_instance.cpp
    201, projects/composablekernel/library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_wmma_universal_bf16_bf16_bf16/device_gemm_wmma_universal_bf16_bf16_bf16_km_kn_mn_comp_default_instance.cpp
    ...
    201, projects/miopen/src/kernels/dynamic_igemm/igemm_gtc_xdlops_nhwc_gfx90a/fwd_fp16/igemm_fwd_gtcx2_nhwc_fp16_bx0_ex0_bt128x64x32_wt32x32x8_ws1x1_wr1x2_ta1x16x1x1_1x2x4x32_tb1x8x1x1_1x4x1x64_pta_vs1_gkgs.s
    201, projects/miopen/src/kernels/dynamic_igemm/igemm_gtc_xdlops_nhwc_gfx90a/fwd_fp16/igemm_fwd_gtcx2_nhwc_fp16_bx0_ex1_bt128x64x32_wt32x32x8_ws1x1_wr1x2_ta1x16x1x1_1x2x4x32_tb1x8x1x1_1x4x1x64_pta_vs1_gkgs.s
    ...
    237, projects/composablekernel/library/src/tensor_operation_instance/gpu/gemm_universal_preshuffle/device_gemm_xdl_universal_preshuffle_f8_f8_bf16/device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma16x16_nk_mn_comp_default_instance_p5.cpp
    237, projects/composablekernel/library/src/tensor_operation_instance/gpu/gemm_universal_preshuffle/device_gemm_xdl_universal_preshuffle_f8_f8_bf16/device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma16x16_nk_mn_comp_default_instance_p6.cpp
    Error: 485 source paths are longer than 200 characters.
      Long paths can be problematic when building on Windows.
      Please look at the output above and trim the paths.
    ```
"""

import argparse
from pathlib import Path
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(description="Path length checker")
    parser.add_argument("root_dir", help="", type=Path)
    parser.add_argument(
        "--limit", help="Path length limit (inclusive)", type=int, default=200
    )
    parser.add_argument(
        "--verbose",
        help="Outputs detailed information about path lengths",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    return args


def check_path_lengths(root_dir: Path, limit: int, verbose: bool):
    longest_path_length = -1
    long_paths = []
    short_paths = []

    for dirpath, _, filenames in root_dir.walk():
        for filename in filenames:
            path = (dirpath / filename).relative_to(root_dir).as_posix()
            if len(path) > args.limit:
                long_paths.append(path)
            else:
                short_paths.append(path)
            longest_path_length = max(longest_path_length, len(path))
    long_paths.sort(key=len)
    short_paths.sort(key=len)

    if args.verbose and short_paths:
        print(f"These paths are shorter than the limit of {args.limit} characters:")
        for path in short_paths:
            print("{:3d}, {}".format(len(path), path))
        print("")

    if long_paths:
        print(f"These paths are longer than the limit of {args.limit} characters:")
        for path in long_paths:
            print("{:3d}, {}".format(len(path), path))
        print(
            f"Error: {len(long_paths)} source paths are longer than {args.limit} characters."
        )
        print("  Long paths can be problematic when building on Windows.")
        print("  Please look at the output above and trim the paths.")
        sys.exit(1)
    else:
        print(f"All path lengths are under the limit of {args.limit} characters.")


if __name__ == "__main__":
    args = parse_arguments()
    check_path_lengths(root_dir=args.root_dir, limit=args.limit, verbose=args.verbose)
