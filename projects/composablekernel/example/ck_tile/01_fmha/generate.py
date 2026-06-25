# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# generate kernel instances to speed up compilation

import argparse
from enum import IntEnum
from pathlib import Path
import pkgutil
from typing import List, Optional

import codegen.ops
from codegen.cmake_config import GEN_DIR


class HandlerId(IntEnum):
    LIST_BLOBS = 0
    WRITE_BLOBS = 1


# inspect all modules under 'codegen.ops' and register API handlers
ops = []
for importer, module_name, _ in pkgutil.iter_modules(codegen.ops.__path__):
    full_module_name = "%s.%s" % (codegen.ops.__name__, module_name)
    ops.append(importer.find_spec(module_name).loader.load_module(module_name))
unwanted_prefix = "fmha_"
handlers = dict(
    [
        (
            op.__name__[len(unwanted_prefix) :]
            if op.__name__.startswith(unwanted_prefix)
            else op.__name__,
            (op.list_blobs, op.write_blobs),
        )
        for op in ops
    ]
)
assert 0 < len(handlers)


def write_blobs(
    targets: List[str],
    output_dir: Optional[str],
    api_list: List[str],
    filters_list: List[str],
    optdim_list: List[int],
    receipt,
    mask_impl,
) -> None:
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir) / GEN_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    for api, kernel_filter in zip(api_list, filters_list):
        handler = handlers[api][HandlerId.WRITE_BLOBS]
        handler(targets, output_dir, kernel_filter, receipt, optdim_list, mask_impl)


# list all the files that will be generated
def list_blobs(
    targets: List[str],
    output_file: Optional[str],
    api_list: List[str],
    filters_list: List[str],
    optdim_list: List[int],
    receipt,
    mask_impl,
) -> None:
    assert output_file is not None
    file_path = Path(output_file)

    # create an empty file / drop its contents if it exists
    open(file_path, "w").close()

    for api, kernel_filter in zip(api_list, filters_list):
        handler = handlers[api][HandlerId.LIST_BLOBS]
        handler(targets, file_path, kernel_filter, receipt, optdim_list, mask_impl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for CK fmha kernel",
    )
    parser.add_argument(
        "--targets",
        default="gfx9,gfx950",
        required=False,
        help="list of GPU targets, separated by comma.",
    )
    parser.add_argument(
        "-d",
        "--direction",  # we keep 'direction' option for backward compatibility
        "-a",
        "--api",
        default="fwd",
        required=False,
        help="supply API(s) to generate (default: fwd). separated by comma.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="write all the blobs into a directory",
    )
    parser.add_argument(
        "-l", "--list_blobs", required=False, help="list all the kernels to a file"
    )
    # TODO: if using filter, must apply same value to output_dir and list_blobs
    parser.add_argument(
        "-f",
        "--filter",
        default="",
        required=False,
        help="filter out kernels that need to generate, using fnmatch module",
    )

    parser.add_argument(
        "-m",
        "--mask",
        default="simplified",
        required=False,
        help="mask implementation, simplified/generic",
    )

    parser.add_argument(
        "-r",
        "--receipt",
        default=0,
        required=False,
        help="codegen receipt. 0: generate only 8xhdim coverage\n"
        + "  1: generate more instance to cover all hdim\n"
        + "  2: Only generate instance for Flash attention integration\n"
        + "  4: Only generate instance for PyTorch integration\n"
        + "  100-199: Only generate instance for Aiter(mha_fwd) integration\n"
        + "  200-299: Only generate instance for Aiter(mha_varlen_fwd) integration\n"
        + "  300-399: Only generate instance for Aiter(mha_bwd) integration\n"
        + "  400-499: Only generate instance for Aiter(mha_varlen_bwd) integration\n"
        + "  600-699: Only generate instance for aiter::mha_fwd && aiter::mha_fwd_splitkv && aiter::mha_bwd C++ api integration",
    )

    parser.add_argument(
        "--optdim",
        default="-1",
        required=False,
        help="only optimize the hdim in the list. separated by comma. -1 is the default choice"
        + "eg. --optdim=32,64,128,256",
    )

    args = parser.parse_args()
    targets = args.targets.split(",")
    api_list = args.direction.split(",")
    filter_list = args.filter.split(",")
    filter_list.extend([""] * (len(api_list) - len(filter_list)))
    optdim_list = [int(hdim) for hdim in args.optdim.split(",")]

    if args.list_blobs is not None:
        list_blobs(
            targets,
            args.list_blobs,
            api_list,
            filter_list,
            optdim_list,
            int(args.receipt),
            mask_impl=args.mask,
        )
    else:
        write_blobs(
            targets,
            args.output_dir,
            api_list,
            filter_list,
            optdim_list,
            int(args.receipt),
            mask_impl=args.mask,
        )
