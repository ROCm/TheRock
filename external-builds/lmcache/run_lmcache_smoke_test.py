#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import importlib
import sys

import torch


NATIVE_MODULES = (
    "lmcache.c_ops",
    "lmcache.native_storage_ops",
    "lmcache.lmcache_fs",
    "lmcache.lmcache_redis",
)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test an LMCache ROCm wheel on an AMD GPU"
    )
    parser.add_argument(
        "--expected-arch",
        help="Expected concrete GPU architecture, such as gfx942",
    )
    args = parser.parse_args(argv)

    if torch.version.hip is None:
        raise RuntimeError("PyTorch was not built with ROCm support")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access a ROCm GPU")

    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    actual_arch = properties.gcnArchName.split(":", 1)[0]
    if args.expected_arch and actual_arch != args.expected_arch:
        raise RuntimeError(
            f"Expected GPU architecture {args.expected_arch}, found {actual_arch}"
        )

    tensor = torch.arange(16, dtype=torch.float32, device="cuda")
    if tensor.sum().item() != 120:
        raise RuntimeError("Unexpected result from the GPU smoke test")

    for module_name in NATIVE_MODULES:
        importlib.import_module(module_name)

    print(f"PyTorch: {torch.__version__}")
    print(f"HIP: {torch.version.hip}")
    print(f"GPU: {properties.name} ({actual_arch})")
    print("LMCache native extensions imported successfully")


if __name__ == "__main__":
    main(sys.argv[1:])
