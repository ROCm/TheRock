# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compile or execute a Triton kernel that uses the HIP hostcall protocol."""

import argparse

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def hostcall_kernel():
    tl.device_print("TheRock hostcall smoke")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.compile_only:
        triton.compile(
            ASTSource(fn=hostcall_kernel, signature={}),
            target=GPUTarget("hip", "gfx942", 64),
            options={"num_warps": 1},
        )
        print("hostcall compiled", flush=True)
        return

    # The wheel build runs --compile-only before PyTorch is built. Keep this
    # import in the execution path so compilation does not require torch.
    import torch

    torch.empty(1, device="cuda")
    hostcall_kernel[(1,)](num_warps=1)
    torch.cuda.synchronize()
    print("hostcall synchronized", flush=True)


if __name__ == "__main__":
    main()
