#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path

LLVM18_PREBUILT = r"""
# Install prebuilt LLVM 18.1.8 for JAX releases using clang_local.
RUN --mount=type=cache,target=/var/cache/dnf \
    dnf install -y wget xz ncurses-compat-libs && \
    dnf clean all

RUN wget -q \
      "https://github.com/llvm/llvm-project/releases/download/llvmorg-18.1.8/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04.tar.xz" \
      -O /tmp/llvm.tar.xz && \
    echo "54ec30358afcc9fb8aa74307db3046f5187f9fb89fb37064cdde906e062ebf36  /tmp/llvm.tar.xz" \
      | sha256sum -c - && \
    mkdir -p /usr/lib/llvm-18 && \
    tar -xJf /tmp/llvm.tar.xz \
      -C /usr/lib/llvm-18 \
      --strip-components=1 && \
    rm /tmp/llvm.tar.xz && \
    printf '%s\n' \
      '--gcc-toolchain=/opt/rh/gcc-toolset-14/root/usr' \
      > /usr/lib/llvm-18/bin/clang.cfg && \
    cp /usr/lib/llvm-18/bin/clang.cfg \
      /usr/lib/llvm-18/bin/clang++.cfg && \
    /usr/lib/llvm-18/bin/clang --version
"""


LOCAL_LLVM_REFS = {
    "rocm-jaxlib-v0.10.0",
    "rocm-jaxlib-v0.10.1",
}

HERMETIC_LLVM_REFS = {
    "rocm-jaxlib-v0.10.2",
    "rocm-jaxlib-v0.11.0",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dockerfile", type=Path, required=True)
    parser.add_argument("--jax-ref", required=True)
    args = parser.parse_args()

    dockerfile = args.dockerfile.read_text(encoding="utf-8")

    start_marker = "# Install LLVM 18 from source"
    end_marker = "# ld.lld (hermetic linker)"

    start = dockerfile.find(start_marker)
    end = dockerfile.find(end_marker, start)

    if start == -1 or end == -1:
        raise RuntimeError("Could not find LLVM source-build block")

    if args.jax_ref in LOCAL_LLVM_REFS:
        print(f"{args.jax_ref}: using prebuilt LLVM 18.1.8")
        replacement = LLVM18_PREBUILT + "\n"

    elif args.jax_ref in HERMETIC_LLVM_REFS:
        print(f"{args.jax_ref}: using hermetic LLVM")
        replacement = ""

    else:
        raise RuntimeError(f"Unsupported JAX ref: {args.jax_ref}")

    args.dockerfile.write_text(
        dockerfile[:start] + replacement + dockerfile[end:],
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
