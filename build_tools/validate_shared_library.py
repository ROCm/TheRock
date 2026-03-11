#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validates that a shared library can be loaded."""

import argparse
import ctypes
import sys


def run(args: argparse.Namespace):
    for shared_lib in args.shared_libs:
        print(f"Validating shared library: {shared_lib}", end="")
        so = ctypes.cdll.LoadLibrary(shared_lib)
        print(" :", so)


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("shared_libs", nargs="*", help="Shared libraries to validate")
    args = p.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
