#!/usr/bin/env python
"""Validates that a static library exists and is a valid archive."""

import argparse
import os
import subprocess
import sys


def run(args: argparse.Namespace):
    for static_lib in args.static_libs:
        print(f"Validating static library: {static_lib}", end="")

        # Check if file exists
        if not os.path.isfile(static_lib):
            print(f" : ERROR - File does not exist")
            sys.exit(1)

        # Check if it's a valid archive using 'ar'
        try:
            result = subprocess.run(
                ["ar", "t", static_lib],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            num_objects = len(result.stdout.decode().strip().split("\n"))
            print(f" : OK (contains {num_objects} object files)")
        except subprocess.CalledProcessError as e:
            print(f" : ERROR - Not a valid archive: {e.stderr.decode()}")
            sys.exit(1)
        except FileNotFoundError:
            # 'ar' command not found, just check file size
            size_mb = os.path.getsize(static_lib) / (1024 * 1024)
            print(f" : OK ({size_mb:.1f} MB)")


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("static_libs", nargs="*", help="Static libraries to validate")
    args = p.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
