# patch_third_party_source.py is designed to act as the PATCH_COMMAND for
#  therock_subproject_fetch
#  ExternalProject_Add
# from CMake.
# See https://cmake.org/cmake/help/latest/module/ExternalProject.html#patch-step-options
#
# SYNOPSIS: patch_third_party_source.py PATCHES_DIR
#
# Uses `patch -p1 -i PATCH` for the actual patching.
# Assumes the current working directory is the source directory of the extracted tarball.

import sys
import os
import subprocess
from pathlib import Path


def run_command(cmd_list, cwd=None):
    print(f"\n--- Executing: {' '.join(map(str, cmd_list))} ---", flush=True)
    try:
        process = subprocess.run(cmd_list, cwd=cwd, check=True, text=True)
    except FileNotFoundError:
        print(
            f"ERROR: Command not found: {cmd_list[0]}. Is it installed and in PATH?",
            file=sys.stderr,
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)
    return process


def create_stamp_file(filename):
    with open(filename, "w") as f:
        pass


def main(args):
    try:
        (patches_dir,) = args
    except ValueError:
        sys.stderr.write("usage: patch_third_party_source PATCHES_DIR\n")
        sys.exit(2)

    stamp_filename = "patch_third_party_source.stamp"
    if os.path.exists(stamp_filename):
        sys.exit(0)

    patches = sorted(os.listdir(patches_dir))
    patches_dir = Path(patches_dir)
    for i in patches:
        p = patches_dir / i
        run_command(["patch", "-p1", "-i", p])
    create_stamp_file(stamp_filename)


if __name__ == "__main__":
    main(sys.argv[1:])
