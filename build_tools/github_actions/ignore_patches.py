"""
This script provides the ability to ignore specific patches while applying other patches from TheRock

As we land fixes for patches to other repositories and attempt to patch an outdated patch, this creates conflicts,
resulting in broken builds and iterations. This script will remediate those conflicts.

This is a temporary solution, as we continue to land patches and prevent blocks. This script will become obsolete after patches are no longer required.

Usage:
python3 ignore_patches.py [-h] --patch-dir PATCH_DIR [--ignore-patches IGNORE_PATCHES [IGNORE_PATCHES ...]]

Example:
- python3 ignore_patches.py --patch-dir ./patches/ --ignore-patches "0001_rocprim_patch" = applies all patches from `./patches/` directory except "0001_rocprim_patch"
"""
import argparse
import os
from pathlib import Path
import subprocess


def run(args):
    patch_dir = args.patch_dir
    ignore_patches = args.ignore_patches

    # Retrieving files from the patch directory
    patch_files = os.listdir(patch_dir)

    # Filter function to check if a string from array "ignore_patches" is matched in a patch file name
    def keep_file(patch_file):
        return not any(file_name in patch_file for file_name in ignore_patches)

    # Filtering out patch files that are listed in "ignore_patches"
    filtered_patch_files = list(filter(keep_file, patch_files))

    for patch_file in filtered_patch_files:
        cmd = [
            "git",
            "-c",
            "user.name='therockbot'",
            "-c",
            "user.email='therockbot@amd.com'",
            "am",
            "--whitespace=nowarn",
            str(patch_dir / patch_file),
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TheRock Patch script")
    parser.add_argument(
        "--patch-dir",
        type=Path,
        required=True,
        help="Patch directory containing patch",
    )
    parser.add_argument(
        "--ignore-patches",
        nargs="+",
        help="List of patches to ignore separated by space",
    )
    args = parser.parse_args()
    run(args)
