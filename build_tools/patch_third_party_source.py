# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# patch_third_party_source.py is designed to act as the PATCH_COMMAND for
#  therock_subproject_fetch
#  ExternalProject_Add
# from CMake.
# See https://cmake.org/cmake/help/latest/module/ExternalProject.html#patch-step-options
#
# SYNOPSIS: patch_third_party_source.py PATCHES_DIR
#
# Uses `patch -p1 -i PATCH` for the actual patching.  On Windows, the patch file
# is fed to `patch --binary -p1` on stdin instead, with LF line endings.
# Assumes the current working directory is the source directory of the extracted tarball.

import sys
import os
import platform
import subprocess
from pathlib import Path


def run_command(cmd_list, cwd=None, stdin_data=None):
    print(f"\n--- Executing: {' '.join(map(str, cmd_list))} ---", flush=True)
    try:
        process = subprocess.run(cmd_list, cwd=cwd, check=True, input=stdin_data)
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


def apply_patch(patch_file, source_dir):
    """Apply PATCH_FILE, a Path, to the source tree in SOURCE_DIR with
    `patch -p1`.  On Windows, CRLF line endings in PATCH_FILE are first turned
    into LF.  Source files with CRLF line endings cannot be patched.  Exits the
    process if the patch program fails."""
    if platform.system() != "Windows":
        run_command(["patch", "-p1", "-i", patch_file.absolute()], cwd=source_dir)
        return
    # On Windows, which patch program comes first on PATH varies, and so do the
    # patch files' line endings.  They are LF in the repository, which has no
    # .gitattributes, so a checkout made with core.autocrlf=true, Git for
    # Windows' default, has them CRLF.  The patch programs disagree on those,
    # as observed with these versions:
    #
    #   - Strawberry Perl's (GNU patch 2.5.9), which windows_support.md
    #     suggests, fails an assertion ("Expression: hunk") on an LF patch file
    #     without --binary.  It applies a CRLF one, but rewrites the patched
    #     files with CRLF line endings.
    #   - Git for Windows' (GNU patch 2.7.6) fails on a CRLF patch file given
    #     --binary, with "different line endings".
    #
    # Both apply an LF patch file with --binary, and leave the sources' LF line
    # endings alone.  So undo what autocrlf did, and hand the patch program the
    # repository's LF patch file, on stdin.  That also strips a CR that a patch
    # for a CRLF source file carries as content, which is why such a file
    # cannot be patched.
    print(f"--- Applying {patch_file} with LF line endings ---", flush=True)
    run_command(
        ["patch", "--binary", "-p1"],
        cwd=source_dir,
        stdin_data=patch_file.read_bytes().replace(b"\r\n", b"\n"),
    )


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
        apply_patch(patches_dir / i, Path.cwd())
    create_stamp_file(stamp_filename)


if __name__ == "__main__":
    main(sys.argv[1:])
