#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Post-install fixup script for the libiberty build (source: binutils tarball).

Creates a versioned libiberty shared library from the PIC-compiled static
archive, and removes build artifacts that are not needed at runtime.
"""

import glob
import os
import shutil
import subprocess
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: patch_install.py <install-prefix>", file=sys.stderr)
        sys.exit(1)

    prefix = sys.argv[1]
    patchelf = os.environ.get("PATCHELF", "patchelf")
    cc = os.environ.get("CC", "cc")

    print("Patching libiberty install...")

    lib_dir = os.path.join(prefix, "lib")
    lib64_dir = os.path.join(prefix, "lib64")

    # Copy lib64 to lib if it exists.  libiberty installs only flat
    # library files (.a, .la) into lib(64) — no subdirectories — so
    # shutil.copy2 is sufficient here.
    if os.path.isdir(lib64_dir):
        for item in os.listdir(lib64_dir):
            shutil.copy2(os.path.join(lib64_dir, item), lib_dir)
        shutil.rmtree(lib64_dir)

    # Remove libtool descriptors.
    for la_file in glob.glob(os.path.join(lib_dir, "*.la")):
        os.remove(la_file)

    # Create a versioned libiberty shared library from the PIC-compiled static
    # archive. libiberty is LGPL-licensed so it must be dynamically linked (not
    # statically) in this MIT-licensed project.
    #
    # We follow the standard Linux SO versioning convention:
    #   libiberty.so.0  - the actual versioned shared library (SONAME libiberty.so.0)
    #   libiberty.so    - unversioned symlink used at link time
    #
    # The SONAME libiberty.so.0 is embedded in the binary so that consumers
    # encode the SONAME (not the build-tree path) in their DT_NEEDED entry,
    # allowing the runtime linker to resolve it via RPATH.
    libiberty_a = os.path.join(lib_dir, "libiberty.a")
    if os.path.isfile(libiberty_a):
        print("Creating libiberty.so.0 from libiberty.a...")
        so_versioned = os.path.join(lib_dir, "libiberty.so.0")
        so_link = os.path.join(lib_dir, "libiberty.so")
        subprocess.run(
            [
                cc,
                "-shared",
                "-Wl,-soname,libiberty.so.0",
                "-Wl,--whole-archive",
                libiberty_a,
                "-Wl,--no-whole-archive",
                "-o",
                so_versioned,
            ],
            check=True,
        )
        if not os.path.isfile(so_versioned) or os.path.getsize(so_versioned) == 0:
            raise RuntimeError(
                f"Expected {so_versioned} to be created but it is missing or empty"
            )
        subprocess.run(
            [patchelf, "--set-rpath", "$ORIGIN", so_versioned],
            check=True,
        )
        # Create the unversioned symlink for link-time use.
        if os.path.lexists(so_link):
            os.remove(so_link)
        os.symlink("libiberty.so.0", so_link)
        print(f"Created {so_versioned}")
        print(f"Created symlink {so_link} -> libiberty.so.0")
        # Remove the static archive so downstream consumers cannot
        # accidentally link libiberty statically instead of via the .so.
        os.remove(libiberty_a)
        print(f"Removed {libiberty_a}")

    # Remove internal build artifacts we do not need.
    for subdir in ["bfd-plugins", "gprofng"]:
        path = os.path.join(lib_dir, subdir)
        if os.path.isdir(path):
            shutil.rmtree(path)

    for topdir in ["bin", "share", "etc"]:
        path = os.path.join(prefix, topdir)
        if os.path.isdir(path):
            shutil.rmtree(path)

    # Remove the architecture-specific sysroot directory (e.g. x86_64-pc-linux-gnu).
    for entry in glob.glob(os.path.join(prefix, "*-*-*")):
        if os.path.isdir(entry):
            shutil.rmtree(entry)

    print("Done patching libiberty install.")


if __name__ == "__main__":
    main()
