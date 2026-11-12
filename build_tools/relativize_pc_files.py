# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Make the pkg-config files installed under a prefix relocatable.

Used by the sysdeps' patch_install scripts, on every host.
"""

from pathlib import Path

from patch_linux_so import relativize_pc_file


def relativize_pc_files(install_prefix: Path) -> list[Path]:
    """Relativize each .pc file under install_prefix/lib/pkgconfig.

    Rewrites each file so that it locates the prefix relative to the file
    itself (prefix=${pcfiledir}/../..), with the other occurrences of the
    build-time prefix as ${prefix}, instead of by the absolute path it was
    built with.  A prefix with no lib/pkgconfig is left alone.

    Returns the files rewritten, sorted.
    """
    pkgconfig_dir = Path(install_prefix) / "lib" / "pkgconfig"
    if not pkgconfig_dir.is_dir():
        return []
    pc_files = sorted(pkgconfig_dir.glob("*.pc"))
    for pc_file in pc_files:
        relativize_pc_file(pc_file)
    return pc_files
