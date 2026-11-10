# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Post-install fixups for the lttng-tools sysdep:
# - drop static and libtool files;
# - give liblttng-ctl its SONAME-named file and a liblttng-ctl.so dev symlink;
# - set RPATHs so the programs find the bundled libraries (liblttng-ctl, lttng-ust-ctl,
#   urcu, popt, libxml2) wherever the install tree is moved: bin/ -> ../lib, the consumer
#   daemon in lib/lttng/libexec/ -> ../.., the library -> its own directory;
# - make the pkg-config file relocatable.
#
# The session daemon has the consumer daemon's install path compiled in. When the tree
# is moved, start it with --consumerd64-path=<prefix>/lib/lttng/libexec/lttng-consumerd
# (or set LTTNG_CONSUMERD64_BIN), as the out-of-process collector does.

from pathlib import Path
import os
import platform
import shutil
import subprocess
import sys

repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo_root / "build_tools"))
from patch_linux_so import update_library_links, relativize_pc_file

prefix = sys.argv[1] if len(sys.argv) > 1 else None
if not prefix:
    sys.exit("Error: Expected install prefix argument")
patchelf_exe = os.environ.get("PATCHELF")
if not patchelf_exe:
    sys.exit("Error: PATCHELF not defined")


def set_rpath(path: Path, rpath: str) -> None:
    subprocess.run([patchelf_exe, "--set-rpath", rpath, str(path)], check=True)


if platform.system() == "Linux":
    prefix_dir = Path(prefix)
    lib_dir = prefix_dir / "lib"
    for file_path in list(lib_dir.rglob("*.a")) + list(lib_dir.rglob("*.la")):
        file_path.unlink(missing_ok=True)

    update_library_links(
        lib_dir / "librocm_sysdeps_lttng_ctl.so", "liblttng-ctl.so", patchelf_exe
    )
    set_rpath((lib_dir / "liblttng-ctl.so").resolve(), "$ORIGIN")

    for program in (prefix_dir / "bin").iterdir():
        if program.is_file() and not program.is_symlink():
            set_rpath(program, "$ORIGIN/../lib")
    for program in (lib_dir / "lttng" / "libexec").iterdir():
        if program.is_file() and not program.is_symlink():
            set_rpath(program, "$ORIGIN/../..")

    for pc_file in (lib_dir / "pkgconfig").glob("*.pc"):
        relativize_pc_file(pc_file)

    for unused in ("share/man", "share/doc"):
        shutil.rmtree(prefix_dir / unused, ignore_errors=True)
