# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import os
import platform
import shutil
import subprocess
import sys


def relativize_pc_file(pc_file: Path) -> None:
    """Make a .pc file relocatable by using pcfiledir-relative paths.

    Replaces the absolute prefix= line with a pcfiledir-relative path,
    then replaces all other occurrences of the absolute prefix with ${prefix}.
    Assumes the .pc file is located at $PREFIX/lib/pkgconfig/.
    """
    content = pc_file.read_text()

    # Find the original absolute prefix value.
    original_prefix = None
    for line in content.splitlines():
        if line.startswith("prefix="):
            original_prefix = line[len("prefix=") :]
            break

    if not original_prefix:
        return

    # Replace the prefix line with pcfiledir-relative path.
    # .pc files are in $PREFIX/lib/pkgconfig, so go up 2 levels.
    content = content.replace(f"prefix={original_prefix}", "prefix=${pcfiledir}/../..")
    # Replace all other occurrences of the absolute path with ${prefix}.
    # Use trailing / to avoid partial matches.
    content = content.replace(f"{original_prefix}/", "${prefix}/")
    pc_file.write_text(content)


def update_library_links(
    libfile: Path, linker_name: str, patchelf: str = "patchelf"
) -> None:
    """
    Normalize a shared library so that its real file is named exactly as its ELF SONAME,
    and ensure a canonical linker-visible symlink exists.

    This function is used when a library has been installed under a prefixed or
    non‑standard filename (e.g., librocm_sysdeps_nl_3.so).
    It performs the following operations:
    - Extracts the library's SONAME using `patchelf --print-soname`.
    - Resolves the underlying real file (following symlinks).
    - Renames the real file to match its SONAME if it does not already.
    - Creates or updates a symlink named `linker_name` pointing to the SONAME file.
    - Removes or renames the original file or symlink as appropriate.
    """
    # Ensure file exists
    if not libfile.exists():
        print(f"Warning: File '{libfile}' not found, skipping", flush=True)
        return

    dir_path = libfile.parent
    # Get SONAME
    try:
        lib_soname = subprocess.check_output(
            [patchelf, "--print-soname", str(libfile)],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        lib_soname = ""

    # Resolve real file path
    try:
        realname = libfile.resolve(strict=True)
    except FileNotFoundError:
        realname = None

    if not lib_soname or realname is None:
        if not lib_soname:
            print(f"Error: No SONAME found in '{libfile}'", flush=True)
        if realname is None:
            print(f"Error: resolve() failed for '{libfile}'", flush=True)
        return

    target_real = dir_path / lib_soname
    if realname != target_real:
        # Move real file to $dir/$soname
        shutil.move(str(realname), str(target_real))

        # Create/update symlink
        symlink_path = dir_path / linker_name
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()
        symlink_path.symlink_to(lib_soname)

        # Remove the original symlink or file
        if libfile.is_symlink() or libfile.exists():
            libfile.unlink()
    else:
        # Rename symlink in the same directory
        new_path = dir_path / linker_name
        if new_path.exists():
            new_path.unlink()
        libfile.rename(new_path)


# Fetch an environment variable or exit if it is not found.
def get_env_or_exit(var_name):
    value = os.environ.get(var_name)
    if value is None:
        print(f"Error: {var_name} not defined")
        sys.exit(1)
    return value


# Validate the install prefix argument.
prefix = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if not prefix:
    print("Error: Expected install prefix argument")
    sys.exit(1)

# 1st argument is the installation prefix.
install_prefix = sys.argv[1]

patchelf_exe = get_env_or_exit("PATCHELF")

if platform.system() == "Linux":
    # Specify the directory containing the libraries.
    lib_dir = Path(install_prefix) / "lib"
    pkgconfig_dir = lib_dir / "pkgconfig"

    # Remove static libs (*.a) and descriptors (*.la).
    for file_path in lib_dir.iterdir():
        if file_path.suffix in (".a", ".la"):
            file_path.unlink(missing_ok=True)

    # Update library linking for each libnl library
    libraries = [
        ("librocm_sysdeps_nl_3.so", "libnl-3.so"),
        ("librocm_sysdeps_nl_genl_3.so", "libnl-genl-3.so"),
        ("librocm_sysdeps_nl_route_3.so", "libnl-route-3.so"),
        ("librocm_sysdeps_nl_idiag_3.so", "libnl-idiag-3.so"),
        ("librocm_sysdeps_nl_nf_3.so", "libnl-nf-3.so"),
        ("librocm_sysdeps_nl_xfrm_3.so", "libnl-xfrm-3.so"),
        ("librocm_sysdeps_nl_cli_3.so", "libnl-cli-3.so"),
    ]

    for source_name, linker_name in libraries:
        source = lib_dir / source_name
        if source.exists():
            update_library_links(source, linker_name)

            # Clean up RUNPATH to only contain $ORIGIN
            target_lib = lib_dir / linker_name
            if target_lib.exists():
                try:
                    subprocess.run(
                        [patchelf_exe, "--set-rpath", "$ORIGIN", str(target_lib)],
                        check=True,
                    )
                except subprocess.CalledProcessError as e:
                    print(
                        f"Warning: Failed to set RPATH on {target_lib}: {e}", flush=True
                    )

    # Make .pc files relocatable
    pc_files = [
        "libnl-3.0.pc",
        "libnl-genl-3.0.pc",
        "libnl-route-3.0.pc",
        "libnl-idiag-3.0.pc",
        "libnl-nf-3.0.pc",
        "libnl-xfrm-3.0.pc",
        "libnl-cli-3.0.pc",
    ]

    for pc_name in pc_files:
        pc_file = pkgconfig_dir / pc_name
        if pc_file.exists():
            relativize_pc_file(pc_file)

    # Create header symlinks for test compatibility
    # Headers are installed in libnl3/netlink/, but tests expect netlink/
    include_dir = Path(install_prefix) / "include"
    libnl3_dir = include_dir / "libnl3"

    if libnl3_dir.exists():
        # Create symlink from include/netlink to include/libnl3/netlink
        netlink_symlink = include_dir / "netlink"
        if netlink_symlink.exists() or netlink_symlink.is_symlink():
            netlink_symlink.unlink()

        libnl3_netlink = libnl3_dir / "netlink"
        if libnl3_netlink.exists():
            netlink_symlink.symlink_to("libnl3/netlink", target_is_directory=True)
