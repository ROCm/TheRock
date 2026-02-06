from pathlib import Path
import os
import platform
import shutil
import subprocess
import sys


def rename_pc_files(pc_file: Path) -> None:
    """Rename a .pc file by removing the 'rocm_sysdeps_' prefix if present.

    This function is used when ROCm‑patched pkg‑config files need to be
    restored to their original upstream names. If the filename begins with the
    prefix 'rocm_sysdeps_', the prefix is stripped and the file is renamed in place.

    Parameters
    ----------
    pc_file : Path Path object pointing to the .pc file to be examined and possibly renamed.

    Returns
    -------
    None The function performs an in‑place rename and does not return a value.
    """
    prefix = "rocm_sysdeps_"
    if pc_file.name.startswith(prefix):
        new_name = pc_file.name[len(prefix) :]
        new_path = pc_file.with_name(new_name)
        pc_file.rename(new_path)


def relativize_pc_file(pc_file: Path) -> None:
    """Make a .pc file relocatable by using pcfiledir-relative paths.

    Replaces the absolute prefix= line with a pcfiledir-relative path,
    then replaces all other occurrences of the absolute prefix with ${prefix}.
    Also remove the string "rocm_sysdeps_"
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
    # Remove "rocm_sysdeps_"
    content = content.replace(f"rocm_sysdeps_", "")
    pc_file.write_text(content)


def update_library_links(libfile: Path) -> None:
    """
    Normalize a shared library so that its real file is named exactly as its ELF SONAME,
    and ensure a canonical linker-visible symlink exists.

    This function is used when a library has been installed under a prefixed or
    non‑standard filename (e.g., librocm_sysdeps_ncursesw.so).
    It performs the following operations:
    - Extracts the library's SONAME using `patchelf --print-soname`.
    - Resolves the underlying real file (following symlinks).
    - Renames the real file to match its SONAME if it does not already.
    - Creates or updates a symlink named `linker_name` pointing to the SONAME file.
    - Removes or renames the original file or symlink as appropriate.

    Parameters ----------
    libfile : Path
    Path to the library file or symlink to normalize.
    Example: /prefix/lib/librocm_sysdeps_ncursesw.so

    """
    # Ensure file exists
    if not libfile.exists():
        raise FileNotFoundError(f"File '{libfile}' not found")

    dir_path = libfile.parent
    # Get SONAME
    try:
        lib_soname = subprocess.check_output(
            [patchelf_exe, "--print-soname", str(libfile)],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        print(f"Error: No SONAME found in '{libfile}'", flush=True)
        sys.exit(1)

    # Resolve real file path
    try:
        realname = libfile.resolve(strict=True)
    except FileNotFoundError:
        print(f"Error: resolve() failed for '{libfile}'", flush=True)
        sys.exit(1)

    linker_name = libfile.name.replace("librocm_sysdeps_", "lib")
    target_real = dir_path / lib_soname
    symlink_path = dir_path / linker_name

    if realname != target_real:
        # Move real file to $dir/$soname
        shutil.move(str(realname), str(target_real))

        # Create/update symlink
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()
        symlink_path.symlink_to(lib_soname)

        # Remove the original symlink or file
        if libfile.is_symlink() or libfile.exists():
            libfile.unlink()
    else:
        # Rename symlink in the same directory
        if symlink_path.exists():
            symlink_path.unlink()
        libfile.rename(symlink_path)


# Fetch an environment variable or exit if it is not found.
def get_env_or_exit(var_name):
    value = os.environ.get(var_name)
    if value is None:
        print(f"Error: {var_name} not defined")
        sys.exit(1)
    return value


# Validate the install prefix argument.
if len(sys.argv) < 2:
    print("Error: Expected install prefix argument")
    sys.exit(1)

# 1st argument is the installation prefix.
install_prefix = Path(sys.argv[1])

# Required environment variables.
patchelf_exe = get_env_or_exit("PATCHELF")

if platform.system() == "Linux":
    # Specify the directory
    lib_dir = Path(install_prefix) / "lib"

    # Remove static libs (*.a) and descriptors (*.la).
    for file_path in lib_dir.iterdir():
        if file_path.suffix in (".a", ".la"):
            file_path.unlink(missing_ok=True)

    # Update library linking
    for so_file in lib_dir.glob("librocm_sysdeps_*.so"):
        update_library_links(so_file)

    # Fix .pc files to use relocatable paths.
    pkgconfig_dir = lib_dir / "pkgconfig"
    if pkgconfig_dir.exists():
        for pc_file in pkgconfig_dir.glob("*.pc"):
            relativize_pc_file(pc_file)
            rename_pc_files(pc_file)

elif platform.system() == "Windows":
    # Do nothing for now.
    sys.exit(0)
