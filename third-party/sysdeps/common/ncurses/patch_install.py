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

# Required environment variables.
therock_source_dir = Path(get_env_or_exit("THEROCK_SOURCE_DIR"))
python_exe = get_env_or_exit("Python3_EXECUTABLE")
patchelf_exe = get_env_or_exit("PATCHELF")

if platform.system() == "Linux":
    # Specify the directory
    lib_dir = Path(install_prefix) / "lib"
    LIB_VERSION_SUFFIX = "6"

    # Remove static libs (*.a) and descriptors (*.la).
    for file_path in lib_dir.iterdir():
        if file_path.suffix in (".a", ".la"):
            file_path.unlink(missing_ok=True)

    # Now adjust the shared libraries according to our sysdeps rules.
    script_path = therock_source_dir / "build_tools" / "patch_linux_so.py"

    # Iterate over all shared libraries.
    for lib_path in lib_dir.glob("*.so"):
        # Patch the shared library and add our sysdeps prefix.
        patch_cmd = [
            python_exe,
            str(script_path),
            "--patchelf",
            patchelf_exe,
            "--add-prefix",
            "rocm_sysdeps_",
            str(lib_path),
        ]

        try:
            subprocess.run(patch_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: Failed to patch {lib_path.name} (Exit: {e.returncode})")
            sys.exit(e.returncode)

        # Create symbolic links for versioned library files, as the executables
        # and libraries expect that naming.
        # For now we need to add the .MAJOR_VERSION suffixes. This may need to be
        # updated in the future as the library gets updated.
        #
        # We ensure these are RELATIVE so the installation is relocatable.
        link_path = lib_path.with_name(f"{lib_path.name}.{LIB_VERSION_SUFFIX}")

        # Remove existing link/file if it exists.
        link_path.unlink(missing_ok=True)

        try:
            # Passing lib_path.name (the filename) creates a relative link.
            link_path.symlink_to(lib_path.name)
        except OSError as e:
            print(
                f"Error: Could not create symlink for {lib_path.name} (Exit: {e.returncode})"
            )
            sys.exit(1)

    # Fix .pc files to use relocatable paths.
    pkgconfig_dir = lib_dir / "pkgconfig"
    if pkgconfig_dir.exists():
        for pc_file in pkgconfig_dir.glob("*.pc"):
            relativize_pc_file(pc_file)

elif platform.system() == "Windows":
    # Do nothing for now.
    sys.exit(0)
