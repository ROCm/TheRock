#!/usr/bin/env python3
"""Post-install patching for libpciaccess."""
import re, subprocess, sys
from pathlib import Path


def relativize_pc_file(pc_file: Path) -> None:
    if not pc_file.exists():
        return
    content = pc_file.read_text()
    content = re.sub(
        r"^prefix=.*$", r"prefix=${pcfiledir}/../..", content, flags=re.MULTILINE
    )
    pc_file.write_text(content)
    print(f"  Made relocatable: {pc_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: patch_install.py <install_prefix>")
        sys.exit(1)
    install_prefix = Path(sys.argv[1])
    patchelf = subprocess.os.environ.get("PATCHELF")
    if not patchelf:
        print("ERROR: PATCHELF environment variable not set")
        sys.exit(1)
    lib_dir = install_prefix / "lib"
    print("Patching libpciaccess installation...")
    libpciaccess = lib_dir / "libpciaccess.so.0"
    if libpciaccess.exists():
        subprocess.run(
            [
                patchelf,
                "--set-soname",
                "librocm_sysdeps_pciaccess.so.0",
                str(libpciaccess),
            ],
            check=True,
        )
        print(f"  Patched SONAME: {libpciaccess}")
        new_name = lib_dir / "librocm_sysdeps_pciaccess.so.0"
        libpciaccess.rename(new_name)
        print(f"  Renamed: {libpciaccess} -> {new_name}")
        symlink = lib_dir / "libpciaccess.so"
        if symlink.is_symlink():
            symlink.unlink()
            symlink.symlink_to(new_name.name)
            print(f"  Updated symlink: {symlink} -> {new_name.name}")
    pkgconfig_dir = lib_dir / "pkgconfig"
    if pkgconfig_dir.exists():
        for pc_file in pkgconfig_dir.glob("*.pc"):
            relativize_pc_file(pc_file)
    print("Patching complete.")


if __name__ == "__main__":
    main()
