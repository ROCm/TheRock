import os
from pathlib import Path
import platform
import subprocess
import sys

PREFIX = sys.argv[1]
PATCHELF = os.getenv("PATCHELF")
THEROCK_SOURCE_DIR = os.getenv("THEROCK_SOURCE_DIR")

if not THEROCK_SOURCE_DIR:
    raise ValueError("Exepcted THEROCK_SOURCE_DIR env var")

if platform.system() == "Linux":
    if not PATCHELF:
        raise ValueError("Exepcted PATCHELF env var")
    # Patch libz.so
    subprocess.check_call(
        [
            sys.executable,
            str(Path(THEROCK_SOURCE_DIR) / "build_tools" / "patch_linux_so.py"),
            "--patchelf",
            PATCHELF,
            "--add-prefix",
            "rocm_sysdeps_",
            str(Path(PREFIX) / "lib" / "libz.so"),
        ]
    )
    # We don't want the static lib on Linux.
    (Path(PREFIX) / "lib" / "libz.a").unlink()
elif platform.system() == "Windows":
    # We don't want the libz.dll on Windows.
    (Path(PREFIX) / "bin" / "zlib.dll").unlink()
