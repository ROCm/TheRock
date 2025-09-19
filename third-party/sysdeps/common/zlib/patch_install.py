import os
from pathlib import Path
import platform
import subprocess
import sys

PREFIX = sys.argv[1]

if platform.system() == "Linux":
    source = str(Path(PREFIX) / "lib" / "librocm_sysdeps_z.so")
    destination = str(Path(PREFIX) / "lib" / "libz.so")
    subprocess.run(["mv", source, destination], check=True)
    # We don't want the static lib on Linux.
    (Path(PREFIX) / "lib" / "librocm_sysdeps_z.a").unlink()
elif platform.system() == "Windows":
    # We don't want the libz.dll on Windows.
    (Path(PREFIX) / "bin" / "zlib.dll").unlink()
    (Path(PREFIX) / "lib" / "zlib.lib").unlink()
