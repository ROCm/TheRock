from pathlib import Path
import platform
import shutil
import sys

PREFIX = sys.argv[1]

if platform.system() == "Linux":
    source = str(Path(PREFIX) / "lib" / "librocm_sysdeps_liblzma.so")
    destination = str(Path(PREFIX) / "lib" / "liblzma.so")
    shutil.move(source, destination)
    # We don't want the static lib on Linux - delete it if it is there
    static_lib = Path(PREFIX) / "lib" / "librocm_sysdeps_liblzma.a"
    if static_lib.exists():
        static_lib.unlink()
elif platform.system() == "Windows":
    # We don't want the .dll on Windows.
    (Path(PREFIX) / "bin" / "liblzma.dll").unlink()
    (Path(PREFIX) / "lib" / "liblzma.lib").unlink()
