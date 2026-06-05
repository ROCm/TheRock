import os
import platform
import subprocess
import sys
import logging
from pathlib import Path


def prepend_env_path(env: dict, var_name: str, new_path: str):
    """
    Prepend a new path to an environment variable that contains a list of paths (e.g., PATH, LD_LIBRARY_PATH).
    """
    existing = env.get(var_name)
    if existing:
        env[var_name] = f"{new_path}{os.pathsep}{existing}"
    else:
        env[var_name] = new_path


def build_rocm_loader_env(artifacts_path: Path) -> dict:
    """Return a copy of os.environ with the ROCm shared-library loader path
    prepended to the platform-appropriate variable.

    Linux prepends ``<artifacts_path>/lib`` to ``LD_LIBRARY_PATH``.
    Windows prepends ``<artifacts_path>/bin`` to ``PATH`` (ROCm DLLs live in
    ``bin/`` on Windows).
    """
    env = os.environ.copy()
    if platform.system() == "Windows":
        prepend_env_path(env, "PATH", str(artifacts_path / "bin"))
    else:
        prepend_env_path(env, "LD_LIBRARY_PATH", str(artifacts_path / "lib"))
    return env


def add_windows_dll_search_dirs(
    env: dict, dll_dirs: list[Path], shim_dir: Path
) -> None:
    """Register ROCm DLL directories for a subprocess Python interpreter.

    CPython >= 3.8 loads extension modules with LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
    which excludes PATH. A .pyd that links ROCm runtime DLLs therefore cannot
    resolve them from a PATH entry (what build_rocm_loader_env prepends), and the
    import fails with a DLL-load error. A sitecustomize.py placed on PYTHONPATH is
    imported at interpreter startup, before any test code imports the extension,
    and registers the dirs via os.add_dll_directory (the supported replacement for
    PATH-based search).

    Windows-only. ``shim_dir`` is a writable scratch directory for the shim.
    """
    present = [d for d in dll_dirs if d.is_dir()]
    shim_dir.mkdir(parents=True, exist_ok=True)
    lines = ["import os"]
    lines += [f"os.add_dll_directory({str(d)!r})" for d in present]
    (shim_dir / "sitecustomize.py").write_text("\n".join(lines) + "\n")

    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{shim_dir}{os.pathsep}{existing}" if existing else str(shim_dir)
    )


def get_gpu_architecture_portable(therock_build_dir):
    """
    Executes rocm_agent_enumerator for Linux and offload-arch for Windows and returns last line of the output.

    Returns:
        str: The gfx architecture of the running system, or None if not available.
    """
    therock_build_dir = str(therock_build_dir)
    file_ending = ".exe" if platform.system() == "Windows" else ""
    try:
        executable = therock_build_dir + f"/lib/llvm/bin/offload-arch{file_ending}"
        result = subprocess.run(
            [executable], capture_output=True, text=True, check=True
        )
        lines = result.stdout.strip().split("\n")
        logging.info(f"DEBUG:{lines}")
        return lines[-1]

    except subprocess.CalledProcessError as e:
        print(f"Error executing offload-arch: {e}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        return None
    except FileNotFoundError:
        print("Error: offload-arch command not found", file=sys.stderr)
        return None
