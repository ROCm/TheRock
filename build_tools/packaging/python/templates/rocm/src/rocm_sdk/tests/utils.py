# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Test utilities."""

import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import sysconfig

is_windows = platform.system() == "Windows"
exe_suffix = ".exe" if is_windows else ""


def run_command(args: list[str | Path], cwd: Path | None = None, capture: bool = False):
    args = [str(arg) for arg in args]
    if cwd is None:
        cwd = Path.cwd()
    print(f"++ Exec [{cwd}]$ {shlex.join(args)}")
    sys.stdout.flush()
    if capture:
        return subprocess.check_output(args, cwd=str(cwd), stdin=subprocess.DEVNULL)
    else:
        subprocess.check_call(args, cwd=str(cwd), stdin=subprocess.DEVNULL)


def assert_is_physical_package(mod):
    """Asserts that the given module is a non namespace module on disk defined
    by an __init__.py file."""
    assert (
        mod.__file__ is not None
    ), f"The `{mod.__name__}` module does not exist as a physical directory (__file__ is None)"
    assert (
        Path(mod.__file__).name == "__init__.py"
    ), f"Expected `{mod.__name__}` to be a non-namespace package"


def get_module_shared_libraries(mod) -> list[Path]:
    path = Path(mod.__file__).parent
    if is_windows:
        so_paths = [p for p in path.glob("**/*.dll") if p.is_file()]
    else:
        so_paths = [p for p in path.glob("**/*.so.*") if p.is_file()] + [
            p for p in path.glob("**/*.so") if p.is_file()
        ]

    return so_paths


def subprocess_kwargs_without_host_rocm(script_name: str) -> dict:
    """Extra subprocess kwargs for console scripts affected by stale HIP SDK env.

    Self-hosted Windows runners often export HIP SDK paths that break hipcc when
    published rocm-sdk-core wheels predate llvm 0007 / an updated _cli trampoline.
    Only hipcc/hipconfig need a sanitized environment; other tools keep the
    inherited process environment.
    """
    if not is_windows or script_name not in ("hipcc", "hipconfig"):
        return {}
    env = os.environ.copy()
    env.pop("ROCM_PATH", None)
    env.pop("HIP_PATH", None)
    return {"env": env}


def find_console_script(script_name: str) -> Path | None:
    scripts_paths = [sysconfig.get_path("scripts")]
    if is_windows:
        scripts_paths.append(sysconfig.get_path("scripts", "nt_user"))
    else:
        scripts_paths.append(sysconfig.get_path("scripts", "posix_user"))
    for scripts_path in scripts_paths:
        script_path = (Path(scripts_path) / script_name).with_suffix(exe_suffix)
        if script_path.exists():
            return script_path
    return None
