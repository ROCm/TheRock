#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Configure a throwaway CMake project against an installed ROCm prefix.

Phase 1 of https://github.com/ROCm/TheRock/issues/9056. Call this after the
native DEB/RPM install in ``test_native_linux_packages_install.yml``. It does
not install packages and does not compile or run a HIP program.

The generated project calls ``find_package(hip CONFIG REQUIRED)`` and
``find_package(hipblaslt CONFIG REQUIRED)``. hipBLASLt's installed config
calls ``find_dependency(origami)`` and searches only beside its own CMake
directory, so a missing origami ``dev`` component fails here.

Exit codes:
    0: both ``find_package`` calls configured.
    1: the prefix is missing, ``cmake`` is missing, the temporary project
       would be written inside the prefix, or configure returned non-zero.
    2: ``argparse`` usage error (missing prefix argument).

This module has no skip path. The workflow ``if`` is what omits the step for
``install`` and ``simulate``. Unit tests live in
``build_tools/packaging/linux/tests/downstream_cmake_smoke_test.py`` and run
from ``.github/workflows/unit_tests.yml``.

Example::

    python3 build_tools/packaging/linux/downstream_cmake_smoke.py \\
        --install-prefix /opt/rocm/core
"""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

CMAKE_MINIMUM_VERSION = "3.21"
# Bounds a hung configure in CI. cmake is not a binutils tool.
CONFIGURE_TIMEOUT_SEC = 120
# find_package(hip CONFIG) still requires HIP_PLATFORM.
# https://github.com/ROCm/TheRock/issues/68
HIP_PLATFORM = "amd"
# hipblaslt's config finds origami. hipblas and rocblas are not required here.
SMOKE_PACKAGES = ("hip", "hipblaslt")

_CLI_EXAMPLES_EPILOG = """
examples:
  # Installed native prefix (workflow default is /opt/rocm/core)
  python3 build_tools/packaging/linux/downstream_cmake_smoke.py \\
      --install-prefix /opt/rocm/core

  # Prefix from the install-test environment
  INSTALL_PREFIX=/opt/rocm/core \\
      python3 build_tools/packaging/linux/downstream_cmake_smoke.py

  # Newer cmake than the distro package, for example on RHEL 8
  python3 build_tools/packaging/linux/downstream_cmake_smoke.py \\
      --install-prefix /opt/rocm/core \\
      --cmake-executable /tmp/test-venv/bin/cmake

  # Distro g++. Omit --cxx-compiler to let CMake search PATH.
  python3 build_tools/packaging/linux/downstream_cmake_smoke.py \\
      --install-prefix /opt/rocm/core \\
      --cxx-compiler g++

  # Clang, or the ROCm compiler installed under the prefix
  python3 build_tools/packaging/linux/downstream_cmake_smoke.py \\
      --install-prefix /opt/rocm/core \\
      --cxx-compiler clang++
  python3 build_tools/packaging/linux/downstream_cmake_smoke.py \\
      --install-prefix /opt/rocm/core \\
      --cxx-compiler /opt/rocm/core/bin/amdclang++
"""


@dataclass(frozen=True)
class ConfigureResult:
    """Stdout, stderr, and status from one ``cmake`` configure."""

    returncode: int
    stdout: str
    stderr: str


ConfigureRunner = Callable[[list[str]], ConfigureResult]


def render_cmake_lists() -> str:
    """Return the configure-only consumer project.

    ``LANGUAGES CXX`` is required. ``hip-config`` calls
    ``find_dependency(Threads)``, and CMake's ``FindThreads`` errors when
    neither C nor CXX is enabled. The step still does not add a HIP target
    or compile a program. ``message(STATUS)`` prints ``hip_VERSION`` and
    ``hipblaslt_VERSION`` on cmake stdout after each ``find_package``.

    Returns:
        ``CMakeLists.txt`` text.
    """
    lines = [
        f"cmake_minimum_required(VERSION {CMAKE_MINIMUM_VERSION})",
        "project(install_smoke_test LANGUAGES CXX)",
        "",
    ]
    for package_name in SMOKE_PACKAGES:
        lines.append(f"find_package({package_name} CONFIG REQUIRED)")
        lines.append(
            f'message(STATUS "{package_name}_VERSION=${{{package_name}_VERSION}}")'
        )
    lines.append("")
    return "\n".join(lines)


def build_configure_command(
    *,
    cmake_executable: str,
    source_dir: Path,
    build_dir: Path,
    install_prefix: Path,
    cxx_compiler: str | None = None,
) -> list[str]:
    """Build the list-form ``cmake`` configure command.

    Args:
        cmake_executable: ``cmake`` name or absolute path.
        source_dir: Directory that contains ``CMakeLists.txt``.
        build_dir: Out-of-source build directory.
        install_prefix: Resolved ROCm prefix passed as ``CMAKE_PREFIX_PATH``.
        cxx_compiler: Resolved compiler for ``CMAKE_CXX_COMPILER``. Unset
            leaves the compiler choice to CMake.

    Returns:
        argv for one configure invocation. No generator flag, so CMake uses
        Unix Makefiles and ``make`` must be on PATH when CXX is enabled. The
        step does not build, so Ninja is not required.
    """
    command = [
        cmake_executable,
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        f"-DCMAKE_PREFIX_PATH={install_prefix}",
        f"-DHIP_PLATFORM={HIP_PLATFORM}",
    ]
    if cxx_compiler:
        command.append(f"-DCMAKE_CXX_COMPILER={cxx_compiler}")
    return command


def path_is_inside(child: Path, parent: Path) -> bool:
    """Return whether ``child`` is ``parent`` or a path under it.

    Both paths are resolved, so a symlink prefix and its real directory
    compare as the same tree.
    """
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _resolve_executable(executable: str) -> str | None:
    """Return an executable path, or None when it cannot be run.

    An absolute path is accepted only when that file exists and is executable.
    Any other value is looked up on ``PATH``.
    """
    candidate = Path(executable)
    if candidate.is_absolute():
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
        return None
    return shutil.which(executable)


def _resolve_cmake(cmake_executable: str) -> str | None:
    """Return the ``cmake`` binary, or None when it cannot be run."""
    return _resolve_executable(cmake_executable)


def _resolve_cxx_compiler(cxx_compiler: str) -> str | None:
    """Return the C++ compiler, or None when it cannot be run."""
    return _resolve_executable(cxx_compiler)


def _captured_text(value: str | bytes | None) -> str:
    """Normalize a captured subprocess stream to text."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _run_cmake(command: list[str]) -> ConfigureResult:
    """Run configure and capture both streams.

    A timeout or a missing binary is a failed configure (return code 1),
    not a skipped test. ``message(STATUS)`` lines arrive on stdout.
    """
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=CONFIGURE_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired as exc:
        return ConfigureResult(
            returncode=1,
            stdout=_captured_text(exc.stdout),
            stderr=_captured_text(exc.stderr)
            or f"cmake timed out after {CONFIGURE_TIMEOUT_SEC} seconds",
        )
    except FileNotFoundError as exc:
        return ConfigureResult(returncode=1, stdout="", stderr=str(exc))
    return ConfigureResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _print_failure(
    command: list[str], install_prefix: Path, result: ConfigureResult
) -> None:
    """Print the configure command, prefix, and captured CMake output."""
    print("[FAIL] Downstream find_package smoke test failed", file=sys.stderr)
    print(f"Prefix: {install_prefix}", file=sys.stderr)
    print(f"Command: {shlex.join(command)}", file=sys.stderr)
    if result.stdout:
        print(result.stdout, file=sys.stderr)
    if result.stderr:
        print(result.stderr, file=sys.stderr)


def _print_versions(stdout: str) -> None:
    """Print ``hip_VERSION`` and ``hipblaslt_VERSION`` status lines from stdout."""
    for line in stdout.splitlines():
        if any(f"{package_name}_VERSION=" in line for package_name in SMOKE_PACKAGES):
            print(line)


@contextmanager
def _project_root(work_dir: Path | None) -> Iterator[Path]:
    """Yield the directory that holds the generated CMake project.

    When ``work_dir`` is unset, this is a temporary directory and it is
    removed when the context exits. A caller-supplied directory is created
    if needed and left in place so tests can inspect it.
    """
    if work_dir is None:
        with tempfile.TemporaryDirectory(
            prefix="rocm-install-smoke-"
        ) as temporary_name:
            yield Path(temporary_name)
        return
    project_root = Path(work_dir)
    project_root.mkdir(parents=True, exist_ok=True)
    yield project_root


def _write_project(project_root: Path) -> tuple[Path, Path]:
    """Write ``src/CMakeLists.txt`` and return the source and build directories.

    Raises:
        RuntimeError: The lists file is missing or empty after the write.
    """
    source_dir = project_root / "src"
    build_dir = project_root / "build"
    source_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    cmake_lists = source_dir / "CMakeLists.txt"
    cmake_lists.write_text(render_cmake_lists(), encoding="utf-8")
    if not cmake_lists.is_file() or cmake_lists.stat().st_size == 0:
        raise RuntimeError(f"Failed to write smoke project: {cmake_lists}")
    return source_dir, build_dir


def run_smoke(
    install_prefix: Path,
    *,
    cmake_executable: str = "cmake",
    cxx_compiler: str | None = None,
    work_dir: Path | None = None,
    run_configure: ConfigureRunner | None = None,
) -> int:
    """Configure the smoke project against ``install_prefix``.

    Args:
        install_prefix: ROCm prefix. Symlinks are resolved before configure.
        cmake_executable: ``cmake`` name or path. Resolved with ``shutil.which``.
        cxx_compiler: Compiler name or path for ``CMAKE_CXX_COMPILER``. Unset
            lets CMake search ``PATH``. ``g++`` selects the distro C++ compiler
            and ``clang++`` selects Clang.
        work_dir: Project directory. Unset uses a temporary directory that is
            removed before this function returns.
        run_configure: Test seam. Production callers leave this unset so
            ``_run_cmake`` runs.

    Returns:
        0 on success, 1 on a missing prefix, a missing ``cmake``, a missing
        requested compiler, a project directory inside the prefix, or a
        non-zero configure.
    """
    prefix = Path(install_prefix).resolve()
    if not prefix.is_dir():
        print(f"[FAIL] Install prefix is not a directory: {prefix}", file=sys.stderr)
        return 1

    resolved_cmake = _resolve_cmake(cmake_executable)
    if resolved_cmake is None:
        print(
            f"[FAIL] cmake executable not found: {cmake_executable}",
            file=sys.stderr,
        )
        return 1

    resolved_cxx = None
    if cxx_compiler:
        resolved_cxx = _resolve_cxx_compiler(cxx_compiler)
        if resolved_cxx is None:
            print(
                f"[FAIL] C++ compiler not found: {cxx_compiler}",
                file=sys.stderr,
            )
            return 1

    with _project_root(work_dir) as project_root:
        if path_is_inside(project_root, prefix):
            print(
                "[FAIL] Smoke project directory is inside the install prefix: "
                f"{project_root}",
                file=sys.stderr,
            )
            return 1
        source_dir, build_dir = _write_project(project_root)
        command = build_configure_command(
            cmake_executable=resolved_cmake,
            source_dir=source_dir,
            build_dir=build_dir,
            install_prefix=prefix,
            cxx_compiler=resolved_cxx,
        )
        configure = run_configure or _run_cmake
        result = configure(command)
        if result.returncode != 0:
            _print_failure(command, prefix, result)
            return 1
        print(f"[PASS] Downstream find_package smoke test passed for {prefix}")
        _print_versions(result.stdout)
        return 0


def _nonempty_path(value: str) -> Path:
    """Convert a CLI path argument, rejecting a blank value."""
    stripped = value.strip()
    if not stripped:
        raise argparse.ArgumentTypeError("path must not be empty")
    return Path(stripped)


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments. ``INSTALL_PREFIX`` supplies the default prefix.

    Args:
        argv: Argument list, typically ``sys.argv[1:]``.

    Returns:
        Parsed arguments. ``argparse`` exits with status 2 when the prefix
        was neither passed nor set in the environment.
    """
    env_prefix = os.environ.get("INSTALL_PREFIX", "").strip()
    parser = argparse.ArgumentParser(
        description=(
            "Configure a downstream find_package() smoke project against an "
            "installed ROCm prefix. Run this after packages are installed."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_CLI_EXAMPLES_EPILOG,
    )
    parser.add_argument(
        "--install-prefix",
        type=_nonempty_path,
        default=Path(env_prefix) if env_prefix else None,
        help=(
            "ROCm install prefix. Defaults to the INSTALL_PREFIX environment "
            "variable."
        ),
    )
    parser.add_argument(
        "--cmake-executable",
        default="cmake",
        help=(
            "cmake binary name or absolute path. Defaults to cmake on PATH. "
            "Use this when the distro cmake is older than the installed "
            "ROCm configs require."
        ),
    )
    parser.add_argument(
        "--cxx-compiler",
        default=None,
        help=(
            "C++ compiler passed as -DCMAKE_CXX_COMPILER. Omit to let CMake "
            "search PATH. Pass g++ for the distro C++ compiler or clang++ for "
            "Clang. An absolute path is accepted, for example "
            "/opt/rocm/core/bin/amdclang++. make must still be on PATH."
        ),
    )
    args = parser.parse_args(argv)
    if args.install_prefix is None:
        parser.error("Install prefix is required (--install-prefix or INSTALL_PREFIX)")
    return args


def main(argv: list[str]) -> int:
    """CLI entry point.

    Args:
        argv: Argument list, typically ``sys.argv[1:]``.

    Returns:
        Process exit code. Usage errors exit through ``argparse``.
    """
    args = parse_args(argv)
    return run_smoke(
        args.install_prefix,
        cmake_executable=args.cmake_executable,
        cxx_compiler=args.cxx_compiler,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
