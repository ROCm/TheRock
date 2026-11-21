#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run examples/cpp-sdk-user against an installed ROCm prefix.

The install includes ``amdrocm`` and ``amdrocm-hpc-sdk``, so every
``THEROCK_ENABLE_*`` option stays at the project's default of ON. A missing
config fails inside ``find_package``. This launcher does not scan the prefix
or pass those flags.

The host file ``hip-host-test.cpp`` includes the installed HIP headers. Those
headers need a compiler newer than RHEL 8's GCC 8, so the host compile uses
the prefix Clang unless ``--cxx-compiler`` is set.
"""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _repo_root() -> Path | None:
    """Return the checkout root when this file lives under ``build_tools/``.

    ``build_tools/packaging/linux/<this file>`` is four levels below the
    checkout. A copy placed somewhere shallower, such as ``/tmp``, has no
    checkout root; callers then pass ``--source-dir`` and ``--driver``.

    Returns:
        The checkout root, or None when this file is not inside one.
    """
    try:
        return Path(__file__).resolve().parents[3]
    except IndexError:
        return None


REPO_ROOT = _repo_root()


def _cmake_path(path: Path | str) -> str:
    """Render ``path`` with the forward slashes cmake accepts.

    ``pathlib.Path`` uses backslashes when this file is imported on Windows.
    The packaging unit test compares these command strings on that runner.
    """
    return Path(path).as_posix()


def host_cxx_compiler(prefix: Path, override: str | None = None) -> str | None:
    """Choose the compiler that builds ``hip-host-test``.

    The installed HIP headers use ``__builtin_bit_cast``. RHEL 8's system
    GCC 8 does not provide that builtin, so the default host compiler is
    the Clang shipped in the prefix. ``--cxx-compiler`` replaces that
    default. When the prefix has no Clang, cmake chooses the compiler.

    Args:
        prefix: Installed ROCm prefix.
        override: Compiler from ``--cxx-compiler``, when the caller set one.

    Returns:
        A compiler path, or None to leave the choice to cmake.
    """
    if override:
        return override
    clangxx = prefix / "lib" / "llvm" / "bin" / "clang++"
    if clangxx.is_file():
        return _cmake_path(clangxx)
    return None


def build_command(
    *,
    cmake: str,
    source_dir: Path,
    binary_dir: Path,
    driver: Path,
    prefix: Path,
    generator: str,
    cxx_compiler: str | None = None,
) -> list[str]:
    """Build the ``cmake -E env`` invocation of the existing driver.

    ``CMAKE_PREFIX_PATH`` is an environment variable, matching
    ``examples/CMakeLists.txt``. ``THEROCK_ENABLE_*`` stays at the project
    default, so this command does not pass those flags.

    Args:
        cmake: cmake executable.
        source_dir: ``examples/cpp-sdk-user``.
        binary_dir: Build directory outside the install prefix.
        driver: ``examples/clean_configure_test_project.cmake``.
        prefix: Installed ROCm prefix.
        generator: CMake generator. The install job uses ``Unix Makefiles``.
        cxx_compiler: Host C++ compiler. None leaves the choice to cmake.

    Returns:
        The argv list, with no shell quoting.
    """
    child = [
        cmake,
        f"-DSOURCE_DIR={_cmake_path(source_dir)}",
        f"-DBINARY_DIR={_cmake_path(binary_dir)}",
        f"-DGENERATOR={generator}",
    ]
    if cxx_compiler:
        child.append(f"-DCMAKE_CXX_COMPILER={_cmake_path(cxx_compiler)}")
    child.extend(["-P", _cmake_path(driver)])
    return [
        cmake,
        "-E",
        "env",
        f"CMAKE_PREFIX_PATH={_cmake_path(prefix)}",
        "--",
        *child,
    ]


def _inside_prefix(path: Path, prefix: Path) -> bool:
    """Return True when ``path`` is ``prefix`` or a directory inside it.

    Args:
        path: Candidate build directory. It does not need to exist yet.
        prefix: Installed ROCm prefix.

    Returns:
        True when resolving ``path`` stays under ``prefix``.
    """
    return path.resolve().is_relative_to(prefix.resolve())


def validate_layout(
    prefix: Path,
    binary_dir: Path,
    source_dir: Path | None,
    driver: Path | None,
) -> None:
    """Fail closed before cmake if the prefix or project paths are unusable.

    Args:
        prefix: Installed ROCm prefix. Must be an existing directory.
        binary_dir: Build directory. Must not be inside ``prefix``.
        source_dir: ``examples/cpp-sdk-user``, containing ``CMakeLists.txt``.
        driver: ``clean_configure_test_project.cmake``.

    Raises:
        ValueError: A path is missing, or the build directory is inside the
            install prefix.
    """
    if not prefix.is_dir():
        raise ValueError(f"Install prefix does not exist: {prefix}")
    if source_dir is None or not (source_dir / "CMakeLists.txt").is_file():
        raise ValueError(f"cpp-sdk-user project not found: {source_dir}")
    if driver is None or not driver.is_file():
        raise ValueError(f"CMake driver not found: {driver}")
    if _inside_prefix(binary_dir, prefix):
        raise ValueError(
            f"Build directory must not be inside the install prefix: {binary_dir}"
        )


def resolve_cmake(cmake: str) -> str:
    """Return a usable cmake path.

    Args:
        cmake: Absolute path, or a name looked up on ``PATH``.

    Returns:
        The executable path.

    Raises:
        ValueError: No executable exists at that path and ``PATH`` has no match.
    """
    candidate = Path(cmake)
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    found = shutil.which(cmake)
    if not found:
        raise ValueError(f"cmake executable not found: {cmake}")
    return found


def _nonempty_path(value: str) -> Path:
    """Convert a CLI path argument, rejecting a blank value."""
    stripped = value.strip()
    if not stripped:
        raise argparse.ArgumentTypeError("path must not be empty")
    return Path(stripped)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse launcher arguments.

    Args:
        argv: Argument vector without the program name. None reads ``sys.argv``.

    Returns:
        The parsed namespace. ``--prefix`` is required.
    """
    parser = argparse.ArgumentParser(
        description="Run examples/cpp-sdk-user against an installed ROCm prefix."
    )
    parser.add_argument(
        "--prefix",
        required=True,
        type=_nonempty_path,
        help="Installed ROCm prefix.",
    )
    parser.add_argument(
        "--binary-dir",
        type=Path,
        help="Build directory. Must be outside --prefix. Defaults to a temp dir.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None if REPO_ROOT is None else REPO_ROOT / "examples" / "cpp-sdk-user",
        help="examples/cpp-sdk-user source directory.",
    )
    parser.add_argument(
        "--driver",
        type=Path,
        default=(
            None
            if REPO_ROOT is None
            else REPO_ROOT / "examples" / "clean_configure_test_project.cmake"
        ),
        help="clean_configure_test_project.cmake driver.",
    )
    parser.add_argument("--generator", default="Unix Makefiles")
    parser.add_argument("--cmake", default="cmake", help="cmake executable.")
    parser.add_argument(
        "--cxx-compiler",
        help=(
            "Host C++ compiler for hip-host-test. Omit to use the prefix "
            "Clang, which can compile the HIP headers on RHEL 8."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run cpp-sdk-user against the installed prefix.

    Args:
        argv: Argument vector without the program name. None reads ``sys.argv``.

    Returns:
        Zero when the existing CMake driver passes. One when the prefix or the
        project path is unusable, or when cmake cannot be started. The driver's
        own exit code when CMake runs and fails.
    """
    args = parse_args(argv)
    binary_dir = args.binary_dir
    if binary_dir is None:
        binary_dir = Path(tempfile.mkdtemp(prefix="cpp-sdk-user-"))
    source_dir = args.source_dir
    driver = args.driver
    try:
        validate_layout(args.prefix, binary_dir, source_dir, driver)
        if source_dir is None or driver is None:
            raise ValueError(f"cpp-sdk-user project not found: {source_dir}")
        cmake = resolve_cmake(args.cmake)
        cxx_compiler = host_cxx_compiler(args.prefix, args.cxx_compiler)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Installed cpp-sdk-user prefix: {args.prefix}", flush=True)
    if cxx_compiler:
        print(f"CMAKE_CXX_COMPILER={cxx_compiler}", flush=True)
    command = build_command(
        cmake=cmake,
        source_dir=source_dir,
        binary_dir=binary_dir,
        driver=driver,
        prefix=args.prefix,
        generator=args.generator,
        cxx_compiler=cxx_compiler,
    )
    print("Running:", shlex.join(command), flush=True)
    try:
        completed = subprocess.run(command)
    except OSError as exc:
        print(f"error: could not run cmake: {exc}", file=sys.stderr)
        return 1
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
