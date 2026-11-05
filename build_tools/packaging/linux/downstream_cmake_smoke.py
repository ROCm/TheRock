#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run examples/cpp-sdk-user against an installed ROCm prefix.

The existing project defaults every THEROCK_ENABLE_* option to ON. A native
install can omit a component (hipTensor is published, but the amdrocm
metapackage does not depend on it; some GPU families never build it). This
launcher sets each option from the CMake configs under <prefix>/lib/cmake,
then calls examples/clean_configure_test_project.cmake.

hip and hipblaslt are required. A group that is only partly installed fails
instead of being switched off, so a present hipblaslt with a missing sibling
still reaches find_package. An entirely absent optional group is OFF.

The install job does not pass an architecture. Those runners have a GPU,
so CMake detects one. ``--hip-arch`` remains available for a GPU-less run:
an artifact group such as ``gfx94X-dcgpu`` becomes one concrete target
from ``cmake/therock_amdgpu_targets.cmake`` and is passed as
``CMAKE_HIP_ARCHITECTURES``.

The host file ``hip-host-test.cpp`` includes the installed HIP headers. Those
headers need a compiler newer than RHEL 8's GCC 8, so the host compile uses
the prefix Clang unless ``--cxx-compiler`` is set.
"""

import argparse
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

# build_tools/ is two levels above packaging/linux. The family map lives in
# _therock_utils and reads cmake/therock_amdgpu_targets.cmake.
_BUILD_TOOLS_DIR = Path(__file__).resolve().parents[2]
if str(_BUILD_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from _therock_utils.cmake_amdgpu_targets import (  # noqa: E402
    amdgpu_family_map,
    expand_families,
)


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

# Packages the install job must contain. Turning these flags OFF would hide
# the missing-origami failure this test exists to catch. Every other
# find_package name is read from examples/cpp-sdk-user/CMakeLists.txt.
REQUIRED_PACKAGES: tuple[str, ...] = ("hip", "hipblaslt")

# (THEROCK_ENABLE_* flag, find_package names that flag covers), in the order
# those flags appear in examples/cpp-sdk-user/CMakeLists.txt.
EnableGroups: TypeAlias = tuple[tuple[str, tuple[str, ...]], ...]

# Line-oriented matches against examples/cpp-sdk-user/CMakeLists.txt.
# Compiled once. Comments are stripped before matching.
_ENABLE_IF = re.compile(r"^if\s*\(\s*(THEROCK_ENABLE_[A-Za-z0-9_]+)\s*\)\s*$")
_ANY_IF = re.compile(r"^if\s*\(")
_ENDIF = re.compile(r"^endif\s*\(")
_FIND_PACKAGE = re.compile(r"^find_package\s*\(\s*([A-Za-z0-9_]+)\b")


@dataclass(frozen=True)
class EnableFlag:
    """One ``THEROCK_ENABLE_*`` value passed through to the CMake driver.

    Attributes:
        name: CMake option, for example ``THEROCK_ENABLE_HIPTENSOR``.
        value: ``ON`` when every package in the group is installed, else ``OFF``.
        missing: find_package names absent from the prefix. Empty when value is ON.
    """

    name: str
    value: str
    missing: tuple[str, ...] = ()


def cmake_config_filenames(package: str) -> tuple[str, str]:
    """Return the two filenames ``find_package(<package> CONFIG)`` searches for.

    Args:
        package: The ``find_package`` name, including its original case.

    Returns:
        ``<Name>Config.cmake`` and ``<lowercase>-config.cmake``.
    """
    return (f"{package}Config.cmake", f"{package.lower()}-config.cmake")


def config_filename_index(packages: set[str]) -> dict[str, str]:
    """Map CMake config filenames to find_package names.

    Built once per prefix scan so each file under ``lib/cmake`` is a single
    dictionary lookup.

    Args:
        packages: find_package names taken from cpp-sdk-user.

    Returns:
        Filename to package name. ``rocwmma-config.cmake`` maps to ``rocWMMA``.
    """
    index: dict[str, str] = {}
    for package in packages:
        for filename in cmake_config_filenames(package):
            index.setdefault(filename, package)
    return index


def parse_enable_groups(cmake_text: str) -> EnableGroups:
    """Read ``if(THEROCK_ENABLE_*)`` / ``find_package`` groups from cpp-sdk-user.

    Nested conditions such as ``if(NOT WIN32)`` stay inside the enclosing flag.
    This Linux install job therefore keeps ``hipsparselt`` under
    ``THEROCK_ENABLE_SPARSE``. A ``find_package`` outside those blocks, or an
    unbalanced ``if`` / ``endif``, is an error.

    Args:
        cmake_text: Contents of ``examples/cpp-sdk-user/CMakeLists.txt``.

    Returns:
        Flags in source order, each with the find_package names inside its block.

    Raises:
        ValueError: The file has a find_package outside ``THEROCK_ENABLE_*``,
            or its ``if`` / ``endif`` pairs do not balance.
    """
    groups: dict[str, list[str]] = {}
    order: list[str] = []
    stack: list[str | None] = []
    for line_number, raw_line in enumerate(cmake_text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        enable_if = _ENABLE_IF.match(line)
        if enable_if:
            flag_name = enable_if.group(1)
            stack.append(flag_name)
            if flag_name not in groups:
                groups[flag_name] = []
                order.append(flag_name)
            continue
        if _ANY_IF.match(line):
            stack.append(None)
            continue
        if _ENDIF.match(line):
            if not stack:
                raise ValueError(
                    f"cpp-sdk-user CMakeLists has endif() without if() at line {line_number}"
                )
            stack.pop()
            continue
        found = _FIND_PACKAGE.match(line)
        if not found:
            continue
        flag_name = next((name for name in reversed(stack) if name), None)
        if flag_name is None:
            raise ValueError(
                "cpp-sdk-user find_package is outside THEROCK_ENABLE_* "
                f"at line {line_number}: {found.group(1)}"
            )
        package = found.group(1)
        if package not in groups[flag_name]:
            groups[flag_name].append(package)
    if stack:
        raise ValueError("cpp-sdk-user CMakeLists has an unclosed if()")
    return tuple((name, tuple(groups[name])) for name in order)


def package_name_from_config(path: Path, packages: set[str]) -> str | None:
    """Return the find_package name for ``path`` when it is one of ``packages``.

    Only the filename is compared. ``rocwmma-config.cmake`` satisfies
    ``find_package(rocWMMA)``. ``hip-lang-config.cmake`` does not satisfy ``hip``.

    Args:
        path: Config file. The directory is ignored.
        packages: find_package names taken from cpp-sdk-user.

    Returns:
        The matching package name, or None.
    """
    return config_filename_index(packages).get(path.name)


def installed_cmake_packages(prefix: Path, groups: EnableGroups) -> set[str]:
    """Collect cpp-sdk-user find_package names present under ``<prefix>/lib/cmake``.

    Args:
        prefix: Installed ROCm prefix.
        groups: Flags and package names from :func:`parse_enable_groups`.

    Returns:
        Package names whose ``<Name>Config.cmake`` or ``<lower>-config.cmake``
        exists. An empty set when ``lib/cmake`` is absent.
    """
    known = {package for _flag_name, group in groups for package in group}
    index = config_filename_index(known)
    cmake_dir = prefix / "lib" / "cmake"
    if not cmake_dir.is_dir():
        return set()
    names: set[str] = set()
    for path in cmake_dir.rglob("*"):
        if not path.is_file():
            continue
        package = index.get(path.name)
        if package:
            names.add(package)
    return names


def enable_flags(packages: set[str], groups: EnableGroups) -> list[EnableFlag]:
    """Map installed config names to THEROCK_ENABLE_* ON/OFF values.

    Args:
        packages: find_package names found under the install prefix.
        groups: Flags and package names from :func:`parse_enable_groups`.

    Returns:
        One flag per group. A group with no installed package is OFF.

    Raises:
        ValueError: ``hip`` or ``hipblaslt`` is missing, or a group is only
            partly installed.
    """
    missing_required = sorted(
        name for name in REQUIRED_PACKAGES if name not in packages
    )
    if missing_required:
        listed = ", ".join(missing_required)
        raise ValueError(
            f"Installed prefix is missing required CMake packages: {listed}"
        )

    flags: list[EnableFlag] = []
    for flag_name, group in groups:
        missing = tuple(sorted(name for name in group if name not in packages))
        found = tuple(sorted(name for name in group if name in packages))
        if missing and found:
            raise ValueError(
                f"{flag_name} is only partly installed: "
                f"found {', '.join(found)}; missing {', '.join(missing)}"
            )
        value = "OFF" if missing else "ON"
        flags.append(EnableFlag(flag_name, value, missing))
    return flags


def first_hip_architecture(spec: str) -> str:
    """Return one concrete gfx target for ``CMAKE_HIP_ARCHITECTURES``.

    ``examples/CMakeLists.txt`` pins the first target of the build so HIP
    language setup does not detect a GPU. The install job names a family
    (``gfx94X-dcgpu``) or a concrete target (``gfx942``). Both are keys in
    ``cmake/therock_amdgpu_targets.cmake``. A comma or semicolon list uses
    only the first entry. The workflow's lowered ``GFX_ARCH`` value
    (``gfx94x``) is not a family name and is rejected.

    Args:
        spec: Artifact group or concrete target from the install job.

    Returns:
        One gfx target, such as ``gfx942``.

    Raises:
        ValueError: ``spec`` is empty or names no family in the CMake map.
    """
    token = spec.replace(";", ",").split(",")[0].strip()
    if not token:
        raise ValueError("HIP architecture spec is empty")
    targets = expand_families([token], amdgpu_family_map())
    return targets[0]


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


def _cmake_path(path: Path | str) -> str:
    """Render ``path`` with the forward slashes cmake accepts.

    ``pathlib.Path`` uses backslashes when this file is imported on Windows.
    The packaging unit test compares these command strings on that runner.
    """
    return Path(path).as_posix()


def build_command(
    *,
    cmake: str,
    source_dir: Path,
    binary_dir: Path,
    driver: Path,
    prefix: Path,
    generator: str,
    flags: list[EnableFlag],
    cxx_compiler: str | None = None,
    hip_arch: str | None = None,
) -> list[str]:
    """Build the ``cmake -E env`` invocation of the existing driver.

    ``CMAKE_PREFIX_PATH`` is an environment variable, matching
    ``examples/CMakeLists.txt``. The driver forwards each ``-D`` flag,
    including ``CMAKE_HIP_ARCHITECTURES`` when ``hip_arch`` is set. Every
    path is written with forward slashes so the same command matches on
    the Windows unit-test runner.

    Args:
        cmake: cmake executable.
        source_dir: ``examples/cpp-sdk-user``.
        binary_dir: Build directory outside the install prefix.
        driver: ``examples/clean_configure_test_project.cmake``.
        prefix: Installed ROCm prefix.
        generator: CMake generator. The install job uses ``Unix Makefiles``.
        flags: ``THEROCK_ENABLE_*`` values for this prefix.
        cxx_compiler: Host C++ compiler. None leaves the choice to cmake.
        hip_arch: One concrete gfx target. Omit to leave detection to CMake.

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
    child.extend(f"-D{flag.name}={flag.value}" for flag in flags)
    if hip_arch:
        child.append(f"-DCMAKE_HIP_ARCHITECTURES={hip_arch}")
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
        ValueError: No such file exists and ``PATH`` has no match.
    """
    candidate = Path(cmake)
    if candidate.is_file():
        return str(candidate)
    found = shutil.which(cmake)
    if not found:
        raise ValueError(f"cmake executable not found: {cmake}")
    return found


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
        "--prefix", required=True, type=Path, help="Installed ROCm prefix."
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
    parser.add_argument(
        "--hip-arch",
        help=(
            "Artifact group or concrete gfx target. The first target of that "
            "family is passed as CMAKE_HIP_ARCHITECTURES. Example: gfx94X-dcgpu."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Set ``THEROCK_ENABLE_*`` from the prefix and run cpp-sdk-user.

    Args:
        argv: Argument vector without the program name. None reads ``sys.argv``.

    Returns:
        Zero when the existing CMake driver passes. One when the prefix, the
        project, or a required package is unusable. The driver's own exit code
        when CMake itself fails.
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
        project_cmake = source_dir / "CMakeLists.txt"
        groups = parse_enable_groups(project_cmake.read_text())
        if not groups:
            raise ValueError(
                f"No THEROCK_ENABLE_* find_package groups in {project_cmake}"
            )
        flags = enable_flags(installed_cmake_packages(args.prefix, groups), groups)
        hip_arch = first_hip_architecture(args.hip_arch) if args.hip_arch else None
        cxx_compiler = host_cxx_compiler(args.prefix, args.cxx_compiler)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Installed cpp-sdk-user prefix: {args.prefix}", flush=True)
    if cxx_compiler:
        print(f"CMAKE_CXX_COMPILER={cxx_compiler}", flush=True)
    if hip_arch:
        print(f"CMAKE_HIP_ARCHITECTURES={hip_arch}", flush=True)
    for flag in flags:
        if flag.value == "ON":
            print(f"{flag.name}=ON", flush=True)
        else:
            print(f"{flag.name}=OFF (missing {', '.join(flag.missing)})", flush=True)
    command = build_command(
        cmake=cmake,
        source_dir=source_dir,
        binary_dir=binary_dir,
        driver=driver,
        prefix=args.prefix,
        generator=args.generator,
        flags=flags,
        cxx_compiler=cxx_compiler,
        hip_arch=hip_arch,
    )
    print("Running:", shlex.join(command), flush=True)
    completed = subprocess.run(command)
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
