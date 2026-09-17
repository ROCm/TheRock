#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Install and validate a rocshmem4py wheel in TheRock CI."""

import argparse
from importlib.metadata import distributions
from pathlib import Path
import subprocess
import sys
import time
import venv

_BUILD_TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BUILD_TOOLS_DIR))
_VALIDATION_DEPS_DIR = _BUILD_TOOLS_DIR.parent / ".rocshmem4py-validation-deps"


def _packaging_types():
    # The gpu-smoke-test subcommand runs inside the intentionally minimal target
    # environment, so keep validation-only dependencies out of module imports.
    sys.path.insert(0, str(_VALIDATION_DEPS_DIR))

    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    from packaging.version import Version

    return Requirement, canonicalize_name, Version


def build_install_command(
    *,
    venv_dir: Path,
    rocm_version: str,
    index_url: str,
    find_links: str,
) -> list[str]:
    _, _, Version = _packaging_types()
    # Isolated mode prevents the selected checkout from shadowing pip.
    command = [
        sys.executable,
        "-I",
        "-m",
        "pip",
        "--python",
        str(venv_dir),
        "install",
        "--disable-pip-version-check",
        "--no-cache-dir",
        "--only-binary=:all:",
        "--pre",
    ]
    if index_url:
        command.extend(["--index-url", index_url])
    else:
        command.append("--no-index")
    if find_links:
        command.extend(["--find-links", find_links])
    command.extend(
        [
            "rocshmem4py",
            f"rocm-sdk-core=={Version(rocm_version)}",
        ]
    )
    return command


def run_with_retries(
    command: list[str], *, attempts: int = 13, wait_seconds: int = 15
) -> None:
    for attempt in range(1, attempts + 1):
        try:
            subprocess.run(command, check=True)
            return
        except subprocess.CalledProcessError:
            if attempt == attempts:
                raise
            print(
                f"Install failed (attempt {attempt}/{attempts}); "
                f"retrying in {wait_seconds} seconds",
                flush=True,
            )
            time.sleep(wait_seconds)


def install_validation_requirements() -> None:
    run_with_retries(
        [
            sys.executable,
            "-I",
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-cache-dir",
            "--only-binary=:all:",
            "--target",
            str(_VALIDATION_DEPS_DIR),
            "--upgrade",
            "packaging==25.0",
        ]
    )


def install_wheel(
    *,
    venv_dir: Path,
    rocm_version: str,
    index_url: str,
    find_links: str,
) -> None:
    install_validation_requirements()
    # Keep the target environment pip-less so only the wheel and its declared
    # runtime dependencies are present. The runner's pip installs into it.
    venv.EnvBuilder(with_pip=False, clear=True).create(venv_dir)
    command = build_install_command(
        venv_dir=venv_dir,
        rocm_version=rocm_version,
        index_url=index_url,
        find_links=find_links,
    )
    run_with_retries(command)


def find_site_packages(target_python: Path) -> Path:
    result = subprocess.run(
        [
            str(target_python),
            "-c",
            "import sysconfig; print(sysconfig.get_path('purelib'))",
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return Path(result.stdout.strip())


def validate_metadata(site_packages: Path, rocm_version: str) -> None:
    Requirement, canonicalize_name, Version = _packaging_types()
    from github_actions.determine_version import derive_version_suffix

    installed = {
        canonicalize_name(dist.metadata["Name"]): dist
        for dist in distributions(path=[str(site_packages)])
    }
    rocshmem4py_dist = installed["rocshmem4py"]
    core_dist = installed["rocm-sdk-core"]

    expected_rocm_version = Version(rocm_version)
    expected_suffix = derive_version_suffix(str(expected_rocm_version))
    expected_local = Version(f"0{expected_suffix}").local
    wheel_version = Version(rocshmem4py_dist.version)
    if wheel_version.local != expected_local:
        raise RuntimeError(
            f"rocshmem4py local version is {wheel_version.local!r}; "
            f"expected {expected_local!r}"
        )
    if Version(core_dist.version) != expected_rocm_version:
        raise RuntimeError(
            f"rocm-sdk-core version is {core_dist.version!r}; "
            f"expected {expected_rocm_version}"
        )

    core_requirements = [
        requirement
        for requirement in map(Requirement, rocshmem4py_dist.requires or [])
        if canonicalize_name(requirement.name) == "rocm-sdk-core"
    ]
    expected_specifier = f"=={expected_rocm_version}"
    if len(core_requirements) != 1 or str(core_requirements[0].specifier) != (
        expected_specifier
    ):
        raise RuntimeError(
            "rocshmem4py must have exactly one rocm-sdk-core requirement "
            f"with specifier {expected_specifier}; found {core_requirements}"
        )


def run_gpu_smoke_test() -> None:
    import _rocshmem4py  # noqa: F401
    import rocshmem4py

    if not rocshmem4py.__rocshmem_version__:
        raise RuntimeError("rocshmem4py does not report its linked rocSHMEM version")

    rocshmem4py.set_hip_device_from_env()
    unique_id = rocshmem4py.rocshmem_get_uniqueid()
    rocshmem4py.rocshmem_init_attr(0, 1, unique_id)
    try:
        buffer = rocshmem4py.rocshmem_create_buffer(4096)
        if buffer.ptr <= 0:
            raise RuntimeError("rocshmem4py returned an invalid GPU buffer")
        buffer.free()
    finally:
        rocshmem4py.hip_device_synchronize()
        rocshmem4py.rocshmem_barrier_all()
        rocshmem4py.rocshmem_finalize()


def validate_install(*, venv_dir: Path, rocm_version: str) -> None:
    target_python = venv_dir / "bin" / "python"
    validate_metadata(find_site_packages(target_python), rocm_version)
    subprocess.run(
        [str(target_python), str(Path(__file__).resolve()), "gpu-smoke-test"],
        check=True,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    install_parser = subparsers.add_parser("install")
    install_parser.add_argument("--venv-dir", type=Path, required=True)
    install_parser.add_argument("--rocm-version", required=True)
    install_parser.add_argument("--index-url", default="")
    install_parser.add_argument("--find-links", default="")

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--venv-dir", type=Path, required=True)
    validate_parser.add_argument("--rocm-version", required=True)

    subparsers.add_parser("gpu-smoke-test")

    args = parser.parse_args(argv)
    if args.command == "install":
        install_wheel(
            venv_dir=args.venv_dir,
            rocm_version=args.rocm_version,
            index_url=args.index_url,
            find_links=args.find_links,
        )
    elif args.command == "validate":
        validate_install(venv_dir=args.venv_dir, rocm_version=args.rocm_version)
    else:
        run_gpu_smoke_test()


if __name__ == "__main__":
    main()
