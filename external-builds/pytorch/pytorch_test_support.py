# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Support steps that upstream test.sh runs around test/run_test.py."""

import os
import shutil
import site
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


PIN_DIRECTORIES = (
    Path(".github/ci_commit_pins"),
    Path(".ci/docker/ci_commit_pins"),
)

PINNED_PACKAGE_REPOSITORIES = {
    "spmd_types": "https://github.com/meta-pytorch/spmd_types.git",
    "torchcomms": "https://github.com/meta-pytorch/torchcomms.git",
}


@dataclass(frozen=True)
class CustomTest:
    source_directory: str
    python_test: str
    export_script: str
    export_argument: str
    executable: str


CUSTOM_TESTS = (
    CustomTest(
        source_directory="custom_operator",
        python_test="test_custom_ops.py",
        export_script="model.py",
        export_argument="--export-script-module-to=model.pt",
        executable="test_custom_ops",
    ),
    CustomTest(
        source_directory="custom_backend",
        python_test="test_custom_backend.py",
        export_script="backend.py",
        export_argument="--export-module-to=model.pt",
        executable="test_custom_backend",
    ),
)


def read_pinned_commit(pytorch_dir: Path, package: str) -> str:
    """Read a dependency pin across supported PyTorch branch layouts."""
    for relative_directory in PIN_DIRECTORIES:
        pin_path = pytorch_dir / relative_directory / f"{package}.txt"
        if pin_path.is_file():
            return pin_path.read_text(encoding="utf-8").strip()
    locations = ", ".join(os.fspath(path) for path in PIN_DIRECTORIES)
    raise FileNotFoundError(f"No {package} pin found under {locations}")


def install_distributed_test_dependencies(pytorch_dir: Path) -> None:
    """Install the pinned packages required by distributed tests."""
    for package, repository in PINNED_PACKAGE_REPOSITORIES.items():
        try:
            commit = read_pinned_commit(pytorch_dir, package)
        except FileNotFoundError:
            print(f"Skipping {package}: this PyTorch ref does not pin it")
            continue
        env = os.environ.copy()
        if package == "torchcomms":
            env.update(
                {
                    "USE_GLOO": "1",
                    "USE_NCCL": "0",
                    "USE_NCCLX": "0",
                    "USE_TRANSPORT": "0",
                }
            )

        with tempfile.TemporaryDirectory(prefix=f"{package}-wheel-") as wheel_dir:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    "--no-build-isolation",
                    "--no-deps",
                    "--wheel-dir",
                    wheel_dir,
                    f"git+{repository}@{commit}",
                ],
                env=env,
            )
            wheels = list(Path(wheel_dir).glob("*.whl"))
            if len(wheels) != 1:
                raise RuntimeError(f"Expected one {package} wheel, found {len(wheels)}")
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-index",
                    "--no-deps",
                    os.fspath(wheels[0]),
                ]
            )


def should_run_custom_tests(
    *,
    test_config: str,
    shard: int,
    num_shards: int,
    included_tests: list[str] | None,
) -> bool:
    """Match test.sh's custom-test placement without expanding core runs."""
    if test_config != "default" or included_tests:
        return False
    if num_shards <= 1:
        return True
    return shard == 2


def cmake_configure_command(
    pytorch_dir: Path, source_dir: Path, build_dir: Path
) -> list[str]:
    """Build the CMake command used for PyTorch's custom test binaries."""
    site_packages = site.getsitepackages()
    prefix_paths = site_packages + [f"{path}/torch" for path in site_packages]
    return [
        "cmake",
        "-S",
        os.fspath(source_dir),
        "-B",
        os.fspath(build_dir),
        "-G",
        "Ninja",
        f"-DCMAKE_PREFIX_PATH={';'.join(prefix_paths)}",
        f"-DPython_EXECUTABLE={sys.executable}",
        f"-DCMAKE_MODULE_PATH={pytorch_dir / 'cmake' / 'public'}",
        "-DUSE_ROCM=ON",
    ]


def _run_custom_test(pytorch_dir: Path, test: CustomTest) -> None:
    source_dir = pytorch_dir / "test" / test.source_directory
    checkout_build_dir = source_dir / "build"
    model_path = source_dir / "model.pt"

    with tempfile.TemporaryDirectory(prefix=f"{test.source_directory}-") as tmp:
        build_dir = Path(tmp) / "build"
        subprocess.check_call(
            cmake_configure_command(pytorch_dir, source_dir, build_dir)
        )
        subprocess.check_call(["cmake", "--build", os.fspath(build_dir), "--verbose"])

        shutil.rmtree(checkout_build_dir, ignore_errors=True)
        shutil.copytree(build_dir, checkout_build_dir)
        executable = checkout_build_dir / test.executable
        if os.name == "nt":
            executable = executable.with_suffix(".exe")

        try:
            subprocess.check_call(
                [sys.executable, test.python_test, "-v"], cwd=source_dir
            )
            subprocess.check_call(
                [sys.executable, test.export_script, test.export_argument],
                cwd=source_dir,
            )
            subprocess.check_call([os.fspath(executable), "./model.pt"], cwd=source_dir)
        finally:
            model_path.unlink(missing_ok=True)
            shutil.rmtree(checkout_build_dir, ignore_errors=True)


def run_custom_tests(pytorch_dir: Path) -> None:
    """Build and run upstream's custom operator and backend tests."""
    for test in CUSTOM_TESTS:
        _run_custom_test(pytorch_dir, test)
