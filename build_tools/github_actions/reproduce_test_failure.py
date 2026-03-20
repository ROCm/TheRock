# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Reproduces a test failure from CI.

Usage:
    # Linux (uses Docker)
    python reproduce_test_failure.py --run-id 12345678 --repository ROCm/TheRock \
        --amdgpu-family gfx94X --test-script "python test.py" --platform linux

    # Windows (bare metal, requires admin PowerShell)
    python reproduce_test_failure.py --run-id 12345678 --repository ROCm/TheRock \
        --amdgpu-family gfx110X --test-script "python test.py" --platform windows

    # Setup only (drops into shell)
    python reproduce_test_failure.py --run-id 12345678 --repository ROCm/TheRock \
        --amdgpu-family gfx94X --test-script "python test.py" --setup-only
"""

import argparse
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_CONTAINER_IMAGE = "ghcr.io/rocm/no_rocm_image_ubuntu24_04:latest"

is_windows = platform.system() == "Windows"


def check_windows_package_installed(package: str) -> bool:
    """Check if a Windows package is already installed."""
    if platform.system() != "Windows":
        return False

    # Map package names to their executables or check commands
    checks = {
        "chocolatey": ["choco", "--version"],
        "git": ["git", "--version"],
        "python": ["python", "--version"],
        "cmake": ["cmake", "--version"],
        "ninja": ["ninja", "--version"],
        "ccache": ["ccache", "--version"],
        "uv": ["uv", "--version"],
    }

    if package not in checks:
        return False

    try:
        result = subprocess.run(
            checks[package],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def check_docker() -> bool:
    if not shutil.which("docker"):
        return False
    result = subprocess.run(["docker", "info"], capture_output=True)
    return result.returncode == 0


def build_reproduction_command(args: argparse.Namespace) -> str:
    """Build the command string for reproduction."""
    cmd = (
        f"python build_tools/github_actions/reproduce_test_failure.py "
        f"--run-id {args.run_id} "
        f"--repository {args.repository} "
        f"--amdgpu-family {args.amdgpu_family} "
        f'--test-script "{args.test_script}" '
    )
    if args.shard_index != "1":
        cmd += f" --shard-index {args.shard_index}"
    if args.total_shards != "1":
        cmd += f" --total-shards {args.total_shards}"
    if args.test_type != "full":
        cmd += f" --test-type {args.test_type}"
    if args.fetch_artifact_args:
        cmd += f' --fetch-artifact-args="{args.fetch_artifact_args}"'
    return cmd


def run_linux(args: argparse.Namespace) -> int:
    """Run reproduction in Docker for Linux."""
    if not check_docker():
        print("ERROR: Docker is not available. Install Docker and try again.")
        return 1

    fetch_cmd = (
        f"GITHUB_REPOSITORY={args.repository} "
        f"python build_tools/install_rocm_from_artifacts.py "
        f"--run-id {args.run_id} "
        f"--amdgpu-family {args.amdgpu_family}"
    )
    if args.fetch_artifact_args:
        fetch_cmd += f" {args.fetch_artifact_args}"

    steps = [
        (
            "Installing uv",
            "curl -LsSf https://astral.sh/uv/install.sh | bash && source $HOME/.local/bin/env",
        ),
        (
            "Cloning TheRock",
            "git clone https://github.com/ROCm/TheRock.git && cd TheRock",
        ),
        ("Creating virtual environment", "uv venv .venv && source .venv/bin/activate"),
        ("Installing dependencies", "uv pip install -r requirements-test.txt"),
        ("Downloading artifacts", fetch_cmd),
        (
            "Setting environment variables",
            " && ".join(
                [
                    "export THEROCK_BIN_DIR=./therock-build/bin",
                    "export OUTPUT_ARTIFACTS_DIR=./therock-build",
                    f"export SHARD_INDEX={args.shard_index}",
                    f"export TOTAL_SHARDS={args.total_shards}",
                    f"export TEST_TYPE={args.test_type}",
                ]
            ),
        ),
    ]

    if args.setup_only:
        steps.append(("Setup complete", f"echo 'Run: {args.test_script}'"))
    else:
        steps.append(
            (
                "Running test",
                f"{args.test_script} || echo 'Test failed with exit code '$?",
            )
        )

    total = len(steps)
    lines = ["set -e"]
    for i, (desc, cmd) in enumerate(steps, 1):
        lines.append(f"echo '[{i}/{total}] {desc}'")
        if i == total:
            lines.append("set +e")
        lines.append(cmd)
    lines.append("exec /bin/bash")

    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "-it",
        "--ipc",
        "host",
        "--group-add",
        "video",
        "--device",
        "/dev/kfd",
        "--device",
        "/dev/dri",
        args.container_image,
        "/bin/bash",
        "-c",
        "\n".join(lines),
    ]

    try:
        return subprocess.run(docker_cmd).returncode
    except KeyboardInterrupt:
        return 130


def run_windows(args: argparse.Namespace) -> int:
    """Run reproduction on bare metal for Windows."""
    all_packages = ["chocolatey", "git", "python", "cmake", "ninja", "ccache", "uv"]

    # Check which packages are already installed
    installed = []
    missing = []
    for pkg in all_packages:
        if check_windows_package_installed(pkg):
            installed.append(pkg)
        else:
            missing.append(pkg)

    print("=" * 60)
    print("WINDOWS REPRODUCTION (bare metal)")
    print("=" * 60)
    print()

    if installed:
        print("Already installed:")
        for pkg in installed:
            print(f"  - {pkg}")
        print()

    # Only prompt if there are packages to install
    if missing:
        print("This will install the following packages on your system:")
        for pkg in missing:
            print(f"  - {pkg}")
        print()
        response = input("Continue? [y/N] ").strip().lower()
        if response not in ("y", "yes"):
            print("Aborted.")
            return 1
        print()
    else:
        print("All required packages are already installed.")
        print()

    fetch_cmd = (
        f"$env:GITHUB_REPOSITORY='{args.repository}'; "
        f"python build_tools/install_rocm_from_artifacts.py "
        f"--run-id {args.run_id} "
        f"--amdgpu-family {args.amdgpu_family}"
    )
    if args.fetch_artifact_args:
        fetch_cmd += f" {args.fetch_artifact_args}"

    # Build steps conditionally based on what's already installed
    steps = []

    if "chocolatey" in missing:
        steps.append(
            (
                "Installing chocolatey",
                (
                    "Set-ExecutionPolicy Bypass -Scope Process -Force; "
                    "[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; "
                    "iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"
                ),
            )
        )
        # Must use Import-Module to refresh env since [System.Environment] can be stale
        steps.append(
            (
                "Refreshing environment",
                "Import-Module $env:ChocolateyInstall\\helpers\\chocolateyProfile.psm1; refreshenv",
            )
        )

    # Check which choco packages need installing
    choco_packages = [
        p for p in ["git", "python", "cmake", "ninja", "ccache"] if p in missing
    ]
    if choco_packages:
        steps.append(
            ("Installing dependencies", f"choco install -y {' '.join(choco_packages)}")
        )
        steps.append(
            (
                "Refreshing environment",
                "Import-Module $env:ChocolateyInstall\\helpers\\chocolateyProfile.psm1; refreshenv",
            )
        )

    if "uv" in missing:
        steps.append(("Installing uv", "irm https://astral.sh/uv/install.ps1 | iex"))
        # Refresh PATH to pick up uv installation
        steps.append(
            (
                "Refreshing environment for uv",
                "$env:Path = [System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User') + ';' + \"$env:USERPROFILE\\.local\\bin\"",
            )
        )

    steps.extend(
        [
            (
                "Creating virtual environment",
                "uv venv .venv; .venv\\Scripts\\Activate.ps1",
            ),
            (
                "Installing Python dependencies",
                "uv pip install -r requirements-test.txt",
            ),
            ("Downloading artifacts", fetch_cmd),
            (
                "Setting environment variables",
                "; ".join(
                    [
                        "$env:THEROCK_BIN_DIR='./therock-build/bin'",
                        "$env:OUTPUT_ARTIFACTS_DIR='./therock-build'",
                        f"$env:SHARD_INDEX='{args.shard_index}'",
                        f"$env:TOTAL_SHARDS='{args.total_shards}'",
                        f"$env:TEST_TYPE='{args.test_type}'",
                    ]
                ),
            ),
        ]
    )

    if args.setup_only:
        steps.append(("Setup complete", f"Write-Host 'Run: {args.test_script}'"))
    else:
        steps.append(("Running test", args.test_script))

    total = len(steps)
    lines = ["$ErrorActionPreference = 'Stop'"]
    for i, (desc, cmd) in enumerate(steps, 1):
        lines.append(f"Write-Host '[{i}/{total}] {desc}'")
        if i == total:
            lines.append("$ErrorActionPreference = 'Continue'")
        lines.append(cmd)

    script_content = "\n".join(lines)

    # Write to temp file and execute
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ps1", delete=False) as f:
        f.write(script_content)
        script_path = f.name

    try:
        if platform.system() != "Windows":
            print("NOTE: Not running on Windows. Printing script instead:")
            print("-" * 60)
            print(script_content)
            print("-" * 60)
            print(f"\nScript saved to: {script_path}")
            print("Copy this script to your Windows machine and run:")
            print(
                f"  powershell -ExecutionPolicy Bypass -File {Path(script_path).name}"
            )
            return 0

        result = subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", script_path],
        )
        return result.returncode
    except KeyboardInterrupt:
        return 130
    finally:
        if platform.system() == "Windows":
            Path(script_path).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce a test failure from CI")
    parser.add_argument("--run-id", required=True, help="GitHub Actions run ID")
    parser.add_argument("--repository", required=True, help="GitHub repository")
    parser.add_argument("--amdgpu-family", required=True, help="AMDGPU family")
    parser.add_argument("--test-script", required=True, help="Test script to run")
    parser.add_argument("--shard-index", default="1", help="Shard index")
    parser.add_argument("--total-shards", default="1", help="Total shards")
    parser.add_argument("--test-type", default="full", help="Test type")
    parser.add_argument(
        "--container-image",
        default=DEFAULT_CONTAINER_IMAGE,
        help="Docker image (Linux only)",
    )
    parser.add_argument(
        "--fetch-artifact-args", nargs="?", default="", help="Extra artifact args"
    )
    parser.add_argument(
        "--setup-only", action="store_true", help="Setup only, don't run test"
    )
    parser.add_argument(
        "--print-cmd", action="store_true", help="Print reproduction command"
    )

    args = parser.parse_args()

    if args.print_cmd:
        print("To reproduce this failure, run:")
        print("  git clone https://github.com/ROCm/TheRock.git && cd TheRock")
        print(f"  {build_reproduction_command(args)}")
        return 0

    if is_windows:
        return run_windows(args)
    else:
        return run_linux(args)


if __name__ == "__main__":
    sys.exit(main())
