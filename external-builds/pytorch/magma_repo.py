#!/usr/bin/env python
"""Manages the MAGMA repository checkout for PyTorch builds with ROCm.

MAGMA (Matrix Algebra on GPU and Multicore Architectures) provides GPU-accelerated
linear algebra routines used by PyTorch for operations like torch.linalg.eig,
torch.linalg.svd, and other dense linear algebra operations.

Usage:
    python magma_repo.py checkout [--checkout-dir DIR] [--repo-hashtag REF]
    python magma_repo.py build [--magma-dir DIR] [--install-dir DIR]
    python magma_repo.py info
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent

is_windows = platform.system() == "Windows"

# MAGMA repository configuration
# Using the k-artem fork that PyTorch CI uses, which has ROCm fixes
DEFAULT_MAGMA_REPO = "https://github.com/k-artem/magma.git"
# This commit is from PyTorch's .ci/magma-rocm/build_magma.sh
# https://github.com/icl-utk-edu/magma/pull/77
DEFAULT_MAGMA_COMMIT = "a68b9257ac435afa1ebdfb8f50d67668950aef61"


def run_command(
    args: list[str | Path],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Run a command and check for errors."""
    args = [str(arg) for arg in args]
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    print(f"++ [{cwd or '.'}]$ {' '.join(args)}")
    subprocess.check_call(args, cwd=str(cwd) if cwd else None, env=full_env)


def capture_output(args: list[str | Path], cwd: Path | None = None) -> str:
    """Run a command and capture its output."""
    args = [str(arg) for arg in args]
    return subprocess.check_output(
        args, cwd=str(cwd) if cwd else None, text=True
    ).strip()


def do_checkout(args: argparse.Namespace) -> None:
    """Checkout the MAGMA repository."""
    checkout_dir: Path = args.checkout_dir
    repo_url = args.gitrepo_origin
    repo_hashtag = args.repo_hashtag

    print(f"=== Checking out MAGMA ===")
    print(f"  Repository: {repo_url}")
    print(f"  Ref: {repo_hashtag}")
    print(f"  Directory: {checkout_dir}")

    if checkout_dir.exists():
        if args.force:
            print(f"  Removing existing directory: {checkout_dir}")
            shutil.rmtree(checkout_dir)
        else:
            # Check if it's already at the right commit
            try:
                current_ref = capture_output(
                    ["git", "rev-parse", "HEAD"], cwd=checkout_dir
                )
                print(f"  Directory exists, current HEAD: {current_ref[:12]}")

                # Try to fetch and check if we're at target
                run_command(["git", "fetch", "origin"], cwd=checkout_dir)

                # Check if repo_hashtag is a commit hash or branch/tag
                try:
                    target_ref = capture_output(
                        ["git", "rev-parse", repo_hashtag], cwd=checkout_dir
                    )
                except subprocess.CalledProcessError:
                    target_ref = capture_output(
                        ["git", "rev-parse", f"origin/{repo_hashtag}"], cwd=checkout_dir
                    )

                if current_ref == target_ref:
                    print(f"  Already at target ref, skipping checkout")
                    return
                else:
                    print(f"  Checking out to: {target_ref[:12]}")
                    run_command(["git", "checkout", repo_hashtag], cwd=checkout_dir)
                    return
            except subprocess.CalledProcessError:
                print(
                    f"  Directory exists but is not a valid git repo, "
                    "please remove it or use --force"
                )
                sys.exit(1)

    # Clone the repository
    checkout_dir.parent.mkdir(parents=True, exist_ok=True)
    run_command(["git", "clone", repo_url, checkout_dir])

    # Checkout the specific ref
    run_command(["git", "checkout", repo_hashtag], cwd=checkout_dir)

    # Tag the checkout point
    try:
        run_command(
            ["git", "tag", "-f", "THEROCK_MAGMA_CHECKOUT", "HEAD"], cwd=checkout_dir
        )
    except subprocess.CalledProcessError:
        pass  # Tag already exists, that's fine

    print(f"=== MAGMA checkout complete ===")


def do_build(args: argparse.Namespace) -> Path:
    """Build MAGMA with ROCm/HIP support.

    Returns the path to the MAGMA installation directory.
    """
    magma_dir: Path = args.magma_dir
    install_dir: Path = args.install_dir
    rocm_path: Path = args.rocm_path
    gpu_targets: str = args.gpu_targets

    print(f"=== Building MAGMA ===")
    print(f"  Source directory: {magma_dir}")
    print(f"  Install directory: {install_dir}")
    print(f"  ROCm path: {rocm_path}")
    print(f"  GPU targets: {gpu_targets}")

    if not magma_dir.exists():
        print(f"ERROR: MAGMA source directory does not exist: {magma_dir}")
        print("Run 'python magma_repo.py checkout' first")
        sys.exit(1)

    if not rocm_path.exists():
        print(f"ERROR: ROCm path does not exist: {rocm_path}")
        sys.exit(1)

    # Create install directory
    install_dir.mkdir(parents=True, exist_ok=True)

    # Set up environment
    env = {
        "ROCM_PATH": str(rocm_path),
        "HIP_PATH": str(rocm_path),
        "PATH": f"{rocm_path}/bin:{os.environ.get('PATH', '')}",
    }

    # Find OpenBLAS from TheRock's host-math or system
    host_math_path = rocm_path / "lib" / "host-math"
    if host_math_path.exists():
        env["OpenBLAS_HOME"] = str(host_math_path)
        print(f"  Using OpenBLAS from: {host_math_path}")

    # Create make.inc configuration
    make_inc_content = generate_make_inc(rocm_path, gpu_targets, host_math_path)
    make_inc_path = magma_dir / "make.inc"
    print(f"  Writing make.inc to: {make_inc_path}")
    make_inc_path.write_text(make_inc_content)

    # Generate HIP sources
    print("+++ Generating hipMAGMA sources")
    run_command(
        ["make", "-f", "make.gen.hipMAGMA", f"-j{os.cpu_count()}"],
        cwd=magma_dir,
        env=env,
    )

    # Build the library
    print("+++ Building libmagma.so")
    run_command(
        ["make", "lib/libmagma.so", f"-j{os.cpu_count()}"],
        cwd=magma_dir,
        env=env,
    )

    # Copy files to install directory
    print("+++ Installing MAGMA")
    lib_dir = install_dir / "lib"
    include_dir = install_dir / "include"
    lib_dir.mkdir(parents=True, exist_ok=True)
    include_dir.mkdir(parents=True, exist_ok=True)

    # Copy library
    for lib_file in (magma_dir / "lib").glob("libmagma*"):
        print(f"  Copying {lib_file.name}")
        shutil.copy2(lib_file, lib_dir)

    # Copy headers
    for header in (magma_dir / "include").glob("*.h"):
        shutil.copy2(header, include_dir)

    # Copy generated config header if it exists
    build_include = magma_dir / "build" / "include"
    if build_include.exists():
        for header in build_include.glob("*.h"):
            shutil.copy2(header, include_dir)

    print(f"=== MAGMA build complete ===")
    print(f"  MAGMA_HOME={install_dir}")

    return install_dir


def generate_make_inc(
    rocm_path: Path, gpu_targets: str, openblas_path: Path | None
) -> str:
    """Generate the make.inc configuration file for MAGMA."""
    # Convert comma-separated targets to space-separated
    gpu_list = gpu_targets.replace(",", " ").replace(";", " ")

    # Build the offload-arch flags
    arch_flags = ""
    for arch in gpu_list.split():
        arch = arch.strip()
        if arch:
            arch_flags += f"DEVCCFLAGS += --offload-arch={arch}\n"

    # Determine BLAS library configuration
    if openblas_path and openblas_path.exists():
        blas_config = f"""
# OpenBLAS configuration (from TheRock)
OPENBLASDIR = {openblas_path}
LIB = -L$(OPENBLASDIR)/lib -lrocm-openblas -lpthread -lstdc++ -lm -lgomp -lhipblas -lhipsparse
INC += -I$(OPENBLASDIR)/include
LIBDIR += -L$(OPENBLASDIR)/lib
"""
    else:
        # Fall back to system OpenBLAS or MKL
        blas_config = """
# System BLAS configuration
LIB = -lopenblas -lpthread -lstdc++ -lm -lgomp -lhipblas -lhipsparse
"""

    make_inc = f"""#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 2.x) --
#      Built by TheRock for ROCm/HIP
#//////////////////////////////////////////////////////////////////////////////

# ROCm/HIP backend configuration
BACKEND      = hip
ROCM_PATH    = {rocm_path}

# Compilers
CC           = hipcc
CXX          = hipcc
FORT        ?= gfortran
HIPCC       ?= hipcc
DEVCC        = $(HIPCC)

# Utilities
ARCH        ?= ar
ARCHFLAGS   ?= cr
RANLIB      ?= ranlib

# GPU targets
GPU_TARGET   = {gpu_list}

# Compiler flags
FPIC         = -fPIC
# Note: -fopenmp can cause issues with hipcc, so we disable it
#FOPENMP     = -fopenmp
FOPENMP      =

CFLAGS       = -O3 $(FPIC) $(FOPENMP) -DNDEBUG -DADD_ -Wall -std=c99
CXXFLAGS     = -O3 $(FPIC) $(FOPENMP) -DNDEBUG -DADD_ -Wall -std=c++14
FFLAGS       = -O3 $(FPIC) -DNDEBUG -DADD_ -Wall -Wno-unused-dummy-argument
F90FLAGS     = -O3 $(FPIC) -DNDEBUG -DADD_ -Wall -Wno-unused-dummy-argument -x f95-cpp-input
LDFLAGS      =     $(FPIC) $(FOPENMP)

# Device compiler flags
DEVCCFLAGS   = -O3 -DNDEBUG -DADD_ $(FPIC) -std=c++14
DEVCCFLAGS  += --gpu-max-threads-per-block=256

# GPU architecture flags
{arch_flags}

{blas_config}

# ROCm library paths
INC         += -I$(ROCM_PATH)/include
LIBDIR      += -L$(ROCM_PATH)/lib

# Add rpath for runtime library loading
LIB         += -Wl,--enable-new-dtags -Wl,--rpath,$(ROCM_PATH)/lib -ldl

# Include checks
-include make.check-hip
"""
    return make_inc


def do_info(args: argparse.Namespace) -> None:
    """Print information about the MAGMA configuration."""
    print("MAGMA Repository Configuration for TheRock:")
    print(f"  Repository: {DEFAULT_MAGMA_REPO}")
    print(f"  Commit: {DEFAULT_MAGMA_COMMIT}")
    print()
    print("This is the k-artem fork used by PyTorch CI with ROCm fixes.")
    print()
    print("To checkout and build MAGMA:")
    print("  python magma_repo.py checkout")
    print("  python magma_repo.py build --rocm-path /path/to/rocm")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="magma_repo.py",
        description="Manage MAGMA repository and builds for PyTorch with ROCm",
    )

    subparsers = parser.add_subparsers(required=True, dest="command")

    # Checkout subcommand
    checkout_parser = subparsers.add_parser(
        "checkout", help="Checkout the MAGMA repository"
    )
    checkout_parser.add_argument(
        "--checkout-dir",
        type=Path,
        default=script_dir / "magma",
        help="Directory to checkout MAGMA into (default: ./magma)",
    )
    checkout_parser.add_argument(
        "--gitrepo-origin",
        default=DEFAULT_MAGMA_REPO,
        help=f"Git repository URL (default: {DEFAULT_MAGMA_REPO})",
    )
    checkout_parser.add_argument(
        "--repo-hashtag",
        default=DEFAULT_MAGMA_COMMIT,
        help=f"Git ref to checkout (default: {DEFAULT_MAGMA_COMMIT})",
    )
    checkout_parser.add_argument(
        "--force",
        action="store_true",
        help="Force checkout, removing existing directory if present",
    )
    checkout_parser.set_defaults(func=do_checkout)

    # Build subcommand
    build_parser = subparsers.add_parser("build", help="Build MAGMA with ROCm/HIP")
    build_parser.add_argument(
        "--magma-dir",
        type=Path,
        default=script_dir / "magma",
        help="MAGMA source directory (default: ./magma)",
    )
    build_parser.add_argument(
        "--install-dir",
        type=Path,
        default=script_dir / "magma_install",
        help="MAGMA installation directory (default: ./magma_install)",
    )
    build_parser.add_argument(
        "--rocm-path",
        type=Path,
        default=Path("/opt/rocm"),
        help="Path to ROCm installation (default: /opt/rocm)",
    )
    build_parser.add_argument(
        "--gpu-targets",
        default="gfx942",
        help="GPU architectures to build for (default: gfx942)",
    )
    build_parser.set_defaults(func=do_build)

    # Info subcommand
    info_parser = subparsers.add_parser(
        "info", help="Print MAGMA repository configuration info"
    )
    info_parser.set_defaults(func=do_info)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])




















