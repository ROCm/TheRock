#!/usr/bin/env python
# Fetches sources from a specified branch/set of projects.
# This script is available for users, but it is primarily the mechanism
# the CI uses to get to a clean state.

import argparse
import hashlib
from pathlib import Path
import platform
import shlex
import subprocess
import sys

THIS_SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = THIS_SCRIPT_DIR.parent
PATCHES_DIR = THEROCK_DIR / "patches"


def is_windows() -> bool:
    return platform.system() == "Windows"


def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def exec(args: list[str | Path], cwd: Path):
    args = [str(arg) for arg in args]
    log(f"++ Exec [{cwd}]$ {shlex.join(args)}")
    sys.stdout.flush()
    subprocess.check_call(args, cwd=str(cwd), stdin=subprocess.DEVNULL)


def get_enabled_projects(args) -> list[str]:
    projects = []
    if args.include_system_projects:
        projects.extend(args.system_projects)
    if args.include_compilers:
        projects.extend(args.compiler_projects)
    if args.include_math_libs:
        projects.extend(args.math_lib_projects)
    if args.include_ml_frameworks:
        projects.extend(args.ml_framework_projects)
    return projects


def run(args):
    projects = get_enabled_projects(args)
    submodule_paths = [get_submodule_path(project) for project in projects]
    # TODO(scotttodd): Check for git lfs?
    update_args = []
    if args.depth:
        update_args += ["--depth", str(args.depth)]
    if args.jobs:
        update_args += ["--jobs", str(args.jobs)]
    if args.remote:
        update_args += ["--remote"]
    if args.update_submodules:
        exec(
            ["git", "submodule", "update", "--init"]
            + update_args
            + ["--"]
            + submodule_paths,
            cwd=THEROCK_DIR,
        )

    # Because we allow local patches, if a submodule is in a patched state,
    # we manually set it to skip-worktree since recording the commit is
    # then meaningless. Here on each fetch, we reset the flag so that if
    # patches are aged out, the tree is restored to normal.
    submodule_paths = [get_submodule_path(name) for name in projects]
    exec(
        ["git", "update-index", "--no-skip-worktree", "--"] + submodule_paths,
        cwd=THEROCK_DIR,
    )

    populate_ancillary_sources(args)

    # Remove any stale .smrev files.
    remove_smrev_files(args, projects)

    if args.apply_patches:
        apply_patches(args, projects)


def remove_smrev_files(args, projects):
    for project in projects:
        submodule_path = get_submodule_path(project)
        project_dir = THEROCK_DIR / submodule_path
        project_revision_file = project_dir.with_name(f".{project_dir.name}.smrev")
        if project_revision_file.exists():
            print(f"Remove stale project revision file: {project_revision_file}")
            project_revision_file.unlink()


def apply_patches(args, projects):
    if not args.patch_tag:
        log("Not patching (no --patch-tag specified)")
    patch_version_dir: Path = PATCHES_DIR / args.patch_tag
    if not patch_version_dir.exists():
        log(f"ERROR: Patch directory {patch_version_dir} does not exist")
    for patch_project_dir in patch_version_dir.iterdir():
        log(f"* Processing project patch directory {patch_project_dir}:")
        # Check that project patch directory was included
        if not patch_project_dir.name in projects:
            log(
                f"* Project patch directory {patch_project_dir.name} was not included. Skipping."
            )
            continue
        submodule_path = get_submodule_path(patch_project_dir.name)
        submodule_url = get_submodule_url(patch_project_dir.name)
        submodule_revision = get_submodule_revision(submodule_path)
        project_dir = THEROCK_DIR / submodule_path
        project_revision_file = project_dir.with_name(f".{project_dir.name}.smrev")

        if not project_dir.exists():
            log(f"WARNING: Source directory {project_dir} does not exist. Skipping.")
            continue
        patch_files = list(patch_project_dir.glob("*.patch"))
        patch_files.sort()
        log(f"Applying {len(patch_files)} patches")
        exec(
            [
                "git",
                "-c",
                "user.name=therockbot",
                "-c",
                "user.email=therockbot@amd.com",
                "am",
                "--whitespace=nowarn",
            ]
            + patch_files,
            cwd=project_dir,
        )
        # Since it is in a patched state, make it invisible to changes.
        exec(
            ["git", "update-index", "--skip-worktree", "--", submodule_path],
            cwd=THEROCK_DIR,
        )

        # Generate the .smrev patch state file.
        # This file consists of two lines: The git origin and a summary of the
        # state of the source tree that was checked out. This can be consumed
        # by individual build steps in lieu of heuristics for asking git. If
        # the tree is in a patched state, the commit hashes of HEAD may be
        # different from checkout-to-checkout, but the .smrev file will have
        # stable contents so long as the submodule pin and contents of the
        # hashes are the same.
        # Note that this does not track the dirty state of the tree. If full
        # fidelity hashes of the tree state are needed for development/dirty
        # trees, then another mechanism must be used.
        patches_hash = hashlib.sha1()
        for patch_file in patch_files:
            patch_contents = Path(patch_file).read_bytes()
            patches_hash.update(patch_contents)
        patches_digest = patches_hash.digest().hex()
        project_revision_file.write_text(
            f"{submodule_url}\n{submodule_revision}+PATCHED:{patches_digest}\n"
        )


# Gets the the relative path to a submodule given its name.
# Raises an exception on failure.
def get_submodule_path(name: str) -> str:
    relpath = (
        subprocess.check_output(
            [
                "git",
                "config",
                "--file",
                ".gitmodules",
                "--get",
                f"submodule.{name}.path",
            ],
            cwd=str(THEROCK_DIR),
        )
        .decode()
        .strip()
    )
    return relpath


# Gets the the relative path to a submodule given its name.
# Raises an exception on failure.
def get_submodule_url(name: str) -> str:
    relpath = (
        subprocess.check_output(
            [
                "git",
                "config",
                "--file",
                ".gitmodules",
                "--get",
                f"submodule.{name}.url",
            ],
            cwd=str(THEROCK_DIR),
        )
        .decode()
        .strip()
    )
    return relpath


def get_submodule_revision(submodule_path: str) -> str:
    # Generates a line like:
    #   160000 5e2093d23f7d34c372a788a6f2b7df8bc1c97947 0       compiler/amd-llvm
    ls_line = (
        subprocess.check_output(
            ["git", "ls-files", "--stage", submodule_path], cwd=str(THEROCK_DIR)
        )
        .decode()
        .strip()
    )
    return ls_line.split()[1]


def populate_ancillary_sources(args):
    """Various subprojects have their own mechanisms for populating ancillary sources
    needed to build. There is often something in CMake that attempts to automate it,
    but it is also often broken. So we just do the right thing here as a transitionary
    step to fixing the underlying software packages."""
    populate_submodules_if_exists(args, THEROCK_DIR / "base" / "rocprofiler-register")
    populate_submodules_if_exists(args, THEROCK_DIR / "profiler" / "rocprofiler-sdk")

    # TODO(#36): Enable once rocprofiler-systems can be checked out on Windows
    #     error: invalid path 'src/counter_analysis_toolkit/scripts/sample_data/L2_RQSTS:ALL_DEMAND_REFERENCES.data.reads.stat'
    #  Upstream issues:
    #   https://github.com/ROCm/rocprofiler-systems/issues/105
    #   https://github.com/icl-utk-edu/papi/issues/321
    if not is_windows():
        populate_submodules_if_exists(
            args, THEROCK_DIR / "profiler" / "rocprofiler-systems"
        )


def populate_submodules_if_exists(args, git_dir: Path):
    if not git_dir.exists():
        print(f"Not populating submodules for {git_dir} (does not exist)")
        return
    print(f"Populating submodules for {git_dir}:")
    update_args = []
    if args.depth is not None:
        update_args = ["--depth", str(args.depth)]
    if args.jobs:
        update_args += ["--jobs", str(args.jobs)]
    exec(["git", "submodule", "update", "--init"] + update_args, cwd=git_dir)


def main(argv):
    parser = argparse.ArgumentParser(prog="fetch_sources")
    parser.add_argument(
        "--patch-tag",
        type=str,
        default="amd-mainline",
        help="Patch tag to apply to sources after sync",
    )
    parser.add_argument(
        "--update-submodules",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Updates submodules",
    )
    parser.add_argument(
        "--remote",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Updates submodules from remote vs current",
    )
    parser.add_argument(
        "--apply-patches",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Apply patches",
    )
    parser.add_argument(
        "--depth", type=int, help="Git depth when updating submodules", default=None
    )
    parser.add_argument(
        "--jobs",
        type=int,
        help="Number of jobs to use for updating submodules",
        default=None,
    )
    parser.add_argument(
        "--include-system-projects",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Include systems projects",
    )
    parser.add_argument(
        "--include-compilers",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Include compilers",
    )
    parser.add_argument(
        "--include-math-libs",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Include supported math libraries",
    )
    parser.add_argument(
        "--include-ml-frameworks",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Include machine learning frameworks that are part of ROCM",
    )
    parser.add_argument(
        "--system-projects",
        nargs="+",
        type=str,
        default=[
            "aqlprofile",
            "clr",
            "half",
            "HIP",
            "rccl",
            "rccl-tests",
            "rocm_smi_lib",
            "rocm-cmake",
            "rocm-core",
            "rocminfo",
            "rocprofiler-register",
            # TODO: Re-enable when used.
            # "rocprofiler-compute",
            "rocprofiler-sdk",
            "rocprof-trace-decoder",
            # TODO: Re-enable when used.
            # "rocprofiler-systems",
            "roctracer",
            "ROCR-Runtime",
        ]
        + (
            [
                "amdgpu-windows-interop",
            ]
            if is_windows()
            else []
        ),
    )
    parser.add_argument(
        "--compiler-projects",
        nargs="+",
        type=str,
        default=[
            "HIPIFY",
            "llvm-project",
        ],
    )
    parser.add_argument(
        "--math-lib-projects",
        nargs="+",
        type=str,
        default=[
            "hipBLAS-common",
            "hipBLAS",
            "hipBLASLt",
            "hipCUB",
            "hipFFT",
            "hipRAND",
            "hipSOLVER",
            "hipSPARSE",
            "mxDataGenerator",
            "Tensile",
            "rocBLAS",
            "rocFFT",
            "rocPRIM",
            "rocRAND",
            "rocRoller",
            "rocSOLVER",
            "rocSPARSE",
            "rocThrust",
        ],
    )
    parser.add_argument(
        "--ml-framework-projects",
        nargs="+",
        type=str,
        default=[
            "MIOpen",
        ]
        + (
            []
            if is_windows()
            else [
                # Linux only projects.
                "composable_kernel",
            ]
        ),
    )
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
