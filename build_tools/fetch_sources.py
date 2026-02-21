#!/usr/bin/env python
# Fetches sources from a specified branch/set of projects.
# This script is available for users, but it is primarily the mechanism
# the CI uses to get to a clean state.
#
# Stage-aware fetching:
#   Use --stage <stage_name> to fetch only submodules needed for a build stage.
#   This uses BUILD_TOPOLOGY.toml to determine which submodules are required.
#
# Legacy flag-based fetching:
#   Use --include-* flags to control which project groups to fetch.
#   This is the original behavior and is still supported.
#
# External repository support:
#   When building external repositories (rocm-libraries, rocm-systems) that are
#   checked out separately instead of as git submodules, exclude the submodule
#   and provide the external path:
#
#   Example:
#     python build_tools/fetch_sources.py \
#       --no-include-rocm-libraries \
#       --rocm-libraries-path /home/user/rocm-libraries

import argparse
import hashlib
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
from typing import List
import os

THIS_SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = THIS_SCRIPT_DIR.parent
PATCHES_DIR = THEROCK_DIR / "patches"
TOPOLOGY_PATH = THEROCK_DIR / "BUILD_TOPOLOGY.toml"
ALWAYS_SUBMODULE_PATHS = [
    "base/rocm-kpack",
]


def is_windows() -> bool:
    return platform.system() == "Windows"


def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def run_command(args: list[str | Path], cwd: Path, env: dict[str, str] | None = None):
    args = [str(arg) for arg in args]
    log(f"++ Exec [{cwd}]$ {shlex.join(args)}")
    sys.stdout.flush()

    full_env = {**os.environ, **(env or {})}
    subprocess.check_call(args, cwd=str(cwd), env=full_env, stdin=subprocess.DEVNULL)


def get_projects_from_topology(stage: str) -> List[str]:
    """Get submodule names for a build stage from BUILD_TOPOLOGY.toml."""
    from _therock_utils.build_topology import BuildTopology

    if not TOPOLOGY_PATH.exists():
        raise FileNotFoundError(f"BUILD_TOPOLOGY.toml not found at {TOPOLOGY_PATH}")

    topology = BuildTopology(str(TOPOLOGY_PATH))
    current_platform = platform.system().lower()
    submodules = topology.get_submodules_for_stage(stage, platform=current_platform)
    return [s.name for s in submodules]


def get_available_stages() -> List[str]:
    """Get list of available build stages from BUILD_TOPOLOGY.toml."""
    from _therock_utils.build_topology import BuildTopology

    if not TOPOLOGY_PATH.exists():
        return []

    topology = BuildTopology(str(TOPOLOGY_PATH))
    return [s.name for s in topology.get_build_stages()]


def parse_nested_submodules(input):
    """Parse nested submodules string like 'iree:flatcc,something' into ("iree", ["flatcc", "something"])."""
    project, nested = input.split(":", 1)
    nested_list = [n.strip() for n in nested.split(",")] if nested else []
    return (project, nested_list)


def get_enabled_projects(args) -> List[str]:
    """Get list of submodule names to fetch.

    If --stage is provided, uses BUILD_TOPOLOGY.toml to determine submodules.
    Otherwise, uses the legacy --include-* flags.
    """
    # Stage-aware mode: use topology
    if args.stage:
        projects = get_projects_from_topology(args.stage)
        log(f"Stage '{args.stage}' requires submodules: {projects}")
        return projects

    # Legacy flag-based mode
    projects = []
    if args.include_system_projects:
        projects.extend(args.system_projects)
    if args.include_compilers:
        projects.extend(args.compiler_projects)
    if args.include_debug_tools:
        projects.extend(args.debug_tools)
    if args.include_rocm_libraries:
        projects.extend(["rocm-libraries"])
    if args.include_rocm_systems:
        projects.extend(["rocm-systems"])
    if args.include_ml_frameworks:
        projects.extend(args.ml_framework_projects)
    if args.include_media_libs:
        projects.extend(args.media_libs_projects)
    if args.include_iree_libs:
        projects.extend(args.iree_libs_projects)
    if args.include_math_libraries:
        projects.extend(args.math_library_projects)
    return projects


def fetch_nested_submodules(args, projects):
    """Fetch nested submodules for projects specified in --nested-submodules."""
    update_args = []
    if args.depth:
        update_args += ["--depth", str(args.depth)]
    if args.progress:
        update_args += ["--progress"]
    if args.jobs:
        update_args += ["--jobs", str(args.jobs)]
    if args.remote:
        update_args += ["--remote"]

    for parent, nested_submodules in dict(args.nested_submodules).items():
        if len(nested_submodules) == 0:
            continue

        # Skip if parent project wasn't fetched
        if parent not in projects:
            continue

        # Fetch the nested submodules
        parent_dir = THEROCK_DIR / get_submodule_path(parent)
        nested_submodule_paths = [
            get_submodule_path(nested_submodule, cwd=parent_dir)
            for nested_submodule in nested_submodules
        ]
        run_command(
            ["git", "submodule", "update", "--init"]
            + update_args
            + ["--"]
            + nested_submodule_paths,
            cwd=parent_dir,
        )


def _detect_external_source_override(args):
    """Detect external source checkout configuration from command-line arguments.

    When external repos (rocm-libraries, rocm-systems) are checked out separately
    instead of as git submodules, this function identifies which project name
    corresponds to the submodule being replaced and where the external source is located.

    Args:
        args: Parsed command-line arguments

    Returns:
        tuple: (external_repo_name: str|None, external_source_dir: Path|None)
            external_repo_name: Name of the external repo (e.g., "rocm-libraries")
            external_source_dir: Path to the external repo checkout directory
    """
    # Check for rocm-libraries external path
    if args.rocm_libraries_path:
        external_repo_name = "rocm-libraries"
        external_source_dir = Path(args.rocm_libraries_path)
        log(f"External source detected: {external_repo_name}")
        log(f"External source path: {external_source_dir}")
        log(f"Skipping submodule update for: {external_repo_name}")
        return external_repo_name, external_source_dir

    # Check for rocm-systems external path
    if args.rocm_systems_path:
        external_repo_name = "rocm-systems"
        external_source_dir = Path(args.rocm_systems_path)
        log(f"External source detected: {external_repo_name}")
        log(f"External source path: {external_source_dir}")
        log(f"Skipping submodule update for: {external_repo_name}")
        return external_repo_name, external_source_dir

    return None, None


def run(args):
    # Detect if we're using an external repo checkout instead of a git submodule
    # external_repo_name: name of the external repo (e.g., "rocm-libraries")
    # external_source_dir: path to the external repo checkout directory
    external_repo_name, external_source_dir = _detect_external_source_override(args)

    projects = get_enabled_projects(args)

    # Build submodule list, excluding external repo if present
    submodule_paths = ALWAYS_SUBMODULE_PATHS + [
        get_submodule_path(project)
        for project in projects
        if project != external_repo_name
    ]

    # TODO(scotttodd): Check for git lfs?
    update_args = []
    if args.depth:
        update_args += ["--depth", str(args.depth)]
    if args.progress:
        update_args += ["--progress"]
    if args.jobs:
        update_args += ["--jobs", str(args.jobs)]
    if args.remote:
        update_args += ["--remote"]
    if args.update_submodules:
        run_command(
            ["git", "submodule", "update", "--init"]
            + update_args
            + ["--"]
            + submodule_paths,
            cwd=THEROCK_DIR,
        )

    # Fetch nested submodules
    if args.update_submodules:
        fetch_nested_submodules(args, projects)

    # Because we allow local patches, if a submodule is in a patched state,
    # we manually set it to skip-worktree since recording the commit is
    # then meaningless. Here on each fetch, we reset the flag so that if
    # patches are aged out, the tree is restored to normal.
    # Skip external repos since they're not submodules.
    submodule_paths = [
        get_submodule_path(name) for name in projects if name != external_repo_name
    ]
    if submodule_paths:
        run_command(
            ["git", "update-index", "--no-skip-worktree", "--"] + submodule_paths,
            cwd=THEROCK_DIR,
        )

    # Remove any stale .smrev files.
    remove_smrev_files(args, projects, external_repo_name)

    if args.apply_patches:
        apply_patches(args, projects, external_repo_name, external_source_dir)

    if args.dvc_projects:
        pull_large_files(
            args.dvc_projects, projects, external_repo_name, external_source_dir
        )


def pull_large_files(
    dvc_projects, projects, external_repo_name=None, external_source_dir=None
):
    if not dvc_projects:
        print("No DVC projects specified, skipping large file pull.")
        return
    dvc_missing = shutil.which("dvc") is None
    if dvc_missing:
        if is_windows():
            print("Could not find `dvc` on PATH so large files could not be fetched")
            print("Visit https://dvc.org/doc/install for installation instructions.")
            sys.exit(1)
        else:
            print("`dvc` not found, skipping large file pull on Linux.")
            return
    for project in dvc_projects:
        if not project in projects:
            continue

        # Use external source path if this is the external repo, otherwise use submodule path
        if project == external_repo_name and external_source_dir:
            project_dir = external_source_dir
        else:
            submodule_path = get_submodule_path(project)
            project_dir = THEROCK_DIR / submodule_path

        dvc_config_file = project_dir / ".dvc" / "config"
        if dvc_config_file.exists():
            print(f"dvc detected in {project_dir}, running dvc pull")
            run_command(["dvc", "pull"], cwd=project_dir)
        else:
            log(f"WARNING: dvc config not found in {project_dir}, when expected.")


def remove_smrev_files(args, projects, external_repo_name=None):
    for project in projects:
        if project == external_repo_name:
            continue  # Skip external repos
        submodule_path = get_submodule_path(project)
        project_dir = THEROCK_DIR / submodule_path
        project_revision_file = project_dir.with_name(f".{project_dir.name}.smrev")
        if project_revision_file.exists():
            print(f"Remove stale project revision file: {project_revision_file}")
            project_revision_file.unlink()


def apply_patches(args, projects, external_repo_name=None, external_source_dir=None):
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

        is_external_repo = (
            patch_project_dir.name == external_repo_name and external_source_dir
        )

        # Use external source path if this is the external repo, otherwise use submodule path
        if is_external_repo:
            project_dir = external_source_dir
            # For external sources, we don't have submodule metadata
            submodule_path = None
            submodule_url = None
            submodule_revision = None
        else:
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

        if not patch_files:
            log(f"No patches to apply (all skipped or none exist)")
            continue

        log(f"Applying {len(patch_files)} patches")
        run_command(
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
            env={
                "GIT_COMMITTER_DATE": "Thu, 1 Jan 2099 00:00:00 +0000",
            },
        )
        # Since it is in a patched state, make it invisible to changes.
        # Skip this for external sources since they're not submodules
        if not is_external_repo:
            run_command(
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
        # Only generate .smrev files for submodules, not external repos
        if submodule_revision:
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
def get_submodule_path(name: str, cwd=THEROCK_DIR) -> str:
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
            cwd=cwd,
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


def main(argv):
    parser = argparse.ArgumentParser(
        prog="fetch_sources",
        description="Fetch sources for TheRock build. Use --stage for stage-aware "
        "fetching or --include-* flags for legacy mode.",
    )

    # Stage-aware fetching (preferred for CI)
    available_stages = get_available_stages()
    parser.add_argument(
        "--stage",
        type=str,
        choices=available_stages if available_stages else None,
        help=f"Build stage to fetch sources for. Uses BUILD_TOPOLOGY.toml. "
        f"Available: {', '.join(available_stages) if available_stages else 'none'}",
    )
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="List available build stages and their submodules, then exit",
    )

    # Legacy options
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
        "--progress",
        default=False,
        action="store_true",
        help="Git progress displayed when updating submodules",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        help="Number of jobs to use for updating submodules",
        default=None,
    )
    parser.add_argument(
        "--nested-submodules",
        nargs="+",
        type=parse_nested_submodules,
        default=[
            (
                "iree",
                [
                    "third_party/flatcc",
                    "third_party/benchmark",
                    "third_party/llvm-project",
                    "third_party/torch-mlir",
                ],
            )
        ],
        help="Specify which nested submodules to fetch (e.g., project1:nested_in_project1_1,nested_in_project1_2 project2:nested_in_project2)",
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
        "--include-debug-tools",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Include ROCm debugging tools",
    )
    parser.add_argument(
        "--include-rocm-libraries",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Include supported rocm-libraries projects",
    )
    parser.add_argument(
        "--include-rocm-systems",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Include supported rocm-systems projects",
    )
    parser.add_argument(
        "--include-ml-frameworks",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Include machine learning frameworks that are part of ROCM",
    )
    parser.add_argument(
        "--include-media-libs",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Include media projects that are part of ROCM",
    )
    parser.add_argument(
        "--include-iree-libs",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Include IREE and related libraries",
    )
    parser.add_argument(
        "--include-math-libraries",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Include math libraries that are part of ROCM",
    )
    parser.add_argument(
        "--system-projects",
        nargs="+",
        type=str,
        default=[
            "half",
            "rocm-cmake",
            "rocprof-trace-decoder",
        ],
    )
    parser.add_argument(
        "--compiler-projects",
        nargs="+",
        type=str,
        default=[
            "HIPIFY",
            "llvm-project",
            "spirv-llvm-translator",
        ],
    )
    parser.add_argument(
        "--ml-framework-projects",
        nargs="+",
        type=str,
        default=[],
    )
    parser.add_argument(
        "--media-libs-projects",
        nargs="+",
        type=str,
        default=(
            []
            if is_windows()
            else [
                # Linux only projects.
                "amd-mesa",
            ]
        ),
    )
    parser.add_argument(
        "--iree-libs-projects",
        nargs="+",
        type=str,
        default=[
            "iree",
            "fusilli",
        ],
    )
    parser.add_argument(
        # projects that use DVC to manage large files
        "--dvc-projects",
        nargs="+",
        type=str,
        default=(
            [
                "rocm-libraries",
                "rocm-systems",
            ]
            if is_windows()
            else [
                "rocm-libraries",
            ]
        ),
    )
    parser.add_argument(
        "--debug-tools",
        nargs="+",
        type=str,
        default=(
            []
            if is_windows()
            else [
                # Linux only projects.
                "rocgdb",
            ]
        ),
    )
    parser.add_argument(
        "--math-library-projects",
        nargs="+",
        type=str,
        default=(
            []
            if is_windows()
            else [
                # Linux only projects.
                "libhipcxx",
            ]
        ),
    )
    parser.add_argument(
        "--rocm-libraries-path",
        type=str,
        default=None,
        help="Path to external rocm-libraries checkout (use with --no-include-rocm-libraries)",
    )
    parser.add_argument(
        "--rocm-systems-path",
        type=str,
        default=None,
        help="Path to external rocm-systems checkout (use with --no-include-rocm-systems)",
    )
    args = parser.parse_args(argv)

    # Handle --list-stages
    if args.list_stages:
        from _therock_utils.build_topology import BuildTopology

        if not TOPOLOGY_PATH.exists():
            print(f"BUILD_TOPOLOGY.toml not found at {TOPOLOGY_PATH}")
            sys.exit(1)

        topology = BuildTopology(str(TOPOLOGY_PATH))
        print("Available build stages and their submodules:\n")
        for stage in topology.get_build_stages():
            submodules = topology.get_submodules_for_stage(stage.name)
            submodule_names = [s.name for s in submodules]
            print(f"  {stage.name} ({stage.type}):")
            print(f"    {stage.description}")
            print(
                f"    Submodules: {', '.join(submodule_names) if submodule_names else '(none)'}"
            )
            print()
        sys.exit(0)

    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
