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


def exec(args: list[str | Path], cwd: Path, env: dict[str, str] | None = None):
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
    if args.include_rocm_media:
        projects.extend(args.rocm_media_projects)
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
        exec(
            ["git", "submodule", "update", "--init"]
            + update_args
            + ["--"]
            + nested_submodule_paths,
            cwd=parent_dir,
        )


def extract_submodule_name_from_repo_url(repo_url: str) -> str:
    """Extract submodule name from repository URL.

    Examples:
        "ROCm/rocm-libraries" -> "rocm-libraries"
        "https://github.com/ROCm/rocm-systems" -> "rocm-systems"
    """
    # Remove .git suffix if present
    repo_url = repo_url.rstrip("/")
    if repo_url.endswith(".git"):
        repo_url = repo_url[:-4]

    # Extract the last component (repo name)
    return repo_url.split("/")[-1]


def move_external_source_to_submodule(temp_path: str, submodule_name: str):
    """Move pre-checked-out external source to its submodule directory.

    This is used when GitHub Actions checks out the external repo (handling
    fork PRs correctly), and we need to move it to the submodule location.

    Args:
        temp_path: Path where GitHub Actions checked out the source
        submodule_name: Name of the submodule (e.g., "rocm-libraries")
    """
    temp_dir = THEROCK_DIR / temp_path
    if not temp_dir.exists():
        log(f"External source temp directory not found: {temp_dir}")
        return

    submodule_path = get_submodule_path(submodule_name)
    target_dir = THEROCK_DIR / submodule_path

    log(f"Moving external source from {temp_dir} to {target_dir}")

    # Remove existing directory if it exists
    if target_dir.exists():
        log(f"Removing existing directory: {target_dir}")
        shutil.rmtree(target_dir)

    # Move the checked-out source to submodule location
    shutil.move(str(temp_dir), str(target_dir))
    log(f"Successfully moved external source to {target_dir}")


def auto_detect_and_pull_dvc():
    """Auto-detect directories with .dvc/ and run dvc pull in them.

    This eliminates the need for explicit DVC project lists and conditionals.
    """
    dvc_missing = shutil.which("dvc") is None
    if dvc_missing:
        if is_windows():
            print("Could not find `dvc` on PATH so large files could not be fetched")
            print("Visit https://dvc.org/doc/install for installation instructions.")
            sys.exit(1)
        else:
            print("`dvc` not found, skipping large file pull on Linux.")
            return

    log("Auto-detecting directories with DVC...")
    dvc_dirs = []

    # Scan for .dvc directories in submodules
    for item in THEROCK_DIR.iterdir():
        if not item.is_dir():
            continue
        dvc_config = item / ".dvc" / "config"
        if dvc_config.exists():
            dvc_dirs.append(item)

    if not dvc_dirs:
        log("No DVC directories detected")
        return

    log(f"Found {len(dvc_dirs)} directories with DVC: {[d.name for d in dvc_dirs]}")

    for dvc_dir in dvc_dirs:
        log(f"Running dvc pull in {dvc_dir.name}")
        exec(["dvc", "pull"], cwd=dvc_dir)


def remove_smrev_files(args, projects):
    for project in projects:
        submodule_path = get_submodule_path(project)
        project_dir = THEROCK_DIR / submodule_path
        project_revision_file = project_dir.with_name(f".{project_dir.name}.smrev")
        if project_revision_file.exists():
            print(f"Remove stale project revision file: {project_revision_file}")
            project_revision_file.unlink()


def auto_detect_and_apply_patches(args, projects):
    """Auto-detect which repository we're in and apply appropriate patches.

    For external repos (rocm-libraries, rocm-systems), only apply patches
    for that specific repo. For TheRock, apply patches for all projects.
    """
    if not args.apply_patches:
        log("Patch application disabled via --no-apply-patches")
        return

    if not args.patch_tag:
        log("Not patching (no --patch-tag specified)")
        return

    patch_version_dir: Path = PATCHES_DIR / args.patch_tag
    if not patch_version_dir.exists():
        log(f"ERROR: Patch directory {patch_version_dir} does not exist")
        return

    # Determine which repos to patch based on EXTERNAL_SOURCE_CHECKOUT
    external_source_checkout = (
        os.environ.get("EXTERNAL_SOURCE_CHECKOUT", "false").lower() == "true"
    )

    if external_source_checkout:
        # External repo mode: detect which external repo by scanning projects
        # For external repos, we only patch that specific external repo
        external_repos = ["rocm-libraries", "rocm-systems"]
        detected_external_repo = None

        for external_repo in external_repos:
            if external_repo in projects:
                detected_external_repo = external_repo
                break

        if detected_external_repo:
            projects_to_patch = [detected_external_repo]
            log(
                f"External repo mode: Will only apply patches for {detected_external_repo}"
            )
        else:
            projects_to_patch = []
            log("External repo mode: No external repo detected in projects list")
    else:
        # TheRock mode: apply patches for all projects
        projects_to_patch = projects
        log(f"TheRock mode: Will apply patches for all enabled projects")

    for patch_project_dir in patch_version_dir.iterdir():
        if not patch_project_dir.is_dir():
            continue

        log(f"* Processing project patch directory {patch_project_dir.name}:")

        # Check that project patch directory was included
        if not patch_project_dir.name in projects_to_patch:
            log(
                f"* Project patch directory {patch_project_dir.name} not in enabled projects. Skipping."
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
            env={
                "GIT_COMMITTER_DATE": "Thu, 1 Jan 2099 00:00:00 +0000",
            },
        )

        # Since it is in a patched state, make it invisible to changes.
        exec(
            ["git", "update-index", "--skip-worktree", "--", submodule_path],
            cwd=THEROCK_DIR,
        )

        # Generate the .smrev patch state file.
        patches_hash = hashlib.sha1()
        for patch_file in patch_files:
            patch_contents = Path(patch_file).read_bytes()
            patches_hash.update(patch_contents)
        patches_digest = patches_hash.digest().hex()
        project_revision_file.write_text(
            f"{submodule_url}\n{submodule_revision}+PATCHED:{patches_digest}\n"
        )


def run(args):
    # Check for external repository override from environment
    external_source_checkout = (
        os.environ.get("EXTERNAL_SOURCE_CHECKOUT", "false").lower() == "true"
    )
    external_source_temp_path = os.environ.get("EXTERNAL_SOURCE_TEMP_PATH", "")

    projects = get_enabled_projects(args)

    # Determine which submodule (if any) is being overridden
    override_submodule = None

    # Check if external source was pre-checked-out by GitHub Actions (handles fork PRs)
    if external_source_checkout and external_source_temp_path:
        # GitHub Actions checked out the external repo - determine which one
        # by looking at the .git/config in the temp directory
        temp_dir = THEROCK_DIR / external_source_temp_path
        if temp_dir.exists():
            try:
                # Read the remote URL to determine which repo this is
                result = subprocess.run(
                    ["git", "config", "--get", "remote.origin.url"],
                    cwd=str(temp_dir),
                    capture_output=True,
                    text=True,
                    check=True,
                )
                repo_url = result.stdout.strip()
                override_submodule = extract_submodule_name_from_repo_url(repo_url)
                log(
                    f"External source detected from GitHub Actions checkout: {override_submodule}"
                )

                # Move the checked-out source to submodule directory
                move_external_source_to_submodule(
                    external_source_temp_path, override_submodule
                )
            except subprocess.CalledProcessError as e:
                log(
                    f"Warning: Could not determine external repo from temp checkout: {e}"
                )

    # Build list of submodule paths to initialize (excluding override)
    submodule_paths = ALWAYS_SUBMODULE_PATHS + [
        get_submodule_path(project)
        for project in projects
        if project != override_submodule  # Skip submodule init for override repo
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

    # Initialize submodules (excluding any override)
    if args.update_submodules and submodule_paths:
        log(f"Initializing {len(submodule_paths)} submodules (excluding any override)")
        exec(
            ["git", "submodule", "update", "--init"]
            + update_args
            + ["--"]
            + submodule_paths,
            cwd=THEROCK_DIR,
        )

    # Auto-detect and pull DVC for all directories
    auto_detect_and_pull_dvc()

    # Fetch nested submodules
    if args.update_submodules:
        fetch_nested_submodules(args, projects)

    # Because we allow local patches, if a submodule is in a patched state,
    # we manually set it to skip-worktree since recording the commit is
    # then meaningless. Here on each fetch, we reset the flag so that if
    # patches are aged out, the tree is restored to normal.
    all_submodule_paths = [get_submodule_path(name) for name in projects]
    exec(
        ["git", "update-index", "--no-skip-worktree", "--"] + all_submodule_paths,
        cwd=THEROCK_DIR,
    )

    # Remove any stale .smrev files.
    remove_smrev_files(args, projects)

    # Auto-detect and apply patches
    auto_detect_and_apply_patches(args, projects)


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
        default=[("iree", ["third_party/flatcc", "third_party/benchmark"])],
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
        "--include-rocm-media",
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
            "rccl",
            "rccl-tests",
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
        "--rocm-media-projects",
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
                "amd-dbgapi",
                "rocr-debug-agent",
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
