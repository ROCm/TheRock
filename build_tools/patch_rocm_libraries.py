#!/usr/bin/env python
"""Applies patches to a monorepo.
This script is available for users, but it is primarily the mechanism
the CI uses to apply patches to a monorepo.

A monorepo like `rocm-libraries` can be fetch with the `fetch_repo.py`
script, whereas patches carried in TheRock would be applied with this
script.

Example usage:

    # Apply all patches to `rocm-libraries/projects`:
    python patch_monorepo.py --repo /tmp/rocm-libraries

    # Apply patches to `rocm-libraries/projects/rocblas`:
    python patch_monorepo.py --repo /tmp/rocm-libraries --projects rocBLAS

    # Apply patches to `rocm-libraries/projects/{rocblas,rocthrust}`
    python patch_monorepo.py --repo /tmp/rocm-libraries --projects rocBLAS rocThrust

"""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

THIS_SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = THIS_SCRIPT_DIR.parent
PATCHES_DIR = THEROCK_DIR / "patches"


def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def exec(args: list[str | Path], cwd: Path):
    args = [str(arg) for arg in args]
    log(f"++ Exec [{cwd}]$ {shlex.join(args)}")
    sys.stdout.flush()
    subprocess.check_call(args, cwd=str(cwd), stdin=subprocess.DEVNULL)


def get_monorepo_path(repo: Path, category: str, name: str) -> Path:
    relpath = repo / category / Path(name.lower())
    return relpath


def run(args):
    projects = list(args.projects)
    shared = list()
    if args.include_shared:
        shared = list(args.shared)

    # TODO: This is take over from `fetch_sources` and likley only applies
    #   to submodules. Re-evaluate here if needed.
    # Because we allow local patches, if a submodule is in a patched state,
    # we manually set it to skip-worktree since recording the commit is
    # then meaningless. Here on each fetch, we reset the flag so that if
    # patches are aged out, the tree is restored to normal.
    exec(
        ["git", "update-index", "--skip-worktree"],
        cwd=args.repo,
    )

    patch_version_dir: Path = PATCHES_DIR / args.patch_tag
    if not patch_version_dir.exists():
        log(f"ERROR: Patch directory {patch_version_dir} does not exist")
    for patch_project_dir in patch_version_dir.iterdir():
        log(f"* Processing project patch directory {patch_project_dir}:")
        # Check that project patch directory was included and set the category
        project_to_patch = patch_project_dir.name
        if project_to_patch in projects:
            category = "projects"
        elif project_to_patch in shared:
            category = "shared"
        else:
            log(
                f"* Project patch directory {patch_project_dir.name} was not included. Skipping."
            )
            continue
        project_path = get_monorepo_path(args.repo, category, project_to_patch)
        patch_files = list(patch_project_dir.glob("*.patch"))
        patch_files.sort()
        log(f"Applying {len(patch_files)} patches to {project_to_patch}")
        apply_directory = str(project_path.relative_to(args.repo))
        exec(
            [
                "git",
                "-c",
                "user.name=therockbot",
                "-c",
                "user.email=therockbot@amd.com",
                "am",
                "--whitespace=nowarn",
                "--directory",
                f"{apply_directory}",
            ]
            + patch_files,
            cwd=args.repo,
        )

    # TODO: This is take over from `fetch_sources` and likley only applies
    #   to submodules. Re-evaluate here if needed.
    # Since it is in a patched state, make it invisible to changes.
    exec(
        ["git", "update-index", "--skip-worktree"],
        cwd=args.repo,
    )


def main(argv):
    parser = argparse.ArgumentParser(prog="patch_monorepo")

    parser.add_argument(
        "--patch-tag",
        type=str,
        default="amd-mainline",
        help="Patch tag to apply to sources after sync",
    )
    parser.add_argument(
        "--projects",
        nargs="+",
        type=str,
        default=[
            "hipSOLVER",
            "rocSOLVER",
            "composable_kernel",
        ],
    )
    parser.add_argument(
        "--shared",
        nargs="+",
        type=str,
        default=[
            "Tensile",
        ],
    )
    parser.add_argument(
        "--include-shared",
        # TODO: Set to True once the patchset applies to Tensile
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Include shared projects",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        required=True,
        help="Path to the monorepo",
    )
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
