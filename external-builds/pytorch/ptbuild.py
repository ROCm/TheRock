#!/usr/bin/env python
"""Checks out and builds PyTorch against a built from source ROCM SDK.

There is nothing that this script does which you couldn't do by hand, but because of
the following, getting PyTorch sources ready to build with ToT TheRock built SDKs
consists of multiple steps:

* Sources must be pre-processed with HIPIFY, creating dirty git trees that are hard
  to develop on further.
* Both the ROCM SDK and PyTorch are moving targets that are eventually consistent.
  We maintain patches for recent PyTorch revisions to adapt to packaging and library
  compatibility differences until all releases are done and available.

Primary usage:

    ./ptbuild.py checkout
    ./ptbuild.py develop

The checkout process combines the following activities:

* Clones the pytorch repository into `src/` with a requested `--pytorch-ref`
  tag (default to latest release).
* Configures PyTorch submodules to be ignored for any local changes (so that
  the result is suitable for development with local patches).
* Applies "base" patches to the pytorch repo and any submodules (by using
  `git am` with patches from `patches/<pytorch-ref>/<repo-name>/base`).
* Runs `hipify` to prepare sources for AMD GPU and commits the result to the
  main repo and any modified submodules.
* Applies "hipified" patches to the pytorch repo and any submodules (by using
  `git am` with patches from `patches/<pytorch-ref>/<repo-name>/hipified`).
* Records some tag information for subsequent activities.

For one-shot builds and CI use, the above is sufficient. But this tool can also
be used to develop. Any commits made to PyTorch or any of its submodules can
be saved locally in TheRock by running `./pybuild.py save-patches`. If checked
in, CI runs for that revision will incorporate them the same as anyone
interactively using this tool.
"""

import argparse
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys

THIS_DIR = Path(__file__).resolve().parent
PATCHES_DIR = THIS_DIR / "patches"
REPO_ROOT = THIS_DIR.parent.parent
FILESET_TOOL = REPO_ROOT / "build_tools" / "fileset_tool.py"
TAG_UPSTREAM_DIFFBASE = "THEROCK_UPSTREAM_DIFFBASE"
TAG_HIPIFY_DIFFBASE = "THEROCK_HIPIFY_DIFFBASE"
HIPIFY_COMMIT_MESSAGE = "DO NOT SUBMIT: HIPIFY"


def exec(args: list[str | Path], cwd: Path, *, stdout_devnull: bool = False):
    args = [str(arg) for arg in args]
    print(f"++ Exec [{cwd}]$ {shlex.join(args)}")
    subprocess.check_call(
        args,
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL if stdout_devnull else None,
    )


def rev_parse(repo_path: Path, rev: str) -> str | None:
    """Parses a revision to a commit hash, returning None if not found."""
    try:
        raw_output = subprocess.check_output(
            ["git", "rev-parse", rev], cwd=str(repo_path), stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        return None
    return raw_output.decode().strip()


def rev_list(repo_path: Path, revlist: str) -> list[str]:
    raw_output = subprocess.check_output(
        ["git", "rev-list", revlist], cwd=str(repo_path)
    )
    return raw_output.decode().splitlines()


def list_submodules(
    repo_path: Path, *, relative: bool = False, recursive: bool = True
) -> list[Path]:
    """Gets paths of all submodules (recursively) in the repository."""
    recursive_args = ["--recursive"] if recursive else []
    raw_output = subprocess.check_output(
        ["git", "submodule", "status"] + recursive_args,
        cwd=str(repo_path),
    )
    lines = raw_output.decode().splitlines()
    relative_paths = [PurePosixPath(line.strip().split()[1]) for line in lines]
    if relative:
        return relative_paths
    return [repo_path / p for p in relative_paths]


def list_status(repo_path: Path) -> list[tuple[str, str]]:
    """Gets the status as a list of (status_type, relative_path)."""
    raw_output = subprocess.check_output(
        ["git", "status", "--porcelain", "-u", "--ignore-submodules"],
        cwd=str(repo_path),
    )
    lines = raw_output.decode().splitlines()
    return [tuple(line.strip().split()) for line in lines]


def get_all_repositories(root_path: Path) -> list[Path]:
    """Gets all repository paths, starting with the root and then including all
    recursive submodules."""
    all_paths = list_submodules(root_path)
    all_paths.insert(0, root_path)
    return all_paths


def git_config_ignore_submodules(repo_path: Path):
    """Sets the `submodule.<name>.ignore = true` git config option for all submodules.

    This causes all submodules to not show up in status or diff reports, which is
    appropriate for our case, since we make arbitrary changes and patches to them.
    Note that pytorch seems to somewhat arbitrarily have some already set this way.
    We just set them all.
    """
    config_names = (
        subprocess.check_output(
            [
                "git",
                "config",
                "--file",
                ".gitmodules",
                "--name-only",
                "--get-regexp",
                "\\.path$",
            ],
            cwd=str(repo_path),
        )
        .decode()
        .splitlines()
    )
    for config_name in config_names:
        ignore_name = config_name.removesuffix(".path") + ".ignore"
        exec(["git", "config", ignore_name, "all"], cwd=repo_path)
    submodule_paths = list_submodules(repo_path, relative=True, recursive=False)
    exec(["git", "update-index", "--skip-worktree"] + submodule_paths, cwd=repo_path)


def save_repo_patches(repo_path: Path, patches_path: Path):
    """Updates the patches directory with any patches committed to the repository."""
    if patches_path.exists():
        shutil.rmtree(patches_path)
    # Get key revisions.
    upstream_rev = rev_parse(repo_path, TAG_UPSTREAM_DIFFBASE)
    hipify_rev = rev_parse(repo_path, TAG_HIPIFY_DIFFBASE)
    if upstream_rev is None:
        print(f"error: Could not find upstream diffbase tag {TAG_UPSTREAM_DIFFBASE}")
        sys.exit(1)
    hipified_count = 0
    if hipify_rev:
        hipified_revlist = f"{hipify_rev}..HEAD"
        base_revlist = f"{upstream_rev}..{hipify_rev}^"
        hipified_count = len(rev_list(repo_path, hipified_revlist))
    else:
        hipified_revlist = None
        base_revlist = f"{upstream_rev}..HEAD"
    base_count = len(rev_list(repo_path, base_revlist))
    if hipified_count == 0 and base_count == 0:
        return
    print(
        f"Saving {patches_path} patches: {base_count} base, {hipified_count} hipified"
    )
    if base_count > 0:
        base_path = patches_path / "base"
        base_path.mkdir(parents=True, exist_ok=True)
        exec(["git", "format-patch", "-o", base_path, base_revlist], cwd=repo_path)
    if hipified_count > 0:
        hipified_path = patches_path / "hipified"
        hipified_path.mkdir(parents=True, exist_ok=True)
        exec(
            ["git", "format-patch", "-o", hipified_path, hipified_revlist],
            cwd=repo_path,
        )


def apply_repo_patches(repo_path: Path, patches_path: Path):
    """Applies patches to a repository from the given patches directory."""
    patch_files = list(patches_path.glob("*.patch"))
    if not patch_files:
        return
    patch_files.sort(key=lambda p: p.name)
    exec(
        ["git", "am", "--whitespace=nowarn", "--committer-date-is-author-date"]
        + patch_files,
        cwd=repo_path,
    )


def apply_all_patches(root_repo_path: Path, patches_path: Path, patchset_name: str):
    relative_sm_paths = list_submodules(root_repo_path, relative=True)
    # Apply base patches.
    apply_repo_patches(root_repo_path, patches_path / "pytorch" / patchset_name)
    for relative_sm_path in relative_sm_paths:
        apply_repo_patches(
            root_repo_path / relative_sm_path,
            patches_path / relative_sm_path / patchset_name,
        )


def do_checkout(args: argparse.Namespace):
    repo_dir: Path = args.repo
    check_git_dir = repo_dir / ".git"
    if check_git_dir.exists():
        print(f"Not cloning repository ({check_git_dir} exists)")
    else:
        print(f"Cloning repository at {args.pytorch_ref}")
        repo_dir.mkdir(parents=True, exist_ok=True)
        exec(["git", "init", "--initial-branch=main"], cwd=repo_dir)
        exec(["git", "config", "advice.detachedHead", "false"], cwd=repo_dir)
        exec(["git", "remote", "add", "origin", args.pytorch_origin], cwd=repo_dir)

    # Fetch and checkout.
    fetch_args = []
    if args.depth is not None:
        fetch_args.extend(["--depth", str(args.depth)])
    if args.jobs:
        fetch_args.extend(["-j", str(args.jobs)])
    exec(["git", "fetch"] + fetch_args + ["origin", args.pytorch_ref], cwd=repo_dir)
    exec(["git", "checkout", "FETCH_HEAD"], cwd=repo_dir)
    exec(["git", "tag", "-f", TAG_UPSTREAM_DIFFBASE], cwd=repo_dir)
    exec(
        ["git", "submodule", "update", "--init", "--recursive"] + fetch_args,
        cwd=repo_dir,
    )
    exec(
        [
            "git",
            "submodule",
            "foreach",
            "--recursive",
            f"git tag -f {TAG_UPSTREAM_DIFFBASE}",
        ],
        cwd=repo_dir,
        stdout_devnull=True,
    )
    git_config_ignore_submodules(repo_dir)

    # Base patches.
    if args.patch:
        apply_all_patches(repo_dir, PATCHES_DIR / args.pytorch_ref, "base")

    # Hipify.
    if args.hipify:
        do_hipify(args)

    # Hipified patches.
    if args.patch:
        apply_all_patches(repo_dir, PATCHES_DIR / args.pytorch_ref, "hipified")


def do_hipify(args: argparse.Namespace):
    repo_dir: Path = args.repo
    print(f"Hipifying {repo_dir}")
    build_amd_path = repo_dir / "tools" / "amd_build" / "build_amd.py"
    exec([sys.executable, build_amd_path], cwd=repo_dir)
    # Iterate over the base repository and all submodules. Because we process
    # the root repo first, it will not add submodule changes.
    all_paths = get_all_repositories(repo_dir)
    for module_path in all_paths:
        status = list_status(module_path)
        if not status:
            continue
        print(f"HIPIFY made changes to {module_path}: Committing")
        exec(["git", "add", "-A"], cwd=module_path)
        exec(["git", "commit", "-m", HIPIFY_COMMIT_MESSAGE], cwd=module_path)
        exec(["git", "tag", "-f", TAG_HIPIFY_DIFFBASE], cwd=module_path)


def do_save_patches(args: argparse.Namespace):
    patches_dir = PATCHES_DIR / args.pytorch_ref
    save_repo_patches(args.repo, patches_dir / "pytorch")
    relative_sm_paths = list_submodules(args.repo, relative=True)
    for relative_sm_path in relative_sm_paths:
        save_repo_patches(args.repo / relative_sm_path, patches_dir / relative_sm_path)


def main(cl_args: list[str]):
    def add_common(command_parser: argparse.ArgumentParser):
        command_parser.add_argument(
            "--repo",
            type=Path,
            default=THIS_DIR / "src",
            help="PyTorch repository path",
        )

    p = argparse.ArgumentParser("ptbuild.py")
    default_tag = "v2.6.0"
    sub_p = p.add_subparsers(required=True)
    checkout_p = sub_p.add_parser("checkout", help="Clone PyTorch locally and checkout")
    add_common(checkout_p)
    checkout_p.add_argument(
        "--pytorch-origin",
        default="https://github.com/pytorch/pytorch.git",
        help="PyTorch git origin",
    )
    checkout_p.add_argument(
        "--pytorch-ref", default=default_tag, help="PyTorch ref/tag to checkout"
    )
    checkout_p.add_argument("--depth", type=int, help="Fetch depth")
    checkout_p.add_argument("--jobs", type=int, help="Number of fetch jobs")
    checkout_p.add_argument(
        "--hipify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run hipify",
    )
    checkout_p.add_argument(
        "--patch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply patches for the pytorch-ref",
    )
    checkout_p.set_defaults(func=do_checkout)

    hipify_p = sub_p.add_parser("hipify", help="Run HIPIFY on the project")
    add_common(hipify_p)
    hipify_p.set_defaults(func=do_hipify)

    save_patches_p = sub_p.add_parser(
        "save-patches", help="Save local commits as patch files for later application"
    )
    add_common(save_patches_p)
    save_patches_p.add_argument(
        "--pytorch-ref", default=default_tag, help="PyTorch ref/tag to checkout"
    )
    save_patches_p.set_defaults(func=do_save_patches)

    args = p.parse_args(cl_args)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
