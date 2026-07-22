# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys
import os

TAG_UPSTREAM_DIFFBASE = "THEROCK_UPSTREAM_DIFFBASE"
TAG_HIPIFY_DIFFBASE = "THEROCK_HIPIFY_DIFFBASE"
HIPIFY_COMMIT_MESSAGE = "DO NOT SUBMIT: HIPIFY"


def add_checkout_options(
    command_parser: argparse.ArgumentParser, *, default_hipify: bool
) -> None:
    command_parser.add_argument("--depth", type=int, help="Fetch depth")
    command_parser.add_argument("--jobs", type=int, help="Number of fetch jobs")
    command_parser.add_argument(
        "--submodules",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Checkout git submodules",
    )
    command_parser.add_argument(
        "--hipify",
        action=argparse.BooleanOptionalAction,
        default=default_hipify,
        help="Run hipify",
    )
    command_parser.add_argument(
        "--commit-hipify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Commit HIPIFY changes after running hipify",
    )


def run_command(args: list[str | Path], cwd: Path, *, stdout_devnull: bool = False):
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
    appropriate for our case, since we make arbitrary changes to them with hipify.
    Note that pytorch seems to somewhat arbitrarily have some already set this way.
    We just set them all.
    """
    file_path = repo_path / ".gitmodules"
    if os.path.exists(file_path):
        try:
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
                run_command(["git", "config", ignore_name, "all"], cwd=repo_path)
            submodule_paths = list_submodules(repo_path, relative=True, recursive=False)
            run_command(
                ["git", "update-index", "--skip-worktree"] + submodule_paths,
                cwd=repo_path,
            )
        except Exception as e:
            # pytorch audio has empty .gitmodules file which can cause exception
            pass


def save_repo_patches(repo_path: Path, patches_path: Path):
    """Updates the patches directory with any patches committed to the repository.

    Splits commits into a `base` set (upstream..hipify^) and a `hipified` set
    (hipify..HEAD) using the diffbase tags recorded during checkout, mirroring
    the `uccl` and `fetch_sources` conventions.
    """
    if patches_path.exists():
        shutil.rmtree(patches_path)
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
        run_command(
            ["git", "format-patch", "-o", base_path, base_revlist], cwd=repo_path
        )
    if hipified_count > 0:
        hipified_path = patches_path / "hipified"
        hipified_path.mkdir(parents=True, exist_ok=True)
        run_command(
            ["git", "format-patch", "-o", hipified_path, hipified_revlist],
            cwd=repo_path,
        )


def apply_repo_patches(repo_path: Path, patches_path: Path):
    """Applies patches to a single repository from the given patches directory.

    Uses `git am` with the same flags as `fetch_sources.py`/`uccl`. A missing or
    empty directory is a no-op.
    """
    patch_files = list(patches_path.glob("*.patch"))
    if not patch_files:
        return
    patch_files.sort(key=lambda p: p.name)
    print(f"Applying {len(patch_files)} patches to {repo_path}")
    run_command(
        [
            "git",
            "am",
            "--whitespace=nowarn",
            "--committer-date-is-author-date",
            "--no-gpg-sign",
        ]
        + patch_files,
        cwd=repo_path,
    )


def apply_all_patches(
    root_repo_path: Path, patches_path: Path, repo_name: str, patchset_name: str
):
    """Applies a patchset to the root repo and every submodule.

    Patches are laid out as
    `patches_path/<repo_name-or-submodule-relpath>/<patchset_name>/*.patch`.
    Missing directories are a no-op, so only the refs/submodules that actually
    need patches have to provide them (e.g. `nightly`, whose gloo is already
    native, matches no directory and is left untouched).
    """
    relative_sm_paths = list_submodules(root_repo_path, relative=True)
    # Root repo patches.
    apply_repo_patches(root_repo_path, patches_path / repo_name / patchset_name)
    # Submodule patches.
    for relative_sm_path in relative_sm_paths:
        apply_repo_patches(
            root_repo_path / relative_sm_path,
            patches_path / relative_sm_path / patchset_name,
        )


# repo_hashtag_to_patches_dir_name('release/2.12') -> '2.12'
# repo_hashtag_to_patches_dir_name('2.7.0-rc9') -> '2.7.0'
def repo_hashtag_to_patches_dir_name(version_ref: str) -> str:
    name = version_ref.removeprefix("release/")
    pos = name.find("-")
    if pos != -1:
        return name[:pos]
    return name


def get_patches_dir_name(args: argparse.Namespace) -> str | None:
    """Resolves the patchset directory name from --patchset or the repo hashtag."""
    patchset_name = getattr(args, "patchset", None)
    if patchset_name is not None:
        return patchset_name
    hashtag = getattr(args, "repo_hashtag", None)
    if hashtag is not None:
        return repo_hashtag_to_patches_dir_name(hashtag)
    return None


def do_hipify(args: argparse.Namespace):
    repo_dir: Path = args.checkout_dir
    print(f"Hipifying {repo_dir}")
    build_amd_path = repo_dir / "tools" / "amd_build" / "build_amd.py"
    if build_amd_path.exists():
        run_command([sys.executable, build_amd_path], cwd=repo_dir)


def tag_hipify_diffbase(module_path: Path):
    """Apply the HIPIFY diffbase tag to the module."""
    try:
        run_command(
            ["git", "tag", "-f", TAG_HIPIFY_DIFFBASE, "--no-sign"],
            cwd=module_path,
        )
    except subprocess.CalledProcessError as e:
        print(f"Failed to apply tag {TAG_HIPIFY_DIFFBASE} in {module_path}: {e}")
        raise


def commit_hipify_module(module_path: Path):
    """Handle HIPIFY commit for a single module."""
    status = list_status(module_path)
    if not status:
        return

    print(f"HIPIFY made changes to {module_path}: Committing")
    run_command(["git", "add", "-A"], cwd=module_path)

    # Check if there are staged changes
    try:
        staged_result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(module_path),
            capture_output=True,
            stdin=subprocess.DEVNULL,
        )
        if staged_result.returncode == 0:
            print(f"No staged changes in {module_path} after git add: Skipping commit")
            tag_hipify_diffbase(module_path)
            return
    except subprocess.CalledProcessError as e:
        print(f"Error checking staged changes in {module_path}: {e}")
        return

    # Attempt to commit changes
    try:
        run_command(
            ["git", "commit", "-m", HIPIFY_COMMIT_MESSAGE, "--no-gpg-sign"],
            cwd=module_path,
        )
    except subprocess.CalledProcessError as e:
        if e.returncode == 1:
            print(f"No changes to commit in {module_path} (git returned exit code 1)")
        else:
            print(f"Commit failed in {module_path}: {e}")
            raise

    tag_hipify_diffbase(module_path)


def commit_hipify(args: argparse.Namespace):
    repo_dir: Path = args.checkout_dir
    all_paths = get_all_repositories(repo_dir) if args.submodules else [repo_dir]
    for module_path in all_paths:
        commit_hipify_module(module_path)


def do_checkout(args: argparse.Namespace, custom_hipify=do_hipify):
    repo_dir: Path = args.checkout_dir
    check_git_dir = repo_dir / ".git"
    if check_git_dir.exists():
        print(f"Not cloning repository ({check_git_dir} exists)")
        run_command(
            ["git", "remote", "set-url", "origin", args.gitrepo_origin], cwd=repo_dir
        )
    else:
        print(f"Cloning repository at {args.repo_hashtag}")
        repo_dir.mkdir(parents=True, exist_ok=True)
        run_command(["git", "init", "--initial-branch=main"], cwd=repo_dir)
        run_command(["git", "config", "advice.detachedHead", "false"], cwd=repo_dir)
        run_command(
            ["git", "remote", "add", "origin", args.gitrepo_origin], cwd=repo_dir
        )

    # Fetch and checkout.
    fetch_args = []
    if args.depth is not None:
        fetch_args.extend(["--depth", str(args.depth)])
    if args.jobs:
        fetch_args.extend(["-j", str(args.jobs)])
    if not args.submodules:
        fetch_args.extend(["--no-recurse-submodules"])
    run_command(
        ["git", "fetch"] + fetch_args + ["origin", args.repo_hashtag], cwd=repo_dir
    )
    run_command(["git", "checkout", "FETCH_HEAD"], cwd=repo_dir)
    run_command(["git", "tag", "-f", TAG_UPSTREAM_DIFFBASE, "--no-sign"], cwd=repo_dir)
    if args.submodules:
        try:
            run_command(
                ["git", "submodule", "update", "--init", "--recursive"] + fetch_args,
                cwd=repo_dir,
            )
        except subprocess.CalledProcessError:
            print("Failed to fetch git submodules")
            sys.exit(1)
        run_command(
            [
                "git",
                "submodule",
                "foreach",
                "--recursive",
                f"git tag -f {TAG_UPSTREAM_DIFFBASE} --no-sign",
            ],
            cwd=repo_dir,
            stdout_devnull=True,
        )
        git_config_ignore_submodules(repo_dir)

    # Resolve carried-patch parameters. Only opted-in checkouts (currently the
    # torch repo) define these; siblings (audio/vision/apex/triton) leave them
    # unset and are treated as no-ops via getattr defaults.
    patch_enabled = bool(getattr(args, "patch", False))
    patch_dir: Path | None = getattr(args, "patch_dir", None)
    patchset_name = get_patches_dir_name(args) if patch_enabled else None
    repo_name = str(getattr(args, "repo_name", repo_dir.name))
    apply_patches = patch_enabled and patch_dir is not None and bool(patchset_name)

    # Base patches: applied after `submodule update` and before hipify so the
    # hipify commit captures the patched sources. Requires submodules.
    if apply_patches and args.submodules:
        apply_all_patches(repo_dir, patch_dir / patchset_name, repo_name, "base")

    # Hipify.
    if args.hipify:
        custom_hipify(args)
        if args.commit_hipify:
            commit_hipify(args)

    # Hipified patches: applied after hipify. Requires submodules.
    if apply_patches and args.hipify and args.submodules:
        apply_all_patches(repo_dir, patch_dir / patchset_name, repo_name, "hipified")


def do_save_patches(args: argparse.Namespace):
    """Saves local commits (base + hipified) of the repo and its submodules as
    patch files under `<patch_dir>/<patchset>/...`, mirroring `uccl`."""
    repo_dir: Path = args.checkout_dir
    repo_name = str(getattr(args, "repo_name", repo_dir.name))
    patch_dir: Path = args.patch_dir
    patchset_name = get_patches_dir_name(args)
    if not patchset_name:
        print("error: Could not determine patchset name (pass --patchset)")
        sys.exit(1)
    patches_dir = patch_dir / patchset_name
    save_repo_patches(repo_dir, patches_dir / repo_name)
    for relative_sm_path in list_submodules(repo_dir, relative=True):
        save_repo_patches(repo_dir / relative_sm_path, patches_dir / relative_sm_path)


# Reads the ROCm maintained "related_commits" file from the given pytorch dir.
# If present, selects the given os and project, returning origin and hashtag.
# Otherwise, returns the given defaults.
def read_pytorch_rocm_pins(
    pytorch_dir: Path,
    os: str,
    project: str,
    *,
    default_origin: str,
    default_hashtag: str | None,
) -> tuple[str, str | None, bool]:
    related_commits_file = pytorch_dir / "related_commits"
    if related_commits_file.exists():
        lines = related_commits_file.read_text().splitlines()
        for line in lines:
            try:
                (
                    rec_os,
                    rec_source,
                    rec_project,
                    rec_branch,
                    rec_commit,
                    rec_origin,
                ) = line.split("|")
            except ValueError:
                print(f"WARNING: Could not parse related_commits line: {line}")
            if rec_os == os and rec_project == project:
                return rec_origin, rec_commit, True

    # Not found.
    return default_origin, default_hashtag, False
