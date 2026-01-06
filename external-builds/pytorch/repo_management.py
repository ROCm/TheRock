import argparse
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import os

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
                exec(["git", "config", ignore_name, "all"], cwd=repo_path)
            submodule_paths = list_submodules(repo_path, relative=True, recursive=False)
            exec(
                ["git", "update-index", "--skip-worktree"] + submodule_paths,
                cwd=repo_path,
            )
        except Exception as e:
            # pytorch audio has empty .gitmodules file which can cause exception
            pass


def do_hipify(args: argparse.Namespace):
    repo_dir: Path = args.checkout_dir
    print(f"Hipifying {repo_dir}")
    build_amd_path = repo_dir / "tools" / "amd_build" / "build_amd.py"
    if build_amd_path.exists():
        exec([sys.executable, build_amd_path], cwd=repo_dir)


def tag_hipify_diffbase(module_path: Path):
    """Apply the HIPIFY diffbase tag to the module."""
    try:
        exec(
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
    exec(["git", "add", "-A"], cwd=module_path)

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
        exec(
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
    all_paths = get_all_repositories(repo_dir)
    for module_path in all_paths:
        commit_hipify_module(module_path)


def do_checkout(args: argparse.Namespace, custom_hipify=do_hipify):
    repo_dir: Path = args.checkout_dir
    check_git_dir = repo_dir / ".git"
    if check_git_dir.exists():
        print(f"Not cloning repository ({check_git_dir} exists)")
        exec(["git", "remote", "set-url", "origin", args.gitrepo_origin], cwd=repo_dir)
    else:
        print(f"Cloning repository at {args.repo_hashtag}")
        repo_dir.mkdir(parents=True, exist_ok=True)
        exec(["git", "init", "--initial-branch=main"], cwd=repo_dir)
        exec(["git", "config", "advice.detachedHead", "false"], cwd=repo_dir)
        exec(["git", "remote", "add", "origin", args.gitrepo_origin], cwd=repo_dir)

    # Fetch and checkout.
    fetch_args = []
    if args.depth is not None:
        fetch_args.extend(["--depth", str(args.depth)])
    if args.jobs:
        fetch_args.extend(["-j", str(args.jobs)])
    exec(["git", "fetch"] + fetch_args + ["origin", args.repo_hashtag], cwd=repo_dir)
    exec(["git", "checkout", "FETCH_HEAD"], cwd=repo_dir)
    exec(["git", "tag", "-f", TAG_UPSTREAM_DIFFBASE, "--no-sign"], cwd=repo_dir)
    try:
        exec(
            ["git", "submodule", "update", "--init", "--recursive"] + fetch_args,
            cwd=repo_dir,
        )
    except subprocess.CalledProcessError:
        print("Failed to fetch git submodules")
        sys.exit(1)
    exec(
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

    # Hipify.
    if args.hipify:
        custom_hipify(args)
        commit_hipify(args)


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
