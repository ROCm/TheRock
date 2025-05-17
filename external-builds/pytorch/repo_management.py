from pathlib import Path, PurePosixPath
import shlex
import shutil
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
    appropriate for our case, since we make arbitrary changes and patches to them.
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
    print("repo_path: " + str(repo_path) + ", patches_path: " + str(patches_path))
    if not patch_files:
        return
    patch_files.sort(key=lambda p: p.name)
    exec(
        ["git", "am", "--whitespace=nowarn", "--committer-date-is-author-date"]
        + patch_files,
        cwd=repo_path,
    )


def apply_all_patches(root_repo_path: Path, patches_path: Path, patchset_name: str):
    repo_name = root_repo_path.name
    relative_sm_paths = list_submodules(root_repo_path, relative=True)
    # Apply base patches.
    apply_repo_patches(root_repo_path, patches_path / repo_name / patchset_name)
    for relative_sm_path in relative_sm_paths:
        apply_repo_patches(
            root_repo_path / relative_sm_path,
            patches_path / relative_sm_path / patchset_name,
        )


def repo_ref_to_patches_dir_name(version_ref: str) -> str:
    """Converts a git ref like `2.7.0-rc9` to a folder name like `2.7.0`."""
    pos = version_ref.find("-")
    if pos != -1:
        return version_ref[:pos]
    return version_ref


def do_checkout(
    src_dir: Path,
    patch_dir: Path,
    remote_url: str,
    ref: str,
    depth: bool,
    jobs: int,
    patch: bool,
    hipify: bool,
):
    check_git_dir = src_dir / ".git"
    if check_git_dir.exists():
        print(f"Not cloning repository ({check_git_dir} exists)")
    else:
        print(f"Cloning {remote_url} repository at {ref}")
        src_dir.mkdir(parents=True, exist_ok=True)
        exec(["git", "init", "--initial-branch=main"], cwd=src_dir)
        exec(["git", "config", "advice.detachedHead", "false"], cwd=src_dir)
        exec(["git", "remote", "add", "origin", remote_url], cwd=src_dir)

    # Fetch and checkout.
    fetch_args = []
    if depth is not None:
        fetch_args.extend(["--depth", str(depth)])
    if jobs:
        fetch_args.extend(["-j", str(jobs)])
    exec(["git", "fetch"] + fetch_args + ["origin", ref], cwd=src_dir)
    exec(["git", "checkout", "FETCH_HEAD"], cwd=src_dir)
    exec(["git", "tag", "-f", TAG_UPSTREAM_DIFFBASE, "-m", '""'], cwd=src_dir)
    try:
        exec(
            ["git", "submodule", "update", "--init", "--recursive"] + fetch_args,
            cwd=src_dir,
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
            f'git tag -f {TAG_UPSTREAM_DIFFBASE} -m ""',
        ],
        cwd=src_dir,
        stdout_devnull=True,
    )
    git_config_ignore_submodules(src_dir)

    # Base patches.
    if patch:
        apply_all_patches(
            src_dir,
            patch_dir / repo_ref_to_patches_dir_name(ref),
            "base",
        )

    # Hipify.
    if hipify:
        do_hipify(src_dir=src_dir)

    # Hipified patches.
    if patch:
        apply_all_patches(
            src_dir,
            patch_dir / repo_ref_to_patches_dir_name(ref),
            "hipified",
        )


def do_hipify(src_dir: Path):
    print(f"Hipifying {src_dir}")
    build_amd_path = src_dir / "tools" / "amd_build" / "build_amd.py"
    if os.path.exists(build_amd_path):
        exec([sys.executable, build_amd_path], cwd=src_dir)
    # Iterate over the base repository and all submodules. Because we process
    # the root repo first, it will not add submodule changes.
    all_paths = get_all_repositories(src_dir)
    for module_path in all_paths:
        status = list_status(module_path)
        if not status:
            continue
        print(f"HIPIFY made changes to {module_path}: Committing")
        exec(["git", "add", "-A"], cwd=module_path)
        exec(["git", "commit", "-m", HIPIFY_COMMIT_MESSAGE], cwd=module_path)
        exec(["git", "tag", "-f", TAG_HIPIFY_DIFFBASE, "-m", '""'], cwd=module_path)


def do_save_patches(
    src_dir: Path,
    patch_dir: Path,
    ref: str,
):
    repo_name = src_dir.name
    patches_dir = patch_dir / repo_ref_to_patches_dir_name(ref)
    save_repo_patches(src_dir, patches_dir / repo_name)
    relative_sm_paths = list_submodules(src_dir, relative=True)
    for relative_sm_path in relative_sm_paths:
        save_repo_patches(src_dir / relative_sm_path, patches_dir / relative_sm_path)
