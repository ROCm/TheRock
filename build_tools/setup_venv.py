#!/usr/bin/env python

"""Sets up a Python venv and optionally installs rocm packages into it.

* https://docs.python.org/3/library/venv.html
* https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-and-use-virtual-environments

There are a few modes this can be used in:

* Minimally, this is equivalent to `python -m venv .venv`:

    ```
    python setup_venv.py .venv
    ```

* To install the latest nightly rocm packages for gfx110X-all into the venv:

    ```
    python setup_venv.py .venv --packages rocm[libraries,devel] \
        --index-name nightly --index-subdir gfx110X-all
    ```

    This is roughly equivalent to:

    ```
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install rocm[libraries,devel] --index-url=https://.../gfx110X-all
    deactivate
    ```
"""

import argparse
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import re

from github_actions.github_actions_utils import *

is_windows = platform.system() == "Windows"

ROCM_INDEX_URLS_MAP = {
    "stable": "https://repo.amd.com/rocm/whl/",
    "prerelease": "https://rocm.prereleases.amd.com/whl",
    "nightly": "https://rocm.nightlies.amd.com/v2",
    "dev": "https://rocm.devreleases.amd.com/v2",
}


def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def run_command(args: list[str | Path], cwd: Path = Path.cwd()):
    args = [str(arg) for arg in args]
    log(f"++ Exec [{cwd}]$ {shlex.join(args)}")
    subprocess.check_call(args, cwd=str(cwd), stdin=subprocess.DEVNULL)


def find_venv_python_exe(venv_path: Path) -> Path | None:
    """Finds the python executable under |venv_path|, if one exists."""
    paths = [venv_path / "bin" / "python", venv_path / "Scripts" / "python.exe"]
    for p in paths:
        if p.exists():
            return p
    return None


def create_venv(venv_dir: Path, use_uv: bool = False):
    """Creates a Python venv at |venv_dir|.

    No-op if venv_dir is already an initialized venv (has a python executable).
    """
    log(f"Creating venv at '{venv_dir}'")

    # Log some other variations of the path too.
    try:
        venv_dir_relative = venv_dir.relative_to(Path.cwd())
    except ValueError:
        venv_dir_relative = venv_dir
    venv_dir_resolved = venv_dir.resolve()
    log(f"  Dir relative to CWD: '{venv_dir_relative}'")
    log(f"  Dir fully resolved : '{venv_dir_resolved}'")
    log("")

    # Create with 'python -m venv' as needed.
    python_exe = find_venv_python_exe(venv_dir_resolved)
    if python_exe:
        log(f"  Found existing python executable at '{python_exe}', skipping creation")
        log("  Run again with --clean to clear the existing directory instead")
        return

    if use_uv:
        run_command(["uv", "venv", str(venv_dir_resolved)])
    else:
        run_command([sys.executable, "-m", "venv", str(venv_dir_resolved)])


def update_venv(venv_dir: Path, use_uv: bool = False):
    if use_uv:
        # No updates needed.
        return

    # pip logs warnings about wanting to update, so we'll do that for it.
    log("")
    python_exe = find_venv_python_exe(venv_dir)
    run_command([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])


def activate_venv_in_gha(venv_dir: Path):
    """Activates the venv in venv_dir for future GitHub Actions workflow steps.

    This is a (useful) hack that modifies the PATH and VIRTUAL_ENV env vars
    rather than call the platform-specific 'activate' scripts.
    """

    log("")
    log(f"Activating venv for future GitHub Actions workflow steps")
    gha_warn_if_not_running_on_ci()

    # See https://docs.python.org/3/library/venv.html#how-venvs-work.
    #
    # The usual way to activate a venv is to run the platform-specific command:
    #   POSIX bash         : `source <venv>/bin/activate`
    #   Windows cmd.exe    : `<venv>\Scripts\activate.bat`
    #   Windows powershell : `<venv>\Scripts\Activate.ps1`
    #   etc.
    #
    # What these scripts actually do is a combination of setting environment
    # variables, which we can't normally do (persistently) from a Python script.
    # However, in the context of a GitHub Actions workflow, we *can* set
    # environment variables (and job outputs, and step summaries, etc.) using
    # https://docs.github.com/en/actions/reference/workflow-commands-for-github-actions.

    if is_windows:
        gha_add_to_path(venv_dir / "Scripts")
    else:
        gha_add_to_path(venv_dir / "bin")
    gha_set_env({"VIRTUAL_ENV": venv_dir})


def install_packages_into_venv(
    venv_dir: Path,
    packages: list[str],
    use_uv: bool = False,
    index_url: str | None = None,
    index_name: str | None = None,
    index_subdir: str | None = None,
    find_links: str | None = None,
    pre: bool = False,
    disable_cache: bool = False,
):
    """Installs packages into venv_dir using the provided options.

    Args:
        venv_dir: The venv to install into
        packages: The list of packages to install
        use_uv: True to use 'uv', uses 'pip' otherwise
        index_url: URL for '--index-url' command argument
        index_name: Shorthand for a base index_url (e.g. 'nightly')
        index_subdir: Subdirectory for 'index_url' or 'index_name'
        find_links: URL for '--find-links' command argument
        pre: Allow pre-release packages (pip: --pre, uv: --prerelease=allow)
        disable_cache: Disable package cache (pip: --no-cache-dir, uv: --no-cache)
    """
    log("")

    venv_python_exe = find_venv_python_exe(venv_dir)
    assert venv_python_exe is not None, f"No python executable found in {venv_dir}"
    pip_install_cmd = (
        [str(venv_python_exe), "-m", "pip", "install"]
        if not use_uv
        else ["uv", "pip", "install", "--python", str(venv_python_exe)]
    )

    if index_url and index_name:
        raise ValueError("Can't set both index_url and index_name")

    if index_name:
        # Look up known index name.
        index_url = ROCM_INDEX_URLS_MAP[index_name]

    if index_url:
        # Join index with subdir.
        if index_subdir:
            index_url = f"{index_url.rstrip('/')}/{index_subdir.strip('/')}"

        pip_install_cmd.append(f"--index-url={index_url}")

    if find_links:
        pip_install_cmd.append(f"--find-links={find_links}")

    if pre:
        pip_install_cmd.append("--prerelease=allow" if use_uv else "--pre")

    if disable_cache:
        pip_install_cmd.append("--no-cache" if use_uv else "--no-cache-dir")

    pip_install_cmd.extend(packages)

    run_command(pip_install_cmd)


def log_venv_activate_instructions(venv_dir: Path):
    """Logs platform-specific instructions for activating a venv."""
    log("")
    log(f"Setup complete at '{venv_dir}'! Activate the venv with:")
    if is_windows:
        log(f"  {venv_dir}\\Scripts\\activate.bat")
    else:
        log(f"  source {venv_dir}/bin/activate")


def run(args: argparse.Namespace):
    venv_dir = args.venv_dir
    use_uv = args.use_uv

    if args.clean and venv_dir.exists():
        log(f"Clearing existing venv_dir '{venv_dir}'")
        shutil.rmtree(venv_dir)

    create_venv(venv_dir, use_uv)
    update_venv(venv_dir, use_uv)

    if args.packages:
        install_packages_into_venv(
            venv_dir=venv_dir,
            packages=args.packages.split(),
            use_uv=use_uv,
            index_url=args.index_url,
            index_subdir=args.index_subdir,
            index_name=args.index_name,
            find_links=args.find_links,
            pre=args.pre,
            disable_cache=args.disable_cache,
        )

    if args.activate_in_future_github_actions_steps:
        activate_venv_in_gha(venv_dir)
    else:
        log_venv_activate_instructions(venv_dir)


GFX_TARGET_REGEX = r'(gfx(?:\d{2,3}X|\d{3,4})(?:-[^<"/]*)?)</a>'


def _scrape_rocm_index_subdirs() -> set[str] | None:
    """Scrapes available subdirs from all known indexes, returns union of all."""
    try:
        import requests
    except ImportError:
        return

    all_subdirs: set[str] = set()

    for index_url in ROCM_INDEX_URLS_MAP.values():
        index_url = index_url.rstrip("/") + "/"
        try:
            response = requests.get(index_url)
            response.raise_for_status()
        except Exception as e:
            print(f"[ERROR]: fetching subdirs from {index_url} failed: {e}")
            continue

        # Extract gfx targets from <a> elements.
        matches = re.findall(GFX_TARGET_REGEX, response.text)
        all_subdirs.update(matches)

    return all_subdirs if all_subdirs else None


def main(argv: list[str]):
    p = argparse.ArgumentParser("setup_venv.py")
    p.add_argument(
        "venv_dir",
        type=Path,
        help="Directory in which to create the venv, such as '.venv'",
    )

    general_options = p.add_argument_group("General options")
    general_options.add_argument(
        "--clean",
        action=argparse.BooleanOptionalAction,
        help="If the venv directory already exists, clear it and start fresh",
    )
    general_options.add_argument(
        "--pre",
        action=argparse.BooleanOptionalAction,
        help="Allow installing pre-release packages",
    )
    general_options.add_argument(
        "--disable-cache",
        action=argparse.BooleanOptionalAction,
        help="Disable the pip/uv package cache",
    )
    general_options.add_argument(
        "--activate-in-future-github-actions-steps",
        action=argparse.BooleanOptionalAction,
        help="Attempts to activate the venv persistently when running in a GitHub Action. This is less reliable than running the official activate command",
    )
    general_options.add_argument(
        "--use-uv",
        action=argparse.BooleanOptionalAction,
        help="Uses uv instead of pip/venv, see more at: https://docs.astral.sh/uv/",
    )

    install_options = p.add_argument_group("Install options")

    # TODO(#1036): Other flags or helper scripts to help map between versions,
    #              git commits/refs, workflow runs, etc.
    #              I'd like a shorthand for "install packages from commit abcde"
    #              Maybe use find_artifacts_for_commit.py
    install_options.add_argument(
        "--packages",
        type=str,
        help="Packages to install, including any extras or explicit versions (e.g. 'rocm[libraries,devel]==1.0')",
    )
    # TODO(#1036): add "auto" mode here that infers the index from the version?
    install_options.add_argument(
        "--index-url",
        type=str,
        help="Package index URL for pip --index-url (complete URL, or base URL with --index-subdir)",
    )
    install_options.add_argument(
        "--index-name",
        type=str,
        choices=["stable", "prerelease", "nightly", "dev"],
        help="Shorthand for a named index (requires --index-subdir)",
    )
    install_options.add_argument(
        "--find-links",
        type=str,
        help="Package location URL for pip --find-links (compatible with --index-url)",
    )

    # Scrape available subdirs for --index-subdir choices.
    available_subdirs = _scrape_rocm_index_subdirs()
    install_options.add_argument(
        "--index-subdir",
        "--index-subdirectory",
        type=str,
        help="Index subdirectory, such as 'gfx110X-all'",
        choices=available_subdirs,
    )

    args = p.parse_args(argv)

    if args.venv_dir.exists() and not args.venv_dir.is_dir():
        p.error(f"venv_dir '{args.venv_dir}' exists and is not a directory")
    if args.index_name and not args.index_subdir:
        p.error("--index-subdir must be set when using --index-name")

    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
