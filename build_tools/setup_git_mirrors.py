#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Create and maintain bare git mirror repositories for TheRock submodules.

These mirrors serve as git reference repositories that speed up
`git submodule update --init` by providing a local source of git objects.
When used with `fetch_sources.py --reference-dir`, submodule clones read
objects from the local mirror instead of fetching them over the network.

Usage:
    # Initial setup (creates all mirrors)
    python setup_git_mirrors.py --mirror-dir ~/.rocm-git-mirrors

    # Update existing mirrors (fetch new refs from remotes)
    python setup_git_mirrors.py --mirror-dir ~/.rocm-git-mirrors --update

    # Verify mirror integrity
    python setup_git_mirrors.py --mirror-dir ~/.rocm-git-mirrors --verify

After creating mirrors, use them with fetch_sources.py:
    python fetch_sources.py --reference-dir ~/.rocm-git-mirrors
"""

import argparse
import concurrent.futures
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from _therock_utils.git_mirrors import MIRROR_DIR_ENV, url_to_mirror_relpath

THIS_SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = THIS_SCRIPT_DIR.parent
RETRY_BASE_DELAY_SECONDS = 2
MAX_RETRY_DELAY_SECONDS = 30


def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


@dataclass
class SubmoduleInfo:
    """A git submodule entry parsed from .gitmodules."""

    name: str
    path: str
    url: str
    mirror_path: Path


@dataclass
class MirrorResult:
    """Outcome of a single mirror create/update/verify operation."""

    submodule: SubmoduleInfo
    success: bool
    action: str  # "created", "updated", "skipped", "verified", "failed"
    elapsed_seconds: float
    error: str | None = None


def run_git(
    args: list[str | Path],
    cwd: Path,
    *,
    capture: bool = False,
) -> subprocess.CompletedProcess | None:
    """Run a git command, optionally capturing output."""
    str_args = [str(a) for a in args]
    log(f"++ Exec [{cwd}]$ {shlex.join(str_args)}")
    sys.stdout.flush()
    if capture:
        return subprocess.run(
            str_args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
    subprocess.check_call(str_args, cwd=str(cwd), stdin=subprocess.DEVNULL)
    return None


def discover_submodules(
    mirror_dir: Path,
    gitmodules_path: Path | None = None,
) -> list[SubmoduleInfo]:
    """Parse .gitmodules to discover all submodules and their remote URLs."""
    if gitmodules_path is None:
        gitmodules_path = THEROCK_DIR / ".gitmodules"
    if not gitmodules_path.exists():
        raise FileNotFoundError(f".gitmodules not found at {gitmodules_path}")

    result = subprocess.run(
        [
            "git",
            "config",
            "--file",
            str(gitmodules_path),
            "--get-regexp",
            r"submodule\..*\.url",
        ],
        capture_output=True,
        text=True,
        cwd=str(gitmodules_path.parent),
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to parse .gitmodules: {result.stderr}")

    submodules: list[SubmoduleInfo] = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        # Line format: "submodule.<name>.url <url>"
        key, url = line.split(None, 1)
        name = key.split(".")[1]

        path_result = subprocess.run(
            [
                "git",
                "config",
                "--file",
                str(gitmodules_path),
                "--get",
                f"submodule.{name}.path",
            ],
            capture_output=True,
            text=True,
            cwd=str(gitmodules_path.parent),
        )
        path = path_result.stdout.strip() if path_result.returncode == 0 else name
        mirror_relpath = url_to_mirror_relpath(url)
        submodules.append(
            SubmoduleInfo(
                name=name,
                path=path,
                url=url,
                mirror_path=mirror_dir / mirror_relpath,
            )
        )

    return submodules


def _parse_ref_lines(output: str) -> dict[str, str]:
    """Parse git show-ref or ls-remote output into {ref: sha}.

    Both commands output "<sha> <ref>" (show-ref uses space, ls-remote
    uses tab). We normalize to a dict keyed by ref name, skipping the
    bare HEAD symref that ls-remote includes but show-ref does not.
    """
    refs: dict[str, str] = {}
    for line in output.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        sha, ref = parts[0], parts[1]
        if ref == "HEAD":
            continue
        refs[ref] = sha
    return refs


def needs_update(mirror_path: Path, remote_url: str) -> bool:
    """Check if remote has refs not present in local mirror."""
    try:
        local = subprocess.run(
            ["git", "show-ref"],
            cwd=str(mirror_path),
            capture_output=True,
            text=True,
        )
        remote = subprocess.run(
            ["git", "ls-remote", "--refs", remote_url],
            capture_output=True,
            text=True,
        )
        if remote.returncode != 0:
            log(f"  Could not ls-remote {remote_url}, will update to be safe")
            return True

        local_refs = _parse_ref_lines(local.stdout)
        remote_refs = _parse_ref_lines(remote.stdout)
        return local_refs != remote_refs
    except (subprocess.CalledProcessError, OSError) as e:
        log(f"  Could not compare refs for {remote_url}: {e}")
        return True


def create_mirror(
    submodule: SubmoduleInfo,
    retries: int = 3,
) -> MirrorResult:
    """Create a new bare mirror clone of a submodule's remote."""
    start = time.monotonic()
    mirror_path = submodule.mirror_path
    mirror_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        try:
            log(f"\n=== Creating mirror: {submodule.name} -> {mirror_path}")
            run_git(
                ["git", "clone", "--mirror", submodule.url, str(mirror_path)],
                cwd=mirror_path.parent,
            )
            return MirrorResult(
                submodule=submodule,
                success=True,
                action="created",
                elapsed_seconds=time.monotonic() - start,
            )
        except subprocess.CalledProcessError as e:
            if mirror_path.exists():
                shutil.rmtree(mirror_path, ignore_errors=True)
            if attempt < retries:
                delay = min(
                    RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)),
                    MAX_RETRY_DELAY_SECONDS,
                )
                log(f"  Attempt {attempt}/{retries} failed, retrying in {delay}s...")
                time.sleep(delay)
            else:
                return MirrorResult(
                    submodule=submodule,
                    success=False,
                    action="failed",
                    elapsed_seconds=time.monotonic() - start,
                    error=str(e),
                )

    return MirrorResult(
        submodule=submodule,
        success=False,
        action="failed",
        elapsed_seconds=time.monotonic() - start,
        error="Exhausted retries",
    )


def update_mirror(
    submodule: SubmoduleInfo,
    retries: int = 3,
    *,
    skip_up_to_date: bool = True,
    force: bool = False,
) -> MirrorResult:
    """Update an existing bare mirror or create it if missing.

    Args:
        submodule: Submodule information including mirror path and URL
        retries: Number of retry attempts for network operations
        skip_up_to_date: Skip update if mirror refs match remote (ignored if force=True)
        force: Force update even if mirror appears up-to-date
    """
    start = time.monotonic()

    if not submodule.mirror_path.exists():
        return create_mirror(submodule, retries=retries)

    if (
        not force
        and skip_up_to_date
        and not needs_update(submodule.mirror_path, submodule.url)
    ):
        log(f"  Mirror up-to-date: {submodule.name}")
        return MirrorResult(
            submodule=submodule,
            success=True,
            action="skipped",
            elapsed_seconds=time.monotonic() - start,
        )

    for attempt in range(1, retries + 1):
        try:
            log(f"\n=== Updating mirror: {submodule.name}")
            run_git(
                ["git", "remote", "update", "--prune"],
                cwd=submodule.mirror_path,
            )
            return MirrorResult(
                submodule=submodule,
                success=True,
                action="updated",
                elapsed_seconds=time.monotonic() - start,
            )
        except subprocess.CalledProcessError as e:
            if attempt < retries:
                delay = min(
                    RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)),
                    MAX_RETRY_DELAY_SECONDS,
                )
                log(f"  Attempt {attempt}/{retries} failed, retrying in {delay}s...")
                time.sleep(delay)
            else:
                return MirrorResult(
                    submodule=submodule,
                    success=False,
                    action="failed",
                    elapsed_seconds=time.monotonic() - start,
                    error=str(e),
                )

    return MirrorResult(
        submodule=submodule,
        success=False,
        action="failed",
        elapsed_seconds=time.monotonic() - start,
        error="Exhausted retries",
    )


def verify_mirror(submodule: SubmoduleInfo) -> MirrorResult:
    """Verify mirror integrity with git fsck."""
    start = time.monotonic()
    if not submodule.mirror_path.exists():
        return MirrorResult(
            submodule=submodule,
            success=False,
            action="failed",
            elapsed_seconds=time.monotonic() - start,
            error=f"Mirror not found at {submodule.mirror_path}",
        )

    log(f"\n=== Verifying: {submodule.name}")
    result = run_git(
        ["git", "fsck", "--no-dangling"],
        cwd=submodule.mirror_path,
        capture=True,
    )
    elapsed = time.monotonic() - start
    if result and result.returncode != 0:
        return MirrorResult(
            submodule=submodule,
            success=False,
            action="failed",
            elapsed_seconds=elapsed,
            error=result.stderr,
        )
    return MirrorResult(
        submodule=submodule,
        success=True,
        action="verified",
        elapsed_seconds=elapsed,
    )


def run_operation(
    submodules: list[SubmoduleInfo],
    operation: str,
    jobs: int,
    retries: int,
    force: bool = False,
) -> list[MirrorResult]:
    """Run a mirror operation across submodules, optionally in parallel.

    Args:
        submodules: List of submodules to process
        operation: Operation type ("verify", "update", or "create")
        jobs: Number of parallel worker threads
        retries: Number of retry attempts for network operations
        force: Force update even if mirrors appear up-to-date (update only)
    """

    def do_one(sub: SubmoduleInfo) -> MirrorResult:
        if operation == "verify":
            return verify_mirror(sub)
        elif operation == "update":
            return update_mirror(sub, retries=retries, force=force)
        else:
            return create_mirror(sub, retries=retries)

    results: list[MirrorResult] = []
    if jobs <= 1:
        for sub in submodules:
            results.append(do_one(sub))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(do_one, sub): sub for sub in submodules}
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
    return results


def print_summary(results: list[MirrorResult]) -> None:
    """Print a summary table of mirror operations."""
    log("\n" + "=" * 72)
    log("Mirror Operation Summary")
    log("=" * 72)
    log(f"{'Submodule':<35} {'Action':<10} {'Status':<8} {'Time':>8}")
    log("-" * 72)
    total_time = 0.0
    failures = 0
    for r in sorted(results, key=lambda x: x.submodule.name):
        status = "OK" if r.success else "FAIL"
        time_str = f"{r.elapsed_seconds:.1f}s"
        log(f"{r.submodule.name:<35} {r.action:<10} {status:<8} {time_str:>8}")
        total_time += r.elapsed_seconds
        if not r.success and r.error:
            log(f"  Error: {r.error}")
        failures += int(not r.success)
    log("-" * 72)
    log(
        f"Total: {len(results)} mirrors, "
        f"{failures} failures, "
        f"{total_time:.1f}s elapsed"
    )
    log("=" * 72)


def prune_stale_mirrors(
    mirror_dir: Path,
    active_submodules: list[SubmoduleInfo],
) -> None:
    """Remove mirror directories no longer referenced by .gitmodules."""
    active_paths = {s.mirror_path for s in active_submodules}
    if not mirror_dir.exists():
        return
    for org_dir in sorted(mirror_dir.iterdir()):
        if not org_dir.is_dir():
            continue
        for repo_dir in sorted(org_dir.iterdir()):
            if repo_dir.is_dir() and repo_dir not in active_paths:
                log(f"Pruning stale mirror: {repo_dir}")
                shutil.rmtree(repo_dir, ignore_errors=True)
        # Remove org dir if now empty
        if org_dir.is_dir() and not any(org_dir.iterdir()):
            org_dir.rmdir()


def main(argv: list[str]) -> int:
    env_default = os.environ.get(MIRROR_DIR_ENV)
    default_dir = Path(env_default) if env_default else None

    parser = argparse.ArgumentParser(
        prog="setup_git_mirrors",
        description=(
            "Create and maintain bare git mirror repositories for TheRock "
            "submodules. These mirrors serve as local reference repositories "
            "to speed up git submodule init/update operations."
        ),
        epilog=(
            "After setup, use with: " "fetch_sources.py --reference-dir <mirror-dir>"
        ),
    )
    parser.add_argument(
        "--mirror-dir",
        type=Path,
        default=default_dir,
        required=default_dir is None,
        help=(
            "Directory to store bare mirror repositories "
            f"(default: ${MIRROR_DIR_ENV} env var)"
        ),
    )
    parser.add_argument(
        "--update",
        default=False,
        action="store_true",
        help=(
            "Update existing mirrors by fetching new refs from remotes. "
            "Creates any mirrors that don't yet exist."
        ),
    )
    parser.add_argument(
        "--force",
        default=False,
        action="store_true",
        help=(
            "Force update all mirrors even if they appear up-to-date. "
            "Only applies when used with --update."
        ),
    )
    parser.add_argument(
        "--verify",
        default=False,
        action="store_true",
        help="Verify mirror integrity with git fsck",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Number of parallel mirror operations (default: 4)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retry attempts for network operations (default: 3)",
    )
    parser.add_argument(
        "--prune",
        default=False,
        action="store_true",
        help="Remove mirrors for repos no longer listed in .gitmodules",
    )
    args = parser.parse_args(argv)

    mirror_dir = args.mirror_dir.resolve()
    log(f"Mirror directory: {mirror_dir}")

    submodules = discover_submodules(mirror_dir)
    log(f"Discovered {len(submodules)} submodules from .gitmodules")
    for sub in submodules:
        log(f"  {sub.name}: {sub.url}")

    if args.prune:
        prune_stale_mirrors(mirror_dir, submodules)

    if args.verify:
        results = run_operation(submodules, "verify", args.jobs, args.retries)
    elif args.update:
        if args.force:
            log("Force update enabled - updating all mirrors regardless of status")
        results = run_operation(
            submodules, "update", args.jobs, args.retries, force=args.force
        )
    else:
        if args.force:
            log("WARNING: --force only applies when used with --update, ignoring")
        to_create = [s for s in submodules if not s.mirror_path.exists()]
        already_exist = [s for s in submodules if s.mirror_path.exists()]
        if already_exist:
            log(
                f"\nSkipping {len(already_exist)} existing mirrors "
                f"(use --update to refresh them)"
            )
        if not to_create:
            log("All mirrors already exist. Use --update to refresh.")
            print_summary([])
            return 0
        results = run_operation(to_create, "create", args.jobs, args.retries)

    print_summary(results)

    failures = [r for r in results if not r.success]
    if failures:
        log(f"\nERROR: {len(failures)} mirror operations failed")
        return 1

    log(f"\nMirror directory ready: {mirror_dir}")
    log(
        f"Use with fetch_sources.py:\n"
        f"  export {MIRROR_DIR_ENV}={mirror_dir}\n"
        f"  ./build_tools/fetch_sources.py --reference-dir {mirror_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
