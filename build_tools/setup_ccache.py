#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Sets up ccache in a way that is compatible with the project.

Building ROCm involves bootstrapping various compiler tools and is therefore a
relatively complicated piece of software to configure ccache properly for. While
users can certainly forgo any special configuration, they will likely get less
than anticipated cache hit rates, especially for device code. This utility
centralizes ccache configuration by writing a config file and doing other cache
setup chores.

By default, the ccache config and any local cache will be setup under the
`.ccache` directory in the repo root:

* `.ccache/ccache.conf` : Configuration file.
* `.ccache/local` : Local cache (if configured for local caching).

In order to develop/debug this facility, run the `hack/ccache/test_ccache_sanity.sh`
script.

Typical usage for the current shell (will set the CCACHE_CONFIGPATH var):
    eval "$(./build_tools/setup_ccache.py)"
"""

import argparse
from pathlib import Path
import platform
import sys
import subprocess

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
IS_WINDOWS = platform.system() == "Windows"
POSIX_CCACHE_COMPILER_CHECK_PATH = THIS_DIR / "posix_ccache_compiler_check.py"
POSIX_COMPILER_CHECK_SCRIPT = (
    POSIX_CCACHE_COMPILER_CHECK_PATH.read_text() if not IS_WINDOWS else None
)
CACHE_SRV_DEV = "http://bazelremote-svc.bazelremote-ns.svc.cluster.local:8080|layout=bazel|connect-timeout=50"
CACHE_SRV_REL = "http://bazelremote-svc-rel.bazelremote-ns.svc.cluster.local:8080|layout=bazel|connect-timeout=50"

DEFAULT_LOG_DIR = REPO_ROOT / "build" / "logs" / "ccache"

# See https://ccache.dev/manual/4.6.1.html#_configuration
# log_file and stats_log are set dynamically in gen_config() using --log-dir
# so that Windows workflows can direct logs to BUILD_DIR/logs/ccache/ (B:\build)
# instead of REPO_ROOT/build/logs/ccache/ (C: drive).
CONFIG_PRESETS_MAP = {
    "local": {},
    # Dev and release use separate cache servers to avoid cache pollution.
    # We may later split these further into presubmit (PR) vs postsubmit
    # (post-merge) presets — presubmit serves varied code at mixed trust
    # levels while postsubmit serves a uniform stream of approved commits,
    # so separating them improves both cache hit rates and data integrity.
    "github-oss-dev": {
        "secondary_storage": CACHE_SRV_DEV,
        "max_size": "5G",
    },
    "github-oss-release": {
        "secondary_storage": CACHE_SRV_REL,
        "max_size": "5G",
    },
}


def gen_config(dir: Path, compiler_check_file: Path, args: argparse.Namespace):
    lines = []

    config_preset: str = args.config_preset
    selected_config = CONFIG_PRESETS_MAP[config_preset]
    for k, v in selected_config.items():
        lines.append(f"{k} = {v}")

    # Log paths: use --log-dir if provided, otherwise default to
    # REPO_ROOT/build/logs/ccache. On Windows CI the build dir is on
    # a separate drive (B:\build) from the source checkout (C: drive),
    # so workflows must pass --log-dir to place logs where the upload
    # scripts expect them.
    if config_preset != "local":
        ccache_log_dir: Path = args.log_dir if args.log_dir else DEFAULT_LOG_DIR
        ccache_log_dir.mkdir(parents=True, exist_ok=True)
        lines.append(f"log_file = {ccache_log_dir / 'ccache.log'}")
        lines.append(f"stats_log = {ccache_log_dir / 'ccache_stats.log'}")

    # (TODO:consider https://ccache.dev/manual/4.6.1.html#_storage_interaction)
    # Switch based on cache type.
    if args.remote:
        if not args.remote_storage:
            raise ValueError(f"Expected --remote-storage with --remote option")
        lines.append(f"remote_storage = {args.remote_storage}")
        lines.append(f"remote_only = true")
    else:
        # Default, local.
        local_path: Path = args.local_path
        if local_path is None:
            local_path = dir / "local"
        local_path.mkdir(parents=True, exist_ok=True)
        lines.append(f"cache_dir = {local_path}")

    # Compiler check: on POSIX we use a custom script that fingerprints the
    # compiler binary and its shared libraries via ldd + sha256sum. On Windows
    # (MSVC) those tools don't exist; ccache's default mtime check works well.
    if not IS_WINDOWS:
        lines.append(
            f"compiler_check = {sys.executable} {compiler_check_file} "
            f"{dir / 'compiler_check_cache'} %compiler%"
        )

    # Slop settings.
    # Creating a hard link to a file increasing the link count, which triggers
    # a ctime update (since ctime tracks changes to the inode metadata) for
    # *all* links to the file. Since we are basically always creating hard
    # link farms in parallel as part of sandboxing, we have to disable this
    # check as it is never valid for our build system and will result in
    # spurious ccache panics where it randomly falls back to the real compiler
    # if the ccache invocation happens to coincide with parallel sandbox
    # creation for another sub-project.
    lines.append(f"sloppiness = include_file_ctime")

    # End with blank line.
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace):
    dir: Path = args.dir
    config_file = dir / "ccache.conf"
    compiler_check_file = dir / "compiler_check.py"

    config_contents = gen_config(dir, compiler_check_file, args)
    if args.init or not config_file.exists():
        print(f"Initializing ccache dir: {dir}", file=sys.stderr)
        dir.mkdir(parents=True, exist_ok=True)
        config_file.write_text(config_contents)
        if not IS_WINDOWS:
            compiler_check_file.write_text(POSIX_COMPILER_CHECK_SCRIPT)

    else:
        # Check to see if updated.
        if config_file.read_text() != config_contents:
            print(
                f"NOTE: {config_file} does not match expected. Run with --init to regenerate",
                file=sys.stderr,
            )
        if not IS_WINDOWS and (
            not compiler_check_file.exists()
            or compiler_check_file.read_text() != POSIX_COMPILER_CHECK_SCRIPT
        ):
            print(
                f"NOTE: {compiler_check_file} does not match expected. Run with --init to regenerate it",
                file=sys.stderr,
            )

    # Reset statistic counters
    if args.reset_stats:
        try:
            proc_ccache = subprocess.run(
                ["ccache", "--zero-stats"], capture_output=True, text=True
            )
            proc_ccache.check_returncode()
            print(proc_ccache.stdout, end="", file=sys.stderr)

        except subprocess.CalledProcessError:
            print(
                f"ERROR! Zeroing statistic counters failed. Message: {proc_ccache.stderr}",
                file=sys.stderr,
            )
    # Output options.
    if IS_WINDOWS:
        print(f"set CCACHE_CONFIGPATH={config_file}")
    else:
        print(f"export CCACHE_CONFIGPATH={config_file}")


def main(argv: list[str]):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dir",
        type=Path,
        default=REPO_ROOT / ".ccache",
        help="Location of the .ccache directory (defaults to ../.ccache)",
    )
    p.add_argument(
        "--reset-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Zeros statistics counters (default to enabled).",
    )
    command_group = p.add_mutually_exclusive_group()
    command_group.add_argument(
        "--init",
        action="store_true",
        help="Initialize a ccache directory",
    )

    type_group = p.add_mutually_exclusive_group()
    type_group.add_argument(
        "--local", action="store_true", help="Use a local cache (default)"
    )
    type_group.add_argument("--remote", action="store_true", help="Use a remote cache")

    p.add_argument(
        "--local-path",
        type=Path,
        help="Use a non-default local ccache directory (defaults to 'local/' in --dir)",
    )

    p.add_argument("--remote-storage", help="Remote storage configuration/URL")

    p.add_argument(
        "--log-dir",
        type=Path,
        help="Directory for ccache log files. Defaults to REPO_ROOT/build/logs/ccache. "
        "On Windows CI, pass BUILD_DIR/logs/ccache so logs land in the build tree.",
    )

    preset_group = p.add_mutually_exclusive_group()
    preset_group.add_argument(
        "--config-preset",
        type=str,
        default="local",
        choices=list(CONFIG_PRESETS_MAP.keys()),
        help="Predefined set of configurations for ccache by environment.",
    )
    preset_group.add_argument(
        "--release-type",
        type=str,
        choices=["", "dev", "nightly", "prerelease"],
        help='Shorthand for --config-preset: "" and "dev" map to github-oss-dev, '
        "others map to github-oss-release.",
    )

    args = p.parse_args(argv)
    if args.release_type is not None:
        if args.release_type in ("", "dev"):
            args.config_preset = "github-oss-dev"
        else:
            args.config_preset = "github-oss-release"
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
