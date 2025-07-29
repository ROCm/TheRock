#!/usr/bin/env python
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
import sys

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
POSIX_CCACHE_COMPILER_CHECK_PATH = THIS_DIR / "posix_ccache_compiler_check.py"
POSIX_COMPILER_CHECK_SCRIPT = POSIX_CCACHE_COMPILER_CHECK_PATH.read_text()
CACHE_SRV = "http://bazelremote-svc.bazelremote-ns.svc.cluster.local:8080|layout=bazel|connect-timeout=50"

# See https://ccache.dev/manual/4.6.1.html#_configuration
CONFIG_PRESETS_MAP = {
    "local": {},
    # Moving build_*_packages.yml CCACHE Env variables to here (linux for now)
    # For initial implementation, pre and post submit will be the same
    "github-oss-presubmit": {
        "secondary_storage": CACHE_SRV,
        "log_file": REPO_ROOT / "build/logs/ccache.log",
        "stats_log": REPO_ROOT / "build/logs/ccache_stats.log",
        "max_size": "5G",
    },
    "github-oss-postsubmit": {
        "secondary_storage": CACHE_SRV,
        "log_file": REPO_ROOT / "build/logs/ccache.log",
        "stats_log": REPO_ROOT / "build/logs/ccache_stats.log",
        "max_size": "5G",
    },
}


def gen_config(dir: Path, compiler_check_file: Path, args: argparse.Namespace):
    lines = []

    # Initial implementation of presets will maintain current yml behavior
    # which allows both "local" and "remote" cache (see `storage interaction`)
    # and inserts all ccache env var configs along side below's local defaults
    config_preset: str = args.config_preset
    selected_config = CONFIG_PRESETS_MAP[config_preset]
    for k, v in selected_config.items():
        lines.append(f"{k} = {v}")
        # Ensure full dir path for logs exists, else ccache will fail and stop CI
        if k == "log_file" or k == "stats_log":
            log_dir = v.parent.absolute()
            if not log_dir.exists():
                log_dir.mkdir(parents=True, exist_ok=True)

    # (TODO:consider https://ccache.dev/manual/4.6.1.html#_storage_interaction)
    # Switch based on cache type.
    if False:
        # Placeholder for other cache type switches.
        ...
    else:
        # Default, local.
        local_path: Path = args.local_path
        if local_path is None:
            local_path = dir / "local"
        local_path.mkdir(parents=True, exist_ok=True)
        lines.append(f"cache_dir = {local_path}")

    # Compiler check.
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
    compiler_check_script = POSIX_COMPILER_CHECK_SCRIPT
    if args.init or not config_file.exists():
        print(f"Initializing ccache dir: {dir}", file=sys.stderr)
        dir.mkdir(parents=True, exist_ok=True)
        config_file.write_text(config_contents)
        compiler_check_file.write_text(compiler_check_script)
    else:
        # Check to see if updated.
        if config_file.read_text() != config_contents:
            print(
                f"NOTE: {config_file} does not match expected. Run with --init to regenerate",
                file=sys.stderr,
            )
        if (
            not compiler_check_file.exists()
            or compiler_check_file.read_text() != compiler_check_script
        ):
            print(
                f"NOTE: {compiler_check_file} does not match expected. Run with --init to regenerate it",
                file=sys.stderr,
            )

    # Output options.
    print(f"export CCACHE_CONFIGPATH={config_file}")


def main(argv: list[str]):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dir",
        type=Path,
        default=REPO_ROOT / ".ccache",
        help="Location of the .ccache directory (defaults to ../.ccache)",
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

    p.add_argument(
        "--local-path",
        type=Path,
        help="Use a non-default local ccache directory (defaults to 'local/' in --dir)",
    )

    preset_group = p.add_mutually_exclusive_group()
    preset_group.add_argument(
        "--config-preset",
        type=str,
        default="local",
        choices=["local", "github-oss-presubmit", "github-oss-postsubmit"],
        help="Predefined set of configurations for ccache by enviroment.",
    )

    args = p.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
