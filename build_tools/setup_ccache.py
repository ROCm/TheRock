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

REPO_ROOT = Path(__file__).resolve().parent.parent

# We use a compiler check script which can deal with our compiler bootstrapping
# process. This script will be written to the .ccache directory next to the
# ccache.conf file.
COMPILER_CHECK_SCRIPT_POSIX = r"""
import hashlib
from pathlib import Path
import sys

compiler_exe = Path(sys.argv[1]).resolve()
compiler_exe_stat = compiler_exe.stat()
compiler_hash_cache_dir = Path(__file__).parent / "compiler_check_cache"

# First hash the canonical path of the compiler and the mtime. We use this to
# store a full compiler check output in the compiler_hash dir by this hash.
hasher = hashlib.sha256()
hasher.update(f"{compiler_exe_stat.st_mtime},".encode())
hasher.update(str(compiler_exe).encode())
compiler_exe_path_hash = hasher.hexdigest()
compiler_exe_path_hash_file = compiler_hash_cache_dir / compiler_exe_path_hash

# Common case: We have previously computed this. Just read the file and print it.
if compiler_exe_path_hash_file.exists():
    print(compiler_exe_path_hash_file.read_text())
    sys.exit(0)


# Cache miss: compute a full content hash.
import os
import re
import subprocess
def compute_compiler_hash():
    ldd_lines = subprocess.check_output(["ldd", str(compiler_exe)]).decode().splitlines()
    ldd_pattern = re.compile(r"^(.+ => )?(.+) \(.+\)$")
    lib_paths = [str(compiler_exe)]
    for ldd_line in ldd_lines:
        m = re.match(ldd_pattern, ldd_line)
        if not m:
            print(f"Could not match ldd output: {ldd_line}")
            sys.exit(1)
        lib_path_str = m.group(2).strip()
        lib_path = Path(lib_path_str)
        if not lib_path.is_absolute():
            # Skip loaders like vdso.
            continue
        lib_paths.append(lib_path_str)

    hash = subprocess.check_output(["sha256sum"] + lib_paths).decode().strip()
    return hash

compiler_hash = compute_compiler_hash()

# Atomically write the hash cache file using rename. It is ok if this is
# racy: we just need it to be atomic.
hash_commit_file = Path(f"{compiler_exe_path_hash_file}.tmp{os.getpid()}")
hash_commit_file.parent.mkdir(parents=True, exist_ok=True)
try:
    hash_commit_file.write_text(compiler_hash)
    os.rename(hash_commit_file, compiler_exe_path_hash_file)
except OSError:
    # Ignore.
    ...

try:
    hash_commit_file.unlink()
except OSError:
    # Ignore.
    ...

print(compiler_hash)
"""


def gen_config(dir: Path, compiler_check_file: Path, args: argparse.Namespace):
    lines = []

    # Switch based on cache type.
    if False:
        # Placeholder for other cache type switches.
        ...
    else:
        # Default, local.
        local_path = dir / "local"
        local_path.mkdir(parents=True, exist_ok=True)
        lines.append(f"cache_dir = {local_path}")

    # Compiler check.
    lines.append(f"compiler_check = {sys.executable} {compiler_check_file} %compiler%")

    # End with blank line.
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace):
    dir: Path = args.dir
    config_file = dir / "ccache.conf"
    compiler_check_file = dir / "compiler_check.py"

    if args.init or not config_file.exists():
        print(f"Initializing ccache dir: {dir}", file=sys.stderr)
        dir.mkdir(parents=True, exist_ok=True)
        config_file.write_text(gen_config(dir, compiler_check_file, args))
        compiler_check_file.write_text(COMPILER_CHECK_SCRIPT_POSIX)

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

    args = p.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
