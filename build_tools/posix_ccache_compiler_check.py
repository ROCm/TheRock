"""Script meant to be used in the ccache compiler_check directive on POSIX.

Syntax:
  posix_ccache_compiler_check.py compiler_check_cache_dir compiler_path

It expects one argument for the compiler path to check and will print a
fingerprint for the compiler that should be stable across builds. CCache then
will hash this fingerprint and include it in the cache key for any uses of the
compiler.

Currently, the algorithm simply passes the compiler binary and every valid
shared library (skipping loader libraries) to sha256sum and post processes
it to eliminate absolute path names that cause false cache misses. This result
should be deterministic for all compilers built in the same way.

Because this script is very hot (it is invoked for every invocation of ccache),
we take some extra pains to maintain directory of fingerprint caches. This
directory consists of sha256sums of the mtime and full path to the compiler
and merely serves to save excess fingerprinting. It is always safe to
delete this cache, and it doesn't cost much to store excess variants, so we
take no effort to maintain it.

This script is typically snapshotted into the ccache directory and hardcoded
in configurations.

If making changes to this script, please validate it by running
./hack/ccache/test_ccache_sanity.sh and studying logs. If you don't know how
ccache works, consult with someone who does.
"""

import hashlib
from pathlib import Path
import sys

compiler_hash_cache_dir = Path(sys.argv[1])
compiler_exe = Path(sys.argv[2]).resolve()
compiler_exe_stat = compiler_exe.stat()

# First hash the canonical path of the compiler and the mtime. We use this to
# store a full compiler check output in the compiler_hash dir by this hash.
fingerprint_version = 1
hasher = hashlib.sha256()
hasher.update(f"{fingerprint_version},{compiler_exe_stat.st_mtime},".encode())
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


def compute_compiler_fingerprint():
    ldd_lines = (
        subprocess.check_output(["ldd", str(compiler_exe)]).decode().splitlines()
    )
    # Matching ldd output lines like:
    #   linux-vdso.so.1 (0x00007481bc7ed000)
    #   libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007481bc400000)
    #   /lib64/ld-linux-x86-64.so.2 (0x00007481bc7ef000)
    # Ignores optional soname (group1). Captures path (group2). Ignores addr.
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

    def _norm_fingerprint_path(line: str):
        # Line is: {hash} {path}
        # Rewrite the path to just be the basename.
        hash, path_str = line.split(None, 2)
        path_str = str(Path(path_str).name)
        return f"{hash}\t{path_str}"

    raw_fingerprint_lines = (
        subprocess.check_output(["sha256sum"] + lib_paths).decode().splitlines()
    )
    fingerprint_lines = [_norm_fingerprint_path(line) for line in raw_fingerprint_lines]
    return "\n".join(fingerprint_lines)


compiler_fingerprint = compute_compiler_fingerprint()

# Atomically write the hash cache file using rename. It is ok if this is
# racy: we just need it to be atomic.
hash_commit_file = Path(f"{compiler_exe_path_hash_file}.tmp{os.getpid()}")
hash_commit_file.parent.mkdir(parents=True, exist_ok=True)
try:
    hash_commit_file.write_text(compiler_fingerprint)
    os.rename(hash_commit_file, compiler_exe_path_hash_file)
except OSError:
    # Ignore.
    ...

try:
    hash_commit_file.unlink()
except OSError:
    # Ignore.
    ...

print(compiler_fingerprint)
