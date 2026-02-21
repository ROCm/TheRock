#!/usr/bin/env python3
"""Start a GNU Make jobserver and build TheRock.

Location:
  build_tools/ninja_jobserver/build_therock_withJobserver.py

What it does:
  - Computes N (from $JOBS or CPU count)
  - Runs: make -jN therock  (from repo root)
  - GNU make creates the jobserver and exports MAKEFLAGS=...--jobserver-auth=...
  - The Makefile's 'therock' target should call build_the_rock.py, which will
    detect the jobserver and avoid passing -j to Ninja, so Ninja joins the pool.
"""

import os
import shlex
import subprocess
import multiprocessing
from pathlib import Path


def log(msg: str) -> None:
    print(msg, flush=True)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def jobs() -> int:
    v = os.environ.get("JOBS")
    if v:
        try:
            return int(v)
        except ValueError:
            log(f"Warning: invalid JOBS='{v}', ignoring.")
    return multiprocessing.cpu_count() or 1


def run(cmd: list[str], cwd: str) -> None:
    log("Running: " + " ".join(shlex.quote(c) for c in cmd))
    subprocess.check_call(cmd, cwd=cwd)


def main() -> None:
    root = repo_root()
    j = jobs()
    log("=== TheRock: GNU Make jobserver build ===")
    log(f"Repo root: {root}")
    log(f"Parallelism: -j{j}\n")
    run(["make", f"-j{j}", "therock"], cwd=str(root))


if __name__ == "__main__":
    main()
