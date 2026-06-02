#!/usr/bin/env python3
"""TheRock build driver (jobserver-aware).

Location:
  build_tools/ninja_jobserver/build_the_rock.py

What it does:
  - Configures TheRock with CMake + Ninja.
  - Builds via `cmake --build`.
  - If a GNU: Make jobserver is detected (MAKEFLAGS contains --jobserver-auth=...),
    it will NOT pass -j to Ninja so Ninja can join the jobserver pool.
  - If no jobserver is detected, it will pass -j<cpu_count> (or $JOBS) to Ninja.

Environment variables:
  THEROCK_CMAKE_ARGS : Extra CMake cache args (string; shell-split), e.g.
                       "-DTHEROCK_AMDGPU_FAMILIES=gfx1100 -DCMAKE_BUILD_TYPE=Release"
  BUILD_DIR          : Build directory (default: <repo_root>/build)
  CMAKE_GENERATOR    : CMake generator (default: Ninja)
  JOBS               : Used only when NO jobserver is present (default: cpu_count)
"""

import os
import sys
import shlex
import subprocess
from pathlib import Path


def log(msg: str) -> None:
    print(msg, flush=True)


def has_jobserver() -> bool:
    """True if we appear to be running under GNU make -j jobserver."""
    return "--jobserver-auth=" in os.environ.get("MAKEFLAGS", "")


def run(cmd: list[str], cwd: str | None = None) -> None:
    log("Running: " + " ".join(shlex.quote(c) for c in cmd))
    subprocess.check_call(cmd, cwd=cwd)


def repo_root() -> Path:
    # build_tools/ninja_jobserver/build_the_rock.py -> repo root is 3 levels up
    return Path(__file__).resolve().parents[2]


def build_dir(root: Path) -> Path:
    v = os.environ.get("BUILD_DIR")
    return Path(v) if v else (root / "build")


def generator() -> str:
    return os.environ.get("CMAKE_GENERATOR", "Ninja")


def jobs() -> int:
    v = os.environ.get("JOBS")
    if v:
        try:
            return int(v)
        except ValueError:
            log(f"Warning: invalid JOBS='{v}', falling back to CPU count.")
    return os.cpu_count() or 1


def extra_cmake_args() -> list[str]:
    raw = os.environ.get("THEROCK_CMAKE_ARGS", "")
    return shlex.split(raw) if raw.strip() else []


def cmake_configure(root: Path, bdir: Path) -> None:
    log(f"==> Configuring TheRock in {bdir}")
    bdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "cmake",
        "-S", str(root),
        "-B", str(bdir),
        "-G", generator(),
    ] + extra_cmake_args()
    run(cmd)


def cmake_build(bdir: Path, ninja_args: list[str]) -> None:
    log("==> Building TheRock")
    if has_jobserver():
        log("    Jobserver detected -> NOT passing -j to Ninja (lets Ninja join jobserver)")
        cmd = ["cmake", "--build", str(bdir), "--"] + ninja_args
    else:
        j = jobs()
        log(f"    No jobserver detected -> using -j{j}")
        cmd = ["cmake", "--build", str(bdir), "--", f"-j{j}"] + ninja_args
    run(cmd)


def usage(prog: str) -> None:
    print(
        f"""Usage: {prog} [configure|build|all] [extra ninja args/targets...]

Examples:
  {prog} all
  {prog} configure
  {prog} build
  {prog} build install
""".strip()
    )


def main(argv: list[str]) -> int:
    prog = argv[0]
    cmd = argv[1] if len(argv) > 1 else "all"
    extra = argv[2:]

    root = repo_root()
    bdir = build_dir(root)

    if cmd == "configure":
        cmake_configure(root, bdir)
    elif cmd == "build":
        cmake_build(bdir, extra)
    elif cmd == "all":
        cmake_configure(root, bdir)
        cmake_build(bdir, extra)
    else:
        usage(prog)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
