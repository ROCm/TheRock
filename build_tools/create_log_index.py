#!/usr/bin/env python3

import os
import subprocess
from pathlib import Path
import sys
import platform
import argparse


def log(*args):
    print(*args)
    sys.stdout.flush()


def is_windows():
    return platform.system().lower() == "windows"


def normalize_path(p: Path) -> str:
    return str(p).replace("\\", "/") if is_windows() else str(p)


def get_indexer_path() -> Path:
    indexer_env_path = os.getenv("INDEXER_PATH")
    if indexer_env_path:
        indexer_path = Path(indexer_env_path).resolve()
        if indexer_path.is_file():
            log(f"[INFO] Using INDEXER_PATH from environment: {indexer_path}")
            return indexer_path
        else:
            log(
                f"[ERROR] INDEXER_PATH is set but does not point to a file: {indexer_path}"
            )
            sys.exit(2)
    else:
        log("[ERROR] INDEXER_PATH environment variable not set")
        sys.exit(2)


def index_log_files(build_dir: Path, amdgpu_family: str):
    log_dir = build_dir / "logs"
    index_file = log_dir / "index.html"

    if not log_dir.is_dir():
        log(f"[WARN] Log directory '{log_dir}' not found. Skipping indexing.")
        return

    indexer_path = get_indexer_path()

    log(f"[INFO] Found '{log_dir}' directory. Indexing '*.log' files...")
    try:
        subprocess.run(
            ["python", str(indexer_path), "-f", "*.log", normalize_path(log_dir)],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        log(f"[ERROR] Failed to run indexer.py: {e}")
        sys.exit(2)

    if index_file.exists():
        log(
            f"[INFO] Rewriting links in '{index_file}' with AMDGPU_FAMILIES={amdgpu_family}..."
        )
        content = index_file.read_text()
        updated = content.replace(
            'a href=".."', f'a href="../../index-{amdgpu_family}.html"'
        )
        index_file.write_text(updated)
        log("[INFO] Log index links updated.")
    else:
        log(f"[WARN] '{index_file}' not found. Skipping link rewrite.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create HTML index for log files.")
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(os.getenv("BUILD_DIR", "build")),
        help="Build directory containing logs (default: 'build' or $BUILD_DIR)",
    )
    parser.add_argument(
        "--amdgpu-family",
        type=str,
        default=os.getenv("AMDGPU_FAMILIES"),
        help="AMDGPU family name (default: $AMDGPU_FAMILIES)",
    )
    args = parser.parse_args()

    if not args.amdgpu_family:
        log("[ERROR] --amdgpu-family not provided and AMDGPU_FAMILIES env var not set")
        sys.exit(1)

    index_log_files(args.build_dir, args.amdgpu_family)
