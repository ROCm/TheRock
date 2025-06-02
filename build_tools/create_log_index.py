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
    """
    Resolve to the local third-party/indexer/indexer.py copy.
    """
    indexer_path = (
        Path(__file__).resolve().parent.parent / "third-party/indexer/indexer.py"
    )
    if indexer_path.is_file():
        log(f"[INFO] Using bundled indexer.py: {indexer_path}")
        return indexer_path
    else:
        log(f"[ERROR] Bundled indexer.py not found at: {indexer_path}")
        sys.exit(2)


def index_log_files(log_dir: Path, amdgpu_family: str):
    index_file = log_dir / "index.html"

    if not log_dir.is_dir():
        log(f"[WARN] Log directory '{log_dir}' not found. Skipping indexing.")
        return

    indexer_path = get_indexer_path()
    log(
        f"[INFO] Found '{log_dir}' directory. Indexing '*.log' files using indexer: {indexer_path}"
    )

    try:
        subprocess.run(
            [sys.executable, str(indexer_path), "-f", "*.log", normalize_path(log_dir)],
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

        # Ensure full write and flush to disk
        with open(index_file, "w", encoding="utf-8") as f:
            f.write(updated)
            f.flush()
            os.fsync(f.fileno())

        log("[INFO] Log index links updated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create HTML index for log files.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(os.getenv("LOG_DIR", "build/logs")),
        help="Directory containing log files (default: 'build/logs' or $LOG_DIR)",
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

    index_log_files(args.log_dir, args.amdgpu_family)
