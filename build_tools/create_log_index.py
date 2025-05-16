#!/usr/bin/env python3

import os
import subprocess
from pathlib import Path
import sys


def log(*args):
    print(*args)
    sys.stdout.flush()


def index_log_files(build_dir: Path, amdgpu_family: str):
    log_dir = build_dir / "logs"
    index_file = log_dir / "index.html"

    if log_dir.is_dir():
        log(f"[INFO] Found '{log_dir}' directory. Indexing '*.log' files...")
        subprocess.run(
            ["python", str(build_dir / "indexer.py"), "-f", "*.log", str(log_dir)],
            check=True,
        )
    else:
        log(f"[WARN] Log directory '{log_dir}' not found. Skipping indexing.")
        return

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
    build_dir = Path(os.getenv("BUILD_DIR", "build"))
    amdgpu_family = os.getenv("AMDGPU_FAMILIES")
    if not amdgpu_family:
        print("[ERROR] AMDGPU_FAMILIES not set")
        sys.exit(1)
    index_log_files(build_dir, amdgpu_family)
