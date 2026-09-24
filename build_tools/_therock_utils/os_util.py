# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""OS-level utility functions.

These include workarounds for platform-specific quirks, particularly Windows
file locking.
"""

import os
from pathlib import Path
import shutil
import sys
import time

# Maximum number of attempts to retry removing a directory.
RMTREE_MAX_ATTEMPTS: int = 10
# Base delay between retry attempts in seconds (multiplied by attempt + 2).
RMTREE_RETRY_DELAY_SECONDS: float = 0.5

# GPU paravirtualization device presented to a WSL2 guest. See is_wsl_gpu().
DXG_DEVICE: str = "/dev/dxg"


def is_wsl_gpu() -> bool:
    """True when running inside a WSL2 guest with a paravirtualized GPU.

    WSL reports platform.system() == "Linux", so callers that branch on the OS
    alone will treat a WSL runner as bare-metal Linux. That is wrong for
    anything touching the GPU: WSL exposes it through GPU-PV as /dev/dxg, with
    no amdgpu kernel driver, no /dev/kfd and no /dev/dri.

    Both signals are required. A /proc/version check alone would also match a
    WSL instance with no GPU passed through, where GPU checks should still fail
    loudly rather than be silently skipped.
    """
    if not os.path.exists(DXG_DEVICE):
        return False
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def rmtree_with_retry(
    path: Path,
    *,
    verbose: bool = False,
    max_attempts: int = RMTREE_MAX_ATTEMPTS,
    retry_delay_seconds: float = RMTREE_RETRY_DELAY_SECONDS,
) -> None:
    """Remove a directory tree, retrying on PermissionError (e.g. Windows locks).

    On Windows, files may be temporarily locked by antivirus scanners, search
    indexers, or other processes. A short retry loop avoids flaky failures in
    CI builds where parallel jobs can hold transient file handles.
    """
    for attempt in range(max_attempts):
        try:
            shutil.rmtree(path)
            if verbose:
                print(f"rmtree {path}", file=sys.stderr)
            return
        except PermissionError:
            wait_time = retry_delay_seconds * (attempt + 2)
            if verbose:
                print(
                    f"PermissionError calling shutil.rmtree('{path}') "
                    f"retrying after {wait_time}s",
                    file=sys.stderr,
                )
            time.sleep(wait_time)
            if attempt == max_attempts - 1:
                if verbose:
                    print(
                        f"rmtree failed after {max_attempts} attempts, failing",
                        file=sys.stderr,
                    )
                raise
