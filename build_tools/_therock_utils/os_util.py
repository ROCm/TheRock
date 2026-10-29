# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""OS-level utility functions.

These include workarounds for platform-specific quirks, particularly Windows
file locking.
"""

from pathlib import Path
import shutil
import sys
import time

# Maximum number of attempts to retry removing a directory.
RMTREE_MAX_ATTEMPTS: int = 10
# Base delay between retry attempts in seconds (multiplied by attempt + 2).
RMTREE_RETRY_DELAY_SECONDS: float = 0.5


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
