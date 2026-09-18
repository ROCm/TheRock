#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Copy runner-local EngFlow client credentials into $RUNNER_TEMP.

XLA and JAX CI keep the mTLS files at `/data/ci-cert.crt` and
`/data/ci-cert.key` on the runner, where every job can read them — including
pull requests from forks. GitHub Actions secrets are not used: they are
invisible to fork PRs and would be a third credential path next to the
runner-local AWS files and OIDC roles the other caches already use.

The copy lands under $RUNNER_TEMP rather than the workspace, which CI
publishes as artifacts. Missing source files are not an error: the wheel
build runs without a remote cache.

Used by `.github/workflows/multi_arch_build_linux_jax_wheels_ci.yml`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from github_actions_api import gha_set_output

DEFAULT_SOURCE_DIR = Path("/data")
CERT_NAME = "ci-cert.crt"
KEY_NAME = "ci-cert.key"
DEST_DIRNAME = "bazel-cache-credentials"


def _log(msg: str) -> None:
    print(f"[jax-bazel-cache] {msg}", file=sys.stderr)


def _restrict_private(path: Path, is_dir: bool) -> None:
    """Owner-only access; ignore chmod failures on filesystems that lack mode bits."""
    mode = stat.S_IRWXU if is_dir else stat.S_IRUSR | stat.S_IWUSR
    try:
        os.chmod(path, mode)
    except OSError as e:
        _log(f"Could not chmod {path}: {e}")


def materialize_credentials(source_dir: Path, dest_dir: Path) -> Path | None:
    """Copy the runner's cert/key into dest_dir, or return None if they are absent."""
    certificate = source_dir / CERT_NAME
    key = source_dir / KEY_NAME
    missing = [
        p for p in (certificate, key) if not p.is_file() or p.stat().st_size == 0
    ]
    if missing:
        _log(
            "No cache credentials at "
            + ", ".join(str(p) for p in missing)
            + "; building without a remote cache."
        )
        return None

    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True)
    _restrict_private(dest_dir, is_dir=True)

    for src, name in ((certificate, CERT_NAME), (key, KEY_NAME)):
        dest = dest_dir / name
        shutil.copyfile(src, dest)
        _restrict_private(dest, is_dir=False)

    _log(f"Cache credentials copied to {dest_dir}")
    return dest_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Copy runner-local EngFlow credentials into $RUNNER_TEMP."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(
            os.environ.get("JAX_BAZEL_CACHE_CREDENTIALS_DIR") or DEFAULT_SOURCE_DIR
        ),
        help="Directory that already holds ci-cert.crt and ci-cert.key on the runner.",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=None,
        help="Directory to write the copy into. Defaults to $RUNNER_TEMP/"
        f"{DEST_DIRNAME}.",
    )
    args = parser.parse_args(argv)

    dest_dir = args.dest_dir
    if dest_dir is None:
        runner_temp = os.environ.get("RUNNER_TEMP")
        if not runner_temp:
            _log("RUNNER_TEMP is unset; building without a remote cache.")
            gha_set_output({"credentials_dir": ""})
            return
        dest_dir = Path(runner_temp) / DEST_DIRNAME

    copied = materialize_credentials(args.source_dir, dest_dir)
    gha_set_output({"credentials_dir": str(copied) if copied else ""})


if __name__ == "__main__":
    main()
