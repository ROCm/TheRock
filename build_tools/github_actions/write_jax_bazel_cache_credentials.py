#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Copy the runner-local EngFlow cache token into $RUNNER_TEMP.

The token is an EngFlow-issued JWT kept on the runner, where every job can read
it — including pull requests from forks. That is safe because the token's
`cache-reader` role is enforced by EngFlow, not by the client. GitHub Actions
secrets are not used: they are invisible to fork PRs and would be a third
credential path next to the runner-local AWS files and OIDC roles the other
caches already use.

The copy lands under $RUNNER_TEMP rather than the workspace, which CI
publishes as artifacts. A missing token is not an error: the wheel build runs
without a remote cache. A token file that is present but empty is.

Used by `.github/workflows/multi_arch_build_linux_jax_wheels_ci.yml`.
"""

import argparse
import os
import shutil
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from github_actions_api import gha_set_output

DEFAULT_SOURCE_DIR = Path("/data")
# Also the name configure_jax_bazel_cache.py reads inside the build container.
TOKEN_NAME = "engflow-cache.jwt"
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
    """Copy the runner's token into dest_dir, or return None if it is absent."""
    token = source_dir / TOKEN_NAME
    if not token.is_file():
        _log(f"No cache token at {token}; building without a remote cache.")
        return None
    if token.stat().st_size == 0:
        raise ValueError(f"Cache token is present but empty: {token}")

    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True)
    _restrict_private(dest_dir, is_dir=True)

    dest = dest_dir / TOKEN_NAME
    shutil.copyfile(token, dest)
    _restrict_private(dest, is_dir=False)

    _log(f"Cache token copied to {dest_dir}")
    return dest_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Copy the runner-local EngFlow cache token into $RUNNER_TEMP."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(
            os.environ.get("JAX_BAZEL_CACHE_CREDENTIALS_DIR") or DEFAULT_SOURCE_DIR
        ),
        help=f"Directory that already holds {TOKEN_NAME} on the runner.",
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
            raise RuntimeError("RUNNER_TEMP is unset; pass --dest-dir")
        dest_dir = Path(runner_temp) / DEST_DIRNAME

    copied = materialize_credentials(args.source_dir, dest_dir)
    gha_set_output({"credentials_dir": str(copied) if copied else ""})


if __name__ == "__main__":
    main()
