#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stage the EngFlow cache token for the JAX wheel build in $RUNNER_TEMP.

Two EngFlow-issued JWTs exist, and EngFlow enforces each one's role:

- The `cache-reader` token sits on the runner, where every job can read it —
  including pull requests from forks, which is safe because it cannot write.
  This copies it.
- The `cache-writer` token is never on a runner. Only ordinary pushes to `main`
  in ROCm/TheRock assume `WRITER_ROLE` through GitHub OIDC (`--plan` names the
  role for the workflow) and fetch it from AWS Secrets Manager.

GitHub Actions secrets are not used: they are invisible to fork PRs and would be
a third credential path next to the runner-local files and OIDC roles the other
caches already use.

The copy lands under $RUNNER_TEMP rather than the workspace, which CI
publishes as artifacts. A missing reader token is not an error: the wheel build
runs without a remote cache. A token that is present but empty is, and so is a
writer token that cannot be fetched.

Used by `.github/workflows/multi_arch_build_linux_jax_wheels_ci.yml`.
"""

import argparse
import json
import os
import shutil
import stat
import sys
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configure_jax_bazel_cache import upload_allowed
from github_actions_api import gha_set_output

DEFAULT_SOURCE_DIR = Path("/data")
# Also the name configure_jax_bazel_cache.py reads inside the build container.
TOKEN_NAME = "engflow-cache.jwt"
DEST_DIRNAME = "bazel-cache-credentials"

WRITER_REPOSITORY = "ROCm/TheRock"
WRITER_ROLE = "arn:aws:iam::692859939525:role/therock-jax-cache-writer-main"
WRITER_SECRET_ID = "/ossci/ci/ROCm/TheRock/main/jax-bazel-cache-writer"
WRITER_SECRET_KEY = "token"
WRITER_REGION = "us-east-2"


def _log(msg: str) -> None:
    print(f"[jax-bazel-cache] {msg}", file=sys.stderr)


def _restrict_private(path: Path, is_dir: bool) -> None:
    """Owner-only access; ignore chmod failures on filesystems that lack mode bits."""
    mode = stat.S_IRWXU if is_dir else stat.S_IRUSR | stat.S_IWUSR
    try:
        os.chmod(path, mode)
    except OSError as e:
        _log(f"Could not chmod {path}: {e}")


def uses_writer_token(event_name: str, ref: str, repository: str) -> bool:
    """Only the pushes to main that upload get the writer; forks never do."""
    return repository == WRITER_REPOSITORY and upload_allowed(event_name, ref)


def _fresh_dest_dir(dest_dir: Path) -> Path:
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True)
    _restrict_private(dest_dir, is_dir=True)
    return dest_dir / TOKEN_NAME


def materialize_credentials(source_dir: Path, dest_dir: Path) -> Path | None:
    """Copy the runner's token into dest_dir, or return None if it is absent."""
    token = source_dir / TOKEN_NAME
    if not token.is_file():
        _log(f"No cache token at {token}; building without a remote cache.")
        return None
    if token.stat().st_size == 0:
        raise ValueError(f"Cache token is present but empty: {token}")

    dest = _fresh_dest_dir(dest_dir)
    shutil.copyfile(token, dest)
    _restrict_private(dest, is_dir=False)

    _log(f"Cache token copied to {dest_dir}")
    return dest_dir


def fetch_writer_token() -> str:
    """Reads the writer JWT with the writer role's credentials.

    The workflow passes those credentials to this step's environment only.
    """
    client = boto3.client("secretsmanager", region_name=WRITER_REGION)
    secret = client.get_secret_value(SecretId=WRITER_SECRET_ID)
    token = json.loads(secret["SecretString"]).get(WRITER_SECRET_KEY, "").strip()
    if not token:
        raise ValueError(f"{WRITER_SECRET_ID} has no {WRITER_SECRET_KEY!r} value")
    if os.environ.get("GITHUB_ACTIONS") == "true":
        # Nothing prints the token; this keeps a stray echo out of the log.
        print(f"::add-mask::{token}")
    return token


def stage_writer_token(token: str, dest_dir: Path) -> Path:
    dest = _fresh_dest_dir(dest_dir)
    dest.write_text(token)
    _restrict_private(dest, is_dir=False)
    _log(f"Cache writer token staged in {dest_dir}")
    return dest_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Stage the EngFlow cache token in $RUNNER_TEMP."
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Only print which AWS role to assume (writer_role, aws_region); "
        "empty when the run reads with the runner's token.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(
            os.environ.get("JAX_BAZEL_CACHE_CREDENTIALS_DIR") or DEFAULT_SOURCE_DIR
        ),
        help=f"Directory that already holds the reader {TOKEN_NAME} on the runner.",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=None,
        help="Directory to write the copy into. Defaults to $RUNNER_TEMP/"
        f"{DEST_DIRNAME}.",
    )
    parser.add_argument(
        "--event-name",
        default=os.environ.get("GITHUB_EVENT_NAME", ""),
        help="GitHub event that started the run; only a push to main writes.",
    )
    parser.add_argument(
        "--ref",
        default=os.environ.get("GITHUB_REF", ""),
        help="Git ref the run is for, e.g. refs/heads/main.",
    )
    parser.add_argument(
        "--repository",
        default=os.environ.get("GITHUB_REPOSITORY", ""),
        help="Repository the run is in; forks never get the writer token.",
    )
    args = parser.parse_args(argv)

    writer = uses_writer_token(args.event_name, args.ref, args.repository)
    if args.plan:
        gha_set_output(
            {
                "writer_role": WRITER_ROLE if writer else "",
                "aws_region": WRITER_REGION if writer else "",
            }
        )
        return

    dest_dir = args.dest_dir
    if dest_dir is None:
        runner_temp = os.environ.get("RUNNER_TEMP")
        if not runner_temp:
            raise RuntimeError("RUNNER_TEMP is unset; pass --dest-dir")
        dest_dir = Path(runner_temp) / DEST_DIRNAME

    if writer:
        staged = stage_writer_token(fetch_writer_token(), dest_dir)
    else:
        staged = materialize_credentials(args.source_dir, dest_dir)
    gha_set_output({"credentials_dir": str(staged) if staged else ""})


if __name__ == "__main__":
    main()
