#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Collect and upload one DASan log snapshot with fresh job-scoped credentials.

Call this once per heartbeat. Credentials stay in the upload child's environment;
they are never written to logs, GITHUB_ENV, or shared AWS configuration files.
The workflow's final blocking log upload still runs independently.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import boto3
from botocore import UNSIGNED
from botocore.config import Config

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _therock_utils.s3_buckets import get_artifacts_bucket_config_for_workflow_run
from github_actions.collect_dasan_logs import collect_logs


def fresh_upload_environment() -> dict[str, str]:
    """Assume the same artifact role as the workflow's credential action."""
    environment = os.environ.copy()
    config = get_artifacts_bucket_config_for_workflow_run(
        github_repository=environment.get("GITHUB_REPOSITORY", "ROCm/TheRock"),
        release_type="ci",
    )
    role = config.write_access_iam_role
    if not role:
        # Preserve the existing fork policy: use runner base credentials.
        return environment

    request_url = environment.get("ACTIONS_ID_TOKEN_REQUEST_URL")
    request_token = environment.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
    if not request_url or not request_token:
        raise RuntimeError("DASan log refresh requires job permission id-token: write")
    parts = urlsplit(request_url)
    query = [(k, v) for k, v in parse_qsl(parts.query) if k != "audience"]
    query.append(("audience", "sts.amazonaws.com"))
    request = Request(
        urlunsplit(parts._replace(query=urlencode(query))),
        headers={"Authorization": f"Bearer {request_token}"},
    )
    try:
        with urlopen(request, timeout=20) as response:
            identity_token = json.load(response)["value"]
        # Web identity authentication must not depend on the expired credentials
        # inherited from the initial configure-aws-credentials step.
        sts = boto3.client(
            "sts",
            region_name=config.region,
            config=Config(
                signature_version=UNSIGNED,
                connect_timeout=20,
                read_timeout=20,
                retries={"mode": "standard", "total_max_attempts": 3},
            ),
        )
        credentials = sts.assume_role_with_web_identity(
            RoleArn=role,
            RoleSessionName="dasan-log-upload",
            WebIdentityToken=identity_token,
            DurationSeconds=3600,
        )["Credentials"]
        environment.update(
            AWS_ACCESS_KEY_ID=credentials["AccessKeyId"],
            AWS_SECRET_ACCESS_KEY=credentials["SecretAccessKey"],
            AWS_SESSION_TOKEN=credentials["SessionToken"],
            AWS_REGION=config.region,
            AWS_DEFAULT_REGION=config.region,
        )
    except Exception:
        # Authentication response bodies can contain sensitive data. Do not
        # propagate them to the heartbeat log (which may itself be uploaded).
        raise RuntimeError("Unable to refresh DASan log upload credentials") from None
    return environment


def upload_snapshot(output_dir: Path, stage: str, family: str, run_id: str) -> None:
    collect_logs(output_dir)
    environment = fresh_upload_environment()
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("post_stage_upload.py")),
            "--run-id",
            run_id,
            "--stage",
            stage,
            "--build-dir",
            str(output_dir / "build"),
            "--amdgpu-family",
            family,
        ],
        env=environment,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--amdgpu-family", default="")
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID"))
    args = parser.parse_args()
    if not args.run_id:
        parser.error("--run-id or GITHUB_RUN_ID is required")
    upload_snapshot(args.output_dir, args.stage, args.amdgpu_family, args.run_id)


if __name__ == "__main__":
    main()
