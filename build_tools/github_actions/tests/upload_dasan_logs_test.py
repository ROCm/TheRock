# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import io
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _therock_utils.s3_buckets import S3BucketConfig
from github_actions import upload_dasan_logs as uploader


class UploadDASanLogsTest(unittest.TestCase):
    def setUp(self):
        self.environment = {
            "GITHUB_REPOSITORY": "ROCm/TheRock",
            "ACTIONS_ID_TOKEN_REQUEST_URL": "https://oidc.example/token?x=1&audience=old",
            "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "request-secret",
            "AWS_ACCESS_KEY_ID": "expired-key",
            "AWS_SECRET_ACCESS_KEY": "expired-secret",
            "AWS_SESSION_TOKEN": "expired-session",
        }
        self.config = S3BucketConfig("therock-ci-artifacts", iam_role="therock-ci")

    def test_each_snapshot_refreshes_and_limits_credentials_to_child(self):
        credentials = [
            {
                "AccessKeyId": f"key-{i}",
                "SecretAccessKey": "secret",
                "SessionToken": "session",
            }
            for i in range(2)
        ]
        with (
            mock.patch.dict(os.environ, self.environment, clear=True),
            mock.patch.object(
                uploader,
                "get_artifacts_bucket_config_for_workflow_run",
                return_value=self.config,
            ),
            mock.patch.object(
                uploader,
                "urlopen",
                side_effect=[
                    io.BytesIO(b'{"value":"oidc-1"}'),
                    io.BytesIO(b'{"value":"oidc-2"}'),
                ],
            ) as get,
            mock.patch.object(uploader.boto3, "client") as client,
            mock.patch.object(uploader, "collect_logs") as collect,
            mock.patch.object(uploader.subprocess, "run") as run,
        ):
            client.return_value.assume_role_with_web_identity.side_effect = [
                {"Credentials": c} for c in credentials
            ]
            for _ in range(2):
                uploader.upload_snapshot(Path("output"), "comm-libs", "gfx950", "123")
            self.assertEqual(os.environ["AWS_ACCESS_KEY_ID"], "expired-key")
            self.assertNotIn("GITHUB_ENV", os.environ)
        self.assertEqual(collect.call_count, 2)
        self.assertEqual(get.call_count, 2)
        request = get.call_args.args[0]
        self.assertEqual(
            parse_qs(urlsplit(request.full_url).query)["audience"],
            ["sts.amazonaws.com"],
        )
        self.assertEqual(request.get_header("Authorization"), "Bearer request-secret")
        self.assertIs(
            client.call_args.kwargs["config"].signature_version, uploader.UNSIGNED
        )
        for i, call in enumerate(run.call_args_list):
            self.assertEqual(call.kwargs["env"]["AWS_ACCESS_KEY_ID"], f"key-{i}")
            self.assertTrue(call.kwargs["check"])
            self.assertNotIn("secret", " ".join(call.args[0]))
        assume = client.return_value.assume_role_with_web_identity
        self.assertEqual(
            assume.call_args.kwargs["RoleArn"], self.config.write_access_iam_role
        )
        self.assertEqual(assume.call_args.kwargs["WebIdentityToken"], "oidc-2")

    def test_refresh_failure_does_not_upload_with_expired_credentials(self):
        with (
            mock.patch.dict(os.environ, self.environment, clear=True),
            mock.patch.object(
                uploader,
                "get_artifacts_bucket_config_for_workflow_run",
                return_value=self.config,
            ),
            mock.patch.object(
                uploader, "urlopen", side_effect=RuntimeError("sensitive-token")
            ),
            mock.patch.object(uploader, "collect_logs"),
            mock.patch.object(uploader.subprocess, "run") as run,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "^Unable to refresh DASan log upload credentials$"
            ):
                uploader.upload_snapshot(Path("output"), "math-libs", "gfx950", "123")
        run.assert_not_called()

    def test_missing_oidc_permission_fails_clearly(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(
                uploader,
                "get_artifacts_bucket_config_for_workflow_run",
                return_value=self.config,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "id-token: write"):
                uploader.fresh_upload_environment()

    def test_no_role_preserves_runner_credentials_policy(self):
        with (
            mock.patch.dict(os.environ, {"AWS_PROFILE": "runner"}, clear=True),
            mock.patch.object(
                uploader,
                "get_artifacts_bucket_config_for_workflow_run",
                return_value=S3BucketConfig("external", iam_role=None),
            ),
            mock.patch.object(uploader, "urlopen") as get,
        ):
            self.assertEqual(
                uploader.fresh_upload_environment(), {"AWS_PROFILE": "runner"}
            )
        get.assert_not_called()

    def test_upload_failure_propagates_to_caller(self):
        with (
            mock.patch.object(uploader, "collect_logs"),
            mock.patch.object(uploader, "fresh_upload_environment", return_value={}),
            mock.patch.object(
                uploader.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(9, ["upload"]),
            ),
        ):
            with self.assertRaises(subprocess.CalledProcessError) as raised:
                uploader.upload_snapshot(Path("output"), "debug-tools", "", "123")
        self.assertEqual(raised.exception.returncode, 9)


if __name__ == "__main__":
    unittest.main()
