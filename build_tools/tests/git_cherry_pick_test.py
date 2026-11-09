# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import base64
from contextlib import redirect_stderr
import io
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


sys.path.insert(0, os.fspath(Path(__file__).resolve().parents[1]))

from git_cherry_pick import (
    CherryPickError,
    push_branch,
    resolve_commits,
)


TOKEN = "test-app-token"  # nosec B105


class GitPrimitiveValidationTest(unittest.TestCase):
    def test_commit_resolution_preserves_repeated_commits(self) -> None:
        resolved = subprocess.CompletedProcess(
            args=["git", "rev-parse"],
            returncode=0,
            stdout=f"{'a' * 40}\n",
            stderr="",
        )
        with patch("git_cherry_pick._run_git", return_value=resolved):
            self.assertEqual(
                resolve_commits(Path("/unused"), ["aaaaaaa", "aaaaaaaa"]),
                ["a" * 40, "a" * 40],
            )


class PushSanitizationTest(unittest.TestCase):
    def test_push_failure_reports_diagnostics_without_credentials(self) -> None:
        remote_url = "https://github.com/ROCm/target.git"
        encoded_credentials = base64.b64encode(
            f"x-access-token:{TOKEN}".encode("utf-8")
        ).decode("ascii")
        result = subprocess.CompletedProcess(
            args=["git", "push"],
            returncode=1,
            stdout="",
            stderr=(
                "remote: GH013: repository rule violations; credentials were "
                f"{TOKEN} and {encoded_credentials}"
            ),
        )
        with patch.dict(
            os.environ,
            {
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "core.attributesFile",
                "GIT_CONFIG_VALUE_0": os.devnull,
            },
            clear=True,
        ), patch("git_cherry_pick.subprocess.run", return_value=result) as run:
            stderr = io.StringIO()
            with redirect_stderr(stderr), self.assertRaises(CherryPickError) as raised:
                push_branch(
                    repo=Path("/temporary/repo"),
                    remote_url=remote_url,
                    remote_description="ROCm/target",
                    head="a" * 40,
                    branch="shared/cherry-pick/therock-10.2-airdel-12",
                    token=TOKEN,
                )
        command = run.call_args.args[0]
        command_text = "\0".join(command)
        environment = run.call_args.kwargs["env"]
        self.assertIn(remote_url, command)
        self.assertNotIn(TOKEN, command_text)
        self.assertNotIn(encoded_credentials, command_text)
        self.assertEqual(environment["GIT_CONFIG_COUNT"], "2")
        self.assertEqual(
            environment["GIT_CONFIG_KEY_1"],
            "http.https://github.com/.extraheader",
        )
        self.assertEqual(
            environment["GIT_CONFIG_VALUE_1"],
            f"AUTHORIZATION: basic {encoded_credentials}",
        )
        self.assertIn("GH013: repository rule violations", stderr.getvalue())
        self.assertNotIn(TOKEN, stderr.getvalue())
        self.assertNotIn(encoded_credentials, stderr.getvalue())
        self.assertNotIn(TOKEN, str(raised.exception))


if __name__ == "__main__":
    unittest.main()
