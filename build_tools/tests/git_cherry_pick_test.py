# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
    def test_commit_resolution_failure_is_reported(self) -> None:
        ambiguous = subprocess.CompletedProcess(
            args=["git", "rev-parse"],
            returncode=128,
            stdout="",
            stderr="error: short object ID abcdef0 is ambiguous",
        )
        with patch("git_cherry_pick._run_git", return_value=ambiguous):
            with self.assertRaisesRegex(CherryPickError, "exactly one target commit"):
                resolve_commits(Path("/unused"), ["abcdef0"])


class PushSanitizationTest(unittest.TestCase):
    def test_push_requires_creation_confirmation_for_exact_refspec(self) -> None:
        head = "a" * 40
        branch = "shared/cherry-pick/therock-10.2-airdel-12"
        refspec = f"{head}:refs/heads/{branch}"
        remote_url = f"https://x-access-token:{TOKEN}@github.com/ROCm/target.git"
        for status_output in (
            "",
            f"*\t{head}:refs/heads/another-branch\t[new branch]\n",
            f"*\t{refspec}\n",
            f"*\t{refspec}\t[new branch]\n*\t{refspec}\t[new branch]\n",
            f"=\t{refspec}\t[up to date]\n",
        ):
            result = subprocess.CompletedProcess(
                args=["git", "push"],
                returncode=0,
                stdout=f"To {remote_url}\n{status_output}Done\n",
                stderr=f"diagnostic containing {TOKEN}",
            )
            with self.subTest(status_output=status_output):
                with patch(
                    "git_cherry_pick.subprocess.run", return_value=result
                ), self.assertRaises(CherryPickError) as raised:
                    push_branch(
                        repo=Path("/temporary/repo"),
                        remote_url=remote_url,
                        remote_description="ROCm/target",
                        head=head,
                        branch=branch,
                    )
                self.assertNotIn(TOKEN, str(raised.exception))
                self.assertNotIn("x-access-token", str(raised.exception))
                self.assertIn(branch, str(raised.exception))

    def test_push_failure_does_not_include_credentials(self) -> None:
        remote_url = f"https://x-access-token:{TOKEN}@github.com/ROCm/target.git"
        result = subprocess.CompletedProcess(
            args=["git", "push"],
            returncode=1,
            stdout="",
            stderr=f"fatal: URL included {TOKEN}",
        )
        with patch("git_cherry_pick.subprocess.run", return_value=result):
            with self.assertRaises(CherryPickError) as raised:
                push_branch(
                    repo=Path("/temporary/repo"),
                    remote_url=remote_url,
                    remote_description="ROCm/target",
                    head="a" * 40,
                    branch="shared/cherry-pick/therock-10.2-airdel-12",
                )
        self.assertNotIn(TOKEN, str(raised.exception))
        self.assertNotIn("x-access-token", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
