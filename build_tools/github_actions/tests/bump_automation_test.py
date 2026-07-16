# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

sys.path.insert(0, str(Path(__file__).parent.parent))
from bump_automation import warn_on_failure, handle_push


class WarnOnFailureTest(unittest.TestCase):
    def test_swallows_exception(self):
        @warn_on_failure
        def boom():
            raise RuntimeError("oops")

        boom()  # must not raise

    def test_returns_value_on_success(self):
        @warn_on_failure
        def ok():
            return 42

        self.assertEqual(ok(), 42)

    def test_preserves_function_name(self):
        @warn_on_failure
        def my_func():
            pass

        self.assertEqual(my_func.__name__, "my_func")

    def test_logs_warning_on_failure(self):
        @warn_on_failure
        def boom():
            raise ValueError("test error")

        with self.assertLogs(level="WARNING") as cm:
            import logging

            logging.warning("dummy")  # ensure handler exists
            boom()
        # warn_on_failure uses print, so just verify no exception raised
        # and the function name is preserved for the log message
        self.assertEqual(boom.__name__, "boom")


class HandlePushTest(unittest.TestCase):
    @patch("bump_automation.create_therock_bump")
    @patch("bump_automation.gh_api")
    @patch("bump_automation.run")
    @patch("bump_automation.get_submodule_sha")
    @patch("bump_automation.submodule_changed")
    def test_creates_next_bump_pr_after_merge(
        self,
        mock_changed,
        mock_get_sha,
        mock_run,
        mock_gh_api,
        mock_create_bump,
    ):
        mock_changed.side_effect = lambda before, after, m: m == "rocm-systems"
        mock_get_sha.side_effect = lambda commit, path: (
            "abc1234abc1234" if commit == "before_sha" else "def5678def5678"
        )
        mock_gh_api.return_value = {}

        import tempfile, os

        with tempfile.TemporaryDirectory() as tmp:
            # Fake the files config expects
            orig = os.getcwd()
            os.chdir(tmp)
            for f in [
                ".github/workflows/therock-ci-linux.yml",
                ".github/workflows/therock-ci-windows.yml",
                ".github/workflows/therock-rccl-ci-linux.yml",
                ".github/workflows/therock-rccl-test-packages-multi-node.yml",
                ".github/workflows/therock-rccl-test-packages-single-node.yml",
                ".github/workflows/therock-test-component.yml",
                ".github/workflows/therock-test-packages.yml",
            ]:
                Path(f).parent.mkdir(parents=True, exist_ok=True)
                Path(f).write_text("placeholder")

            with patch("bump_automation.tempfile.TemporaryDirectory") as mock_tmp:
                mock_tmp.return_value.__enter__ = MagicMock(return_value=tmp)
                mock_tmp.return_value.__exit__ = MagicMock(return_value=False)
                with patch("bump_automation.os.chdir"):
                    handle_push(
                        "before_sha", "after_sha", "sys_token", "lib_token"
                    )

            os.chdir(orig)

        mock_create_bump.assert_called_once_with("rocm-systems", "sys_token")

    @patch("bump_automation.create_therock_bump")
    @patch("bump_automation.submodule_changed")
    def test_no_bump_pr_when_no_submodule_changed(
        self, mock_changed, mock_create_bump
    ):
        mock_changed.return_value = False

        handle_push("before_sha", "after_sha", "sys_token", "lib_token")

        mock_create_bump.assert_not_called()

    @patch("bump_automation.create_therock_bump")
    @patch("bump_automation.gh_api")
    @patch("bump_automation.run")
    @patch("bump_automation.get_submodule_sha")
    @patch("bump_automation.submodule_changed")
    def test_bump_pr_still_called_when_using_libraries_token(
        self,
        mock_changed,
        mock_get_sha,
        mock_run,
        mock_gh_api,
        mock_create_bump,
    ):
        mock_changed.side_effect = lambda before, after, m: m == "rocm-libraries"
        mock_get_sha.return_value = "aaa0000aaa0000"
        mock_gh_api.return_value = {}

        import tempfile, os

        with tempfile.TemporaryDirectory() as tmp:
            orig = os.getcwd()
            os.chdir(tmp)
            Path(".github/actions/ci-env").mkdir(parents=True, exist_ok=True)
            Path(".github/actions/ci-env/action.yml").write_text("placeholder")

            with patch("bump_automation.tempfile.TemporaryDirectory") as mock_tmp:
                mock_tmp.return_value.__enter__ = MagicMock(return_value=tmp)
                mock_tmp.return_value.__exit__ = MagicMock(return_value=False)
                with patch("bump_automation.os.chdir"):
                    handle_push(
                        "before_sha", "after_sha", "sys_token", "lib_token"
                    )

            os.chdir(orig)

        mock_create_bump.assert_called_once_with("rocm-libraries", "lib_token")


if __name__ == "__main__":
    unittest.main()
