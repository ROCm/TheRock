# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import io
import time
import unittest
from collections.abc import Iterable
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from build_tools.github_actions.monitor_workflow_run import (
    GitHub,
    MonitorError,
    UncertainRerun,
    monitor,
    rate_limit_delay,
    main,
)


def run(
    attempt: int = 1, conclusion: str | None = None, status: str = "completed"
) -> dict[str, object]:
    return dict(run_attempt=attempt, conclusion=conclusion, status=status)


class FakeAPI:
    def __init__(
        self, states: Iterable[dict[str, object]], uncertain: bool = False
    ) -> None:
        self.states = iter(states)
        self.posts = 0
        self.pauses: list[float] = []
        self.uncertain = uncertain

    def request(self, method: str = "GET") -> dict[str, object]:
        if method == "POST":
            self.posts += 1
            if self.uncertain:
                raise UncertainRerun("response lost")
            return {}
        return next(self.states)

    def pause(self, seconds: float) -> None:
        self.pauses.append(seconds)


class MonitorTests(unittest.TestCase):
    def test_initial_success(self) -> None:
        api = FakeAPI([run(conclusion="success")])
        self.assertEqual(monitor(api, 2, 60), 0)
        self.assertEqual(api.posts, 0)

    def test_retry_success_waits_for_new_attempt(self) -> None:
        api = FakeAPI(
            [
                run(conclusion="failure"),
                run(conclusion="failure"),
                run(2, status="queued"),
                run(2, "success"),
            ]
        )
        self.assertEqual(monitor(api, 2, 60), 0)
        self.assertEqual(api.posts, 1)
        self.assertEqual(api.pauses, [10, 60])

    def test_exhausted(self) -> None:
        api = FakeAPI([run(1, "failure"), run(2, "timed_out"), run(3, "failure")])
        self.assertEqual(monitor(api, 2, 60), 1)
        self.assertEqual(api.posts, 2)

    def test_zero_retries(self) -> None:
        api = FakeAPI([run(conclusion="failure")])
        self.assertEqual(monitor(api, 0, 60), 1)
        self.assertEqual(api.posts, 0)

    def test_cancelled(self) -> None:
        api = FakeAPI([run(conclusion="cancelled")])
        with self.assertRaises(MonitorError):
            monitor(api, 2, 60)
        self.assertEqual(api.posts, 0)

    def test_uncertain_post_not_repeated(self) -> None:
        api = FakeAPI(
            [run(1, "failure"), run(1, "failure"), run(2, "success")], uncertain=True
        )
        self.assertEqual(monitor(api, 2, 60), 0)
        self.assertEqual(api.posts, 1)

    def test_external_rerun(self) -> None:
        api = FakeAPI([run(1, status="in_progress"), run(2, "success")])
        with self.assertRaises(MonitorError):
            monitor(api, 2, 60)

    def test_registration_timeout(self) -> None:
        api = FakeAPI([run(1, "failure"), run(1, "failure")])
        with patch(
            "build_tools.github_actions.monitor_workflow_run.time.monotonic",
            side_effect=[0, 301],
        ):
            with self.assertRaisesRegex(MonitorError, "did not register"):
                monitor(api, 2, 60)
        self.assertEqual(api.posts, 1)

    def test_deadline(self) -> None:
        api = GitHub("owner/repo", "123", "test-token", time.monotonic() - 1)
        with self.assertRaisesRegex(MonitorError, "deadline"):
            api.request()

    def test_rate_limit_headers(self) -> None:
        self.assertEqual(rate_limit_delay({"Retry-After": "120"}, ""), 120)
        with patch(
            "build_tools.github_actions.monitor_workflow_run.time.time",
            return_value=100,
        ):
            self.assertEqual(
                rate_limit_delay(
                    {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "200"}, ""
                ),
                105,
            )
        self.assertIsNone(
            rate_limit_delay({}, "Resource not accessible by integration")
        )

    def test_permission_error_is_not_retried(self) -> None:
        api = GitHub("owner/repo", "123", "test-token", time.monotonic() + 1000)
        error = HTTPError(api.url, 403, "Forbidden", {}, io.BytesIO(b"access denied"))
        with patch(
            "build_tools.github_actions.monitor_workflow_run.urlopen", side_effect=error
        ) as request:
            with self.assertRaisesRegex(MonitorError, "HTTP 403"):
                api.request()
            self.assertEqual(request.call_count, 1)

    def test_rate_limited_read_recovers(self) -> None:
        api = GitHub("owner/repo", "123", "test-token", time.monotonic() + 1000)
        error = HTTPError(
            api.url, 403, "Forbidden", {"Retry-After": "60"}, io.BytesIO(b"rate limit")
        )
        response = unittest.mock.MagicMock()
        response.__enter__.return_value.read.return_value = b'{"status":"completed"}'
        with patch(
            "build_tools.github_actions.monitor_workflow_run.urlopen",
            side_effect=[error, response],
        ):
            with patch.object(api, "pause") as pause:
                self.assertEqual(api.request()["status"], "completed")
                pause.assert_called_once_with(60)

    def test_network_error_on_post_is_uncertain(self) -> None:
        api = GitHub("owner/repo", "123", "test-token", time.monotonic() + 1000)
        with patch(
            "build_tools.github_actions.monitor_workflow_run.urlopen",
            side_effect=URLError("lost"),
        ) as request:
            with self.assertRaises(UncertainRerun):
                api.request("POST")
            self.assertEqual(request.call_count, 1)

    def test_rate_limited_post_recovers(self) -> None:
        api = GitHub("owner/repo", "123", "test-token", time.monotonic() + 1000)
        error = HTTPError(
            api.url, 403, "Forbidden", {"Retry-After": "60"}, io.BytesIO(b"rate limit")
        )
        response = unittest.mock.MagicMock()
        response.__enter__.return_value.read.return_value = b""
        with patch(
            "build_tools.github_actions.monitor_workflow_run.urlopen",
            side_effect=[error, response],
        ) as request:
            with patch.object(api, "pause") as pause:
                self.assertEqual(api.request("POST"), {})
                self.assertEqual(request.call_count, 2)
                pause.assert_called_once_with(60)

    def test_429_without_headers_recovers(self) -> None:
        api = GitHub("owner/repo", "123", "test-token", time.monotonic() + 1000)
        error = HTTPError(api.url, 429, "Too Many Requests", {}, io.BytesIO(b""))
        response = unittest.mock.MagicMock()
        response.__enter__.return_value.read.return_value = b"{}"
        with patch(
            "build_tools.github_actions.monitor_workflow_run.urlopen",
            side_effect=[error, response],
        ):
            with patch.object(api, "pause") as pause:
                self.assertEqual(api.request(), {})
                pause.assert_called_once_with(10)

    def test_transient_get_recovers(self) -> None:
        api = GitHub("owner/repo", "123", "test-token", time.monotonic() + 1000)
        error = HTTPError(api.url, 503, "Unavailable", {}, io.BytesIO(b""))
        response = unittest.mock.MagicMock()
        response.__enter__.return_value.read.return_value = b"{}"
        with patch(
            "build_tools.github_actions.monitor_workflow_run.urlopen",
            side_effect=[error, URLError("lost"), response],
        ):
            with patch.object(api, "pause") as pause:
                self.assertEqual(api.request(), {})
                self.assertEqual(
                    pause.call_args_list,
                    [unittest.mock.call(10), unittest.mock.call(20)],
                )

    def test_server_error_post_not_repeated(self) -> None:
        api = GitHub("owner/repo", "123", "test-token", time.monotonic() + 1000)
        error = HTTPError(api.url, 503, "Unavailable", {}, io.BytesIO(b""))
        with patch(
            "build_tools.github_actions.monitor_workflow_run.urlopen", side_effect=error
        ) as request:
            with self.assertRaises(UncertainRerun):
                api.request("POST")
            self.assertEqual(request.call_count, 1)

    def test_invalid_cli(self) -> None:
        base = ["--repository", "owner/repo", "--run-id", "123"]
        for extra in [
            ["--max-retries", "-1"],
            ["--max-retries", "abc"],
            ["--poll-interval", "0"],
            ["--timeout-minutes", "351"],
            ["--run-id", "0"],
            ["--repository", "bad"],
        ]:
            with self.subTest(extra=extra), patch(
                "sys.stderr", new_callable=io.StringIO
            ):
                with self.assertRaises(SystemExit) as error:
                    main(base + extra)
                self.assertEqual(error.exception.code, 2)

    def test_cli_token_and_defaults(self) -> None:
        with patch.dict("os.environ", {"GITHUB_TOKEN": "test-token"}, clear=True):
            with patch(
                "build_tools.github_actions.monitor_workflow_run.monitor",
                return_value=0,
            ) as handler:
                self.assertEqual(
                    main(["--repository", "owner/repo", "--run-id", "123"]), 0
                )
                self.assertEqual(handler.call_args.args[1:], (2, 60))
                self.assertEqual(handler.call_args.args[0].token, "test-token")

    def test_missing_token(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(main(["--repository", "owner/repo", "--run-id", "123"]), 1)

    def test_malformed_rate_limit_headers(self) -> None:
        self.assertEqual(rate_limit_delay({"Retry-After": "nonsense"}, ""), 60)
        self.assertEqual(
            rate_limit_delay(
                {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "nonsense"}, ""
            ),
            60,
        )


if __name__ == "__main__":
    unittest.main()
