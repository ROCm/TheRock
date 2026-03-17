# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for runner_overrides.py."""

import json
import os
import unittest
from unittest.mock import MagicMock, patch

import runner_overrides


class TestRunnerOverrides(unittest.TestCase):
    """Tests for runner_overrides module."""

    def setUp(self):
        runner_overrides.reset_cache()
        os.environ.pop("THEROCK_RUNNER_OVERRIDE_URL", None)
        os.environ.pop("THEROCK_DISABLE_RUNNER_OVERRIDES", None)

    def tearDown(self):
        runner_overrides.reset_cache()
        os.environ.pop("THEROCK_RUNNER_OVERRIDE_URL", None)
        os.environ.pop("THEROCK_DISABLE_RUNNER_OVERRIDES", None)

    def _mock_urlopen(self, mock, data):
        """Helper to set up urlopen mock with given data."""
        resp = MagicMock()
        resp.read.return_value = json.dumps(data).encode("utf-8")
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock.return_value = resp

    @patch("runner_overrides.urlopen")
    def test_fetch_success_and_caching(self, mock_urlopen):
        """Test successful fetch and result caching."""
        self._mock_urlopen(mock_urlopen, {"overrides": {"gfx94x": {"linux": {}}}})

        result1 = runner_overrides.fetch_overrides()
        result2 = runner_overrides.fetch_overrides()

        self.assertEqual(result1, {"gfx94x": {"linux": {}}})
        self.assertEqual(result1, result2)
        self.assertEqual(mock_urlopen.call_count, 1)  # Cached

    @patch("runner_overrides.urlopen")
    def test_fetch_errors_return_empty(self, mock_urlopen):
        """Test that network/parse errors return empty dict."""
        from urllib.error import URLError

        mock_urlopen.side_effect = URLError("Connection refused")
        self.assertEqual(runner_overrides.fetch_overrides(), {})

    def test_fetch_disabled(self):
        """Test fetch skipped when disabled."""
        os.environ["THEROCK_DISABLE_RUNNER_OVERRIDES"] = "true"
        with patch("runner_overrides.urlopen") as mock:
            self.assertEqual(runner_overrides.fetch_overrides(), {})
            mock.assert_not_called()

    @patch("runner_overrides.fetch_overrides")
    def test_apply_sparse_merge(self, mock_fetch):
        """Test overrides are sparsely merged without mutating original."""
        mock_fetch.return_value = {"gfx94x": {"linux": {"test-runs-on": "new-runner"}}}
        original = {
            "gfx94x": {
                "linux": {"test-runs-on": "old-runner", "family": "gfx94X-dcgpu"}
            }
        }

        result = runner_overrides.apply_overrides(original)

        # Override applied
        self.assertEqual(result["gfx94x"]["linux"]["test-runs-on"], "new-runner")
        # Other fields preserved
        self.assertEqual(result["gfx94x"]["linux"]["family"], "gfx94X-dcgpu")
        # Original unchanged
        self.assertEqual(original["gfx94x"]["linux"]["test-runs-on"], "old-runner")

    @patch("runner_overrides.fetch_overrides")
    def test_apply_ignores_unknown_families(self, mock_fetch):
        """Test unknown families/platforms in overrides are ignored."""
        mock_fetch.return_value = {"unknown": {"linux": {"test-runs-on": "x"}}}
        original = {"gfx94x": {"linux": {"test-runs-on": "runner"}}}

        result = runner_overrides.apply_overrides(original)

        self.assertNotIn("unknown", result)
        self.assertEqual(result["gfx94x"]["linux"]["test-runs-on"], "runner")


if __name__ == "__main__":
    unittest.main()
