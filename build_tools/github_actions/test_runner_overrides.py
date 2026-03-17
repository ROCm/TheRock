# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for runner_overrides.py."""

import json
import os
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

import runner_overrides


class TestFetchOverrides(unittest.TestCase):
    """Tests for fetch_overrides() function."""

    def setUp(self):
        """Reset module cache before each test."""
        runner_overrides.reset_cache()
        # Clear any environment variables that might affect tests
        for env_var in [
            "THEROCK_RUNNER_OVERRIDE_URL",
            "THEROCK_DISABLE_RUNNER_OVERRIDES",
        ]:
            if env_var in os.environ:
                del os.environ[env_var]

    def tearDown(self):
        """Clean up after each test."""
        runner_overrides.reset_cache()
        for env_var in [
            "THEROCK_RUNNER_OVERRIDE_URL",
            "THEROCK_DISABLE_RUNNER_OVERRIDES",
        ]:
            if env_var in os.environ:
                del os.environ[env_var]

    @patch("runner_overrides.urlopen")
    def test_fetch_overrides_success(self, mock_urlopen):
        """Test successful fetch of overrides from S3."""
        override_data = {
            "version": 1,
            "overrides": {
                "gfx94x": {
                    "linux": {
                        "test-runs-on": "linux-mi325-1gpu-ossci-rocm",
                    }
                }
            },
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(override_data).encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = runner_overrides.fetch_overrides()

        self.assertEqual(result, override_data["overrides"])
        mock_urlopen.assert_called_once()

    @patch("runner_overrides.urlopen")
    def test_fetch_overrides_caches_result(self, mock_urlopen):
        """Test that fetch_overrides caches the result and doesn't fetch twice."""
        override_data = {"version": 1, "overrides": {"gfx94x": {"linux": {}}}}
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(override_data).encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        # First call
        result1 = runner_overrides.fetch_overrides()
        # Second call
        result2 = runner_overrides.fetch_overrides()

        self.assertEqual(result1, result2)
        # Should only call urlopen once due to caching
        self.assertEqual(mock_urlopen.call_count, 1)

    @patch("runner_overrides.urlopen")
    def test_fetch_overrides_http_error(self, mock_urlopen):
        """Test graceful handling of HTTP errors."""
        from urllib.error import HTTPError

        mock_urlopen.side_effect = HTTPError(
            url="http://test", code=404, msg="Not Found", hdrs={}, fp=None
        )

        result = runner_overrides.fetch_overrides()

        self.assertEqual(result, {})

    @patch("runner_overrides.urlopen")
    def test_fetch_overrides_url_error(self, mock_urlopen):
        """Test graceful handling of URL/network errors."""
        from urllib.error import URLError

        mock_urlopen.side_effect = URLError("Connection refused")

        result = runner_overrides.fetch_overrides()

        self.assertEqual(result, {})

    @patch("runner_overrides.urlopen")
    def test_fetch_overrides_timeout(self, mock_urlopen):
        """Test graceful handling of timeout errors."""
        mock_urlopen.side_effect = TimeoutError("Connection timed out")

        result = runner_overrides.fetch_overrides()

        self.assertEqual(result, {})

    @patch("runner_overrides.urlopen")
    def test_fetch_overrides_invalid_json(self, mock_urlopen):
        """Test graceful handling of invalid JSON response."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"not valid json"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = runner_overrides.fetch_overrides()

        self.assertEqual(result, {})

    @patch("runner_overrides.urlopen")
    def test_fetch_overrides_missing_overrides_key(self, mock_urlopen):
        """Test handling of valid JSON without 'overrides' key."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"version": 1}).encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = runner_overrides.fetch_overrides()

        self.assertEqual(result, {})

    def test_fetch_overrides_disabled(self):
        """Test that fetch is skipped when disabled via environment variable."""
        os.environ["THEROCK_DISABLE_RUNNER_OVERRIDES"] = "1"

        with patch("runner_overrides.urlopen") as mock_urlopen:
            result = runner_overrides.fetch_overrides()

            self.assertEqual(result, {})
            mock_urlopen.assert_not_called()

    @patch("runner_overrides.urlopen")
    def test_fetch_overrides_custom_url(self, mock_urlopen):
        """Test that custom URL can be specified via environment variable."""
        custom_url = "https://custom-bucket.s3.amazonaws.com/overrides.json"
        os.environ["THEROCK_RUNNER_OVERRIDE_URL"] = custom_url

        override_data = {"version": 1, "overrides": {}}
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(override_data).encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        runner_overrides.fetch_overrides()

        # Verify the custom URL was used
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        self.assertEqual(request.full_url, custom_url)


class TestApplyOverrides(unittest.TestCase):
    """Tests for apply_overrides() function."""

    def setUp(self):
        """Reset module cache before each test."""
        runner_overrides.reset_cache()
        for env_var in [
            "THEROCK_RUNNER_OVERRIDE_URL",
            "THEROCK_DISABLE_RUNNER_OVERRIDES",
        ]:
            if env_var in os.environ:
                del os.environ[env_var]

    def tearDown(self):
        """Clean up after each test."""
        runner_overrides.reset_cache()
        for env_var in [
            "THEROCK_RUNNER_OVERRIDE_URL",
            "THEROCK_DISABLE_RUNNER_OVERRIDES",
        ]:
            if env_var in os.environ:
                del os.environ[env_var]

    @patch("runner_overrides.fetch_overrides")
    def test_apply_overrides_sparse_merge(self, mock_fetch):
        """Test that overrides are sparsely merged (only specified fields change)."""
        mock_fetch.return_value = {
            "gfx94x": {
                "linux": {
                    "test-runs-on": "new-runner-label",
                }
            }
        }

        family_matrix = {
            "gfx94x": {
                "linux": {
                    "test-runs-on": "old-runner-label",
                    "test-runs-on-multi-gpu": "multi-gpu-runner",
                    "family": "gfx94X-dcgpu",
                }
            }
        }

        result = runner_overrides.apply_overrides(family_matrix)

        # Overridden field should change
        self.assertEqual(result["gfx94x"]["linux"]["test-runs-on"], "new-runner-label")
        # Non-overridden fields should remain
        self.assertEqual(
            result["gfx94x"]["linux"]["test-runs-on-multi-gpu"], "multi-gpu-runner"
        )
        self.assertEqual(result["gfx94x"]["linux"]["family"], "gfx94X-dcgpu")

    @patch("runner_overrides.fetch_overrides")
    def test_apply_overrides_does_not_mutate_original(self, mock_fetch):
        """Test that the original matrix is not mutated."""
        mock_fetch.return_value = {
            "gfx94x": {
                "linux": {
                    "test-runs-on": "new-runner-label",
                }
            }
        }

        original_value = "old-runner-label"
        family_matrix = {
            "gfx94x": {
                "linux": {
                    "test-runs-on": original_value,
                }
            }
        }

        result = runner_overrides.apply_overrides(family_matrix)

        # Result should have new value
        self.assertEqual(result["gfx94x"]["linux"]["test-runs-on"], "new-runner-label")
        # Original should be unchanged
        self.assertEqual(family_matrix["gfx94x"]["linux"]["test-runs-on"], original_value)

    @patch("runner_overrides.fetch_overrides")
    def test_apply_overrides_unknown_family_ignored(self, mock_fetch):
        """Test that overrides for unknown families are ignored."""
        mock_fetch.return_value = {
            "unknown_family": {
                "linux": {
                    "test-runs-on": "some-runner",
                }
            }
        }

        family_matrix = {
            "gfx94x": {
                "linux": {
                    "test-runs-on": "original-runner",
                }
            }
        }

        result = runner_overrides.apply_overrides(family_matrix)

        # Original should be unchanged, no new keys added
        self.assertEqual(result["gfx94x"]["linux"]["test-runs-on"], "original-runner")
        self.assertNotIn("unknown_family", result)

    @patch("runner_overrides.fetch_overrides")
    def test_apply_overrides_unknown_platform_ignored(self, mock_fetch):
        """Test that overrides for unknown platforms are ignored."""
        mock_fetch.return_value = {
            "gfx94x": {
                "unknown_platform": {
                    "test-runs-on": "some-runner",
                }
            }
        }

        family_matrix = {
            "gfx94x": {
                "linux": {
                    "test-runs-on": "original-runner",
                }
            }
        }

        result = runner_overrides.apply_overrides(family_matrix)

        # Original should be unchanged, no new platform added
        self.assertEqual(result["gfx94x"]["linux"]["test-runs-on"], "original-runner")
        self.assertNotIn("unknown_platform", result["gfx94x"])

    @patch("runner_overrides.fetch_overrides")
    def test_apply_overrides_empty_overrides(self, mock_fetch):
        """Test that empty overrides return unchanged matrix."""
        mock_fetch.return_value = {}

        family_matrix = {
            "gfx94x": {
                "linux": {
                    "test-runs-on": "original-runner",
                }
            }
        }

        result = runner_overrides.apply_overrides(family_matrix)

        self.assertEqual(result["gfx94x"]["linux"]["test-runs-on"], "original-runner")

    @patch("runner_overrides.fetch_overrides")
    def test_apply_overrides_multiple_families_and_platforms(self, mock_fetch):
        """Test applying overrides to multiple families and platforms."""
        mock_fetch.return_value = {
            "gfx94x": {
                "linux": {"test-runs-on": "new-linux-runner"},
                "windows": {"test-runs-on": "new-windows-runner"},
            },
            "gfx110x": {
                "linux": {"bypass_tests_for_releases": False},
            },
        }

        family_matrix = {
            "gfx94x": {
                "linux": {"test-runs-on": "old-linux-runner", "family": "gfx94X-dcgpu"},
                "windows": {"test-runs-on": "old-windows-runner"},
            },
            "gfx110x": {
                "linux": {
                    "test-runs-on": "gfx110x-runner",
                    "bypass_tests_for_releases": True,
                },
            },
        }

        result = runner_overrides.apply_overrides(family_matrix)

        self.assertEqual(result["gfx94x"]["linux"]["test-runs-on"], "new-linux-runner")
        self.assertEqual(
            result["gfx94x"]["windows"]["test-runs-on"], "new-windows-runner"
        )
        self.assertEqual(result["gfx110x"]["linux"]["bypass_tests_for_releases"], False)
        # Non-overridden field should remain
        self.assertEqual(result["gfx110x"]["linux"]["test-runs-on"], "gfx110x-runner")

    @patch("runner_overrides.fetch_overrides")
    def test_apply_overrides_can_add_new_fields(self, mock_fetch):
        """Test that overrides can add new fields to existing configs."""
        mock_fetch.return_value = {
            "gfx94x": {
                "linux": {
                    "new-field": "new-value",
                }
            }
        }

        family_matrix = {
            "gfx94x": {
                "linux": {
                    "test-runs-on": "runner",
                }
            }
        }

        result = runner_overrides.apply_overrides(family_matrix)

        self.assertEqual(result["gfx94x"]["linux"]["new-field"], "new-value")
        self.assertEqual(result["gfx94x"]["linux"]["test-runs-on"], "runner")

    @patch("runner_overrides.fetch_overrides")
    def test_apply_overrides_handles_invalid_override_structure(self, mock_fetch):
        """Test that invalid override structures are handled gracefully."""
        # family_overrides is not a dict
        mock_fetch.return_value = {
            "gfx94x": "invalid_not_a_dict",
        }

        family_matrix = {
            "gfx94x": {
                "linux": {
                    "test-runs-on": "original-runner",
                }
            }
        }

        result = runner_overrides.apply_overrides(family_matrix)

        # Should not crash, original value preserved
        self.assertEqual(result["gfx94x"]["linux"]["test-runs-on"], "original-runner")

    @patch("runner_overrides.fetch_overrides")
    def test_apply_overrides_handles_invalid_platform_structure(self, mock_fetch):
        """Test that invalid platform override structures are handled gracefully."""
        # platform_overrides is not a dict
        mock_fetch.return_value = {
            "gfx94x": {
                "linux": "invalid_not_a_dict",
            }
        }

        family_matrix = {
            "gfx94x": {
                "linux": {
                    "test-runs-on": "original-runner",
                }
            }
        }

        result = runner_overrides.apply_overrides(family_matrix)

        # Should not crash, original value preserved
        self.assertEqual(result["gfx94x"]["linux"]["test-runs-on"], "original-runner")


class TestResetCache(unittest.TestCase):
    """Tests for reset_cache() function."""

    def test_reset_cache_allows_refetch(self):
        """Test that reset_cache allows fetching again."""
        with patch("runner_overrides.urlopen") as mock_urlopen:
            override_data = {"version": 1, "overrides": {"v1": {}}}
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(override_data).encode("utf-8")
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            # First fetch
            runner_overrides.fetch_overrides()
            self.assertEqual(mock_urlopen.call_count, 1)

            # Reset cache
            runner_overrides.reset_cache()

            # Update mock to return different data
            override_data2 = {"version": 1, "overrides": {"v2": {}}}
            mock_response.read.return_value = json.dumps(override_data2).encode("utf-8")

            # Second fetch after reset
            result = runner_overrides.fetch_overrides()
            self.assertEqual(mock_urlopen.call_count, 2)
            self.assertEqual(result, {"v2": {}})


if __name__ == "__main__":
    unittest.main()
