#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for _therock_utils.log_utils module.

Tests cover:
- Configurable verbosity levels
- File output configuration
- Subprocess output capture (cross-platform: Linux and Windows)
- TheRockLogger class with GitHub annotation support

Run tests::

    # From TheRock root directory
    python -m pytest build_tools/tests/log_utils_test.py -v

    # Run specific test class
    python -m pytest build_tools/tests/log_utils_test.py::TheRockLoggerTest -v

    # Run with unittest
    python -m unittest build_tools.tests.log_utils_test
"""

import io
import logging
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from _therock_utils.log_utils import (
    ENV_LOG_ENABLED,
    ENV_LOG_LEVEL,
    ENV_GITHUB_ACTIONS,
    capture_console,
    configure_logging,
    disable_logger,
    set_verbosity,
    TheRockLogger,
    vlog,
)


class ConfigureLoggingTest(unittest.TestCase):
    """Tests for configure_logging function."""

    def setUp(self):
        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(logging.WARNING)
        logging.disable(logging.NOTSET)

        self._env_backup = {}
        for key in [ENV_LOG_ENABLED, ENV_LOG_LEVEL]:
            self._env_backup[key] = os.environ.pop(key, None)

    def tearDown(self):
        for key, value in self._env_backup.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]
        logging.disable(logging.NOTSET)

    def test_default_and_custom_levels(self):
        """configure_logging sets INFO by default, respects custom level."""
        stream = io.StringIO()
        configure_logging(stream=stream)
        self.assertEqual(logging.getLogger().level, logging.INFO)

        configure_logging(level=logging.WARNING, stream=stream)
        self.assertEqual(logging.getLogger().level, logging.WARNING)

    def test_verbose_enables_debug(self):
        """configure_logging with verbose=True enables DEBUG level."""
        stream = io.StringIO()
        configure_logging(verbose=True, stream=stream)
        self.assertEqual(logging.getLogger().level, logging.DEBUG)

    def test_env_level_override(self):
        """THEROCK_LOG_LEVEL environment variable overrides level parameter."""
        os.environ[ENV_LOG_LEVEL] = "WARNING"
        stream = io.StringIO()
        configure_logging(level=logging.DEBUG, stream=stream)
        self.assertEqual(logging.getLogger().level, logging.WARNING)

    def test_enabled_false_disables(self):
        """configure_logging with enabled=False disables all logging."""
        stream = io.StringIO()
        configure_logging(enabled=False, stream=stream)

        logging.getLogger("test").error("This should not appear")
        self.assertEqual(stream.getvalue(), "")


class VerbosityTest(unittest.TestCase):
    """Tests for set_verbosity and vlog functions."""

    def setUp(self):
        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(logging.WARNING)
        logging.disable(logging.NOTSET)

        self.stream = io.StringIO()
        configure_logging(stream=self.stream)

    def tearDown(self):
        logging.disable(logging.NOTSET)
        set_verbosity(0)

    def test_verbosity_levels(self):
        """set_verbosity controls logging level: 0=INFO, 1+=DEBUG, -1=disabled."""
        set_verbosity(0)
        self.assertEqual(logging.getLogger().level, logging.INFO)

        set_verbosity(1)
        self.assertEqual(logging.getLogger().level, logging.DEBUG)

        set_verbosity(-1)
        logging.getLogger("test").error("should not appear")
        self.assertEqual(self.stream.getvalue(), "")

    def test_vlog_respects_verbosity_threshold(self):
        """vlog only outputs when verbosity >= message level."""
        set_verbosity(1)

        vlog("level 0 message", level=0)
        vlog("level 1 message", level=1)
        vlog("level 2 message", level=2)

        output = self.stream.getvalue()
        self.assertIn("level 0 message", output)
        self.assertIn("level 1 message", output)
        self.assertNotIn("level 2 message", output)


class DisableLoggerTest(unittest.TestCase):
    """Tests for disable_logger function."""

    def setUp(self):
        root = logging.getLogger()
        root.handlers.clear()
        logging.disable(logging.NOTSET)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_disable_logger_silences_named_logger(self):
        """disable_logger silences the specified logger."""
        stream = io.StringIO()
        configure_logging(stream=stream)

        logger = logging.getLogger("noisy.module")
        logger.info("before disable")
        self.assertIn("before disable", stream.getvalue())

        stream.truncate(0)
        stream.seek(0)

        disable_logger("noisy.module")
        logger.info("after disable")
        self.assertEqual(stream.getvalue(), "")


class CaptureConsoleTest(unittest.TestCase):
    """Tests for capture_console context manager."""

    def test_file_output_configuration(self):
        """capture_console creates log file and parent directories."""
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "subdir" / "build.log"

            with capture_console(log_path, also_to_console=False):
                print("Build output", flush=True)

            self.assertTrue(log_path.exists())
            self.assertIn("Build output", log_path.read_text())

    def test_disabled_skips_capture(self):
        """capture_console skips capture when disabled via param or env."""
        with tempfile.TemporaryDirectory() as tmp:
            log1 = Path(tmp) / "test1.log"
            log2 = Path(tmp) / "test2.log"

            with capture_console(log1, enabled=False):
                print("should not be captured")
            self.assertFalse(log1.exists())

            with mock.patch.dict(os.environ, {ENV_LOG_ENABLED: "0"}):
                with capture_console(log2):
                    print("should not be captured")
            self.assertFalse(log2.exists())

    def test_basic_capture_writes_to_file(self):
        """capture_console writes output to file (platform-agnostic public API)."""
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "build.log"

            with capture_console(log_path):
                print("hello", flush=True)

            self.assertIn("hello", log_path.read_text())

    @unittest.skipIf(sys.platform == "win32", "POSIX-only path")
    def test_posix_capture(self):
        """capture_console captures subprocess output on POSIX via fd redirection."""
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "build.log"

            with capture_console(log_path):
                print("Python output", flush=True)
                subprocess.run(
                    [sys.executable, "-c", "print('subprocess stdout')"],
                    check=True,
                )
                subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import sys; sys.stderr.write('subprocess stderr\\n')",
                    ],
                    check=True,
                )

            content = log_path.read_text()
            self.assertIn("Python output", content)
            self.assertIn("subprocess stdout", content)
            self.assertIn("subprocess stderr", content)

    @unittest.skipUnless(sys.platform == "win32", "Windows-only path")
    def test_windows_capture(self):
        """capture_console captures subprocess output on Windows via SetStdHandle."""
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "build.log"

            with capture_console(log_path):
                print("Python output", flush=True)
                subprocess.run(
                    [sys.executable, "-c", "print('subprocess stdout')"],
                    check=True,
                )
                subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import sys; sys.stderr.write('subprocess stderr\\n')",
                    ],
                    check=True,
                )

            content = log_path.read_text()
            self.assertIn("Python output", content)
            self.assertIn("subprocess stdout", content)
            self.assertIn("subprocess stderr", content)


class TheRockLoggerTest(unittest.TestCase):
    """Tests for TheRockLogger class."""

    def setUp(self):
        root = logging.getLogger()
        root.handlers.clear()
        logging.disable(logging.NOTSET)

        self.stream = io.StringIO()
        configure_logging(stream=self.stream)

        self._env_backup = os.environ.get(ENV_GITHUB_ACTIONS)
        os.environ.pop(ENV_GITHUB_ACTIONS, None)

    def tearDown(self):
        logging.disable(logging.NOTSET)
        if self._env_backup is not None:
            os.environ[ENV_GITHUB_ACTIONS] = self._env_backup
        else:
            os.environ.pop(ENV_GITHUB_ACTIONS, None)

    def test_logger_methods(self):
        """TheRockLogger provides info/warning/error methods."""
        logger = TheRockLogger("test_module")

        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")

        output = self.stream.getvalue()
        self.assertIn("info message", output)
        self.assertIn("warning message", output)
        self.assertIn("error message", output)

    def test_logger_with_github_annotation(self):
        """TheRockLogger supports github=True for annotations."""
        os.environ[ENV_GITHUB_ACTIONS] = "true"
        logger = TheRockLogger("test_module")

        with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            logger.warning("GitHub warning", github=True)
            logger.error("GitHub error", github=True, file="test.py", line=5)

            output = mock_stdout.getvalue()
            self.assertIn("::warning::GitHub warning", output)
            self.assertIn("::error file=test.py,line=5::GitHub error", output)

    def test_logger_github_skipped_when_not_ci(self):
        """TheRockLogger skips GitHub annotation when not in CI."""
        os.environ.pop(ENV_GITHUB_ACTIONS, None)
        logger = TheRockLogger("test_module")

        with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            logger.warning("should not appear", github=True)
            self.assertEqual(mock_stdout.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
