#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for _therock_utils.log_utils module.

Tests cover:
- Configurable verbosity levels
- File output configuration
- Subprocess output capture (cross-platform: Linux and Windows)
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
    capture_console,
    configure_logging,
    disable_logger,
    set_verbosity,
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

    def test_captures_subprocess_and_python_output(self):
        """capture_console captures Python prints and subprocess stdout/stderr.

        Works on both Linux (via fd redirection) and Windows (via SetStdHandle).
        """
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


if __name__ == "__main__":
    unittest.main()
