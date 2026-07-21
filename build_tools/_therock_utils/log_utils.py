"""Unified logging utilities for TheRock build tools.

Provides a consistent logging API across all Python scripts with:
- Console output capture to files (Python print + subprocess output)
- CI-friendly formatting (GitHub Actions groups, annotations)
- Verbosity control via environment variables or API
- Thread-safe stream handling

Usage::

    from _therock_utils.log_utils import configure_logging, get_logger, capture_console

    # One-time setup in main()
    configure_logging(verbose=args.verbose)

    # Get logger for module
    logger = get_logger(__name__)
    logger.info("Processing artifacts")

    # Capture all output to file
    with capture_console("build.log"):
        logger.info("Building...")
        subprocess.run(["cmake", ".."])  # Also captured
"""

import io
import logging
import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_FORMAT = "[%(asctime)s][%(levelname)-8s]: %(message)s"
DEFAULT_DATE_FORMAT = "%y-%m-%d %H:%M:%S"
CI_FORMAT = "[%(levelname)-8s] %(name)s: %(message)s"

# Environment variables
ENV_LOG_ENABLED = "THEROCK_LOG_ENABLED"
ENV_LOG_LEVEL = "THEROCK_LOG_LEVEL"
ENV_GITHUB_ACTIONS = "GITHUB_ACTIONS"

# Module state
_configured = False
_verbosity = 0
_capture_depth = threading.local()  # Per-thread capture depth counter
_MAX_CAPTURE_DEPTH = 3


def _get_capture_depth() -> int:
    """Get capture depth for current thread."""
    return getattr(_capture_depth, "value", 0)


def _increment_capture_depth() -> int:
    """Increment and return capture depth for current thread."""
    depth = getattr(_capture_depth, "value", 0) + 1
    _capture_depth.value = depth
    return depth


def _decrement_capture_depth() -> None:
    """Decrement capture depth for current thread."""
    _capture_depth.value = max(0, getattr(_capture_depth, "value", 0) - 1)


# ---------------------------------------------------------------------------
# CI Detection
# ---------------------------------------------------------------------------


def is_ci() -> bool:
    """Check if running in CI (GitHub Actions)."""
    return os.environ.get(ENV_GITHUB_ACTIONS, "").lower() == "true"


def is_logging_enabled() -> bool:
    """Check if logging is globally enabled."""
    return os.environ.get(ENV_LOG_ENABLED, "1") != "0"


# ---------------------------------------------------------------------------
# Flushing Handler
# ---------------------------------------------------------------------------


class FlushingStreamHandler(logging.StreamHandler):
    """StreamHandler that flushes after every emit for CI compatibility."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def configure_logging(
    *,
    enabled: bool | None = None,
    level: int | str = logging.INFO,
    verbose: bool = False,
    ci_mode: bool | None = None,
    stream: TextIO | None = None,
) -> None:
    """Configure root logger for script execution.

    Call once at the start of main(). Replaces logging.basicConfig().

    Args:
        enabled: Enable/disable logging. None = check THEROCK_LOG_ENABLED env.
        level: Base logging level (default INFO).
        verbose: If True, sets level to DEBUG.
        ci_mode: CI-specific formatting. None = auto-detect from GITHUB_ACTIONS.
        stream: Output stream (default sys.stderr).
    """
    global _configured

    # Check if globally disabled
    if enabled is None:
        enabled = is_logging_enabled()

    if not enabled:
        logging.disable(logging.CRITICAL)
        _configured = True
        return

    # Determine level
    env_level = os.environ.get(ENV_LOG_LEVEL, "").upper()
    if env_level:
        level = getattr(logging, env_level, logging.INFO)
    elif verbose:
        level = logging.DEBUG

    # Determine CI mode
    if ci_mode is None:
        ci_mode = is_ci()

    # Select format
    fmt = CI_FORMAT if ci_mode else DEFAULT_FORMAT
    datefmt = None if ci_mode else DEFAULT_DATE_FORMAT

    # Configure root logger
    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Add new handler
    handler = FlushingStreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Configured Logger instance.
    """
    if not _configured:
        configure_logging()
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Verbosity Control
# ---------------------------------------------------------------------------


def set_verbosity(level: int) -> None:
    """Set global verbosity level.

    Args:
        level: -1 = silent (disabled), 0 = normal (INFO), 1+ = verbose (DEBUG)
    """
    global _verbosity
    _verbosity = level

    if level < 0:
        logging.disable(logging.CRITICAL)
    else:
        logging.disable(logging.NOTSET)
        root = logging.getLogger()
        if level >= 1:
            root.setLevel(logging.DEBUG)
        else:
            root.setLevel(logging.INFO)


def vlog(message: str, *args, level: int = 0, **kwargs) -> None:
    """Log with verbosity level (py_packaging compatibility).

    Args:
        message: Log message.
        level: Minimum verbosity level required to show this message.
        *args, **kwargs: Passed to logger.debug().
    """
    if _verbosity >= level:
        logger = get_logger("therock")
        logger.debug(message, *args, **kwargs)


# ---------------------------------------------------------------------------
# TeeStream - Thread-safe dual output
# ---------------------------------------------------------------------------


class TeeStream:
    """Thread-safe stream that writes to two destinations.

    Used by capture_console() to write to both console and file.
    """

    def __init__(self, stream1: TextIO, stream2: TextIO):
        self.stream1 = stream1
        self.stream2 = stream2
        self._lock = threading.Lock()

    def write(self, data: str) -> int:
        with self._lock:
            if isinstance(data, bytes):
                data = data.decode("utf-8", errors="replace")

            self.stream1.write(data)
            self.stream2.write(data)
            self.stream1.flush()
            self.stream2.flush()
            return len(data)

    def flush(self) -> None:
        with self._lock:
            self.stream1.flush()
            self.stream2.flush()

    def fileno(self) -> int:
        """Return file descriptor for subprocess compatibility."""
        return self.stream1.fileno()

    def isatty(self) -> bool:
        return self.stream1.isatty()

    @property
    def encoding(self) -> str:
        return getattr(self.stream1, "encoding", "utf-8")

    @property
    def errors(self) -> str:
        return getattr(self.stream1, "errors", "replace")


# ---------------------------------------------------------------------------
# Console Capture with OS-level file descriptor redirection
# ---------------------------------------------------------------------------


class _TeeWriter(threading.Thread):
    """Background thread that reads from a pipe and writes to multiple destinations.

    This enables capturing subprocess output that writes directly to file descriptors.
    """

    def __init__(self, read_fd: int, outputs: list[TextIO], name: str = "TeeWriter"):
        super().__init__(daemon=True, name=name)
        self.read_fd = read_fd
        self.outputs = outputs
        self._stop_event = threading.Event()

    def run(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    data = os.read(self.read_fd, 4096)
                    if not data:
                        break
                    text = data.decode("utf-8", errors="replace")
                    for out in self.outputs:
                        try:
                            out.write(text)
                            out.flush()
                        except (IOError, ValueError):
                            pass
                except OSError:
                    break
        finally:
            try:
                os.close(self.read_fd)
            except OSError:
                pass

    def stop(self) -> None:
        self._stop_event.set()


@contextmanager
def capture_console(
    log_file: Path | str,
    *,
    also_to_console: bool = True,
    enabled: bool | None = None,
) -> Iterator[Path]:
    """Capture all console output to a file.

    Uses OS-level file descriptor redirection to capture:
    - Python print() statements
    - Python logging calls
    - Subprocess stdout/stderr (even direct fd writes)
    - C library output (printf, etc.)

    Supports nesting up to 3 levels for per-component + combined logs.

    Args:
        log_file: Path to output file.
        also_to_console: If True (default), output also appears on console.
        enabled: Enable/disable capture. None = check THEROCK_LOG_ENABLED env.

    Yields:
        Path to the log file.

    Example::

        with capture_console("build.log"):
            print("Building...")  # Goes to console AND build.log
            subprocess.run(["make"])  # Subprocess output also captured
    """
    # Check if enabled
    if enabled is None:
        enabled = is_logging_enabled()

    log_path = Path(log_file)

    if not enabled:
        yield log_path
        return

    # Check nesting depth (thread-safe via thread-local storage)
    if _get_capture_depth() >= _MAX_CAPTURE_DEPTH:
        log = get_logger(__name__)
        log.warning(
            f"capture_console: max nesting depth ({_MAX_CAPTURE_DEPTH}) reached, "
            f"skipping capture to {log_path}"
        )
        yield log_path
        return

    _increment_capture_depth()

    # Ensure parent directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Save original file descriptors
    orig_stdout_fd = os.dup(1)
    orig_stderr_fd = os.dup(2)
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    # Create pipes for capturing
    stdout_read_fd, stdout_write_fd = os.pipe()
    stderr_read_fd, stderr_write_fd = os.pipe()

    tee_threads = []
    log_fh = None

    try:
        log_fh = open(log_path, "w", encoding="utf-8")

        # Determine output destinations
        if also_to_console:
            stdout_outputs = [os.fdopen(orig_stdout_fd, "w", closefd=False), log_fh]
            stderr_outputs = [os.fdopen(orig_stderr_fd, "w", closefd=False), log_fh]
        else:
            stdout_outputs = [log_fh]
            stderr_outputs = [log_fh]

        # Start tee threads to read from pipes and write to outputs
        stdout_tee = _TeeWriter(stdout_read_fd, stdout_outputs, "StdoutTee")
        stderr_tee = _TeeWriter(stderr_read_fd, stderr_outputs, "StderrTee")
        tee_threads = [stdout_tee, stderr_tee]

        stdout_tee.start()
        stderr_tee.start()

        # Redirect file descriptors 1 and 2 to write ends of pipes
        os.dup2(stdout_write_fd, 1)
        os.dup2(stderr_write_fd, 2)
        os.close(stdout_write_fd)
        os.close(stderr_write_fd)

        # Redirect Python's sys.stdout/stderr to the new fd 1/2
        sys.stdout = io.TextIOWrapper(
            io.FileIO(1, "w", closefd=False),
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )
        sys.stderr = io.TextIOWrapper(
            io.FileIO(2, "w", closefd=False),
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )

        # Update logging handlers
        root = logging.getLogger()
        old_handlers = []
        for handler in root.handlers:
            if isinstance(handler, logging.StreamHandler):
                old_handlers.append((handler, handler.stream))
                handler.stream = sys.stderr

        try:
            yield log_path
        finally:
            # Flush Python streams
            sys.stdout.flush()
            sys.stderr.flush()

            # Restore logging handlers
            for handler, old_stream in old_handlers:
                handler.stream = old_stream

    finally:
        # Restore original file descriptors
        os.dup2(orig_stdout_fd, 1)
        os.dup2(orig_stderr_fd, 2)
        os.close(orig_stdout_fd)
        os.close(orig_stderr_fd)

        # Restore Python streams
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr

        # Wait for tee threads to finish (they'll exit when pipes close)
        for t in tee_threads:
            t.stop()
            t.join(timeout=1.0)

        # Close log file
        if log_fh:
            log_fh.close()

        _decrement_capture_depth()


# ---------------------------------------------------------------------------
# GitHub Actions Integration
# ---------------------------------------------------------------------------


@contextmanager
def github_group(title: str) -> Iterator[None]:
    """Create a collapsible group in GitHub Actions logs.

    No-op when not running in CI.

    Args:
        title: Group title shown in the UI.

    Example::

        with github_group("Building ROCm Components"):
            for component in components:
                build(component)
    """
    if is_ci():
        print(f"::group::{title}", flush=True)
    try:
        yield
    finally:
        if is_ci():
            print("::endgroup::", flush=True)


def github_warning(
    message: str, *, file: str | None = None, line: int | None = None
) -> None:
    """Emit a warning annotation in GitHub Actions.

    Args:
        message: Warning message.
        file: Optional file path to annotate.
        line: Optional line number.
    """
    if not is_ci():
        return

    params = []
    if file:
        params.append(f"file={file}")
    if line:
        params.append(f"line={line}")

    param_str = " " + ",".join(params) if params else ""
    print(f"::warning{param_str}::{message}", flush=True)


def github_error(
    message: str, *, file: str | None = None, line: int | None = None
) -> None:
    """Emit an error annotation in GitHub Actions.

    Args:
        message: Error message.
        file: Optional file path to annotate.
        line: Optional line number.
    """
    if not is_ci():
        return

    params = []
    if file:
        params.append(f"file={file}")
    if line:
        params.append(f"line={line}")

    param_str = " " + ",".join(params) if params else ""
    print(f"::error{param_str}::{message}", flush=True)


# ---------------------------------------------------------------------------
# Logger Disable
# ---------------------------------------------------------------------------


def disable_logger(name: str) -> None:
    """Disable a specific logger by name.

    Args:
        name: Logger name to disable (e.g., 'therock.packaging').
    """
    logging.getLogger(name).disabled = True
