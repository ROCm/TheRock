# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Optional, invocation-scoped diagnostics; never write to report stdout.

This module has no database, network or build imports. Instrumentation is inert
without a logging session. File logs use exclusive creation and are flushed
after every record; a later logging failure must not undo cache operations.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
import json
from pathlib import Path
import sys
import time


_active = ContextVar("artifact_cache_diagnostics", default=None)


def _fields(values):
    return " ".join(
        f"{name}={json.dumps(value, default=str, ensure_ascii=True)}"
        for name, value in values.items() if value is not None
    )


class _Session:
    def __init__(self, verbose, stream):
        self.verbose = verbose
        self.stream = stream
        self.started = time.monotonic()
        self.sequence = 0

    def record(self, event, *, console=True, **fields):
        self.sequence += 1
        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        elapsed = time.monotonic() - self.started
        line = (f"[artifact-cache] {now} +{elapsed:.3f}s #{self.sequence} "
                f"{event} {_fields(fields)}").rstrip()
        if self.verbose and console:
            try:
                print(line, file=sys.stderr, flush=True)
            except (OSError, ValueError):
                self.verbose = False
        if self.stream is not None:
            try:
                print(line, file=self.stream, flush=True)
            except (OSError, ValueError):
                self.stream = None
                try:
                    print("[artifact-cache] Diagnostic file write failed; continuing without file logging.",
                          file=sys.stderr, flush=True)
                except (OSError, ValueError):
                    pass


def enabled():
    return _active.get() is not None


def trace(event, **fields):
    session = _active.get()
    if session is not None:
        session.record(event, **fields)


def record_message(message):
    """Retain an already displayed phase/final line without printing it twice."""
    session = _active.get()
    if session is not None:
        session.record("REPORT", console=False, message=message)


@contextmanager
def span(event, **fields):
    if not enabled():
        yield
        return
    started = time.monotonic()
    trace(event + "_START", **fields)
    try:
        yield
    except BaseException as exc:
        trace(event + "_FAILED", **fields, error_type=type(exc).__name__,
              reason=str(exc), duration_seconds=round(time.monotonic() - started, 3))
        raise
    else:
        trace(event + "_DONE", **fields, duration_seconds=round(time.monotonic() - started, 3))


def traced(event):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with span(event):
                return function(*args, **kwargs)
        return wrapped
    return decorate


class ByteProgress:
    """At most one update per five seconds, including within a large file.

    Counts bytes processed by this particular pass, not elapsed build work or
    network traffic. No worker thread, extra file reads, or per-file log dump.
    """

    def __init__(self, event, *, total_bytes=None, **fields):
        self.event, self.fields = event, fields
        self.total_bytes = total_bytes
        self.processed = 0
        self.last = time.monotonic() if enabled() else None

    def advance(self, count):
        if self.last is None:
            return
        self.processed += count
        now = time.monotonic()
        if now - self.last >= 5:
            trace(self.event, **self.fields, processed_bytes=self.processed,
                  total_bytes=self.total_bytes)
            self.last = now


@contextmanager
def logging_session(*, verbose=False, log_file=None):
    if not verbose and log_file is None:
        yield
        return
    stream = None
    if log_file is not None:
        path = Path(log_file).expanduser().absolute()
        path.parent.mkdir(parents=True, exist_ok=True)
        # A typo must not append diagnostics to a DB, payload, or source file.
        stream = path.open("x", encoding="utf-8")
    token = _active.set(_Session(verbose, stream))
    try:
        yield
    finally:
        _active.reset(token)
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass


def add_logging_args(parser):
    parser.add_argument("--verbose", action="store_true",
                        help="show detailed cache events with UTC timestamps and elapsed time on stderr")
    parser.add_argument("--log-file", type=Path,
                        help="save detailed cache events to a NEW UTF-8 file (does not require --verbose)")


def run_logged(args, execute):
    """Wrap a CLI operation, including its final report and failure exit code."""
    try:
        with logging_session(verbose=args.verbose, log_file=args.log_file):
            details = {name: getattr(args, name) for name in (
                "db", "build", "build_dir", "cache", "cache_root", "expected",
                "artifacts_dir", "variant", "destination", "log_file",
            ) if hasattr(args, name)}
            trace("RUN_START", operation=args.command, **details)
            try:
                code = execute(args)
            except BaseException as exc:
                trace("RUN_FAILED", operation=args.command, error_type=type(exc).__name__, reason=str(exc))
                raise
            trace("RUN_END", operation=args.command, exit_code=code,
                  result="completed" if code == 0 else "incomplete_or_failed")
            return code
    except OSError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
