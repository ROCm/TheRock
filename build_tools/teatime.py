#!/usr/bin/env python
"""teatime.py - Combination of "tee" and "time" for managing build logs, while
indicating that if you are watching them, it might be time for tea.

Usage in a pipeline:
  teatime.py args...

Usage to manage a subprocess:
  teatime.py args... -- child_command child_args...

With no further arguments, teatime will just forward the child output to its
output (if launching a child process, it will combine stderr and stdout).

Like with `tee`, a log file can be passed as `teatime.py output.log -- child`,
causing output to be written to the log file in addition to the console. Some
additional information, like execution command line are also written to the
log file, so it is not a verbatim copy of the output.

Other arguments:

* --label 'some label': Prefixes console output lines with `[label] ` and writes
  a summary line with total execution time.
* --no-interactive: Disables interactive console output. Output will only be
  written to the console if the child returns a non zero exit code.
* --log-timestamps: Log lines will be written with a starting column of the
  time in seconds since start, and a header/trailer will be added with more
  timing information.

CI systems can set `TEATIME_LABEL_GH_GROUP=1` in the environment, which will
cause labeled console output to be printed using GitHub Actions group markers
instead of line prefixes. This causes the output to show in the log viewer
with interactive group handles.
"""

import argparse
import io
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


class OutputSink:
    def __init__(self, args: argparse.Namespace):
        self.start_time = time.time()
        self.interactive: bool = args.interactive
        if self.interactive:
            self.out = sys.stdout.buffer
        else:
            self.out = io.BytesIO()

        # Label management.
        self.label: str | None = args.label
        if self.label is not None:
            self.label = self.label.encode()
        self.gh_group_enable = False
        try:
            self.gh_group_enable = bool(int(os.getenv("TEATIME_LABEL_GH_GROUP", "0")))
        except ValueError:
            print(
                "warning: TEATIME_LABEL_GH_GROUP env var must be an integer "
                "(not emitting GH actions friendly groups)",
                file=sys.stderr,
            )
            self.gh_group_enable = False
        self.gh_group_label: bytes | None = None
        self.interactive_prefix: bytes | None = None
        if self.label is not None:
            if self.gh_group_enable:
                self.gh_group_label = self.label
            else:
                self.interactive_prefix = b"[" + self.label + b"] "

        # Log file.
        self.log_path: Path | None = args.file
        self.log_file = None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_file = open(self.log_path, "wb")
        self.log_timestamps: bool = args.log_timestamps

    def start(self):
        if self.gh_group_label is not None:
            self.out.write(b"::group::" + self.gh_group_label + b"\n")
        if self.log_file and self.log_timestamps:
            self.log_file.write(f"BEGIN\t{self.start_time}\n".encode())

    def finish(self, rc: int):
        end_time = time.time()
        if self.log_file is not None:
            if self.log_timestamps:
                self.log_file.write(
                    f"END\t{end_time}\t{end_time - self.start_time}\t{rc}\n".encode()
                )
            self.log_file.close()
        if self.gh_group_label is not None:
            self.out.write(b"::endgroup::\n")
        elif self.interactive_prefix is not None and self.label is not None:
            run_pretty = f"{round(end_time - self.start_time)} seconds"
            if rc == 0:
                status_msg = f" SUCCEEDED in "
            else:
                status_msg = f" FAILED WITH CODE {rc} in "
            self.out.write(
                b"[" + self.label + status_msg.encode() + run_pretty.encode() + b"]\n"
            )

    def writeline(self, line: bytes):
        if self.interactive_prefix is not None:
            self.out.write(self.interactive_prefix)
        self.out.write(line)
        if self.interactive:
            self.out.flush()
        if self.log_file is not None:
            if self.log_timestamps:
                now = time.time()
                self.log_file.write(
                    f"{round((now - self.start_time) * 10) / 10}\t".encode()
                )
            self.log_file.write(line)
            self.log_file.flush()


def run(args: argparse.Namespace, child_arg_list: list[str] | None, sink: OutputSink):
    child: subprocess.Popen | None = None
    if child_arg_list is None:
        # Pipeline mode.
        child_stream = sys.stdin.buffer
    else:
        # Subprocess mode.
        if sink.log_file:
            child_arg_list_pretty = shlex.join(child_arg_list)
            sink.log_file.write(
                f"EXEC\t{os.getcwd()}\t{child_arg_list_pretty}\n".encode()
            )
        child = subprocess.Popen(
            child_arg_list, stderr=subprocess.STDOUT, stdout=subprocess.PIPE
        )
        child_stream = child.stdout

    try:
        for line in child_stream:
            sink.writeline(line)
    except KeyboardInterrupt:
        if child:
            child.terminate()
    if child:
        rc = child.wait()
        if rc != 0 and not args.interactive:
            # Dump all output on failure.
            sys.stdout.buffer.write(sink.out.getvalue())
        sys.exit(rc)


def main(cl_args: list[str]):
    # If the command line contains a "--" then we are in subprocess execution
    # mode: capture the child arguments explicitly before parsing ours.
    child_arg_list: list[str] | None = None
    try:
        child_sep_pos = cl_args.index("--")
    except ValueError:
        pass
    else:
        child_arg_list = cl_args[child_sep_pos + 1 :]
        cl_args = cl_args[0:child_sep_pos]

    p = argparse.ArgumentParser(
        "teatime.py", usage="teatime.py {command} [-- {child args...}]"
    )
    p.add_argument("--label", help="Apply a label prefix to interactive output")
    p.add_argument(
        "--interactive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable interactive output (if disabled console output will only be emitted on failure)",
    )
    p.add_argument(
        "--log-timestamps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log timestamps along with log lines to the log file",
    )
    p.add_argument("file", type=Path, help="Also log output to this file")
    args = p.parse_args(cl_args)

    # Allow some things to be overriden by env vars.
    force_interactive = os.getenv("TEATIME_FORCE_INTERACTIVE")
    if force_interactive:
        try:
            force_interactive = int(force_interactive)
        except ValueError as e:
            raise ValueError(
                "Expected 'TEATIME_FORCE_INTERACTIVE' env var to be an int"
            ) from e
        args.interactive = bool(force_interactive)

    sink = OutputSink(args)
    sink.start()
    rc = 0
    try:
        run(args, child_arg_list, sink)
    except SystemExit as e:
        # See docs for interpretation of e.code
        if e.code is None:
            rc = 0
        elif isinstance(e.code, int):
            rc = e.code
        else:
            rc = 1
        raise
    finally:
        sink.finish(rc)


if __name__ == "__main__":
    main(sys.argv[1:])
