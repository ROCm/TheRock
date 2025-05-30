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
* --skip-index: Suppresses automatic indexing and uploading of logs to S3.

Environment Variables:
* TEATIME_FORCE_INTERACTIVE: If set to 1, forces console output regardless of
  --interactive flag.
* TEATIME_S3_UPLOAD: If "1", enables automatic S3 upload and log indexing.
* TEATIME_S3_BUCKET: S3 bucket name for log uploads.
* TEATIME_S3_SUBDIR: Subdirectory in the bucket for organizing uploads.
* TEATIME_FAIL_ON_UPLOAD_ERROR: If "1", treat S3 upload or indexing errors as fatal.
* BASE_BUILD_DIR: Root of the build directory. Used to find logs and pass to
  indexing/upload scripts.
* AMDGPU_FAMILIES: Required by `create_log_index.py` to organize logs.

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
        self.skip_index = args.skip_index
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

        # S3 upload configuration
        try:
            self.upload_to_s3 = bool(int(os.getenv("TEATIME_S3_UPLOAD", "0")))
        except ValueError:
            print(
                "warning: TEATIME_S3_UPLOAD env var must be an integer "
                "(skipping S3 log upload)",
                file=sys.stderr,
            )
            self.upload_to_s3 = False

        self.s3_bucket = os.getenv("TEATIME_S3_BUCKET")
        if not self.s3_bucket and self.upload_to_s3:
            print(
                "warning: TEATIME_S3_BUCKET is not set (S3 upload will likely fail)",
                file=sys.stderr,
            )

        self.s3_subdir = os.getenv("TEATIME_S3_SUBDIR")
        if not self.s3_subdir and self.upload_to_s3:
            print(
                "warning: TEATIME_S3_SUBDIR is not set (S3 upload will likely fail)",
                file=sys.stderr,
            )

        self.log_dir = os.getenv("LOG_DIR", "build/logs")

    def start(self):
        if self.gh_group_label is not None:
            self.out.write(b"::group::" + self.gh_group_label + b"\n")
        if self.log_file and self.log_timestamps:
            self.log_file.write(f"BEGIN\t{self.start_time}\n".encode())

    def finish(self):
        end_time = time.time()
        if self.log_file is not None:
            if self.log_timestamps:
                self.log_file.write(
                    f"END\t{end_time}\t{end_time - self.start_time}\n".encode()
                )
            self.log_file.close()

        if self.gh_group_label is not None:
            self.out.write(b"::endgroup::\n")
        elif self.interactive_prefix is not None and self.label is not None:
            run_pretty = f"{round(end_time - self.start_time)} seconds"
            self.out.write(
                b"[" + self.label + b" completed in " + run_pretty.encode() + b"]\n"
            )

        # Call coordinate_index_and_logs if indexing is enabled.
        if (
            not self.skip_index
            and self.upload_to_s3
            and self.s3_bucket
            and self.log_dir
        ):
            self.coordinate_index_and_logs()

    def coordinate_index_and_logs(self):
        # Determine paths
        log_dir = self.log_dir
        amdgpu_family = os.getenv("AMDGPU_FAMILIES")

        # Step 1: Run create_log_index.py (if AMDGPU_FAMILIES is defined)
        if amdgpu_family:
            try:
                index_script = (
                    Path(__file__).resolve().parent.parent
                    / "build_tools"
                    / "create_log_index.py"
                )
                print(f"[TEATIME] Indexing logs for AMDGPU_FAMILIES={amdgpu_family}")
                subprocess.run(
                    [
                        sys.executable,
                        str(index_script),
                        "--log-dir",
                        str(log_dir),
                        "--amdgpu-family",
                        amdgpu_family,
                    ],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                print(f"[WARN] create_log_index.py failed: {e}", file=sys.stderr)
            except Exception as e:
                print(
                    f"[WARN] Unexpected error during log indexing: {e}", file=sys.stderr
                )
        else:
            print("[WARN] AMDGPU_FAMILIES not set; skipping log indexing")

        # Step 2: S3 upload using --bucket and --subdir
        try:
            upload_script = (
                Path(__file__).resolve().parent.parent
                / "build_tools"
                / "upload_logs_to_s3.py"
            )
            print(f"[TEATIME] Uploading logs to s3://{self.s3_bucket}/{self.s3_subdir}")
            subprocess.run(
                [
                    sys.executable,
                    str(upload_script),
                    "--log-dir",
                    str(log_dir),
                    "--bucket",
                    self.s3_bucket,
                    "--subdir",
                    self.s3_subdir,
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"[WARN] Log upload failed: {e}", file=sys.stderr)
            if os.getenv("TEATIME_FAIL_ON_UPLOAD_ERROR") == "1":
                raise
        except Exception as e:
            print(f"[WARN] Unexpected error during log upload: {e}", file=sys.stderr)
            if os.getenv("TEATIME_FAIL_ON_UPLOAD_ERROR") == "1":
                raise

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
    p.add_argument(
        "--skip-index", action="store_true", help="Skip indexing and uploading logs"
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
    try:
        run(args, child_arg_list, sink)
    finally:
        sink.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
