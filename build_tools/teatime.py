#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
* --diagnostics: Logs process IDs, one-second resource samples, peak child-tree
  memory, and the exact return code in decimal and hexadecimal.
* --diagnostic-path: With diagnostics enabled, records path metadata before and
  after the child command. May be specified more than once.

CI systems can set `TEATIME_LABEL_GH_GROUP=1` in the environment, which will
cause labeled console output to be printed using GitHub Actions group markers
instead of line prefixes. This causes the output to show in the log viewer
with interactive group handles.
"""

import argparse
from dataclasses import dataclass
import io
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import threading
import time

try:
    import psutil
except ModuleNotFoundError:
    psutil = None


_DIAGNOSTIC_INTERVAL_SECONDS = 1.0
_WINDOWS_STATUS_NAMES = {
    0xC0000005: "STATUS_ACCESS_VIOLATION",
    0xC0000017: "STATUS_NO_MEMORY",
    0xC00000FD: "STATUS_STACK_OVERFLOW",
    0xC0000135: "STATUS_DLL_NOT_FOUND",
    0xC0000142: "STATUS_DLL_INIT_FAILED",
    0xC0000374: "STATUS_HEAP_CORRUPTION",
    0xC0000409: "STATUS_STACK_BUFFER_OVERRUN",
}


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    if value < 1024**3:
        return f"{value / (1024**2):.2f}MiB"
    return f"{value / (1024**3):.2f}GiB"


def format_returncode(returncode: int) -> str:
    """Formats a process return code without losing Windows NTSTATUS bits."""
    unsigned = returncode & 0xFFFFFFFF
    fields = [f"decimal={returncode}", f"hex=0x{unsigned:08X}"]
    status_name = _WINDOWS_STATUS_NAMES.get(unsigned)
    if status_name is not None:
        fields.append(f"status={status_name}")
    elif returncode < 0 and os.name != "nt":
        fields.append(f"signal={-returncode}")
    return " ".join(fields)


def _get_windows_commit_bytes() -> tuple[int | None, int | None, int | None]:
    if platform.system() != "Windows":
        return None, None, None

    try:
        import ctypes

        class PerformanceInformation(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("CommitTotal", ctypes.c_size_t),
                ("CommitLimit", ctypes.c_size_t),
                ("CommitPeak", ctypes.c_size_t),
                ("PhysicalTotal", ctypes.c_size_t),
                ("PhysicalAvailable", ctypes.c_size_t),
                ("SystemCache", ctypes.c_size_t),
                ("KernelTotal", ctypes.c_size_t),
                ("KernelPaged", ctypes.c_size_t),
                ("KernelNonpaged", ctypes.c_size_t),
                ("PageSize", ctypes.c_size_t),
                ("HandleCount", ctypes.c_ulong),
                ("ProcessCount", ctypes.c_ulong),
                ("ThreadCount", ctypes.c_ulong),
            ]

        info = PerformanceInformation()
        info.cb = ctypes.sizeof(info)
        if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info), info.cb):
            return None, None, None
        return (
            info.CommitTotal * info.PageSize,
            info.CommitLimit * info.PageSize,
            info.CommitPeak * info.PageSize,
        )
    except Exception:
        return None, None, None


@dataclass(frozen=True)
class ResourceSnapshot:
    system_memory_percent: float | None = None
    system_memory_available: int | None = None
    swap_used: int | None = None
    swap_total: int | None = None
    commit_total: int | None = None
    commit_limit: int | None = None
    commit_peak: int | None = None
    process_rss: int | None = None
    process_vms: int | None = None
    process_threads: int | None = None
    process_read_bytes: int | None = None
    process_write_bytes: int | None = None
    descendant_count: int | None = None
    descendant_rss: int | None = None
    descendant_read_bytes: int | None = None
    descendant_write_bytes: int | None = None

    def format(self) -> str:
        memory_percent = (
            "unknown"
            if self.system_memory_percent is None
            else f"{self.system_memory_percent:.1f}%"
        )
        return " ".join(
            [
                f"system_memory={memory_percent}",
                f"system_available={_format_bytes(self.system_memory_available)}",
                f"swap={_format_bytes(self.swap_used)}/{_format_bytes(self.swap_total)}",
                f"commit={_format_bytes(self.commit_total)}/{_format_bytes(self.commit_limit)}",
                f"commit_peak={_format_bytes(self.commit_peak)}",
                f"process_rss={_format_bytes(self.process_rss)}",
                f"process_vms={_format_bytes(self.process_vms)}",
                f"process_threads={self.process_threads if self.process_threads is not None else 'unknown'}",
                f"process_read={_format_bytes(self.process_read_bytes)}",
                f"process_write={_format_bytes(self.process_write_bytes)}",
                f"descendants={self.descendant_count if self.descendant_count is not None else 'unknown'}",
                f"descendant_rss={_format_bytes(self.descendant_rss)}",
                f"descendant_read={_format_bytes(self.descendant_read_bytes)}",
                f"descendant_write={_format_bytes(self.descendant_write_bytes)}",
            ]
        )


def _collect_resource_snapshot(pid: int) -> ResourceSnapshot:
    if psutil is None:
        return ResourceSnapshot()

    virtual_memory = psutil.virtual_memory()
    swap_memory = psutil.swap_memory()
    commit_total, commit_limit, commit_peak = _get_windows_commit_bytes()

    process_rss = None
    process_vms = None
    process_threads = None
    process_read_bytes = None
    process_write_bytes = None
    descendant_count = None
    descendant_rss = None
    descendant_read_bytes = None
    descendant_write_bytes = None
    try:
        process = psutil.Process(pid)
        process_memory = process.memory_info()
        process_rss = process_memory.rss
        process_vms = process_memory.vms
        process_threads = process.num_threads()
        process_io = process.io_counters()
        process_read_bytes = process_io.read_bytes
        process_write_bytes = process_io.write_bytes
        descendants = process.children(recursive=True)
        descendant_count = len(descendants)
        descendant_rss = 0
        descendant_read_bytes = 0
        descendant_write_bytes = 0
        for descendant in descendants:
            try:
                descendant_rss += descendant.memory_info().rss
                descendant_io = descendant.io_counters()
                descendant_read_bytes += descendant_io.read_bytes
                descendant_write_bytes += descendant_io.write_bytes
            except psutil.Error:
                pass
    except psutil.Error:
        pass

    return ResourceSnapshot(
        system_memory_percent=virtual_memory.percent,
        system_memory_available=virtual_memory.available,
        swap_used=swap_memory.used,
        swap_total=swap_memory.total,
        commit_total=commit_total,
        commit_limit=commit_limit,
        commit_peak=commit_peak,
        process_rss=process_rss,
        process_vms=process_vms,
        process_threads=process_threads,
        process_read_bytes=process_read_bytes,
        process_write_bytes=process_write_bytes,
        descendant_count=descendant_count,
        descendant_rss=descendant_rss,
        descendant_read_bytes=descendant_read_bytes,
        descendant_write_bytes=descendant_write_bytes,
    )


def _collect_top_processes(limit: int = 10) -> list[str]:
    if psutil is None:
        return []

    processes: list[tuple[int, str]] = []
    for process in psutil.process_iter(
        ["pid", "ppid", "name", "memory_info", "num_threads"]
    ):
        try:
            info = process.info
            memory_info = info["memory_info"]
            rss = memory_info.rss if memory_info is not None else 0
            description = (
                f"pid={info['pid']} ppid={info['ppid']} name={info['name']} "
                f"rss={_format_bytes(rss)} threads={info['num_threads']}"
            )
            processes.append((rss, description))
        except (psutil.Error, KeyError):
            pass
    processes.sort(key=lambda item: item[0], reverse=True)
    return [description for _, description in processes[:limit]]


class OutputSink:
    def __init__(self, args: argparse.Namespace):
        self.start_time = time.time()
        self.lock = threading.Lock()
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
        with self.lock:
            if self.gh_group_label is not None:
                self.out.write(b"::group::" + self.gh_group_label + b"\n")
            if self.log_file and self.log_timestamps:
                self.log_file.write(f"BEGIN\t{self.start_time}\n".encode())
                self.log_file.flush()

    def finish(self, rc: int):
        end_time = time.time()
        with self.lock:
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
                    b"["
                    + self.label
                    + status_msg.encode()
                    + run_pretty.encode()
                    + b"]\n"
                )
            if not self.interactive and rc != 0:
                sys.stdout.buffer.write(self.out.getvalue())
                sys.stdout.buffer.flush()

    def writeline(self, line: bytes):
        with self.lock:
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

    def write_log_record(self, record: str) -> None:
        if self.log_file is None:
            return
        with self.lock:
            self.log_file.write(record.encode())
            self.log_file.flush()


class DiagnosticMonitor:
    def __init__(
        self,
        sink: OutputSink,
        child: subprocess.Popen,
        diagnostic_paths: list[Path],
    ):
        self.sink = sink
        self.child = child
        self.diagnostic_paths = diagnostic_paths
        self.stop_event = threading.Event()
        self.max_process_rss = 0
        self.max_descendant_rss = 0
        self.max_system_memory_percent = 0.0
        self.max_commit_total: int | None = None
        self.commit_limit: int | None = None

    def _emit_snapshot(self, phase: str) -> None:
        try:
            snapshot = _collect_resource_snapshot(self.child.pid)
            self.max_process_rss = max(self.max_process_rss, snapshot.process_rss or 0)
            self.max_descendant_rss = max(
                self.max_descendant_rss, snapshot.descendant_rss or 0
            )
            self.max_system_memory_percent = max(
                self.max_system_memory_percent, snapshot.system_memory_percent or 0.0
            )
            if snapshot.commit_total is not None:
                self.max_commit_total = max(
                    self.max_commit_total or 0, snapshot.commit_total
                )
            if snapshot.commit_limit is not None:
                self.commit_limit = max(self.commit_limit or 0, snapshot.commit_limit)
            self.sink.writeline(
                f"TEATIME_RESOURCE phase={phase} {snapshot.format()}\n".encode()
            )
        except Exception as e:
            self.sink.writeline(
                f"TEATIME_RESOURCE phase={phase} collection_error={e!r}\n".encode()
            )

    def _monitor(self) -> None:
        while not self.stop_event.wait(_DIAGNOSTIC_INTERVAL_SECONDS):
            self._emit_snapshot("periodic")

    def _emit_path_status(self, phase: str) -> None:
        for path in self.diagnostic_paths:
            try:
                exists = path.exists()
                is_symlink = path.is_symlink()
                stat_result = path.lstat() if exists or is_symlink else None
                self.sink.writeline(
                    (
                        f"TEATIME_PATH phase={phase} path={path.absolute()} "
                        f"exists={exists} is_file={path.is_file()} "
                        f"is_dir={path.is_dir()} is_symlink={is_symlink} "
                        f"parent_exists={path.parent.exists()} "
                        f"size={stat_result.st_size if stat_result is not None else 'unknown'} "
                        f"mtime_ns={stat_result.st_mtime_ns if stat_result is not None else 'unknown'}\n"
                    ).encode()
                )
            except OSError as e:
                self.sink.writeline(
                    f"TEATIME_PATH phase={phase} path={path.absolute()} error={e!r}\n".encode()
                )

    def start(self) -> None:
        try:
            self.sink.writeline(
                (
                    "TEATIME_PROCESS "
                    f"wrapper_pid={os.getpid()} wrapper_parent_pid={os.getppid()} "
                    f"child_pid={self.child.pid} platform={platform.platform()} "
                    f"python={sys.version.split()[0]} executable={sys.executable}\n"
                ).encode()
            )
            if psutil is None:
                self.sink.writeline(
                    b"TEATIME_RESOURCE unavailable=psutil-not-installed\n"
                )
                self._emit_path_status("start")
                return
            self._emit_snapshot("start")
            self._emit_path_status("start")
            self.thread = threading.Thread(target=self._monitor, daemon=True)
            self.thread.start()
        except Exception as e:
            self.sink.writeline(f"TEATIME_DIAGNOSTIC start_error={e!r}\n".encode())

    def stop(self, returncode: int) -> None:
        try:
            self.stop_event.set()
            if hasattr(self, "thread"):
                self.thread.join(timeout=2)
            if psutil is not None:
                self._emit_snapshot("end")
                for rank, process_description in enumerate(
                    _collect_top_processes(), start=1
                ):
                    self.sink.writeline(
                        f"TEATIME_TOP_PROCESS rank={rank} {process_description}\n".encode()
                    )
            self._emit_path_status("end")
        except Exception as e:
            self.sink.writeline(f"TEATIME_DIAGNOSTIC stop_error={e!r}\n".encode())
        self.sink.writeline(
            (
                f"TEATIME_RESULT {format_returncode(returncode)} "
                f"max_process_rss={_format_bytes(self.max_process_rss)} "
                f"max_descendant_rss={_format_bytes(self.max_descendant_rss)} "
                f"max_system_memory={self.max_system_memory_percent:.1f}% "
                f"max_commit={_format_bytes(self.max_commit_total)}/"
                f"{_format_bytes(self.commit_limit)}\n"
            ).encode()
        )


def run(
    args: argparse.Namespace, child_arg_list: list[str] | None, sink: OutputSink
) -> int:
    child: subprocess.Popen | None = None
    if child_arg_list is None:
        # Pipeline mode.
        child_stream = sys.stdin.buffer
    else:
        # Subprocess mode.
        if sink.log_file:
            child_arg_list_pretty = shlex.join(child_arg_list)
            sink.write_log_record(f"EXEC\t{os.getcwd()}\t{child_arg_list_pretty}\n")
        child = subprocess.Popen(
            child_arg_list, stderr=subprocess.STDOUT, stdout=subprocess.PIPE
        )
        child_stream = child.stdout

    monitor = None
    if child is not None and args.diagnostics:
        monitor = DiagnosticMonitor(sink, child, args.diagnostic_path)
        monitor.start()

    try:
        for line in child_stream:
            sink.writeline(line)
    except KeyboardInterrupt:
        if child:
            child.terminate()
    if child:
        rc = child.wait()
        if monitor is not None:
            monitor.stop(rc)
        return rc
    return 0


def main(cl_args: list[str]) -> int:
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
        "--diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record child process, resource, and raw return-code diagnostics",
    )
    p.add_argument(
        "--diagnostic-path",
        action="append",
        default=[],
        type=Path,
        help="Record path metadata before and after the child command",
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
    rc = 1
    try:
        rc = run(args, child_arg_list, sink)
        return rc
    finally:
        sink.finish(rc)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
