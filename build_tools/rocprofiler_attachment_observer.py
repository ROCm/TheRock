# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Bounded read-only observer for two installed attachment-test executables."""

import errno
import json
from pathlib import Path
import re
import stat
import time

from normal_ci_input_identity import visible_cgroups

PROVIDERS = re.compile(
    r"lib(?:amdhip64|hsa-runtime64|rocprofiler-sdk|rocprofiler-register|rocjitsu|amd_comgr|hiprtc).*\.so(?:\..*)?$"
)


def proc_identity(pid: int) -> dict:
    text = Path(f"/proc/{pid}/stat").read_text()
    fields = text[text.rfind(")") + 2 :].split()
    return {
        "pid": pid,
        "ppid": int(fields[1]),
        "start_ticks": int(fields[19]),
        "state": fields[0],
        "own_user_cpu_ticks": int(fields[11]),
        "own_system_cpu_ticks": int(fields[12]),
        "threads": int(fields[17]),
    }


def descendant(pid: int, root: int) -> bool:
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == root:
            return True
        seen.add(pid)
        pid = proc_identity(pid)["ppid"]
    return False


def bounded_text(path: Path, limit: int = 32768) -> str:
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("read-size limit exceeded")
    return data.decode("utf-8", errors="replace")


def roctx_markers(command_line: str, allowed_root: Path) -> dict:
    args = command_line.rstrip("\0").split("\0")
    if len(args) != 3 or args[1] not in {"normal", "selected"}:
        raise ValueError("unexpected ROCTx target arguments")
    trigger = Path(args[2]).resolve()
    if not trigger.is_relative_to(allowed_root.resolve()):
        raise ValueError("ROCTx marker path is outside installed test build")
    result = dict(
        mode=args[1],
        trigger_path=str(trigger),
        markers={},
        scope="Positive -ready marker proves HIP allocation/init completed; absence alone does not prove incomplete startup",
    )
    for suffix in ["-ready", "-complete"]:
        path = Path(str(trigger) + suffix)
        try:
            value = path.lstat()
            if not stat.S_ISREG(value.st_mode):
                raise ValueError("nonregular/symlink ROCTx marker rejected")
            result["markers"][suffix] = dict(
                exists=True, bytes=value.st_size, mtime_ns=value.st_mtime_ns
            )
        except FileNotFoundError:
            result["markers"][suffix] = dict(exists=False)
    return result


class TargetObserver:
    def __init__(
        self,
        output: Path,
        root_pid: int,
        executables: set[Path],
        fingerprint,
        marker_root: Path | None = None,
    ):
        self.output, self.root_pid = output, root_pid
        self.executables = {path.resolve() for path in executables}
        self.fingerprint = fingerprint
        self.marker_root = marker_root
        self.targets, self.errors, self.rejected = {}, [], []
        self.offset, self.partial, self.polls = 0, b"", 0
        self.mapped_files = {}
        self.sample_count = 0
        self.output_bytes = 0
        self.groups, self.group_scope = visible_cgroups()

    def emit(self, filename: str, data: dict) -> None:
        encoded = json.dumps(data, sort_keys=True) + "\n"
        size = len(encoded.encode())
        if self.output_bytes + size > 128 * 1024**2:
            raise ValueError("observer output budget exceeded")
        with (self.output / filename).open("a", encoding="utf-8") as stream:
            stream.write(encoded)
        self.output_bytes += size

    def consume(self) -> None:
        path = self.output / "poll-events.tsv"
        if not path.exists():
            return
        with path.open("rb") as stream:
            stream.seek(self.offset)
            data = stream.read(1024 * 1024)
            self.offset += len(data)
        if self.offset > 32 * 1024**2:
            raise ValueError("raw poll-event budget exceeded")
        lines = (self.partial + data).split(b"\n")
        self.partial = lines.pop()
        for line in lines:
            self.polls += 1
            if self.polls > 100000:
                raise ValueError("poll-record limit exceeded")
            begin, end, shell, target, status, error_path = line.decode().split("\t")
            shell, target, status = int(shell), int(target), int(status)
            row = dict(
                begin_epoch_s=float(begin),
                end_epoch_s=float(end),
                shell_pid=shell,
                target_pid=target,
                original_grep_status=status,
                observed_monotonic_s=time.monotonic(),
            )
            error = Path(error_path)
            if (
                error.parent != self.output / "poll-errors"
                or error.is_symlink()
                or not re.fullmatch(rf"{shell}-{target}-\d+\.txt", error.name)
            ):
                raise ValueError("poll error path is outside owned capture")
            try:
                row["original_grep_stderr"] = bounded_text(error, 65536)
                if not descendant(shell, self.root_pid) or not descendant(
                    target, self.root_pid
                ):
                    raise ValueError("PID is not in the owned test-command tree")
                identity = proc_identity(target)
                executable = Path(f"/proc/{target}/exe").resolve(strict=True)
                if executable not in self.executables:
                    raise ValueError(
                        "target executable is outside the two-file allowlist"
                    )
                identity["executable"] = str(executable)
                identity["executable_sha256"] = self.fingerprint(executable)
                key = (target, identity["start_ticks"])
                if len(self.targets) >= 128 and key not in self.targets:
                    raise ValueError("target-identity count budget exceeded")
                self.targets.setdefault(key, identity)
                row["identity"] = identity
                row["attributed"] = True
            except (OSError, ValueError) as exc:
                row.update(
                    attributed=False,
                    error=str(exc),
                    scope="Unattributed/exited/unreadable candidate; no thread/maps reads performed",
                )
                self.rejected.append(row)
                if len(self.rejected) > 256:
                    raise ValueError("unattributed/error record budget exceeded")
            self.emit("polls.jsonl", row)

    def snapshot(self) -> None:
        if len(self.errors) > 256:
            raise ValueError("observer error record budget exceeded")
        if self.targets:
            groups = []
            for path in self.groups:
                files = {}
                for name in [
                    "pids.max",
                    "pids.current",
                    "pids.events",
                    "cpu.stat",
                    "cpu.pressure",
                    "memory.pressure",
                    "io.pressure",
                ]:
                    try:
                        files[name] = bounded_text(path / name).strip()
                    except OSError as exc:
                        files[name] = {"unavailable": str(exc)}
                groups.append(dict(path=str(path), files=files))
            self.emit(
                "visible-cgroup-samples.jsonl",
                dict(
                    monotonic_s=time.monotonic(),
                    epoch_s=time.time(),
                    resolution=self.group_scope,
                    groups=groups,
                ),
            )
        for key, original in list(self.targets.items()):
            pid, start = key
            row = dict(
                identity=original, monotonic_s=time.monotonic(), epoch_s=time.time()
            )
            try:
                current = proc_identity(pid)
                if current["start_ticks"] != start:
                    row["state"] = "PID-reused; no further reads"
                    self.targets.pop(key)
                    self.errors.append(row)
                    self.emit("samples.jsonl", row)
                    continue
                row["process"] = current
                if (
                    self.marker_root is not None
                    and Path(original["executable"]).name == "roctx-attach-pause-resume"
                ):
                    row["existing_application_markers"] = roctx_markers(
                        bounded_text(Path(f"/proc/{pid}/cmdline"), 8192),
                        self.marker_root,
                    )
                row["cgroup_membership"] = bounded_text(
                    Path(f"/proc/{pid}/cgroup"), 8192
                ).strip()
                row["same_visible_cgroup_as_observer"] = (
                    row["cgroup_membership"]
                    == bounded_text(Path("/proc/self/cgroup"), 8192).strip()
                )
                tasks = list(Path(f"/proc/{pid}/task").iterdir())
                if len(tasks) > 4096:
                    raise ValueError("thread-count limit exceeded")
                row["threads"] = []
                for task in sorted(tasks, key=lambda path: int(path.name)):
                    thread = {"tid": int(task.name)}
                    try:
                        thread["comm"] = bounded_text(task / "comm", 1024).strip()
                        thread["wchan"] = bounded_text(task / "wchan", 1024).strip()
                        values = bounded_text(task / "status").splitlines()
                        thread["status"] = [
                            value
                            for value in values
                            if value.split(":", 1)[0]
                            in {
                                "Name",
                                "State",
                                "Tgid",
                                "Pid",
                                "PPid",
                                "TracerPid",
                                "Threads",
                                "SigQ",
                                "SigPnd",
                                "ShdPnd",
                                "SigBlk",
                                "SigIgn",
                                "SigCgt",
                                "voluntary_ctxt_switches",
                                "nonvoluntary_ctxt_switches",
                            }
                        ]
                    except OSError as exc:
                        thread["read_error"] = str(exc)
                        if exc.errno != errno.ENOENT:
                            self.errors.append(
                                dict(pid=pid, tid=thread["tid"], error=str(exc))
                            )
                    row["threads"].append(thread)
                maps = bounded_text(Path(f"/proc/{pid}/maps"), 2 * 1024 * 1024)
                row["selected_maps"] = []
                for line in maps.splitlines():
                    parts = line.split(None, 5)
                    if len(parts) != 6 or not parts[5].startswith("/"):
                        continue
                    path = Path(parts[5])
                    if not PROVIDERS.fullmatch(path.name):
                        continue
                    row["selected_maps"].append(line)
                    if str(path) not in self.mapped_files:
                        self.mapped_files[str(path)] = dict(
                            path=str(path), sha256=self.fingerprint(path)
                        )
                row["state"] = "observed"
                self.sample_count += 1
            except FileNotFoundError as exc:
                if Path(f"/proc/{pid}").exists():
                    row.update(state="file-read-error-in-live-target", error=str(exc))
                    self.errors.append(row)
                else:
                    row["state"] = "exited-after-identity"
                    self.targets.pop(key)
            except (OSError, ValueError) as exc:
                row.update(state="read-error", error=str(exc))
                self.errors.append(row)
            self.emit("samples.jsonl", row)

    def summary(self) -> dict:
        return dict(
            polls=self.polls,
            samples=self.sample_count,
            output_bytes=self.output_bytes,
            rejected=self.rejected,
            errors=self.errors,
            mapped_provider_files=list(self.mapped_files.values()),
            complete=bool(self.polls and self.sample_count)
            and not self.rejected
            and not self.errors
            and not self.partial,
            scope="Allowlisted descendant targets; PID+creation identity. No environ or arbitrary process reads. Missing/rejected attribution is not clean evidence.",
        )
