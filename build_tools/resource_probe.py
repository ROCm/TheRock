# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Low-overhead, opt-in resource telemetry for CI build commands.

This wrapper deliberately records counters and fixed process categories, not
process names, command lines, or environment values. It is dependency-free so
that it can run before TheRock's Python requirements are installed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Iterable

try:
    import resource
except ImportError:  # pragma: no cover - unavailable on Windows
    resource = None


SCHEMA = "therock.resource_probe.v1"
PROBE_VERSION = "1"
PHASE_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
UNLIMITED_SENTINEL = 1 << 60
PROCESS_CATEGORY_PATTERNS = (
    ("build_tool", re.compile(r"^(?:cmake|ninja|make|meson)$")),
    (
        "compiler",
        re.compile(r"^(?:cc|c\+\+|cc1|cc1plus|gcc|g\+\+|clang|clang\+\+|clang-\d+)$"),
    ),
    ("linker", re.compile(r"^(?:ld|ld\.lld|lld|collect2)$")),
    (
        "binary_tool",
        re.compile(r"^(?:ar|ranlib|strip|objcopy|llvm-ar|llvm-ranlib)$"),
    ),
    (
        "runtime",
        re.compile(r"^(?:python(?:\d+(?:\.\d+)?)?|pypy\d*|perl|ruby|node)$"),
    ),
    ("shell", re.compile(r"^(?:bash|dash|fish|sh|zsh)$")),
    (
        "transfer_or_cache_tool",
        re.compile(r"^(?:ccache|curl|dvc|gh|git|rsync|sccache|tar|wget)$"),
    ),
)


def _utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return None


def _read_int(path: Path) -> int | None:
    value = _read_text(path)
    if value is None:
        return None
    value = value.strip()
    if not value or value == "max":
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return None if parsed >= UNLIMITED_SENTINEL else parsed


def parse_key_values(text: str | None) -> dict[str, int]:
    result: dict[str, int] = {}
    if not text:
        return result
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            result[fields[0].rstrip(":")] = int(fields[1])
        except ValueError:
            continue
    return result


def parse_cpu_list(value: str | None) -> list[int]:
    cpus: set[int] = set()
    if not value:
        return []
    for item in value.strip().split(","):
        try:
            if "-" in item:
                first, last = (int(part) for part in item.split("-", 1))
                if first <= last:
                    cpus.update(range(first, last + 1))
            elif item:
                cpus.add(int(item))
        except ValueError:
            continue
    return sorted(cpus)


def parse_psi(text: str | None) -> dict[str, dict[str, float | int]] | None:
    if not text:
        return None
    result: dict[str, dict[str, float | int]] = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        values: dict[str, float | int] = {}
        for field in fields[1:]:
            if "=" not in field:
                continue
            key, raw = field.split("=", 1)
            try:
                values[key] = int(raw) if key == "total" else float(raw)
            except ValueError:
                continue
        result[fields[0]] = values
    return result or None


def parse_io_stat(text: str | None) -> dict[str, int] | None:
    if not text:
        return None
    totals = {key: 0 for key in ("rbytes", "wbytes", "rios", "wios", "dbytes", "dios")}
    found = False
    for line in text.splitlines():
        for field in line.split()[1:]:
            if "=" not in field:
                continue
            key, raw = field.split("=", 1)
            if key not in totals:
                continue
            try:
                totals[key] += int(raw)
                found = True
            except ValueError:
                continue
    return totals if found else None


def parse_io_stat_devices(text: str | None) -> dict[str, dict[str, int]]:
    """Parse cgroup io.stat without persisting host device names."""
    result: dict[str, dict[str, int]] = {}
    if not text:
        return result
    allowed = ("rbytes", "wbytes", "rios", "wios", "dbytes", "dios")
    for line in text.splitlines():
        fields = line.split()
        if not fields or not re.fullmatch(r"\d+:\d+", fields[0]):
            continue
        counters: dict[str, int] = {}
        for field in fields[1:]:
            if "=" not in field:
                continue
            key, raw = field.split("=", 1)
            if key not in allowed:
                continue
            try:
                counters[key] = int(raw)
            except ValueError:
                continue
        if counters:
            result[fields[0]] = counters
    return result


def parse_diskstats(text: str | None) -> dict[str, dict[str, int]]:
    """Return host disk counters keyed only by stable major:minor numbers."""
    result: dict[str, dict[str, int]] = {}
    if not text:
        return result
    keys = (
        "reads_completed",
        "reads_merged",
        "sectors_read",
        "read_time_ms",
        "writes_completed",
        "writes_merged",
        "sectors_written",
        "write_time_ms",
        "io_in_progress",
        "io_time_ms",
        "weighted_io_time_ms",
    )
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 14:
            continue
        try:
            device_id = f"{int(fields[0])}:{int(fields[1])}"
            values = [int(value) for value in fields[3:14]]
        except ValueError:
            continue
        result[device_id] = dict(zip(keys, values))
    return result


def parse_numa_stat(text: str | None) -> dict[str, Any] | None:
    """Aggregate cgroup NUMA counters without retaining source text or paths."""
    if not text:
        return None
    totals: dict[str, int] = {}
    nodes: set[int] = set()
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        metric_total = 0
        found = False
        for field in fields[1:]:
            match = re.fullmatch(r"N(\d+)=(\d+)", field)
            if not match:
                continue
            nodes.add(int(match.group(1)))
            metric_total += int(match.group(2))
            found = True
        if found:
            totals[fields[0]] = metric_total
    return {"node_count": len(nodes), "totals_bytes": totals} if totals else None


def parse_net_dev(text: str | None) -> dict[str, int] | None:
    if not text:
        return None
    totals = {
        "rx_bytes": 0,
        "rx_packets": 0,
        "rx_errors": 0,
        "rx_drops": 0,
        "tx_bytes": 0,
        "tx_packets": 0,
        "tx_errors": 0,
        "tx_drops": 0,
    }
    found = False
    for line in text.splitlines()[2:]:
        if ":" not in line:
            continue
        interface, raw = line.split(":", 1)
        if interface.strip() == "lo":
            continue
        fields = raw.split()
        if len(fields) < 12:
            continue
        try:
            values = [int(item) for item in fields]
        except ValueError:
            continue
        for key, value in zip(
            totals,
            (
                values[0],
                values[1],
                values[2],
                values[3],
                values[8],
                values[9],
                values[10],
                values[11],
            ),
        ):
            totals[key] += value
        found = True
    return totals if found else None


def _parse_mountinfo(proc_root: Path, cgroup_path: str) -> Path | None:
    text = _read_text(proc_root / "self" / "mountinfo")
    if not text:
        return None
    for line in text.splitlines():
        before, separator, after = line.partition(" - ")
        if not separator or not after.startswith("cgroup2 "):
            continue
        fields = before.split()
        if len(fields) < 5:
            continue
        mount_root = fields[3].rstrip("/") or "/"
        mount_point = Path(fields[4])
        if mount_root != "/" and cgroup_path.startswith(mount_root + "/"):
            relative = cgroup_path[len(mount_root) :].lstrip("/")
        elif mount_root == "/":
            relative = cgroup_path.lstrip("/")
        elif cgroup_path == mount_root:
            relative = ""
        else:
            continue
        return mount_point / relative
    return None


def discover_cgroup_v2(proc_root: Path = Path("/proc")) -> Path | None:
    membership = _read_text(proc_root / "self" / "cgroup")
    if not membership:
        return None
    for line in membership.splitlines():
        fields = line.split(":", 2)
        if len(fields) == 3 and fields[0] == "0" and fields[1] == "":
            discovered = _parse_mountinfo(proc_root, fields[2] or "/")
            if discovered is not None:
                return discovered
    fallback = Path("/sys/fs/cgroup")
    return fallback if (fallback / "cgroup.controllers").exists() else None


def _process_category(value: str) -> str:
    for category, pattern in PROCESS_CATEGORY_PATTERNS:
        if pattern.fullmatch(value):
            return category
    return "other"


def _delta(current: int | None, previous: int | None) -> int | None:
    if current is None or previous is None or current < previous:
        return None
    return current - previous


class Collector:
    def __init__(
        self,
        storage_path: Path,
        proc_root: Path = Path("/proc"),
        cgroup_root: Path | None = None,
        detail_profile: str = "basic",
    ):
        self.storage_path = storage_path
        self.proc_root = proc_root
        self.cgroup_root = (
            cgroup_root if cgroup_root is not None else discover_cgroup_v2(proc_root)
        )
        self.previous_cpu_usage: int | None = None
        self.previous_sample_ns: int | None = None
        self.page_size = getattr(os, "sysconf", lambda _key: 4096)("SC_PAGE_SIZE")
        self.clock_ticks = getattr(os, "sysconf", lambda _key: 100)("SC_CLK_TCK")
        self.detail_profile = detail_profile
        self.previous_processes: dict[tuple[int, int], dict[str, Any]] = {}
        self.tracked_processes: set[tuple[int, int]] = set()
        self.child_identity: tuple[int, int] | None = None
        self.previous_io_devices: dict[str, dict[str, int]] = {}
        self.previous_diskstats: dict[str, dict[str, int]] = {}

    def _cgroup_text(self, name: str) -> str | None:
        return _read_text(self.cgroup_root / name) if self.cgroup_root else None

    def _cgroup_int(self, name: str) -> int | None:
        return _read_int(self.cgroup_root / name) if self.cgroup_root else None

    def _ancestor_limits(self) -> dict[str, Any]:
        if not self.cgroup_root:
            return {
                "scope": "cgroup_ancestors",
                "levels_observed": 0,
                "effective_cpu_quota_cores": None,
                "effective_memory_limit_bytes": None,
                "effective_swap_limit_bytes": None,
            }
        cpu_limits: list[float] = []
        memory_limits: list[int] = []
        swap_limits: list[int] = []
        levels = 0
        current = self.cgroup_root
        while (current / "cgroup.controllers").exists():
            levels += 1
            cpu_max = _read_text(current / "cpu.max")
            if cpu_max:
                fields = cpu_max.split()
                try:
                    if len(fields) >= 2 and fields[0] != "max" and int(fields[1]) > 0:
                        cpu_limits.append(int(fields[0]) / int(fields[1]))
                except ValueError:
                    pass
            for name, destination in (
                ("memory.max", memory_limits),
                ("memory.swap.max", swap_limits),
            ):
                value = _read_int(current / name)
                if value is not None:
                    destination.append(value)
            if current == current.parent:
                break
            current = current.parent
        return {
            "scope": "cgroup_ancestors",
            "levels_observed": levels,
            "effective_cpu_quota_cores": min(cpu_limits) if cpu_limits else None,
            "effective_memory_limit_bytes": (
                min(memory_limits) if memory_limits else None
            ),
            "effective_swap_limit_bytes": min(swap_limits) if swap_limits else None,
        }

    def _numa_capacity(self) -> dict[str, Any]:
        node_root = self.proc_root.parent / "sys" / "devices" / "system" / "node"
        nodes: list[dict[str, int]] = []
        try:
            entries = list(node_root.iterdir())
        except OSError:
            entries = []
        for entry in entries:
            match = re.fullmatch(r"node(\d+)", entry.name)
            if not match:
                continue
            cpus = parse_cpu_list(_read_text(entry / "cpulist"))
            nodes.append({"node": int(match.group(1)), "cpu_count": len(cpus)})
        return {
            "available": bool(nodes),
            "node_count": len(nodes) if nodes else None,
            "nodes": sorted(nodes, key=lambda item: item["node"]) or None,
        }

    def capacity(self) -> dict[str, Any]:
        os_count = os.cpu_count()
        ancestor_limits = self._ancestor_limits()
        try:
            affinity = sorted(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            affinity = list(range(os_count or 0))

        cpuset = parse_cpu_list(self._cgroup_text("cpuset.cpus.effective"))
        if not cpuset:
            cpuset = parse_cpu_list(self._cgroup_text("cpuset.cpus"))

        quota = period = None
        cpu_max = self._cgroup_text("cpu.max")
        if cpu_max:
            fields = cpu_max.split()
            if len(fields) >= 2:
                try:
                    quota = None if fields[0] == "max" else int(fields[0])
                    period = int(fields[1])
                except ValueError:
                    quota = period = None

        candidates: list[float] = []
        if affinity:
            candidates.append(float(len(affinity)))
        if cpuset:
            candidates.append(float(len(cpuset)))
        if quota is not None and period:
            candidates.append(quota / period)
        ancestor_cpu_limit = ancestor_limits["effective_cpu_quota_cores"]
        if isinstance(ancestor_cpu_limit, (int, float)):
            candidates.append(float(ancestor_cpu_limit))
        effective = min(candidates) if candidates else None

        fs_total = None
        try:
            fs = os.statvfs(self.storage_path)
            fs_total = fs.f_blocks * fs.f_frsize
        except (AttributeError, OSError):
            pass

        return {
            "os_cpu_count": os_count,
            "affinity_cpu_count": len(affinity) if affinity else None,
            "affinity_cpus": affinity,
            "cpuset_cpu_count": len(cpuset) if cpuset else None,
            "cpuset_cpus": cpuset or None,
            "cgroup_cpu_quota_usec": quota,
            "cgroup_cpu_period_usec": period,
            "effective_cpu_count": effective,
            "cgroup_memory_limit_bytes": self._cgroup_int("memory.max"),
            "cgroup_swap_limit_bytes": self._cgroup_int("memory.swap.max"),
            "filesystem_total_bytes": fs_total,
            "cgroup_ancestor_limits": ancestor_limits,
            "numa": self._numa_capacity(),
        }

    def capabilities(self) -> dict[str, bool]:
        cgroup = self.cgroup_root
        return {
            "cgroup_v2": bool(cgroup and (cgroup / "cgroup.controllers").exists()),
            "cgroup_cpu": bool(cgroup and (cgroup / "cpu.stat").exists()),
            "cgroup_memory": bool(cgroup and (cgroup / "memory.current").exists()),
            "cgroup_io": bool(cgroup and parse_io_stat(self._cgroup_text("io.stat"))),
            "psi_cpu": bool(
                (cgroup and (cgroup / "cpu.pressure").exists())
                or (self.proc_root / "pressure" / "cpu").exists()
            ),
            "psi_memory": bool(
                (cgroup and (cgroup / "memory.pressure").exists())
                or (self.proc_root / "pressure" / "memory").exists()
            ),
            "psi_io": bool(
                (cgroup and (cgroup / "io.pressure").exists())
                or (self.proc_root / "pressure" / "io").exists()
            ),
            "network": (self.proc_root / "net" / "dev").exists(),
            "filesystem": self.storage_path.exists(),
            "processes": self.proc_root.exists() and os.name == "posix",
            "numa": bool(
                self._numa_capacity()["available"]
                or (cgroup and (cgroup / "memory.numa_stat").exists())
            ),
            "numa_topology": bool(self._numa_capacity()["available"]),
            "cgroup_numa": bool(cgroup and (cgroup / "memory.numa_stat").exists()),
            "host_diskstats": (self.proc_root / "diskstats").exists(),
        }

    def diagnostic_collector_status(self) -> dict[str, dict[str, Any]]:
        """Declare optional privileged collectors; never acquire capabilities."""
        paranoid = _read_int(self.proc_root / "sys" / "kernel" / "perf_event_paranoid")
        return {
            "perf": {
                "status": "unsupported",
                "reason": "collection_not_implemented_no_privileges_requested",
                "perf_event_paranoid": paranoid,
            },
            "ebpf": {
                "status": "unsupported",
                "reason": "collection_not_implemented_no_privileges_requested",
            },
        }

    def _pressure(self, resource_name: str) -> tuple[str | None, dict[str, Any] | None]:
        cgroup_text = self._cgroup_text(f"{resource_name}.pressure")
        if cgroup_text is not None:
            return "cgroup", parse_psi(cgroup_text)
        host_text = _read_text(self.proc_root / "pressure" / resource_name)
        return ("host", parse_psi(host_text)) if host_text is not None else (None, None)

    def _cpu(self, now_ns: int) -> dict[str, Any]:
        raw = parse_key_values(self._cgroup_text("cpu.stat"))
        usage = raw.get("usage_usec")
        interval_ns = (
            now_ns - self.previous_sample_ns
            if self.previous_sample_ns is not None
            else None
        )
        usage_delta = _delta(usage, self.previous_cpu_usage)
        cores_used = (
            usage_delta * 1000 / interval_ns
            if usage_delta is not None and interval_ns and interval_ns > 0
            else None
        )
        self.previous_cpu_usage = usage
        self.previous_sample_ns = now_ns

        load = _read_text(self.proc_root / "loadavg")
        load_values: dict[str, Any] | None = None
        if load:
            fields = load.split()
            try:
                runnable, tasks = fields[3].split("/", 1)
                load_values = {
                    "load1": float(fields[0]),
                    "load5": float(fields[1]),
                    "load15": float(fields[2]),
                    "runnable_tasks": int(runnable),
                    "total_tasks": int(tasks),
                }
            except (IndexError, ValueError):
                pass

        host_stat = parse_key_values(_read_text(self.proc_root / "stat"))
        return {
            "usage_usec": usage,
            "user_usec": raw.get("user_usec"),
            "system_usec": raw.get("system_usec"),
            "nr_periods": raw.get("nr_periods"),
            "nr_throttled": raw.get("nr_throttled"),
            "throttled_usec": raw.get("throttled_usec"),
            "cores_used": cores_used,
            "interval_ns": interval_ns,
            "load": load_values,
            "host": {
                "context_switches": host_stat.get("ctxt"),
                "processes_created": host_stat.get("processes"),
                "processes_running": host_stat.get("procs_running"),
                "processes_blocked": host_stat.get("procs_blocked"),
            },
        }

    def _memory(self) -> dict[str, Any]:
        memory_stat = parse_key_values(self._cgroup_text("memory.stat"))
        events = parse_key_values(self._cgroup_text("memory.events"))
        lifetime_peak = self._cgroup_int("memory.peak")
        meminfo: dict[str, int] = {}
        host_text = _read_text(self.proc_root / "meminfo")
        if host_text:
            for line in host_text.splitlines():
                key, separator, raw = line.partition(":")
                if not separator:
                    continue
                fields = raw.split()
                try:
                    meminfo[key] = int(fields[0]) * (1024 if len(fields) > 1 else 1)
                except (IndexError, ValueError):
                    continue
        return {
            "current_bytes": self._cgroup_int("memory.current"),
            "cgroup_lifetime_peak_bytes": lifetime_peak,
            "peak_scope": "cgroup_lifetime" if lifetime_peak is not None else None,
            "max_bytes": self._cgroup_int("memory.max"),
            "swap_current_bytes": self._cgroup_int("memory.swap.current"),
            "swap_max_bytes": self._cgroup_int("memory.swap.max"),
            "anon_bytes": memory_stat.get("anon"),
            "file_bytes": memory_stat.get("file"),
            "slab_bytes": memory_stat.get("slab"),
            "page_faults": memory_stat.get("pgfault"),
            "major_page_faults": memory_stat.get("pgmajfault"),
            "events": events or None,
            "host": {
                "total_bytes": meminfo.get("MemTotal"),
                "available_bytes": meminfo.get("MemAvailable"),
                "swap_total_bytes": meminfo.get("SwapTotal"),
                "swap_free_bytes": meminfo.get("SwapFree"),
            },
        }

    def _filesystem(self) -> dict[str, int] | None:
        try:
            fs = os.statvfs(self.storage_path)
        except (AttributeError, OSError):
            return None
        return {
            "total_bytes": fs.f_blocks * fs.f_frsize,
            "free_bytes": fs.f_bfree * fs.f_frsize,
            "available_bytes": fs.f_bavail * fs.f_frsize,
            "total_inodes": fs.f_files,
            "free_inodes": fs.f_ffree,
        }

    def _read_processes(self) -> dict[int, dict[str, Any]]:
        processes: dict[int, dict[str, Any]] = {}
        try:
            entries = list(self.proc_root.iterdir())
        except OSError:
            return processes
        for entry in entries:
            if not entry.name.isdigit():
                continue
            stat_text = _read_text(entry / "stat")
            if not stat_text:
                continue
            close = stat_text.rfind(")")
            open_ = stat_text.find("(")
            if open_ < 0 or close < open_:
                continue
            fields = stat_text[close + 1 :].split()
            try:
                pid = int(stat_text[:open_].strip())
                status = parse_key_values(_read_text(entry / "status"))
                io = parse_key_values(_read_text(entry / "io"))
                processes[pid] = {
                    "pid": pid,
                    "ppid": int(fields[1]),
                    "state": fields[0],
                    "category": _process_category(stat_text[open_ + 1 : close]),
                    "minor_faults": int(fields[7]),
                    "major_faults": int(fields[9]),
                    "user_ticks": int(fields[11]),
                    "system_ticks": int(fields[12]),
                    "cpu_ticks": int(fields[11]) + int(fields[12]),
                    "threads": int(fields[17]),
                    "starttime_ticks": int(fields[19]),
                    "rss_bytes": max(0, int(fields[21])) * int(self.page_size),
                    "voluntary_context_switches": status.get("voluntary_ctxt_switches"),
                    "involuntary_context_switches": status.get(
                        "nonvoluntary_ctxt_switches"
                    ),
                    "read_bytes": io.get("read_bytes"),
                    "write_bytes": io.get("write_bytes"),
                }
            except (IndexError, ValueError):
                continue
        return processes

    def _processes(self, child_pid: int | None) -> dict[str, Any] | None:
        if child_pid is None or os.name != "posix":
            return None
        processes = self._read_processes()

        root = processes.get(child_pid)
        if root is not None and self.child_identity is None:
            self.child_identity = (child_pid, root["starttime_ticks"])
        selected: set[int] = set()
        if root is not None and self.child_identity == (
            child_pid,
            root["starttime_ticks"],
        ):
            selected.add(child_pid)
        for pid, item in processes.items():
            identity = (pid, item["starttime_ticks"])
            if identity in self.tracked_processes:
                selected.add(pid)
        changed = True
        while changed:
            changed = False
            for pid, item in processes.items():
                if pid not in selected and item["ppid"] in selected:
                    selected.add(pid)
                    changed = True
        descendants = [processes[pid] for pid in selected if pid in processes]
        current = {(item["pid"], item["starttime_ticks"]): item for item in descendants}
        previous = self.previous_processes
        current_ids = set(current)
        previous_ids = set(previous)
        self.tracked_processes.update(current_ids)
        top: list[dict[str, Any]] = []
        selected_pids: set[int] = set()
        # Preserve representatives for distinct bottleneck dimensions. A pure
        # CPU sort can otherwise hide a linker or generator doing all of the I/O.
        for metric in ("cpu_ticks", "rss_bytes", "write_bytes", "read_bytes"):
            ranked = sorted(
                descendants,
                key=lambda item: item.get(metric) or 0,
                reverse=True,
            )
            for item in ranked[:2]:
                if item["pid"] not in selected_pids:
                    top.append(item)
                    selected_pids.add(item["pid"])
        for item in sorted(
            descendants,
            key=lambda item: (item["cpu_ticks"], item["rss_bytes"]),
            reverse=True,
        ):
            if len(top) >= 8:
                break
            if item["pid"] not in selected_pids:
                top.append(item)
                selected_pids.add(item["pid"])
        top = [
            {
                key: item.get(key)
                for key in (
                    "pid",
                    "ppid",
                    "category",
                    "rss_bytes",
                    "cpu_ticks",
                    "read_bytes",
                    "write_bytes",
                )
            }
            for item in top[:8]
        ]
        result = {
            "scope": "wrapped_process_tree",
            "count": len(descendants),
            "aggregate_rss_bytes": sum(item["rss_bytes"] for item in descendants),
            "top": top,
        }
        if self.detail_profile in ("process", "diagnostic"):
            categories: dict[str, dict[str, Any]] = {}
            for identity, item in current.items():
                prior = previous.get(identity, {})
                category = item["category"]
                aggregate = categories.setdefault(
                    category,
                    {
                        "category": category,
                        "process_count": 0,
                        "rss_bytes": 0,
                        "thread_count": 0,
                        "state_counts": {},
                        "user_cpu_ticks_delta": 0,
                        "system_cpu_ticks_delta": 0,
                        "minor_faults_delta": 0,
                        "major_faults_delta": 0,
                        "voluntary_context_switches_delta": 0,
                        "involuntary_context_switches_delta": 0,
                        "read_bytes_delta": 0,
                        "write_bytes_delta": 0,
                        "counter_delta_process_count": 0,
                        "new_process_count": 0,
                        "counter_scope": "sampled_survivors_lower_bound",
                    },
                )
                aggregate["process_count"] += 1
                aggregate["rss_bytes"] += item["rss_bytes"]
                aggregate["thread_count"] += item["threads"]
                if prior:
                    aggregate["counter_delta_process_count"] += 1
                else:
                    aggregate["new_process_count"] += 1
                states = aggregate["state_counts"]
                states[item["state"]] = states.get(item["state"], 0) + 1
                for source, destination in (
                    ("user_ticks", "user_cpu_ticks_delta"),
                    ("system_ticks", "system_cpu_ticks_delta"),
                    ("minor_faults", "minor_faults_delta"),
                    ("major_faults", "major_faults_delta"),
                    (
                        "voluntary_context_switches",
                        "voluntary_context_switches_delta",
                    ),
                    (
                        "involuntary_context_switches",
                        "involuntary_context_switches_delta",
                    ),
                    ("read_bytes", "read_bytes_delta"),
                    ("write_bytes", "write_bytes_delta"),
                ):
                    value = _delta(item.get(source), prior.get(source))
                    if value is not None:
                        aggregate[destination] += value
            result.update(
                {
                    "clock_ticks_per_second": self.clock_ticks,
                    "counter_scope": "sampled_survivors_lower_bound",
                    "counter_scope_note": (
                        "Counters omit processes that start and exit between samples; "
                        "newly observed process counters begin at their first observation."
                    ),
                    "category_aggregates": sorted(
                        categories.values(), key=lambda item: item["category"]
                    ),
                    "lifecycle": {
                        "scope": "sample_observed_process_identities",
                        "started_since_previous_sample": len(
                            current_ids - previous_ids
                        ),
                        "exited_since_previous_sample": len(previous_ids - current_ids),
                        "identity": "pid_and_starttime_ticks",
                    },
                }
            )
            self.previous_processes = current
        return result

    def _diagnostic_io(self) -> dict[str, Any]:
        current_cgroup = parse_io_stat_devices(self._cgroup_text("io.stat"))
        current_host = parse_diskstats(_read_text(self.proc_root / "diskstats"))
        device_ids = sorted(current_cgroup)

        def device_deltas(
            current: dict[str, dict[str, int]],
            previous: dict[str, dict[str, int]],
            selected: list[str],
        ) -> list[dict[str, Any]]:
            values: list[dict[str, Any]] = []
            for device_id in selected:
                counters = current.get(device_id, {})
                prior = previous.get(device_id, {})
                values.append(
                    {
                        "device_id": device_id,
                        "deltas": {
                            key: _delta(value, prior.get(key))
                            for key, value in counters.items()
                        },
                    }
                )
            return values

        result = {
            "cgroup_devices": device_deltas(
                current_cgroup, self.previous_io_devices, device_ids
            ),
            "host_diskstats": {
                "scope": "host_counters_for_cgroup_device_ids",
                "devices": device_deltas(
                    current_host, self.previous_diskstats, device_ids
                ),
            },
        }
        self.previous_io_devices = current_cgroup
        self.previous_diskstats = current_host
        return result

    def sample(
        self, now_ns: int, child_pid: int | None, include_processes: bool
    ) -> dict[str, Any]:
        cpu_psi_scope, cpu_psi = self._pressure("cpu")
        memory_psi_scope, memory_psi = self._pressure("memory")
        io_psi_scope, io_psi = self._pressure("io")
        result = {
            "cpu": self._cpu(now_ns),
            "memory": self._memory(),
            "io": parse_io_stat(self._cgroup_text("io.stat")),
            "network": parse_net_dev(_read_text(self.proc_root / "net" / "dev")),
            "filesystem": self._filesystem(),
            "psi": {
                "cpu": cpu_psi,
                "memory": memory_psi,
                "io": io_psi,
                "scope": {
                    "cpu": cpu_psi_scope,
                    "memory": memory_psi_scope,
                    "io": io_psi_scope,
                },
            },
            "processes": self._processes(child_pid) if include_processes else None,
        }
        if self.detail_profile == "diagnostic":
            result["diagnostic_io"] = self._diagnostic_io()
            result["numa"] = parse_numa_stat(self._cgroup_text("memory.numa_stat"))
        return result


class JsonlWriter:
    def __init__(self, path: Path):
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        self.stream = os.fdopen(descriptor, "w", encoding="utf-8", buffering=1)

    def write(self, record: dict[str, Any]) -> None:
        self.stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self.stream.flush()

    def close(self) -> None:
        self.stream.close()


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - index) + ordered[high] * (index - low)


def _nested_int(record: dict[str, Any], *keys: str) -> int | None:
    value: Any = record
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _summary(
    *,
    phase: str,
    sequence: int,
    start_utc: str,
    start_ns: int,
    end_ns: int,
    samples: list[dict[str, Any]],
    child_code: int,
    child_signal: int | None,
    forwarded_signal: int | None,
    collection_durations: list[int],
    missed_deadlines: int,
    collector_errors: int,
    self_cpu_ns: int,
    self_peak_rss_bytes: int | None,
    jsonl_bytes_before_summary: int | None,
) -> dict[str, Any]:
    cores = [
        value
        for value in (_nested_number(item, "cpu", "cores_used") for item in samples)
        if value is not None
    ]
    memory_values = [
        value
        for value in (_nested_int(item, "memory", "current_bytes") for item in samples)
        if value is not None
    ]
    filesystem_values = [
        value
        for value in (
            _nested_int(item, "filesystem", "available_bytes") for item in samples
        )
        if value is not None
    ]
    first = samples[0] if samples else {}
    last = samples[-1] if samples else {}
    process_samples = [
        item["processes"] for item in samples if isinstance(item.get("processes"), dict)
    ]
    process_totals: dict[str, dict[str, Any]] = {}
    process_interval_totals: dict[str, dict[str, Any]] = {}
    process_lifecycle = {"started": 0, "exited": 0}
    for process_sample in process_samples:
        for process in process_sample.get("top", []):
            category = process.get("category")
            if not isinstance(category, str):
                continue
            aggregate = process_totals.setdefault(
                category,
                {
                    "category": category,
                    "observed_max_cpu_ticks": 0,
                    "observed_peak_rss_bytes": 0,
                    "observed_max_read_bytes": 0,
                    "observed_max_write_bytes": 0,
                },
            )
            for source, destination in (
                ("cpu_ticks", "observed_max_cpu_ticks"),
                ("rss_bytes", "observed_peak_rss_bytes"),
                ("read_bytes", "observed_max_read_bytes"),
                ("write_bytes", "observed_max_write_bytes"),
            ):
                value = process.get(source)
                if isinstance(value, int):
                    aggregate[destination] = max(aggregate[destination], value)
        lifecycle = process_sample.get("lifecycle")
        if isinstance(lifecycle, dict):
            for source, destination in (
                ("started_since_previous_sample", "started"),
                ("exited_since_previous_sample", "exited"),
            ):
                value = lifecycle.get(source)
                if isinstance(value, int):
                    process_lifecycle[destination] += value
        for category_sample in process_sample.get("category_aggregates", []):
            category = category_sample.get("category")
            if not isinstance(category, str):
                continue
            aggregate = process_interval_totals.setdefault(
                category,
                {
                    "category": category,
                    "observed_peak_process_count": 0,
                    "observed_peak_rss_bytes": 0,
                    "observed_peak_thread_count": 0,
                    "user_cpu_ticks_delta": 0,
                    "system_cpu_ticks_delta": 0,
                    "minor_faults_delta": 0,
                    "major_faults_delta": 0,
                    "voluntary_context_switches_delta": 0,
                    "involuntary_context_switches_delta": 0,
                    "read_bytes_delta": 0,
                    "write_bytes_delta": 0,
                    "counter_delta_process_count": 0,
                    "new_process_count": 0,
                },
            )
            for source, destination in (
                ("process_count", "observed_peak_process_count"),
                ("rss_bytes", "observed_peak_rss_bytes"),
                ("thread_count", "observed_peak_thread_count"),
            ):
                value = category_sample.get(source)
                if isinstance(value, int):
                    aggregate[destination] = max(aggregate[destination], value)
            for key in (
                "user_cpu_ticks_delta",
                "system_cpu_ticks_delta",
                "minor_faults_delta",
                "major_faults_delta",
                "voluntary_context_switches_delta",
                "involuntary_context_switches_delta",
                "read_bytes_delta",
                "write_bytes_delta",
                "counter_delta_process_count",
                "new_process_count",
            ):
                value = category_sample.get(key)
                if isinstance(value, int):
                    aggregate[key] += value

    def counter_delta(section: str, key: str) -> int | None:
        return _delta(_nested_int(last, section, key), _nested_int(first, section, key))

    cpu_usage_delta = counter_delta("cpu", "usage_usec")
    sample_elapsed_ns = _delta(
        _nested_int(last, "monotonic_ns"),
        _nested_int(first, "monotonic_ns"),
    )
    weighted_cpu = [
        (cores_used, interval_ns)
        for item in samples
        if (cores_used := _nested_number(item, "cpu", "cores_used")) is not None
        and (interval_ns := _nested_int(item, "cpu", "interval_ns")) is not None
        and interval_ns > 0
    ]
    if cpu_usage_delta is not None and sample_elapsed_ns and sample_elapsed_ns > 0:
        average_cores_used = cpu_usage_delta * 1000 / sample_elapsed_ns
        average_cores_used_source = "cgroup_usage_delta"
    elif weighted_cpu:
        weighted_duration = sum(interval_ns for _, interval_ns in weighted_cpu)
        average_cores_used = (
            sum(value * interval_ns for value, interval_ns in weighted_cpu)
            / weighted_duration
        )
        average_cores_used_source = "interval_weighted_samples"
    else:
        average_cores_used = statistics.fmean(cores) if cores else None
        average_cores_used_source = "sample_mean" if cores else None

    event_delta: dict[str, int | None] = {}
    for key in ("oom", "oom_kill", "high", "max"):
        event_delta[key] = _delta(
            _nested_int(last, "memory", "events", key),
            _nested_int(first, "memory", "events", key),
        )

    return {
        "schema": SCHEMA,
        "record_type": "summary",
        "phase": phase,
        "timestamp_utc": _utc_now(),
        "monotonic_ns": end_ns,
        "sequence": sequence,
        "start_utc": start_utc,
        "end_utc": _utc_now(),
        "duration_ns": max(0, end_ns - start_ns),
        "sample_count": len(samples),
        "missed_deadline_count": missed_deadlines,
        "collector_error_count": collector_errors,
        "child_exit_code": child_code,
        "child_signal": child_signal,
        "forwarded_signal": forwarded_signal,
        "cpu": {
            "scope": "cgroup",
            "average_cores_used": average_cores_used,
            "average_cores_used_source": average_cores_used_source,
            "p50_cores_used": _percentile(cores, 0.50),
            "p95_cores_used": _percentile(cores, 0.95),
            "max_cores_used": max(cores) if cores else None,
            "usage_usec_delta": cpu_usage_delta,
            "user_usec_delta": counter_delta("cpu", "user_usec"),
            "system_usec_delta": counter_delta("cpu", "system_usec"),
            "nr_throttled_delta": counter_delta("cpu", "nr_throttled"),
            "throttled_usec_delta": counter_delta("cpu", "throttled_usec"),
        },
        "memory": {
            "scope": "cgroup",
            "sampled_peak_bytes": max(memory_values) if memory_values else None,
            "current_end_bytes": memory_values[-1] if memory_values else None,
            "cgroup_lifetime_peak_bytes": max(
                (
                    value
                    for value in (
                        _nested_int(item, "memory", "cgroup_lifetime_peak_bytes")
                        for item in samples
                    )
                    if value is not None
                ),
                default=None,
            ),
            "event_deltas": event_delta,
        },
        "io": {
            "scope": "cgroup",
            **{
                key + "_delta": counter_delta("io", key)
                for key in ("rbytes", "wbytes", "rios", "wios", "dbytes", "dios")
            },
        },
        "network": {
            "scope": "network_namespace",
            **{
                key + "_delta": counter_delta("network", key)
                for key in ("rx_bytes", "tx_bytes", "rx_drops", "tx_drops")
            },
        },
        "psi": {
            resource_name: {
                "scope": _nested_string(last, "psi", "scope", resource_name),
                **{
                    pressure_type
                    + "_total_usec_delta": _delta(
                        _nested_int(last, "psi", resource_name, pressure_type, "total"),
                        _nested_int(
                            first, "psi", resource_name, pressure_type, "total"
                        ),
                    )
                    for pressure_type in ("some", "full")
                },
            }
            for resource_name in ("cpu", "memory", "io")
        },
        "filesystem": {
            "scope": "storage_path_filesystem",
            "minimum_available_bytes": (
                min(filesystem_values) if filesystem_values else None
            ),
            "maximum_used_delta_bytes": (
                max(0, filesystem_values[0] - min(filesystem_values))
                if filesystem_values
                else None
            ),
        },
        "processes": {
            "scope": "child_process_tree",
            "observed_max_count": max(
                (item.get("count", 0) for item in process_samples), default=None
            ),
            "observed_max_aggregate_rss_bytes": max(
                (item.get("aggregate_rss_bytes", 0) for item in process_samples),
                default=None,
            ),
            "top_categories": sorted(
                process_totals.values(),
                key=lambda item: (
                    item["observed_max_cpu_ticks"],
                    item["observed_peak_rss_bytes"],
                ),
                reverse=True,
            )[:8],
            "interval_categories": sorted(
                process_interval_totals.values(), key=lambda item: item["category"]
            ),
            "interval_category_counter_scope": "sampled_survivors_lower_bound",
            "lifecycle": {
                "observed_started_count": process_lifecycle["started"],
                "observed_exited_count": process_lifecycle["exited"],
                "identity": "pid_and_starttime_ticks",
            },
        },
        "probe_overhead": {
            "self_cpu_ns": self_cpu_ns,
            "self_peak_rss_bytes": self_peak_rss_bytes,
            "jsonl_bytes_before_summary": jsonl_bytes_before_summary,
            "average_collection_duration_ns": (
                statistics.fmean(collection_durations) if collection_durations else None
            ),
            "p95_collection_duration_ns": _percentile(
                [float(value) for value in collection_durations], 0.95
            ),
            "max_collection_duration_ns": (
                max(collection_durations) if collection_durations else None
            ),
            "collection_overrun_count": missed_deadlines,
        },
    }


def _nested_number(record: dict[str, Any], *keys: str) -> float | None:
    value: Any = record
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _nested_string(record: dict[str, Any], *keys: str) -> str | None:
    value: Any = record
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value if isinstance(value, str) else None


def _peak_rss_bytes() -> int | None:
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    return int(usage if sys.platform == "darwin" else usage * 1024)


def _write_atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _normalized_return_code(return_code: int) -> tuple[int, int | None]:
    if return_code < 0:
        child_signal = -return_code
        return 128 + child_signal, child_signal
    return return_code, None


def _forward_signal(process: subprocess.Popen[Any], signum: int) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signum)
        else:  # pragma: no cover - Linux is the production path
            process.send_signal(signum)
    except (OSError, ProcessLookupError):
        pass


def _install_forwarders(
    process_getter: Callable[[], subprocess.Popen[Any] | None],
    state: dict[str, int | None],
) -> dict[int, Any]:
    previous: dict[int, Any] = {}

    def handler(signum: int, _frame: Any) -> None:
        if state["forwarded_signal"] is None:
            state["forwarded_signal"] = signum
        process = process_getter()
        if process is not None:
            _forward_signal(process, signum)

    for name in ("SIGINT", "SIGTERM", "SIGHUP", "SIGQUIT"):
        signum = getattr(signal, name, None)
        if signum is None:
            continue
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, handler)
    return previous


def _restore_handlers(previous: dict[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def run_probe(args: argparse.Namespace) -> int:
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("resource_probe: missing command after --", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    samples_path = output_dir / f"{args.phase}.samples.jsonl"
    summary_path = output_dir / f"{args.phase}.summary.json"
    writer: JsonlWriter | None = None
    telemetry_failure_reported = False
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        writer = JsonlWriter(samples_path)
    except OSError as error:
        telemetry_failure_reported = True
        print(
            f"resource_probe: telemetry disabled (output unavailable: {type(error).__name__})",
            file=sys.stderr,
        )

    def disable_writer(error: BaseException) -> None:
        """Permanently disable telemetry without changing the child result."""
        nonlocal telemetry_failure_reported, writer
        current = writer
        writer = None
        if not telemetry_failure_reported:
            telemetry_failure_reported = True
            print(
                "resource_probe: telemetry disabled after write failure "
                f"({type(error).__name__})",
                file=sys.stderr,
            )
        if current is not None:
            try:
                current.close()
            except (OSError, ValueError):
                pass

    def emit(record: dict[str, Any]) -> bool:
        if writer is None:
            return False
        try:
            writer.write(record)
            return True
        except (OSError, ValueError, TypeError) as error:
            disable_writer(error)
            return False

    def close_writer() -> bool:
        nonlocal writer
        current = writer
        writer = None
        if current is None:
            return False
        try:
            current.close()
            return True
        except (OSError, ValueError) as error:
            if not telemetry_failure_reported:
                print(
                    "resource_probe: telemetry disabled after close failure "
                    f"({type(error).__name__})",
                    file=sys.stderr,
                )
            return False

    self_cpu_start = time.process_time_ns()
    collector = Collector(Path(args.storage_path), detail_profile=args.detail_profile)
    capacity = collector.capacity()
    capabilities = collector.capabilities()
    start_ns = time.monotonic_ns()
    start_utc = _utc_now()
    process: subprocess.Popen[Any] | None = None
    signal_state: dict[str, int | None] = {"forwarded_signal": None}
    handlers = _install_forwarders(lambda: process, signal_state)
    samples: list[dict[str, Any]] = []
    durations: list[int] = []
    missed_deadlines = 0
    collector_errors = 0
    sequence = 0
    child_start_error: str | None = None

    try:
        baseline: tuple[int, str, int, dict[str, Any]] | None = None
        baseline_error: Exception | None = None
        if writer is not None:
            baseline_start = time.monotonic_ns()
            baseline_utc = _utc_now()
            try:
                baseline_payload = collector.sample(
                    baseline_start,
                    None,
                    include_processes=False,
                )
                baseline = (
                    baseline_start,
                    baseline_utc,
                    time.monotonic_ns() - baseline_start,
                    baseline_payload,
                )
            except Exception as error:  # keep telemetry failures observational
                baseline_error = error

        raw_return_code: int | None = None
        pending_signal = signal_state["forwarded_signal"]
        if pending_signal is not None:
            raw_return_code = -pending_signal
        else:
            try:
                process = subprocess.Popen(
                    command,
                    start_new_session=(os.name == "posix"),
                )
                # A signal can arrive after the pre-launch check but before
                # Popen returns. Replay it once the child process group exists.
                pending_signal = signal_state["forwarded_signal"]
                if pending_signal is not None:
                    _forward_signal(process, pending_signal)
            except (FileNotFoundError, PermissionError, OSError) as error:
                child_start_error = type(error).__name__
                pending_signal = signal_state["forwarded_signal"]
                raw_return_code = -pending_signal if pending_signal else 127

        metadata = {
            "schema": SCHEMA,
            "record_type": "metadata",
            "phase": args.phase,
            "timestamp_utc": start_utc,
            "monotonic_ns": start_ns,
            "sequence": sequence,
            "probe_version": PROBE_VERSION,
            "interval_seconds": args.interval_seconds,
            "detail_profile": args.detail_profile,
            "probe_pid": os.getpid(),
            "child_pid": process.pid if process else None,
            "child_process_category": _process_category(Path(command[0]).name),
            "child_start_error": child_start_error,
            "platform": {
                "system": sys.platform,
                "kernel_release": os.uname().release if hasattr(os, "uname") else None,
                "machine": os.uname().machine if hasattr(os, "uname") else None,
                "python_version": sys.version.split()[0],
                "page_size": collector.page_size,
            },
            "cgroup": {"version": 2 if collector.cgroup_root else None},
            "capacity": capacity,
            "capabilities": capabilities,
        }
        if args.detail_profile == "diagnostic":
            metadata["optional_collectors"] = collector.diagnostic_collector_status()
        emit(metadata)
        sequence += 1
        if writer is not None and baseline is not None:
            baseline_start, baseline_utc, duration, payload = baseline
            baseline_record = {
                "schema": SCHEMA,
                "record_type": "sample",
                "sample_kind": "baseline",
                "phase": args.phase,
                "timestamp_utc": baseline_utc,
                "monotonic_ns": baseline_start,
                "sequence": sequence,
                "elapsed_ns": max(0, baseline_start - start_ns),
                "collection_duration_ns": duration,
                **payload,
            }
            samples.append(baseline_record)
            durations.append(duration)
            emit(baseline_record)
            sequence += 1
        elif writer is not None and baseline_error is not None:
            collector_errors += 1
            if emit(
                {
                    "schema": SCHEMA,
                    "record_type": "probe_error",
                    "phase": args.phase,
                    "timestamp_utc": _utc_now(),
                    "monotonic_ns": time.monotonic_ns(),
                    "sequence": sequence,
                    "error_type": type(baseline_error).__name__,
                }
            ):
                sequence += 1

        interval_ns = int(args.interval_seconds * 1_000_000_000)
        deadline = start_ns
        while raw_return_code is None:
            now_ns = time.monotonic_ns()
            if writer is not None and now_ns >= deadline:
                collection_start = time.monotonic_ns()
                try:
                    payload = collector.sample(
                        collection_start,
                        process.pid if process else None,
                        include_processes=(
                            args.detail_profile in ("process", "diagnostic")
                            or len(samples) % 3 == 0
                        ),
                    )
                    duration = time.monotonic_ns() - collection_start
                    record = {
                        "schema": SCHEMA,
                        "record_type": "sample",
                        "sample_kind": "periodic",
                        "phase": args.phase,
                        "timestamp_utc": _utc_now(),
                        "monotonic_ns": collection_start,
                        "sequence": sequence,
                        "elapsed_ns": max(0, collection_start - start_ns),
                        "collection_duration_ns": duration,
                        **payload,
                    }
                    samples.append(record)
                    durations.append(duration)
                    emit(record)
                    sequence += 1
                except Exception as error:  # keep telemetry failures observational
                    collector_errors += 1
                    if collector_errors <= 8 and emit(
                        {
                            "schema": SCHEMA,
                            "record_type": "probe_error",
                            "phase": args.phase,
                            "timestamp_utc": _utc_now(),
                            "monotonic_ns": time.monotonic_ns(),
                            "sequence": sequence,
                            "error_type": type(error).__name__,
                        }
                    ):
                        sequence += 1
                deadline += interval_ns
                now_after = time.monotonic_ns()
                if now_after >= deadline:
                    skipped = (now_after - deadline) // interval_ns + 1
                    missed_deadlines += int(skipped)
                    deadline += skipped * interval_ns

            raw_return_code = process.poll() if process else raw_return_code
            if raw_return_code is None:
                remaining = (
                    max(0.0, (deadline - time.monotonic_ns()) / 1e9)
                    if writer is not None
                    else 0.25
                )
                try:
                    process.wait(timeout=min(remaining, 0.25))
                except subprocess.TimeoutExpired:
                    pass
                raw_return_code = process.poll()

        if writer is not None:
            final_start = time.monotonic_ns()
            try:
                payload = collector.sample(
                    final_start,
                    process.pid if process else None,
                    include_processes=True,
                )
                duration = time.monotonic_ns() - final_start
                final_record = {
                    "schema": SCHEMA,
                    "record_type": "sample",
                    "sample_kind": "final",
                    "phase": args.phase,
                    "timestamp_utc": _utc_now(),
                    "monotonic_ns": final_start,
                    "sequence": sequence,
                    "elapsed_ns": max(0, final_start - start_ns),
                    "collection_duration_ns": duration,
                    **payload,
                }
                samples.append(final_record)
                durations.append(duration)
                emit(final_record)
                sequence += 1
            except Exception:
                collector_errors += 1

        normalized, child_signal = _normalized_return_code(raw_return_code)
        end_ns = time.monotonic_ns()
        jsonl_bytes_before_summary = None
        if writer:
            try:
                jsonl_bytes_before_summary = samples_path.stat().st_size
            except OSError:
                pass
        summary = _summary(
            phase=args.phase,
            sequence=sequence,
            start_utc=start_utc,
            start_ns=start_ns,
            end_ns=end_ns,
            samples=samples,
            child_code=normalized,
            child_signal=child_signal,
            forwarded_signal=signal_state["forwarded_signal"],
            collection_durations=durations,
            missed_deadlines=missed_deadlines,
            collector_errors=collector_errors,
            self_cpu_ns=max(0, time.process_time_ns() - self_cpu_start),
            self_peak_rss_bytes=_peak_rss_bytes(),
            jsonl_bytes_before_summary=jsonl_bytes_before_summary,
        )
        if writer is not None:
            summary_emitted = emit(summary)
            writer_closed = close_writer()
            if summary_emitted and writer_closed:
                try:
                    _write_atomic_json(summary_path, summary)
                except (OSError, ValueError, TypeError) as error:
                    print(
                        f"resource_probe: summary unavailable ({type(error).__name__})",
                        file=sys.stderr,
                    )
        print(
            f"resource_probe: phase={args.phase} exit={normalized} samples={len(samples)}",
            file=sys.stderr,
        )
        return normalized
    finally:
        _restore_handlers(handlers)
        close_writer()


def _interval(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("interval must be a number") from error
    if not 1 <= parsed <= 60:
        raise argparse.ArgumentTypeError("interval must be between 1 and 60 seconds")
    return parsed


def _phase(value: str) -> str:
    if not PHASE_RE.fullmatch(value):
        raise argparse.ArgumentTypeError("invalid phase name")
    return value


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, type=_phase)
    parser.add_argument("--interval-seconds", type=_interval, default=5.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--storage-path", default=".")
    parser.add_argument(
        "--detail-profile",
        choices=("basic", "process", "diagnostic"),
        default="basic",
        help=(
            "basic preserves low-overhead sampling; process adds per-interval "
            "privacy-safe descendant aggregates; diagnostic also adds "
            "unprivileged NUMA and per-device I/O indicators"
        ),
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    return run_probe(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
