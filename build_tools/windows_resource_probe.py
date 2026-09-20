# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Privacy-safe, dependency-free resource telemetry for Windows CI commands.

The collector intentionally emits fixed process categories and numeric counters
only. It never records executable names, command lines, paths, environment
values, interface names, or volume labels. All Win32 calls used by the basic and
process profiles are available to an unprivileged process. The diagnostic
profile additionally attempts English-name PDH counters and reports capability
failure instead of failing the wrapped command.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import datetime as dt
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time
from typing import Any, Protocol


SCHEMA = "therock.windows_resource_probe.v1"
PROBE_VERSION = "1"
PHASE_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
PROCESS_CATEGORY_PATTERNS = (
    ("build_tool", re.compile(r"^(?:cmake|ninja|make|msbuild|devenv)$")),
    (
        "compiler",
        re.compile(r"^(?:cl|clang|clang\+\+|clang-cl|gcc|g\+\+|cc1|cc1plus)$"),
    ),
    ("linker", re.compile(r"^(?:link|lld-link|ld|ld\.lld)$")),
    ("binary_tool", re.compile(r"^(?:lib|llvm-ar|llvm-ranlib|mt|rc|strip)$")),
    ("runtime", re.compile(r"^(?:python(?:\d+(?:\.\d+)?)?|pypy\d*|perl|ruby|node)$")),
    ("shell", re.compile(r"^(?:bash|cmd|pwsh|powershell|sh)$")),
    (
        "transfer_or_cache_tool",
        re.compile(r"^(?:ccache|curl|dvc|gh|git|robocopy|sccache|tar|wget)$"),
    ),
)


def utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def categorize_executable(value: str) -> str:
    """Return an allowlisted category; the executable value is never emitted."""
    name = Path(value).name.lower()
    if name.endswith(".exe"):
        name = name[:-4]
    for category, pattern in PROCESS_CATEGORY_PATTERNS:
        if pattern.fullmatch(name):
            return category
    return "other"


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.5)))
    return ordered[index]


class Provider(Protocol):
    def capabilities(self, profile: str) -> dict[str, Any]: ...
    def host_snapshot(self, storage_path: str, profile: str) -> dict[str, Any]: ...
    def process_snapshot(self, root_pid: int) -> list[dict[str, Any]]: ...
    def close(self) -> None: ...


if sys.platform == "win32":  # pragma: no branch - declarations are Windows-only
    ULONG_PTR = wintypes.WPARAM
    MAX_PATH = 260
    TH32CS_SNAPPROCESS = 0x00000002
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    PROCESS_QUERY_INFORMATION = 0x0400
    PROCESS_VM_READ = 0x0010
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

    class FILETIME(ctypes.Structure):
        _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ULONG_PTR),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * MAX_PATH),
        ]

    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    class PERFORMANCE_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
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
            ("HandleCount", wintypes.DWORD),
            ("ProcessCount", wintypes.DWORD),
            ("ThreadCount", wintypes.DWORD),
        ]

    MAXLEN_IFDESCR = 256
    MAXLEN_PHYSADDR = 8

    class MIB_IFROW(ctypes.Structure):
        _fields_ = [
            ("wszName", wintypes.WCHAR * 256),
            ("dwIndex", wintypes.DWORD),
            ("dwType", wintypes.DWORD),
            ("dwMtu", wintypes.DWORD),
            ("dwSpeed", wintypes.DWORD),
            ("dwPhysAddrLen", wintypes.DWORD),
            ("bPhysAddr", ctypes.c_ubyte * MAXLEN_PHYSADDR),
            ("dwAdminStatus", wintypes.DWORD),
            ("dwOperStatus", wintypes.DWORD),
            ("dwLastChange", wintypes.DWORD),
            ("dwInOctets", wintypes.DWORD),
            ("dwInUcastPkts", wintypes.DWORD),
            ("dwInNUcastPkts", wintypes.DWORD),
            ("dwInDiscards", wintypes.DWORD),
            ("dwInErrors", wintypes.DWORD),
            ("dwInUnknownProtos", wintypes.DWORD),
            ("dwOutOctets", wintypes.DWORD),
            ("dwOutUcastPkts", wintypes.DWORD),
            ("dwOutNUcastPkts", wintypes.DWORD),
            ("dwOutDiscards", wintypes.DWORD),
            ("dwOutErrors", wintypes.DWORD),
            ("dwOutQLen", wintypes.DWORD),
            ("dwDescrLen", wintypes.DWORD),
            ("bDescr", ctypes.c_ubyte * MAXLEN_IFDESCR),
        ]


def _filetime_value(value: Any) -> int:
    return (int(value.high) << 32) | int(value.low)


class WindowsProvider:
    """Win32 provider. Construct only on Windows."""

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise OSError("WindowsProvider requires Windows")
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.psapi = ctypes.WinDLL("psapi", use_last_error=True)
        self.iphlpapi = ctypes.WinDLL("iphlpapi", use_last_error=True)
        self.kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        self.kernel32.CreateToolhelp32Snapshot.argtypes = [
            wintypes.DWORD,
            wintypes.DWORD,
        ]
        self.kernel32.OpenProcess.restype = wintypes.HANDLE
        self.kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        self.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        self.kernel32.GetCurrentProcess.argtypes = []
        self.kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel32.CloseHandle.restype = wintypes.BOOL
        self.kernel32.GetActiveProcessorGroupCount.argtypes = []
        self.kernel32.GetActiveProcessorGroupCount.restype = wintypes.WORD
        self.kernel32.GetActiveProcessorCount.argtypes = [wintypes.WORD]
        self.kernel32.GetActiveProcessorCount.restype = wintypes.DWORD
        self.kernel32.GetProcessAffinityMask.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.kernel32.GetProcessAffinityMask.restype = wintypes.BOOL
        self.kernel32.GetNumaHighestNodeNumber.argtypes = [
            ctypes.POINTER(wintypes.ULONG)
        ]
        self.kernel32.GetNumaHighestNodeNumber.restype = wintypes.BOOL
        self.kernel32.GetSystemTimes.argtypes = [
            ctypes.POINTER(FILETIME),
            ctypes.POINTER(FILETIME),
            ctypes.POINTER(FILETIME),
        ]
        self.kernel32.GetSystemTimes.restype = wintypes.BOOL
        self.kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
        self.kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
        self.kernel32.GetDiskFreeSpaceExW.argtypes = [
            wintypes.LPCWSTR,
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_ulonglong),
        ]
        self.kernel32.GetDiskFreeSpaceExW.restype = wintypes.BOOL
        self.kernel32.GetProcessTimes.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(FILETIME),
            ctypes.POINTER(FILETIME),
            ctypes.POINTER(FILETIME),
            ctypes.POINTER(FILETIME),
        ]
        self.kernel32.GetProcessTimes.restype = wintypes.BOOL
        self.kernel32.GetProcessIoCounters.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(IO_COUNTERS),
        ]
        self.kernel32.GetProcessIoCounters.restype = wintypes.BOOL
        self.kernel32.Process32FirstW.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESSENTRY32W),
        ]
        self.kernel32.Process32FirstW.restype = wintypes.BOOL
        self.kernel32.Process32NextW.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESSENTRY32W),
        ]
        self.kernel32.Process32NextW.restype = wintypes.BOOL
        self.psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
            wintypes.DWORD,
        ]
        self.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        self.psapi.GetPerformanceInfo.argtypes = [
            ctypes.POINTER(PERFORMANCE_INFORMATION),
            wintypes.DWORD,
        ]
        self.psapi.GetPerformanceInfo.restype = wintypes.BOOL
        self.iphlpapi.GetIfTable.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(wintypes.ULONG),
            wintypes.BOOL,
        ]
        self.iphlpapi.GetIfTable.restype = wintypes.DWORD
        self._pdh: Any = None
        self._pdh_query: Any = None
        self._pdh_counters: dict[str, Any] = {}
        self._tracked_processes: dict[int, int] = {}

    def close(self) -> None:
        if self._pdh is not None and self._pdh_query:
            self._pdh.PdhCloseQuery(self._pdh_query)
            self._pdh_query = None

    def _pdh_setup(self) -> tuple[bool, str | None]:
        if self._pdh is not None:
            return bool(self._pdh_query), (
                None if self._pdh_query else "initialization_failed"
            )
        try:
            self._pdh = ctypes.WinDLL("pdh", use_last_error=True)
            self._pdh.PdhOpenQueryW.argtypes = [
                wintypes.LPCWSTR,
                ctypes.c_size_t,
                ctypes.POINTER(wintypes.HANDLE),
            ]
            self._pdh.PdhOpenQueryW.restype = wintypes.LONG
            self._pdh.PdhAddEnglishCounterW.argtypes = [
                wintypes.HANDLE,
                wintypes.LPCWSTR,
                ctypes.c_size_t,
                ctypes.POINTER(wintypes.HANDLE),
            ]
            self._pdh.PdhAddEnglishCounterW.restype = wintypes.LONG
            self._pdh.PdhCollectQueryData.argtypes = [wintypes.HANDLE]
            self._pdh.PdhCollectQueryData.restype = wintypes.LONG
            self._pdh.PdhCloseQuery.argtypes = [wintypes.HANDLE]
            self._pdh.PdhCloseQuery.restype = wintypes.LONG
            query = wintypes.HANDLE()
            if self._pdh.PdhOpenQueryW(None, 0, ctypes.byref(query)) != 0:
                return False, "PdhOpenQueryW_failed"
            requested = {
                "cpu_percent": r"\Processor(_Total)\% Processor Time",
                "processor_queue_length": r"\System\Processor Queue Length",
                "disk_bytes_per_second": r"\PhysicalDisk(_Total)\Disk Bytes/sec",
                "disk_seconds_per_transfer": r"\PhysicalDisk(_Total)\Avg. Disk sec/Transfer",
                "context_switches_per_second": r"\System\Context Switches/sec",
            }
            for key, path in requested.items():
                counter = wintypes.HANDLE()
                if (
                    self._pdh.PdhAddEnglishCounterW(
                        query, path, 0, ctypes.byref(counter)
                    )
                    == 0
                ):
                    self._pdh_counters[key] = counter
            if not self._pdh_counters:
                self._pdh.PdhCloseQuery(query)
                return False, "no_English_counters_available"
            self._pdh_query = query
            self._pdh.PdhCollectQueryData(query)
            return True, None
        except (AttributeError, OSError):
            return False, "pdh_unavailable"

    def capabilities(self, profile: str) -> dict[str, Any]:
        pdh_supported, pdh_reason = (
            self._pdh_setup()
            if profile == "diagnostic"
            else (False, "profile_not_diagnostic")
        )
        return {
            "win32_process_tree": {"supported": profile != "basic"},
            "cpu_groups_affinity_numa": {"supported": True},
            "physical_commit_memory": {"supported": True},
            "volume_free_space": {"supported": True},
            "host_network_bytes": {"supported": True, "scope": "host"},
            "pdh_english_counters": {"supported": pdh_supported, "reason": pdh_reason},
            "etw": {"supported": False, "reason": "not_collected_non_admin_probe"},
            "pmu": {"supported": False, "reason": "privilege_and_hardware_dependent"},
            "per_process_network": {
                "supported": False,
                "reason": "not_available_from_non_admin_win32_counters",
            },
            "per_process_context_switches": {
                "supported": False,
                "reason": "not_available_from_documented_non_admin_process_api",
            },
        }

    def _cpu_topology(self) -> dict[str, Any]:
        groups = int(self.kernel32.GetActiveProcessorGroupCount())
        per_group = [
            int(self.kernel32.GetActiveProcessorCount(i)) for i in range(groups)
        ]
        process_mask = ctypes.c_size_t()
        system_mask = ctypes.c_size_t()
        affinity_ok = bool(
            self.kernel32.GetProcessAffinityMask(
                self.kernel32.GetCurrentProcess(),
                ctypes.byref(process_mask),
                ctypes.byref(system_mask),
            )
        )
        highest_node = wintypes.ULONG()
        numa_ok = bool(
            self.kernel32.GetNumaHighestNodeNumber(ctypes.byref(highest_node))
        )
        idle = FILETIME()
        kernel = FILETIME()
        user = FILETIME()
        times_ok = bool(
            self.kernel32.GetSystemTimes(
                ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
            )
        )
        return {
            "logical_processors": sum(per_group),
            "processor_group_count": groups,
            "logical_processors_per_group": per_group,
            "process_affinity_mask_bits_current_group": (
                int(process_mask.value).bit_count() if affinity_ok else None
            ),
            "system_affinity_mask_bits_current_group": (
                int(system_mask.value).bit_count() if affinity_ok else None
            ),
            "numa_node_count": int(highest_node.value) + 1 if numa_ok else None,
            "system_idle_100ns": _filetime_value(idle) if times_ok else None,
            "system_kernel_100ns": _filetime_value(kernel) if times_ok else None,
            "system_user_100ns": _filetime_value(user) if times_ok else None,
        }

    def _memory(self) -> dict[str, Any]:
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(status)
        status_ok = bool(self.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)))
        perf = PERFORMANCE_INFORMATION()
        perf.cb = ctypes.sizeof(perf)
        perf_ok = bool(self.psapi.GetPerformanceInfo(ctypes.byref(perf), perf.cb))
        page = int(perf.PageSize) if perf_ok else 0
        return {
            "status": "available" if status_ok else "unavailable",
            "physical_total_bytes": int(status.ullTotalPhys) if status_ok else None,
            "physical_available_bytes": int(status.ullAvailPhys) if status_ok else None,
            "memory_load_percent": int(status.dwMemoryLoad) if status_ok else None,
            "commit_total_bytes": int(perf.CommitTotal) * page if perf_ok else None,
            "commit_limit_bytes": int(perf.CommitLimit) * page if perf_ok else None,
            "commit_peak_bytes": int(perf.CommitPeak) * page if perf_ok else None,
        }

    def _network(self) -> dict[str, Any]:
        # GetIfTable supplies host aggregate DWORD counters. Interface identity,
        # description, and address fields are deliberately discarded.
        size = wintypes.ULONG(0)
        self.iphlpapi.GetIfTable(None, ctypes.byref(size), False)
        if not size.value:
            return {
                "scope": "host",
                "status": "unavailable",
                "reason": "size_query_failed",
            }
        buffer = ctypes.create_string_buffer(size.value)
        if self.iphlpapi.GetIfTable(buffer, ctypes.byref(size), False) != 0:
            return {
                "scope": "host",
                "status": "unavailable",
                "reason": "GetIfTable_failed",
            }
        count = ctypes.cast(buffer, ctypes.POINTER(wintypes.DWORD))[0]
        total_in = total_out = 0
        base = ctypes.addressof(buffer) + 4
        for index in range(int(count)):
            row = ctypes.cast(
                base + index * ctypes.sizeof(MIB_IFROW), ctypes.POINTER(MIB_IFROW)
            ).contents
            total_in += int(row.dwInOctets)
            total_out += int(row.dwOutOctets)
        return {
            "scope": "host",
            "status": "available",
            "counter_width_bits": 32,
            "received_bytes": total_in,
            "sent_bytes": total_out,
        }

    def _pdh_values(self) -> dict[str, float] | None:
        if not self._pdh_query:
            return None
        if self._pdh.PdhCollectQueryData(self._pdh_query) != 0:
            return None
        values: dict[str, float] = {}
        for key, counter in self._pdh_counters.items():
            counter_type = wintypes.DWORD()
            value = ctypes.c_double()

            # PDH_FMT_DOUBLE = 0x200; the union's double begins after CStatus/padding.
            class PDH_FMT_COUNTERVALUE_DOUBLE(ctypes.Structure):
                _fields_ = [
                    ("CStatus", wintypes.DWORD),
                    ("doubleValue", ctypes.c_double),
                ]

            formatted = PDH_FMT_COUNTERVALUE_DOUBLE()
            self._pdh.PdhGetFormattedCounterValue.argtypes = [
                wintypes.HANDLE,
                wintypes.DWORD,
                ctypes.POINTER(wintypes.DWORD),
                ctypes.c_void_p,
            ]
            self._pdh.PdhGetFormattedCounterValue.restype = wintypes.LONG
            if (
                self._pdh.PdhGetFormattedCounterValue(
                    counter, 0x200, ctypes.byref(counter_type), ctypes.byref(formatted)
                )
                == 0
                and formatted.CStatus == 0
            ):
                values[key] = float(formatted.doubleValue)
        return (
            {"scope": "host", "status": "available", "values": values}
            if values
            else {"scope": "host", "status": "unavailable"}
        )

    def host_snapshot(self, storage_path: str, profile: str) -> dict[str, Any]:
        free = ctypes.c_ulonglong()
        total = ctypes.c_ulonglong()
        total_free = ctypes.c_ulonglong()
        volume = f"{Path(storage_path).drive or 'B:'}\\"
        disk_ok = bool(
            self.kernel32.GetDiskFreeSpaceExW(
                volume,
                ctypes.byref(free),
                ctypes.byref(total),
                ctypes.byref(total_free),
            )
        )
        result = {
            "cpu": self._cpu_topology(),
            "memory": self._memory(),
            "storage": (
                {
                    "scope": "selected_volume",
                    "status": "available",
                    "free_bytes": int(free.value),
                    "total_bytes": int(total.value),
                }
                if disk_ok
                else {"scope": "selected_volume", "status": "unavailable"}
            ),
            "network": self._network(),
        }
        if profile == "diagnostic":
            result["pdh"] = self._pdh_values()
        return result

    def process_snapshot(self, root_pid: int) -> list[dict[str, Any]]:
        snapshot = self.kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        if snapshot == INVALID_HANDLE_VALUE:
            return []
        entries: list[tuple[int, int, int, str]] = []
        try:
            entry = PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(entry)
            ok = self.kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
            while ok:
                entries.append(
                    (
                        int(entry.th32ProcessID),
                        int(entry.th32ParentProcessID),
                        int(entry.cntThreads),
                        categorize_executable(entry.szExeFile),
                    )
                )
                ok = self.kernel32.Process32NextW(snapshot, ctypes.byref(entry))
        finally:
            self.kernel32.CloseHandle(snapshot)
        # Validate retained PID+creation identities before using them as parent
        # anchors. This prevents PID reuse from attaching unrelated processes.
        live_tracked: dict[int, int] = {}
        for tracked_pid, tracked_creation in self._tracked_processes.items():
            tracked_handle = self.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, tracked_pid
            )
            if not tracked_handle:
                continue
            try:
                creation = FILETIME()
                exit_time = FILETIME()
                kernel = FILETIME()
                user = FILETIME()
                if self.kernel32.GetProcessTimes(
                    tracked_handle,
                    ctypes.byref(creation),
                    ctypes.byref(exit_time),
                    ctypes.byref(kernel),
                    ctypes.byref(user),
                ):
                    if _filetime_value(creation) == tracked_creation:
                        live_tracked[tracked_pid] = tracked_creation
            finally:
                self.kernel32.CloseHandle(tracked_handle)
        self._tracked_processes = live_tracked
        selected = {root_pid, *live_tracked.keys()}
        changed = True
        while changed:
            changed = False
            for pid, parent, _, _ in entries:
                if parent in selected and pid not in selected:
                    selected.add(pid)
                    changed = True
        output = []
        for pid, _, threads, category in entries:
            if pid not in selected:
                continue
            memory_access = True
            handle = self.kernel32.OpenProcess(
                PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid
            )
            if not handle:
                memory_access = False
                handle = self.kernel32.OpenProcess(
                    PROCESS_QUERY_LIMITED_INFORMATION, False, pid
                )
            if not handle:
                continue
            try:
                creation = FILETIME()
                exit_time = FILETIME()
                kernel = FILETIME()
                user = FILETIME()
                mem = PROCESS_MEMORY_COUNTERS_EX()
                mem.cb = ctypes.sizeof(mem)
                io = IO_COUNTERS()
                times_ok = bool(
                    self.kernel32.GetProcessTimes(
                        handle,
                        ctypes.byref(creation),
                        ctypes.byref(exit_time),
                        ctypes.byref(kernel),
                        ctypes.byref(user),
                    )
                )
                if not times_ok:
                    continue
                mem_ok = memory_access and bool(
                    self.psapi.GetProcessMemoryInfo(handle, ctypes.byref(mem), mem.cb)
                )
                io_ok = bool(
                    self.kernel32.GetProcessIoCounters(handle, ctypes.byref(io))
                )
                creation_value = _filetime_value(creation)
                if (
                    pid in self._tracked_processes
                    and self._tracked_processes[pid] != creation_value
                ):
                    continue
                self._tracked_processes[pid] = creation_value
                output.append(
                    {
                        "identity": f"{pid}:{creation_value}",
                        "pid": pid,
                        "category": category,
                        "threads": threads,
                        "cpu_100ns": (
                            _filetime_value(kernel) + _filetime_value(user)
                            if times_ok
                            else None
                        ),
                        "memory_status": "available" if mem_ok else "unavailable",
                        "rss_bytes": int(mem.WorkingSetSize) if mem_ok else None,
                        "private_bytes": int(mem.PrivateUsage) if mem_ok else None,
                        "page_faults": int(mem.PageFaultCount) if mem_ok else None,
                        "read_bytes": int(io.ReadTransferCount) if io_ok else None,
                        "write_bytes": int(io.WriteTransferCount) if io_ok else None,
                    }
                )
            finally:
                self.kernel32.CloseHandle(handle)
        return output


def aggregate_processes(processes: list[dict[str, Any]]) -> dict[str, Any]:
    categories: dict[str, dict[str, int]] = {}
    total = {
        key: 0
        for key in (
            "processes",
            "threads",
            "cpu_100ns",
            "rss_bytes",
            "private_bytes",
            "page_faults",
            "read_bytes",
            "write_bytes",
            "memory_available_processes",
            "memory_unavailable_processes",
        )
    }
    for item in processes:
        category = item["category"]
        bucket = categories.setdefault(category, {key: 0 for key in total})
        bucket["processes"] += 1
        total["processes"] += 1
        memory_available = item.get("memory_status") == "available" or (
            "memory_status" not in item and item.get("rss_bytes") is not None
        )
        memory_key = (
            "memory_available_processes"
            if memory_available
            else "memory_unavailable_processes"
        )
        bucket[memory_key] += 1
        total[memory_key] += 1
        for key in total:
            if (
                key
                not in (
                    "processes",
                    "memory_available_processes",
                    "memory_unavailable_processes",
                )
                and item.get(key) is not None
            ):
                value = int(item[key])
                bucket[key] += value
                total[key] += value
    return {"total": total, "categories": categories}


def process_interval(
    processes: list[dict[str, Any]],
    previous: dict[str, dict[str, Any]],
    elapsed_seconds: float,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Aggregate current gauges and interval deltas without emitting identities."""
    current = {str(item["identity"]): item for item in processes}
    result = aggregate_processes(processes)
    delta_keys = ("cpu_100ns", "page_faults", "read_bytes", "write_bytes")
    interval_total = {key: 0 for key in delta_keys}
    interval_categories: dict[str, dict[str, int]] = {}
    for identity, item in current.items():
        old = previous.get(identity, {})
        category = item["category"]
        bucket = interval_categories.setdefault(
            category, {key: 0 for key in delta_keys}
        )
        for key in delta_keys:
            value = item.get(key)
            if value is None:
                continue
            delta = max(0, int(value) - int(old.get(key) or 0))
            interval_total[key] += delta
            bucket[key] += delta
    interval_total["average_cores_used"] = (
        interval_total["cpu_100ns"] / 10_000_000.0 / elapsed_seconds
        if elapsed_seconds > 0
        else None
    )
    result["interval"] = {
        "elapsed_seconds": elapsed_seconds,
        "total": interval_total,
        "categories": interval_categories,
        "processes_started": len(set(current) - set(previous)),
        "processes_exited": len(set(previous) - set(current)),
    }
    return result, current


def add_host_intervals(
    host: dict[str, Any], previous: dict[str, Any] | None, elapsed_seconds: float
) -> None:
    cpu = host.get("cpu") or {}
    old_cpu = (previous or {}).get("cpu") or {}
    fields = ("system_idle_100ns", "system_kernel_100ns", "system_user_100ns")
    if not previous or any(
        cpu.get(k) is None or old_cpu.get(k) is None for k in fields
    ):
        cpu["interval"] = {"status": "baseline"}
        return
    idle = max(0, int(cpu[fields[0]]) - int(old_cpu[fields[0]]))
    kernel = max(0, int(cpu[fields[1]]) - int(old_cpu[fields[1]]))
    user = max(0, int(cpu[fields[2]]) - int(old_cpu[fields[2]]))
    total = kernel + user
    busy = max(0, total - idle)
    logical = int(cpu.get("logical_processors") or 0)
    cores = busy / 10_000_000.0 / elapsed_seconds if elapsed_seconds > 0 else None
    cpu["interval"] = {
        "status": "available",
        "elapsed_seconds": elapsed_seconds,
        "busy_100ns": busy,
        "total_100ns": total,
        "average_cores_used": cores,
        "capacity_percent": (
            (cores / logical * 100.0) if cores is not None and logical else None
        ),
    }


def build_summary(
    phase: str,
    profile: str,
    started: str,
    ended: str,
    duration: float,
    exit_code: int,
    samples: list[dict[str, Any]],
    capabilities: dict[str, Any],
    collection_ms: list[float],
    errors: int,
) -> dict[str, Any]:
    memory_samples = [
        s
        for s in samples
        if s.get("process_tree")
        and s["process_tree"]["total"].get("memory_available_processes", 0) > 0
    ]
    rss = [float(s["process_tree"]["total"]["rss_bytes"]) for s in memory_samples]
    private = [
        float(s["process_tree"]["total"]["private_bytes"]) for s in memory_samples
    ]
    free = [
        float(s["host"]["storage"]["free_bytes"])
        for s in samples
        if s["host"].get("storage", {}).get("status") == "available"
    ]
    host_cores = [
        float(s["host"]["cpu"]["interval"]["average_cores_used"])
        for s in samples
        if s["host"]["cpu"].get("interval", {}).get("average_cores_used") is not None
    ]
    tree_cores = [
        float(s["process_tree"]["interval"]["total"]["average_cores_used"])
        for s in samples
        if s.get("process_tree")
        and s["process_tree"]["interval"]["total"].get("average_cores_used") is not None
    ]
    interval_keys = ("cpu_100ns", "page_faults", "read_bytes", "write_bytes")
    interval_totals: dict[str, Any] = {key: 0 for key in interval_keys}
    category_totals: dict[str, dict[str, int]] = {}
    lifecycle = {"processes_started": 0, "processes_exited": 0}
    for sample in samples:
        tree = sample.get("process_tree")
        if not tree:
            continue
        interval = tree["interval"]
        for key in interval_keys:
            interval_totals[key] += int(interval["total"].get(key) or 0)
        for key in lifecycle:
            lifecycle[key] += int(interval.get(key) or 0)
        for category, values in interval["categories"].items():
            bucket = category_totals.setdefault(
                category, {key: 0 for key in interval_keys}
            )
            for key in interval_keys:
                bucket[key] += int(values.get(key) or 0)
    return {
        "schema": SCHEMA,
        "probe_version": PROBE_VERSION,
        "platform": "windows",
        "phase": phase,
        "detail_profile": profile,
        "started_at": started,
        "ended_at": ended,
        "duration_seconds": duration,
        "command_exit_code": exit_code,
        "sample_count": len(samples),
        "collection_error_count": errors,
        "capabilities": capabilities,
        "peaks": {
            "rss_bytes": max(rss) if rss else None,
            "private_bytes": max(private) if private else None,
            "storage_min_free_bytes": min(free) if free else None,
        },
        "host_cpu_cores_used": {
            "average": statistics.fmean(host_cores) if host_cores else None,
            "p50": percentile(host_cores, 0.50),
            "p95": percentile(host_cores, 0.95),
            "max": max(host_cores) if host_cores else None,
        },
        "process_tree_cpu_cores_used": {
            "average": statistics.fmean(tree_cores) if tree_cores else None,
            "p50": percentile(tree_cores, 0.50),
            "p95": percentile(tree_cores, 0.95),
            "max": max(tree_cores) if tree_cores else None,
        },
        "process_tree_interval_totals": {
            **interval_totals,
            **lifecycle,
            "categories": category_totals,
        },
        "counter_scopes": {
            "cpu": "host_and_wrapped_process_tree",
            "memory": "host_and_wrapped_process_tree",
            "storage": "selected_volume",
            "network": "host_all_interfaces",
            "pdh": "host",
        },
        "probe_overhead": {
            "collection_ms_p50": (
                statistics.median(collection_ms) if collection_ms else None
            ),
            "collection_ms_p95": percentile(collection_ms, 0.95),
            "collection_ms_max": max(collection_ms) if collection_ms else None,
        },
        "privacy": {
            "process_names": False,
            "command_lines": False,
            "paths": False,
            "environment": False,
            "interface_names": False,
            "fixed_process_categories_only": True,
        },
    }


def run_probe(args: argparse.Namespace, provider: Provider) -> int:
    output_dir = Path(args.output_dir)
    jsonl_path = output_dir / f"{args.phase}.jsonl"
    summary_path = output_dir / f"{args.phase}.summary.json"
    started_at = utc_now()
    start = time.monotonic()
    samples: list[dict[str, Any]] = []
    collection_ms: list[float] = []
    errors = 0
    capabilities = provider.capabilities(args.detail_profile)
    stream = None
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        stream = jsonl_path.open("w", encoding="utf-8")
    except OSError as error:
        errors += 1
        print(
            f"windows_resource_probe: telemetry output unavailable ({type(error).__name__})",
            file=sys.stderr,
        )
    previous_host: dict[str, Any] | None = None
    previous_processes: dict[str, dict[str, Any]] = {}

    def collect(kind: str, root_pid: int | None) -> None:
        nonlocal errors, previous_host, previous_processes, stream
        collection_start = time.perf_counter()
        try:
            now = time.monotonic()
            host = provider.host_snapshot(args.storage_path, args.detail_profile)
            elapsed = now - collect.last_time if collect.last_time is not None else 0.0
            add_host_intervals(host, previous_host, elapsed)
            tree = None
            if args.detail_profile != "basic" and root_pid is not None:
                tree, previous_processes = process_interval(
                    provider.process_snapshot(root_pid), previous_processes, elapsed
                )
            sample = {
                "schema": SCHEMA,
                "phase": args.phase,
                "sample_kind": kind,
                "timestamp": utc_now(),
                "elapsed_seconds": now - start,
                "host": host,
                "process_tree": tree,
            }
            if stream is not None:
                stream.write(
                    json.dumps(sample, sort_keys=True, separators=(",", ":")) + "\n"
                )
                stream.flush()
            samples.append(sample)
            previous_host = host
            collect.last_time = now
        except Exception as error:
            errors += 1
            print(
                f"windows_resource_probe: sample unavailable ({type(error).__name__})",
                file=sys.stderr,
            )
        collection_ms.append((time.perf_counter() - collection_start) * 1000.0)

    collect.last_time = None  # type: ignore[attr-defined]

    collect("baseline", None)
    creationflags = (
        getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        if sys.platform == "win32"
        else 0
    )
    child = subprocess.Popen(args.command, creationflags=creationflags)
    try:
        while True:
            collect("running", child.pid)
            if child.poll() is not None:
                break
            time.sleep(max(0.1, args.interval_seconds))
        collect("final", child.pid)
    except KeyboardInterrupt:
        if sys.platform == "win32" and hasattr(signal, "CTRL_BREAK_EVENT"):
            try:
                child.send_signal(signal.CTRL_BREAK_EVENT)
            except OSError:
                pass
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
    finally:
        if stream is not None:
            try:
                stream.close()
            except OSError:
                errors += 1
    exit_code = int(child.returncode)
    ended_at = utc_now()
    duration = time.monotonic() - start
    summary = build_summary(
        args.phase,
        args.detail_profile,
        started_at,
        ended_at,
        duration,
        exit_code,
        samples,
        capabilities,
        collection_ms,
        errors,
    )
    try:
        temp_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
        temp_summary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temp_summary.replace(summary_path)
    except OSError as error:
        print(
            f"windows_resource_probe: summary unavailable ({type(error).__name__})",
            file=sys.stderr,
        )
    finally:
        provider.close()
    return exit_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True)
    parser.add_argument(
        "--detail-profile",
        choices=("basic", "process", "diagnostic"),
        default="process",
    )
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--storage-path", default="B:\\")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not PHASE_RE.fullmatch(args.phase):
        parser.error("--phase must be a privacy-safe label")
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if sys.platform != "win32":
        print("windows_resource_probe.py requires Windows", file=sys.stderr)
        return 2
    return run_probe(args, WindowsProvider())


if __name__ == "__main__":
    raise SystemExit(main())
