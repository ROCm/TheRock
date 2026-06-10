#!/usr/bin/env python3
"""Resource monitor for CI builds.

Monitors memory, GPU, storage, and CPU during command execution.

Usage:
    python build_tools/memory_monitor.py -- cmake --build build

Output:
    [09:00:03Z] Mem: 24.5/32.0GB (77%) [WARNING] | CPU: 85% | Load: 14/16 | Disk: 150GB free
               Top: clang:foo.cpp(12.3%m,95%c), ninja:rocblas(8.1%m,50%c), link:libhip.so(5.2%m,0%c)

When memory or CPU exceeds 75%, top processes are shown with memory and CPU percentages.
Load shows system load average vs CPU count; when overloaded shows multiplier (e.g., "2.5x overload").
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import psutil


def is_in_container() -> bool:
    """Detect if we're running inside a container."""
    # Check for Docker
    if Path("/.dockerenv").exists():
        return True
    # Check for container cgroup
    try:
        cgroup = Path("/proc/1/cgroup").read_text()
        if "docker" in cgroup or "kubepods" in cgroup or "containerd" in cgroup:
            return True
    except (OSError, IOError):
        pass
    return False


# Constants
GB = 1024**3
DEFAULT_INTERVAL = 30.0
WARN_PERCENT = 75
CRIT_PERCENT = 90


def get_gpu_memory() -> list[dict]:
    """Get AMD GPU memory using rocm-smi."""
    gpus = []
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            for card_id, card_data in data.items():
                if card_id.startswith("card"):
                    used = int(card_data.get("VRAM Total Used Memory (B)", 0))
                    total = int(card_data.get("VRAM Total Memory (B)", 0))
                    if total > 0:
                        gpus.append(
                            {
                                "id": card_id,
                                "used_gb": used / GB,
                                "total_gb": total / GB,
                                "percent": (used / total) * 100,
                            }
                        )
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass
    return gpus


def get_storage_info(path: str = ".") -> dict:
    """Get storage usage for the given path."""
    try:
        usage = shutil.disk_usage(path)
        return {
            "used_gb": usage.used / GB,
            "total_gb": usage.total / GB,
            "free_gb": usage.free / GB,
            "percent": (usage.used / usage.total) * 100,
        }
    except OSError:
        return {}


def get_thread_info() -> dict:
    """Get thread/CPU information."""
    info = {
        "cpu_count": psutil.cpu_count(),
        "cpu_percent": psutil.cpu_percent(interval=None),
    }
    try:
        # psutil.getloadavg() works on all platforms (emulated on Windows)
        load1, load5, load15 = psutil.getloadavg()
        info["load_1m"] = load1
        info["load_5m"] = load5
    except (OSError, AttributeError):
        pass
    return info


def get_top_processes(top_n: int = 4) -> list[dict]:
    """Get top processes by memory and CPU usage.

    Returns a list of dicts with process info, sorted by memory usage descending.
    """
    processes = []
    for proc in psutil.process_iter(
        ["pid", "name", "memory_percent", "cpu_percent", "cmdline"]
    ):
        try:
            pinfo = proc.info
            # Skip processes with negligible resource usage
            mem_pct = pinfo.get("memory_percent") or 0
            cpu_pct = pinfo.get("cpu_percent") or 0
            if mem_pct < 0.1 and cpu_pct < 1.0:
                continue

            # Extract a useful command name from cmdline
            cmdline = pinfo.get("cmdline") or []
            name = pinfo.get("name") or "unknown"

            # Try to get a meaningful command description
            cmd_desc = _extract_command_description(name, cmdline)

            processes.append(
                {
                    "pid": pinfo["pid"],
                    "name": name,
                    "cmd": cmd_desc,
                    "mem_pct": mem_pct,
                    "cpu_pct": cpu_pct,
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    # Sort by memory usage descending
    processes.sort(key=lambda p: p["mem_pct"], reverse=True)
    return processes[:top_n]


def _extract_command_description(name: str, cmdline: list[str]) -> str:
    """Extract a useful short description from command line arguments.

    Prioritizes showing the actual build target or script being run.
    """
    if not cmdline:
        return name

    # For common build tools, try to extract the target
    cmd_str = " ".join(cmdline)

    # Ninja: look for the target
    if "ninja" in name.lower() or "ninja" in cmd_str.lower():
        for i, arg in enumerate(cmdline):
            if arg == "-C" and i + 1 < len(cmdline):
                # Skip -C path, look for target after
                continue
            if not arg.startswith("-") and i > 0:
                return f"ninja:{arg}"
        return "ninja"

    # CMake: show the build directory or command
    if "cmake" in name.lower():
        for i, arg in enumerate(cmdline):
            if arg == "--build" and i + 1 < len(cmdline):
                return f"cmake:build"
        return "cmake"

    # Clang/compiler: show the source file being compiled
    if any(comp in name.lower() for comp in ["clang", "gcc", "cc", "c++"]):
        for arg in cmdline:
            if arg.endswith((".cpp", ".c", ".cc", ".cxx")):
                # Get just the filename
                return f"{name}:{arg.split('/')[-1]}"
        return name

    # ld/linker: indicate linking
    if name in ("ld", "ld.lld", "lld", "gold"):
        for arg in cmdline:
            if arg == "-o" or arg.startswith("-o"):
                continue
            if not arg.startswith("-") and "/" in arg:
                return f"link:{arg.split('/')[-1]}"
        return "linking"

    # Python scripts
    if "python" in name.lower():
        for arg in cmdline:
            if arg.endswith(".py"):
                return f"py:{arg.split('/')[-1]}"
        return "python"

    # Default: return the process name
    return name


class ResourceMonitor:
    """Monitors system resources in a background thread."""

    def __init__(
        self,
        interval: float = DEFAULT_INTERVAL,
        phase: str = "Build",
        monitor_gpu: bool = True,
        monitor_storage: bool = True,
        storage_path: str = ".",
    ):
        self.interval = interval
        self.phase = phase
        self.monitor_gpu = monitor_gpu
        self.monitor_storage = monitor_storage
        self.storage_path = storage_path
        self.stop_event = threading.Event()
        self.samples: list[dict] = []
        self.start_time: Optional[float] = None
        self.lock = threading.Lock()

    def _collect_stats(self) -> dict:
        """Collect current resource statistics."""
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()

        stats = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mem_used_gb": vm.used / GB,
            "mem_total_gb": vm.total / GB,
            "mem_percent": vm.percent,
            "swap_used_gb": swap.used / GB,
            "swap_percent": swap.percent,
        }

        # CPU/threads
        thread_info = get_thread_info()
        stats.update(thread_info)

        # Top processes
        stats["top_procs"] = get_top_processes(top_n=4)

        # GPU memory
        if self.monitor_gpu:
            stats["gpus"] = get_gpu_memory()

        # Storage
        if self.monitor_storage:
            stats["storage"] = get_storage_info(self.storage_path)

        return stats

    def _format_stats(self, stats: dict) -> str:
        """Format stats as a single line with optional top processes on next line."""
        parts = []

        # Memory
        warn = ""
        if stats["mem_percent"] >= CRIT_PERCENT:
            warn = " [CRITICAL]"
        elif stats["mem_percent"] >= WARN_PERCENT:
            warn = " [WARNING]"
        parts.append(
            f"Mem: {stats['mem_used_gb']:.1f}/{stats['mem_total_gb']:.1f}GB ({stats['mem_percent']:.0f}%){warn}"
        )

        # Swap (only if used)
        if stats["swap_used_gb"] > 0.1:
            parts.append(f"Swap: {stats['swap_used_gb']:.1f}GB")

        # CPU as cores in use (more intuitive than percentage)
        cpu_count = stats.get("cpu_count", 1)
        if "cpu_percent" in stats and stats["cpu_percent"] > 0:
            cores_in_use = (stats["cpu_percent"] / 100) * cpu_count
            parts.append(f"CPU: {cores_in_use:.0f}/{cpu_count} cores")

        # GPU
        for gpu in stats.get("gpus", []):
            parts.append(
                f"GPU{gpu['id'][-1]}: {gpu['used_gb']:.1f}/{gpu['total_gb']:.1f}GB"
            )

        # Storage
        storage = stats.get("storage", {})
        if storage:
            parts.append(f"Disk: {storage['free_gb']:.0f}GB free")

        main_line = " | ".join(parts)

        # Always show top process, more when resources are high
        top_procs = stats.get("top_procs", [])
        in_container = is_in_container()
        if top_procs:
            # Show more processes when memory or CPU is high
            show_count = (
                4
                if (
                    stats["mem_percent"] >= WARN_PERCENT
                    or stats.get("cpu_percent", 0) >= WARN_PERCENT
                )
                else 1
            )
            proc_strs = []
            for p in top_procs[:show_count]:
                # Convert raw cpu_percent (can exceed 100% on multi-core) to cores
                cores_used = p["cpu_pct"] / 100
                proc_strs.append(
                    f"{p['cmd']}({p['mem_pct']:.1f}% mem, {cores_used:.1f} CPUs)"
                )
            # Clarify scope when in container (process list is container-local)
            prefix = "Container top" if in_container else "Top"
            main_line += f"\n           {prefix}: {', '.join(proc_strs)}"

        return main_line

    def _log_stats(self, stats: dict) -> None:
        """Print resource stats to stdout."""
        line = self._format_stats(stats)
        print(f"[{stats['timestamp'][11:19]}Z] {line}", flush=True)

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while not self.stop_event.wait(timeout=self.interval):
            stats = self._collect_stats()
            with self.lock:
                self.samples.append(stats)
            self._log_stats(stats)

    def start(self) -> None:
        """Start background monitoring."""
        self.start_time = time.time()
        self.stop_event.clear()
        # Collect initial sample
        stats = self._collect_stats()
        self.samples.append(stats)
        self._log_stats(stats)
        # Start background thread
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop monitoring and print summary."""
        self.stop_event.set()
        if hasattr(self, "thread"):
            self.thread.join(timeout=2)
        self._print_summary()

    def _print_summary(self) -> None:
        """Print summary statistics."""
        with self.lock:
            samples = list(self.samples)

        if not samples:
            return

        duration = time.time() - self.start_time if self.start_time else 0

        # Memory stats
        max_mem = max(s["mem_percent"] for s in samples)
        avg_mem = sum(s["mem_percent"] for s in samples) / len(samples)
        max_mem_gb = max(s["mem_used_gb"] for s in samples)
        max_swap = max(s["swap_percent"] for s in samples)

        # CPU stats
        cpu_count = samples[0].get("cpu_count", 1) if samples else 1
        avg_cpu = sum(s.get("cpu_percent", 0) for s in samples) / len(samples)

        # Aggregate top processes across all samples
        proc_mem_totals: dict[str, list[float]] = {}
        proc_cpu_totals: dict[str, list[float]] = {}
        for s in samples:
            for p in s.get("top_procs", []):
                cmd = p["cmd"]
                proc_mem_totals.setdefault(cmd, []).append(p["mem_pct"])
                proc_cpu_totals.setdefault(cmd, []).append(p["cpu_pct"])

        print("\n" + "=" * 70)
        print(f"Resource Summary - {self.phase}")
        print("=" * 70)
        print(f"Duration:     {duration / 60:.1f} min ({len(samples)} samples)")
        print(
            f"Memory:       {max_mem:.0f}% peak ({max_mem_gb:.1f} GB), {avg_mem:.0f}% avg"
        )
        if max_swap > 1:
            print(f"Swap:         {max_swap:.0f}% peak")
        avg_cores = (avg_cpu / 100) * cpu_count
        max_cores = (max(s.get("cpu_percent", 0) for s in samples) / 100) * cpu_count
        print(
            f"CPU:          {avg_cores:.0f}/{cpu_count} cores avg, {max_cores:.0f} peak"
        )

        # Top memory consumers (by peak memory usage)
        if proc_mem_totals:
            top_mem_procs = sorted(
                [(cmd, max(mems)) for cmd, mems in proc_mem_totals.items()],
                key=lambda x: x[1],
                reverse=True,
            )[:4]
            if top_mem_procs and top_mem_procs[0][1] > 1.0:
                print(
                    "Top memory:   "
                    + ", ".join(f"{cmd}({pct:.1f}%)" for cmd, pct in top_mem_procs)
                )

        # Top CPU consumers (by average CPU usage, shown as cores)
        if proc_cpu_totals:
            top_cpu_procs = sorted(
                [(cmd, sum(cpus) / len(cpus)) for cmd, cpus in proc_cpu_totals.items()],
                key=lambda x: x[1],
                reverse=True,
            )[:4]
            if top_cpu_procs and top_cpu_procs[0][1] > 10.0:
                print(
                    "Top CPU:      "
                    + ", ".join(
                        f"{cmd}({pct / 100:.1f} cores)" for cmd, pct in top_cpu_procs
                    )
                )

        # GPU summary
        gpu_maxes = {}
        for s in samples:
            for gpu in s.get("gpus", []):
                gid = gpu["id"]
                if gid not in gpu_maxes or gpu["used_gb"] > gpu_maxes[gid]["used_gb"]:
                    gpu_maxes[gid] = gpu
        for gid, gpu in gpu_maxes.items():
            print(
                f"GPU {gid}:       {gpu['used_gb']:.1f}/{gpu['total_gb']:.1f} GB peak ({gpu['percent']:.0f}%)"
            )

        # Storage summary
        storage_samples = [s.get("storage", {}) for s in samples if s.get("storage")]
        if storage_samples:
            min_free = min(s["free_gb"] for s in storage_samples)
            print(f"Storage:      {min_free:.0f} GB min free")

        print("=" * 70)
        print(f"Max memory usage was {max_mem:.0f}%")
        print("=" * 70 + "\n")


def run_with_monitor(
    command: list[str], interval: float, phase: str, storage_path: str
) -> int:
    """Run a command with resource monitoring."""
    monitor = ResourceMonitor(interval=interval, phase=phase, storage_path=storage_path)

    def handle_signal(signum, frame):
        monitor.stop()
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    monitor.start()
    try:
        result = subprocess.run(command)
        return result.returncode
    finally:
        monitor.stop()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Monitor resources during command execution"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=float(os.environ.get("MONITOR_INTERVAL", DEFAULT_INTERVAL)),
    )
    parser.add_argument("--phase", default=os.environ.get("MONITOR_PHASE", "Build"))
    parser.add_argument(
        "--storage-path", default=os.environ.get("MONITOR_STORAGE_PATH", ".")
    )
    parser.add_argument("command", nargs="*")

    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]

    if command:
        return run_with_monitor(command, args.interval, args.phase, args.storage_path)
    else:
        # One-shot mode
        monitor = ResourceMonitor(phase=args.phase, storage_path=args.storage_path)
        stats = monitor._collect_stats()
        monitor._log_stats(stats)
        return 0


if __name__ == "__main__":
    sys.exit(main())
