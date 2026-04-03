#!/usr/bin/env python3
"""Resource monitor for CI builds.

Monitors memory, GPU, storage, and CPU during command execution.

Usage:
    python build_tools/memory_monitor.py -- cmake --build build

Output:
    [09:00:03Z] Mem: 24.5/32.0GB (77%) [WARNING] | CPU: 85% | Jobs: ~14/16 | Disk: 150GB free
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
from typing import Optional

import psutil

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

        # GPU memory
        if self.monitor_gpu:
            stats["gpus"] = get_gpu_memory()

        # Storage
        if self.monitor_storage:
            stats["storage"] = get_storage_info(self.storage_path)

        return stats

    def _format_stats(self, stats: dict) -> str:
        """Format stats as a single line."""
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

        # CPU
        if "cpu_percent" in stats and stats["cpu_percent"] > 0:
            parts.append(f"CPU: {stats['cpu_percent']:.0f}%")
        if "load_1m" in stats:
            cpu_count = stats.get("cpu_count", 1)
            parts.append(f"Jobs: ~{stats['load_1m']:.0f}/{cpu_count}")

        # GPU
        for gpu in stats.get("gpus", []):
            parts.append(
                f"GPU{gpu['id'][-1]}: {gpu['used_gb']:.1f}/{gpu['total_gb']:.1f}GB"
            )

        # Storage
        storage = stats.get("storage", {})
        if storage:
            parts.append(f"Disk: {storage['free_gb']:.0f}GB free")

        return " | ".join(parts)

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
        max_load = max((s.get("load_1m", 0) for s in samples), default=0)
        avg_cpu = sum(s.get("cpu_percent", 0) for s in samples) / len(samples)

        print("\n" + "=" * 70)
        print(f"Resource Summary - {self.phase}")
        print("=" * 70)
        print(f"Duration:     {duration / 60:.1f} min ({len(samples)} samples)")
        print(
            f"Memory:       {max_mem:.0f}% peak ({max_mem_gb:.1f} GB), {avg_mem:.0f}% avg"
        )
        if max_swap > 1:
            print(f"Swap:         {max_swap:.0f}% peak")
        print(f"CPU:          {avg_cpu:.0f}% avg")

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
