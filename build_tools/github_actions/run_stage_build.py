#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Run a single TheRock build stage under Ninja while collecting resource
and timing telemetry.

This script wraps the stage build with `/usr/bin/time -v`, enables Ninja's
`--profile` output, parses the resulting trace, and emits a JSON summary.
Metrics are written to build/logs/ by default so that post_stage_upload.py
includes them in the S3 log index automatically.

Example (from GitHub Actions):

    python build_tools/github_actions/run_stage_build.py \
        --stage math-libs \
        --build-dir build \
        --amdgpu-family gfx94X-dcgpu
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, help="Stage name (e.g. compiler-runtime, math-libs)")
    parser.add_argument("--build-dir", default="build", help="CMake/Ninja build directory")
    parser.add_argument(
        "--metrics-dir",
        default=None,
        help="Directory to store metrics (defaults to <build-dir>/logs)",
    )
    parser.add_argument(
        "--variant",
        default=os.environ.get("THEROCK_BUILD_VARIANT", ""),
        help="Build variant label (asan, host-asan, etc.)",
    )
    parser.add_argument(
        "--amdgpu-family",
        default="",
        help="amdgpu family if this is a per-arch stage (e.g. gfx94X-dcgpu)",
    )
    parser.add_argument(
        "--profile-name",
        default=None,
        help="Optional explicit name suffix for profile files (defaults to timestamp)",
    )
    parser.add_argument(
        "targets",
        nargs="*",
        help="Override Ninja targets (default: stage-<stage> therock-artifacts)",
    )
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _run(cmd: Sequence[str], *, env: Mapping[str, str] | None = None, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=False,
        env=dict(env) if env is not None else None,
        text=True,
        capture_output=capture_output,
    )


def _parse_time_report(time_path: Path) -> Dict[str, Any]:
    """
    Parse `/usr/bin/time -v` output into a dict.
    """
    metrics: Dict[str, Any] = {}
    if not time_path.exists():
        return metrics

    pattern = re.compile(r"^([^:]+):\s*(.*)$")
    with time_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            match = pattern.match(line.strip())
            if not match:
                continue
            key, value = match.groups()
            metrics[key] = value

    # Convenience parsing for a few common fields
    def _maybe_float(key: str) -> float | None:
        raw = metrics.get(key)
        if raw is None:
            return None
        try:
            return float(raw.split()[0])
        except ValueError:
            return None

    def _maybe_int(key: str) -> int | None:
        raw = metrics.get(key)
        if raw is None:
            return None
        try:
            return int(raw.split()[0])
        except ValueError:
            return None

    parsed: Dict[str, Any] = {
        "raw": metrics,
        "user_seconds": _maybe_float("User time (seconds)"),
        "system_seconds": _maybe_float("System time (seconds)"),
        "percent_cpu": _maybe_float("Percent of CPU this job got"),
        "elapsed_time_hms": metrics.get("Elapsed (wall clock) time (h:mm:ss or m:ss)", None),
        "max_rss_kb": _maybe_int("Maximum resident set size (kbytes)"),
        "average_rss_kb": _maybe_int("Average resident set size (kbytes)"),
        "major_page_faults": _maybe_int("Major (requiring I/O) page faults"),
        "minor_page_faults": _maybe_int("Minor (reclaiming a frame) page faults"),
        "voluntary_context_switches": _maybe_int("Voluntary context switches"),
        "involuntary_context_switches": _maybe_int("Involuntary context switches"),
        "filesystem_inputs": _maybe_int("File system inputs"),
        "filesystem_outputs": _maybe_int("File system outputs"),
        "exit_status": _maybe_int("Exit status"),
    }
    if parsed["max_rss_kb"] is not None:
        parsed["max_rss_gb"] = parsed["max_rss_kb"] / (1024 ** 2)
    return parsed


def _capture_command_output(cmd: Sequence[str], output_path: Path) -> Dict[str, Any] | None:
    """
    Execute a diagnostic command (when available) and write stdout/stderr to disk.
    Returns structured metadata or None if the binary was missing.
    """
    binary = cmd[0]
    if shutil.which(binary) is None:
        return None

    result = _run(cmd, capture_output=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                f"Command: {' '.join(shlex.quote(token) for token in cmd)}",
                f"Return code: {result.returncode}",
                "",
                "STDOUT:",
                result.stdout or "",
                "",
                "STDERR:",
                result.stderr or "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "command": list(cmd),
        "return_code": result.returncode,
        "output_path": str(output_path),
    }


def _capture_ccache_stats(path: Path) -> Dict[str, Any] | None:
    """
    Attempt to gather ccache statistics, trying richer commands before falling back.
    """
    candidates: Sequence[Sequence[str]] = (
        ["ccache", "--show-stats"],
        ["ccache", "--print-stats"],
        ["ccache", "-s"],
    )
    for cmd in candidates:
        meta = _capture_command_output(cmd, path)
        if meta and meta["return_code"] == 0:
            try:
                contents = path.read_text()
            except OSError:
                contents = ""
            if contents.strip():
                meta["command"] = list(cmd)
                return meta
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass
    return None


def _capture_diskstats(path: Path) -> Dict[str, Any] | None:
    """
    Capture /proc/diskstats as a fallback when iostat is unavailable.
    """
    try:
        data = Path("/proc/diskstats").read_text()
    except OSError:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")
    return {
        "command": ["cat", "/proc/diskstats"],
        "return_code": 0,
        "output_path": str(path),
    }


def _du_kilobytes(path: Path) -> int | None:
    if not path.exists():
        return None
    result = _run(["du", "-sk", str(path)], capture_output=True)
    if result.returncode != 0 or not result.stdout:
        return None
    try:
        return int(result.stdout.split()[0])
    except (IndexError, ValueError):
        return None


def _capture_command_output(cmd: Sequence[str], output_path: Path) -> Dict[str, Any] | None:
    """
    Execute a diagnostic command (when available) and write stdout/stderr to disk.
    Returns structured metadata or None if the binary was missing.
    """
    binary = cmd[0]
    if shutil.which(binary) is None:
        return None

    result = _run(cmd, capture_output=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                f"Command: {' '.join(shlex.quote(token) for token in cmd)}",
                f"Return code: {result.returncode}",
                "",
                "STDOUT:",
                result.stdout or "",
                "",
                "STDERR:",
                result.stderr or "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "command": cmd,
        "return_code": result.returncode,
        "output_path": str(output_path),
    }


def _du_kilobytes(path: Path) -> int | None:
    if not path.exists():
        return None
    result = _run(["du", "-sk", str(path)], capture_output=True)
    if result.returncode != 0 or not result.stdout:
        return None
    try:
        return int(result.stdout.split()[0])
    except (IndexError, ValueError):
        return None


def _load_profile_json(build_dir: Path, profile_path: Path, output_json_path: Path, output_text_path: Path) -> Tuple[Dict[str, Any] | None, str | None]:
    """
    Convert the Ninja profile data to JSON (if supported) and capture textual fallback.
    Returns (json_data, profile_format).
    """
    if not profile_path.exists():
        return None, None

    # Try JSON first (requires ninja >= 1.11)
    json_cmd = ["ninja", "-C", str(build_dir), "-t", "profile", "--json", str(profile_path)]
    json_result = _run(json_cmd, capture_output=True)
    if json_result.returncode == 0 and json_result.stdout:
        try:
            data = json.loads(json_result.stdout)
        except json.JSONDecodeError:
            data = None
        else:
            output_json_path.write_text(json_result.stdout, encoding="utf-8")
            return data, "chrome-trace-json"

    # Fallback to textual output
    text_cmd = ["ninja", "-C", str(build_dir), "-t", "profile", str(profile_path)]
    text_result = _run(text_cmd, capture_output=True)
    output_text_path.write_text(
        f"Command: {' '.join(shlex.quote(c) for c in text_cmd)}\n"
        f"Return code: {text_result.returncode}\n\n"
        f"STDOUT:\n{text_result.stdout}\n\nSTDERR:\n{text_result.stderr}",
        encoding="utf-8",
    )
    return None, "text"


def _bucket_from_output(output_path: str, command: str) -> str:
    """
    Collapse a file output path into a coarse bucket (subproject/component).
    """
    candidate = output_path or ""
    candidate = candidate.replace("\\", "/")
    parts = [part for part in candidate.split("/") if part]
    if not parts:
        # Fall back to command heuristic
        if command:
            cmd_parts = [part for part in command.split() if "/" in part]
            if cmd_parts:
                parts = [p for p in cmd_parts[0].split("/") if p]
    if not parts:
        return "unknown"

    # Recognize common repo layouts for better grouping
    if parts[0] in {"rocm-libraries", "rocm-systems", "math-libs", "ml-libs"} and len(parts) >= 3:
        return "/".join(parts[:3])
    if parts[0] == "projects" and len(parts) >= 2:
        return "/".join(parts[:2])
    return "/".join(parts[:2]) if len(parts) >= 2 else parts[0]


def _seconds_from_duration(duration_value: Any) -> float | None:
    if duration_value is None:
        return None
    try:
        # Ninja chrome trace durations are microseconds.
        return float(duration_value) / 1_000_000.0
    except (TypeError, ValueError):
        return None


def _analyze_profile_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce aggregated metrics from Ninja's profile JSON (chrome trace format).
    """
    trace_events: Iterable[Mapping[str, Any]] = data.get("traceEvents", [])  # type: ignore[assignment]
    buckets: Dict[str, float] = {}
    command_entries: List[Dict[str, Any]] = []

    for event in trace_events:
        if event.get("ph") != "X":
            continue
        duration_seconds = _seconds_from_duration(event.get("dur"))
        if duration_seconds is None or duration_seconds <= 0.0:
            continue
        args = event.get("args") or {}
        output_path = args.get("output", "") or ""
        command = args.get("command", "") or event.get("name", "")
        bucket = _bucket_from_output(output_path, command)
        buckets[bucket] = buckets.get(bucket, 0.0) + duration_seconds
        command_entries.append(
            {
                "output": output_path,
                "command": command,
                "duration_seconds": duration_seconds,
            }
        )

    total_profiled_seconds = sum(buckets.values())
    top_buckets = sorted(
        ({"bucket": name, "duration_seconds": value, "percent": (value / total_profiled_seconds * 100.0) if total_profiled_seconds else None} for name, value in buckets.items()),
        key=lambda item: item["duration_seconds"],
        reverse=True,
    )
    top_commands = sorted(command_entries, key=lambda item: item["duration_seconds"], reverse=True)[:50]

    return {
        "total_profiled_seconds": total_profiled_seconds,
        "bucket_count": len(buckets),
        "top_buckets": top_buckets[:50],
        "top_commands": top_commands,
        "trace_event_count": len(command_entries),
    }


def _gather_env_snapshot() -> Dict[str, Any]:
    keys = [
        "CMAKE_BUILD_PARALLEL_LEVEL",
        "MAX_JOBS",
        "NINJAFLAGS",
        "NINJA_STATUS",
        "GITHUB_RUN_ID",
        "GITHUB_RUN_ATTEMPT",
        "GITHUB_JOB",
        "RUNNER_NAME",
        "RUNNER_TRACKING_ID",
    ]
    snapshot = {key: os.environ.get(key) for key in keys}
    snapshot["nproc"] = os.cpu_count()
    return snapshot


def main() -> int:
    args = _parse_args()
    build_dir = Path(args.build_dir).resolve()
    if not build_dir.exists():
        print(f"[run_stage_build] Build directory '{build_dir}' does not exist", file=sys.stderr)
        return 2

    metrics_root = (
        Path(args.metrics_dir).resolve()
        if args.metrics_dir
        else build_dir / "logs"
    )
    _ensure_dir(metrics_root)

    stamp = args.profile_name or _timestamp()
    profile_path = metrics_root / f"{args.stage}-ninja-profile-{stamp}.prof"
    time_path = metrics_root / f"{args.stage}-usrbin-time-{stamp}.txt"
    profile_json_path = metrics_root / f"{args.stage}-ninja-profile-{stamp}.json"
    profile_text_path = metrics_root / f"{args.stage}-ninja-profile-{stamp}.txt"
    summary_path = metrics_root / f"{args.stage}-summary-{stamp}.json"
    timing_log_path = metrics_root / f"{args.stage}-rule-timings-{stamp}.jsonl"
    ccache_pre_path = metrics_root / f"{args.stage}-ccache-before-{stamp}.txt"
    ccache_post_path = metrics_root / f"{args.stage}-ccache-after-{stamp}.txt"
    vmstat_path = metrics_root / f"{args.stage}-vmstat-{stamp}.txt"
    iostat_path = metrics_root / f"{args.stage}-iostat-{stamp}.txt"
    diskstats_path = metrics_root / f"{args.stage}-diskstats-{stamp}.txt"
    disk_usage_path = metrics_root / f"{args.stage}-disk-usage-{stamp}.json"

    targets = args.targets or [f"stage-{args.stage}", "therock-artifacts"]
    ninja_cmd = ["ninja", "-C", str(build_dir), "-k", "0"]

    parallel_env = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL")
    parallel_level: int | None = None
    if parallel_env:
        try:
            parsed = int(parallel_env)
        except ValueError:
            parsed = None
        if parsed and parsed > 0:
            parallel_level = parsed
            ninja_cmd.extend(["-j", str(parsed)])

    ninja_cmd.extend(targets)

    time_binary = shutil.which("time")
    if time_binary is None and Path("/usr/bin/time").exists():
        time_binary = "/usr/bin/time"

    # Wire in per-command timing instrumentation unless the workflow already provided overrides.
    timing_script = Path(__file__).resolve().with_name("record_rule_launch.py")
    if timing_script.exists():
        os.environ.setdefault("RUN_STAGE_TIMING_LOG", str(timing_log_path))
        os.environ.setdefault("RUN_STAGE_VARIANT", args.variant or "")
        if not os.environ.get("CMAKE_RULE_LAUNCH_COMPILE"):
            os.environ["CMAKE_RULE_LAUNCH_COMPILE"] = f"{sys.executable} {timing_script} compile"
        if not os.environ.get("CMAKE_RULE_LAUNCH_LINK"):
            os.environ["CMAKE_RULE_LAUNCH_LINK"] = f"{sys.executable} {timing_script} link"

    trace_supported: bool = False
    ninja_help_result = _run(["ninja", "--help"], capture_output=True)
    try:
        ninja_help_text = ninja_help_result.stdout or ""
    except AttributeError:
        ninja_help_text = ""

    if ninja_help_result.returncode == 0 and "--profile=" in ninja_help_text:
        trace_supported = True

    if trace_supported and profile_path:
        ninja_cmd.insert(3, f"--profile={profile_path}")

    if time_binary:
        time_cmd = [time_binary, "-v", "-o", str(time_path), *ninja_cmd]
    else:
        print("[run_stage_build] WARNING: 'time' binary not found; skipping detailed resource metrics")
        time_cmd = ninja_cmd

    print(f"[run_stage_build] Executing: {' '.join(shlex.quote(tok) for tok in time_cmd)}")

    tooling_notes: List[str] = []
    disk_usage_before_kb = _du_kilobytes(build_dir)
    if disk_usage_before_kb is None:
        tooling_notes.append("du_missing")

    ccache_pre_meta = _capture_ccache_stats(ccache_pre_path)
    if ccache_pre_meta is None:
        tooling_notes.append("ccache_stats_unavailable")

    vmstat_meta = _capture_command_output(["vmstat", "-s"], vmstat_path)
    if vmstat_meta is None:
        tooling_notes.append("vmstat_missing")

    iostat_meta = _capture_command_output(["iostat", "-dx"], iostat_path)
    diskstats_meta = None
    if iostat_meta is None:
        tooling_notes.append("iostat_missing")
        diskstats_meta = _capture_diskstats(diskstats_path)
        if diskstats_meta is None:
            tooling_notes.append("diskstats_missing")

    start = datetime.now(tz=timezone.utc)
    result = _run(time_cmd)
    end = datetime.now(tz=timezone.utc)

    if result.returncode != 0:
        print(f"[run_stage_build] Build exited with code {result.returncode}", file=sys.stderr)

    # Collect profile data.
    profile_json: Dict[str, Any] | None = None
    profile_format: str | None = None
    if profile_path.exists():
        profile_json, profile_format = _load_profile_json(build_dir, profile_path, profile_json_path, profile_text_path)

    profile_summary: Dict[str, Any] | None = None
    if profile_json:
        profile_summary = _analyze_profile_json(profile_json)

    time_summary = _parse_time_report(time_path) if time_binary else {}
    env_snapshot = _gather_env_snapshot()

    summary: Dict[str, Any] = {
        "stage": args.stage,
        "amdgpu_family": args.amdgpu_family or None,
        "variant": args.variant or None,
        "build_dir": str(build_dir),
        "metrics_dir": str(metrics_root),
        "timestamp": stamp,
        "started_at_utc": start.isoformat(),
        "finished_at_utc": end.isoformat(),
        "ninja_targets": targets,
        "ninja_parallel_level": parallel_level,
        "ninja_exit_code": result.returncode,
        "profile_path": str(profile_path) if profile_path.exists() else None,
        "profile_format": profile_format,
        "profile_json_path": str(profile_json_path) if profile_json_path.exists() else None,
        "profile_text_path": str(profile_text_path) if profile_text_path.exists() else None,
        "profile_summary": profile_summary,
        "time_report_path": str(time_path) if time_path.exists() else None,
        "time_summary": time_summary,
        "environment": env_snapshot,
        "resource_metrics_available": bool(time_binary),
        "timing_log_path": str(timing_log_path) if timing_log_path.exists() else None,
        "ccache_pre_path": str(ccache_pre_path) if ccache_pre_path.exists() else None,
        "ccache_post_path": None,
        "vmstat_path": str(vmstat_path) if vmstat_path.exists() else None,
        "iostat_path": str(iostat_path) if iostat_path.exists() else None,
        "diskstats_path": str(diskstats_path) if diskstats_path.exists() else None,
        "disk_usage_before_kb": disk_usage_before_kb,
        "disk_usage_after_kb": None,
        "disk_usage_delta_kb": None,
        "ccache_pre_metadata": ccache_pre_meta,
        "vmstat_metadata": vmstat_meta,
        "iostat_metadata": iostat_meta,
        "diskstats_metadata": diskstats_meta,
        "tooling_notes": tooling_notes or None,
    }

    disk_usage_after_kb = _du_kilobytes(build_dir)
    if disk_usage_after_kb is not None and disk_usage_before_kb is not None:
        summary["disk_usage_after_kb"] = disk_usage_after_kb
        summary["disk_usage_delta_kb"] = disk_usage_after_kb - disk_usage_before_kb
        disk_usage_path.write_text(
            json.dumps(
                {
                    "before_kb": disk_usage_before_kb,
                    "after_kb": disk_usage_after_kb,
                    "delta_kb": disk_usage_after_kb - disk_usage_before_kb,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        summary["disk_usage_path"] = str(disk_usage_path)

    ccache_post_meta = _capture_ccache_stats(ccache_post_path)
    if ccache_post_meta:
        summary["ccache_post_path"] = str(ccache_post_path)
        summary["ccache_post_metadata"] = ccache_post_meta

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[run_stage_build] Summary written to {summary_path}")
    if profile_summary:
        top_bucket = profile_summary["top_buckets"][0] if profile_summary["top_buckets"] else None
        if top_bucket:
            print(
                f"[run_stage_build] Top bucket: {top_bucket['bucket']} "
                f"({top_bucket['duration_seconds']:.1f}s, {top_bucket.get('percent', 0.0):.1f}%)"
            )
    if time_summary.get("max_rss_gb") is not None:
        print(f"[run_stage_build] Max RSS: {time_summary['max_rss_gb']:.2f} GiB")
    if disk_usage_after_kb is not None and disk_usage_before_kb is not None:
        delta_gib = (disk_usage_after_kb - disk_usage_before_kb) * 1024 / (1024 ** 3)
        print(f"[run_stage_build] Disk delta: {delta_gib:+.2f} GiB ({disk_usage_before_kb/1024/1024:.2f} → {disk_usage_after_kb/1024/1024:.2f} GiB)")
    if summary.get("timing_log_path"):
        print(f"[run_stage_build] Timing log captured: {summary['timing_log_path']}")
    else:
        print("[run_stage_build] Timing log missing (check CMAKE_RULE_LAUNCH_* wiring)")
    if tooling_notes:
        print(f"[run_stage_build] Tooling notes: {', '.join(tooling_notes)}")

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
