#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Measure Linux subproject commands and collect downloadable build metrics.

CPU is waited-child user/system time. RAM is sampled aggregate process-tree RSS
(shared pages may be counted repeatedly), plus the largest individual child RSS.
Neither CPU nor RAM can be inferred from Ninja timings.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def measure(args: argparse.Namespace) -> int:
    import resource
    import psutil

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("A command after -- is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record_id = uuid.uuid4().hex
    start = time.monotonic()
    started_at = time.time()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    peak_rss = 0
    process = subprocess.Popen(command, start_new_session=True)
    try:
        root = psutil.Process(process.pid)
    except psutil.NoSuchProcess:
        root = None

    def forward_signal(signum: int, frame: object) -> None:
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass

    old_handlers = {
        signum: signal.signal(signum, forward_signal)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        with (args.output_dir / f"{record_id}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(["elapsed_seconds", "process_tree_rss_bytes"])
            while True:
                rss = 0
                try:
                    processes = [root, *root.children(recursive=True)] if root else []
                except psutil.Error:
                    processes = []
                for child in processes:
                    try:
                        rss += child.memory_info().rss
                    except psutil.Error:
                        pass
                peak_rss = max(peak_rss, rss)
                writer.writerow([time.monotonic() - start, rss])
                stream.flush()
                try:
                    exit_code = process.wait(timeout=args.interval)
                    break
                except subprocess.TimeoutExpired:
                    pass
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    elapsed = time.monotonic() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    user = after.ru_utime - before.ru_utime
    system = after.ru_stime - before.ru_stime
    record = {
        "component": args.component,
        "stage": args.stage,
        "started_at_unix": started_at,
        "wall_seconds": elapsed,
        "user_cpu_seconds": user,
        "system_cpu_seconds": system,
        "average_cpu_percent": 100 * (user + system) / elapsed,
        "sampled_peak_process_tree_rss_bytes": peak_rss,
        "max_child_rss_bytes": after.ru_maxrss * 1024,
        "sample_interval_seconds": args.interval,
        "exit_code": exit_code,
        "command": command,
        "samples_file": f"{record_id}.csv",
    }
    (args.output_dir / f"{record_id}.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return exit_code if exit_code >= 0 else 128 - exit_code


def ninja_stages(log: Path) -> list[dict]:
    """Keep command timings, deduplicating absolute/relative stamp aliases.

    Logs may contain incremental invocations; rows are not summed across builds.
    Stamp timings include copy/touch overhead absent from command measurements.
    """
    rows = []
    seen = set()
    if not log.exists():
        return rows
    for line in log.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) != 5:
            continue
        start, end, mtime, output, command_hash = fields
        path = Path(output)
        if path.parent.name != "stamp" or path.suffix != ".stamp":
            continue
        try:
            int(start), int(end)
        except ValueError:
            continue  # An interrupted writer may leave a partial final record.
        component = path.parent.parent
        if component.is_absolute():
            component = component.relative_to(log.parent)
        key = (start, end, mtime, command_hash, str(component), path.stem)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "component_path": str(component),
                "stage": path.stem,
                "start_ms": int(start),
                "end_ms": int(end),
                "wall_seconds": (int(end) - int(start)) / 1000,
            }
        )
    return rows


def binary_sizes(build_dir: Path) -> list[dict]:
    """Inventory staged ELF/archive file bytes, excluding symlinks and merged dist."""
    rows = []
    for stage in sorted(build_dir.rglob("stage")):
        if not stage.is_dir() or stage.is_symlink():
            continue
        if not (stage.parent / "stamp").is_dir():
            continue
        component = str(stage.parent.relative_to(build_dir))
        for path in sorted(stage.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            with path.open("rb") as stream:
                magic = stream.read(8)
            if magic.startswith(b"\x7fELF") or magic == b"!<arch>\n":
                rows.append(
                    {
                        "component_path": component,
                        "path": str(path.relative_to(stage)),
                        "bytes": path.stat().st_size,
                    }
                )
    return rows


def report(build_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    source = build_dir / "logs" / "metrics"
    records = []
    if source.exists():
        shutil.copytree(source, output_dir / "raw", dirs_exist_ok=True)
        records = [json.loads(p.read_text()) for p in sorted(source.glob("*.json"))]
    fields = [
        "component",
        "stage",
        "started_at_unix",
        "wall_seconds",
        "user_cpu_seconds",
        "system_cpu_seconds",
        "average_cpu_percent",
        "sampled_peak_process_tree_rss_bytes",
        "max_child_rss_bytes",
        "exit_code",
    ]
    write_csv(output_dir / "stages.csv", records, fields)
    (output_dir / "stages.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(
        output_dir / "ninja_stages.csv",
        ninja_stages(build_dir / ".ninja_log"),
        ["component_path", "stage", "start_ms", "end_ms", "wall_seconds"],
    )
    for log in sorted(build_dir.rglob(".ninja_log")):
        target = output_dir / "ninja" / log.relative_to(build_dir).parent / "ninja.log"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(log, target)
    sizes = binary_sizes(build_dir)
    write_csv(
        output_dir / "binary_sizes.csv", sizes, ["component_path", "path", "bytes"]
    )
    totals = {}
    for row in sizes:
        component = row["component_path"]
        totals[component] = totals.get(component, 0) + row["bytes"]
    write_csv(
        output_dir / "binary_totals.csv",
        [
            {"component_path": key, "bytes": value}
            for key, value in sorted(totals.items())
        ],
        ["component_path", "bytes"],
    )
    (output_dir / "binary_sizes.json").write_text(
        json.dumps(sizes, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    run_parser = subparsers.add_parser("run", help="Measure one subproject command")
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--component", required=True)
    run_parser.add_argument("--stage", required=True)
    run_parser.add_argument("--interval", type=float, default=0.2)
    run_parser.add_argument("command", nargs=argparse.REMAINDER)
    report_parser = subparsers.add_parser("report", help="Collect downloadable metrics")
    report_parser.add_argument("--build-dir", type=Path, required=True)
    report_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "run":
        if args.interval <= 0:
            parser.error("--interval must be positive")
        if sys.platform != "linux":
            parser.error("Command measurement requires Linux")
        return measure(args)
    report(args.build_dir.resolve(), args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
