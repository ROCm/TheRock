#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Read-only ELF probe benchmark: --root output/artifacts --output diagnostics --blocks 3.

Repeated ABBA, fresh process/pass. A=Path.read_bytes()[:4]; B=open().read(4).
Never calls patchelf/readelf, drops caches, or modifies the artifact tree.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import psutil


def digest(paths):
    h = hashlib.sha256()
    for name in sorted(paths):
        b = name.encode("utf-8", errors="surrogateescape")
        h.update(len(b).to_bytes(8, "big"))
        h.update(b)
    return h.hexdigest()


def peak_rss():
    if sys.platform == "win32":
        return psutil.Process().memory_info().peak_wset
    import resource

    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return maximum if sys.platform == "darwin" else maximum * 1024


def scan(root, method):
    process = psutil.Process()
    io0 = process.io_counters()._asdict()
    cpu0, wall0 = time.process_time(), time.perf_counter()
    count = returned = requested = sized = symlinks = 0
    elves, errors, manifest = [], [], []
    for directory, dirs, files in os.walk(root, topdown=True, followlinks=True):
        dirs.sort()
        for name in sorted(files):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                symlinks += 1
                continue
            try:
                before = path.stat()
                if method == "A":
                    data = path.read_bytes()
                    magic, actual = data[:4], len(data)
                    requested += before.st_size
                    del data
                else:
                    with path.open("rb") as stream:
                        magic = stream.read(4)
                    actual = len(magic)
                    requested += 4
                count += 1
                returned += actual
                sized += before.st_size
                after = path.stat()
                manifest.append(f"{relative}\0{before.st_size}\0{before.st_mtime_ns}")
                if (before.st_size, before.st_mtime_ns) != (
                    after.st_size,
                    after.st_mtime_ns,
                ):
                    errors.append(relative + ":changed_during_read")
                if magic == b"\x7fELF":
                    elves.append(relative)
            except OSError as error:
                errors.append(relative + ":" + type(error).__name__)
    wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
    io1 = process.io_counters()._asdict()
    return dict(
        method=method,
        wall_s=wall,
        cpu_s=cpu,
        peak_rss_bytes=peak_rss(),
        files_read=count,
        symlink_files_skipped=symlinks,
        file_sizes_sum_bytes=sized,
        application_requested_bytes=requested,
        application_returned_bytes=returned,
        elf_count=len(elves),
        elf_paths_sha256=digest(elves),
        input_manifest_sha256=digest(manifest),
        errors=errors,
        errors_sha256=digest(errors),
        process_io_delta={k: io1[k] - v for k, v in io0.items()},
        io_note="Application bytes differ from buffered syscall bytes and physical disk bytes; OS counters separate.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--worker", choices=["A", "B"])
    args = p.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        p.error("--root must be an existing directory")
    if args.worker:
        print(json.dumps(scan(root, args.worker)))
        return
    if args.output is None or args.blocks < 1:
        p.error("provide --output and positive --blocks")
    output = args.output.resolve()
    if output == root or root in output.parents:
        p.error("--output must be outside the read-only input tree")
    output.mkdir(parents=True, exist_ok=True)
    meta = dict(
        root=str(root),
        ci={
            key: os.environ.get(key)
            for key in (
                "GITHUB_SHA",
                "GITHUB_REPOSITORY",
                "GITHUB_RUN_ID",
                "GITHUB_RUN_ATTEMPT",
                "RUNNER_NAME",
                "RUNNER_OS",
            )
        },
        platform=platform.platform(),
        python=sys.version,
        psutil=psutil.__version__,
        blocks=args.blocks,
        order="ABBA repeated; fresh Python process each pass",
        cache_state="Never cleared. First pass current post-fetch state; subsequent passes naturally warmed.",
        scope="Magic classification only; no readelf, patchelf, config edit, or package creation.",
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    (output / "metadata.json").write_text(json.dumps(meta, indent=2))
    results = []
    for block in range(args.blocks):
        for position, method in enumerate("ABBA"):
            started = datetime.datetime.now(datetime.timezone.utc).isoformat()
            run = subprocess.run(
                [sys.executable, __file__, "--root", str(root), "--worker", method],
                capture_output=True,
                text=True,
            )
            if run.returncode:
                (output / "worker-failure.json").write_text(
                    json.dumps(
                        dict(
                            block=block,
                            position=position,
                            method=method,
                            returncode=run.returncode,
                            stdout=run.stdout,
                            stderr=run.stderr,
                        ),
                        indent=2,
                    )
                )
                raise SystemExit(
                    f"{method} worker failed: {run.returncode}; see worker-failure.json"
                )
            value = json.loads(run.stdout)
            value.update(block=block, position=position, started_utc=started)
            results.append(value)
            (output / "passes.json").write_text(json.dumps(results, indent=2))
            print(json.dumps(value), flush=True)
    invariants = [
        "elf_paths_sha256",
        "elf_count",
        "input_manifest_sha256",
        "errors_sha256",
        "files_read",
        "symlink_files_skipped",
        "file_sizes_sum_bytes",
    ]
    matches = all(all(r[k] == results[0][k] for k in invariants) for r in results)
    summary = dict(
        classification_and_manifest_match=matches,
        complete_input_scan=all(
            not row["errors"] and row["files_read"] > 0 for row in results
        ),
        pass_count=len(results),
        first_pass_reported_separately=results[0],
        input_invariants=invariants,
        scope="Read-only scan; savings are not full RPATH conversion or packaging savings.",
    )
    for method in "AB":
        rows = [r for r in results if r["method"] == method]
        summary[method] = {
            metric: dict(
                median=statistics.median(r[metric] for r in rows),
                minimum=min(r[metric] for r in rows),
                maximum=max(r[metric] for r in rows),
            )
            for metric in [
                "wall_s",
                "cpu_s",
                "peak_rss_bytes",
                "application_returned_bytes",
            ]
        }
    summary["scan_only_median_wall_reduction_s"] = (
        summary["A"]["wall_s"]["median"] - summary["B"]["wall_s"]["median"]
    )
    summary["scan_only_median_speed_ratio_A_over_B"] = (
        summary["A"]["wall_s"]["median"] / summary["B"]["wall_s"]["median"]
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    if not matches:
        raise SystemExit(
            "Input/classification mismatch; comparisons invalid; inspect passes.json"
        )
    if not summary["complete_input_scan"]:
        raise SystemExit(
            "Empty or incomplete input scan; comparisons invalid; inspect passes.json"
        )


if __name__ == "__main__":
    main()
