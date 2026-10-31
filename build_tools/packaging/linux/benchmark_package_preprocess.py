#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Compare actual RPATH preprocessing on pristine-derived copies, ABBA once.

Only conversion is timed. Resetting, full hashing and dynamic-tag inspection are
outside that interval. Requires Linux, readelf/patchelf and generous disk space.
No package generation or publication is performed.
"""

import argparse
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
from typing import Any


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def emit_hash(hasher: Any, record: dict[str, Any]) -> None:
    hasher.update((json.dumps(record, sort_keys=True) + "\n").encode())


def check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise RuntimeError("Diagnostic runtime budget exhausted")


def check_links(root: Path) -> None:
    """Reject escaping directory/config links and cycles before conversion.

    Other file symlinks are preserved: production skips their ELF probe and the
    verifier records the link itself without following it.
    """
    root = root.resolve()
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(directory) / name
            if path.is_symlink() and (path.is_dir() or path.name == "rocm.cfg"):
                target = path.resolve()
                if target != root and root not in target.parents:
                    raise RuntimeError(
                        f"Symlink escapes input copy: {path} -> {target}"
                    )

    def visit(directory: Path, ancestors: set[tuple[int, int]]) -> None:
        info = directory.stat()
        identity = (info.st_dev, info.st_ino)
        if identity in ancestors:
            raise RuntimeError(f"Directory symlink cycle: {directory}")
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=True):
                    visit(Path(entry.path), ancestors | {identity})

    visit(root, set())


def manifest(
    root: Path, output: Path, deadline: float, dynamic: bool = False
) -> dict[str, Any]:
    """Hash physical entries; post-conversion equivalence excludes timestamps."""
    started = time.monotonic()
    content, mtimes = hashlib.sha256(), hashlib.sha256()
    dynamic_hash = hashlib.sha256()
    counts = dict(files=0, directories=0, symlinks=0, elf=0, configs=0, bytes=0)
    readelf_errors = []
    with output.open("w", encoding="utf-8") as stream:
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs.sort()
            files.sort()
            paths = [Path(directory)] + [Path(directory) / name for name in files]
            paths += [
                Path(directory) / name
                for name in dirs
                if (Path(directory) / name).is_symlink()
            ]
            for path in paths:
                check_deadline(deadline)
                info = path.lstat()
                relative = path.relative_to(root).as_posix()
                record = dict(path=relative, mode=stat.S_IMODE(info.st_mode))
                if stat.S_ISLNK(info.st_mode):
                    record.update(kind="symlink", target=os.readlink(path))
                    counts["symlinks"] += 1
                elif stat.S_ISDIR(info.st_mode):
                    record.update(kind="directory")
                    counts["directories"] += 1
                elif stat.S_ISREG(info.st_mode):
                    file_hash = hashlib.sha256()
                    with path.open("rb") as source:
                        first = source.read(1024 * 1024)
                        file_hash.update(first)
                        while chunk := source.read(1024 * 1024):
                            check_deadline(deadline)
                            file_hash.update(chunk)
                    record.update(
                        kind="file", size=info.st_size, sha256=file_hash.hexdigest()
                    )
                    counts["files"] += 1
                    counts["bytes"] += info.st_size
                    emit_hash(mtimes, dict(path=relative, mtime_ns=info.st_mtime_ns))
                    if path.name == "rocm.cfg":
                        counts["configs"] += 1
                        record["compiler_config"] = True
                    if first[:4] == b"\x7fELF":
                        counts["elf"] += 1
                        if dynamic:
                            run = subprocess.run(
                                ["readelf", "-d", str(path)],
                                capture_output=True,
                                timeout=30,
                            )
                            tags = re.findall(
                                rb"\((R(?:UN)?PATH)\).*?\[(.*?)\]", run.stdout
                            )
                            entry = dict(
                                path=relative,
                                returncode=run.returncode,
                                stdout_sha256=digest(run.stdout),
                                stderr_sha256=digest(run.stderr),
                                rpath_tags=[
                                    (a.decode(), b.decode(errors="replace"))
                                    for a, b in tags
                                ],
                            )
                            record["dynamic"] = entry
                            emit_hash(dynamic_hash, entry)
                            if run.returncode:
                                readelf_errors.append(entry)
                else:
                    raise RuntimeError(f"Unsupported special file in artifacts: {path}")
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                # Dynamic tags are checked independently; compare exact file content too.
                emit_hash(content, {k: v for k, v in record.items() if k != "dynamic"})
    return dict(
        content_sha256=content.hexdigest(),
        mtime_sha256=mtimes.hexdigest(),
        dynamic_sha256=dynamic_hash.hexdigest(),
        readelf_errors=readelf_errors,
        counts=counts,
        wall_s=time.monotonic() - started,
    )


def copy_tree(
    pristine: Path,
    work: Path,
    expected_bytes: int,
    deadline: float,
    owned: bool = False,
) -> dict[str, Any]:
    if (
        work.resolve() == pristine.resolve()
        or work.parent.resolve() != pristine.parent.resolve()
    ):
        raise RuntimeError("Work copy must be a distinct sibling of pristine artifacts")
    if work.exists():
        if not owned or work.is_symlink():
            raise RuntimeError(
                "Refusing to remove a work path not created by this process"
            )
        shutil.rmtree(work)
    required = expected_bytes + max(5 * 1024**3, expected_bytes // 10)
    free = shutil.disk_usage(pristine.parent).free
    if free < required:
        raise RuntimeError(
            f"Insufficient free disk: {free} bytes; require {required} for an independent copy and safety margin"
        )
    counters = dict(
        reflink_files=0,
        copied_files=0,
        fsynced_files=0,
        free_before_bytes=free,
        required_bytes=required,
    )
    started = time.monotonic()

    def independent_copy(source: str, destination: str) -> str:
        check_deadline(deadline)
        import fcntl

        try:
            with open(source, "rb") as src, open(destination, "wb") as dst:
                fcntl.ioctl(dst.fileno(), 0x40049409, src.fileno())  # Linux FICLONE
            counters["reflink_files"] += 1
        except OSError as error:
            if error.errno not in (
                errno.EOPNOTSUPP,
                errno.ENOTTY,
                errno.EINVAL,
                errno.EXDEV,
                errno.EPERM,
            ):
                raise
            with open(source, "rb") as src, open(destination, "wb") as dst:
                while chunk := src.read(1024 * 1024):
                    check_deadline(deadline)
                    dst.write(chunk)
            counters["copied_files"] += 1
        shutil.copystat(source, destination, follow_symlinks=False)
        # Keep deferred copy-data writes outside the timed conversion as well.
        with open(destination, "rb") as target:
            os.fsync(target.fileno())
        counters["fsynced_files"] += 1
        if os.stat(destination).st_nlink != 1 or os.path.samefile(source, destination):
            raise RuntimeError(
                "Independent copy verification failed; hard links are prohibited"
            )
        return destination

    shutil.copytree(pristine, work, symlinks=True, copy_function=independent_copy)
    check_links(work)
    counters["wall_s"] = time.monotonic() - started
    return counters


def prepare_sources(output: Path) -> dict[str, Path]:
    original = (
        Path(__file__).with_name("runpath_to_rpath.py").read_text(encoding="utf-8")
    )
    old = '                if filepath.read_bytes()[:4] != b"\\x7fELF":'
    new = '                with filepath.open("rb") as stream:\n                    magic = stream.read(4)\n                if magic != b"\\x7fELF":'
    if original.count(old) != 1:
        raise RuntimeError(
            "Expected exactly the pinned whole-file probe; review source before experimenting"
        )
    sources = output / "sources"
    sources.mkdir()
    paths = {}
    for method, text in [("A", original), ("B", original.replace(old, new))]:
        paths[method] = sources / f"runpath_{method}.py"
        paths[method].write_text(text, encoding="utf-8")
    return paths


def convert_worker(source: Path, root: Path, metrics: Path) -> None:
    import resource

    spec = importlib.util.spec_from_file_location("actual_runpath", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    before_self = resource.getrusage(resource.RUSAGE_SELF)
    before_children = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    try:
        module.convert_runpath_to_rpath(root)
    finally:
        wall = time.monotonic() - started
        after_self = resource.getrusage(resource.RUSAGE_SELF)
        after_children = resource.getrusage(resource.RUSAGE_CHILDREN)
        own = (after_self.ru_utime - before_self.ru_utime) + (
            after_self.ru_stime - before_self.ru_stime
        )
        child = (after_children.ru_utime - before_children.ru_utime) + (
            after_children.ru_stime - before_children.ru_stime
        )
        dump(
            metrics,
            dict(
                conversion_wall_s=wall,
                self_cpu_s=own,
                waited_child_cpu_s=child,
                aggregate_cpu_s=own + child,
                peak_self_rss_bytes=after_self.ru_maxrss * 1024,
            ),
        )


def run_conversion(
    source: Path, root: Path, output: Path, timeout: float
) -> dict[str, Any]:
    profile = Path(__file__).resolve().parents[2] / "profile_build.py"
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("ROCM_PATH", "ROCM_DIR", "HIP_PATH", "HIP_DIR")
    }
    command = [
        sys.executable,
        str(profile),
        "--output",
        str(output / "profile"),
        "--",
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        str(source),
        "--root",
        str(root),
        "--output",
        str(output / "conversion.json"),
    ]
    output.mkdir(parents=True)
    with (output / "stdout.log").open("wb") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise RuntimeError(
                "Conversion exceeded its runtime budget; process group terminated"
            )
    if code:
        raise RuntimeError(
            f"Conversion/profile command failed with code {code}; see {output}/stdout.log"
        )
    lines = (output / "stdout.log").read_text(errors="replace").splitlines()
    errors = [
        line
        for line in lines
        if "patchelf failed" in line or "Couldn't update rocm.cfg" in line
    ]
    if errors:
        dump(output / "conversion-errors.json", errors)
        raise RuntimeError("Actual conversion reported patchelf/configuration errors")
    value = json.loads((output / "conversion.json").read_text())
    value.update(
        stdout_multiset_sha256=digest("\n".join(sorted(lines)).encode()),
        converted_log_lines=sum(
            "DT_RUNPATH changed to DT_RPATH:" in line for line in lines
        ),
        error_count=len(errors),
    )
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--max-wall-seconds", type=int, default=12600)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Actual preprocessing diagnostics require Linux")
    if args.worker:
        convert_worker(args.worker, args.root, args.output)
        return
    pristine, output = args.root.resolve(), args.output.resolve()
    if not pristine.is_dir() or output == pristine or pristine in output.parents:
        parser.error("Output must be outside an existing pristine artifact tree")
    work = pristine.parent / "preprocess-work"
    if work.exists() or output.exists():
        parser.error(
            "Use fresh diagnostic output/work paths; existing paths are never adopted"
        )
    output.mkdir(parents=True)
    started = time.monotonic()
    deadline = started + args.max_wall_seconds
    try:
        tools = {}
        for tool in ["readelf", "patchelf"]:
            if not shutil.which(tool):
                raise RuntimeError(f"Required tool missing: {tool}")
            tools[tool] = subprocess.check_output(
                [tool, "--version"], text=True
            ).splitlines()[0]
        check_links(pristine)
        source_paths = prepare_sources(output)
        initial = manifest(pristine, output / "pristine-before.jsonl", deadline)
        if not initial["counts"]["configs"]:
            raise RuntimeError(
                "No rocm.cfg in the artifact tree; reject external ROCM_PATH fallback"
            )
        metadata = dict(
            tools=tools,
            root=str(pristine),
            work=str(work),
            order="ABBA",
            initial=initial,
            source_sha256={k: digest(v.read_bytes()) for k, v in source_paths.items()},
            ci={
                k: os.getenv(k) for k in ["GITHUB_SHA", "GITHUB_RUN_ID", "RUNNER_NAME"]
            },
            scope="Actual conversion only timed; copies, full hashes and dynamic verification outside",
            cache_state="No cache dropping; all copies/reset verification precede each timed pass",
            max_wall_seconds=args.max_wall_seconds,
        )
        dump(output / "metadata.json", metadata)
        rows = []
        for index, method in enumerate("ABBA"):
            check_deadline(deadline)
            copies = copy_tree(
                pristine, work, initial["counts"]["bytes"], deadline, owned=index > 0
            )
            before = manifest(work, output / f"pass-{index}-input.jsonl", deadline)
            if before["content_sha256"] != initial["content_sha256"]:
                raise RuntimeError(
                    "Reset work copy does not match pristine content/modes/links"
                )
            if index == 0:
                estimate = (
                    (time.monotonic() - started)
                    + 3 * (copies["wall_s"] + before["wall_s"])
                    + 5 * initial["wall_s"]
                    + 4 * 720
                )
                dump(
                    output / "planning-estimate.json",
                    dict(
                        estimated_total_s=estimate,
                        note="Conservative planning estimate, not measured conversion performance",
                    ),
                )
                if estimate > args.max_wall_seconds:
                    raise RuntimeError(
                        "Observed copy/hash costs exceed planned runtime budget; use the retained procedure"
                    )
            pass_output = output / f"pass-{index}-{method}"
            conversion = run_conversion(
                source_paths[method],
                work,
                pass_output,
                timeout=min(1800, max(1, deadline - time.monotonic())),
            )
            after = manifest(
                work, output / f"pass-{index}-post.jsonl", deadline, dynamic=True
            )
            row = dict(
                index=index,
                method=method,
                reset=copies,
                input=before,
                conversion=conversion,
                post=after,
            )
            rows.append(row)
            dump(output / "passes.json", rows)
            print(
                json.dumps(dict(index=index, method=method, conversion=conversion)),
                flush=True,
            )
        expected = rows[0]
        for row in rows:
            for key in ["content_sha256", "dynamic_sha256", "readelf_errors", "counts"]:
                if row["post"][key] != expected["post"][key]:
                    raise RuntimeError(f"Post-conversion equivalence failed: {key}")
            if (
                row["conversion"]["stdout_multiset_sha256"]
                != expected["conversion"]["stdout_multiset_sha256"]
            ):
                raise RuntimeError(
                    "Conversion decisions/log messages differ across passes"
                )
        final = manifest(pristine, output / "pristine-after.jsonl", deadline)
        if any(
            final[k] != initial[k] for k in ["content_sha256", "mtime_sha256", "counts"]
        ):
            raise RuntimeError("Pristine input changed during experiment")
        dump(
            output / "summary.json",
            dict(
                validated=True,
                passes=rows,
                pristine_unchanged=True,
                post_content_modes_links_and_dynamic_tags_equal=True,
                total_diagnostic_wall_s=time.monotonic() - started,
                limitation="Conversion-only comparison, not complete packaging or CI speedup",
            ),
        )
    except Exception as error:
        dump(
            output / "failure.json",
            dict(error=str(error), elapsed_s=time.monotonic() - started),
        )
        raise


if __name__ == "__main__":
    main()
