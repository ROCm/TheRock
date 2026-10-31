#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Preserve generated Ninja evidence outside the timed build.

Only build files and command data are captured, never environment variables.
Ninja compdb expands generated rule response files. Referenced RCCL custom
link_objects.rsp bodies are retained separately under strict build-path guards.
"""

import argparse
import hashlib
import io
import json
import os
import re
import stat
from pathlib import Path
import subprocess
import tarfile
import tempfile
import time

MAX_CUSTOM_RSP_FILES = 128
MAX_CUSTOM_RSP_BYTES = 8 * 1024 * 1024
MAX_CUSTOM_RSP_TOTAL_BYTES = 32 * 1024 * 1024


def custom_response_paths(commands: str, tree: Path) -> set[Path]:
    paths = set()
    # Custom Python link commands use @device_build/<arch>/link_objects.rsp;
    # compdb -x cannot reconstruct this Python-written body from a Ninja rsp rule.
    for match in re.finditer(r"(?:^|\s)@(?:\"([^\"]+)\"|'([^']+)'|([^\s]+))", commands):
        value = next(v for v in match.groups() if v is not None)
        path = Path(value)
        if path.name == "link_objects.rsp":
            paths.add(path if path.is_absolute() else tree / path)
    return paths


def read_custom_response(path: Path, build_dir: Path) -> bytes:
    if ".." in path.parts or (path != build_dir and build_dir not in path.parents):
        raise ValueError("custom response path escapes build directory")
    current = path
    while current != build_dir:
        if current.is_symlink():
            raise ValueError("custom response path contains a symlink")
        current = current.parent
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError("custom response must be a regular single-link file")
    if before.st_size > MAX_CUSTOM_RSP_BYTES:
        raise ValueError("custom response exceeds per-file byte bound")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        data = stream.read(MAX_CUSTOM_RSP_BYTES + 1)
        after = os.fstat(stream.fileno())

    def identity(value):
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if (
        len(data) != before.st_size
        or identity(before) != identity(opened)
        or identity(opened) != identity(after)
        or identity(after) != identity(path.stat())
    ):
        raise ValueError("custom response changed during capture")
    return data


def capture(build_dir: Path, output: Path) -> None:
    build_dir = build_dir.resolve()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    failures: list[dict[str, str | int]] = []
    trees = sorted({p.parent for p in build_dir.rglob("build.ninja")})
    custom_paths = set()
    custom_receipts = []
    custom_bytes = 0
    with tempfile.TemporaryDirectory(prefix="therock-graph-") as temporary:
        temporary_dir = Path(temporary)
        with tarfile.open(output, "w:gz") as archive:
            for tree in trees:
                relative = tree.relative_to(build_dir)
                for tool, arguments in (
                    ("commands", ["commands"]),
                    ("compdb", ["compdb", "-x"]),
                    ("targets", ["targets", "all"]),
                ):
                    destination = temporary_dir / relative / f"{tool}.txt"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open("wb") as handle:
                        result = subprocess.run(
                            ["ninja", "-C", str(tree), "-t", *arguments],
                            stdout=handle,
                            stderr=subprocess.PIPE,
                            check=False,
                        )
                    if result.returncode:
                        failures.append(
                            {
                                "tree": str(relative),
                                "tool": tool,
                                "returncode": result.returncode,
                                "stderr": result.stderr.decode(errors="replace"),
                            }
                        )
                    archive.add(destination, arcname=str(relative / destination.name))
                    if tool == "commands" and not result.returncode:
                        custom_paths.update(
                            custom_response_paths(
                                destination.read_text(errors="replace"), tree
                            )
                        )
            if len(custom_paths) > MAX_CUSTOM_RSP_FILES:
                failures.append(
                    dict(
                        tool="custom-response-files",
                        error="custom response count exceeds bound",
                        count=len(custom_paths),
                    )
                )
            else:
                for path in sorted(custom_paths):
                    row = dict(path=str(path))
                    try:
                        payload = read_custom_response(path, build_dir)
                        if custom_bytes + len(payload) > MAX_CUSTOM_RSP_TOTAL_BYTES:
                            raise ValueError("custom responses exceed total byte bound")
                        relative = path.relative_to(build_dir)
                        member = (Path("custom-response-files") / relative).as_posix()
                        info = tarfile.TarInfo(member)
                        info.size = len(payload)
                        info.mode = 0o600
                        archive.addfile(info, io.BytesIO(payload))
                        custom_bytes += len(payload)
                        row.update(
                            status="retained",
                            member=member,
                            bytes=len(payload),
                            sha256=hashlib.sha256(payload).hexdigest(),
                        )
                    except (OSError, ValueError) as error:
                        row.update(status="unavailable-or-rejected", error=str(error))
                        failures.append(
                            dict(
                                tool="custom-response-file",
                                path=str(path),
                                error=str(error),
                            )
                        )
                    custom_receipts.append(row)
            for path in sorted(build_dir.rglob("*")):
                # Source trees can contain inaccessible Windows entries. Only
                # inspect files in the diagnostic's declared generated-file set.
                if (
                    path.name == ".ninja_log" or path.suffix == ".ninja"
                ) and path.is_file():
                    archive.add(
                        path,
                        arcname=str(Path("generated") / path.relative_to(build_dir)),
                    )
            manifest = {
                "schema": 1,
                "build_dir": str(build_dir),
                "ninja_tree_count": len(trees),
                "elapsed_seconds": time.monotonic() - started,
                "github_sha": os.environ.get("GITHUB_SHA"),
                "github_run_id": os.environ.get("GITHUB_RUN_ID"),
                "stage": os.environ.get("STAGE_NAME"),
                "amdgpu_family": os.environ.get("AMDGPU_FAMILIES"),
                "failures": failures,
                "custom_response_files": custom_receipts,
                "custom_response_file_count": len(custom_paths),
                "custom_response_bytes": custom_bytes,
                "custom_response_files_complete": len(custom_paths)
                == len(custom_receipts)
                and all(row["status"] == "retained" for row in custom_receipts),
                "custom_response_scope": "Only command-referenced link_objects.rsp; missing/unsafe/changed/oversize bodies stay explicit, never a complete-parity claim",
            }
            manifest_path = temporary_dir / "capture-manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            archive.add(manifest_path, arcname=manifest_path.name)
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture(args.build_dir, args.output)


if __name__ == "__main__":
    main()
