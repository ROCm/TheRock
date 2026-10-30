#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Validate an explicitly selected prior build; never rebuild or reattribute it."""

import argparse
from collections import OrderedDict
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build_tools"))
from package_dasan_stack import NoRedirect, github_artifact_request


class RemoteZip(io.RawIOBase):
    def __init__(
        self, size, url_provider, block_size=16 * 1024**2, cache_blocks=4, timeout=10800
    ):
        self.size, self.provider = size, url_provider
        self.block_size, self.cache_blocks = block_size, cache_blocks
        self.position, self.network_bytes = 0, 0
        self.cache = OrderedDict()
        self.byte_budget = 128 * 1024**2
        self.deadline = time.monotonic() + timeout
        self.progress = time.monotonic()
        self.url = None

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        target = offset + (
            self.position if whence == 1 else self.size if whence == 2 else 0
        )
        if not 0 <= target <= self.size:
            raise ValueError("ZIP seek outside artifact")
        self.position = target
        return target

    def _block(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        start = index * self.block_size
        end = min(start + self.block_size, self.size) - 1
        for attempt in range(3):
            try:
                remaining = self.deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("Delivery transfer time budget exhausted")
                if not self.url:
                    self.url = self.provider()
                request = urllib.request.Request(
                    self.url, headers={"Range": f"bytes={start}-{end}"}
                )
                with urllib.request.urlopen(
                    request, timeout=min(30, remaining)
                ) as response:
                    if (
                        response.status != 206
                        or response.headers.get("Content-Range")
                        != f"bytes {start}-{end}/{self.size}"
                    ):
                        raise RuntimeError("Server rejected exact bounded ZIP range")
                    if (
                        int(response.headers.get("Content-Length", end - start + 1))
                        != end - start + 1
                    ):
                        raise RuntimeError("ZIP range length mismatch")
                    chunks = []
                    received = 0
                    while received < end - start + 1:
                        if time.monotonic() >= self.deadline:
                            raise RuntimeError(
                                "Delivery transfer time budget exhausted"
                            )
                        maximum = min(1024**2, end - start + 1 - received)
                        if self.network_bytes + maximum > self.byte_budget:
                            raise RuntimeError("Selected-member byte budget exhausted")
                        chunk = response.read1(maximum)
                        if not chunk:
                            raise OSError("Short ZIP range response")
                        chunks.append(chunk)
                        received += len(chunk)
                        self.network_bytes += len(chunk)
                        if time.monotonic() - self.progress >= 30:
                            print(
                                f"Delivery network progress: {self.network_bytes} bytes (bounded selected-member transfer)",
                                flush=True,
                            )
                            self.progress = time.monotonic()
                data = b"".join(chunks)
                self.cache[index] = data
                if len(self.cache) > self.cache_blocks:
                    self.cache.popitem(last=False)
                return data
            except RuntimeError:
                raise
            except Exception as error:
                self.url = None
                if attempt == 2:
                    raise RuntimeError(
                        f"ZIP range transfer failed ({type(error).__name__})"
                    ) from None
                print(
                    f"Retrying exact ZIP range {start}-{end} ({type(error).__name__})",
                    flush=True,
                )
        raise RuntimeError("ZIP transfer exhausted attempts")

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        if size > 32 * 1024**2:
            raise RuntimeError("ZIP consumer requested an unbounded read")
        result = []
        while size:
            index, offset = divmod(self.position, self.block_size)
            block = self._block(index)
            take = min(size, len(block) - offset)
            result.append(block[offset : offset + take])
            self.position += take
            size -= take
        return b"".join(result)


def api_json(path, token):
    request = urllib.request.Request(
        "https://api.github.com/repos/ROCm/TheRock/" + path,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.build_opener(NoRedirect).open(request, timeout=30) as response:
        return json.load(response)


def metadata(archive, directory, origin_run, origin_sha):
    directory.mkdir(parents=True, exist_ok=True)
    infos = archive.infolist()
    tests = [
        i for i in infos if "-tests-" in i.filename and i.filename.endswith(".tar.gz")
    ]
    if len(tests) != 1:
        raise RuntimeError("Expected exactly one full-tests SDK archive")
    if Path(tests[0].filename).name != tests[0].filename:
        raise RuntimeError("Unexpected SDK member path")
    names = set()
    for item in infos:
        if item.is_dir() or item.filename.endswith(".tar.gz"):
            continue
        if (
            item.filename in names
            or Path(item.filename).name != item.filename
            or item.file_size > 2 * 1024**2
        ):
            raise RuntimeError("Duplicate, unsafe or oversized metadata member")
        names.add(item.filename)
        (directory / item.filename).write_bytes(archive.read(item))
    manifest = json.loads((directory / "source-config-manifest.json").read_text())
    if (
        str(manifest["run_id"]) != origin_run
        or manifest["repository"] != "ROCm/TheRock"
        or manifest["therock_sha"] != origin_sha
        or manifest["installation_archive"] != tests[0].filename
    ):
        raise RuntimeError("Delivery provenance differs from selected original build")
    checksums = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9.+_-]+)", line)
        if not match or match[2] in checksums:
            raise RuntimeError("Invalid master checksum entry")
        checksums[match[2]] = match[1]
    for name in names - {"SHA256SUMS"}:
        if (
            checksums.get(name)
            != hashlib.sha256((directory / name).read_bytes()).hexdigest()
        ):
            raise RuntimeError(f"Metadata checksum mismatch: {name}")
    if any(name not in names for name in checksums if not name.endswith(".tar.gz")):
        raise RuntimeError("Master checksum names a missing metadata file")
    if tests[0].filename not in checksums:
        raise RuntimeError("Missing full-tests SDK checksum")
    return tests[0], checksums[tests[0].filename], manifest


def download_selected(reader, directory, origin_run, origin_sha):
    with zipfile.ZipFile(reader) as archive:
        selected, expected, manifest = metadata(
            archive, directory, origin_run, origin_sha
        )
        # Historical delivery metadata has no unpacked-size field. This is an
        # explicit conservative workspace reserve, not a claimed exact size.
        required = selected.file_size + (300 + 15) * 1024**3
        if shutil.disk_usage(directory).free < required:
            raise RuntimeError(
                f"Insufficient disk: require {required} bytes including 300 GiB extraction reserve and 15 GiB scratch"
            )
        reader.byte_budget = (
            reader.network_bytes + selected.compress_size + 256 * 1024**2
        )
        target = directory / selected.filename
        partial = target.with_name(target.name + ".part")
        digest = hashlib.sha256()
        count = 0
        progress = time.monotonic()
        with archive.open(selected) as source, partial.open("wb") as output:
            while chunk := source.read(1024**2):
                output.write(chunk)
                digest.update(chunk)
                count += len(chunk)
                if time.monotonic() - progress >= 30:
                    print(
                        f"Tests SDK: {count}/{selected.file_size} bytes; network {reader.network_bytes} bytes",
                        flush=True,
                    )
                    progress = time.monotonic()
        # ZipExtFile checks ZIP CRC while reading; the independent master
        # SHA256 additionally authenticates the selected SDK's exact bytes.
        if count != selected.file_size or digest.hexdigest() != expected:
            raise RuntimeError("Full-tests SDK size/SHA256 mismatch")
        partial.replace(target)
        (directory / "SELECTED_SHA256SUMS").write_text(
            f"{expected}  {selected.filename}\n"
        )
        return {
            "archive": selected.filename,
            "archive_bytes": count,
            "archive_sha256": expected,
            "network_bytes": reader.network_bytes,
            "delivery_zip_bytes": reader.size,
            "original_build_run_id": origin_run,
            "original_build_sha": origin_sha,
            "disk_preflight_bytes": required,
            "extraction_reserve_gib": 300,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin-run-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validation-dir", required=True, type=Path)
    args = parser.parse_args()
    if not args.origin_run_id.isdecimal():
        raise RuntimeError("Original build run ID must be numeric")
    token = os.environ["GITHUB_TOKEN"]
    provenance = {
        "original_build_run_id": args.origin_run_id,
        "verification_run_id": os.environ.get("GITHUB_RUN_ID"),
        "verification_tools_sha": os.environ.get("GITHUB_SHA"),
        "status": "running",
    }
    try:
        run = api_json(f"actions/runs/{args.origin_run_id}", token)
        artifacts = api_json(
            f"actions/runs/{args.origin_run_id}/artifacts?per_page=100&name=dasan-stack-sdk-gfx950",
            token,
        )["artifacts"]
        items = [
            a
            for a in artifacts
            if a["name"] == "dasan-stack-sdk-gfx950" and not a["expired"]
        ]
        if len(items) != 1:
            raise RuntimeError(
                "Original run must have exactly one retained combined SDK delivery"
            )
        item = items[0]
        if str(item["workflow_run"]["id"]) != args.origin_run_id:
            raise RuntimeError("Artifact belongs to another original build")
        reader = RemoteZip(
            item["size_in_bytes"],
            lambda: github_artifact_request(str(item["id"]), token, location=True),
        )
        provenance.update(
            download_selected(
                reader, args.output_dir, args.origin_run_id, run["head_sha"]
            ),
            sdk_artifact_id=item["id"],
            status="passed",
        )
        print(json.dumps(provenance, indent=2), flush=True)
    except Exception as error:
        provenance.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        args.validation_dir.mkdir(parents=True, exist_ok=True)
        (args.validation_dir / "delivery-provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
