#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pack a pre-built Python package directory into a wheel.

This script creates a PEP 427 wheel from an already-installed package
directory (containing an extension module, __init__.py, etc.) and infers
the platform tag from the extension module filename.

Usage:
    python pack_python_wheel.py \
        --pkg-dir  /path/to/stage/hipdnn_frontend \
        --name hipdnn-frontend \
        --version 1.0.0 \
        --wheel-dir /path/to/output
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import re
import sys
import zipfile
from pathlib import Path

GENERATOR = "pack_python_wheel 0.1"


def _infer_platform_tag(pkg_dir: Path) -> tuple[str, str, str] | None:
    for fpath in sorted(pkg_dir.rglob("*")):
        fname = fpath.name
        if fname.endswith(".so"):
            m = re.search(r"\.cpython-(\d+)-(\w+)-linux-\w+\.so$", fname)
            if m:
                ver, arch = m.group(1), m.group(2)
                return f"cp{ver}", f"cp{ver}", f"linux_{arch}"
        elif fname.endswith(".pyd"):
            m = re.search(r"\.cp(\d+)-(win_\w+)\.pyd$", fname)
            if m:
                ver, plat = m.group(1), m.group(2)
                return f"cp{ver}", f"cp{ver}", plat
    return None


def _sha256_record(data: bytes) -> tuple[str, str]:
    digest = (
        base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    )
    return f"sha256={digest}", str(len(data))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pkg-dir",
        required=True,
        type=Path,
        help="Directory containing the built package files",
    )
    parser.add_argument("--name", required=True, help="Package name")
    parser.add_argument("--version", required=True, help="Package version")
    parser.add_argument(
        "--wheel-dir",
        required=True,
        type=Path,
        help="Output directory for the .whl file",
    )
    parser.add_argument(
        "--requires-python",
        default=">=3.9",
        help="Requires-Python specifier (default: >=3.9)",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Optional Summary field for METADATA",
    )
    parser.add_argument(
        "--homepage",
        default=None,
        help="Optional Home-page field for METADATA",
    )
    parser.add_argument(
        "--author",
        default=None,
        help="Optional Author field for METADATA",
    )
    parser.add_argument(
        "--license",
        default=None,
        help="Optional License field for METADATA (e.g., 'MIT')",
    )
    args = parser.parse_args()

    pkg_dir: Path = args.pkg_dir
    wheel_dir: Path = args.wheel_dir
    name: str = args.name
    version: str = args.version
    norm_name = re.sub(r"[-_.]+", "_", name)
    pkg_name = pkg_dir.name

    tag_parts = _infer_platform_tag(pkg_dir)
    if tag_parts is None:
        raise SystemExit(
            f"No native extension module (.so/.pyd) found under {pkg_dir}. "
            "This packer is for native bindings wheels; refusing to emit a "
            "py3-none-any wheel for a Root-Is-Purelib: false package."
        )
    py_tag, abi_tag, plat_tag = tag_parts
    tag = f"{py_tag}-{abi_tag}-{plat_tag}"
    wheel_name = f"{norm_name}-{version}-{tag}.whl"

    wheel_dir.mkdir(parents=True, exist_ok=True)
    wheel_path = wheel_dir / wheel_name

    dist_info = f"{norm_name}-{version}.dist-info"
    records: list[tuple[str, str, str]] = []

    with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in sorted(p for p in pkg_dir.rglob("*") if p.is_file()):
            arcname = f"{pkg_name}/{fpath.relative_to(pkg_dir).as_posix()}"
            data = fpath.read_bytes()
            compress_type = (
                zipfile.ZIP_STORED
                if fpath.suffix in (".so", ".pyd")
                else zipfile.ZIP_DEFLATED
            )
            zf.writestr(arcname, data, compress_type=compress_type)
            records.append((arcname, *_sha256_record(data)))

        metadata_lines = [
            "Metadata-Version: 2.1",
            f"Name: {name}",
            f"Version: {version}",
            f"Requires-Python: {args.requires_python}",
        ]
        if args.summary:
            metadata_lines.append(f"Summary: {args.summary}")
        if args.homepage:
            metadata_lines.append(f"Home-page: {args.homepage}")
        if args.author:
            metadata_lines.append(f"Author: {args.author}")
        if args.license:
            metadata_lines.append(f"License: {args.license}")
        metadata = "\n".join(metadata_lines) + "\n"
        meta_path = f"{dist_info}/METADATA"
        zf.writestr(meta_path, metadata)
        records.append((meta_path, *_sha256_record(metadata.encode())))

        wheel_meta = (
            f"Wheel-Version: 1.0\n"
            f"Generator: {GENERATOR}\n"
            f"Root-Is-Purelib: false\n"
            f"Tag: {tag}\n"
        )
        wheel_meta_path = f"{dist_info}/WHEEL"
        zf.writestr(wheel_meta_path, wheel_meta)
        records.append((wheel_meta_path, *_sha256_record(wheel_meta.encode())))

        record_path = f"{dist_info}/RECORD"
        buf = io.StringIO()
        writer = csv.writer(buf)
        for row in records:
            writer.writerow(row)
        writer.writerow((record_path, "", ""))
        zf.writestr(record_path, buf.getvalue())

    print(f"Created {wheel_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
