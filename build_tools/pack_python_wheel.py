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

import argparse
import hashlib
import base64
import csv
import glob
import io
import os
import re
import sys
import zipfile


def _infer_platform_tag(pkg_dir):
    for path in glob.glob(os.path.join(pkg_dir, "*.so")):
        # Linux: module.cpython-312-x86_64-linux-gnu.so
        m = re.search(r"\.cpython-(\d+)-(\w+)-linux-\w+\.so$", os.path.basename(path))
        if m:
            ver, arch = m.group(1), m.group(2)
            return f"cp{ver}", f"cp{ver}", f"linux_{arch}"
    for path in glob.glob(os.path.join(pkg_dir, "*.pyd")):
        # Windows: module.cp312-win_amd64.pyd
        m = re.search(r"\.cp(\d+)-(win_\w+)\.pyd$", os.path.basename(path))
        if m:
            ver, plat = m.group(1), m.group(2)
            return f"cp{ver}", f"cp{ver}", plat
    return "py3", "none", "any"


def _sha256_record(data: bytes):
    digest = (
        base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    )
    return f"sha256={digest}", str(len(data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pkg-dir", required=True, help="Directory containing the built package files"
    )
    parser.add_argument("--name", required=True, help="Package name")
    parser.add_argument("--version", required=True, help="Package version")
    parser.add_argument(
        "--wheel-dir", required=True, help="Output directory for the .whl file"
    )
    args = parser.parse_args()

    name, version = args.name, args.version
    norm_name = re.sub(r"[-_.]+", "_", name)
    pkg_name = os.path.basename(args.pkg_dir)

    py_tag, abi_tag, plat_tag = _infer_platform_tag(args.pkg_dir)
    tag = f"{py_tag}-{abi_tag}-{plat_tag}"
    wheel_name = f"{norm_name}-{version}-{tag}.whl"

    os.makedirs(args.wheel_dir, exist_ok=True)
    wheel_path = os.path.join(args.wheel_dir, wheel_name)

    dist_info = f"{norm_name}-{version}.dist-info"
    records = []

    with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(args.pkg_dir):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                arcname = os.path.join(pkg_name, os.path.relpath(fpath, args.pkg_dir))
                data = open(fpath, "rb").read()
                zf.writestr(arcname, data)
                records.append((arcname, *_sha256_record(data)))

        metadata = f"Metadata-Version: 2.1\n" f"Name: {name}\n" f"Version: {version}\n"
        meta_path = f"{dist_info}/METADATA"
        zf.writestr(meta_path, metadata)
        records.append((meta_path, *_sha256_record(metadata.encode())))

        wheel_meta = (
            f"Wheel-Version: 1.0\n"
            f"Generator: pack_python_wheel\n"
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
