#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Measure native Windows hardlink sharing during staged cache publication.

Holding a DELETE-access handle lengthens a possible rename window. This is a
filesystem semantics probe, not evidence that CI takes that interleaving.
"""

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--directory",
        type=Path,
        help="Parent for a unique task-owned temporary directory",
    )
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("requires native Windows Python")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL
    read = kernel.ReadFile
    read.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.c_void_p,
    ]
    read.restype = wintypes.BOOL
    volume_info = kernel.GetVolumeInformationW
    volume_info.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPWSTR,
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.LPWSTR,
        wintypes.DWORD,
    ]
    volume_info.restype = wintypes.BOOL
    invalid = ctypes.c_void_p(-1).value
    read_access, delete_access = 0x80000000, 0x00010000
    share_all, open_existing = 7, 3
    records = []
    payload = b"verified shared tensor contents"
    expected = hashlib.md5(payload, usedforsecurity=False).hexdigest()

    def open_native(path, access):
        handle = create(str(path), access, share_all, None, open_existing, 0x80, None)
        if handle == invalid:
            raise ctypes.WinError(ctypes.get_last_error())
        return handle

    def attempt(phase, path):
        record = {"phase": phase, "alias": path.name}
        try:
            record["md5"] = hashlib.md5(
                path.read_bytes(), usedforsecurity=False
            ).hexdigest()
            record["passed"] = record["md5"] == expected
        except OSError as err:
            record.update(
                passed=False,
                exception=type(err).__name__,
                errno=err.errno,
                winerror=getattr(err, "winerror", None),
                message=str(err),
            )
        records.append(record)

    if args.directory:
        args.directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="toga-7-cache-sharing-", dir=args.directory
    ) as tmp:
        root = Path(tmp)
        # Record filesystem and Windows version: AWS containers may differ
        # from hosted runners even when both expose native Python.
        fs_name = ctypes.create_unicode_buffer(256)
        volume = ctypes.create_unicode_buffer(256)
        if not volume_info(
            str(root.anchor), volume, 256, None, None, None, fs_name, 256
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        for access_name, access in (
            ("zero-access-control", 0),
            ("read-control", read_access),
            ("delete-handle", delete_access),
        ):
            case = root / access_name
            case.mkdir()
            source, staging, cache = (case / n for n in ("source", "staging", "cache"))
            source.write_bytes(payload)
            os.link(source, staging)
            independent = case / "independent-copy"
            shutil.copyfile(source, independent)
            handle = open_native(staging, access)
            try:
                attempt(access_name + "-staged", source)
                attempt(access_name + "-staged", independent)
                # Real first publication (no destination exists). Holding the
                # handle tests whether cache readers can open the new alias.
                try:
                    os.rename(staging, cache)
                except OSError as err:
                    records.append(
                        {
                            "phase": access_name + "-rename",
                            "passed": False,
                            "error": str(err),
                            "winerror": getattr(err, "winerror", None),
                        }
                    )
                    continue
                attempt(access_name + "-published", cache)
                attempt(access_name + "-published", source)
                # Win32 reader sharing DELETE is a positive sharing control.
                reader = open_native(cache, read_access)
                try:
                    data = ctypes.create_string_buffer(len(payload))
                    count = wintypes.DWORD()
                    if not read(reader, data, len(payload), ctypes.byref(count), None):
                        raise ctypes.WinError(ctypes.get_last_error())
                    correct = data.raw[: count.value] == payload
                finally:
                    close(reader)
                records.append(
                    {"phase": access_name + "-shared-native-reader", "passed": correct}
                )
            finally:
                close(handle)
            attempt(access_name + "-closed", cache)

        # Normal Python readers themselves deny DELETE access. This verifies
        # the reciprocal sharing rule without involving a rename.
        control = root / "python-reader-control"
        control.write_bytes(payload)
        with control.open("rb"):
            try:
                handle = open_native(control, delete_access)
            except OSError as err:
                records.append(
                    {
                        "phase": "python-reader-denies-delete",
                        "passed": True,
                        "winerror": err.winerror,
                    }
                )
            else:
                close(handle)
                records.append(
                    {"phase": "python-reader-denies-delete", "passed": False}
                )

    result = {
        "platform": sys.platform,
        "python": platform.python_version(),
        "windows": platform.platform(),
        "filesystem": fs_name.value,
        "directory": str(args.directory) if args.directory else tempfile.gettempdir(),
        "records": records,
        "interpretation": "Held DELETE handle models a window; actual CI operation remains unproven.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    # Preserve every observation. A behavior difference is a research result,
    # not a reason to discard the artifact as a failed expected-output test.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
