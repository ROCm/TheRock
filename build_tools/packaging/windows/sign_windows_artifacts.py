#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Sign Portable Executable files in downloaded TheRock artifacts.

This script only signs files already present in a local directory. Fetch and
upload artifacts separately with the artifact tooling appropriate for the
calling environment.

Files ending in .exe, .dll, or .pyd are checked for PE magic bytes before they
are signed. Pass ``--all-files`` to check every file for PE magic bytes and
include extensionless PE files. Each file is verified and signed with
signtool.exe.

Examples::

    python sign_windows_artifacts.py \
        --input-dir therock-sign/artifacts \
        --thumbprint <SHA1>

    python sign_windows_artifacts.py \
        --input-dir therock-sign/artifacts \
        --thumbprint <SHA1> \
        --cert-file C:\\actions-runner\\cert.cer \
        --csp "eToken Base Cryptographic Provider" \
        --key <key>
"""

import argparse
import json
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

TIMESTAMP_URL = "http://timestamp.digicert.com"
CODE_SIGNING_OID = "1.3.6.1.5.5.7.3.3"
PE_EXTENSIONS = {".exe", ".dll", ".pyd"}


@dataclass(frozen=True)
class SigningSummary:
    total: int
    signed: int
    skipped: int
    failed: int


def detect_codesign_certs() -> list[dict[str, str]]:
    """Return code-signing certificates from the Windows certificate stores."""
    script = (
        "$certs = @(); "
        "foreach ($store in @('CurrentUser', 'LocalMachine')) { "
        "  try { "
        '    Get-ChildItem -Path "Cert:\\$store\\My" -ErrorAction Stop | '
        "    Where-Object { $_.EnhancedKeyUsageList.ObjectId -contains '"
        + CODE_SIGNING_OID
        + "' } | "
        "    ForEach-Object { "
        "      $certs += [PSCustomObject]@{ "
        "        Thumbprint = $_.Thumbprint; "
        "        Subject    = $_.Subject; "
        "        Issuer     = $_.Issuer; "
        "        NotAfter   = $_.NotAfter.ToString('yyyy-MM-dd'); "
        "        Store      = $store "
        "      } "
        "    } "
        "  } catch {} "
        "} "
        "$certs | ConvertTo-Json -Depth 2"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        print(
            "Warning: powershell not found; cannot auto-detect certificates.",
            file=sys.stderr,
        )
        return []

    if result.returncode != 0:
        print(
            f"Warning: PowerShell exited {result.returncode}; "
            "cannot auto-detect certificates.",
            file=sys.stderr,
        )
        return []

    output = result.stdout.strip()
    if not output:
        return []

    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        print(
            "Warning: unexpected PowerShell output; "
            "cannot auto-detect certificates.",
            file=sys.stderr,
        )
        return []

    if isinstance(data, dict):
        data = [data]
    return [
        {
            "thumbprint": entry["Thumbprint"],
            "subject": entry["Subject"],
            "issuer": entry["Issuer"],
            "not_after": entry["NotAfter"],
            "store": entry["Store"],
        }
        for entry in data
    ]


def resolve_cert(
    thumbprint: str | None,
    subject: str | None,
    machine_store: bool,
) -> tuple[str, bool]:
    """Resolve the certificate thumbprint and certificate store."""
    if thumbprint:
        return thumbprint, machine_store

    certs = detect_codesign_certs()
    if subject:
        subject_lower = subject.lower()
        certs = [cert for cert in certs if subject_lower in cert["subject"].lower()]

    if not certs:
        message = "No code-signing certificate found in the Windows certificate store."
        if subject:
            message += f" Subject filter: {subject!r}."
        raise RuntimeError(message)

    if len(certs) > 1:
        choices = "\n".join(
            f"  {cert['thumbprint']}  {cert['subject']}  ({cert['store']})"
            for cert in certs
        )
        raise RuntimeError(
            "Multiple code-signing certificates found. "
            f"Select one with --thumbprint:\n{choices}"
        )

    cert = certs[0]
    print("Auto-detected certificate:")
    print(f"  Thumbprint : {cert['thumbprint']}")
    print(f"  Subject    : {cert['subject']}")
    print(f"  Issuer     : {cert['issuer']}")
    print(f"  Expires    : {cert['not_after']}")
    print(f"  Store      : {cert['store']}")
    return cert["thumbprint"], cert["store"] == "LocalMachine"


def find_signtool(user_path: Path | None) -> Path:
    """Find signtool.exe from an explicit path or the Windows SDK."""
    if user_path is not None:
        if user_path.is_file():
            return user_path.resolve()
        raise FileNotFoundError(f"signtool path not found: {user_path}")

    kits_root = Path("C:/Program Files (x86)/Windows Kits/10/bin")
    if kits_root.is_dir():
        for kit in sorted(kits_root.iterdir(), reverse=True):
            if not kit.is_dir():
                continue
            for arch in ("x64", "x86"):
                candidate = kit / arch / "signtool.exe"
                if candidate.is_file():
                    return candidate

    raise FileNotFoundError(
        "signtool.exe not found. Install the Windows SDK or pass --signtool."
    )


def is_pe(path: Path) -> bool:
    """Return whether a file has valid DOS and PE signatures."""
    try:
        with path.open("rb") as file:
            if file.read(2) != b"MZ":
                return False
            file.seek(0x3C)
            offset_bytes = file.read(4)
            if len(offset_bytes) < 4:
                return False
            pe_offset = struct.unpack_from("<I", offset_bytes)[0]
            file.seek(pe_offset)
            return file.read(4) == b"PE\x00\x00"
    except OSError:
        return False


def iter_pe_files(root: Path, *, all_files: bool) -> list[Path]:
    """Return confirmed PE files below root in deterministic path order."""
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if not all_files and path.suffix.lower() not in PE_EXTENSIONS:
            continue
        if is_pe(path):
            files.append(path)
    return files


def is_signed(path: Path, signtool: Path) -> bool:
    """Return whether a file has a valid Authenticode signature."""
    result = subprocess.run(
        [str(signtool), "verify", "/pa", str(path)],
        capture_output=True,
    )
    return result.returncode == 0


def build_signtool_cmd(
    *,
    signtool: Path,
    path: Path,
    thumbprint: str,
    machine_store: bool,
    cert_file: str | None,
    csp: str | None,
    key: str | None,
) -> list[str]:
    """Build the signtool command used by the existing tarball signer."""
    command = [
        str(signtool),
        "sign",
        "/fd",
        "sha256",
        "/tr",
        TIMESTAMP_URL,
        "/td",
        "sha256",
        "/sha1",
        thumbprint,
    ]
    if cert_file:
        command += ["/f", cert_file]
    if csp:
        command += ["/csp", csp]
    if key:
        command += ["/k", key]
    if machine_store:
        command += ["/sm"]
    command.append(str(path))
    return command


def sign_file(
    *,
    signtool: Path,
    path: Path,
    thumbprint: str,
    machine_store: bool,
    cert_file: str | None,
    csp: str | None,
    key: str | None,
) -> bool:
    """Sign one file and return whether signtool succeeded."""
    command = build_signtool_cmd(
        signtool=signtool,
        path=path,
        thumbprint=thumbprint,
        machine_store=machine_store,
        cert_file=cert_file,
        csp=csp,
        key=key,
    )
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode == 0:
        return True

    output = (result.stderr or result.stdout or "(no output)").strip()
    print(f"Failed to sign {path}:\n{output}", file=sys.stderr)
    return False


def sign_artifacts(
    *,
    files: list[Path],
    signtool: Path,
    thumbprint: str,
    machine_store: bool,
    cert_file: str | None,
    csp: str | None,
    key: str | None,
    skip_signed: bool,
) -> SigningSummary:
    """Verify and sign PE files."""
    signed = 0
    skipped = 0
    failed = 0
    total = len(files)
    batch_number = 1
    batch_start_index = 1
    batch_start_time = time.monotonic()

    for index, path in enumerate(files, start=1):
        if skip_signed and is_signed(path, signtool):
            skipped += 1
        elif sign_file(
            signtool=signtool,
            path=path,
            thumbprint=thumbprint,
            machine_store=machine_store,
            cert_file=cert_file,
            csp=csp,
            key=key,
        ):
            signed += 1
        else:
            failed += 1

        if index % 100 == 0 or index == total:
            batch_elapsed = time.monotonic() - batch_start_time
            batch_file_count = index - batch_start_index + 1
            print(
                f"Processed batch {batch_number} ({batch_file_count} files) "
                f"in {batch_elapsed:.1f} seconds; {index}/{total} total: "
                f"{signed} signed, {skipped} skipped (already signed), {failed} failed"
            )
            batch_number += 1
            batch_start_index = index + 1
            batch_start_time = time.monotonic()

    return SigningSummary(
        total=total,
        signed=signed,
        skipped=skipped,
        failed=failed,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sign PE files in downloaded TheRock artifact directories."
    )
    parser.add_argument(
        "--input-dir",
        "--input",
        "-i",
        dest="input_dir",
        type=Path,
        required=True,
        help="Directory containing extracted artifact directories",
    )

    cert_group = parser.add_mutually_exclusive_group()
    cert_group.add_argument(
        "--thumbprint",
        metavar="SHA1",
        help="Certificate thumbprint from the Windows certificate store",
    )
    cert_group.add_argument(
        "--subject",
        "-n",
        metavar="NAME",
        help="Partial certificate subject name",
    )
    parser.add_argument(
        "--machine-store",
        action="store_true",
        help="Use the local machine certificate store",
    )
    parser.add_argument(
        "--signtool",
        type=Path,
        help="Path to signtool.exe; auto-detected from the Windows SDK if omitted",
    )
    parser.add_argument(
        "--cert-file",
        metavar="PATH",
        help="Public certificate file for token/CSP signing",
    )
    parser.add_argument(
        "--csp",
        metavar="NAME",
        help="Cryptographic Service Provider for the token",
    )
    parser.add_argument(
        "--key",
        metavar="NAME",
        help="Private key name on the token",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Check every file for PE magic bytes, including extensionless files",
    )
    parser.add_argument(
        "--skip-signed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip files that already have a valid signature (default: enabled)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and list PE files without signing",
    )
    args = parser.parse_args(argv)
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if not args.input_dir.is_dir():
        print(f"Input directory not found: {args.input_dir}", file=sys.stderr)
        return 1

    try:
        thumbprint, machine_store = resolve_cert(
            args.thumbprint,
            args.subject,
            args.machine_store,
        )
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    scan_description = (
        "all files" if args.all_files else ", ".join(sorted(PE_EXTENSIONS))
    )
    print(f"Scanning {args.input_dir} for PE files ({scan_description})...")
    scan_start_time = time.monotonic()
    files = iter_pe_files(args.input_dir, all_files=args.all_files)
    scan_elapsed = time.monotonic() - scan_start_time
    print(f"Found {len(files)} PE file(s).")
    print(f"File scan time: {scan_elapsed:.1f} seconds")
    if not files:
        return 0

    if args.dry_run:
        print("Dry run; files that would be processed:")
        for path in files:
            print(f"  {path}")
        return 0

    try:
        signtool = find_signtool(args.signtool)
    except FileNotFoundError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    store = "local machine" if machine_store else "current user"
    print(f"Using signtool: {signtool}")
    print(f"Certificate: {thumbprint} ({store} store)")
    print(f"Timestamp URL: {TIMESTAMP_URL}")
    print("Signing files serially with one signtool invocation per file.")

    start_time = datetime.now().astimezone()
    start_monotonic = time.monotonic()
    print(f"Signing started at: {start_time.isoformat(timespec='seconds')}")
    summary = sign_artifacts(
        files=files,
        signtool=signtool,
        thumbprint=thumbprint,
        machine_store=machine_store,
        cert_file=args.cert_file,
        csp=args.csp,
        key=args.key,
        skip_signed=args.skip_signed,
    )
    end_time = datetime.now().astimezone()
    total_elapsed = time.monotonic() - start_monotonic
    print(f"Signing ended at: {end_time.isoformat(timespec='seconds')}")
    print(f"Total signing time: {total_elapsed:.1f} seconds")
    print(
        f"Done: {summary.signed} signed, {summary.skipped} skipped (already signed), "
        f"{summary.failed} failed out of {summary.total}."
    )
    return 1 if summary.failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
