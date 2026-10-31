# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Exact manifests and common clock controls for the opt-in DEB diagnostic.

No normalization: raw DEBs, control and payload bytes must agree. These controls
never run on the normal packaging path; performance/log clocks stay real.
"""

from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import time
from types import SimpleNamespace

EPOCH = 1793404800  # 2026-10-31 00:00:00 UTC; common to both allocations.
VERSION = "10.3.0~20261031"
SUFFIX = "41147801272"
INDEX_URL = (
    "https://therock-nightly-artifacts.s3.amazonaws.com/41147801272-linux/"
    "packages/deb/dists/stable/main/binary-amd64/Packages.gz"
)


def file_hash(path: Path) -> str:
    hashed = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hashed.update(chunk)
    return hashed.hexdigest()


def hardlink_groups(root: Path, manifest: Path) -> list[list[str]]:
    groups = {}
    with manifest.open() as stream:
        for line in stream:
            record = json.loads(line)
            if record["kind"] == "file":
                info = (root / record["path"]).stat()
                groups.setdefault((info.st_dev, info.st_ino), []).append(record["path"])
    return sorted(sorted(paths) for paths in groups.values() if len(paths) > 1)


def metadata_controls(deb, summary, repo, templates: Path) -> dict:
    """Patch module-local renderers, not global datetime or the system clock.

    install(1) resets copied changelog mtime: SOURCE_DATE_EPOCH + fixed date text
    alone are insufficient. gzip -n is a common diagnostic-only header control.
    """

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls.fromtimestamp(EPOCH, tz)

    deb.datetime = summary.datetime = FixedDateTime
    repo.datetime = SimpleNamespace(datetime=FixedDateTime, timezone=timezone)
    templates.mkdir()
    shutil.copytree(deb.SCRIPT_DIR / "template", templates / "template")
    rules = templates / "template/debian_rules.j2"
    original = rules.read_text()
    needle = "gzip -9 -f $(CURDIR)"
    if original.count(needle) != 1:
        raise RuntimeError("Review changed changelog gzip template")
    fallback = "|| dh_install;"
    if original.count(fallback) != 1:
        raise RuntimeError("Review changed hardlink install fallback")
    # Passing cp -al command/flags are byte-identical. Reject unexpected extra
    # full-variant copy BEFORE invoking fallback. Empty-package else unchanged.
    guard = "|| { echo THEROCK_DEB_PROFILE_FALLBACK_REJECTED >&2; exit 86; };"
    controlled = original.replace(needle, "gzip -n -9 -f $(CURDIR)").replace(
        fallback, guard
    )
    (templates / "template/debian_rules.uncontrolled.j2").write_text(original)
    rules.write_text(controlled)
    deb.SCRIPT_DIR = templates
    stock_run = repo.run_command

    def run_command(cmd, cwd=None, stdout=None):
        if cmd and cmd[0] == "gzip":
            if cmd != ["gzip", "-9c", "Packages"]:
                raise RuntimeError("Review changed repository gzip command")
            cmd = ["gzip", "-n", "-9c", "Packages"]
        return stock_run(cmd, cwd=cwd, stdout=stdout)

    repo.run_command = run_command
    os.environ["SOURCE_DATE_EPOCH"] = str(EPOCH)
    return dict(
        epoch=EPOCH,
        date=datetime.fromtimestamp(EPOCH, timezone.utc).isoformat(),
        clock_scope="Only deb_package/packaging_summary/repo metadata rendering",
        monotonic_and_log_clocks="Unmodified",
        changelog_gzip="Diagnostic template gzip -n; no header filename/mtime",
        repository_gzip="Diagnostic gzip -n; no header filename/mtime",
        install_guard="Common fail-fast replaces ONLY cp-al fallback; firing invalidates case; normal cp-al flags and empty-package dh_install unchanged",
        normal_hardlink_command_unchanged=True,
        guard_original=fallback,
        guard_controlled=guard,
        template_original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        template_controlled_sha256=file_hash(rules),
    )


def hardlink_capability(case: Path, output: Path) -> dict:
    """Prove normal cp -al behavior on the actual per-case temp filesystem."""
    fixture = case / "hardlink-capability"
    source, destination = fixture / "source", fixture / "destination"
    source.mkdir(parents=True)
    destination.mkdir()
    payload = source / "payload"
    payload.write_bytes(b"capability proof, never an artifact input\n")
    payload.chmod(0o640)
    os.utime(payload, (EPOCH, EPOCH))
    (source / "alias").symlink_to("payload")
    subprocess.run(
        ["cp", "-al", str(source) + "/.", str(destination) + "/"], check=True
    )
    a, b = payload.stat(), (destination / "payload").stat()
    expected = (
        a.st_dev,
        a.st_ino,
        a.st_mode,
        a.st_uid,
        a.st_gid,
        a.st_mtime_ns,
        a.st_size,
    )
    observed = (
        b.st_dev,
        b.st_ino,
        b.st_mode,
        b.st_uid,
        b.st_gid,
        b.st_mtime_ns,
        b.st_size,
    )
    if expected != observed or (destination / "alias").readlink() != Path("payload"):
        raise RuntimeError("Stock cp-al hardlink/metadata capability proof failed")
    result = dict(
        cp_command=["cp", "-al", "source/.", "destination/"],
        same_filesystem=a.st_dev == b.st_dev,
        shared_inode=a.st_ino == b.st_ino,
        regular_payload_copies=1,
        symlink_preserved=True,
        source_stat=expected,
        destination_stat=observed,
        fixture_scope="Actual case TMPDIR filesystem; no original inputs touched",
    )
    output.write_text(json.dumps(result, indent=2))
    shutil.rmtree(fixture)
    return result


def tar_manifest(stream, output: Path) -> dict:
    """Streaming logical extraction; no package files/links installed/executed."""
    records, payload = [], 0
    with tarfile.open(fileobj=stream, mode="r|") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError(f"Unsafe package member: {member.name}")
            record = dict(
                path=path.as_posix(),
                type=member.type.decode("ascii"),
                mode=member.mode,
                uid=member.uid,
                gid=member.gid,
                uname=member.uname,
                gname=member.gname,
                size=member.size,
                mtime=member.mtime,
                linkname=member.linkname,
                pax_headers=member.pax_headers,
                devmajor=member.devmajor,
                devminor=member.devminor,
            )
            if member.isfile():
                hashed = hashlib.sha256()
                with archive.extractfile(member) as source:
                    while chunk := source.read(1024 * 1024):
                        hashed.update(chunk)
                        payload += len(chunk)
                record["sha256"] = hashed.hexdigest()
            records.append(record)
    names = [r["path"] for r in records]
    if len(names) != len(set(names)):
        raise RuntimeError("Duplicate package tar path")
    records.sort(key=lambda r: r["path"])
    with output.open("w", encoding="utf-8") as target:
        for record in records:
            target.write(json.dumps(record, sort_keys=True) + "\n")
    return dict(entries=len(records), payload_bytes=payload, sha256=file_hash(output))


def ar_headers(path: Path) -> list[dict]:
    records = []
    with path.open("rb") as stream:
        if stream.read(8) != b"!<arch>\n":
            raise RuntimeError("Not a DEB ar container")
        while header := stream.read(60):
            if len(header) != 60 or header[58:] != b"`\n":
                raise RuntimeError("Malformed DEB ar header")
            size = int(header[48:58])
            records.append(
                dict(header_hex=header.hex(), member_bytes=size, offset=stream.tell())
            )
            stream.seek(size + size % 2, 1)
        if stream.tell() != path.stat().st_size:
            raise RuntimeError("DEB ar length mismatch")
    return records


def inspect_packages(packages: Path, output: Path) -> dict:
    """Capture every raw/control/payload digest, field and archive header."""
    output.mkdir()
    debs = sorted(packages.glob("*.deb"))
    if not debs:
        raise RuntimeError("No complete DEB payload to inspect")
    inventory = []
    for path in debs:
        package_output = output / path.name
        package_output.mkdir()
        fields = subprocess.check_output(["dpkg-deb", "--field", str(path)])
        (package_output / "fields.txt").write_bytes(fields)
        record = dict(
            name=path.name,
            size=path.stat().st_size,
            sha256=file_hash(path),
            ar_headers=ar_headers(path),
            fields_sha256=hashlib.sha256(fields).hexdigest(),
        )
        for kind, option in [("control", "--ctrl-tarfile"), ("data", "--fsys-tarfile")]:
            with (package_output / f"{kind}-stderr.log").open("wb") as log:
                process = subprocess.Popen(
                    ["dpkg-deb", option, str(path)], stdout=subprocess.PIPE, stderr=log
                )
                try:
                    record[kind] = tar_manifest(
                        process.stdout, package_output / f"{kind}.jsonl"
                    )
                    process.stdout.close()
                    if process.wait():
                        raise RuntimeError(f"dpkg-deb extraction failed: {path.name}")
                finally:
                    process.stdout.close()
                    if process.poll() is None:
                        process.kill()
                        process.wait()
        inventory.append(record)
    encoded = json.dumps(inventory, sort_keys=True, indent=2).encode()
    (output / "inventory.json").write_bytes(encoded)
    return dict(
        packages=len(inventory),
        raw_bytes=sum(r["size"] for r in inventory),
        inventory_sha256=hashlib.sha256(encoded).hexdigest(),
        inventory=inventory,
    )


def deb822(data: str) -> list[dict[str, str]]:
    entries = []
    for paragraph in data.strip().split("\n\n"):
        record, key = {}, None
        for line in paragraph.splitlines():
            if line.startswith((" ", "\t")) and key:
                record[key] += "\n" + line
            else:
                key, value = line.split(":", 1)
                if key in record:
                    raise RuntimeError("Duplicate repository field")
                record[key] = value.strip()
        entries.append(record)
    return entries


def verify_repository(packages: Path, inventory: list[dict]) -> dict:
    index = packages / "dists/stable/main/binary-amd64/Packages"
    compressed, release = index.with_suffix(".gz"), packages / "dists/stable/Release"
    if gzip.decompress(compressed.read_bytes()) != index.read_bytes():
        raise RuntimeError("Packages.gz does not match Packages")
    records = deb822(index.read_text())
    expected, observed = {r["name"]: r for r in inventory}, {}
    for record in records:
        name = record["Filename"]
        if name != "pool/main/" + Path(name).name:
            raise RuntimeError("Unsafe/unexpected index Filename")
        base = Path(name).name
        if base in observed or base not in expected:
            raise RuntimeError("Duplicate/unexpected repository package")
        value = expected[base]
        if (
            int(record["Size"]) != value["size"]
            or record["SHA256"] != value["sha256"]
            or file_hash(packages / name) != value["sha256"]
        ):
            raise RuntimeError("Repository package size/checksum mismatch")
        observed[base] = record
    if observed.keys() != expected.keys():
        raise RuntimeError("Repository index inventory differs")
    sections = {"MD5Sum": "md5", "SHA1": "sha1", "SHA256": "sha256"}
    section, checks = None, {key: set() for key in sections}
    for line in release.read_text().splitlines():
        if line.rstrip(":") in sections and line.endswith(":"):
            section = line[:-1]
        elif section and line.startswith(" "):
            digest, size, name = line.split()
            if name not in [
                "main/binary-amd64/Packages",
                "main/binary-amd64/Packages.gz",
            ]:
                raise RuntimeError("Unexpected Release checksum path")
            data = (packages / "dists/stable" / name).read_bytes()
            actual = hashlib.new(
                sections[section], data, usedforsecurity=False
            ).hexdigest()
            if digest != actual or int(size) != len(data) or name in checks[section]:
                raise RuntimeError("Release checksum mismatch/duplicate")
            checks[section].add(name)
        else:
            section = None
    if any(len(values) != 2 for values in checks.values()):
        raise RuntimeError("Incomplete Release checksum sections")
    return dict(
        indexed=len(records),
        files={
            str(p.relative_to(packages)): file_hash(p)
            for p in [index, compressed, release]
        },
    )


def floor_check(path: Path, reserve: int, extra: int = 0) -> dict:
    free, required = shutil.disk_usage(path).free, reserve + extra
    record = dict(
        free_bytes=free, required_bytes=required, missing_bytes=max(0, required - free)
    )
    if free < required:
        raise RuntimeError("Full-DEB disk capacity gate: " + json.dumps(record))
    return record


def monitored_run(
    command: list[str],
    output: Path,
    deadline: float,
    reserve: int,
    disk: Path,
    env=None,
) -> dict:
    """No shell; terminate the complete owned process group on budget failure."""
    output.mkdir()
    started, minimum = time.monotonic(), shutil.disk_usage(disk).free
    own_cpu_started = time.process_time()
    started_utc = datetime.now(timezone.utc).isoformat()
    heartbeat = started
    with (output / "stdout.log").open("wb") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Full-DEB runtime budget exhausted")
                value = floor_check(disk, reserve)
                minimum = min(minimum, value["free_bytes"])
                if time.monotonic() - heartbeat >= 30:
                    print(
                        json.dumps(
                            dict(
                                utc=datetime.now(timezone.utc).isoformat(),
                                phase=output.name,
                                wall_s=time.monotonic() - started,
                                free_bytes=value["free_bytes"],
                                stdout_bytes=(output / "stdout.log").stat().st_size,
                            )
                        ),
                        flush=True,
                    )
                    heartbeat = time.monotonic()
                time.sleep(0.2)
            code = process.returncode
        except BaseException:
            import signal

            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        finally:
            (output / "command.json").write_text(
                json.dumps(
                    dict(
                        command=command,
                        started_utc=started_utc,
                        ended_utc=datetime.now(timezone.utc).isoformat(),
                        wall_s=time.monotonic() - started,
                        controller_cpu_s=time.process_time() - own_cpu_started,
                        exit_code=process.returncode,
                        minimum_observed_free_bytes=minimum,
                        disk_poll_interval_s=0.2,
                    ),
                    indent=2,
                )
            )
    if code:
        raise RuntimeError(f"Command failed ({code}); see {output}/stdout.log")
    return json.loads((output / "command.json").read_text())
