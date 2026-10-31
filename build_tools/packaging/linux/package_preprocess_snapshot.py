# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Owned, verified compact resets for the package preprocessing diagnostic."""

from decimal import Decimal
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import stat
import subprocess
import tarfile
import tempfile
import time


def deadline_check(deadline):
    if time.monotonic() >= deadline:
        raise RuntimeError("Snapshot/reset runtime budget exhausted")


def validate_ownership(root: Path, receipt: Path):
    """A receipt is written by this workflow immediately after its own fetch."""
    workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve()
    if root != workspace / "output/artifacts" or root.is_symlink():
        raise RuntimeError("Snapshot removal requires this workflow's output/artifacts")
    if receipt.resolve() != workspace / "output/scan-diagnostics/input-ownership.json":
        raise RuntimeError("Unexpected ownership receipt path")
    record = json.loads(receipt.read_text())
    expected = dict(
        repository="ROCm/TheRock",
        run_id=os.environ["GITHUB_RUN_ID"],
        sha=os.environ["GITHUB_SHA"],
        root=str(root),
    )
    if record != expected:
        raise RuntimeError(
            "Ownership receipt does not match this run, SHA and input root"
        )
    return record


class SnapshotReset:
    """Only removes roots whose exact paths and inode identities it owns."""

    def __init__(self, root, work, output, initial, initial_manifest, deadline):
        self.root, self.work, self.output = (
            root.resolve(),
            work.resolve(),
            output.resolve(),
        )
        self.archive = self.root.parent / "preprocess-pristine.tar.zst"
        if (
            self.work != self.root.parent / "preprocess-work"
            or self.work.exists()
            or self.archive.exists()
            or self.root.is_symlink()
            or self.work.is_symlink()
            or self.archive.is_symlink()
        ):
            raise RuntimeError("Snapshot/work paths must be fresh owned siblings")
        self.initial, self.deadline = initial, deadline
        self.records = {
            r["path"]: r
            for r in map(json.loads, initial_manifest.read_text().splitlines())
        }
        self.owned = {self.root: self.inode(self.root)}
        self.original_removed = False
        self.archive_sha256 = None
        self.operations = []
        self.floor = max(5 * 1024**3, initial["counts"]["bytes"] // 10)
        self.floor += initial["largest_elf_bytes"]
        self.output.mkdir(exist_ok=True)

    @staticmethod
    def inode(path):
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise RuntimeError("Owned tree must be a physical directory")
        return info.st_dev, info.st_ino

    def record(self, operation, started, **values):
        result = dict(operation=operation, wall_s=time.monotonic() - started, **values)
        self.operations.append(result)
        (self.output / "operations.json").write_text(
            json.dumps(self.operations, indent=2)
        )
        return result

    def disk_check(self, expanded=False):
        free = shutil.disk_usage(self.root.parent).free
        required = self.floor + (self.initial["counts"]["bytes"] if expanded else 0)
        if free < required:
            raise RuntimeError(
                f"Snapshot disk reserve: free {free}, require {required}"
            )
        return free, required

    def run(self, command, log):
        deadline_check(self.deadline)
        with log.open("wb") as stream:
            process = subprocess.Popen(
                command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                while process.poll() is None:
                    deadline_check(self.deadline)
                    self.disk_check()
                    time.sleep(0.2)
                if process.returncode:
                    raise RuntimeError(
                        f"Snapshot command failed ({process.returncode}); see {log}"
                    )
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise

    def hash_archive(self):
        deadline_check(self.deadline)
        started = time.monotonic()
        hashed = hashlib.sha256()
        with self.archive.open("rb") as f:
            while chunk := f.read(1024 * 1024):
                deadline_check(self.deadline)
                hashed.update(chunk)
        value = hashed.hexdigest()
        self.record(
            "archive-sha256", started, bytes=self.archive.stat().st_size, sha256=value
        )
        if self.archive_sha256 is not None and value != self.archive_sha256:
            raise RuntimeError("Immutable snapshot SHA256 changed")
        return value

    def verify_stream(self):
        """Compare every archive payload, mode, link and nanosecond mtime."""
        started = time.monotonic()
        observed, payload = set(), 0
        with (self.output / "verify-zstd.log").open("wb") as log:
            process = subprocess.Popen(
                ["zstd", "-d", "-c", str(self.archive)],
                stdout=subprocess.PIPE,
                stderr=log,
            )
            try:
                with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
                    for member in archive:
                        deadline_check(self.deadline)
                        path = PurePosixPath(member.name)
                        if path.is_absolute() or ".." in path.parts:
                            raise RuntimeError("Unsafe path in compact snapshot")
                        name = path.as_posix()
                        if name.startswith("./"):
                            name = name[2:]
                        if name not in self.records or name in observed:
                            raise RuntimeError(
                                f"Unexpected/duplicate archive path: {name}"
                            )
                        expected = self.records[name]
                        mtime = int(
                            Decimal(str(member.pax_headers.get("mtime", member.mtime)))
                            * 10**9
                        )
                        if (
                            member.mode != expected["mode"]
                            or mtime != expected["mtime_ns"]
                        ):
                            raise RuntimeError(f"Snapshot mode/mtime mismatch: {name}")
                        if member.issym():
                            valid = (
                                expected["kind"] == "symlink"
                                and member.linkname == expected["target"]
                            )
                        elif member.isdir():
                            valid = expected["kind"] == "directory"
                        elif member.isfile():
                            hashed = hashlib.sha256()
                            with archive.extractfile(member) as stream:
                                while chunk := stream.read(1024 * 1024):
                                    deadline_check(self.deadline)
                                    hashed.update(chunk)
                                    payload += len(chunk)
                            valid = (
                                expected["kind"] == "file"
                                and member.size == expected["size"]
                                and hashed.hexdigest() == expected["sha256"]
                            )
                        else:
                            # Hardlinks and special files are never restored.
                            valid = False
                        if not valid:
                            raise RuntimeError(
                                f"Snapshot payload/type/link mismatch: {name}"
                            )
                        observed.add(name)
                process.stdout.close()
                if process.wait(timeout=max(1, self.deadline - time.monotonic())):
                    raise RuntimeError("Snapshot decompression failed")
            except BaseException:
                process.kill()
                process.wait()
                raise
            finally:
                process.stdout.close()
        if observed != self.records.keys():
            raise RuntimeError("Snapshot did not contain the complete physical tree")
        return self.record(
            "archive-stream-verification",
            started,
            entries=len(observed),
            payload_bytes=payload,
            matched=True,
        )

    def prepare(self):
        deadline_check(self.deadline)
        for record in self.records.values():
            if record["kind"] == "symlink" and os.path.isabs(record["target"]):
                path = self.root / record["path"]
                if path.is_dir() or path.name == "rocm.cfg":
                    raise RuntimeError(
                        "Absolute directory/config symlink changes meaning after snapshot reset"
                    )
        free, required = self.disk_check()
        # Exercise the actual inherited filesystem group before archiving or
        # deleting input. A normal-user chmod can silently clear setgid.
        self.preflight_directory_modes()
        started = time.monotonic()
        command = [
            "tar",
            "--create",
            "--format=pax",
            "--sort=name",
            "--hard-dereference",
            "--use-compress-program=zstd -T1 -3",
            "--file",
            str(self.archive),
            "--directory",
            str(self.root),
            ".",
        ]
        self.run(command, self.output / "create-tar.log")
        with self.archive.open("rb") as f:
            os.fsync(f.fileno())
        self.record(
            "snapshot-create",
            started,
            command=command,
            archive_bytes=self.archive.stat().st_size,
            free_before_bytes=free,
            reserve_bytes=required,
        )
        self.verify_stream()
        self.archive_sha256 = self.hash_archive()
        self.archive.chmod(0o444)
        # Logical bytes can overstate recoverable space for sparse/hardlinked files.
        free = shutil.disk_usage(self.root.parent).free
        recoverable = self.initial["unique_allocated_file_bytes"]
        if free + recoverable < self.initial["counts"]["bytes"] + self.floor:
            raise RuntimeError(
                "Verified snapshot plus expanded restore cannot fit with disk reserve"
            )
        self.original_removed = True
        self.remove_owned(self.root)
        self.disk_check(expanded=True)

    def sudo_directory_modes(self, destination, paths, mode, log):
        """Privileged chmod only, on physical directories this instance owns."""
        if (
            destination not in self.owned
            or destination.resolve() != destination
            or self.inode(destination) != self.owned[destination]
        ):
            raise RuntimeError("Unowned directory-mode restoration root")
        for path in paths:
            if (
                path.resolve() != path
                or (path != destination and destination not in path.parents)
                or not stat.S_ISDIR(path.lstat().st_mode)
            ):
                raise RuntimeError("Unsafe privileged directory-mode path")
        command = [
            "/usr/bin/sudo",
            "-n",
            "--",
            "/usr/bin/chmod",
            format(mode, "o"),
            "--",
            *map(str, paths),
        ]
        self.run(command, log)
        for path in paths:
            if stat.S_IMODE(path.lstat().st_mode) != mode:
                raise RuntimeError(
                    "Privileged chmod could not restore exact directory mode"
                )
        return command

    def preflight_directory_modes(self):
        started = time.monotonic()
        modes = sorted(
            {
                r["mode"]
                for r in self.records.values()
                if r["kind"] == "directory" and r["mode"] & 0o6000
            }
        )
        result = dict(
            status="passed",
            euid=os.geteuid(),
            egid=os.getegid(),
            groups=os.getgroups(),
            special_modes=modes,
            privileged_commands=[],
        )
        try:
            with tempfile.TemporaryDirectory(
                prefix=".preprocess-mode-probe-", dir=self.root.parent
            ) as temporary:
                owned = Path(temporary).resolve()
                self.owned[owned] = self.inode(owned)
                try:
                    probe = owned / "directory"
                    probe.mkdir()
                    result["inherited_gid"] = probe.stat().st_gid
                    for index, mode in enumerate(modes):
                        probe.chmod(mode)
                        if stat.S_IMODE(probe.stat().st_mode) != mode:
                            result["privileged_commands"].append(
                                self.sudo_directory_modes(
                                    owned,
                                    [probe],
                                    mode,
                                    self.output / f"mode-preflight-{index}.log",
                                )
                            )
                finally:
                    self.owned.pop(owned, None)
        except Exception as error:
            result.update(status="failed", error=str(error))
            raise
        finally:
            self.record("directory-modes-preflight", started, **result)

    def remove_owned(self, path):
        deadline_check(self.deadline)
        if path not in [self.root, self.work] or path not in self.owned:
            raise RuntimeError("Refusing removal of an unowned snapshot tree")
        if path.resolve() != path or self.inode(path) != self.owned[path]:
            raise RuntimeError("Owned snapshot tree path/inode changed")
        started = time.monotonic()
        shutil.rmtree(path)
        self.owned.pop(path)
        self.record("remove-owned-tree", started, path=str(path))

    def restore(self, destination):
        deadline_check(self.deadline)
        if destination not in [self.work, self.root]:
            raise RuntimeError("Unexpected snapshot restore path")
        if destination.exists() or destination.is_symlink():
            if destination != self.work:
                raise RuntimeError(
                    "Refusing to replace existing pristine recovery path"
                )
            self.remove_owned(destination)
        self.hash_archive()
        free, required = self.disk_check(expanded=True)
        destination.mkdir()
        self.owned[destination] = self.inode(destination)
        started = time.monotonic()
        command = [
            "tar",
            "--extract",
            "--preserve-permissions",
            "--delay-directory-restore",
            "--use-compress-program=zstd -T1 -d",
            "--file",
            str(self.archive),
            "--directory",
            str(destination),
        ]
        self.run(
            command, self.output / ("restore-" + str(len(self.operations)) + ".log")
        )
        # GNU tar can clear setgid on directories when extracting without
        # ownership privileges. Reapply the exact verified archive modes after
        # all descendants exist; the caller still checks the full manifest.
        restored_directory_modes = 0
        privileged_modes = defaultdict(list)
        for name, record in sorted(
            self.records.items(),
            key=lambda item: len(PurePosixPath(item[0]).parts),
            reverse=True,
        ):
            if record["kind"] != "directory":
                continue
            deadline_check(self.deadline)
            path = destination / name
            info = path.lstat()
            if not stat.S_ISDIR(info.st_mode):
                raise RuntimeError("Restored directory changed type")
            if stat.S_IMODE(info.st_mode) != record["mode"]:
                try:
                    path.chmod(record["mode"])
                except PermissionError:
                    pass
                restored_directory_modes += 1
                if stat.S_IMODE(path.lstat().st_mode) != record["mode"]:
                    privileged_modes[record["mode"]].append(path)
        commands = []
        for mode, paths in privileged_modes.items():
            for begin in range(0, len(paths), 64):
                commands.append(
                    self.sudo_directory_modes(
                        destination,
                        paths[begin : begin + 64],
                        mode,
                        self.output
                        / f"mode-restore-{len(self.operations)}-{len(commands)}.log",
                    )
                )
        # Keep deferred extraction writes outside the actual conversion timer.
        for directory, _, files in os.walk(destination, followlinks=False):
            for name in files:
                deadline_check(self.deadline)
                path = Path(directory) / name
                if not path.is_symlink():
                    if path.stat().st_nlink != 1:
                        raise RuntimeError(
                            "Snapshot restore produced a regular-file hardlink"
                        )
                    with path.open("rb") as f:
                        os.fsync(f.fileno())
        return self.record(
            "snapshot-restore",
            started,
            path=str(destination),
            command=command,
            free_before_bytes=free,
            required_bytes=required,
            restored_directory_modes=restored_directory_modes,
            privileged_directory_modes=sum(map(len, privileged_modes.values())),
            privileged_chmod_commands=commands,
        )

    def recover_pristine(self):
        if not self.original_removed:
            return
        if self.work.exists() or self.work.is_symlink():
            self.remove_owned(self.work)
        # A failed removal or failed recovery extraction may leave a partial
        # tree. Only retry removal when this instance still owns that inode.
        if self.root.exists() or self.root.is_symlink():
            self.remove_owned(self.root)
        self.restore(self.root)
        self.original_removed = False
