# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from contextlib import redirect_stdout
import importlib.util
import io
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "packaging/linux"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark = load("benchmark_package_preprocess")
snapshot_module = load("package_preprocess_snapshot")
LINUX_TOOLS = sys.platform == "linux" and all(shutil.which(t) for t in ["tar", "zstd"])


class SnapshotTest(unittest.TestCase):
    def create(self, base):
        root = base / "output/artifacts"
        (root / "sub").mkdir(parents=True)
        (root / "sub/payload").write_bytes(b"content" * 1000)
        (root / "rocm.cfg").write_text("--enable-new-dtags\n")
        (root / "sub/payload").chmod(0o640)
        if sys.platform == "linux":
            (root / "sub").chmod(0o2755)
            os.link(root / "sub/payload", root / "hardlink")
            (root / "alias").symlink_to("sub", target_is_directory=True)
            (root / "filelink").symlink_to("sub/payload")
            for offset, path in enumerate(sorted(root.rglob("*")) + [root]):
                ns = 1700000000123456789 + offset * 1000
                os.utime(path, ns=(ns, ns), follow_symlinks=False)
        output = base / "diagnostics"
        output.mkdir()
        deadline = time.monotonic() + 90
        manifest_path = output / "initial.jsonl"
        initial = benchmark.manifest(root, manifest_path, deadline)
        reset = snapshot_module.SnapshotReset(
            root,
            root.parent / "preprocess-work",
            output / "snapshot",
            initial,
            manifest_path,
            deadline,
        )
        return root, output, initial, reset

    def receipt(self, base, root):
        receipt = base / "output/scan-diagnostics/input-ownership.json"
        receipt.parent.mkdir(parents=True)
        receipt.write_text(
            json.dumps(
                dict(
                    repository="ROCm/TheRock",
                    run_id="fixture",
                    sha="fixture-sha",
                    root=str(root),
                )
            )
        )
        return receipt

    def test_receipt_requires_exact_run_sha_workspace_and_owned_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root, _, _, _ = self.create(base)
            receipt = self.receipt(base, root)
            env = dict(
                GITHUB_WORKSPACE=str(base),
                GITHUB_RUN_ID="fixture",
                GITHUB_SHA="fixture-sha",
            )
            with patch.dict(os.environ, env):
                snapshot_module.validate_ownership(root, receipt)
                with patch.dict(os.environ, {"GITHUB_SHA": "other"}):
                    with self.assertRaisesRegex(RuntimeError, "Ownership receipt"):
                        snapshot_module.validate_ownership(root, receipt)
                with self.assertRaisesRegex(RuntimeError, "output/artifacts"):
                    snapshot_module.validate_ownership(base, receipt)
                with self.assertRaisesRegex(RuntimeError, "receipt path"):
                    snapshot_module.validate_ownership(root, base / "other.json")

    @unittest.skipUnless(LINUX_TOOLS, "requires Linux GNU tar and zstd")
    def test_roundtrip_preserves_modes_links_nanoseconds_and_recovers_pristine(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, output, initial, reset = self.create(Path(temporary))
            reset.prepare()
            self.assertFalse(root.exists())
            digest = reset.archive_sha256
            reset.restore(reset.work)
            benchmark.check_links(reset.work)
            restored = benchmark.manifest(
                reset.work, output / "restored.jsonl", reset.deadline
            )
            for key in ["content_sha256", "mtime_sha256", "counts"]:
                self.assertEqual(restored[key], initial[key], key)
            self.assertNotEqual(
                (reset.work / "sub/payload").stat().st_ino,
                (reset.work / "hardlink").stat().st_ino,
            )
            self.assertEqual((reset.work / "sub/payload").stat().st_nlink, 1)
            (reset.work / "sub/payload").write_text("mutated work only")
            reset.recover_pristine()
            recovered = benchmark.manifest(
                root, output / "recovered.jsonl", reset.deadline
            )
            for key in ["content_sha256", "mtime_sha256", "counts"]:
                self.assertEqual(recovered[key], initial[key], key)
            self.assertFalse(reset.work.exists())
            self.assertEqual(reset.hash_archive(), digest)

    @unittest.skipUnless(LINUX_TOOLS, "requires Linux GNU tar and zstd")
    def test_privilege_failure_is_preflighted_before_archive_or_input_removal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, _, _, reset = self.create(Path(temporary))
            chmod = Path.chmod

            def strip_setgid(path, mode, *args, **kwargs):
                if path.name == "directory" and mode & 0o2000:
                    mode &= ~0o2000
                return chmod(path, mode, *args, **kwargs)

            with patch.object(Path, "chmod", strip_setgid), patch.object(
                reset, "run", side_effect=RuntimeError("fixture sudo unavailable")
            ):
                with self.assertRaisesRegex(RuntimeError, "sudo unavailable"):
                    reset.prepare()
            self.assertTrue(root.exists())
            self.assertFalse(reset.archive.exists())
            self.assertFalse(reset.original_removed)
            self.assertEqual(reset.operations[-1]["status"], "failed")

    @unittest.skipUnless(LINUX_TOOLS, "requires Linux GNU tar and zstd")
    def test_privileged_mode_paths_reject_escape_and_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, _, _, reset = self.create(Path(temporary))
            for path in [root.parent, root / "alias"]:
                with self.assertRaisesRegex(RuntimeError, "Unsafe privileged"):
                    reset.sudo_directory_modes(
                        root, [path], 0o2755, root.parent / "log"
                    )
            reset.owned[root] = (-1, -1)
            with self.assertRaisesRegex(RuntimeError, "Unowned"):
                reset.sudo_directory_modes(root, [root], 0o2755, root.parent / "log")

    @unittest.skipUnless(
        LINUX_TOOLS
        and hasattr(os, "geteuid")
        and os.geteuid() == 0
        and Path("/usr/bin/sudo").exists(),
        "root fixture needed to create a foreign-group directory without changing sudo policy",
    )
    def test_real_unprivileged_foreign_group_roundtrip_with_guarded_chmod(self):
        # The parent executes only the child's guarded chmod requests. The child
        # has a real non-root UID and no supplementary groups; no sudoers edits.
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, output, initial, reset = self.create(base)
            for path in [base, *base.rglob("*")]:
                mode = path.lstat().st_mode & 0o7777
                os.chown(path, 65533, 65534, follow_symlinks=False)
                if not path.is_symlink():
                    os.chmod(path, mode)
            root.parent.chmod(0o2755)
            parent, child = multiprocessing.Pipe()
            pid = os.fork()
            if pid == 0:
                parent.close()
                try:
                    os.setgroups([])
                    os.setgid(65533)
                    os.setuid(65533)
                    original = reset.run

                    def mediated(command, log):
                        if command[0] == "/usr/bin/sudo":
                            child.send(dict(command=command))
                            if not child.recv():
                                raise RuntimeError("Privileged fixture chmod failed")
                            log.write_text("fixture parent executed guarded chmod\n")
                        else:
                            original(command, log)

                    reset.run = mediated
                    reset.prepare()
                    reset.restore(reset.work)
                    restored = benchmark.manifest(
                        reset.work, output / "restored.jsonl", reset.deadline
                    )
                    for key in ["content_sha256", "mtime_sha256", "counts"]:
                        assert restored[key] == initial[key], key
                    reset.recover_pristine()
                    recovered = benchmark.manifest(
                        root, output / "recovered.jsonl", reset.deadline
                    )
                    for key in ["content_sha256", "mtime_sha256", "counts"]:
                        assert recovered[key] == initial[key], key
                    child.send(dict(done=True))
                    child.close()
                    os._exit(0)
                except BaseException as error:
                    child.send(dict(error=str(error)))
                    child.close()
                    os._exit(1)
            child.close()
            count = 0
            finished = False
            try:
                while True:
                    self.assertTrue(parent.poll(30), "Foreign-group fixture timed out")
                    message = parent.recv()
                    self.assertNotIn("error", message, message)
                    if message.get("done"):
                        finished = True
                        break
                    command = message["command"]
                    self.assertEqual(
                        command[:6],
                        ["/usr/bin/sudo", "-n", "--", "/usr/bin/chmod", "2755", "--"],
                    )
                    for name in command[6:]:
                        path = Path(name)
                        self.assertTrue(
                            base in path.resolve().parents and path.is_dir()
                        )
                    completed = subprocess.run(command, capture_output=True)
                    parent.send(completed.returncode == 0)
                    count += 1
                self.assertGreaterEqual(count, 3)  # preflight + work + recovery
            finally:
                parent.close()
                if not finished:
                    os.kill(pid, signal.SIGKILL)
                _, status = os.waitpid(pid, 0)
            self.assertEqual(os.waitstatus_to_exitcode(status), 0)

    @unittest.skipUnless(LINUX_TOOLS, "requires Linux GNU tar and zstd")
    def test_restore_reapplies_setgid_modes_cleared_by_tar(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, output, initial, reset = self.create(Path(temporary))
            reset.prepare()
            original = reset.run

            def drop_setgid(command, log):
                original(command, log)
                if "--extract" in command:
                    (reset.work / "sub").chmod(0o755)

            with patch.object(reset, "run", side_effect=drop_setgid):
                operation = reset.restore(reset.work)
            self.assertEqual(operation["restored_directory_modes"], 1)
            restored = benchmark.manifest(
                reset.work, output / "restored.jsonl", reset.deadline
            )
            for key in ["content_sha256", "mtime_sha256", "counts"]:
                self.assertEqual(restored[key], initial[key], key)
            reset.recover_pristine()
            self.assertEqual((root / "sub").stat().st_mode & 0o7777, 0o2755)

    @unittest.skipUnless(LINUX_TOOLS, "requires Linux GNU tar and zstd")
    def test_corrupt_stream_never_deletes_original_and_changed_archive_cannot_restore(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root, _, _, reset = self.create(Path(temporary))
            original = reset.verify_stream

            def corrupt_then_verify():
                reset.archive.write_bytes(b"corrupt archive")
                return original()

            with patch.object(reset, "verify_stream", side_effect=corrupt_then_verify):
                with self.assertRaises((RuntimeError, tarfile.ReadError)):
                    reset.prepare()
            self.assertTrue(root.exists())
            self.assertFalse(reset.original_removed)
        with tempfile.TemporaryDirectory() as temporary:
            root, _, _, reset = self.create(Path(temporary))
            reset.prepare()
            reset.archive.chmod(0o644)
            reset.archive.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "SHA256 changed"):
                reset.restore(reset.work)
            self.assertFalse(reset.work.exists())

    @unittest.skipUnless(LINUX_TOOLS, "requires Linux GNU tar and zstd")
    def test_unsafe_archive_and_absolute_directory_links_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, output, _, reset = self.create(Path(temporary))
            plain = output / "unsafe.tar"
            with tarfile.open(plain, "w") as archive:
                member = tarfile.TarInfo("../escape")
                member.size = 1
                archive.addfile(member, io.BytesIO(b"x"))
            with reset.archive.open("wb") as stream:
                subprocess.run(["zstd", "-c", str(plain)], stdout=stream, check=True)
            with self.assertRaisesRegex(RuntimeError, "Unsafe path"):
                reset.verify_stream()
            self.assertTrue(root.exists())
        with tempfile.TemporaryDirectory() as temporary:
            root, output, _, _ = self.create(Path(temporary))
            (root / "absolute-dir").symlink_to(root / "sub", target_is_directory=True)
            initial = benchmark.manifest(
                root, output / "absolute.jsonl", time.monotonic() + 30
            )
            reset = snapshot_module.SnapshotReset(
                root,
                root.parent / "preprocess-work",
                output / "other-snapshot",
                initial,
                output / "absolute.jsonl",
                time.monotonic() + 30,
            )
            with self.assertRaisesRegex(RuntimeError, "Absolute directory"):
                reset.prepare()
            self.assertTrue(root.exists())
            self.assertFalse(reset.archive.exists())

    @unittest.skipUnless(LINUX_TOOLS, "requires Linux GNU tar and zstd")
    def test_partial_owned_removal_is_recovered_from_verified_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, output, initial, reset = self.create(Path(temporary))

            def interrupted_removal(path):
                (path / "rocm.cfg").unlink()
                raise OSError("fixture interrupted removal")

            with patch.object(shutil, "rmtree", side_effect=interrupted_removal):
                with self.assertRaisesRegex(OSError, "interrupted removal"):
                    reset.prepare()
            self.assertTrue(reset.original_removed)
            reset.recover_pristine()
            recovered = benchmark.manifest(
                root, output / "recovery.jsonl", reset.deadline
            )
            for key in ["content_sha256", "mtime_sha256", "counts"]:
                self.assertEqual(recovered[key], initial[key])

    @unittest.skipUnless(LINUX_TOOLS, "requires Linux GNU tar and zstd")
    def test_disk_deadline_and_unowned_deletion_fail_without_removing_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, _, _, reset = self.create(Path(temporary))
            with patch.object(
                shutil, "disk_usage", return_value=type("Disk", (), {"free": 0})()
            ):
                with self.assertRaisesRegex(RuntimeError, "disk reserve"):
                    reset.prepare()
            self.assertTrue(root.exists())
            self.assertFalse(reset.archive.exists())
            reset.deadline = time.monotonic() - 1
            with self.assertRaisesRegex(RuntimeError, "runtime budget"):
                reset.prepare()
            self.assertTrue(root.exists())
            reset.deadline = time.monotonic() + 30
            with self.assertRaisesRegex(RuntimeError, "unowned"):
                reset.remove_owned(root.parent)
            original = root.with_name("saved")
            root.rename(original)
            root.mkdir()
            with self.assertRaisesRegex(RuntimeError, "path/inode changed"):
                reset.remove_owned(root)
            self.assertTrue(original.exists())
        with tempfile.TemporaryDirectory() as temporary:
            root, output, initial, reset = self.create(Path(temporary))
            reset.prepare()
            digest = reset.archive_sha256
            reset.deadline = time.monotonic() - 1
            with self.assertRaisesRegex(RuntimeError, "runtime budget"):
                reset.restore(reset.work)
            self.assertTrue(reset.archive.exists())
            self.assertFalse(reset.work.exists())
            reset.deadline = time.monotonic() + 30
            reset.recover_pristine()
            recovered = benchmark.manifest(
                root, output / "after-stop.jsonl", reset.deadline
            )
            self.assertEqual(recovered["content_sha256"], initial["content_sha256"])
            self.assertEqual(reset.hash_archive(), digest)

    @unittest.skipUnless(
        LINUX_TOOLS and all(shutil.which(t) for t in ["cc", "readelf", "patchelf"]),
        "requires real Linux ELF conversion tools",
    )
    def test_actual_abba_snapshot_mode_and_failure_recover_identical_pristine(self):
        for fail_conversion in [False, True]:
            with self.subTest(
                fail_conversion=fail_conversion
            ), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary).resolve()
                root, _, _, _ = self.create(base)
                source = base / "fixture.c"
                source.write_text("int fixture(void) { return 42; }\n")
                subprocess.run(
                    [
                        "cc",
                        "-shared",
                        "-fPIC",
                        str(source),
                        "-Wl,-rpath,$ORIGIN",
                        "-o",
                        str(root / "sub/fixture.so"),
                    ],
                    check=True,
                )
                receipt = self.receipt(base, root)
                before = benchmark.manifest(
                    root, base / "before.jsonl", time.monotonic() + 30
                )
                output = base / "output/scan-diagnostics/preprocess"
                argv = [
                    str(SCRIPTS / "benchmark_package_preprocess.py"),
                    "--root",
                    str(root),
                    "--output",
                    str(output),
                    "--snapshot-reset",
                    "--input-ownership-file",
                    str(receipt),
                ]
                env = dict(
                    GITHUB_WORKSPACE=str(base),
                    GITHUB_RUN_ID="fixture",
                    GITHUB_SHA="fixture-sha",
                )
                with patch.dict(os.environ, env), patch.object(
                    sys, "argv", argv
                ), patch.object(benchmark, "require_rock_ci"), redirect_stdout(
                    io.StringIO()
                ):
                    if fail_conversion:
                        with patch.object(
                            benchmark,
                            "run_conversion",
                            side_effect=RuntimeError("fixture conversion failure"),
                        ):
                            with self.assertRaisesRegex(
                                RuntimeError, "fixture conversion failure"
                            ):
                                benchmark.main()
                        failure = json.loads((output / "failure.json").read_text())
                        self.assertEqual(
                            failure["recovery"], dict(restored=True, matched=True)
                        )
                    else:
                        benchmark.main()
                        summary = json.loads((output / "summary.json").read_text())
                        self.assertTrue(summary["validated"])
                        self.assertTrue(
                            summary["pristine_recovered_from_verified_snapshot"]
                        )
                        self.assertEqual(
                            [r["method"] for r in summary["passes"]], list("ABBA")
                        )
                after = benchmark.manifest(
                    root, base / "after.jsonl", time.monotonic() + 30
                )
                for key in ["content_sha256", "mtime_sha256", "counts"]:
                    self.assertEqual(before[key], after[key], key)
                self.assertFalse((root.parent / "preprocess-work").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
