# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from contextlib import ExitStack
from datetime import datetime, timezone
import gzip
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "packaging/linux"
sys.path.insert(0, str(SCRIPTS))
import benchmark_full_deb as full
import deb_validation_support as support
import benchmark_package_preprocess as pre
import build_package as build
import deb_package as deb
import packaging_summary as summary
import build_package_repo as repo

LINUX = sys.platform == "linux"
DEB_TOOLS = LINUX and all(
    shutil.which(p) for p in ["dpkg-deb", "dpkg-scanpackages", "gzip"]
)


class FullDebTest(unittest.TestCase):
    def controls(self, directory, stack):
        for module, name in [
            (deb, "datetime"),
            (deb, "SCRIPT_DIR"),
            (summary, "datetime"),
            (repo, "datetime"),
            (repo, "run_command"),
        ]:
            stack.enter_context(patch.object(module, name, getattr(module, name)))
        stack.enter_context(patch.dict(os.environ))
        return support.metadata_controls(deb, summary, repo, directory)

    def test_metadata_clock_is_common_and_monotonic_stays_real(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            base = Path(temporary)
            before = time.monotonic()
            controls = self.controls(base / "templates", stack)
            self.assertEqual(deb.datetime.now(timezone.utc).timestamp(), support.EPOCH)
            self.assertEqual(
                summary.datetime.now(timezone.utc).timestamp(), support.EPOCH
            )
            self.assertEqual(
                repo.datetime.datetime.now(timezone.utc).timestamp(), support.EPOCH
            )
            self.assertGreater(time.monotonic(), before)
            self.assertNotEqual(datetime.now(timezone.utc).timestamp(), support.EPOCH)
            self.assertEqual(os.environ["SOURCE_DATE_EPOCH"], str(support.EPOCH))
            cfg = build.PackageConfig(
                artifacts_dir=base,
                dest_dir=base,
                pkg_type="deb",
                rocm_version=support.VERSION,
                version_suffix=support.SUFFIX,
                install_prefix="/opt/rocm/core-10.3",
                gfx_arch="",
            )
            out = base / "debian"
            out.mkdir()
            deb.generate_changelog_file(
                build.get_package_info("amdrocm-llvm"), out, cfg
            )
            text = (out / "changelog").read_text()
            self.assertIn("Sat, 31 Oct 2026 00:00:00 +0000", text)
            self.assertIn("10.3.0~20261031-41147801272", text)
            deb.generate_rules_file(build.get_package_info("amdrocm-llvm"), out, cfg)
            self.assertIn("gzip -n -9 -f", (out / "rules").read_text())
            self.assertNotEqual(
                controls["template_original_sha256"],
                controls["template_controlled_sha256"],
            )

    def test_tar_manifest_preserves_bytes_metadata_and_rejects_unsafe_duplicates(self):
        def archive(names):
            target = io.BytesIO()
            with tarfile.open(fileobj=target, mode="w") as tar:
                for name in names:
                    entry = tarfile.TarInfo(name)
                    entry.uid, entry.gid, entry.mode, entry.mtime = 12, 34, 0o640, 12345
                    entry.size = 4
                    tar.addfile(entry, io.BytesIO(b"data"))
            target.seek(0)
            return target

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "manifest"
            actual = support.tar_manifest(archive(["x"]), output)
            row = json.loads(output.read_text())
            self.assertEqual(
                (row["uid"], row["gid"], row["mode"], row["mtime"]),
                (12, 34, 0o640, 12345),
            )
            self.assertEqual(actual["payload_bytes"], 4)
            for names, pattern in [
                (["../escape"], "Unsafe"),
                (["x", "x"], "Duplicate"),
            ]:
                with self.assertRaisesRegex(RuntimeError, pattern):
                    support.tar_manifest(archive(names), output)

    def test_capacity_and_owned_removal_fail_before_deletion(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            env = {"GITHUB_WORKSPACE": str(base)}
            with patch.dict(os.environ, env):
                root, work, _, case = full.paths()
                root.mkdir(parents=True)
                state = dict(deadline=time.monotonic() + 10, reserve_bytes=5 * 1024**3)
                inode = list(full.snapshots.SnapshotReset.inode(root))
                with patch.object(
                    shutil,
                    "disk_usage",
                    return_value=shutil._ntuple_diskusage(100, 99, 1),
                ):
                    with self.assertRaisesRegex(RuntimeError, "capacity gate"):
                        full.remove_owned(root, inode, state)
                self.assertTrue(root.exists())
                with self.assertRaisesRegex(RuntimeError, "outside exact"):
                    full.remove_owned(
                        base, list(full.snapshots.SnapshotReset.inode(base)), state
                    )
                with self.assertRaisesRegex(RuntimeError, "inode changed"):
                    full.remove_owned(root, [inode[0], inode[1] + 1], state)
                state["deadline"] = time.monotonic() - 1
                with self.assertRaisesRegex(RuntimeError, "budget"):
                    full.remove_owned(root, inode, state)
                self.assertTrue(root.exists())

    def test_run_scoped_state_rejects_other_run_and_changed_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            env = dict(
                GITHUB_REPOSITORY="ROCm/TheRock",
                GITHUB_RUN_ID="1",
                GITHUB_RUN_ATTEMPT="1",
                GITHUB_SHA="sha",
                GITHUB_WORKSPACE=str(base),
            )
            with patch.dict(os.environ, env):
                output = full.paths()[2]
                output.mkdir(parents=True)
                state = dict(
                    owner=full.owner(),
                    paths=list(map(str, full.paths())),
                    output_inode=list(full.snapshots.SnapshotReset.inode(output)),
                    helper_sha256=support.file_hash(full.SCRIPT),
                )
                full.save_state(state)
                self.assertEqual(full.read_state(), state)
                with patch.dict(os.environ, {"GITHUB_RUN_ID": "other"}):
                    with self.assertRaisesRegex(RuntimeError, "different run"):
                        full.read_state()
                state["paths"][0] = str(base)
                full.save_state(state)
                with self.assertRaisesRegex(RuntimeError, "paths changed"):
                    full.read_state()

    def test_workflow_is_default_off_two_orders_the_rock_only_no_publication(self):
        source = (
            SCRIPTS.parents[2] / ".github/workflows/multi_arch_ci.yml"
        ).read_text()
        job = source.split("\n  profile_package_deb:\n", 1)[1].split(
            "\n  linux_build_and_test:", 1
        )[0]
        for required in [
            "github.repository == 'ROCm/TheRock'",
            "runs-on: aws-linux-scale-rocm-small",
            "order: [AB, BA]",
            "compression-level: 0",
            "retention-days: 7",
            "steps.first_packages.outputs.artifact-id",
            "steps.second_packages.outputs.artifact-id",
        ]:
            self.assertIn(required, job)
        for forbidden in [
            "id-token:",
            "configure_aws",
            "upload_package_repo",
            "upload_repo_setup",
            "Quartz",
            "secrets:",
            "cmake --build",
        ]:
            self.assertNotIn(forbidden, job)
        self.assertIn(
            "profile_package_deb:\n        type: boolean\n        default: false",
            source,
        )
        command = full.fetch_command() if "GITHUB_WORKSPACE" in os.environ else None
        with patch.dict(os.environ, {"GITHUB_WORKSPACE": str(Path.cwd())}):
            command = full.fetch_command()
        self.assertIn("--run-github-repo=ROCm/rockrel", command)
        self.assertEqual(len(full.FAMILIES.split(";")), 17)
        for repository in ["ROCm/rockrel", "other/TheRock"]:
            with patch.dict(
                os.environ, {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": repository}
            ):
                with self.assertRaisesRegex(RuntimeError, "ROCm/TheRock CI"):
                    pre.require_rock_ci()

    def test_stock_component_selection_drives_retained_staging_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "artifacts"
            source = root / "core-amdsmi_lib_generic/core/amdsmi/stage/lib"
            source.mkdir(parents=True)
            (source / "payload").write_bytes(b"x" * 100)
            (root / "core-amdsmi_lib_generic/artifact_manifest.txt").write_text(
                "core/amdsmi/stage\n"
            )
            initial = base / "manifest.jsonl"
            pre.manifest(root, initial, time.monotonic() + 20)
            plan = full.component_plan(root, initial, 1000)
            component = next(
                row
                for row in plan["components"]
                if row["component"] == "amdrocm-amdsmi"
            )
            self.assertEqual(component["retained_source_bytes"], 100)
            self.assertEqual(component["max_current_variant_bytes"], 100)
            self.assertIn(
                "core-amdsmi_lib_generic/core/amdsmi/stage/",
                component["variants"][0]["roots"],
            )
            self.assertGreaterEqual(
                plan["extra_case_headroom_bytes"],
                plan["retained_two_worker_bytes"]
                + plan["two_current_variant_copy_fallback_bytes"]
                + plan["output_estimate_bytes"],
            )

    @unittest.skipUnless(LINUX, "nanosecond metadata and symbolic links")
    def test_refetch_metadata_restore_then_full_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "root"
            root.mkdir()
            (root / "payload").write_bytes(b"pristine")
            os.link(root / "payload", root / "hardlink")
            (root / "link").symlink_to("payload")
            initial_path = base / "initial.jsonl"
            initial = pre.manifest(root, initial_path, time.monotonic() + 20)
            self.assertEqual(
                support.hardlink_groups(root, initial_path), [["hardlink", "payload"]]
            )
            for path in [root, root / "payload", root / "link"]:
                os.utime(path, ns=(1000000000, 1000000000), follow_symlinks=False)
            before = pre.manifest(root, base / "before.jsonl", time.monotonic() + 20)
            self.assertEqual(initial["content_sha256"], before["content_sha256"])
            self.assertNotEqual(initial["mtime_sha256"], before["mtime_sha256"])
            full.restore_refetch_times(root, initial_path)
            after = pre.manifest(root, base / "after.jsonl", time.monotonic() + 20)
            self.assertEqual(initial["mtime_sha256"], after["mtime_sha256"])
            (root / "hardlink").unlink()
            (root / "hardlink").write_bytes(b"pristine")
            self.assertEqual(support.hardlink_groups(root, initial_path), [])

    @unittest.skipUnless(LINUX and shutil.which("zstd"), "Linux compact snapshot")
    def test_compact_checkpoint_reload_preserves_ownership_and_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            env = dict(
                GITHUB_WORKSPACE=str(base),
                GITHUB_REPOSITORY="ROCm/TheRock",
                GITHUB_RUN_ID="1",
                GITHUB_RUN_ATTEMPT="1",
                GITHUB_SHA="sha",
            )
            with patch.dict(os.environ, env):
                root, work, output, _ = full.paths()
                root.mkdir(parents=True)
                output.mkdir(parents=True)
                (root / "payload").write_bytes(b"exact snapshot input")
                initial = pre.manifest(
                    root, output / "pristine-before.jsonl", time.monotonic() + 90
                )
                state = dict(
                    owner=full.owner(),
                    paths=list(map(str, full.paths())),
                    output_inode=list(full.snapshots.SnapshotReset.inode(output)),
                    helper_sha256=support.file_hash(full.SCRIPT),
                    initial=initial,
                    deadline=time.monotonic() + 90,
                    reserve_bytes=5 * 1024**3,
                )
                reset = full.snapshots.SnapshotReset(
                    root,
                    work,
                    output / "snapshot",
                    initial,
                    output / "pristine-before.jsonl",
                    state["deadline"],
                )
                reset.prepare()
                full.save_snapshot(state, reset)
                resumed = full.load_snapshot(full.read_state())
                self.assertEqual(resumed.hash_archive(), reset.archive_sha256)
                resumed.restore(work)
                full.save_snapshot(state, resumed)
                again = full.load_snapshot(full.read_state())
                again.recover_pristine()
                self.assertFalse(work.exists())
                self.assertEqual(
                    (root / "payload").read_bytes(), b"exact snapshot input"
                )
                final = pre.manifest(
                    root, output / "final.jsonl", time.monotonic() + 20
                )
                self.assertEqual(initial["content_sha256"], final["content_sha256"])
                self.assertEqual(initial["mtime_sha256"], final["mtime_sha256"])

    @unittest.skipUnless(DEB_TOOLS, "Linux DEB tools")
    def test_real_deb_complete_manifests_repo_checks_and_payload_mutation(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            base = Path(temporary)
            packages = base / "packages"
            packages.mkdir()
            root = base / "fixture"
            (root / "DEBIAN").mkdir(parents=True)
            (root / "opt/fixture").mkdir(parents=True)
            (root / "DEBIAN/control").write_text(
                "Package: therock-deb-profile-fixture\nVersion: 1.0\nArchitecture: all\nMaintainer: fixture <fixture@example.invalid>\nDescription: tiny manifest fixture\n"
            )
            (root / "DEBIAN/postinst").write_text("#!/bin/sh\nexit 0\n")
            (root / "DEBIAN/postinst").chmod(0o755)
            (root / "DEBIAN/conffiles").write_text("/opt/fixture/config\n")
            (root / "DEBIAN/triggers").write_text("interest-noawait /opt/fixture\n")
            payload = root / "opt/fixture/config"
            payload.write_bytes(b"exact payload\n")
            payload.chmod(0o640)
            (payload.parent / "alias").symlink_to("config")
            self.controls(base / "templates", stack)
            debfile = packages / "therock-deb-profile-fixture_1.0_all.deb"
            subprocess.run(
                ["dpkg-deb", "--root-owner-group", "--build", str(root), str(debfile)],
                check=True,
                capture_output=True,
            )
            first = support.inspect_packages(packages, base / "first")
            original_deb = debfile.read_bytes()
            # Deliberately change every source mtime ABOVE the fixed epoch.
            # dpkg's epoch control must make the complete second DEB identical.
            for entry in [root, *root.rglob("*")]:
                os.utime(
                    entry,
                    ns=((support.EPOCH + 100) * 10**9,) * 2,
                    follow_symlinks=False,
                )
            subprocess.run(
                ["dpkg-deb", "--root-owner-group", "--build", str(root), str(debfile)],
                check=True,
                capture_output=True,
            )
            self.assertEqual(original_deb, debfile.read_bytes())
            control = (base / "first" / debfile.name / "control.jsonl").read_text()
            for name in ["postinst", "conffiles", "triggers"]:
                self.assertIn(name, control)
            data = [
                json.loads(line)
                for line in (base / "first" / debfile.name / "data.jsonl")
                .read_text()
                .splitlines()
            ]
            config = [r for r in data if r["path"].endswith("/config")][0]
            self.assertEqual(config["mode"], 0o640)
            self.assertEqual(config["uid"], 0)
            second = support.inspect_packages(packages, base / "second")
            self.assertEqual(first["inventory"], second["inventory"])
            repo.create_deb_repo(packages, "ci")
            checked = support.verify_repository(packages, first["inventory"])
            self.assertEqual(checked["indexed"], 1)
            release = (packages / "dists/stable/Release").read_text()
            self.assertIn("Date: Sat, 31 Oct 2026 00:00:00 UTC", release)
            compressed = packages / "dists/stable/main/binary-amd64/Packages.gz"
            self.assertEqual(compressed.read_bytes()[4:8], b"\0\0\0\0")
            target = packages / "pool/main" / debfile.name
            with target.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                support.verify_repository(packages, first["inventory"])

    @unittest.skipUnless(LINUX, "POSIX process groups")
    def test_runtime_budget_terminates_subprocess_and_retains_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "runtime budget"):
                support.monitored_run(
                    [sys.executable, "-c", "import time;time.sleep(20)"],
                    base / "log",
                    time.monotonic() + 0.3,
                    0,
                    base,
                )
            command = json.loads((base / "log/command.json").read_text())
            self.assertNotEqual(command["exit_code"], 0)
            self.assertLess(command["wall_s"], 8)
            self.assertGreaterEqual(command["controller_cpu_s"], 0)

    @unittest.skipUnless(LINUX and shutil.which("make"), "Linux make/cp")
    def test_no_fallback_guard_preserves_happy_path_and_rejects_exdev_before_copy(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            base = Path(temporary)
            cases = []
            for name in ["stock", "controlled"]:
                directory = base / name
                (directory / "debian").mkdir(parents=True)
                source = directory / "opt/fixture"
                source.mkdir(parents=True)
                (source / "payload").write_bytes(b"same payload")
                (source / "payload").chmod(0o640)
                os.utime(source / "payload", (support.EPOCH, support.EPOCH))
                (source / "alias").symlink_to("payload")
                cfg = build.PackageConfig(
                    artifacts_dir=base,
                    dest_dir=base,
                    pkg_type="deb",
                    rocm_version="10.3.0",
                    version_suffix="1",
                    install_prefix="/opt/fixture",
                    gfx_arch="",
                )
                if name == "controlled":
                    self.controls(base / "templates", stack)
                deb.generate_rules_file(
                    build.get_package_info("amdrocm-llvm"), directory / "debian", cfg
                )
                result = subprocess.run(
                    ["make", "-f", "debian/rules", "override_dh_install"],
                    cwd=directory,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                copied = directory / "debian/amdrocm-llvm10.3/opt/fixture/payload"
                self.assertEqual(
                    copied.stat().st_ino, (source / "payload").stat().st_ino
                )
                self.assertEqual(copied.read_bytes(), b"same payload")
                info = copied.stat()
                cases.append(
                    (
                        info.st_mode,
                        info.st_uid,
                        info.st_gid,
                        info.st_size,
                        info.st_mtime_ns,
                        (copied.parent / "alias").readlink(),
                    )
                )
            self.assertEqual(cases[0], cases[1])
            capability = support.hardlink_capability(base, base / "capability.json")
            self.assertTrue(capability["same_filesystem"])
            # Force cp to fail with an EXDEV-like status. A fake dh_install would
            # allocate a sentinel if reached; the controlled branch must not run it.
            binary = base / "bin"
            binary.mkdir()
            (binary / "cp").write_text(
                "#!/bin/sh\necho 'Invalid cross-device link' >&2\nexit 18\n"
            )
            sentinel = base / "fallback-allocation"
            (binary / "dh_install").write_text(
                "#!/bin/sh\nprintf copied > '" + str(sentinel) + "'\n"
            )
            for path in binary.iterdir():
                path.chmod(0o755)
            directory = base / "controlled"
            shutil.rmtree(directory / "debian/amdrocm-llvm10.3")
            env = dict(os.environ, PATH=str(binary) + ":" + os.environ["PATH"])
            result = subprocess.run(
                ["make", "-f", "debian/rules", "override_dh_install"],
                cwd=directory,
                env=env,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"THEROCK_DEB_PROFILE_FALLBACK_REJECTED\n", result.stderr)
            self.assertFalse(sentinel.exists())
            self.assertFalse(
                (directory / "debian/amdrocm-llvm10.3/opt/fixture/payload").exists()
            )
            # Empty/metapackage branch still invokes the unchanged dh_install.
            shutil.rmtree(directory / "opt")
            result = subprocess.run(
                ["make", "-f", "debian/rules", "override_dh_install"],
                cwd=directory,
                env=env,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(sentinel.read_text(), "copied")


if __name__ == "__main__":
    unittest.main(verbosity=2)
