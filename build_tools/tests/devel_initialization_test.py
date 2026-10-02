# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for initialization of the generic devel package."""

import csv
import json
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import unittest

# Import the runtime module straight from the package template source tree.
ROCM_SDK_SRC = (
    Path(__file__).resolve().parent.parent
    / "packaging"
    / "python"
    / "templates"
    / "rocm"
    / "src"
)
sys.path.insert(0, os.fspath(ROCM_SDK_SRC))

from rocm_sdk import _devel  # noqa: E402
from rocm_sdk import _dist_info as di  # noqa: E402


class DevelInitializationTest(unittest.TestCase):
    """Test explicit and concurrent generic devel initialization."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.site = Path(self._tmp.name)
        self.pure_dir = self.site / "rocm_sdk_devel"
        devel_py_pkg_name = di.ALL_PACKAGES["devel"].get_py_package_name()
        self.platform_dir = self.site / devel_py_pkg_name
        self.libraries_dir = self.site / "_rocm_sdk_libraries_test"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # ----- helpers --------------------------------------------------------

    def _create_link_manifest(
        self, link_count: int
    ) -> tuple[Path, list[dict[str, str]], set[str]]:
        """Create disposable link inputs and their initial RECORD."""
        # Create generic link targets.
        relpaths = [f"lib/component-{i}/libexample-{i}.so" for i in range(link_count)]
        links: list[dict[str, str]] = []
        for relpath in relpaths:
            target = self.libraries_dir / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f"runtime payload for {relpath}")
            parent_count = len(PurePosixPath(relpath).parts)
            target_name = "/".join(
                [".."] * parent_count + [self.libraries_dir.name, relpath]
            )
            links.append({"relpath": relpath, "target": target_name})

        # Create the link manifest but leave the package uninitialized.
        manifest_path = self.pure_dir / ".devel_links/devel.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(
            json.dumps({"version": di.__version__, "links": links})
        )

        # Create a RECORD containing one wheel-owned entry.
        dist_info_dir = self.site / f"rocm_sdk_devel-{di.__version__}.dist-info"
        dist_info_dir.mkdir()
        record_path = dist_info_dir / "RECORD"
        manifest_record_name = manifest_path.relative_to(self.site).as_posix()
        record_path.write_text(f"{manifest_record_name},sha256=test,1\n")
        return record_path, links, {manifest_record_name}

    def _create_installed_package(self) -> tuple[Path, dict[str, str]]:
        """Create a disposable devel installation for public CLI tests."""
        record_path, [link], _ = self._create_link_manifest(link_count=1)
        # Provide the installed metadata used by the public init command.
        init_path = self.pure_dir / "__init__.py"
        init_path.touch()
        metadata_path = record_path.parent / "METADATA"
        metadata_path.write_text(
            "Metadata-Version: 2.1\n"
            "Name: rocm-sdk-devel\n"
            f"Version: {di.__version__}\n"
        )
        top_level_path = record_path.parent / "top_level.txt"
        top_level_path.write_text(f"{self.pure_dir.name}\n")
        record_path.write_text(
            record_path.read_text()
            + f"{init_path.relative_to(self.site).as_posix()},sha256=test,1\n"
            + f"{metadata_path.relative_to(self.site).as_posix()},sha256=test,1\n"
            + f"{top_level_path.relative_to(self.site).as_posix()},sha256=test,1\n"
            + f"{record_path.relative_to(self.site).as_posix()},,\n"
        )

        return record_path, link

    def _run_init(self) -> subprocess.CompletedProcess[str]:
        """Run public SDK initialization against the disposable installation."""
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [os.fspath(self.site), os.fspath(ROCM_SDK_SRC)]
        )
        command = [sys.executable, "-m", "rocm_sdk", "init", "--quiet"]
        return subprocess.run(
            command,
            cwd=Path(__file__).resolve().parent.parent,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    # ----- tests ----------------------------------------------------------

    def test_cli_init_rejects_mismatched_manifest_version(self) -> None:
        """Reject a mismatched manifest without changes, then retry after repair."""
        record_path, link = self._create_installed_package()
        manifest = self.pure_dir / ".devel_links/devel.json"
        manifest.write_text(json.dumps({"version": "0.0.0", "links": [link]}))
        original_record = record_path.read_bytes()

        completed = self._run_init()

        self.assertNotEqual(completed.returncode, 0, msg=completed.stdout)
        self.assertIn("does not match installed ROCm version", completed.stdout)
        self.assertEqual(record_path.read_bytes(), original_record)
        self.assertFalse((self.pure_dir / di.DEVEL_INITIALIZED).exists())
        self.assertFalse(self.platform_dir.exists())

        manifest.write_text(json.dumps({"version": di.__version__, "links": [link]}))
        completed = self._run_init()
        self.assertEqual(completed.returncode, 0, msg=completed.stdout)
        self.assertTrue((self.pure_dir / di.DEVEL_INITIALIZED).is_file())
        dest = self.platform_dir / link["relpath"]
        self.assertTrue(dest.samefile(dest.parent / link["target"]))

    def test_cli_init_rejects_wheel_owned_link_destination(self) -> None:
        """Reject a link collision without changing the wheel-owned file or RECORD."""
        record_path, link = self._create_installed_package()
        dest = self.platform_dir / link["relpath"]
        dest.parent.mkdir(parents=True)
        dest.write_text("wheel-owned payload")
        with record_path.open("a") as record_file:
            record_file.write(
                f"{dest.relative_to(self.site).as_posix()},sha256=test,{dest.stat().st_size}\n"
            )
        original_record = record_path.read_bytes()

        completed = self._run_init()

        self.assertNotEqual(completed.returncode, 0, msg=completed.stdout)
        self.assertIn("path is owned by the wheel", completed.stdout)
        self.assertEqual(dest.read_text(), "wheel-owned payload")
        self.assertEqual(record_path.read_bytes(), original_record)
        self.assertFalse((self.pure_dir / di.DEVEL_INITIALIZED).exists())

    def test_cli_init_reruns_completed_initialization(self) -> None:
        """Explicit init repairs links despite an existing completion marker."""
        _, link = self._create_installed_package()
        (self.pure_dir / di.DEVEL_INITIALIZED).touch()

        completed = self._run_init()
        self.assertEqual(completed.returncode, 0, msg=completed.stdout)

        dest = self.platform_dir / link["relpath"]
        target = dest.parent / link["target"]
        self.assertTrue(dest.samefile(target), "rocm-sdk init did not repair the link")

    def test_concurrent_initialization_produces_complete_installation(self) -> None:
        """Multiple initializers leave one complete, recorded link tree.

        This is a basic multi-process sanity check. Process scheduling may
        serialize the initializers, so the test does not guarantee contention.
        """
        # Use enough links for independently started processes to overlap.
        link_count = 64
        initializer_count = 4
        # Bound child-process cleanup so a failed test cannot hang the suite.
        process_timeout_seconds = 30
        record_path, links, wheel_owned_path_names = self._create_link_manifest(
            link_count=link_count
        )
        context = multiprocessing.get_context("spawn")

        # Start every initializer before waiting for any of them.
        processes = [
            context.Process(
                target=_devel._initialize_devel_links,
                kwargs={
                    "site_lib_path": self.site,
                    "rocm_sdk_devel_path": self.pure_dir,
                    "record_path": record_path,
                    "wheel_owned_path_names": wheel_owned_path_names,
                },
            )
            for _ in range(initializer_count)
        ]
        for process in processes:
            process.start()
        try:
            for process in processes:
                process.join(process_timeout_seconds)
                self.assertIsNotNone(
                    process.exitcode, "concurrent devel initialization did not complete"
                )
        finally:
            for process in processes:
                process.kill()
                process.join()

        # Windows uses a nonblocking lock, so overlapping workers may fail.
        accepted_exit_codes = {0, 1} if sys.platform == "win32" else {0}
        for process in processes:
            self.assertIn(process.exitcode, accepted_exit_codes)

        # Inspect the completed initialization directly.
        marker_path = self.pure_dir / di.DEVEL_INITIALIZED
        self.assertTrue(marker_path.is_file(), "initialization marker is missing")
        for link in links:
            dest = self.platform_dir / link["relpath"]
            target = dest.parent / link["target"]
            self.assertTrue(
                dest.samefile(target),
                f"generated link does not resolve to its target: {dest}",
            )

        # Verify generated paths are recorded once and wheel ownership is preserved.
        with record_path.open(newline="") as record_file:
            record_rows = list(csv.reader(record_file))
        record_names = [row[0] for row in record_rows]
        expected_generated_names = [
            f"{self.platform_dir.name}/{link['relpath']}" for link in links
        ] + [
            f"{self.pure_dir.name}/{di.DEVEL_INITIALIZED.as_posix()}",
            f"{self.pure_dir.name}/.devel_links/devel.lock",
        ]
        for name in expected_generated_names:
            self.assertIn([name, "", ""], record_rows)
        self.assertEqual(
            len(record_names), len(set(record_names)), "RECORD contains duplicate paths"
        )
        manifest_record_name = (
            (self.pure_dir / ".devel_links/devel.json")
            .relative_to(self.site)
            .as_posix()
        )
        self.assertIn([manifest_record_name, "sha256=test", "1"], record_rows)


if __name__ == "__main__":
    unittest.main()
