# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for rocm_sdk._devel._reconcile_device_links().

These exercise the install-time reconcile that mirrors per-ISA device files
(.kpack archives, kernel DBs, per-arch .so) from the rocm-sdk-libraries overlay
into the expanded rocm-sdk-devel tree, driven by each installed
`rocm-sdk-device-*` wheel's `_devel_links` manifest.

The reconcile is unit-tested against a synthetic site-packages directory: fake
`.dist-info` directories (METADATA + RECORD) are created so importlib.metadata
discovers them via `distributions(path=[site])`, alongside a libraries overlay
holding the real device files and the manifests.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path, PurePosixPath

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


VERSION = "0.0.1.test"
LIBS_NAME = "_rocm_sdk_libraries_test"
DEVEL_NAME = "_rocm_sdk_devel_test"


def _target_for(relpath: str) -> str:
    """Relative hardlink target from the devel tree into the libraries overlay."""
    n = len(PurePosixPath(relpath).parts)
    return "/".join([".."] * n + [LIBS_NAME, relpath])


class ReconcileDeviceLinksTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.site = Path(self._tmp.name)
        # The expanded (generic) devel platform dir must already exist.
        self.devel_dir = self.site / DEVEL_NAME
        self.devel_dir.mkdir(parents=True)
        (self.devel_dir / "__init__.py").touch()

    def tearDown(self):
        self._tmp.cleanup()

    # ----- helpers --------------------------------------------------------

    def _add_device_wheel(
        self, target_family: str, files: dict[str, str], version: str = VERSION
    ):
        """Materialize an installed device wheel into the synthetic site-packages.

        Writes the device files into the libraries overlay, a `_devel_links`
        manifest, and a `.dist-info` with METADATA + RECORD listing all of them.
        """
        # Device files overlay into the libraries package dir.
        for relpath, content in files.items():
            p = self.site / LIBS_NAME / relpath
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)

        # Manifest, named per target so device wheels don't collide.
        manifest_relpath = f"{LIBS_NAME}/.devel_links/{target_family}.json"
        manifest = {
            "version": version,
            "links": [
                {"relpath": relpath, "target": _target_for(relpath)}
                for relpath in files
            ],
        }
        manifest_path = self.site / manifest_relpath
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest))

        # .dist-info with METADATA and a RECORD listing the shipped files.
        dist_info = self.site / f"rocm_sdk_device_{target_family}-{version}.dist-info"
        dist_info.mkdir(parents=True, exist_ok=True)
        (dist_info / "METADATA").write_text(
            "Metadata-Version: 2.1\n"
            f"Name: rocm-sdk-device-{target_family}\n"
            f"Version: {version}\n"
        )
        record_lines = [f"{LIBS_NAME}/{r},," for r in files]
        record_lines.append(f"{manifest_relpath},,")
        record_lines.append(
            f"rocm_sdk_device_{target_family}-{version}.dist-info/METADATA,,"
        )
        record_lines.append(
            f"rocm_sdk_device_{target_family}-{version}.dist-info/RECORD,,"
        )
        (dist_info / "RECORD").write_text("\n".join(record_lines) + "\n")
        return dist_info / "RECORD"

    def _record_paths(self, record_path: Path) -> list[str]:
        return [
            line.split(",", 1)[0]
            for line in record_path.read_text().splitlines()
            if line.strip()
        ]

    def _reconcile(self) -> int:
        return _devel._reconcile_device_links(self.site, self.devel_dir, VERSION)

    # ----- tests ----------------------------------------------------------

    def test_links_created_and_recorded(self):
        record = self._add_device_wheel(
            "gfx942",
            {
                ".kpack/blas_lib_gfx942.kpack": "kpack data",
                "lib/rocblas/library/Foo_gfx942.co": "kernel object",
            },
        )

        created = self._reconcile()
        self.assertEqual(created, 2)

        for relpath in (
            ".kpack/blas_lib_gfx942.kpack",
            "lib/rocblas/library/Foo_gfx942.co",
        ):
            devel_file = self.devel_dir / relpath
            libs_file = self.site / LIBS_NAME / relpath
            self.assertTrue(devel_file.is_file(), f"{relpath} not linked into devel")
            self.assertTrue(
                devel_file.samefile(libs_file),
                f"{relpath} must be a hardlink to the libraries overlay file",
            )
            # The created devel path must be recorded in the device wheel RECORD
            # so `pip uninstall` removes it.
            self.assertIn(f"{DEVEL_NAME}/{relpath}", self._record_paths(record))

    def test_idempotent_no_duplicate_records(self):
        record = self._add_device_wheel(
            "gfx942", {".kpack/blas_lib_gfx942.kpack": "kpack data"}
        )

        self.assertEqual(self._reconcile(), 1)
        before = record.read_text()
        # Second run creates nothing new and must not duplicate RECORD lines.
        self.assertEqual(self._reconcile(), 0)
        after = record.read_text()
        self.assertEqual(before, after)
        paths = self._record_paths(record)
        self.assertEqual(len(paths), len(set(paths)), "duplicate RECORD entries")

    def test_second_arch_added_later(self):
        self._add_device_wheel(
            "gfx950", {".kpack/blas_lib_gfx950.kpack": "gfx950 kpack"}
        )
        cached_site_mtime = self.site.stat().st_mtime
        self.assertEqual(self._reconcile(), 1)

        # gfx942 installed afterwards: re-running reconcile links only gfx942.
        self._add_device_wheel(
            "gfx942", {".kpack/blas_lib_gfx942.kpack": "gfx942 kpack"}
        )
        # Simulate a filesystem with coarse directory mtime resolution, where
        # importlib.metadata's path cache would not otherwise notice the new dist.
        os.utime(self.site, (cached_site_mtime, cached_site_mtime))
        self.assertEqual(self._reconcile(), 1)

        self.assertTrue((self.devel_dir / ".kpack/blas_lib_gfx950.kpack").is_file())
        self.assertTrue((self.devel_dir / ".kpack/blas_lib_gfx942.kpack").is_file())

    def test_version_mismatch_skipped(self):
        self._add_device_wheel(
            "gfx942",
            {".kpack/blas_lib_gfx942.kpack": "kpack data"},
            version="9.9.9",
        )
        # Device wheel version does not match the SDK version -> skip entirely.
        self.assertEqual(self._reconcile(), 0)
        self.assertFalse((self.devel_dir / ".kpack/blas_lib_gfx942.kpack").exists())

    def test_prune_semantics_record_lists_links(self):
        record = self._add_device_wheel(
            "gfx942", {".kpack/blas_lib_gfx942.kpack": "kpack data"}
        )
        self._reconcile()

        # Simulate `pip uninstall` by removing every path the RECORD owns.
        for rel in self._record_paths(record):
            p = self.site / rel
            if p.is_file():
                p.unlink()

        self.assertFalse(
            (self.devel_dir / ".kpack/blas_lib_gfx942.kpack").exists(),
            "devel link must be RECORD-owned so pip uninstall removes it",
        )

    def test_record_repaired_when_link_exists_but_unrecorded(self):
        # Simulate an interrupted prior run: the hardlink was created but RECORD
        # was never updated. The fast path must not short-circuit on link
        # presence alone - it must repair the missing RECORD entry.
        record = self._add_device_wheel(
            "gfx942", {".kpack/blas_lib_gfx942.kpack": "kpack data"}
        )
        relpath = ".kpack/blas_lib_gfx942.kpack"
        rec_name = f"{DEVEL_NAME}/{relpath}"
        dest = self.devel_dir / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.link(self.site / LIBS_NAME / relpath, dest)
        self.assertNotIn(rec_name, self._record_paths(record))

        self.assertEqual(self._reconcile(), 0)  # no new links, only RECORD repair
        paths = self._record_paths(record)
        self.assertIn(rec_name, paths)
        self.assertEqual(len(paths), len(set(paths)), "duplicate RECORD entries")

    def test_existing_duplicate_record_entry_is_removed(self):
        # A duplicate RECORD row (e.g. from a botched prior write) must be
        # repaired even when there is nothing new to link.
        relpath = ".kpack/blas_lib_gfx942.kpack"
        record = self._add_device_wheel("gfx942", {relpath: "kpack data"})
        self.assertEqual(self._reconcile(), 1)

        record_name = f"{DEVEL_NAME}/{relpath}"
        with record.open("a") as f:
            f.write(f"{record_name},,\n")
        self.assertEqual(self._record_paths(record).count(record_name), 2)

        self.assertEqual(self._reconcile(), 0)
        self.assertEqual(self._record_paths(record).count(record_name), 1)

    def test_symlink_dest_replaced_with_hardlink(self):
        # A symlink that resolves to the right file must still be replaced by a
        # true hardlink (the reconciler guarantees hardlinks).
        record = self._add_device_wheel(
            "gfx942", {".kpack/blas_lib_gfx942.kpack": "kpack data"}
        )
        relpath = ".kpack/blas_lib_gfx942.kpack"
        dest = self.devel_dir / relpath
        src = self.site / LIBS_NAME / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            dest.symlink_to(os.path.relpath(src, dest.parent))
        except OSError as e:
            self.skipTest(f"cannot create symlink on this platform: {e}")
        self.assertTrue(dest.is_symlink())

        self._reconcile()
        self.assertFalse(dest.is_symlink(), "symlink dest must be replaced")
        self.assertTrue(
            dest.samefile(src), "dest must be a hardlink to the libraries file"
        )
        self.assertIn(f"{DEVEL_NAME}/{relpath}", self._record_paths(record))

    def test_stale_file_at_dest_is_overwritten(self):
        # Policy: manifest-driven device relpaths are authoritative, so a stale
        # unrelated file at a device relpath is overwritten with the correct
        # hardlink. (Device relpaths are arch-specific and assumed collision-free
        # with generic devel content.)
        self._add_device_wheel("gfx942", {".kpack/blas_lib_gfx942.kpack": "real kpack"})
        relpath = ".kpack/blas_lib_gfx942.kpack"
        dest = self.devel_dir / relpath
        src = self.site / LIBS_NAME / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("stale unrelated content")
        self.assertFalse(dest.samefile(src))

        self._reconcile()
        self.assertTrue(dest.samefile(src))
        self.assertEqual(dest.read_text(), "real kpack")


if __name__ == "__main__":
    unittest.main()
