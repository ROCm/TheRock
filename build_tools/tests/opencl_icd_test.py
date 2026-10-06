# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for OpenCL ICD packaging.

Covers the OpenCL runtime symlinks, relocatable ICD registration, and the
intentional artifact exclusions associated with issue #7667.
"""

import os
import sys
import tempfile
import tomllib
import types
import unittest
from pathlib import Path

import build_python_packages

THEROCK_DIR = Path(__file__).resolve().parent.parent.parent
CORE_OCL_DESCRIPTOR = THEROCK_DIR / "core" / "artifact-core-ocl.toml"

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


def make_tree() -> Path:
    root = Path(tempfile.mkdtemp())
    (root / "lib" / "opencl").mkdir(parents=True)
    (root / "etc" / "OpenCL" / "vendors").mkdir(parents=True)
    return root


class EnsureOpenCLLibrarySymlinksTest(unittest.TestCase):
    def ensure(self, root: Path):
        # Only .platform_dir is consulted.
        build_python_packages.ensure_opencl_library_symlinks(
            types.SimpleNamespace(platform_dir=root)
        )

    def test_creates_unversioned_names(self):
        root = make_tree()
        (root / "lib" / "opencl" / "libamdocl64.so.2").write_text("ELF")
        (root / "lib" / "libOpenCL.so.1").write_text("ELF")

        self.ensure(root)

        amdocl = root / "lib" / "opencl" / "libamdocl64.so"
        loader = root / "lib" / "libOpenCL.so"
        self.assertTrue(amdocl.is_symlink())
        self.assertEqual(os.readlink(amdocl), "libamdocl64.so.2")
        self.assertTrue(loader.is_symlink())
        self.assertEqual(os.readlink(loader), "libOpenCL.so.1")

    def test_is_idempotent(self):
        root = make_tree()
        (root / "lib" / "opencl" / "libamdocl64.so.2").write_text("ELF")

        self.ensure(root)
        self.ensure(root)

        self.assertEqual(
            os.readlink(root / "lib" / "opencl" / "libamdocl64.so"), "libamdocl64.so.2"
        )

    def test_leaves_existing_file_alone(self):
        root = make_tree()
        (root / "lib" / "opencl" / "libamdocl64.so.2").write_text("ELF")
        real = root / "lib" / "opencl" / "libamdocl64.so"
        real.write_text("a real file, not a link")

        self.ensure(root)

        self.assertFalse(real.is_symlink())
        self.assertEqual(real.read_text(), "a real file, not a link")

    def test_tolerates_missing_opencl_dir(self):
        root = Path(tempfile.mkdtemp())
        self.ensure(root)  # must not raise

    def test_ignores_stray_trailing_dot_entry(self):
        # Some builds stage a stray "libamdocl64.so." alongside the real files;
        # it must not be mistaken for a versioned library.
        root = make_tree()
        (root / "lib" / "opencl" / "libamdocl64.so.").write_text("")

        self.ensure(root)

        self.assertFalse((root / "lib" / "opencl" / "libamdocl64.so").exists())


class RegisterOpenCLICDTest(unittest.TestCase):
    def write_icd(self, root: Path, contents: str = "libamdocl64.so\n") -> Path:
        icd = root / "etc" / "OpenCL" / "vendors" / "amdocl64.icd"
        icd.write_text(contents)
        return icd

    def test_rewrites_bare_name_to_absolute_soname_path(self):
        root = make_tree()
        lib = root / "lib" / "opencl" / "libamdocl64.so.2"
        lib.write_text("ELF")
        (root / "lib" / "opencl" / "libamdocl64.so").symlink_to("libamdocl64.so.2")
        icd = self.write_icd(root)

        vendors_dir = _devel.register_opencl_icd(root)

        self.assertEqual(vendors_dir, root / "etc" / "OpenCL" / "vendors")
        self.assertEqual(icd.read_text(), f"{lib}\n")

    def test_replaces_symlink_without_touching_its_target(self):
        """Replace a devel-tree ICD symlink without modifying its target."""
        root = make_tree()
        (root / "lib" / "opencl" / "libamdocl64.so.2").write_text("ELF")
        core_icd = root / "core-amdocl64.icd"
        core_icd.write_text("libamdocl64.so\n")
        icd = root / "etc" / "OpenCL" / "vendors" / "amdocl64.icd"
        icd.symlink_to(os.path.relpath(core_icd, icd.parent))

        _devel.register_opencl_icd(root)

        self.assertFalse(icd.is_symlink())
        self.assertEqual(core_icd.read_text(), "libamdocl64.so\n")

    def test_is_idempotent(self):
        root = make_tree()
        (root / "lib" / "opencl" / "libamdocl64.so.2").write_text("ELF")
        icd = self.write_icd(root)

        _devel.register_opencl_icd(root)
        first = icd.read_text()
        _devel.register_opencl_icd(root)

        self.assertEqual(icd.read_text(), first)

    def test_prefers_highest_soname_version(self):
        root = make_tree()
        (root / "lib" / "opencl" / "libamdocl64.so.2").write_text("ELF")
        (root / "lib" / "opencl" / "libamdocl64.so.10").write_text("ELF")
        icd = self.write_icd(root)

        _devel.register_opencl_icd(root)

        self.assertEqual(Path(icd.read_text().strip()).name, "libamdocl64.so.10")

    def test_falls_back_to_unversioned_library(self):
        root = make_tree()
        lib = root / "lib" / "opencl" / "libamdocl64.so"
        lib.write_text("ELF")
        icd = self.write_icd(root)

        _devel.register_opencl_icd(root)

        self.assertEqual(icd.read_text(), f"{lib}\n")

    def test_returns_none_without_runtime(self):
        root = make_tree()
        self.write_icd(root)
        self.assertIsNone(_devel.register_opencl_icd(root))

    def test_returns_none_without_icd(self):
        root = make_tree()
        (root / "lib" / "opencl" / "libamdocl64.so.2").write_text("ELF")
        self.assertIsNone(_devel.register_opencl_icd(root))


class OpenCLArtifactDescriptorTest(unittest.TestCase):
    """Verify intentional exclusions in core/artifact-core-ocl.toml."""

    # Mirrors the ocl-clr install tree, limited to the paths under test.
    STAGE_FILES = [
        "bin/clinfo",
        "etc/OpenCL/vendors/amdocl64.icd",
        "etc/ld.so.conf.d/10-rocm-opencl.conf",
        "include/CL/opencl.h",
        "lib/opencl/libamdocl64.so",
        "lib/opencl/libamdocl64.so.2",
        "share/doc/opencl/LICENSE.md",
        "share/opencl/ocltst/liboclperf.so",
        "share/opencl/ocltst/liboclruntime.so",
    ]

    def setUp(self):
        from _therock_utils.artifact_builder import ArtifactDescriptor, ComponentScanner

        self.ArtifactDescriptor = ArtifactDescriptor
        self.ComponentScanner = ComponentScanner
        self.record = tomllib.loads(CORE_OCL_DESCRIPTOR.read_text())

        self.root = Path(tempfile.mkdtemp())
        stage = self.root / "core" / "ocl-clr" / "stage"
        for relpath in self.STAGE_FILES:
            path = stage / relpath
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x")

    def scan(self, record: dict):
        descriptor = self.ArtifactDescriptor(record, artifact_name="core-ocl")
        return self.ComponentScanner(self.root, descriptor)

    def component_of(self, scanner, relpath: str) -> str | None:
        for name, contents in scanner.components.items():
            for pm in contents.basedir_contents.values():
                if relpath in pm.all:
                    return name
        return None

    def test_ld_so_conf_fragment_ships_in_no_component(self):
        scanner = self.scan(self.record)
        self.assertIsNone(
            self.component_of(scanner, "etc/ld.so.conf.d/10-rocm-opencl.conf")
        )

    def test_dropping_the_fragment_is_declared_not_accidental(self):
        """Require the ld.so.conf.d omission to be explicitly declared."""
        self.scan(self.record).verify()  # declared: no error

        undeclared = tomllib.loads(CORE_OCL_DESCRIPTOR.read_text())
        del undeclared["options"]
        with self.assertRaises(ValueError) as raised:
            self.scan(undeclared).verify()
        self.assertIn("10-rocm-opencl.conf", str(raised.exception))

    def test_opencl_runtime_and_icd_ship_in_lib(self):
        scanner = self.scan(self.record)
        for relpath in (
            "lib/opencl/libamdocl64.so",
            "lib/opencl/libamdocl64.so.2",
            "etc/OpenCL/vendors/amdocl64.icd",
        ):
            self.assertEqual(self.component_of(scanner, relpath), "lib", relpath)

    def test_ocltst_harness_libraries_ship_in_test_not_lib(self):
        scanner = self.scan(self.record)
        for relpath in (
            "share/opencl/ocltst/liboclperf.so",
            "share/opencl/ocltst/liboclruntime.so",
        ):
            self.assertEqual(self.component_of(scanner, relpath), "test", relpath)


if __name__ == "__main__":
    unittest.main()
