# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for OpenCL ICD packaging.

Covers the ICD loader reaching rocm-sdk-core, relocatable ICD registration,
and the intentional artifact routing associated with issue #7667.
"""

import os
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

import build_python_packages

from _therock_utils.artifacts import ArtifactName

THEROCK_DIR = Path(__file__).resolve().parent.parent.parent
CORE_OCL_DESCRIPTOR = THEROCK_DIR / "core" / "artifact-core-ocl.toml"
CORE_OCL_ICD_DESCRIPTOR = THEROCK_DIR / "core" / "artifact-core-ocl-icd.toml"

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


class CoreArtifactFilterTest(unittest.TestCase):
    """rocm-sdk-core must carry the ICD loader, not just the vendor runtime.

    Without `core-ocl-icd` the wheel ships `lib/opencl/libamdocl64.so.2` and no
    `libOpenCL.so` at all, so nothing can load the vendor runtime unless the
    host happens to provide a loader. That was the state of every published
    ROCm 10.0.0 wheel (issue #7667).
    """

    def accepts(self, name: str, component: str) -> bool:
        return build_python_packages.core_artifact_filter(
            ArtifactName(name, component, "generic")
        )

    def test_includes_opencl_icd_loader(self):
        self.assertTrue(self.accepts("core-ocl-icd", "lib"))

    def test_includes_opencl_vendor_runtime(self):
        self.assertTrue(self.accepts("core-ocl", "lib"))

    def test_excludes_loader_development_and_test_components(self):
        for component in ("dev", "doc", "dbg", "test"):
            self.assertFalse(self.accepts("core-ocl-icd", component), component)

    def test_artifact_name_parsing_keeps_icd_distinct_from_core_ocl(self):
        """`core-ocl` must not be read as a prefix match for `core-ocl-icd`.

        The artifact regex splits on '_', so the two names are distinct
        entries; a substring-style filter would silently conflate them.
        """
        parsed = ArtifactName.from_filename("core-ocl-icd_lib_generic.tar.zst")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.name, "core-ocl-icd")
        self.assertEqual(parsed.component, "lib")


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
        """Replace a devel-tree ICD symlink without modifying its target.

        In an expanded devel tree this path is a symlink into the sibling
        rocm-sdk-core package. Writing through it would mutate another
        distribution's installed files and invalidate its RECORD.
        """
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

    @unittest.skipIf(
        sys.platform == "win32",
        "Win32 strips trailing dots from filenames, so the stray entry cannot be "
        "created distinctly from the real unversioned name.",
    )
    def test_ignores_stray_trailing_dot_entry(self):
        """`libamdocl64.so.` is the real file in a dist tree; `.so.2` links to it.

        It must never be written into the ICD, because the loader would then
        hold a name that is not a SONAME.
        """
        root = make_tree()
        (root / "lib" / "opencl" / "libamdocl64.so.").write_text("ELF")
        (root / "lib" / "opencl" / "libamdocl64.so.2").write_text("ELF")
        icd = self.write_icd(root)

        _devel.register_opencl_icd(root)

        self.assertEqual(Path(icd.read_text().strip()).name, "libamdocl64.so.2")

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


class OpenCLICDVendorsEnvValueTest(unittest.TestCase):
    """OCL_ICD_VENDORS must carry a trailing separator.

    The Khronos loader concatenates the value with each ".icd" file name
    rather than joining paths, so "<dir>" yields "<dir>amdocl64.icd" and
    enumerates zero platforms with no error. Verified against the shipped
    libOpenCL.so.1: `OCL_ICD_VENDORS=<dir>` reports -1001 while
    `OCL_ICD_VENDORS=<dir>/` reports the AMD platform.
    """

    def test_appends_trailing_separator(self):
        root = make_tree()
        (root / "lib" / "opencl" / "libamdocl64.so.2").write_text("ELF")
        (root / "etc" / "OpenCL" / "vendors" / "amdocl64.icd").write_text(
            "libamdocl64.so\n"
        )

        vendors_dir = _devel.register_opencl_icd(root)
        value = _devel.opencl_icd_vendors_env_value(vendors_dir)

        self.assertTrue(value.endswith(os.sep), value)
        self.assertEqual(value, f"{vendors_dir}{os.sep}")

    def test_concatenation_yields_the_icd_path(self):
        """Mirrors what the loader actually does with the value."""
        vendors_dir = Path("/opt/example/etc/OpenCL/vendors")
        value = _devel.opencl_icd_vendors_env_value(vendors_dir)
        self.assertEqual(Path(value + "amdocl64.icd"), vendors_dir / "amdocl64.icd")


class _DescriptorTestCase(unittest.TestCase):
    """Scans a real artifact descriptor against a synthetic stage tree."""

    DESCRIPTOR: Path
    ARTIFACT_NAME: str
    STAGE_SUBDIR: str
    STAGE_FILES: list[str] = []

    def setUp(self):
        from _therock_utils.artifact_builder import ArtifactDescriptor, ComponentScanner

        self.ArtifactDescriptor = ArtifactDescriptor
        self.ComponentScanner = ComponentScanner
        self.record = tomllib.loads(self.DESCRIPTOR.read_text())

        self.root = Path(tempfile.mkdtemp())
        stage = self.root / self.STAGE_SUBDIR
        for relpath in self.STAGE_FILES:
            path = stage / relpath
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x")

    def scan(self):
        descriptor = self.ArtifactDescriptor(
            self.record, artifact_name=self.ARTIFACT_NAME
        )
        return self.ComponentScanner(self.root, descriptor)

    def component_of(self, scanner, relpath: str) -> str | None:
        for name, contents in scanner.components.items():
            for pm in contents.basedir_contents.values():
                if relpath in pm.all:
                    return name
        return None


class OpenCLLoaderArtifactDescriptorTest(_DescriptorTestCase):
    """The loader lives in core-ocl-icd, separately from the vendor runtime."""

    DESCRIPTOR = CORE_OCL_ICD_DESCRIPTOR
    ARTIFACT_NAME = "core-ocl-icd"
    STAGE_SUBDIR = "core/ocl-icd/stage"
    STAGE_FILES = [
        "lib/libOpenCL.so",
        "lib/libOpenCL.so.1",
        "lib/libOpenCL.so.1.2",
    ]

    def test_loader_ships_in_lib(self):
        """`libOpenCL.so.1` is the SONAME consumers dlopen.

        populate_runtime_files() keeps the SONAME-named file, so this is the
        name that must reach rocm-sdk-core. Asserting the whole family guards
        against a glob that would pass on the wrong member.
        """
        scanner = self.scan()
        for relpath in self.STAGE_FILES:
            self.assertEqual(self.component_of(scanner, relpath), "lib", relpath)


class OpenCLArtifactDescriptorTest(_DescriptorTestCase):
    """Verify intentional exclusions in core/artifact-core-ocl.toml."""

    DESCRIPTOR = CORE_OCL_DESCRIPTOR
    ARTIFACT_NAME = "core-ocl"
    STAGE_SUBDIR = "core/ocl-clr/stage"
    # Mirrors the ocl-clr install tree, limited to the paths under test.
    STAGE_FILES = [
        "bin/clinfo",
        "etc/OpenCL/vendors/amdocl64.icd",
        "etc/ld.so.conf.d/10-rocm-opencl.conf",
        "include/CL/opencl.h",
        "lib/liboclperf.so",
        "lib/liboclruntime.so",
        "lib/opencl/libamdocl64.so",
        "lib/opencl/libamdocl64.so.2",
        "share/doc/opencl/LICENSE.md",
    ]

    def test_system_config_files_ship_in_sys(self):
        """The deb/rpm flow rewrites these in place, so they must be shipped.

        rename_opencl_etc_files() points the fragment at the real runtime dir
        and gives both files versioned names; dropping either from the artifact
        would silently disable ICD registration for native packages.
        """
        scanner = self.scan()
        for relpath in (
            "etc/ld.so.conf.d/10-rocm-opencl.conf",
            "etc/OpenCL/vendors/amdocl64.icd",
        ):
            self.assertEqual(self.component_of(scanner, relpath), "sys", relpath)

    def test_opencl_runtime_ships_in_lib(self):
        scanner = self.scan()
        for relpath in (
            "lib/opencl/libamdocl64.so",
            "lib/opencl/libamdocl64.so.2",
        ):
            self.assertEqual(self.component_of(scanner, relpath), "lib", relpath)

    def test_ocltst_harness_libraries_ship_in_test_not_lib(self):
        """These match the 'lib' default "**/*.so", which is applied first."""
        scanner = self.scan()
        for relpath in (
            "lib/liboclperf.so",
            "lib/liboclruntime.so",
        ):
            self.assertEqual(self.component_of(scanner, relpath), "test", relpath)


if __name__ == "__main__":
    unittest.main()
