# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for _therock_utils/py_packaging.py.

These tests cover:
  - PopulatedFiles: per-instance isolation, dedup semantics
  - Multi-arch packaging: each library package independently tracks its own files
  - params.populated_packages: registration and cross-package search helpers
"""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.artifacts import ArtifactCatalog
from _therock_utils.py_packaging import Parameters, PopulatedDistPackage, PopulatedFiles
from build_python_packages import validate_kpack_split_target_completeness


class TmpDirTestCase(unittest.TestCase):
    def setUp(self):
        override_temp = os.getenv("TEST_TMPDIR")
        if override_temp is not None:
            self.temp_context = None
            self.temp_dir = Path(override_temp)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.temp_context = tempfile.TemporaryDirectory()
            self.temp_dir = Path(self.temp_context.name)

    def tearDown(self):
        if self.temp_context:
            self.temp_context.cleanup()

    def write_file(self, relpath: str, content: str = ""):
        p = self.temp_dir / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return p


# ---------------------------------------------------------------------------
# Pure unit tests for PopulatedFiles
# ---------------------------------------------------------------------------


class PopulatedFilesTest(unittest.TestCase):
    """Tests for PopulatedFiles in isolation — no disk I/O."""

    def _fake_package(self):
        """Minimal stand-in for the package argument to mark_populated."""
        import types

        return types.SimpleNamespace(platform_dir=Path("/fake"))

    def test_has_returns_false_when_empty(self):
        files = PopulatedFiles()
        self.assertFalse(files.has("lib/libfoo.so.1"))

    def test_has_returns_true_after_mark_populated(self):
        files = PopulatedFiles()
        files.mark_populated(self._fake_package(), "lib/libfoo.so.1", Path("/dest"))
        self.assertTrue(files.has("lib/libfoo.so.1"))

    def test_mark_populated_stores_package_and_path(self):
        files = PopulatedFiles()
        pkg = self._fake_package()
        dest = Path("/dest/lib/libfoo.so.1")
        files.mark_populated(pkg, "lib/libfoo.so.1", dest)
        stored_pkg, stored_path = files.materialized_relpaths["lib/libfoo.so.1"]
        self.assertIs(stored_pkg, pkg)
        self.assertEqual(stored_path, dest)

    def test_mark_populated_raises_on_duplicate(self):
        """Populating the same relpath twice within one package is always a bug."""
        files = PopulatedFiles()
        pkg = self._fake_package()
        files.mark_populated(pkg, "lib/libfoo.so.1", Path("/dest"))
        with self.assertRaises(AssertionError):
            files.mark_populated(pkg, "lib/libfoo.so.1", Path("/dest"))

    def test_two_instances_are_independent(self):
        """Regression test for the multi-arch dedup bug.

        Before the fix, all packages shared a single params.files instance.
        Whichever target family iterated first would claim every shared relpath,
        leaving the other with an incomplete (empty) package.

        After the fix, each PopulatedDistPackage has its own self.files, so
        both packages can independently own the same relpath.
        """
        f1 = PopulatedFiles()
        f2 = PopulatedFiles()
        pkg1 = self._fake_package()
        pkg2 = self._fake_package()

        dest1 = Path("/pkg1/lib/librocblas.so.5")
        dest2 = Path("/pkg2/lib/librocblas.so.5")

        f1.mark_populated(pkg1, "lib/librocblas.so.5", dest1)

        # f2 must not be affected by f1's population.
        self.assertFalse(f2.has("lib/librocblas.so.5"))
        f2.mark_populated(pkg2, "lib/librocblas.so.5", dest2)
        self.assertTrue(f2.has("lib/librocblas.so.5"))

        # f1 retains its own path, unmodified.
        _, path = f1.materialized_relpaths["lib/librocblas.so.5"]
        self.assertEqual(path, dest1)

    def test_soname_aliases_are_per_instance(self):
        """soname_aliases dict is per-instance, not shared."""
        f1 = PopulatedFiles()
        f2 = PopulatedFiles()
        f1.soname_aliases["lib/libfoo.so"] = "libfoo.so.1"
        self.assertNotIn("lib/libfoo.so", f2.soname_aliases)


# ---------------------------------------------------------------------------
# Integration tests: real artifact directories, real Parameters/PopulatedDistPackage
# ---------------------------------------------------------------------------


class MultiArchPackagingTest(TmpDirTestCase):
    """Integration tests verifying multi-arch library packaging behaviour.

    These tests create minimal artifact directories on disk (text files, no ELF
    binaries) so that ArtifactCatalog and populate_runtime_files work end-to-end
    without patchelf being invoked.
    """

    def _add_artifact(
        self,
        artifact_dir: Path,
        name: str,
        component: str,
        target_family: str,
        files: dict[str, str],
    ):
        """Create a minimal artifact directory with the given files under stage/."""
        subdir = artifact_dir / f"{name}_{component}_{target_family}"
        stage = subdir / "stage"
        for relpath, content in files.items():
            f = stage / relpath
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(content)
        (subdir / "artifact_manifest.txt").write_text("stage\n")

    def _make_params(self, artifact_dir: Path) -> Parameters:
        dest_dir = self.temp_dir / "packages"
        dest_dir.mkdir(parents=True, exist_ok=True)
        return Parameters(
            dest_dir=dest_dir,
            version="0.0.1.test",
            version_suffix="",
            artifacts=ArtifactCatalog(artifact_dir),
        )

    def test_each_library_package_independently_owns_shared_relpaths(self):
        """Both arch-specific library packages must contain all their runtime files.

        This is the end-to-end regression test for the global params.files dedup bug.
        When gfx120X-all and gfx94X-dcgpu artifacts share relpaths (e.g.
        lib/librocblas.txt), both packages must end up with that file in their own
        self.files — neither should silently skip it because the other got there first.
        """
        artifact_dir = self.temp_dir / "artifacts"
        shared_files = {
            "lib/librocblas.txt": "arch-specific rocblas",
            "lib/libhipblas.txt": "arch-neutral hipblas wrapper",
        }
        self._add_artifact(artifact_dir, "blas", "lib", "gfx120X-all", shared_files)
        self._add_artifact(artifact_dir, "blas", "lib", "gfx94X-dcgpu", shared_files)

        params = self._make_params(artifact_dir)

        for target_family in sorted(params.all_target_families):
            lib = PopulatedDistPackage(
                params, logical_name="libraries", target_family=target_family
            )
            lib.populate_runtime_files(
                params.filter_artifacts(
                    lambda an, tf=target_family: an.name == "blas"
                    and an.target_family == tf
                )
            )

        self.assertEqual(len(params.populated_packages), 2)
        pkg_gfx120X = next(
            p for p in params.populated_packages if p.target_family == "gfx120X-all"
        )
        pkg_gfx94X = next(
            p for p in params.populated_packages if p.target_family == "gfx94X-dcgpu"
        )

        for relpath in shared_files:
            self.assertTrue(
                pkg_gfx120X.files.has(relpath),
                f"gfx120X-all missing {relpath}",
            )
            self.assertTrue(
                pkg_gfx94X.files.has(relpath),
                f"gfx94X-dcgpu missing {relpath}",
            )

    def test_populate_runtime_files_registers_package(self):
        """Package is appended to params.populated_packages after populate_runtime_files."""
        artifact_dir = self.temp_dir / "artifacts"
        self._add_artifact(
            artifact_dir, "blas", "lib", "gfx120X-all", {"lib/foo.txt": "x"}
        )

        params = self._make_params(artifact_dir)
        self.assertEqual(len(params.populated_packages), 0)

        lib = PopulatedDistPackage(
            params, logical_name="libraries", target_family="gfx120X-all"
        )
        lib.populate_runtime_files(
            params.filter_artifacts(lambda an: an.name == "blas")
        )

        self.assertEqual(len(params.populated_packages), 1)
        self.assertIs(params.populated_packages[0], lib)

    def test_find_populated_searches_across_packages(self):
        """_find_populated locates a file regardless of which package owns it."""
        artifact_dir = self.temp_dir / "artifacts"
        self._add_artifact(
            artifact_dir,
            "blas",
            "lib",
            "gfx120X-all",
            {"lib/librocblas.txt": "gfx120X"},
        )
        self._add_artifact(
            artifact_dir,
            "blas",
            "lib",
            "gfx94X-dcgpu",
            {"lib/librocsolver.txt": "gfx94X"},
        )

        params = self._make_params(artifact_dir)
        lib1 = PopulatedDistPackage(
            params, logical_name="libraries", target_family="gfx120X-all"
        )
        lib1.populate_runtime_files(
            params.filter_artifacts(lambda an: an.target_family == "gfx120X-all")
        )
        lib2 = PopulatedDistPackage(
            params, logical_name="libraries", target_family="gfx94X-dcgpu"
        )
        lib2.populate_runtime_files(
            params.filter_artifacts(lambda an: an.target_family == "gfx94X-dcgpu")
        )

        # Create a devel package and use it as the search context (as in real use).
        devel = PopulatedDistPackage(params, logical_name="devel")

        result = devel._find_populated("lib/librocblas.txt")
        self.assertIsNotNone(result)
        owner, _ = result
        self.assertIs(owner, lib1)

        result = devel._find_populated("lib/librocsolver.txt")
        self.assertIsNotNone(result)
        owner, _ = result
        self.assertIs(owner, lib2)

        self.assertIsNone(devel._find_populated("lib/nonexistent.txt"))

    def test_find_soname_alias_searches_across_packages(self):
        """_find_soname_alias finds an alias from any registered package."""
        artifact_dir = self.temp_dir / "artifacts"
        self._add_artifact(
            artifact_dir, "blas", "lib", "gfx120X-all", {"lib/placeholder.txt": "x"}
        )

        params = self._make_params(artifact_dir)
        lib = PopulatedDistPackage(
            params, logical_name="libraries", target_family="gfx120X-all"
        )
        lib.populate_runtime_files(
            params.filter_artifacts(lambda an: an.name == "blas")
        )

        # Inject a soname alias as populate_runtime_files does for real .so symlinks.
        lib.files.soname_aliases["lib/librocblas.so"] = "librocblas.so.5"

        devel = PopulatedDistPackage(params, logical_name="devel")
        self.assertEqual(
            devel._find_soname_alias("lib/librocblas.so"), "librocblas.so.5"
        )
        self.assertIsNone(devel._find_soname_alias("lib/nonexistent.so"))

    def test_find_populated_prefers_matching_target_family(self):
        """A target-specific devel package skips runtime pkgs from a different arch.

        When gfx120X-all and gfx94X-dcgpu both own the same relpath, a devel
        package for gfx94X-dcgpu must return the gfx94X entry, not gfx120X.
        """
        artifact_dir = self.temp_dir / "artifacts"
        shared_relpath = "lib/librocblas.txt"
        self._add_artifact(
            artifact_dir,
            "blas",
            "lib",
            "gfx120X-all",
            {shared_relpath: "gfx120X rocblas"},
        )
        self._add_artifact(
            artifact_dir,
            "blas",
            "lib",
            "gfx94X-dcgpu",
            {shared_relpath: "gfx94X rocblas"},
        )

        params = self._make_params(artifact_dir)
        lib_gfx120 = PopulatedDistPackage(
            params, logical_name="libraries", target_family="gfx120X-all"
        )
        lib_gfx120.populate_runtime_files(
            params.filter_artifacts(lambda an: an.target_family == "gfx120X-all")
        )
        lib_gfx94 = PopulatedDistPackage(
            params, logical_name="libraries", target_family="gfx94X-dcgpu"
        )
        lib_gfx94.populate_runtime_files(
            params.filter_artifacts(lambda an: an.target_family == "gfx94X-dcgpu")
        )

        devel_gfx94 = PopulatedDistPackage(
            params, logical_name="devel", target_family="gfx94X-dcgpu"
        )
        result = devel_gfx94._find_populated(shared_relpath)
        self.assertIsNotNone(result)
        owner, _ = result
        self.assertIs(owner, lib_gfx94, "gfx94X devel must link to gfx94X libraries")

    def test_find_populated_falls_back_to_generic_package(self):
        """A target-specific devel package can find files from a generic (core) package.

        target_family=None on a package means it is arch-neutral; it must never
        be skipped by the target-family filter.
        """
        artifact_dir = self.temp_dir / "artifacts"
        self._add_artifact(
            artifact_dir,
            "base",
            "lib",
            "generic",
            {"lib/librocm_core.txt": "core lib"},
        )
        self._add_artifact(
            artifact_dir,
            "blas",
            "lib",
            "gfx94X-dcgpu",
            {"lib/librocblas.txt": "gfx94X rocblas"},
        )

        params = self._make_params(artifact_dir)

        # core package: target_family=None (no target family)
        core = PopulatedDistPackage(params, logical_name="core")
        core.populate_runtime_files(
            params.filter_artifacts(lambda an: an.name == "base")
        )

        lib_gfx94 = PopulatedDistPackage(
            params, logical_name="libraries", target_family="gfx94X-dcgpu"
        )
        lib_gfx94.populate_runtime_files(
            params.filter_artifacts(lambda an: an.name == "blas")
        )

        devel_gfx94 = PopulatedDistPackage(
            params, logical_name="devel", target_family="gfx94X-dcgpu"
        )

        # The core file (generic package) must be found by the gfx94X devel.
        result = devel_gfx94._find_populated("lib/librocm_core.txt")
        self.assertIsNotNone(result)
        owner, _ = result
        self.assertIs(
            owner,
            core,
            "core (generic) file must be reachable from arch-specific devel",
        )


# ---------------------------------------------------------------------------
# Tests for Parameters construction edge cases
# ---------------------------------------------------------------------------


class ParametersConstructionTest(TmpDirTestCase):
    def test_no_arch_specific_artifacts_does_not_crash(self):
        # Regression: Parameters.__init__ raised IndexError when all_target_families
        # was empty because it did sorted(...)[0] unconditionally.
        artifact_dir = self.temp_dir / "artifacts"
        artifact_dir.mkdir()
        params = Parameters(
            dest_dir=self.temp_dir / "packages",
            version="0.0.1.test",
            version_suffix="",
            artifacts=ArtifactCatalog(artifact_dir),
        )
        self.assertIsNone(params.default_target_family)


# ---------------------------------------------------------------------------
# Tests for kpack-split device packaging
# ---------------------------------------------------------------------------


class DevicePackagingTest(TmpDirTestCase):
    """Tests for kpack-split mode: arch-neutral libraries + per-ISA device wheels."""

    def _add_artifact(
        self,
        artifact_dir: Path,
        name: str,
        component: str,
        target_family: str,
        files: dict[str, str],
    ):
        """Create a minimal artifact directory with the given files under stage/."""
        subdir = artifact_dir / f"{name}_{component}_{target_family}"
        stage = subdir / "stage"
        for relpath, content in files.items():
            f = stage / relpath
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(content)
        (subdir / "artifact_manifest.txt").write_text("stage\n")

    def _make_params(
        self,
        artifact_dir: Path,
        kpack_split: bool = False,
        version: str = "0.0.1.test",
    ) -> Parameters:
        dest_dir = self.temp_dir / "packages"
        dest_dir.mkdir(parents=True, exist_ok=True)
        return Parameters(
            dest_dir=dest_dir,
            version=version,
            version_suffix="",
            artifacts=ArtifactCatalog(artifact_dir),
            kpack_split=kpack_split,
        )

    def _setup_kpack_split_artifacts(self) -> Path:
        """Create a minimal set of generic + per-ISA artifacts."""
        artifact_dir = self.temp_dir / "artifacts"
        # Generic library artifact (host code)
        self._add_artifact(
            artifact_dir,
            "blas",
            "lib",
            "generic",
            {"lib/librocblas.txt": "host library"},
        )
        # Per-ISA device artifact
        self._add_artifact(
            artifact_dir,
            "blas",
            "lib",
            "gfx942",
            {
                ".kpack/blas_lib_gfx942.kpack": "kpack data",
                "lib/rocblas/library/Foo_gfx942.co": "kernel object",
            },
        )
        return artifact_dir

    def test_kpack_split_libraries_is_arch_neutral(self):
        """kpack_split=True makes the libraries entry non-target-specific."""
        artifact_dir = self._setup_kpack_split_artifacts()
        params = self._make_params(artifact_dir, kpack_split=True)

        libraries_entry = params.dist_info.ALL_PACKAGES["libraries"]
        # Should not raise — libraries is no longer target-specific.
        dist_name = libraries_entry.get_dist_package_name(target_family=None)
        self.assertEqual(dist_name, "rocm-sdk-libraries")
        # py_package_name should also work without a target.
        py_name = libraries_entry.get_py_package_name(target_family=None)
        self.assertTrue(py_name.startswith("_rocm_sdk_libraries"))

    def test_kpack_split_libraries_package_creates_without_target(self):
        """Libraries package with target_family=None should not raise in kpack-split."""
        artifact_dir = self._setup_kpack_split_artifacts()
        params = self._make_params(artifact_dir, kpack_split=True)

        # Should not raise.
        lib = PopulatedDistPackage(params, logical_name="libraries", target_family=None)
        self.assertIsNotNone(lib.path)

    def test_kpack_split_libraries_setup_uses_unsuffixed_pure_package(self):
        """setup.py must not turn target_family=None into a package name suffix."""
        artifact_dir = self._setup_kpack_split_artifacts()
        params = self._make_params(
            artifact_dir,
            kpack_split=True,
            version="0.0.1.dev0",
        )

        lib = PopulatedDistPackage(params, logical_name="libraries", target_family=None)
        result = subprocess.run(
            [sys.executable, "setup.py", "--name"],
            cwd=lib.path,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        self.assertIn(
            "Found packages: ['rocm_sdk_libraries', '_rocm_sdk_libraries']",
            result.stdout,
        )
        self.assertNotIn("rocm_sdk_libraries_None", result.stdout)

    def test_populate_device_files_copies_all_files(self):
        """populate_device_files() should copy .kpack and kernel DB files."""
        artifact_dir = self._setup_kpack_split_artifacts()
        params = self._make_params(artifact_dir, kpack_split=True)

        dev = PopulatedDistPackage(
            params, logical_name="device", target_family="gfx942"
        )
        dev.populate_device_files(
            params.filter_artifacts(
                lambda an: an.name == "blas"
                and an.component == "lib"
                and an.target_family == "gfx942"
            )
        )

        # Both files should be materialized.
        self.assertTrue(dev.files.has(".kpack/blas_lib_gfx942.kpack"))
        self.assertTrue(dev.files.has("lib/rocblas/library/Foo_gfx942.co"))

        # Files should exist on disk in the platform dir.
        platform_dir = dev._platform_dir
        self.assertTrue((platform_dir / ".kpack" / "blas_lib_gfx942.kpack").exists())
        self.assertTrue(
            (platform_dir / "lib" / "rocblas" / "library" / "Foo_gfx942.co").exists()
        )

    def test_populate_device_files_emits_devel_links_manifest(self):
        """Device wheel must ship a `_devel_links` manifest mapping each device
        file to its relative hardlink target into the libraries overlay dir.

        This manifest is what `rocm-sdk init` consumes to mirror per-ISA device
        files (.kpack/.co/.dat/...) into the generic rocm-sdk-devel tree.
        """
        artifact_dir = self._setup_kpack_split_artifacts()
        params = self._make_params(artifact_dir, kpack_split=True)

        dev = PopulatedDistPackage(
            params, logical_name="device", target_family="gfx942"
        )
        dev.populate_device_files(
            params.filter_artifacts(
                lambda an: an.name == "blas"
                and an.component == "lib"
                and an.target_family == "gfx942"
            )
        )

        libs_name = dev._platform_dir.name
        manifest_path = dev._platform_dir / ".devel_links" / "gfx942.json"
        self.assertTrue(
            manifest_path.is_file(),
            "device wheel must emit a .devel_links/<target>.json manifest",
        )

        data = json.loads(manifest_path.read_text())
        self.assertEqual(data["version"], "0.0.1.test")
        links = {entry["relpath"]: entry["target"] for entry in data["links"]}

        # Every device file must have an entry with a relative target that
        # backtracks out of the devel tree and into the libraries overlay.
        self.assertEqual(
            links[".kpack/blas_lib_gfx942.kpack"],
            f"../../{libs_name}/.kpack/blas_lib_gfx942.kpack",
        )
        self.assertEqual(
            links["lib/rocblas/library/Foo_gfx942.co"],
            f"../../../../{libs_name}/lib/rocblas/library/Foo_gfx942.co",
        )

        # The manifest must not list itself as a device file to link.
        self.assertNotIn(".devel_links/gfx942.json", links)

    def test_device_platform_dir_overlays_libraries(self):
        """Device package platform dir must match libraries package platform dir name."""
        artifact_dir = self._setup_kpack_split_artifacts()
        params = self._make_params(artifact_dir, kpack_split=True)

        lib = PopulatedDistPackage(params, logical_name="libraries", target_family=None)
        dev = PopulatedDistPackage(
            params, logical_name="device", target_family="gfx942"
        )

        # The platform dir names must match for the overlay to work.
        self.assertEqual(lib._platform_dir.name, dev._platform_dir.name)

    def test_device_artifact_filter(self):
        """device_artifact_filter selects only per-ISA lib artifacts."""
        # Import the filter from the build script.
        sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
        from build_python_packages import device_artifact_filter

        from _therock_utils.artifacts import ArtifactName

        # Should match per-ISA lib artifact.
        an_gfx942 = ArtifactName("blas", "lib", "gfx942")
        self.assertTrue(device_artifact_filter("gfx942", an_gfx942))

        # rccl is built TARGET_NEUTRAL but BUILD_TOPOLOGY marks it
        # target-specific so kpack splits produce per-arch rccl_lib_<arch>
        # artifacts that must land in the device wheel.
        an_rccl = ArtifactName("rccl", "lib", "gfx942")
        self.assertTrue(device_artifact_filter("gfx942", an_rccl))

        # Should NOT match generic.
        an_generic = ArtifactName("blas", "lib", "generic")
        self.assertFalse(device_artifact_filter("gfx942", an_generic))

        # Should NOT match wrong ISA.
        an_gfx1100 = ArtifactName("blas", "lib", "gfx1100")
        self.assertFalse(device_artifact_filter("gfx942", an_gfx1100))

        # Should NOT match test component.
        an_test = ArtifactName("blas", "test", "gfx942")
        self.assertFalse(device_artifact_filter("gfx942", an_test))

        # Should NOT match non-library artifact name.
        an_core = ArtifactName("core-hip", "lib", "gfx942")
        self.assertFalse(device_artifact_filter("gfx942", an_core))

    def test_device_dist_info_has_libraries_py_package_name(self):
        """Device package _dist_info.py must contain LIBRARIES_PY_PACKAGE_NAME."""
        artifact_dir = self._setup_kpack_split_artifacts()
        params = self._make_params(artifact_dir, kpack_split=True)

        dev = PopulatedDistPackage(
            params, logical_name="device", target_family="gfx942"
        )

        dist_info_path = (
            dev.path / "src" / dev.entry.pure_py_package_name / "_dist_info.py"
        )
        content = dist_info_path.read_text()
        self.assertIn("LIBRARIES_PY_PACKAGE_NAME", content)
        self.assertIn("_rocm_sdk_libraries", content)


class KpackSplitCompletenessTest(TmpDirTestCase):
    """Tests for validating kpack-split artifact coverage before packaging."""

    def _add_artifact(
        self,
        artifact_dir: Path,
        name: str,
        component: str,
        target_family: str,
    ) -> None:
        subdir = artifact_dir / f"{name}_{component}_{target_family}"
        stage = subdir / "stage"
        stage.mkdir(parents=True, exist_ok=True)
        (stage / "placeholder.txt").write_text("x")
        (subdir / "artifact_manifest.txt").write_text("stage\n")

    def _make_artifact_catalog(self, target_families: list[str]) -> ArtifactCatalog:
        artifact_dir = self.temp_dir / "artifacts"
        for target_family in target_families:
            self._add_artifact(
                artifact_dir=artifact_dir,
                name="blas",
                component="lib",
                target_family=target_family,
            )
        return ArtifactCatalog(artifact_dir)

    def _validate_completeness(
        self,
        *,
        kpack_split: bool,
        artifacts: ArtifactCatalog,
        linux_targets: list[str] | None,
        windows_targets: list[str] | None,
        platform_name: str,
    ) -> None:
        validate_kpack_split_target_completeness(
            kpack_split=kpack_split,
            artifact_dir=self.temp_dir / "artifacts",
            artifacts=artifacts,
            linux_targets=linux_targets,
            windows_targets=windows_targets,
            platform_name=platform_name,
        )

    def test_linux_completeness_passes_when_targets_match(self):
        artifacts = self._make_artifact_catalog(["gfx1100", "gfx1101"])

        self._validate_completeness(
            kpack_split=True,
            artifacts=artifacts,
            linux_targets=["gfx1100", "gfx1101"],
            windows_targets=None,
            platform_name="linux",
        )

    def test_linux_completeness_fails_when_target_is_missing(self):
        artifacts = self._make_artifact_catalog(["gfx1100"])

        with self.assertRaisesRegex(RuntimeError, "gfx1101"):
            self._validate_completeness(
                kpack_split=True,
                artifacts=artifacts,
                linux_targets=["gfx1100", "gfx1101"],
                windows_targets=None,
                platform_name="linux",
            )

    def test_windows_completeness_uses_windows_targets(self):
        artifacts = self._make_artifact_catalog(["gfx1200"])

        self._validate_completeness(
            kpack_split=True,
            artifacts=artifacts,
            linux_targets=["gfx1100"],
            windows_targets=["gfx1200"],
            platform_name="win32",
        )

    def test_completeness_skips_without_platform_target_input(self):
        artifacts = self._make_artifact_catalog(["gfx1100"])

        self._validate_completeness(
            kpack_split=True,
            artifacts=artifacts,
            linux_targets=None,
            windows_targets=["gfx1200"],
            platform_name="linux",
        )

    def test_completeness_skips_when_kpack_split_is_disabled(self):
        artifacts = self._make_artifact_catalog(["gfx1100"])

        self._validate_completeness(
            kpack_split=False,
            artifacts=artifacts,
            linux_targets=["gfx1100", "gfx1101"],
            windows_targets=None,
            platform_name="linux",
        )


class RequiredDistPackagesTest(TmpDirTestCase):
    """Tests for validating required files in the final dist directory."""

    version = "0.0.1.test"
    wheel_tag = "py3-none-linux_x86_64"

    def _add_artifact(
        self,
        artifact_dir: Path,
        name: str,
        component: str,
        target_family: str,
    ) -> None:
        subdir = artifact_dir / f"{name}_{component}_{target_family}"
        stage = subdir / "stage"
        stage.mkdir(parents=True, exist_ok=True)
        (stage / "placeholder.txt").write_text("x")
        (subdir / "artifact_manifest.txt").write_text("stage\n")

    def _make_artifact_catalog(
        self, artifact_specs: list[tuple[str, str, str]]
    ) -> ArtifactCatalog:
        artifact_dir = self.temp_dir / "artifacts"
        for name, component, target_family in artifact_specs:
            self._add_artifact(
                artifact_dir=artifact_dir,
                name=name,
                component=component,
                target_family=target_family,
            )
        return ArtifactCatalog(artifact_dir)

    def _write_dist_file(self, filename: str) -> None:
        dist_dir = self.temp_dir / "packages" / "dist"
        dist_dir.mkdir(parents=True, exist_ok=True)
        (dist_dir / filename).write_text("package")

    def _write_required_kpack_split_runtime_files(self, target: str) -> None:
        self._write_dist_file(f"rocm-{self.version}.tar.gz")
        self._write_dist_file(f"rocm_sdk_core-{self.version}-{self.wheel_tag}.whl")
        self._write_dist_file(f"rocm_sdk_libraries-{self.version}-{self.wheel_tag}.whl")
        self._write_dist_file(
            f"rocm_sdk_device_{target}-{self.version}-{self.wheel_tag}.whl"
        )

    def _validate_required_dist_packages(
        self,
        *,
        artifacts: ArtifactCatalog,
        kpack_split: bool,
        linux_targets: list[str] | None,
        windows_targets: list[str] | None,
        platform_name: str,
    ) -> None:
        sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
        from build_python_packages import validate_required_dist_packages

        validate_required_dist_packages(
            dest_dir=self.temp_dir / "packages",
            version=self.version,
            artifacts=artifacts,
            kpack_split=kpack_split,
            linux_targets=linux_targets,
            windows_targets=windows_targets,
            platform_name=platform_name,
        )

    def test_required_kpack_split_dist_packages_pass(self):
        artifacts = self._make_artifact_catalog([("blas", "lib", "gfx1100")])
        self._write_required_kpack_split_runtime_files("gfx1100")

        self._validate_required_dist_packages(
            artifacts=artifacts,
            kpack_split=True,
            linux_targets=["gfx1100"],
            windows_targets=None,
            platform_name="linux",
        )

    def test_required_dist_packages_fail_when_rocm_sdist_is_missing(self):
        artifacts = self._make_artifact_catalog([("blas", "lib", "gfx1100")])
        self._write_dist_file(f"rocm_sdk_core-{self.version}-{self.wheel_tag}.whl")
        self._write_dist_file(f"rocm_sdk_libraries-{self.version}-{self.wheel_tag}.whl")
        self._write_dist_file(
            f"rocm_sdk_device_gfx1100-{self.version}-{self.wheel_tag}.whl"
        )

        with self.assertRaisesRegex(RuntimeError, f"rocm-{self.version}.tar.gz"):
            self._validate_required_dist_packages(
                artifacts=artifacts,
                kpack_split=True,
                linux_targets=["gfx1100"],
                windows_targets=None,
                platform_name="linux",
            )

    def test_required_dist_packages_fail_when_device_wheel_is_missing(self):
        artifacts = self._make_artifact_catalog([("blas", "lib", "gfx1100")])
        self._write_dist_file(f"rocm-{self.version}.tar.gz")
        self._write_dist_file(f"rocm_sdk_core-{self.version}-{self.wheel_tag}.whl")
        self._write_dist_file(f"rocm_sdk_libraries-{self.version}-{self.wheel_tag}.whl")

        with self.assertRaisesRegex(RuntimeError, "rocm_sdk_device_gfx1100"):
            self._validate_required_dist_packages(
                artifacts=artifacts,
                kpack_split=True,
                linux_targets=["gfx1100"],
                windows_targets=None,
                platform_name="linux",
            )

    def test_required_dist_packages_require_devel_when_dev_artifacts_exist(self):
        artifacts = self._make_artifact_catalog(
            [
                ("blas", "lib", "gfx1100"),
                ("core-hip", "dev", "generic"),
            ]
        )
        self._write_required_kpack_split_runtime_files("gfx1100")

        with self.assertRaisesRegex(RuntimeError, "rocm_sdk_devel"):
            self._validate_required_dist_packages(
                artifacts=artifacts,
                kpack_split=True,
                linux_targets=["gfx1100"],
                windows_targets=None,
                platform_name="linux",
            )

    def test_required_dist_packages_skip_devel_without_dev_artifacts(self):
        artifacts = self._make_artifact_catalog([("blas", "lib", "gfx1100")])
        self._write_required_kpack_split_runtime_files("gfx1100")

        self._validate_required_dist_packages(
            artifacts=artifacts,
            kpack_split=True,
            linux_targets=["gfx1100"],
            windows_targets=None,
            platform_name="linux",
        )


# ---------------------------------------------------------------------------
# Unit tests for restrict_families (per-family meta package)
# ---------------------------------------------------------------------------


class RestrictFamiliesTest(TmpDirTestCase):
    """Tests for restrict_families=True in PopulatedDistPackage.

    These tests verify that per-family meta (rocm) packages bake the correct
    DEFAULT_TARGET_FAMILY and AVAILABLE_TARGET_FAMILIES into _dist_info.py.
    """

    def _add_artifact(
        self,
        artifact_dir: Path,
        name: str,
        component: str,
        target_family: str,
    ):
        """Create a minimal artifact directory (no files needed)."""
        subdir = artifact_dir / f"{name}_{component}_{target_family}"
        stage = subdir / "stage"
        stage.mkdir(parents=True, exist_ok=True)
        (subdir / "artifact_manifest.txt").write_text("stage\n")

    def _make_params(self, artifact_dir: Path) -> Parameters:
        dest_dir = self.temp_dir / "packages"
        dest_dir.mkdir(parents=True, exist_ok=True)
        return Parameters(
            dest_dir=dest_dir,
            version="0.0.1.test",
            version_suffix="",
            artifacts=ArtifactCatalog(artifact_dir),
        )

    def _exec_dist_info(self, meta: PopulatedDistPackage) -> dict:
        """Read and exec the generated _dist_info.py; return the namespace."""
        dist_info_path = (
            meta.path / "src" / meta.entry.pure_py_package_name / "_dist_info.py"
        )
        content = dist_info_path.read_text()
        ns: dict = {}
        exec(content, ns)
        return ns

    def _make_two_family_params(self) -> Parameters:
        artifact_dir = self.temp_dir / "artifacts"
        self._add_artifact(artifact_dir, "base", "lib", "gfx120X-all")
        self._add_artifact(artifact_dir, "base", "lib", "gfx94X-dcgpu")
        return self._make_params(artifact_dir)

    def test_restrict_families_gfx120x_only(self):
        """restrict_families=True limits _dist_info.py to the requested family."""
        params = self._make_two_family_params()

        meta = PopulatedDistPackage(
            params,
            logical_name="meta",
            target_family="gfx120X-all",
            restrict_families=True,
        )

        ns = self._exec_dist_info(meta)
        self.assertEqual(ns["DEFAULT_TARGET_FAMILY"], "gfx120X-all")
        self.assertEqual(ns["AVAILABLE_TARGET_FAMILIES"], ["gfx120X-all"])

    def test_restrict_families_gfx94x_only(self):
        """restrict_families=True works for the second family as well."""
        params = self._make_two_family_params()

        meta = PopulatedDistPackage(
            params,
            logical_name="meta",
            target_family="gfx94X-dcgpu",
            restrict_families=True,
        )

        ns = self._exec_dist_info(meta)
        self.assertEqual(ns["DEFAULT_TARGET_FAMILY"], "gfx94X-dcgpu")
        self.assertEqual(ns["AVAILABLE_TARGET_FAMILIES"], ["gfx94X-dcgpu"])

    def test_no_restrict_families_lists_all(self):
        """Without restrict_families, _dist_info.py still lists all built families."""
        params = self._make_two_family_params()

        meta = PopulatedDistPackage(
            params,
            logical_name="meta",
            target_family="gfx120X-all",
            restrict_families=False,
        )

        ns = self._exec_dist_info(meta)
        self.assertIn("gfx120X-all", ns["AVAILABLE_TARGET_FAMILIES"])
        self.assertIn("gfx94X-dcgpu", ns["AVAILABLE_TARGET_FAMILIES"])
        self.assertEqual(len(ns["AVAILABLE_TARGET_FAMILIES"]), 2)

    def test_restrict_families_single_arch_build(self):
        """In a single-arch build restrict_families is a no-op (only one family anyway)."""
        artifact_dir = self.temp_dir / "artifacts"
        self._add_artifact(artifact_dir, "base", "lib", "gfx120X-all")
        params = self._make_params(artifact_dir)

        meta = PopulatedDistPackage(
            params,
            logical_name="meta",
            target_family="gfx120X-all",
            restrict_families=True,
        )

        ns = self._exec_dist_info(meta)
        self.assertEqual(ns["DEFAULT_TARGET_FAMILY"], "gfx120X-all")
        self.assertEqual(ns["AVAILABLE_TARGET_FAMILIES"], ["gfx120X-all"])

    def test_restrict_families_ignored_when_target_family_is_none(self):
        """restrict_families=True with target_family=None must not modify families."""
        params = self._make_two_family_params()

        # This is a degenerate call (meta without a target family) but must not crash
        # and must not restrict families (since there is no specific family to restrict to).
        meta = PopulatedDistPackage(
            params,
            logical_name="meta",
            target_family=None,
            restrict_families=True,
        )

        ns = self._exec_dist_info(meta)
        # Both families must still be present — the guard condition prevented restriction.
        self.assertIn("gfx120X-all", ns["AVAILABLE_TARGET_FAMILIES"])
        self.assertIn("gfx94X-dcgpu", ns["AVAILABLE_TARGET_FAMILIES"])
        self.assertEqual(len(ns["AVAILABLE_TARGET_FAMILIES"]), 2)

    def test_restrict_families_no_dead_writes(self):
        """restrict_families=True must not produce dead writes in _dist_info.py.

        The generated file must contain no AVAILABLE_TARGET_FAMILIES.clear() and
        must not append the non-selected family at all.
        """
        params = self._make_two_family_params()

        meta = PopulatedDistPackage(
            params,
            logical_name="meta",
            target_family="gfx120X-all",
            restrict_families=True,
        )

        dist_info_path = (
            meta.path / "src" / meta.entry.pure_py_package_name / "_dist_info.py"
        )
        content = dist_info_path.read_text()
        self.assertNotIn("AVAILABLE_TARGET_FAMILIES.clear()", content)
        self.assertNotIn("gfx94X-dcgpu", content)


# ---------------------------------------------------------------------------
# Tests for cross-platform family awareness in the rocm sdist
# ---------------------------------------------------------------------------


class CrossPlatformFamiliesTest(TmpDirTestCase):
    """Tests for linux_target_families / windows_target_families kwargs.

    When a multi-arch release pipeline knows the full union of GPU targets
    across both Linux and Windows builds, the rocm sdist must (a) advertise
    that union in AVAILABLE_TARGET_FAMILIES, (b) record each platform's
    contribution separately so setup.py can attach sys_platform markers,
    (c) pick a DEFAULT_TARGET_FAMILY that resolves on either OS, and
    (d) produce identical dist_info_contents on both platforms so the
    metadata that drives the published device extras matches regardless
    of which platform's job uploaded the sdist last.

    Note on naming: the kwargs and the AVAILABLE_TARGET_FAMILIES constant
    use the historical "target_family" label, but in kpack-split mode
    (the only mode that exhibits this bug) the values are GPU **targets**
    like gfx942 / gfx1100, matching the device wheel suffixes the
    artifact catalog produces and the `rocm-sdk-device-*` package names
    published to the index. Tests use realistic target names accordingly.

    The on-disk artifact catalog is irrelevant under these kwargs; tests
    use an empty artifact tree except where the on-disk view is itself
    under test (backward compat + identity-across-disjoint-artifacts).
    """

    def _add_minimal_artifact(self, artifact_dir: Path, target_family: str):
        subdir = artifact_dir / f"base_lib_{target_family}"
        (subdir / "stage").mkdir(parents=True, exist_ok=True)
        (subdir / "artifact_manifest.txt").write_text("stage\n")

    def _make_params(
        self,
        *,
        on_disk_families: list[str] | None = None,
        linux_target_families: list[str] | None = None,
        windows_target_families: list[str] | None = None,
    ) -> Parameters:
        artifact_dir = self.temp_dir / "artifacts"
        artifact_dir.mkdir(exist_ok=True)
        for tf in on_disk_families or []:
            self._add_minimal_artifact(artifact_dir, tf)
        dest_dir = self.temp_dir / "packages"
        dest_dir.mkdir(parents=True, exist_ok=True)
        return Parameters(
            dest_dir=dest_dir,
            version="0.0.1.test",
            version_suffix="",
            artifacts=ArtifactCatalog(artifact_dir),
            linux_target_families=linux_target_families,
            windows_target_families=windows_target_families,
        )

    def _exec_dist_info(self, params: Parameters) -> dict:
        ns: dict = {}
        exec(params.dist_info_contents, ns)
        return ns

    # ----- Backward compat: no kwargs => on-disk artifact view ------------

    def test_no_kwargs_uses_on_disk_artifact_view(self):
        """Without the new kwargs, available_target_families reflects the
        artifact catalog. Single-platform builds keep their existing
        behavior unchanged.
        """
        params = self._make_params(on_disk_families=["gfx942", "gfx1100"])
        self.assertEqual(
            sorted(params.available_target_families),
            ["gfx1100", "gfx942"],
        )
        self.assertEqual(params.linux_target_families, [])
        self.assertEqual(params.windows_target_families, [])

    def test_dist_info_omits_per_platform_appends_when_kwargs_omitted(self):
        """No LINUX/WINDOWS_TARGET_FAMILIES.append() lines emitted when
        neither kwarg is passed, so the generated _dist_info.py for
        single-platform builds is byte-equivalent to today's.
        """
        params = self._make_params(on_disk_families=["gfx942"])
        ns = self._exec_dist_info(params)
        # Predeclared constants must load as empty lists.
        self.assertEqual(ns["LINUX_TARGET_FAMILIES"], [])
        self.assertEqual(ns["WINDOWS_TARGET_FAMILIES"], [])
        self.assertNotIn("LINUX_TARGET_FAMILIES.append", params.dist_info_contents)
        self.assertNotIn("WINDOWS_TARGET_FAMILIES.append", params.dist_info_contents)

    # ----- Union semantics ------------------------------------------------

    def test_available_target_families_is_sorted_union(self):
        """available_target_families is the sorted union of linux + windows."""
        params = self._make_params(
            linux_target_families=["gfx942", "gfx950", "gfx1100"],
            windows_target_families=["gfx1100", "gfx1102"],
        )
        self.assertEqual(
            params.available_target_families,
            ["gfx1100", "gfx1102", "gfx942", "gfx950"],
        )

    def test_kwargs_override_on_disk_view(self):
        """When kwargs are passed, available_target_families ignores the
        on-disk artifact list. The kwargs are the source of truth.
        """
        params = self._make_params(
            on_disk_families=["gfx942"],
            linux_target_families=["gfx1100"],
            windows_target_families=["gfx1100"],
        )
        self.assertEqual(params.available_target_families, ["gfx1100"])

    def test_per_platform_lists_sorted_and_deduped(self):
        """Each platform's list is sorted and deduped on Parameters."""
        params = self._make_params(
            linux_target_families=["gfx950", "gfx942", "gfx942"],
            windows_target_families=["gfx1100"],
        )
        self.assertEqual(params.linux_target_families, ["gfx942", "gfx950"])
        self.assertEqual(params.windows_target_families, ["gfx1100"])

    def test_only_linux_provided_union_equals_linux(self):
        """When only linux is provided, union equals linux and windows is empty."""
        params = self._make_params(
            linux_target_families=["gfx942", "gfx950"],
        )
        self.assertEqual(params.available_target_families, ["gfx942", "gfx950"])
        self.assertEqual(params.linux_target_families, ["gfx942", "gfx950"])
        self.assertEqual(params.windows_target_families, [])

    # ----- default_target_family must be cross-platform-safe --------------
    # determine_target_family() in _dist_info.py falls back to
    # DEFAULT_TARGET_FAMILY when neither ROCM_SDK_TARGET_FAMILY nor
    # offload-arch resolves. setup.py then plugs that target into every
    # target-specific extras Requires-Dist. If DEFAULT is Linux-only, a
    # Windows user without env var or offload-arch fails to resolve
    # `rocm-sdk-libraries-{DEFAULT}` because no win_amd64 wheel exists
    # for that target. Hence: prefer the intersection.

    def test_default_target_family_prefers_intersection(self):
        """DEFAULT must come from the linux ∩ windows intersection when
        non-empty, so a user without env var / offload-arch gets a target
        that has wheels for their OS.
        """
        params = self._make_params(
            linux_target_families=["gfx942", "gfx1100", "gfx950"],
            windows_target_families=["gfx1100", "gfx1102"],
        )
        # Intersection = {gfx1100}; that must be the DEFAULT.
        self.assertEqual(params.default_target_family, "gfx1100")

    def test_default_target_family_avoids_linux_only_alpha_first(self):
        """Discriminating case: alphabetical-first of the union is a
        Linux-only target but the intersection points elsewhere. The
        intersection must win, not the alphabetical-first.

        Configuration is artificial: gfx900 / gfx942 are Linux-only in
        practice; the test puts gfx942 in the Windows list purely to
        construct a disagreement between sort-order and intersection.
        """
        # Union sorted = [gfx900, gfx942] - alpha-first is gfx900 (Linux-only).
        # Intersection = [gfx942] - DEFAULT must be gfx942.
        params = self._make_params(
            linux_target_families=["gfx900", "gfx942"],
            windows_target_families=["gfx942"],
        )
        self.assertEqual(params.default_target_family, "gfx942")

    def test_default_target_family_intersection_is_sorted(self):
        """When the intersection has multiple targets, DEFAULT is the
        sorted-first of the intersection (deterministic).
        """
        params = self._make_params(
            linux_target_families=["gfx942", "gfx1100", "gfx1200"],
            windows_target_families=["gfx1200", "gfx1100"],
        )
        # Intersection sorted = [gfx1100, gfx1200] => gfx1100.
        self.assertEqual(params.default_target_family, "gfx1100")

    def test_default_target_family_falls_back_to_union_when_disjoint(self):
        """When linux and windows lists are disjoint (no cross-platform
        target possible), DEFAULT falls back to sorted-first-of-union as
        a best-effort. Users on the "wrong" OS for that DEFAULT will need
        ROCM_SDK_TARGET_FAMILY or offload-arch.
        """
        params = self._make_params(
            linux_target_families=["gfx942"],
            windows_target_families=["gfx1100"],
        )
        # No intersection => sorted-union[0] = gfx1100.
        self.assertEqual(params.default_target_family, "gfx1100")

    def test_default_target_family_from_only_linux(self):
        """Only linux provided: DEFAULT is first of sorted linux list."""
        params = self._make_params(
            linux_target_families=["gfx950", "gfx942"],
        )
        self.assertEqual(params.default_target_family, "gfx942")

    def test_default_target_family_from_only_windows(self):
        """Only windows provided: DEFAULT is first of sorted windows list."""
        params = self._make_params(
            windows_target_families=["gfx1200", "gfx1100"],
        )
        self.assertEqual(params.default_target_family, "gfx1100")

    # ----- Generated _dist_info.py shape ----------------------------------

    def test_dist_info_exposes_per_platform_constants(self):
        """LINUX_TARGET_FAMILIES / WINDOWS_TARGET_FAMILIES baked into
        _dist_info.py so setup.py can inspect them at install time.
        """
        params = self._make_params(
            linux_target_families=["gfx942", "gfx950"],
            windows_target_families=["gfx1100"],
        )
        ns = self._exec_dist_info(params)
        self.assertEqual(sorted(ns["LINUX_TARGET_FAMILIES"]), ["gfx942", "gfx950"])
        self.assertEqual(ns["WINDOWS_TARGET_FAMILIES"], ["gfx1100"])
        self.assertEqual(
            sorted(ns["AVAILABLE_TARGET_FAMILIES"]),
            ["gfx1100", "gfx942", "gfx950"],
        )

    def test_dist_info_identical_across_disjoint_artifact_sets(self):
        """Core invariant: same cross-platform inputs produce identical
        dist_info_contents on both Linux and Windows machines, even when
        their on-disk artifact catalogs are disjoint. Without this, the
        last-writer-wins upload of rocm-X.Y.Z.tar.gz silently drops one
        platform's targets from the published device extras.
        """
        # Linux machine has the full Linux target set on disk.
        linux_params = self._make_params(
            on_disk_families=["gfx942", "gfx950", "gfx1100"],
            linux_target_families=["gfx942", "gfx950", "gfx1100"],
            windows_target_families=["gfx1100"],
        )
        # Windows machine only has Windows-supported targets on disk.
        windows_params = self._make_params(
            on_disk_families=["gfx1100"],
            linux_target_families=["gfx942", "gfx950", "gfx1100"],
            windows_target_families=["gfx1100"],
        )
        self.assertEqual(
            linux_params.dist_info_contents,
            windows_params.dist_info_contents,
        )


# ---------------------------------------------------------------------------
# Tests for platform marker helper
# ---------------------------------------------------------------------------


class PlatformMarkerTest(TmpDirTestCase):
    """Tests for get_target_family_platform_marker() in _dist_info.py.

    Called by the rocm setup.py at install time to decide whether a
    device-gfx* Requires-Dist needs a PEP 508 sys_platform marker.
    Returns the marker string for platform-exclusive families, "" for
    cross-platform families or when the per-platform breakdown is
    unknown (single-platform builds).
    """

    def _make_dist_info(
        self,
        *,
        linux_target_families: list[str] | None,
        windows_target_families: list[str] | None,
    ):
        artifact_dir = self.temp_dir / "artifacts"
        artifact_dir.mkdir(exist_ok=True)
        dest_dir = self.temp_dir / "packages"
        dest_dir.mkdir(parents=True, exist_ok=True)
        params = Parameters(
            dest_dir=dest_dir,
            version="0.0.1.test",
            version_suffix="",
            artifacts=ArtifactCatalog(artifact_dir),
            linux_target_families=linux_target_families,
            windows_target_families=windows_target_families,
        )
        return params.dist_info

    def test_markers_in_mixed_platform_config(self):
        """In a realistic multi-arch config with Linux-only, cross-platform,
        and Windows-only targets, each category gets the right marker.
        Mirrors the production case where Linux and Windows ship
        overlapping but unequal target sets.
        """
        dist_info = self._make_dist_info(
            linux_target_families=["gfx942", "gfx1100"],
            windows_target_families=["gfx1100", "gfx1102"],
        )
        # Linux-only target gets a linux marker.
        self.assertEqual(
            dist_info.get_target_family_platform_marker("gfx942"),
            'sys_platform == "linux"',
        )
        # Windows-only target gets a win32 marker.
        self.assertEqual(
            dist_info.get_target_family_platform_marker("gfx1102"),
            'sys_platform == "win32"',
        )
        # Cross-platform target has no marker.
        self.assertEqual(dist_info.get_target_family_platform_marker("gfx1100"), "")

    def test_no_marker_when_per_platform_lists_unknown(self):
        """Single-platform builds don't pass the new kwargs; no markers
        are added, so existing (non-multi-arch) sdists are unchanged.
        """
        dist_info = self._make_dist_info(
            linux_target_families=None, windows_target_families=None
        )
        self.assertEqual(dist_info.get_target_family_platform_marker("gfx942"), "")
        self.assertEqual(dist_info.get_target_family_platform_marker("gfx1100"), "")

    def test_no_marker_when_only_one_platform_participates(self):
        """When only one platform's families are declared (the other side
        is skipped in a multi-arch run, or a Linux-only build flows through
        the cross-platform kwargs), markers are not attached: there is no
        cross-platform variant to disambiguate from.
        """
        # Linux declared, Windows not.
        dist_info = self._make_dist_info(
            linux_target_families=["gfx942", "gfx1100"],
            windows_target_families=None,
        )
        self.assertEqual(dist_info.get_target_family_platform_marker("gfx942"), "")
        self.assertEqual(dist_info.get_target_family_platform_marker("gfx1100"), "")
        # Windows declared, Linux not.
        dist_info = self._make_dist_info(
            linux_target_families=None,
            windows_target_families=["gfx1100", "gfx1102"],
        )
        self.assertEqual(dist_info.get_target_family_platform_marker("gfx1100"), "")
        self.assertEqual(dist_info.get_target_family_platform_marker("gfx1102"), "")


# ---------------------------------------------------------------------------
# Tests for per-target device extras builder
# ---------------------------------------------------------------------------


class PerTargetExtrasTest(TmpDirTestCase):
    """Tests for build_per_target_extras() in _dist_info.py.

    The helper produces the device-gfx* and device-all entries for the
    rocm meta sdist's EXTRAS_REQUIRE. Cross-platform-aware: targets
    listed only in LINUX_TARGET_FAMILIES or only in WINDOWS_TARGET_FAMILIES
    get a sys_platform marker so `pip install rocm[device-all]` resolves
    only to wheels actually published for the user's OS.

    All tests run with kpack_split=True to mirror the multi-arch release
    pipeline (the only mode that exhibits the cross-platform divergence).
    In legacy mode the libraries package is also target-specific and the
    helper would additionally emit libraries-{target} extras; that path
    is not exercised here.
    """

    def _make_dist_info(
        self,
        *,
        linux_target_families: list[str] | None,
        windows_target_families: list[str] | None,
    ):
        artifact_dir = self.temp_dir / "artifacts"
        artifact_dir.mkdir(exist_ok=True)
        dest_dir = self.temp_dir / "packages"
        dest_dir.mkdir(parents=True, exist_ok=True)
        params = Parameters(
            dest_dir=dest_dir,
            version="0.0.1.test",
            version_suffix="",
            artifacts=ArtifactCatalog(artifact_dir),
            kpack_split=True,
            linux_target_families=linux_target_families,
            windows_target_families=windows_target_families,
        )
        return params.dist_info

    def test_empty_when_single_target(self):
        """No per-target extras for distributions with one target (legacy)."""
        dist_info = self._make_dist_info(
            linux_target_families=["gfx942"],
            windows_target_families=["gfx942"],
        )
        self.assertEqual(dist_info.build_per_target_extras(), {})

    def test_extras_emitted_for_mixed_platform_config(self):
        """In a realistic multi-arch config, the helper emits one
        device-{target} entry per available target, plus a device-all
        aggregating them. Linux-only and Windows-only targets carry
        sys_platform markers; cross-platform targets do not.
        """
        dist_info = self._make_dist_info(
            linux_target_families=["gfx942", "gfx1100"],
            windows_target_families=["gfx1100", "gfx1102"],
        )
        extras = dist_info.build_per_target_extras()

        # One entry per target plus the aggregate.
        self.assertEqual(
            sorted(extras.keys()),
            sorted(
                [
                    "device-gfx1100",
                    "device-gfx1102",
                    "device-gfx942",
                    "device-all",
                ]
            ),
        )

        # Linux-only target carries the linux marker.
        self.assertTrue(
            extras["device-gfx942"][0].endswith('; sys_platform == "linux"'),
            f"Expected linux marker on Linux-only target, got: "
            f"{extras['device-gfx942'][0]}",
        )
        # Windows-only target carries the win32 marker.
        self.assertTrue(
            extras["device-gfx1102"][0].endswith('; sys_platform == "win32"'),
            f"Expected win32 marker on Windows-only target, got: "
            f"{extras['device-gfx1102'][0]}",
        )
        # Cross-platform target has no marker.
        self.assertNotIn(";", extras["device-gfx1100"][0])
        self.assertNotIn("sys_platform", extras["device-gfx1100"][0])

        # device-all aggregates every per-target requirement verbatim.
        self.assertEqual(
            sorted(extras["device-all"]),
            sorted(
                extras["device-gfx942"]
                + extras["device-gfx1100"]
                + extras["device-gfx1102"]
            ),
        )

    def test_no_markers_when_per_platform_lists_unknown(self):
        """Without per-platform kwargs (single-platform builds), no markers
        attach so existing single-platform sdists stay unchanged.
        """
        # Populate two targets via the artifact catalog so the helper
        # actually emits per-target extras.
        artifact_dir = self.temp_dir / "artifacts"
        for t in ("gfx942", "gfx1100"):
            subdir = artifact_dir / f"base_lib_{t}"
            (subdir / "stage").mkdir(parents=True, exist_ok=True)
            (subdir / "artifact_manifest.txt").write_text("stage\n")
        dest_dir = self.temp_dir / "packages"
        dest_dir.mkdir(parents=True, exist_ok=True)
        params = Parameters(
            dest_dir=dest_dir,
            version="0.0.1.test",
            version_suffix="",
            artifacts=ArtifactCatalog(artifact_dir),
            kpack_split=True,
        )
        extras = params.dist_info.build_per_target_extras()
        for extra_name, requires in extras.items():
            for req in requires:
                self.assertNotIn(
                    "sys_platform",
                    req,
                    f"Single-platform build leaked a marker into {extra_name}: {req}",
                )

    def test_requires_dist_pins_version_and_uses_target_in_name(self):
        """Requires-Dist strings start with 'rocm-sdk-device-{target}==<ver>',
        matching the canonical PackageEntry.get_dist_package_require() shape.
        """
        dist_info = self._make_dist_info(
            linux_target_families=["gfx942", "gfx1100"],
            windows_target_families=["gfx1100"],
        )
        extras = dist_info.build_per_target_extras()
        req = extras["device-gfx942"][0]
        self.assertTrue(
            req.startswith("rocm-sdk-device-gfx942==0.0.1.test"),
            f"Unexpected Requires-Dist shape: {req}",
        )


if __name__ == "__main__":
    unittest.main()
