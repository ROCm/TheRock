# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for _therock_utils/py_packaging.py.

These tests cover:
  - PopulatedFiles: per-instance isolation, dedup semantics
  - Multi-arch packaging: each library package independently tracks its own files
  - params.populated_packages: registration and cross-package search helpers
"""

import os
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.artifacts import ArtifactCatalog
from _therock_utils.py_packaging import Parameters, PopulatedDistPackage, PopulatedFiles


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


if __name__ == "__main__":
    unittest.main()
