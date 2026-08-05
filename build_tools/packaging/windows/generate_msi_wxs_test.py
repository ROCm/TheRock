#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for generate_msi_wxs.py."""

import argparse
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from generate_msi_wxs import (
    PACKAGES,
    PackageDef,
    collect_files_from_catalog,
    make_id,
    build_wxs,
    _read_rocm_version,
)

WXS_NS = "http://wixtoolset.org/schemas/v4/wxs"


def _ns(tag: str) -> str:
    return f"{{{WXS_NS}}}{tag}"


def _make_artifact_dir(
    artifacts_root: Path,
    artifact_name: str,
    component: str,
    basedir: str,
    files: list[str],
) -> Path:
    """Create an extracted artifact directory with manifest and files.

    Layout: artifacts_root/{artifact_name}_{component}_generic/
              artifact_manifest.txt  <- contains basedir
              {basedir}/
                {file1}
                {file2}
                ...
    """
    artifact_dir = artifacts_root / f"{artifact_name}_{component}_generic"
    stage = artifact_dir / basedir
    stage.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "artifact_manifest.txt").write_text(basedir)
    for rel in files:
        p = stage / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"placeholder")
    return artifact_dir


class TestMakeId(unittest.TestCase):
    def test_deterministic(self):
        p = Path("bin/hipcc.exe")
        self.assertEqual(make_id(p, "f"), make_id(p, "f"))

    def test_different_prefix(self):
        p = Path("bin/hipcc.exe")
        self.assertNotEqual(make_id(p, "f"), make_id(p, "c"))

    def test_collision_resistance(self):
        # foo-bar and foo_bar sanitize to the same safe string but must not collide
        a = make_id(Path("bin/foo-bar.dll"), "f")
        b = make_id(Path("bin/foo_bar.dll"), "f")
        self.assertNotEqual(a, b)

    def test_wix_legal_chars(self):
        import re
        result = make_id(Path("lib/llvm/lib/clang/17/include/stddef.h"), "f")
        self.assertRegex(result, r'^[A-Za-z0-9_]+$')

    def test_max_length(self):
        result = make_id(Path("a/very/deeply/nested/path/to/some/file/name.dll"), "f")
        self.assertLessEqual(len(result), 72)


class TestReadRocmVersion(unittest.TestCase):
    def test_reads_version_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "version.json").write_text('{"rocm-version": "9.1.2"}')
            self.assertEqual(_read_rocm_version(root), "9.1.2")

    def test_missing_file_returns_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(_read_rocm_version(Path(tmp)), "7.0.0")

    def test_malformed_json_returns_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "version.json").write_text("not json")
            self.assertEqual(_read_rocm_version(root), "7.0.0")

    def test_missing_key_returns_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "version.json").write_text('{"other-key": "1.0.0"}')
            self.assertEqual(_read_rocm_version(root), "7.0.0")


class TestCollectFilesFromCatalog(unittest.TestCase):
    """Each test creates extracted artifact directories with artifact_manifest.txt
    and verifies that collect_files_from_catalog returns the expected files."""

    BASEDIR = "some/stage"

    def _pkg(self, artifacts: list[str]) -> PackageDef:
        return PackageDef(
            product_name="Test",
            artifacts=artifacts,
            output_stem="test",
            install_subdir="test-{version}",
            upgrade_code="00000000-0000-0000-0000-000000000000",
            feature_id="Test",
            feature_title="Test",
            registry_key="Software\\Test\\{version}",
            description="test",
        )

    def _names(self, files):
        return [install_rel.name for install_rel, _ in files]

    def _install_rels(self, files):
        return [str(install_rel) for install_rel, _ in files]

    def test_collects_run_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            _make_artifact_dir(artifacts, "foo", "run", self.BASEDIR, ["bin/tool.exe"])
            files = collect_files_from_catalog(artifacts, self._pkg(["foo"]))
            self.assertEqual(self._names(files), ["tool.exe"])

    def test_collects_lib_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            _make_artifact_dir(artifacts, "foo", "lib", self.BASEDIR, ["lib/foo.dll"])
            files = collect_files_from_catalog(artifacts, self._pkg(["foo"]))
            self.assertEqual(self._names(files), ["foo.dll"])

    def test_install_rel_is_flattened(self):
        """install_rel must be relative to the stage root, not include basedir."""
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            _make_artifact_dir(artifacts, "foo", "run", self.BASEDIR, ["bin/tool.exe"])
            files = collect_files_from_catalog(artifacts, self._pkg(["foo"]))
            self.assertEqual(self._install_rels(files), [str(Path("bin/tool.exe"))])

    def test_stage_scoping_prevents_bleed(self):
        """Files in one artifact's stage must not appear in another artifact's results."""
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            _make_artifact_dir(artifacts, "foo", "run", self.BASEDIR, ["bin/foo.exe"])
            _make_artifact_dir(artifacts, "bar", "run", self.BASEDIR, ["bin/bar.exe"])
            files = collect_files_from_catalog(artifacts, self._pkg(["foo"]))
            self.assertEqual(self._names(files), ["foo.exe"])

    def test_deduplication_across_artifacts(self):
        """Same file in two artifacts counts once."""
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            _make_artifact_dir(artifacts, "a", "run", self.BASEDIR, ["bin/shared.exe"])
            _make_artifact_dir(artifacts, "b", "run", self.BASEDIR, ["bin/shared.exe"])
            files = collect_files_from_catalog(artifacts, self._pkg(["a", "b"]))
            self.assertEqual(len(files), 1)

    def test_dev_component_excluded(self):
        """dev component must not be included (only run and lib are packaged)."""
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            _make_artifact_dir(artifacts, "foo", "run", self.BASEDIR, ["bin/tool.exe"])
            _make_artifact_dir(artifacts, "foo", "dev", self.BASEDIR, ["include/foo.h"])
            files = collect_files_from_catalog(artifacts, self._pkg(["foo"]))
            self.assertNotIn("foo.h", self._names(files))
            self.assertIn("tool.exe", self._names(files))

    def test_multiple_artifacts_collected(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            _make_artifact_dir(artifacts, "a", "run", self.BASEDIR, ["bin/aaa.exe"])
            _make_artifact_dir(artifacts, "b", "run", self.BASEDIR, ["bin/zzz.exe"])
            files = collect_files_from_catalog(artifacts, self._pkg(["a", "b"]))
            self.assertEqual(sorted(self._names(files)), ["aaa.exe", "zzz.exe"])

    def test_missing_artifact_dir_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            nonexistent = Path(tmp) / "no_such_dir"
            import io
            from contextlib import redirect_stderr
            buf = io.StringIO()
            with redirect_stderr(buf):
                files = collect_files_from_catalog(nonexistent, self._pkg(["foo"]))
            self.assertEqual(files, [])
            self.assertIn("Warning", buf.getvalue())

    def test_empty_artifact_dir_returns_empty(self):
        """Artifact dir exists but has no matching artifacts."""
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            import io
            from contextlib import redirect_stderr
            buf = io.StringIO()
            with redirect_stderr(buf):
                files = collect_files_from_catalog(artifacts, self._pkg(["foo"]))
            self.assertEqual(files, [])


class TestBuildWxs(unittest.TestCase):
    """Integration tests: run build_wxs and parse the resulting XML."""

    BASEDIR = "some/stage"

    def _run(self, tmp: str, artifact_specs: dict, package: str = "hip-runtime",
             extra_args: dict = None):
        """Set up artifact dirs and run build_wxs.

        artifact_specs: {artifact_name: {component: [files]}}
        """
        root = Path(tmp)
        artifacts = root / "artifacts"
        build = root / "build"
        out = root / "out.wxs"
        artifacts.mkdir()
        build.mkdir()
        (root / "version.json").write_text('{"rocm-version": "1.2.3"}')

        for artifact_name, components in artifact_specs.items():
            for component, files in components.items():
                _make_artifact_dir(artifacts, artifact_name, component,
                                   self.BASEDIR, files)

        defaults = dict(
            package=package,
            build_root=build,
            output=out,
            install_root="ProgramFilesFolder",
            product_dir="AMD",
            version_dir="ROCm",
            package_version="1.2.3",
            artifacts_url=None,
            artifacts_cache_dir=root / ".artifact-cache",
        )
        defaults.update(extra_args or {})
        # Override build_root so artifacts/ is under it
        defaults["build_root"] = root
        args = argparse.Namespace(**defaults)
        build_wxs(args)
        return ET.parse(out).getroot()

    def _minimal_specs(self, package: str) -> dict:
        """Return artifact specs with empty run components for all package artifacts."""
        return {name: {"run": []} for name in PACKAGES[package].artifacts}

    def test_produces_valid_xml(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._run(tmp, self._minimal_specs("hip-runtime"))
            self.assertEqual(root.tag, _ns("Wix"))

    def test_package_element_attributes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._run(tmp, self._minimal_specs("hip-runtime"))
            pkg = root.find(_ns("Package"))
            self.assertEqual(pkg.get("Version"), "1.2.3")
            self.assertEqual(pkg.get("Manufacturer"), "Advanced Micro Devices, Inc.")
            self.assertEqual(pkg.get("UpgradeCode"), PACKAGES["hip-runtime"].upgrade_code)

    def test_install_subdir_uses_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._run(tmp, self._minimal_specs("hip-runtime"))
            names = [d.get("Name") for d in root.iter(_ns("Directory"))]
            self.assertIn("hip-runtime-1.2.3", names)

    def test_file_components_emitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            specs = {
                "core-hip": {"run": ["bin/hipcc.exe"]},
                "core-kpack": {"run": []},
                "core-hipinfo": {"run": []},
            }
            root = self._run(tmp, specs)
            names = [f.get("Name") for f in root.iter(_ns("File"))]
            self.assertIn("hipcc.exe", names)

    def test_stage_scoping_in_wxs(self):
        """Files from one artifact's stage must not appear via another artifact."""
        with tempfile.TemporaryDirectory() as tmp:
            specs = {
                "core-hip": {"run": ["bin/hipcc.exe"]},
                "core-kpack": {"run": []},
                "core-hipinfo": {"run": []},
            }
            root = self._run(tmp, specs)
            names = [f.get("Name") for f in root.iter(_ns("File"))]
            self.assertIn("hipcc.exe", names)
            self.assertNotIn("foreign.exe", names)

    def test_install_layout_is_flat(self):
        """Directory tree in WXS must be flat (bin/, lib/) not nested with basedir."""
        with tempfile.TemporaryDirectory() as tmp:
            specs = {
                "core-hip": {"run": ["bin/hipcc.exe"]},
                "core-kpack": {"run": []},
                "core-hipinfo": {"run": []},
            }
            root = self._run(tmp, specs)
            dir_names = [d.get("Name") for d in root.iter(_ns("Directory"))]
            for part in self.BASEDIR.split("/"):
                self.assertNotIn(part, dir_names)

    def test_no_files_emits_warning_not_error(self):
        """When artifact dir is missing entirely, a warning is emitted and
        an empty but valid WXS is still produced."""
        import io
        from contextlib import redirect_stderr
        with tempfile.TemporaryDirectory() as tmp:
            root_path = Path(tmp)
            # Use a build_root that has no artifacts/ subdir
            empty_build = root_path / "empty_build"
            empty_build.mkdir()
            out = root_path / "out.wxs"
            buf = io.StringIO()
            args = argparse.Namespace(
                package="hip-runtime",
                build_root=empty_build,
                output=out,
                install_root="ProgramFilesFolder",
                product_dir="AMD",
                version_dir="ROCm",
                package_version="1.2.3",
                artifacts_url=None,
                artifacts_cache_dir=root_path / ".artifact-cache",
            )
            with redirect_stderr(buf):
                build_wxs(args)
            self.assertIn("Warning", buf.getvalue())
            root = ET.parse(out).getroot()
            self.assertEqual(root.tag, _ns("Wix"))

    def test_path_component_added_when_bin_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            specs = {
                "core-hip": {"run": ["bin/hipcc.exe"]},
                "core-kpack": {"run": []},
                "core-hipinfo": {"run": []},
            }
            root = self._run(tmp, specs)
            comp_ids = [c.get("Id") for c in root.iter(_ns("Component"))]
            self.assertIn("EnvPath", comp_ids)

    def test_path_component_absent_when_no_bin(self):
        with tempfile.TemporaryDirectory() as tmp:
            specs = {
                "core-hip": {"lib": ["lib/foo.dll"]},
                "core-kpack": {"run": []},
                "core-hipinfo": {"run": []},
            }
            root = self._run(tmp, specs)
            comp_ids = [c.get("Id") for c in root.iter(_ns("Component"))]
            self.assertNotIn("EnvPath", comp_ids)

    def test_long_paths_feature_always_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._run(tmp, self._minimal_specs("hip-runtime"))
            features = [f.get("Id") for f in root.iter(_ns("Feature"))]
            self.assertIn("LongPaths", features)

    def test_installfolder_property_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._run(tmp, self._minimal_specs("hip-runtime"))
            prop_ids = [p.get("Id") for p in root.iter(_ns("Property"))]
            self.assertIn("INSTALLFOLDER", prop_ids)
            set_dirs = [s.get("Id") for s in root.iter(_ns("SetDirectory"))]
            self.assertIn("InstallDir", set_dirs)

    def test_no_legacy_system32_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._run(tmp, self._minimal_specs("hip-runtime"))
            actions = [ca.get("Id") for ca in root.iter(_ns("CustomAction"))]
            self.assertNotIn("RemoveLegacyROCmDlls", actions)

    def test_component_refs_match_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            specs = {
                "core-hip": {"run": ["bin/a.exe", "bin/b.exe"]},
                "core-kpack": {"run": []},
                "core-hipinfo": {"run": []},
            }
            root = self._run(tmp, specs)
            comp_ids = {c.get("Id") for c in root.iter(_ns("Component"))}
            ref_ids = {r.get("Id") for r in root.iter(_ns("ComponentRef"))}
            self.assertTrue(ref_ids.issubset(comp_ids))

    def test_runtimes_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._run(tmp, self._minimal_specs("runtimes"), package="runtimes")
            pkg = root.find(_ns("Package"))
            self.assertEqual(pkg.get("UpgradeCode"), PACKAGES["runtimes"].upgrade_code)
            names = [d.get("Name") for d in root.iter(_ns("Directory"))]
            self.assertIn("runtimes-1.2.3", names)

    def test_custom_install_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._run(tmp, self._minimal_specs("hip-runtime"),
                             extra_args={"install_root": "C:\\MyROCm"})
            targetdirs = [d.get("Id") for d in root.iter(_ns("Directory"))]
            self.assertIn("TARGETDIR", targetdirs)


class TestPackageDefs(unittest.TestCase):
    def test_all_upgrade_codes_unique(self):
        codes = [p.upgrade_code for p in PACKAGES.values()]
        self.assertEqual(len(codes), len(set(codes)))

    def test_all_feature_ids_unique(self):
        ids = [p.feature_id for p in PACKAGES.values()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_all_output_stems_unique(self):
        stems = [p.output_stem for p in PACKAGES.values()]
        self.assertEqual(len(stems), len(set(stems)))

    def test_install_subdir_contains_version_placeholder(self):
        for name, pkg in PACKAGES.items():
            self.assertIn("{version}", pkg.install_subdir, msg=f"{name} missing {{version}}")

    def test_registry_key_contains_version_placeholder(self):
        for name, pkg in PACKAGES.items():
            self.assertIn("{version}", pkg.registry_key, msg=f"{name} missing {{version}}")


if __name__ == "__main__":
    unittest.main()
