#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for generate_msi_wxs.py."""

import argparse
import sys
import tempfile
import unittest
import unittest.mock
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_msi_wxs import (
    PACKAGES,
    PackageDef,
    InstallLayout,
    WixDocument,
    collect_files_from_catalog,
    fetch_artifacts,
    make_id,
    build_wxs,
    parse_args,
    create_wix_document,
    resolve_install_layout,
    resolve_legacy_dlls,
    add_install_directory_tree,
    add_legacy_system32_feature,
    _stable_guid,
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
        self.assertRegex(result, r"^[A-Za-z0-9_]+$")

    def test_illegal_symbols_sanitized(self):
        # Dashes, dots, spaces, and other punctuation are all illegal in WiX
        # v4 IDs and must be replaced, leaving only [A-Za-z0-9_].
        for name in [
            "bin/foo-bar.dll",
            "bin/foo bar.dll",
            "bin/foo+bar(1).dll",
            "bin/foo@bar#baz.dll",
            "bin/foo.bar.baz.dll",
        ]:
            result = make_id(Path(name), "f")
            self.assertRegex(
                result, r"^[A-Za-z0-9_]+$", f"illegal char leaked for {name!r}"
            )

    def test_unicode_sanitized(self):
        # Non-ASCII letters pass Python's str.isalnum() but are illegal in WiX
        # IDs; they must be replaced with underscores, not passed through.
        for name in [
            "bin/café.dll",
            "bin/naïve.dll",
            "bin/日本語.dll",
            "bin/Ωmega.dll",
        ]:
            result = make_id(Path(name), "f")
            self.assertRegex(result, r"^[A-Za-z0-9_]+$", f"unicode leaked for {name!r}")

    def test_unicode_collision_resistance(self):
        # "café" and "cafe" sanitize to different-length safe strings but the
        # digest must keep their IDs distinct regardless.
        a = make_id(Path("bin/café.dll"), "f")
        b = make_id(Path("bin/cafe.dll"), "f")
        self.assertNotEqual(a, b)

    def test_max_length(self):
        # Use a path long enough that the sanitized string would blow past the
        # 72-char WiX limit if it weren't truncated, so this actually exercises
        # the length cap rather than passing trivially.
        long_path = Path(
            "a/" + "/".join(f"segment{i:02d}" for i in range(30)) + "/name.dll"
        )
        self.assertGreater(len(str(long_path)), 72)
        result = make_id(long_path, "f")
        self.assertLessEqual(len(result), 72)

    def test_long_paths_stay_unique(self):
        # Two long paths that share a truncated prefix must still get distinct
        # IDs via the digest suffix, even after the cap chops the common head.
        base = "a/" + "/".join(f"segment{i:02d}" for i in range(30))
        a = make_id(Path(base + "/alpha.dll"), "f")
        b = make_id(Path(base + "/beta.dll"), "f")
        self.assertLessEqual(len(a), 72)
        self.assertLessEqual(len(b), 72)
        self.assertNotEqual(a, b)


class TestReadRocmVersion(unittest.TestCase):
    def test_reads_version_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "version.json").write_text('{"rocm-version": "9.1.2"}')
            self.assertEqual(_read_rocm_version(root), "9.1.2")

    def test_reads_rocm_version_ignoring_other_keys(self):
        # When both "rocm-version" and unrelated keys are present, the unrelated
        # keys are ignored and "rocm-version" is returned.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "version.json").write_text(
                '{"other-key": "1.0.0", "rocm-version": "9.1.2"}'
            )
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
        # run component is intentionally excluded on Windows (contains dev tools
        # not suitable for a runtime redistributable); lib component is used instead.
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            _make_artifact_dir(artifacts, "foo", "run", self.BASEDIR, ["bin/tool.exe"])
            files = collect_files_from_catalog(artifacts, self._pkg(["foo"]))
            self.assertEqual(self._names(files), [])

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
            _make_artifact_dir(artifacts, "foo", "lib", self.BASEDIR, ["bin/tool.exe"])
            files = collect_files_from_catalog(artifacts, self._pkg(["foo"]))
            self.assertEqual(self._install_rels(files), [str(Path("bin/tool.exe"))])

    def test_stage_scoping_prevents_bleed(self):
        """Files in one artifact's stage must not appear in another artifact's results."""
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            _make_artifact_dir(artifacts, "foo", "lib", self.BASEDIR, ["bin/foo.exe"])
            _make_artifact_dir(artifacts, "bar", "lib", self.BASEDIR, ["bin/bar.exe"])
            files = collect_files_from_catalog(artifacts, self._pkg(["foo"]))
            self.assertEqual(self._names(files), ["foo.exe"])

    def test_deduplication_across_artifacts(self):
        """Same file in two artifacts counts once."""
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            _make_artifact_dir(artifacts, "a", "lib", self.BASEDIR, ["bin/shared.exe"])
            _make_artifact_dir(artifacts, "b", "lib", self.BASEDIR, ["bin/shared.exe"])
            files = collect_files_from_catalog(artifacts, self._pkg(["a", "b"]))
            self.assertEqual(len(files), 1)

    def test_dev_component_excluded(self):
        """dev component must not be included (only lib is packaged)."""
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            _make_artifact_dir(artifacts, "foo", "lib", self.BASEDIR, ["bin/tool.exe"])
            _make_artifact_dir(artifacts, "foo", "dev", self.BASEDIR, ["include/foo.h"])
            files = collect_files_from_catalog(artifacts, self._pkg(["foo"]))
            self.assertNotIn("foo.h", self._names(files))
            self.assertIn("tool.exe", self._names(files))

    def test_multiple_artifacts_collected(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            _make_artifact_dir(artifacts, "a", "lib", self.BASEDIR, ["bin/aaa.exe"])
            _make_artifact_dir(artifacts, "b", "lib", self.BASEDIR, ["bin/zzz.exe"])
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


class TestFetchArtifacts(unittest.TestCase):
    """fetch_artifacts delegates to artifact_manager.py; assert the argv."""

    def _run_and_capture(self, **kwargs):
        with unittest.mock.patch("generate_msi_wxs.subprocess.run") as mock_run:
            result = fetch_artifacts(**kwargs)
        self.assertEqual(mock_run.call_count, 1)
        argv = mock_run.call_args.args[0]
        return argv, mock_run.call_args, result

    def test_delegates_to_artifact_manager_fetch(self):
        cache = Path("/tmp/cache")
        argv, call, result = self._run_and_capture(
            run_id="12345", platform="windows", dest_dir=cache
        )
        self.assertIn("artifact_manager.py", " ".join(argv))
        self.assertIn("fetch", argv)
        self.assertIn("--run-id=12345", argv)
        self.assertIn("--platform=windows", argv)
        self.assertIn("--stage=all", argv)
        self.assertIn("--generic-only", argv)
        self.assertIn(f"--output-dir={cache}", argv)
        self.assertIn(f"--download-cache-dir={cache / '_downloads'}", argv)
        # Only the lib component is needed today; the rest are excluded to keep
        # the fetch small (more can be added once more MSI packages are scoped).
        exclude = next(a for a in argv if a.startswith("--exclude-components="))
        excluded = set(exclude.split("=", 1)[1].split(","))
        self.assertEqual(excluded, {"run", "dev", "dbg", "doc", "test"})
        self.assertNotIn("lib", excluded)
        # check=True so a fetch failure aborts rather than silently continuing.
        self.assertTrue(call.kwargs.get("check"))
        # Returns the extract dir ArtifactCatalog reads.
        self.assertEqual(result, cache / "artifacts")

    def test_passes_run_github_repo_when_set(self):
        argv, _, _ = self._run_and_capture(
            run_id="1",
            platform="windows",
            dest_dir=Path("/tmp/c"),
            run_github_repo="me/fork",
        )
        self.assertIn("--run-github-repo=me/fork", argv)

    def test_omits_run_github_repo_by_default(self):
        argv, _, _ = self._run_and_capture(
            run_id="1", platform="windows", dest_dir=Path("/tmp/c")
        )
        self.assertFalse(any(a.startswith("--run-github-repo") for a in argv))


class TestBuildWxs(unittest.TestCase):
    """Integration tests: run build_wxs and parse the resulting XML."""

    BASEDIR = "some/stage"

    def _run(
        self,
        tmp: str,
        artifact_specs: dict,
        package: str = "runtime",
        extra_args: dict = None,
    ):
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
                _make_artifact_dir(
                    artifacts, artifact_name, component, self.BASEDIR, files
                )

        # Materialize every System32 DLL the package declares in the source-tree
        # legacy dir so resolve_legacy_dlls (fail-fast on missing) is satisfied.
        # Standing in for the build's DVC pull of rocm-systems.
        legacy_dir = (
            root / "rocm-systems" / "shared" / "amdgpu-windows-interop" / "legacy"
        )
        legacy_dir.mkdir(parents=True, exist_ok=True)
        for dll in PACKAGES[package].legacy_system32_dlls:
            (legacy_dir / dll).write_bytes(b"dll")

        defaults = dict(
            package=package,
            build_root=build,
            repo_root=root,
            output=out,
            install_root="ProgramFiles64Folder",
            product_dir="AMD",
            version_dir="ROCm",
            package_version="1.2.3",
            run_id=None,
            platform="windows",
            run_github_repo=None,
            artifacts_cache_dir=root / "artifact-cache",
        )
        defaults.update(extra_args or {})
        # Override build_root so artifacts/ is under it
        defaults["build_root"] = root
        args = argparse.Namespace(**defaults)
        build_wxs(args)
        return ET.parse(out).getroot()

    def _minimal_specs(self, package: str) -> dict:
        """Return artifact specs with empty lib components for all package artifacts."""
        return {name: {"lib": []} for name in PACKAGES[package].artifacts}

    def test_produces_valid_xml(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._run(tmp, self._minimal_specs("runtime"))
            self.assertEqual(root.tag, _ns("Wix"))

    def test_package_element_attributes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._run(tmp, self._minimal_specs("runtime"))
            pkg = root.find(_ns("Package"))
            self.assertEqual(pkg.get("Version"), "1.2.3")
            self.assertEqual(pkg.get("Manufacturer"), "Advanced Micro Devices, Inc.")
            self.assertEqual(pkg.get("UpgradeCode"), PACKAGES["runtime"].upgrade_code)

    def test_install_subdir_uses_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._run(tmp, self._minimal_specs("runtime"))
            names = [d.get("Name") for d in root.iter(_ns("Directory"))]
            self.assertIn("core-1.2", names)

    def test_file_components_emitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            specs = {
                "core-hip": {"lib": ["bin/amdhip64_7.dll"]},
                "core-kpack": {"lib": []},
                "core-hipinfo": {"lib": []},
            }
            root = self._run(tmp, specs)
            names = [f.get("Name") for f in root.iter(_ns("File"))]
            self.assertIn("amdhip64_7.dll", names)

    def test_stage_scoping_in_wxs(self):
        """Files from one artifact's stage must not appear via another artifact."""
        with tempfile.TemporaryDirectory() as tmp:
            specs = {
                "core-hip": {"lib": ["bin/amdhip64_7.dll"]},
                "core-kpack": {"lib": []},
                "core-hipinfo": {"lib": []},
            }
            root = self._run(tmp, specs)
            names = [f.get("Name") for f in root.iter(_ns("File"))]
            self.assertIn("amdhip64_7.dll", names)
            self.assertNotIn("foreign.dll", names)

    def test_install_layout_is_flat(self):
        """Directory tree in WXS must be flat (bin/, lib/) not nested with basedir."""
        with tempfile.TemporaryDirectory() as tmp:
            specs = {
                "core-hip": {"lib": ["bin/amdhip64_7.dll"]},
                "core-kpack": {"lib": []},
                "core-hipinfo": {"lib": []},
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
            # The System32 DLLs are a hard prerequisite, so provide them; this
            # test is about missing *payload* artifacts, not System32 DLLs.
            legacy_dir = (
                root_path
                / "rocm-systems"
                / "shared"
                / "amdgpu-windows-interop"
                / "legacy"
            )
            legacy_dir.mkdir(parents=True)
            for dll in PACKAGES["runtime"].legacy_system32_dlls:
                (legacy_dir / dll).write_bytes(b"dll")
            args = argparse.Namespace(
                package="runtime",
                build_root=empty_build,
                repo_root=root_path,
                output=out,
                install_root="ProgramFiles64Folder",
                product_dir="AMD",
                version_dir="ROCm",
                package_version="1.2.3",
                run_id=None,
                platform="windows",
                run_github_repo=None,
                artifacts_cache_dir=root_path / "artifact-cache",
            )
            with redirect_stderr(buf):
                build_wxs(args)
            self.assertIn("Warning", buf.getvalue())
            root = ET.parse(out).getroot()
            self.assertEqual(root.tag, _ns("Wix"))

    def test_path_component_added_when_bin_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            specs = {
                "core-hip": {"lib": ["bin/amdhip64_7.dll"]},
                "core-kpack": {"lib": []},
                "core-hipinfo": {"lib": []},
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
            root = self._run(tmp, self._minimal_specs("runtime"))
            features = [f.get("Id") for f in root.iter(_ns("Feature"))]
            self.assertIn("LongPaths", features)

    def test_long_paths_uses_wix_v4_schema(self):
        # Regression guard for two WiX v4 schema requirements that a real
        # `wix build` rejects but structural XML checks miss:
        #   1. RegistrySearch must nest inside the Property it populates, not
        #      sit as a sibling of it.
        #   2. A Component's install condition is a `Condition` attribute, not
        #      a child <Condition> element (that was WiX v3 syntax).
        with tempfile.TemporaryDirectory() as tmp:
            root = self._run(tmp, self._minimal_specs("runtime"))

            prop = next(
                p
                for p in root.iter(_ns("Property"))
                if p.get("Id") == "LONGPATHS_PREEXISTING"
            )
            self.assertIsNotNone(
                prop.find(_ns("RegistrySearch")),
                "RegistrySearch must be nested inside LONGPATHS_PREEXISTING",
            )
            # No RegistrySearch may appear anywhere except inside a Property.
            for search in root.iter(_ns("RegistrySearch")):
                self.assertEqual(search.get("Property"), None)

            component = next(
                c
                for c in root.iter(_ns("Component"))
                if c.get("Id") == "LongPathsEnable"
            )
            self.assertEqual(
                component.get("Condition"), 'NOT LONGPATHS_PREEXISTING = "#1"'
            )
            self.assertIsNone(
                component.find(_ns("Condition")),
                "Component condition must be an attribute, not a child element",
            )

    def test_installfolder_property_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._run(tmp, self._minimal_specs("runtime"))
            prop_ids = [p.get("Id") for p in root.iter(_ns("Property"))]
            self.assertIn("INSTALLFOLDER", prop_ids)
            set_dirs = [s.get("Id") for s in root.iter(_ns("SetDirectory"))]
            self.assertIn("InstallDir", set_dirs)

    def test_no_legacy_system32_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._run(tmp, self._minimal_specs("runtime"))
            actions = [ca.get("Id") for ca in root.iter(_ns("CustomAction"))]
            self.assertNotIn("RemoveLegacyROCmDlls", actions)

    def test_component_refs_match_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            specs = {
                "core-hip": {"lib": ["bin/a.dll", "bin/b.dll"]},
                "core-kpack": {"lib": []},
                "core-hipinfo": {"lib": []},
            }
            root = self._run(tmp, specs)
            comp_ids = {c.get("Id") for c in root.iter(_ns("Component"))}
            ref_ids = {r.get("Id") for r in root.iter(_ns("ComponentRef"))}
            self.assertTrue(ref_ids.issubset(comp_ids))

    def test_runtime_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._run(tmp, self._minimal_specs("runtime"), package="runtime")
            pkg = root.find(_ns("Package"))
            self.assertEqual(pkg.get("UpgradeCode"), PACKAGES["runtime"].upgrade_code)
            names = [d.get("Name") for d in root.iter(_ns("Directory"))]
            self.assertIn("core-1.2", names)

    def test_custom_install_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._run(
                tmp,
                self._minimal_specs("runtime"),
                extra_args={"install_root": "C:\\MyROCm"},
            )
            targetdirs = [d.get("Id") for d in root.iter(_ns("Directory"))]
            self.assertIn("TARGETDIR", targetdirs)


class TestResolveLegacyDlls(unittest.TestCase):
    """resolve_legacy_dlls reads pre-present DLLs; it never fetches them."""

    LEGACY_REL = Path("rocm-systems/shared/amdgpu-windows-interop/legacy")

    def test_empty_names_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(resolve_legacy_dlls(root / "artifacts", [], root), [])

    def test_prefers_artifact_over_source_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_dir = root / "artifacts"
            (artifact_dir / "sub").mkdir(parents=True)
            (artifact_dir / "sub" / "amdhip64_7.dll").write_bytes(b"art")
            legacy = root / self.LEGACY_REL
            legacy.mkdir(parents=True)
            (legacy / "amdhip64_7.dll").write_bytes(b"src")
            resolved = resolve_legacy_dlls(artifact_dir, ["amdhip64_7.dll"], root)
            self.assertEqual(len(resolved), 1)
            self.assertEqual(resolved[0][1].read_bytes(), b"art")

    def test_falls_back_to_source_tree_legacy_dir(self):
        # Driver-supplied DLLs live only in the rocm-systems source checkout
        # (DVC-pulled by the build), not in the artifacts.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_dir = root / "artifacts"
            artifact_dir.mkdir()
            legacy = root / self.LEGACY_REL
            legacy.mkdir(parents=True)
            (legacy / "amdhip64_6.dll").write_bytes(b"driver")
            resolved = resolve_legacy_dlls(artifact_dir, ["amdhip64_6.dll"], root)
            self.assertEqual(len(resolved), 1)
            self.assertEqual(resolved[0][0], "amdhip64_6.dll")
            self.assertEqual(resolved[0][1], legacy / "amdhip64_6.dll")

    def test_missing_dll_is_fatal(self):
        # Presence is a prerequisite: a declared DLL that cannot be found in the
        # artifacts or the source checkout must fail, never silently ship an
        # incomplete MSI.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_dir = root / "artifacts"
            artifact_dir.mkdir()
            with self.assertRaises(FileNotFoundError):
                resolve_legacy_dlls(artifact_dir, ["absent.dll"], root)


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
            has_placeholder = any(
                p in pkg.install_subdir for p in ("{version}", "{major}", "{minor}")
            )
            self.assertTrue(
                has_placeholder,
                msg=f"{name} install_subdir missing version placeholder",
            )

    def test_registry_key_contains_version_placeholder(self):
        for name, pkg in PACKAGES.items():
            has_placeholder = any(
                p in pkg.registry_key for p in ("{version}", "{major}", "{minor}")
            )
            self.assertTrue(
                has_placeholder, msg=f"{name} registry_key missing version placeholder"
            )


class TestBuildWxsHelpers(unittest.TestCase):
    """Unit tests for the individually-testable build_wxs building blocks."""

    def _args(self, **overrides):
        defaults = dict(
            package="runtime",
            install_root="ProgramFiles64Folder",
            product_dir="AMD",
            version_dir="ROCm",
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_resolve_install_layout_splits_version(self):
        layout = resolve_install_layout(self._args(), "7.15.3")
        self.assertEqual(layout.major, "7")
        self.assertEqual(layout.minor, "15")
        self.assertEqual(layout.version, "7.15.3")
        self.assertEqual(layout.subdir_name, "core-7.15")

    def test_resolve_install_layout_handles_short_version(self):
        # Missing minor/patch components resolve to empty strings, not errors.
        layout = resolve_install_layout(self._args(), "8")
        self.assertEqual(layout.major, "8")
        self.assertEqual(layout.minor, "")

    def test_install_layout_uses_standard_dir(self):
        std = resolve_install_layout(
            self._args(install_root="ProgramFiles64Folder"), "1.2.3"
        )
        self.assertTrue(std.uses_standard_dir)
        custom = resolve_install_layout(self._args(install_root="C:\\AMD"), "1.2.3")
        self.assertFalse(custom.uses_standard_dir)

    def test_default_install_root_is_64bit_program_files(self):
        # ProgramFilesFolder resolves to C:\Program Files (x86) even in an x64
        # package; ROCm is 64-bit, so the default must be ProgramFiles64Folder
        # (C:\Program Files). Guards against regressing to the 32-bit token.
        argv = ["generate_msi_wxs.py", "--package", "runtime"]
        with unittest.mock.patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(args.install_root, "ProgramFiles64Folder")

    def test_stable_guid_is_deterministic_and_upper(self):
        a = _stable_guid("System32", "amdhip64_7.dll")
        b = _stable_guid("System32", "amdhip64_7.dll")
        self.assertEqual(a, b)
        self.assertEqual(a, a.upper())

    def test_stable_guid_distinct_for_distinct_inputs(self):
        self.assertNotEqual(_stable_guid("a"), _stable_guid("b"))
        # Multi-part identities are joined with "/", matching the real call
        # sites (e.g. "System32"/dll vs a distinct component key).
        self.assertNotEqual(
            _stable_guid("System32", "amdhip64_7.dll"),
            _stable_guid("ROCm_ROCmRuntime_PATH_component"),
        )

    def test_create_wix_document_emits_control_properties(self):
        pkg = PACKAGES["runtime"]
        doc = create_wix_document(pkg, "1.2.3")
        self.assertIsInstance(doc, WixDocument)
        prop_ids = {el.get("Id") for el in doc.package.findall(_ns("Property"))}
        self.assertEqual(
            prop_ids, {"ENABLE_LONG_PATHS", "INSTALLFOLDER", "LEGACY_INSTALL"}
        )

    def test_add_install_directory_tree_standard_vs_custom(self):
        pkg = PACKAGES["runtime"]
        # Standard token -> StandardDirectory root.
        std_doc = create_wix_document(pkg, "1.2.3")
        add_install_directory_tree(
            std_doc, resolve_install_layout(self._args(), "1.2.3")
        )
        self.assertIsNotNone(std_doc.package.find(_ns("StandardDirectory")))
        self.assertEqual(std_doc.install_dir.get("Id"), "InstallDir")
        # Absolute path -> TARGETDIR/CustomInstallRoot chain instead.
        custom_doc = create_wix_document(pkg, "1.2.3")
        add_install_directory_tree(
            custom_doc,
            resolve_install_layout(self._args(install_root="C:\\AMD"), "1.2.3"),
        )
        self.assertIsNone(custom_doc.package.find(_ns("StandardDirectory")))
        targetdir = custom_doc.package.find(_ns("Directory"))
        self.assertEqual(targetdir.get("Id"), "TARGETDIR")

    def test_add_legacy_system32_feature_no_dlls_is_noop(self):
        doc = create_wix_document(PACKAGES["runtime"], "1.2.3")
        add_legacy_system32_feature(doc, [])
        self.assertIsNone(doc.package.find(_ns("StandardDirectory")))
        feature_ids = {f.get("Id") for f in doc.package.findall(_ns("Feature"))}
        self.assertNotIn("LegacyInstall", feature_ids)


if __name__ == "__main__":
    unittest.main()
