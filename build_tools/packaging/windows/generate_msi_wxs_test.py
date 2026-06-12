#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for generate_msi_wxs.py."""

import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from generate_msi_wxs import (
    PACKAGES,
    collect_files_from_artifacts,
    make_id,
    build_wxs,
    _read_rocm_version,
)

WXS_NS = "http://wixtoolset.org/schemas/v4/wxs"
# Basedir used in test TOMLs — stage files live at build_root/STAGE_BASEDIR/
STAGE_BASEDIR = "some/stage"


def _ns(tag: str) -> str:
    return f"{{{WXS_NS}}}{tag}"


def _write_toml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _make_stage(build_root: Path, basedir: str, *relpaths: str) -> None:
    """Create files in a stage dir and mirror them to a dist root sibling."""
    stage = build_root / basedir
    for rel in relpaths:
        p = stage / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()


def _mirror_to_dist(build_root: Path, basedir: str, dist_root: Path) -> None:
    """Copy all files from a stage dir into dist_root (flat merge)."""
    stage = build_root / basedir
    if not stage.exists():
        return
    for src in stage.rglob("*"):
        if src.is_file():
            rel = src.relative_to(stage)
            dst = dist_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())


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


class TestCollectFilesFromArtifacts(unittest.TestCase):
    """Each test creates files under build_root/STAGE_BASEDIR/ and mirrors
    them to dist_root so both the scoping (stage) and Source= paths (dist)
    are valid."""

    def _setup(self, tmp: str):
        root = Path(tmp)
        repo = root / "repo"
        build = root / "build"
        dist = root / "dist"
        repo.mkdir()
        build.mkdir()
        dist.mkdir()
        return repo, build, dist

    def _collect(self, repo, build, dist, artifacts, components=None):
        if components is None:
            components = {"run", "lib"}
        return collect_files_from_artifacts(dist, artifacts, repo, components, build)

    def _names(self, files):
        return [install_rel.name for install_rel, _ in files]

    def _install_rels(self, files):
        return [str(install_rel) for install_rel, _ in files]

    def test_explicit_include(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, build, dist = self._setup(tmp)
            _write_toml(repo / "artifact-foo.toml", f"""
[components.run."{STAGE_BASEDIR}"]
include = ["bin/tool.exe"]
""")
            _make_stage(build, STAGE_BASEDIR, "bin/tool.exe", "bin/other.exe")
            _mirror_to_dist(build, STAGE_BASEDIR, dist)
            files = self._collect(repo, build, dist, ["foo"])
            self.assertEqual(self._names(files), ["tool.exe"])

    def test_install_rel_is_flattened(self):
        """install_rel must be relative to the stage root, not include basedir."""
        with tempfile.TemporaryDirectory() as tmp:
            repo, build, dist = self._setup(tmp)
            _write_toml(repo / "artifact-foo.toml", f"""
[components.run."{STAGE_BASEDIR}"]
include = ["bin/tool.exe"]
""")
            _make_stage(build, STAGE_BASEDIR, "bin/tool.exe")
            _mirror_to_dist(build, STAGE_BASEDIR, dist)
            files = self._collect(repo, build, dist, ["foo"])
            install_rels = self._install_rels(files)
            # Must be bin/tool.exe, not some/stage/bin/tool.exe
            self.assertEqual(install_rels, [str(Path("bin/tool.exe"))])

    def test_default_lib_picks_up_dlls(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, build, dist = self._setup(tmp)
            _write_toml(repo / "artifact-foo.toml", f"""
[components.lib."{STAGE_BASEDIR}"]
""")
            _make_stage(build, STAGE_BASEDIR, "lib/foo.dll", "lib/foo.lib", "lib/foo.h")
            _mirror_to_dist(build, STAGE_BASEDIR, dist)
            files = self._collect(repo, build, dist, ["foo"])
            self.assertEqual(self._names(files), ["foo.dll"])

    def test_stage_scoping_prevents_dist_bleed(self):
        """Files from other artifacts in dist must not be picked up if absent
        from the artifact's own stage dir."""
        with tempfile.TemporaryDirectory() as tmp:
            repo, build, dist = self._setup(tmp)
            _write_toml(repo / "artifact-foo.toml", f"""
[components.run."{STAGE_BASEDIR}"]
include = ["bin/**"]
""")
            _make_stage(build, STAGE_BASEDIR, "bin/tool.exe")
            _mirror_to_dist(build, STAGE_BASEDIR, dist)
            (dist / "bin" / "foreign.exe").touch()
            files = self._collect(repo, build, dist, ["foo"])
            self.assertEqual(self._names(files), ["tool.exe"])

    def test_exclude_applied(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, build, dist = self._setup(tmp)
            _write_toml(repo / "artifact-foo.toml", f"""
[components.run."{STAGE_BASEDIR}"]
include = ["bin/**"]
exclude = ["bin/skip.exe"]
""")
            _make_stage(build, STAGE_BASEDIR, "bin/keep.exe", "bin/skip.exe")
            _mirror_to_dist(build, STAGE_BASEDIR, dist)
            files = self._collect(repo, build, dist, ["foo"])
            self.assertEqual(self._names(files), ["keep.exe"])

    def test_force_include_bypasses_exclude(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, build, dist = self._setup(tmp)
            _write_toml(repo / "artifact-foo.toml", f"""
[components.lib."{STAGE_BASEDIR}"]
include = ["lib/**"]
exclude = ["lib/clang/**"]
force_include = ["lib/clang/**"]
""")
            _make_stage(build, STAGE_BASEDIR, "lib/foo.dll", "lib/clang/resource.h")
            _mirror_to_dist(build, STAGE_BASEDIR, dist)
            files = self._collect(repo, build, dist, ["foo"])
            names = {install_rel.name for install_rel, _ in files}
            self.assertIn("resource.h", names)
            self.assertIn("foo.dll", names)

    def test_deduplication_across_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, build, dist = self._setup(tmp)
            for art in ("a", "b"):
                _write_toml(repo / f"artifact-{art}.toml", f"""
[components.run."{STAGE_BASEDIR}"]
include = ["bin/shared.exe"]
""")
            _make_stage(build, STAGE_BASEDIR, "bin/shared.exe")
            _mirror_to_dist(build, STAGE_BASEDIR, dist)
            files = self._collect(repo, build, dist, ["a", "b"])
            self.assertEqual(len(files), 1)

    def test_component_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, build, dist = self._setup(tmp)
            _write_toml(repo / "artifact-foo.toml", f"""
[components.run."{STAGE_BASEDIR}"]
include = ["bin/tool.exe"]
[components.dev."{STAGE_BASEDIR}"]
include = ["include/foo.h"]
""")
            _make_stage(build, STAGE_BASEDIR, "bin/tool.exe", "include/foo.h")
            _mirror_to_dist(build, STAGE_BASEDIR, dist)
            files = self._collect(repo, build, dist, ["foo"], components={"run"})
            self.assertEqual(self._names(files), ["tool.exe"])

    def test_default_patterns_false_skips_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, build, dist = self._setup(tmp)
            _write_toml(repo / "artifact-foo.toml", f"""
[components.lib."{STAGE_BASEDIR}"]
default_patterns = false
""")
            _make_stage(build, STAGE_BASEDIR, "lib/foo.dll")
            _mirror_to_dist(build, STAGE_BASEDIR, dist)
            files = self._collect(repo, build, dist, ["foo"])
            self.assertEqual(files, [])

    def test_missing_descriptor_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, build, dist = self._setup(tmp)
            with self.assertRaises(SystemExit):
                self._collect(repo, build, dist, ["nonexistent"])

    def test_missing_stage_dir_produces_no_files(self):
        """If a stage dir doesn't exist yet (partial build), skip it gracefully."""
        with tempfile.TemporaryDirectory() as tmp:
            repo, build, dist = self._setup(tmp)
            _write_toml(repo / "artifact-foo.toml", f"""
[components.run."{STAGE_BASEDIR}"]
include = ["bin/**"]
""")
            files = self._collect(repo, build, dist, ["foo"])
            self.assertEqual(files, [])

    def test_multiple_artifacts_collected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, build, dist = self._setup(tmp)
            for art, fname in (("a", "bin/aaa.exe"), ("b", "bin/zzz.exe")):
                _write_toml(repo / f"artifact-{art}.toml", f"""
[components.run."{STAGE_BASEDIR}"]
include = ["{fname}"]
""")
                _make_stage(build, STAGE_BASEDIR, fname)
            _mirror_to_dist(build, STAGE_BASEDIR, dist)
            files = self._collect(repo, build, dist, ["a", "b"])
            self.assertEqual(sorted(self._names(files)), ["aaa.exe", "zzz.exe"])


class TestBuildWxs(unittest.TestCase):
    """Integration tests: run build_wxs and parse the resulting XML."""

    def _run(self, tmp: str, artifacts_toml: dict[str, str],
             stage_files: list[str], package: str = "hip-runtime",
             extra_args: dict = None):
        root = Path(tmp)
        repo = root / "repo"
        build = root / "build"
        dist = root / "dist"
        out = root / "out.wxs"
        repo.mkdir(); build.mkdir(); dist.mkdir()
        (repo / "version.json").write_text('{"rocm-version": "1.2.3"}')

        for name, content in artifacts_toml.items():
            _write_toml(repo / f"artifact-{name}.toml", content)
        _make_stage(build, STAGE_BASEDIR, *stage_files)
        _mirror_to_dist(build, STAGE_BASEDIR, dist)

        import argparse
        defaults = dict(
            package=package,
            dist_root=dist,
            build_root=build,
            output=out,
            repo_root=repo,
            install_root="ProgramFilesFolder",
            product_dir="AMD",
            version_dir="ROCm",
            package_version="1.2.3",
            artifacts_url=None,
        )
        defaults.update(extra_args or {})
        args = argparse.Namespace(**defaults)
        build_wxs(args)
        return ET.parse(out).getroot()

    def _minimal_toml(self, package: str) -> dict[str, str]:
        return {
            name: f'[components.run."{STAGE_BASEDIR}"]\n'
            for name in PACKAGES[package].artifacts
        }

    def test_produces_valid_xml(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._run(tmp, self._minimal_toml("hip-runtime"), [])
            self.assertEqual(root.tag, _ns("Wix"))

    def test_package_element_attributes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._run(tmp, self._minimal_toml("hip-runtime"), [])
            pkg = root.find(_ns("Package"))
            self.assertEqual(pkg.get("Version"), "1.2.3")
            self.assertEqual(pkg.get("Manufacturer"), "Advanced Micro Devices, Inc.")
            self.assertEqual(pkg.get("UpgradeCode"), PACKAGES["hip-runtime"].upgrade_code)

    def test_install_subdir_uses_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._run(tmp, self._minimal_toml("hip-runtime"), [])
            names = [d.get("Name") for d in root.iter(_ns("Directory"))]
            self.assertIn("hip-runtime-1.2.3", names)

    def test_file_components_emitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            tomls = {
                "core-hip": f'[components.run."{STAGE_BASEDIR}"]\ninclude = ["bin/hipcc.exe"]\n',
                "core-kpack": f'[components.run."{STAGE_BASEDIR}"]\n',
                "core-hipinfo": f'[components.run."{STAGE_BASEDIR}"]\n',
            }
            root = self._run(tmp, tomls, ["bin/hipcc.exe"])
            names = [f.get("Name") for f in root.iter(_ns("File"))]
            self.assertIn("hipcc.exe", names)

    def test_stage_scoping_in_wxs(self):
        """Files from dist not covered by any artifact's stage must not appear."""
        with tempfile.TemporaryDirectory() as tmp:
            root_path = Path(tmp)
            repo = root_path / "repo"; build = root_path / "build"
            dist = root_path / "dist"; out = root_path / "out.wxs"
            repo.mkdir(); build.mkdir(); dist.mkdir()
            (repo / "version.json").write_text('{"rocm-version": "1.2.3"}')

            _write_toml(repo / "artifact-core-hip.toml", f"""
[components.run."{STAGE_BASEDIR}"]
include = ["bin/hipcc.exe"]
""")
            for art in ("core-kpack", "core-hipinfo"):
                _write_toml(repo / f"artifact-{art}.toml",
                            f'[components.run."{STAGE_BASEDIR}"]\n')

            _make_stage(build, STAGE_BASEDIR, "bin/hipcc.exe")
            _mirror_to_dist(build, STAGE_BASEDIR, dist)
            (dist / "bin" / "foreign.exe").touch()

            import argparse
            args = argparse.Namespace(
                package="hip-runtime", dist_root=dist, build_root=build,
                output=out, repo_root=repo, install_root="ProgramFilesFolder",
                product_dir="AMD", version_dir="ROCm", package_version="1.2.3",
                artifacts_url=None,
            )
            build_wxs(args)
            root = ET.parse(out).getroot()
            names = [f.get("Name") for f in root.iter(_ns("File"))]
            self.assertIn("hipcc.exe", names)
            self.assertNotIn("foreign.exe", names)

    def test_install_layout_is_flat(self):
        """Directory tree in WXS must be flat (bin/, lib/) not nested with basedir."""
        with tempfile.TemporaryDirectory() as tmp:
            tomls = {
                "core-hip": f'[components.run."{STAGE_BASEDIR}"]\ninclude = ["bin/hipcc.exe"]\n',
                "core-kpack": f'[components.run."{STAGE_BASEDIR}"]\n',
                "core-hipinfo": f'[components.run."{STAGE_BASEDIR}"]\n',
            }
            root = self._run(tmp, tomls, ["bin/hipcc.exe"])
            dir_names = [d.get("Name") for d in root.iter(_ns("Directory"))]
            # basedir components must not appear as directory names
            for part in STAGE_BASEDIR.split("/"):
                self.assertNotIn(part, dir_names)

    def test_no_files_emits_warning_not_error(self):
        import io
        from contextlib import redirect_stderr
        with tempfile.TemporaryDirectory() as tmp:
            buf = io.StringIO()
            with redirect_stderr(buf):
                root = self._run(tmp, self._minimal_toml("hip-runtime"), [])
            self.assertIn("Warning", buf.getvalue())
            self.assertEqual(root.tag, _ns("Wix"))

    def test_path_component_added_when_bin_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            tomls = {
                "core-hip": f'[components.run."{STAGE_BASEDIR}"]\ninclude = ["bin/hipcc.exe"]\n',
                "core-kpack": f'[components.run."{STAGE_BASEDIR}"]\n',
                "core-hipinfo": f'[components.run."{STAGE_BASEDIR}"]\n',
            }
            root = self._run(tmp, tomls, ["bin/hipcc.exe"])
            comp_ids = [c.get("Id") for c in root.iter(_ns("Component"))]
            self.assertIn("EnvPath", comp_ids)

    def test_path_component_absent_when_no_bin(self):
        with tempfile.TemporaryDirectory() as tmp:
            tomls = {
                "core-hip": f'[components.lib."{STAGE_BASEDIR}"]\ninclude = ["lib/foo.dll"]\n',
                "core-kpack": f'[components.run."{STAGE_BASEDIR}"]\n',
                "core-hipinfo": f'[components.run."{STAGE_BASEDIR}"]\n',
            }
            root = self._run(tmp, tomls, ["lib/foo.dll"])
            comp_ids = [c.get("Id") for c in root.iter(_ns("Component"))]
            self.assertNotIn("EnvPath", comp_ids)

    def test_long_paths_feature_always_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._run(tmp, self._minimal_toml("hip-runtime"), [])
            features = [f.get("Id") for f in root.iter(_ns("Feature"))]
            self.assertIn("LongPaths", features)

    def test_installfolder_property_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._run(tmp, self._minimal_toml("hip-runtime"), [])
            prop_ids = [p.get("Id") for p in root.iter(_ns("Property"))]
            self.assertIn("INSTALLFOLDER", prop_ids)
            set_dirs = [s.get("Id") for s in root.iter(_ns("SetDirectory"))]
            self.assertIn("InstallDir", set_dirs)

    def test_no_legacy_system32_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._run(tmp, self._minimal_toml("hip-runtime"), [])
            actions = [ca.get("Id") for ca in root.iter(_ns("CustomAction"))]
            self.assertNotIn("RemoveLegacyROCmDlls", actions)

    def test_component_refs_match_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            tomls = {
                "core-hip": f'[components.run."{STAGE_BASEDIR}"]\ninclude = ["bin/a.exe", "bin/b.exe"]\n',
                "core-kpack": f'[components.run."{STAGE_BASEDIR}"]\n',
                "core-hipinfo": f'[components.run."{STAGE_BASEDIR}"]\n',
            }
            root = self._run(tmp, tomls, ["bin/a.exe", "bin/b.exe"])
            comp_ids = {c.get("Id") for c in root.iter(_ns("Component"))}
            ref_ids = {r.get("Id") for r in root.iter(_ns("ComponentRef"))}
            self.assertTrue(ref_ids.issubset(comp_ids))

    def test_runtimes_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._run(tmp, self._minimal_toml("runtimes"), [], package="runtimes")
            pkg = root.find(_ns("Package"))
            self.assertEqual(pkg.get("UpgradeCode"), PACKAGES["runtimes"].upgrade_code)
            names = [d.get("Name") for d in root.iter(_ns("Directory"))]
            self.assertIn("runtimes-1.2.3", names)

    def test_custom_install_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._run(tmp, self._minimal_toml("hip-runtime"), [],
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
