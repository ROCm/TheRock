#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

"""Unit tests for ``build_package.py`` using API-driven fixtures and real outputs.

Tests stage artifact trees by walking ``package.json`` via ``get_package_info()`` and
the same ``{Artifact}_{Component}_{suffix}`` layout consumed by
``filter_components_fromartifactory``. Config is built via ``create_package_config()``.
DEB generation runs through real ``create_versioned_deb_package()``; only
``package_with_dpkg_build`` and ``move_packages_to_destination`` are mocked.

Run::

    python3.12 build_tools/packaging/linux/tests/build_package_test.py -v

Requires Python 3.10+ (``packaging_utils`` type syntax).
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

_LINUX_DIR = Path(__file__).resolve().parent.parent
_BUILD_TOOLS_DIR = _LINUX_DIR.parent.parent
for _path in (_BUILD_TOOLS_DIR, _LINUX_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

_BP_PATH = _LINUX_DIR / "build_package.py"
_BP_SPEC = importlib.util.spec_from_file_location("build_package", _BP_PATH)
build_package = importlib.util.module_from_spec(_BP_SPEC)
_BP_SPEC.loader.exec_module(build_package)

_DEB_PATH = _LINUX_DIR / "deb_package.py"
_DEB_SPEC = importlib.util.spec_from_file_location("deb_package", _DEB_PATH)
deb_package = importlib.util.module_from_spec(_DEB_SPEC)
_DEB_SPEC.loader.exec_module(deb_package)

from packaging_utils import (  # noqa: E402
    GFX_HOST,
    GFX_META,
    filter_components_fromartifactory,
    get_package_info,
    is_gfxarch_package,
    is_key_defined,
    update_package_name,
)


# ---------------------------------------------------------------------------
# CLI / manifest helpers
# ---------------------------------------------------------------------------


def _args(tmp: Path, **overrides) -> Namespace:
    """Build an ``argparse.Namespace`` mirroring ``build_package.py`` CLI flags."""
    artifacts = tmp / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    defaults = {
        "artifacts_dir": artifacts,
        "dest_dir": tmp / "output",
        "target": ["gfx1100", "gfx942"],
        "pkg_type": "deb",
        "rocm_version": "7.1.0",
        "version_suffix": "daily",
        "install_prefix": "/opt/rocm/core",
        "rpath_pkg": False,
        "enable_kpack": False,
        "pkg_names": None,
        "clean_build": False,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def _write_kpack_manifest(artifacts_dir: Path) -> None:
    """Write ``therock_manifest.json`` with ``KPACK_SPLIT_ARTIFACTS`` enabled."""
    manifest_dir = artifacts_dir / "pkg"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "therock_manifest.json").write_text(
        json.dumps({"flags": {"KPACK_SPLIT_ARTIFACTS": True}}),
        encoding="utf-8",
    )


def _control_field(control_text: str, field: str) -> str:
    """Return the value of a ``debian/control`` field (e.g. ``Package``)."""
    prefix = f"{field}:"
    for line in control_text.splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    raise AssertionError(f"Field {field!r} not found in control file:\n{control_text}")


# ---------------------------------------------------------------------------
# API-driven artifact staging (mirrors filter_components_fromartifactory)
# ---------------------------------------------------------------------------


def _artifact_suffix_for_staging(
    pkg_info: dict,
    artifact: dict,
    gfx_arch: str,
    *,
    enable_kpack: bool,
    artifacts_dir: Path,
) -> str | None:
    """Compute artifact directory suffix using packaging naming rules."""
    if enable_kpack:
        if gfx_arch == GFX_META:
            return None
        if gfx_arch == GFX_HOST:
            dir_suffix = "generic"
        elif is_gfxarch_package(pkg_info, enable_kpack, artifacts_dir):
            dir_suffix = gfx_arch
        elif is_key_defined(pkg_info, "Gfxarch"):
            # Staging device trees before gfx dirs exist: Gfxarch metadata still
            # implies arch-specific suffixes once kpack splits are in play.
            dir_suffix = gfx_arch
        else:
            dir_suffix = "generic"
    else:
        dir_suffix = (
            gfx_arch
            if is_gfxarch_package(pkg_info, enable_kpack, artifacts_dir)
            else "generic"
        )

    if "Artifact_Gfxarch" in artifact:
        is_artifact_gfx = str(artifact["Artifact_Gfxarch"]).lower() == "true"
        if (
            enable_kpack
            and gfx_arch not in (GFX_HOST, GFX_META)
            and not is_artifact_gfx
        ):
            return None
        return gfx_arch if is_artifact_gfx else "generic"
    return dir_suffix


def _stage_package_artifacts(
    pkg_name: str,
    artifacts_dir: Path,
    gfx_arch: str,
    *,
    enable_kpack: bool = True,
) -> list[Path]:
    """Stage artifact dirs + manifests from ``package.json`` via ``get_package_info``."""
    pkg_info = get_package_info(pkg_name)
    if enable_kpack and gfx_arch == GFX_META:
        return []

    created: list[Path] = []
    for artifact in pkg_info.get("Artifactory", []):
        suffix = _artifact_suffix_for_staging(
            pkg_info,
            artifact,
            gfx_arch,
            enable_kpack=enable_kpack,
            artifacts_dir=artifacts_dir,
        )
        if suffix is None:
            continue

        prefix = artifact["Artifact"]
        for subdir in artifact["Artifact_Subdir"]:
            subdir_name = subdir["Name"]
            for component in subdir["Components"]:
                artifact_dir = artifacts_dir / f"{prefix}_{component}_{suffix}"
                rel_path = f"{subdir_name}/{component}/libdummy.so"
                payload = artifact_dir / rel_path
                payload.parent.mkdir(parents=True, exist_ok=True)
                payload.write_bytes(b"\x00")
                manifest = artifact_dir / "artifact_manifest.txt"
                manifest.write_text(f"{rel_path}\n", encoding="utf-8")
                created.append(artifact_dir)
    return created


def _kpack_config(tmp: Path, **overrides) -> build_package.PackageConfig:
    """Build kpack ``PackageConfig`` via ``create_package_config`` (not hand-built)."""
    root = Path(tmp)
    _write_kpack_manifest(root / "artifacts")
    args = _args(root, enable_kpack=True, **overrides)
    return build_package.create_package_config(args)


def _control_path(pkg_name: str, config: build_package.PackageConfig) -> Path:
    """Return path to generated ``debian/control`` for a versioned DEB build."""
    updated = update_package_name(pkg_name, replace(config, versioned_pkg=True))
    return Path(config.dest_dir) / config.pkg_type / updated / "debian" / "control"


# ---------------------------------------------------------------------------
# Artifact staging — validates production discovery APIs
# ---------------------------------------------------------------------------
class ArtifactStagingTest(unittest.TestCase):
    """``_stage_package_artifacts`` produces trees ``filter_components`` accepts."""

    def test_staged_fft_host_artifacts_discovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            _stage_package_artifacts("amdrocm-fft", artifacts, GFX_HOST)
            sourcedirs = filter_components_fromartifactory(
                "amdrocm-fft", artifacts, GFX_HOST, enable_kpack=True
            )
            self.assertTrue(sourcedirs)

    def test_staged_fft_device_artifacts_discovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            _stage_package_artifacts("amdrocm-fft", artifacts, "gfx1100")
            sourcedirs = filter_components_fromartifactory(
                "amdrocm-fft", artifacts, "gfx1100", enable_kpack=True
            )
            self.assertTrue(sourcedirs)
            self.assertTrue(
                is_gfxarch_package(
                    get_package_info("amdrocm-fft"),
                    enable_kpack=True,
                    artifacts_dir=artifacts,
                )
            )


# ---------------------------------------------------------------------------
# create_versioned_deb_package — real control file generation
# ---------------------------------------------------------------------------
class CreateVersionedDebPackageTest(unittest.TestCase):
    """Real ``create_versioned_deb_package`` with API-staged artifacts."""

    @patch.object(deb_package, "move_packages_to_destination", return_value=[])
    @patch.object(deb_package, "package_with_dpkg_build")
    def test_fft_host_generates_control_file(self, _mock_dpkg, _mock_move):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = _kpack_config(root)
            # Host naming requires gfx dirs on disk (#5874); runtime dep needs artifacts
            # in pkg_list for convert_to_versiondependency.
            _stage_package_artifacts("amdrocm-fft", cfg.artifacts_dir, GFX_HOST)
            _stage_package_artifacts("amdrocm-fft", cfg.artifacts_dir, "gfx1100")
            _stage_package_artifacts("amdrocm-runtime", cfg.artifacts_dir, GFX_HOST)
            host_cfg = replace(cfg, gfx_arch=GFX_HOST)

            deb_package.create_versioned_deb_package("amdrocm-fft", host_cfg)

            control = _control_path("amdrocm-fft", host_cfg).read_text(encoding="utf-8")
            self.assertEqual(_control_field(control, "Package"), "amdrocm-fft-host7.1")
            self.assertEqual(_control_field(control, "Architecture"), "amd64")
            self.assertIn("amdrocm-runtime", _control_field(control, "Depends"))

    @patch.object(deb_package, "move_packages_to_destination", return_value=[])
    @patch.object(deb_package, "package_with_dpkg_build")
    def test_fft_device_generates_control_file(self, _mock_dpkg, _mock_move):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = _kpack_config(root)
            _stage_package_artifacts("amdrocm-fft", cfg.artifacts_dir, "gfx1100")
            device_cfg = replace(cfg, gfx_arch="gfx1100")

            deb_package.create_versioned_deb_package("amdrocm-fft", device_cfg)

            control = _control_path("amdrocm-fft", device_cfg).read_text(
                encoding="utf-8"
            )
            self.assertEqual(
                _control_field(control, "Package"), "amdrocm-fft7.1-gfx1100"
            )

    @patch.object(deb_package, "move_packages_to_destination", return_value=[])
    @patch.object(deb_package, "package_with_dpkg_build")
    def test_fft_meta_generates_control_file(self, _mock_dpkg, _mock_move):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = _kpack_config(root)
            meta_cfg = replace(cfg, gfx_arch=GFX_META)

            deb_package.create_versioned_deb_package("amdrocm-fft", meta_cfg)

            control = _control_path("amdrocm-fft", meta_cfg).read_text(encoding="utf-8")
            self.assertEqual(_control_field(control, "Package"), "amdrocm-fft7.1")

    @patch.object(deb_package, "move_packages_to_destination", return_value=[])
    @patch.object(deb_package, "package_with_dpkg_build")
    def test_core_sdk_device_meta_generates_control_file(self, _mock_dpkg, _mock_move):
        """Gfx-arch metapackage ``amdrocm-core-sdk`` device variant (#6093)."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = _kpack_config(root)
            device_cfg = replace(cfg, gfx_arch="gfx1100")

            deb_package.create_versioned_deb_package("amdrocm-core-sdk", device_cfg)

            control = _control_path("amdrocm-core-sdk", device_cfg).read_text(
                encoding="utf-8"
            )
            self.assertEqual(
                _control_field(control, "Package"), "amdrocm-core-sdk7.1-gfx1100"
            )
            # debian_replace_devel_name maps -devel → -dev in DEB Depends
            self.assertIn("amdrocm-core-dev", _control_field(control, "Depends"))

    @patch.object(deb_package, "move_packages_to_destination", return_value=[])
    @patch.object(deb_package, "package_with_dpkg_build")
    def test_developer_tools_versioned_metapackage_control(
        self, _mock_dpkg, _mock_move
    ):
        """Simple kpack metapackage with no Artifactory entries."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = _kpack_config(root)
            # convert_to_versiondependency only keeps amdrocm deps present in pkg_list
            _stage_package_artifacts("amdrocm-debugger", cfg.artifacts_dir, "")
            versioned_cfg = replace(cfg, gfx_arch="")

            deb_package.create_versioned_deb_package(
                "amdrocm-developer-tools", versioned_cfg
            )

            control = _control_path("amdrocm-developer-tools", versioned_cfg).read_text(
                encoding="utf-8"
            )
            self.assertEqual(
                _control_field(control, "Package"), "amdrocm-developer-tools7.1"
            )
            self.assertIn("amdrocm-debugger", _control_field(control, "Depends"))


# ---------------------------------------------------------------------------
# build_package_variants — routing with real artifact detection (#5874)
# ---------------------------------------------------------------------------
class BuildPackageVariantsRoutingTest(unittest.TestCase):
    """Top-level routing using real ``is_gfxarch_package`` / staged artifacts."""

    @patch.object(build_package, "build_gfxarch_package_variants", return_value=[])
    @patch.object(build_package, "build_simple_package_variants")
    def test_fft_with_staged_artifacts_routes_to_gfxarch(
        self, mock_simple, mock_gfxarch
    ):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _kpack_config(tmp)
            _stage_package_artifacts("amdrocm-fft", cfg.artifacts_dir, "gfx1100")
            build_package.build_package_variants("amdrocm-fft", cfg)
            mock_gfxarch.assert_called_once_with("amdrocm-fft", cfg)
            mock_simple.assert_not_called()

    @patch.object(build_package, "build_gfxarch_package_variants")
    @patch.object(build_package, "build_simple_package_variants", return_value=[])
    def test_fft_without_gfx_artifacts_routes_to_simple(
        self, mock_simple, mock_gfxarch
    ):
        """#5874: Gfxarch metadata alone must not trigger gfx splits without artifacts."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _kpack_config(tmp)
            pkg_info = get_package_info("amdrocm-fft")
            self.assertFalse(
                is_gfxarch_package(
                    pkg_info,
                    enable_kpack=True,
                    artifacts_dir=cfg.artifacts_dir,
                )
            )
            build_package.build_package_variants("amdrocm-fft", cfg)
            mock_simple.assert_called_once_with("amdrocm-fft", cfg)
            mock_gfxarch.assert_not_called()

    @patch.object(build_package, "build_gfxarch_package_variants", return_value=[])
    @patch.object(build_package, "build_simple_package_variants")
    def test_core_sdk_metapackage_routes_to_gfxarch(self, mock_simple, mock_gfxarch):
        """Gfx-arch metapackage routes to gfxarch builder even without artifacts (#6093)."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _kpack_config(tmp)
            build_package.build_package_variants("amdrocm-core-sdk", cfg)
            mock_gfxarch.assert_called_once_with("amdrocm-core-sdk", cfg)
            mock_simple.assert_not_called()

    @patch.object(build_package, "build_gfxarch_package_variants")
    @patch.object(build_package, "build_simple_package_variants", return_value=[])
    def test_developer_tools_routes_to_simple(self, mock_simple, mock_gfxarch):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _kpack_config(tmp)
            build_package.build_package_variants("amdrocm-developer-tools", cfg)
            mock_simple.assert_called_once_with("amdrocm-developer-tools", cfg)
            mock_gfxarch.assert_not_called()


# ---------------------------------------------------------------------------
# create_package_config — CLI → PackageConfig (real function, no hand-built config)
# ---------------------------------------------------------------------------
class CreatePackageConfigTest(unittest.TestCase):
    def test_explicit_targets_in_kpack_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = build_package.create_package_config(
                _args(Path(tmp), enable_kpack=True, target=["gfx1100", "gfx942"])
            )
            self.assertTrue(cfg.enable_kpack)
            self.assertEqual(cfg.gfxarch_list, ("gfx1100", "gfx942"))

    def test_auto_detect_kpack_from_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_kpack_manifest(root / "artifacts")
            cfg = build_package.create_package_config(_args(root))
            self.assertTrue(cfg.enable_kpack)

    def test_invalid_rocm_version_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                build_package.create_package_config(
                    _args(Path(tmp), rocm_version="7"),
                )


# ---------------------------------------------------------------------------
# load_kpack_from_manifest
# ---------------------------------------------------------------------------
class LoadKpackFromManifestTest(unittest.TestCase):
    def test_true_when_kpack_flag_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_kpack_manifest(Path(tmp))
            self.assertTrue(build_package.load_kpack_from_manifest(Path(tmp)))

    def test_false_when_no_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(build_package.load_kpack_from_manifest(Path(tmp)))


# ---------------------------------------------------------------------------
# parse_input_package_list — real package.json names
# ---------------------------------------------------------------------------
class ParseInputPackageListTest(unittest.TestCase):
    def test_explicit_pkg_names_filter_package_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg_list, skipped = build_package.parse_input_package_list(
                ["amdrocm-core-sdk", "amdrocm-ck", "no-such-package"],
                Path(tmp),
            )
            self.assertEqual(set(pkg_list), {"amdrocm-core-sdk", "amdrocm-ck"})
            self.assertEqual(skipped, [])


if __name__ == "__main__":
    unittest.main()
