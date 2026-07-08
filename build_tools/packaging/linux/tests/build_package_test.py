#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

"""Unit tests for ``build_package.py`` CLI parameters and packaging dispatch.

Scope
-----
Validates how ``build_package.py`` maps CLI flags to ``PackageConfig``, detects kpack
mode from manifests, resolves ``--pkg-names``, and routes each package to the correct
variant builder (gfx-arch kpack, simple kpack, or single-arch). Covers metapackages
vs regular packages, DEB vs RPM, ``--enable-kpack``, and ``--rpath-pkg``.

Test strategy
-------------
* **Config / parsing tests** — call real functions with synthetic ``argparse.Namespace``
  values built by ``_args()``; no subprocess, no real package builds.
* **Routing tests** — mock high-level builders to assert the correct code path is chosen
  (regression guard for issues like #6093 where metapackages were mis-routed).
* **Variant-builder tests** — mock ``create_versioned_*`` / ``create_nonversioned_*`` to
  count which variants are produced without invoking ``dpkg`` / ``rpmbuild``.
* **Integration with ``package.json``** — ``parse_input_package_list`` and routing tests
  use real package names (``amdrocm-core-sdk``, ``amdrocm-ck``, ``amdrocm-fft``, etc.).

Run::

    python3.12 build_tools/packaging/linux/tests/build_package_test.py -v

Requires Python 3.10+ (``packaging_utils`` type syntax).
"""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from argparse import Namespace
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

from packaging_utils import GFX_HOST, GFX_META, PackageConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Test fixtures — synthetic CLI/config and minimal artifact trees
# ---------------------------------------------------------------------------


# Build synthetic argparse.Namespace (CLI flags) for create_package_config / run tests.
def _args(tmp: Path, **overrides) -> Namespace:
    """Build an ``argparse.Namespace`` mirroring ``build_package.py`` CLI flags.

    Creates a temp artifact directory under ``tmp``. Defaults match a typical
    single-arch DEB build; pass ``**overrides`` to simulate other invocations.

    Default arguments:
        ``--artifacts-dir``   ``tmp/artifacts`` (created on disk)
        ``--dest-dir``        ``tmp/output``
        ``--target``          ``["gfx1100", "gfx942"]``
        ``--pkg-type``        ``deb``
        ``--rocm-version``    ``7.1.0``
        ``--version-suffix``  ``daily``
        ``--install-prefix``  ``/opt/rocm/core``
        ``--rpath-pkg``       ``False``
        ``--enable-kpack``    ``False``
        ``--pkg-names``       ``None`` (build all eligible packages)
        ``--clean-build``     ``False``

    Args:
        tmp: Root temp directory for artifact and output paths.
        **overrides: Any CLI field to replace in the namespace.

    Returns:
        Namespace consumed by ``create_package_config`` and ``run``.
    """
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


# Build synthetic PackageConfig for variant-builder tests (default: kpack DEB).
def _config(tmp: Path, **overrides) -> PackageConfig:
    """Build a minimal ``PackageConfig`` for variant-builder unit tests.

    Default config simulates kpack DEB mode with two GPU targets. Override fields
    to test single-arch, RPM, or ``--rpath-pkg`` paths.

    Args:
        tmp: Root temp directory; ``artifacts_dir`` and ``dest_dir`` are placed under it.
        **overrides: Any ``PackageConfig`` field to replace.

    Returns:
        Frozen ``PackageConfig`` passed to ``build_*_package_variants`` helpers.
    """
    defaults = {
        "artifacts_dir": tmp / "artifacts",
        "dest_dir": tmp / "output",
        "pkg_type": "deb",
        "rocm_version": "7.1.0",
        "version_suffix": "daily",
        "install_prefix": "/opt/rocm/core-7.1",
        "gfx_arch": "",
        "enable_rpath": False,
        "versioned_pkg": True,
        "enable_kpack": True,
        "gfxarch_list": ("gfx1100", "gfx942"),
    }
    defaults.update(overrides)
    return PackageConfig(**defaults)


# Write therock_manifest.json with KPACK_SPLIT_ARTIFACTS for kpack auto-detect.
def _write_kpack_manifest(artifacts_dir: Path) -> None:
    """Write a ``therock_manifest.json`` that enables kpack split artifacts.

    Args:
        artifacts_dir: Root scanned by ``load_kpack_from_manifest`` (recursive).
    """
    manifest_dir = artifacts_dir / "pkg"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "therock_manifest.json").write_text(
        json.dumps({"flags": {"KPACK_SPLIT_ARTIFACTS": True}}),
        encoding="utf-8",
    )


# Create empty gfx-arch artifact dir matching production {Artifact}_{Component}_gfx* naming.
def _write_gfx_artifact_dir(artifacts_dir: Path, dirname: str) -> None:
    """Create a minimal gfx-specific artifact directory under ``artifacts_dir``.

    Production kpack trees use ``{Artifact}_{Component}_gfx*`` directory names;
    an empty directory is enough for ``_has_arch_specific_artifacts`` detection.

    Args:
        artifacts_dir: Artifact tree root passed as ``PackageConfig.artifacts_dir``.
        dirname: Gfx-arch artifact directory name (e.g. ``fft_run_gfx1100``).
    """
    (artifacts_dir / dirname).mkdir(parents=True, exist_ok=True)


# Fixture: composable-kernel_run_gfx1100 dir so amdrocm-ck passes is_gfxarch_package.
def _ck_gfx_artifacts(artifacts_dir: Path) -> None:
    """Gfx dirs for ``amdrocm-ck`` → ``composable-kernel_run_gfx1100``."""
    _write_gfx_artifact_dir(artifacts_dir, "composable-kernel_run_gfx1100")


# Fixture: fft_run_gfx1100 dir so amdrocm-fft passes is_gfxarch_package.
def _fft_gfx_artifacts(artifacts_dir: Path) -> None:
    """Gfx dirs for ``amdrocm-fft`` → ``fft_run_gfx1100``."""
    _write_gfx_artifact_dir(artifacts_dir, "fft_run_gfx1100")


# ---------------------------------------------------------------------------
# create_package_config — CLI → PackageConfig
# ---------------------------------------------------------------------------
class CreatePackageConfigTest(unittest.TestCase):
    """``create_package_config(args)`` — CLI → ``PackageConfig`` mapping.

    Use case: every ``build_package.py`` invocation first normalizes flags into a
    ``PackageConfig``. These tests lock in argument handling for ``--pkg-type``,
    ``--target``, ``--enable-kpack``, ``--rpath-pkg``, ``--rocm-version``,
    ``--install-prefix``, and ``--version-suffix``.
    """

    # DEB --pkg-type is lower-cased in PackageConfig.
    def test_deb_pkg_type_normalized(self):
        """``--pkg-type DEB`` is lower-cased to ``deb`` in ``PackageConfig.pkg_type``."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = build_package.create_package_config(_args(Path(tmp), pkg_type="DEB"))
            self.assertEqual(cfg.pkg_type, "deb")

    # RPM --pkg-type is lower-cased in PackageConfig.
    def test_rpm_pkg_type_normalized(self):
        """``--pkg-type RPM`` is lower-cased to ``rpm`` in ``PackageConfig.pkg_type``."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = build_package.create_package_config(_args(Path(tmp), pkg_type="RPM"))
            self.assertEqual(cfg.pkg_type, "rpm")

    # Unsupported --pkg-type raises ValueError.
    def test_invalid_pkg_type_raises(self):
        """Unsupported ``--pkg-type`` (e.g. tgz) raises ``ValueError`` listing deb/rpm."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as ctx:
                build_package.create_package_config(
                    _args(Path(tmp), pkg_type="tgz"),
                )
            self.assertIn("deb", str(ctx.exception))
            self.assertIn("rpm", str(ctx.exception))

    # Malformed --rocm-version raises ValueError.
    def test_invalid_rocm_version_raises(self):
        """``--rocm-version`` without major.minor (e.g. ``7``) raises ``ValueError``."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                build_package.create_package_config(
                    _args(Path(tmp), rocm_version="7"),
                )

    # Default install prefix gets -major.minor suffix.
    def test_default_install_prefix_appends_major_minor(self):
        """Default ``--install-prefix /opt/rocm/core`` becomes ``/opt/rocm/core-7.1``."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = build_package.create_package_config(_args(Path(tmp)))
            self.assertEqual(cfg.install_prefix, "/opt/rocm/core-7.1")

    # Custom install prefix is not version-suffixed.
    def test_custom_install_prefix_unchanged(self):
        """Non-default ``--install-prefix`` is copied verbatim (no version suffix)."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = build_package.create_package_config(
                _args(Path(tmp), install_prefix="/custom/prefix"),
            )
            self.assertEqual(cfg.install_prefix, "/custom/prefix")

    # Kpack mode stores --target list in gfxarch_list.
    def test_explicit_targets_in_kpack_mode(self):
        """``--enable-kpack`` + ``--target gfx1100 gfx942`` fills ``gfxarch_list``."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = build_package.create_package_config(
                _args(
                    Path(tmp),
                    enable_kpack=True,
                    target=["gfx1100", "gfx942"],
                ),
            )
            self.assertTrue(cfg.enable_kpack)
            self.assertEqual(cfg.gfxarch_list, ("gfx1100", "gfx942"))
            self.assertEqual(cfg.gfx_arch, "")

    # Non-kpack mode uses first --target as gfx_arch only.
    def test_single_arch_uses_first_target(self):
        """Without kpack, ``gfx_arch`` is the first ``--target``; ``gfxarch_list`` empty."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = build_package.create_package_config(
                _args(Path(tmp), target=["gfx1100", "gfx942"]),
            )
            self.assertFalse(cfg.enable_kpack)
            self.assertEqual(cfg.gfx_arch, "gfx1100")
            self.assertEqual(cfg.gfxarch_list, ())

    # Missing --target auto-detects from artifact scan.
    @patch.object(build_package, "get_all_target_families", return_value=["gfx1200"])
    def test_auto_detect_targets_when_target_omitted(self, _mock_detect):
        """Omitted ``--target`` auto-detects via ``get_all_target_families(artifacts_dir)``."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = build_package.create_package_config(_args(Path(tmp), target=None))
            self.assertEqual(cfg.gfx_arch, "gfx1200")

    # Kpack enabled when manifest has KPACK_SPLIT_ARTIFACTS.
    def test_auto_detect_kpack_from_manifest(self):
        """Kpack enabled when ``therock_manifest.json`` has ``KPACK_SPLIT_ARTIFACTS: true``."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_kpack_manifest(root / "artifacts")
            cfg = build_package.create_package_config(_args(root))
            self.assertTrue(cfg.enable_kpack)

    # CLI --enable-kpack wins without manifest flag.
    def test_explicit_enable_kpack_overrides_absent_manifest(self):
        """``--enable-kpack`` sets kpack even when no manifest flag is present."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = build_package.create_package_config(
                _args(Path(tmp), enable_kpack=True),
            )
            self.assertTrue(cfg.enable_kpack)

    # --rpath-pkg maps to PackageConfig.enable_rpath.
    def test_rpath_pkg_flag(self):
        """``--rpath-pkg`` maps to ``PackageConfig.enable_rpath=True``."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = build_package.create_package_config(
                _args(Path(tmp), rpath_pkg=True),
            )
            self.assertTrue(cfg.enable_rpath)

    # --version-suffix is stored on PackageConfig.
    def test_version_suffix_preserved(self):
        """``--version-suffix nightly`` is stored on ``PackageConfig.version_suffix``."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = build_package.create_package_config(
                _args(Path(tmp), version_suffix="nightly"),
            )
            self.assertEqual(cfg.version_suffix, "nightly")

    # Writes PACKAGING_ARCH_LIST to GITHUB_OUTPUT for CI.
    def test_writes_packaging_arch_list_to_github_output(self):
        """When ``GITHUB_OUTPUT`` is set, writes ``PACKAGING_ARCH_LIST`` for CI matrix."""
        with tempfile.TemporaryDirectory() as tmp:
            github_env = Path(tmp) / "github_output"
            github_env.write_text("", encoding="utf-8")
            with patch.dict(os.environ, {"GITHUB_OUTPUT": str(github_env)}):
                build_package.create_package_config(
                    _args(Path(tmp), target=["gfx1100"]),
                )
            self.assertIn("PACKAGING_ARCH_LIST=gfx1100", github_env.read_text())


# ---------------------------------------------------------------------------
# load_kpack_from_manifest — kpack detection from artifact tree
# ---------------------------------------------------------------------------
class LoadKpackFromManifestTest(unittest.TestCase):
    """``load_kpack_from_manifest(artifacts_dir)`` — kpack auto-detection.

    Use case: CI artifact trees carry ``therock_manifest.json``; packaging should
    enter host+device split mode without requiring ``--enable-kpack`` on the CLI.
    """

    # load_kpack_from_manifest returns True when flag present.
    def test_true_when_kpack_flag_set(self):
        """Returns True when any nested manifest has ``flags.KPACK_SPLIT_ARTIFACTS``."""
        with tempfile.TemporaryDirectory() as tmp:
            _write_kpack_manifest(Path(tmp))
            self.assertTrue(build_package.load_kpack_from_manifest(Path(tmp)))

    # load_kpack_from_manifest returns False with no manifest.
    def test_false_when_no_manifest(self):
        """Empty artifact tree with no manifest files returns False."""
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(build_package.load_kpack_from_manifest(Path(tmp)))

    # load_kpack_from_manifest returns False without kpack flag.
    def test_false_when_manifest_missing_flag(self):
        """Manifest present but without ``KPACK_SPLIT_ARTIFACTS`` returns False."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "therock_manifest.json").write_text("{}", encoding="utf-8")
            self.assertFalse(build_package.load_kpack_from_manifest(root))


# ---------------------------------------------------------------------------
# parse_input_package_list — --pkg-names filtering
# ---------------------------------------------------------------------------
class ParseInputPackageListTest(unittest.TestCase):
    """``parse_input_package_list(pkg_names, artifact_dir)`` — ``--pkg-names`` filtering.

    Use case: operators pass ``--pkg-names amdrocm-ck amdrocm-core-sdk`` to build a
    subset; omitting it builds every package eligible from the artifact tree.
    """

    @patch.object(
        build_package, "get_package_list", return_value=(["pkg-a"], ["skip-a"])
    )
    # pkg_names=None delegates to get_package_list.
    def test_none_pkg_names_loads_all_from_artifacts(self, mock_get_list):
        """``pkg_names=None`` delegates to ``get_package_list(artifact_dir)``."""
        with tempfile.TemporaryDirectory() as tmp:
            art = Path(tmp)
            pkg_list, skipped = build_package.parse_input_package_list(None, art)
            mock_get_list.assert_called_once_with(art)
            self.assertEqual(pkg_list, ["pkg-a"])
            self.assertEqual(skipped, ["skip-a"])

    # Unknown --pkg-names entries are dropped.
    def test_explicit_pkg_names_filter_package_json(self):
        """Named packages are matched against ``package.json``; unknown names dropped."""
        with tempfile.TemporaryDirectory() as tmp:
            pkg_list, skipped = build_package.parse_input_package_list(
                ["amdrocm-core-sdk", "amdrocm-ck", "no-such-package"],
                Path(tmp),
            )
            self.assertEqual(set(pkg_list), {"amdrocm-core-sdk", "amdrocm-ck"})
            self.assertEqual(skipped, [])


# ---------------------------------------------------------------------------
# build_package_variants — top-level routing (gfxarch / simple / singlearch)
# ---------------------------------------------------------------------------
class BuildPackageVariantsRoutingTest(unittest.TestCase):
    """``build_package_variants(pkg_name, config)`` — top-level builder selection.

    Use case: the first routing decision determines host/device/meta splits vs simple
    versioned packages. Mis-routing a gfx-arch metapackage (e.g. ``amdrocm-core-sdk``)
    to the simple builder caused #6093 (missing per-GPU SDK dependencies).
    """

    @patch.object(
        build_package, "build_gfxarch_package_variants", return_value=["meta.deb"]
    )
    # Kpack gfx metapackage amdrocm-core-sdk → gfxarch builder (#6093).
    def test_kpack_metapackage_routes_to_gfxarch_builder(self, mock_gfx):
        """Kpack + gfx-arch metapackage ``amdrocm-core-sdk`` → ``build_gfxarch_package_variants``.

        Args (via ``_config``): ``enable_kpack=True``, ``pkg_type=deb``.
        Package: ``amdrocm-core-sdk`` (``Metapackage`` + ``Gfxarch`` in ``package.json``).
        """
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(Path(tmp))
            result = build_package.build_package_variants("amdrocm-core-sdk", cfg)
            mock_gfx.assert_called_once_with("amdrocm-core-sdk", cfg)
            self.assertEqual(result, ["meta.deb"])

    @patch.object(
        build_package, "build_gfxarch_package_variants", return_value=["ck.deb"]
    )
    # Kpack amdrocm-ck with gfx dirs → gfxarch builder (#5874).
    def test_kpack_gfxarch_non_meta_routes_to_gfxarch_builder(self, mock_gfx):
        """Kpack + gfx-arch regular package ``amdrocm-ck`` → gfx-arch builder.

        Requires gfx artifact dir ``composable-kernel_run_gfx1100`` on disk so
        ``is_gfxarch_package`` returns True after #5874 artifact verification.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _ck_gfx_artifacts(root / "artifacts")
            cfg = _config(root)
            result = build_package.build_package_variants("amdrocm-ck", cfg)
            mock_gfx.assert_called_once_with("amdrocm-ck", cfg)
            self.assertEqual(result, ["ck.deb"])

    @patch.object(
        build_package, "build_gfxarch_package_variants", return_value=["fft.deb"]
    )
    # Kpack amdrocm-fft with gfx dirs → gfxarch builder.
    def test_kpack_gfxarch_fft_routes_to_gfxarch_builder(self, mock_gfx):
        """Kpack + gfx-arch regular package ``amdrocm-fft`` → gfx-arch builder.

        Requires gfx artifact dir ``fft_run_gfx1100`` on disk (production math-lib
        naming) so ``is_gfxarch_package`` returns True after #5874 verification.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fft_gfx_artifacts(root / "artifacts")
            cfg = _config(root)
            result = build_package.build_package_variants("amdrocm-fft", cfg)
            mock_gfx.assert_called_once_with("amdrocm-fft", cfg)
            self.assertEqual(result, ["fft.deb"])

    @patch.object(build_package, "build_gfxarch_package_variants")
    @patch.object(
        build_package, "build_simple_package_variants", return_value=["fft.deb"]
    )
    # Kpack amdrocm-fft without gfx dirs → simple builder (#5874 negative).
    def test_kpack_gfxarch_without_artifacts_routes_to_simple_builder(
        self, mock_simple, mock_gfx
    ):
        """Kpack + ``amdrocm-fft`` with Gfxarch metadata but no ``_gfx*`` dirs → simple builder.

        Regression guard for #5874: metadata alone must not trigger gfx-arch splits when
        arch-specific artifact directories are absent on disk.
        """
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(Path(tmp))
            result = build_package.build_package_variants("amdrocm-fft", cfg)
            mock_simple.assert_called_once_with("amdrocm-fft", cfg)
            mock_gfx.assert_not_called()
            self.assertEqual(result, ["fft.deb"])

    @patch.object(
        build_package, "build_simple_package_variants", return_value=["tools.deb"]
    )
    # Kpack non-gfx package → simple builder.
    def test_kpack_non_gfxarch_routes_to_simple_builder(self, mock_simple):
        """Kpack + non-gfx-arch ``amdrocm-developer-tools`` → ``build_simple_package_variants``.

        Package has ``Gfxarch: False`` in ``package.json`` — versioned + non-versioned only.
        """
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(Path(tmp))
            result = build_package.build_package_variants(
                "amdrocm-developer-tools",
                cfg,
            )
            mock_simple.assert_called_once_with("amdrocm-developer-tools", cfg)
            self.assertEqual(result, ["tools.deb"])

    @patch.object(
        build_package, "build_singlearch_package_variants", return_value=["single.deb"]
    )
    # enable_kpack=False → singlearch builder.
    def test_single_arch_routes_to_singlearch_builder(self, mock_single):
        """``enable_kpack=False`` always uses ``build_singlearch_package_variants``.

        Args: ``enable_kpack=False``, ``gfx_arch=gfx1100`` (first ``--target``).
        """
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(Path(tmp), enable_kpack=False, gfx_arch="gfx1100")
            result = build_package.build_package_variants("amdrocm-core-sdk", cfg)
            mock_single.assert_called_once_with("amdrocm-core-sdk", cfg)
            self.assertEqual(result, ["single.deb"])


# ---------------------------------------------------------------------------
# build_gfxarch_package_variants — host / device / meta / non-versioned counts
# ---------------------------------------------------------------------------
class BuildGfxarchPackageVariantsTest(unittest.TestCase):
    """``build_gfxarch_package_variants`` — kpack host / device / meta / non-versioned.

    Use case: gfx-arch packages in kpack mode produce multiple DEB/RPM variants.
    Metapackages skip the host variant (no artifacts); regular packages include it.
    ``--rpath-pkg`` suppresses the user-facing non-versioned metapackage variant.
    """

    @patch.object(build_package, "cleanup_build_directory")
    @patch.object(
        build_package, "create_nonversioned_deb_package", return_value=["nv.deb"]
    )
    # Gfx metapackage builds device+meta+nv only (no host).
    @patch.object(build_package, "create_versioned_deb_package", return_value=["v.deb"])
    def test_metapackage_skips_host_builds_device_meta_nonversioned(
        self, mock_versioned, mock_nonversioned, _mock_cleanup
    ):
        """Metapackage ``amdrocm-core-sdk``: 2 device + 1 meta + 1 non-versioned (no host).

        ``gfxarch_list=("gfx1100", "gfx942")`` → two ``create_versioned_deb_package`` calls
        for devices, one for meta, one ``create_nonversioned_deb_package``.
        """
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(Path(tmp), pkg_type="deb")
            built = build_package.build_gfxarch_package_variants(
                "amdrocm-core-sdk",
                cfg,
            )
            self.assertEqual(mock_versioned.call_count, 3)
            self.assertEqual(mock_nonversioned.call_count, 1)
            self.assertEqual(len(built), 4)

    @patch.object(build_package, "cleanup_build_directory")
    @patch.object(
        build_package, "create_nonversioned_deb_package", return_value=["nv.deb"]
    )
    # Gfx regular package amdrocm-ck builds host+device+meta+nv.
    @patch.object(build_package, "create_versioned_deb_package", return_value=["v.deb"])
    def test_non_meta_includes_host_device_meta_nonversioned(
        self, mock_versioned, mock_nonversioned, _mock_cleanup
    ):
        """Regular gfx-arch ``amdrocm-ck``: host + 2 device + meta + non-versioned = 5 outputs."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _ck_gfx_artifacts(root / "artifacts")
            cfg = _config(root, pkg_type="deb")
            built = build_package.build_gfxarch_package_variants("amdrocm-ck", cfg)
            self.assertEqual(mock_versioned.call_count, 4)
            self.assertEqual(mock_nonversioned.call_count, 1)
            self.assertEqual(len(built), 5)

    @patch.object(build_package, "cleanup_build_directory")
    @patch.object(
        build_package, "create_nonversioned_deb_package", return_value=["nv.deb"]
    )
    # Gfx regular package amdrocm-fft builds host+device+meta+nv.
    @patch.object(build_package, "create_versioned_deb_package", return_value=["v.deb"])
    def test_fft_includes_host_device_meta_nonversioned(
        self, mock_versioned, mock_nonversioned, _mock_cleanup
    ):
        """Regular gfx-arch ``amdrocm-fft``: host + 2 device + meta + non-versioned = 5 outputs."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fft_gfx_artifacts(root / "artifacts")
            cfg = _config(root, pkg_type="deb")
            built = build_package.build_gfxarch_package_variants("amdrocm-fft", cfg)
            self.assertEqual(mock_versioned.call_count, 4)
            self.assertEqual(mock_nonversioned.call_count, 1)
            self.assertEqual(len(built), 5)

    @patch.object(build_package, "cleanup_build_directory")
    @patch.object(
        build_package, "create_nonversioned_rpm_package", return_value=["nv.rpm"]
    )
    # Gfxarch variant path uses RPM create_* helpers when pkg_type=rpm.
    @patch.object(build_package, "create_versioned_rpm_package", return_value=["v.rpm"])
    def test_rpm_pkg_type_for_gfxarch_variants(
        self, mock_versioned, mock_nonversioned, _mock_cleanup
    ):
        """``pkg_type=rpm`` dispatches to ``create_versioned_rpm_package`` / ``create_nonversioned_rpm_package``."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(Path(tmp), pkg_type="rpm")
            build_package.build_gfxarch_package_variants("amdrocm-core-sdk", cfg)
            mock_versioned.assert_called()
            mock_nonversioned.assert_called_once()
            self.assertNotIn("deb", str(mock_versioned.call_args))

    # --rpath-pkg skips non-versioned gfxarch variant.
    @patch.object(build_package, "cleanup_build_directory")
    @patch.object(build_package, "create_nonversioned_deb_package")
    @patch.object(build_package, "create_versioned_deb_package", return_value=["v.deb"])
    def test_rpath_pkg_skips_nonversioned_gfxarch_variant(
        self, mock_versioned, mock_nonversioned, _mock_cleanup
    ):
        """``enable_rpath=True`` builds only versioned variants (device + meta for metapackage)."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(Path(tmp), enable_rpath=True)
            built = build_package.build_gfxarch_package_variants(
                "amdrocm-core-sdk",
                cfg,
            )
            mock_nonversioned.assert_not_called()
            self.assertEqual(len(built), 3)
            self.assertEqual(mock_versioned.call_count, 3)


# ---------------------------------------------------------------------------
# build_simple_package_variants — kpack non-gfx-arch packages
# ---------------------------------------------------------------------------
class BuildSimplePackageVariantsTest(unittest.TestCase):
    """``build_simple_package_variants`` — kpack non-gfx-arch packages.

    Use case: packages like ``amdrocm-developer-tools`` get a versioned package
    (e.g. ``amdrocm-developer-tools7.1``) and a user-facing non-versioned metapackage.
    """

    @patch.object(build_package, "cleanup_build_directory")
    @patch.object(
        build_package, "create_nonversioned_deb_package", return_value=["nv.deb"]
    )
    # Simple kpack DEB builds versioned + non-versioned packages.
    @patch.object(build_package, "create_versioned_deb_package", return_value=["v.deb"])
    def test_deb_versioned_and_nonversioned(
        self, mock_versioned, mock_nonversioned, _mock_cleanup
    ):
        """DEB kpack simple path: one versioned + one non-versioned package."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(Path(tmp), pkg_type="deb")
            built = build_package.build_simple_package_variants(
                "amdrocm-developer-tools",
                cfg,
            )
            mock_versioned.assert_called_once()
            mock_nonversioned.assert_called_once()
            self.assertEqual(built, ["v.deb", "nv.deb"])

    @patch.object(build_package, "cleanup_build_directory")
    @patch.object(
        build_package, "create_nonversioned_rpm_package", return_value=["nv.rpm"]
    )
    # Simple kpack RPM builds versioned + non-versioned packages.
    @patch.object(build_package, "create_versioned_rpm_package", return_value=["v.rpm"])
    def test_rpm_versioned_and_nonversioned(
        self, mock_versioned, mock_nonversioned, _mock_cleanup
    ):
        """RPM kpack simple path: one versioned + one non-versioned package."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(Path(tmp), pkg_type="rpm")
            built = build_package.build_simple_package_variants(
                "amdrocm-developer-tools",
                cfg,
            )
            mock_versioned.assert_called_once()
            mock_nonversioned.assert_called_once()
            self.assertEqual(built, ["v.rpm", "nv.rpm"])

    # --rpath-pkg skips non-versioned on simple kpack path.
    @patch.object(build_package, "cleanup_build_directory")
    @patch.object(build_package, "create_nonversioned_deb_package")
    @patch.object(build_package, "create_versioned_deb_package", return_value=["v.deb"])
    def test_rpath_skips_nonversioned_simple_variant(
        self, mock_versioned, mock_nonversioned, _mock_cleanup
    ):
        """``enable_rpath=True`` on simple kpack path emits only the versioned package."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(Path(tmp), enable_rpath=True)
            built = build_package.build_simple_package_variants(
                "amdrocm-developer-tools",
                cfg,
            )
            mock_versioned.assert_called_once()
            mock_nonversioned.assert_not_called()
            self.assertEqual(built, ["v.deb"])


# ---------------------------------------------------------------------------
# build_singlearch_package_variants — non-kpack single-target builds
# ---------------------------------------------------------------------------
class BuildSinglearchPackageVariantsTest(unittest.TestCase):
    """``build_singlearch_package_variants`` — non-kpack (single ``--target``) builds.

    Use case: legacy single-GPU packaging produces one versioned and one non-versioned
    variant for the selected architecture (``gfx_arch`` from first ``--target``).
    """

    @patch.object(build_package, "cleanup_build_directory")
    @patch.object(
        build_package, "create_nonversioned_deb_package", return_value=["nv.deb"]
    )
    # Single-arch DEB builds versioned + non-versioned.
    @patch.object(build_package, "create_versioned_deb_package", return_value=["v.deb"])
    def test_deb_single_arch_builds_both_variants(
        self, mock_versioned, mock_nonversioned, _mock_cleanup
    ):
        """Single-arch DEB: ``enable_kpack=False``, ``gfx_arch=gfx1100`` → versioned + non-versioned."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(
                Path(tmp),
                enable_kpack=False,
                gfx_arch="gfx1100",
                pkg_type="deb",
            )
            built = build_package.build_singlearch_package_variants(
                "amdrocm-core-sdk",
                cfg,
            )
            mock_versioned.assert_called_once()
            mock_nonversioned.assert_called_once()
            self.assertEqual(built, ["v.deb", "nv.deb"])

    @patch.object(build_package, "cleanup_build_directory")
    @patch.object(
        build_package, "create_nonversioned_rpm_package", return_value=["nv.rpm"]
    )
    # Single-arch RPM builds versioned + non-versioned.
    @patch.object(build_package, "create_versioned_rpm_package", return_value=["v.rpm"])
    def test_rpm_single_arch_builds_both_variants(
        self, mock_versioned, mock_nonversioned, _mock_cleanup
    ):
        """Single-arch RPM: same as DEB but via ``create_*_rpm_package``."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(
                Path(tmp),
                enable_kpack=False,
                gfx_arch="gfx1100",
                pkg_type="rpm",
            )
            built = build_package.build_singlearch_package_variants(
                "amdrocm-core-sdk",
                cfg,
            )
            mock_versioned.assert_called_once()
            mock_nonversioned.assert_called_once()
            self.assertEqual(built, ["v.rpm", "nv.rpm"])

    # --rpath-pkg skips non-versioned on single-arch path.
    @patch.object(build_package, "cleanup_build_directory")
    @patch.object(build_package, "create_nonversioned_deb_package")
    @patch.object(build_package, "create_versioned_deb_package", return_value=["v.deb"])
    def test_rpath_skips_nonversioned_single_arch(
        self, mock_versioned, mock_nonversioned, _mock_cleanup
    ):
        """``enable_rpath=True`` in single-arch mode skips non-versioned output."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(
                Path(tmp),
                enable_kpack=False,
                gfx_arch="gfx1100",
                enable_rpath=True,
            )
            built = build_package.build_singlearch_package_variants(
                "amdrocm-core-sdk",
                cfg,
            )
            mock_versioned.assert_called_once()
            mock_nonversioned.assert_not_called()
            self.assertEqual(built, ["v.deb"])


# ---------------------------------------------------------------------------
# Low-level build_*_package helpers — gfx_arch / pkg_type wiring
# ---------------------------------------------------------------------------
class VariantBuilderDispatchTest(unittest.TestCase):
    """Low-level ``build_*_package`` helpers — ``gfx_arch`` and ``pkg_type`` wiring.

    Use case: each variant builder must pass the correct ``PackageConfig`` fields
    (``GFX_HOST``, device arch, ``GFX_META``, ``versioned_pkg``) into the DEB/RPM
    creator for dependency naming and artifact selection.
    """

    @patch.object(
        build_package, "create_versioned_deb_package", return_value=["host.deb"]
    )
    # build_host_package sets gfx_arch=GFX_HOST for DEB.
    def test_build_host_package_deb(self, mock_create):
        """``build_host_package`` sets ``gfx_arch=GFX_HOST`` and ``versioned_pkg=True``."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(Path(tmp), pkg_type="deb")
            result = build_package.build_host_package("amdrocm-ck", cfg)
            self.assertEqual(result, ["host.deb"])
            passed = mock_create.call_args[0][1]
            self.assertEqual(passed.gfx_arch, GFX_HOST)
            self.assertTrue(passed.versioned_pkg)

    @patch.object(
        build_package, "create_versioned_rpm_package", return_value=["dev.rpm"]
    )
    # build_device_package sets per-device gfx_arch for RPM.
    def test_build_device_package_rpm(self, mock_create):
        """``build_device_package(..., device_arch)`` sets ``gfx_arch`` to the device token."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(Path(tmp), pkg_type="rpm")
            result = build_package.build_device_package("amdrocm-ck", cfg, "gfx1100")
            self.assertEqual(result, ["dev.rpm"])
            passed = mock_create.call_args[0][1]
            self.assertEqual(passed.gfx_arch, "gfx1100")

    @patch.object(
        build_package, "create_versioned_deb_package", return_value=["meta.deb"]
    )
    # build_meta_package sets gfx_arch=GFX_META for DEB.
    def test_build_meta_package_deb(self, mock_create):
        """``build_meta_package`` sets ``gfx_arch=GFX_META`` for the versioned meta DEB."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(Path(tmp), pkg_type="deb")
            build_package.build_meta_package("amdrocm-ck", cfg)
            passed = mock_create.call_args[0][1]
            self.assertEqual(passed.gfx_arch, GFX_META)

    @patch.object(
        build_package, "create_versioned_deb_package", return_value=["ver.deb"]
    )
    # build_versioned_package clears gfx_arch on simple path.
    def test_build_versioned_package_clears_gfx_arch(self, mock_create):
        """``build_versioned_package`` (non-gfx simple path) clears ``gfx_arch`` to ``""``."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(Path(tmp), gfx_arch="gfx1100")
            build_package.build_versioned_package("amdrocm-developer-tools", cfg)
            passed = mock_create.call_args[0][1]
            self.assertEqual(passed.gfx_arch, "")

    @patch.object(
        build_package, "create_nonversioned_rpm_package", return_value=["nv.rpm"]
    )
    # build_nonversioned_package sets versioned_pkg=False for RPM.
    def test_build_nonversioned_package_rpm(self, mock_create):
        """``build_nonversioned_package`` sets ``versioned_pkg=False``; preserves caller ``gfx_arch``."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(Path(tmp), pkg_type="rpm", gfx_arch=GFX_META)
            build_package.build_nonversioned_package("amdrocm-ck", cfg)
            passed = mock_create.call_args[0][1]
            self.assertFalse(passed.versioned_pkg)
            self.assertEqual(passed.gfx_arch, GFX_META)


# ---------------------------------------------------------------------------
# main() / run() — CLI orchestration entry points
# ---------------------------------------------------------------------------
class MainAndRunTest(unittest.TestCase):
    """``main(argv)`` and ``run(args)`` — end-to-end CLI orchestration (mocked builds).

    Use case: verify the public entry points wire argparse → config → package list →
    per-package ``build_package_variants`` without running a full packaging job.
    """

    # main() parses argv and delegates to run().
    def test_main_delegates_to_run(self):
        """``main`` parses required flags and passes the resulting namespace to ``run``.

        CLI exercised: ``--artifacts-dir``, ``--dest-dir``, ``--pkg-type deb``,
        ``--rocm-version 7.1.0``, ``--target gfx1100``.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            art = root / "artifacts"
            out = root / "output"
            art.mkdir()
            out.mkdir()
            argv = [
                "--artifacts-dir",
                str(art),
                "--dest-dir",
                str(out),
                "--pkg-type",
                "deb",
                "--rocm-version",
                "7.1.0",
                "--target",
                "gfx1100",
            ]
            with patch.object(build_package, "run") as mock_run:
                build_package.main(argv)
            mock_run.assert_called_once()
            passed = mock_run.call_args[0][0]
            self.assertEqual(passed.pkg_type, "deb")
            self.assertEqual(passed.rocm_version, "7.1.0")

    @patch.object(build_package, "print_build_summary")
    @patch.object(build_package, "cleanup_packaging_environment")
    @patch.object(build_package, "build_package_variants", return_value=["out.deb"])
    @patch.object(
        build_package,
        "parse_input_package_list",
        return_value=(["amdrocm-core-sdk"], []),
    )
    # run() builds each package from resolved --pkg-names.
    @patch.object(build_package, "create_package_config")
    def test_run_builds_requested_packages(
        self,
        mock_create_cfg,
        mock_parse_list,
        mock_build_variants,
        _mock_cleanup,
        _mock_summary,
    ):
        """``run`` calls ``build_package_variants`` once per resolved ``--pkg-names`` entry.

        Args: ``pkg_names=["amdrocm-core-sdk"]``, ``pkg_type=deb``.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = _config(root)
            mock_create_cfg.return_value = cfg
            args = _args(root, pkg_names=["amdrocm-core-sdk"], pkg_type="deb")
            build_package.run(args)
            mock_parse_list.assert_called_once_with(
                ["amdrocm-core-sdk"], cfg.artifacts_dir
            )
            mock_build_variants.assert_called_once_with("amdrocm-core-sdk", cfg)

    # run() exits 1 when package list is empty.
    @patch.object(build_package, "print_build_summary")
    @patch.object(build_package, "cleanup_packaging_environment")
    @patch.object(build_package, "parse_input_package_list", return_value=([], []))
    @patch.object(build_package, "create_package_config")
    def test_run_exits_when_package_list_empty(
        self,
        mock_create_cfg,
        _mock_parse_list,
        _mock_cleanup,
        _mock_summary,
    ):
        """``run`` exits with code 1 when no packages remain after list resolution."""
        with tempfile.TemporaryDirectory() as tmp:
            mock_create_cfg.return_value = _config(Path(tmp))
            args = _args(Path(tmp))
            with self.assertRaises(SystemExit) as ctx:
                build_package.run(args)
            self.assertEqual(ctx.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
