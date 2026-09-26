#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``build_package.py`` using API-driven fixtures and real outputs.

Tests stage artifact trees by walking ``package.json`` via ``get_package_info()`` and
the same ``{Artifact}_{Component}_{suffix}`` layout consumed by
``filter_components_fromartifactory``. Config is built via ``create_package_config()``.
DEB generation runs through real ``create_versioned_deb_package()``; RPM generation
runs through real ``create_versioned_rpm_package()``; only ``package_with_dpkg_build`` /
``package_with_rpmbuild`` and ``move_packages_to_destination`` are mocked.

Run::

    python3.12 build_tools/packaging/linux/tests/build_package_test.py -v

Requires Python 3.10+ (``packaging_utils`` type syntax).
"""

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

THIS_SCRIPT_DIR = Path(__file__).resolve().parent
LINUX_DIR = THIS_SCRIPT_DIR.parent
BUILD_TOOLS_DIR = LINUX_DIR.parent.parent

# Test fixture defaults (avoid unexplained literals in assertions and helpers).
TEST_ROCM_VERSION = "7.1.0"
TEST_VERSION_SUFFIX = "daily"
TEST_INSTALL_PREFIX = "/opt/rocm/core"
TEST_GFX_TARGET = "gfx1100"
TEST_GFX_TARGET_ALT = "gfx942"
TEST_PKG_TYPE_DEB = "deb"
TEST_PKG_TYPE_RPM = "rpm"
TEST_BUILD_ARCH = "x86_64"
EMPTY_GFX_ARCH = ""

PKG_FFT = "amdrocm-fft"
PKG_CORE_SDK = "amdrocm-core-sdk"
PKG_DEVELOPER_TOOLS = "amdrocm-developer-tools"
PKG_RUNTIME = "amdrocm-runtime"
PKG_DEBUGGER = "amdrocm-debugger"
PKG_CK = "amdrocm-ck"

FFT_HOST_PACKAGE = "amdrocm-fft-host7.1"
FFT_DEVICE_PACKAGE = "amdrocm-fft7.1-gfx1100"
FFT_META_PACKAGE = "amdrocm-fft7.1"
CORE_SDK_DEVICE_PACKAGE = "amdrocm-core-sdk7.1-gfx1100"
DEVELOPER_TOOLS_PACKAGE = "amdrocm-developer-tools7.1"

STAGING_PAYLOAD_NAME = "libdummy.so"
STAGING_PAYLOAD_BYTES = b"\x00"


def _setup_import_path() -> None:
    """Add packaging paths so modules resolve from any working directory."""
    for path in (BUILD_TOOLS_DIR, LINUX_DIR):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _load_module(name: str, path: Path) -> types.ModuleType:
    """Load a packaging script by path for unit tests, not CLI execution.

    Uses ``importlib`` so ``build_package`` / ``deb_package`` / ``rpm_package``
    resolve from ``LINUX_DIR`` regardless of cwd, without invoking each script's
    ``main()`` or ``if __name__ == \"__main__\"`` entry point.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_setup_import_path()
build_package = _load_module("build_package", LINUX_DIR / "build_package.py")
deb_package = _load_module("deb_package", LINUX_DIR / "deb_package.py")
rpm_package = _load_module("rpm_package", LINUX_DIR / "rpm_package.py")

# packaging_utils depends on sys.path setup above.
from packaging_utils import (  # noqa: E402
    GFX_HOST,
    GFX_META,
    PackageConfig,
    filter_components_fromartifactory,
    get_package_info,
    is_gfxarch_package,
    is_key_defined,
    process_devel_dependencies_kpack,
    process_nonversioned_dependencies,
    update_package_name,
    PackageCollisionError,
    validate_package_roots,
    expand_kpack_meta_dependencies,
    process_main_dependencies_kpack,
)


class BuildPackageTestCase(unittest.TestCase):
    """Base test case with a per-test temporary directory."""

    def setUp(self) -> None:
        self._temp_context = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._temp_context.name)

    def tearDown(self) -> None:
        self._temp_context.cleanup()

    def artifacts_dir(self) -> Path:
        """Return the artifact root under the temp directory."""
        artifacts = self.temp_dir / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        return artifacts


def _args(tmp: Path, **overrides: object) -> Namespace:
    """Build an ``argparse.Namespace`` mirroring ``build_package.py`` CLI flags."""
    artifacts = tmp / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    defaults: dict[str, object] = {
        "artifacts_dir": artifacts,
        "dest_dir": tmp / "output",
        "target": [TEST_GFX_TARGET, TEST_GFX_TARGET_ALT],
        "pkg_type": TEST_PKG_TYPE_DEB,
        "rocm_version": TEST_ROCM_VERSION,
        "version_suffix": TEST_VERSION_SUFFIX,
        "install_prefix": TEST_INSTALL_PREFIX,
        "runpath_pkg": False,
        "enable_kpack": False,
        "pkg_names": None,
        "clean_build": False,
        "build_variant": "",
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def _write_kpack_manifest(artifacts_dir: Path) -> None:
    """Write ``therock_manifest.json`` with ``KPACK_SPLIT_ARTIFACTS`` enabled."""
    manifest_dir = artifacts_dir / "pkg"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "therock_manifest.json"
    manifest_path.write_text(
        json.dumps({"flags": {"KPACK_SPLIT_ARTIFACTS": True}}),
        encoding="utf-8",
    )
    if not manifest_path.exists():
        raise RuntimeError(f"Failed to write kpack manifest: {manifest_path}")


def _metadata_field(metadata_text: str, field: str) -> str:
    """Return the value of a DEB control or RPM spec field (e.g. ``Package`` / ``Name``)."""
    prefix = f"{field}:"
    for line in metadata_text.splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    raise AssertionError(
        f"Field {field!r} not found in metadata file:\n{metadata_text}"
    )


def _control_field(control_text: str, field: str) -> str:
    """Return the value of a ``debian/control`` field (e.g. ``Package``)."""
    return _metadata_field(control_text, field)


def _spec_field(spec_text: str, field: str) -> str:
    """Return the value of an RPM ``specfile`` field (e.g. ``Name``)."""
    return _metadata_field(spec_text, field)


def _artifact_suffix_for_staging(
    pkg_info: dict[str, object],
    artifact: dict[str, object],
    gfx_arch: str,
    *,
    enable_kpack: bool,
    artifacts_dir: Path,
) -> str | None:
    """Compute artifact directory suffix using packaging naming rules.

    Duplicates suffix routing from ``filter_components_fromartifactory`` so staged
    trees use ``{Artifact}_{Component}_{suffix}`` paths production expects. If that
    function's kpack/gfxarch rules change, update this helper in the same change;
    otherwise tests may stage the wrong layout and still pass (false positive).
    ``ArtifactStagingTest`` catches gross mismatches but not every routing edge case.
    """
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
    payload_name: str = STAGING_PAYLOAD_NAME,
) -> list[Path]:
    """Stage artifacts from ``package.json`` and return their manifest stage roots."""
    pkg_info = get_package_info(pkg_name)
    if enable_kpack and gfx_arch == GFX_META:
        return []

    created: list[Path] = []
    artifactory = pkg_info.get("Artifactory", [])
    if not isinstance(artifactory, list):
        return created

    for artifact in artifactory:
        if not isinstance(artifact, dict):
            continue
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
        subdirs = artifact.get("Artifact_Subdir", [])
        if not isinstance(subdirs, list):
            continue

        for subdir in subdirs:
            if not isinstance(subdir, dict):
                continue
            subdir_name = subdir["Name"]
            components = subdir.get("Components", [])
            if not isinstance(components, list):
                continue

            for component in components:
                artifact_dir = artifacts_dir / f"{prefix}_{component}_{suffix}"
                rel_path = f"{subdir_name}/{component}/{payload_name}"
                payload = artifact_dir / rel_path
                payload.parent.mkdir(parents=True, exist_ok=True)
                payload.write_bytes(STAGING_PAYLOAD_BYTES)
                manifest = artifact_dir / "artifact_manifest.txt"
                # Artifact manifests list stage roots, not individual files.
                manifest.write_text(f"{subdir_name}/{component}/\n", encoding="utf-8")
                if not manifest.exists():
                    raise RuntimeError(f"Failed to write artifact manifest: {manifest}")
                created.append(artifact_dir / subdir_name / component)
    return created


def _kpack_config(tmp: Path, **overrides: object) -> PackageConfig:
    """Build kpack ``PackageConfig`` via ``create_package_config`` (not hand-built)."""
    root = Path(tmp)
    _write_kpack_manifest(root / "artifacts")
    args = _args(root, enable_kpack=True, **overrides)
    return build_package.create_package_config(args)


def _control_path(pkg_name: str, config: PackageConfig) -> Path:
    """Return path to generated ``debian/control`` for a versioned DEB build."""
    updated = update_package_name(pkg_name, replace(config, versioned_pkg=True))
    return Path(config.dest_dir) / config.pkg_type / updated / "debian" / "control"


def _read_control_file(pkg_name: str, config: PackageConfig) -> str:
    """Read generated ``debian/control`` after validating it was created."""
    control_path = _control_path(pkg_name=pkg_name, config=config)
    if not control_path.exists():
        raise AssertionError(f"Expected control file was not created: {control_path}")
    return control_path.read_text(encoding="utf-8")


def _spec_path(pkg_name: str, config: PackageConfig) -> Path:
    """Return path to generated RPM ``specfile`` for a versioned RPM build."""
    updated = update_package_name(pkg_name, replace(config, versioned_pkg=True))
    return Path(config.dest_dir) / config.pkg_type / updated / "specfile"


def _read_spec_file(pkg_name: str, config: PackageConfig) -> str:
    """Read generated RPM ``specfile`` after validating it was created."""
    spec_path = _spec_path(pkg_name=pkg_name, config=config)
    if not spec_path.exists():
        raise AssertionError(f"Expected spec file was not created: {spec_path}")
    return spec_path.read_text(encoding="utf-8")


def _stage_fft_kpack_tree(artifacts_dir: Path, *, include_host: bool = False) -> None:
    """Stage FFT artifacts for kpack tests; optionally include host/generic tree."""
    _stage_package_artifacts(
        pkg_name=PKG_FFT,
        artifacts_dir=artifacts_dir,
        gfx_arch=TEST_GFX_TARGET,
        enable_kpack=True,
    )
    if include_host:
        _stage_package_artifacts(
            pkg_name=PKG_FFT,
            artifacts_dir=artifacts_dir,
            gfx_arch=GFX_HOST,
            enable_kpack=True,
        )


# ---------------------------------------------------------------------------
# Artifact staging — validates production discovery APIs
# ---------------------------------------------------------------------------
class SharedOwnerPackagingTest(BuildPackageTestCase):
    """Keep selected payload identities inside one native package per owner."""

    def _stage_targets(
        self, targets: tuple[str, ...], pkg_type: str = "deb"
    ) -> PackageConfig:
        cfg = _kpack_config(
            self.temp_dir / ";".join(targets) / pkg_type,
            target=list(targets),
            pkg_type=pkg_type,
        )
        # Supply both targets to ensure selection does not implicitly include the other.
        self.stage_roots: dict[str, list[Path]] = {}
        for target in ("gfx1250", "gfx1250-strict"):
            self.stage_roots[target] = _stage_package_artifacts(
                PKG_FFT,
                cfg.artifacts_dir,
                target,
                payload_name=f".kpack/fft_{target}.kpack",
            )
        return replace(cfg, gfx_arch="gfx1250", versioned_pkg=True)

    @patch.object(deb_package, "move_packages_to_destination", return_value=[])
    @patch.object(deb_package, "package_with_dpkg_build")
    def test_deb_retains_only_selected_payloads(self, mock_build, _mock_move):
        for targets in (
            ("gfx1250",),
            ("gfx1250-strict",),
            ("gfx1250", "gfx1250-strict"),
        ):
            with self.subTest(targets=targets):
                cfg = self._stage_targets(targets)
                deb_package.create_versioned_deb_package(PKG_FFT, cfg)
                control = _read_control_file(PKG_FFT, cfg)
                self.assertEqual(
                    _control_field(control, "Package"), "amdrocm-fft7.1-gfx1250"
                )
                package_dir = mock_build.call_args.args[0]
                payload_dir = package_dir / cfg.install_prefix.lstrip("/") / ".kpack"
                self.assertEqual(
                    {path.name for path in payload_dir.iterdir()},
                    {f"fft_{target}.kpack" for target in targets},
                )
                for target in targets:
                    self.assertEqual(
                        (payload_dir / f"fft_{target}.kpack").read_bytes(),
                        STAGING_PAYLOAD_BYTES,
                    )

    @patch.object(rpm_package, "move_packages_to_destination", return_value=[])
    @patch.object(rpm_package, "package_with_rpmbuild")
    def test_rpm_spec_includes_only_selected_roots(self, _mock_build, _mock_move):
        for targets in (
            ("gfx1250",),
            ("gfx1250-strict",),
            ("gfx1250", "gfx1250-strict"),
        ):
            with self.subTest(targets=targets):
                cfg = self._stage_targets(targets, "rpm")
                rpm_package.create_versioned_rpm_package(PKG_FFT, cfg)
                spec = _read_spec_file(PKG_FFT, cfg)
                self.assertEqual(_spec_field(spec, "Name"), "amdrocm-fft7.1-gfx1250")
                for target in ("gfx1250", "gfx1250-strict"):
                    artifact_path = (
                        str(cfg.artifacts_dir / f"fft_lib_{target}") + os.sep
                    )
                    self.assertEqual(artifact_path in spec, target in targets)

    @patch.object(build_package, "build_nonversioned_package", return_value=[])
    @patch.object(build_package, "build_meta_package", return_value=[])
    @patch.object(build_package, "build_host_package", return_value=[])
    @patch.object(build_package, "build_device_package", return_value=[])
    def test_device_package_built_once(self, device, _host, _meta, _nonversioned):
        cfg = self._stage_targets(("gfx1250", "gfx1250-strict"))
        build_package.build_gfxarch_package_variants(PKG_FFT, cfg)
        device.assert_called_once_with(PKG_FFT, cfg, "gfx1250")

    def test_meta_dependency_lists_owner_once(self):
        cfg = self._stage_targets(("gfx1250", "gfx1250-strict"))
        self.assertEqual(
            expand_kpack_meta_dependencies(PKG_FFT, cfg.gfxarch_list, cfg),
            ["amdrocm-fft-host7.1", "amdrocm-fft7.1-gfx1250"],
        )

    def test_device_dependency_uses_selected_member_availability(self):
        cfg = self._stage_targets(("gfx1250-strict",))
        _stage_package_artifacts("amdrocm-rand", cfg.artifacts_dir, "gfx1250-strict")
        _stage_package_artifacts(
            "amdrocm-rocalution", cfg.artifacts_dir, "gfx1250-strict"
        )
        info = get_package_info("amdrocm-rocalution")
        for field in ("DEBDepends", "RPMRequires"):
            with self.subTest(field=field):
                deps = process_main_dependencies_kpack(info, field, cfg)
                self.assertIn("amdrocm-rand7.1-gfx1250", deps)
                self.assertNotIn("gfx1250-strict", deps)

    def _stage_conflicting_targets(self, pkg_type: str) -> PackageConfig:
        cfg = self._stage_targets(("gfx1250", "gfx1250-strict"), pkg_type)
        for target, roots in self.stage_roots.items():
            for root in roots:
                (root / "shared").write_text(target)
        return cfg

    @patch.object(
        build_package,
        "create_versioned_deb_package",
        deb_package.create_versioned_deb_package,
    )
    @patch.object(deb_package, "package_with_dpkg_build")
    def test_deb_conflict_propagates_before_building(self, external_build):
        cfg = self._stage_conflicting_targets("deb")
        with self.assertRaisesRegex(
            PackageCollisionError, "Conflicting package path shared"
        ):
            build_package.build_device_package(PKG_FFT, cfg, "gfx1250")
        external_build.assert_not_called()

    @patch.object(
        build_package,
        "create_versioned_rpm_package",
        rpm_package.create_versioned_rpm_package,
    )
    @patch.object(rpm_package, "package_with_rpmbuild")
    def test_rpm_conflict_propagates_before_building(self, external_build):
        cfg = self._stage_conflicting_targets("rpm")
        with self.assertRaisesRegex(
            PackageCollisionError, "Conflicting package path shared"
        ):
            build_package.build_device_package(PKG_FFT, cfg, "gfx1250")
        external_build.assert_not_called()


class PackageFailureExitTest(BuildPackageTestCase):
    def _run_failing_build(self, error: Exception) -> None:
        config = _kpack_config(self.temp_dir)
        _stage_fft_kpack_tree(config.artifacts_dir)
        args = _args(self.temp_dir, runpath_pkg=True, pkg_names=[PKG_FFT], parallel=1)
        # Exercise the real worker and orchestration without external package builds.
        with (
            patch.object(build_package, "create_package_config", return_value=config),
            patch.object(build_package, "cleanup_packaging_environment"),
            patch.object(build_package, "ProcessPoolExecutor", ThreadPoolExecutor),
            patch.object(build_package, "build_package_variants", side_effect=error),
        ):
            build_package.run(args)

    def test_collision_exits_nonzero(self):
        with self.assertRaises(SystemExit) as raised:
            self._run_failing_build(
                PackageCollisionError("Conflicting package path shared")
            )
        self.assertEqual(raised.exception.code, 1)

    def test_other_failure_keeps_existing_exit_behavior(self):
        self._run_failing_build(RuntimeError("Package build failed"))


class PackageRootCollisionTest(BuildPackageTestCase):
    def setUp(self):
        super().setUp()
        self.roots = [self.temp_dir / "first", self.temp_dir / "second"]
        for root in self.roots:
            root.mkdir()

    def test_file_directory_collision_is_rejected(self):
        (self.roots[0] / "shared").write_bytes(b"file")
        (self.roots[1] / "shared").mkdir()
        with self.assertRaisesRegex(
            PackageCollisionError, "Conflicting package path shared"
        ):
            validate_package_roots(self.roots)

    def test_identical_dangling_symlinks_are_accepted(self):
        for root in self.roots:
            (root / "shared").symlink_to("external-target")
        validate_package_roots(self.roots)

    def test_conflicting_symlinks_are_rejected(self):
        (self.roots[0] / "shared").symlink_to("first-target")
        (self.roots[1] / "shared").symlink_to("second-target")
        with self.assertRaisesRegex(
            PackageCollisionError, "Conflicting package path shared"
        ):
            validate_package_roots(self.roots)


class ArtifactStagingTest(BuildPackageTestCase):
    """``_stage_package_artifacts`` produces trees ``filter_components`` accepts."""

    def test_staged_fft_host_artifacts_discovered(self) -> None:
        artifacts = self.artifacts_dir()
        _stage_package_artifacts(
            pkg_name=PKG_FFT,
            artifacts_dir=artifacts,
            gfx_arch=GFX_HOST,
            enable_kpack=True,
        )
        sourcedirs = filter_components_fromartifactory(
            pkg_name=PKG_FFT,
            artifacts_dir=artifacts,
            gfx_arch=GFX_HOST,
            enable_kpack=True,
        )
        self.assertTrue(sourcedirs)

    def test_staged_fft_device_artifacts_discovered(self) -> None:
        artifacts = self.artifacts_dir()
        _stage_package_artifacts(
            pkg_name=PKG_FFT,
            artifacts_dir=artifacts,
            gfx_arch=TEST_GFX_TARGET,
            enable_kpack=True,
        )
        sourcedirs = filter_components_fromartifactory(
            pkg_name=PKG_FFT,
            artifacts_dir=artifacts,
            gfx_arch=TEST_GFX_TARGET,
            enable_kpack=True,
        )
        self.assertTrue(sourcedirs)
        self.assertTrue(
            is_gfxarch_package(
                pkg_info=get_package_info(PKG_FFT),
                enable_kpack=True,
                artifacts_dir=artifacts,
            )
        )


# ---------------------------------------------------------------------------
# create_versioned_deb_package — real control file generation
# ---------------------------------------------------------------------------
class CreateVersionedDebPackageTest(BuildPackageTestCase):
    """Real ``create_versioned_deb_package`` with API-staged artifacts."""

    @patch.object(deb_package, "move_packages_to_destination", return_value=[])
    @patch.object(deb_package, "package_with_dpkg_build")
    def test_fft_host_generates_control_file(
        self, _mock_dpkg: object, _mock_move: object
    ) -> None:
        cfg = _kpack_config(self.temp_dir)
        # Host naming requires gfx dirs on disk (#5874); runtime dep needs artifacts
        # in pkg_list for convert_to_versiondependency.
        _stage_package_artifacts(
            pkg_name=PKG_FFT,
            artifacts_dir=cfg.artifacts_dir,
            gfx_arch=GFX_HOST,
            enable_kpack=True,
        )
        _stage_package_artifacts(
            pkg_name=PKG_FFT,
            artifacts_dir=cfg.artifacts_dir,
            gfx_arch=TEST_GFX_TARGET,
            enable_kpack=True,
        )
        _stage_package_artifacts(
            pkg_name=PKG_RUNTIME,
            artifacts_dir=cfg.artifacts_dir,
            gfx_arch=GFX_HOST,
            enable_kpack=True,
        )
        host_cfg = replace(cfg, gfx_arch=GFX_HOST)

        deb_package.create_versioned_deb_package(pkg_name=PKG_FFT, config=host_cfg)

        control = _read_control_file(pkg_name=PKG_FFT, config=host_cfg)
        self.assertEqual(_control_field(control, "Package"), FFT_HOST_PACKAGE)
        self.assertEqual(_control_field(control, "Architecture"), "amd64")
        self.assertIn("amdrocm-runtime", _control_field(control, "Depends"))

    @patch.object(deb_package, "move_packages_to_destination", return_value=[])
    @patch.object(deb_package, "package_with_dpkg_build")
    def test_fft_device_generates_control_file(
        self, _mock_dpkg: object, _mock_move: object
    ) -> None:
        cfg = _kpack_config(self.temp_dir)
        _stage_package_artifacts(
            pkg_name=PKG_FFT,
            artifacts_dir=cfg.artifacts_dir,
            gfx_arch=TEST_GFX_TARGET,
            enable_kpack=True,
        )
        device_cfg = replace(cfg, gfx_arch=TEST_GFX_TARGET)

        deb_package.create_versioned_deb_package(pkg_name=PKG_FFT, config=device_cfg)

        control = _read_control_file(pkg_name=PKG_FFT, config=device_cfg)
        self.assertEqual(_control_field(control, "Package"), FFT_DEVICE_PACKAGE)

    @patch.object(deb_package, "move_packages_to_destination", return_value=[])
    @patch.object(deb_package, "package_with_dpkg_build")
    def test_fft_meta_generates_control_file(
        self, _mock_dpkg: object, _mock_move: object
    ) -> None:
        cfg = _kpack_config(self.temp_dir)
        meta_cfg = replace(cfg, gfx_arch=GFX_META)

        deb_package.create_versioned_deb_package(pkg_name=PKG_FFT, config=meta_cfg)

        control = _read_control_file(pkg_name=PKG_FFT, config=meta_cfg)
        self.assertEqual(_control_field(control, "Package"), FFT_META_PACKAGE)

    @patch.object(deb_package, "move_packages_to_destination", return_value=[])
    @patch.object(deb_package, "package_with_dpkg_build")
    def test_core_sdk_device_meta_generates_control_file(
        self, _mock_dpkg: object, _mock_move: object
    ) -> None:
        """Gfx-arch metapackage ``amdrocm-core-sdk`` device variant (#6093)."""
        cfg = _kpack_config(self.temp_dir)
        device_cfg = replace(cfg, gfx_arch=TEST_GFX_TARGET)

        deb_package.create_versioned_deb_package(
            pkg_name=PKG_CORE_SDK, config=device_cfg
        )

        control = _read_control_file(pkg_name=PKG_CORE_SDK, config=device_cfg)
        self.assertEqual(_control_field(control, "Package"), CORE_SDK_DEVICE_PACKAGE)
        # debian_replace_devel_name maps -devel → -dev in DEB Depends.
        self.assertIn("amdrocm-core-dev", _control_field(control, "Depends"))

    @patch.object(deb_package, "move_packages_to_destination", return_value=[])
    @patch.object(deb_package, "package_with_dpkg_build")
    def test_developer_tools_versioned_metapackage_control(
        self, _mock_dpkg: object, _mock_move: object
    ) -> None:
        """Simple kpack metapackage with no Artifactory entries."""
        cfg = _kpack_config(self.temp_dir)
        # convert_to_versiondependency only keeps amdrocm deps present in pkg_list.
        _stage_package_artifacts(
            pkg_name=PKG_DEBUGGER,
            artifacts_dir=cfg.artifacts_dir,
            gfx_arch=EMPTY_GFX_ARCH,
            enable_kpack=True,
        )
        versioned_cfg = replace(cfg, gfx_arch=EMPTY_GFX_ARCH)

        deb_package.create_versioned_deb_package(
            pkg_name=PKG_DEVELOPER_TOOLS, config=versioned_cfg
        )

        control = _read_control_file(pkg_name=PKG_DEVELOPER_TOOLS, config=versioned_cfg)
        self.assertEqual(_control_field(control, "Package"), DEVELOPER_TOOLS_PACKAGE)
        self.assertIn("amdrocm-debugger", _control_field(control, "Depends"))


# ---------------------------------------------------------------------------
# create_versioned_rpm_package — real spec file generation
# ---------------------------------------------------------------------------
class CreateVersionedRpmPackageTest(BuildPackageTestCase):
    """Real ``create_versioned_rpm_package`` with API-staged artifacts."""

    @patch.object(rpm_package, "move_packages_to_destination", return_value=[])
    @patch.object(rpm_package, "package_with_rpmbuild")
    def test_fft_host_generates_spec_file(
        self, _mock_rpmbuild: object, _mock_move: object
    ) -> None:
        cfg = _kpack_config(self.temp_dir, pkg_type=TEST_PKG_TYPE_RPM)
        _stage_package_artifacts(
            pkg_name=PKG_FFT,
            artifacts_dir=cfg.artifacts_dir,
            gfx_arch=GFX_HOST,
            enable_kpack=True,
        )
        _stage_package_artifacts(
            pkg_name=PKG_FFT,
            artifacts_dir=cfg.artifacts_dir,
            gfx_arch=TEST_GFX_TARGET,
            enable_kpack=True,
        )
        _stage_package_artifacts(
            pkg_name=PKG_RUNTIME,
            artifacts_dir=cfg.artifacts_dir,
            gfx_arch=GFX_HOST,
            enable_kpack=True,
        )
        host_cfg = replace(cfg, gfx_arch=GFX_HOST)

        rpm_package.create_versioned_rpm_package(pkg_name=PKG_FFT, config=host_cfg)

        spec = _read_spec_file(pkg_name=PKG_FFT, config=host_cfg)
        self.assertEqual(_spec_field(spec, "Name"), FFT_HOST_PACKAGE)
        self.assertEqual(_spec_field(spec, "BuildArch"), TEST_BUILD_ARCH)
        self.assertIn("amdrocm-runtime", _spec_field(spec, "Requires"))

    @patch.object(rpm_package, "move_packages_to_destination", return_value=[])
    @patch.object(rpm_package, "package_with_rpmbuild")
    def test_fft_device_generates_spec_file(
        self, _mock_rpmbuild: object, _mock_move: object
    ) -> None:
        cfg = _kpack_config(self.temp_dir, pkg_type=TEST_PKG_TYPE_RPM)
        _stage_package_artifacts(
            pkg_name=PKG_FFT,
            artifacts_dir=cfg.artifacts_dir,
            gfx_arch=TEST_GFX_TARGET,
            enable_kpack=True,
        )
        device_cfg = replace(cfg, gfx_arch=TEST_GFX_TARGET)

        rpm_package.create_versioned_rpm_package(pkg_name=PKG_FFT, config=device_cfg)

        spec = _read_spec_file(pkg_name=PKG_FFT, config=device_cfg)
        self.assertEqual(_spec_field(spec, "Name"), FFT_DEVICE_PACKAGE)

    @patch.object(rpm_package, "move_packages_to_destination", return_value=[])
    @patch.object(rpm_package, "package_with_rpmbuild")
    def test_fft_meta_generates_spec_file(
        self, _mock_rpmbuild: object, _mock_move: object
    ) -> None:
        cfg = _kpack_config(self.temp_dir, pkg_type=TEST_PKG_TYPE_RPM)
        meta_cfg = replace(cfg, gfx_arch=GFX_META)

        rpm_package.create_versioned_rpm_package(pkg_name=PKG_FFT, config=meta_cfg)

        spec = _read_spec_file(pkg_name=PKG_FFT, config=meta_cfg)
        self.assertEqual(_spec_field(spec, "Name"), FFT_META_PACKAGE)

    @patch.object(rpm_package, "move_packages_to_destination", return_value=[])
    @patch.object(rpm_package, "package_with_rpmbuild")
    def test_core_sdk_device_meta_generates_spec_file(
        self, _mock_rpmbuild: object, _mock_move: object
    ) -> None:
        """Gfx-arch metapackage ``amdrocm-core-sdk`` device variant (#6093)."""
        cfg = _kpack_config(self.temp_dir, pkg_type=TEST_PKG_TYPE_RPM)
        device_cfg = replace(cfg, gfx_arch=TEST_GFX_TARGET)

        rpm_package.create_versioned_rpm_package(
            pkg_name=PKG_CORE_SDK, config=device_cfg
        )

        spec = _read_spec_file(pkg_name=PKG_CORE_SDK, config=device_cfg)
        self.assertEqual(_spec_field(spec, "Name"), CORE_SDK_DEVICE_PACKAGE)
        # RPM keeps -devel naming (no debian_replace_devel_name mapping).
        self.assertIn("amdrocm-core-devel", _spec_field(spec, "Requires"))


# ---------------------------------------------------------------------------
# build_package_variants — routing with real artifact detection (#5874)
# ---------------------------------------------------------------------------
class BuildPackageVariantsRoutingTest(BuildPackageTestCase):
    """Top-level routing using real ``is_gfxarch_package`` / staged artifacts."""

    @patch.object(build_package, "build_gfxarch_package_variants", return_value=[])
    @patch.object(build_package, "build_simple_package_variants")
    def test_fft_with_staged_artifacts_routes_to_gfxarch(
        self, mock_simple: object, mock_gfxarch: object
    ) -> None:
        cfg = _kpack_config(self.temp_dir)
        _stage_fft_kpack_tree(cfg.artifacts_dir)
        build_package.build_package_variants(pkg_name=PKG_FFT, config=cfg)
        mock_gfxarch.assert_called_once_with(PKG_FFT, cfg)
        mock_simple.assert_not_called()

    @patch.object(build_package, "build_gfxarch_package_variants")
    @patch.object(build_package, "build_simple_package_variants", return_value=[])
    def test_fft_without_gfx_artifacts_routes_to_simple(
        self, mock_simple: object, mock_gfxarch: object
    ) -> None:
        """#5874: Gfxarch metadata alone must not trigger gfx splits without artifacts."""
        cfg = _kpack_config(self.temp_dir)
        pkg_info = get_package_info(PKG_FFT)
        self.assertFalse(
            is_gfxarch_package(
                pkg_info=pkg_info,
                enable_kpack=True,
                artifacts_dir=cfg.artifacts_dir,
            )
        )
        build_package.build_package_variants(pkg_name=PKG_FFT, config=cfg)
        mock_simple.assert_called_once_with(PKG_FFT, cfg)
        mock_gfxarch.assert_not_called()

    @patch.object(build_package, "build_gfxarch_package_variants", return_value=[])
    @patch.object(build_package, "build_simple_package_variants")
    def test_core_sdk_metapackage_routes_to_gfxarch(
        self, mock_simple: object, mock_gfxarch: object
    ) -> None:
        """Gfx-arch metapackage routes to gfxarch builder even without artifacts (#6093)."""
        cfg = _kpack_config(self.temp_dir)
        build_package.build_package_variants(pkg_name=PKG_CORE_SDK, config=cfg)
        mock_gfxarch.assert_called_once_with(PKG_CORE_SDK, cfg)
        mock_simple.assert_not_called()

    @patch.object(build_package, "build_gfxarch_package_variants")
    @patch.object(build_package, "build_simple_package_variants", return_value=[])
    def test_developer_tools_routes_to_simple(
        self, mock_simple: object, mock_gfxarch: object
    ) -> None:
        cfg = _kpack_config(self.temp_dir)
        build_package.build_package_variants(pkg_name=PKG_DEVELOPER_TOOLS, config=cfg)
        mock_simple.assert_called_once_with(PKG_DEVELOPER_TOOLS, cfg)
        mock_gfxarch.assert_not_called()


# ---------------------------------------------------------------------------
# create_package_config — CLI → PackageConfig (real function, no hand-built config)
# ---------------------------------------------------------------------------
class CreatePackageConfigTest(BuildPackageTestCase):
    """``create_package_config`` maps CLI args to ``PackageConfig``."""

    def test_explicit_targets_in_kpack_mode(self) -> None:
        cfg = build_package.create_package_config(
            _args(
                self.temp_dir,
                enable_kpack=True,
                target=[TEST_GFX_TARGET, TEST_GFX_TARGET_ALT],
            )
        )
        self.assertTrue(cfg.enable_kpack)
        self.assertEqual(cfg.gfxarch_list, (TEST_GFX_TARGET, TEST_GFX_TARGET_ALT))

    def test_auto_detect_kpack_from_manifest(self) -> None:
        _write_kpack_manifest(self.artifacts_dir())
        cfg = build_package.create_package_config(_args(self.temp_dir))
        self.assertTrue(cfg.enable_kpack)

    def test_invalid_rocm_version_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_package.create_package_config(
                _args(self.temp_dir, rocm_version="7"),
            )

    def test_asan_family_variants_get_asan_install_prefix(self) -> None:
        """asan, host-asan, and their "-debug" counterparts must all get the
        same "-asan-MAJOR.MINOR" install prefix suffix as full asan."""
        for build_variant in ("asan", "host-asan", "asan-debug", "host-asan-debug"):
            with self.subTest(build_variant=build_variant):
                cfg = build_package.create_package_config(
                    _args(
                        self.temp_dir,
                        build_variant=build_variant,
                        install_prefix=build_package.DEFAULT_INSTALL_PREFIX,
                    )
                )
                self.assertTrue(cfg.install_prefix.endswith("-asan-7.1"))

    def test_release_variant_gets_plain_install_prefix(self) -> None:
        cfg = build_package.create_package_config(
            _args(
                self.temp_dir,
                build_variant="release",
                install_prefix=build_package.DEFAULT_INSTALL_PREFIX,
            )
        )
        self.assertFalse("asan" in cfg.install_prefix)
        self.assertTrue(cfg.install_prefix.endswith("-7.1"))


# ---------------------------------------------------------------------------
# update_package_name
# ---------------------------------------------------------------------------
class UpdatePackageNameAsanSuffixTest(BuildPackageTestCase):
    """``update_package_name`` must add the "-asan" suffix for every ASan-family
    build_variant, including the "-debug" variants."""

    def test_asan_family_variants_get_asan_name_suffix(self) -> None:
        for build_variant in ("asan", "host-asan", "asan-debug", "host-asan-debug"):
            with self.subTest(build_variant=build_variant):
                cfg = build_package.create_package_config(
                    _args(self.temp_dir, build_variant=build_variant)
                )
                updated = update_package_name(
                    "amdrocm-core", replace(cfg, versioned_pkg=True)
                )
                self.assertEqual(updated, f"amdrocm-core-asan7.1-{TEST_GFX_TARGET}")

    def test_release_variant_gets_plain_name(self) -> None:
        cfg = build_package.create_package_config(
            _args(self.temp_dir, build_variant="release")
        )
        updated = update_package_name("amdrocm-core", replace(cfg, versioned_pkg=True))
        self.assertEqual(updated, f"amdrocm-core7.1-{TEST_GFX_TARGET}")


# ---------------------------------------------------------------------------
# load_kpack_from_manifest
# ---------------------------------------------------------------------------
class LoadKpackFromManifestTest(BuildPackageTestCase):
    """``load_kpack_from_manifest`` reads ``therock_manifest.json`` kpack flags."""

    def test_true_when_kpack_flag_set(self) -> None:
        _write_kpack_manifest(self.artifacts_dir())
        self.assertTrue(
            build_package.load_kpack_from_manifest(artifacts_dir=self.artifacts_dir())
        )

    def test_false_when_no_manifest(self) -> None:
        self.assertFalse(
            build_package.load_kpack_from_manifest(artifacts_dir=self.artifacts_dir())
        )


# ---------------------------------------------------------------------------
# parse_input_package_list — real package.json names
# ---------------------------------------------------------------------------
class ParseInputPackageListTest(BuildPackageTestCase):
    """``parse_input_package_list`` filters names against ``package.json``."""

    def test_explicit_pkg_names_filter_package_json(self) -> None:
        pkg_list, skipped = build_package.parse_input_package_list(
            pkg_name=[PKG_CORE_SDK, PKG_CK, "no-such-package"],
            artifact_dir=self.artifacts_dir(),
        )
        self.assertEqual(set(pkg_list), {PKG_CORE_SDK, PKG_CK})
        self.assertEqual(skipped, [])


# ---------------------------------------------------------------------------
# copy_package_contents — symlink handling
# ---------------------------------------------------------------------------
class CopyPackageContentsTest(BuildPackageTestCase):
    """``copy_package_contents`` preserves symlinks correctly.

    Regression test: in ROCm 10.1, symlinks pointing to valid directories
    (e.g., llvm -> lib/llvm) were incorrectly expanded via copytree because
    Path.is_dir() follows symlinks and returns True.
    """

    def test_identical_nested_entries_merge(self) -> None:
        destination = self.temp_dir / "dest"
        for name in ("first", "second"):
            source = self.temp_dir / name
            nested = source / ".kpack"
            nested.mkdir(parents=True)
            (nested / "shared").write_bytes(b"shared payload")
            (nested / "shared-link").symlink_to("shared")
            deb_package.copy_package_contents(source, destination)
        self.assertEqual(
            (destination / ".kpack/shared").read_bytes(), b"shared payload"
        )
        self.assertEqual(
            (destination / ".kpack/shared-link").readlink(), Path("shared")
        )

    def test_valid_symlink_to_directory_preserved(self) -> None:
        """Symlink to existing directory must remain a symlink.

        In ROCm 10.1, llvm -> lib/llvm points to a valid directory. Without
        the fix, is_dir() returns True and copytree expands it into a directory.
        """
        source = self.temp_dir / "source"
        dest = self.temp_dir / "dest"
        source.mkdir()

        # Create lib/llvm structure (the real directory)
        lib_llvm = source / "lib" / "llvm"
        lib_llvm.mkdir(parents=True)
        (lib_llvm / "bin").mkdir()
        (lib_llvm / "bin" / "clang").write_bytes(b"\x00")

        # Valid symlink: target exists
        (source / "llvm").symlink_to("lib/llvm")

        deb_package.copy_package_contents(source, dest)

        self.assertTrue(
            (dest / "llvm").is_symlink(),
            "valid symlink should be preserved, not expanded",
        )
        self.assertEqual((dest / "llvm").readlink(), Path("lib/llvm"))

    def test_dangling_symlink_preserved(self) -> None:
        """Dangling symlinks must be preserved.

        In ROCm 10.0, symlinks like amdgcn -> lib/llvm/amdgcn were dangling
        at copy time (target didn't exist yet).
        """
        source = self.temp_dir / "source"
        dest = self.temp_dir / "dest"
        source.mkdir()

        # Dangling symlink: target doesn't exist
        (source / "amdgcn").symlink_to("lib/llvm/amdgcn")

        deb_package.copy_package_contents(source, dest)

        self.assertTrue(
            (dest / "amdgcn").is_symlink(),
            "dangling symlink should be preserved",
        )
        self.assertEqual((dest / "amdgcn").readlink(), Path("lib/llvm/amdgcn"))


# ---------------------------------------------------------------------------
# Devel package dependency resolution (#6391)
# ---------------------------------------------------------------------------
class DevelDependencyResolutionTest(BuildPackageTestCase):
    """Devel and non-versioned package dependency resolution in kpack mode."""

    def test_devel_gfxarch_depends_on_full_meta(self) -> None:
        """GfxArch devel depends on full meta (e.g., amdrocm-fft7.1), not host."""
        cfg = _kpack_config(self.temp_dir)
        _stage_package_artifacts(
            pkg_name=PKG_FFT,
            artifacts_dir=cfg.artifacts_dir,
            gfx_arch=TEST_GFX_TARGET,
            enable_kpack=True,
        )
        devel_cfg = replace(cfg, gfx_arch=EMPTY_GFX_ARCH)

        pkg_info = get_package_info("amdrocm-fft-devel")
        deps = process_devel_dependencies_kpack(pkg_info, "DEBDepends", devel_cfg)

        self.assertIn("amdrocm-fft7.1", deps)
        self.assertNotIn("amdrocm-fft-host", deps)

    def test_devel_meta_includes_version_constraints(self) -> None:
        """Meta devel packages include version constraints on dependencies."""
        cfg = _kpack_config(self.temp_dir)
        devel_cfg = replace(cfg, gfx_arch=EMPTY_GFX_ARCH)

        pkg_info = get_package_info("amdrocm-core-devel")
        deps = process_devel_dependencies_kpack(pkg_info, "DEBDepends", devel_cfg)

        self.assertRegex(deps, r"\(.*=.*\)")

    def test_nonversioned_gfxarch_depends_on_versioned_meta(self) -> None:
        """Non-versioned gfxarch package depends on versioned meta in kpack mode."""
        cfg = _kpack_config(self.temp_dir)
        nonversioned_cfg = replace(cfg, versioned_pkg=False, gfx_arch=GFX_META)

        pkg_info = get_package_info(PKG_FFT)
        deps = process_nonversioned_dependencies(pkg_info, nonversioned_cfg)

        self.assertEqual(deps, "amdrocm-fft7.1")

    def test_nonversioned_devel_depends_on_versioned_devel(self) -> None:
        """Non-versioned devel depends on versioned devel in kpack mode."""
        cfg = _kpack_config(self.temp_dir)
        nonversioned_cfg = replace(cfg, versioned_pkg=False, gfx_arch=EMPTY_GFX_ARCH)

        pkg_info = get_package_info("amdrocm-fft-devel")
        deps = process_nonversioned_dependencies(pkg_info, nonversioned_cfg)

        self.assertEqual(deps, "amdrocm-fft-dev7.1")


if __name__ == "__main__":
    unittest.main()
