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
import re
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
    filter_dependencies_by_artifacts,
    get_package_info,
    has_artifact_for_arch,
    is_gfxarch_package,
    is_key_defined,
    is_meta_package,
    read_package_json_file,
    update_package_name,
    PackageCollisionError,
    validate_package_roots,
    expand_kpack_meta_dependencies,
    process_main_dependencies_kpack,
    _artifact_dir_name_index,
    has_generic_content,
    omits_generic_variants,
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
    dir_suffix_override: str | None = None,
) -> list[Path]:
    """Stage artifacts from ``package.json`` and return their manifest stage roots.

    ``dir_suffix_override``, when given, replaces the computed directory suffix
    (e.g. stage under ``..._gfx942:xnack+`` while still resolving dependencies
    for plain ``gfx_arch="gfx942"``) - used to reproduce components that are
    only ever built with an xnack-suffixed artifact directory.
    """
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
        if dir_suffix_override is not None:
            suffix = dir_suffix_override

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
                dir_path = f"{subdir_name}/{component}"
                payload = artifact_dir / dir_path / payload_name
                payload.parent.mkdir(parents=True, exist_ok=True)
                payload.write_bytes(STAGING_PAYLOAD_BYTES)
                manifest = artifact_dir / "artifact_manifest.txt"
                # Append to manifest: multiple subdirs may write to the same artifact dir
                # within a single _stage_package_artifacts() call. This is safe because:
                # 1. Each test gets an isolated temp directory (BuildPackageTestCase.setUp)
                # 2. Tests run sequentially (standard unittest behavior)
                # 3. Appends within one call are single-threaded
                with manifest.open("a", encoding="utf-8") as f:
                    f.write(f"{dir_path}\n")
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


def _rules_path(pkg_name: str, config: PackageConfig) -> Path:
    """Return path to generated ``debian/rules`` for a versioned DEB build."""
    updated = update_package_name(pkg_name, replace(config, versioned_pkg=True))
    return Path(config.dest_dir) / config.pkg_type / updated / "debian" / "rules"


def _read_rules_file(pkg_name: str, config: PackageConfig) -> str:
    """Read generated ``debian/rules`` after validating it was created."""
    rules_path = _rules_path(pkg_name=pkg_name, config=config)
    if not rules_path.exists():
        raise AssertionError(f"Expected rules file was not created: {rules_path}")
    return rules_path.read_text(encoding="utf-8")


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

    @unittest.skipIf(
        sys.platform == "win32",
        "xnack-suffixed artifact dirs use ':', which NTFS reserves and cannot "
        "represent as a real path; this naming only ever occurs on the Linux "
        "packaging path, so Windows coverage is a no-op by construction.",
    )
    def test_has_artifact_for_arch_matches_xnack_suffixed_directory(self):
        """rand/solver/hiptensor/rocalution ship gfx942/gfx950 only as ':xnack+'
        directories - has_artifact_for_arch() must recognize those, not just an
        exact (non-suffixed) directory match."""
        cfg = _kpack_config(self.temp_dir, target=[TEST_GFX_TARGET_ALT])
        _stage_package_artifacts(
            "amdrocm-rand",
            cfg.artifacts_dir,
            TEST_GFX_TARGET_ALT,
            dir_suffix_override=f"{TEST_GFX_TARGET_ALT}:xnack+",
        )
        self.assertTrue(
            has_artifact_for_arch(
                "amdrocm-rand", cfg.artifacts_dir, TEST_GFX_TARGET_ALT
            )
        )

    def test_has_artifact_for_arch_missing_when_not_staged_for_that_arch(self):
        """Artifacts staged for one arch but not another - has_artifact_for_arch()
        must report False for the arch that was never staged, rather than
        (incorrectly) skipping the check because *some* gfx-arch artifacts
        exist elsewhere for this package."""
        cfg = _kpack_config(
            self.temp_dir, target=[TEST_GFX_TARGET, TEST_GFX_TARGET_ALT]
        )
        _stage_package_artifacts("amdrocm-rand", cfg.artifacts_dir, TEST_GFX_TARGET)
        self.assertFalse(
            has_artifact_for_arch(
                "amdrocm-rand", cfg.artifacts_dir, TEST_GFX_TARGET_ALT
            )
        )

    @unittest.skipIf(
        sys.platform == "win32",
        "xnack-suffixed artifact dirs use ':', which NTFS reserves and cannot "
        "represent as a real path; this naming only ever occurs on the Linux "
        "packaging path, so Windows coverage is a no-op by construction.",
    )
    def test_meta_dependency_includes_xnack_only_device_arch(self):
        """Reproduces the original bug end-to-end: a meta package must still
        depend on a device arch whose only artifact directory is xnack-suffixed."""
        cfg = _kpack_config(self.temp_dir, target=[TEST_GFX_TARGET_ALT])
        _stage_package_artifacts(
            "amdrocm-rand",
            cfg.artifacts_dir,
            GFX_HOST,
        )
        _stage_package_artifacts(
            "amdrocm-rand",
            cfg.artifacts_dir,
            TEST_GFX_TARGET_ALT,
            dir_suffix_override=f"{TEST_GFX_TARGET_ALT}:xnack+",
        )
        deps = expand_kpack_meta_dependencies("amdrocm-rand", cfg.gfxarch_list, cfg)
        self.assertEqual(
            deps,
            ["amdrocm-rand-host7.1", f"amdrocm-rand7.1-{TEST_GFX_TARGET_ALT}"],
        )

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
    def test_rules_file_excludes_code_objects_from_makeshlibs(
        self, _mock_dpkg: object, _mock_move: object
    ) -> None:
        """The rendered ``debian/rules`` must override dh_makeshlibs to skip
        ``.co`` GPU code objects. hip-tests ships an intentionally-corrupted ELF
        fixture (elf_bad_shoff.co) that dh_makeshlibs cannot parse; without the
        ``-X.co`` exclusion it aborts dpkg-buildpackage for the whole package.
        """
        cfg = _kpack_config(self.temp_dir)
        _stage_package_artifacts(
            pkg_name=PKG_FFT,
            artifacts_dir=cfg.artifacts_dir,
            gfx_arch=TEST_GFX_TARGET,
            enable_kpack=True,
        )
        device_cfg = replace(cfg, gfx_arch=TEST_GFX_TARGET)

        deb_package.create_versioned_deb_package(pkg_name=PKG_FFT, config=device_cfg)

        rules = _read_rules_file(pkg_name=PKG_FFT, config=device_cfg)
        self.assertIn("override_dh_makeshlibs:", rules)
        self.assertIn("dh_makeshlibs -X.co", rules)

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


# Empty package skipping (#6766)
# ---------------------------------------------------------------------------
class EmptyPackageSkippingTest(BuildPackageTestCase):
    """Packages with no artifacts are skipped in kpack mode."""

    def test_skips_empty_non_meta_package(self) -> None:
        """DEB/RPM generation skips gfxarch packages with no artifacts."""
        test_cases = [
            (
                TEST_PKG_TYPE_DEB,
                deb_package,
                "package_with_dpkg_build",
                "create_versioned_deb_package",
            ),
            (
                TEST_PKG_TYPE_RPM,
                rpm_package,
                "package_with_rpmbuild",
                "create_versioned_rpm_package",
            ),
        ]
        for pkg_type, module, build_func, create_func in test_cases:
            with self.subTest(pkg_type=pkg_type):
                cfg = _kpack_config(self.temp_dir, pkg_type=pkg_type)
                device_cfg = replace(cfg, gfx_arch=TEST_GFX_TARGET)

                with (
                    patch.object(
                        module, "move_packages_to_destination", return_value=[]
                    ),
                    patch.object(module, build_func) as mock_build,
                ):
                    result = getattr(module, create_func)(
                        pkg_name=PKG_FFT, config=device_cfg
                    )

                self.assertEqual(result, [])
                mock_build.assert_not_called()

    def test_builds_meta_package_without_artifacts(self) -> None:
        """DEB/RPM generation builds meta packages (they never have artifacts)."""
        test_cases = [
            (
                TEST_PKG_TYPE_DEB,
                deb_package,
                "package_with_dpkg_build",
                "create_versioned_deb_package",
            ),
            (
                TEST_PKG_TYPE_RPM,
                rpm_package,
                "package_with_rpmbuild",
                "create_versioned_rpm_package",
            ),
        ]
        for pkg_type, module, build_func, create_func in test_cases:
            with self.subTest(pkg_type=pkg_type):
                cfg = _kpack_config(self.temp_dir, pkg_type=pkg_type)
                meta_cfg = replace(cfg, gfx_arch=GFX_META)

                with (
                    patch.object(
                        module, "move_packages_to_destination", return_value=[]
                    ),
                    patch.object(module, build_func) as mock_build,
                ):
                    getattr(module, create_func)(pkg_name=PKG_FFT, config=meta_cfg)

                mock_build.assert_called_once()


# ---------------------------------------------------------------------------
# Host fallback mechanism (#6766)
#
# When a gfx-specific dependency has no artifacts for the target architecture
# but does have host (generic) artifacts, the package should fall back to
# depending on the host version instead of failing or omitting the dependency.
#
# Test scenario:
#   - Building amdrocm-blas for gfx1100
#   - amdrocm-blas depends on amdrocm-solver (both are gfxarch packages)
#   - amdrocm-solver has artifacts for gfx942 but NOT for gfx1100
#   - amdrocm-solver DOES have host (generic) artifacts
#
#   amdrocm-blas (gfx1100)
#       └── depends on amdrocm-solver
#                         ├── gfx1100 artifacts? ❌ NO
#                         └── host artifacts?    ✅ YES → use host fallback
#
# Expected: Generated package should depend on amdrocm-solver-host7.1
#           (not amdrocm-solver7.1-gfx1100 which doesn't exist)
# ---------------------------------------------------------------------------
PKG_BLAS = "amdrocm-blas"
PKG_SOLVER = "amdrocm-solver"


class HostFallbackTest(BuildPackageTestCase):
    """Integration tests for host fallback when gfx-specific artifacts are missing."""

    @patch.object(deb_package, "move_packages_to_destination", return_value=[])
    @patch.object(deb_package, "package_with_dpkg_build")
    def test_deb_device_package_uses_host_fallback_dependency(
        self, _mock_dpkg: object, _mock_move: object
    ) -> None:
        """DEB device package falls back to host dependency when gfx artifacts missing.

        Also verifies host packages include gfxarch dependencies as host-versioned form
        to ensure version-locked dependencies rather than version-ambiguous library deps.
        """
        cfg = _kpack_config(self.temp_dir)

        # Stage BLAS for gfx1100 (the package we're building)
        _stage_package_artifacts(
            pkg_name=PKG_BLAS,
            artifacts_dir=cfg.artifacts_dir,
            gfx_arch=TEST_GFX_TARGET,
            enable_kpack=True,
        )
        # Stage BLAS host artifacts (needed for host package test)
        _stage_package_artifacts(
            pkg_name=PKG_BLAS,
            artifacts_dir=cfg.artifacts_dir,
            gfx_arch=GFX_HOST,
            enable_kpack=True,
        )
        # Stage SOLVER for gfx942 only (NOT gfx1100) - makes it a gfxarch package
        _stage_package_artifacts(
            pkg_name=PKG_SOLVER,
            artifacts_dir=cfg.artifacts_dir,
            gfx_arch=TEST_GFX_TARGET_ALT,
            enable_kpack=True,
        )
        # Stage SOLVER host artifacts - this is what we'll fall back to
        _stage_package_artifacts(
            pkg_name=PKG_SOLVER,
            artifacts_dir=cfg.artifacts_dir,
            gfx_arch=GFX_HOST,
            enable_kpack=True,
        )
        # Stage RUNTIME (non-gfxarch dependency needed by BLAS)
        _stage_package_artifacts(
            pkg_name=PKG_RUNTIME,
            artifacts_dir=cfg.artifacts_dir,
            gfx_arch=GFX_HOST,
            enable_kpack=True,
        )

        device_cfg = replace(cfg, gfx_arch=TEST_GFX_TARGET)
        deb_package.create_versioned_deb_package(pkg_name=PKG_BLAS, config=device_cfg)

        # Verify: BLAS device depends on SOLVER-host (fallback), not SOLVER-gfx1100
        control = _read_control_file(pkg_name=PKG_BLAS, config=device_cfg)
        depends = _control_field(control, "Depends")
        self.assertIn("amdrocm-solver-host", depends)

        # Also verify: BLAS host includes SOLVER-host (gfxarch dep as host-versioned)
        host_cfg = replace(cfg, gfx_arch=GFX_HOST)
        deb_package.create_versioned_deb_package(pkg_name=PKG_BLAS, config=host_cfg)
        control = _read_control_file(pkg_name=PKG_BLAS, config=host_cfg)
        depends = _control_field(control, "Depends")
        self.assertIn("amdrocm-solver-host", depends)

    @patch.object(rpm_package, "move_packages_to_destination", return_value=[])
    @patch.object(rpm_package, "package_with_rpmbuild")
    def test_rpm_device_package_uses_host_fallback_dependency(
        self, _mock_rpmbuild: object, _mock_move: object
    ) -> None:
        """RPM device package falls back to host dependency when gfx artifacts missing.

        Also verifies host packages include gfxarch dependencies as host-versioned form
        to ensure version-locked dependencies rather than version-ambiguous library deps.
        """
        cfg = _kpack_config(self.temp_dir, pkg_type=TEST_PKG_TYPE_RPM)

        # Stage BLAS for gfx1100 (the package we're building)
        _stage_package_artifacts(
            pkg_name=PKG_BLAS,
            artifacts_dir=cfg.artifacts_dir,
            gfx_arch=TEST_GFX_TARGET,
            enable_kpack=True,
        )
        # Stage BLAS host artifacts (needed for host package test)
        _stage_package_artifacts(
            pkg_name=PKG_BLAS,
            artifacts_dir=cfg.artifacts_dir,
            gfx_arch=GFX_HOST,
            enable_kpack=True,
        )
        # Stage SOLVER for gfx942 only (NOT gfx1100) - makes it a gfxarch package
        _stage_package_artifacts(
            pkg_name=PKG_SOLVER,
            artifacts_dir=cfg.artifacts_dir,
            gfx_arch=TEST_GFX_TARGET_ALT,
            enable_kpack=True,
        )
        # Stage SOLVER host artifacts - this is what we'll fall back to
        _stage_package_artifacts(
            pkg_name=PKG_SOLVER,
            artifacts_dir=cfg.artifacts_dir,
            gfx_arch=GFX_HOST,
            enable_kpack=True,
        )
        # Stage RUNTIME (non-gfxarch dependency needed by BLAS)
        _stage_package_artifacts(
            pkg_name=PKG_RUNTIME,
            artifacts_dir=cfg.artifacts_dir,
            gfx_arch=GFX_HOST,
            enable_kpack=True,
        )

        device_cfg = replace(cfg, gfx_arch=TEST_GFX_TARGET)
        rpm_package.create_versioned_rpm_package(pkg_name=PKG_BLAS, config=device_cfg)

        # Verify: BLAS device requires SOLVER-host (fallback), not SOLVER-gfx1100
        spec = _read_spec_file(pkg_name=PKG_BLAS, config=device_cfg)
        requires = _spec_field(spec, "Requires")
        self.assertIn("amdrocm-solver-host", requires)

        # Also verify: BLAS host includes SOLVER-host (gfxarch dep as host-versioned)
        host_cfg = replace(cfg, gfx_arch=GFX_HOST)
        rpm_package.create_versioned_rpm_package(pkg_name=PKG_BLAS, config=host_cfg)
        spec = _read_spec_file(pkg_name=PKG_BLAS, config=host_cfg)
        requires = _spec_field(spec, "Requires")
        self.assertIn("amdrocm-solver-host", requires)


# Generic kpack packaging logic, tested on a synthetic package.json.
class _KpackBuildTestCase(BuildPackageTestCase):
    """``_build`` returns the control/spec text of each package built (tools mocked)."""

    def setUp(self) -> None:
        super().setUp()
        _artifact_dir_name_index.cache_clear()

    def _config(
        self, targets: tuple[str, ...], pkg_type: str = "deb", label: str = "tree"
    ) -> PackageConfig:
        return _kpack_config(
            self.temp_dir / label, target=list(targets), pkg_type=pkg_type
        )

    def _build(self, cfg: PackageConfig, pkg_names) -> dict[str, str]:
        built: dict[str, str] = {}
        deb = sys.modules["deb_package"]
        rpm = sys.modules["rpm_package"]

        def record_deb(package_dir) -> None:
            control = (Path(package_dir) / "debian" / "control").read_text("utf-8")
            built[_control_field(control, "Package")] = control

        def record_rpm(specfile) -> None:
            spec = Path(specfile).read_text("utf-8")
            built[_spec_field(spec, "Name")] = spec

        with (
            patch.object(deb, "package_with_dpkg_build", side_effect=record_deb),
            patch.object(deb, "move_packages_to_destination", return_value=[]),
            patch.object(rpm, "package_with_rpmbuild", side_effect=record_rpm),
            patch.object(rpm, "move_packages_to_destination", return_value=[]),
        ):
            for pkg_name in pkg_names:
                build_package.build_package_variants(pkg_name, cfg)
        return built

    @staticmethod
    def _depends(built: dict[str, str], name: str) -> list[str]:
        """Dependency names (versions stripped) from all Depends:/Requires: fields;
        an empty field yields [], empty elements of a non-empty field are kept."""
        fields: list[str] = []
        in_field = False
        for line in built[name].splitlines():
            if line.startswith(("Depends:", "Requires:")):
                fields.append(line.split(":", 1)[1])
                in_field = True
            elif in_field and line[:1] in (" ", "\t"):
                fields[-1] += " " + line
            else:
                in_field = False
        joined = ",".join(fields)
        if not joined.strip():
            return []
        return [re.split(r"[\s(]", dep.strip())[0] for dep in joined.split(",")]


FAKE_TARGETS = ("gfx942", "gfx1100")
FAKE_UNBUILT_TARGET = "gfx1030"
FAKE_PKG_TYPES = (("deb", "dev"), ("rpm", "devel"))
FAKE_VER = ".".join(TEST_ROCM_VERSION.split(".")[:2])  # version in package names

FAKE_BASE = "amdrocm-fakebase"  # Gfxarch; generic and per-arch content
FAKE_SUPPORT = "amdrocm-fakesupport"  # not Gfxarch; generic content
FAKE_OPT_DEVEL = "amdrocm-fakeopt-devel"  # Gfxarch + OptionalContent -devel
FAKE_BARE_DEVEL = "amdrocm-fakebare-devel"  # as FAKE_OPT_DEVEL, no dependencies
FAKE_OPT_SIMPLE = "amdrocm-fakeoptsimple"  # OptionalContent, not Gfxarch
FAKE_OPT_META = "amdrocm-fakeoptmeta-devel"  # Metapackage + OptionalContent
FAKE_STRICT = "amdrocm-fakestrict"  # Gfxarch, no OptionalContent
FAKE_PLAIN = "amdrocm-fakeplain"  # no Gfxarch, no OptionalContent
FAKE_NOGFX_DEVEL = "amdrocm-fakenogfx-devel"  # -devel with OptionalContent, no Gfxarch
FAKE_GFX_DEVEL = "amdrocm-fakegfx-devel"  # -devel with Gfxarch, no OptionalContent
FAKE_SUITE = "amdrocm-fakesuite-devel"  # Gfxarch metapackage

# package -> (artifact prefix, artifact subdir name, component)
FAKE_ARTIFACTS = {
    FAKE_BASE: ("fakebase", "FakeBase", "lib"),
    FAKE_SUPPORT: ("fakesupport", "FakeSupport", "lib"),
    FAKE_OPT_DEVEL: ("fakeopt", "FakeOpt", "dev"),
    FAKE_BARE_DEVEL: ("fakebare", "FakeBare", "dev"),
    FAKE_OPT_SIMPLE: ("fakeoptsimple", "FakeOptSimple", "lib"),
    FAKE_STRICT: ("fakestrict", "FakeStrict", "lib"),
    FAKE_PLAIN: ("fakeplain", "FakePlain", "lib"),
    FAKE_NOGFX_DEVEL: ("fakenogfx", "FakeNoGfx", "dev"),
    FAKE_GFX_DEVEL: ("fakegfx", "FakeGfx", "dev"),
}


def _fake_package(
    name: str,
    *,
    deps: tuple[str, ...] = (),
    gfxarch: bool = False,
    optional: bool = False,
    meta: bool = False,
) -> dict:
    """Return a synthetic ``package.json`` entry; artifacts come from FAKE_ARTIFACTS."""
    info: dict = {
        "Package": name,
        "Version": "",
        "Architecture": "amd64",
        "BuildArch": "x86_64",
        "DEBDepends": list(deps),
        "RPMRequires": list(deps),
        "Maintainer": "Test <test@example.com>",
        "Description_Short": f"{name} synthetic test package",
        "Description_Long": f"{name} synthetic test package",
        "Section": "devel",
        "Priority": "optional",
        "Group": "unknown",
        "License": "MIT",
        "Vendor": "Test",
        "Homepage": "https://example.com",
    }
    if name in FAKE_ARTIFACTS:
        artifact, subdir, component = FAKE_ARTIFACTS[name]
        info["Artifactory"] = [
            {
                "Artifact": artifact,
                "Artifact_Subdir": [{"Name": subdir, "Components": [component]}],
            }
        ]
    if gfxarch:
        info["Gfxarch"] = "True"
    if optional:
        info["OptionalContent"] = "True"
    if meta:
        info["Metapackage"] = "True"
    return info


FAKE_PACKAGES = [
    _fake_package(FAKE_BASE, gfxarch=True),
    _fake_package(FAKE_SUPPORT),
    _fake_package(
        FAKE_OPT_DEVEL,
        deps=(FAKE_BASE, FAKE_SUPPORT, FAKE_OPT_SIMPLE),
        gfxarch=True,
        optional=True,
    ),
    _fake_package(FAKE_BARE_DEVEL, gfxarch=True, optional=True),
    _fake_package(FAKE_OPT_SIMPLE, optional=True),
    _fake_package(FAKE_OPT_META, gfxarch=True, optional=True, meta=True),
    _fake_package(FAKE_STRICT, deps=(FAKE_SUPPORT,), gfxarch=True),
    _fake_package(FAKE_PLAIN),
    _fake_package(FAKE_NOGFX_DEVEL, optional=True),
    _fake_package(FAKE_GFX_DEVEL, gfxarch=True),
    _fake_package(
        FAKE_SUITE,
        deps=(FAKE_OPT_DEVEL, FAKE_OPT_SIMPLE, FAKE_PLAIN),
        gfxarch=True,
        meta=True,
    ),
]


def _stage_fake(
    artifacts_dir: Path, pkg_name: str, suffix: str, content: str = "files"
) -> None:
    """Stage ``<artifact>_<component>_<suffix>``; ``content`` is "files",
    "empty_dir" or "manifest_only" (manifest line, no directory)."""
    artifact, subdir, component = FAKE_ARTIFACTS[pkg_name]
    artifact_dir = artifacts_dir / f"{artifact}_{component}_{suffix}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stage_path = f"{subdir}/{component}"
    with (artifact_dir / "artifact_manifest.txt").open("a", encoding="utf-8") as f:
        f.write(f"{stage_path}\n")
    if content == "manifest_only":
        return
    (artifact_dir / stage_path).mkdir(parents=True, exist_ok=True)
    if content == "files":
        (artifact_dir / stage_path / STAGING_PAYLOAD_NAME).write_bytes(
            STAGING_PAYLOAD_BYTES
        )


class _FakePackagesTestCase(_KpackBuildTestCase):
    def setUp(self) -> None:
        super().setUp()
        # Patch package.json in every module namespace; other tests may load copies.
        namespaces: dict[int, dict] = {}
        for module in (
            build_package,
            deb_package,
            rpm_package,
            sys.modules["deb_package"],
            sys.modules["rpm_package"],
            sys.modules["packaging_utils"],
        ):
            for namespace in (module.__dict__, module.get_package_info.__globals__):
                if "read_package_json_file" in namespace:
                    namespaces[id(namespace)] = namespace
        for namespace in namespaces.values():
            patcher = patch.dict(
                namespace,
                {"read_package_json_file": lambda: list(FAKE_PACKAGES)},
            )
            patcher.start()
            self.addCleanup(patcher.stop)

    def _stage_base_and_support(
        self, cfg: PackageConfig, base_targets: tuple[str, ...] = FAKE_TARGETS
    ) -> None:
        _stage_fake(cfg.artifacts_dir, FAKE_SUPPORT, "generic")
        _stage_fake(cfg.artifacts_dir, FAKE_BASE, "generic")
        for target in base_targets:
            _stage_fake(cfg.artifacts_dir, FAKE_BASE, target)


class GenericContentTest(_FakePackagesTestCase):
    def test_generic_content_requires_a_non_empty_directory_for_the_subdir(
        self,
    ) -> None:
        cfg = self._config(FAKE_TARGETS)
        _stage_fake(cfg.artifacts_dir, FAKE_OPT_DEVEL, "generic", "manifest_only")
        _stage_fake(cfg.artifacts_dir, FAKE_OPT_SIMPLE, "generic", "empty_dir")
        _stage_fake(cfg.artifacts_dir, FAKE_STRICT, "generic", "files")
        other = cfg.artifacts_dir / "fakeplain_lib_generic"
        (other / "Other/lib").mkdir(parents=True)
        (other / "Other/lib" / STAGING_PAYLOAD_NAME).write_bytes(STAGING_PAYLOAD_BYTES)
        (other / "artifact_manifest.txt").write_text("Other/lib\n", encoding="utf-8")

        expected = {
            FAKE_OPT_DEVEL: False,  # manifest line without a directory
            FAKE_OPT_SIMPLE: False,  # existing but empty directory
            FAKE_STRICT: True,
            FAKE_PLAIN: False,  # content under another subdir
            FAKE_BARE_DEVEL: False,  # nothing staged
        }
        for pkg_name, has_content in expected.items():
            self.assertEqual(
                has_generic_content(get_package_info(pkg_name), cfg.artifacts_dir),
                has_content,
                pkg_name,
            )

    def test_omitted_only_for_optional_non_meta_package_without_content(self) -> None:
        cfg = self._config(FAKE_TARGETS)
        _stage_fake(cfg.artifacts_dir, FAKE_OPT_DEVEL, "generic", "manifest_only")
        _stage_fake(cfg.artifacts_dir, FAKE_OPT_SIMPLE, "generic", "files")
        _stage_fake(cfg.artifacts_dir, FAKE_STRICT, "generic", "manifest_only")
        expected = {
            FAKE_OPT_DEVEL: True,
            FAKE_OPT_SIMPLE: False,  # has content
            FAKE_STRICT: False,  # not OptionalContent
            FAKE_PLAIN: False,  # not OptionalContent, nothing staged
            FAKE_OPT_META: False,  # metapackages never omit
        }
        for pkg_name, omitted in expected.items():
            self.assertEqual(
                omits_generic_variants(get_package_info(pkg_name), cfg.artifacts_dir),
                omitted,
                pkg_name,
            )


class DevelGfxarchRuleTest(_FakePackagesTestCase):
    def _is_gfxarch(self, cfg: PackageConfig, pkg_name: str) -> bool:
        return is_gfxarch_package(get_package_info(pkg_name), True, cfg.artifacts_dir)

    def test_devel_with_gfxarch_follows_per_arch_artifacts(self) -> None:
        cfg = self._config(FAKE_TARGETS)
        _stage_fake(cfg.artifacts_dir, FAKE_OPT_DEVEL, "generic")
        self.assertFalse(self._is_gfxarch(cfg, FAKE_OPT_DEVEL))
        _stage_fake(cfg.artifacts_dir, FAKE_OPT_DEVEL, "gfx942")
        _artifact_dir_name_index.cache_clear()
        self.assertTrue(self._is_gfxarch(cfg, FAKE_OPT_DEVEL))

    @unittest.skipIf(
        sys.platform == "win32",
        "xnack-suffixed artifact dirs use ':', which NTFS reserves and cannot "
        "create; the packaging flow that consumes them is Linux-only",
    )
    def test_devel_with_gfxarch_matches_xnack_suffixed_artifacts(self) -> None:
        cfg = self._config(FAKE_TARGETS)
        _stage_fake(cfg.artifacts_dir, FAKE_OPT_DEVEL, "gfx942:xnack+")
        self.assertTrue(self._is_gfxarch(cfg, FAKE_OPT_DEVEL))

    def test_devel_with_gfxarch_needs_an_artifacts_dir(self) -> None:
        self.assertFalse(
            is_gfxarch_package(get_package_info(FAKE_OPT_DEVEL), True, None)
        )

    def test_devel_without_gfxarch_stays_generic_with_per_arch_artifacts(self) -> None:
        cfg = self._config(FAKE_TARGETS)
        _stage_fake(cfg.artifacts_dir, FAKE_NOGFX_DEVEL, "generic")
        _stage_fake(cfg.artifacts_dir, FAKE_NOGFX_DEVEL, "gfx942")
        self.assertFalse(self._is_gfxarch(cfg, FAKE_NOGFX_DEVEL))

    def test_devel_without_optional_content_stays_generic_with_per_arch_artifacts(
        self,
    ) -> None:
        cfg = self._config(FAKE_TARGETS)
        _stage_fake(cfg.artifacts_dir, FAKE_GFX_DEVEL, "generic")
        _stage_fake(cfg.artifacts_dir, FAKE_GFX_DEVEL, "gfx942")
        self.assertFalse(self._is_gfxarch(cfg, FAKE_GFX_DEVEL))

    def test_devel_with_manifest_only_per_arch_artifacts_stays_generic(self) -> None:
        cfg = self._config(FAKE_TARGETS)
        _stage_fake(cfg.artifacts_dir, FAKE_OPT_DEVEL, "gfx942", "manifest_only")
        self.assertFalse(self._is_gfxarch(cfg, FAKE_OPT_DEVEL))


class OptionalContentDependencyFilterTest(_FakePackagesTestCase):
    def test_only_optional_packages_without_content_are_dropped(self) -> None:
        cfg = self._config(FAKE_TARGETS)
        _stage_fake(cfg.artifacts_dir, FAKE_OPT_SIMPLE, "generic", "manifest_only")
        _stage_fake(cfg.artifacts_dir, FAKE_PLAIN, "generic", "empty_dir")
        _stage_fake(cfg.artifacts_dir, FAKE_SUPPORT, "generic")
        deps = [FAKE_OPT_SIMPLE, FAKE_PLAIN, FAKE_SUPPORT, FAKE_OPT_META]
        kept, _ = filter_dependencies_by_artifacts(deps, cfg.artifacts_dir, GFX_HOST)
        self.assertCountEqual(kept, [FAKE_PLAIN, FAKE_SUPPORT, FAKE_OPT_META])


class HostlessVariantsTest(_FakePackagesTestCase):
    def _stage_opt_devel(
        self,
        cfg: PackageConfig,
        targets: tuple[str, ...] = FAKE_TARGETS,
        generic: str = "manifest_only",
    ) -> None:
        self._stage_base_and_support(cfg)
        _stage_fake(cfg.artifacts_dir, FAKE_OPT_DEVEL, "generic", generic)
        for target in targets:
            _stage_fake(cfg.artifacts_dir, FAKE_OPT_DEVEL, target)

    def test_per_target_content_builds_device_packages_without_host(self) -> None:
        targets = FAKE_TARGETS + (FAKE_UNBUILT_TARGET,)
        for pkg_type, dev in FAKE_PKG_TYPES:
            for generic in ("manifest_only", "empty_dir"):
                with self.subTest(pkg_type=pkg_type, generic=generic):
                    cfg = self._config(targets, pkg_type, f"{pkg_type}-{generic}")
                    # Per-target content for FAKE_TARGETS only.
                    self._stage_opt_devel(cfg, generic=generic)
                    built = self._build(cfg, [FAKE_OPT_DEVEL])
                    name = f"amdrocm-fakeopt-{dev}"
                    self.assertTrue(
                        {
                            f"{name}{FAKE_VER}-gfx942",
                            f"{name}{FAKE_VER}-gfx1100",
                            f"{name}{FAKE_VER}",
                            name,
                        }
                        <= set(built)
                    )
                    self.assertFalse([pkg for pkg in built if "-host" in pkg])
                    self.assertFalse(
                        [pkg for pkg in built if FAKE_UNBUILT_TARGET in pkg]
                    )
                    for target in FAKE_TARGETS:
                        deps = self._depends(built, f"{name}{FAKE_VER}-{target}")
                        self.assertIn(f"amdrocm-fakesupport{FAKE_VER}", deps)
                        self.assertIn(f"amdrocm-fakebase{FAKE_VER}-{target}", deps)
                        self.assertFalse([dep for dep in deps if "-host" in dep])
                    self.assertCountEqual(
                        self._depends(built, f"{name}{FAKE_VER}"),
                        [f"{name}{FAKE_VER}-gfx1100", f"{name}{FAKE_VER}-gfx942"],
                    )

    def test_arch_dependency_without_artifacts_for_a_target_uses_its_host(self) -> None:
        cfg = self._config(FAKE_TARGETS)
        self._stage_base_and_support(cfg, base_targets=("gfx942",))
        _stage_fake(cfg.artifacts_dir, FAKE_OPT_DEVEL, "generic", "manifest_only")
        for target in FAKE_TARGETS:
            _stage_fake(cfg.artifacts_dir, FAKE_OPT_DEVEL, target)
        built = self._build(cfg, [FAKE_OPT_DEVEL])
        with_arch = self._depends(built, f"amdrocm-fakeopt-dev{FAKE_VER}-gfx942")
        without_arch = self._depends(built, f"amdrocm-fakeopt-dev{FAKE_VER}-gfx1100")
        self.assertIn(f"amdrocm-fakebase{FAKE_VER}-gfx942", with_arch)
        self.assertIn(f"amdrocm-fakebase-host{FAKE_VER}", without_arch)
        self.assertNotIn(f"amdrocm-fakebase{FAKE_VER}-gfx1100", without_arch)
        self.assertIn(f"amdrocm-fakesupport{FAKE_VER}", without_arch)

    def test_optional_dependency_without_content_is_not_carried(self) -> None:
        for has_content, label in ((False, "empty"), (True, "present")):
            with self.subTest(dependency=label):
                cfg = self._config(FAKE_TARGETS, label=label)
                self._stage_opt_devel(cfg)
                _stage_fake(
                    cfg.artifacts_dir,
                    FAKE_OPT_SIMPLE,
                    "generic",
                    "files" if has_content else "manifest_only",
                )
                built = self._build(cfg, [FAKE_OPT_DEVEL])
                for target in FAKE_TARGETS:
                    deps = self._depends(
                        built, f"amdrocm-fakeopt-dev{FAKE_VER}-{target}"
                    )
                    self.assertEqual(
                        f"amdrocm-fakeoptsimple{FAKE_VER}" in deps, has_content
                    )

    def test_package_without_dependencies_gets_no_dependency_line(self) -> None:
        cfg = self._config(FAKE_TARGETS)
        _stage_fake(cfg.artifacts_dir, FAKE_BARE_DEVEL, "generic", "manifest_only")
        for target in FAKE_TARGETS:
            _stage_fake(cfg.artifacts_dir, FAKE_BARE_DEVEL, target)
        built = self._build(cfg, [FAKE_BARE_DEVEL])
        for target in FAKE_TARGETS:
            name = f"amdrocm-fakebare-dev{FAKE_VER}-{target}"
            self.assertIn(name, built)
            self.assertEqual(self._depends(built, name), [])

    def test_generic_content_adds_a_host_that_device_packages_depend_on(self) -> None:
        cfg = self._config(FAKE_TARGETS)
        self._stage_opt_devel(cfg, generic="files")
        built = self._build(cfg, [FAKE_OPT_DEVEL])
        host = f"amdrocm-fakeopt-dev-host{FAKE_VER}"
        self.assertIn(host, built)
        self.assertIn(host, self._depends(built, f"amdrocm-fakeopt-dev{FAKE_VER}"))
        for target in FAKE_TARGETS:
            self.assertIn(
                host, self._depends(built, f"amdrocm-fakeopt-dev{FAKE_VER}-{target}")
            )

    def test_no_content_at_all_builds_nothing(self) -> None:
        for content in ("manifest_only", "empty_dir"):
            for pkg_name in (FAKE_OPT_DEVEL, FAKE_OPT_SIMPLE):
                with self.subTest(content=content, pkg=pkg_name):
                    cfg = self._config(FAKE_TARGETS, label=f"{content}-{pkg_name}")
                    self._stage_base_and_support(cfg)
                    _stage_fake(cfg.artifacts_dir, pkg_name, "generic", content)
                    built = self._build(cfg, [pkg_name])
                    self.assertFalse(
                        [pkg for pkg in built if pkg.startswith("amdrocm-fakeopt")]
                    )

    def test_suite_depends_on_per_arch_package_only_for_targets_with_it(self) -> None:
        targets = FAKE_TARGETS + (FAKE_UNBUILT_TARGET,)
        cfg = self._config(targets)
        self._stage_opt_devel(cfg)
        _stage_fake(cfg.artifacts_dir, FAKE_OPT_SIMPLE, "generic")
        _stage_fake(cfg.artifacts_dir, FAKE_PLAIN, "generic")
        built = self._build(cfg, [FAKE_OPT_DEVEL, FAKE_SUITE])
        for target in FAKE_TARGETS:
            deps = self._depends(built, f"amdrocm-fakesuite-dev{FAKE_VER}-{target}")
            self.assertIn(f"amdrocm-fakeopt-dev{FAKE_VER}-{target}", deps)
        unbuilt = self._depends(
            built, f"amdrocm-fakesuite-dev{FAKE_VER}-{FAKE_UNBUILT_TARGET}"
        )
        self.assertFalse(
            [dep for dep in unbuilt if dep.startswith("amdrocm-fakeopt-dev")]
        )

    def test_suite_has_no_dependency_on_package_without_content(self) -> None:
        cfg = self._config(FAKE_TARGETS)
        self._stage_base_and_support(cfg)
        _stage_fake(cfg.artifacts_dir, FAKE_OPT_DEVEL, "generic", "manifest_only")
        _stage_fake(cfg.artifacts_dir, FAKE_OPT_SIMPLE, "generic", "manifest_only")
        _stage_fake(cfg.artifacts_dir, FAKE_PLAIN, "generic")
        built = self._build(cfg, [FAKE_OPT_DEVEL, FAKE_SUITE])
        for target in FAKE_TARGETS:
            deps = self._depends(built, f"amdrocm-fakesuite-dev{FAKE_VER}-{target}")
            self.assertFalse([dep for dep in deps if dep.startswith("amdrocm-fakeopt")])
            self.assertIn(f"amdrocm-fakeplain{FAKE_VER}", deps)


class NonOptionalPackagesKeepVariantsTest(_FakePackagesTestCase):
    def test_gfxarch_package_without_generic_content_keeps_host_edges(self) -> None:
        cfg = self._config(FAKE_TARGETS)
        _stage_fake(cfg.artifacts_dir, FAKE_SUPPORT, "generic")
        _stage_fake(cfg.artifacts_dir, FAKE_STRICT, "generic", "manifest_only")
        for target in FAKE_TARGETS:
            _stage_fake(cfg.artifacts_dir, FAKE_STRICT, target)
        built = self._build(cfg, [FAKE_STRICT])
        host = f"amdrocm-fakestrict-host{FAKE_VER}"
        self.assertIn(host, self._depends(built, f"amdrocm-fakestrict{FAKE_VER}"))
        for target in FAKE_TARGETS:
            self.assertIn(
                host, self._depends(built, f"amdrocm-fakestrict{FAKE_VER}-{target}")
            )

    def test_package_without_the_key_keeps_alias_when_empty(self) -> None:
        cfg = self._config(FAKE_TARGETS)
        _stage_fake(cfg.artifacts_dir, FAKE_PLAIN, "generic", "empty_dir")
        _stage_fake(cfg.artifacts_dir, FAKE_OPT_SIMPLE, "generic", "manifest_only")
        built = self._build(cfg, [FAKE_PLAIN, FAKE_OPT_SIMPLE])
        self.assertEqual(
            self._depends(built, "amdrocm-fakeplain"), [f"amdrocm-fakeplain{FAKE_VER}"]
        )
        self.assertFalse(
            [name for name in built if name.startswith("amdrocm-fakeoptsimple")]
        )

    def test_optional_package_with_content_keeps_alias_and_edge(self) -> None:
        cfg = self._config(FAKE_TARGETS)
        _stage_fake(cfg.artifacts_dir, FAKE_OPT_SIMPLE, "generic")
        _stage_fake(cfg.artifacts_dir, FAKE_PLAIN, "generic")
        built = self._build(cfg, [FAKE_OPT_SIMPLE, FAKE_SUITE])
        self.assertEqual(
            self._depends(built, "amdrocm-fakeoptsimple"),
            [f"amdrocm-fakeoptsimple{FAKE_VER}"],
        )
        for target in FAKE_TARGETS:
            self.assertIn(
                f"amdrocm-fakeoptsimple{FAKE_VER}",
                self._depends(built, f"amdrocm-fakesuite-dev{FAKE_VER}-{target}"),
            )

    def test_failed_versioned_build_still_gets_alias_when_content_exists(self) -> None:
        cfg = self._config(FAKE_TARGETS)
        _stage_fake(cfg.artifacts_dir, FAKE_OPT_SIMPLE, "generic")
        with patch.object(build_package, "build_versioned_package", return_value=[]):
            built = self._build(cfg, [FAKE_OPT_SIMPLE])
        self.assertIn("amdrocm-fakeoptsimple", built)
        self.assertNotIn(f"amdrocm-fakeoptsimple{FAKE_VER}", built)


class PackagesWithoutContentPartitionTest(_FakePackagesTestCase):
    def test_only_optional_packages_without_any_content_are_skipped(self) -> None:
        # label, package, staged (suffix, content) pairs, kpack mode, skipped
        rows = (
            ("interim", FAKE_OPT_DEVEL, (("generic", "manifest_only"),), True, True),
            (
                "per_arch",
                FAKE_OPT_DEVEL,
                (("generic", "manifest_only"), ("gfx942", "files")),
                True,
                False,
            ),
            ("generic", FAKE_OPT_SIMPLE, (("generic", "files"),), True, False),
            ("plain_empty", FAKE_PLAIN, (("generic", "empty_dir"),), True, False),
            (
                "single_arch",
                FAKE_OPT_DEVEL,
                (("generic", "manifest_only"),),
                False,
                False,
            ),
        )
        for label, pkg_name, staged, kpack, skipped in rows:
            with self.subTest(label):
                cfg = self._config(FAKE_TARGETS, label=label)
                for suffix, content in staged:
                    _stage_fake(cfg.artifacts_dir, pkg_name, suffix, content)
                cfg = replace(cfg, enable_kpack=kpack)
                expected = ([], [pkg_name]) if skipped else ([pkg_name], [])
                self.assertEqual(
                    build_package.partition_packages_without_content([pkg_name], cfg),
                    expected,
                )

    def test_package_whose_content_check_raises_is_kept(self) -> None:
        cfg = self._config(FAKE_TARGETS)
        broken = _fake_package(FAKE_OPT_DEVEL, gfxarch=True, optional=True)
        del broken["Artifactory"][0]["Artifact_Subdir"]
        with patch.object(build_package, "get_package_info", return_value=broken):
            self.assertEqual(
                build_package.partition_packages_without_content([FAKE_OPT_DEVEL], cfg),
                ([FAKE_OPT_DEVEL], []),
            )


class RunReportsPackagesWithoutContentTest(_FakePackagesTestCase):
    def test_package_without_content_is_reported_as_skipped(self) -> None:
        cfg = self._config(FAKE_TARGETS)
        _stage_fake(cfg.artifacts_dir, FAKE_OPT_DEVEL, "generic", "manifest_only")
        self.assertFalse(cfg.dest_dir.exists())
        args = _args(
            self.temp_dir, runpath_pkg=True, pkg_names=[FAKE_OPT_DEVEL], parallel=1
        )
        # Exercise the real worker and orchestration without external package builds.
        with (
            patch.object(build_package, "create_package_config", return_value=cfg),
            patch.object(build_package, "cleanup_packaging_environment"),
            patch.object(build_package, "ProcessPoolExecutor", ThreadPoolExecutor),
        ):
            build_package.run(args)
        manifest_file = cfg.dest_dir / "built_packages.txt"
        self.assertTrue(manifest_file.exists())
        manifest = manifest_file.read_text("utf-8")
        self.assertIn("# Skipped base packages: 1\n", manifest)
        self.assertIn("# Failed to build: 0\n", manifest)


class PackageJsonConsistencyTest(unittest.TestCase):
    DEPENDENCY_FIELD_SUFFIXES = ("Depends", "Requires", "Recommends", "Suggests")

    def setUp(self) -> None:
        self.packages = read_package_json_file()

    def test_amdrocm_dependency_entries_are_exact_package_names(self) -> None:
        names = {pkg["Package"] for pkg in self.packages}
        misses = []
        for pkg in self.packages:
            for field, value in pkg.items():
                if not field.endswith(self.DEPENDENCY_FIELD_SUFFIXES):
                    continue
                for element in value if isinstance(value, list) else [value]:
                    for entry in element.split(","):
                        entry = entry.strip()
                        if "amdrocm" in entry and entry not in names:
                            misses.append(
                                f"{pkg['Package']}: {field} entry {entry!r} is "
                                "not an exact package name"
                            )
        self.assertEqual([], misses)

    def test_optional_content_only_on_non_meta_packages_with_artifacts(self) -> None:
        for pkg in self.packages:
            if not is_key_defined(pkg, "OptionalContent"):
                continue
            self.assertFalse(
                is_meta_package(pkg),
                f"{pkg['Package']}: OptionalContent is set on a metapackage",
            )
            self.assertTrue(
                pkg.get("Artifactory"),
                f"{pkg['Package']}: OptionalContent is set without Artifactory",
            )


if __name__ == "__main__":
    unittest.main()
