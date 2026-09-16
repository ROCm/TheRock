# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Strict target selection and SDK ownership regressions (no component builds)."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.artifacts import ArtifactCatalog
from _therock_utils.cmake_amdgpu_targets import (
    amdgpu_family_map,
    parse_amdgpu_targets_cmake,
)
from _therock_utils.py_packaging import Parameters, PopulatedDistPackage, build_packages
from _therock_utils.sdk_targets import (
    canonical_target,
    group_package_targets,
    package_owner,
)
from build_python_packages import (
    _run_kpack_split,
    validate_kpack_split_target_completeness,
    validate_required_dist_packages,
)

ROOT = Path(__file__).resolve().parents[2]


class StrictSelectionTest(unittest.TestCase):
    def test_registration(self):
        infos = parse_amdgpu_targets_cmake(ROOT / "cmake/therock_amdgpu_targets.cmake")
        strict = next(info for info in infos if info.gfx_target == "gfx1250-strict")
        self.assertTrue(strict.exclude_default_tests)
        self.assertEqual(strict.families, [])
        families = amdgpu_family_map()
        self.assertEqual(families["gfx1250"], ["gfx1250"])
        self.assertEqual(families["gfx1250-strict"], ["gfx1250-strict"])
        for family in ("gfx125X-all", "gfx125X-dcgpu", "dcgpu-all"):
            self.assertNotIn("gfx1250-strict", families[family])

    def test_cmake_selection_and_defaults(self):
        cases = [
            ("set(THEROCK_AMDGPU_FAMILIES gfx1250)", "gfx1250", "gfx1250", False),
            ("set(THEROCK_AMDGPU_FAMILIES gfx125X-all)", "gfx1250", "gfx1250", False),
            (
                "set(THEROCK_AMDGPU_TARGETS gfx1250)",
                "gfx1250",
                "THEROCK_DIST_AMDGPU_TARGETS-NOTFOUND",
                False,
            ),
            (
                "set(THEROCK_AMDGPU_FAMILIES gfx1250-strict)\nset(THEROCK_TEST_AMDGPU_FAMILIES gfx1250-strict)",
                "gfx1250-strict",
                "gfx1250-strict",
                True,
            ),
            (
                "set(THEROCK_AMDGPU_TARGETS gfx1250-strict)\nset(THEROCK_DIST_AMDGPU_TARGETS gfx1250-strict)\nset(THEROCK_TEST_AMDGPU_TARGETS gfx1250-strict)",
                "gfx1250-strict",
                "gfx1250-strict",
                True,
            ),
            (
                "set(THEROCK_AMDGPU_FAMILIES gfx1250-strict)",
                "gfx1250-strict",
                "gfx1250-strict",
                False,
            ),
        ]
        for settings, build, dist, explicit_tests in cases:
            with self.subTest(settings=settings), tempfile.TemporaryDirectory() as tmp:
                script = Path(tmp) / "probe.cmake"
                script.write_text(
                    f"""cmake_minimum_required(VERSION 3.25)
include("{ROOT}/cmake/therock_amdgpu_targets.cmake")
{settings}
therock_validate_amdgpu_targets()
if(NOT THEROCK_AMDGPU_TARGETS STREQUAL "{build}")
  message(FATAL_ERROR "Wrong build targets: ${{THEROCK_AMDGPU_TARGETS}}")
endif()
if(NOT THEROCK_DIST_AMDGPU_TARGETS STREQUAL "{dist}")
  message(FATAL_ERROR "Wrong dist targets: ${{THEROCK_DIST_AMDGPU_TARGETS}}")
endif()
"""
                )
                with script.open("a") as f:
                    if explicit_tests:
                        f.write(
                            'if(NOT THEROCK_TEST_AMDGPU_TARGETS STREQUAL "gfx1250-strict")\nmessage(FATAL_ERROR "Explicit strict tests lost")\nendif()\n'
                        )
                    else:
                        f.write(
                            'get_property(expected GLOBAL PROPERTY THEROCK_AMDGPU_TARGETS)\nlist(REMOVE_ITEM expected gfx1250-strict)\nif(NOT THEROCK_TEST_AMDGPU_TARGETS STREQUAL expected)\nmessage(FATAL_ERROR "Default test enumeration changed")\nendif()\n'
                        )
                subprocess.run(
                    ["cmake", "-P", str(script)],
                    check=True,
                    capture_output=True,
                    text=True,
                )


class StrictPackagingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir()

    def artifact(self, target: str, files: dict[str, str], name: str = "blas") -> Path:
        root = self.artifacts / f"{name}_lib_{target}"
        stage = root / "stage"
        stage.mkdir(parents=True)
        (root / "artifact_manifest.txt").write_text("stage\n")
        for path, content in files.items():
            dest = stage / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content)
        return stage

    def params(
        self,
        linux_target_families: list[str] | None = None,
        windows_target_families: list[str] | None = None,
    ) -> Parameters:
        return Parameters(
            self.root / "packages",
            "1.0.0",
            "",
            ArtifactCatalog(self.artifacts),
            kpack_split=True,
            linux_target_families=linux_target_families,
            windows_target_families=windows_target_families,
        )

    def test_owner_features_and_supplied_members(self):
        targets = ["gfx1250-strict", "gfx1250", "gfx1250-strict", "gfx950:xnack+"]
        self.assertEqual(
            group_package_targets(targets),
            {"gfx1250": ["gfx1250-strict", "gfx1250"], "gfx950": ["gfx950:xnack+"]},
        )
        for feature in (":xnack+", "-xnack-", ":sramecc+:xnack-"):
            self.assertEqual(
                canonical_target("gfx1250-strict" + feature), "gfx1250-strict"
            )
            self.assertEqual(package_owner("gfx1250-strict" + feature), "gfx1250")
        self.assertEqual(package_owner("gfx1250-unknown"), "gfx1250-unknown")
        self.assertEqual(package_owner("gfx1250:unknown+"), "gfx1250:unknown+")

    def test_base_strict_and_combined_population(self):
        for targets in (["gfx1250"], ["gfx1250-strict"], ["gfx1250", "gfx1250-strict"]):
            with self.subTest(targets=targets), tempfile.TemporaryDirectory() as tmp:
                self.root = Path(tmp)
                self.artifacts = self.root / "artifacts"
                self.artifacts.mkdir()
                for target in targets:
                    self.artifact(
                        target,
                        {
                            f".kpack/blas_{target}.kpack": target,
                            "lib/shared.dat": "identical",
                        },
                    )
                params = self.params()
                core = PopulatedDistPackage(params, logical_name="core")
                args = argparse.Namespace(
                    build_packages=False,
                    dest_dir=params.dest_dir,
                    devel_tarball_compression=False,
                )
                _run_kpack_split(args, params, core, None)
                devices = [
                    p for p in params.populated_packages if p.logical_name == "device"
                ]
                self.assertEqual(len(devices), 1)
                device = devices[0]
                self.assertEqual(
                    device.entry.get_dist_package_name(device.target_family),
                    "rocm-sdk-device-gfx1250",
                )
                for target in targets:
                    self.assertEqual(
                        (
                            device.platform_dir / f".kpack/blas_{target}.kpack"
                        ).read_text(),
                        target,
                    )
                self.assertEqual(params.all_target_families, set(targets))
                manifest = json.loads(
                    (device.platform_dir / ".devel_links/gfx1250.json").read_text()
                )
                links = [link["relpath"] for link in manifest["links"]]
                self.assertEqual(links.count("lib/shared.dat"), 1)
                self.assertEqual(
                    set(links),
                    {"lib/shared.dat"} | {f".kpack/blas_{t}.kpack" for t in targets},
                )
                self.assertFalse(list(params.dest_dir.glob("*strict*")))
                build_packages(params.dest_dir, package_dirs=[device.path])
                wheels = list((params.dest_dir / "dist").glob("*.whl"))
                self.assertEqual(len(wheels), 1)
                self.assertTrue(wheels[0].name.startswith("rocm_sdk_device_gfx1250-"))
                with zipfile.ZipFile(wheels[0]) as wheel:
                    payloads = {
                        name for name in wheel.namelist() if name.endswith(".kpack")
                    }
                    self.assertEqual(
                        payloads,
                        {f"_rocm_sdk_libraries/.kpack/blas_{t}.kpack" for t in targets},
                    )
                    metadata_name = next(
                        name for name in wheel.namelist() if name.endswith("/METADATA")
                    )
                    metadata = wheel.read(metadata_name).decode()
                    self.assertIn("Name: rocm-sdk-device-gfx1250", metadata)
                    self.assertIn("Requires-Dist: rocm-sdk-libraries==1.0.0", metadata)
                    self.assertNotIn("strict", metadata)

    def test_missing_requested_artifact_not_satisfied_by_owner(self):
        for present, requested in (
            ("gfx1250", "gfx1250-strict"),
            ("gfx1250-strict", "gfx1250"),
        ):
            with self.subTest(present=present):
                self.artifact(present, {"lib/a.dat": "data"}, name=present)
                catalog = ArtifactCatalog(
                    self.artifacts, filter=lambda an: an.target_family == present
                )
                with self.assertRaisesRegex(
                    RuntimeError, "missing fetched artifact targets"
                ):
                    validate_kpack_split_target_completeness(
                        kpack_split=True,
                        artifacts=catalog,
                        artifact_dir=self.artifacts,
                        linux_targets=[requested],
                        windows_targets=None,
                        platform_name="linux",
                    )

    def test_owner_metadata_and_required_wheels(self):
        self.artifact("gfx1250-strict", {"lib/a.dat": "data"})
        params = self.params(
            linux_target_families=["gfx1250-strict", "gfx942"],
            windows_target_families=["gfx1250"],
        )
        info = params.dist_info
        self.assertEqual(
            info.ALL_PACKAGES["device"].get_dist_package_require("gfx1250-strict"),
            "rocm-sdk-device-gfx1250==1.0.0",
        )
        self.assertEqual(info.get_target_family_platform_marker("gfx1250-strict"), "")
        extras = info.build_per_target_extras()
        self.assertIn("device-gfx1250", extras)
        self.assertNotIn("strict", str(extras))
        with mock.patch.dict(os.environ, {"ROCM_SDK_TARGET_FAMILY": "gfx1250-strict"}):
            self.assertEqual(info.determine_target_family(), "gfx1250-strict")
        dist = params.dest_dir / "dist"
        dist.mkdir(parents=True)
        for filename in (
            "rocm-1.0.0.tar.gz",
            "rocm_sdk_core-1.0.0-py3-none-linux_x86_64.whl",
            "rocm_sdk_libraries-1.0.0-py3-none-linux_x86_64.whl",
            "rocm_sdk_device_gfx1250-1.0.0-py3-none-linux_x86_64.whl",
        ):
            (dist / filename).write_text("fixture")
        validate_required_dist_packages(
            dest_dir=params.dest_dir,
            version=params.version,
            artifacts=params.artifacts,
            kpack_split=True,
            linux_targets=["gfx1250", "gfx1250-strict"],
            windows_targets=None,
            platform_name="linux",
        )

    def test_architectural_classification_is_independent_of_build_membership(self):
        params = self.params(linux_target_families=["gfx125X-all"])
        info = params.dist_info
        for target in ("gfx1250", "gfx1250-strict"):
            with self.subTest(target=target), mock.patch.object(
                info.subprocess, "check_output", return_value=target + "\n"
            ):
                self.assertEqual(info.discover_current_target_family(), "gfx125X-all")

    def test_owner_default_uses_cross_platform_package_availability(self):
        params = self.params(
            linux_target_families=["gfx1100", "gfx1250-strict"],
            windows_target_families=["gfx1250"],
        )
        self.assertEqual(params.default_target_family, "gfx1250-strict")

    def test_collisions_fail_before_population(self):
        self.artifact("gfx1250", {"lib/shared.dat": "base"})
        self.artifact("gfx1250-strict", {"lib/shared.dat": "strict"})
        params = self.params()
        dev = PopulatedDistPackage(
            params, logical_name="device", target_family="gfx1250"
        )
        with self.assertRaisesRegex(
            ValueError, "Conflicting device path lib/shared.dat"
        ):
            dev.populate_device_files(params.artifacts)
        self.assertFalse((dev.platform_dir / "lib/shared.dat").exists())

    def test_entry_types_and_symlink_content(self):
        first = self.artifact("gfx1250", {"lib/data": "base"})
        second = self.artifact("gfx1250-strict", {"lib/data": "base"})
        (first / "lib/link").symlink_to("data")
        (second / "lib/link").write_text("base")
        with self.assertRaisesRegex(ValueError, "lib/link"):
            ArtifactCatalog(self.artifacts).validated_matches()
        (second / "lib/link").unlink()
        (second / "lib/link").symlink_to("data")
        self.assertIn("lib/link", ArtifactCatalog(self.artifacts).validated_matches())
        (second / "lib/data").unlink()
        (second / "lib/data").mkdir()
        with self.assertRaisesRegex(ValueError, "Conflicting device path"):
            ArtifactCatalog(self.artifacts).validated_matches()


if __name__ == "__main__":
    unittest.main()
