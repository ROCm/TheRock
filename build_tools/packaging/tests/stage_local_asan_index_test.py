#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the strictly local ASan wheel index."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from io import BytesIO

sys.path.insert(0, os.fspath(Path(__file__).parent.parent / "python"))

from stage_local_asan_index import (  # noqa: E402
    PHASE1_PROJECTS,
    main,
    normalize_project_name,
    stage_index,
    verify_index,
)


def _wheel_filename(project: str, version: str) -> str:
    distribution = project.replace("-", "_")
    return f"{distribution}-{version}-py3-none-any.whl"


def _write_wheel(
    directory: Path,
    project: str,
    version: str = "10.1.0+asan.20260807",
    *,
    marker: str = "",
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _wheel_filename(project, version)
    dist_info = f"{project.replace('-', '_')}-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr(
            f"{dist_info}/METADATA",
            "Metadata-Version: 2.1\n"
            f"Name: {project}\n"
            f"Version: {version}\n\n{marker}",
        )
        wheel.writestr(
            f"{dist_info}/WHEEL",
            "Wheel-Version: 1.0\nTag: py3-none-any\n",
        )
        wheel.writestr(f"{dist_info}/RECORD", "")
    return path


def _write_sdist(
    directory: Path,
    project: str,
    version: str = "10.1.0+asan.20260807",
    *,
    nested_metadata: bytes | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{project}-{version}.tar.gz"
    package_root = f"{project}-{version}"
    metadata = (
        "Metadata-Version: 2.1\n"
        f"Name: {project}\n"
        f"Version: {version}\n"
    ).encode()
    info = tarfile.TarInfo(f"{package_root}/PKG-INFO")
    info.size = len(metadata)
    with tarfile.open(path, "w:gz") as sdist:
        sdist.addfile(info, BytesIO(metadata))
        nested_info = tarfile.TarInfo(
            f"{package_root}/src/{project}.egg-info/PKG-INFO"
        )
        nested_payload = metadata if nested_metadata is None else nested_metadata
        nested_info.size = len(nested_payload)
        sdist.addfile(nested_info, BytesIO(nested_payload))
    return path


class TestProjectNormalization(unittest.TestCase):
    def test_pep503_normalization(self):
        self.assertEqual(normalize_project_name("ROCm_SDK.Device"), "rocm-sdk-device")


class TestStageLocalAsanIndex(unittest.TestCase):
    def test_accepts_matching_setuptools_egg_info_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sdist = _write_sdist(root / "packages", "rocm")

            index_dir = stage_index(root / "packages", root / "local")

            self.assertTrue((index_dir / sdist.name).is_file())

    def test_rejects_disagreeing_setuptools_egg_info_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_sdist(
                root / "packages",
                "rocm",
                nested_metadata=(
                    b"Metadata-Version: 2.1\n"
                    b"Name: rocm\n"
                    b"Version: 10.1.0+asan.different\n"
                ),
            )

            with self.assertRaisesRegex(ValueError, "disagrees with"):
                stage_index(root / "packages", root / "local")

    def test_rejects_multiple_top_level_metadata_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            packages = root / "packages"
            packages.mkdir()
            path = packages / "rocm-10.1.0+asan.20260807.tar.gz"
            metadata = (
                b"Metadata-Version: 2.1\n"
                b"Name: rocm\n"
                b"Version: 10.1.0+asan.20260807\n"
            )
            with tarfile.open(path, "w:gz") as sdist:
                for package_root in ("rocm-first", "rocm-second"):
                    info = tarfile.TarInfo(f"{package_root}/PKG-INFO")
                    info.size = len(metadata)
                    sdist.addfile(info, BytesIO(metadata))

            with self.assertRaisesRegex(ValueError, "top-level.*found 2"):
                stage_index(packages, root / "local")

    def test_pip_resolves_flat_and_simple_indexes_without_network(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_wheel(root / "packages", "rocm-sdk-core")
            index_dir = stage_index(root / "packages", root / "local")

            common = [
                sys.executable,
                "-m",
                "pip",
                "--isolated",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
            ]
            find_links = subprocess.run(
                [
                    *common,
                    "--no-index",
                    "--find-links",
                    os.fspath(index_dir / "index.html"),
                    "--target",
                    os.fspath(root / "find-links-install"),
                    "rocm-sdk-core==10.1.0+asan.20260807",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(find_links.returncode, 0, find_links.stderr)

            simple = subprocess.run(
                [
                    *common,
                    "--index-url",
                    (index_dir / "simple").as_uri() + "/",
                    "--target",
                    os.fspath(root / "simple-install"),
                    "rocm-sdk-core==10.1.0+asan.20260807",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(simple.returncode, 0, simple.stderr)

    def test_cli_stage_and_verify_complete_set(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            packages = root / "packages" / "dist"
            for project in PHASE1_PROJECTS - {"rocm"}:
                _write_wheel(packages, project)
            _write_sdist(packages, "rocm")

            self.assertEqual(
                main(
                    [
                        "stage",
                        "--input-dir",
                        os.fspath(packages.parent),
                        "--output-root",
                        os.fspath(root / "local"),
                        "--require-phase1-set",
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "verify",
                        "--output-root",
                        os.fspath(root / "local"),
                        "--require-phase1-set",
                    ]
                ),
                0,
            )

    def test_stages_find_links_simple_index_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            packages = root / "packages" / "dist"
            output = root / "local"
            wheel = _write_wheel(packages, "rocm-sdk-core")

            index_dir = stage_index(packages.parent, output)

            staged_wheel = index_dir / wheel.name
            self.assertEqual(staged_wheel.read_bytes(), wheel.read_bytes())

            find_links = (index_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn(wheel.name.replace("+", "%2B"), find_links)

            simple_root = (index_dir / "simple" / "index.html").read_text(
                encoding="utf-8"
            )
            self.assertIn('href="./rocm-sdk-core/"', simple_root)

            project_index = (
                index_dir / "simple" / "rocm-sdk-core" / "index.html"
            ).read_text(encoding="utf-8")
            self.assertIn("../../rocm_sdk_core-10.1.0%2Basan.20260807", project_index)
            self.assertIn("#sha256=", project_index)

            manifest = json.loads(
                (index_dir / "index-manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["index_kind"], "local-only")
            self.assertEqual(manifest["relative_path"], "whl-asan/gfx942-all")
            self.assertEqual(manifest["package_count"], 1)
            self.assertEqual(manifest["packages"][0]["project"], "rocm-sdk-core")
            self.assertEqual(verify_index(output), index_dir)

    def test_rejects_non_asan_version(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_wheel(root / "packages", "rocm-sdk-core", "10.1.0")

            with self.assertRaisesRegex(ValueError, "expected prefix"):
                stage_index(root / "packages", root / "local")

    def test_refuses_different_wheel_with_same_filename(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            packages = root / "packages"
            _write_wheel(packages, "rocm-sdk-core", marker="first")
            stage_index(packages, root / "local")
            _write_wheel(packages, "rocm-sdk-core", marker="second")

            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                stage_index(packages, root / "local")

    def test_requires_complete_phase1_project_set_when_requested(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            packages = root / "packages"
            _write_wheel(packages, "rocm-sdk-core")

            with self.assertRaisesRegex(ValueError, "missing required Phase 1"):
                stage_index(
                    packages,
                    root / "local",
                    require_phase1_set=True,
                )

            for project in PHASE1_PROJECTS - {"rocm-sdk-core", "rocm"}:
                _write_wheel(packages, project)
            _write_sdist(packages, "rocm")
            index_dir = stage_index(
                packages,
                root / "complete",
                require_phase1_set=True,
            )
            self.assertEqual(
                len(list(index_dir.glob("*.whl"))), len(PHASE1_PROJECTS) - 1
            )
            self.assertEqual(len(list(index_dir.glob("*.tar.gz"))), 1)

    def test_phase1_set_requires_rocm_selector_as_sdist(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            packages = root / "packages"
            for project in PHASE1_PROJECTS:
                _write_wheel(packages, project)

            with self.assertRaisesRegex(ValueError, "rocm requires a sdist"):
                stage_index(
                    packages,
                    root / "local",
                    require_phase1_set=True,
                )

    def test_verify_detects_tampered_wheel(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wheel = _write_wheel(root / "packages", "rocm-sdk-core")
            index_dir = stage_index(root / "packages", root / "local")
            (index_dir / wheel.name).write_bytes(b"tampered")

            with self.assertRaises(ValueError):
                verify_index(root / "local")

    def test_rejects_mixed_asan_build_versions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            packages = root / "packages"
            _write_wheel(packages, "rocm-sdk-core")
            _write_wheel(
                packages,
                "rocm-sdk-devel",
                "10.1.0+asan.20260808",
            )

            with self.assertRaisesRegex(ValueError, "exactly one package version"):
                stage_index(packages, root / "local")

    def test_rejects_unsafe_family_name(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_wheel(root / "packages", "rocm-sdk-core")
            with self.assertRaisesRegex(ValueError, "Invalid family"):
                stage_index(root / "packages", root / "local", family="../escape")


if __name__ == "__main__":
    unittest.main()
