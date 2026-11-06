#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for pre-upload verify and ``built_packages`` claim inputs.

Covers:
- Presence checks against ``built_packages.txt``
- Claim ↔ live control-field checks against ``built_packages.json``
  (Version, Depends/Requires, and related fields)
- ``packaging_summary.write_build_manifest`` JSON claim shape

Run::

    python3.12 -m unittest build_tools.packaging.linux.tests.native_linux_package_pre_upload_ut_test -v
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

THIS_SCRIPT_DIR = Path(__file__).resolve().parent
LINUX_DIR = THIS_SCRIPT_DIR.parent
BUILD_TOOLS_DIR = LINUX_DIR.parent.parent

for _path in (BUILD_TOOLS_DIR, LINUX_DIR):
    path_str = os.fspath(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import native_linux_package_pre_upload_test as verify  # noqa: E402
import packaging_summary as summary  # noqa: E402
from packaging_utils import PackageConfig  # noqa: E402

SAMPLE_MANIFEST = """# Built Packages Manifest
# Package Type: DEB
# Successfully built: 3
# Failed to build: 1

# Created Packages:
amdrocm-core-sdk7.15_7.15.0-1_amd64.deb
amdrocm-core-sdk7.15-gfx1100_7.15.0-1_amd64.deb
amdrocm-fft7.15_7.15.0-1_amd64.deb

# Failed Packages:
amdrocm-ck

# Skipped Packages:
# Note: Package names shown are base names from package.json
amdrocm-optional
"""

_SAMPLE_CREATED = (
    "amdrocm-core-sdk7.15_7.15.0-1_amd64.deb",
    "amdrocm-core-sdk7.15-gfx1100_7.15.0-1_amd64.deb",
    "amdrocm-fft7.15_7.15.0-1_amd64.deb",
)


def _cli_args(packages_dir: Path, *extra: str) -> list[str]:
    """Build minimal CLI argv for ``verify.main`` against ``packages_dir``."""
    return [
        "--packages-dir",
        str(packages_dir),
        "--pkg-type",
        "deb",
        *extra,
    ]


def _package_config(dest: Path, *, pkg_type: str = "deb") -> PackageConfig:
    return PackageConfig(
        artifacts_dir=dest / "artifacts",
        dest_dir=dest,
        pkg_type=pkg_type,
        rocm_version="7.15.0",
        version_suffix="1",
        install_prefix="/opt/rocm",
        gfx_arch="gfx1100",
        versioned_pkg=True,
    )


def _deb_control(**overrides: str) -> dict[str, str]:
    """Minimal DEB claim ``control`` dict matching ``DEB_CONTROL_FIELDS`` keys."""
    control = {name: "" for name in summary.DEB_CONTROL_FIELDS}
    control.update(
        {
            "Package": "amdrocm-fft7.15",
            "Version": "7.15.0-1",
            "Architecture": "amd64",
            "Depends": "amdrocm-runtime7.15, libc6",
            "Priority": "optional",
            "Section": "devel",
            "Maintainer": "AMD",
            "Homepage": "https://rocm.docs.amd.com",
        }
    )
    control.update(overrides)
    return control


def _write_claim(
    dest: Path,
    packages: list[dict[str, object]],
    *,
    pkg_type: str = "deb",
) -> Path:
    """Write a schema-valid ``built_packages.json`` for CLI/API tests."""
    path = dest / summary.MANIFEST_JSON_NAME
    doc = {
        "schema_version": summary.MANIFEST_SCHEMA_VERSION,
        "pkg_type": pkg_type,
        "rocm_version": "7.15.0",
        "version_suffix": "1",
        "gfx_arch": "gfx1100",
        "build_date_utc": "2026-01-01T00:00:00+00:00",
        "packages": packages,
        "failed": [],
        "skipped": [],
    }
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


class ParseBuiltPackageNamesTest(unittest.TestCase):
    """Manifest parsing keeps created filenames and ignores failed/skipped."""

    def test_parses_created_package_filenames(self) -> None:
        """Created ``.deb`` lines are returned; failed/skipped base names are not."""
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "built_packages.txt"
            manifest.write_text(SAMPLE_MANIFEST, encoding="utf-8")
            names = verify.parse_built_package_names(manifest)

        self.assertEqual(
            names,
            [
                "amdrocm-core-sdk7.15-gfx1100_7.15.0-1_amd64.deb",
                "amdrocm-core-sdk7.15_7.15.0-1_amd64.deb",
                "amdrocm-fft7.15_7.15.0-1_amd64.deb",
            ],
        )

    def test_pkg_type_filters_extension(self) -> None:
        """``pkg_type=deb`` drops ``.rpm`` filenames from the result."""
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "built_packages.txt"
            manifest.write_text(
                "# Created Packages:\na.deb\nb.rpm\n",
                encoding="utf-8",
            )
            names = verify.parse_built_package_names(manifest, pkg_type="deb")
        self.assertEqual(names, ["a.deb"])

    def test_name_prefix_filter(self) -> None:
        """``name_prefix`` keeps only matching Created filenames."""
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "built_packages.txt"
            manifest.write_text(SAMPLE_MANIFEST, encoding="utf-8")
            names = verify.parse_built_package_names(
                manifest, name_prefix="amdrocm-fft"
            )
        self.assertEqual(names, ["amdrocm-fft7.15_7.15.0-1_amd64.deb"])

    def test_missing_manifest(self) -> None:
        """Missing manifest path raises ``FileNotFoundError``."""
        with self.assertRaises(FileNotFoundError):
            verify.parse_built_package_names(Path("/no/such/built_packages.txt"))


class VerifyManifestPackagesTest(unittest.TestCase):
    """Presence check and path-safety behavior."""

    def test_found_and_missing(self) -> None:
        """Present files go to ``found``; absent names go to ``missing``."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.deb").write_bytes(b"x")
            result = verify.verify_manifest_packages(root, ["a.deb", "b.deb"])
        self.assertEqual(result.found, ["a.deb"])
        self.assertEqual(result.missing, ["b.deb"])
        self.assertFalse(result.passed)

    def test_path_traversal_rejected(self) -> None:
        """Filenames that escape ``packages_dir`` raise ``ValueError``."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                verify.verify_manifest_packages(Path(tmp), ["../escape.deb"])


class FormatReportTextTest(unittest.TestCase):
    """Human-readable report content."""

    def test_includes_overall(self) -> None:
        """PASS report includes overall status and the manifest path."""
        result = verify.ManifestVerifyResult(expected=["a.deb"], found=["a.deb"])
        text = verify.format_report_text(result, Path("/tmp/built_packages.txt"))
        self.assertIn("Overall result: PASS", text)
        self.assertIn("built_packages.txt", text)

    def test_includes_control_mismatches(self) -> None:
        """Control section lists field mismatches when present."""
        presence = verify.ManifestVerifyResult(expected=["a.deb"], found=["a.deb"])
        control = verify.ControlVerifyResult(
            checked=["a.deb"],
            mismatches=[
                verify.ControlFieldMismatch(
                    filename="a.deb",
                    field="Depends",
                    claimed="libc6",
                    actual="libc6 (>= 2.34)",
                )
            ],
        )
        text = verify.format_report_text(
            presence,
            Path("/tmp/built_packages.txt"),
            control=control,
            claim_path=Path("/tmp/built_packages.json"),
        )
        self.assertIn("Overall result: FAIL", text)
        self.assertIn("Depends", text)
        self.assertIn("claimed='libc6'", text)


class RunCliTest(unittest.TestCase):
    """CLI exit codes for presence + claim loading."""

    @patch("native_linux_package_pre_upload_test.summary.read_package_control_fields")
    def test_pass_when_files_exist(self, mock_read: MagicMock) -> None:
        """Exit ``0`` when Created files exist and control fields match the claim."""
        control = _deb_control()
        mock_read.return_value = dict(control)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "built_packages.txt").write_text(SAMPLE_MANIFEST, encoding="utf-8")
            packages = []
            for name in _SAMPLE_CREATED:
                (root / name).write_bytes(b"x")
                packages.append({"filename": name, "control": dict(control)})
            _write_claim(root, packages)
            self.assertEqual(verify.main(_cli_args(root)), 0)

    def test_missing_package_fails(self) -> None:
        """Exit ``1`` when Created filenames are listed but files are absent."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "built_packages.txt").write_text(SAMPLE_MANIFEST, encoding="utf-8")
            _write_claim(
                root,
                [
                    {"filename": name, "control": _deb_control()}
                    for name in _SAMPLE_CREATED
                ],
            )
            self.assertEqual(verify.main(_cli_args(root)), 1)

    def test_missing_manifest_is_config_error(self) -> None:
        """Exit ``2`` when ``built_packages.txt`` is missing."""
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(verify.main(_cli_args(Path(tmp))), 2)

    def test_missing_claim_json_is_config_error(self) -> None:
        """Exit ``2`` when ``built_packages.json`` is absent."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "built_packages.txt").write_text(
                "# Created Packages:\na.deb\n",
                encoding="utf-8",
            )
            (root / "a.deb").write_bytes(b"x")
            self.assertEqual(verify.main(_cli_args(root)), 2)


class ControlFieldVerifyTest(unittest.TestCase):
    """Claim ↔ live control-field comparison (Depends, Version, related fields)."""

    @patch("native_linux_package_pre_upload_test.summary.read_package_control_fields")
    def test_cli_pass_when_claim_matches_live(self, mock_read: MagicMock) -> None:
        """Exit ``0`` when Depends/Version and related fields match the claim."""
        control = _deb_control()
        mock_read.return_value = dict(control)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fname = "amdrocm-fft7.15_7.15.0-1_amd64.deb"
            (root / "built_packages.txt").write_text(
                f"# Created Packages:\n{fname}\n",
                encoding="utf-8",
            )
            (root / fname).write_bytes(b"x")
            _write_claim(root, [{"filename": fname, "control": control}])
            self.assertEqual(verify.main(_cli_args(root)), 0)
            mock_read.assert_called()

    @patch("native_linux_package_pre_upload_test.summary.read_package_control_fields")
    def test_depends_mismatch_fails(self, mock_read: MagicMock) -> None:
        """Exit ``1`` when live Depends differs from the claim."""
        claimed = _deb_control(Depends="amdrocm-runtime7.15, libc6")
        mock_read.return_value = _deb_control(Depends="amdrocm-runtime7.15")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fname = "amdrocm-fft7.15_7.15.0-1_amd64.deb"
            (root / "built_packages.txt").write_text(
                f"# Created Packages:\n{fname}\n",
                encoding="utf-8",
            )
            (root / fname).write_bytes(b"x")
            _write_claim(root, [{"filename": fname, "control": claimed}])
            self.assertEqual(verify.main(_cli_args(root)), 1)

    @patch("native_linux_package_pre_upload_test.summary.read_package_control_fields")
    def test_claim_probe_error_fails_verify(self, mock_read: MagicMock) -> None:
        """Exit ``1`` when the claim recorded a soft-fail probe error."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fname = "broken.deb"
            (root / "built_packages.txt").write_text(
                f"# Created Packages:\n{fname}\n",
                encoding="utf-8",
            )
            (root / fname).write_bytes(b"x")
            _write_claim(
                root,
                [{"filename": fname, "error": "dpkg-deb failed"}],
            )
            self.assertEqual(verify.main(_cli_args(root)), 1)
            mock_read.assert_not_called()

    def test_verify_control_fields_reports_version_mismatch(self) -> None:
        """Direct API: Version mismatch is recorded with claimed vs actual."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fname = "a.deb"
            (root / fname).write_bytes(b"x")
            claims = {
                fname: {
                    "filename": fname,
                    "control": _deb_control(Version="7.15.0-1"),
                }
            }
            with patch(
                "native_linux_package_pre_upload_test.summary.read_package_control_fields",
                return_value=_deb_control(Version="7.15.0-2"),
            ):
                result = verify.verify_control_fields(
                    root,
                    [fname],
                    claims,
                    pkg_type="deb",
                )
        self.assertFalse(result.passed)
        self.assertEqual(len(result.mismatches), 1)
        self.assertEqual(result.mismatches[0].field, "Version")
        self.assertEqual(result.mismatches[0].claimed, "7.15.0-1")
        self.assertEqual(result.mismatches[0].actual, "7.15.0-2")


class EnrichedPackageSummaryClaimTest(unittest.TestCase):
    """``packaging_summary.write_build_manifest`` JSON claim shape for pre-upload."""

    @patch("packaging_summary.read_package_control_fields")
    def test_write_build_manifest_emits_json_claim_for_pre_upload(
        self, mock_read: MagicMock
    ) -> None:
        """TXT inventory stays filename-only; JSON carries control fields."""
        mock_read.return_value = _deb_control(
            Depends="amdrocm-runtime7.15, libc6",
            Recommends="amdrocm-optional7.15",
        )
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            fname = "amdrocm-fft7.15_7.15.0-1_amd64.deb"
            (dest / fname).write_bytes(b"x")
            summary.write_build_manifest(
                _package_config(dest, pkg_type="deb"),
                summary.PackageList(
                    total=["amdrocm-fft"],
                    built=[fname],
                    skipped=["amdrocm-optional"],
                    failed=["amdrocm-ck"],
                ),
            )

            txt = (dest / summary.MANIFEST_TXT_NAME).read_text(encoding="utf-8")
            self.assertIn("# Created Packages:", txt)
            self.assertIn(fname, txt)

            doc = json.loads(
                (dest / summary.MANIFEST_JSON_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(doc["schema_version"], summary.MANIFEST_SCHEMA_VERSION)
            self.assertEqual(doc["pkg_type"], "deb")
            self.assertEqual(doc["rocm_version"], "7.15.0")
            self.assertEqual(doc["version_suffix"], "1")
            self.assertEqual(doc["failed"], ["amdrocm-ck"])
            self.assertEqual(doc["skipped"], ["amdrocm-optional"])
            self.assertEqual(len(doc["packages"]), 1)
            pkg = doc["packages"][0]
            self.assertEqual(pkg["filename"], fname)
            control = pkg["control"]
            self.assertEqual(control["Package"], "amdrocm-fft7.15")
            self.assertEqual(control["Version"], "7.15.0-1")
            self.assertEqual(control["Depends"], "amdrocm-runtime7.15, libc6")
            self.assertEqual(control["Recommends"], "amdrocm-optional7.15")
            for key in summary.DEB_CONTROL_FIELDS:
                self.assertIn(key, control)

    @patch("packaging_summary.subprocess.run")
    def test_rpm_claim_includes_requires_and_related_fields(
        self, mock_run: MagicMock
    ) -> None:
        """RPM JSON claim includes Requires/Provides/Recommends for pre-upload."""
        scalar = "amdrocm-fft7.15\n7.15.0\n1\nx86_64\nDevelopment/Libraries\nMIT\nAMD\n"

        def _run(cmd, **_kwargs):
            m = MagicMock(returncode=0, stdout="", stderr="")
            if len(cmd) >= 3 and cmd[2] == "--queryformat":
                m.stdout = scalar
            elif len(cmd) >= 2 and cmd[1] == "-qpR":
                m.stdout = (
                    "amdrocm-runtime7.15 = 7.15.0-1\nrpmlib(CompressedFileNames)\n"
                )
            elif "--provides" in cmd:
                m.stdout = "amdrocm-fft7.15 = 7.15.0-1\n"
            elif "--recommends" in cmd:
                m.stdout = "amdrocm-extra7.15\n"
            return m

        mock_run.side_effect = _run
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            fname = "amdrocm-fft7.15-7.15.0-1.x86_64.rpm"
            (dest / fname).write_bytes(b"x")
            summary.write_build_manifest(
                _package_config(dest, pkg_type="rpm"),
                summary.PackageList(
                    total=["amdrocm-fft"],
                    built=[fname],
                    skipped=[],
                    failed=[],
                ),
            )
            doc = json.loads(
                (dest / summary.MANIFEST_JSON_NAME).read_text(encoding="utf-8")
            )
            control = doc["packages"][0]["control"]
            self.assertEqual(doc["pkg_type"], "rpm")
            self.assertEqual(control["Name"], "amdrocm-fft7.15")
            self.assertEqual(control["Version"], "7.15.0")
            self.assertEqual(control["Requires"], "amdrocm-runtime7.15 = 7.15.0-1")
            self.assertNotIn("rpmlib(", control["Requires"])
            self.assertEqual(control["Provides"], "amdrocm-fft7.15 = 7.15.0-1")
            self.assertEqual(control["Recommends"], "amdrocm-extra7.15")
            for name, _ in summary.RPM_LIST_QUERIES:
                self.assertIn(name, control)

    @patch("packaging_summary.read_package_control_fields")
    def test_control_read_failure_recorded_as_error(self, mock_read: MagicMock) -> None:
        """Unreadable package still appears in the claim with ``error`` (no crash)."""
        mock_read.side_effect = RuntimeError("dpkg-deb failed")
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            fname = "broken.deb"
            (dest / fname).write_bytes(b"x")
            summary.write_build_manifest(
                _package_config(dest),
                summary.PackageList(
                    total=["broken"], built=[fname], skipped=[], failed=[]
                ),
            )
            pkg = json.loads(
                (dest / summary.MANIFEST_JSON_NAME).read_text(encoding="utf-8")
            )["packages"][0]
            self.assertEqual(pkg["filename"], fname)
            self.assertIn("error", pkg)
            self.assertNotIn("control", pkg)


if __name__ == "__main__":
    unittest.main()
