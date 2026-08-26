#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``build_package_repo.py``.

Regression guards for local repo metadata creation (Phase 2 split from upload).

Coverage:

  - ``generate_release_file_with_checksums`` — DEB ``Release`` has checksum sections

Prerequisites:

  - Python 3.10 or newer
  - Run from TheROCK repository root (modules resolved via ``__file__``)
  - Stdlib only (no boto3 / live ``createrepo_c`` in these tests)

Run::

    python3.12 build_tools/packaging/linux/tests/build_package_repo_test.py -v

    python3.12 -m unittest discover -s build_tools/packaging/linux/tests \\
        -p 'build_package_repo_test.py' -v
"""

import gzip
import os
import sys
import tempfile
import unittest
from pathlib import Path

THIS_SCRIPT_DIR = Path(__file__).resolve().parent
LINUX_DIR = THIS_SCRIPT_DIR.parent

if os.fspath(LINUX_DIR) not in sys.path:
    sys.path.insert(0, os.fspath(LINUX_DIR))

import build_package_repo as repo_builder  # noqa: E402

TEST_JOB_TYPE = "nightly"


class GenerateReleaseFileTest(unittest.TestCase):
    """Tests for ``generate_release_file_with_checksums()`` (DEB upload-ready Release)."""

    def test_release_includes_checksum_sections(self) -> None:
        """Release must include MD5/SHA256 sections before S3 upload."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dists_dir = Path(temp_dir) / "main" / "binary-amd64"
            dists_dir.mkdir(parents=True)
            (dists_dir / "Packages").write_text("Package: demo\n", encoding="utf-8")
            with gzip.open(dists_dir / "Packages.gz", "wb") as handle:
                handle.write(b"Package: demo\n")

            release_file = Path(temp_dir) / "Release"
            repo_builder.generate_release_file_with_checksums(
                release_file, TEST_JOB_TYPE, dists_dir
            )
            release_text = release_file.read_text(encoding="utf-8")

        self.assertIn("MD5Sum:", release_text)
        self.assertIn("SHA256:", release_text)
        self.assertIn("main/binary-amd64/Packages", release_text)


if __name__ == "__main__":
    unittest.main()
