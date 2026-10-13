# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from packaging.version import Version

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import resolve_jax_version as m
from determine_version import derive_version_suffix

ROCM_VERSION = "7.14.0a20261002"


def _checkout_committed_at(path: Path, iso_date: str) -> Path:
    """Makes a one-commit git repository whose HEAD has the given dates."""
    env = {
        **os.environ,
        "GIT_AUTHOR_DATE": iso_date,
        "GIT_COMMITTER_DATE": iso_date,
    }
    git = ["git", "-c", "user.name=test", "-c", "user.email=test@example.com"]
    subprocess.run([*git, "init", "-q"], cwd=path, check=True)
    subprocess.run(
        [*git, "commit", "-q", "--allow-empty", "-m", "tip"],
        cwd=path,
        env=env,
        check=True,
    )
    return path


class ResolveJaxVersionTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.checkout = _checkout_committed_at(
            Path(tmp.name), "2026-10-02T09:51:21-07:00"
        )

    def test_release_is_the_rocm_suffix_alone(self):
        self.assertEqual(
            m.ml_wheel_version_suffix("release", ROCM_VERSION, self.checkout),
            derive_version_suffix(ROCM_VERSION),
        )

    def test_nightly_dates_the_suffix_from_the_head_commit(self):
        self.assertEqual(
            m.ml_wheel_version_suffix("nightly", ROCM_VERSION, self.checkout),
            ".dev20261002+rocm7.14.0a20261002",
        )

    def test_the_commit_date_is_taken_in_utc(self):
        # 23:30 in UTC-7 is already the next day in UTC; the runner's own
        # timezone must not move the date.
        with tempfile.TemporaryDirectory() as tmp:
            late = _checkout_committed_at(Path(tmp), "2026-10-02T23:30:00-07:00")
            self.assertEqual(m.commit_date(late), "20261003")

    def test_nightly_versions_are_prereleases_and_release_ones_are_not(self):
        # pip install without --pre must keep resolving real releases.
        nightly = m.ml_wheel_version_suffix("nightly", ROCM_VERSION, self.checkout)
        release = m.ml_wheel_version_suffix("release", "7.14.0", self.checkout)
        self.assertTrue(Version(f"0.12.0{nightly}").is_prerelease)
        self.assertFalse(Version(f"0.11.2{release}").is_prerelease)

    def test_nightly_builds_jax_and_jaxlib_from_the_same_checkout(self):
        self.assertEqual(
            m.WHEELS["nightly"],
            ("jax", "jaxlib", "jax-rocm-plugin", "jax-rocm-pjrt"),
        )
        self.assertEqual(m.WHEELS["release"], ("jax-rocm-plugin", "jax-rocm-pjrt"))

    def test_an_unknown_wheel_type_is_rejected(self):
        with self.assertRaises(SystemExit):
            m.main(
                [
                    "--wheel-type",
                    "custom",
                    "--rocm-version",
                    ROCM_VERSION,
                    "--jax-source-dir",
                    os.fspath(self.checkout),
                ]
            )

    def test_main_writes_the_suffix_and_wheels_to_github_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "github_output"
            with mock.patch.dict(os.environ, {"GITHUB_OUTPUT": os.fspath(output)}):
                m.main(
                    [
                        "--wheel-type",
                        "nightly",
                        "--rocm-version",
                        ROCM_VERSION,
                        "--jax-source-dir",
                        os.fspath(self.checkout),
                    ]
                )
            self.assertEqual(
                output.read_text().splitlines(),
                [
                    "ml_wheel_version_suffix=.dev20261002+rocm7.14.0a20261002",
                    "wheels=jax,jaxlib,jax-rocm-plugin,jax-rocm-pjrt",
                ],
            )


if __name__ == "__main__":
    unittest.main()
