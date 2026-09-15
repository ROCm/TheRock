# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import resolve_jax_nightly_version as m

_INDEX_PREFIX = "https://us-python.pkg.dev/ml-oss-artifacts-published/jax/jaxlib/"


def _index_html(filenames: list[str]) -> str:
    """Renders a PEP 503 simple index page like the JAX nightly index serves."""
    anchors = "\n".join(
        f'<a href="{_INDEX_PREFIX}{name}#sha256=abc123">{name}</a><br/>'
        for name in filenames
    )
    return f"<!DOCTYPE html><html><body><h1>Links for jaxlib</h1>\n{anchors}\n</body></html>"


# The index carries every base version ever published, in no particular
# order, for several Python versions and platforms.
_INDEX_FILENAMES = [
    "jaxlib-0.11.2.dev20260914-cp312-cp312-manylinux_2_27_x86_64.whl",
    "jaxlib-0.11.2.dev20260914-cp313-cp313-manylinux_2_27_x86_64.whl",
    "jaxlib-0.11.2.dev20260914-cp313-cp313t-manylinux_2_27_x86_64.whl",
    "jaxlib-0.11.2.dev20260914-cp312-cp312-macosx_11_0_arm64.whl",
    "jaxlib-0.11.2.dev20260914-cp312-cp312-manylinux_2_27_aarch64.whl",
    "jaxlib-0.11.2.dev20260913-cp312-cp312-manylinux_2_27_x86_64.whl",
    # A day with only a Windows wheel for this Python version.
    "jaxlib-0.11.2.dev20260915-cp312-cp312-win_amd64.whl",
    # A day published only for another Python version.
    "jaxlib-0.11.2.dev20260916-cp313-cp313-manylinux_2_27_x86_64.whl",
    "jaxlib-0.11.1.dev20260830-cp312-cp312-manylinux_2_27_x86_64.whl",
    "jaxlib-0.11.1-cp312-cp312-manylinux_2_27_x86_64.whl",
    "jaxlib-0.9.2.dev20260309-cp312-cp312-win_amd64.whl",
    "jaxlib-0.5.3-cp313-cp313t-manylinux2014_x86_64.whl",
]


class ParseIndexTest(unittest.TestCase):
    def test_parses_wheel_filenames_from_anchor_hrefs(self):
        filenames = m.parse_index_wheel_filenames(_index_html(_INDEX_FILENAMES))
        self.assertEqual(filenames, _INDEX_FILENAMES)

    def test_ignores_non_wheel_links_and_decodes_percent_escapes(self):
        html = (
            '<a href="../">up</a>'
            '<a href="https://x/y/jaxlib-0.11.2.dev20260914-cp312-cp312-manylinux_2_27_x86_64.whl#sha256=1">w</a>'
            '<a href="https://x/y/jaxlib-0.11.2.tar.gz">s</a>'
            '<a href="https://x/y/jaxlib-0.11.2.dev20260913-cp312-cp312-manylinux_2_27_x86_64%2Ewhl">e</a>'
        )
        self.assertEqual(
            m.parse_index_wheel_filenames(html),
            [
                "jaxlib-0.11.2.dev20260914-cp312-cp312-manylinux_2_27_x86_64.whl",
                "jaxlib-0.11.2.dev20260913-cp312-cp312-manylinux_2_27_x86_64.whl",
            ],
        )


class ParseNightlyWheelsTest(unittest.TestCase):
    def test_keeps_only_dated_dev_releases_of_jaxlib(self):
        nightlies = m.parse_nightly_wheels(
            [
                "jaxlib-0.11.2.dev20260914-cp312-cp312-manylinux_2_27_x86_64.whl",
                # Not a nightly: a release, a dev number that is not a date,
                # a local version, a pre-release, another project, junk.
                "jaxlib-0.11.1-cp312-cp312-manylinux_2_27_x86_64.whl",
                "jaxlib-0.11.2.dev0-cp312-cp312-manylinux_2_27_x86_64.whl",
                "jaxlib-0.11.2.dev20261399-cp312-cp312-manylinux_2_27_x86_64.whl",
                "jaxlib-0.11.2.dev20260914+abc123-cp312-cp312-manylinux_2_27_x86_64.whl",
                "jaxlib-0.11.2rc1.dev20260914-cp312-cp312-manylinux_2_27_x86_64.whl",
                "jax-0.11.2.dev20260914-py3-none-any.whl",
                "not-a-wheel.whl",
            ]
        )
        self.assertEqual(
            nightlies,
            [
                m.NightlyWheel(
                    base_version="0.11.2",
                    build_date="20260914",
                    python_tag="cp312-cp312",
                    platform="manylinux_2_27_x86_64",
                )
            ],
        )

    def test_free_threaded_abi_is_a_distinct_tag(self):
        nightlies = m.parse_nightly_wheels(
            ["jaxlib-0.11.2.dev20260914-cp313-cp313t-manylinux_2_27_x86_64.whl"]
        )
        self.assertEqual(nightlies[0].python_tag, "cp313-cp313t")


class ResolveBuildDateTest(unittest.TestCase):
    def setUp(self):
        self.nightlies = m.parse_nightly_wheels(_INDEX_FILENAMES)

    def test_picks_newest_date_with_a_manylinux_wheel_for_the_python_version(self):
        # 20260915 has only a Windows wheel and 20260916 only a cp313 wheel,
        # so the newest date the cp312 test job can install is 20260914.
        self.assertEqual(
            m.resolve_nightly_build_date(
                self.nightlies, base_version="0.11.2", python_version="3.12"
            ),
            "20260914",
        )
        self.assertEqual(
            m.resolve_nightly_build_date(
                self.nightlies, base_version="0.11.2", python_version="3.13"
            ),
            "20260916",
        )

    def test_matches_base_version_exactly(self):
        self.assertEqual(
            m.resolve_nightly_build_date(
                self.nightlies, base_version="0.11.1", python_version="3.12"
            ),
            "20260830",
        )

    def test_no_matching_nightly_names_what_exists(self):
        with self.assertRaises(m.ResolveError) as ctx:
            m.resolve_nightly_build_date(
                self.nightlies, base_version="0.12.0", python_version="3.12"
            )
        self.assertIn("0.12.0.devYYYYMMDD", str(ctx.exception))
        self.assertIn("0.11.2", str(ctx.exception))

    def test_python_tag_for_version(self):
        self.assertEqual(m.python_tag_for_version("3.12"), "cp312-cp312")
        self.assertEqual(m.python_tag_for_version("3.9"), "cp39-cp39")


class ReadJaxBaseVersionTest(unittest.TestCase):
    def _write_checkout(self, tmp: str, version_py: str) -> Path:
        jax_dir = Path(tmp) / "jax"
        jax_dir.mkdir()
        (jax_dir / "version.py").write_text(version_py)
        return Path(tmp)

    def test_reads_version_assignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = self._write_checkout(
                tmp,
                textwrap.dedent(
                    """\
                    import os

                    _version = "0.11.2"

                    # The following line is overwritten by build scripts.
                    _release_version: str | None = None
                    """
                ),
            )
            self.assertEqual(m.read_jax_base_version(checkout), "0.11.2")

    def test_missing_assignment_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = self._write_checkout(tmp, "_release_version = None\n")
            with self.assertRaises(m.ResolveError):
                m.read_jax_base_version(checkout)

    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(m.ResolveError):
                m.read_jax_base_version(Path(tmp))


class MainTest(unittest.TestCase):
    def test_writes_build_date_and_version_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "jax").mkdir()
            (Path(tmp) / "jax" / "version.py").write_text('_version = "0.11.2"\n')
            output_file = Path(tmp) / "github_output"
            with (
                mock.patch.object(
                    m,
                    "fetch_index_page",
                    return_value=_index_html(_INDEX_FILENAMES),
                ) as fetch,
                mock.patch.dict(os.environ, {"GITHUB_OUTPUT": os.fspath(output_file)}),
            ):
                rc = m.main(
                    [
                        "--jax-source-dir",
                        tmp,
                        "--python-version",
                        "3.12",
                        "--index-url",
                        "https://example.test/simple/",
                    ]
                )

            self.assertEqual(rc, 0)
            fetch.assert_called_once_with("https://example.test/simple/", "jaxlib", 60)
            self.assertEqual(
                output_file.read_text().splitlines(),
                ["build_date=20260914", "jax_nightly_version=0.11.2.dev20260914"],
            )

    def test_no_nightly_is_a_clear_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "jax").mkdir()
            (Path(tmp) / "jax" / "version.py").write_text('_version = "0.12.0"\n')
            with mock.patch.object(
                m, "fetch_index_page", return_value=_index_html(_INDEX_FILENAMES)
            ):
                rc = m.main(["--jax-source-dir", tmp, "--python-version", "3.12"])
            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
