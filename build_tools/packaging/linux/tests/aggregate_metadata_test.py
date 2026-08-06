#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for aggregate_deb_metadata.py and aggregate_rpm_metadata.py."""

import argparse
import bz2
import gzip
import lzma
import sys
import textwrap
import unittest
import urllib.error
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, call, patch

# Add parent directory to path so we can import the scripts directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import aggregate_deb_metadata as deb
import aggregate_rpm_metadata as rpm


# ---------------------------------------------------------------------------
# DEB tests
# ---------------------------------------------------------------------------


class TestDebParseSource(unittest.TestCase):
    def test_valid(self):
        result = deb.parse_source("core,https://example.com/deb,dists,pool/core")
        self.assertEqual(result["name"], "core")
        self.assertEqual(result["base_url"], "https://example.com/deb")
        self.assertEqual(result["style"], "dists")
        self.assertEqual(result["pool_prefix"], "pool/core")

    def test_valid_flat(self):
        result = deb.parse_source("rvs,https://example.com/rvs/deb,flat,pool/rvs")
        self.assertEqual(result["style"], "flat")

    def test_strips_whitespace(self):
        result = deb.parse_source(" core , https://example.com , dists , pool/core ")
        self.assertEqual(result["name"], "core")
        self.assertEqual(result["base_url"], "https://example.com")

    def test_wrong_field_count(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            deb.parse_source("core,https://example.com,dists")

    def test_invalid_style(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            deb.parse_source("core,https://example.com,zip,pool/core")

    def test_url_with_commas_in_prefix(self):
        # pool_prefix is the 4th field — earlier commas should be consumed by split(,3)
        result = deb.parse_source("core,https://example.com,dists,pool/core/extra")
        self.assertEqual(result["pool_prefix"], "pool/core/extra")


class TestDebParseStanzas(unittest.TestCase):
    def _stanzas(self, text):
        return deb.parse_stanzas(textwrap.dedent(text).strip())

    def test_single_stanza(self):
        stanzas = self._stanzas(
            """
            Package: mypkg
            Version: 1.0
            Architecture: amd64
        """
        )
        self.assertEqual(len(stanzas), 1)
        self.assertEqual(stanzas[0]["Package"], "mypkg")
        self.assertEqual(stanzas[0]["Version"], "1.0")

    def test_multi_stanza(self):
        stanzas = self._stanzas(
            """
            Package: pkg1
            Version: 1.0

            Package: pkg2
            Version: 2.0
        """
        )
        self.assertEqual(len(stanzas), 2)
        self.assertEqual(stanzas[0]["Package"], "pkg1")
        self.assertEqual(stanzas[1]["Package"], "pkg2")

    def test_continuation_line(self):
        stanzas = self._stanzas(
            """
            Package: mypkg
            Description: Short desc
             Long continuation line
        """
        )
        self.assertIn("Long continuation line", stanzas[0]["Description"])

    def test_empty_blocks_ignored(self):
        stanzas = self._stanzas(
            """
            Package: pkg1
            Version: 1.0


            Package: pkg2
            Version: 2.0
        """
        )
        self.assertEqual(len(stanzas), 2)

    def test_empty_input(self):
        self.assertEqual(deb.parse_stanzas(""), [])


class TestDebRewriteFilename(unittest.TestCase):
    def test_dists_style_path(self):
        stanza = {
            "Package": "mypkg",
            "Filename": "pool/main/m/mypkg/mypkg_1.0_amd64.deb",
        }
        result = deb.rewrite_filename(stanza, "https://example.com", "pool/core")
        self.assertEqual(result["Filename"], "pool/core/mypkg_1.0_amd64.deb")

    def test_flat_style_basename_only(self):
        stanza = {"Package": "mypkg", "Filename": "mypkg_1.0_amd64.deb"}
        result = deb.rewrite_filename(stanza, "https://example.com", "pool/rvs")
        self.assertEqual(result["Filename"], "pool/rvs/mypkg_1.0_amd64.deb")

    def test_original_stanza_not_mutated(self):
        stanza = {"Package": "mypkg", "Filename": "pool/main/mypkg_1.0_amd64.deb"}
        deb.rewrite_filename(stanza, "https://example.com", "pool/core")
        self.assertEqual(stanza["Filename"], "pool/main/mypkg_1.0_amd64.deb")

    def test_missing_filename_field(self):
        stanza = {"Package": "mypkg"}
        result = deb.rewrite_filename(stanza, "https://example.com", "pool/core")
        self.assertEqual(result["Filename"], "pool/core/")


class TestDebFetchPackages(unittest.TestCase):
    def _make_http_error(self, code):
        return urllib.error.HTTPError(
            url="http://x", code=code, msg="", hdrs={}, fp=None
        )

    def test_xz_preferred(self):
        xz_data = lzma.compress(b"Package: pkg\nVersion: 1.0\n")
        with patch.object(deb, "fetch_bytes", return_value=xz_data) as mock_fetch:
            result = deb.fetch_packages(
                "https://example.com", "dists", "stable", "main", "amd64"
            )
        self.assertEqual(result, b"Package: pkg\nVersion: 1.0\n")
        # Should have fetched the .xz URL first
        self.assertIn(".xz", mock_fetch.call_args[0][0])

    def test_falls_back_to_gz_when_xz_and_bz2_404(self):
        gz_data = gzip.compress(b"Package: pkg\nVersion: 2.0\n")

        def side_effect(url):
            if url.endswith(".xz") or url.endswith(".bz2"):
                raise self._make_http_error(404)
            return gz_data

        with patch.object(deb, "fetch_bytes", side_effect=side_effect):
            result = deb.fetch_packages(
                "https://example.com", "dists", "stable", "main", "amd64"
            )
        self.assertEqual(result, b"Package: pkg\nVersion: 2.0\n")

    def test_falls_back_through_all_to_uncompressed(self):
        raw = b"Package: pkg\nVersion: 3.0\n"

        def side_effect(url):
            if url.endswith((".xz", ".bz2", ".gz")):
                raise self._make_http_error(404)
            return raw

        with patch.object(deb, "fetch_bytes", side_effect=side_effect):
            result = deb.fetch_packages(
                "https://example.com", "flat", "stable", "main", "amd64"
            )
        self.assertEqual(result, raw)

    def test_raises_when_all_404(self):
        with patch.object(deb, "fetch_bytes", side_effect=self._make_http_error(404)):
            with self.assertRaises(RuntimeError):
                deb.fetch_packages(
                    "https://example.com", "dists", "stable", "main", "amd64"
                )

    def test_non_404_error_reraises(self):
        with patch.object(deb, "fetch_bytes", side_effect=self._make_http_error(500)):
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                deb.fetch_packages(
                    "https://example.com", "dists", "stable", "main", "amd64"
                )
            self.assertEqual(ctx.exception.code, 500)

    def test_flat_style_url_construction(self):
        with patch.object(deb, "fetch_bytes", side_effect=self._make_http_error(404)):
            with self.assertRaises(RuntimeError):
                deb.fetch_packages(
                    "https://example.com/rvs", "flat", "stable", "main", "amd64"
                )


class TestDebGenerateRelease(unittest.TestCase):
    def _release(self, **kwargs):
        defaults = dict(
            suite="stable",
            component="main",
            arch="amd64",
            packages_bytes=b"Package: p\n",
            packages_gz_bytes=gzip.compress(b"Package: p\n"),
        )
        defaults.update(kwargs)
        return deb.generate_release(**defaults)

    def test_required_fields_present(self):
        text = self._release()
        self.assertIn("Suite: stable", text)
        self.assertIn("Codename: stable", text)
        self.assertIn("Components: main", text)
        self.assertIn("Architectures: amd64", text)
        self.assertIn("MD5Sum:", text)
        self.assertIn("SHA1:", text)
        self.assertIn("SHA256:", text)

    def test_date_format_has_plus0000(self):
        text = self._release()
        # APT requires +0000 not UTC
        self.assertIn("+0000", text)
        self.assertNotIn("UTC", text)

    def test_package_paths_listed(self):
        text = self._release()
        self.assertIn("main/binary-amd64/Packages", text)
        self.assertIn("main/binary-amd64/Packages.gz", text)


class TestDebNevraCollision(unittest.TestCase):
    """Test NEVRA collision detection in the merge loop."""

    def _make_stanza(self, name, version, arch):
        return {
            "Package": name,
            "Version": version,
            "Architecture": arch,
            "Filename": f"{name}.deb",
        }

    def test_same_nevra_different_sources_raises(self):
        declared = {"pkg1_1.0_amd64": "core"}
        stanza = self._make_stanza("pkg1", "1.0", "amd64")
        nevra_key = f"{stanza['Package']}_{stanza['Version']}_{stanza['Architecture']}"
        self.assertIn(nevra_key, declared)
        self.assertNotEqual(declared[nevra_key], "rvs")

    def test_same_nevra_same_source_allowed(self):
        declared = {"pkg1_1.0_amd64": "core"}
        self.assertEqual(declared.get("pkg1_1.0_amd64"), "core")

    def test_different_versions_no_collision(self):
        declared = {"pkg1_1.0_amd64": "core"}
        nevra = "pkg1_2.0_amd64"
        self.assertNotIn(nevra, declared)


class TestDebStanzaRoundTrip(unittest.TestCase):
    def test_roundtrip(self):
        original = "Package: mypkg\nVersion: 1.0\nArchitecture: amd64"
        stanzas = deb.parse_stanzas(original)
        self.assertEqual(len(stanzas), 1)
        text = deb.stanza_to_text(stanzas[0])
        self.assertIn("Package: mypkg", text)
        self.assertIn("Version: 1.0", text)


# ---------------------------------------------------------------------------
# RPM tests
# ---------------------------------------------------------------------------

NS_COMMON = "http://linux.duke.edu/metadata/common"
NS_REPO = "http://linux.duke.edu/metadata/repo"


class TestRpmParseSource(unittest.TestCase):
    def test_valid(self):
        result = rpm.parse_source("core,https://example.com/rpm/x86_64,core")
        self.assertEqual(result["name"], "core")
        self.assertEqual(result["base_url"], "https://example.com/rpm/x86_64")
        self.assertEqual(result["href_prefix"], "core")

    def test_strips_whitespace(self):
        result = rpm.parse_source(" rvs , https://example.com/rvs , rvs ")
        self.assertEqual(result["name"], "rvs")

    def test_wrong_field_count_too_few(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            rpm.parse_source("core,https://example.com")

    def test_wrong_field_count_too_many(self):
        # Only 3 fields expected; split(,2) handles extra commas in base_url
        result = rpm.parse_source("core,https://example.com/a,b,extra,core")
        # href_prefix gets everything after 2nd comma
        self.assertEqual(result["href_prefix"], "b,extra,core")


class TestRpmRewriteLocationHref(unittest.TestCase):
    def _make_pkg(self, href):
        pkg = ET.Element(f"{{{NS_COMMON}}}package", type="rpm")
        ET.SubElement(pkg, f"{{{NS_COMMON}}}name").text = "mypkg"
        loc = ET.SubElement(pkg, f"{{{NS_COMMON}}}location")
        loc.set("href", href)
        return pkg

    def test_rewrites_simple_basename(self):
        pkg = self._make_pkg("mypkg-1.0.x86_64.rpm")
        rpm.rewrite_location_href(pkg, "core")
        ns = {"c": NS_COMMON}
        loc = pkg.find("c:location", ns)
        self.assertEqual(loc.get("href"), "core/mypkg-1.0.x86_64.rpm")

    def test_rewrites_path_with_subdirectory(self):
        pkg = self._make_pkg("Packages/mypkg-1.0.x86_64.rpm")
        rpm.rewrite_location_href(pkg, "rvs")
        ns = {"c": NS_COMMON}
        loc = pkg.find("c:location", ns)
        self.assertEqual(loc.get("href"), "rvs/mypkg-1.0.x86_64.rpm")

    def test_missing_location_raises(self):
        pkg = ET.Element(f"{{{NS_COMMON}}}package", type="rpm")
        ET.SubElement(pkg, f"{{{NS_COMMON}}}name").text = "mypkg"
        with self.assertRaises(RuntimeError) as ctx:
            rpm.rewrite_location_href(pkg, "core")
        self.assertIn("mypkg", str(ctx.exception))
        self.assertIn("location", str(ctx.exception).lower())


class TestRpmFetchPrimaryXml(unittest.TestCase):
    def _make_repomd(self, primary_href):
        repomd = ET.Element(f"{{{NS_REPO}}}repomd")
        data = ET.SubElement(repomd, f"{{{NS_REPO}}}data", type="primary")
        loc = ET.SubElement(data, f"{{{NS_REPO}}}location")
        loc.set("href", primary_href)
        return ET.tostring(repomd, encoding="UTF-8", xml_declaration=True)

    def test_fetches_and_decompresses(self):
        primary_content = b"<metadata/>"
        primary_gz = gzip.compress(primary_content)
        repomd_bytes = self._make_repomd("repodata/primary.xml.gz")

        def side_effect(url):
            if "repomd.xml" in url:
                return repomd_bytes
            return primary_gz

        with patch.object(rpm, "fetch_bytes", side_effect=side_effect):
            result = rpm.fetch_primary_xml("https://example.com/rpm")
        self.assertEqual(result, primary_content)

    def test_raises_when_no_primary_in_repomd(self):
        repomd = ET.Element(f"{{{NS_REPO}}}repomd")
        repomd_bytes = ET.tostring(repomd)

        with patch.object(rpm, "fetch_bytes", return_value=repomd_bytes):
            with self.assertRaises(RuntimeError) as ctx:
                rpm.fetch_primary_xml("https://example.com/rpm")
        self.assertIn("primary", str(ctx.exception).lower())


class TestRpmNevraCollision(unittest.TestCase):
    def _make_pkg(self, name, epoch, ver, rel, arch):
        pkg = ET.Element(f"{{{NS_COMMON}}}package", type="rpm")
        ET.SubElement(pkg, f"{{{NS_COMMON}}}name").text = name
        ET.SubElement(pkg, f"{{{NS_COMMON}}}arch").text = arch
        ver_el = ET.SubElement(pkg, f"{{{NS_COMMON}}}version")
        ver_el.set("epoch", epoch)
        ver_el.set("ver", ver)
        ver_el.set("rel", rel)
        loc = ET.SubElement(pkg, f"{{{NS_COMMON}}}location")
        loc.set("href", f"{name}-{ver}.{arch}.rpm")
        return pkg

    def _nevra_key(self, pkg):
        ns = {"c": NS_COMMON}
        name = pkg.find("c:name", ns).text
        arch = pkg.find("c:arch", ns).text
        ver_el = pkg.find("c:version", ns)
        ver_str = f"{ver_el.get('epoch')}:{ver_el.get('ver')}-{ver_el.get('rel')}"
        return f"{name}-{ver_str}.{arch}"

    def test_same_nevra_different_sources_detected(self):
        pkg = self._make_pkg("mypkg", "0", "1.0", "1.el8", "x86_64")
        declared = {self._nevra_key(pkg): "core"}
        nevra = self._nevra_key(pkg)
        self.assertIn(nevra, declared)
        self.assertNotEqual(declared[nevra], "rvs")

    def test_different_versions_no_collision(self):
        pkg1 = self._make_pkg("mypkg", "0", "1.0", "1.el8", "x86_64")
        pkg2 = self._make_pkg("mypkg", "0", "2.0", "1.el8", "x86_64")
        declared = {self._nevra_key(pkg1): "core"}
        self.assertNotIn(self._nevra_key(pkg2), declared)


class TestRpmBuildRepomdXml(unittest.TestCase):
    def test_output_is_valid_xml(self):
        primary_xml = b"<metadata/>"
        primary_gz = gzip.compress(primary_xml)
        result = rpm.build_repomd_xml(primary_gz, primary_xml, timestamp=1000000)
        root = ET.fromstring(result)
        self.assertIsNotNone(root)

    def test_contains_primary_data(self):
        primary_xml = b"<metadata/>"
        primary_gz = gzip.compress(primary_xml)
        result = rpm.build_repomd_xml(primary_gz, primary_xml, timestamp=1000000)
        self.assertIn(b"primary", result)
        self.assertIn(b"primary.xml.gz", result)

    def test_checksums_present(self):
        primary_xml = b"<metadata/>"
        primary_gz = gzip.compress(primary_xml)
        result = rpm.build_repomd_xml(primary_gz, primary_xml, timestamp=1000000)
        self.assertIn(b"sha256", result)


if __name__ == "__main__":
    unittest.main()
