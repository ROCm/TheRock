# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
from gitleaks import (
    _CONFIG_PATH,
    _GITLEAKS_TARBALL_SHA256,
    _LEAK_SECURITY_SEVERITY_HIGH,
    _ReportTarget,
    _determine_log_opts,
    _enrich_sarif_with_security_severity,
    _parse_report_formats,
    _resolve_config_path,
    _sha256_of,
)


class ParseReportFormatsTest(unittest.TestCase):
    """Tests for `_parse_report_formats`."""

    def test_default_sarif(self):
        targets = _parse_report_formats("sarif")
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].fmt, "sarif")
        self.assertEqual(targets[0].path, Path("gitleaks-report.sarif"))

    def test_multiple_formats_with_whitespace_and_dedup(self):
        targets = _parse_report_formats(" sarif , json , sarif ,csv ")
        self.assertEqual([t.fmt for t in targets], ["sarif", "json", "csv"])
        self.assertEqual(
            [t.path for t in targets],
            [
                Path("gitleaks-report.sarif"),
                Path("gitleaks-report.json"),
                Path("gitleaks-report.csv"),
            ],
        )

    def test_junit_uses_xml_extension(self):
        targets = _parse_report_formats("junit")
        self.assertEqual(targets[0].path, Path("gitleaks-report.xml"))

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parse_report_formats("")
        self.assertIn("report_formats is empty", str(ctx.exception))

    def test_only_whitespace_raises(self):
        with self.assertRaises(ValueError):
            _parse_report_formats(" , , ")

    def test_unknown_format_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parse_report_formats("sarif,xml")
        self.assertIn("'xml'", str(ctx.exception))


class DetermineLogOptsTest(unittest.TestCase):
    """Tests for `_determine_log_opts`."""

    def test_scan_mode_all_returns_empty(self):
        self.assertEqual(_determine_log_opts("all", "pull_request", {}), "")
        self.assertEqual(_determine_log_opts("all", "release", {"unrelated": 1}), "")

    def test_pull_request_returns_sha_range_without_no_merges(self):
        event = {"pull_request": {"base": {"sha": "aaa"}, "head": {"sha": "bbb"}}}
        log_opts = _determine_log_opts("changed", "pull_request", event)
        self.assertEqual(log_opts, "aaa..bbb")
        self.assertNotIn("--no-merges", log_opts)

    def test_pull_request_target_is_explicitly_rejected(self):
        event = {"pull_request": {"base": {"sha": "aaa"}, "head": {"sha": "bbb"}}}
        with self.assertRaises(ValueError) as ctx:
            _determine_log_opts("changed", "pull_request_target", event)
        self.assertIn("pull_request_target is not supported", str(ctx.exception))

    def test_push_returns_sha_range_without_no_merges(self):
        log_opts = _determine_log_opts(
            "changed", "push", {"before": "xxx", "after": "yyy"}
        )
        self.assertEqual(log_opts, "xxx..yyy")
        self.assertNotIn("--no-merges", log_opts)

    def test_push_new_ref_returns_empty(self):
        log_opts = _determine_log_opts(
            "changed", "push", {"before": "0" * 40, "after": "yyy"}
        )
        self.assertEqual(log_opts, "")

    def test_unknown_event_type_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _determine_log_opts("changed", "release", {})
        self.assertIn("'release'", str(ctx.exception))
        self.assertIn("scan_mode='all'", str(ctx.exception))

    def test_unset_event_name_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _determine_log_opts("changed", "", {})
        self.assertIn("'<unset>'", str(ctx.exception))

    def test_pull_request_malformed_payload_raises_key_error(self):
        with self.assertRaises(KeyError):
            _determine_log_opts("changed", "pull_request", {"pull_request": {}})

    def test_push_malformed_payload_raises_key_error(self):
        with self.assertRaises(KeyError):
            _determine_log_opts("changed", "push", {})


class EnrichSarifTest(unittest.TestCase):
    """Tests for `_enrich_sarif_with_security_severity`."""

    def _write_sarif(self, payload: object) -> Path:
        fd, name = tempfile.mkstemp(suffix=".sarif")
        os.close(fd)
        path = Path(name)
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.addCleanup(path.unlink, missing_ok=True)
        return path

    def test_backfills_level_and_security_severity(self):
        path = self._write_sarif(
            {"runs": [{"results": [{"message": {"text": "leak"}}]}]}
        )
        _enrich_sarif_with_security_severity(path)
        data = json.loads(path.read_text())
        result = data["runs"][0]["results"][0]
        self.assertEqual(result["level"], "error")
        self.assertEqual(
            result["properties"]["security-severity"],
            _LEAK_SECURITY_SEVERITY_HIGH,
        )

    def test_preserves_existing_level(self):
        path = self._write_sarif(
            {"runs": [{"results": [{"level": "warning"}]}]}
        )
        _enrich_sarif_with_security_severity(path)
        data = json.loads(path.read_text())
        self.assertEqual(data["runs"][0]["results"][0]["level"], "warning")

    def test_preserves_existing_security_severity(self):
        path = self._write_sarif(
            {"runs": [{"results": [{"properties": {"security-severity": "3.5"}}]}]}
        )
        _enrich_sarif_with_security_severity(path)
        data = json.loads(path.read_text())
        self.assertEqual(
            data["runs"][0]["results"][0]["properties"]["security-severity"],
            "3.5",
        )

    def test_empty_runs_is_a_noop(self):
        path = self._write_sarif({"runs": []})
        original = path.read_text()
        _enrich_sarif_with_security_severity(path)
        self.assertEqual(path.read_text(), original)

    def test_malformed_top_level_is_skipped(self):
        path = self._write_sarif(["not", "a", "dict"])
        original = path.read_text()
        _enrich_sarif_with_security_severity(path)
        # File should be left unchanged when payload is unexpectedly shaped.
        self.assertEqual(path.read_text(), original)

    def test_missing_file_is_skipped(self):
        path = Path(tempfile.gettempdir()) / "does-not-exist.sarif"
        if path.exists():
            path.unlink()
        # Should not raise; just warn-and-skip.
        _enrich_sarif_with_security_severity(path)


class ResolveConfigPathTest(unittest.TestCase):
    """Tests for `_resolve_config_path`."""

    def setUp(self):
        # `_resolve_config_path` resolves _CONFIG_PATH relative to cwd, so
        # each test runs in its own tempdir to isolate the lookup.
        self._original_cwd = Path.cwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._original_cwd)
        self._tmp.cleanup()

    def test_returns_config_path_when_present(self):
        Path(_CONFIG_PATH).write_text("# stub config", encoding="utf-8")
        self.assertEqual(_resolve_config_path(), _CONFIG_PATH)

    def test_returns_none_when_missing(self):
        self.assertIsNone(_resolve_config_path())


class Sha256OfTest(unittest.TestCase):
    """Tests for `_sha256_of` and the pinned `_GITLEAKS_TARBALL_SHA256`."""

    def test_matches_hashlib_for_known_content(self):
        payload = b"the rock"
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(payload)
            path = Path(f.name)
        self.addCleanup(path.unlink, missing_ok=True)
        self.assertEqual(_sha256_of(path), hashlib.sha256(payload).hexdigest())

    def test_pinned_constant_is_a_valid_sha256_hex_string(self):
        # Guards against typos / accidental truncation in the constant.
        self.assertEqual(len(_GITLEAKS_TARBALL_SHA256), 64)
        int(_GITLEAKS_TARBALL_SHA256, 16)  # raises ValueError on non-hex


if __name__ == "__main__":
    unittest.main()
