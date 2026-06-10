# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
from gitleaks import (
    _CONFIG_PATH,
    _LEAK_SECURITY_SEVERITY_HIGH,
    _determine_log_opts,
    _enrich_sarif_with_security_severity,
    _md_code_fence,
    _parse_report_formats,
    _resolve_config_path,
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
        with mock.patch("gitleaks.subprocess.run") as run:
            # `_determine_log_opts` does a best-effort fetch and then
            # verifies the base commit is reachable with `rev-parse`.
            run.side_effect = [
                mock.Mock(returncode=0, stderr=""),
                mock.Mock(returncode=0, stderr=""),
            ]
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
        path = self._write_sarif({"runs": [{"results": [{"level": "warning"}]}]})
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

    def test_empty_runs_raises(self):
        path = self._write_sarif({"runs": []})
        original = path.read_text()
        with self.assertRaises(ValueError) as ctx:
            _enrich_sarif_with_security_severity(path)
        self.assertIn("empty 'runs' array", str(ctx.exception))
        # File is left untouched when we bail out.
        self.assertEqual(path.read_text(), original)

    def test_clean_scan_with_empty_results_is_valid(self):
        # Gitleaks emits {"runs": [{"results": [], ...}]} on a clean scan;
        # that's a valid SARIF and must NOT raise (this is the normal,
        # no-leaks-found path).
        path = self._write_sarif({"runs": [{"results": []}]})
        _enrich_sarif_with_security_severity(path)
        data = json.loads(path.read_text())
        self.assertEqual(data["runs"][0]["results"], [])

    def test_malformed_top_level_raises(self):
        path = self._write_sarif(["not", "a", "dict"])
        original = path.read_text()
        with self.assertRaises(ValueError) as ctx:
            _enrich_sarif_with_security_severity(path)
        self.assertIn("top-level must be a JSON object", str(ctx.exception))
        # File should be left unchanged when payload is unexpectedly shaped.
        self.assertEqual(path.read_text(), original)

    def test_invalid_json_raises(self):
        fd, name = tempfile.mkstemp(suffix=".sarif")
        os.close(fd)
        path = Path(name)
        self.addCleanup(path.unlink, missing_ok=True)
        path.write_text("{not valid json", encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            _enrich_sarif_with_security_severity(path)
        self.assertIn("not valid JSON", str(ctx.exception))

    def test_runs_must_be_a_list(self):
        path = self._write_sarif({"runs": "oops"})
        with self.assertRaises(ValueError) as ctx:
            _enrich_sarif_with_security_severity(path)
        self.assertIn("'runs' must be a list", str(ctx.exception))

    def test_missing_file_raises(self):
        path = Path(tempfile.gettempdir()) / "does-not-exist.sarif"
        if path.exists():
            path.unlink()
        with self.assertRaises(FileNotFoundError):
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

    def test_raises_when_missing(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            _resolve_config_path()
        self.assertIn(_CONFIG_PATH, str(ctx.exception))


class MdCodeFenceTest(unittest.TestCase):
    """Tests for `_md_code_fence`."""

    def test_default_three_backticks_when_no_backticks(self):
        self.assertEqual(_md_code_fence("plain,csv,content"), "```")

    def test_three_backticks_when_content_has_short_runs(self):
        self.assertEqual(_md_code_fence("a `b` c"), "```")

    def test_grows_beyond_triple_backtick_run(self):
        self.assertEqual(_md_code_fence("before ``` after"), "````")

    def test_grows_to_longest_run(self):
        self.assertEqual(_md_code_fence("x ````` y"), "``````")

    def test_fence_actually_wraps_content(self):
        content = "with ``` inside"
        fence = _md_code_fence(content)
        block = f"{fence}\n{content}\n{fence}"
        # The closing fence must be on its own line and not appear within
        # the content, so the block is unambiguous.
        self.assertNotIn(fence, content)
        self.assertTrue(block.startswith(fence + "\n"))
        self.assertTrue(block.endswith("\n" + fence))


if __name__ == "__main__":
    unittest.main()
