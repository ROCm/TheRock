# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
from zizmor import (
    _CONFIG_PATH,
    _SEVERITY_ORDER,
    _determine_changed_audited_files,
    _diff_range,
    _enrich_sarif_with_security_severity,
    _is_audited_path,
    _parse_report_formats,
    _resolve_config_path,
    _tally_findings_by_severity,
    _tally_findings_from_sarif,
)


class ParseReportFormatsTest(unittest.TestCase):
    """Tests for `_parse_report_formats`."""

    def test_default_sarif(self):
        targets = _parse_report_formats("sarif")
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].fmt, "sarif")
        self.assertEqual(targets[0].path, Path("zizmor-report.sarif"))

    def test_multiple_formats_with_whitespace_and_dedup(self):
        targets = _parse_report_formats(" sarif , json , sarif ,plain,github ")
        self.assertEqual([t.fmt for t in targets], ["sarif", "json", "plain", "github"])
        self.assertEqual(
            [t.path for t in targets],
            [
                Path("zizmor-report.sarif"),
                Path("zizmor-report.json"),
                Path("zizmor-report.txt"),
                Path("zizmor-report.txt"),
            ],
        )

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parse_report_formats("")
        self.assertIn("report_formats is empty", str(ctx.exception))

    def test_unknown_format_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parse_report_formats("sarif,xml")
        self.assertIn("'xml'", str(ctx.exception))


class DiffRangeTest(unittest.TestCase):
    """Tests for `_diff_range`."""

    def test_pull_request_returns_base_head(self):
        event = {"pull_request": {"base": {"sha": "aaa"}, "head": {"sha": "bbb"}}}
        self.assertEqual(_diff_range("pull_request", event), ("aaa", "bbb"))

    def test_pull_request_missing_shas_returns_none(self):
        event = {"pull_request": {"base": {}, "head": {}}}
        self.assertIsNone(_diff_range("pull_request", event))

    def test_push_returns_before_after(self):
        self.assertEqual(
            _diff_range("push", {"before": "abc", "after": "def"}),
            ("abc", "def"),
        )

    def test_push_new_ref_returns_none(self):
        self.assertIsNone(_diff_range("push", {"before": "0" * 40, "after": "abc"}))

    def test_unknown_event_returns_none(self):
        self.assertIsNone(_diff_range("workflow_dispatch", {}))


class IsAuditedPathTest(unittest.TestCase):
    """Tests for `_is_audited_path`."""

    def test_matches_workflow_paths(self):
        self.assertTrue(_is_audited_path(".github/workflows/ci.yml"))
        self.assertTrue(_is_audited_path(".github/workflows/release.yaml"))

    def test_matches_action_files_anywhere(self):
        self.assertTrue(_is_audited_path("action.yml"))
        self.assertTrue(_is_audited_path("tools/my-action/action.yaml"))

    def test_matches_dependabot(self):
        self.assertTrue(_is_audited_path(".github/dependabot.yml"))
        self.assertTrue(_is_audited_path(".github/dependabot.yaml"))

    def test_rejects_non_audited_files(self):
        self.assertFalse(_is_audited_path("README.md"))
        self.assertFalse(_is_audited_path("build_tools/script.py"))


class DetermineChangedAuditedFilesTest(unittest.TestCase):
    """Tests for `_determine_changed_audited_files`."""

    def setUp(self):
        self._original_cwd = Path.cwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)
        Path(".github/workflows").mkdir(parents=True)
        Path("docs").mkdir(parents=True)
        Path(".github/workflows/ci.yml").write_text("name: ci\n", encoding="utf-8")
        Path("docs/readme.md").write_text("# docs\n", encoding="utf-8")

    def tearDown(self):
        os.chdir(self._original_cwd)
        self._tmp.cleanup()

    def test_returns_only_changed_audited_files_under_scan_path(self):
        with mock.patch("zizmor.subprocess.run") as run:
            run.side_effect = [
                mock.Mock(returncode=0, stdout="", stderr=""),
                mock.Mock(
                    returncode=0,
                    stdout=".github/workflows/ci.yml\ndocs/readme.md\n",
                    stderr="",
                ),
            ]
            files = _determine_changed_audited_files(
                event_name="push",
                event={"before": "abc", "after": "def"},
                scan_path=Path("."),
            )
        self.assertEqual(files, [Path(".github/workflows/ci.yml")])

    def test_returns_empty_list_when_diff_has_no_audited_files(self):
        with mock.patch("zizmor.subprocess.run") as run:
            run.side_effect = [
                mock.Mock(returncode=0, stdout="", stderr=""),
                mock.Mock(returncode=0, stdout="docs/readme.md\n", stderr=""),
            ]
            files = _determine_changed_audited_files(
                event_name="push",
                event={"before": "abc", "after": "def"},
                scan_path=Path("."),
            )
        self.assertEqual(files, [])

    def test_returns_none_when_diff_command_fails(self):
        with mock.patch("zizmor.subprocess.run") as run:
            run.side_effect = [
                mock.Mock(returncode=0, stdout="", stderr=""),
                subprocess.CalledProcessError(
                    returncode=1, cmd=["git", "diff"], stderr="bad range"
                ),
            ]
            files = _determine_changed_audited_files(
                event_name="push",
                event={"before": "abc", "after": "def"},
                scan_path=Path("."),
            )
        self.assertIsNone(files)


class EnrichSarifSeverityTest(unittest.TestCase):
    """Tests for `_enrich_sarif_with_security_severity`."""

    def _write_sarif(self, payload: object) -> Path:
        fd, name = tempfile.mkstemp(suffix=".sarif")
        os.close(fd)
        path = Path(name)
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.addCleanup(path.unlink, missing_ok=True)
        return path

    def test_injects_security_severity_from_zizmor_severity(self):
        path = self._write_sarif(
            {
                "runs": [
                    {
                        "results": [
                            {"properties": {"zizmor/severity": "High"}},
                            {"properties": {"zizmor/severity": "informational"}},
                        ]
                    }
                ]
            }
        )
        _enrich_sarif_with_security_severity(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        results = data["runs"][0]["results"]
        self.assertEqual(results[0]["properties"]["security-severity"], "8.5")
        self.assertEqual(results[1]["properties"]["security-severity"], "0.3")

    def test_unknown_severity_is_left_unmapped(self):
        path = self._write_sarif(
            {"runs": [{"results": [{"properties": {"zizmor/severity": "unknown"}}]}]}
        )
        _enrich_sarif_with_security_severity(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        props = data["runs"][0]["results"][0]["properties"]
        self.assertNotIn("security-severity", props)


class TallyFindingsBySeverityTest(unittest.TestCase):
    """Tests for `_tally_findings_by_severity`."""

    def test_counts_known_and_unknown_severities(self):
        fd, name = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        path = Path(name)
        self.addCleanup(path.unlink, missing_ok=True)
        path.write_text(
            json.dumps(
                [
                    {"determinations": {"severity": "High"}},
                    {"determinations": {"severity": "Medium"}},
                    {"determinations": {"severity": "high"}},
                    {"determinations": {"severity": "Unknown"}},
                    {},
                ]
            ),
            encoding="utf-8",
        )
        counts = _tally_findings_by_severity(path)
        self.assertEqual(counts["HIGH"], 2)
        self.assertEqual(counts["MEDIUM"], 1)
        self.assertEqual(counts["UNKNOWN"], 2)
        for sev in _SEVERITY_ORDER:
            self.assertIn(sev, counts)


class TallyFindingsFromSarifTest(unittest.TestCase):
    """Tests for `_tally_findings_from_sarif`."""

    def _write_sarif(self, payload: object) -> Path:
        fd, name = tempfile.mkstemp(suffix=".sarif")
        os.close(fd)
        path = Path(name)
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.addCleanup(path.unlink, missing_ok=True)
        return path

    def test_counts_known_and_unknown_severities_across_runs(self):
        path = self._write_sarif(
            {
                "runs": [
                    {
                        "results": [
                            {"properties": {"zizmor/severity": "High"}},
                            {"properties": {"zizmor/severity": "medium"}},
                        ]
                    },
                    {"results": [{"properties": {"zizmor/severity": "High"}}, {}]},
                ]
            }
        )
        counts = _tally_findings_from_sarif(path)
        self.assertEqual(counts["HIGH"], 2)
        self.assertEqual(counts["MEDIUM"], 1)
        self.assertEqual(counts["UNKNOWN"], 1)
        for sev in _SEVERITY_ORDER:
            self.assertIn(sev, counts)

    def test_no_results_returns_zero_counts(self):
        path = self._write_sarif({"runs": [{"results": []}]})
        counts = _tally_findings_from_sarif(path)
        self.assertEqual(sum(counts.values()), 0)

    def test_invalid_json_raises(self):
        fd, name = tempfile.mkstemp(suffix=".sarif")
        os.close(fd)
        path = Path(name)
        path.write_text("not json", encoding="utf-8")
        self.addCleanup(path.unlink, missing_ok=True)
        with self.assertRaises(RuntimeError):
            _tally_findings_from_sarif(path)


class ResolveConfigPathTest(unittest.TestCase):
    """Tests for `_resolve_config_path`."""

    def setUp(self):
        self._original_cwd = Path.cwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._original_cwd)
        self._tmp.cleanup()

    def test_returns_config_path_when_present(self):
        Path(_CONFIG_PATH).write_text("rules: {}\n", encoding="utf-8")
        self.assertEqual(_resolve_config_path(), _CONFIG_PATH)

    def test_raises_when_missing(self):
        with self.assertRaises(FileNotFoundError):
            _resolve_config_path()


if __name__ == "__main__":
    unittest.main()
