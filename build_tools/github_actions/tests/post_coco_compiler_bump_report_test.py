# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for post_coco_compiler_bump_report.py."""

import argparse
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import post_coco_compiler_bump_report as report
from github_actions_api import GitHubAPIError

API = report.API
REPO = report.THEROCK_REPO


def make_components(llvm_tree="tree-llvm", llvm_commit="c-llvm"):
    return {
        "compiler/amd-llvm": {
            "repo": "ROCm/llvm-project",
            "commit": llvm_commit,
            "tree": llvm_tree,
        },
        "compiler/hipify": {"repo": "ROCm/HIPIFY", "commit": "c-hip", "tree": "t-hip"},
        "compiler/spirv-llvm-translator": {
            "repo": "ROCm/SPIRV-LLVM-Translator",
            "commit": "c-spirv",
            "tree": "t-spirv",
        },
    }


def make_manifest(**kwargs):
    return {
        "schema": report.MANIFEST_SCHEMA,
        "run_id": "123",
        "run_url": f"https://github.com/{REPO}/actions/runs/123",
        "branch": "amd-compiler-2026-11-17-x",
        "components": make_components(**kwargs),
        "rocm_systems": "rs-bump",
        "jobs": ["oci-rccl-compiler-bump"],
    }


class FakeAPI:
    """Routes GETs to canned responses and records every write."""

    def __init__(self, responses=None, comments=None):
        self.responses = responses or {}
        self.comments = comments or []
        self.writes = []

    def send_request(self, url, timeout_seconds=300, *, method="GET", body=None):
        if method != "GET":
            self.writes.append((method, url, body))
            return {}
        if "/issues/" in url and "/comments?" in url:
            return self.comments if "page=1" in url else []
        if url not in self.responses:
            raise GitHubAPIError(f"404 {url}")
        return self.responses[url]


def comment(comment_id, body, login=report.COMMENT_AUTHOR):
    return {"id": comment_id, "body": body, "user": {"login": login}}


class CollectReportsTest(unittest.TestCase):
    def test_keeps_latest_attempt_per_job_and_ignores_other_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, text in [
                ("coco-run-oci-rccl-compiler-bump-9-2-output", "attempt 2"),
                ("coco-run-oci-rccl-compiler-bump-9-10-output", "attempt 10"),
                ("coco-run-ruby64-rccl-compiler-bump-9-1-output", "ruby"),
                ("coco-run-oci-rccl-compiler-bump-9-1-console", "console"),
            ]:
                summary = root / name / "reports" / "compiler-bump"
                summary.mkdir(parents=True)
                (summary / "compiler-bump_summary.md").write_text(text)
            (root / "coco-run-empty-9-1-output").mkdir()

            self.assertEqual(
                report.collect_reports(root),
                {
                    "oci-rccl-compiler-bump": "attempt 10",
                    "ruby64-rccl-compiler-bump": "ruby",
                },
            )


class ReportDirTest(unittest.TestCase):
    def test_round_trip(self):
        manifest = make_manifest()
        with tempfile.TemporaryDirectory() as tmp:
            report.write_report_dir(Path(tmp), manifest, {"oci": "# oci\n"})
            read_manifest, reports = report.read_report_dir(Path(tmp))
        self.assertEqual(read_manifest["jobs"], ["oci"])
        self.assertEqual(reports, {"oci": "# oci\n"})

    def test_rejects_unknown_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            report.write_report_dir(Path(tmp), dict(make_manifest(), schema=99), {})
            with self.assertRaises(ValueError):
                report.read_report_dir(Path(tmp))


class MatchingTest(unittest.TestCase):
    def test_matches_on_trees_not_commits(self):
        self.assertTrue(
            report.components_match(
                make_components(llvm_commit="a"), make_components(llvm_commit="b")
            )
        )
        self.assertFalse(
            report.components_match(
                make_components(llvm_tree="x"), make_components(llvm_tree="y")
            )
        )

    def test_artifact_name_uses_llvm_tree(self):
        self.assertEqual(
            report.artifact_name(make_components(llvm_tree="abc")),
            "coco-compiler-bump-report-abc",
        )


class ProvenanceTest(unittest.TestCase):
    def test_notes_commit_and_rocm_systems_differences(self):
        text = report.build_provenance(
            make_manifest(llvm_commit="bump-commit"),
            make_components(llvm_commit="pr-commit"),
            rocm_systems="rs-pr",
        )
        self.assertIn("`compiler/amd-llvm` was built as `bump-commit`", text)
        self.assertIn("this PR's `pr-commit`", text)
        self.assertIn("`rocm-systems` differs", text)
        self.assertNotIn("`compiler/hipify` was built as", text)

    def test_quiet_when_everything_is_identical(self):
        text = report.build_provenance(make_manifest(), make_components(), "rs-bump")
        self.assertNotIn("was built as", text)
        self.assertNotIn("differs", text)

    def test_long_reports_are_truncated(self):
        body = report.build_report_body(
            "oci", "x" * 70000, make_manifest(), make_components(), "rs-bump"
        )
        self.assertLess(len(body), 65536)
        self.assertIn("Truncated", body)


class SyncPrCommentsTest(unittest.TestCase):
    def test_match_posts_each_job_and_deletes_status(self):
        api = FakeAPI(comments=[comment(7, f"{report.STATUS_MARKER}\nold")])
        found = (make_manifest(), {"oci": "# oci", "ruby64": "# ruby64"})

        report.sync_pr_comments(1, make_components(), "rs-bump", found, api)

        methods = [(m, url.rsplit("/", 1)[-1]) for m, url, _ in api.writes]
        self.assertEqual(
            methods, [("POST", "comments"), ("POST", "comments"), ("DELETE", "7")]
        )
        self.assertIn(report.report_marker("oci"), api.writes[0][2]["body"])

    def test_match_updates_existing_job_comment_in_place(self):
        api = FakeAPI(comments=[comment(5, f"{report.report_marker('oci')}\nold")])
        found = (make_manifest(), {"oci": "# oci"})

        report.sync_pr_comments(1, make_components(), "rs-bump", found, api)

        self.assertEqual(len(api.writes), 1)
        method, url, _ = api.writes[0]
        self.assertEqual(
            (method, url), ("PATCH", f"{API}/repos/{REPO}/issues/comments/5")
        )

    def test_unchanged_comment_is_not_rewritten(self):
        found = (make_manifest(), {"oci": "# oci"})
        body = report.build_report_body(
            "oci", "# oci", found[0], make_components(), "rs-bump"
        )
        api = FakeAPI(comments=[comment(5, body)])

        report.sync_pr_comments(1, make_components(), "rs-bump", found, api)

        self.assertEqual(api.writes, [])

    def test_no_match_posts_status_and_marks_old_reports_outdated(self):
        old = f"{report.report_marker('oci')}\n# old report"
        api = FakeAPI(comments=[comment(5, old)])

        report.sync_pr_comments(1, make_components(), "rs", None, api)

        self.assertEqual(api.writes[0][0], "POST")
        self.assertIn(report.STATUS_MARKER, api.writes[0][2]["body"])
        self.assertEqual(api.writes[1][0], "PATCH")
        patched = api.writes[1][2]["body"]
        self.assertTrue(patched.startswith(report.report_marker("oci")))
        self.assertIn(report.OUTDATED_NOTE, patched)
        self.assertIn("# old report", patched)

    def test_ignores_marked_comments_from_other_authors(self):
        quoted = comment(9, f"> {report.STATUS_MARKER}", login="someone")
        api = FakeAPI(comments=[quoted])

        report.sync_pr_comments(1, make_components(), "rs", None, api)

        self.assertEqual([m for m, _, _ in api.writes], ["POST"])


class FindReportTest(unittest.TestCase):
    def test_picks_newest_unexpired_matching_artifact(self):
        components = make_components()
        name = report.artifact_name(components)
        api = FakeAPI(
            responses={
                f"{API}/repos/{REPO}/actions/artifacts?name={name}&per_page=100": {
                    "artifacts": [
                        {"id": 1, "created_at": "2026-11-17T00:00:00Z"},
                        {
                            "id": 2,
                            "created_at": "2026-11-19T00:00:00Z",
                            "expired": True,
                        },
                        {"id": 3, "created_at": "2026-11-18T00:00:00Z"},
                    ]
                }
            }
        )

        def fake_download(artifact_id, dest):
            report.write_report_dir(
                dest, dict(make_manifest(), run_id=str(artifact_id)), {"oci": "x"}
            )

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            report, "download_artifact", side_effect=fake_download
        ) as download:
            manifest, _ = report.find_report(components, api, Path(tmp))

        self.assertEqual(manifest["run_id"], "3")
        download.assert_called_once()

    def test_none_when_no_artifact(self):
        components = make_components()
        name = report.artifact_name(components)
        api = FakeAPI(
            responses={
                f"{API}/repos/{REPO}/actions/artifacts?name={name}&per_page=100": {
                    "artifacts": []
                }
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(report.find_report(components, api, Path(tmp)))


class DownloadArtifactTest(unittest.TestCase):
    def test_token_is_not_forwarded_on_redirect(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("manifest.json", "{}")
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = buffer.getvalue()

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {"GITHUB_TOKEN": "secret"}
        ), mock.patch.object(report, "urlopen", return_value=response) as urlopen:
            report.download_artifact(42, Path(tmp))
            self.assertTrue((Path(tmp) / "manifest.json").exists())

        request = urlopen.call_args.args[0]
        self.assertTrue(request.full_url.endswith("/actions/artifacts/42/zip"))
        self.assertEqual(request.unredirected_hdrs["Authorization"], "Bearer secret")
        self.assertNotIn("Authorization", request.headers)


class PostOpenPrsTest(unittest.TestCase):
    def test_posts_only_on_matching_prs_and_reports_failures(self):
        manifest = make_manifest()
        components = make_components()

        def resolve_llvm(head, path, api):
            if head == "broken":
                raise GitHubAPIError("404")
            return (
                {"match": "c-llvm", "other": "c-other"}[head]
                if path.endswith("amd-llvm")
                else "rs"
            )

        def tree(repo, commit, api):
            return {"c-llvm": "tree-llvm", "c-other": "tree-other"}[commit]

        with tempfile.TemporaryDirectory() as tmp, mock.patch.multiple(
            report,
            list_open_compiler_prs=mock.Mock(
                return_value=[(1, "match"), (2, "other"), (3, "broken")]
            ),
            get_submodule_commit=mock.Mock(side_effect=resolve_llvm),
            get_tree=mock.Mock(side_effect=tree),
            resolve_components=mock.Mock(return_value=components),
            sync_pr_comments=mock.Mock(),
        ):
            report.write_report_dir(Path(tmp), manifest, {"oci": "x"})
            args = argparse.Namespace(report_dir=tmp, dry_run=False)
            with self.assertRaisesRegex(RuntimeError, r"\[3\]"):
                report.cmd_post_open_prs(args, FakeAPI())
            synced = [c.args[0] for c in report.sync_pr_comments.call_args_list]

        self.assertEqual(synced, [1])


if __name__ == "__main__":
    unittest.main()
