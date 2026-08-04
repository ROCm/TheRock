#!/usr/bin/env python
"""Unit tests for s3_buckets.py."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.s3_buckets import (
    BUCKET_REGISTRY_ENV_VAR,
    BucketRegistryError,
    CdnRule,
    S3BucketConfig,
    all_bucket_configs,
    get_artifacts_bucket_config,
    get_artifacts_bucket_config_for_workflow_run,
    get_release_bucket_config,
    lookup_bucket_config,
    require_bucket_config,
    reset_bucket_registry,
    resolve_public_url,
    set_bucket_config_file,
)


# ---------------------------------------------------------------------------
# get_artifacts_bucket_config
# ---------------------------------------------------------------------------


class TestGetArtifactsBucketConfig(unittest.TestCase):
    def test_ci_rocm_therock(self):
        config = get_artifacts_bucket_config(
            release_type="ci", repository="ROCm/TheRock", is_pr_from_fork=False
        )
        self.assertEqual(config.name, "therock-ci-artifacts")
        self.assertEqual(
            config.write_access_iam_role,
            "arn:aws:iam::692859939525:role/therock-ci",
        )

    def test_ci_fork_pr(self):
        config = get_artifacts_bucket_config(
            release_type="ci", repository="ROCm/TheRock", is_pr_from_fork=True
        )
        self.assertEqual(config.name, "therock-ci-artifacts-external")
        # The raw lookup returns the external role for forks; the OIDC skip for
        # forks happens in get_artifacts_bucket_config_for_workflow_run, not here.
        self.assertEqual(config.iam_role, "therock-ci-external")

    def test_ci_external_repo(self):
        config = get_artifacts_bucket_config(
            release_type="ci", repository="ROCm/rocm-libraries", is_pr_from_fork=False
        )
        self.assertEqual(config.name, "therock-ci-artifacts-external")
        self.assertEqual(config.iam_role, "therock-ci-external")

    def test_release_type_dev(self):
        config = get_artifacts_bucket_config(
            release_type="dev", repository="ROCm/TheRock", is_pr_from_fork=False
        )
        self.assertEqual(config.name, "therock-dev-artifacts")
        self.assertEqual(config.iam_role, "therock-dev")

    def test_release_type_from_rockrel(self):
        config = get_artifacts_bucket_config(
            release_type="nightly", repository="ROCm/rockrel", is_pr_from_fork=False
        )
        self.assertEqual(config.name, "therock-nightly-artifacts")

    def test_release_type_invalid_raises(self):
        with self.assertRaises(ValueError) as cm:
            get_artifacts_bucket_config(
                release_type="bogus",
                repository="ROCm/TheRock",
                is_pr_from_fork=False,
            )
        self.assertIn("bogus", str(cm.exception))

    def test_empty_release_type_raises(self):
        with self.assertRaises(ValueError):
            get_artifacts_bucket_config(
                release_type="",
                repository="ROCm/TheRock",
                is_pr_from_fork=False,
            )


# ---------------------------------------------------------------------------
# get_release_bucket_config
# ---------------------------------------------------------------------------


class TestGetReleaseBucketConfig(unittest.TestCase):
    def test_dev_tarball(self):
        config = get_release_bucket_config(release_type="dev", bucket_type="tarball")
        self.assertEqual(config.name, "therock-dev-tarball")
        self.assertEqual(config.iam_role, "therock-dev")
        self.assertEqual(
            config.write_access_iam_role,
            "arn:aws:iam::692859939525:role/therock-dev",
        )

    def test_nightly_python(self):
        config = get_release_bucket_config(release_type="nightly", bucket_type="python")
        self.assertEqual(config.name, "therock-nightly-python")
        self.assertEqual(config.iam_role, "therock-nightly")

    def test_prerelease_packages(self):
        config = get_release_bucket_config(
            release_type="prerelease", bucket_type="packages"
        )
        self.assertEqual(config.name, "therock-prerelease-packages")
        self.assertEqual(config.iam_role, "therock-prerelease")

    def test_all_combinations_exist(self):
        for release_type in ("dev", "nightly", "prerelease"):
            for bucket_type in ("tarball", "python", "packages"):
                config = get_release_bucket_config(release_type, bucket_type)
                self.assertEqual(config.name, f"therock-{release_type}-{bucket_type}")

    def test_invalid_release_type_raises(self):
        with self.assertRaises(ValueError) as cm:
            get_release_bucket_config(release_type="bogus", bucket_type="tarball")
        self.assertIn("bogus", str(cm.exception))

    def test_empty_release_type_raises(self):
        with self.assertRaises(ValueError):
            get_release_bucket_config(release_type="", bucket_type="tarball")

    def test_ci_release_type_raises(self):
        with self.assertRaises(ValueError):
            get_release_bucket_config(release_type="ci", bucket_type="tarball")

    def test_invalid_bucket_type_raises(self):
        with self.assertRaises(ValueError) as cm:
            get_release_bucket_config(release_type="dev", bucket_type="wheels")
        self.assertIn("wheels", str(cm.exception))


# ---------------------------------------------------------------------------
# get_artifacts_bucket_config_for_workflow_run
# ---------------------------------------------------------------------------


class TestGetArtifactsBucketConfigForWorkflowRun(unittest.TestCase):
    """Test the workflow-run-aware wrapper."""

    def setUp(self):
        self.api_patcher = mock.patch(
            "github_actions.github_actions_api.gha_query_workflow_run_by_id"
        )
        self.mock_api = self.api_patcher.start()

        self.env_patcher = mock.patch.dict(os.environ)
        self.env_patcher.start()
        os.environ.pop("GITHUB_REPOSITORY", None)
        os.environ.pop("GITHUB_EVENT_NAME", None)
        os.environ.pop("GITHUB_EVENT_PATH", None)
        os.environ.pop("RELEASE_TYPE", None)

    def tearDown(self):
        self.env_patcher.stop()
        self.api_patcher.stop()

    def test_default_ci(self):
        config = get_artifacts_bucket_config_for_workflow_run(
            github_repository="ROCm/TheRock"
        )
        self.assertEqual(config.name, "therock-ci-artifacts")

    def test_explicit_release_type(self):
        config = get_artifacts_bucket_config_for_workflow_run(
            github_repository="ROCm/TheRock", release_type="nightly"
        )
        self.assertEqual(config.name, "therock-nightly-artifacts")

    def test_release_type_from_env(self):
        os.environ["RELEASE_TYPE"] = "dev"
        config = get_artifacts_bucket_config_for_workflow_run(
            github_repository="ROCm/TheRock"
        )
        self.assertEqual(config.name, "therock-dev-artifacts")

    def test_explicit_release_type_overrides_env(self):
        os.environ["RELEASE_TYPE"] = "dev"
        config = get_artifacts_bucket_config_for_workflow_run(
            github_repository="ROCm/TheRock", release_type="nightly"
        )
        self.assertEqual(config.name, "therock-nightly-artifacts")

    def test_workflow_run_same_repo(self):
        fake_run = {
            "id": 12345,
            "head_repository": {"full_name": "ROCm/TheRock"},
        }
        config = get_artifacts_bucket_config_for_workflow_run(
            github_repository="ROCm/TheRock", workflow_run=fake_run
        )
        self.assertEqual(config.name, "therock-ci-artifacts")

    def test_workflow_run_from_fork(self):
        fake_run = {
            "id": 12345,
            "head_repository": {"full_name": "SomeUser/TheRock"},
        }
        config = get_artifacts_bucket_config_for_workflow_run(
            github_repository="ROCm/TheRock", workflow_run=fake_run
        )
        self.assertEqual(config.name, "therock-ci-artifacts-external")
        # Fork PRs cannot assume an IAM role via OIDC (no trust relationship),
        # so the wrapper must strip the role and fall back to runner base
        # credentials. Regression coverage for #5654.
        self.assertIsNone(config.iam_role)
        self.assertIsNone(config.write_access_iam_role)

    def test_workflow_run_external_repo_uses_oidc(self):
        # An external (non-fork) repo such as rocm-libraries keeps the
        # therock-ci-external role so it can authenticate via OIDC.
        fake_run = {
            "id": 12345,
            "head_repository": {"full_name": "ROCm/rocm-libraries"},
        }
        config = get_artifacts_bucket_config_for_workflow_run(
            github_repository="ROCm/rocm-libraries", workflow_run=fake_run
        )
        self.assertEqual(config.name, "therock-ci-artifacts-external")
        self.assertEqual(config.iam_role, "therock-ci-external")

    def test_workflow_run_same_repo_keeps_internal_role(self):
        # A same-repo (non-fork) ROCm/TheRock PR is unaffected by the fork skip
        # and keeps the internal therock-ci role.
        fake_run = {
            "id": 12345,
            "head_repository": {"full_name": "ROCm/TheRock"},
        }
        config = get_artifacts_bucket_config_for_workflow_run(
            github_repository="ROCm/TheRock", workflow_run=fake_run
        )
        self.assertEqual(config.name, "therock-ci-artifacts")
        self.assertEqual(config.iam_role, "therock-ci")

    def test_workflow_run_id_triggers_api_call(self):
        self.mock_api.return_value = {
            "id": 12345,
            "head_repository": {"full_name": "ROCm/TheRock"},
        }
        config = get_artifacts_bucket_config_for_workflow_run(
            github_repository="ROCm/TheRock", workflow_run_id="12345"
        )
        self.mock_api.assert_called_once_with("ROCm/TheRock", "12345")
        self.assertEqual(config.name, "therock-ci-artifacts")

    def _write_event(self, event: dict) -> str:
        """Write a synthetic GitHub event payload to a temp file.

        Returns the path. Caller must os.unlink() after use.
        Uses delete=False because NamedTemporaryFile(delete=True) holds an
        exclusive lock on Windows, preventing the code under test from reading.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(event, f)
            return f.name

    def test_fork_pr_from_event_payload(self):
        """Fork PR detected via event payload (no workflow_run dict)."""
        event_path = self._write_event(
            {"pull_request": {"head": {"repo": {"fork": True}}}}
        )
        try:
            os.environ["GITHUB_EVENT_NAME"] = "pull_request"
            os.environ["GITHUB_EVENT_PATH"] = event_path
            config = get_artifacts_bucket_config_for_workflow_run(
                github_repository="ROCm/TheRock"
            )
            self.assertEqual(config.name, "therock-ci-artifacts-external")
        finally:
            os.unlink(event_path)

    def test_same_repo_pr_from_event_payload(self):
        """Same-repo PR detected via event payload (no workflow_run dict)."""
        event_path = self._write_event(
            {"pull_request": {"head": {"repo": {"fork": False}}}}
        )
        try:
            os.environ["GITHUB_EVENT_NAME"] = "pull_request"
            os.environ["GITHUB_EVENT_PATH"] = event_path
            config = get_artifacts_bucket_config_for_workflow_run(
                github_repository="ROCm/TheRock"
            )
            self.assertEqual(config.name, "therock-ci-artifacts")
        finally:
            os.unlink(event_path)

    def test_workflow_run_id_ignored_when_workflow_run_provided(self):
        """workflow_run takes priority over workflow_run_id."""
        fake_run = {
            "id": 12345,
            "head_repository": {"full_name": "ROCm/TheRock"},
        }
        config = get_artifacts_bucket_config_for_workflow_run(
            github_repository="ROCm/TheRock",
            workflow_run=fake_run,
            workflow_run_id="99999",
        )
        self.mock_api.assert_not_called()
        self.assertEqual(config.name, "therock-ci-artifacts")


# ---------------------------------------------------------------------------
# CdnRule / resolve_public_url
# ---------------------------------------------------------------------------


class TestCdnRule(unittest.TestCase):
    def test_appends_trailing_slashes(self):
        rule = CdnRule("v4/whl", "https://cdn.example.com/whl")
        self.assertEqual(rule.key_prefix, "v4/whl/")
        self.assertEqual(rule.url_prefix, "https://cdn.example.com/whl/")

    def test_keeps_existing_trailing_slashes(self):
        rule = CdnRule("v4/whl/", "https://cdn.example.com/whl/")
        self.assertEqual(rule.key_prefix, "v4/whl/")
        self.assertEqual(rule.url_prefix, "https://cdn.example.com/whl/")

    def test_empty_key_prefix_matches_whole_bucket(self):
        rule = CdnRule("", "https://cdn.example.com/")
        self.assertEqual(rule.key_prefix, "")

    def test_leading_slash_key_prefix_raises(self):
        with self.assertRaises(ValueError):
            CdnRule("/v4/whl/", "https://cdn.example.com/")

    def test_non_https_url_prefix_raises(self):
        with self.assertRaises(ValueError):
            CdnRule("v4/whl/", "http://cdn.example.com/")

    def test_frozen(self):
        rule = CdnRule("v4/whl/", "https://cdn.example.com/")
        with self.assertRaises(Exception):
            rule.key_prefix = "other/"


class TestResolvePublicUrl(unittest.TestCase):
    DEFAULT = "https://bucket.s3.amazonaws.com/some/key"

    def test_unknown_bucket_falls_back(self):
        self.assertEqual(
            resolve_public_url("not-a-real-bucket", "some/key", default=self.DEFAULT),
            self.DEFAULT,
        )

    def test_known_bucket_without_matching_rule_falls_back(self):
        # therock-dev-python only maps v4/whl/.
        self.assertEqual(
            resolve_public_url(
                "therock-dev-python", "v4/other/thing.whl", default=self.DEFAULT
            ),
            self.DEFAULT,
        )

    def test_bucket_with_no_rules_falls_back(self):
        self.assertEqual(
            resolve_public_url(
                "therock-ci-artifacts", "12345-linux/x.tar.xz", default=self.DEFAULT
            ),
            self.DEFAULT,
        )

    def test_whl_round_trips(self):
        self.assertEqual(
            resolve_public_url(
                "therock-nightly-python", "v4/whl/gfx110X/x.whl", default=self.DEFAULT
            ),
            "https://rocm.nightlies.amd.com/whl-multi-arch/gfx110X/x.whl",
        )

    def test_tarball_round_trips(self):
        self.assertEqual(
            resolve_public_url(
                "therock-dev-tarball", "v4/tarball/x.tar.gz", default=self.DEFAULT
            ),
            "https://rocm.devreleases.amd.com/tarball-multi-arch/x.tar.gz",
        )

    def test_deb_and_rpm_round_trip(self):
        self.assertEqual(
            resolve_public_url(
                "therock-dev-packages", "v4/deb/20260101-1/x.deb", default=self.DEFAULT
            ),
            "https://rocm.devreleases.amd.com/packages-multi-arch/deb/20260101-1/x.deb",
        )
        self.assertEqual(
            resolve_public_url(
                "therock-dev-packages", "v4/rpm/20260101-1/x.rpm", default=self.DEFAULT
            ),
            "https://rocm.devreleases.amd.com/packages-multi-arch/rpm/20260101-1/x.rpm",
        )

    def test_every_seeded_rule_round_trips(self):
        for config in all_bucket_configs():
            for rule in config.cdn_rules:
                with self.subTest(bucket=config.name, prefix=rule.key_prefix):
                    self.assertEqual(
                        resolve_public_url(
                            config.name, rule.key_prefix + "a/b.txt", default="fallback"
                        ),
                        rule.url_prefix + "a/b.txt",
                    )

    def test_longest_prefix_wins(self):
        config = S3BucketConfig(
            "prefix-test",
            cdn_rules=(
                CdnRule("", "https://cdn.example.com/root/"),
                CdnRule("v4/whl/", "https://cdn.example.com/whl/"),
            ),
        )
        with mock.patch("_therock_utils.s3_buckets.s3_bucket_configs", [config]):
            reset_bucket_registry()
            self.assertEqual(
                resolve_public_url("prefix-test", "v4/whl/x.whl", default="fallback"),
                "https://cdn.example.com/whl/x.whl",
            )
            self.assertEqual(
                resolve_public_url("prefix-test", "other/x.txt", default="fallback"),
                "https://cdn.example.com/root/other/x.txt",
            )


class TestArtifactsBucketsHaveNoCdn(unittest.TestCase):
    """Artifacts bucket URLs are machine-consumed and read directly from S3.

    Adding a CDN rule to one of these buckets would silently retarget every CI
    job-summary link and every artifact fetch. If that is ever intended, delete
    this test deliberately rather than discovering the change in production.
    """

    def test_no_artifacts_bucket_has_cdn_rules(self):
        for config in all_bucket_configs():
            if config.name.endswith("-artifacts") or config.name.endswith(
                "-artifacts-external"
            ):
                with self.subTest(bucket=config.name):
                    self.assertEqual(config.cdn_rules, ())


class TestForkPrConfigPreservesNewFields(unittest.TestCase):
    """Fork PRs drop the IAM role, and must drop nothing else.

    get_artifacts_bucket_config_for_workflow_run used to rebuild the config
    field by field, which reset every field it did not name.
    """

    def test_fork_pr_keeps_all_other_fields(self):
        original = require_bucket_config("therock-ci-artifacts-external")
        config = get_artifacts_bucket_config_for_workflow_run(
            "ROCm/TheRock",
            workflow_run={
                "id": 123,
                "head_repository": {"full_name": "someuser/TheRock"},
            },
        )
        self.assertIsNone(config.iam_role)
        self.assertEqual(config.name, original.name)
        self.assertEqual(config.region, original.region)
        self.assertEqual(config.iam_account, original.iam_account)
        self.assertEqual(config.key_prefix, original.key_prefix)
        self.assertEqual(config.cdn_rules, original.cdn_rules)
        self.assertEqual(
            config.namespace_external_repos, original.namespace_external_repos
        )

    def test_covers_every_field(self):
        """Guard against a new field being added without extending the test above."""
        import dataclasses

        self.assertEqual(
            {f.name for f in dataclasses.fields(S3BucketConfig)},
            {
                "name",
                "region",
                "iam_account",
                "iam_role",
                "key_prefix",
                "cdn_rules",
                "namespace_external_repos",
            },
        )


# ---------------------------------------------------------------------------
# Registry file loading
# ---------------------------------------------------------------------------


class RegistryFileTestCase(unittest.TestCase):
    """Base class that writes registry files and always restores global state."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(set_bucket_config_file, None)

    def write_registry(self, data, name="buckets.json") -> Path:
        path = Path(self._tmp.name) / name
        path.write_text(
            data if isinstance(data, str) else json.dumps(data), encoding="utf-8"
        )
        return path

    def use(self, data, *, via_env=False, name="buckets.json") -> Path:
        path = self.write_registry(data, name=name)
        if via_env:
            patcher = mock.patch.dict(os.environ, {BUCKET_REGISTRY_ENV_VAR: str(path)})
            patcher.start()
            self.addCleanup(patcher.stop)
            reset_bucket_registry()
        else:
            set_bucket_config_file(path)
        return path


_MINIMAL_DOWNSTREAM = {
    "version": 1,
    "buckets": [
        {
            "name": "downstream-artifacts",
            "iam_role": "downstream",
            "key_prefix": "v3/artifacts",
            "cdn_rules": [
                {
                    "key_prefix": "v3/artifacts/",
                    "url_prefix": "https://cdn.example.com/artifacts/",
                }
            ],
        }
    ],
}


class TestRegistryFileLoading(RegistryFileTestCase):
    def test_registers_new_bucket(self):
        self.use(_MINIMAL_DOWNSTREAM)
        config = require_bucket_config("downstream-artifacts")
        self.assertEqual(config.iam_role, "downstream")
        # key_prefix is normalized with a trailing slash on load.
        self.assertEqual(config.key_prefix, "v3/artifacts/")
        self.assertEqual(
            resolve_public_url(
                "downstream-artifacts", "v3/artifacts/123-linux/x.tar.xz", default="raw"
            ),
            "https://cdn.example.com/artifacts/123-linux/x.tar.xz",
        )

    def test_in_tree_buckets_still_present(self):
        self.use(_MINIMAL_DOWNSTREAM)
        self.assertIsNotNone(lookup_bucket_config("therock-ci-artifacts"))

    def test_env_var_path_works(self):
        self.use(_MINIMAL_DOWNSTREAM, via_env=True)
        self.assertIsNotNone(lookup_bucket_config("downstream-artifacts"))

    def test_explicit_file_beats_env_var(self):
        env_path = self.write_registry(
            {"version": 1, "buckets": [{"name": "from-env"}]}, name="env.json"
        )
        flag_path = self.write_registry(
            {"version": 1, "buckets": [{"name": "from-flag"}]}, name="flag.json"
        )
        with mock.patch.dict(os.environ, {BUCKET_REGISTRY_ENV_VAR: str(env_path)}):
            set_bucket_config_file(flag_path)
            self.assertIsNotNone(lookup_bucket_config("from-flag"))
            self.assertIsNone(lookup_bucket_config("from-env"))

    def test_clearing_explicit_file_restores_env_var(self):
        env_path = self.write_registry(
            {"version": 1, "buckets": [{"name": "from-env"}]}, name="env.json"
        )
        flag_path = self.write_registry(
            {"version": 1, "buckets": [{"name": "from-flag"}]}, name="flag.json"
        )
        with mock.patch.dict(os.environ, {BUCKET_REGISTRY_ENV_VAR: str(env_path)}):
            set_bucket_config_file(flag_path)
            set_bucket_config_file(None)
            self.assertIsNotNone(lookup_bucket_config("from-env"))

    def test_defaults_fill_in(self):
        self.use({"version": 1, "buckets": [{"name": "bare"}]})
        config = require_bucket_config("bare")
        self.assertEqual(config.region, "us-east-2")
        self.assertIsNone(config.iam_role)
        self.assertEqual(config.key_prefix, "")
        self.assertEqual(config.cdn_rules, ())
        self.assertFalse(config.namespace_external_repos)


class TestRegistryFileErrors(RegistryFileTestCase):
    def assert_load_error(self, data, *needles):
        path = self.use(data)
        with self.assertRaises(BucketRegistryError) as cm:
            all_bucket_configs()
        message = str(cm.exception)
        # Every error names the offending file, so the user knows what to edit.
        self.assertIn(str(path), message)
        for needle in needles:
            self.assertIn(needle, message)

    def test_missing_file(self):
        set_bucket_config_file(Path(self._tmp.name) / "nope.json")
        with self.assertRaises(BucketRegistryError) as cm:
            all_bucket_configs()
        self.assertIn("nope.json", str(cm.exception))

    def test_invalid_json(self):
        self.assert_load_error("{not json", "invalid JSON")

    def test_top_level_not_an_object(self):
        self.assert_load_error([], "top level")

    def test_missing_version(self):
        self.assert_load_error({"buckets": []}, "version")

    def test_unknown_version(self):
        self.assert_load_error({"version": 99, "buckets": []}, "version")

    def test_unknown_top_level_key(self):
        self.assert_load_error(
            {"version": 1, "bucket": []}, "unknown top-level key", "bucket"
        )

    def test_unknown_bucket_key(self):
        self.assert_load_error(
            {"version": 1, "buckets": [{"name": "x", "cdn_rule": []}]},
            "cdn_rule",
        )

    def test_unknown_cdn_rule_key(self):
        self.assert_load_error(
            {
                "version": 1,
                "buckets": [
                    {
                        "name": "x",
                        "cdn_rules": [
                            {
                                "key_prefix": "a/",
                                "url_prefix": "https://x/",
                                "urlprefix": "https://y/",
                            }
                        ],
                    }
                ],
            },
            "urlprefix",
        )

    def test_cdn_rule_missing_field(self):
        self.assert_load_error(
            {
                "version": 1,
                "buckets": [{"name": "x", "cdn_rules": [{"key_prefix": "a/"}]}],
            },
            "url_prefix",
        )

    def test_cdn_rule_non_https(self):
        self.assert_load_error(
            {
                "version": 1,
                "buckets": [
                    {
                        "name": "x",
                        "cdn_rules": [
                            {"key_prefix": "a/", "url_prefix": "http://insecure/"}
                        ],
                    }
                ],
            },
            "https://",
        )

    def test_bucket_missing_name(self):
        self.assert_load_error({"version": 1, "buckets": [{}]}, "name")

    def test_duplicate_bucket_in_same_file(self):
        self.assert_load_error(
            {"version": 1, "buckets": [{"name": "dup"}, {"name": "dup"}]},
            "more than once",
        )

    def test_key_prefix_with_leading_slash(self):
        self.assert_load_error(
            {"version": 1, "buckets": [{"name": "x", "key_prefix": "/v3/"}]},
            "must not start with '/'",
        )

    def test_shadowing_in_tree_bucket_without_override(self):
        self.assert_load_error(
            {"version": 1, "buckets": [{"name": "therock-ci-artifacts"}]},
            "override",
        )

    def test_selection_override_naming_unregistered_bucket(self):
        self.assert_load_error(
            {"version": 1, "artifacts_buckets": {"ci": "nowhere"}},
            "nowhere",
        )

    def test_unknown_artifacts_selection_slot(self):
        self.assert_load_error(
            {"version": 1, "artifacts_buckets": {"staging": "therock-ci-artifacts"}},
            "staging",
        )

    def test_unknown_release_selection_slot(self):
        self.assert_load_error(
            {"version": 1, "release_buckets": {"ci": {}}},
            "ci",
        )

    def test_unknown_release_bucket_type(self):
        self.assert_load_error(
            {
                "version": 1,
                "release_buckets": {"dev": {"wheels": "therock-dev-python"}},
            },
            "wheels",
        )


class TestRegistryFileOverride(RegistryFileTestCase):
    def test_override_replaces_in_tree_bucket_wholesale(self):
        self.use(
            {
                "version": 1,
                "buckets": [
                    {
                        "name": "therock-dev-python",
                        "override": True,
                        "key_prefix": "v9/",
                    }
                ],
            }
        )
        config = require_bucket_config("therock-dev-python")
        self.assertEqual(config.key_prefix, "v9/")
        # Full replacement, not a field-wise merge: the in-tree iam_role and
        # cdn_rules are gone rather than inherited.
        self.assertIsNone(config.iam_role)
        self.assertEqual(config.cdn_rules, ())


class TestSelectionOverrides(RegistryFileTestCase):
    REGISTRY = {
        "version": 1,
        "buckets": [
            {"name": "downstream-artifacts", "key_prefix": "v3/artifacts/"},
            {"name": "downstream-external", "namespace_external_repos": True},
            {"name": "downstream-python"},
        ],
        "artifacts_buckets": {
            "ci": "downstream-artifacts",
            "ci-external": "downstream-external",
        },
        "release_buckets": {"nightly": {"python": "downstream-python"}},
    }

    def test_artifacts_ci_slot_is_redirected(self):
        self.use(self.REGISTRY)
        config = get_artifacts_bucket_config("ci", "ROCm/TheRock", False)
        self.assertEqual(config.name, "downstream-artifacts")
        self.assertEqual(config.key_prefix, "v3/artifacts/")

    def test_artifacts_ci_external_slot_is_separate(self):
        self.use(self.REGISTRY)
        config = get_artifacts_bucket_config("ci", "ROCm/TheRock", True)
        self.assertEqual(config.name, "downstream-external")

    def test_unset_slot_keeps_built_in_formula(self):
        self.use(self.REGISTRY)
        self.assertEqual(
            get_artifacts_bucket_config("nightly", "ROCm/TheRock", False).name,
            "therock-nightly-artifacts",
        )

    def test_downstream_repo_uses_the_ci_external_slot(self):
        """A non-TheRock repo is 'external' even when the PR is not from a fork.

        This is why a downstream registry has to name both CI slots; the NPI
        round trip below is the case that motivated the selection override.
        """
        self.use(self.REGISTRY)
        self.assertEqual(
            get_artifacts_bucket_config(
                "ci", "AMD-ROCm-Internal/rocm-npi-dev", False
            ).name,
            "downstream-external",
        )

    def test_key_prefix_reaches_the_public_url(self):
        """Full round trip: prefix present in the S3 key, absent from the CDN URL."""
        from _therock_utils.workflow_outputs import WorkflowOutputRoot

        self.use(
            {
                "version": 1,
                "buckets": [
                    {
                        "name": "downstream-artifacts",
                        "key_prefix": "v3/artifacts/",
                        "cdn_rules": [
                            {
                                "key_prefix": "v3/artifacts/",
                                "url_prefix": "https://cdn.example.com/artifacts/",
                            }
                        ],
                    }
                ],
                "artifacts_buckets": {"ci-external": "downstream-artifacts"},
            }
        )
        config = get_artifacts_bucket_config("ci", "downstream/repo", False)
        root = WorkflowOutputRoot(
            bucket=config.name,
            external_repo="",
            run_id="4242",
            platform="linux",
            key_prefix=config.key_prefix,
        )
        location = root.artifact_index()
        self.assertEqual(
            location.s3_uri,
            "s3://downstream-artifacts/v3/artifacts/4242-linux/index.html",
        )
        self.assertEqual(
            location.public_url,
            "https://cdn.example.com/artifacts/4242-linux/index.html",
        )

    def test_release_slot_is_redirected(self):
        self.use(self.REGISTRY)
        self.assertEqual(
            get_release_bucket_config("nightly", "python").name, "downstream-python"
        )

    def test_unset_release_slot_keeps_built_in_formula(self):
        self.use(self.REGISTRY)
        self.assertEqual(
            get_release_bucket_config("nightly", "tarball").name,
            "therock-nightly-tarball",
        )
        self.assertEqual(
            get_release_bucket_config("dev", "python").name, "therock-dev-python"
        )

    def test_overriding_ci_alone_does_not_leak_into_ci_external(self):
        """Fork uploads must not silently land in the trusted bucket."""
        self.use(
            {
                "version": 1,
                "buckets": [{"name": "downstream-artifacts"}],
                "artifacts_buckets": {"ci": "downstream-artifacts"},
            }
        )
        self.assertEqual(
            get_artifacts_bucket_config("ci", "ROCm/TheRock", True).name,
            "therock-ci-artifacts-external",
        )

    def test_invalid_release_type_still_rejected(self):
        self.use(self.REGISTRY)
        with self.assertRaises(ValueError):
            get_artifacts_bucket_config("bogus", "ROCm/TheRock", False)


class TestRequireBucketConfigError(unittest.TestCase):
    def test_message_names_known_buckets_and_injection_mechanism(self):
        with self.assertRaises(KeyError) as cm:
            require_bucket_config("nope")
        message = str(cm.exception)
        self.assertIn("therock-ci-artifacts", message)
        self.assertIn(BUCKET_REGISTRY_ENV_VAR, message)
        self.assertIn("--bucket-config-file", message)


if __name__ == "__main__":
    unittest.main()
