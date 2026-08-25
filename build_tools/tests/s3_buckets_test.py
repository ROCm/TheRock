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
    cdn_url_for,
    get_artifacts_bucket_config,
    get_artifacts_bucket_config_for_workflow_run,
    get_index_release_stream,
    get_legacy_release_index_url,
    get_product_release_bucket_config,
    get_release_package_index_url,
    get_release_stream,
    get_release_tarball_index_url,
    get_release_bucket_config,
    lookup_bucket_config,
    require_bucket_config,
    reset_bucket_registry,
    release_stream_url,
    resolve_public_url,
    set_bucket_config_file,
)
from _therock_utils.storage_location import StorageLocation
from setup_venv import INDEX_NAME_TO_RELEASE_TYPE


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

    def test_bkc_release_types_use_existing_artifacts_buckets(self):
        for release_type, bucket_name in (
            ("dev-bkc", "therock-dev-artifacts"),
            ("nightly-bkc", "therock-nightly-artifacts"),
        ):
            with self.subTest(release_type=release_type):
                config = get_artifacts_bucket_config(
                    release_type=release_type,
                    repository="ROCm/TheRock",
                    is_pr_from_fork=False,
                )
                self.assertEqual(config.name, bucket_name)

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

    def test_bkc_release_types_use_existing_release_buckets(self):
        for release_type, bucket_name in (
            ("dev-bkc", "therock-dev-python"),
            ("nightly-bkc", "therock-nightly-python"),
        ):
            with self.subTest(release_type=release_type):
                config = get_release_bucket_config(
                    release_type=release_type, bucket_type="python"
                )
                self.assertEqual(config.name, bucket_name)

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
# product release helpers
# ---------------------------------------------------------------------------


class TestProductReleaseHelpers(unittest.TestCase):
    def test_release_type_maps_to_external_stream(self):
        expected_streams = {
            "dev": "dev",
            "nightly": "nightly",
            "prerelease": "rc",
            "dev-bkc": "bkc",
            "nightly-bkc": "bkc",
        }
        for release_type, expected_stream in expected_streams.items():
            with self.subTest(release_type=release_type):
                self.assertEqual(get_release_stream(release_type), expected_stream)

    def test_product_release_bucket_config(self):
        for release_type, stream in (
            ("dev", "dev"),
            ("nightly", "nightly"),
            ("prerelease", "rc"),
            ("dev-bkc", "bkc"),
            ("nightly-bkc", "bkc"),
        ):
            for product in ("core", "pytorch", "jax"):
                with self.subTest(release_type=release_type, product=product):
                    config = get_product_release_bucket_config(release_type, product)
                    self.assertEqual(
                        config.name, f"therock-repo-amd-{stream}-{product}"
                    )
                    self.assertEqual(config.region, "us-east-2")
                    self.assertEqual(config.iam_account, "324352301041")
                    self.assertEqual(
                        config.iam_role, f"therock-repo-{stream}-{product}"
                    )
                    self.assertEqual(
                        config.write_access_iam_role,
                        f"arn:aws:iam::324352301041:role/therock-repo-{stream}-{product}",
                    )

    def test_release_package_index_url_is_aggregate_index(self):
        expected_urls = {
            "dev": "https://dev.repo.amd.com/rocm/whl-next/",
            "nightly": "https://nightly.repo.amd.com/rocm/whl-next/",
            "prerelease": "https://rc.repo.amd.com/rocm/whl-next/",
            "dev-bkc": "https://bkc.repo.amd.com/rocm/whl-next/",
            "nightly-bkc": "https://bkc.repo.amd.com/rocm/whl-next/",
        }
        for release_type, expected_url in expected_urls.items():
            with self.subTest(release_type=release_type):
                self.assertEqual(
                    get_release_package_index_url(release_type), expected_url
                )

    def test_invalid_product_release_type_raises(self):
        with self.assertRaises(ValueError) as cm:
            get_product_release_bucket_config("weekly", "core")
        self.assertIn("weekly", str(cm.exception))

    def test_invalid_product_raises(self):
        with self.assertRaises(ValueError) as cm:
            get_product_release_bucket_config("dev", "python")
        self.assertIn("python", str(cm.exception))


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


class TestS3BucketConfigKeyPrefix(unittest.TestCase):
    """The trailing-slash invariant documented on WorkflowOutputRoot.key_prefix.

    Enforced on the dataclass, not only in the registry parser, so a config built
    directly in Python cannot produce 'v34242-linux' out of key_prefix='v3'.
    """

    def test_appends_trailing_slash(self):
        self.assertEqual(S3BucketConfig(name="b", key_prefix="v3").key_prefix, "v3/")

    def test_keeps_existing_trailing_slash(self):
        self.assertEqual(S3BucketConfig(name="b", key_prefix="v3/").key_prefix, "v3/")

    def test_empty_key_prefix_stays_empty(self):
        self.assertEqual(S3BucketConfig(name="b").key_prefix, "")

    def test_leading_slash_raises(self):
        with self.assertRaises(ValueError):
            S3BucketConfig(name="b", key_prefix="/v3/")


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


class TestProductReleaseBuckets(unittest.TestCase):
    """The repo.amd.com product buckets, now registered rather than synthesized.

    Registering them is what lets StorageLocation derive their public URLs; the
    price is that get_product_release_bucket_config must keep returning exactly
    what it built inline before, which the first test below pins field by field.
    """

    # What get_product_release_bucket_config constructed before these buckets
    # joined the inventory. Written out rather than imported so that a change to
    # the inventory cannot quietly redefine what "unchanged" means.
    EXPECTED = {
        "dev": ("therock-repo-amd-dev-core", "therock-repo-dev-core"),
        "dev-bkc": ("therock-repo-amd-bkc-core", "therock-repo-bkc-core"),
        "nightly": ("therock-repo-amd-nightly-core", "therock-repo-nightly-core"),
        "nightly-bkc": ("therock-repo-amd-bkc-core", "therock-repo-bkc-core"),
        "prerelease": ("therock-repo-amd-rc-core", "therock-repo-rc-core"),
    }

    def test_identical_to_the_previously_synthesized_config(self):
        for release_type, (name, role) in self.EXPECTED.items():
            with self.subTest(release_type=release_type):
                config = get_product_release_bucket_config(release_type, "core")
                self.assertEqual(config.name, name)
                self.assertEqual(config.iam_role, role)
                self.assertEqual(config.iam_account, "324352301041")
                self.assertEqual(config.region, "us-east-2")
                self.assertEqual(
                    config.write_access_iam_role,
                    f"arn:aws:iam::324352301041:role/{role}",
                )

    def test_v5_prefix_is_stripped_in_the_public_url(self):
        """The whole point of registering them: v5/ in the bucket, gone on the CDN.

        Asserted as literal URLs rather than rebuilt from release_stream_url, so
        a wrong formula cannot agree with itself.
        """
        cases = [
            (
                "therock-repo-amd-dev-core",
                "v5/rocm/core/tarball/therock-dist-linux-gfx94X-dcgpu-7.10.0.tar.gz",
                "https://dev.repo.amd.com/rocm/core/tarball/"
                "therock-dist-linux-gfx94X-dcgpu-7.10.0.tar.gz",
            ),
            (
                "therock-repo-amd-nightly-core",
                "v5/rocm/core/whl-next/rocm-sdk-core/rocm_sdk_core-7.13.0.whl",
                "https://nightly.repo.amd.com/rocm/core/whl-next/rocm-sdk-core/"
                "rocm_sdk_core-7.13.0.whl",
            ),
            (
                "therock-repo-amd-rc-core",
                "v5/rocm/core/packages/deb/",
                "https://rc.repo.amd.com/rocm/core/packages/deb/",
            ),
            (
                "therock-repo-amd-bkc-core",
                "v5/rocm/core/tarball/therock-dist-linux-gfx950-dcgpu-7.10.0.tar.gz",
                "https://bkc.repo.amd.com/rocm/core/tarball/"
                "therock-dist-linux-gfx950-dcgpu-7.10.0.tar.gz",
            ),
            (
                "therock-repo-amd-nightly-pytorch",
                "v5/rocm/pytorch/whl-next/torch/torch-2.9.0.whl",
                "https://nightly.repo.amd.com/rocm/pytorch/whl-next/torch/"
                "torch-2.9.0.whl",
            ),
        ]
        for bucket, key, expected in cases:
            with self.subTest(bucket=bucket):
                self.assertEqual(StorageLocation(bucket, key).public_url, expected)

    def test_bkc_streams_resolve_like_any_other(self):
        """Both BKC release types share one stream, and it rewrites normally."""
        for release_type in ("dev-bkc", "nightly-bkc"):
            with self.subTest(release_type=release_type):
                config = get_product_release_bucket_config(release_type, "core")
                self.assertEqual(config.name, "therock-repo-amd-bkc-core")
                self.assertEqual(
                    cdn_url_for(config.name, "v5/rocm/core/tarball/x.tar.gz"),
                    "https://bkc.repo.amd.com/rocm/core/tarball/x.tar.gz",
                )

    def test_release_stream_url(self):
        for stream in ("dev", "nightly", "weekly", "rc", "stable", "bkc"):
            with self.subTest(stream=stream):
                self.assertEqual(
                    release_stream_url(stream), f"https://{stream}.repo.amd.com/"
                )

    def test_every_product_bucket_has_a_rule(self):
        """No stream is a special case; the formula covers the whole inventory."""
        for config in all_bucket_configs():
            if config.name.startswith("therock-repo-amd-"):
                with self.subTest(bucket=config.name):
                    self.assertEqual(len(config.cdn_rules), 1)
                    self.assertEqual(config.key_prefix, "v5/")

    def test_invalid_product_is_rejected(self):
        with self.assertRaises(ValueError):
            get_product_release_bucket_config("nightly", "tensorflow")


class TestReleaseIndexUrls(unittest.TestCase):
    """The public index URLs users install from.

    Every expected value is written out in full rather than rebuilt from the
    same streams and CdnRules the implementation reads. These URLs appear in
    install instructions and are passed to pip, so a test that derived them
    would agree with any rule the code happens to hold, including a wrong one.
    """

    def test_aggregate_pip_index(self):
        expected = {
            "dev": "https://dev.repo.amd.com/rocm/whl-next/",
            "dev-bkc": "https://bkc.repo.amd.com/rocm/whl-next/",
            "nightly": "https://nightly.repo.amd.com/rocm/whl-next/",
            "nightly-bkc": "https://bkc.repo.amd.com/rocm/whl-next/",
            "prerelease": "https://rc.repo.amd.com/rocm/whl-next/",
            "release": "https://stable.repo.amd.com/rocm/whl-next/",
        }
        for release_type, url in expected.items():
            with self.subTest(release_type=release_type):
                self.assertEqual(get_release_package_index_url(release_type), url)

    def test_setup_venv_index_names_are_unchanged(self):
        """The four --index-name values still resolve to what they did before.

        setup_venv.py held these as a literal map; it now derives them. Same
        strings either way, so no command line changes.
        """
        expected = {
            "stable": "https://stable.repo.amd.com/rocm/whl-next/",
            "prerelease": "https://rc.repo.amd.com/rocm/whl-next/",
            "nightly": "https://nightly.repo.amd.com/rocm/whl-next/",
            "dev": "https://dev.repo.amd.com/rocm/whl-next/",
        }
        for index_name, url in expected.items():
            with self.subTest(index_name=index_name):
                self.assertEqual(
                    get_release_package_index_url(
                        INDEX_NAME_TO_RELEASE_TYPE[index_name]
                    ),
                    url,
                )

    def test_product_local_index(self):
        self.assertEqual(
            get_release_package_index_url("nightly", "pytorch"),
            "https://nightly.repo.amd.com/rocm/pytorch/whl-next/",
        )
        self.assertEqual(
            get_release_package_index_url("dev", "jax"),
            "https://dev.repo.amd.com/rocm/jax/whl-next/",
        )

    def test_tarball_index(self):
        self.assertEqual(
            get_release_tarball_index_url("nightly"),
            "https://nightly.repo.amd.com/rocm/core/tarball/",
        )
        self.assertEqual(
            get_release_tarball_index_url("release"),
            "https://stable.repo.amd.com/rocm/core/tarball/",
        )

    def test_legacy_index_urls_match_the_maps_they_replaced(self):
        """Byte-identical to the LEGACY_MULTI_ARCH_INDEX_URLS copies deleted here.

        Those lived in publish_pytorch_to_release_bucket.py and
        publish_jax_to_release_bucket.py, identical in both.
        """
        expected = {
            "dev": "https://rocm.devreleases.amd.com/whl-multi-arch/",
            "dev-bkc": "https://rocm.devreleases.amd.com/whl-multi-arch/",
            "nightly": "https://rocm.nightlies.amd.com/whl-multi-arch/",
            "nightly-bkc": "https://rocm.nightlies.amd.com/whl-multi-arch/",
            "prerelease": "https://rocm.prereleases.amd.com/whl-multi-arch/",
        }
        for release_type, url in expected.items():
            with self.subTest(release_type=release_type):
                self.assertEqual(get_legacy_release_index_url(release_type), url)

    def test_legacy_tarball_index(self):
        self.assertEqual(
            get_legacy_release_index_url("nightly", "tarball"),
            "https://rocm.nightlies.amd.com/tarball-multi-arch/",
        )
        self.assertEqual(
            get_legacy_release_index_url("release", "tarball"),
            "https://repo.amd.com/rocm/tarball-multi-arch/",
        )

    def test_release_is_installable_from_but_not_publishable_to(self):
        """The read set is wider than the write set, on purpose.

        The stable buckets have no automated upload credentials, so the publish
        path must keep rejecting "release" - but it is the channel most users
        install from.
        """
        self.assertEqual(get_index_release_stream("release"), "stable")
        with self.assertRaises(ValueError):
            get_release_stream("release")
        with self.assertRaises(ValueError):
            get_release_bucket_config("release", "python")

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            get_release_package_index_url("bogus")
        with self.assertRaises(ValueError):
            get_release_package_index_url("nightly", "tensorflow")
        with self.assertRaises(ValueError):
            get_legacy_release_index_url("bogus")
        with self.assertRaises(ValueError):
            # "packages" has no single index: that CDN serves a
            # distro-partitioned apt/dnf repo, not a prefix rewrite.
            get_legacy_release_index_url("nightly", "packages")


class TestAnonymousS3Read(unittest.TestCase):
    """download_url picks raw S3 or the CDN from a fact about the bucket.

    Verified against the live buckets on 2026-08-25 by requesting a key that
    does not exist: a public bucket answers 404 NoSuchKey, a private one answers
    403 AccessDenied. Every therock-*-packages bucket, both prerelease and
    release python/tarball buckets, and all repo.amd.com product buckets
    answered 403.
    """

    def test_public_buckets_keep_the_raw_s3_url(self):
        # Every URL TheRock's CI hands to pip or apt today resolves to one of
        # these, so this is the assertion that says "nothing changed".
        for bucket in (
            "therock-ci-artifacts",
            "therock-ci-artifacts-external",
            "therock-dev-artifacts",
            "therock-nightly-artifacts",
            "therock-prerelease-artifacts",
        ):
            with self.subTest(bucket=bucket):
                location = StorageLocation(bucket, "12345-linux/python/index.html")
                self.assertTrue(require_bucket_config(bucket).anonymous_s3_read)
                self.assertEqual(location.download_url, location.https_url)

    def test_private_buckets_fall_back_to_the_cdn(self):
        location = StorageLocation("therock-prerelease-tarball", "v4/tarball/x.tar.gz")
        self.assertEqual(
            location.download_url,
            "https://rocm.prereleases.amd.com/tarball-multi-arch/x.tar.gz",
        )

    def test_private_bucket_with_no_rule_raises(self):
        """A 403 in a build log is a worse way to learn this than an exception.

        therock-prerelease-packages is private and deliberately carries no rule
        (its CDN serves a distro-partitioned repo, not a prefix rewrite), so it
        has no public download URL at all.
        """
        location = StorageLocation("therock-prerelease-packages", "v4/deb/x.deb")
        with self.assertRaises(ValueError) as cm:
            location.download_url
        message = str(cm.exception)
        self.assertIn("therock-prerelease-packages", message)
        self.assertIn("anonymous", message)

    def test_unknown_bucket_keeps_the_raw_s3_url(self):
        """An unregistered bucket behaves exactly as it did before this field."""
        location = StorageLocation("some-other-bucket", "path/file.txt")
        self.assertEqual(location.download_url, location.https_url)


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
        self.assertEqual(config.anonymous_s3_read, original.anonymous_s3_read)

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
                "anonymous_s3_read",
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

    def test_bkc_slots_are_redirectable_independently(self):
        """A BKC channel shares dev's bucket but not dev's selection slot.

        Redirecting "dev-bkc" must not drag "dev" with it, or a downstream repo
        could not separate the two.
        """
        self.use(
            {
                "version": 1,
                "buckets": [{"name": "downstream-bkc-artifacts"}],
                "artifacts_buckets": {"dev-bkc": "downstream-bkc-artifacts"},
            }
        )
        self.assertEqual(
            get_artifacts_bucket_config("dev-bkc", "ROCm/TheRock", False).name,
            "downstream-bkc-artifacts",
        )
        self.assertEqual(
            get_artifacts_bucket_config("dev", "ROCm/TheRock", False).name,
            "therock-dev-artifacts",
        )

    def test_product_release_slot_is_redirected(self):
        """The v5 product buckets are selected by formula too, so they need a slot.

        ``therock-repo-amd-{stream}-{product}`` is no more guessable by a
        downstream repo than ``therock-{release_type}-artifacts`` is.
        """
        self.use(
            {
                "version": 1,
                "buckets": [{"name": "downstream-core"}],
                "product_release_buckets": {"nightly": {"core": "downstream-core"}},
            }
        )
        self.assertEqual(
            get_product_release_bucket_config("nightly", "core").name,
            "downstream-core",
        )
        # An unset (release_type, product) pair keeps the built-in formula.
        self.assertEqual(
            get_product_release_bucket_config("nightly", "jax").name,
            "therock-repo-amd-nightly-jax",
        )


class TestDownstreamRegistryRoundTrip(RegistryFileTestCase):
    """The shape rocm-npi-dev needs, pinned in TheRock's own suite.

    That repo currently reaches the same result by monkey-patching
    WorkflowOutputRoot.from_workflow_run and string-replacing S3 hostnames in
    job summaries, in two scripts each. Both copies are installed by sniffing
    TheRock's module layout at run time, so an upstream refactor breaks them
    silently. Encoding the contract here is what makes that detectable.

    Three buckets share one CDN host and each strips a different ``v3/…``
    prefix - the case a single per-bucket CDN URL cannot express, because it
    supplies a URL to prepend but no prefix length to strip. It is the same
    shape as RFC0012's ``v5/`` rewrite.
    """

    REGISTRY = {
        "version": 1,
        "buckets": [
            {
                "name": "therock-npi-artifacts",
                "key_prefix": "v3/artifacts/",
                "anonymous_s3_read": False,
                "cdn_rules": [
                    {
                        "key_prefix": "v3/artifacts/",
                        "url_prefix": "https://rocm.genesis.amd.com/artifacts/",
                    }
                ],
            },
            {
                "name": "therock-npi-tarball",
                "key_prefix": "v3/tarball/",
                "anonymous_s3_read": False,
                "cdn_rules": [
                    {
                        "key_prefix": "v3/tarball/",
                        "url_prefix": "https://rocm.genesis.amd.com/tarball/",
                    }
                ],
            },
            {
                "name": "therock-npi-python",
                "key_prefix": "v3/whl/",
                "anonymous_s3_read": False,
                "cdn_rules": [
                    {
                        "key_prefix": "v3/whl/",
                        "url_prefix": "https://rocm.genesis.amd.com/whl/",
                    }
                ],
            },
        ],
        "artifacts_buckets": {
            "ci": "therock-npi-artifacts",
            "ci-external": "therock-npi-artifacts",
        },
        "release_buckets": {"nightly": {"python": "therock-npi-python"}},
    }

    def test_each_bucket_rewrites_its_own_prefix(self):
        """Literal URLs, matching the rewrite table those wrapper scripts hold."""
        self.use(self.REGISTRY)
        cases = [
            (
                "therock-npi-artifacts",
                "v3/artifacts/4242-linux/index.html",
                "https://rocm.genesis.amd.com/artifacts/4242-linux/index.html",
            ),
            (
                "therock-npi-tarball",
                "v3/tarball/therock-dist-linux-gfx1250-7.0.0.tar.gz",
                "https://rocm.genesis.amd.com/tarball/"
                "therock-dist-linux-gfx1250-7.0.0.tar.gz",
            ),
            (
                "therock-npi-python",
                "v3/whl/gfx1250/rocm-7.0.0.whl",
                "https://rocm.genesis.amd.com/whl/gfx1250/rocm-7.0.0.whl",
            ),
        ]
        for bucket, key, expected in cases:
            with self.subTest(bucket=bucket):
                self.assertEqual(StorageLocation(bucket, key).public_url, expected)

    def test_private_buckets_get_the_cdn_for_machine_urls(self):
        """The pip index must be the CDN, because raw S3 answers 403 there.

        TheRock's own buckets are readable and keep the raw S3 URL for egress
        cost. Both answers come from the same field, so neither repo has to
        override a policy set for the other.
        """
        self.use(self.REGISTRY)
        location = StorageLocation("therock-npi-python", "v3/whl/gfx1250/index.html")
        self.assertEqual(
            location.download_url,
            "https://rocm.genesis.amd.com/whl/gfx1250/index.html",
        )
        self.assertEqual(
            StorageLocation(
                "therock-ci-artifacts", "4242-linux/python/index.html"
            ).download_url,
            "https://therock-ci-artifacts.s3.amazonaws.com/4242-linux/python/index.html",
        )

    def test_key_prefix_is_folded_into_the_s3_key(self):
        """The prefix reaches the key without going through ``external_repo``.

        Storing it in external_repo is the abuse this replaces: that field means
        "fork namespace", and a run in a fork would have overwritten it.
        """
        from _therock_utils.workflow_outputs import WorkflowOutputRoot

        self.use(self.REGISTRY)
        root = WorkflowOutputRoot.from_workflow_run(
            run_id="4242",
            platform="linux",
            github_repository="AMD-ROCm-Internal/rocm-npi-dev",
            workflow_run={
                "id": 4242,
                "head_repository": {"full_name": "AMD-ROCm-Internal/rocm-npi-dev"},
            },
        )
        location = root.artifact_index()
        self.assertTrue(location.relative_path.startswith("v3/artifacts/"))
        self.assertTrue(
            location.public_url.startswith("https://rocm.genesis.amd.com/artifacts/")
        )
        # The prefix is stripped on the way out, not carried into the CDN path.
        self.assertNotIn("v3/artifacts", location.public_url)


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
