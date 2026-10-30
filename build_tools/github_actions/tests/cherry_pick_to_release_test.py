# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
from cherry_pick_to_release import (
    bot_identity,
    build_commit_message,
    build_pr_body,
    build_pr_title,
    check_git_version,
    clone_target,
    _conflict_paths,
    CherryPickError,
    CommitInfo,
    create_pull_request,
    derive_pr_branch,
    execute_cherry_pick,
    main,
    parse_commit_list,
    parse_dry_run,
    parse_jira_id,
    parse_release_branch,
    parse_repository,
    push_branch,
    resolve_commits,
    RunResult,
    target_metadata,
    validate_ticket_requirement,
)

from github_actions_api import GitHubAPIError


THEROCK_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_PATH = THEROCK_ROOT / ".github/workflows/cherry-pick-to-release.yml"
RUN_URL = "https://github.com/ROCm/TheRock/actions/runs/123456"
TOKEN = "test-app-token"  # nosec B105


FIXTURE_COMMIT_ENV = {
    "GIT_AUTHOR_NAME": "Original Author",
    "GIT_AUTHOR_EMAIL": "original@example.com",
    "GIT_AUTHOR_DATE": "2026-10-25T10:20:30+00:00",
    "GIT_COMMITTER_NAME": "Fixture Committer",
    "GIT_COMMITTER_EMAIL": "fixture@example.com",
    "GIT_COMMITTER_DATE": "2026-10-25T10:20:30+00:00",
}


def _isolated_git_environment() -> dict[str, str]:
    """Keep OS settings while isolating Git from the caller's repositories/config."""
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "core.attributesFile",
            "GIT_CONFIG_VALUE_0": os.devnull,
        }
    )
    return environment


def _git(
    *args: str,
    cwd: Path | None = None,
    input_text: str | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = ["git", *args]
    command_env = _isolated_git_environment()
    if env is not None:
        command_env.update(env)
    return subprocess.run(
        command,
        cwd=cwd,
        input=input_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=command_env,
        check=check,
    )


@dataclass(frozen=True)
class APICall:
    url: str
    method: str
    body: object | None


class FakeGitHubAPI:
    def __init__(
        self,
        *,
        default_branch: str = "develop",
        bot_response: object | None = None,
        pr_response: object | None = None,
        pr_error: Exception | None = None,
    ) -> None:
        self.default_branch = default_branch
        self.bot_response = (
            {
                "login": "systems-assistant[bot]",
                "id": 45123,
                "email": "systems-assistant@example.com",
            }
            if bot_response is None
            else bot_response
        )
        self.pr_response = (
            {"html_url": "https://github.com/ROCm/rocm-systems/pull/99"}
            if pr_response is None
            else pr_response
        )
        self.pr_error = pr_error
        self.calls: list[APICall] = []

    def send_request(
        self,
        url: str,
        timeout_seconds: int = 300,
        *,
        method: str = "GET",
        body: object | None = None,
    ) -> object:
        del timeout_seconds
        self.calls.append(APICall(url=url, method=method, body=body))
        if url == "https://api.github.com/repos/ROCm/rocm-systems":
            return {"default_branch": self.default_branch}
        if url == "https://api.github.com/users/systems-assistant%5Bbot%5D":
            return self.bot_response
        if url == "https://api.github.com/repos/ROCm/rocm-systems/pulls":
            if self.pr_error is not None:
                raise self.pr_error
            return self.pr_response
        raise AssertionError(f"Unexpected API request: {method} {url}")

    @property
    def pull_requests(self) -> list[APICall]:
        return [call for call in self.calls if call.method == "POST"]


def _commit_info(subject: str = "Fix the release") -> CommitInfo:
    return CommitInfo(
        sha="a" * 40,
        parent="b" * 40,
        subject=subject,
        message=f"{subject}\n",
        author_name="Release Author",
        author_email="author@example.com",
        author_date="2026-10-25T10:00:00+00:00",
    )


class InputParsingTest(unittest.TestCase):
    def test_repository_accepts_only_supported_choices(self) -> None:
        self.assertEqual(parse_repository("TheRock"), "TheRock")
        self.assertEqual(parse_repository("rocm-systems"), "rocm-systems")
        self.assertEqual(parse_repository("rocm-libraries"), "rocm-libraries")
        for value in ("therock", "ROCm/rocm-systems", "", " rocm-systems "):
            with self.subTest(value=value), self.assertRaises(CherryPickError):
                parse_repository(value)

    def test_commit_list_accepts_space_comma_and_newline_separators(self) -> None:
        commits = parse_commit_list("abcdef0, 12345678\nABCDEF012345")
        self.assertEqual(commits, ["abcdef0", "12345678", "ABCDEF012345"])

    def test_commit_list_rejects_empty_non_hex_and_short_values(self) -> None:
        for value in ("", "   ,\n", "abcdef", "abcdefg", "abc123!", "a" * 41):
            with self.subTest(value=value), self.assertRaises(CherryPickError):
                parse_commit_list(value)

    def test_jira_id_accepts_expected_keys_and_trims_whitespace(self) -> None:
        accepted = {
            "AIRDEL-1234": "AIRDEL-1234",
            " AIRDEL-1234 ": "AIRDEL-1234",
            "ROCM-42": "ROCM-42",
            "AB_1-7": "AB_1-7",
            "": None,
            " \n": None,
        }
        for value, expected in accepted.items():
            with self.subTest(value=value):
                self.assertEqual(parse_jira_id(value), expected)

    def test_jira_id_rejects_invalid_values(self) -> None:
        for value in (
            "1234",
            "airdel-1234",
            "JIRA ID: AIRDEL-1234",
            "AIRDEL-0123",
            "AIRDEL-",
            "AIRDEL-12a",
            "1AB-5",
            "AIRDEL-1 AIRDEL-2",
        ):
            with self.subTest(value=value), self.assertRaises(CherryPickError):
                parse_jira_id(value)

    def test_release_branch_families_produce_expected_labels(self) -> None:
        self.assertEqual(parse_release_branch("release/therock-10.2").label, "10.2")
        self.assertEqual(
            parse_release_branch("release/bkc/therock-10.2-20261013").label,
            "10.2 BKC 20261013",
        )

    def test_release_branch_rejects_other_families(self) -> None:
        for branch in (
            "release/rocm-rel-7.0",
            "release/docs-hipdnn-1.0",
            "release/therock-10.2.1",
            "release/bkc/therock-10.2",
            "main",
        ):
            with self.subTest(branch=branch), self.assertRaises(CherryPickError):
                parse_release_branch(branch)

    def test_ticket_requirement_depends_on_release_family(self) -> None:
        regular = parse_release_branch("release/therock-10.2")
        bkc = parse_release_branch("release/bkc/therock-10.2-20261013")
        with self.assertRaisesRegex(CherryPickError, "JIRA ID is required"):
            validate_ticket_requirement(regular, None)
        validate_ticket_requirement(regular, "AIRDEL-1234")
        validate_ticket_requirement(bkc, None)

    def test_dry_run_parser_has_no_truthy_fallback(self) -> None:
        self.assertTrue(parse_dry_run("true"))
        self.assertTrue(parse_dry_run(" TRUE "))
        self.assertFalse(parse_dry_run("false"))
        for value in ("", "yes", "1", "truthy"):
            with self.subTest(value=value), self.assertRaises(CherryPickError):
                parse_dry_run(value)


class ContentBuilderTest(unittest.TestCase):
    def test_ticketless_bkc_names_and_content_have_no_jira_line(self) -> None:
        release = parse_release_branch("release/bkc/therock-10.2-20261013")
        commit = _commit_info()
        branch = derive_pr_branch(release, None, commit.sha)
        message = build_commit_message(commit.sha, release.label, None, commit.message)
        title = build_pr_title(release.label, None, [commit, commit])
        body = build_pr_body(release, None, [commit], RUN_URL, "release-operator")

        self.assertEqual(branch, "shared/cherry-pick/bkc-therock-10.2-20261013-aaaaaaa")
        self.assertNotIn("JIRA", message)
        self.assertEqual(title, "[ROCm 10.2 BKC 20261013] cherry-pick 2 commits")
        self.assertNotIn("JIRA", body)
        self.assertIn(RUN_URL, body)

    def test_pr_titles_for_one_and_multiple_commits(self) -> None:
        commits = [_commit_info("First fix"), _commit_info("Second fix")]
        self.assertEqual(
            build_pr_title("10.2", "AIRDEL-12", commits[:1]), "[ROCm 10.2] First fix"
        )
        self.assertEqual(
            build_pr_title("10.2", "AIRDEL-12", commits),
            "[ROCm 10.2] AIRDEL-12: cherry-pick 2 commits",
        )

    def test_pr_body_links_to_therock_run_and_lists_full_sha(self) -> None:
        release = parse_release_branch("release/therock-10.2")
        body = build_pr_body(
            release,
            "AIRDEL-12",
            [_commit_info()],
            RUN_URL,
            "release-operator",
        )
        self.assertIn(f"`{'a' * 40}` Fix the release", body)
        self.assertIn("JIRA ID: AIRDEL-12", body)
        self.assertIn(RUN_URL, body)
        self.assertNotIn("rocm-systems/actions/runs", body)


class GitHubOperationsTest(unittest.TestCase):
    def test_target_metadata_rejects_missing_or_malformed_default_branch(self) -> None:
        class MetadataAPI:
            def __init__(self, response: object) -> None:
                self.response = response

            def send_request(
                self,
                url: str,
                timeout_seconds: int = 300,
                *,
                method: str = "GET",
                body: object | None = None,
            ) -> object:
                del url, timeout_seconds, method, body
                return self.response

        for response in (None, [], {}, {"default_branch": ""}, {"default_branch": 5}):
            with self.subTest(response=response), self.assertRaises(CherryPickError):
                target_metadata(MetadataAPI(response), "rocm-systems")

    def test_bot_identity_uses_email_or_noreply_fallback(self) -> None:
        api = FakeGitHubAPI()
        identity = bot_identity(api, "systems-assistant")
        self.assertEqual(identity.name, "systems-assistant[bot]")
        self.assertEqual(identity.email, "systems-assistant@example.com")

        fallback_api = FakeGitHubAPI(
            bot_response={
                "login": "systems-assistant[bot]",
                "id": 45123,
                "email": None,
            }
        )
        identity = bot_identity(fallback_api, "systems-assistant")
        self.assertEqual(
            identity.email,
            "45123+systems-assistant[bot]@users.noreply.github.com",
        )

    def test_bot_identity_rejects_incomplete_response(self) -> None:
        for response in (
            [],
            {},
            {"login": "someone-else[bot]", "id": 1},
            {"login": "systems-assistant[bot]", "email": None},
            {"login": "systems-assistant[bot]", "id": 1, "email": ""},
            {"login": "systems-assistant[bot]", "id": 1, "email": 5},
        ):
            api = FakeGitHubAPI(bot_response=response)
            with self.subTest(response=response), self.assertRaises(CherryPickError):
                bot_identity(api, "systems-assistant")

    def test_create_pull_request_rejects_malformed_response(self) -> None:
        for response in (
            [],
            {},
            {"html_url": ""},
            {"html_url": 5},
            {"html_url": "https://example.com/ROCm/rocm-systems/pull/1"},
            {"html_url": "https://github.com/ROCm/another-repo/pull/1"},
        ):
            api = FakeGitHubAPI(pr_response=response)
            with self.subTest(response=response), self.assertRaises(CherryPickError):
                create_pull_request(
                    api,
                    "rocm-systems",
                    "release/therock-10.2",
                    "shared/cherry-pick/therock-10.2-airdel-12",
                    "title",
                    "body",
                )


class GitPrimitiveValidationTest(unittest.TestCase):
    def test_commit_resolution_failure_is_reported(self) -> None:
        ambiguous = subprocess.CompletedProcess(
            args=["git", "rev-parse"],
            returncode=128,
            stdout="",
            stderr="error: short object ID abcdef0 is ambiguous",
        )
        with patch("cherry_pick_to_release._run_git", return_value=ambiguous):
            with self.assertRaisesRegex(CherryPickError, "exactly one target commit"):
                resolve_commits(Path("/unused"), ["abcdef0"])

    def test_git_older_than_attribute_source_requirement_is_rejected(self) -> None:
        old_version = subprocess.CompletedProcess(
            args=["git", "--version"],
            returncode=0,
            stdout="git version 2.40.9\n",
            stderr="",
        )
        with patch("cherry_pick_to_release._run_git", return_value=old_version):
            with self.assertRaisesRegex(CherryPickError, "Git 2.41 or newer"):
                check_git_version()

    def test_git_with_attribute_source_support_is_accepted(self) -> None:
        supported_version = subprocess.CompletedProcess(
            args=["git", "--version"],
            returncode=0,
            stdout="git version 2.41.0\n",
            stderr="",
        )
        with patch("cherry_pick_to_release._run_git", return_value=supported_version):
            self.assertEqual(check_git_version(), (2, 41))


class CommandLineTest(unittest.TestCase):
    def test_dry_run_defaults_environment_and_explicit_overrides(self) -> None:
        cases = (
            (None, [], True),
            ("true", [], True),
            ("false", [], False),
            ("false", ["--dry-run"], True),
            ("true", ["--no-dry-run"], False),
            (None, ["--no-dry-run"], False),
        )
        environment = {
            "CHERRY_PICK_REPOSITORY": "rocm-systems",
            "CHERRY_PICK_COMMITS": "a" * 40,
            "CHERRY_PICK_RELEASE_BRANCH": "release/therock-10.2",
            "CHERRY_PICK_JIRA_ID": "AIRDEL-1234",
            "GITHUB_APP_TOKEN": TOKEN,
            "GITHUB_APP_SLUG": "systems-assistant",
            "WORKFLOW_RUN_URL": RUN_URL,
            "WORKFLOW_ACTOR": "release-operator",
        }
        for dry_run_value, argv, expected in cases:
            with self.subTest(environment=dry_run_value, argv=argv):
                with patch.dict(os.environ, environment, clear=True), patch(
                    "cherry_pick_to_release.execute_cherry_pick"
                ) as execute, patch("cherry_pick_to_release.GitHubAPI") as api:
                    if dry_run_value is not None:
                        os.environ["CHERRY_PICK_DRY_RUN"] = dry_run_value
                    self.assertEqual(main(argv), 0)
                    execute.assert_called_once_with(
                        repository_value="rocm-systems",
                        commits_value="a" * 40,
                        release_branch_value="release/therock-10.2",
                        jira_id_value="AIRDEL-1234",
                        dry_run=expected,
                        app_token=TOKEN,
                        app_slug="systems-assistant",
                        workflow_run_url=RUN_URL,
                        workflow_actor="release-operator",
                        api=api.return_value,
                    )
                    api.assert_called_once_with(github_token=TOKEN)


class ConflictOutputTest(unittest.TestCase):
    def test_only_path_section_is_reported_and_path_whitespace_is_preserved(
        self,
    ) -> None:
        paths = [" binary-like.txt ", "nested/module", "warning: real filename"]
        diagnostics = (
            "warning: Cannot merge binary files: binary-like.txt",
            "CONFLICT (binary): Merge conflict in binary-like.txt",
            "CONFLICT (submodule): Merge conflict in nested/module",
        )
        output = "\n".join(["a" * 40, *paths, "", *diagnostics]) + "\n"
        self.assertEqual(_conflict_paths(output), paths)


class WorkflowWiringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = yaml.load(
            WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader
        )
        cls.steps = cls.workflow["jobs"]["cherry_pick"]["steps"]
        cls.cherry_pick_step = next(
            step for step in cls.steps if step.get("name") == "Cherry-pick commits"
        )

    def test_dispatch_inputs_have_safe_defaults(self) -> None:
        inputs = self.workflow["on"]["workflow_dispatch"]["inputs"]
        self.assertEqual(
            set(inputs["repository"]["options"]),
            {"TheRock", "rocm-systems", "rocm-libraries"},
        )
        self.assertEqual(inputs["dry_run"]["default"], "true")
        self.assertEqual(inputs["jira_id"]["required"], "false")

    def test_workflow_routes_each_repository_to_only_its_app(self) -> None:
        token_steps = {
            step["id"]: step
            for step in self.steps
            if step.get("uses", "").startswith("actions/create-github-app-token@")
        }
        app_routes = {
            "therock-token": ("TheRock", "ROCM_SYSTEMS"),
            "systems-token": ("rocm-systems", "ROCM_SYSTEMS"),
            "libraries-token": ("rocm-libraries", "ROCM_LIBRARIES"),
        }
        self.assertEqual(set(token_steps), set(app_routes))
        for step_id, (repository, secret_prefix) in app_routes.items():
            with self.subTest(repository=repository):
                step = token_steps[step_id]
                self.assertEqual(step["if"], f"inputs.repository == '{repository}'")
                settings = step["with"]
                self.assertEqual(settings["repositories"], repository)
                self.assertEqual(settings["owner"], "ROCm")
                self.assertEqual(
                    settings["app-id"], "${{ secrets." + secret_prefix + "_APP_ID }}"
                )
                self.assertEqual(
                    settings["private-key"],
                    "${{ secrets." + secret_prefix + "_APP_PRIVATE_KEY }}",
                )
                for permission in ("contents", "pull-requests", "workflows"):
                    self.assertEqual(settings[f"permission-{permission}"], "write")
                self.assertNotIn("permission-actions", settings)
                self.assertIn(
                    f"steps.{step_id}.outputs.token",
                    self.cherry_pick_step["env"]["GITHUB_APP_TOKEN"],
                )
                self.assertIn(
                    f"steps.{step_id}.outputs.app-slug",
                    self.cherry_pick_step["env"]["GITHUB_APP_SLUG"],
                )

    def test_workflow_passes_inputs_through_environment(self) -> None:
        environment = self.cherry_pick_step["env"]
        for input_name in (
            "repository",
            "commits",
            "release_branch",
            "jira_id",
            "dry_run",
        ):
            with self.subTest(input_name=input_name):
                self.assertEqual(
                    environment[f"CHERRY_PICK_{input_name.upper()}"],
                    "${{ inputs." + input_name + " }}",
                )
        self.assertEqual(environment["WORKFLOW_ACTOR"], "${{ github.actor }}")
        self.assertEqual(
            environment["WORKFLOW_RUN_URL"],
            "${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}",
        )
        for step in self.steps:
            if "run" in step:
                self.assertNotIn("${{ inputs.", step["run"])

    def test_workflow_uses_pinned_actions_and_read_only_default_permissions(
        self,
    ) -> None:
        self.assertEqual(self.workflow["permissions"], {"contents": "read"})
        for step in self.steps:
            if "uses" in step:
                action_ref = step["uses"].rsplit("@", maxsplit=1)[1]
                self.assertRegex(action_ref, r"^[0-9a-f]{40}$")

    def test_checkout_does_not_persist_credentials_or_fetch_submodules(self) -> None:
        checkout = next(
            step
            for step in self.steps
            if step.get("uses", "").startswith("actions/checkout@")
        )
        self.assertEqual(checkout["with"]["persist-credentials"], "false")
        self.assertEqual(checkout["with"]["submodules"], "false")


class LocalTargetRepository:
    """Temporary source and filter-capable bare remote for integration tests."""

    def __init__(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name)
        self.source = self.root / "source"
        self.remote = self.root / "target.git"
        self.source.mkdir()
        _git("init", "-b", "develop", cwd=self.source)
        _git("config", "user.name", "Fixture Committer", cwd=self.source)
        _git("config", "user.email", "fixture@example.com", cwd=self.source)
        (self.source / "common.txt").write_text("common\n", encoding="utf-8")
        (self.source / "delete-me.txt").write_text("delete me\n", encoding="utf-8")
        nested = self.source / "nested"
        nested.mkdir()
        (nested / "union.txt").write_text("base\n", encoding="utf-8")
        (self.source / "binary-like.txt").write_text(
            "base first\nbase middle\nbase last\n", encoding="utf-8"
        )
        (self.source / "attribute-chain.txt").write_text(
            "base first\nbase middle\nbase last\n", encoding="utf-8"
        )
        _git("add", ".", cwd=self.source)
        _git("commit", "-m", "Initial files", cwd=self.source)
        self.base = self.rev_parse("HEAD")
        _git("branch", "release/therock-10.2", self.base, cwd=self.source)
        _git(
            "branch",
            "release/bkc/therock-10.2-20261013",
            self.base,
            cwd=self.source,
        )
        self.first = self.commit(
            "feature-one.txt",
            "first fix\n",
            "First release fix\n\n# Keep this heading\n",
        )
        _git("init", "--bare", self.remote.as_posix())
        _git(
            "--git-dir",
            self.remote.as_posix(),
            "config",
            "uploadpack.allowFilter",
            "true",
        )
        _git(
            "--git-dir",
            self.remote.as_posix(),
            "config",
            "uploadpack.allowAnySHA1InWant",
            "true",
        )
        _git("remote", "add", "origin", self.remote.as_uri(), cwd=self.source)
        self.push_all()
        _git(
            "--git-dir",
            self.remote.as_posix(),
            "symbolic-ref",
            "HEAD",
            "refs/heads/develop",
        )

    def cleanup(self) -> None:
        self._temporary_directory.cleanup()

    def rev_parse(self, revision: str) -> str:
        return _git("rev-parse", revision, cwd=self.source).stdout.strip()

    def commit(self, relative_path: str, content: str, message: str) -> str:
        path = self.source / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        _git("add", relative_path, cwd=self.source)
        _git(
            "commit",
            "-F",
            "-",
            cwd=self.source,
            input_text=message,
            env=FIXTURE_COMMIT_ENV,
        )
        return self.rev_parse("HEAD")

    def commit_with_exact_message(
        self, relative_path: str, content: str, message: str
    ) -> str:
        path = self.source / relative_path
        path.write_text(content, encoding="utf-8")
        _git("add", relative_path, cwd=self.source)
        tree = _git("write-tree", cwd=self.source).stdout.strip()
        parent = self.rev_parse("HEAD")
        commit = _git(
            "commit-tree",
            tree,
            "-p",
            parent,
            "-F",
            "-",
            cwd=self.source,
            input_text=message,
            env=FIXTURE_COMMIT_ENV,
        ).stdout.strip()
        _git("update-ref", "refs/heads/develop", commit, cwd=self.source)
        return commit

    def switch(self, branch: str) -> None:
        _git("switch", branch, cwd=self.source)

    def push_all(self) -> None:
        _git("push", "origin", "--all", cwd=self.source)
        _git("push", "origin", "--tags", cwd=self.source)

    def comparison_clone(self, name: str) -> Path:
        clone = self.root / name
        _git("clone", self.remote.as_uri(), clone.as_posix())
        _git("config", "user.name", "Comparison Committer", cwd=clone)
        _git("config", "user.email", "comparison@example.com", cwd=clone)
        _git("switch", "release/therock-10.2", cwd=clone)
        return clone

    def remote_ref(self, branch: str) -> str | None:
        result = _git(
            "--git-dir",
            self.remote.as_posix(),
            "rev-parse",
            "--verify",
            "--quiet",
            f"refs/heads/{branch}",
            check=False,
        )
        if result.returncode == 1:
            return None
        result.check_returncode()
        return result.stdout.strip()

    def remote_git(self, *args: str) -> str:
        return _git("--git-dir", self.remote.as_posix(), *args).stdout


class CherryPickIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        # Production subprocesses must use the same isolation as fixture commands.
        environment = patch.dict(os.environ, _isolated_git_environment(), clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.target = LocalTargetRepository()
        self.addCleanup(self.target.cleanup)
        self.api = FakeGitHubAPI()
        self.summaries: list[str] = []

    def execute(
        self,
        commits: str | None = None,
        *,
        release_branch: str = "release/therock-10.2",
        jira_id: str = "AIRDEL-1234",
        dry_run: bool = True,
        api: FakeGitHubAPI | None = None,
    ) -> RunResult:
        return execute_cherry_pick(
            repository_value="rocm-systems",
            commits_value=self.target.first if commits is None else commits,
            release_branch_value=release_branch,
            jira_id_value=jira_id,
            dry_run=dry_run,
            app_token=TOKEN,
            app_slug="systems-assistant",
            workflow_run_url=RUN_URL,
            workflow_actor="release-operator",
            api=self.api if api is None else api,
            summary_writer=self.summaries.append,
            clone_url=self.target.remote.as_uri(),
            push_url=self.target.remote.as_uri(),
        )

    def test_dry_run_builds_commit_and_reports_pr_without_writing(self) -> None:
        result = self.execute()
        self.assertTrue(result.dry_run)
        self.assertEqual(
            result.pr_branch, "shared/cherry-pick/therock-10.2-airdel-1234"
        )
        self.assertEqual(len(result.built_commits), 1)
        self.assertIsNone(result.pr_url)
        self.assertIsNone(self.target.remote_ref(result.pr_branch))
        self.assertEqual(self.api.pull_requests, [])
        summary = self.summaries[0]
        self.assertIn("Dry run", summary)
        self.assertIn("No branch was pushed", summary)
        self.assertIn(result.built_commits[0].sha, summary)
        self.assertIn(result.pr_title, summary)
        self.assertIn(result.pr_body, summary)

    def test_live_run_pushes_unsigned_commit_and_creates_pr(self) -> None:
        result = self.execute(dry_run=False)
        self.assertFalse(result.dry_run)
        self.assertEqual(result.pr_url, "https://github.com/ROCm/rocm-systems/pull/99")
        self.assertEqual(
            self.target.remote_ref(result.pr_branch), result.built_commits[0].sha
        )
        metadata = self.target.remote_git(
            "log",
            "-1",
            "--format=%an%x00%ae%x00%aI%x00%cn%x00%ce%x00%G?",
            f"refs/heads/{result.pr_branch}",
        ).removesuffix("\n")
        self.assertEqual(
            metadata.split("\0"),
            [
                "Original Author",
                "original@example.com",
                "2026-10-25T10:20:30+00:00",
                "systems-assistant[bot]",
                "systems-assistant@example.com",
                "N",
            ],
        )
        request = self.api.pull_requests[0]
        self.assertEqual(request.body["base"], "release/therock-10.2")
        self.assertEqual(request.body["head"], result.pr_branch)
        self.assertEqual(request.body["title"], result.pr_title)
        self.assertEqual(request.body["body"], result.pr_body)
        self.assertNotIn("labels", request.body)
        self.assertIn(RUN_URL, request.body["body"])
        self.assertIn(result.pr_url, self.summaries[0])

    def test_multiple_commits_follow_input_order_and_dedupe_mixed_shas(self) -> None:
        second = self.target.commit(
            "feature-two.txt", "second fix\n", "Second release fix\n"
        )
        self.target.push_all()
        result = self.execute(
            f"{second[:8]}, {second} {self.target.first}", dry_run=False
        )
        self.assertEqual(
            [item.source.sha for item in result.built_commits],
            [second, self.target.first],
        )
        messages = self.target.remote_git(
            "log",
            "-2",
            "--format=%B%x00",
            f"refs/heads/{result.pr_branch}",
        )
        newest, older, _ = messages.split("\0")
        self.assertIn(f"Cherry-picks commit {self.target.first}", newest)
        self.assertIn(f"Cherry-picks commit {second}", older)

    def test_original_hash_lines_and_trailing_blank_lines_are_preserved(self) -> None:
        original_message = "Exact subject\n\n# Keep this line\n\n\n"
        exact_commit = self.target.commit_with_exact_message(
            "exact-message.txt", "content\n", original_message
        )
        source_object = _git(
            "cat-file", "commit", exact_commit, cwd=self.target.source
        ).stdout
        self.assertEqual(source_object.partition("\n\n")[2], original_message)
        self.target.push_all()

        result = self.execute(exact_commit, dry_run=False)
        generated_object = self.target.remote_git(
            "cat-file", "commit", result.built_commits[0].sha
        )
        generated_message = generated_object.partition("\n\n")[2]
        self.assertEqual(
            generated_message,
            f"Cherry-picks commit {exact_commit} into the ROCm 10.2 branch.\n\n"
            "JIRA ID: AIRDEL-1234\n\n"
            f"{original_message}",
        )

    def test_ticketless_bkc_dry_run_uses_sha_branch_and_omits_jira(self) -> None:
        result = self.execute(
            release_branch="release/bkc/therock-10.2-20261013",
            jira_id="",
        )
        self.assertEqual(
            result.pr_branch,
            f"shared/cherry-pick/bkc-therock-10.2-20261013-{self.target.first[:7]}",
        )
        self.assertNotIn("JIRA ID", result.pr_body)
        self.assertNotIn("JIRA ID", self.summaries[0])

    def test_conflict_fails_without_push_and_reports_path(self) -> None:
        self.target.switch("release/therock-10.2")
        self.target.commit(
            "feature-one.txt", "release version\n", "Divergent release change\n"
        )
        self.target.push_all()
        with self.assertRaisesRegex(CherryPickError, "conflicts"):
            self.execute(dry_run=False)
        self.assertIsNone(
            self.target.remote_ref("shared/cherry-pick/therock-10.2-airdel-1234")
        )
        self.assertEqual(self.api.pull_requests, [])
        self.assertIn("feature-one.txt", self.summaries[0])
        self.assertIn("ROCm/rocm-systems", self.summaries[0])
        self.assertIn("release/therock-10.2", self.summaries[0])
        self.assertIn("Nothing was pushed", self.summaries[0])

    def test_nested_union_attribute_matches_ordinary_cherry_pick(self) -> None:
        self.target.switch("release/therock-10.2")
        (self.target.source / "nested/.gitattributes").write_text(
            "union.txt merge=union\n", encoding="utf-8"
        )
        (self.target.source / "nested/union.txt").write_text(
            "release\n", encoding="utf-8"
        )
        _git("add", "nested", cwd=self.target.source)
        _git("commit", "-m", "Use union merge for nested file", cwd=self.target.source)

        self.target.switch("develop")
        source_commit = self.target.commit(
            "nested/union.txt", "source\n", "Update nested union file\n"
        )
        self.target.push_all()

        result = self.execute(source_commit, dry_run=False)
        generated_tree = self.target.remote_git(
            "rev-parse", f"refs/heads/{result.pr_branch}^{{tree}}"
        ).strip()

        normal_clone = self.target.comparison_clone("union-normal-clone")
        _git("cherry-pick", source_commit, cwd=normal_clone)
        cherry_pick_tree = _git(
            "rev-parse", "HEAD^{tree}", cwd=normal_clone
        ).stdout.strip()
        self.assertEqual(generated_tree, cherry_pick_tree)

    def test_release_binary_attribute_conflicts_without_push(self) -> None:
        self.target.switch("release/therock-10.2")
        (self.target.source / ".gitattributes").write_text(
            "binary-like.txt binary\n", encoding="utf-8"
        )
        (self.target.source / "binary-like.txt").write_text(
            "release first\nbase middle\nbase last\n", encoding="utf-8"
        )
        _git("add", ".gitattributes", "binary-like.txt", cwd=self.target.source)
        _git("commit", "-m", "Treat release data as binary", cwd=self.target.source)

        self.target.switch("develop")
        source_commit = self.target.commit(
            "binary-like.txt",
            "base first\nbase middle\nsource last\n",
            "Update binary-marked data\n",
        )
        self.target.push_all()

        normal_clone = self.target.comparison_clone("binary-normal-clone")
        ordinary_pick = _git(
            "cherry-pick", source_commit, cwd=normal_clone, check=False
        )
        self.assertNotEqual(ordinary_pick.returncode, 0)

        with self.assertRaisesRegex(CherryPickError, "conflicts"):
            self.execute(source_commit, dry_run=False)
        self.assertIsNone(
            self.target.remote_ref("shared/cherry-pick/therock-10.2-airdel-1234")
        )
        self.assertEqual(self.api.pull_requests, [])
        self.assertIn("binary-like.txt", self.summaries[0])
        self.assertNotIn("warning:", self.summaries[0])
        self.assertNotIn("CONFLICT", self.summaries[0])
        self.assertNotIn("Auto-merging", self.summaries[0])

    def test_later_pick_uses_attributes_added_by_earlier_pick(self) -> None:
        self.target.switch("release/therock-10.2")
        (self.target.source / "attribute-chain.txt").write_text(
            "release first\nbase middle\nbase last\n", encoding="utf-8"
        )
        _git("add", "attribute-chain.txt", cwd=self.target.source)
        _git("commit", "-m", "Change release attribute fixture", cwd=self.target.source)

        self.target.switch("develop")
        attribute_path = self.target.source / ".gitattributes"
        attribute_path.write_text("attribute-chain.txt binary\n", encoding="utf-8")
        _git("add", ".gitattributes", cwd=self.target.source)
        _git("commit", "-m", "Mark chained data as binary", cwd=self.target.source)
        attribute_commit = self.target.rev_parse("HEAD")

        (self.target.source / "attribute-chain.txt").write_text(
            "base first\nbase middle\nsource last\n", encoding="utf-8"
        )
        attribute_path.unlink()
        _git("add", "-A", cwd=self.target.source)
        _git("commit", "-m", "Update chained data", cwd=self.target.source)
        content_commit = self.target.rev_parse("HEAD")
        self.target.push_all()

        normal_clone = self.target.comparison_clone("attribute-chain-normal-clone")
        _git("cherry-pick", attribute_commit, cwd=normal_clone)
        ordinary_pick = _git(
            "cherry-pick", content_commit, cwd=normal_clone, check=False
        )
        self.assertNotEqual(ordinary_pick.returncode, 0)

        with self.assertRaisesRegex(CherryPickError, "conflicts"):
            self.execute(
                f"{attribute_commit} {content_commit}",
                dry_run=False,
            )
        self.assertIsNone(
            self.target.remote_ref("shared/cherry-pick/therock-10.2-airdel-1234")
        )
        self.assertEqual(self.api.pull_requests, [])
        self.assertIn("attribute-chain.txt", self.summaries[0])

    def test_target_default_branch_comes_from_api_not_therock_metadata(self) -> None:
        _git("branch", "-m", "develop", "target-mainline", cwd=self.target.source)
        _git(
            "--git-dir",
            self.target.remote.as_posix(),
            "symbolic-ref",
            "HEAD",
            "refs/heads/target-mainline",
        )
        _git("push", "origin", "target-mainline", cwd=self.target.source)
        _git("push", "origin", "--delete", "develop", cwd=self.target.source)
        api = FakeGitHubAPI(default_branch="target-mainline")
        result = self.execute(api=api)
        self.assertEqual(result.built_commits[0].source.sha, self.target.first)

    def test_empty_pick_fails_without_push(self) -> None:
        self.target.switch("release/therock-10.2")
        _git("cherry-pick", self.target.first, cwd=self.target.source)
        self.target.push_all()
        with self.assertRaisesRegex(CherryPickError, "is empty"):
            self.execute(dry_run=False)
        self.assertIsNone(
            self.target.remote_ref("shared/cherry-pick/therock-10.2-airdel-1234")
        )
        self.assertEqual(self.api.pull_requests, [])

    def test_existing_pr_branch_is_preserved_without_opening_pr(self) -> None:
        _git(
            "branch",
            "shared/cherry-pick/therock-10.2-airdel-1234",
            self.target.base,
            cwd=self.target.source,
        )
        self.target.push_all()
        with self.assertRaisesRegex(CherryPickError, "already exists"):
            self.execute(dry_run=False)
        self.assertEqual(
            self.target.remote_ref("shared/cherry-pick/therock-10.2-airdel-1234"),
            self.target.base,
        )
        self.assertEqual(self.api.pull_requests, [])

    def _assert_concurrent_branch_creation_is_rejected(
        self, *, identical_head: bool
    ) -> None:
        pr_branch = "shared/cherry-pick/therock-10.2-airdel-1234"
        concurrent_head: str | None = None

        def push_after_concurrent_creation(
            repo: Path,
            repository: str,
            token: str,
            head: str,
            pr_branch: str,
            push_url: str | None = None,
        ) -> None:
            nonlocal concurrent_head
            concurrent_head = head if identical_head else self.target.base
            _git(
                "push",
                self.target.remote.as_uri(),
                f"{concurrent_head}:refs/heads/{pr_branch}",
                cwd=repo,
            )
            push_branch(repo, repository, token, head, pr_branch, push_url=push_url)

        expected_error = "already exists" if identical_head else "Failed to push"
        with patch(
            "cherry_pick_to_release.push_branch",
            side_effect=push_after_concurrent_creation,
        ), self.assertRaisesRegex(CherryPickError, expected_error):
            self.execute(dry_run=False)
        self.assertEqual(self.target.remote_ref(pr_branch), concurrent_head)
        self.assertEqual(self.api.pull_requests, [])

    def test_concurrent_branch_at_ancestor_is_not_updated(self) -> None:
        self._assert_concurrent_branch_creation_is_rejected(identical_head=False)

    def test_concurrent_branch_at_identical_head_does_not_open_pr(self) -> None:
        self._assert_concurrent_branch_creation_is_rejected(identical_head=True)

    def test_commit_only_on_feature_branch_is_rejected(self) -> None:
        _git(
            "switch",
            "-c",
            "feature/not-develop",
            self.target.base,
            cwd=self.target.source,
        )
        feature_commit = self.target.commit(
            "feature-only.txt", "feature\n", "Feature-only commit\n"
        )
        self.target.push_all()
        with self.assertRaisesRegex(CherryPickError, "not reachable.*origin/develop"):
            self.execute(feature_commit)
        self.assertEqual(self.api.pull_requests, [])

    def test_commit_only_reachable_from_tag_is_rejected(self) -> None:
        _git("switch", "--detach", self.target.base, cwd=self.target.source)
        tag_commit = self.target.commit("tag-only.txt", "tag\n", "Tag-only commit\n")
        _git("tag", "tag-only", tag_commit, cwd=self.target.source)
        self.target.push_all()
        with self.assertRaisesRegex(CherryPickError, "not reachable.*origin/develop"):
            self.execute(tag_commit)

    def test_one_invalid_commit_rejects_whole_list_before_build(self) -> None:
        _git(
            "switch",
            "-c",
            "release/other-line",
            self.target.base,
            cwd=self.target.source,
        )
        other_release = self.target.commit(
            "other-release.txt", "other\n", "Other release only\n"
        )
        self.target.push_all()
        with patch(
            "cherry_pick_to_release.build_cherry_pick_commits"
        ) as build_commits, self.assertRaisesRegex(CherryPickError, "not reachable"):
            self.execute(f"{self.target.first} {other_release}")
        build_commits.assert_not_called()

    def test_merge_commit_is_rejected(self) -> None:
        _git("switch", "develop", cwd=self.target.source)
        _git("switch", "-c", "merge-side", self.target.base, cwd=self.target.source)
        self.target.commit("merge-side.txt", "side\n", "Side commit\n")
        _git("switch", "develop", cwd=self.target.source)
        _git(
            "merge", "--no-ff", "merge-side", "-m", "Merge side", cwd=self.target.source
        )
        merge_commit = self.target.rev_parse("HEAD")
        self.target.push_all()
        with self.assertRaisesRegex(CherryPickError, "has 2 parents"):
            self.execute(merge_commit)

    def test_root_commit_is_rejected(self) -> None:
        with self.assertRaisesRegex(CherryPickError, "has 0 parents"):
            self.execute(self.target.base)

    def test_commit_already_on_release_branch_is_rejected(self) -> None:
        _git(
            "branch",
            "-f",
            "release/therock-10.2",
            self.target.first,
            cwd=self.target.source,
        )
        self.target.push_all()
        with self.assertRaisesRegex(CherryPickError, "already an ancestor"):
            self.execute()

    def test_prior_automation_cherry_pick_is_rejected(self) -> None:
        self.target.switch("release/therock-10.2")
        _git(
            "commit",
            "--allow-empty",
            "-m",
            f"Cherry-picks commit {self.target.first} into the ROCm 10.2 branch.",
            cwd=self.target.source,
        )
        self.target.push_all()
        with self.assertRaisesRegex(CherryPickError, "already cherry-picked"):
            self.execute()

    def test_pr_failure_leaves_pushed_branch_and_reports_it(self) -> None:
        failing_api = FakeGitHubAPI(pr_error=GitHubAPIError("API unavailable"))
        with self.assertRaisesRegex(
            CherryPickError, "Branch.*was pushed.*pull request creation failed"
        ):
            self.execute(dry_run=False, api=failing_api)
        self.assertIsNotNone(
            self.target.remote_ref("shared/cherry-pick/therock-10.2-airdel-1234")
        )

    def test_same_ticket_can_target_multiple_releases(self) -> None:
        _git("branch", "release/therock-10.3", self.target.base, cwd=self.target.source)
        self.target.push_all()
        targets = {
            "release/therock-10.2": "shared/cherry-pick/therock-10.2-airdel-1234",
            "release/therock-10.3": "shared/cherry-pick/therock-10.3-airdel-1234",
            "release/bkc/therock-10.2-20261013": (
                "shared/cherry-pick/bkc-therock-10.2-20261013-airdel-1234"
            ),
        }
        for release_branch, pr_branch in targets.items():
            with self.subTest(release_branch=release_branch):
                result = self.execute(release_branch=release_branch, dry_run=False)
                self.assertEqual(result.pr_branch, pr_branch)
                self.assertEqual(
                    self.target.remote_ref(pr_branch), result.built_commits[0].sha
                )
                self.assertEqual(
                    self.api.pull_requests[-1].body["base"], release_branch
                )
                self.assertEqual(self.api.pull_requests[-1].body["head"], pr_branch)
        self.assertEqual(len(self.api.pull_requests), len(targets))

    def test_unexpected_pr_failure_preserves_exception_and_recovery_context(
        self,
    ) -> None:
        error = RuntimeError("Unexpected implementation failure")
        failing_api = FakeGitHubAPI(pr_error=error)
        with self.assertRaises(RuntimeError) as raised:
            self.execute(dry_run=False, api=failing_api)
        self.assertIs(raised.exception, error)
        branch = "shared/cherry-pick/therock-10.2-airdel-1234"
        self.assertIsNotNone(self.target.remote_ref(branch))
        notes = "\n".join(error.__notes__)
        self.assertIn(branch, notes)
        self.assertIn("was pushed", notes)
        self.assertIn("ROCm/rocm-systems", notes)

    def test_merge_tree_matches_git_cherry_pick_for_rename_and_delete(self) -> None:
        self.target.switch("develop")
        (self.target.source / "common.txt").rename(self.target.source / "renamed.txt")
        (self.target.source / "delete-me.txt").unlink()
        _git("add", "-A", cwd=self.target.source)
        _git("commit", "-m", "Rename and delete files", cwd=self.target.source)
        source_commit = self.target.rev_parse("HEAD")
        self.target.push_all()

        result = self.execute(source_commit, dry_run=False)
        generated_tree = self.target.remote_git(
            "rev-parse", f"refs/heads/{result.pr_branch}^{{tree}}"
        ).strip()

        normal_clone = self.target.comparison_clone("normal-clone")
        _git("cherry-pick", source_commit, cwd=normal_clone)
        cherry_pick_tree = _git(
            "rev-parse", "HEAD^{tree}", cwd=normal_clone
        ).stdout.strip()
        self.assertEqual(generated_tree, cherry_pick_tree)

    def test_partial_clone_configuration_and_no_checkout(self) -> None:
        clone_path = self.target.root / "inspection-clone"
        clone_target("rocm-systems", clone_path, clone_url=self.target.remote.as_uri())
        self.assertEqual(
            _git(
                "config", "--get", "remote.origin.promisor", cwd=clone_path
            ).stdout.strip(),
            "true",
        )
        self.assertEqual(
            _git(
                "config", "--get", "remote.origin.partialclonefilter", cwd=clone_path
            ).stdout.strip(),
            "tree:0",
        )
        self.assertEqual({path.name for path in clone_path.iterdir()}, {".git"})


class FailureSanitizationTest(unittest.TestCase):
    def test_push_requires_creation_confirmation_for_exact_refspec(self) -> None:
        head = "a" * 40
        branch = "shared/cherry-pick/therock-10.2-airdel-12"
        refspec = f"{head}:refs/heads/{branch}"
        for status_output in (
            "",
            f"*\t{head}:refs/heads/another-branch\t[new branch]\n",
            f"*\t{refspec}\n",
            f"*\t{refspec}\t[new branch]\n*\t{refspec}\t[new branch]\n",
            f"=\t{refspec}\t[up to date]\n",
        ):
            result = subprocess.CompletedProcess(
                args=["git", "push"],
                returncode=0,
                stdout=f"To https://x-access-token:{TOKEN}@github.com/ROCm/rocm-systems.git\n"
                f"{status_output}Done\n",
                stderr=f"diagnostic containing {TOKEN}",
            )
            with self.subTest(status_output=status_output):
                with patch(
                    "cherry_pick_to_release.subprocess.run", return_value=result
                ), self.assertRaises(CherryPickError) as raised:
                    push_branch(
                        Path("/temporary/repo"), "rocm-systems", TOKEN, head, branch
                    )
                self.assertNotIn(TOKEN, str(raised.exception))
                self.assertNotIn("x-access-token", str(raised.exception))
                self.assertIn(branch, str(raised.exception))

    def test_push_failure_does_not_include_credentials(self) -> None:
        result = subprocess.CompletedProcess(
            args=["git", "push"],
            returncode=1,
            stdout="",
            stderr=f"fatal: URL included {TOKEN}",
        )
        with patch("cherry_pick_to_release.subprocess.run", return_value=result):
            with self.assertRaises(CherryPickError) as raised:
                push_branch(
                    Path("/temporary/repo"),
                    "rocm-systems",
                    TOKEN,
                    "a" * 40,
                    "shared/cherry-pick/therock-10.2-airdel-12",
                )
        self.assertNotIn(TOKEN, str(raised.exception))
        self.assertNotIn("x-access-token", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
