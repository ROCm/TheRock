# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Parse the external source configuration passed between TheRock CI jobs.

Data flow:
    1. A repository calling TheRock's reusable workflows supplies an external_repo
       JSON input, for example:

           {
               "repository": "ROCm/rocm-libraries",
               "ref": "0123456789abcdef0123456789abcdef01234567",
               "extra_cmake_options": "-DTHEROCK_FLAG_HIPKERNELPROVIDER_ENABLE_ROCKE=ON"
           }

       ref is the commit selected for the run, including the merge SHA for PRs.

    2. Early in CI setup, github_actions/detect_external_repo_config.py reads
       that input and resolves repository to its TheRock submodule path. Its
       config_json output adds submodule_path and preserves repository, ref,
       and caller build options, adding defaults for omitted options:

           {
               "repository": "ROCm/rocm-libraries",
               "submodule_path": "rocm-libraries",
               "ref": "0123456789abcdef0123456789abcdef01234567",
               "extra_cmake_options": "-DTHEROCK_FLAG_HIPKERNELPROVIDER_ENABLE_ROCKE=ON",
               "projects": "",
               "family_overrides": {}
           }

    3. Downstream workflows pass the resolved JSON as EXTERNAL_REPO_CONFIG.
       Scripts use this module to parse it and access the resolved submodule
       path, ref, and other configuration fields needed by each consumer.

This module parses both shapes:

    parse_external_repo() handles the caller-supplied input from step 1,
    returning an ExternalRepoInput with the diff coordinates (event_name,
    base_sha/head_sha, pr_number) CI needs to attribute a change.

    parse_and_validate_external_repo_config() handles the resolved JSON from
    step 2, validating the shared submodule_path and ref fields and preserving
    repository and additional options without interpreting them.

Repository-to-path resolution happens during CI setup; consumers decide how to
use the parsed configuration.

Separately, load_repos_config()/subtree_prefixes() parse the calling repo's own
repos-config.json, the registry of subtrees it syncs from upstream repositories.
Those prefixes are what a diff is attributed to when scoping a run to projects;
see configure_ci_path_filters.determine_changed_projects().
"""

import json
import logging
import typing
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class ExternalRepoInput:
    """Schema for the step-1 external_repo JSON blob passed by calling repos.

    Distinct from the step-2 resolved config, which carries submodule_path and
    is modeled by detect_external_repo_config.ExternalRepoConfig.

    Fields:
        repository: "owner/name" of the calling repo (e.g. "ROCm/rocm-libraries").
        ref: Ref of the calling repo to check out.
        event_name: Caller's github.event_name; "schedule"/"workflow_dispatch" have
            no diff.
        base_sha: Base commit SHA for computing the diff.
        head_sha: Head commit SHA for computing the diff.
        pr_number: Pull request number in the calling repo.
        base_ref: Fallback base for the local git-diff path.
        skip_ci_config: Path to the caller's skip-CI TOML.
        family_overrides: Per-family overrides; consumed by setup_multi_arch.yml.
    """

    repository: str
    ref: str | None = None
    event_name: str = ""
    base_sha: str | None = None
    head_sha: str | None = None
    pr_number: int | None = None
    base_ref: str | None = None
    skip_ci_config: str | None = None
    family_overrides: dict | None = None

    # Expected types for each field, used by __post_init__ for validation.
    _FIELD_TYPES: typing.ClassVar[dict[str, tuple[type, ...]]] = {
        "repository": (str,),
        "ref": (str,),
        "event_name": (str,),
        "base_sha": (str,),
        "head_sha": (str,),
        "pr_number": (int,),
        "base_ref": (str,),
        "skip_ci_config": (str,),
        "family_overrides": (dict,),
    }

    def __post_init__(self):
        if not self.repository or "/" not in self.repository:
            raise ValueError(
                f"external_repo.repository must be 'owner/name', "
                f"got {self.repository!r}"
            )
        for f in fields(self):
            if f.name.startswith("_"):
                continue
            value = getattr(self, f.name)
            if value is None:
                continue
            expected = self._FIELD_TYPES.get(f.name)
            if expected and not isinstance(value, expected):
                raise ValueError(
                    f"external_repo.{f.name}: expected "
                    f"{'/'.join(t.__name__ for t in expected)}, "
                    f"got {type(value).__name__}: {value!r}"
                )

    @property
    def repo_name(self) -> str:
        """Short repo name (the part after the slash)."""
        return self.repository.split("/")[-1]


def parse_external_repo(raw: str) -> ExternalRepoInput:
    """Parse and validate the external_repo JSON blob.

    Raises ValueError for malformed JSON, non-object payloads, missing or
    invalid ``repository``, or fields with wrong types.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"Invalid external_repo JSON: {raw!r}") from e

    if not isinstance(data, dict):
        raise ValueError(
            f"external_repo must be a JSON object, got: {type(data).__name__}"
        )

    # Coerce pr_number from string to int (callers may interpolate via ${{ }}).
    pr_number = data.get("pr_number")
    if isinstance(pr_number, str):
        try:
            data["pr_number"] = int(pr_number)
        except ValueError:
            raise ValueError(
                f"external_repo.pr_number: cannot convert {pr_number!r} to int"
            )

    known_fields = {
        f.name for f in fields(ExternalRepoInput) if not f.name.startswith("_")
    }
    filtered = {k: v for k, v in data.items() if k in known_fields}
    unknown = set(data.keys()) - known_fields
    if unknown:
        logging.getLogger(__name__).debug(
            "Ignoring unknown external_repo fields: %s", sorted(unknown)
        )

    try:
        return ExternalRepoInput(**filtered)
    except TypeError as e:
        raise ValueError(f"Invalid external_repo payload: {e}") from e


def parse_and_validate_external_repo_config(raw: str | None) -> dict | None:
    """Decode and validate a serialized EXTERNAL_REPO_CONFIG value.

    ``raw`` is JSON text, typically supplied through the environment. None or an
    empty string returns None; otherwise require an object with nonblank string
    submodule_path/ref fields and preserve additional options.

    Malformed JSON raises json.JSONDecodeError; invalid fields raise ValueError.
    Validation does not establish a full commit SHA, matching submodule path, or
    remote reachability.
    """
    if not raw:
        return None
    config = json.loads(raw)
    if not isinstance(config, dict):
        raise ValueError("EXTERNAL_REPO_CONFIG must be a JSON object")
    for field in ("submodule_path", "ref"):
        value = config.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"EXTERNAL_REPO_CONFIG '{field}' must be a non-empty string"
            )
    return config


# Path of the subtree registry within a calling repository.
REPOS_CONFIG_PATH = ".github/repos-config.json"


@dataclass
class RepoEntry:
    """One upstream repository subtree-synced into a calling monorepo."""

    name: str
    url: str
    branch: str
    category: str
    auto_subtree_pull: bool = False
    auto_subtree_push: bool = False
    monorepo_source_of_truth: bool = False


def load_repos_config(config_path: str | Path) -> list[RepoEntry]:
    """Parse a calling repo's repos-config.json into RepoEntry records.

    Returns an empty list when the file is missing or unreadable, which callers
    must treat as "no subtrees known" rather than "no subtrees changed".
    """
    logger = logging.getLogger(__name__)
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.warning("repos-config.json not found: %s", config_path)
        return []
    except (OSError, json.JSONDecodeError) as e:
        logger.error("Failed to load %s: %s", config_path, e)
        return []

    # Unknown keys are dropped so a newer config schema stays loadable here.
    known_fields = {f.name for f in fields(RepoEntry)}
    return [
        RepoEntry(**{k: v for k, v in entry.items() if k in known_fields})
        for entry in data.get("repositories", [])
    ]


def subtree_prefixes(entries: list[RepoEntry]) -> set[str]:
    """Directory prefixes the configured subtrees occupy, e.g. projects/rocblas."""
    return {f"{entry.category}/{entry.name}" for entry in entries}
