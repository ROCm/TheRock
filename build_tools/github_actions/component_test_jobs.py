# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Component-owned test job layouts ("jobs manifests").

A test_matrix entry in fetch_test_configurations.py may declare ``jobs_manifest``
to split its single test job into several jobs. The layout comes from a TOML file
in the component's own source tree, read at the source revision that produced the
artifacts under test, so the jobs always match the tests in the artifact:

    "jobs_manifest": {
        "submodule": "rocm-libraries",
        "path": "dnn-providers/hip-kernel-provider/ci_jobs.toml",
    }

Each group selects its tests by CTest label. One extra "remainder" job runs every
test that no group claims, so a test missing from the manifest still runs.

Manifest schema (version 1)::

    schema = 1

    [defaults]                       # optional
    shards = 1                       # <tier value>
    timeout_minutes = 30             # <tier value>

    [remainder]                      # optional
    name = "unit"                    # job suffix, default "unit"
    shards = 1                       # <tier value>
    timeout_minutes = 45             # <tier value>

    [[group]]
    name = "asm-sdpa"                # job: <component job name>-asm-sdpa
    labels = ["^ASM_SDPA_ENGINE$"]   # a test belongs to the group if any of its
                                     # CTest labels matches one of these regexes
    shards = { standard = 2 }        # <tier value>, optional
    timeout_minutes = 30             # <tier value>, optional
    allow_empty = false              # true: matching no tests is not an error

    [[group.exclude_family]]         # optional, repeatable
    platform = "windows"
    families = ["gfx110X-all"]       # family group strings or gfx targets
    reason = "ships no kernels for gfx11"

A <tier value> is an integer, or a table keyed by tier (quick, standard,
comprehensive, full) whose values are an integer or a table keyed by platform
(linux, windows). Unset tiers and platforms fall back to [defaults], then to
1 shard / 30 minutes.

The label filters reach test_runner.py as environment assignments prepended to
the component's test_script (see TEST_LABEL_* in test_runner.py).
"""

import json
import logging
import os
import re
import shlex
import sys
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

SCHEMA_VERSION = 1
TIERS = ("quick", "standard", "comprehensive", "full")
PLATFORMS = ("linux", "windows")
DEFAULT_SHARDS = 1
DEFAULT_TIMEOUT_MINUTES = 30
MAX_SHARDS_PER_JOB = 4
MAX_SHARDS_PER_COMPONENT = 12
MAX_TIMEOUT_MINUTES = 180

_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
# The regex subset that means the same thing to Python's re and to CMake's
# string(REGEX), which ctest -L/-LE use.
_LABEL_REGEX_RE = re.compile(r"^[A-Za-z0-9_.^$*+?()|\[\]-]+$")
_FAMILY_RE = re.compile(r"^gfx[0-9A-Za-z]+(-[a-z]+)?$")


class ManifestError(ValueError):
    """The manifest is present but invalid."""


@dataclass(frozen=True)
class FamilyExclusion:
    platform: str
    families: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class JobSpec:
    name: str
    labels: tuple[str, ...]  # empty for the remainder job
    shards: object  # <tier value> or None
    timeout_minutes: object  # <tier value> or None
    allow_empty: bool
    exclude_family: tuple[FamilyExclusion, ...]

    @property
    def label_regex(self) -> str:
        return "|".join(self.labels)


@dataclass(frozen=True)
class JobsManifest:
    default_shards: object
    default_timeout_minutes: object
    remainder: JobSpec
    groups: tuple[JobSpec, ...]


# ---------------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------------


def _check_keys(table: dict, allowed: set[str], where: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ManifestError(f"{where}: unknown key(s) {unknown}")


def _check_int(value, where: str, low: int, high: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ManifestError(f"{where}: expected an integer, got {value!r}")
    if not low <= value <= high:
        raise ManifestError(f"{where}: {value} is outside [{low}, {high}]")


def _check_tier_value(value, where: str, low: int, high: int) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        _check_int(value, where, low, high)
        return
    _check_keys(value, set(TIERS), where)
    for tier, per_tier in value.items():
        if isinstance(per_tier, dict):
            _check_keys(per_tier, set(PLATFORMS), f"{where}.{tier}")
            for platform, v in per_tier.items():
                _check_int(v, f"{where}.{tier}.{platform}", low, high)
        else:
            _check_int(per_tier, f"{where}.{tier}", low, high)


def resolve_tier_value(value, tier: str, platform: str):
    """Return the value for (tier, platform), or None when it is not set."""
    if value is None:
        return None
    if not isinstance(value, dict):
        return value
    per_tier = value.get(tier)
    if isinstance(per_tier, dict):
        return per_tier.get(platform)
    return per_tier


def _parse_job(table: dict, where: str, is_remainder: bool) -> JobSpec:
    allowed = {"name", "shards", "timeout_minutes"}
    if not is_remainder:
        allowed |= {"labels", "allow_empty", "exclude_family"}
    _check_keys(table, allowed, where)

    name = table.get("name", "unit" if is_remainder else None)
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ManifestError(f"{where}.name: {name!r} must match {_NAME_RE.pattern}")

    labels: tuple[str, ...] = ()
    if not is_remainder:
        raw_labels = table.get("labels")
        if not isinstance(raw_labels, list) or not raw_labels:
            raise ManifestError(f"{where}.labels: expected a non-empty list")
        for regex in raw_labels:
            if not isinstance(regex, str) or not _LABEL_REGEX_RE.match(regex):
                raise ManifestError(
                    f"{where}.labels: {regex!r} must use only "
                    f"{_LABEL_REGEX_RE.pattern[1:-1]}"
                )
            try:
                re.compile(regex)
            except re.error as exc:
                raise ManifestError(f"{where}.labels: {regex!r}: {exc}") from exc
        labels = tuple(raw_labels)

    _check_tier_value(table.get("shards"), f"{where}.shards", 1, MAX_SHARDS_PER_JOB)
    _check_tier_value(
        table.get("timeout_minutes"),
        f"{where}.timeout_minutes",
        1,
        MAX_TIMEOUT_MINUTES,
    )

    allow_empty = table.get("allow_empty", False)
    if not isinstance(allow_empty, bool):
        raise ManifestError(f"{where}.allow_empty: expected true or false")

    exclusions = []
    for i, entry in enumerate(table.get("exclude_family", [])):
        ewhere = f"{where}.exclude_family[{i}]"
        if not isinstance(entry, dict):
            raise ManifestError(f"{ewhere}: expected a table")
        _check_keys(entry, {"platform", "families", "reason"}, ewhere)
        platform = entry.get("platform")
        if platform not in PLATFORMS:
            raise ManifestError(f"{ewhere}.platform: expected one of {PLATFORMS}")
        families = entry.get("families")
        if (
            not isinstance(families, list)
            or not families
            or not all(isinstance(f, str) and _FAMILY_RE.match(f) for f in families)
        ):
            raise ManifestError(
                f"{ewhere}.families: expected a non-empty list of gfx family names"
            )
        reason = entry.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ManifestError(f"{ewhere}.reason: a reason is required")
        exclusions.append(FamilyExclusion(platform, tuple(families), reason))

    return JobSpec(
        name=name,
        labels=labels,
        shards=table.get("shards"),
        timeout_minutes=table.get("timeout_minutes"),
        allow_empty=allow_empty,
        exclude_family=tuple(exclusions),
    )


def parse_manifest(text: str, source: str = "jobs manifest") -> JobsManifest:
    """Parse and validate a jobs manifest. Raises ManifestError when invalid."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ManifestError(f"{source}: {exc}") from exc

    _check_keys(data, {"schema", "defaults", "remainder", "group"}, source)
    if data.get("schema") != SCHEMA_VERSION:
        raise ManifestError(
            f"{source}: schema must be {SCHEMA_VERSION}, got {data.get('schema')!r}"
        )

    defaults = data.get("defaults", {})
    _check_keys(defaults, {"shards", "timeout_minutes"}, f"{source}: defaults")
    _check_tier_value(
        defaults.get("shards"), f"{source}: defaults.shards", 1, MAX_SHARDS_PER_JOB
    )
    _check_tier_value(
        defaults.get("timeout_minutes"),
        f"{source}: defaults.timeout_minutes",
        1,
        MAX_TIMEOUT_MINUTES,
    )

    remainder = _parse_job(
        data.get("remainder", {}), f"{source}: remainder", is_remainder=True
    )
    groups = tuple(
        _parse_job(g, f"{source}: group[{i}]", is_remainder=False)
        for i, g in enumerate(data.get("group", []))
    )

    names = [remainder.name] + [g.name for g in groups]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ManifestError(f"{source}: duplicate job name(s) {duplicates}")

    manifest = JobsManifest(
        default_shards=defaults.get("shards"),
        default_timeout_minutes=defaults.get("timeout_minutes"),
        remainder=remainder,
        groups=groups,
    )

    # Total shards for the component, worst case (no family exclusions).
    for tier in TIERS:
        for platform in PLATFORMS:
            total = sum(
                job_shards(manifest, job, tier, platform)
                for job in (manifest.remainder, *manifest.groups)
            )
            if total > MAX_SHARDS_PER_COMPONENT:
                raise ManifestError(
                    f"{source}: {total} shards on {tier}/{platform} exceeds "
                    f"{MAX_SHARDS_PER_COMPONENT}"
                )
    return manifest


def job_shards(manifest: JobsManifest, job: JobSpec, tier: str, platform: str) -> int:
    for value in (job.shards, manifest.default_shards):
        resolved = resolve_tier_value(value, tier, platform)
        if resolved is not None:
            return resolved
    return DEFAULT_SHARDS


def job_timeout_minutes(
    manifest: JobsManifest, job: JobSpec, tier: str, platform: str
) -> int:
    for value in (job.timeout_minutes, manifest.default_timeout_minutes):
        resolved = resolve_tier_value(value, tier, platform)
        if resolved is not None:
            return resolved
    return DEFAULT_TIMEOUT_MINUTES


# ---------------------------------------------------------------------------
# Expansion into test matrix entries
# ---------------------------------------------------------------------------


def _env_prefix(env: dict[str, str]) -> str:
    return "".join(f"{key}={shlex.quote(value)} " for key, value in env.items())


def expand_jobs(
    base: dict,
    manifest: JobsManifest,
    *,
    platform: str,
    test_type: str,
    is_excluded: Callable[[list[str]], bool],
) -> list[dict]:
    """Expand one resolved test matrix entry into the manifest's jobs.

    ``base`` is the component's job config as fetch_test_configurations.py would
    emit it. ``is_excluded(families)`` reports whether the current family matches
    any entry of ``families``.
    """
    base_name = base["job_name"]
    jobs = []

    def make(job: JobSpec, env: dict[str, str]) -> dict:
        shards = job_shards(manifest, job, test_type, platform)
        config = dict(base)
        config["job_name"] = f"{base_name}-{job.name}"
        config["total_shards"] = shards
        config["shard_arr"] = list(range(1, shards + 1))
        config["timeout_minutes"] = job_timeout_minutes(
            manifest, job, test_type, platform
        )
        # TEST_COMPONENT keeps every per-component setting in test_runner.py
        # (test directory, sharding mode) keyed on the original component.
        env = {"TEST_COMPONENT": base_name, **env}
        config["test_script"] = _env_prefix(env) + base["test_script"]
        return config

    selected_groups = []
    for group in manifest.groups:
        excluded_by = next(
            (
                e
                for e in group.exclude_family
                if e.platform == platform and is_excluded(list(e.families))
            ),
            None,
        )
        if excluded_by:
            logging.info(
                f"Excluding job {base_name}-{group.name}: {excluded_by.reason}"
            )
            continue
        selected_groups.append(group)

    # The remainder excludes every group, including groups excluded on this
    # family: their tests do not belong to the remainder anywhere.
    all_group_regex = "|".join(g.label_regex for g in manifest.groups)
    remainder_env = {"TEST_LABEL_EXCLUDE": all_group_regex} if all_group_regex else {}
    jobs.append(make(manifest.remainder, remainder_env))

    for group in selected_groups:
        others = "|".join(
            g.label_regex for g in manifest.groups if g.name != group.name
        )
        env = {"TEST_LABEL_INCLUDE": group.label_regex}
        if others:
            env["TEST_LABEL_DISJOINT"] = others
        if group.allow_empty:
            env["TEST_LABEL_ALLOW_EMPTY"] = "1"
        jobs.append(make(group, env))
    return jobs


# ---------------------------------------------------------------------------
# Locating the manifest for a workflow run
# ---------------------------------------------------------------------------


def override_env_var(component_key: str) -> str:
    """Env var naming a local manifest file that replaces the fetched one."""
    return "TEST_JOBS_MANIFEST_" + re.sub(r"[^A-Za-z0-9]", "_", component_key).upper()


def _http_get(url: str) -> bytes | None:
    """GET a URL; None on 404, raises on other failures."""
    request = urllib.request.Request(url, headers={"User-Agent": "therock-ci"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def _github_raw_url(submodule_url: str, sha: str, path: str) -> str:
    match = re.match(r"^https://github\.com/([^/]+/[^/]+?)(\.git)?/?$", submodule_url)
    if not match:
        raise ValueError(f"not a github.com repository URL: {submodule_url}")
    return f"https://raw.githubusercontent.com/{match.group(1)}/{sha}/{path}"


def fetch_manifest_text(
    spec: dict,
    *,
    run_id: str,
    platform: str,
    artifact_group: str | None,
    github_repository: str | None,
) -> str | None:
    """Read the manifest at the source revision recorded for a workflow run.

    The run's therock_manifest.json records the commit of every submodule that
    was built, including the external repository's merge commit for PR runs.
    Returns None when the run or the source tree has no manifest.
    """
    sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))
    from _therock_utils.workflow_outputs import WorkflowOutputRoot

    root = WorkflowOutputRoot.from_workflow_run(
        run_id=run_id,
        platform=platform,
        github_repository=github_repository,
        lookup_workflow_run=True,
    )
    candidates = [root.manifest_root()]
    if artifact_group:
        candidates.append(root.manifest(artifact_group))

    therock_manifest = None
    for location in candidates:
        body = _http_get(location.https_url)
        if body is not None:
            therock_manifest = json.loads(body)
            break
    if therock_manifest is None:
        logging.warning(f"No therock_manifest.json for run {run_id} ({platform})")
        return None

    submodule = next(
        (
            s
            for s in therock_manifest.get("submodules", [])
            if s.get("submodule_path") == spec["submodule"]
        ),
        None,
    )
    if submodule is None:
        logging.warning(
            f"therock_manifest.json for run {run_id} has no submodule "
            f"{spec['submodule']!r}"
        )
        return None

    url = _github_raw_url(
        submodule["submodule_url"], submodule["pin_sha"], spec["path"]
    )
    logging.info(f"Reading jobs manifest {url}")
    body = _http_get(url)
    return body.decode("utf-8") if body is not None else None


def load_manifest(
    component_key: str,
    spec: dict,
    *,
    platform: str,
    artifact_group: str | None,
) -> JobsManifest | None:
    """Return the component's manifest for this run, or None to keep one job.

    An override file (see override_env_var) wins. Otherwise the manifest is
    fetched only inside CI, where ARTIFACT_RUN_ID names the run under test.
    A manifest that cannot be fetched keeps the single job; one that is fetched
    but invalid raises ManifestError.
    """
    override = os.environ.get(override_env_var(component_key))
    if override:
        path = Path(override)
        return parse_manifest(path.read_text(encoding="utf-8"), os.fspath(path))

    run_id = os.environ.get("ARTIFACT_RUN_ID")
    if not run_id:
        return None
    try:
        text = fetch_manifest_text(
            spec,
            run_id=run_id,
            platform=platform,
            artifact_group=artifact_group,
            github_repository=os.environ.get("GITHUB_REPOSITORY"),
        )
    except Exception as exc:  # network or lookup failure: keep the single job
        logging.warning(f"Could not fetch the {component_key} jobs manifest: {exc}")
        return None
    if text is None:
        logging.info(f"No jobs manifest for {component_key}; keeping one job")
        return None
    return parse_manifest(text, f"{spec['submodule']}/{spec['path']}")
