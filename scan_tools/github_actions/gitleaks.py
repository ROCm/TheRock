#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run gitleaks against the current repository checkout.

This script is the implementation behind ``.github/workflows/gitleaks.yml``
and any other reusable callers. It performs every step the CI workflow
needs in a single invocation:

* Downloads + caches the gitleaks release binary at the requested version
  under ``$RUNNER_TEMP`` and smoke-tests it with ``gitleaks version``
  before any scan runs.
* Validates the caller-supplied inputs (scan mode, report formats,
  config path, scan path) and resolves the gitleaks config file:
  the default path falls back gracefully to gitleaks's built-in rules
  when missing, while an explicit non-default path that's missing is a
  hard error so a typo can't silently disable custom rules.
* Optionally writes a starter ``gitleaks.toml.template`` for review and
  refuses to overwrite an existing one (``--generate-config-template``).
* Derives a ``--log-opts`` git range from the current GitHub Actions
  event context so that ``scan_mode=changed`` only inspects new commits,
  while ``scan_mode=all`` scans the entire repository history. A *set*
  but malformed ``GITHUB_EVENT_PATH`` is a hard error rather than a
  silent fallback to "scan everything".
* Runs ``gitleaks detect`` once per requested report format. NOTE:
  gitleaks only emits a single report per invocation today, so we
  re-run the scan per format. Revisit once the upstream multi-output PR
  lands: https://github.com/gitleaks/gitleaks/pull/1232
* Writes ``sarif_path``, ``non_sarif_paths``, and
  ``config_template_path`` to ``$GITHUB_OUTPUT`` (each empty when not
  applicable) so the calling workflow can upload SARIF to code scanning
  and the rest as build artifacts.
* Post-processes any SARIF output to backfill ``result.level`` (set to
  ``error``) and ``result.properties.security-severity`` (set to
  ``_LEAK_SECURITY_SEVERITY``). Gitleaks's native SARIF leaves both
  fields unset, which would tier findings at *Medium* in the GitHub
  Security tab; every gitleaks hit is a leaked secret, so we override
  to *High* uniformly and make findings filterable in the Security
  tab's severity dropdown the same way CodeQL alerts are.
  ``tool.driver.name = "gitleaks"`` (set by gitleaks itself) plus the
  ``category: gitleaks`` argument the reusable workflow passes to
  ``upload-sarif`` keep the resulting alerts filterable by tool,
  separately from any other scanner.

Exit codes:

* ``0`` — clean run, no leaks, no input issues.
* ``1`` — gitleaks found one or more potential secrets, or
  ``--report-formats`` was empty / contained an unknown format.
* ``2`` — input/config error before or during the scan: scan path
  doesn't exist, explicit config path is missing, ``GITHUB_EVENT_PATH``
  is malformed, refused to overwrite an existing template, or gitleaks
  itself errored unexpectedly.

Inputs are read from CLI flags (with ``GITLEAKS_*`` environment-variable
defaults so they round-trip cleanly through the workflow's ``env:``
block).
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

log = logging.getLogger(__name__)

# Mapping of the gitleaks `--report-format` values we expose to the file
# extension we use for the on-disk artifact. Keep this list in sync with
# the `report_formats` input documented in `.github/workflows/gitleaks.yml`.
_SUPPORTED_FORMATS: dict[str, str] = {
    "sarif": "sarif",
    "json": "json",
    "csv": "csv",
    "junit": "xml",
}
_DEFAULT_GITLEAKS_VERSION = "8.30.1"
# Default config path is also the "graceful fallback" sentinel: when the
# resolved value still equals this constant we silently fall back to the
# gitleaks built-in rules if the file is missing (so the workflow works
# in repos without a custom config). Any other value is treated as an
# explicit override and a missing file becomes a hard error.
_DEFAULT_CONFIG_PATH = "gitleaks.toml"
# Path the config-template generator writes to. We use a `.template`
# suffix so we never silently overwrite an in-tree gitleaks.toml; the
# user is expected to review the artifact, copy/edit it, and commit the
# result as the canonical config.
_CONFIG_TEMPLATE_PATH = "gitleaks.toml.template"
# `gitleaks detect --exit-code N` makes the binary exit with N when it
# finds leaks. We pin this to 1 so we can tell "clean run" (rc=0) apart
# from "leaks found" (rc=1) and from "gitleaks itself errored" (rc>1).
_LEAK_EXIT_CODE = 1
_DOWNLOAD_TIMEOUT_SECONDS = 60
# Numeric `security-severity` (CVSS-like 0.1-10.0) injected into every
# SARIF result so the GitHub code-scanning Security tab tiers gitleaks
# findings uniformly at *High*. Unlike a code-quality scanner (which
# can have low/medium/high tiers per rule), every gitleaks hit is a
# leaked secret -- there is no useful "low-severity secret" case.
# 8.5 lands squarely inside GitHub's High tier (7.0-8.9). Bump to 9.0+
# if exposed secrets should appear under the *Critical* tier instead.
_LEAK_SECURITY_SEVERITY = "8.5"


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a GitHub Actions-style boolean env var.

    GitHub renders ``type: boolean`` workflow inputs as the literal
    strings ``"true"`` / ``"false"`` in the job environment, so we accept
    that pair plus the usual ``1``/``yes``/``on`` aliases for CLI use.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("true", "1", "yes", "on")


def _render_config_template(gitleaks_version: str) -> str:
    """Return the contents of a starter ``gitleaks.toml`` for review.

    The template extends gitleaks's built-in default ruleset and seeds
    common allowlist patterns + a worked custom-rule example, all
    commented out. The intent is to give the reviewer a working starting
    point rather than a wall of opinionated defaults.
    """
    return f"""\
# gitleaks configuration template
#
# Generated by scan_tools/github_actions/gitleaks.py for gitleaks
# v{gitleaks_version}. To use it:
#   1. Review the contents below and customize as needed.
#   2. Rename to `gitleaks.toml` and commit at the repository root.
#   3. The CI workflow picks it up automatically (no input change needed).
#
# References
#   - Full default ruleset for v{gitleaks_version}:
#     https://github.com/gitleaks/gitleaks/blob/v{gitleaks_version}/config/gitleaks.toml
#   - Configuration schema:
#     https://github.com/gitleaks/gitleaks#configuration

# Inherit gitleaks's built-in default rules. Setting `useDefault = false`
# disables them and forces you to define every rule from scratch (rarely
# what you want).
[extend]
useDefault = true

# Optional: pull in additional rule packs by URL or local path.
# path = "shared-rules.toml"
# url  = "https://example.com/rules/gitleaks.toml"

# ---------------------------------------------------------------------------
# Project-wide allowlist. Matches here are excluded from ALL rules.
# Prefer narrow, targeted entries (paths, file globs, specific commits) over
# broad regex allowlists that can mask real findings.
# ---------------------------------------------------------------------------
[allowlist]
description = "Project allowlist"

# Paths (regex match against full path, relative to the repo root).
paths = [
    # '''(?i)tests?/fixtures/.*''',
    # '''(?i)docs?/.*\\.md$''',
    # '''package-lock\\.json$''',
    # '''yarn\\.lock$''',
]

# Specific finding fingerprints (commit:file:rule:start_line) to ignore.
# stopwords = []

# Whole commits to ignore (use sparingly -- prefer fixing leaked secrets).
# commits = ['''<full-commit-sha>''']

# ---------------------------------------------------------------------------
# Custom rules. The default ruleset already covers most common secret types
# (AWS keys, GitHub tokens, JWTs, private keys, etc.); add rules here only
# for tokens specific to your organization.
# ---------------------------------------------------------------------------
# [[rules]]
# id          = "internal-token"
# description = "Internal service access token"
# regex       = '''(?i)(internal[-_]?token)\\s*[:=]\\s*["']?([a-z0-9]{{32,}})["']?'''
# tags        = ["secret", "token"]
# # Per-rule allowlist, only applies to this rule:
# [rules.allowlist]
# regexes = ['''dummy[-_]?token''']
"""


@dataclass(frozen=True)
class _ReportTarget:
    """A single ``(format, on-disk path)`` pair the runner will produce."""

    fmt: str
    path: Path


def _gitleaks_release_url(version: str) -> str:
    return (
        f"https://github.com/gitleaks/gitleaks/releases/download/v{version}/"
        f"gitleaks_{version}_linux_x64.tar.gz"
    )


def _ensure_gitleaks(version: str) -> Path:
    """Return the path to a gitleaks binary, downloading it if needed.

    The binary is cached under ``$RUNNER_TEMP`` (when running on a GitHub
    Actions runner) or the system temp dir otherwise, keyed by version.
    """
    cache_root = Path(os.environ.get("RUNNER_TEMP") or tempfile.gettempdir())
    install_dir = cache_root / f"gitleaks-{version}"
    binary = install_dir / "gitleaks"
    if binary.is_file() and os.access(binary, os.X_OK):
        log.info("Using cached gitleaks binary at %s", binary)
        return binary

    install_dir.mkdir(parents=True, exist_ok=True)
    url = _gitleaks_release_url(version)
    log.info("Downloading gitleaks v%s from %s", version, url)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tarball_path = Path(tmp.name)
    try:
        with (
            urlopen(Request(url), timeout=_DOWNLOAD_TIMEOUT_SECONDS) as resp,
            open(tarball_path, "wb") as out,
        ):
            shutil.copyfileobj(resp, out)
        with tarfile.open(tarball_path, mode="r:gz") as tar:
            # `filter='data'` rejects unsafe member metadata (absolute
            # paths, traversal, special files)
            member = tar.getmember("gitleaks")
            tar.extract(member, path=install_dir, filter="data")
    finally:
        tarball_path.unlink(missing_ok=True)

    if not binary.is_file():
        raise RuntimeError(
            f"gitleaks tarball for v{version} did not contain a 'gitleaks' file at {binary}"
        )
    binary.chmod(0o755)

    # Smoke-test the freshly installed binary so we fail fast if the
    # download is corrupt or the platform is unsupported, rather than
    # surfacing a confusing failure inside the actual scan loop.
    try:
        result = subprocess.run(
            [str(binary), "version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"gitleaks at {binary} failed to execute after install: {exc}"
        ) from exc
    log.info("Installed gitleaks %s at %s", result.stdout.strip(), binary)
    return binary


def _parse_report_formats(raw: str) -> list[_ReportTarget]:
    """Parse a comma-separated ``report_formats`` value into report targets.

    Whitespace is trimmed, duplicates collapse to the first occurrence,
    and unknown formats raise :class:`ValueError`.
    """
    targets: list[_ReportTarget] = []
    seen: set[str] = set()
    for raw_fmt in raw.split(","):
        fmt = raw_fmt.strip()
        if not fmt or fmt in seen:
            continue
        seen.add(fmt)
        ext = _SUPPORTED_FORMATS.get(fmt)
        if ext is None:
            raise ValueError(
                f"Invalid report_format '{fmt}' "
                f"(expected one of: {', '.join(sorted(_SUPPORTED_FORMATS))})"
            )
        targets.append(_ReportTarget(fmt=fmt, path=Path(f"gitleaks-report.{ext}")))
    if not targets:
        raise ValueError(
            "report_formats is empty (expected one or more of: "
            f"{', '.join(sorted(_SUPPORTED_FORMATS))})"
        )
    return targets


def _resolve_config_path(raw: str, *, is_default: bool) -> str | None:
    """Return the existing config path, or ``None`` to use gitleaks defaults.

    When the caller did not override the default path, a missing file is
    treated as graceful fallback (warn + use gitleaks built-in rules) so
    the workflow keeps working in repos without a custom config. When
    the caller *did* set a non-default path, a missing file is a hard
    error (raises :class:`FileNotFoundError`) so a typo can't silently
    disable custom rules.
    """
    if not raw:
        log.warning("No config_path provided; using gitleaks built-in default rules")
        return None
    if Path(raw).is_file():
        log.info("Using gitleaks config: %s", raw)
        return raw
    if is_default:
        log.warning(
            "No config found at default path '%s'; using gitleaks built-in default rules",
            raw,
        )
        return None
    raise FileNotFoundError(
        f"config_path '{raw}' was explicitly set but no such file exists"
    )


def _load_github_event() -> dict[str, Any]:
    """Load the JSON event payload pointed to by ``$GITHUB_EVENT_PATH``.

    An unset variable is a legitimate case (the script may be invoked
    locally for development), so we silently return an empty dict. A
    *set* variable that points at a missing or malformed file is a CI
    misconfiguration and raises rather than silently degrading the scan
    range to "full history".
    """
    raw = os.environ.get("GITHUB_EVENT_PATH", "")
    if not raw:
        return {}
    event_path = Path(raw)
    if not event_path.is_file():
        raise FileNotFoundError(
            f"GITHUB_EVENT_PATH is set to '{event_path}' but no such file exists"
        )
    try:
        with open(event_path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"GITHUB_EVENT_PATH '{event_path}' contains invalid JSON: {exc}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"Cannot read GITHUB_EVENT_PATH '{event_path}': {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"GITHUB_EVENT_PATH '{event_path}' must contain a JSON object, "
            f"got {type(data).__name__}"
        )
    return data


def _determine_log_opts(scan_mode: str, event_name: str, event: dict[str, Any]) -> str:
    """Build the ``--log-opts`` value for ``gitleaks detect``.

    Returns an empty string to indicate "scan the entire history" (i.e.
    don't pass ``--log-opts`` at all). Unknown event types fall back to
    scanning ``HEAD`` only, which is the safest "changed-ish" range we
    can construct without a real diff base.
    """
    if scan_mode == "all":
        return ""

    if event_name in ("pull_request", "pull_request_target"):
        pr = event.get("pull_request") or {}
        base_sha = ((pr.get("base") or {}).get("sha")) or ""
        head_sha = ((pr.get("head") or {}).get("sha")) or ""
        if not base_sha or not head_sha:
            log.warning("PR event missing base/head SHA; falling back to full history")
            return ""
        # Best-effort fetch of the base SHA so the range is reachable.
        # `actions/checkout` with `fetch-depth: 0` fetches HEAD's lineage,
        # but the PR base may live on a ref that wasn't explicitly fetched.
        subprocess.run(
            ["git", "fetch", "--no-tags", "--depth=1", "origin", base_sha],
            check=False,
            capture_output=True,
        )
        return f"--no-merges {base_sha}..{head_sha}"

    if event_name == "push":
        before = event.get("before") or ""
        after = event.get("after") or ""
        if not before or not after:
            log.warning("Push event missing before/after SHA; falling back to full history")
            return ""
        # GitHub uses a 0-only SHA for "no previous commit" (new ref);
        # there's nothing to diff against so we scan everything.
        if set(before) <= {"0"}:
            log.info("Push created a new ref; falling back to full history scan")
            return ""
        return f"--no-merges {before}..{after}"

    log.info("Event '%s' has no diff range; scanning HEAD only", event_name or "<unset>")
    return "-1 HEAD"


def _enrich_sarif_with_security_severity(sarif_path: Path) -> None:
    """Mark every gitleaks SARIF result as High severity for code scanning.

    Gitleaks's native SARIF formatter leaves ``result.level`` unset
    (defaults to SARIF's ``warning`` -> GitHub's Medium tier) and does
    not emit ``properties.security-severity``. Every gitleaks hit is a
    leaked secret -- there's no useful low/medium tier for that -- so
    we backfill both fields uniformly:

    * ``level = "error"`` (so SARIF viewers that don't read
      ``security-severity`` still treat the finding as high priority).
    * ``properties.security-severity = _LEAK_SECURITY_SEVERITY``
      (places the finding in GitHub code scanning's *High* tier and
      makes it filterable in the Security tab's severity dropdown the
      same way CodeQL alerts are).

    Pre-existing values for either field are preserved verbatim (so a
    future gitleaks version that emits them natively keeps control).
    Failures during enrichment are logged at WARNING and don't
    propagate -- a missing ``security-severity`` is benign (the
    Security tab still ingests the SARIF), so we'd rather emit a
    slightly-less-rich SARIF than fail the scan job.
    """
    try:
        with open(sarif_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("SARIF severity enrichment skipped (%s): %s", sarif_path, exc)
        return
    if not isinstance(data, dict):
        log.warning(
            "SARIF severity enrichment skipped: %s is not a JSON object",
            sarif_path,
        )
        return

    enriched_level = 0
    enriched_score = 0
    preserved_level = 0
    preserved_score = 0
    for run in data.get("runs") or []:
        if not isinstance(run, dict):
            continue
        for result in run.get("results") or []:
            if not isinstance(result, dict):
                continue
            if result.get("level") is None:
                result["level"] = "error"
                enriched_level += 1
            else:
                preserved_level += 1
            props = result.get("properties")
            if not isinstance(props, dict):
                props = {}
                result["properties"] = props
            if props.get("security-severity") is None:
                props["security-severity"] = _LEAK_SECURITY_SEVERITY
                enriched_score += 1
            else:
                preserved_score += 1

    if enriched_level == 0 and enriched_score == 0:
        log.debug(
            "SARIF severity enrichment: nothing to add (%d level preserved, "
            "%d score preserved) in %s",
            preserved_level,
            preserved_score,
            sarif_path,
        )
        return

    try:
        with open(sarif_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError as exc:
        log.warning("Failed to write enriched SARIF to %s: %s", sarif_path, exc)
        return

    log.info(
        "SARIF severity enrichment: set level=error on %d result(s) and "
        "security-severity=%s on %d result(s) in %s",
        enriched_level,
        _LEAK_SECURITY_SEVERITY,
        enriched_score,
        sarif_path,
    )


def _run_gitleaks(
    binary: Path,
    targets: list[_ReportTarget],
    *,
    config_path: str | None,
    log_opts: str,
    source_dir: Path,
) -> bool:
    """Run gitleaks once per target. Return ``True`` if any leaks were found.

    Raises :class:`RuntimeError` for unexpected gitleaks exit codes.
    """
    base_args: list[str] = [
        str(binary),
        "detect",
        "--source", str(source_dir),
        "--redact",
        "--verbose",
        "--no-banner",
        "--exit-code", str(_LEAK_EXIT_CODE),
    ]
    if config_path:
        base_args.extend(["--config", config_path])
    if log_opts:
        base_args.append(f"--log-opts={log_opts}")

    leaks_found = False
    # NOTE: gitleaks emits a single report per invocation, so we re-run
    # per format. Revisit when https://github.com/gitleaks/gitleaks/pull/1232
    # is merged.
    for tgt in targets:
        cmd = [*base_args, "--report-format", tgt.fmt, "--report-path", str(tgt.path)]
        log.info("Running: %s", " ".join(cmd))
        rc = subprocess.run(cmd, check=False).returncode
        if rc == 0 or rc == _LEAK_EXIT_CODE:
            if rc == _LEAK_EXIT_CODE:
                leaks_found = True
            # Post-process SARIF reports to align with the GitHub
            # Security tab's severity tiers (gitleaks leaves both
            # `level` and `security-severity` unset by default).
            if tgt.fmt == "sarif" and tgt.path.is_file():
                _enrich_sarif_with_security_severity(tgt.path)
            continue
        raise RuntimeError(
            f"gitleaks exited unexpectedly with code {rc} for format '{tgt.fmt}'"
        )
    return leaks_found


def _write_github_output(**values: str) -> None:
    """Append step outputs to ``$GITHUB_OUTPUT`` using the heredoc form for multiline."""
    out_path = os.environ.get("GITHUB_OUTPUT")
    if not out_path:
        log.debug("GITHUB_OUTPUT is unset; skipping step output emission")
        return
    with open(out_path, "a", encoding="utf-8") as f:
        for key, raw in values.items():
            value = "" if raw is None else str(raw)
            if "\n" in value:
                f.write(f"{key}<<EOF\n{value}\nEOF\n")
            else:
                f.write(f"{key}={value}\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scan-mode",
        default=os.environ.get("GITLEAKS_SCAN_MODE", "changed"),
        choices=("changed", "all"),
        help=(
            "'changed' (default) scans only commits introduced by the calling "
            "event (PR commits or push range). 'all' scans the full repository "
            "history."
        ),
    )
    p.add_argument(
        "--report-formats",
        default=os.environ.get("GITLEAKS_REPORT_FORMATS", "sarif"),
        help=(
            "Comma-separated list of gitleaks report formats. Allowed values: "
            f"{', '.join(sorted(_SUPPORTED_FORMATS))}."
        ),
    )
    p.add_argument(
        "--config-path",
        default=os.environ.get("GITLEAKS_CONFIG_PATH", _DEFAULT_CONFIG_PATH),
        help=(
            "Path to a gitleaks config TOML. Defaults to "
            f"'{_DEFAULT_CONFIG_PATH}' at the repository root, in which case "
            "a missing file gracefully falls back to gitleaks's built-in "
            "default rules. When set to any non-default value the file "
            "MUST exist or the script exits non-zero, so a typo can't "
            "silently disable custom rules."
        ),
    )
    p.add_argument(
        "--gitleaks-version",
        default=os.environ.get("GITLEAKS_VERSION", _DEFAULT_GITLEAKS_VERSION),
        help="Gitleaks release version to download (default %(default)s).",
    )
    p.add_argument(
        "--source-dir",
        default=os.environ.get("GITLEAKS_SOURCE_DIR", "."),
        help=(
            "Path to scan (default %(default)s). Set to a subdirectory of the "
            "checkout to restrict the scan to that subtree; gitleaks's "
            "--source flag combines naturally with --log-opts so the "
            "'changed' scan mode still works for partial-tree scans. The "
            "path must exist."
        ),
    )
    p.add_argument(
        "--generate-config-template",
        default=_env_bool("GITLEAKS_GENERATE_CONFIG_TEMPLATE"),
        action=argparse.BooleanOptionalAction,
        help=(
            "When set, write a starter gitleaks config to "
            f"'{_CONFIG_TEMPLATE_PATH}' alongside the scan. The CI workflow "
            "uploads it as the 'gitleaks-config-template' artifact for "
            "review; the user is expected to copy/edit/rename it to "
            f"'{_DEFAULT_CONFIG_PATH}' and commit the result. Refuses to "
            "overwrite an existing file at the template path."
        ),
    )
    return p


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    args = build_parser().parse_args(argv)

    try:
        targets = _parse_report_formats(args.report_formats)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    source_dir = Path(args.source_dir)
    if not source_dir.is_dir():
        log.error(
            "scan path '%s' does not exist or is not a directory "
            "(did the checkout step fetch it?)",
            source_dir,
        )
        return 2

    try:
        config_path = _resolve_config_path(
            args.config_path,
            is_default=(args.config_path == _DEFAULT_CONFIG_PATH),
        )
        event = _load_github_event()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        log.error("%s", exc)
        return 2

    log_opts = _determine_log_opts(
        scan_mode=args.scan_mode,
        event_name=os.environ.get("GITHUB_EVENT_NAME", ""),
        event=event,
    )
    log.info("Gitleaks scope: %s", log_opts or "<full repository history>")
    log.info("Gitleaks source: %s", source_dir)
    log.info(
        "Gitleaks formats: %s",
        ", ".join(f"{t.fmt}->{t.path}" for t in targets),
    )

    config_template_output = ""
    if args.generate_config_template:
        template_path = Path(_CONFIG_TEMPLATE_PATH)
        if template_path.exists():
            log.error(
                "refusing to overwrite existing '%s'; remove it first or "
                "disable --generate-config-template",
                template_path,
            )
            return 2
        template_path.write_text(
            _render_config_template(args.gitleaks_version), encoding="utf-8"
        )
        log.info("Wrote gitleaks config template to %s", template_path)
        config_template_output = str(template_path)

    # Emit step outputs up-front so the workflow's upload steps know
    # which paths to look at even if gitleaks fails partway through.
    sarif_target = next((t for t in targets if t.fmt == "sarif"), None)
    non_sarif = [t for t in targets if t.fmt != "sarif"]
    _write_github_output(
        sarif_path="" if sarif_target is None else str(sarif_target.path),
        non_sarif_paths="\n".join(str(t.path) for t in non_sarif),
        config_template_path=config_template_output,
    )

    try:
        binary = _ensure_gitleaks(args.gitleaks_version)
        leaks_found = _run_gitleaks(
            binary,
            targets,
            config_path=config_path,
            log_opts=log_opts,
            source_dir=source_dir,
        )
    except RuntimeError as exc:
        log.error("%s", exc)
        return 2

    if leaks_found:
        log.error("gitleaks found one or more potential secrets; see report artifacts")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
