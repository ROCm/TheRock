#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run gitleaks against the current repository checkout.

* Download + cache the gitleaks binary from TheRock's third-party S3
  mirror, verify its SHA-256, and smoke-test it before scanning.
* Resolve the config at `_CONFIG_PATH`; fall back to gitleaks' built-in
  rules if it is missing.
* Derive `--log-opts` from the GitHub Actions event so `scan_mode=changed`
  scans only new commits and `scan_mode=all` scans the full history. A
  malformed `GITHUB_EVENT_PATH` is a hard error.
* Run `gitleaks detect` once per requested report format (gitleaks emits
  one report per invocation; see https://github.com/gitleaks/gitleaks/pull/1232).
* Write `sarif_path` and `non_sarif_paths` to `$GITHUB_OUTPUT` and echo
  each non-SARIF report into the job log and `$GITHUB_STEP_SUMMARY`.
* Post-process SARIF: set `result.level = "error"` and
  `result.properties.security-severity = _LEAK_SECURITY_SEVERITY_HIGH`
  so findings land in the GitHub Security tab's High tier (gitleaks
  leaves both fields unset, defaulting to Medium).

Exit codes:

* `0` - no leaks, clean run.
* `1` - gitleaks found leaks, or `--report-formats` was empty/unknown.
* `2` - input error: scan path missing, `GITHUB_EVENT_PATH` malformed,
  or gitleaks itself errored.

Inputs come from CLI flags or matching `GITLEAKS_*` env vars set by the
workflow.
"""

import argparse
import json
import logging
import os
import hashlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

# Reach the shared GitHub Actions helpers under `build_tools/`. Mirrors
# the import pattern used by `build_tools/generate_manifest_diff_report.py`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "build_tools"))
from github_actions.github_actions_api import (  # noqa: E402
    gha_append_step_summary,
    gha_load_github_event,
)

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
# Gitleaks release mirrored to the rocm-third-party-deps S3 bucket (see
# docs/development/git_chores.md "Updating a third-party mirror") so the
# CI runner doesn't depend on github.com/gitleaks/gitleaks being reachable
# or untampered with at scan time. The expected SHA256 is verified after
# every download (see `_ensure_gitleaks`).
#
# To bump the gitleaks version:
#   1. Download gitleaks_<new>_linux_x64.tar.gz + gitleaks_<new>_checksums.txt
#      from https://github.com/gitleaks/gitleaks/releases/tag/v<new>
#   2. Verify sha256sum matches the checksums.txt line for linux_x64.
#   3. Upload the tarball to https://us-east-2.console.aws.amazon.com/s3/buckets/rocm-third-party-deps
#   4. Update all three constants below in the same commit.
_GITLEAKS_VERSION = "8.30.1"
# Originally mirrored from: https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_linux_x64.tar.gz
_GITLEAKS_TARBALL_URL = (
    "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/"
    f"gitleaks_{_GITLEAKS_VERSION}_linux_x64.tar.gz"
)
_GITLEAKS_TARBALL_SHA256 = (
    "551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb"
)
# Repository-root config path. Hardcoded (no input override): every
# caller of this script points at the same file, so exposing a path
# input just invited drift between callers. A missing file gracefully
# falls back to gitleaks's built-in default rules so the workflow keeps
# working in fresh checkouts before a config has been committed.
_CONFIG_PATH = "gitleaks.toml"
# `gitleaks detect --exit-code N` makes the binary exit with N when it
# finds leaks. We pin this to 1 so we can tell "clean run" (rc=0) apart
# from "leaks found" (rc=1) and from "gitleaks itself errored" (rc>1).
_LEAK_EXIT_CODE = 1
_LEAK_SECURITY_SEVERITY_HIGH = "8.5"


@dataclass(frozen=True)
class _ReportTarget:
    """A single `(format, on-disk path)` pair the runner will produce."""

    fmt: str
    path: Path


def _sha256_of(path: Path) -> str:
    """Return the SHA-256 of `path` as a lowercase hex string."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _ensure_gitleaks() -> Path:
    """Return the path to a verified gitleaks binary, downloading it if needed.

    The binary is installed under `$RUNNER_TEMP` on a GitHub Actions
    runner (the system temp dir otherwise). That directory is per-job
    storage which GitHub wipes between runs, so each scan job
    re-downloads; the in-job existence check just avoids redundant work
    within a single `python gitleaks.py` invocation. The downloaded
    tarball is checked against `_GITLEAKS_TARBALL_SHA256` before
    extraction; a mismatch raises `RuntimeError` and the partial
    download is discarded.
    """
    install_root = Path(os.environ.get("RUNNER_TEMP") or tempfile.gettempdir())
    install_dir = install_root / f"gitleaks-{_GITLEAKS_VERSION}"
    binary = install_dir / "gitleaks"
    if binary.is_file() and os.access(binary, os.X_OK):
        log.info("Found gitleaks binary at %s", binary)
        return binary

    install_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "Downloading gitleaks v%s from %s",
        _GITLEAKS_VERSION,
        _GITLEAKS_TARBALL_URL,
    )
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tarball_path = Path(tmp.name)
    try:
        with (
            urlopen(Request(_GITLEAKS_TARBALL_URL), timeout=60) as resp,
            open(tarball_path, "wb") as out,
        ):
            shutil.copyfileobj(resp, out)
        actual_sha = _sha256_of(tarball_path)
        if actual_sha != _GITLEAKS_TARBALL_SHA256:
            raise RuntimeError(
                f"gitleaks tarball SHA256 mismatch: expected "
                f"{_GITLEAKS_TARBALL_SHA256}, got {actual_sha} "
                f"(downloaded from {_GITLEAKS_TARBALL_URL})"
            )
        with tarfile.open(tarball_path, mode="r:gz") as tar:
            # `filter='data'` rejects unsafe member metadata (absolute
            # paths, traversal, special files)
            member = tar.getmember("gitleaks")
            tar.extract(member, path=install_dir, filter="data")
    finally:
        tarball_path.unlink(missing_ok=True)

    if not binary.is_file():
        raise RuntimeError(
            f"gitleaks tarball for v{_GITLEAKS_VERSION} did not contain "
            f"a 'gitleaks' file at {binary}"
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
    """Parse a comma-separated `report_formats` value into report targets.

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


def _resolve_config_path() -> str | None:
    """Return `_CONFIG_PATH` if it exists, else `None` (gitleaks built-ins)."""
    if Path(_CONFIG_PATH).is_file():
        log.info("Using gitleaks config: %s", _CONFIG_PATH)
        return _CONFIG_PATH
    log.warning(
        "No config found at '%s'; using gitleaks built-in default rules",
        _CONFIG_PATH,
    )
    return None


def _determine_log_opts(scan_mode: str, event_name: str, event: dict[str, Any]) -> str:
    """Build the `--log-opts` value for `gitleaks detect`.

    Returns an empty string to indicate "scan the entire history" (i.e.
    don't pass `--log-opts` at all). Raises `ValueError` for event types
    we don't know how to derive a diff range from (including local runs
    without `$GITHUB_EVENT_NAME`); the caller should pass `--scan-mode all`
    for those instead of silently scanning something unexpected.
    """
    if scan_mode == "all":
        return ""

    if event_name in ("pull_request", "pull_request_target"):
        # GitHub guarantees both base.sha and head.sha on pull_request*
        # events, so a KeyError here is a real payload-format problem
        # rather than something we should try to paper over.
        pr = event["pull_request"]
        base_sha = pr["base"]["sha"]
        head_sha = pr["head"]["sha"]
        # Best-effort fetch of the base SHA so the range is reachable.
        # `actions/checkout` with `fetch-depth: 0` fetches HEAD's lineage,
        # but the PR base may live on a ref that wasn't explicitly fetched.
        subprocess.run(
            ["git", "fetch", "--no-tags", "--depth=1", "origin", base_sha],
            check=False,
            capture_output=True,
        )
        # Note: no `--no-merges`. A merge commit can introduce a secret
        # (the merge resolution diff) that none of its parent commits
        # contained individually, so dropping merges from the scan would
        # leave that class of leak undetected on long-running branches.
        return f"{base_sha}..{head_sha}"

    if event_name == "push":
        # GitHub guarantees `before` and `after` on push events.
        before = event["before"]
        after = event["after"]
        # GitHub uses a 0-only SHA for "no previous commit" (new ref);
        # there's nothing to diff against so we scan everything.
        if set(before) <= {"0"}:
            log.info("Push created a new ref; falling back to full history scan")
            return ""
        # See PR branch above for why `--no-merges` is intentionally absent.
        return f"{before}..{after}"

    raise ValueError(
        f"Cannot derive a diff range for event "
        f"'{event_name or '<unset>'}'. Pass --scan-mode all "
        f"(or set scan_mode='all' in the workflow input) to scan the "
        f"full repository history."
    )


def _enrich_sarif_with_security_severity(sarif_path: Path) -> None:
    """Mark every gitleaks SARIF result as High severity for code scanning.

    Gitleaks's native SARIF formatter leaves `result.level` unset
    (defaults to SARIF's `warning` -> GitHub's Medium tier) and does
    not emit `properties.security-severity`. Every gitleaks hit is a
    leaked secret -- there's no useful low/medium tier for that -- so
    we backfill both fields uniformly:

    * `level = "error"` (so SARIF viewers that don't read
      `security-severity` still treat the finding as high priority).
    * `properties.security-severity = _LEAK_SECURITY_SEVERITY_HIGH`
      (places the finding in GitHub code scanning's *High* tier and
      makes it filterable in the Security tab's severity dropdown the
      same way CodeQL alerts are).

    Pre-existing values for either field are preserved verbatim (so a
    future gitleaks version that emits them natively keeps control).
    Failures during enrichment are logged at WARNING and don't
    propagate -- a missing `security-severity` is benign (the
    Security tab still ingests the SARIF), so we'd rather emit a
    slightly-less-rich SARIF than fail the scan job.
    """
    levels_set_count = 0
    levels_kept_count = 0
    scores_set_count = 0
    scores_kept_count = 0
    try:
        with open(sarif_path, encoding="utf-8") as f:
            data = json.load(f)
        for run in data.get("runs") or []:
            for result in run.get("results") or []:
                if result.get("level") is None:
                    result["level"] = "error"
                    levels_set_count += 1
                else:
                    levels_kept_count += 1
                props = result.setdefault("properties", {})
                if props.get("security-severity") is None:
                    props["security-severity"] = _LEAK_SECURITY_SEVERITY_HIGH
                    scores_set_count += 1
                else:
                    scores_kept_count += 1
    except (OSError, json.JSONDecodeError, AttributeError, KeyError, TypeError) as exc:
        log.warning("SARIF severity enrichment skipped (%s): %s", sarif_path, exc)
        return

    if levels_set_count == 0 and scores_set_count == 0:
        log.debug(
            "SARIF severity enrichment: nothing to add (%d level preserved, "
            "%d score preserved) in %s",
            levels_kept_count,
            scores_kept_count,
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
        levels_set_count,
        _LEAK_SECURITY_SEVERITY_HIGH,
        scores_set_count,
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
    """Run gitleaks once per target. Return `True` if any leaks were found.

    Raises :class:`RuntimeError` for unexpected gitleaks exit codes.
    """
    base_args: list[str] = [
        str(binary),
        "detect",
        "--source",
        str(source_dir),
        "--redact",
        "--verbose",
        "--no-banner",
        "--exit-code",
        str(_LEAK_EXIT_CODE),
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
    """Append step outputs to `$GITHUB_OUTPUT` using the heredoc form for multiline."""
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


def _emit_non_sarif_reports(non_sarif: list[_ReportTarget]) -> None:
    """Surface each non-SARIF report in the workflow run.

    Writes each report's content to two places so PR reviewers don't
    have to download the `gitleaks-report` artifact, unzip it, and open
    the file locally:

    1. stdout, wrapped in `::group::`/`::endgroup::` workflow
       commands so the live job log stays scannable.
    2. `$GITHUB_STEP_SUMMARY` via `gha_append_step_summary` so the
       report is one click away from the run page.

    Missing files are skipped with a warning rather than failing the
    job; they typically mean gitleaks exited before producing that
    format.
    """
    summary_chunks: list[str] = []
    for target in non_sarif:
        path = target.path
        if not path.is_file():
            log.warning(
                "non-SARIF report '%s' missing; skipping log + summary emission",
                path,
            )
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        print(f"::group::Gitleaks report: {path}")
        print(content)
        print("::endgroup::")
        summary_chunks.append(
            f"### Gitleaks report: `{path}`\n\n```\n{content}\n```"
        )
    if summary_chunks:
        gha_append_step_summary("\n\n".join(summary_chunks))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scan-mode",
        default=os.environ.get("GITLEAKS_SCAN_MODE", "changed"),
        choices=("changed", "all"),
        help=(
            "'changed' (default) scans only commits introduced by the calling "
            "event; requires a pull_request*, pull_request_target, or push "
            "event payload at $GITHUB_EVENT_PATH and hard-fails otherwise. "
            "'all' scans the full repository history and is required for "
            "schedule, workflow_dispatch, release, and any other event."
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

    config_path = _resolve_config_path()
    try:
        event = gha_load_github_event()
        log_opts = _determine_log_opts(
            scan_mode=args.scan_mode,
            event_name=os.environ.get("GITHUB_EVENT_NAME", ""),
            event=event,
        )
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        log.error("%s", exc)
        return 2
    log.info("Gitleaks scope: %s", log_opts or "<full repository history>")
    log.info("Gitleaks source: %s", source_dir)
    log.info(
        "Gitleaks formats: %s",
        ", ".join(f"{t.fmt}->{t.path}" for t in targets),
    )

    # Emit step outputs up-front so the workflow's upload steps know
    # which paths to look at even if gitleaks fails partway through.
    sarif_target = next((t for t in targets if t.fmt == "sarif"), None)
    non_sarif = [t for t in targets if t.fmt != "sarif"]
    _write_github_output(
        sarif_path="" if sarif_target is None else str(sarif_target.path),
        non_sarif_paths="\n".join(str(t.path) for t in non_sarif),
    )

    try:
        binary = _ensure_gitleaks()
        leaks_found = _run_gitleaks(
            binary,
            targets,
            config_path=config_path,
            log_opts=log_opts,
            source_dir=source_dir,
        )
    except RuntimeError as exc:
        log.error("%s", exc)
        _emit_non_sarif_reports(non_sarif)
        return 2

    _emit_non_sarif_reports(non_sarif)

    if leaks_found:
        log.error("gitleaks found one or more potential secrets; see report artifacts")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
