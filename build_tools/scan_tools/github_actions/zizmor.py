#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run zizmor for the current checkout.

- Install the pinned `zizmor` release and verify it.
- Require `zizmor.yml` at the repo root (hard error if missing).
- Derive change sets from the GitHub event for changed/all scans; run per
  requested format and emit SARIF/non-SARIF paths plus severity tally.

Exit codes: 0 clean/empty changed set; 1 findings at/above threshold or
bad formats; 2 input/config/runtime errors.

Inputs come from CLI flags with `ZIZMOR_*` env var defaults.
"""

import argparse
import fnmatch
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

THEROCK_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Add build_tools to path for github_actions imports.
sys.path.insert(0, str(THEROCK_DIR / "build_tools"))
from github_actions.github_actions_api import (  # noqa: E402
    gha_append_step_summary,
    gha_load_github_event,
    gha_set_output,
)

log = logging.getLogger(__name__)

# Map user-visible report formats to file extensions.
_SUPPORTED_FORMATS: dict[str, str] = {
    "sarif": "sarif",
    "json": "json",
    "plain": "txt",
    "github": "txt",
}
_ZIZMOR_VERSION = "1.24.1"
_CONFIG_PATH = "zizmor.yml"
# Ascending severity order; threshold comparisons rely on it.
_SEVERITY_ORDER: tuple[str, ...] = ("INFORMATIONAL", "LOW", "MEDIUM", "HIGH")
_SEVERITY_CHOICES: tuple[str, ...] = tuple(s.lower() for s in _SEVERITY_ORDER)
_DEFAULT_SEVERITY_THRESHOLD = "high"
_PERSONA_CHOICES: tuple[str, ...] = ("regular", "pedantic", "auditor")
_DEFAULT_PERSONA = "regular"
# Internal JSON tally pass output; cleaned up before returning.
_INTERNAL_TALLY_PATH = "zizmor-tally.json"
# Diff filter for 'changed' mode; mirrors zizmor's --collect=default set.
_AUDITED_PATTERNS: tuple[str, ...] = (
    ".github/workflows/*.yml",
    ".github/workflows/*.yaml",
    ".github/dependabot.yml",
    ".github/dependabot.yaml",
    "**/action.yml",
    "**/action.yaml",
    "action.yml",
    "action.yaml",
)
# Map zizmor severities to GitHub security-severity values.
_ZIZMOR_SECURITY_SEVERITY: dict[str, str] = {
    "HIGH": "8.5",
    "MEDIUM": "5.0",
    "LOW": "1.0",
    "INFORMATIONAL": "0.3",
}


@dataclass(frozen=True)
class _ReportTarget:
    """A single `(format, on-disk path)` pair the runner will produce."""

    fmt: str
    path: Path


def _md_code_fence(content: str) -> str:
    """Return a fence longer than any backtick run in `content`."""
    longest = max((len(m) for m in re.findall(r"`+", content)), default=0)
    return "`" * max(3, longest + 1)


def _emit_non_sarif_reports(non_sarif: list[_ReportTarget]) -> None:
    """Surface non-SARIF reports in logs and step summary."""
    summary_chunks: list[str] = []
    for target in non_sarif:
        path = target.path
        if not path.is_file():
            log.warning("Non-SARIF report '%s' missing; skipping", path)
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        print(f"::group::Zizmor report: {path}")
        print(content)
        print("::endgroup::")
        fence = _md_code_fence(content)
        summary_chunks.append(
            f"### Zizmor report: `{path}`\n\n{fence}\n{content}\n{fence}"
        )
    if summary_chunks:
        gha_append_step_summary("\n\n".join(summary_chunks))


def _ensure_zizmor() -> Path:
    spec = f"zizmor=={_ZIZMOR_VERSION}"
    log.info("Installing %s", spec)
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", spec],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to install {spec}: {exc}") from exc

    # Console scripts land next to sys.executable, even for an unactivated venv.
    binary_path = Path(sys.executable).parent / "zizmor"
    if not binary_path.is_file():
        found = shutil.which("zizmor")
        if found is None:
            raise RuntimeError(
                f"zizmor CLI not found at {binary_path} or on PATH after "
                f"installing {spec}; is the active Python environment writable?"
            )
        binary_path = Path(found)

    try:
        result = subprocess.run(
            [str(binary_path), "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"zizmor at {binary_path} failed to execute after install: {exc}"
        ) from exc
    log.info(
        "Installed %s at %s",
        result.stdout.strip().splitlines()[0] if result.stdout.strip() else "zizmor",
        binary_path,
    )
    return binary_path


def _parse_report_formats(raw: str) -> list[_ReportTarget]:
    """Parse a comma-separated `report_formats` value into report targets."""
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
        targets.append(_ReportTarget(fmt=fmt, path=Path(f"zizmor-report.{ext}")))
    if not targets:
        raise ValueError(
            "report_formats is empty (expected one or more of: "
            f"{', '.join(sorted(_SUPPORTED_FORMATS))})"
        )
    return targets


def _resolve_config_path() -> str:
    if not Path(_CONFIG_PATH).is_file():
        raise FileNotFoundError(
            f"zizmor config not found at '{_CONFIG_PATH}'. "
            "Run from the repo root so the config is resolvable."
        )
    log.info("Using zizmor config: %s", _CONFIG_PATH)
    return _CONFIG_PATH


def _event_str(event: dict[str, object], *keys: str) -> str:
    """Return the nested string at `event[keys[0]][keys[1]]...`, or `""`."""
    current: object = event
    for key in keys:
        if not isinstance(current, dict):
            return ""
        current = current.get(key)
    return current if isinstance(current, str) else ""


def _diff_range(event_name: str, event: dict[str, object]) -> tuple[str, str] | None:
    """Return `(base_sha, head_sha)` for the calling event, or `None`."""

    if event_name in ("pull_request", "pull_request_target"):
        base = _event_str(event, "pull_request", "base", "sha")
        head = _event_str(event, "pull_request", "head", "sha")
        if not base or not head:
            log.warning("PR event missing base/head SHA; falling back to full scan")
            return None
        return (base, head)
    if event_name == "push":
        before = _event_str(event, "before")
        after = _event_str(event, "after")
        if not before or not after:
            log.warning(
                "Push event missing before/after SHA; falling back to full scan"
            )
            return None
        # All-zero SHA means a new ref: nothing to diff against.
        if set(before) <= {"0"}:
            log.info("Push created a new ref; falling back to full scan")
            return None
        return (before, after)
    log.info(
        "Event '%s' has no diff range; falling back to full scan",
        event_name or "<unset>",
    )
    return None


def _segments_match(path_parts: list[str], pattern_parts: list[str]) -> bool:
    if not pattern_parts:
        return not path_parts
    head, rest = pattern_parts[0], pattern_parts[1:]
    if head == "**":
        # Try consuming zero segments, then one-or-more.
        if _segments_match(path_parts, rest):
            return True
        return bool(path_parts) and _segments_match(path_parts[1:], pattern_parts)
    if not path_parts:
        return False
    if not fnmatch.fnmatchcase(path_parts[0], head):
        return False
    return _segments_match(path_parts[1:], rest)


def _is_audited_path(relpath: str) -> bool:
    norm = relpath.replace(os.sep, "/")
    parts = norm.split("/")
    return any(
        _segments_match(parts, pattern.split("/")) for pattern in _AUDITED_PATTERNS
    )


def _determine_changed_audited_files(
    event_name: str,
    event: dict[str, object],
    scan_path: Path,
) -> list[Path] | None:
    """Return the changed audited files inside `scan_path`, or `None`."""
    diff = _diff_range(event_name, event)
    if diff is None:
        return None
    base_sha, head_sha = diff

    # Best-effort fetch so the base SHA is reachable for the diff.
    subprocess.run(
        ["git", "fetch", "--no-tags", "--depth=1", "origin", base_sha],
        check=False,
        capture_output=True,
    )

    try:
        result = subprocess.run(
            [
                "git",
                "diff",
                "--name-only",
                "--diff-filter=ACMR",
                f"{base_sha}..{head_sha}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        log.warning(
            "git diff %s..%s failed (rc=%s, stderr=%r); falling back to full scan",
            base_sha,
            head_sha,
            exc.returncode,
            exc.stderr.strip() if exc.stderr else "",
        )
        return None

    scan_root = scan_path.resolve()
    files: list[Path] = []
    for raw in result.stdout.splitlines():
        relpath = raw.strip()
        if not relpath or not _is_audited_path(relpath):
            continue
        candidate = Path(relpath)
        try:
            candidate.resolve().relative_to(scan_root)
        except ValueError:
            continue
        if not candidate.is_file():
            continue
        files.append(candidate)
    return files


def _run_zizmor(
    binary: Path,
    user_targets: list[_ReportTarget],
    *,
    config_path: str,
    persona: str,
    files: list[Path] | None,
    scan_path: Path,
) -> dict[str, int]:
    """Run zizmor for each user-requested format and return a severity
    tally."""
    base_args: list[str] = [
        str(binary),
        "--no-exit-codes",
        "--persona",
        persona,
        "--quiet",
    ]
    base_args.extend(["--config", config_path])
    if files is None:
        base_args.append(str(scan_path))
    else:
        base_args.extend(str(p) for p in files)

    user_json = next((t for t in user_targets if t.fmt == "json"), None)
    user_sarif = next((t for t in user_targets if t.fmt == "sarif"), None)
    internal_tally: _ReportTarget | None = None
    if user_json is not None or user_sarif is not None:
        runs: list[_ReportTarget] = list(user_targets)
    else:
        internal_tally = _ReportTarget(fmt="json", path=Path(_INTERNAL_TALLY_PATH))
        runs = [*user_targets, internal_tally]

    try:
        for tgt in runs:
            cmd = [*base_args, "--format", tgt.fmt]
            log.info("Running: %s > %s", " ".join(cmd), tgt.path)
            try:
                completed = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                )
            except OSError as exc:
                raise RuntimeError(
                    f"zizmor invocation for format '{tgt.fmt}' failed to start: {exc}"
                ) from exc
            if completed.returncode != 0:
                raise RuntimeError(
                    f"zizmor exited unexpectedly with code {completed.returncode} "
                    f"for format '{tgt.fmt}'; stderr: "
                    f"{completed.stderr.strip() if completed.stderr else '<empty>'}"
                )
            try:
                tgt.path.write_text(completed.stdout, encoding="utf-8")
            except OSError as exc:
                raise RuntimeError(
                    f"Failed to write zizmor {tgt.fmt} report to {tgt.path}: {exc}"
                ) from exc

        # Enrich SARIF severity for the Security tab.
        if user_sarif is not None and user_sarif.path.is_file():
            _enrich_sarif_with_security_severity(user_sarif.path)

        if user_json is not None:
            return _tally_findings_by_severity(user_json.path)
        if user_sarif is not None:
            return _tally_findings_from_sarif(user_sarif.path)
        assert internal_tally is not None
        return _tally_findings_by_severity(internal_tally.path)
    finally:
        if internal_tally is not None:
            internal_tally.path.unlink(missing_ok=True)


def _enrich_sarif_with_security_severity(sarif_path: Path) -> None:
    """Inject `security-severity` into each SARIF result so the
    GitHub Security tab tiers zizmor findings the same way it tiers
    CodeQL.
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

    enriched = 0
    unknown = 0
    for run in data.get("runs") or []:
        if not isinstance(run, dict):
            continue
        for result in run.get("results") or []:
            if not isinstance(result, dict):
                continue
            props = result.get("properties")
            if not isinstance(props, dict):
                props = {}
                result["properties"] = props
            sev = str(props.get("zizmor/severity") or "").upper()
            score = _ZIZMOR_SECURITY_SEVERITY.get(sev)
            if score is None:
                unknown += 1
                continue
            props["security-severity"] = score
            enriched += 1

    if enriched == 0:
        log.debug(
            "SARIF severity enrichment: nothing to add (%d unknown) in %s",
            unknown,
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
        "SARIF severity enrichment: %d result(s) given security-severity "
        "(%d had unknown severity) in %s",
        enriched,
        unknown,
        sarif_path,
    )


def _tally_findings_by_severity(json_path: Path) -> dict[str, int]:
    """Read zizmor's JSON output and tally findings by severity."""
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Failed to read zizmor JSON tally at {json_path}: {exc}"
        ) from exc
    if not isinstance(data, list):
        raise RuntimeError(
            f"zizmor JSON tally at {json_path} is not a JSON array "
            f"(got {type(data).__name__}); is `--format=json` schema v1?"
        )
    counts: dict[str, int] = {sev: 0 for sev in _SEVERITY_ORDER}
    for issue in data:
        if not isinstance(issue, dict):
            continue
        determ = issue.get("determinations") or {}
        sev = str(determ.get("severity") or "UNKNOWN").upper()
        counts[sev] = counts.get(sev, 0) + 1
    return counts


def _tally_findings_from_sarif(sarif_path: Path) -> dict[str, int]:
    """Tally findings by severity from a SARIF report's `zizmor/severity`
    result properties (see `_enrich_sarif_with_security_severity`), so
    the tally doesn't require a separate zizmor JSON invocation.
    """
    try:
        with open(sarif_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Failed to read zizmor SARIF report at {sarif_path}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"zizmor SARIF report at {sarif_path} is not a JSON object")

    counts: dict[str, int] = {sev: 0 for sev in _SEVERITY_ORDER}
    for run in data.get("runs") or []:
        if not isinstance(run, dict):
            continue
        for result in run.get("results") or []:
            if not isinstance(result, dict):
                continue
            props = result.get("properties") or {}
            sev = str(props.get("zizmor/severity") or "UNKNOWN").upper()
            counts[sev] = counts.get(sev, 0) + 1
    return counts


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scan-mode",
        default=os.environ.get("ZIZMOR_SCAN_MODE", "changed"),
        choices=("changed", "all"),
        help=(
            "'changed' (default) audits only workflow/action/dependabot "
            "files modified by the calling event (PR commits or push "
            "range). 'all' audits every such file under --source-dir; "
            "zizmor's own --collect walker filters to workflows, "
            "composite actions, and dependabot configs automatically."
        ),
    )
    p.add_argument(
        "--report-formats",
        default=os.environ.get("ZIZMOR_REPORT_FORMATS", "sarif"),
        help=(
            "Comma-separated list of zizmor report formats. Allowed values: "
            f"{', '.join(sorted(_SUPPORTED_FORMATS))}."
        ),
    )
    p.add_argument(
        "--source-dir",
        default=os.environ.get("ZIZMOR_SOURCE_DIR", "."),
        help=(
            "Path to audit (default %(default)s). Set to a subdirectory of "
            "the checkout to restrict the audit to that subtree; the "
            "'changed' scan mode further restricts to only the "
            "workflow / action / dependabot files modified inside this "
            "path. The path must exist."
        ),
    )
    p.add_argument(
        "--severity-threshold",
        default=os.environ.get(
            "ZIZMOR_SEVERITY_THRESHOLD", _DEFAULT_SEVERITY_THRESHOLD
        ),
        choices=_SEVERITY_CHOICES,
        help=(
            "Minimum zizmor severity that fails the job. Zizmor reports "
            "every finding regardless of threshold so the uploaded "
            "reports keep the full picture; after the scan the script "
            "tallies findings by severity and exits 1 only if any are "
            "at or above this threshold. Zizmor categorises findings "
            "as INFORMATIONAL / LOW / MEDIUM / HIGH (no 'critical' "
            "tier). Default '%(default)s' fails only on HIGH findings; "
            "set 'informational' to fail on any finding."
        ),
    )
    p.add_argument(
        "--persona",
        default=os.environ.get("ZIZMOR_PERSONA", _DEFAULT_PERSONA),
        choices=_PERSONA_CHOICES,
        help=(
            "Zizmor audit persona. 'regular' (default) surfaces "
            "high-signal, actionable findings. 'pedantic' adds code "
            "smells suitable for cleanup PRs. 'auditor' surfaces "
            "everything zizmor knows about, including likely false "
            "positives -- intended for security reviews, not CI."
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
        config_path = _resolve_config_path()
        event = gha_load_github_event()
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        log.error("%s", exc)
        return 2

    if args.scan_mode == "all":
        files: list[Path] | None = None
    else:
        files = _determine_changed_audited_files(
            event_name=os.environ.get("GITHUB_EVENT_NAME", ""),
            event=event,
            scan_path=source_dir,
        )

    if files is None:
        log.info(
            "Zizmor scope: recursive audit of %s (zizmor filters to "
            "workflow / action / dependabot files itself)",
            source_dir,
        )
    elif files:
        log.info(
            "Zizmor scope: %d changed audited file(s) under %s",
            len(files),
            source_dir,
        )
        for f in files:
            log.info("  - %s", f)
    else:
        log.info(
            "Zizmor scope: no audited files changed under %s; nothing to scan",
            source_dir,
        )

    log.info(
        "Zizmor formats: %s",
        ", ".join(f"{t.fmt}->{t.path}" for t in targets),
    )
    log.info("Zizmor persona: %s", args.persona)
    log.info(
        "Zizmor fail threshold: severity >= %s (lower-severity findings "
        "still appear in reports)",
        args.severity_threshold.upper(),
    )

    sarif_target = next((t for t in targets if t.fmt == "sarif"), None)
    non_sarif = [t for t in targets if t.fmt != "sarif"]

    if files is not None and not files:
        gha_set_output({"sarif_path": "", "non_sarif_paths": ""})
        return 0

    try:
        binary = _ensure_zizmor()
        counts = _run_zizmor(
            binary,
            targets,
            config_path=config_path,
            persona=args.persona,
            files=files,
            scan_path=source_dir,
        )
        gha_set_output(
            {
                "sarif_path": (
                    str(sarif_target.path)
                    if sarif_target is not None and sarif_target.path.is_file()
                    else ""
                ),
                "non_sarif_paths": "\n".join(
                    str(t.path) for t in non_sarif if t.path.is_file()
                ),
            }
        )
    except RuntimeError as exc:
        log.error("%s", exc)
        _emit_non_sarif_reports(non_sarif)
        return 2

    threshold = args.severity_threshold.upper()
    threshold_idx = _SEVERITY_ORDER.index(threshold)
    failing = sum(counts[sev] for sev in _SEVERITY_ORDER[threshold_idx:])
    summary = ", ".join(f"{sev}={counts[sev]}" for sev in _SEVERITY_ORDER)
    extra = {sev: n for sev, n in counts.items() if sev not in _SEVERITY_ORDER and n}
    if extra:
        summary += ", " + ", ".join(f"{sev}={n}" for sev, n in extra.items())
    log.info("Zizmor findings: %s", summary)

    _emit_non_sarif_reports(non_sarif)

    if failing:
        log.error(
            "zizmor reported %d finding(s) at severity >= %s; failing the job "
            "(see report artifacts for the full set, including lower-severity "
            "findings that didn't trigger the threshold)",
            failing,
            threshold,
        )
        return 1
    log.info(
        "All findings below threshold '%s'; passing (lower-severity findings "
        "are still listed in the uploaded reports)",
        threshold,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
