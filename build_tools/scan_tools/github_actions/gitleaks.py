#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run gitleaks against the current repository checkout.

Exit codes:

* `0` - no leaks, clean run.
* `1` - gitleaks found leaks, or `--report-formats` was empty/unknown.
* `2` - input error: scan path missing, `gitleaks.toml` missing,
  `GITHUB_EVENT_PATH` malformed, or gitleaks itself errored.

Inputs come from CLI flags or matching `GITLEAKS_*` env vars set by the
workflow.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

THEROCK_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Add build_tools to path for github_actions imports.
sys.path.insert(0, str(THEROCK_DIR / "build_tools"))
from github_actions.github_actions_api import (  # noqa: E402
    gha_append_step_summary,
    gha_load_github_event,
    gha_set_output,
)

log = logging.getLogger(__name__)

# Keep in sync with the `report_formats` input in
# `.github/workflows/gitleaks.yml`.
_SUPPORTED_FORMATS: dict[str, str] = {
    "sarif": "sarif",
    "json": "json",
    "csv": "csv",
    "junit": "xml",
}
# Mirrored to the rocm-third-party-deps S3 bucket (see
# docs/development/git_chores.md) so CI doesn't depend on github.com.
# Original source: https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_linux_x64.tar.gz
_GITLEAKS_VERSION = "8.30.1"
_GITLEAKS_TARBALL_URL = (
    "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/"
    f"gitleaks_{_GITLEAKS_VERSION}_linux_x64.tar.gz"
)
_GITLEAKS_TARBALL_SHA256 = (
    "551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb"
)
_GITLEAKS_MAX_TARBALL_BYTES = 100 * 1024 * 1024  # 100 MiB guardrail
_CONFIG_PATH = "gitleaks.toml"
# Pin --exit-code to 1 so we can tell clean (0) from leaks (1) from a
# gitleaks error (>1).
_LEAK_EXIT_CODE = 1
_LEAK_SECURITY_SEVERITY_HIGH = "8.5"

# Null SHA-1 git uses for "no previous commit" (a newly created ref).
Z40 = "0" * 40


@dataclass(frozen=True)
class _ReportTarget:
    """A single `(format, on-disk path)` pair the runner will produce."""

    fmt: str
    path: Path


def _sha256_of(path: Path) -> str:
    """Return the SHA-256 of `path` as a lowercase hex string."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def get_gitleaks_binary() -> Path:
    """Return a verified gitleaks binary in RUNNER_TEMP/gitleaks-<ver>.

    Downloads and validates it if missing.
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
            written = 0
            chunk = resp.read(1024 * 1024)
            while chunk:
                if written + len(chunk) > _GITLEAKS_MAX_TARBALL_BYTES:
                    raise RuntimeError(
                        f"gitleaks tarball exceeds {_GITLEAKS_MAX_TARBALL_BYTES} bytes "
                        f"(source: {_GITLEAKS_TARBALL_URL})"
                    )
                out.write(chunk)
                written += len(chunk)
                chunk = resp.read(1024 * 1024)
        actual_sha = _sha256_of(tarball_path)
        if actual_sha != _GITLEAKS_TARBALL_SHA256:
            raise RuntimeError(
                f"gitleaks tarball SHA256 mismatch: expected "
                f"{_GITLEAKS_TARBALL_SHA256}, got {actual_sha} "
                f"(downloaded from {_GITLEAKS_TARBALL_URL})"
            )
        with tarfile.open(tarball_path, mode="r:gz") as tar:
            # filter="data" rejects unsafe members (traversal, abs paths, devices).
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
    installed_version = result.stdout.strip().lstrip("v")
    if installed_version != _GITLEAKS_VERSION:
        raise RuntimeError(
            f"gitleaks at {binary} reports version {installed_version!r}, "
            f"expected {_GITLEAKS_VERSION!r}"
        )
    log.info("Installed gitleaks %s at %s", installed_version, binary)
    return binary


def _parse_report_formats(raw: str) -> list[_ReportTarget]:
    """Parse comma-separated report formats into unique report targets."""
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


def _resolve_config_path() -> str:
    if not Path(_CONFIG_PATH).is_file():
        raise FileNotFoundError(
            f"gitleaks config not found at '{_CONFIG_PATH}'. "
            "Run from the repo root so the config is resolvable."
        )
    log.info("Using gitleaks config: %s", _CONFIG_PATH)
    return _CONFIG_PATH


def _determine_log_opts(scan_mode: str, event_name: str, event: dict[str, Any]) -> str:
    """Build the `--log-opts` value for `gitleaks detect`.

    Returns '' to scan the full history; otherwise returns a git range
    derived from the triggering event or raises when unavailable.
    """
    if scan_mode == "all":
        return ""

    if event_name == "pull_request_target":
        raise ValueError(
            "pull_request_target is not supported for scan_mode=changed. "
            "Use pull_request for untrusted PRs, or set scan_mode='all' "
            "for trusted post-merge/manual scans."
        )

    if event_name == "pull_request":
        pr = event["pull_request"]
        base_sha = pr["base"]["sha"]
        head_sha = pr["head"]["sha"]
        fetch_result = subprocess.run(
            ["git", "fetch", "--no-tags", "--depth=1", "origin", base_sha],
            check=False,
            capture_output=True,
            text=True,
        )
        if fetch_result.returncode != 0:
            log.warning(
                "git fetch of PR base %s exited %d: %s",
                base_sha,
                fetch_result.returncode,
                (fetch_result.stderr or "").strip() or "(no stderr)",
            )
        rev_parse = subprocess.run(
            ["git", "rev-parse", "--verify", f"{base_sha}^{{commit}}"],
            check=False,
            capture_output=True,
            text=True,
        )
        if rev_parse.returncode != 0:
            raise RuntimeError(
                f"PR base commit {base_sha} is not reachable in the local "
                "checkout (fetch failed and the commit isn't in the pack). "
                "Increase the checkout `fetch-depth` or ensure the base ref "
                "is fetchable."
            )
        return f"{base_sha}..{head_sha}"

    if event_name == "push":
        # GitHub guarantees `before` and `after` on push events.
        before = event["before"]
        after = event["after"]
        if before == Z40:
            log.info("Push created a new ref; falling back to full history scan")
            return ""
        return f"{before}..{after}"

    raise ValueError(
        f"Cannot derive a diff range for event "
        f"'{event_name or '<unset>'}'. Pass --scan-mode all "
        f"(or set scan_mode='all' in the workflow input) to scan the "
        f"full repository history."
    )


def _enrich_sarif_with_security_severity(sarif_path: Path) -> None:
    """Mark every gitleaks SARIF result as High severity for code scanning.

    Gitleaks leaves `level` and `security-severity` unset; we backfill to
    match GitHub's severity tiers. Pre-existing values are preserved.
    """
    try:
        with open(sarif_path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"SARIF file '{sarif_path}' is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"SARIF file '{sarif_path}' top-level must be a JSON object, "
            f"got {type(data).__name__}"
        )
    runs = data.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError(
            f"SARIF file '{sarif_path}' field 'runs' must be a list, "
            f"got {type(runs).__name__}"
        )
    if not runs:
        raise ValueError(
            f"SARIF file '{sarif_path}' has an empty 'runs' array; "
            "gitleaks should always emit at least one run. This usually "
            "indicates the scanner aborted before writing a real report."
        )

    levels_set_count = 0
    levels_kept_count = 0
    scores_set_count = 0
    scores_kept_count = 0
    for run_idx, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ValueError(
                f"SARIF file '{sarif_path}' runs[{run_idx}] must be an "
                f"object, got {type(run).__name__}"
            )
        results = run.get("results", [])
        if not isinstance(results, list):
            raise ValueError(
                f"SARIF file '{sarif_path}' runs[{run_idx}].results must "
                f"be a list, got {type(results).__name__}"
            )
        for res_idx, result in enumerate(results):
            if not isinstance(result, dict):
                raise ValueError(
                    f"SARIF file '{sarif_path}' "
                    f"runs[{run_idx}].results[{res_idx}] must be an "
                    f"object, got {type(result).__name__}"
                )
            if result.get("level") is None:
                result["level"] = "error"
                levels_set_count += 1
            else:
                levels_kept_count += 1
            props = result.setdefault("properties", {})
            if not isinstance(props, dict):
                raise ValueError(
                    f"SARIF file '{sarif_path}' "
                    f"runs[{run_idx}].results[{res_idx}].properties must "
                    f"be an object, got {type(props).__name__}"
                )
            if props.get("security-severity") is None:
                props["security-severity"] = _LEAK_SECURITY_SEVERITY_HIGH
                scores_set_count += 1
            else:
                scores_kept_count += 1

    if levels_set_count == 0 and scores_set_count == 0:
        log.debug(
            "SARIF severity enrichment: nothing to add (%d level preserved, "
            "%d score preserved) in %s",
            levels_kept_count,
            scores_kept_count,
            sarif_path,
        )
        return

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=sarif_path.parent,
            prefix=f"{sarif_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            json.dump(data, tmp, indent=2)
        os.replace(tmp_path, sarif_path)
    except OSError as exc:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to write enriched SARIF to '{sarif_path}': {exc}"
        ) from exc

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
    config_path: str,
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
            if not tgt.path.is_file():
                raise RuntimeError(
                    f"gitleaks exited {rc} but did not write the expected "
                    f"{tgt.fmt} report at '{tgt.path}'."
                )
            # Align SARIF with the GitHub Security tab's severity tiers
            # (gitleaks leaves `level` and `security-severity` unset).
            if tgt.fmt == "sarif":
                _enrich_sarif_with_security_severity(tgt.path)
            continue
        raise RuntimeError(
            f"gitleaks exited unexpectedly with code {rc} for format '{tgt.fmt}'"
        )
    return leaks_found


def _md_code_fence(content: str) -> str:
    """Return a backtick fence longer than any backtick run in `content`.

    Ensures markdown summaries stay intact even when reports contain backticks.
    """
    longest = max((len(m) for m in re.findall(r"`+", content)), default=0)
    return "`" * max(3, longest + 1)


def _emit_non_sarif_reports(non_sarif: list[_ReportTarget]) -> None:
    """Surface each non-SARIF report in the workflow run."""
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
        fence = _md_code_fence(content)
        summary_chunks.append(
            f"### Gitleaks report: `{path}`\n\n{fence}\n{content}\n{fence}"
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
            "event; requires a pull_request or push "
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

    try:
        config_path = _resolve_config_path()
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

    sarif_target = next((t for t in targets if t.fmt == "sarif"), None)
    non_sarif = [t for t in targets if t.fmt != "sarif"]
    gha_set_output(
        {
            "sarif_path": "" if sarif_target is None else str(sarif_target.path),
            "non_sarif_paths": "\n".join(str(t.path) for t in non_sarif),
        }
    )

    try:
        binary = get_gitleaks_binary()
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
