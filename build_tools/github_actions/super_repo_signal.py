# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Render the super-repo source and TheRock checkout used by CI."""


import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from html import escape

from github_actions_api import gha_append_step_summary


@dataclass(frozen=True)
class SuperRepoSignal:
    """Source and workflow references used for an external-repository build."""

    super_repo_repository: str
    requested_source_ref: str
    super_repo_source_sha: str
    therock_repository: str
    requested_therock_ref: str
    resolved_therock_sha: str
    overlay_path: str
    fetch_sources_args: str
    calling_workflow_ref: str
    reusable_workflow_path: str


def _parse_external_repo_config(raw_config: str) -> dict[str, object]:
    if not raw_config.strip():
        raise ValueError("EXTERNAL_REPO_CONFIG_JSON must contain a JSON object")

    try:
        parsed = json.loads(raw_config)
    except json.JSONDecodeError as exc:
        raise ValueError("EXTERNAL_REPO_CONFIG_JSON contains invalid JSON") from exc

    if not isinstance(parsed, dict):
        raise ValueError("EXTERNAL_REPO_CONFIG_JSON must contain a JSON object")

    return parsed


def _as_string(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_super_repo_signal(
    env: Mapping[str, str],
) -> SuperRepoSignal:
    """Load the external-repository signal from environment values."""

    config = _parse_external_repo_config(env.get("EXTERNAL_REPO_CONFIG_JSON", ""))

    return SuperRepoSignal(
        super_repo_repository=_as_string(config.get("repository")),
        requested_source_ref=_as_string(config.get("ref")),
        super_repo_source_sha=env.get(
            "SUPER_REPO_SOURCE_SHA",
            "",
        ).strip(),
        therock_repository=env.get(
            "THEROCK_REPOSITORY",
            "",
        ).strip(),
        requested_therock_ref=env.get(
            "THEROCK_REQUESTED_REF",
            "",
        ).strip(),
        resolved_therock_sha=env.get(
            "THEROCK_RESOLVED_SHA",
            "",
        ).strip(),
        overlay_path=_as_string(config.get("checkout_path")),
        fetch_sources_args=_as_string(config.get("fetch_sources_args")),
        calling_workflow_ref=env.get(
            "CALLING_WORKFLOW_REF",
            "",
        ).strip(),
        reusable_workflow_path=env.get(
            "REUSABLE_WORKFLOW_PATH",
            "",
        ).strip(),
    )


def _markdown_code(value: str) -> str:
    if not value:
        return "_not provided_"

    normalized = " ".join(value.replace("\r", "\n").splitlines())
    escaped = escape(
        normalized,
        quote=False,
    ).replace("|", "&#124;")

    return f"<code>{escaped}</code>"


def render_super_repo_signal(
    signal: SuperRepoSignal,
) -> str:
    """Render the source/ref signal as a Markdown job-summary section."""

    rows = (
        (
            "Super-repo repository",
            signal.super_repo_repository,
        ),
        (
            "Requested source ref",
            signal.requested_source_ref,
        ),
        (
            "Triggering super-repo SHA",
            signal.super_repo_source_sha,
        ),
        (
            "TheRock repository",
            signal.therock_repository,
        ),
        (
            "Requested TheRock ref",
            signal.requested_therock_ref,
        ),
        (
            "Resolved TheRock SHA",
            signal.resolved_therock_sha,
        ),
        (
            "Source overlay path",
            signal.overlay_path,
        ),
        (
            "Source fetch arguments",
            signal.fetch_sources_args,
        ),
        (
            "Calling workflow ref",
            signal.calling_workflow_ref,
        ),
        (
            "Reusable workflow path",
            signal.reusable_workflow_path,
        ),
    )

    lines = [
        "### Super-repo source/ref signal",
        "",
        "| Field | Value |",
        "|---|---|",
    ]

    lines.extend(f"| {name} | {_markdown_code(value)} |" for name, value in rows)

    return "\n".join(lines)


def main(
    env: Mapping[str, str] | None = None,
    *,
    append_summary: Callable[
        [str],
        None,
    ] = gha_append_step_summary,
) -> int:
    """Write the super-repository source/ref job summary."""

    effective_env = env if env is not None else os.environ
    signal = load_super_repo_signal(effective_env)
    append_summary(render_super_repo_signal(signal))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
