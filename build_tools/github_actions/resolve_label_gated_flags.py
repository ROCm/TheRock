#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolve label-gated cmake flags once per multi-arch CI run.

Runs as a step in setup_multi_arch.yml, before both consumers, so there is one
decision and one step summary per run. See label_gated_flags.py for what the
feature does.

Reads GITHUB_EVENT_NAME and GITHUB_EVENT_PATH (ambient on the runner; for an
external-repo run these describe that repository's pull request) and
EXTERNAL_REPO (the caller's `external_repo` JSON, whose `repository` field
picks the section of the map to use).

Sets the outputs `flags` (space-joined), `flags_active` (true/false) and
`matched_labels` (comma-joined).
"""

from __future__ import annotations

import json
import os
import sys

from github_actions_api import gha_append_step_summary, gha_set_output
from label_gated_flags import (
    collect_flags,
    parse_labels_from_event,
    render_summary,
    validate_label_gated_flags,
)


def repo_name_from_external_repo(external_repo_json: str) -> str:
    """Derive the map's repository key from the `external_repo` JSON string.

    Mirrors detect_external_repo_config.py: the short name, lowercased
    ("ROCm/ROCgdb" -> "rocgdb"). Returns "" for a missing or unparseable
    payload; an unknown repository simply matches no labels.
    """
    external_repo_json = (external_repo_json or "").strip()
    if not external_repo_json:
        return ""
    try:
        external_repo = json.loads(external_repo_json)
    except json.JSONDecodeError as e:
        print(f"Warning: failed to parse EXTERNAL_REPO: {e}", file=sys.stderr)
        return ""
    if not isinstance(external_repo, dict):
        return ""
    repository = external_repo.get("repository", "")
    if not isinstance(repository, str):
        return ""
    return repository.split("/")[-1].strip().lower()


def main() -> int:
    # Unconditional, so a malformed map fails every run rather than only the
    # runs carrying the offending label.
    validate_label_gated_flags()

    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    event_path = os.environ.get("GITHUB_EVENT_PATH", "")
    repo_name = repo_name_from_external_repo(os.environ.get("EXTERNAL_REPO", ""))

    labels = parse_labels_from_event(event_path, event_name)
    matched, flags = collect_flags(labels, repo_name)

    print(f"Event: {event_name}", file=sys.stderr)
    print(f"External repository: {repo_name or '(none)'}", file=sys.stderr)
    print(f"Labels: {labels}", file=sys.stderr)
    print(f"Matched gating labels: {matched}", file=sys.stderr)
    print(f"Appended cmake options: {' '.join(flags) or '(none)'}", file=sys.stderr)

    gha_set_output(
        {
            "flags": " ".join(flags),
            "flags_active": "true" if flags else "false",
            "matched_labels": ",".join(matched),
        }
    )
    gha_append_step_summary(
        render_summary(
            event_name=event_name,
            repo_name=repo_name,
            labels=labels,
            matched=matched,
            flags=flags,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
