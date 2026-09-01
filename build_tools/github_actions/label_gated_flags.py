#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Label-gated cmake flags for multi-arch CI.

A label on an external repository's pull request makes that pull request's
multi-arch build add a non-default -DTHEROCK_FLAG_* option. This is how a
default-off code path gets real CI coverage, including GPU tests.

The flag is appended to `extra_cmake_options` in the `external_repo` payload,
which the build workflows already splice onto the cmake line for every stage.

Only the labels present on the pull request matter, never which label changed,
so removing a label needs no special case. Labels are read from the event
payload (GITHUB_EVENT_PATH): no token, and it works for forks.

See docs/development/ci_behavior_manipulation.md#label-gated-cmake-flags.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Iterable, Mapping, Optional

# Labels that turn TheRock cmake flags on for a single pull request.
#
# Ships empty: an entry here changes what CI builds. A label has an effect if
# and only if it is a key in this map; no naming convention is enforced.
#
# The outer key is the external repository's short name, as it appears in
# REPO_CONFIGS in detect_external_repo_config.py; the inner key is the label;
# the value is the options it turns on. A rocm-systems label never picks up a
# rocm-libraries entry.
#
# The flag must already be declared in FLAGS.cmake with
# `therock_declare_flag(... SUB_PROJECTS <proj>)`, or it never reaches the
# subproject.
#
# Example entry (nothing is mapped today):
#     "rocm-libraries": {
#         "ci:miopen-hipdnn-wrapper": [
#             "-DTHEROCK_FLAG_MIOPEN_ENABLE_HIPDNN_WRAPPER=ON",
#         ],
#     },
LABEL_GATED_FLAGS: dict[str, dict[str, list[str]]] = {}

# Only -DTHEROCK_FLAG_<NAME>=ON|OFF. The pattern allows no whitespace and no
# quotes, so a token cannot carry a shell metacharacter into the `run:` block it
# is interpolated into. The prefix matters too: THEROCK_FLAG_<NAME> is the knob
# therock_declare_flag creates, and options in other namespaces (THEROCK_ENABLE_*)
# are overwritten by TheRock's own generated ones.
# See docs/development/ci_behavior_manipulation.md#label-gated-cmake-flags.
FLAG_RE = re.compile(r"^-DTHEROCK_FLAG_[A-Z][A-Z0-9_]*=(ON|OFF)$")


def validate_label_gated_flags(
    mapping: Optional[Mapping[str, object]] = None,
) -> None:
    """Raise ValueError if the label map is malformed."""
    if mapping is None:
        mapping = LABEL_GATED_FLAGS
    if not isinstance(mapping, dict):
        raise ValueError("LABEL_GATED_FLAGS must be a dict")

    for repo_name, labels in mapping.items():
        if not isinstance(repo_name, str) or not repo_name:
            raise ValueError(
                f"LABEL_GATED_FLAGS keys must be non-empty repository names, "
                f"got {repo_name!r}"
            )
        if not isinstance(labels, dict):
            raise ValueError(
                f"LABEL_GATED_FLAGS['{repo_name}'] must be a dict of label -> flags, "
                f"got {type(labels).__name__}"
            )
        for label, flags in labels.items():
            if not isinstance(label, str) or not label:
                raise ValueError(
                    f"LABEL_GATED_FLAGS['{repo_name}'] keys must be non-empty "
                    f"strings, got {label!r}"
                )
            # A bare string here would iterate character by character and
            # produce nonsense flags, so require a list explicitly.
            if not isinstance(flags, list):
                raise ValueError(
                    f"LABEL_GATED_FLAGS['{repo_name}']['{label}'] must be a list of "
                    f"strings, got {type(flags).__name__}"
                )
            if not flags:
                raise ValueError(
                    f"LABEL_GATED_FLAGS['{repo_name}']['{label}'] must not be empty"
                )
            for flag in flags:
                if not isinstance(flag, str) or not FLAG_RE.fullmatch(flag):
                    raise ValueError(
                        f"LABEL_GATED_FLAGS['{repo_name}']['{label}'] contains "
                        f"{flag!r}; entries must match -DTHEROCK_FLAG_<NAME>=ON|OFF"
                    )


def parse_labels_from_event(
    event_path: str | Path | None, event_name: str
) -> list[str]:
    """Return the pull request's label names from the GitHub event payload.

    Returns [] for anything that is not a pull_request event, so labels have no
    effect on dispatch, push, nightly or release runs. Also returns [] for any
    malformed payload rather than raising.
    """
    if event_name != "pull_request":
        return []
    if not event_path:
        return []
    path = Path(event_path)
    if not path.is_file():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return []
    if not isinstance(payload, dict):
        return []
    pull_request = payload.get("pull_request")
    if not isinstance(pull_request, dict):
        return []
    labels = pull_request.get("labels")
    if not isinstance(labels, list):
        return []
    names: list[str] = []
    for entry in labels:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def collect_flags(
    labels: Iterable[str],
    repo_name: str,
    mapping: Optional[Mapping[str, Mapping[str, list[str]]]] = None,
) -> tuple[list[str], list[str]]:
    """Return (matched_labels, flags) for the labels present on the PR.

    Iterates the map rather than the labels, so flag order comes from the map
    and not from GitHub's payload ordering. An unknown repo_name matches
    nothing.
    """
    if mapping is None:
        mapping = LABEL_GATED_FLAGS
    repo_mapping = mapping.get(repo_name) if repo_name else None
    if not repo_mapping:
        return [], []

    label_set = set(labels)
    matched: list[str] = []
    flags: list[str] = []
    # Flag name -> (value, label that set it), to catch two labels asking for
    # opposite values instead of letting map order silently decide.
    seen: dict[str, tuple[str, str]] = {}

    for label, label_flags in repo_mapping.items():
        if label not in label_set:
            continue
        matched.append(label)
        for flag in label_flags:
            name, _, value = flag.partition("=")
            previous = seen.get(name)
            if previous is None:
                seen[name] = (value, label)
                flags.append(flag)
            elif previous[0] != value:
                raise ValueError(
                    f"Labels '{previous[1]}' and '{label}' set {name} to "
                    f"conflicting values ({previous[0]} vs {value}); "
                    "remove one of the labels."
                )

    return matched, flags


def render_summary(
    *,
    event_name: str,
    repo_name: str,
    labels: list[str],
    matched: list[str],
    flags: list[str],
) -> str:
    """Render the step summary. Names the exact labels it acted on."""
    lines: list[str] = []
    lines.append("### Label-gated cmake flags")
    lines.append("")

    if event_name != "pull_request":
        lines.append(
            f"Event is `{event_name}`, not `pull_request`; labels are not read and "
            "the build uses the default configuration."
        )
        return "\n".join(lines) + "\n"

    if not repo_name:
        lines.append(
            "No external repository for this run; labels are not read and the build "
            "uses the default configuration."
        )
        return "\n".join(lines) + "\n"

    label_list = ", ".join(f"`{label}`" for label in labels) if labels else "_none_"
    matched_list = ", ".join(f"`{label}`" for label in matched) if matched else "_none_"
    flag_list = " ".join(f"`{flag}`" for flag in flags) if flags else "_none_"
    lines.append(f"- **External repository:** `{repo_name}`")
    lines.append(f"- **Labels on this pull request:** {label_list}")
    lines.append(f"- **Matched gating labels:** {matched_list}")
    lines.append(f"- **Appended cmake options:** {flag_list}")

    if flags:
        lines.append("")
        lines.append("> [!IMPORTANT]")
        lines.append(
            "> The flag-on build **replaces** the normal build; this run produces no "
            "flag-off signal. Stage reuse is forced off, so every stage is rebuilt."
        )
        lines.append(">")
        lines.append(
            "> Re-running an older run replays the label set from when that run was "
            "first triggered. Push a commit or re-apply the label instead of "
            "re-running a stale run."
        )

    return "\n".join(lines) + "\n"
