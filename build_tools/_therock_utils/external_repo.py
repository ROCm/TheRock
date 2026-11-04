# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Source identity from the existing external-repository CI configuration."""

import json
import re


def parse_and_validate_external_repo_config(raw: str | None) -> dict | None:
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


def external_repo_url(config: dict) -> str | None:
    """Return the GitHub URL when the caller supplies a repository identity."""
    repository = config.get("repository")
    if repository is None:
        return None
    if not isinstance(repository, str) or not re.fullmatch(
        r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository
    ):
        raise ValueError("EXTERNAL_REPO_CONFIG 'repository' must be owner/repo")
    return f"https://github.com/{repository}.git"
