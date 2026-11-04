# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read the source override used to test an external repository with TheRock.

CI setup generates EXTERNAL_REPO_CONFIG to identify the repository and commit
under test and the TheRock submodule it replaces. Build jobs pass this JSON to
source fetching and manifest generation so both use the requested source instead
of TheRock's default pin. This module validates the shared source fields and
constructs the repository's clone URL for those consumers.

Example EXTERNAL_REPO_CONFIG:
    {
        "repository": "my-fork/rocm-libraries",
        "ref": "0123456789abcdef0123456789abcdef01234567",
        "submodule_path": "rocm-libraries"
    }
"""

import json
import re


def parse_and_validate_external_repo_config(raw: str | None) -> dict | None:
    """Decode and validate a serialized EXTERNAL_REPO_CONFIG value.

    ``raw`` is JSON text, typically supplied through the environment. None or an
    empty string returns None; otherwise require an object with nonblank string
    submodule_path/ref fields and preserve additional options.

    Malformed JSON raises json.JSONDecodeError; invalid fields raise ValueError.
    Validation does not establish a full commit SHA, matching submodule path, or
    remote reachability. external_repo_url validates the repository separately.
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


def external_repo_url(config: dict) -> str | None:
    """Return the HTTPS clone URL for config's GitHub owner/repo value.

    Preserve owner/name casing. A missing or null repository returns None; any
    other invalid value raises ValueError. This constructs a URL without
    checking remote existence or commit reachability.
    """
    repository = config.get("repository")
    if repository is None:
        return None
    if not isinstance(repository, str) or not re.fullmatch(
        r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository
    ):
        raise ValueError("EXTERNAL_REPO_CONFIG 'repository' must be owner/repo")
    return f"https://github.com/{repository}.git"
