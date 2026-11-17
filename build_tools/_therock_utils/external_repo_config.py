# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Parse the external source configuration passed between TheRock CI jobs.

Data flow:
    1. A repository calling TheRock's reusable workflows supplies an external_repo
       JSON input, for example:

           {
               "repository": "ROCm/rocm-libraries",
               "ref": "0123456789abcdef0123456789abcdef01234567",
               "extra_cmake_options": "-DTHEROCK_FLAG_HIPKERNELPROVIDER_ENABLE_ROCKE=ON"
           }

       ref is the commit selected for the run, including the merge SHA for PRs.

    2. Early in CI setup, github_actions/detect_external_repo_config.py reads
       that input and resolves repository to its TheRock submodule path. Its
       config_json output adds submodule_path and preserves repository, ref,
       and caller build options, adding defaults for omitted options:

           {
               "repository": "ROCm/rocm-libraries",
               "submodule_path": "rocm-libraries",
               "ref": "0123456789abcdef0123456789abcdef01234567",
               "extra_cmake_options": "-DTHEROCK_FLAG_HIPKERNELPROVIDER_ENABLE_ROCKE=ON",
               "projects": "",
               "family_overrides": {}
           }

    3. Downstream workflows pass the resolved JSON as EXTERNAL_REPO_CONFIG.
       Scripts use this module to parse it and access the resolved submodule
       path, ref, and other configuration fields needed by each consumer.

This module validates the shared submodule_path and ref fields and preserves
repository and additional options without interpreting them. Repository-to-path
resolution happens during CI setup; consumers decide how to use the parsed
configuration.
"""

import json


def parse_and_validate_external_repo_config(raw: str | None) -> dict | None:
    """Decode and validate a serialized EXTERNAL_REPO_CONFIG value.

    ``raw`` is JSON text, typically supplied through the environment. None or an
    empty string returns None; otherwise require an object with nonblank string
    submodule_path/ref fields and preserve additional options.

    Malformed JSON raises json.JSONDecodeError; invalid fields raise ValueError.
    Validation does not establish a full commit SHA, matching submodule path, or
    remote reachability.
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
