# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Omit unsupported strict BLAS and dependent artifacts in this bring-up branch."""

import os
from pathlib import Path
import re
import tomllib


def main():
    families = os.environ.get("AMDGPU_FAMILIES", "").replace(",", ";").split(";")
    if os.environ.get("STAGE_NAME") != "math-libs" or "gfx1250-strict" not in families:
        return
    path = Path("BUILD_TOPOLOGY.toml")
    source = path.read_text()
    artifacts = tomllib.loads(source)["artifacts"]
    excluded = {"blas"}
    while True:
        expanded = excluded | {
            name for name, spec in artifacts.items()
            if excluded.intersection(spec.get("artifact_deps", []))
        }
        if expanded == excluded:
            break
        excluded = expanded
    for name in sorted(excluded):
        source = re.sub(
            r"(?ms)^\[artifacts\." + re.escape(name) + r"\]\n.*?(?=^\[|\Z)",
            "", source,
        )
    path.write_text(source)
    print("Strict-only excluded artifacts: " + ", ".join(sorted(excluded)))


if __name__ == "__main__":
    main()
