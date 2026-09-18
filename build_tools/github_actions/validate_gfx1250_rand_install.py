#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check an installed RAND SDK subset without executing GPU workloads."""

import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys

import _rocm_sdk_libraries
from rocm_sdk import _devel, _dist_info

expected = json.loads(Path(sys.argv[1]).read_text())
selected = sys.argv[2]
if _dist_info.determine_target_family() != selected:
    raise ValueError(f"SDK did not preserve explicit selection {selected}")
distribution = _dist_info.ALL_PACKAGES["device"].get_dist_package_name(selected)
if distribution != "rocm-sdk-device-gfx1250":
    raise ValueError(f"Unexpected owner distribution: {distribution}")
metadata.distribution(distribution)
libraries = Path(_rocm_sdk_libraries.__file__).parent
devel = _devel.get_devel_root()
actual = {}
for path in libraries.rglob("*.kpack"):
    if not path.is_file():
        continue
    if path.name in actual:
        raise ValueError(f"Duplicate installed archive: {path}")
    with path.open("rb") as stream:
        actual[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    if not (devel / path.relative_to(libraries)).samefile(path):
        raise ValueError(f"Missing development link for {path}")
if actual != expected:
    raise ValueError(f"Installed archives differ: {actual}; expected {expected}")
print(
    json.dumps(
        {
            "selected_target": selected,
            "device_distribution": distribution,
            "archive_sha256": actual,
        },
        indent=2,
    )
)
