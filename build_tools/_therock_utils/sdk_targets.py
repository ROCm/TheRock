# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build adapter and SDK embedding for the shared rocm-bootstrap leaf module."""

import importlib.util
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
# TODO(#4865): Import rocm_bootstrap.package_metadata directly once bootstrap
# is available to TheRock packaging tools, replacing this source-path loader.
_METADATA_PATH = (
    Path(os.environ.get("THEROCK_ROCM_SYSTEMS_SOURCE_DIR", _ROOT / "rocm-systems"))
    / "python/rocm-bootstrap/python/rocm_bootstrap/package_metadata.py"
)
_spec = importlib.util.spec_from_file_location("_sdk_target_metadata", _METADATA_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load target metadata from {_METADATA_PATH}")
_metadata = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metadata)
if _metadata.ownership_data()["schema_version"] != 1:
    raise ValueError("Unsupported target ownership schema")

canonical_target = _metadata.canonical_target
package_owner = _metadata.package_owner
group_package_targets = _metadata.group_package_targets
architectural_family = _metadata.architectural_family
ownership_data = _metadata.ownership_data


def render_dist_info(template: Path) -> str:
    """Embed the shared ownership source into standalone SDK metadata."""
    contents = template.read_text()
    start = "# BEGIN SHARED TARGET METADATA\n"
    end = "# END SHARED TARGET METADATA\n"
    if contents.count(start) != 1 or contents.count(end) != 1:
        raise ValueError(f"Invalid target metadata embedding markers in {template}")
    before, rest = contents.split(start)
    _, after = rest.split(end)
    return before + start + _METADATA_PATH.read_text() + "\n" + end + after
