# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build adapter and SDK embedding for local target metadata.

Keep package_metadata.py identical to its counterpart in rocm-systems:
python/rocm-bootstrap/python/rocm_bootstrap/package_metadata.py.
The local copy lets packaging and CI tools run without a submodule checkout
or an installed rocm-bootstrap package.
"""

from pathlib import Path

from .package_metadata import (
    architectural_family,
    canonical_target,
    group_package_targets,
    ownership_data,
    package_owner,
)

_METADATA_PATH = Path(__file__).with_name("package_metadata.py")


def render_dist_info(template: Path) -> str:
    """Embed the local ownership source into standalone SDK metadata."""
    contents = template.read_text()
    start = "# BEGIN SHARED TARGET METADATA\n"
    end = "# END SHARED TARGET METADATA\n"
    if contents.count(start) != 1 or contents.count(end) != 1:
        raise ValueError(f"Invalid target metadata embedding markers in {template}")
    before, rest = contents.split(start)
    _, after = rest.split(end)
    return before + start + _METADATA_PATH.read_text() + "\n" + end + after
