# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build-time adapter to the SDK's self-contained ownership authority.

Keep ownership data in the metadata shipped by the SDK, never in this adapter.
"""

import importlib.util
from pathlib import Path

_METADATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "packaging/python/templates/rocm/src/rocm_sdk/_dist_info.py"
)
_spec = importlib.util.spec_from_file_location("_sdk_target_metadata", _METADATA_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load SDK metadata from {_METADATA_PATH}")
_metadata = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metadata)

canonical_target = _metadata.canonical_target
package_owner = _metadata.package_owner
group_package_targets = _metadata.group_package_targets
