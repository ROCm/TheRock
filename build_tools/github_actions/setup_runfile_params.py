#!/usr/bin/env python
"""Sets up parameters for the Linux runfile installer workflow.

This script reads workflow inputs from environment variables and computes
any derived values (e.g., reading rocm_version from version.json if not provided).

Environment variable inputs:
    * ROCM_VERSION: ROCm version (optional, reads from version.json if empty)
    * GFX_ARCHS: GFX architectures to build
    * PULL_AMDGPU: Version of amdgpu to package
    * PULL_TAG: Build date in YYYYMMDD format
    * PULL_RUN_ID: Workflow run ID
    * PULL_PKG: Base package name

Outputs written to GITHUB_OUTPUT:
    * rocm_version: The ROCm version
    * gfx_archs: GFX architectures (if not "all")
    * pull_amdgpu: AMDGPU version
    * pull_pkg: Package name
    * pull_tag: Build date (if provided)
    * pull_run_id: Run ID (if provided)
    * is_auto_detect_nightly_pull_run_id: "true" if pull_run_id or pull_tag need auto-detection

Example usage:
    python build_tools/github_actions/setup_runfile_params.py
"""

import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from github_actions_utils import gha_set_output


def get_rocm_version_from_file() -> str:
    """Read rocm-version from version.json."""
    version_file = Path(__file__).parent.parent.parent / "version.json"
    if not version_file.exists():
        print(f"ERROR: version.json not found at {version_file}")
        sys.exit(1)

    with open(version_file) as f:
        data = json.load(f)

    version = data.get("rocm-version")
    if not version:
        print("ERROR: rocm-version not found in version.json")
        sys.exit(1)

    return version


def main():
    # Read environment variables
    rocm_version = os.environ.get("ROCM_VERSION", "")
    gfx_archs = os.environ.get("GFX_ARCHS", "all")
    pull_amdgpu = os.environ.get("PULL_AMDGPU", "release,31.10")
    pull_tag = os.environ.get("PULL_TAG", "")
    pull_run_id = os.environ.get("PULL_RUN_ID", "")
    pull_pkg = os.environ.get("PULL_PKG", "amdrocm-core-sdk")

    output = dict()
    # Compute rocm_version if not provided
    if not rocm_version:
        rocm_version = get_rocm_version_from_file()
    print(f"Using ROCM_VERSION={rocm_version}")
    # gha_set_output({"rocm_version": rocm_version})

    output["rocm_version"] = rocm_version

    # Pass through pull_amdgpu and pull_pkg
    print(f"Using PULL_AMDGPU={pull_amdgpu}")
    # gha_set_output("pull_amdgpu", pull_amdgpu)

    output["pull_amdgpu"] = pull_amdgpu

    print(f"Using PULL_PKG={pull_pkg}")
    # gha_set_output("pull_pkg", pull_pkg)

    output["pull_pkg"] = pull_pkg

    output = { "rocm_version": rocm_version,
               "pull_amdgpu": pull_amdgpu,
               "pull_pkg": pull_pkg
             }

    # Handle gfx_archs if explicitly provided (not "all")
    if gfx_archs and gfx_archs != "all":
        print(f"Using GFX_ARCHS={gfx_archs}")
        # gha_set_output("gfx_archs", gfx_archs)

        output["gfx_archs"] = gfx_archs

    # Check if we need auto-detection
    is_auto_detect_nightly_pull_run_id = not (pull_run_id and pull_tag)

    if pull_run_id and pull_tag:
        # Both provided, output them directly
        print(f"Using provided PULL_RUN_ID={pull_run_id}")
        # gha_set_output("pull_run_id", pull_run_id)

        print(f"Using provided PULL_TAG={pull_tag}")
        # gha_set_output("pull_tag", pull_tag)

        # gha_set_output("is_auto_detect_nightly_pull_run_id", "false")

        output["pull_run_id"] = pull_run_id
        output["pull_tag"] = pull_tag
        output["is_auto_detect_nightly_pull_run_id"] = False
    else:
        # Signal that auto-detection is needed
        print("Auto-detection required for pull_run_id and/or pull_tag")
        # gha_set_output("is_auto_detect_nightly_pull_run_id", "true")

        output["is_auto_detect_nightly_pull_run_id"] = True

        # Pass through pull_tag if provided (fetch_nightly_run_id.py will use it)
        if pull_tag:
            print(f"Using provided PULL_TAG={pull_tag} for auto-detection")

    # Output whether gfx_archs needs auto-detection
    if gfx_archs == "all":
        print("GFX architectures will be auto-detected")
        # gha_set_output("is_auto_detect_gfx", "true")
        output["is_auto_detect_gfx"] = True
    else:
        # gha_set_output("is_auto_detect_gfx", "false")
        output["is_auto_detect_gfx"] = False
    
    gha_set_output(output)


if __name__ == "__main__":
    main()
