# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
AMD GPU Family Matrix and runner selection utilities for GitHub workflows.

Architecture:
- SUPPORTED BUILD ARCHITECTURES are defined locally in this file (TheRock repo).
  This is the source of truth for what GPU families can be built.
- CI RUNNER CONFIGURATION is loaded from external config when available
  (https://github.com/ROCm/therock-ci-config). This only provides runner labels
  (test-runs-on, etc.) for where tests can actually be executed.

The external config overlays runner labels onto the local architecture definitions.
A family can exist for building even without CI runners configured for testing.

For presubmit, postsubmit and nightly family selection:

- presubmit runs the targets from presubmit dictionary on pull requests
- postsubmit runs the targets from presubmit and postsubmit dictionaries on pushes to main branch
- nightly runs targets from presubmit, postsubmit and nightly dictionaries

TODO(#2200): clarify AMD GPU family selection
"""

#############################################################################################
# NOTE: when doing changes here, also check that they are done in new_amdgpu_family_matrix.py
#############################################################################################

import copy
import os
import random
import sys
from pathlib import Path


def _log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def load_external_runner_config() -> dict | None:
    """Load external CI runner config from CI_CONFIG_PATH if set.

    The CI config API lives in therock-ci-config repo, which is checked out
    to CI_CONFIG_PATH. Returns None if CI_CONFIG_PATH is not set or config
    doesn't exist.

    The external config only provides runner labels (test-runs-on, etc.),
    not the full family definitions. Build architecture support is defined
    locally in this file.
    """
    ci_config_path = os.environ.get("CI_CONFIG_PATH", "").strip()
    if not ci_config_path:
        _log("CI_CONFIG_PATH not set, using local runner definitions")
        return None
    config_path = Path(ci_config_path)
    sys.path.insert(0, str(config_path))
    try:
        from ci_config_api import get_gpu_runner_labels, load_runner_config
    except ImportError:
        _log(f"CI config API not found at {ci_config_path}, using local fallback")
        return None
    try:
        raw_config = load_runner_config(config_path)
    except Exception as e:
        _log(f"Failed to load CI config from {ci_config_path}: {e}")
        return None
    # Add runner_labels for _overlay_runner_config (extracted from gpu_runner_labels)
    raw_config["runner_labels"] = get_gpu_runner_labels(raw_config)
    _log(f"Loaded external runner config from {ci_config_path}")
    return raw_config


def is_asan():
    """Determines if this is an ASAN build using BUILD_VARIANT env var."""
    BUILD_VARIANT = os.getenv("BUILD_VARIANT", "")
    return BUILD_VARIANT == "asan"


def select_weighted_label(labels_config: list[dict], context_name: str) -> str:
    """Select a runner label based on weighted random selection."""
    rand_val = random.random()
    cumulative = 0.0
    for config in labels_config:
        cumulative += config["weight"]
        if rand_val < cumulative:
            print(
                f"  {context_name}: selected runner (weight={config['weight']}): "
                f"{config['label']}"
            )
            return config["label"]
    # Fallback to last label if rounding errors
    selected = labels_config[-1]
    print(
        f"  {context_name}: selected runner (weight={selected['weight']}): "
        f"{selected['label']}"
    )
    return selected["label"]


# Build runner configuration for Linux builds
# Uses weighted distribution: 100% AWS
# Sanitizer builds (asan/tsan) use ramdisk variants (100% Azure, no AWS yet)
BUILD_RUNNER_LABELS = {
    "linux": {
        "default": [
            {"label": "aws-linux-scale-rocm-prod", "weight": 1.0},
        ],
        "sanitizer": [
            {"label": "azure-linux-scale-rocm-heavy-ramdisk", "weight": 1.0},
        ],
    },
    "windows": {
        "default": [
            {"label": "azure-windows-scale-rocm", "weight": 1.0},
        ],
    },
}


def select_build_runner(platform: str, build_variant: str) -> str:
    """Select a build runner label based on platform and build variant."""
    build_runner_labels = get_build_runner_labels()
    if platform not in build_runner_labels:
        # Platform not configured for weighted selection, return default
        print(f"  No build runner config for platform {platform}, using default")
        return ""

    platform_config = build_runner_labels[platform]

    # Use sanitizer runners for asan/tsan builds
    if "san" in build_variant:
        labels_config = platform_config.get("sanitizer", platform_config["default"])
        context_name = f"build-runner ({platform}, {build_variant})"
    else:
        labels_config = platform_config["default"]
        context_name = f"build-runner ({platform})"

    return select_weighted_label(labels_config, context_name)


all_build_variants = {
    "linux": {
        "release": {
            "build_variant_label": "release",
            "build_variant_suffix": "",
            # TODO: Enable linux-release-package once capacity and rccl link
            # issues are resolved. https://github.com/ROCm/TheRock/issues/1781
            # "build_variant_cmake_preset": "linux-release-package",
            "build_variant_cmake_preset": "",
        },
        # full ASAN builds are run on nightly
        "asan": {
            "build_variant_label": "asan",
            "build_variant_suffix": "asan",
            "build_variant_cmake_preset": "linux-release-asan",
        },
        # host ASAN builds are run on nightly, with intent to run on presubmit and postsubmit
        # host ASAN detects memory errors on host code (excluding kernel binaries), while ASAN sanitizes everything
        "host-asan": {
            "build_variant_label": "host-asan",
            "build_variant_suffix": "host-asan",
            "build_variant_cmake_preset": "linux-release-host-asan",
        },
        "tsan": {
            "build_variant_label": "tsan",
            "build_variant_suffix": "tsan",
            "build_variant_cmake_preset": "linux-release-tsan",
        },
    },
    "windows": {
        "release": {
            "build_variant_label": "release",
            "build_variant_suffix": "",
            "build_variant_cmake_preset": "windows-release",
        },
    },
}

"""
amdgpu_family_info_matrix dictionary fields:
- test-runs-on: (required) GitHub runner label for this architecture
- test-runs-on-labels: (optional) List of runner label configs for load balancing across pools.
    Each entry is a dict with "label" and "weight" (probability 0.0-1.0). Weights must sum to 1.0.
    When present, overrides test-runs-on for runner selection.
- test-runs-on-multi-gpu: (optional) GitHub runner label for multi-GPU tests for this architecture
- test-runs-on-multi-gpu-labels: (optional) List of runner label configs for multi-GPU load balancing.
    Same format as test-runs-on-labels.
- benchmark-runs-on: (optional) GitHub runner label for benchmarks for this architecture
- test-runs-on-kernel: (optional) dict of kernel-specific runner labels, keyed by kernel type (e.g. "oem")
- family: (required) AMD GPU family name, used for test selection and artifact fetching
- fetch-gfx-targets: (required) list of gfx targets to fetch split test artifacts for (e.g. ["gfx942", "gfx942:xnack+"])
- build_variants: (optional) list of build variants to build for this architecture (e.g. ["release", "asan"])
- bypass_tests_for_releases: (optional) if enabled, bypass tests for release builds (e.g. by skipping test steps in the workflow, or by not running tests on release builds in test scripts)
- sanity_check_only_for_family: (optional) if enabled, only run sanity check tests for this architecture
- run-full-tests-only: (optional) if enabled, only run full tests for this architecture
- nightly_check_only_for_family (optional): if enabled, only run CI nightly tests for this architecture
- submodule_bump_tests_only (optional): if enabled, only run tests when submodule changes are detected or on workflow_dispatch (builds always run)
"""
# The 'presubmit' matrix runs on 'pull_request' triggers (on all PRs).
amdgpu_family_info_matrix_presubmit = {
    "gfx94x": {
        "linux": {
            # TODO: Remove multi-label config once we get dedicated set of machines
            # As we are bringing up mi325, we are using a multi-label configuration to distribute load
            "test-runs-on": "linux-gfx942-1gpu-ccs-csp-ossci-rocm",
            "test-runs-on-labels": [
                {
                    "label": "linux-gfx942-1gpu-ccs-ossci-rocm",
                    "weight": 0.1,
                },  # ccs (5)
                {
                    "label": "linux-gfx942-1gpu-ccs-csp-ossci-rocm",
                    "weight": 0.8,
                },  # ccs-csp (28)
                {
                    "label": "linux-gfx942-1gpu-ossci-rocm",
                    "weight": 0.1,
                },  # vultr (5)
            ],
            # TODO(#3433): Remove sandbox label once ASAN tests are passing
            "test-runs-on-sandbox": "linux-mi325-gpu-rocm-cpu-sandbox",
            "test-runs-on-multi-gpu": "linux-gfx942-8gpu-ossci-rocm",
            "test-runs-on-multi-gpu-labels": [
                {
                    "label": "linux-gfx942-8gpu-ossci-rocm",
                    "weight": 1.0,
                },  # (10)
            ],
            # TODO(#2754): Add new benchmark-runs-on runner for benchmarks
            "benchmark-runs-on": "linux-gfx942-8gpu-ossci-rocm",
            "family": "gfx94X-dcgpu",
            # Individual GPU target(s) on the test runner, for fetching split artifacts.
            # TODO(#3444): ASAN variants may need xnack suffix expansion (e.g. gfx942:xnack+).
            "fetch-gfx-targets": ["gfx942"],
            "build_variants": ["release", "asan", "host-asan", "tsan"],
        }
    },
    "gfx110x": {
        "linux": {
            "test-runs-on": "linux-gfx110X-gpu-rocm",
            "family": "gfx110X-all",
            "fetch-gfx-targets": [],
            "bypass_tests_for_releases": True,
            "build_variants": ["release"],
            "nightly_check_only_for_family": True,
        },
        "windows": {
            "test-runs-on": "windows-gfx110X-gpu-rocm",
            "family": "gfx110X-all",
            "fetch-gfx-targets": ["gfx1100", "gfx1101", "gfx1102", "gfx1103"],
            "bypass_tests_for_releases": True,
            "build_variants": ["release"],
        },
    },
    "gfx1151": {
        "linux": {
            "test-runs-on": "linux-gfx1151-gpu-rocm",
            "test-runs-on-kernel": {
                "oem": "linux-strix-halo-gpu-rocm-oem",
            },
            "family": "gfx1151",
            "fetch-gfx-targets": ["gfx1151"],
            "bypass_tests_for_releases": True,
            "build_variants": ["release"],
            "nightly_check_only_for_family": True,
        },
        "windows": {
            "test-runs-on": "windows-gfx1151-gpu-rocm",
            # TODO(#2754): Add new benchmark-runs-on runner for benchmarks
            "benchmark-runs-on": "windows-gfx1151-gpu-rocm",
            "family": "gfx1151",
            "fetch-gfx-targets": ["gfx1151"],
            "build_variants": ["release"],
            # TODO(#3299): Re-enable quick tests once capacity is available for Windows gfx1151
            "nightly_check_only_for_family": True,
        },
    },
    "gfx120x": {
        "linux": {
            "test-runs-on": "linux-gfx120X-gpu-rocm",
            "family": "gfx120X-all",
            "fetch-gfx-targets": ["gfx1200", "gfx1201"],
            "bypass_tests_for_releases": True,
            "build_variants": ["release"],
            "nightly_check_only_for_family": True,
        },
        "windows": {
            "test-runs-on": "windows-gfx120X-gpu-rocm",
            "family": "gfx120X-all",
            "fetch-gfx-targets": ["gfx1200", "gfx1201"],
            "bypass_tests_for_releases": True,
            "build_variants": ["release"],
            "nightly_check_only_for_family": True,
        },
    },
}


# The 'postsubmit' matrix runs on 'push' triggers (for every commit to the default branch).
amdgpu_family_info_matrix_postsubmit = {
    "gfx950": {
        "linux": {
            "test-runs-on": "linux-gfx950-1gpu-ccs-ossci-rocm",
            "test-runs-on-multi-gpu": "linux-gfx950-8gpu-ccs-ossci-rocm",
            "family": "gfx950-dcgpu",
            "fetch-gfx-targets": ["gfx950"],
            "build_variants": ["release", "asan", "tsan"],
            # Only run tests on submodule bumps (builds always run)
            "submodule_bump_tests_only": True,
        }
    },
}

# The 'nightly' matrix runs on 'schedule' triggers.
amdgpu_family_info_matrix_nightly = {
    "gfx900": {
        "linux": {
            # Disabled due to hardware availability
            "test-runs-on": "",
            "family": "gfx900",
            "fetch-gfx-targets": [],
            "sanity_check_only_for_family": True,
            "build_variants": ["release"],
        },
        "windows": {
            "test-runs-on": "",
            "family": "gfx900",
            "fetch-gfx-targets": [],
            "build_variants": ["release"],
        },
    },
    # gfx906/908/90a split into separate families - each has different instruction
    # support (e.g., fp8 variants, WMMA) so CK/MIOpen need to build/test individually.
    "gfx906": {
        "linux": {
            # Disabled due to hardware availability
            "test-runs-on": "",
            "family": "gfx906",
            "fetch-gfx-targets": [],
            "sanity_check_only_for_family": True,
            "build_variants": ["release"],
        },
        # TODO(#1927): Resolve error generating file `torch_hip_generated_int4mm.hip.obj`, to enable PyTorch builds
        "windows": {
            "test-runs-on": "",
            "family": "gfx906",
            "fetch-gfx-targets": [],
            "build_variants": ["release"],
        },
    },
    "gfx908": {
        "linux": {
            # Disabled due to hardware availability
            "test-runs-on": "",
            "family": "gfx908",
            "fetch-gfx-targets": [],
            "sanity_check_only_for_family": True,
            "build_variants": ["release"],
        },
        "windows": {
            "test-runs-on": "",
            "family": "gfx908",
            "fetch-gfx-targets": [],
            "build_variants": ["release"],
        },
    },
    "gfx90a": {
        "linux": {
            "test-runs-on": "linux-gfx90a-gpu-rocm",
            "family": "gfx90a",
            "fetch-gfx-targets": ["gfx90a"],
            "build_variants": ["release"],
            "nightly_check_only_for_family": True,
        },
        "windows": {
            "test-runs-on": "",
            "family": "gfx90a",
            "fetch-gfx-targets": [],
            "build_variants": ["release"],
        },
    },
    "gfx101x": {
        "linux": {
            "test-runs-on": "",
            "family": "gfx101X-dgpu",
            "fetch-gfx-targets": [],
            "build_variants": ["release"],
        },
        "windows": {
            "test-runs-on": "",
            "family": "gfx101X-dgpu",
            "fetch-gfx-targets": [],
            "build_variants": ["release"],
        },
    },
    "gfx103x": {
        "linux": {
            "test-runs-on": "linux-gfx1030-gpu-rocm",
            "family": "gfx103X-all",
            "fetch-gfx-targets": ["gfx1030"],
            "build_variants": ["release"],
            "nightly_check_only_for_family": True,
        },
        "windows": {
            "test-runs-on": "windows-gfx1030-gpu-rocm",
            "family": "gfx103X-all",
            "fetch-gfx-targets": [],
            "build_variants": ["release"],
            "nightly_check_only_for_family": True,
        },
    },
    "gfx1150": {
        "linux": {
            "test-runs-on": "linux-gfx1150-gpu-rocm",
            "family": "gfx1150",
            "fetch-gfx-targets": [],
            "build_variants": ["release"],
            "nightly_check_only_for_family": True,
        },
        "windows": {
            "test-runs-on": "",
            "family": "gfx1150",
            "fetch-gfx-targets": [],
            "build_variants": ["release"],
        },
    },
    "gfx1152": {
        "linux": {
            "test-runs-on": "",
            "family": "gfx1152",
            "fetch-gfx-targets": [],
            "build_variants": ["release"],
        },
        "windows": {
            "test-runs-on": "",
            "family": "gfx1152",
            "fetch-gfx-targets": [],
            "build_variants": ["release"],
        },
    },
    "gfx1153": {
        "linux": {
            "test-runs-on": "linux-gfx1153-gpu-rocm",
            "family": "gfx1153",
            "fetch-gfx-targets": [],
            "build_variants": ["release"],
            "nightly_check_only_for_family": True,
        },
        "windows": {
            "test-runs-on": "",
            "family": "gfx1153",
            "fetch-gfx-targets": [],
            "build_variants": ["release"],
        },
    },
    "gfx125x": {
        "linux": {
            # No hardware available for testing yet; build-only.
            # PyTorch builds are included — workflow_dispatch can be used
            # to trigger manually; nightly schedule runs both ROCm stack
            # and PyTorch builds.
            "test-runs-on": "",
            "family": "gfx125X-dcgpu",
            "fetch-gfx-targets": [],
            "build_variants": ["release"],
        },
    },
}


def _get_local_families_for_trigger_types(trigger_types) -> dict:
    """Returns combined family matrix from local definitions for trigger types."""
    result = {}
    matrix_map = {
        "presubmit": amdgpu_family_info_matrix_presubmit,
        "postsubmit": amdgpu_family_info_matrix_postsubmit,
        "nightly": amdgpu_family_info_matrix_nightly,
    }

    for trigger_type in trigger_types:
        if trigger_type in matrix_map:
            for family_name, family_config in matrix_map[trigger_type].items():
                result[family_name] = family_config

    return result


def _extract_runner_labels_from_v1(external_config: dict) -> dict:
    """Extract runner labels from V1 gpu_families format for backward compatibility."""
    runner_labels: dict = {}
    gpu_families = external_config.get("gpu_families", {})

    # V1 gpu_families mixes build config (family, build_variants, etc.) with runner
    # config. We must filter to only extract runner keys, otherwise we'd overwrite
    # local build definitions. This filter is only needed for V1 backward compat.
    runner_keys = {
        "test-runs-on",
        "test-runs-on-labels",
        "test-runs-on-sandbox",
        "test-runs-on-multi-gpu",
        "test-runs-on-multi-gpu-labels",
        "test-runs-on-kernel",
        "benchmark-runs-on",
    }

    for _trigger, families in gpu_families.items():
        for family_name, family_config in families.items():
            if family_name not in runner_labels:
                runner_labels[family_name] = {}
            for platform, platform_config in family_config.items():
                if platform not in runner_labels[family_name]:
                    runner_labels[family_name][platform] = {}
                for key, value in platform_config.items():
                    if key in runner_keys and value:
                        runner_labels[family_name][platform][key] = value

    return runner_labels


def _overlay_runner_config(families: dict, external_config: dict) -> dict:
    """Overlay external runner configuration onto local family definitions.

    Merges runner labels from external config onto local build definitions.
    Only runner-related keys (test-runs-on, build-runs-on) are overlaid;
    build architecture support (amdgpu_targets, build_variants) stays local.

    Example transformation for gfx94X:
        Before (local):
            "gfx94X": {"linux": {"test-runs-on": "", "amdgpu_targets": "gfx942"}}
        After (with external config):
            "gfx94X": {"linux": {"test-runs-on": "linux-mi300-gpu", "amdgpu_targets": "gfx942"}}
    """
    # V2 format: runner_labels directly available
    # V1 format: extract from gpu_families
    runner_labels = external_config.get("runner_labels", {})
    if not runner_labels:
        runner_labels = _extract_runner_labels_from_v1(external_config)

    if not runner_labels:
        return families

    result = copy.deepcopy(families)
    _log("Overlaying runner labels from external config:")

    for family_name, family_config in result.items():
        if family_name not in runner_labels:
            continue

        external_family = runner_labels[family_name]
        for platform in ["linux", "windows"]:
            if platform not in family_config:
                continue
            if platform not in external_family:
                continue

            # Overlay all keys from runner_labels onto local definitions
            external_platform = external_family[platform]
            for key, value in external_platform.items():
                old_value = family_config[platform].get(key)
                family_config[platform][key] = value
                if old_value != value:
                    _log(f"  {family_name}/{platform}: {key} = {value}")

    return result


def get_all_families_for_trigger_types(trigger_types):
    """Returns combined family matrix for the specified trigger types.

    Local definitions (this file) are the source of truth for build architecture
    support. External config (therock-ci-config) provides runner labels which
    are overlaid onto local definitions when available.
    """
    # Always start with local definitions - source of truth for build support
    result = _get_local_families_for_trigger_types(trigger_types)

    # Overlay external runner config if available
    external_config = load_external_runner_config()
    if external_config is not None:
        result = _overlay_runner_config(result, external_config)

    return result


def get_build_runner_labels():
    """Returns build runner label configuration.

    Attempts to load external config from CI_CONFIG_PATH. Falls back to local
    definitions if external config is unavailable.
    """
    external_config = load_external_runner_config()

    if external_config is not None:
        return external_config.get("build_runners", {})

    return BUILD_RUNNER_LABELS
