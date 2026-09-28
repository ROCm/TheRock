# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
This script determines what test configurations to run.

Outputs (written to $GITHUB_OUTPUT):
  - sanity_component: JSON object for the sanity component, always present as a
    prerequisite that must pass before other components are run. The
    ``test_runner`` field within this object is non-empty only on GPU runners,
    so callers can gate GPU-only steps on that field.
  - components: JSON array of component configs for the regular test matrix
    (excludes sanity, which is output separately above).
  - platform: lowercase OS name derived from RUNNER_OS.

Required environment variables:
  - RUNNER_OS (https://docs.github.com/en/actions/how-tos/writing-workflows/choosing-what-your-workflow-does/store-information-in-variables#detecting-the-operating-system)
"""

import argparse
import ast
import copy
import json
import logging
import os
import platform as platform_module
from pathlib import Path

from github_actions_api import *
from amdgpu_family_matrix import (
    get_all_families_for_trigger_types,
    select_weighted_label,
)
from test_matrix import (
    TEST_LABEL_GROUPS,
    _family_matches,
    test_matrix as _base_test_matrix,
)

logging.basicConfig(level=logging.INFO)

# Note: these paths are relative to the repository root. We could make that
# more explicit, or use absolute paths.
SCRIPT_DIR = Path("./build_tools/github_actions/test_executable_scripts")
OUTPUT_ARTIFACTS_DIR = Path(os.environ.get("OUTPUT_ARTIFACTS_DIR", "build"))


def _get_script_path(script_name: str) -> str:
    # Convert to posix (using `/` instead of `\\`) so test workflows can use
    # 'bash' as the shell on Linux and Windows.
    return (SCRIPT_DIR / script_name).as_posix()


def _get_artifact_path(artifact_path: str) -> str:
    # Convert to posix (using `/` instead of `\\`) so test workflows can use
    # 'bash' as the shell on Linux and Windows.
    return (OUTPUT_ARTIFACTS_DIR / artifact_path).as_posix()


# Base container options applied to all Linux containers
# --ipc host - Allows shared memory between host and container
# --user 0:0 - Running as root, by recommendation of GitHub: https://docs.github.com/en/actions/reference/workflows-and-actions/dockerfile-support#user
# --ulimit memlock=-1:-1 - Prevents memory allocation issues with ROCm inside container
# --ulimit nofile=1048576:1048576 - Increase open file limit for RCCL
# --security-opt seccomp=unconfined - enables memory mapping, and is recommended for containers running in HPC environments
_BASE_CONTAINER_OPTIONS = [
    "--ipc host",
    "--user 0:0",
    "--ulimit memlock=-1:-1",
    "--ulimit nofile=1048576:1048576",
    "--security-opt seccomp=unconfined",
]

# GPU-specific container options (only applied when linux_cpu_runner != True)
# --group-add video - Grants access to GPU video group
# --device /dev/kfd - AMD KFD device for GPU compute
# --device /dev/dri - Direct Rendering Infrastructure devices
# --group-add 993,992,110 - Additional GPU-related groups
# --env-file /etc/podinfo/gha-gpu-isolation-settings - Required for GPU isolation on OSSCI MIXXX runners
# -e ROCR_VISIBLE_DEVICES - Pass host's GPU isolation env var to container (used on ARC runners)
_GPU_CONTAINER_OPTIONS = [
    "--group-add video",
    "--device /dev/kfd",
    "--device /dev/dri",
    "--group-add 993",
    "--group-add 992",
    "--group-add 110",
    "--env-file /etc/podinfo/gha-gpu-isolation-settings",
    "-e ROCR_VISIBLE_DEVICES",
    "-e KUBE_CPU_REQUEST",
]


def _build_container_options(job_config: dict, platform: str) -> dict:
    """
    Build the final container_options string by concatenating base, GPU, and job-specific options.

    Args:
        job_config: The job configuration dictionary
        platform: The platform (e.g., "linux", "windows")

    Returns:
        The modified job_config with updated container_options
    """
    # Containers are Linux-only (test_component.yml gates container.image on
    # platform == 'linux'). On other platforms, collapse container_options to an
    # empty string so `options: ${{ fromJSON(...).container_options }}` doesn't
    # evaluate to a YAML sequence and fail template parsing.
    if platform != "linux":
        job_config["container_options"] = ""
        return job_config

    # Start with base options (always applied on Linux)
    options_parts = _BASE_CONTAINER_OPTIONS.copy()

    # Add GPU-specific options unless this is a CPU-only runner
    if not job_config.get("linux_cpu_runner", False):
        options_parts.extend(_GPU_CONTAINER_OPTIONS)

    # Add any job-specific container options
    if "container_options" in job_config:
        options_parts.extend(job_config["container_options"])

    # Concatenate all parts with a space separator
    job_config["container_options"] = " ".join(options_parts)

    return job_config


# Common settings applied to all jobs
_common_settings = {
    "additional_requirements_files": [],
}


def _build_runtime_test_matrix() -> dict:
    """Build the test_matrix with runtime-specific fields.

    The base test_matrix is imported from test_matrix.py and contains all the
    metadata for component filtering/selection. This function adds the
    runtime-specific fields like test_script and additional_requirements_files
    that depend on the script paths resolved at runtime.
    """
    # Deep copy to avoid modifying the shared module's dict
    matrix = copy.deepcopy(_base_test_matrix)

    # Runtime-specific configurations for each component
    # These use _get_script_path() and _get_artifact_path() which depend on runtime paths
    runtime_configs = {
        "sanity": {
            "test_script": f"python {_get_script_path('test_sanity.py')}",
        },
        "hip-tests": {
            "test_script": f"python {_get_script_path('test_hiptests.py')}",
        },
        "hipfile": {
            "test_script": f"python {_get_script_path('test_hipfile.py')}",
        },
        "rocblas": {
            "test_script": f"python {_get_script_path('test_runner.py')}",
        },
        "rocroller": {
            "test_script": f"python {_get_script_path('test_runner.py')}",
        },
        "tensilelite": {
            "additional_requirements_files": [
                _get_artifact_path("share/hipblaslt/tensilelite/requirements-test.txt"),
            ],
            "test_script": f"python {_get_script_path('pytest_runner.py')}",
        },
        "tensilelite-common": {
            "additional_requirements_files": [
                _get_artifact_path("share/hipblaslt/tensilelite/requirements-test.txt"),
            ],
            "test_script": f"TEST_CATEGORY=hw-common python {_get_script_path('pytest_runner.py')}",
        },
        "origami": {
            "test_script": f"python {_get_script_path('test_origami.py')}",
        },
        "hipblas": {
            "test_script": f"python {_get_script_path('test_runner.py')}",
        },
        "amdsmi": {
            "test_script": f"python {_get_script_path('test_amdsmi.py')}",
        },
        "hipblaslt": {
            "test_script": f"python {_get_script_path('test_runner.py')}",
        },
        "hipsolver": {
            "test_script": f"python {_get_script_path('test_runner.py')}",
        },
        "rocsolver": {
            "test_script": f"python {_get_script_path('test_runner.py')}",
        },
        "rocprim": {
            "test_script": f"python {_get_script_path('test_runner.py')}",
        },
        "hipcub": {
            "test_script": f"python {_get_script_path('test_runner.py')}",
        },
        "rocgdb-cpu": {
            "test_script": "python ./build/tests/rocgdb/test_rocgdb.py --parallel -f 0.25 --tests gdb.dwarf2",
        },
        "rocgdb-gpu": {
            "test_script": "python ./build/tests/rocgdb/test_rocgdb.py --parallel -f 0.25 --toolchain llvm --tests gdb.rocm",
        },
        "rocgdb-corefile": {
            "test_script": "python ./build/tests/rocgdb/test_rocgdb.py --parallel -f 0.25 --toolchain llvm --tests gdb.rocm/runtime-core.exp",
        },
        "rocr-debug-agent": {
            "test_script": "python ./build/tests/rocm-debug-agent/test_rocr-debug-agent.py",
        },
        "rocthrust": {
            "test_script": f"python {_get_script_path('test_runner.py')}",
        },
        "hipsparse": {
            "test_script": f"python {_get_script_path('test_runner.py')}",
        },
        "rocsparse": {
            "test_script": f"python {_get_script_path('test_runner.py')}",
        },
        "hipsparselt": {
            "test_script": f"python {_get_script_path('test_runner.py')}",
        },
        "rocrand": {
            "test_script": f"python {_get_script_path('test_runner.py')}",
        },
        "hiprand": {
            "test_script": f"python {_get_script_path('test_runner.py')}",
        },
        "rocfft": {
            "test_script": f"python {_get_script_path('test_runner.py')}",
        },
        "hipfft": {
            "test_script": f"python {_get_script_path('test_runner.py')}",
        },
        "miopen": {
            "test_script": f"python {_get_script_path('test_runner.py')}",
        },
        "miopen-dbsync": {
            "test_script": "python ./build/share/miopen/bin/run_dbsync_rocjitsu.py",
        },
        "rccl": {
            "test_script": f"pytest {_get_script_path('test_rccl.py')} -v -s --log-cli-level=info",
        },
        "rocshmem": {
            "test_script": f"python {_get_script_path('test_runner.py')}",
        },
        "rocprofiler-sdk": {
            "additional_requirements_files": [
                _get_artifact_path("share/rocprofiler-sdk/tests/requirements.txt"),
            ],
            "test_script": f"python {_get_script_path('test_rocprofiler_sdk.py')} --enable-cdash",
        },
        "hipdnn": {
            "test_script": f"python {_get_script_path('test_runner.py')}",
        },
        "hipdnn_install": {
            "test_script": f"python {_get_script_path('test_hipdnn_install.py')}",
        },
        "hipdnn-integration-tests": {
            "test_script": f"python {_get_script_path('test_runner.py')}",
        },
        "hipdnn-samples": {
            "test_script": f"python {_get_script_path('test_runner.py')}",
        },
        "miopenprovider": {
            "test_script": f"python {_get_script_path('test_runner.py')}",
        },
        "hipblasltprovider": {
            "test_script": f"python {_get_script_path('test_runner.py')}",
        },
        "hipkernelprovider": {
            "additional_requirements_files": [
                "build_tools/github_actions/test_executable_scripts/requirements-test-hipkernelprovider.txt",
            ],
            "test_script": f"python {_get_script_path('test_hipkernelprovider.py')}",
        },
        "rocwmma": {
            "test_script": f"python {_get_script_path('test_runner.py')}",
        },
        "rocalution": {
            "test_script": f"python {_get_script_path('test_runner.py')}",
        },
        "rocprofiler-compute": {
            "additional_requirements_files": [
                _get_artifact_path("libexec/rocprofiler-compute/requirements.txt"),
                _get_artifact_path("libexec/rocprofiler-compute/requirements-test.txt"),
            ],
            "test_script": f"python {_get_script_path('test_runner.py')}",
        },
        "rocprofiler-systems": {
            "additional_requirements_files": [
                _get_artifact_path("share/rocprofiler-systems/tests/requirements.txt"),
            ],
            "test_script": f"python {_get_script_path('test_runner.py')}",
        },
        "libhipcxx_amdclang": {
            "additional_requirements_files": [
                _get_artifact_path("libhipcxx/requirements-test.txt"),
            ],
            "test_script": f"python {_get_script_path('test_libhipcxx_amdclang.py')}",
        },
        "libhipcxx_hiprtc": {
            "additional_requirements_files": [
                _get_artifact_path("libhipcxx/requirements-test.txt"),
            ],
            "test_script": f"python {_get_script_path('test_libhipcxx_hiprtc.py')}",
        },
        "hipthreads": {
            "additional_requirements_files": [
                _get_artifact_path("hipthreads/test/requirements-test.txt"),
            ],
            "test_script": f"python {_get_script_path('test_hipthreads.py')}",
        },
        "hipthreads_examples": {
            "test_script": f"python {_get_script_path('test_hipthreads_examples.py')}",
        },
        "rocdecode": {
            "test_script": f"python {_get_script_path('test_rocdecode.py')}",
        },
        "rocjpeg": {
            "test_script": f"python {_get_script_path('test_rocjpeg.py')}",
        },
        "rpp": {
            "test_script": f"python {_get_script_path('test_rpp.py')}",
        },
        "aqlprofile": {
            "test_script": f"python {_get_script_path('test_aqlprofile.py')}",
        },
        "rocrtst": {
            "test_script": f"python {_get_script_path('test_runner.py')}",
        },
        "hiptensor": {
            "test_script": f"python {_get_script_path('test_runner.py')}",
        },
    }

    # Merge runtime configs into the base matrix
    for key, config in runtime_configs.items():
        if key in matrix:
            matrix[key].update(config)
        else:
            logging.warning(f"Runtime config for unknown test component: {key}")

    return matrix


# Build the test_matrix with runtime-specific fields
test_matrix = _build_runtime_test_matrix()


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--platform",
        type=str,
        default=platform_module.system().lower(),
        help="Platform to configure tests for (linux or windows)",
    )
    args, _ = parser.parse_known_args()
    platform = args.platform
    projects_to_test = os.getenv("PROJECTS_TO_TEST", "*")
    amdgpu_families = os.getenv("AMDGPU_FAMILIES")
    test_type = os.getenv("TEST_TYPE", "standard")
    test_labels = ast.literal_eval(os.getenv("TEST_LABELS") or "[]")
    build_variant = os.getenv("BUILD_VARIANT", "release")

    # Check for ci:run-multi-gpu label to force multi-GPU tests
    enable_multi_gpu_by_label = "ci:run-multi-gpu" in test_labels
    if enable_multi_gpu_by_label:
        logging.info("Multi-GPU tests forced via ci:run-multi-gpu label")

    # Get runner config for per-component runner selection
    # This enables better load distribution across runner pools
    test_runs_on_labels = None
    test_runs_on_default = None
    test_runs_on_multi_gpu_labels = None
    test_runs_on_multi_gpu_default = None
    # For ASAN builds, use the sandbox runner if available
    test_runs_on_sandbox = None
    if amdgpu_families:
        shortened_family = amdgpu_families.split("-")[0].lower()
        all_families = get_all_families_for_trigger_types(
            ["presubmit", "postsubmit", "nightly"]
        )
        if shortened_family in all_families:
            platform_info = all_families[shortened_family].get(platform, {})
            test_runs_on_labels = platform_info.get("test-runs-on-labels")
            test_runs_on_default = platform_info.get("test-runs-on", "")
            test_runs_on_multi_gpu_labels = platform_info.get(
                "test-runs-on-multi-gpu-labels"
            )
            test_runs_on_multi_gpu_default = platform_info.get(
                "test-runs-on-multi-gpu", ""
            )
            test_runs_on_sandbox = platform_info.get("test-runs-on-sandbox", "")

    logging.info(f"Selecting projects: {projects_to_test}")

    logging.info(f"Using test_matrix ({len(test_matrix)} test(s))")

    # This string -> array conversion ensures no partial strings are detected during test selection (ex: "hipblas" in ["hipblaslt", "rocblas"] = false)
    project_array = [item.strip() for item in projects_to_test.split(",")]

    all_components = []
    for key in test_matrix:
        job_name = test_matrix[key]["job_name"]

        # Resolve the individual gfx targets for the current family once, so both
        # include_family and exclude_family can match either the family group
        # string (e.g. "gfx120X-all") passed via AMDGPU_FAMILIES or the individual
        # gfx targets within that family (e.g. "gfx1200", "gfx1201").
        _family_gfx_targets = []
        if amdgpu_families and shortened_family and shortened_family in all_families:
            _family_gfx_targets = (
                all_families[shortened_family]
                .get(platform, {})
                .get("fetch-gfx-targets", [])
            )

        # include_family (opt-in) and exclude_family (opt-out) together decide
        # whether a job runs: it runs only when it matches an include (if any are
        # listed for this platform) and matches no exclude. Matching is exact
        # membership.
        _include_list = test_matrix[key].get("include_family", {}).get(platform, [])
        if _include_list and not _family_matches(
            _include_list, amdgpu_families, _family_gfx_targets
        ):
            logging.info(
                f"Excluding job {job_name} for platform {platform} and family "
                f"{amdgpu_families}: not listed in include_family"
            )
            continue

        _exclude_list = test_matrix[key].get("exclude_family", {}).get(platform, [])
        if _exclude_list and _family_matches(
            _exclude_list, amdgpu_families, _family_gfx_targets
        ):
            logging.info(
                f"Excluding job {job_name} for platform {platform} and family {amdgpu_families}"
            )
            continue

        # If test labels are populated, and the test job name is not in the test labels, skip the test
        # Note: Benchmarks never use test_labels (always empty list)
        # Filter out ci: control labels - they're not test component selectors
        component_test_labels = [c for c in test_labels if not c.startswith("ci:")]
        parsed_test_labels = [c.split("test:")[-1] for c in component_test_labels]
        expanded_test_labels = [
            member
            for label in parsed_test_labels
            for member in TEST_LABEL_GROUPS.get(label, [label])
        ]
        if key != "sanity" and expanded_test_labels and key not in expanded_test_labels:
            logging.info(f"Excluding job {job_name} since it's not in the test labels")
            continue

        # Tier gate: a component may declare which test tiers it runs on via
        # "test_types". Skip it entirely (schedule no job) for any TEST_TYPE not in
        # the list -- e.g. miopen-dbsync runs standard/comprehensive/full only, never
        # quick. Omit the field to run on every tier.
        allowed_test_types = test_matrix[key].get("test_types")
        if allowed_test_types and test_type not in allowed_test_types:
            logging.info(
                f"Excluding job {job_name}: test_type {test_type} not in {allowed_test_types}"
            )
            continue

        # If the test is enabled for a particular platform and a particular (or all) projects are selected.
        # Note: Sanity goes through the same all_components loop as other components, but is separated
        # into its own sanity_component GHA output after the loop (see gha_set_output below).
        if platform in test_matrix[key]["platform"] and (
            key == "sanity" or key in project_array or "*" in project_array
        ):
            logging.info(f"Including job {job_name} with test_type {test_type}")

            # Hip-tests on Windows run with both PAL and ROCR backends.
            # See: https://github.com/ROCm/TheRock/issues/3587
            if key == "hip-tests" and platform == "windows":
                base = test_matrix[key]
                total_shards = base.get("total_shards_dict", {}).get(platform, 1)
                if test_type == "quick":
                    total_shards = 1

                shard_arr = list(range(1, total_shards + 1))

                pal_entry = {
                    **_common_settings,
                    "job_name": "hip-tests (PAL)",
                    "fetch_artifact_args": base["fetch_artifact_args"],
                    "timeout_minutes": base["timeout_minutes"],
                    "test_script": base["test_script"],
                    "platform": base["platform"],
                    "total_shards": total_shards,
                    "test_type": test_type,
                    "shard_arr": shard_arr,
                    "gpu_enable_pal": "1",
                }
                all_components.append(pal_entry)

                rocr_entry = {
                    **_common_settings,
                    "job_name": "hip-tests (ROCR)",
                    "fetch_artifact_args": base["fetch_artifact_args"],
                    "timeout_minutes": base["timeout_minutes"],
                    "test_script": base["test_script"],
                    "platform": base["platform"],
                    "total_shards": total_shards,
                    "test_type": test_type,
                    "shard_arr": shard_arr,
                    "gpu_enable_pal": "0",
                }
                all_components.append(rocr_entry)
                continue

            job_config_data = {**_common_settings, **test_matrix[key]}
            job_config_data["test_type"] = test_type

            # tensilelite: append the tensilelite/tests C++ gtest suite (run via
            # ctest -L <test_type>, driven by the shared test_runner.py) after
            # the existing pytest stage, for every tier except quick -- that
            # component's test_categories.yaml only defines standard/
            # comprehensive/full so far (promote to quick once the standard
            # tier proves stable). See AIHPBLAS-4410.
            #
            # TODO(#7851): this is a temporary special-case. test_runner.py
            # only knows how to run the C++/ctest suite today, so the pytest
            # and ctest stages have to be chained here instead. Fold both
            # into test_runner.py's own dual-mode support and drop this
            # branch once that lands.
            if key == "tensilelite" and test_type != "quick":
                job_config_data["test_script"] = (
                    job_config_data["test_script"]
                    + f" && TEST_COMPONENT=hipblaslt-tensilelite python {_get_script_path('test_runner.py')}"
                )
                # +15 min over the pytest-only baseline for the added ctest
                # stage; re-measure once CI timing is observed and adjust.
                job_config_data["timeout_minutes"] = (
                    job_config_data["timeout_minutes"] + 15
                )

            # For CI testing, we construct a shard array based on "total_shards" from "fetch_test_configurations.py"
            # This way, the test jobs will be split up into X shards. (ex: [1, 2, 3, 4] = 4 test shards)
            # For display purposes, we add "i + 1" for the job name (ex: 1 of 4). During the actual test sharding in the test executable, this array will become 0th index
            total_shards = job_config_data.get("total_shards_dict", {}).get(platform, 1)
            job_config_data["shard_arr"] = [i + 1 for i in range(total_shards)]
            job_config_data["total_shards"] = total_shards

            # If the test type is quick tests, we only need one shard for the test job
            if test_type == "quick":
                job_config_data["total_shards"] = 1
                job_config_data["shard_arr"] = [1]

            # If the test requires multi GPU testing, we use a multi-GPU test runner for this specific test
            # Inside the "multi_gpu" field, we have a mapping of amdgpu_family -> bool (if multi GPU testing is enabled for that family)
            # If the multi GPU test runner is not enabled, we will skip the test
            if "multi_gpu" in test_matrix[key]:
                # Skip multi-GPU tests for quick runs unless enable_multi_gpu_by_label is set.
                # Jobs that require multi-GPU runners (defined via "multi_gpu" in their config)
                # only run on standard, comprehensive, or full tiers by default.
                if test_type == "quick" and not enable_multi_gpu_by_label:
                    logging.info(
                        f"Excluding job {job_name}: multi-GPU tests skipped for quick runs (capacity constraint)"
                    )
                    continue

                # Check if this family has multi-GPU runner support, OR if enable_multi_gpu_by_label is set
                family_has_multi_gpu = (
                    platform in test_matrix[key]["multi_gpu"]
                    and amdgpu_families in test_matrix[key]["multi_gpu"][platform]
                )

                if family_has_multi_gpu or enable_multi_gpu_by_label:
                    # Mark this component as needing a multi-GPU runner.
                    # The actual runner selection is done in the per-component loop below.
                    job_config_data["multi_gpu_runner"] = True
                    logging.info(f"Including job {job_name} for multi-GPU testing")
                else:
                    # If the architecture is not available for multi GPU testing, we skip the test requiring multi GPU
                    logging.info(
                        f"Excluding job {job_name} since multi GPU testing is not available for family {amdgpu_families}"
                    )
                    continue

            all_components.append(job_config_data)

    # Per-component runner selection for better load distribution
    # Each component gets its own independent random draw based on configured weights
    # For ASan builds, use the sandbox runner to isolate potentially failing tests.
    # This matches multiple build variants, including "asan", "host-asan",
    # "asan-debug", and "host-asan-debug".
    is_asan_build = "asan" in build_variant
    components_with_runners = []
    for component in all_components:
        job_name = component.get("job_name", "unknown")
        if "multi_gpu_runner" in component:
            # Multi-GPU components use multi-GPU runner labels
            if test_runs_on_multi_gpu_labels:
                component["multi_gpu_runner"] = select_weighted_label(
                    test_runs_on_multi_gpu_labels, f"{job_name}-multi-gpu"
                )
            elif test_runs_on_multi_gpu_default:
                component["multi_gpu_runner"] = test_runs_on_multi_gpu_default
            else:
                # No multi-GPU runner configured for this family; skip the component
                logging.info(
                    f"Excluding job {job_name}: multi-GPU required but no multi-GPU runner configured"
                )
                continue
        elif "test_runner" not in component:
            # Regular components use standard runner labels.
            # Skip if test_runner is already pre-pinned (e.g. rocgdb-corefile).
            # For ASAN builds, use the sandbox runner if available
            if is_asan_build and test_runs_on_sandbox:
                component["test_runner"] = test_runs_on_sandbox
                logging.info(
                    f"  {job_name}: using ASAN sandbox runner: {test_runs_on_sandbox}"
                )
            elif test_runs_on_labels:
                component["test_runner"] = select_weighted_label(
                    test_runs_on_labels, job_name
                )
            elif test_runs_on_default:
                component["test_runner"] = test_runs_on_default
        components_with_runners.append(component)

    # Build container options for all components (concatenates base, GPU, and job-specific options)
    all_components = [
        _build_container_options(c, platform) for c in components_with_runners
    ]

    # Separate sanity (always a prerequisite) from the regular component matrix.
    sanity_component = next(
        (c for c in all_components if c.get("job_name") == "sanity"), None
    )
    output_matrix = [c for c in all_components if c.get("job_name") != "sanity"]

    gha_set_output(
        {
            "sanity_component": json.dumps(sanity_component),
            "components": json.dumps(output_matrix),
            "platform": platform,
        }
    )


if __name__ == "__main__":
    run()
