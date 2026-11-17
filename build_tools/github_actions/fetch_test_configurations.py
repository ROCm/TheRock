# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
This module owns ``test_matrix``, the definition of every test component, and
turns it into the two things CI needs:

  - ``build_test_plan()`` decides which components run for one GPU family, at
    which tier, and on which runner. configure_multi_arch_ci.py calls it once per
    family during setup and ships the result to test_artifacts.yml, so the test
    workflows need no configuration step of their own.
  - ``describe_component()`` returns the static execution config for a single
    component (test_script, fetch_artifact_args, ...). That config is identical
    for every family, so test_component.yml looks it up on the test runner
    instead of carrying it through workflow outputs.

Running this file directly builds a test plan from the environment, which is how
standalone dispatches of test_artifacts.yml configure themselves:

  - AMDGPU_FAMILIES, AMDGPU_TARGETS, PROJECTS_TO_TEST, TEST_TYPE, TEST_LABELS,
    BUILD_VARIANT, SKIP_STAGES, TEST_RUNS_ON, TEST_RUNS_ON_CPU

Outputs (written to $GITHUB_OUTPUT):
  - test_plan: JSON object; see build_test_plan() for its shape.
  - with --describe-component: the fields listed in describe_component().
"""

import argparse
import ast
import json
import logging
import os
import platform as platform_module
from dataclasses import dataclass, field
from pathlib import Path

from github_actions_api import *
from amdgpu_family_matrix import (
    get_all_families_for_trigger_types,
    get_cpu_test_runner,
    select_weighted_label,
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


# Maps a group label (the part after "test:") to the individual test matrix
# keys it expands to. Use this when a single label should select multiple
# related jobs without relying on name-prefix inference.
TEST_LABEL_GROUPS: dict[str, list[str]] = {
    "rocgdb": ["rocgdb-cpu", "rocgdb-gpu"],
    "tensilelite": ["tensilelite", "tensilelite-common"],
}


def _parse_test_labels(raw: str) -> list[str]:
    """Parse the TEST_LABELS environment variable into a list of strings.

    Accepts either Python list literal syntax (e.g. '["test:rocblas"]') or a
    plain comma-separated string (e.g. 'test:rocblas,test:hipblas').  The
    latter is friendlier for manual workflow_dispatch runs where users are
    unlikely to type valid Python list syntax.
    """
    raw = raw.strip()
    if not raw or raw == "[]":
        return []

    # If it looks like a Python list literal, try ast.literal_eval first.
    if raw.startswith("["):
        try:
            result = ast.literal_eval(raw)
            if isinstance(result, list):
                return [str(item) for item in result]
        except (ValueError, SyntaxError) as exc:
            raise ValueError(
                f"TEST_LABELS looks like a Python list but could not be parsed: {raw!r}. "
                f'Expected format: \'["test:rocblas", "test:hipblas"]\' (Python list literal) '
                f"or 'test:rocblas,test:hipblas' (comma-separated string). "
                f"Original error: {exc}"
            ) from exc

    # Fall back to comma-separated string parsing.
    return [item.strip() for item in raw.split(",") if item.strip()]


# Base container options applied to all Linux containers
# --user 0:0 - Running as root, by recommendation of GitHub: https://docs.github.com/en/actions/reference/workflows-and-actions/dockerfile-support#user
# --ulimit memlock=-1:-1 - Prevents memory allocation issues with ROCm inside container
# --ulimit nofile=1048576:1048576 - Increase open file limit for RCCL
_BASE_CONTAINER_OPTIONS = [
    "--user 0:0",
    "--ulimit memlock=-1:-1",
    "--ulimit nofile=1048576:1048576",
]

# GPU-specific container options per platform (only applied when linux_cpu_runner != True)
# linux:
#   --security-opt seccomp=unconfined - enables memory mapping and device ioctls needed by ROCm/HPC GPU workloads
#   --group-add video - Grants access to GPU video group
#   --device /dev/kfd - AMD KFD device for GPU compute
#   --device /dev/dri - Direct Rendering Infrastructure devices
#   --group-add 993,992,110 - Additional GPU-related groups
#   --env-file /etc/podinfo/gha-gpu-isolation-settings - Required for GPU isolation on OSSCI MIXXX runners
#   -e ROCR_VISIBLE_DEVICES - Pass host's GPU isolation env var to container (used on ARC runners)
# wsl: the GPU is only /dev/dxg (no /dev/kfd or /dev/dri), with its driver
#   libraries in /usr/lib/wsl on the host.
_GPU_CONTAINER_OPTIONS = {
    "linux": [
        "--security-opt seccomp=unconfined",
        "--group-add video",
        "--device /dev/kfd",
        "--device /dev/dri",
        "--group-add 993",
        "--group-add 992",
        "--group-add 110",
        "--env-file /etc/podinfo/gha-gpu-isolation-settings",
        "-e ROCR_VISIBLE_DEVICES",
        "-e KUBE_CPU_REQUEST",
    ],
    "wsl": [
        "--device /dev/dxg",
        "-v /usr/lib/wsl:/usr/lib/wsl",
        "-e LD_LIBRARY_PATH=/usr/lib/wsl/lib",
    ],
}

# Options that Apptainer-backed runners cannot honor: they always run as the
# invoking user and cannot grant capabilities, so requesting them fails the job.
_APPTAINER_FAMILIES = ["gfx125X-dcgpu"]
_APPTAINER_UNSUPPORTED_OPTIONS = ["--user 0:0", "--cap-add SYS_MODULE"]


def _build_container_options(
    platform: str,
    *,
    cpu_only: bool = False,
    job_options: list[str] | None = None,
    amdgpu_families: str = "",
) -> str:
    """Concatenate base, GPU, and job-specific container options.

    Args:
        platform: The platform (e.g., "linux", "windows", "wsl")
        cpu_only: True for components routed to a CPU-only runner
        job_options: Extra options declared by the component itself
        amdgpu_families: The AMDGPU family string being tested

    Nearly every component uses one of the two family-wide defaults (GPU and
    CPU-only), so the test plan emits those once per family and only carries a
    per-component string for the few components that pass ``job_options``.
    """
    # Containers are Linux-only, including WSL (test_component.yml gates
    # container.image on platform != 'windows'). On Windows, collapse
    # container_options to an empty string so `options: ${{ ... }}` doesn't
    # evaluate to a YAML sequence and fail template parsing.
    if platform not in _GPU_CONTAINER_OPTIONS:
        return ""

    # Start with base options (always applied on Linux)
    options_parts = _BASE_CONTAINER_OPTIONS.copy()

    # Add GPU-specific options unless this is a CPU-only runner
    if not cpu_only:
        options_parts.extend(_GPU_CONTAINER_OPTIONS[platform])

    # Add any job-specific container options
    options_parts.extend(job_options or [])

    # Drop options the Apptainer runners for this family cannot honor
    if any(family in amdgpu_families for family in _APPTAINER_FAMILIES):
        options_parts = [
            opt for opt in options_parts if opt not in _APPTAINER_UNSUPPORTED_OPTIONS
        ]

    # Concatenate all parts with a space separator
    return " ".join(options_parts)


def _family_matches(
    family_list: list[str], amdgpu_families: str, family_gfx_targets: list[str]
) -> bool:
    """Returns True if the current AMDGPU family matches any entry in family_list.

    CI may pass either the family group string (e.g. "gfx120X-all") via
    AMDGPU_FAMILIES or refer to the individual gfx targets within that family
    (e.g. "gfx1200", "gfx1201"). Both forms are checked using exact membership.
    """
    return amdgpu_families in family_list or any(
        t in family_list for t in family_gfx_targets
    )


# Common settings applied to all jobs
_common_settings = {
    "additional_requirements_files": [],
    # Components whose tests need no artifacts of their own (hipdnn_install,
    # hipdnn-cudnn-samples, profiler-hub) omit this and fetch nothing.
    "fetch_artifact_args": "",
}

# Common settings for rocgdb jobs
_rocgdb_common = {
    "fetch_artifact_args": "--debug-tools --tests",
    "timeout_minutes": 30,
    "platform": ["linux"],
    "total_shards": 1,
    "container_image": "ghcr.io/rocm/no_rocm_image_ubuntu24_04_rocgdb@sha256:e89ba4458be7add8ee1f8e398da9f3fd8816a0ad56d521054543916d4133184a",
    "container_options": ["--cap-add=SYS_PTRACE"],
}


# Runner assignment for test components
# =====================================
# Most components have their runner selected at runtime by the per-component loop
# below, which draws from the AMDGPU-family runner pool configured in
# amdgpu_family_matrix.py / therock-ci-config.
#
# A component may instead pre-pin its runner by setting "test_runner" directly in
# its test_matrix entry. The loop will detect this and leave the value untouched.
# Use this when a component must run on a specific machine class regardless of the
# GPU family being tested. For example, rocgdb-corefile requires runners that have
# GPU core-dump support enabled, identified by the label
# "linux-gfx942-gpu-rocm-mathlib", which is registered separately in the runner pool.
#
# Similarly, "linux_cpu_runner: True" routes a component to a CPU-only machine
# (currently aws-linux-scale-rocm-prod) via the test_artifacts.yml routing
# expression. "multi_gpu_runner" routes to multi-GPU machines.
#
# A component may also restrict which GPU families it runs on via "include_family"
# (opt-in) and "exclude_family" (opt-out). Each is a map keyed by platform
# ("linux" and/or "windows") whose value is a list of family entries. A job runs
# only when it matches an include (if any are listed for that platform) and
# matches no exclude.
#
# The two filters are evaluated per platform and independently: a list under
# "linux" only affects Linux runs and a list under "windows" only affects Windows
# runs, so a platform with no list (or the empty list) is left unfiltered. This
# means an include scoped to one platform does not gate the other. To gate both,
# list the families under both keys, for example:
#   "include_family": {"linux": ["gfx942"], "windows": ["gfx942"]}
#
# Each entry matches either the family group string passed via AMDGPU_FAMILIES
# (e.g. "gfx120X-all", "gfx950-dcgpu") or one of the individual gfx targets within
# that family (e.g. "gfx1200", "gfx1201"). Some families expose no individual
# targets, so those must be matched by the group string (e.g. "gfx1150",
# "gfx125X-dcgpu"). Targets = AMDGPU_TARGETS if set, else the static fetch-gfx-targets.
# Examples:
#   "exclude_family": {"linux": ["gfx1030"]}                # skip a single target
#   "include_family": {"linux": ["gfx908", "gfx90a", "gfx942"]}  # opt in to a set
#
# A component may restrict which test tiers it runs on via "test_types", a list of
# allowed TEST_TYPE values (any of "quick", "standard", "comprehensive", "full").
# When set, the component is skipped entirely -- no job is scheduled -- for any tier
# not in the list; omit the field to run on every tier (the default). For example, a
# component whose suite is too slow for the quick sanity tier opts out of it with:
#   "test_types": ["standard", "comprehensive", "full"]
#
# A component may mark itself advisory (non-blocking) on specific GPU families via
# "expect_failure", a map keyed by platform whose value is a list of family entries,
# matched exactly like include_family/exclude_family (so either the family group
# string or an individual gfx target works). It is resolved to a single boolean per
# job before the matrix is emitted, which test_component.yml maps to
# continue-on-error plus an "(xfail)" job-name suffix. The job is still scheduled and
# still reports; it just cannot fail the run. Use it to land coverage for an arch
# that is still being qualified without blocking everyone else. For example:
#   "expect_failure": {"linux": ["gfx950"]}
#
# A component's "skip_stages" lists build stages whose absence excludes the test.
# If any listed stage is in SKIP_STAGES, the component is skipped. Reused stages
# still supply artifacts and do not exclude tests. Omit the field for no stage gate.
#
# A component's "skip_build_variant" lists BUILD_VARIANT values (exact match) on which
# it is not scheduled, e.g. ["asan", "host-asan", "asan-debug", "host-asan-debug"] for a
# test that cannot work under ASAN. Omit the field to run on every build variant.

test_matrix = {
    # Sanity runs first when compiler-runtime artifacts are available.
    "sanity": {
        "job_name": "sanity",
        "skip_stages": ["compiler-runtime"],
        "fetch_artifact_args": "--sanity",
        "timeout_minutes": 5,
        "test_script": f"python {_get_script_path('test_sanity.py')}",
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
        # Supply the system OpenCL ICD loader for clinfo (Linux only).
        "container_image": "ghcr.io/rocm/no_rocm_image_ubuntu24_04_ocl_rt@sha256:f7f293a91b8bcf68d7a4afa715578eb252e0455e56e2df3dfab6921bcc5dbd80",
        # Running docker with cap-add and -v /lib/modules, by recommendation of GitHub:
        # https://rocm.docs.amd.com/projects/amdsmi/en/amd-staging/how-to/setup-docker-container.html
        "container_options": ["--cap-add SYS_MODULE", "-v /lib/modules:/lib/modules"],
    },
    # hip-tests
    "hip-tests": {
        "job_name": "hip-tests",
        "fetch_artifact_args": "--hip-tests --tests",
        "timeout_minutes": 120,
        "test_script": f"python {_get_script_path('test_hiptests.py')}",
        # "wsl": the Linux build on a WSL-hosted runner (ci:run-wsl).
        "platform": ["linux", "windows", "wsl"],
        # Advisory on WSL until its results are stable.
        "expect_failure": {"wsl": ["gfx110X-all"]},
        "total_shards_dict": {
            "linux": 4,
            "windows": 4,
            "wsl": 4,
        },
    },
    # hipFile (storage-libs) unit tests. CPU-only (mocked), so they run quickly
    # and do not require a GPU runner.
    "hipfile": {
        "job_name": "hipfile",
        "fetch_artifact_args": "--hipfile --tests",
        "timeout_minutes": 15,
        "test_script": f"python {_get_script_path('test_hipfile.py')}",
        "platform": ["linux"],
        "linux_cpu_runner": True,
        "total_shards_dict": {
            "linux": 1,
        },
    },
    # BLAS tests
    "rocblas": {
        "job_name": "rocblas",
        "fetch_artifact_args": "--blas --tests",
        # Suppress per-test kpack debug logging (test_component.yml defaults it to
        # "1" for diagnostics, which floods this suite's -V ctest output).
        "rocm_kpack_debug": "0",
        # GHA step timeout: max category timeout in rocBLAS should be 24 hours / 6 shards = 4 hours per shard
        # 240 min + 20% margin = 288 min
        "timeout_minutes": 288,
        "test_script": f"python {_get_script_path('test_runner.py')}",
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 6,
            "windows": 6,
        },
        "exclude_family": {
            "linux": [
                # KNOWN FAILURE (rocblas-test_quick_suite crash/no gtest output, cannot filter individual tests)
                # https://github.com/ROCm/TheRock/actions/runs/35820932302/job/107052684531
                "gfx125X-dcgpu",
            ],
        },
    },
    "rocroller": {
        "job_name": "rocroller",
        "fetch_artifact_args": "--blas --tests",
        "timeout_minutes": 60,
        "test_script": f"python {_get_script_path('test_runner.py')}",
        "platform": ["linux"],
        "total_shards_dict": {
            "linux": 5,
            "windows": 5,
        },
        "exclude_family": {
            # rocroller does not support gfx110X architectures (see TheRock#6693)
            # rocroller does not plan to support Linux and Windows gfx115X architectures
            "linux": [
                "gfx1100",
                "gfx1101",
                "gfx1102",
                "gfx1103",
                "gfx1150",
                "gfx1151",
                "gfx1152",
                "gfx1153",
            ],
            "windows": [
                "gfx1100",
                "gfx1101",
                "gfx1102",
                "gfx1103",
                "gfx1150",
                "gfx1151",
                "gfx1152",
                "gfx1153",
            ],
        },
    },
    "tensilelite": {
        "job_name": "tensilelite",
        "fetch_artifact_args": "--blas --tests",
        "timeout_minutes": 15,
        "additional_requirements_files": [
            _get_artifact_path("share/hipblaslt/tensilelite/requirements-test.txt"),
        ],
        # Python/pytest suite only (rocisa + TensileLite unit). The C++ gtest
        # suite (tensilelite/tests) is appended below for TEST_TYPE != quick;
        # see the "tensilelite" special-case in the component loop
        # (AIHPBLAS-4410).
        "test_script": f"python {_get_script_path('pytest_runner.py')}",
        "platform": ["linux"],
        "total_shards_dict": {
            "linux": 1,
        },
        "exclude_family": {
            "linux": [
                # CRITICAL FAILURE (hang): test causes hang during execution
                # https://github.com/ROCm/TheRock/actions/runs/36185080189/job/108239321671
                # gfx1250 only; gfx1250-strict runs.
                "gfx1250",
            ],
        },
    },
    # TensileLite common GEMM tests (Tensile/Tests/common) on real hardware,
    # matching Math CI's `preliminary` `-m common` stage. A separate job rather
    # than another stage chained onto "tensilelite", so a unit-test failure
    # cannot hide the GEMM result.
    #
    # include_family is opt-in on purpose: selection inside the suite works by
    # each config declaring skip-gfxNNNN, and that list only covers the
    # architectures registered in tensilelite's pytest.ini. A family with no
    # declarations (e.g. gfx1103, gfx115X) would try to run all ~417 configs.
    #
    # In Math CI (4 xdist workers) this suite takes up to 2h03 on gfx950 and
    # 64 min on gfx942. Only gfx942 is on the PR path (gfx950 and gfx90a are
    # postsubmit, gfx120X-all is nightly), so it runs unsharded; the timeout is
    # sized for gfx950.
    #
    # Until the pinned rocm-libraries ships the hw-common category,
    # pytest_runner.py skips this job with a warning instead of failing.
    "tensilelite-common": {
        "job_name": "tensilelite-common",
        "fetch_artifact_args": "--blas --tests",
        "timeout_minutes": 180,
        "additional_requirements_files": [
            _get_artifact_path("share/hipblaslt/tensilelite/requirements-test.txt"),
        ],
        "test_script": f"TEST_CATEGORY=hw-common python {_get_script_path('pytest_runner.py')}",
        "platform": ["linux"],
        "total_shards_dict": {
            "linux": 1,
        },
        "include_family": {
            "linux": [
                "gfx90a",
                "gfx94X-dcgpu",
                "gfx950-dcgpu",
                "gfx120X-all",
                "gfx1250-strict",
            ],
        },
    },
    "origami": {
        "job_name": "origami",
        "fetch_artifact_args": "--blas --tests",
        "timeout_minutes": 5,
        "test_script": f"python {_get_script_path('test_origami.py')}",
        "platform": ["linux", "windows"],
        "total_shards": 1,
    },
    "hipblas": {
        "job_name": "hipblas",
        "fetch_artifact_args": "--blas --solver --tests",
        # Suppress per-test kpack debug logging (test_component.yml defaults it to
        # "1" for diagnostics, which floods this suite's -V ctest output).
        "rocm_kpack_debug": "0",
        "timeout_minutes": 30,
        "test_script": f"python {_get_script_path('test_runner.py')}",
        "platform": ["linux", "windows"],
        # TODO(#2616): Enable full tests once known machine issues are resolved
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
    },
    "amdsmi": {
        "job_name": "amdsmi",
        "fetch_artifact_args": "--base-only",
        "timeout_minutes": 10,
        "test_script": f"python {_get_script_path('test_amdsmi.py')}",
        "platform": ["linux"],
        "total_shards_dict": {
            "linux": 1,
        },
    },
    "hipblaslt": {
        "job_name": "hipblaslt",
        "fetch_artifact_args": "--blas --tests",
        "timeout_minutes": 180,
        "test_script": f"python {_get_script_path('test_runner.py')}",
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 6,
            "windows": 1,
        },
        "exclude_family": {
            "linux": [
                # hipBLASLt does not support gfx103X (see TheRock#1062)
                "gfx1030",
                # FAILURE (3275+ gtest failures - too many to filter individually)
                # https://github.com/ROCm/TheRock/actions/runs/35816223373/job/107038470972
                # gfx1250 only; gfx1250-strict runs.
                "gfx1250",
            ],
        },
    },
    # SOLVER tests
    "hipsolver": {
        "job_name": "hipsolver",
        "fetch_artifact_args": "--solver --blas --sparse --tests",
        "timeout_minutes": 5,
        "test_script": f"python {_get_script_path('test_runner.py')}",
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
        "exclude_family": {
            # known failure
            "linux": ["gfx125X-dcgpu"],
        },
    },
    "rocsolver": {
        "job_name": "rocsolver",
        "fetch_artifact_args": "--solver --blas --tests",
        # test_runner.py drives ctest category labels, so it runs a filtered
        # subset rather than the full ~5 hr extended suite.
        # 68350(approx) tests needs 48 mins, so 48 mins / 2 shards = 24 mins per shard
        # 24 mins + 20% margin = 30 mins => ~40 mins (considering gpu delays and lags)
        "timeout_minutes": 60,
        "test_script": f"python {_get_script_path('test_runner.py')}",
        # Issue for adding windows tests: https://github.com/ROCm/TheRock/issues/1770
        "platform": ["linux"],
        "total_shards_dict": {
            "linux": 3,
            "windows": 2,
        },
    },
    # PRIM tests
    "rocprim": {
        "job_name": "rocprim",
        "fetch_artifact_args": "--prim --tests",
        "timeout_minutes": 45,
        "test_script": f"python {_get_script_path('test_runner.py')}",
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 2,
            "windows": 2,
        },
    },
    "hipcub": {
        "job_name": "hipcub",
        "fetch_artifact_args": "--prim --tests",
        "timeout_minutes": 45,
        "test_script": f"python {_get_script_path('test_runner.py')}",
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
    },
    "rocgdb-cpu": {
        **_rocgdb_common,
        "job_name": "rocgdb-cpu",
        "test_script": "python ./build/tests/rocgdb/test_rocgdb.py --parallel -f 0.25 --tests gdb.dwarf2",
        "linux_cpu_runner": True,
    },
    "rocgdb-gpu": {
        **_rocgdb_common,
        "job_name": "rocgdb-gpu",
        "test_script": "python ./build/tests/rocgdb/test_rocgdb.py --parallel -f 0.25 --toolchain llvm --tests gdb.rocm",
        "exclude_family": {
            "linux": [
                # GPU tests do not honor ROCR_VISIBLE_DEVICES and utilizes other gpus during test runs. excluding
                "gfx125X-dcgpu",
            ],
        },
    },
    "rocr-debug-agent": {
        "job_name": "rocr-debug-agent",
        "fetch_artifact_args": "--debug-tools --tests",
        "timeout_minutes": 10,
        "test_script": "python ./build/tests/rocm-debug-agent/test_rocr-debug-agent.py",
        "platform": ["linux"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
        "exclude_family": {
            "linux": [
                # FAILURE (15 test failures, custom test framework doesn't support GTEST_FILTER)
                # https://github.com/ROCm/TheRock/actions/runs/35803014951/job/106997748160
                "gfx125X-dcgpu",
            ],
        },
    },
    "rocthrust": {
        "job_name": "rocthrust",
        "fetch_artifact_args": "--prim --tests",
        "timeout_minutes": 45,
        "test_script": f"python {_get_script_path('test_runner.py')}",
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
        "exclude_family": {
            "linux": [
                # CRITICAL FAILURE (amd-smi hangs): rocthrust test hangs during amd-smi GPU detection
                # https://github.com/ROCm/TheRock/actions/runs/35798263253/job/106982866530
                "gfx125X-dcgpu",
            ],
        },
    },
    # SPARSE tests
    "hipsparse": {
        "job_name": "hipsparse",
        "fetch_artifact_args": "--sparse --blas --tests",
        "timeout_minutes": 30,
        "test_script": f"python {_get_script_path('test_runner.py')}",
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 3,
            "windows": 3,
        },
        "exclude_family": {
            "linux": [
                # CRITICAL FAILURE (hang): test causes hang during execution
                # https://github.com/ROCm/TheRock/actions/runs/36174684654/job/108202860306
                "gfx125X-dcgpu",
            ],
        },
    },
    "rocsparse": {
        "job_name": "rocsparse",
        "fetch_artifact_args": "--sparse --blas --tests",
        # rocsparse now uses 3-way gtest sharding, enabled once the tolerance fix
        # in ROCm/rocm-libraries#8713 landed in TheRock. The full suite is ~240 min
        # single-shard; split across 3 shards that is ~80 min per shard, and 90 min
        # leaves headroom.
        "timeout_minutes": 90,
        "test_script": f"python {_get_script_path('test_runner.py')}",
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 3,
            "windows": 3,
        },
        "exclude_family": {
            "linux": [
                # KNOWN FAILURE: sddmm f16 compute tests fail with tolerance issues
                # individual tests fail but GTEST_FILTER plumbing not available (ctest overrides env var)
                # https://github.com/ROCm/TheRock/actions/runs/35914840519/job/107363781930
                "gfx125X-dcgpu",
            ],
        },
    },
    "hipsparselt": {
        "job_name": "hipsparselt",
        "fetch_artifact_args": "--sparse --blas --tests",
        # GHA step timeout: max category timeout in hipsparselt should be 6 hours / 6 shards = 60 min per shard
        # 60 min + 20% margin = 72 min
        "timeout_minutes": 72,
        "test_script": f"python {_get_script_path('test_runner.py')}",
        "platform": ["linux"],
        "total_shards_dict": {
            "linux": 6,
            "windows": 1,
        },
        "exclude_family": {
            # hipsparselt does not support gfx908, gfx90a (see TheRock#2042)
            # hipsparselt does not support gfx110X architectures (TensileLibrary missing)
            # hipsparselt does not plan to support Linux and Windows gfx115X architectures
            # hipsparselt does not support gfx117X architectures
            # hipsparselt does not support gfx120X (see TheRock#6473)
            "linux": [
                "gfx908",
                "gfx90a",
                "gfx1030",
                "gfx1100",
                "gfx1101",
                "gfx1102",
                "gfx1103",
                "gfx1150",
                "gfx1151",
                "gfx1152",
                "gfx1153",
                "gfx117X-all",
                "gfx1200",
                "gfx1201",
                # KNOWN FAILURE (timeout): Quick suite exceeds 900s on FP16 strided-batched clipped-ReLU
                # Related: ROCM-28013
                "gfx125X-dcgpu",
            ],
            "windows": [
                "gfx908",
                "gfx90a",
                "gfx1030",
                "gfx1100",
                "gfx1101",
                "gfx1102",
                "gfx1103",
                "gfx1150",
                "gfx1151",
                "gfx1152",
                "gfx1153",
                "gfx117X-all",
                "gfx1200",
                "gfx1201",
            ],
        },
    },
    # RAND tests
    "rocrand": {
        "job_name": "rocrand",
        "fetch_artifact_args": "--rand --tests",
        "timeout_minutes": 15,
        "test_script": f"python {_get_script_path('test_runner.py')}",
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
    },
    "hiprand": {
        "job_name": "hiprand",
        "fetch_artifact_args": "--rand --tests",
        "timeout_minutes": 5,
        "test_script": f"python {_get_script_path('test_runner.py')}",
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
    },
    # FFT tests
    "rocfft": {
        "job_name": "rocfft",
        "fetch_artifact_args": "--fft --rand --tests",
        "timeout_minutes": 60,
        "test_script": f"python {_get_script_path('test_runner.py')}",
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 2,
            "windows": 2,
        },
        "exclude_family": {
            "linux": [
                # CRITICAL FAILURE (hang): test causes hang during execution
                "gfx125X-dcgpu",
            ],
        },
    },
    "hipfft": {
        "job_name": "hipfft",
        "fetch_artifact_args": "--fft --rand --tests",
        "timeout_minutes": 60,
        "test_script": f"python {_get_script_path('test_runner.py')}",
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 2,
            "windows": 2,
        },
        "exclude_family": {
            "linux": [
                # CRITICAL FAILURE (hang): test causes hang during execution
                "gfx125X-dcgpu",
            ],
        },
    },
    # hipFFT multi-GPU subset: only the *multi_gpu* hipfft tests, scheduled on
    # 8-GPU runners. Shares the hipfft-test binary/artifacts with the single-GPU
    # "hipfft" job above. test_runner.py rewrites the ctest label to
    # "multigpu_<category>" for this job (see MULTI_GPU_COMPONENTS), so it selects
    # only the multigpu_* labelled tests from hipfft's test_categories.yaml.
    "hipfft-multi-gpu": {
        "job_name": "hipfft-multi-gpu",
        # Selected whenever the base hipfft component is (test:hipfft /
        # PROJECTS_TO_TEST=hipfft), since this variant has no label of its own.
        "base_project": "hipfft",
        "fetch_artifact_args": "--fft --rand --tests",
        "timeout_minutes": 30,
        "test_script": f"python {_get_script_path('test_runner.py')}",
        "platform": ["linux"],
        "total_shards_dict": {
            "linux": 1,
        },
        "test_types": ["comprehensive", "full"],
        "multi_gpu": {"linux": ["gfx94X-dcgpu", "gfx950-dcgpu"]},
    },
    # rocFFT multi-GPU subset: mirror of hipfft-multi-gpu for rocFFT. Selects only
    # the multigpu_* labelled tests from rocfft's test_categories.yaml on 8-GPU
    # runners, sharing the rocfft-test binary/artifacts with the single-GPU
    # "rocfft" job above.
    "rocfft-multi-gpu": {
        "job_name": "rocfft-multi-gpu",
        "base_project": "rocfft",
        "fetch_artifact_args": "--fft --rand --tests",
        "timeout_minutes": 30,
        "test_script": f"python {_get_script_path('test_runner.py')}",
        "platform": ["linux"],
        "total_shards_dict": {
            "linux": 1,
        },
        "test_types": ["comprehensive", "full"],
        "multi_gpu": {"linux": ["gfx94X-dcgpu", "gfx950-dcgpu"]},
    },
    # MIOpen tests
    "miopen": {
        "job_name": "miopen",
        "fetch_artifact_args": "--blas --miopen --rand --tests",
        # GHA step timeout: sized to allow nightly comprehensive runs (~2 hr).
        # Per-test CTest TIMEOUT in rocm-libraries/projects/miopen/test/gtest/
        # test_categories.yaml bounds individual tests (quick: 10 min,
        # standard: 60 min, etc).
        "timeout_minutes": 120,
        "test_script": f"python {_get_script_path('test_runner.py')}",
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 4,
            "windows": 4,
        },
        "exclude_family": {
            "linux": [
                # KNOWN FAILURE: Gemm solver FP16 tests fail on gfx125X
                # individual tests fail but GTEST_FILTER plumbing not available (ctest overrides env var)
                # https://github.com/ROCm/TheRock/actions/runs/35914840519/job/107363782290
                "gfx125X-dcgpu",
            ],
        },
    },
    # MIOpen dbsync (StaticFDBSync) -- GPU-free under the rocjitsu KMD interposer on a CPU runner.
    # The runner ships in the MIOpen dist (share/miopen/bin/run_dbsync_rocjitsu.py, pulled via
    # --miopen; defined in rocm-libraries projects/miopen/test/gtest/dbsync/): it resolves arch + CU
    # list from AMDGPU_FAMILIES, sparse-builds the pinned rocjitsu KMD, and runs StaticFDBSync once
    # per CU with a CU-corrected config. include_family restricts it to the arches rocjitsu has a
    # KMD config for: gfx942, whose FAMILY_MAP entry covers both CU variants -- MI300X (304 CU) and
    # MI300A (228 CU) -- in a single job, gfx950 (MI355X, 256 CU), and gfx90a (MI210, 104 CU, and
    # MI250X GCD, 110 CU -- also in a single job; advisory, see expect_failure below).
    # linux_cpu_runner: no scarce GPU test runner needed; uses the default no_rocm Ubuntu container
    # (the runner sudo-apt-installs cmake/build-essential/libdrm-dev to build rocjitsu).
    "miopen-dbsync": {
        "job_name": "miopen-dbsync",
        "fetch_artifact_args": "--blas --miopen --rand --tests",
        # Standard/comprehensive/full only: "test_types" makes the framework skip this
        # job entirely on the `quick` tier -- no job is scheduled, so no artifact fetch
        # or rocjitsu build is paid for on quick (the runner script also self-skips on
        # TEST_TYPE=quick as a backstop). Runs serially (MIOPEN_DBSYNC_MAX_THREADS=1)
        # under rocjitsu; full set (gfx942 304+228, gfx950 256) + artifact fetch + rocjitsu
        # build measures ~15 min, so 30 gives margin and fails a hung interposer faster.
        "timeout_minutes": 30,
        "test_script": "python ./build/share/miopen/bin/run_dbsync_rocjitsu.py",
        "platform": ["linux"],
        "linux_cpu_runner": True,
        "test_types": ["standard", "comprehensive", "full"],
        # StaticFDBSync skips itself at runtime under ASAN.
        "skip_build_variant": ["asan", "host-asan", "asan-debug", "host-asan-debug"],
        "include_family": {
            "linux": ["gfx942", "gfx950", "gfx90a"],
        },
        # gfx90a is advisory while dbsync under rocjitsu's cdna2 emulation is qualified.
        "expect_failure": {
            "linux": ["gfx90a"],
        },
        "total_shards_dict": {
            "linux": 1,
        },
    },
    # RCCL tests
    "rccl": {
        "job_name": "rccl",
        "fetch_artifact_args": "--rccl --tests",
        "timeout_minutes": 15,
        "test_script": f"pytest {_get_script_path('test_rccl.py')} -v -s --log-cli-level=info",
        "platform": ["linux"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
        # Architectures that we have multi GPU setup for testing
        "multi_gpu": {"linux": ["gfx94X-dcgpu", "gfx950-dcgpu"]},
        # RCCL collective ops use cross-process shared memory via /dev/shm
        "container_options": ["--ipc host"],
    },
    # rocSHMEM tests
    "rocshmem": {
        "job_name": "rocshmem",
        "fetch_artifact_args": "--rocshmem --tests",
        "timeout_minutes": 30,
        "test_script": f"python {_get_script_path('test_runner.py')}",
        "platform": ["linux"],
        "total_shards_dict": {
            "linux": 1,
        },
        # rocSHMEM functional/unit tests launch via mpirun with RANKS 2..64, so
        # they need a multi-GPU runner (same setup as rccl).
        "multi_gpu": {"linux": ["gfx94X-dcgpu", "gfx950-dcgpu"]},
    },
    # rocprofiler-sdk tests
    "rocprofiler-sdk": {
        "job_name": "rocprofiler-sdk",
        "fetch_artifact_args": "--tests",
        "timeout_minutes": 20,
        "additional_requirements_files": [
            _get_artifact_path("share/rocprofiler-sdk/tests/requirements.txt"),
            _get_script_path("requirements-test-ninja.txt"),
        ],
        "test_script": f"python {_get_script_path('test_rocprofiler_sdk.py')} --enable-cdash",
        "platform": ["linux"],
        "container_options": ["--cap-add=SYS_PTRACE"],
        "total_shards_dict": {
            "linux": 1,
        },
        # rocprofv3 mpi-ranks tests gate on find_package(MPI) and launch under
        # mpiexec. OpenMPI is not bundled in TheRock artifacts and is provided via
        # the specialized openmpi image.
        "container_image": "ghcr.io/rocm/no_rocm_image_ubuntu24_04_openmpi@sha256:64f8d6e3e6dfd46e2b1b9321b99aa72fd09140025638e27e5a0d50acb0e69e89",
        "exclude_family": {
            "linux": [
                # CRITICAL FAILURE (pytest hangs): rocprofiler-sdk test hangs during pytest collection
                # https://github.com/ROCm/TheRock/actions/runs/35820932302/job/107052684462
                "gfx125X-dcgpu",
            ],
        },
    },
    # hipDNN tests
    "hipdnn": {
        "job_name": "hipdnn",
        "fetch_artifact_args": "--hipdnn --tests",
        "timeout_minutes": 30,
        "test_script": f"python {_get_script_path('test_runner.py')}",
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
        "exclude_family": {
            "linux": [
                # CRITICAL FAILURE (MES hang): TestGpuLayernormBwdRefValidation.AcceptsValidParamsNormalizeDimThree5D
                # causes MES queue hang. Related: ROCM-31227
                "gfx125X-dcgpu",
            ],
        },
    },
    # hipDNN install/consumption tests
    "hipdnn_install": {
        "job_name": "hipdnn_install",
        "timeout_minutes": 30,
        "additional_requirements_files": [
            _get_script_path("requirements-test-ninja.txt"),
        ],
        "test_script": f"python {_get_script_path('test_hipdnn_install.py')}",
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
    },
    # hipDNN integration tests (unit tests for the integration test harness)
    "hipdnn-integration-tests": {
        "job_name": "hipdnn-integration-tests",
        "fetch_artifact_args": "--hipdnn --hipdnn-integration-tests --tests",
        "timeout_minutes": 30,
        "test_script": f"python {_get_script_path('test_runner.py')}",
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 2,
            "windows": 2,
        },
        "exclude_family": {
            "linux": [
                # GPU tests do not honor ROCR_VISIBLE_DEVICES and utilizes other gpus during test runs. excluding
                "gfx125X-dcgpu",
            ],
        },
    },
    # hipDNN samples tests. Every provider is fetched so that samples whose
    # operations only hipBLASLt (matmul) or hip-kernel-provider (layer norm,
    # resample) implement find an engine instead of skipping.
    "hipdnn-samples": {
        "job_name": "hipdnn-samples",
        "fetch_artifact_args": "--blas --miopen --hipdnn --miopenprovider --hipblasltprovider --hipkernelprovider --hipdnn-samples --tests",
        "timeout_minutes": 30,
        "test_script": f"python {_get_script_path('test_runner.py')}",
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
        "exclude_family": {
            # CRITICAL FAILURE on gfx125X-dcgpu: hipdnn_sample_conv_fprop hangs and
            # becomes a zombie process, blocking the test job indefinitely.
            # See: https://github.com/ROCm/TheRock/actions/runs/15831078820/job/107341707111
            "linux": ["gfx125X-dcgpu"],
        },
    },
    # NVIDIA cuDNN sample corpus built against hipDNN's cuDNN compatibility
    # shim. Distinct from "hipdnn-samples" above, which is hipDNN's own sample
    # suite; these two are easy to confuse and test entirely different things.
    "hipdnn-cudnn-samples": {
        "job_name": "hipdnn-cudnn-samples",
        # No "fetch_artifact_args" on purpose: this job calls
        # find_package(hipdnn_frontend), whose config file and headers live in
        # the "dev" component, and install_rocm_from_artifacts.py expands each
        # selected artifact to <name>_lib plus <name>_test only -- never
        # <name>_dev. Omitting the key takes the fetch-everything branch, which
        # is why "hipdnn_install" omits it too. Do not "fix" this by adding the
        # enumerated artifact flags back; find_package would fail.
        "timeout_minutes": 30,
        "additional_requirements_files": [
            _get_script_path("requirements-test-ninja.txt"),
        ],
        "test_script": f"python {_get_script_path('test_hipdnn_cudnn_samples.py')}",
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
    },
    # profiler-hub install/consumption tests
    "profiler-hub": {
        "job_name": "profiler-hub",
        "timeout_minutes": 5,
        "test_script": f"python {_get_script_path('test_profiler_hub_install.py')}",
        "platform": ["linux"],
        "linux_cpu_runner": True,
        "total_shards_dict": {"linux": 1},
        "exclude_family": {
            # known failure
            "linux": ["gfx125X-dcgpu"],
        },
    },
    # MIOpen provider tests
    "miopenprovider": {
        "job_name": "miopenprovider",
        "fetch_artifact_args": "--blas --miopen --hipdnn --miopenprovider --hipdnn-integration-tests --tests",
        "timeout_minutes": 30,
        "test_script": f"python {_get_script_path('test_runner.py')}",
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 2,
            "windows": 2,
        },
        "exclude_family": {
            "linux": [
                # CRITICAL FAILURE (amd-smi hangs): miopenprovider test hangs during amd-smi GPU detection
                # https://github.com/ROCm/TheRock/actions/runs/35803014951/job/106997748208
                "gfx125X-dcgpu",
            ],
        },
    },
    # hipBLASLt provider tests
    "hipblasltprovider": {
        "job_name": "hipblasltprovider",
        "fetch_artifact_args": "--blas --hipdnn --hipblasltprovider --hipdnn-integration-tests --tests",
        "timeout_minutes": 30,
        "test_script": f"python {_get_script_path('test_runner.py')}",
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
        "exclude_family": {
            "linux": [
                # KNOWN FAILURE: TestGpuMatmulPlan and TestHipblasltMatmulPlanBuilder tests fail
                # individual tests fail but GTEST_FILTER plumbing not available (ctest overrides env var)
                # https://github.com/ROCm/TheRock/actions/runs/35914840519/job/107363782678
                "gfx125X-dcgpu",
            ],
        },
    },
    # hip-kernel-provider tests. test_hipkernelprovider.py installs the staged
    # rocKE wheels, then delegates to test_runner.py.
    "hipkernelprovider": {
        "job_name": "hipkernelprovider",
        "fetch_artifact_args": "--hipdnn --hipkernelprovider --hipdnn-integration-tests --tests",
        "timeout_minutes": 30,
        # TODO: Use the copy in the hipkernelprovider test artifact after rocKE
        # installs its component-owned requirements file.
        "additional_requirements_files": [
            "build_tools/github_actions/test_executable_scripts/requirements-test-hipkernelprovider.txt",
        ],
        "test_script": f"python {_get_script_path('test_hipkernelprovider.py')}",
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 4,
            "windows": 4,
        },
        "exclude_family": {
            "linux": [
                # CRITICAL FAILURE (hang): test hangs during execution
                "gfx125X-dcgpu",
            ],
        },
    },
    # rocWMMA tests
    "rocwmma": {
        "job_name": "rocwmma",
        "fetch_artifact_args": "--rocwmma --tests --blas",
        # Headroom above typical shard runtime; per-test CTest timeouts fail fast on hangs (ROCM-24171).
        "timeout_minutes": 90,
        "test_script": f"python {_get_script_path('test_runner.py')}",
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 5,
            "windows": 2,
        },
        "exclude_family": {
            "linux": [
                # rocWMMA does not support gfx103X (see TheRock#1944)
                "gfx1030",
                # CRITICAL FAILURE (GPU hang): rocwmma test causes GPU hang during parallel test execution
                # https://github.com/ROCm/TheRock/actions/runs/36077293577
                "gfx125X-dcgpu",
            ],
        },
    },
    # rocALUTION tests
    "rocalution": {
        "job_name": "rocalution",
        "fetch_artifact_args": "--rocalution --tests --blas --sparse --rand",
        "timeout_minutes": 30,
        "test_script": f"python {_get_script_path('test_runner.py')}",
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
        "exclude_family": {
            # CRITICAL FAILURE (GPU hang): rocalution test causes MES queue hang during parallel test execution
            # https://github.com/ROCm/TheRock/actions/runs/36077293577
            "linux": ["gfx125X-dcgpu"],
        },
    },
    # profiler tests
    "rocprofiler-compute": {
        "job_name": "rocprofiler-compute",
        "fetch_artifact_args": "--rocprofiler-compute --rocprofiler-sdk --tests",
        "timeout_minutes": 60,
        "additional_requirements_files": [
            _get_artifact_path("libexec/rocprofiler-compute/requirements.txt"),
            _get_artifact_path("libexec/rocprofiler-compute/requirements-test.txt"),
        ],
        "test_script": f"python {_get_script_path('test_runner.py')}",
        "platform": ["linux"],
        "total_shards_dict": {"linux": 1},
        "exclude_family": {
            # rocprofiler-compute supports gfx908, gfx90a, gfx942, gfx950,
            # gfx115X and gfx1250 (see TheRock#2892)
            "linux": [
                "gfx1030",
                "gfx1100",
                "gfx1101",
                "gfx1102",
                "gfx1103",
                "gfx1200",
                "gfx1201",
                # GPU tests do not honor ROCR_VISIBLE_DEVICES and utilizes other gpus during test runs. excluding
                "gfx125X-dcgpu",
            ],
        },
    },
    "rocprofiler-systems": {
        "job_name": "rocprofiler-systems",
        "fetch_artifact_args": "--rocprofiler-systems --rocprofiler-systems-examples --rocprofiler-sdk --tests",
        "timeout_minutes": 60,
        "additional_requirements_files": [
            _get_artifact_path("share/rocprofiler-systems/tests/requirements.txt"),
        ],
        "test_script": f"python {_get_script_path('test_runner.py')}",
        "platform": ["linux"],
        "total_shards_dict": {
            "linux": 1,
        },
        "container_options": ["--cap-add=SYS_PTRACE", "--cap-add=PERFMON"],
        "exclude_family": {
            "linux": [
                # CRITICAL FAILURE (amd-smi hangs): rocprofiler-systems test hangs during amd-smi GPU detection
                # https://github.com/ROCm/TheRock/actions/runs/35820932302/job/107052684535
                "gfx125X-dcgpu",
            ],
        },
    },
    # libhipcxx amdclang++ tests (formerly libhipcxx_hipcc)
    "libhipcxx_amdclang": {
        "job_name": "libhipcxx_amdclang",
        # --libhipcxx --tests also pulls prim_dev (rocThrust/hipCUB/rocPRIM
        # headers and configs) for the c2h tests.
        "fetch_artifact_args": "--libhipcxx --tests",
        "timeout_minutes": 30,
        "additional_requirements_files": [
            _get_artifact_path("libhipcxx/requirements-test.txt"),
            _get_script_path("requirements-test-ninja.txt"),
        ],
        "test_script": f"python {_get_script_path('test_libhipcxx_amdclang.py')}",
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
        "exclude_family": {
            "linux": [
                # GPU tests do not honor ROCR_VISIBLE_DEVICES and utilizes other gpus during test runs. excluding
                "gfx125X-dcgpu",
            ],
        },
    },
    # libhipcxx hiprtc tests
    "libhipcxx_hiprtc": {
        "job_name": "libhipcxx_hiprtc",
        "fetch_artifact_args": "--libhipcxx --tests",
        "timeout_minutes": 20,
        "additional_requirements_files": [
            _get_artifact_path("libhipcxx/requirements-test.txt"),
            _get_script_path("requirements-test-ninja.txt"),
        ],
        "test_script": f"python {_get_script_path('test_libhipcxx_hiprtc.py')}",
        "platform": ["linux"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
        "exclude_family": {
            "linux": [
                # GPU tests do not honor ROCR_VISIBLE_DEVICES and utilizes other gpus during test runs. excluding
                "gfx125X-dcgpu",
            ],
        },
    },
    # hipthreads lit tests
    "hipthreads": {
        "job_name": "hipthreads",
        "fetch_artifact_args": "--hipthreads --tests",
        "timeout_minutes": 30,
        "additional_requirements_files": [
            _get_artifact_path("hipthreads/test/requirements-test.txt"),
        ],
        "test_script": f"python {_get_script_path('test_hipthreads.py')}",
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
        "exclude_family": {
            "linux": [
                # KNOWN FAILURE (hipErrorNoBinaryForGpu/hsa-hotswap errors - fundamental gfx1250 arch issue)
                # https://github.com/ROCm/TheRock/actions/runs/35881596668/job/107251861504
                "gfx125X-dcgpu",
            ],
            "windows": [
                "gfx117X-all",
            ],
        },
    },
    # hipthreads example apps (build + run consumer samples against the artifact).
    "hipthreads_examples": {
        "job_name": "hipthreads_examples",
        # --prim pulls rocThrust/rocPrim (roc::rocthrust); --rand pulls hipRAND
        # (the InOneWeekend example includes <hiprand/hiprand.hpp>).
        "fetch_artifact_args": "--hipthreads --prim --rand --tests",
        "timeout_minutes": 30,
        "additional_requirements_files": [
            _get_script_path("requirements-test-ninja.txt"),
        ],
        "test_script": f"python {_get_script_path('test_hipthreads_examples.py')}",
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
    },
    "rocdecode": {
        "job_name": "rocdecode",
        "fetch_artifact_args": "--rocdecode --tests",
        "timeout_minutes": 10,
        "additional_requirements_files": [
            _get_script_path("requirements-test-ninja.txt"),
        ],
        "test_script": f"python {_get_script_path('test_rocdecode.py')}",
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
        # rocdecode requires FFmpeg dev libraries (libavcodec-dev, libavformat-dev,
        # libavutil-dev) for the extended (FFmpeg-demux) test builds, which run only
        # on Linux via the specialized media image below. Windows runs the always-on
        # tests (raw decode, caps, negative API) with ENABLE_EXTENDED_TESTS=OFF via
        # the vaon12 backend, so it needs no FFmpeg and no container image.
        "container_image": "ghcr.io/rocm/no_rocm_image_ubuntu24_04_media@sha256:7853bb7fc56e8ddd768e849b5594d0b7f2e89aad9618e29943888ee9d916c6d2",
    },
    "rocjpeg": {
        "job_name": "rocjpeg",
        "fetch_artifact_args": "--rocjpeg --tests",
        "timeout_minutes": 10,
        "additional_requirements_files": [
            _get_script_path("requirements-test-ninja.txt"),
        ],
        "test_script": f"python {_get_script_path('test_rocjpeg.py')}",
        "platform": ["linux"],
        "total_shards_dict": {
            "linux": 1,
        },
    },
    "rpp": {
        "job_name": "rpp",
        "fetch_artifact_args": "--rpp --tests",
        # Sized for comprehensive/full, which runs the perf suites serially and
        # upstream allows 4000s each. quick and standard are far under this.
        # TODO(ROCm/rocm-libraries#10187): lower once perf tests are split out
        # of the correctness suite.
        "timeout_minutes": 60,
        "additional_requirements_files": [
            _get_script_path("requirements-test-ninja.txt"),
        ],
        "test_script": f"python {_get_script_path('test_rpp.py')}",
        "platform": ["linux"],
        "total_shards_dict": {
            "linux": 1,
        },
    },
    # aqlprofile tests
    "aqlprofile": {
        "job_name": "aqlprofile",
        "fetch_artifact_args": "--aqlprofile --tests",
        "timeout_minutes": 5,
        "test_script": f"python {_get_script_path('test_aqlprofile.py')}",
        "platform": ["linux"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
    },
    # rocrtst tests
    "rocrtst": {
        "job_name": "rocrtst",
        "fetch_artifact_args": "--rocrtst --tests",
        "timeout_minutes": 15,
        "test_script": f"python {_get_script_path('test_runner.py')}",
        "platform": ["linux", "wsl"],
        # Advisory on WSL until its results are stable.
        "expect_failure": {"wsl": ["gfx110X-all"]},
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
            "wsl": 1,
        },
    },
    # hipTensor tests
    "hiptensor": {
        "job_name": "hiptensor",
        "fetch_artifact_args": "--hiptensor --tests",
        # Suppress per-test kpack debug logging (test_component.yml defaults it to
        # "1" for diagnostics, which floods this suite's -V ctest output).
        "rocm_kpack_debug": "0",
        # Github Actions step timeout, applied to every tier (it does not vary by test_type).
        # Must be sized for the largest tier the nightly runs (comprehensive),
        # not quick/standard -- otherwise the step is killed mid-suite well
        # before ctest's own per-test --timeout 7200 can take effect. See
        # rocm-libraries/projects/hiptensor/test_categories.yaml
        # execution_settings.category_timeouts (full: 7200s = 2h).
        "timeout_minutes": 120,
        "test_script": f"python {_get_script_path('test_runner.py')}",
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
        "exclude_family": {
            # hipTensor requires composable_kernel, which is filtered out on some platforms,
            # so no hipTensor test artifact is produced for that family (see TheRock#2074).
            "linux": [
                "gfx900",
                "gfx90c",
                "gfx906",
                "gfx101X-all",
                "gfx103X-all",
                # CRITICAL FAILURE (test hangs): hiptensor test hangs during execution
                # https://github.com/ROCm/TheRock/actions/runs/35816223373/job/107038471042
                "gfx125X-dcgpu",
            ],
            "windows": ["gfx900", "gfx90c", "gfx906", "gfx101X-all", "gfx103X-all"],
        },
    },
}


# ---------------------------------------------------------------------------
# Test plan construction
# ---------------------------------------------------------------------------
#
# A "test plan" is the compact, per-GPU-family answer to "which components run,
# on which runner, at which tier". configure_multi_arch_ci.py builds one plan
# per family during setup and passes it to test_artifacts.yml, which expands it
# into jobs without any configuration step of its own.
#
# Only family-varying data belongs in a plan, because every byte travels through
# a workflow output. Static per-component execution config (test_script,
# fetch_artifact_args, ...) is identical for every family, so test_component.yml
# re-reads it from this module via describe_component() once it is already on the
# test runner.


# Marks a selected component as needing a multi-GPU runner while the plan is
# being assembled. Popped once a concrete label lands in "test_runner".
_NEEDS_MULTI_GPU = "_needs_multi_gpu"


def _appends_tensilelite_ctest(key: str, test_type: str) -> bool:
    """Whether this component chains the C++/ctest stage after its pytest stage.

    tensilelite runs the tensilelite/tests C++ gtest suite (via ctest -L
    <test_type>, driven by the shared test_runner.py) after the existing pytest
    stage, for every tier except quick -- that component's test_categories.yaml
    only defines standard/comprehensive/full so far (promote to quick once the
    standard tier proves stable). See AIHPBLAS-4410.

    TODO(#7851): this is a temporary special-case. test_runner.py only knows how
    to run the C++/ctest suite today, so the pytest and ctest stages have to be
    chained here instead. Fold both into test_runner.py's own dual-mode support
    and drop this branch once that lands.
    """
    return key == "tensilelite" and test_type != "quick"


def _component_timeout_minutes(key: str, test_type: str) -> int:
    """Job timeout for a component. Must agree with describe_component()."""
    timeout_minutes = test_matrix[key]["timeout_minutes"]
    if _appends_tensilelite_ctest(key, test_type):
        # +15 min over the pytest-only baseline for the added ctest stage;
        # re-measure once CI timing is observed and adjust.
        timeout_minutes += 15
    return timeout_minutes


def describe_component(key: str, platform: str, test_type: str) -> dict:
    """Static execution config for one component, looked up by test_matrix key.

    These fields are the same for every GPU family and every runner, so they are
    resolved inside the test job rather than copied into each family's test plan.
    """
    entry = {**_common_settings, **test_matrix[key]}

    test_script = entry["test_script"]
    if _appends_tensilelite_ctest(key, test_type):
        test_script += (
            " && TEST_COMPONENT=hipblaslt-tensilelite "
            f"python {_get_script_path('test_runner.py')}"
        )

    return {
        "test_script": test_script,
        "fetch_artifact_args": entry["fetch_artifact_args"],
        # Joined here because its only consumer is
        # install_additional_requirements.py --requirements-files.
        "additional_requirements_files": ",".join(
            entry["additional_requirements_files"]
        ),
        # Most components enable the kpack loader's per-test diagnostics; a few
        # opt out via "rocm_kpack_debug": "0" to keep -V ctest logs readable.
        "rocm_kpack_debug": entry.get("rocm_kpack_debug", "1"),
    }


@dataclass
class FamilyRunners:
    """Runner pools for one GPU family, already gated by CI policy.

    configure_multi_arch_ci.py resolves these while expanding its per-family
    build config, where trigger gating, ASAN sandbox routing and kernel-specific
    runners are already applied. An empty ``test_runs_on`` means GPU tests are
    gated off for this family: only CPU-only components are scheduled.
    """

    test_runs_on: str = ""
    test_runs_on_cpu: str = ""
    test_runs_on_labels: list[dict] | None = None
    test_runs_on_multi_gpu: str = ""
    test_runs_on_multi_gpu_labels: list[dict] | None = None
    test_runs_on_sandbox: str = ""


@dataclass
class TestPlanRequest:
    """Everything needed to decide the test plan for one GPU family."""

    platform: str
    runners: FamilyRunners = field(default_factory=FamilyRunners)
    amdgpu_families: str = ""
    # Individual gfx targets for this family, used by the include_family /
    # exclude_family / expect_failure filters alongside the family group string.
    family_gfx_targets: list[str] = field(default_factory=list)
    projects_to_test: str = "*"
    test_type: str = "standard"
    test_labels: list[str] = field(default_factory=list)
    build_variant: str = "release"
    skip_stages: list[str] = field(default_factory=list)


def resolve_family_runners(
    platform: str,
    amdgpu_families: str,
    test_runs_on: str | None,
    test_runs_on_cpu: str,
) -> tuple[FamilyRunners, str, list[str]]:
    """Resolve runner pools for a family from the static family matrix.

    Only the standalone CLI path needs this; configure_multi_arch_ci.py has
    already resolved the same values (and has applied external family overrides
    that this static view does not see).

    Returns the runner pools, the family's forced test type ("" when the family
    does not pin one) and the family's individual gfx targets.
    """
    runners = FamilyRunners(
        test_runs_on_cpu=test_runs_on_cpu or get_cpu_test_runner(platform)
    )
    if not amdgpu_families:
        return runners, "", []

    shortened_family = amdgpu_families.split("-")[0].lower()
    all_families = get_all_families_for_trigger_types(
        ["presubmit", "postsubmit", "nightly"]
    )
    if shortened_family not in all_families:
        return runners, "", []

    # wsl has no entry of its own here; its runner is TEST_RUNS_ON.
    platform_info = all_families[shortened_family].get(platform, {})

    # An explicitly empty TEST_RUNS_ON carries a policy decision (e.g. trigger
    # gating) and gates GPU tests, including multi-GPU ones. Leaving it unset
    # falls back to the static matrix.
    if test_runs_on != "":
        runners.test_runs_on = test_runs_on or platform_info.get("test-runs-on", "")
        runners.test_runs_on_labels = platform_info.get("test-runs-on-labels")
        runners.test_runs_on_multi_gpu = platform_info.get("test-runs-on-multi-gpu", "")
        runners.test_runs_on_multi_gpu_labels = platform_info.get(
            "test-runs-on-multi-gpu-labels"
        )
    runners.test_runs_on_sandbox = platform_info.get("test-runs-on-sandbox", "")

    return (
        runners,
        platform_info.get("test_type_for_family") or "",
        platform_info.get("fetch-gfx-targets", []),
    )


def build_test_plan(request: TestPlanRequest) -> dict:
    """Select the components to test for one GPU family and assign their runners.

    Returns the test plan consumed by test_artifacts.yml:

        {
          "platform": "linux",
          # The two defaults nearly every component uses, emitted once per
          # family; components that declare extra options carry their own
          # "container_options" string instead.
          "container_options": "<GPU default>",
          "container_options_cpu": "<CPU-only default>",
          "sanity": {...} | None,   # must pass before "components" run
          "components": [{...}, ...],
        }
    """
    platform = request.platform
    amdgpu_families = request.amdgpu_families
    test_type = request.test_type
    runners = request.runners

    # Check for ci:run-multi-gpu label to force multi-GPU tests
    enable_multi_gpu_by_label = "ci:run-multi-gpu" in request.test_labels
    if enable_multi_gpu_by_label:
        logging.info("Multi-GPU tests forced via ci:run-multi-gpu label")

    if not runners.test_runs_on:
        logging.info(
            "GPU tests gated (no GPU runner), only CPU-only components will run"
        )

    logging.info(f"Selecting projects: {request.projects_to_test}")
    logging.info(f"Using test_matrix ({len(test_matrix)} test(s))")

    # This string -> array conversion ensures no partial strings are detected
    # during test selection (ex: "hipblas" in ["hipblaslt", "rocblas"] = false)
    project_array = [item.strip() for item in request.projects_to_test.split(",")]

    # Filter out ci: control labels - they're not test component selectors.
    # Note: Benchmarks never use test_labels (always empty list)
    component_test_labels = [c for c in request.test_labels if not c.startswith("ci:")]
    parsed_test_labels = [c.split("test:")[-1] for c in component_test_labels]
    expanded_test_labels = [
        member
        for label in parsed_test_labels
        for member in TEST_LABEL_GROUPS.get(label, [label])
    ]

    selected = []
    for key in test_matrix:
        job_name = test_matrix[key]["job_name"]

        excluded_stages = set(request.skip_stages).intersection(
            test_matrix[key].get("skip_stages", [])
        )
        if excluded_stages:
            logging.info(
                f"Excluding {job_name}: stages {sorted(excluded_stages)} are skipped"
            )
            continue

        if request.build_variant in test_matrix[key].get("skip_build_variant", []):
            logging.info(
                f"Excluding job {job_name}: skip_build_variant lists "
                f"{request.build_variant}"
            )
            continue

        # include_family (opt-in) and exclude_family (opt-out) together decide
        # whether a job runs: it runs only when it matches an include (if any are
        # listed for this platform) and matches no exclude. Matching is exact
        # membership against either the family group string (e.g. "gfx120X-all")
        # or an individual gfx target within that family (e.g. "gfx1200").
        _include_list = test_matrix[key].get("include_family", {}).get(platform, [])
        if _include_list and not _family_matches(
            _include_list, amdgpu_families, request.family_gfx_targets
        ):
            logging.info(
                f"Excluding job {job_name} for platform {platform} and family "
                f"{amdgpu_families}: not listed in include_family"
            )
            continue

        _exclude_list = test_matrix[key].get("exclude_family", {}).get(platform, [])
        if _exclude_list and _family_matches(
            _exclude_list, amdgpu_families, request.family_gfx_targets
        ):
            logging.info(
                f"Excluding job {job_name} for platform {platform} and family "
                f"{amdgpu_families}"
            )
            continue

        # Key used for test-label / project selection. Split jobs (e.g. the
        # multi-GPU variants) declare a "base_project" so that selecting the base
        # component (via `test:hipfft` label or PROJECTS_TO_TEST=hipfft) also
        # picks up the variant, which otherwise has no matching label of its own.
        selection_key = test_matrix[key].get("base_project", key)

        # If test labels are populated, and the test job name is not in the test
        # labels, skip the test.
        if (
            key != "sanity"
            and expanded_test_labels
            and selection_key not in expanded_test_labels
        ):
            logging.info(f"Excluding job {job_name} since it's not in the test labels")
            continue

        # Tier gate: a component may declare which test tiers it runs on via
        # "test_types". Skip it entirely (schedule no job) for any TEST_TYPE not
        # in the list -- e.g. miopen-dbsync runs standard/comprehensive/full
        # only, never quick, and the hipfft/rocfft multi-GPU split jobs run
        # comprehensive/full only. Omit the field to run on every tier.
        allowed_test_types = test_matrix[key].get("test_types")
        if allowed_test_types and test_type not in allowed_test_types:
            logging.info(
                f"Excluding job {job_name}: test_type {test_type} not in "
                f"{allowed_test_types}"
            )
            continue

        # If the test is enabled for a particular platform and a particular (or
        # all) projects are selected. Split jobs match via selection_key
        # (base_project) so PROJECTS_TO_TEST=hipfft also selects hipfft-multi-gpu.
        if platform not in test_matrix[key]["platform"] or not (
            key == "sanity" or selection_key in project_array or "*" in project_array
        ):
            continue

        logging.info(f"Requesting job {job_name} with test_type {test_type}")

        # For CI testing, we construct a shard array based on "total_shards_dict".
        # This way, the test jobs will be split up into X shards. (ex: [1, 2, 3,
        # 4] = 4 test shards). For display purposes, we add "i + 1" for the job
        # name (ex: 1 of 4). During the actual test sharding in the test
        # executable, this array will become 0th index.
        total_shards = test_matrix[key].get("total_shards_dict", {}).get(platform, 1)
        # If the test type is quick tests, we only need one shard for the test job
        if test_type == "quick":
            total_shards = 1

        entry = {
            "key": key,
            "job_name": job_name,
            "timeout_minutes": _component_timeout_minutes(key, test_type),
            "total_shards": total_shards,
            "shard_arr": [i + 1 for i in range(total_shards)],
        }

        # A component declares the families it is advisory on via
        # "expect_failure". Resolve that declaration down to the single boolean
        # the workflows read -- test_component.yml compares it with `== true` to
        # set continue-on-error, so the emitted value must never stay a map.
        _xfail = test_matrix[key].get("expect_failure")
        if _xfail and _family_matches(
            _xfail.get(platform, []), amdgpu_families, request.family_gfx_targets
        ):
            entry["expect_failure"] = True

        # Routes the component to a CPU-only machine and selects the CPU-only
        # container options; test_component.yml also skips its GPU probe.
        if test_matrix[key].get("linux_cpu_runner", False):
            entry["linux_cpu_runner"] = True

        # Components that pre-pin a runner (e.g. rocgdb-corefile needs machines
        # with GPU core-dump support) keep it; the loop below leaves it alone.
        if "test_runner" in test_matrix[key]:
            entry["test_runner"] = test_matrix[key]["test_runner"]

        if "container_image" in test_matrix[key]:
            entry["container_image"] = test_matrix[key]["container_image"]

        # Only components declaring extra options need their own string; the
        # rest fall back to the plan-level defaults.
        if "container_options" in test_matrix[key]:
            entry["container_options"] = _build_container_options(
                platform,
                cpu_only=entry.get("linux_cpu_runner", False),
                job_options=test_matrix[key]["container_options"],
                amdgpu_families=amdgpu_families,
            )

        # If the test requires multi GPU testing, we use a multi-GPU test runner
        # for this specific test. Inside the "multi_gpu" field, we have a mapping
        # of amdgpu_family -> bool (if multi GPU testing is enabled for that
        # family). If the multi GPU test runner is not enabled, we skip the test.
        if "multi_gpu" in test_matrix[key]:
            # Jobs that require multi-GPU runners only run on standard,
            # comprehensive, or full tiers by default (capacity constraint).
            if test_type == "quick" and not enable_multi_gpu_by_label:
                logging.info(
                    f"Excluding job {job_name}: multi-GPU tests skipped for quick runs"
                )
                continue

            family_has_multi_gpu = (
                platform in test_matrix[key]["multi_gpu"]
                and amdgpu_families in test_matrix[key]["multi_gpu"][platform]
            )
            if not (family_has_multi_gpu or enable_multi_gpu_by_label):
                logging.info(
                    f"Excluding job {job_name} since multi GPU testing is not "
                    f"available for family {amdgpu_families}"
                )
                continue

            # Mark this component as needing a multi-GPU runner. The actual
            # runner selection is done in the per-component loop below, which
            # resolves it into "test_runner" like every other component so the
            # workflows have a single field to route on.
            entry[_NEEDS_MULTI_GPU] = True
            logging.info(f"Including job {job_name} for multi-GPU testing")

        # Hip-tests on Windows run with both PAL and ROCR backends.
        # See: https://github.com/ROCm/TheRock/issues/3587
        if key == "hip-tests" and platform == "windows":
            for backend, enable_pal in [("PAL", "1"), ("ROCR", "0")]:
                selected.append(
                    {
                        **entry,
                        "job_name": f"hip-tests ({backend})",
                        "gpu_enable_pal": enable_pal,
                    }
                )
            continue

        selected.append(entry)

    # Per-component runner selection for better load distribution. Each
    # component gets its own independent random draw based on configured weights.
    # For ASan builds, use the sandbox runner to isolate potentially failing
    # tests. This matches multiple build variants, including "asan", "host-asan",
    # "asan-debug", and "host-asan-debug".
    logging.info("")
    logging.info("Assigning runners to requested jobs...")
    is_asan_build = "asan" in request.build_variant
    components = []
    for component in selected:
        job_name = component["job_name"]
        if component.pop(_NEEDS_MULTI_GPU, False):
            # Multi-GPU components use multi-GPU runner labels
            if runners.test_runs_on_multi_gpu_labels:
                component["test_runner"] = select_weighted_label(
                    runners.test_runs_on_multi_gpu_labels, f"{job_name}-multi-gpu"
                )
            elif runners.test_runs_on_multi_gpu:
                component["test_runner"] = runners.test_runs_on_multi_gpu
            else:
                # No multi-GPU runner configured for this family; skip it
                logging.info(
                    f"  Excluding {job_name}: multi-GPU required but no multi-GPU "
                    f"runner configured"
                )
                continue
        elif "test_runner" not in component:
            # Regular components use standard runner labels. Skip if test_runner
            # is already pre-pinned (e.g. rocgdb-corefile).
            if component.get("linux_cpu_runner", False):
                if not runners.test_runs_on_cpu:
                    logging.info(
                        f"  Excluding {job_name}: CPU runner required but none "
                        f"configured"
                    )
                    continue
                component["test_runner"] = runners.test_runs_on_cpu
                logging.info(
                    f"  {job_name}: CPU-only, using runner: {runners.test_runs_on_cpu}"
                )
            elif is_asan_build and runners.test_runs_on_sandbox:
                # For ASAN builds, use the sandbox runner if available
                component["test_runner"] = runners.test_runs_on_sandbox
                logging.info(
                    f"  {job_name}: using ASAN sandbox runner: "
                    f"{runners.test_runs_on_sandbox}"
                )
            elif runners.test_runs_on_labels:
                component["test_runner"] = select_weighted_label(
                    runners.test_runs_on_labels, job_name
                )
            elif runners.test_runs_on:
                component["test_runner"] = runners.test_runs_on
            else:
                # No GPU runner available and component requires GPU - skip it
                logging.info(
                    f"  Excluding {job_name}: GPU runner required but none configured"
                )
                continue
        components.append(component)

    # Separate sanity from the regular component matrix: it must pass before the
    # others run.
    sanity = next((c for c in components if c["key"] == "sanity"), None)

    return {
        "platform": platform,
        # Uniform across the plan: a family either pins its own tier via
        # test_type_for_family or inherits the run-wide one.
        "test_type": test_type,
        "container_options": _build_container_options(
            platform, amdgpu_families=amdgpu_families
        ),
        "container_options_cpu": _build_container_options(
            platform, cpu_only=True, amdgpu_families=amdgpu_families
        ),
        "sanity": sanity,
        "components": [c for c in components if c["key"] != "sanity"],
    }


def dump_test_plan(plan: dict) -> str:
    """Serialize a test plan for a workflow output.

    Compact separators, because a nightly run carries one of these per GPU
    family per platform through a single job's outputs.
    """
    return json.dumps(plan, separators=(",", ":"))


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--platform",
        type=str,
        default=platform_module.system().lower(),
        help=(
            "Platform to configure tests for: linux, windows, or wsl (the Linux"
            " build on a WSL-hosted runner, passed as TEST_RUNS_ON)"
        ),
    )
    parser.add_argument(
        "--describe-component",
        type=str,
        metavar="KEY",
        help=(
            "Instead of building a test plan, emit the static execution config"
            " for one component (its test_matrix key). Used by test_component.yml"
            " so that config does not have to travel through workflow outputs."
        ),
    )
    args, _ = parser.parse_known_args()
    platform = args.platform
    test_type = os.getenv("TEST_TYPE", "standard")

    if args.describe_component:
        gha_set_output(describe_component(args.describe_component, platform, test_type))
        return

    amdgpu_families = os.getenv("AMDGPU_FAMILIES", "")
    # configure_multi_arch_ci.py passes ",".join(fetch-gfx-targets): no spaces to strip.
    amdgpu_targets = [t for t in os.getenv("AMDGPU_TARGETS", "").split(",") if t]

    # Check if a GPU runner was passed from the workflow. This carries the policy
    # decision (e.g. trigger gating). When set to empty, GPU tests are gated but
    # CPU-only tests (linux_cpu_runner: True) can still run.
    runners, test_type_for_family, family_gfx_targets = resolve_family_runners(
        platform=platform,
        amdgpu_families=amdgpu_families,
        test_runs_on=os.getenv("TEST_RUNS_ON"),
        test_runs_on_cpu=os.getenv("TEST_RUNS_ON_CPU", ""),
    )

    # Enforce test_type_for_family if set in the family matrix. This is a strict
    # override - families with limited hardware (e.g. MI455) should always run
    # quick tests regardless of what's passed in TEST_TYPE.
    if test_type_for_family and test_type_for_family != test_type:
        logging.info(
            f"Overriding test_type from '{test_type}' to '{test_type_for_family}' "
            f"(test_type_for_family for {amdgpu_families})"
        )
        test_type = test_type_for_family

    plan = build_test_plan(
        TestPlanRequest(
            platform=platform,
            runners=runners,
            amdgpu_families=amdgpu_families,
            # AMDGPU_TARGETS (fetch-gfx-targets after the caller's family
            # overrides) wins over the static matrix.
            family_gfx_targets=amdgpu_targets or family_gfx_targets,
            projects_to_test=os.getenv("PROJECTS_TO_TEST", "*"),
            test_type=test_type,
            test_labels=_parse_test_labels(os.getenv("TEST_LABELS") or "[]"),
            build_variant=os.getenv("BUILD_VARIANT", "release"),
            skip_stages=os.getenv("SKIP_STAGES", "").split(","),
        )
    )

    gha_set_output({"test_plan": dump_test_plan(plan)})


if __name__ == "__main__":
    run()
