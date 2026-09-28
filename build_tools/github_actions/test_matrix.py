# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared test matrix definitions for CI test configuration.

This module contains the canonical test_matrix dict defining all test components,
TEST_LABEL_GROUPS for label expansion, and the _family_matches helper function.
Used by both configure_multi_arch_ci.py (upfront selection) and
fetch_test_configurations.py (runtime configuration).
"""

# Maps a group label (the part after "test:") to the individual test matrix
# keys it expands to. Use this when a single label should select multiple
# related jobs without relying on name-prefix inference.
TEST_LABEL_GROUPS: dict[str, list[str]] = {
    "rocgdb": ["rocgdb-cpu", "rocgdb-gpu", "rocgdb-corefile"],
    "tensilelite": ["tensilelite", "tensilelite-common"],
}


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


# Common settings for rocgdb jobs
_rocgdb_common = {
    "fetch_artifact_args": "--debug-tools --tests",
    "timeout_minutes": 30,
    "platform": ["linux"],
    "total_shards": 1,
    "container_image": "ghcr.io/rocm/no_rocm_image_ubuntu24_04_rocgdb@sha256:aa3f8966fcdefca04d4c04fb10ae7f8b654d1bb1cc6a894ea7089e5a01953197",  # 2026-07-22T15:21:18.527038581Z
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
# "gfx125X-dcgpu"). Examples:
#   "exclude_family": {"linux": ["gfx1030"]}                # skip a single target
#   "include_family": {"linux": ["gfx908", "gfx90a", "gfx942"]}  # opt in to a set
#
# A component may restrict which test tiers it runs on via "test_types", a list of
# allowed TEST_TYPE values (any of "quick", "standard", "comprehensive", "full").
# When set, the component is skipped entirely -- no job is scheduled -- for any tier
# not in the list; omit the field to run on every tier (the default). For example, a
# component whose suite is too slow for the quick sanity tier opts out of it with:
#   "test_types": ["standard", "comprehensive", "full"]

test_matrix = {
    # Sanity tests - always run first as a prerequisite for other component tests
    "sanity": {
        "job_name": "sanity",
        "fetch_artifact_args": "--base-only",
        "timeout_minutes": 5,
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
        # Running docker with cap-add and -v /lib/modules, by recommendation of GitHub:
        # https://rocm.docs.amd.com/projects/amdsmi/en/amd-staging/how-to/setup-docker-container.html
        "container_options": ["--cap-add SYS_MODULE", "-v /lib/modules:/lib/modules"],
    },
    # hip-tests
    "hip-tests": {
        "job_name": "hip-tests",
        "fetch_artifact_args": "--hip-tests --tests",
        "timeout_minutes": 120,
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 4,
            "windows": 4,
        },
    },
    # hipFile (storage-libs) unit tests. CPU-only (mocked), so they run quickly
    # and do not require a GPU runner.
    "hipfile": {
        "job_name": "hipfile",
        "fetch_artifact_args": "--hipfile --tests",
        "timeout_minutes": 15,
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
        # GHA step timeout: max category timeout in rocBLAS should be 24 hours / 6 shards = 4 hours per shard
        # 240 min + 20% margin = 288 min
        "timeout_minutes": 288,
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 6,
            "windows": 6,
        },
    },
    "rocroller": {
        "job_name": "rocroller",
        "fetch_artifact_args": "--blas --tests",
        "timeout_minutes": 60,
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
        # Python/pytest suite only (rocisa + TensileLite unit). The C++ gtest
        # suite (tensilelite/tests) is appended below for TEST_TYPE != quick;
        # see the "tensilelite" special-case in the component loop
        # (AIHPBLAS-4410).
        "platform": ["linux"],
        "total_shards_dict": {
            "linux": 1,
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
        "platform": ["linux"],
        "total_shards_dict": {
            "linux": 1,
        },
        "include_family": {
            "linux": ["gfx90a", "gfx94X-dcgpu", "gfx950-dcgpu", "gfx120X-all"],
        },
    },
    "origami": {
        "job_name": "origami",
        "fetch_artifact_args": "--blas --tests",
        "timeout_minutes": 5,
        "platform": ["linux", "windows"],
        "total_shards": 1,
    },
    "hipblas": {
        "job_name": "hipblas",
        "fetch_artifact_args": "--blas --solver --tests",
        "timeout_minutes": 30,
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
        "platform": ["linux"],
        "total_shards_dict": {
            "linux": 1,
        },
    },
    "hipblaslt": {
        "job_name": "hipblaslt",
        "fetch_artifact_args": "--blas --tests",
        "timeout_minutes": 180,
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 6,
            "windows": 1,
        },
        "exclude_family": {
            # hipBLASLt does not support gfx103X (see TheRock#1062)
            "linux": ["gfx1030"],
        },
    },
    # SOLVER tests
    "hipsolver": {
        "job_name": "hipsolver",
        "fetch_artifact_args": "--solver --blas --sparse --tests",
        "timeout_minutes": 5,
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
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
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
    },
    "rocgdb-cpu": {
        **_rocgdb_common,
        "job_name": "rocgdb-cpu",
        "linux_cpu_runner": True,
    },
    "rocgdb-gpu": {
        **_rocgdb_common,
        "job_name": "rocgdb-gpu",
    },
    # Corefile tests require specific hardware support (GPU core dump capable runners).
    # test_runner is pre-pinned so the family-based runner selection loop skips it.
    # Only gfx942 has core-dump support, so include_family opts the job in to that
    # family alone rather than enumerating every other architecture to exclude.
    "rocgdb-corefile": {
        **_rocgdb_common,
        "job_name": "rocgdb-corefile",
        "test_runner": "linux-gfx942-gpu-rocm-mathlib",
        "include_family": {
            "linux": ["gfx942"],
        },
    },
    "rocr-debug-agent": {
        "job_name": "rocr-debug-agent",
        "fetch_artifact_args": "--debug-tools --tests",
        "timeout_minutes": 10,
        "platform": ["linux"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
    },
    "rocthrust": {
        "job_name": "rocthrust",
        "fetch_artifact_args": "--prim --tests",
        "timeout_minutes": 45,
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
    },
    # SPARSE tests
    "hipsparse": {
        "job_name": "hipsparse",
        "fetch_artifact_args": "--sparse --blas --tests",
        "timeout_minutes": 30,
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 3,
            "windows": 3,
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
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 3,
            "windows": 3,
        },
    },
    "hipsparselt": {
        "job_name": "hipsparselt",
        "fetch_artifact_args": "--sparse --blas --tests",
        # GHA step timeout: max category timeout in hipsparselt should be 6 hours / 6 shards = 60 min per shard
        # 60 min + 20% margin = 72 min
        "timeout_minutes": 72,
        "platform": ["linux"],
        "total_shards_dict": {
            "linux": 6,
            "windows": 1,
        },
        "exclude_family": {
            # hipsparselt does not support gfx908, gfx90a (see TheRock#2042)
            # hipsparselt does not support gfx110X architectures (TensileLibrary missing)
            # hipsparselt does not plan to support Linux and Windows gfx115X architectures
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
                "gfx1200",
                "gfx1201",
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
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 2,
            "windows": 2,
        },
    },
    "hipfft": {
        "job_name": "hipfft",
        "fetch_artifact_args": "--fft --rand --tests",
        "timeout_minutes": 60,
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 2,
            "windows": 2,
        },
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
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 4,
            "windows": 4,
        },
    },
    # MIOpen dbsync (StaticFDBSync) -- GPU-free under the rocjitsu KMD interposer on a CPU runner.
    # The runner ships in the MIOpen dist (share/miopen/bin/run_dbsync_rocjitsu.py, pulled via
    # --miopen; defined in rocm-libraries projects/miopen/test/gtest/dbsync/): it resolves arch + CU
    # list from AMDGPU_FAMILIES, sparse-builds the pinned rocjitsu KMD, and runs StaticFDBSync once
    # per CU with a CU-corrected config. include_family restricts it to gfx942, whose FAMILY_MAP
    # entry covers both CU variants -- MI300X (304 CU) and MI300A (228 CU) -- in a single job.
    # linux_cpu_runner: no scarce GPU test runner needed; uses the default no_rocm Ubuntu container
    # (the runner sudo-apt-installs cmake/build-essential/libdrm-dev to build rocjitsu).
    "miopen-dbsync": {
        "job_name": "miopen-dbsync",
        "fetch_artifact_args": "--blas --miopen --rand --tests",
        # Standard/comprehensive/full only: "test_types" makes the framework skip this
        # job entirely on the `quick` tier -- no job is scheduled, so no artifact fetch
        # or rocjitsu build is paid for on quick (the runner script also self-skips on
        # TEST_TYPE=quick as a backstop). Runs serially (MIOPEN_DBSYNC_MAX_THREADS=1)
        # under rocjitsu; full set (gfx942 304+228) + artifact fetch + rocjitsu build
        # measures ~15 min, so 30 gives margin and fails a hung interposer faster.
        "timeout_minutes": 30,
        "platform": ["linux"],
        "linux_cpu_runner": True,
        "test_types": ["standard", "comprehensive", "full"],
        "include_family": {
            "linux": ["gfx942"],
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
        "platform": ["linux"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
        # Architectures that we have multi GPU setup for testing
        "multi_gpu": {"linux": ["gfx94X-dcgpu", "gfx950-dcgpu"]},
    },
    # rocSHMEM tests
    "rocshmem": {
        "job_name": "rocshmem",
        "fetch_artifact_args": "--rocshmem --tests",
        "timeout_minutes": 30,
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
        "platform": ["linux"],
        "container_options": ["--cap-add=SYS_PTRACE"],
        "total_shards_dict": {
            "linux": 1,
        },
        # rocprofv3 mpi-ranks tests gate on find_package(MPI) and launch under
        # mpiexec. OpenMPI is not bundled in TheRock artifacts and is provided via
        # the specialized openmpi image.
        "container_image": "ghcr.io/rocm/no_rocm_image_ubuntu24_04_openmpi@sha256:f67d0b02cae8faf0d2f3e4a1de38a01af6bad2eb27f10a5e07bf19748a84d1e6",
    },
    # hipDNN tests
    "hipdnn": {
        "job_name": "hipdnn",
        "fetch_artifact_args": "--hipdnn --tests",
        "timeout_minutes": 30,
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
    },
    # hipDNN install/consumption tests
    "hipdnn_install": {
        "job_name": "hipdnn_install",
        "timeout_minutes": 30,
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
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
    },
    # hipDNN samples tests
    "hipdnn-samples": {
        "job_name": "hipdnn-samples",
        "fetch_artifact_args": "--blas --miopen --hipdnn --miopenprovider --hipdnn-samples --tests",
        "timeout_minutes": 30,
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
    },
    # MIOpen provider tests
    "miopenprovider": {
        "job_name": "miopenprovider",
        "fetch_artifact_args": "--blas --miopen --hipdnn --miopenprovider --hipdnn-integration-tests --tests",
        "timeout_minutes": 30,
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
    },
    # hipBLASLt provider tests
    "hipblasltprovider": {
        "job_name": "hipblasltprovider",
        "fetch_artifact_args": "--blas --hipdnn --hipblasltprovider --hipdnn-integration-tests --tests",
        "timeout_minutes": 30,
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
    },
    # hip-kernel-provider tests. test_hipkernelprovider.py installs the staged
    # rocKE wheels, then delegates to test_runner.py.
    "hipkernelprovider": {
        "job_name": "hipkernelprovider",
        "fetch_artifact_args": "--hipdnn --hipkernelprovider --hipdnn-integration-tests --tests",
        "timeout_minutes": 30,
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
    },
    # rocWMMA tests
    "rocwmma": {
        "job_name": "rocwmma",
        "fetch_artifact_args": "--rocwmma --tests --blas",
        # Headroom above typical shard runtime; per-test CTest timeouts fail fast on hangs (ROCM-24171).
        "timeout_minutes": 90,
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 5,
            "windows": 2,
        },
        "exclude_family": {
            # rocWMMA does not support gfx103X (see TheRock#1944)
            "linux": ["gfx1030"],
        },
    },
    # rocALUTION tests
    "rocalution": {
        "job_name": "rocalution",
        "fetch_artifact_args": "--rocalution --tests --blas --sparse --rand",
        "timeout_minutes": 30,
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
    },
    # profiler tests
    "rocprofiler-compute": {
        "job_name": "rocprofiler-compute",
        "fetch_artifact_args": "--rocprofiler-compute --rocprofiler-sdk --tests",
        "timeout_minutes": 60,
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
            ],
        },
    },
    "rocprofiler-systems": {
        "job_name": "rocprofiler-systems",
        "fetch_artifact_args": "--rocprofiler-systems --rocprofiler-systems-examples --rocprofiler-sdk --tests",
        "timeout_minutes": 60,
        "platform": ["linux"],
        "total_shards_dict": {
            "linux": 1,
        },
        "container_options": ["--cap-add=SYS_PTRACE", "--cap-add=PERFMON"],
    },
    # libhipcxx amdclang++ tests (formerly libhipcxx_hipcc)
    "libhipcxx_amdclang": {
        "job_name": "libhipcxx_amdclang",
        "fetch_artifact_args": "--libhipcxx --tests",
        "timeout_minutes": 30,
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
    },
    # libhipcxx hiprtc tests
    "libhipcxx_hiprtc": {
        "job_name": "libhipcxx_hiprtc",
        "fetch_artifact_args": "--libhipcxx --tests",
        "timeout_minutes": 20,
        "platform": ["linux"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
    },
    # hipthreads lit tests
    "hipthreads": {
        "job_name": "hipthreads",
        "fetch_artifact_args": "--hipthreads --tests",
        "timeout_minutes": 30,
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
    },
    # hipthreads example apps (build + run consumer samples against the artifact).
    "hipthreads_examples": {
        "job_name": "hipthreads_examples",
        # --prim pulls rocThrust/rocPrim (roc::rocthrust); --rand pulls hipRAND
        # (the InOneWeekend example includes <hiprand/hiprand.hpp>).
        "fetch_artifact_args": "--hipthreads --prim --rand --tests",
        "timeout_minutes": 30,
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
        "platform": ["linux"],
        "total_shards_dict": {
            "linux": 1,
        },
        # rocdecode requires FFmpeg dev libraries (libavcodec-dev, libavformat-dev,
        # libavutil-dev) for test builds. These are not bundled in TheRock
        # artifacts and are provided via the specialized media image.
        "container_image": "ghcr.io/rocm/no_rocm_image_ubuntu24_04_media@sha256:d715ae2db664b055c90343e00588ce9ac3eec387513fe359396e5e08e75521ca",
    },
    "rocjpeg": {
        "job_name": "rocjpeg",
        "fetch_artifact_args": "--rocjpeg --tests",
        "timeout_minutes": 10,
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
        "platform": ["linux"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
    },
    # hipTensor tests
    "hiptensor": {
        "job_name": "hiptensor",
        "fetch_artifact_args": "--hiptensor --tests",
        # Github Actions step timeout, applied to every tier (it does not vary by test_type).
        # Must be sized for the largest tier the nightly runs (comprehensive),
        # not quick/standard -- otherwise the step is killed mid-suite well
        # before ctest's own per-test --timeout 7200 can take effect. See
        # rocm-libraries/projects/hiptensor/test_categories.yaml
        # execution_settings.category_timeouts (full: 7200s = 2h).
        "timeout_minutes": 120,
        "platform": ["linux", "windows"],
        "total_shards_dict": {
            "linux": 1,
            "windows": 1,
        },
        "exclude_family": {
            # hipTensor requires composable_kernel, which is filtered out on some platforms,
            # so no hipTensor test artifact is produced for that family (see TheRock#2074).
            "linux": ["gfx900", "gfx90c", "gfx906", "gfx101X-all", "gfx103X-all"],
            "windows": ["gfx900", "gfx90c", "gfx906", "gfx101X-all", "gfx103X-all"],
        },
    },
}
