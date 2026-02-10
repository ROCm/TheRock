"""
Benchmark test matrix definitions.

This module contains the benchmark_matrix dictionary which defines all benchmark tests.
Benchmark tests run only on nightly CI builds and are merged into test_matrix by configure_ci.py.
"""

from pathlib import Path

# Note: these paths are relative to the repository root.
SCRIPT_DIR = Path("build_tools") / "github_actions" / "benchmarks" / "scripts"


def _get_benchmark_script_path(script_name: str) -> str:
    platform_path = SCRIPT_DIR / script_name
    # Convert to posix (using `/` instead of `\\`) so test workflows can use
    # 'bash' as the shell on Linux and Windows.
    posix_path = platform_path.as_posix()
    return str(posix_path)


benchmark_matrix = {
    # BLAS benchmark tests
    "rocblas_bench": {
        "job_name": "rocblas_bench",
        "fetch_artifact_args": "--blas --tests",
        "timeout_minutes": 90,
        "test_script": f"python {_get_benchmark_script_path('test_rocblas_benchmark.py')}",
        # TODO(lajagapp): Add windows support (https://github.com/ROCm/TheRock/issues/2478)
        "platform": ["linux"],
        "total_shards": 1,
        # TODO: Remove xfail once dedicated performance servers are added in "benchmark-runs-on"
        "expect_failure": True,
    },
    "hipblaslt_bench": {
        "job_name": "hipblaslt_bench",
        "fetch_artifact_args": "--blas --tests",
        "timeout_minutes": 60,
        "test_script": f"python {_get_benchmark_script_path('test_hipblaslt_benchmark.py')}",
        "platform": ["linux", "windows"],
        "total_shards": 1,
        # TODO: Remove xfail once dedicated performance servers are added in "benchmark-runs-on"
        "expect_failure": True,
    },
    # SOLVER benchmark tests
    "rocsolver_bench": {
        "job_name": "rocsolver_bench",
        "fetch_artifact_args": "--blas --tests",
        "timeout_minutes": 60,
        "test_script": f"python {_get_benchmark_script_path('test_rocsolver_benchmark.py')}",
        "platform": ["linux", "windows"],
        "total_shards": 1,
        # TODO: Remove xfail once dedicated performance servers are added in "benchmark-runs-on"
        "expect_failure": True,
    },
    # RAND benchmark tests
    "rocrand_bench": {
        "job_name": "rocrand_bench",
        "fetch_artifact_args": "--rand --tests",
        "timeout_minutes": 90,
        "test_script": f"python {_get_benchmark_script_path('test_rocrand_benchmark.py')}",
        "platform": ["linux", "windows"],
        "total_shards": 1,
        # TODO: Remove xfail once dedicated performance servers are added in "benchmark-runs-on"
        "expect_failure": True,
    },
    # FFT benchmark tests
    "rocfft_bench": {
        "job_name": "rocfft_bench",
        "fetch_artifact_args": "--fft --rand --tests",
        "timeout_minutes": 60,
        "test_script": f"python {_get_benchmark_script_path('test_rocfft_benchmark.py')}",
        "platform": ["linux", "windows"],
        "total_shards": 1,
        # TODO: Remove xfail once dedicated performance servers are added in "benchmark-runs-on"
        "expect_failure": True,
    },
    # Communication benchmark tests
    # DISABLED: RCCL Performance Benchmark - Waiting for OpenMPI integration
    # TODO: Enable after OpenMPI is added to TheRock (Issue #2887, blocked by #1284)
    # "rccl_bench": {
    #    "job_name": "rccl_bench",
    #    "fetch_artifact_args": "--rccl --tests",
    #    "timeout_minutes": 90,
    #    "test_script": f"python {_get_benchmark_script_path('test_rccl_benchmark.py')}",
    #    # TODO(lajagapp): Add windows support (https://github.com/ROCm/TheRock/issues/2478)
    #    "platform": ["linux"],
    #    "total_shards": 1,
    #    # TODO: Remove xfail once dedicated performance servers are added in "benchmark-runs-on"
    #    "expect_failure": True,
    # },
}
