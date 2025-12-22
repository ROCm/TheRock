"""
Performance test matrix definitions.

This module contains the performance_matrix dictionary which defines all performance tests.
Performance tests run only on nightly CI builds and are merged into test_matrix by configure_ci.py.
"""

from pathlib import Path

# Note: these paths are relative to the repository root.
SCRIPT_DIR = Path("build_tools") / "github_actions" / "test_framework" / "scripts" / "performance"


def _get_performance_script_path(script_name: str) -> str:
    platform_path = SCRIPT_DIR / script_name
    # Convert to posix (using `/` instead of `\\`) so test workflows can use
    # 'bash' as the shell on Linux and Windows.
    posix_path = platform_path.as_posix()
    return str(posix_path)


performance_matrix = {
    # BLAS performance tests
    "hipblaslt_perf": {
        "job_name": "hipblaslt_perf",
        "fetch_artifact_args": "--blas --tests",
        "timeout_minutes": 60,
        "test_script": f"python {_get_performance_script_path('test_hipblaslt_perf.py')}",
        # TODO(lajagapp): Add windows support (https://github.com/ROCm/TheRock/issues/2478)
        "platform": ["linux"],
        "total_shards": 1,
        # TODO: Remove xfail once dedicated performance servers are added in "benchmark-runs-on"
        "expect_failure": True,
    },
    # SOLVER performance tests
    "rocsolver_perf": {
        "job_name": "rocsolver_perf",
        "fetch_artifact_args": "--blas --tests",
        "timeout_minutes": 60,
        "test_script": f"python {_get_performance_script_path('test_rocsolver_perf.py')}",
        # TODO(lajagapp): Add windows support (https://github.com/ROCm/TheRock/issues/2478)
        "platform": ["linux"],
        "total_shards": 1,
        # TODO: Remove xfail once dedicated performance servers are added in "benchmark-runs-on"
        "expect_failure": True,
    },
    # RAND performance tests
    "rocrand_perf": {
        "job_name": "rocrand_perf",
        "fetch_artifact_args": "--rand --tests",
        "timeout_minutes": 60,
        "test_script": f"python {_get_performance_script_path('test_rocrand_perf.py')}",
        # TODO(lajagapp): Add windows support (https://github.com/ROCm/TheRock/issues/2478)
        "platform": ["linux"],
        "total_shards": 1,
        # TODO: Remove xfail once dedicated performance servers are added in "benchmark-runs-on"
        "expect_failure": True,
    },
    # FFT performance tests
    "rocfft_perf": {
        "job_name": "rocfft_perf",
        "fetch_artifact_args": "--fft --rand --tests",
        "timeout_minutes": 60,
        "test_script": f"python {_get_performance_script_path('test_rocfft_perf.py')}",
        # TODO(lajagapp): Add windows support (https://github.com/ROCm/TheRock/issues/2478)
        "platform": ["linux"],
        "total_shards": 1,
        # TODO: Remove xfail once dedicated performance servers are added in "benchmark-runs-on"
        "expect_failure": True,
    },
}
