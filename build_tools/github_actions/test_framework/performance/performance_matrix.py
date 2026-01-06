"""
Performance test matrix definitions.

Performance tests characterize system performance with detailed metrics,
scaling analysis, and bottleneck identification.
"""

from pathlib import Path

SCRIPT_DIR = Path("build_tools/github_actions/test_framework/performance/scripts")

performance_matrix = {
    # Example performance test (uncomment and customize):
    # "rocblas_performance": {
    #     "job_name": "rocBLAS Performance Characterization",
    #     "fetch_artifact_args": "--blas",
    #     "timeout_minutes": 90,
    #     "test_script": f"python {SCRIPT_DIR}/test_rocblas_performance.py",
    #     "platform": ["linux"],
    #     "metrics": ["tflops", "latency", "bandwidth"],
    #     "targets": {
    #         "tflops_min": 40.0,
    #         "latency_max_ms": 2.0
    #     }
    # },
}
