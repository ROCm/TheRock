"""
Functional test matrix definitions.

This module contains the functional_matrix dictionary which defines all functional tests.
"""

from pathlib import Path

# Note: these paths are relative to the repository root.
SCRIPT_DIR = Path("tests") / "extended_tests" / "functional" / "scripts"


def _get_functional_script_path(script_name: str) -> str:
    platform_path = SCRIPT_DIR / script_name
    # Convert to posix (using `/` instead of `\\`) so test workflows can use
    # 'bash' as the shell on Linux and Windows.
    posix_path = platform_path.as_posix()
    return str(posix_path)


functional_matrix = {
    "test_hip_samples": {
        "job_name": "test_hip_samples",
        "fetch_artifact_args": "--base-only --tests",
        "timeout_minutes": 60,
        "test_script": f"python {_get_functional_script_path('test_hip_samples.py')}",
        # TODO(lajagapp): Add windows support (https://github.com/ROCm/TheRock/issues/3207)
        "platform": ["linux"],
        "total_shards": 1,
    },
    "test_targetid_bit_extract": {
        "job_name": "test_targetid_bit_extract",
        "fetch_artifact_args": "--base-only --tests",
        "timeout_minutes": 30,
        "test_script": (
            "python "
            f"{_get_functional_script_path('test_targetid_bit_extract.py')}"
        ),
        "platform": ["linux"],
        "total_shards": 1,
    },
}
