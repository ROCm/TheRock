import os

from setuptools import find_packages, setup

# Version is defined centrally for all ROCm wheels
from rocm_sdk import _dist_info as dist_info  # type: ignore

version = os.environ.get("ROCM_SDK_VERSION", dist_info.__version__)
if version == "DEFAULT":
    # Allows standalone template builds (Phase 2)
    version = "0.0.0.dev0"

setup(
    name="rocm-sdk-profilers",
    version=version,  # <-- use the computed version
    description="ROCm profiler applications (rocprofiler-systems and rocprofiler-compute)",
    author="AMD",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "rocprof-compute=rocm_sdk_profilers._cli:rocprof_compute",
            "rocprof-sys-avail=rocm_sdk_profilers._cli:rocprof_sys_avail",
            "rocprof-sys-causal=rocm_sdk_profilers._cli:rocprof_sys_causal",
            "rocprof-sys-instrument=rocm_sdk_profilers._cli:rocprof_sys_instrument",
            "rocprof-sys-run=rocm_sdk_profilers._cli:rocprof_sys_run",
            "rocprof-sys-sample=rocm_sdk_profilers._cli:rocprof_sys_sample",
        ],
    },
    python_requires=">=3.8",
)