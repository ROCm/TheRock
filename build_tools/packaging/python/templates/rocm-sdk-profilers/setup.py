# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Main rocm-sdk-profilers (OS specific)."""


import os

from setuptools import find_packages, setup

import importlib.util


# Version is defined centrally for all ROCm wheels via a local _dist_info.py
def _load_local_dist_info():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    dist_info_path = os.path.join(this_dir, "_dist_info.py")
    spec = importlib.util.spec_from_file_location("_dist_info", dist_info_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load local _dist_info module from {dist_info_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dist_info = _load_local_dist_info()

version = os.environ.get("ROCM_SDK_VERSION", dist_info.__version__)
if version == "DEFAULT":
    # Allows standalone template builds (Phase 2)
    version = "0.0.0.dev0"

setup(
    name="rocm-sdk-profilers",
    version=version,  # the computed version
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
