# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Main rocm-sdk-profilers (OS specific)."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from setuptools import find_packages, setup


THIS_DIR = Path(__file__).resolve().parent


def _load_local_dist_info():
    """Load the per-wheel _dist_info.py generated into this template at build time."""
    dist_info_path = THIS_DIR / "_dist_info.py"
    if not dist_info_path.exists():
        raise ImportError(f"Cannot find local _dist_info.py at: {dist_info_path}")

    spec = importlib.util.spec_from_file_location("_dist_info", dist_info_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load local _dist_info module from {dist_info_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dist_info = _load_local_dist_info()

version = os.environ.get("ROCM_SDK_VERSION", dist_info.__version__)
if version == "DEFAULT":
    # Fallback used for standalone template builds during Phase 2 testing.
    # The packaging pipeline will always provide ROCM_SDK_VERSION.
    version = "0.0.0.dev0"

setup(
    name="rocm-sdk-profilers",
    version=version,
    description="ROCm profiler applications (rocprofiler-systems and rocprofiler-compute)",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
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
