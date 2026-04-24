# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Main rocm-profiler (OS specific)."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import platform
import sysconfig

from setuptools import find_packages, setup


THIS_DIR = Path(__file__).resolve().parent


def _load_local_dist_info():
    """Load the per-wheel _dist_info.py generated into this template at build time."""
    dist_info_path = THIS_DIR / "src" / "rocm_profiler" / "_dist_info.py"
    if not dist_info_path.exists():
        raise ImportError(f"Cannot find local _dist_info.py at: {dist_info_path}")

    spec = importlib.util.spec_from_file_location("_dist_info", dist_info_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load local _dist_info module from {dist_info_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dist_info = _load_local_dist_info()
my_package = dist_info.ALL_PACKAGES["profiler"]
packages = find_packages(where="./src")
platform_package_name = my_package.get_py_package_name()
packages.append(platform_package_name)

version = os.environ.get("ROCM_SDK_VERSION")
if version is None:
    version = dist_info.__version__

if version == "DEFAULT":
    version = "0.0.0.dev0"


setup(
    name="rocm-profiler",
    version=version,
    description="ROCm profiler applications (rocprofiler-systems and rocprofiler-compute)",
    packages=packages,
    package_dir={
        "": "src",
        platform_package_name: f"platform/{platform_package_name}",
    },
    include_package_data=True,
    zip_safe=False,
    options={
        "bdist_wheel": {
            "plat_name": os.getenv(
                "ROCM_SDK_WHEEL_PLATFORM_TAG", sysconfig.get_platform()
            ),
        },
    },
    entry_points={
        "console_scripts": (
            [
                "rocprof-compute=rocm_profiler._cli:rocprof_compute",
                "rocprof-sys-attach=rocm_profiler._cli:rocprof_sys_attach",
                "rocprof-sys-avail=rocm_profiler._cli:rocprof_sys_avail",
                "rocprof-sys-causal=rocm_profiler._cli:rocprof_sys_causal",
                "rocprof-sys-instrument=rocm_profiler._cli:rocprof_sys_instrument",
                "rocprof-sys-run=rocm_profiler._cli:rocprof_sys_run",
                "rocprof-sys-sample=rocm_profiler._cli:rocprof_sys_sample",
                "rocprof-sys-python=rocm_profiler._cli:rocprof_sys_python",
            ]
            if platform.system() != "Windows"
            else []
        ),
    },
    python_requires=">=3.8",
)
