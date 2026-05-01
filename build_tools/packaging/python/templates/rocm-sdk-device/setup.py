# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ROCm SDK device package: per-ISA .kpack archives and kernel databases.

Device wheels overlay into the rocm-sdk-libraries package's platform directory
so the kpack runtime finds .kpack files alongside host shared libraries.
"""

import importlib.util
import os
from setuptools import setup
import sys
import sysconfig
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent


def import_dist_info():
    dist_info_path = THIS_DIR / "src" / "rocm_sdk_device" / "_dist_info.py"
    if not dist_info_path.exists():
        raise RuntimeError(f"No _dist_info.py file found: {dist_info_path}")
    module_name = "rocm_sdk_dist_info"
    spec = importlib.util.spec_from_file_location(module_name, dist_info_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


dist_info = import_dist_info()
my_package = dist_info.ALL_PACKAGES["device"]

# The platform directory is named to match the libraries wheel's platform
# package so that device files (.kpack, kernel DBs) overlay into the same
# site-packages directory as the host shared libraries.
# LIBRARIES_PY_PACKAGE_NAME is baked into _dist_info.py by the build system.
platform_py_package = dist_info.LIBRARIES_PY_PACKAGE_NAME
platform_dir = THIS_DIR / "platform" / platform_py_package
packages = [platform_py_package]

# Collect all files in the platform directory as package_data.
# include_package_data=True relies on VCS tracking which doesn't apply here.
# Device files include .kpack archives (in dotfile dirs), kernel databases
# (.co, .dat, .hsaco), MIOpen perf DBs (.kdb, .db.txt), and ML models.
package_data_files = []
if platform_dir.is_dir():
    for path in platform_dir.rglob("*"):
        if path.is_file() and path.name != "__init__.py":
            package_data_files.append(str(path.relative_to(platform_dir)))

setup(
    name=my_package.get_dist_package_name(target_family=dist_info.THIS_TARGET_FAMILY),
    version=dist_info.__version__,
    packages=packages,
    package_dir={
        platform_py_package: f"platform/{platform_py_package}",
    },
    package_data={
        platform_py_package: package_data_files,
    },
    install_requires=[
        f"rocm-sdk-libraries=={dist_info.__version__}",
    ],
    zip_safe=False,
    options={
        "bdist_wheel": {
            "plat_name": os.getenv(
                "ROCM_SDK_WHEEL_PLATFORM_TAG", sysconfig.get_platform()
            ),
        },
    },
)
