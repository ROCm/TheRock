# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pre-built hipdnn_frontend wheel setup.

Mirrors the rocm-sdk-* template setups: plain `setup()` with
`Distribution.has_ext_modules() -> True` so bdist_wheel emits a
`cp{X}{Y}-cp{X}{Y}-<plat>` tag for the CPython-ABI-bound
`hipdnn_frontend_python.so`.
"""

import os
import sysconfig

from setuptools import setup, find_packages, Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


# Must match EXPECTED_PKG_NAME in pack_python_wheel.py; the driver stages
# the source tree under this name into the build dir.
_pkg = "hipdnn_frontend"
_packages = find_packages(where=".", include=[_pkg, f"{_pkg}.*"]) or [_pkg]

setup(
    distclass=BinaryDistribution,
    packages=_packages,
    package_data={p: ["**/*"] for p in _packages},
    exclude_package_data={
        p: ["**/__pycache__/*", "**/*.pyc", "**/*.pyo"] for p in _packages
    },
    include_package_data=False,
    zip_safe=False,
    options={
        "bdist_wheel": {
            "plat_name": os.getenv(
                "ROCM_SDK_WHEEL_PLATFORM_TAG", sysconfig.get_platform()
            ),
        },
    },
)
