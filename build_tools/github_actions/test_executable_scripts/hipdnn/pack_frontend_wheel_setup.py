# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pre-built hipdnn_frontend wheel setup.

Differs from the rocm-sdk-* templates: those templates ship loose ROCm shared
libraries (not CPython extensions) and produce a `py3-none-<plat>` wheel via
`include_package_data=True` + a `platform/<pkg>/` source layout. This wheel,
in contrast, contains `hipdnn_frontend_python.so` — a nanobind extension
linked to a specific CPython ABI — so it must install only on a matching
interpreter.

`Distribution.has_ext_modules() -> True` forces bdist_wheel to emit the
CPython-ABI tag `cp{X}{Y}-cp{X}{Y}-<plat>` even though we are not invoking
any setuptools build extension (the .so is pre-built and staged into the
package directory by the driver script).
"""

import os
import sysconfig

from setuptools import setup, find_packages, Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


# Must match EXPECTED_PKG_NAME in pack_frontend_wheel.py; the driver stages
# the source tree under this name into the build dir.
_pkg = "hipdnn_frontend"
_packages = find_packages(where=".", include=[_pkg, f"{_pkg}.*"])
if not _packages:
    raise RuntimeError(
        f"find_packages found no {_pkg!r} package; wheel staging is broken"
    )

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
