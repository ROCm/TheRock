# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pre-built hipdnn_frontend wheel setup.

Differs from the rocm-sdk-* templates: those templates ship loose ROCm shared
libraries (not CPython extensions) and produce a `py3-none-<plat>` wheel via
`include_package_data=True` + a `platform/<pkg>/` source layout. This wheel,
in contrast, contains the nanobind extension `hipdnn_frontend_python` built
against the CPython Limited API (stable ABI), so it is a platform binary that
installs on any interpreter at or above the ABI floor.

`Distribution.has_ext_modules() -> True` marks this as a platform wheel (so the
tag carries `<plat>` rather than `any`) even though we invoke no setuptools
build extension — the .so/.pyd is pre-built and staged into the package
directory by the driver script. `bdist_wheel`'s `py_limited_api` then sets the
ABI tag to `abi3` with the impl floor `cp{LIMITED}`, yielding
`cp{LIMITED}-abi3-<plat>`. The floor must match the Limited-API version the
extension was compiled against (hipDNN: Python 3.12, via STABLE_ABI +
Development.SABIModule).
"""

import os
import sysconfig

from setuptools import setup, find_packages, Distribution

# CPython Limited-API floor the extension is built against. Must match the
# `find_package(Python 3.12 ... Development.SABIModule)` + STABLE_ABI build.
_PY_LIMITED_API = "cp312"


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
            "py_limited_api": _PY_LIMITED_API,
        },
    },
)
