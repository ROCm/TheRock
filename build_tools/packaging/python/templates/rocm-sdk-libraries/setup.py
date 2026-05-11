# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Main rocm-sdk-libraries (OS specific)."""

import importlib.util
import os
from setuptools import setup, find_packages
import sys
import sysconfig
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent


# The built package contains a pre-generated _dist_info.py file, which would
# normally be accessible at runtime. However, to make it available at
# package build time (here!), we have to dynamically import it.
def import_dist_info():
    dist_info_path = THIS_DIR / "src" / "rocm_sdk_libraries" / "_dist_info.py"
    if not dist_info_path.exists():
        raise RuntimeError(f"No _dist_info.py file found: {dist_info_path}")
    module_name = "rocm_sdk_dist_info"
    spec = importlib.util.spec_from_file_location(module_name, dist_info_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


dist_info = import_dist_info()
my_package = dist_info.ALL_PACKAGES["libraries"]
print(f"Loaded dist_info package: {my_package}")
pure_py_package = f"rocm_sdk_libraries_{dist_info.THIS_TARGET_FAMILY}"
platform_py_package = my_package.get_py_package_name(
    target_family=dist_info.THIS_TARGET_FAMILY
)

# Discover all packages under src/. This finds rocm_sdk_libraries (the core
# pure-Python package) plus any promoted Python packages from library artifacts
# (e.g. hipdnn_frontend).
discovered = find_packages(where="./src")

packages = []
package_dir = {
    f"{platform_py_package}": f"platform/{platform_py_package}",
}
for pkg in discovered:
    if pkg == "rocm_sdk_libraries" or pkg.startswith("rocm_sdk_libraries."):
        # Remap unqualified rocm_sdk_libraries to target-qualified name.
        qualified = pkg.replace("rocm_sdk_libraries", pure_py_package, 1)
        packages.append(qualified)
        package_dir[qualified] = f"src/{pkg.replace('.', '/')}"
    else:
        # Promoted packages (e.g. hipdnn_frontend) keep their name.
        packages.append(pkg)
        package_dir[pkg] = f"src/{pkg.replace('.', '/')}"
packages.append(platform_py_package)
print("Found packages:", packages)

setup(
    name=my_package.get_dist_package_name(target_family=dist_info.THIS_TARGET_FAMILY),
    version=dist_info.__version__,
    packages=packages,
    package_dir=package_dir,
    zip_safe=False,
    include_package_data=True,
    options={
        "bdist_wheel": {
            "plat_name": os.getenv(
                "ROCM_SDK_WHEEL_PLATFORM_TAG", sysconfig.get_platform()
            ),
        },
    },
    entry_points={
        "console_scripts": [],
    },
)
