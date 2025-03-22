"""Main rocm-sdk-libraries (OS specific)."""

import importlib.util
from setuptools import setup, find_packages
import sys
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
packages = [pure_py_package]
platform_py_package = my_package.get_py_package_name(
    target_family=dist_info.THIS_TARGET_FAMILY
)
packages.append(platform_py_package)
print("Found packages:", packages)

setup(
    name=my_package.get_dist_package_name(target_family=dist_info.THIS_TARGET_FAMILY),
    version=dist_info.__version__,
    packages=packages,
    package_dir={
        # Note that in the template, the package is not target qualified but
        # at dist time it is.
        f"{pure_py_package}": f"src/rocm_sdk_libraries",
        f"{platform_py_package}": f"platform/{platform_py_package}",
    },
    zip_safe=False,
    include_package_data=True,
    entry_points={
        "console_scripts": [],
    },
)
