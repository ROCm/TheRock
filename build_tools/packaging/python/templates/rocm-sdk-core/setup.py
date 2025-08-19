"""Main rocm-sdk-core (OS specific)."""

import importlib.util
import os
import platform
from setuptools import setup, find_packages
import sys
import sysconfig
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent


# The built package contains a pre-generated _dist_info.py file, which would
# normally be accessible at runtime. However, to make it available at
# package build time (here!), we have to dynamically import it.
def import_dist_info():
    dist_info_path = THIS_DIR / "src" / "rocm_sdk_core" / "_dist_info.py"
    if not dist_info_path.exists():
        raise RuntimeError(f"No _dist_info.py file found: {dist_info_path}")
    module_name = "rocm_sdk_dist_info"
    spec = importlib.util.spec_from_file_location(module_name, dist_info_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


dist_info = import_dist_info()
my_package = dist_info.ALL_PACKAGES["core"]
print(f"Loaded dist_info package: {my_package}")
packages = find_packages(where="./src")
platform_package_name = my_package.get_py_package_name()
packages.append(platform_package_name)
print("Found packages:", packages)

setup(
    name=f"rocm-sdk-core",
    version=dist_info.__version__,
    packages=packages,
    package_dir={
        "": "src",
        platform_package_name: f"platform/{platform_package_name}",
    },
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
        "console_scripts": [
            "amdclang=rocm_sdk_core._cli:amdclang",
            "amdclang++=rocm_sdk_core._cli:amdclangpp",
            "amdclang-cpp=rocm_sdk_core._cli:amdclang_cpp",
            "amdclang-cl=rocm_sdk_core._cli:amdclang_cl",
            "amdgpu-arch=rocm_sdk_core._cli:amdgpu_arch",
            "amdflang=rocm_sdk_core._cli:amdflang",
            "amdlld=rocm_sdk_core._cli:amdlld",
            "hipcc=rocm_sdk_core._cli:hipcc",
            "hipconfig=rocm_sdk_core._cli:hipconfig",
            "roc-obj=rocm_sdk_core._cli:roc_obj",
            "roc-obj-extract=rocm_sdk_core._cli:roc_obj_extract",
            "roc-obj-ls=rocm_sdk_core._cli:roc_obj_ls",
        ]
        + (
            [
                # These tools are only available on Linux.
                "rocm_agent_enumerator=rocm_sdk_core._cli:rocm_agent_enumerator",
                "rocminfo=rocm_sdk_core._cli:rocm_info",
                "rocm-smi=rocm_sdk_core._cli:rocm_smi",
            ]
            if platform.system() != "Windows"
            else [
                "hipInfo=rocm_sdk_core._cli:hipInfo",
            ]
        ),
    },
)
