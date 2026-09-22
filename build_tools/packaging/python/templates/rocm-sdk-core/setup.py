# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Main rocm-sdk-core (OS specific)."""

import importlib.util
import os
import platform
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as _build_py
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

# The amdsmi Python module travels as ordinary files inside the platform
# payload, at <py_package>/share/amd_smi/amdsmi. Nothing in a wheel install puts
# that directory on sys.path, so `import amdsmi` fails even though the files are
# present. Ship a .pth naming it.
#
# The entry is relative, which site.py resolves against the site-packages
# directory holding the .pth, so the wheel stays relocatable. The module is
# referenced where it lands rather than copied: amdsmi_wrapper.py locates
# libamd_smi.so by a path relative to its own file, so a copy elsewhere would
# bind whichever library the dynamic linker found first.
AMDSMI_PTH_NAME = "amdsmi.pth"
AMDSMI_SHARE_RELPATH = f"{platform_package_name}/share/amd_smi"


def _amdsmi_module_present() -> bool:
    staged = (
        THIS_DIR / "platform" / platform_package_name / "share" / "amd_smi" / "amdsmi"
    )
    return (staged / "__init__.py").is_file()


class build_py(_build_py):
    """Emit amdsmi.pth at the wheel root so the staged module is importable.

    A file copied into build_lib's root becomes a top-level entry in the wheel,
    which pip installs directly into site-packages and records in RECORD, so
    `pip uninstall` removes it again.
    """

    def run(self):
        super().run()
        if not _amdsmi_module_present():
            print("amdsmi module not present in payload; skipping amdsmi.pth")
            return
        os.makedirs(self.build_lib, exist_ok=True)
        target = Path(self.build_lib) / AMDSMI_PTH_NAME
        target.write_text(AMDSMI_SHARE_RELPATH + "\n")
        print(f"Wrote {target} -> {AMDSMI_SHARE_RELPATH}")


WINDOWS_CONSOLE_SCRIPTS = [
    "hipInfo=rocm_sdk_core._cli:hipInfo",
]
if (
    platform.system() == "Windows"
    and (
        THIS_DIR / "platform" / platform_package_name / "bin" / "rocminfo.exe"
    ).exists()
):
    WINDOWS_CONSOLE_SCRIPTS.append("rocminfo=rocm_sdk_core._cli:rocm_info")

setup(
    name=f"rocm-sdk-core",
    version=dist_info.__version__,
    cmdclass={"build_py": build_py},
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
            "amdflang=rocm_sdk_core._cli:amdflang",
            "amdlld=rocm_sdk_core._cli:amdlld",
            "hipcc=rocm_sdk_core._cli:hipcc",
            "hipconfig=rocm_sdk_core._cli:hipconfig",
            "hipify-clang=rocm_sdk_core._cli:hipify_clang",
            "offload-arch=rocm_sdk_core._cli:offload_arch",
            "roc-obj=rocm_sdk_core._cli:roc_obj",
            "roc-obj-extract=rocm_sdk_core._cli:roc_obj_extract",
            "roc-obj-ls=rocm_sdk_core._cli:roc_obj_ls",
        ]
        + (
            [
                # These tools are only available on Linux.
                "amd-smi=rocm_sdk_core._cli:amd_smi",
                "hipify-perl=rocm_sdk_core._cli:hipify_perl",
                "rocm_agent_enumerator=rocm_sdk_core._cli:rocm_agent_enumerator",
                "rocminfo=rocm_sdk_core._cli:rocm_info",
                "roccoremerge=rocm_sdk_core._cli:roccoremerge",
                "rocgdb=rocm_sdk_core._cli:rocgdb",
                "rocpd=rocm_sdk_core._cli:rocpd",
                "rocpd2csv=rocm_sdk_core._cli:rocpd2csv",
                "rocpd2otf2=rocm_sdk_core._cli:rocpd2otf2",
                "rocpd2pftrace=rocm_sdk_core._cli:rocpd2pftrace",
                "rocpd2summary=rocm_sdk_core._cli:rocpd2summary",
                "rocprofv3=rocm_sdk_core._cli:rocprofv3",
                "rocprofv3-attach=rocm_sdk_core._cli:rocprofv3_attach",
                "rocprofv3-avail=rocm_sdk_core._cli:rocprofv3_avail",
            ]
            if platform.system() != "Windows"
            else WINDOWS_CONSOLE_SCRIPTS
        ),
    },
)
