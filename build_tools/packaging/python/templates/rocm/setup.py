# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Main rocm meta package.

This package is a bit unique because we only distribute it as an sdist: it is
intended to be built implicitly on a target machine, where the environment can
be inspected to dynamically determine its deps.

There are also a number of magic environment variables to be used in "full"
installs, docker building, etc to force selection of a certain set of GPU
families for inclusion.

Note that this file is executed for building both sdists and bdists and needs
to be sensical for both.
"""

import importlib.util
from setuptools import setup, find_packages
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent


# The built package contains a pre-generated _dist_info.py file, which would
# normally be accessible at runtime. However, to make it available at
# package build time (here!), we have to dynamically import it.
def import_dist_info():
    dist_info_path = THIS_DIR / "src" / "rocm_sdk" / "_dist_info.py"
    if not dist_info_path.exists():
        raise RuntimeError(f"No _dist_info.py file found: {dist_info_path}")
    module_name = "rocm_sdk_dist_info"
    spec = importlib.util.spec_from_file_location(module_name, dist_info_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


dist_info = import_dist_info()
print(
    f"Loaded rocm dist_info: version={dist_info.__version__}, "
    f"suffix_nonce='{dist_info.PY_PACKAGE_SUFFIX_NONCE}', "
    f"default_target_family='{dist_info.DEFAULT_TARGET_FAMILY}', "
    f"available_target_families={dist_info.AVAILABLE_TARGET_FAMILIES}, "
    f"packages={dist_info.ALL_PACKAGES}"
)


TARGET_FAMILY = dist_info.determine_target_family()
INSTALL_REQUIRES = [
    pkg.get_dist_package_require(target_family=TARGET_FAMILY)
    for pkg in dist_info.ALL_PACKAGES.values()
    if pkg.required
]
print(f"install_requires={INSTALL_REQUIRES}")
EXTRAS_REQUIRE = {
    pkg.logical_name: [pkg.get_dist_package_require(target_family=TARGET_FAMILY)]
    for pkg in dist_info.ALL_PACKAGES.values()
    if not pkg.required
}
# Per-target extras for target-specific packages with multiple available
# targets (e.g. device wheels in kpack-split mode): explicit pip install
# rocm[device-gfx942] plus a device-all aggregate. Cross-platform multi-arch
# builds attach PEP 508 sys_platform markers to platform-exclusive targets
# so `pip install rocm[device-all]` only pulls device wheels published for
# the user's OS.
EXTRAS_REQUIRE.update(dist_info.build_per_target_extras())

# Drop the generic 'device' extra when target resolution would silently fall
# back to DEFAULT_TARGET_FAMILY - e.g. kpack-split CI installs on GPU-less
# runners where offload-arch is not yet available. Callers can still name a
# specific ISA via the per-target extras emitted above. Keep the generic
# 'device' extra when target resolution succeeds (normal install on a GPU
# machine).
device_entry = dist_info.ALL_PACKAGES.get("device")
if device_entry and device_entry.is_target_specific:
    try:
        dist_info.determine_target_family()
    except Exception:
        EXTRAS_REQUIRE.pop("device", None)

print(f"extras_require={EXTRAS_REQUIRE}")
packages = find_packages(where="./src")
print("Found packages:", packages)

setup(
    name="rocm",
    version=dist_info.__version__,
    package_dir={"": "src"},
    packages=packages,
    entry_points={
        "console_scripts": [
            "rocm-sdk = rocm_sdk.__main__:main",
        ],
    },
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
