"""Main rocm-sdk meta package.

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
from setuptools import setup
import sys
import os
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
    f"Loaded rocm-sdk dist_info: version={dist_info.__version__}, "
    f"suffix_nonce='{dist_info.PY_PACKAGE_SUFFIX_NONCE}', "
    f"default_target_family='{dist_info.DEFAULT_TARGET_FAMILY}', "
    f"available_target_families={dist_info.AVAILABLE_TARGET_FAMILIES}, "
    f"packages={dist_info.ALL_PACKAGES}"
)


# Resolve the build target family. This consults a list of things in increasing
# order of specificity:
#   1. "ROCM_SDK_TARGET_FAMILY" environment variable
#   2. Dynamically discovered/most salient target family on the actual system
#   3. dist_info.DEFAULT_TARGET_FAMILY
def discover_current_target_family() -> str | None:
    # TODO: Implement dynamic discovery.
    return None


def determine_target_family() -> str:
    target_family = os.getenv("ROCM_SDK_TARGET_FAMILY")
    if target_family is None:
        target_family = discover_current_target_family()
        if target_family is None:
            target_family = dist_info.DEFAULT_TARGET_FAMILY
    assert target_family is not None
    if target_family not in dist_info.AVAILABLE_TARGET_FAMILIES:
        raise ValueError(
            f"Requested ROCM_SDK_TARGET_FAMILY={target_family} is "
            f"not available in the distribution (available: "
            f"{', '.join(dist_info.AVAILABLE_TARGET_FAMILIES)})"
        )
    print(f"Determined target family: '{target_family}'")
    return target_family


TARGET_FAMILY = determine_target_family()
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
print(f"extras_require={EXTRAS_REQUIRE}")

setup(
    name="rocm-sdk",
    version=dist_info.__version__,
    package_dir={"": "src"},
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
