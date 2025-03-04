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
print(f"extras_require={EXTRAS_REQUIRE}")

setup(
    name="rocm-sdk",
    version=dist_info.__version__,
    package_dir={"": "src"},
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
