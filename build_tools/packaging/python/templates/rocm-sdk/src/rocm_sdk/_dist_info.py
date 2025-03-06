"""Distribution information.

This file is typically auto-generated by the build system, but for the purposes
of bootstrapping, we are including it inline for the moment.
"""

import os
import platform


def os_arch() -> str:
    """Gets the `os_arch` placeholder for the current system."""
    return f"{platform.system().lower()}-{platform.machine()}"


class PackageEntry:
    def __init__(
        self, logical_name: str, package_template: str, *, required: bool = False
    ):
        self.logical_name = logical_name
        self.package_template = package_template
        self.required = required
        if logical_name in ALL_PACKAGES:
            raise ValueError(f"Package already defined: {logical_name}")
        ALL_PACKAGES[logical_name] = self

    @property
    def is_target_specific(self) -> bool:
        return "{target_family}" in self.package_template

    @property
    def is_host_specific(self) -> bool:
        return "{os_arch}" in self.package_template

    def get_dist_package_name(self, target_family: str | None = None) -> str:
        if self.is_target_specific and target_family is None:
            raise ValueError(
                f"Package {self.logical_name} is target specific, but no target specified"
            )
        kwargs = {"os_arch": os_arch()}
        if target_family is not None:
            kwargs["target_family"] = target_family
        return self.package_template.format(**kwargs)

    def get_dist_package_require(self, target_family: str | None = None) -> str:
        return self.get_dist_package_name(target_family) + f"=={__version__}"

    def get_py_package_name(self, target_family: str | None = None) -> str:
        dist_name = self.get_dist_package_name(target_family)
        return "_" + dist_name.replace("-", "_") + PY_PACKAGE_SUFFIX_NONCE

    def __repr__(self):
        return self.package_template


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
            target_family = DEFAULT_TARGET_FAMILY
    assert target_family is not None
    if target_family not in AVAILABLE_TARGET_FAMILIES:
        raise ValueError(
            f"Requested ROCM_SDK_TARGET_FAMILY={target_family} is "
            f"not available in the distribution (available: "
            f"{', '.join(AVAILABLE_TARGET_FAMILIES)})"
        )
    print(f"Determined target family: '{target_family}'")
    return target_family


# All packages that are part of the distribution.
ALL_PACKAGES: dict[str, PackageEntry] = {}

# Always available packages.
PackageEntry("core", "rocm-sdk-core-{os_arch}", required=True)
PackageEntry(
    "libraries", "rocm-sdk-libraries-{target_family}-{os_arch}", required=False
)
PackageEntry("devel", "rocm-sdk-devel-{os_arch}", required=False)

# Overall ROCM package version.
__version__ = "DEFAULT"

# Nonce added to the backend packages which encodes the version. This is
# typically empty for development distributions. Only backend packages with
# a matching nonce will be considered for use by this meta package.
PY_PACKAGE_SUFFIX_NONCE: str = "_DEFAULT"

# If a target family cannot be found or is not relevant (i.e. building devel
# packages on a gpu-less system), this is the default target family.
DEFAULT_TARGET_FAMILY: str = "DEFAULT"

# All available target families that this distribution has available.
AVAILABLE_TARGET_FAMILIES: list[str] = []
