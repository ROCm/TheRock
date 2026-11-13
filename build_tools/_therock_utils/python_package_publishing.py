# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Publishing policy for structured product-local Python package indexes.

Decides which aggregate index (the ``<index>`` segment of
``<product>/<index>/<normalized-package>/<filename>``) a build publishes to,
based on the index type and the CI build variant, and which build variants
support publishing at all. Shared by the release publishers; the resolved
index is passed to the path helpers in ``python_package_paths.py``.
"""

from github_actions.amdgpu_family_matrix import all_build_variants

# Aggregate index types:
# whl: flat pip-installable index (explicit dependency on device-all)
# whl-next: index for gfx-specific builds (explicit device selection)
INDEX_TYPES = ("whl", "whl-next")
DEFAULT_INDEX_TYPE = "whl-next"

# Shared indexes for instrumented builds, as ``whl-next-<suffix>``. There is
# no index per build variant: all ASAN variants (asan, asan-debug, host-asan,
# host-asan-debug) share ``whl-next-asan`` and are told apart by their local
# version (``+asan``, ``+asandebug``, ``+hostasan``, see
# compute_rocm_package_version.py). Only whl-next has these: whl pulls
# device-all by default, which is not wanted for these builds. Add a suffix
# (e.g. tsan, lsan) only once its index is supported; see
# build_variant_index_suffix() for the other changes needed.
INDEX_BUILD_VARIANT_SUFFIXES = ("asan",)


def get_python_index(
    index_type: str = DEFAULT_INDEX_TYPE, build_variant_suffix: str | None = None
) -> str:
    """Return the aggregate index name for an index type and build variant.

    ``get_python_index("whl-next")`` -> ``whl-next``;
    ``get_python_index("whl-next", "asan")`` -> ``whl-next-asan``.

    Raises:
        ValueError: if ``index_type`` or ``build_variant_suffix`` is unknown,
            or the combination is not published (e.g. ``whl`` with ``asan``).
    """
    if index_type not in INDEX_TYPES:
        raise ValueError(
            f"index_type={index_type!r} is invalid, must be one of {INDEX_TYPES}"
        )
    if build_variant_suffix is None:
        return index_type
    if build_variant_suffix not in INDEX_BUILD_VARIANT_SUFFIXES:
        raise ValueError(
            f"build_variant_suffix={build_variant_suffix!r} is invalid, must be "
            f"one of {INDEX_BUILD_VARIANT_SUFFIXES}"
        )
    if index_type != "whl-next":
        raise ValueError(
            f"build_variant_suffix={build_variant_suffix!r} is only published to "
            f"whl-next, not index_type={index_type!r}"
        )
    return f"{index_type}-{build_variant_suffix}"


# Build variants across all platforms, taken from all_build_variants in
# amdgpu_family_matrix.py so new variants are picked up without changes here.
BUILD_VARIANTS = tuple(
    dict.fromkeys(
        variant
        for platform_variants in all_build_variants.values()
        for variant in platform_variants
    )
)


def build_variant_index_suffix(build_variant: str) -> str | None:
    """Return the index suffix for a build variant, for ``get_python_index``.

    Host ASAN and ``-debug`` (RelWithDebInfo) variants share the ASAN index,
    matching tarball-asan: ``build_variant_index_suffix("asan-debug")`` ->
    ``asan``; ``build_variant_index_suffix("release")`` -> ``None``.

    Raises:
        ValueError: if ``build_variant`` is unknown or not supported for
            publishing yet (e.g. ``tsan``).
    """
    if build_variant not in BUILD_VARIANTS:
        raise ValueError(
            f"build_variant={build_variant!r} is invalid, must be one of "
            f"{BUILD_VARIANTS}"
        )
    if "asan" in build_variant:
        return "asan"
    if build_variant == "release":
        return None
    # Other build variants (e.g. tsan, lsan) are not published yet. To add one:
    # map it to an index suffix here (sharing "asan" or adding its own to
    # INDEX_BUILD_VARIANT_SUFFIXES), and give it tarball/native package paths
    # in publish_rocm_to_release_buckets.py, which only knows ASAN and release.
    raise ValueError(
        f"build_variant={build_variant!r} is not supported for publishing yet"
    )
