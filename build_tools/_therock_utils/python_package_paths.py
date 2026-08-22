# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Path/key computation for structured product-local Python package publishing.

The stream-subdomain repository layout places Python artifacts under
product-local package directories:

    <product>/<index>/<normalized-package>/<filename>

where <index> is ``whl`` or ``whl-next``. Release stream (dev/nightly/
prerelease) is selected by the target bucket, never encoded in the path. This
module computes the per-file destination keys the release publishers use when
run with ``--structured``; the generator in ``manage_structured.py`` later
discovers and indexes those directories.

pep503_normalize + package-name extraction here intentionally mirror
manage_structured.py so producer output round-trips through its
discover_packages(). They are duplicated (a few lines) rather than shared
because manage_structured.py runs server-side and is deployed on its own; a
shared import would drag this module into that deployment.
"""

import dataclasses
from pathlib import Path
from re import sub

from _therock_utils.storage_location import StorageLocation
from packaging.utils import (
    InvalidSdistFilename,
    InvalidWheelFilename,
    parse_sdist_filename,
    parse_wheel_filename,
)


# Distribution artifacts placed into package directories.
ACCEPTED_FILE_EXTENSIONS = (".whl", ".tar.gz", ".zip")

# Valid aggregate index names (the second path segment).
INDEX_NAMES = ("whl", "whl-next")
DEFAULT_INDEX = "whl-next"

# repo.amd.com release streams and the per-product bucket naming scheme:
# therock-repo-amd-<stream>-<product>. Centralized here (rather than
# duplicated in download_python_packages.py and upload_release_packages.py)
# so a bucket-name typo like "pytorch" -> "python" can't be introduced in one
# copy and missed in the other.
REPO_STREAMS = ("dev", "nightly", "rc")
REPO_BUCKET_PRODUCT_NAMES = {
    "core": "core",
    "pytorch": "pytorch",
    "jax": "jax",
}

# ROCm Core tarball prefixes under the structured layout, keyed by variant.
CORE_TARBALL_PREFIXES = {
    "release": "v5/rocm/core/tarball/",
    "asan": "v5/rocm/core/tarball-asan/",
}


def repo_product_bucket(stream: str, product: str) -> str:
    """Return the repo.amd.com bucket name for a product on a given stream."""
    return f"therock-repo-amd-{stream}-{REPO_BUCKET_PRODUCT_NAMES[product]}"


def core_tarball_prefix(tarball_variant: str) -> str:
    """Return the structured S3 prefix for a ROCm Core tarball variant."""
    return CORE_TARBALL_PREFIXES[tarball_variant]


def core_tarball_dir_name(tarball_variant: str) -> str:
    """Return the local directory name used for a ROCm Core tarball variant."""
    return "tarball-asan" if tarball_variant == "asan" else "tarball"


def pep503_normalize(name: str) -> str:
    """Normalize a package name per PEP 503.

    Lowercase and collapse runs of "-", "_", and "." to a single "-".
    """
    return sub(r"[-_.]+", "-", name.lower())


def package_name_from_filename(filename: str) -> str:
    """Return the PEP 503-normalized package name for a distribution artifact.

    Uses spec-aware parsing (packaging.utils) so that sdist/zip filenames
    with hyphens in the project name are handled correctly (e.g.
    ``llnl-hatchet-2024.1.tar.gz`` -> ``llnl-hatchet``).

    Raises:
        ValueError: if the filename cannot be parsed as a wheel or sdist.
    """
    if filename.endswith(".whl"):
        try:
            name, _, _, _ = parse_wheel_filename(filename)
            return name
        except InvalidWheelFilename as e:
            raise ValueError(f"Cannot parse wheel filename {filename!r}: {e}") from e
    try:
        name, _ = parse_sdist_filename(filename)
        return name
    except InvalidSdistFilename as e:
        raise ValueError(f"Cannot parse sdist filename {filename!r}: {e}") from e


def is_accepted_artifact(filename: str) -> bool:
    return filename.endswith(ACCEPTED_FILE_EXTENSIONS)


def structured_key(product: str, index: str, filename: str) -> str:
    """Compute the structured S3 key for an artifact.

    Returns ``v5/rocm/<product>/<index>/<normalized-package>/<filename>``.
    ``v5`` is the layout schema version and the S3 origin prefix; ``rocm`` is
    the served tree segment, so the object serves at
    ``<stream>.repo.amd.com/rocm/<product>/<index>/...`` with no rewrite.

    Raises:
        ValueError: if ``index`` is not a valid aggregate index name.
    """
    if index not in INDEX_NAMES:
        raise ValueError(f"index={index!r} is invalid, must be one of {INDEX_NAMES}")
    package = package_name_from_filename(filename)
    return f"v5/rocm/{product}/{index}/{package}/{filename}"


@dataclasses.dataclass
class PlannedUpload:
    """A local artifact to upload into a structured package directory.

    Attributes:
        source: Local path to the artifact.
        dest: Destination location in the structured layout.
    """

    source: Path
    dest: StorageLocation


@dataclasses.dataclass
class PlannedCopy:
    """An S3 artifact to copy into a structured package directory.

    Attributes:
        source: Source location (an existing S3 object).
        dest: Destination location in the structured layout.
    """

    source: StorageLocation
    dest: StorageLocation


def plan_local_uploads(
    source_dir: Path,
    dest_bucket: str,
    product: str,
    index: str,
) -> list[PlannedUpload]:
    """Plan structured uploads for accepted artifacts in a local directory.

    Enumerates top-level artifacts in ``source_dir`` (not recursive: publish
    sources are flat directories of wheels/sdists) and computes their structured
    destinations. Sorted by filename for stable, reviewable output.
    """
    return [
        PlannedUpload(
            source=path,
            dest=StorageLocation(
                dest_bucket, structured_key(product, index, path.name)
            ),
        )
        for path in sorted(source_dir.iterdir())
        if path.is_file() and is_accepted_artifact(path.name)
    ]


def plan_key_copies(
    source_keys: list[str],
    source_bucket: str,
    dest_bucket: str,
    product: str,
    index: str,
) -> list[PlannedCopy]:
    """Plan structured copies for accepted artifacts listed from S3.

    Args:
        source_keys: Full S3 keys returned by a listing (in ``source_bucket``).
        source_bucket: Bucket the source keys live in.
        dest_bucket: Destination bucket name.
        product: Product segment (e.g. ``core``).
        index: Index segment (``whl`` or ``whl-next``).

    Returns:
        Planned copies for accepted artifacts, sorted by source key. Only the
        basename of each source key is used to place the artifact (the
        structured layout is flat within each package directory).
    """
    plans: list[PlannedCopy] = []
    for key in sorted(source_keys):
        filename = key.rsplit("/", 1)[-1]
        if not is_accepted_artifact(filename):
            continue
        plans.append(
            PlannedCopy(
                source=StorageLocation(source_bucket, key),
                dest=StorageLocation(
                    dest_bucket, structured_key(product, index, filename)
                ),
            )
        )
    return plans


# Product classification for the flat local promotion layout: the local
# directory (e.g. <output-dir>/wheels/, see how_to_do_release.md) where
# downloaded core, PyTorch, and JAX artifacts sit side-by-side, unsorted by
# product, before upload_release_packages.py --structured routes each one to
# its own product bucket. Kept as an explicit allowlist (rather than
# defaulting unmatched packages to "core") so an unrecognized or renamed
# package fails loudly instead of silently landing in the wrong product
# bucket.
#
# "rocm-bootstrap" is a real package intentionally not classified here:
# promote_packages.py already special-cases and skips it before this
# classification would ever run.
#
# PYTORCH_PACKAGE_NAMES is an exact-match set (apex/triton share no reliable
# prefix with "torch"); the *_PREFIXES tuples below are str.startswith()
# wildcard-style matches (e.g. "rocm_sdk_device_gfx1100" matches
# "rocm-sdk-device" once pep503-normalized).
PYTORCH_PACKAGE_NAMES = frozenset({"apex", "triton"})
PYTORCH_PACKAGE_PREFIXES = ("torch", "amd-torch")
JAX_PACKAGE_PREFIXES = ("jax", "jaxlib", "jax-rocm")
CORE_PACKAGE_PREFIXES = (
    "rocm-sdk-core",
    "rocm-sdk-devel",
    "rocm-sdk-device",
    "rocm-sdk-libraries",
    "rocm-profiler",
    "rocm",
)


def infer_structured_product(filename: str) -> str:
    """Infer the repo.amd.com product for a flat-layout distribution artifact.

    Args:
        filename: Wheel or sdist filename from the flat local promotion
            layout (core/pytorch/jax artifacts intermixed in one directory).

    Returns:
        One of "pytorch", "jax", "core".

    Raises:
        ValueError: if filename cannot be parsed as a wheel/sdist, or its
            package name doesn't match any known core/pytorch/jax package.
    """
    package_name = package_name_from_filename(filename)
    if package_name in PYTORCH_PACKAGE_NAMES or package_name.startswith(
        PYTORCH_PACKAGE_PREFIXES
    ):
        return "pytorch"
    if package_name.startswith(JAX_PACKAGE_PREFIXES):
        return "jax"
    if package_name.startswith(CORE_PACKAGE_PREFIXES):
        return "core"
    raise ValueError(
        f"Cannot infer structured product for package {package_name!r} "
        f"(file: {filename!r}); expected a known core/pytorch/jax package"
    )
