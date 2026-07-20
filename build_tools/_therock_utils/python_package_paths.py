# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Path/key computation for structured product-local Python package publishing.

The stream-subdomain repository layout places Python artifacts under
product-local package directories:

    <product>/<index>/<normalized-package>/<filename>

where <index> is ``whl`` or ``whl-next``. Release stream (dev/nightly/
prerelease) is selected by the target bucket, never encoded in the path. This
module computes the per-file destination keys the release publishers use when
run with ``--structured``; the generator in
``third_party/s3_management/manage_structured.py`` later discovers and indexes
those directories.

pep503_normalize + package-name extraction here intentionally mirror
manage_structured.py so producer output round-trips through its
discover_packages(). They are duplicated (a few lines) rather than shared to
avoid a dependency from _therock_utils onto third_party/s3_management.
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

    Returns ``v5/<product>/<index>/<normalized-package>/<filename>``. The
    ``v5`` prefix is the layout schema version; CloudFront needs to redirect
    the served path to the current version.

    Raises:
        ValueError: if ``index`` is not a valid aggregate index name.
    """
    if index not in INDEX_NAMES:
        raise ValueError(f"index={index!r} is invalid, must be one of {INDEX_NAMES}")
    package = package_name_from_filename(filename)
    return f"v5/{product}/{index}/{package}/{filename}"


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
