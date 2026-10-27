# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cache epoch, artifact identities and derived keys for reuse estimates.

The relational production cache uses manifest.ArtifactKey(full
component identity, raw CMake fingerprint, epoch=CACHE_EPOCH). The epoch is
persisted separately in artifact_keys and participates in key equality.
Database lookup and publication require ArtifactKey objects. Derived strings
identify reuse estimates and do not establish verified artifact availability.
"""


from dataclasses import dataclass
import hashlib
from typing import Iterable

# Bump deliberately when old productions must not satisfy new requests.
# Newly constructed ArtifactKey objects use this epoch; stored epochs stay fixed.
# Raw .fprint values and payload digests remain independent of it.
CACHE_EPOCH = 1


def validate_cache_epoch(epoch: int) -> int:
    """Require a positive integer representable by SQLite, excluding bool."""
    if type(epoch) is not int or not 1 <= epoch <= 2**63 - 1:
        raise ValueError("epoch must be a positive SQLite integer")
    return epoch


def current_cache_epoch() -> int:
    """Resolve the default when a new key/operation is created."""
    return validate_cache_epoch(CACHE_EPOCH)

# Marks commit-derived fallback fingerprints used in reuse estimates.
# The prefix distinguishes these values from raw CMake fingerprints.
_FALLBACK_FINGERPRINT_PREFIX = "srcfp1"


@dataclass(frozen=True)
class ArtifactIdentity:
    """Identifies *which output* a cache entry is for (not how it was built).

    Separating output identity from the build-input key is important for split
    / per-architecture artifacts: ``cmake/therock_artifacts.cmake`` omits the
    ``.fprint`` for split artifacts and assigns one aggregate fingerprint to the
    components of an unsplit artifact, so the cache needs an explicit output
    identity in addition to the input key.
    """

    name: str  # artifact name, e.g. "blas"
    component: str  # e.g. "lib", "dev", "run", "dbg", "doc", "test"
    target_family: str  # e.g. "gfx94X-dcgpu" or "generic"

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.name, self.component, self.target_family)

    def __str__(self) -> str:
        return f"{self.name}/{self.component}/{self.target_family}"


@dataclass(frozen=True)
class CacheKey:
    """A versioned, variant-scoped content key for one artifact unit."""

    fingerprint: str  # source/build-input fingerprint (real or fallback)
    build_variant: str  # "release", "asan", "debug", ...
    epoch: int = CACHE_EPOCH

    @property
    def value(self) -> str:
        """Return the epoch, build variant and fingerprint as a string."""
        return f"e{self.epoch}:{self.build_variant}:{self.fingerprint}"

    @property
    def short(self) -> str:
        """Human-friendly abbreviation for logs and reports."""
        return f"{self.build_variant}:{self.fingerprint[:12]}"

    @property
    def is_fallback_fingerprint(self) -> bool:
        return self.fingerprint.startswith(_FALLBACK_FINGERPRINT_PREFIX + ":")

    def __str__(self) -> str:
        return self.value


def _sha256_over(parts: Iterable[str]) -> str:
    """Stable SHA-256 over an ordered list of strings.

    Each part is length-prefixed so that ``["a", "bc"]`` and ``["ab", "c"]``
    cannot hash to the same value.
    """
    h = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8")
        h.update(str(len(encoded)).encode("ascii"))
        h.update(b":")
        h.update(encoded)
        h.update(b"\n")
    return h.hexdigest()


def compute_cache_key(
    *,
    source_fingerprint: str,
    build_variant: str,
    cmake_files_hash: str | None = None,
    extra_inputs: Iterable[str] | None = None,
    epoch: int = CACHE_EPOCH,
) -> CacheKey:
    """Compose a derived key for reuse estimates.

    This helper combines source evidence and optional recipe inputs. Database
    lookup and publication use ArtifactKey with the raw CMake fingerprint and
    exact component membership; they do not use this derived string.

    Args:
        source_fingerprint: Raw CMake fingerprint or a commit-derived value
            from derive_fallback_fingerprint, used as input to the estimate.
        build_variant: The build variant, which materially changes the bytes.
        cmake_files_hash: Optional hash of CMake files for the estimate.
        extra_inputs: Optional additional recipe inputs (e.g. docker image id,
            toolchain file hash) to include in the estimate.
        epoch: Global cache epoch; defaults to ``CACHE_EPOCH``.
    """
    parts = [f"fp={source_fingerprint}", f"variant={build_variant}"]
    if cmake_files_hash:
        parts.append(f"cmake={cmake_files_hash}")
    for item in extra_inputs or ():
        parts.append(f"extra={item}")
    fingerprint = _sha256_over(parts)
    # Preserve the fallback marker through hashing so derived keys remain
    # distinguishable from estimates based on raw CMake fingerprints.
    if source_fingerprint.startswith(_FALLBACK_FINGERPRINT_PREFIX + ":"):
        fingerprint = f"{_FALLBACK_FINGERPRINT_PREFIX}:{fingerprint}"
    return CacheKey(fingerprint=fingerprint, build_variant=build_variant, epoch=epoch)


def derive_fallback_fingerprint(
    *,
    source_commit: str,
    artifact: ArtifactIdentity,
    dep_fingerprints: Iterable[str] = (),
    build_variant: str = "release",
) -> str:
    """Derive a *fallback* source fingerprint from coarse signals.

    PLACEHOLDER -- NOT the production key. Real artifact fingerprints
    (``THEROCK_FPRINT``) hash source-file contents, dependency fingerprints,
    toolchain files and build-flag state, and are invalid on a dirty tree. We
    cannot recompute those from historical run metadata (it would require a
    submodule checkout + cmake configure), so this derives a key from the
    coarsest signals that *are* available in run history:

    * the source commit the run was built from (stands in for source identity),
    * the artifact output identity,
    * any dependency fingerprints we happen to know (usually empty for history),
    * the build variant.

    This is deliberately coarser than the real key -- it will over-invalidate
    (a key changes whenever the whole-repo commit changes, even for an unchanged
    subproject). That understates achievable reuse, which is the safe direction
    for a feasibility estimate. The returned value is prefixed so it can never
    collide with a real-fingerprint key.
    """
    body = _sha256_over(
        [
            f"commit={source_commit}",
            f"artifact={artifact}",
            *(f"dep={d}" for d in sorted(dep_fingerprints)),
            f"variant={build_variant}",
        ]
    )
    return f"{_FALLBACK_FINGERPRINT_PREFIX}:{body}"
