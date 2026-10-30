# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cache keys for fingerprint-based artifact reuse.

The central design contract (see issue #8651 and the design gist) is:

    equal keys imply safe substitution of build outputs

A cache key must therefore fold in *everything* that can change an artifact's
bytes in a way that matters to consumers:

* the source identity that produced it (ideally a reproducible fingerprint),
* the build variant (release / ASan / debug / ...),
* the compiler/toolchain and build-recipe inputs (cmake files, flags),
* a global ``CACHE_EPOCH`` so the whole cache can be invalidated at once if we
  discover a key input was missing.

``compute_cache_key`` is the single seam between "what the key is made of" and
"where the source fingerprint comes from". Today TheRock generates artifact
fingerprints at CMake configure time (``THEROCK_FPRINT`` /
``cmake/therock_subproject.cmake``) but does **not** upload them with
artifacts, so historical runs have no stored fingerprint to key on. Until that
until then, we derive a *fallback* fingerprint from signals that do
exist in run history (``derive_fallback_fingerprint``). When real fingerprints are
uploaded, only ``source_fingerprint`` changes -- every other layer of this
module stays the same.
"""


from dataclasses import dataclass
import hashlib
from typing import Iterable

# Bump to invalidate the entire cache (e.g. when a key input is added or a bug
# in key computation is fixed). Stored on every entry so stale-epoch entries are
# simply never matched.
CACHE_EPOCH = 1

# Marks keys whose source fingerprint is a fallback rather than a real
# uploaded artifact fingerprint. This keeps derived keys from ever colliding
# with (future) real-fingerprint keys, so the two can coexist during migration.
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
        """Canonical, stable string form used as the database primary key."""
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
    """Compose the final cache key from its inputs.

    THIS IS THE SEAM. ``source_fingerprint`` is the only thing that changes when
    real uploaded artifact fingerprints replace the fallback. Everything
    else -- epoch, build variant, cmake-file hash, and any extra recipe inputs
    -- is layered on here so the key stays correct regardless of where the
    source fingerprint came from.

    Args:
        source_fingerprint: A fingerprint of the source + dependency closure
            that produced the artifact. In production this is the uploaded
            ``.fprint``; in the fallback path it is ``derive_fallback_fingerprint(...)``.
        build_variant: The build variant, which materially changes the bytes.
        cmake_files_hash: Optional hash of the subproject's cmake files. The
            design notes call this out as a wanted key input beyond source
            hashes; included here when available.
        extra_inputs: Optional additional recipe inputs (e.g. docker image id,
            toolchain file hash) to fold in as they become available.
        epoch: Global cache epoch; defaults to ``CACHE_EPOCH``.
    """
    parts = [f"fp={source_fingerprint}", f"variant={build_variant}"]
    if cmake_files_hash:
        parts.append(f"cmake={cmake_files_hash}")
    for item in extra_inputs or ():
        parts.append(f"extra={item}")
    fingerprint = _sha256_over(parts)
    # Preserve the fallback marker through hashing so derived keys remain
    # distinguishable from (future) real-fingerprint keys and can never collide.
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
