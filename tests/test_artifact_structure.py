# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Structural validation of artifact archives.

These tests scan artifact archives WITHOUT extracting them and check invariants
that the build system should maintain. They run on CPU (no GPU required) and
are designed to catch issues before artifacts are installed or tested on GPU
runners.

Tests:
  - Cross-artifact overlaps: no two artifacts should produce files that
    flatten to the same path (causes silent overwrites or race conditions,
    see https://github.com/ROCm/TheRock/issues/3758).
  - Within-artifact component overlaps: different components (lib, run,
    test, etc.) of the same artifact should contain disjoint files (the
    component scanner should enforce this via the extends chain).
  - Manifest validation: every archive should have artifact_manifest.txt
    as its first member.
  - Package coverage: every artifact/component present in the fetched
    archives should be covered by a Linux package definition in
    build_tools/packaging/linux/package.json, checked per Artifact_Subdir
    so one subdir's gap cannot hide behind its siblings (see #9049).

Usage:
    THEROCK_ARTIFACTS_DIR=/path/to/archives \\
        python -m pytest tests/test_artifact_structure.py -v --log-cli-level=info

THEROCK_ARTIFACTS_DIR should point to a directory containing artifact archives
(*.tar.zst or *.tar.xz) as produced by the build pipeline.
"""

import dataclasses
import functools
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath

import pytest

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "build_tools"))

from _therock_utils.artifacts import ArtifactName
from _therock_utils.archive_util import open_archive_for_read

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = os.getenv("THEROCK_ARTIFACTS_DIR", "")

# Resolve package.json relative to this source file (repo root), NOT the current
# working directory. tests/test_artifact_structure.py lives one directory below
# the repository root.
REPO_ROOT = THIS_DIR.parent
PACKAGE_JSON_PATH = REPO_ROOT / "build_tools" / "packaging" / "linux" / "package.json"

# Components intentionally excluded from packaging coverage checks.
IGNORED_COMPONENTS = {"dbg"}

# Entries known to be intentionally uncovered by package.json today. Two
# arities are accepted:
#   (artifact, component)          -> the artifact/component is not
#       distributed at all (new/experimental artifact, test-only project).
#   (artifact, subdir, component)  -> one Artifact_Subdir is uncovered while
#       its siblings are packaged. Prefer this form for multi-subdir
#       artifacts: a 2-tuple suppresses every subdir and can mask a future
#       origami-shaped gap (#9049).
# Entries should be temporary — remove once real package.json coverage is
# added. All values lowercase.

KNOWN_UNCOVERED_COMPONENTS: set[tuple[str, ...]] = {
    ("base", "rocprofiler-register", "test"),  # install-test binary, not shipped
    ("fftw3", "dev"),  # fftw3 is currently test-only; may be distributed later.
    ("fftw3", "doc"),
    ("fftw3", "run"),
    ("fftw3", "lib"),
    ("hipkernelprovider", "lib"),
    ("hipkernelprovider", "test"),
    ("hipthreads", "dev"),
    ("hipthreads", "test"),
    ("mirage", "dev"),  # new artifact, no packages yet.
    ("rocjitsu", "dev"),  # new artifact, no packages yet.
    ("rocprofiler-systems-examples", "test"),
    (
        "rocwmma",
        "test",
    ),  # rocWMMA ships via amdrocm-math-common (dev/lib); tests not distributed.
    ("rocrtst", "lib"),
    ("support", "dev"),
    ("support", "doc"),
    ("sysdeps-util-linux", "dev"),
    # --- #9049 first-run triage ---
    # Per-subproject LICENSE files under share/doc/<subproject>/; packages
    # carry their own top-level licence metadata.
    # TODO(#NNNN): confirm attribution policy with packaging.
    ("base", "half", "doc"),  # share/doc/half/LICENSE.txt
    ("base", "rocm-cmake", "doc"),  # share/doc/rocm-cmake/LICENSE
    ("blas", "hipblas-common", "doc"),  # share/doc/hipblas-common/LICENSE.md
    ("rccl", "rccl-tests", "doc"),  # share/doc/rccl-tests/LICENSE.txt
    # Packaging smoke-test binary, not shipped to users.
    ("profiler-hub", "profiler-hub-install-test", "test"),  # bin/test_profiler-hub
}

# Manifest subdir names that don't literally match their package.json
# Artifact_Subdir "Name". Comparison is already case-insensitive, so this is
# only for genuine naming drift. Keys and values lowercase.
SUBDIR_PACKAGE_ALIASES: dict[str, str] = {}

# "strict" (default): subdir-level gaps fail.
# "warn": subdir-level gaps log only; artifact-level gaps still fail.
# One-release escape hatch while the first-run backlog is triaged.
VALID_COVERAGE_MODES = ("strict", "warn")

# Trailing prefix segments that are build-layout noise, not a subproject name.
_PREFIX_LEAF_IGNORE = {"stage", "dist", "build"}


def parse_coverage_mode(value: str | None) -> str:
    """Normalize and validate THEROCK_SUBDIR_COVERAGE.

    Fails loudly on typos: silently falling back to a default would make the
    env var untrustworthy in CI (a misspelled 'stict' would appear to work).
    """
    mode = (value or "strict").strip().lower()
    if mode not in VALID_COVERAGE_MODES:
        raise ValueError(
            f"THEROCK_SUBDIR_COVERAGE must be one of "
            f"{', '.join(VALID_COVERAGE_MODES)}; got {value!r}"
        )
    return mode


SUBDIR_COVERAGE_MODE = parse_coverage_mode(os.getenv("THEROCK_SUBDIR_COVERAGE"))


def subdir_from_prefix(prefix: str) -> str:
    """Derive the package.json Artifact_Subdir name from a manifest prefix.

    e.g. "math-libs/BLAS/origami/stage" -> "origami"
    """
    parts = list(PurePosixPath(prefix.strip().strip("/")).parts)
    while parts and parts[-1] in _PREFIX_LEAF_IGNORE:
        parts.pop()
    return parts[-1] if parts else ""


def is_windows_platform() -> bool:
    """Check if the artifacts being validated are Windows artifacts and return bool"""
    return os.getenv("PLATFORM", "linux").lower() == "windows"


@pytest.fixture(scope="session")
def artifacts_dir() -> Path:
    if not ARTIFACTS_DIR:
        pytest.skip("THEROCK_ARTIFACTS_DIR not set")
    path = Path(ARTIFACTS_DIR).resolve()
    if not path.is_dir():
        pytest.fail(f"THEROCK_ARTIFACTS_DIR is not a directory: {path}")
    return path


def list_archive_files(archive_path: Path) -> tuple[list[str], list[tuple[str, str]]]:
    """List the flattened file paths in an artifact archive.

    Reads artifact_manifest.txt (must be the first tar member) to get basedir
    prefixes, then strips those prefixes from each file member to produce
    the flattened output path. The prefix that matched is retained as the
    originating subdir, so coverage can be checked per subproject (#9049).

    Returns:
        (manifest_prefixes, [(subdir_name, flattened_path), ...])

    manifest_prefixes is returned in manifest order; matching internally uses
    longest-first order so nested prefixes win over their parents.
    """
    prefixes: list[str] = []
    entries: list[tuple[str, str]] = []

    with open_archive_for_read(archive_path) as tf:
        # Extract prefixes from the artifact manifest.
        manifest_member = tf.next()
        if manifest_member is None or manifest_member.name != "artifact_manifest.txt":
            raise ValueError(
                f"{archive_path.name}: expected artifact_manifest.txt as first member, "
                f"got {manifest_member.name if manifest_member else 'empty archive'}"
            )
        with tf.extractfile(manifest_member) as mf:
            # The artifact_manifest.txt will contain prefix lines like:
            #   math-libs/BLAS/hipBLAS-common/stage
            #   math-libs/BLAS/rocRoller/stage
            #   math-libs/BLAS/hipBLAS/stage
            prefixes = [
                line for line in mf.read().decode().splitlines() if line.strip()
            ]

        # Precompute prefix -> subdir once: few prefixes, many members.
        prefix_subdirs = {p: subdir_from_prefix(p) for p in prefixes}

        # Longest-first: if one prefix is a path-prefix of another (e.g.
        # "math-libs/BLAS/stage" vs "math-libs/BLAS/hipBLAS/stage"), the more
        # specific one must win.
        match_prefixes = sorted(prefixes, key=len, reverse=True)

        # Strip prefixes from each file member to produce flattened paths.
        # For example:
        #   prefix:     math-libs/BLAS/hipBLAS/stage
        #   member:     math-libs/BLAS/hipBLAS/stage/include/hipblas/hipblas.h
        #   flattened:  include/hipblas/hipblas.h
        while member := tf.next():
            if member.isdir():
                continue
            name = member.name
            for prefix in match_prefixes:
                prefix_slash = prefix + "/"
                if name.startswith(prefix_slash):
                    flattened_path = name[len(prefix_slash) :]
                    if flattened_path:
                        entries.append((prefix_subdirs[prefix], flattened_path))
                    break

    return prefixes, entries


def discover_archives(artifacts_dir: Path) -> list[Path]:
    """Find all artifact archives in a directory."""
    archives = []
    for ext in ("*.tar.zst", "*.tar.xz"):
        archives.extend(artifacts_dir.glob(ext))
    return sorted(archives)


@dataclasses.dataclass
class ArchiveInfo:
    """Metadata and flattened file listing for a single artifact archive."""

    artifact_name: str
    component: str
    filename: str
    # subdir name (from the manifest prefix) -> flattened paths contributed by
    # that subdir. Only subdirs that actually contribute files to THIS
    # component archive appear, so a subdir that legitimately ships nothing
    # for a component is never flagged.
    subdir_paths: dict[str, set[str]]

    # subdir name -> the manifest prefix line it was derived from.
    # packaging_utils.py matches Artifact_Subdir "Name" as a SUBSTRING of the
    # whole manifest line, not against the derived leaf, so the full line must
    # survive to replay that rule faithfully.
    subdir_prefixes: dict[str, str] = dataclasses.field(default_factory=dict)

    @functools.cached_property
    def flattened_paths(self) -> set[str]:
        """Union of all subdirs' paths (the pre-#9049 view)."""
        out: set[str] = set()
        for paths in self.subdir_paths.values():
            out |= paths
        return out

    @property
    def subdir_names(self) -> list[str]:
        return sorted(self.subdir_paths)


@dataclasses.dataclass
class CrossArtifactOverlap:
    """A flattened file path found in multiple artifacts."""

    path: str
    sources: list[tuple[str, str]]  # (artifact_name, archive_filename)


@dataclasses.dataclass
class ComponentOverlap:
    """Files duplicated across components within one artifact."""

    artifact_name: str
    overlaps: dict[str, list[str]]  # path -> [component_name, ...]


# ---------------------------------------------------------------------------
# Package coverage helpers
#
# These cross-reference the artifact archives that were fetched (their name and
# component, derived from the filename via ArtifactName.from_filename) against
# the Linux packaging definitions in build_tools/packaging/linux/package.json.
# ---------------------------------------------------------------------------
def load_package_data(package_json_path: Path = PACKAGE_JSON_PATH) -> list:
    """Load and parse the Linux packaging definitions (package.json)."""
    with open(package_json_path) as f:
        return json.load(f)


def _clean_components(components) -> list[str]:
    """Normalize a component spec to a list with ignored components removed."""
    if isinstance(components, str):
        components = [components]
    return [c for c in components if c not in IGNORED_COMPONENTS]


@dataclasses.dataclass
class PackageCoverage:
    """Components declared by package.json, keyed by subdir AND by artifact."""

    components: dict[str, set[str]]
    # Artifacts declaring Artifact_Subdir entries. For these the collapsed
    # top-level bucket is NOT a valid coverage source — that collapsing is
    # the #9049 bug. Retained only for diagnostics / flat entries.
    subdir_declared_artifacts: set[str]
    # artifact -> [(Name, {components})], in package.json declaration order.
    artifact_subdirs: dict[str, list[tuple[str, set[str]]]] = dataclasses.field(
        default_factory=dict
    )

    def components_for(self, key: str) -> set[str]:
        return self.components.get(key.lower(), set())

    def has_subdir_granularity(self, artifact_name: str) -> bool:
        return artifact_name.lower() in self.subdir_declared_artifacts

    def packager_match(
        self, artifact: str, prefix_line: str, component: str
    ) -> str | None:
        """Replicate packaging_utils.py: (Name + "/") in manifest_line.

        Deliberately mirrors the packager rather than being stricter, so a
        finding means "these files land in no package", not merely "the
        declaration looks unusual". The rule is loose in places --
        Name="sysdeps" legitimately covers "third-party/sysdeps/zlib/stage" --
        but tightening the packager is a separate change.
        """
        line = prefix_line.lower()
        for name, comps in self.artifact_subdirs.get(artifact.lower(), []):
            if component in comps and (name.lower() + "/") in line:
                return name
        return None


def build_packaged_components(package_data: list) -> PackageCoverage:
    """Map each packaged artifact name (lowercased) to the components it ships.

    Handles both the nested ``Artifact_Subdir`` structure and the older flat
    structure, mirroring build_tools/packaging/linux/package.json.

    Returns:
        PackageCoverage, with components keyed by BOTH subdir name and
        top-level artifact name. For artifacts that declare Artifact_Subdir
        entries, only the per-subdir keys are valid for coverage decisions —
        the collapsed artifact key is diagnostics-only (see #9049).

    Raises:
        ValueError: if an Artifact_Subdir entry has no usable "Name".
    """
    packaged: dict[str, set[str]] = defaultdict(set)
    subdir_declared: set[str] = set()
    artifact_subdirs: dict[str, list[tuple[str, set[str]]]] = defaultdict(list)

    all_packages = []
    for item in package_data:
        if "Artifactory" in item:
            all_packages.extend(item["Artifactory"])

    for pkg in all_packages:
        artifact = pkg.get("Artifact")

        if "Artifact_Subdir" in pkg:
            if artifact:
                subdir_declared.add(artifact.lower())
            # Nested structure: each subdir has its own Name + Components.
            for subdir in pkg["Artifact_Subdir"]:
                subdir_name = (subdir.get("Name") or "").lower()
                if not subdir_name:
                    # An empty key would become a catch-all that any archive
                    # with an undetectable subdir could match — reintroducing
                    # the #9049 "unknown means covered" hole.
                    raise ValueError(
                        f"package.json: Artifact_Subdir entry under "
                        f"Artifact={artifact!r} has no 'Name'; cannot resolve "
                        f"per-subdir package coverage"
                    )
                components = _clean_components(subdir.get("Components", ["run"]))
                if artifact:
                    artifact_subdirs[artifact.lower()].append(
                        (subdir["Name"], set(components))
                    )
                for component in components:
                    packaged[subdir_name].add(component)
                    # Collapsed bucket: diagnostics + flat artifacts only.
                    # NOT used to satisfy coverage for subdir-declared
                    # artifacts (see #9049).
                    if artifact:
                        packaged[artifact.lower()].add(component)
        else:
            # Old flat structure without Artifact_Subdir.
            pkg_name = pkg.get("Name", artifact)
            key = (artifact or pkg_name or "").lower()
            for component in _clean_components(pkg.get("Components", ["run"])):
                packaged[key].add(component)

    return PackageCoverage(
        components=dict(packaged),
        subdir_declared_artifacts=subdir_declared,
        artifact_subdirs=dict(artifact_subdirs),
    )


@dataclasses.dataclass(frozen=True)
class UncoveredComponent:
    """One artifact/subdir/component present in archives but not packaged."""

    artifact_name: str
    subdir: str
    component: str
    filename: str
    # "artifact": missing from the artifact entirely (old check caught these).
    # "subdir":   only this subdir lacks it; siblings provide it. New.
    level: str
    reason: str

    @property
    def sort_key(self) -> tuple[str, str, str]:
        return (self.artifact_name, self.subdir, self.component)

    def describe(self) -> str:
        return (
            f"{self.artifact_name}/{self.subdir} [{self.component}] "
            f"({self.filename}): {self.reason}"
        )


def _is_known_uncovered(artifact: str, subdir: str | None, component: str) -> bool:
    if (artifact, component) in KNOWN_UNCOVERED_COMPONENTS:
        return True
    if subdir is not None:
        return (artifact, subdir, component) in KNOWN_UNCOVERED_COMPONENTS
    return False


def find_uncovered_components(
    archive_index: list["ArchiveInfo"], coverage: PackageCoverage
) -> list[UncoveredComponent]:
    """Return artifact/subdir/components not covered by any package.

    For artifacts declared with Artifact_Subdir entries, coverage resolves per
    subdir. The collapsed top-level bucket is deliberately NOT a fallback:
    consulting it is precisely what let origami/dev ship uncovered (#9049)
    behind hipBLAS/rocBLAS/hipBLASLt's dev entries.
    """
    uncovered: list[UncoveredComponent] = []
    for info in archive_index:
        if info.component in IGNORED_COMPONENTS:
            continue
        artifact_key = info.artifact_name.lower()
        if _is_known_uncovered(artifact_key, None, info.component):
            continue

        artifact_covered = info.component in coverage.components_for(artifact_key)

        if not coverage.has_subdir_granularity(artifact_key):
            # Flat package entry: no finer granularity exists to check.
            if not artifact_covered:
                uncovered.append(
                    UncoveredComponent(
                        info.artifact_name,
                        "*",
                        info.component,
                        info.filename,
                        "artifact",
                        "artifact/component has no package.json entry",
                    )
                )
            continue

        for subdir in info.subdir_names:
            raw_key = subdir.lower()
            if not raw_key:
                # The manifest prefix yielded no subproject name (e.g. a bare
                # "stage" prefix). Degrade to the artifact-level verdict rather
                # than resolving against an empty package.json key.
                if not artifact_covered:
                    uncovered.append(
                        UncoveredComponent(
                            info.artifact_name,
                            "*",
                            info.component,
                            info.filename,
                            "artifact",
                            "artifact/component has no package.json entry "
                            "(manifest prefix yielded no subdir name)",
                        )
                    )
                continue
            subdir_key = SUBDIR_PACKAGE_ALIASES.get(raw_key, raw_key)
            if _is_known_uncovered(artifact_key, subdir_key, info.component):
                continue

            # Authoritative check: would packaging_utils.py pick these files
            # up? Fall back to the subdir name when no prefix was recorded.
            prefix_line = info.subdir_prefixes.get(subdir, subdir)
            if coverage.packager_match(artifact_key, prefix_line, info.component):
                continue

            declared = coverage.components.get(subdir_key)
            if declared is None:
                reason = (
                    f"no Artifact_Subdir entry matches manifest prefix "
                    f"{prefix_line!r} for component {info.component!r} "
                    f"(add an entry, or an alias in SUBDIR_PACKAGE_ALIASES)"
                )
            else:
                reason = (
                    f"subdir {subdir!r} ships files for component "
                    f"{info.component!r} but does not list it in Components"
                )
            uncovered.append(
                UncoveredComponent(
                    info.artifact_name,
                    subdir,
                    info.component,
                    info.filename,
                    "artifact" if not artifact_covered else "subdir",
                    reason,
                )
            )
    return uncovered


@pytest.fixture(scope="session")
def archive_index(artifacts_dir: Path) -> list[ArchiveInfo]:
    """Scan all archives once and return a flat list of ArchiveInfo."""
    archives = discover_archives(artifacts_dir)
    if not archives:
        pytest.skip(f"No artifact archives found in {artifacts_dir}")

    index: list[ArchiveInfo] = []
    skipped: list[str] = []

    for archive_path in archives:
        archive_name = archive_path.name
        an = ArtifactName.from_filename(archive_name)
        if an is None:
            logger.warning("Skipping unrecognized archive: %s", archive_name)
            skipped.append(archive_name)
            continue

        logger.info("Listing %s ...", archive_name)
        try:
            prefixes, entries = list_archive_files(archive_path)
        except Exception:
            logger.exception("Failed to read %s", archive_name)
            skipped.append(archive_name)
            continue

        subdir_paths: dict[str, set[str]] = defaultdict(set)
        for subdir, fpath in entries:
            subdir_paths[subdir].add(fpath)

        # One manifest prefix per contributing subdir. If two prefixes reduce
        # to the same leaf, first-in-manifest wins -- deterministic, and either
        # line matches the same Artifact_Subdir entry anyway.
        subdir_prefixes: dict[str, str] = {}
        for prefix in prefixes:
            leaf = subdir_from_prefix(prefix)
            if leaf in subdir_paths:
                subdir_prefixes.setdefault(leaf, prefix)

        index.append(
            ArchiveInfo(
                artifact_name=an.name,
                component=an.component,
                filename=archive_name,
                subdir_paths=dict(subdir_paths),
                subdir_prefixes=subdir_prefixes,
            )
        )

    if skipped:
        logger.warning("Skipped %d archives: %s", len(skipped), skipped)

    artifact_names = {a.artifact_name for a in index}
    logger.info(
        "Indexed %d archives across %d artifacts", len(index), len(artifact_names)
    )
    return index


def _format_cross_artifact_overlaps(overlaps: list[CrossArtifactOverlap]) -> str:
    """Format cross-artifact overlaps into readable summary."""
    lines = []
    for overlap in sorted(overlaps, key=lambda o: o.path):
        lines.append(f"  {overlap.path}")
        for label, archive in sorted(overlap.sources):
            lines.append(f"    - {label} ({archive})")
    return "\n".join(lines)


def _format_component_overlaps(overlaps: list[ComponentOverlap]) -> str:
    """Format within-artifact component overlaps into readable summary."""
    lines = []
    total = 0
    for overlap in sorted(overlaps, key=lambda o: o.artifact_name):
        total += len(overlap.overlaps)
        lines.append(f"  {overlap.artifact_name} ({len(overlap.overlaps)} files):")
        for fpath in sorted(overlap.overlaps):
            comps = sorted(set(overlap.overlaps[fpath]))
            lines.append(f"    {fpath}  [{', '.join(comps)}]")
    return total, "\n".join(lines)


class TestArtifactStructure:
    """Structural validation of artifact archives."""

    def test_no_cross_artifact_overlaps(self, archive_index: list[ArchiveInfo]):
        """No two artifacts should contain files that flatten to the same path.

        This catches both same-basedir overlaps (like #3758) and the subtler
        cross-basedir case where different stage dirs install identically-named
        files (e.g., two subprojects both installing "bin/sequence.yaml").

        See https://github.com/ROCm/TheRock/issues/3796
        """
        # For each flattened path, track which artifacts contain it.
        # flattened_path -> { artifact_name: set of archive filenames }
        path_artifacts: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )

        for info in archive_index:
            for fpath in info.flattened_paths:
                path_artifacts[fpath][info.artifact_name].add(info.filename)

        # A cross-artifact overlap is a path claimed by more than one artifact
        # (e.g. "blas", "support").
        overlaps: list[CrossArtifactOverlap] = []
        for fpath, artifact_archives in path_artifacts.items():
            if len(artifact_archives) > 1:
                sources = []
                for artifact_name, archives in artifact_archives.items():
                    for archive in archives:
                        sources.append((artifact_name, archive))
                overlaps.append(CrossArtifactOverlap(path=fpath, sources=sources))

        if overlaps:
            summary = _format_cross_artifact_overlaps(overlaps)
            pytest.fail(
                f"Found {len(overlaps)} cross-artifact overlap(s) across "
                f"{len(archive_index)} archives "
                f"(see https://github.com/ROCm/TheRock/issues/3796):\n{summary}"
            )

        logger.info(
            "Checked %d unique paths across %d archives, no cross-artifact overlaps",
            len(path_artifacts),
            len(archive_index),
        )

    def test_no_within_artifact_component_overlaps(
        self, archive_index: list[ArchiveInfo]
    ):
        """Components of the same artifact should contain disjoint files.

        The component scanner (artifact_builder.py) enforces disjointness via
        the extends chain (lib -> run -> dbg -> dev -> doc -> test). Overlaps
        can still occur if descriptors misconfigure include/exclude patterns
        (e.g., a bare 'run' entry acting as a catch-all that steals files
        from later components like 'test').

        See https://github.com/ROCm/TheRock/issues/3796
        """
        # Group archives by artifact name, merging target variants per component.
        # artifact_name -> component -> set of flattened paths
        by_artifact: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        for info in archive_index:
            by_artifact[info.artifact_name][info.component].update(info.flattened_paths)

        # A component overlap is a path claimed by more than one component
        # within a single artifact name (e.g. "blas_run", "blas_test").
        overlaps: list[ComponentOverlap] = []
        for artifact_name, component_files in by_artifact.items():
            component_names = sorted(component_files.keys())
            # fpath -> list of component names that contain it
            artifact_overlaps: dict[str, list[str]] = {}
            for i in range(len(component_names)):
                for j in range(i + 1, len(component_names)):
                    c1, c2 = component_names[i], component_names[j]
                    for fpath in component_files[c1] & component_files[c2]:
                        artifact_overlaps.setdefault(fpath, []).extend([c1, c2])

            if artifact_overlaps:
                overlaps.append(
                    ComponentOverlap(
                        artifact_name=artifact_name, overlaps=artifact_overlaps
                    )
                )

        if overlaps:
            total, summary = _format_component_overlaps(overlaps)
            pytest.fail(
                f"Found within-artifact component overlaps in "
                f"{len(overlaps)} artifact(s) ({total} total files). "
                f"Components (_run, _test, etc.) should be disjoint "
                f"(see https://github.com/ROCm/TheRock/issues/3796):\n{summary}"
            )

        artifact_names = {a.artifact_name for a in archive_index}
        logger.info(
            "Checked %d artifacts, no within-artifact component overlaps",
            len(artifact_names),
        )

    @pytest.mark.skipif(
        is_windows_platform(),
        reason="package.json coverage check only applies to Linux artifacts; "
        "package.json (build_tools/packaging/linux/package.json) has no "
        "entries for Windows-only artifacts. ",
    )
    def test_artifacts_covered_by_packages(self, archive_index: list[ArchiveInfo]):
        """Every fetched artifact/component should be covered by a package.

        Cross-references the fetched artifact archives (their name + component,
        derived from the filename) against the Linux packaging definitions in
        build_tools/packaging/linux/package.json. Coverage resolves per
        Artifact_Subdir so one subdir's gap cannot hide behind its siblings
        (see https://github.com/ROCm/TheRock/issues/9049).
        """
        coverage = build_packaged_components(load_package_data())
        uncovered = find_uncovered_components(archive_index, coverage)

        if SUBDIR_COVERAGE_MODE == "warn":
            fatal = [u for u in uncovered if u.level == "artifact"]
            warn_only = [u for u in uncovered if u.level == "subdir"]
        else:
            fatal, warn_only = uncovered, []

        for u in sorted(warn_only, key=lambda u: u.sort_key):
            logger.warning("SUBDIR package coverage gap (warn mode): %s", u.describe())
        for u in sorted(fatal, key=lambda u: u.sort_key):
            logger.error("MISSING package coverage: %s", u.describe())

        if fatal:
            summary = "\n".join(
                f"  [{u.level}] {u.describe()}"
                for u in sorted(fatal, key=lambda u: u.sort_key)
            )
            pytest.fail(
                f"Found {len(fatal)} artifact component(s) not covered by "
                f"package.json (see https://github.com/ROCm/TheRock/issues/9049):"
                f"\n{summary}\n"
                f"Fix: add the component to the subdir's Components list in "
                f"build_tools/packaging/linux/package.json, or add a triaged "
                f"(artifact, subdir, component) entry to "
                f"KNOWN_UNCOVERED_COMPONENTS with a tracking issue."
            )

        logger.info(
            "Checked %d archives (%d subdir-level warnings), all fatal "
            "components covered by package.json",
            len(archive_index),
            len(warn_only),
        )
