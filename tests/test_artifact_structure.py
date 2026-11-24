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

# Trailing prefix segments that are build-layout noise, not a subproject name.
_PREFIX_LEAF_IGNORE = {"stage", "dist", "build"}


def subdir_from_prefix(prefix: str) -> str:
    """Short subproject name for a manifest prefix, used for reporting and
    KNOWN_UNCOVERED_COMPONENTS keys.

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


def list_archive_files(archive_path: Path) -> list[tuple[str, str]]:
    """List the flattened file paths in an artifact archive.

    Reads artifact_manifest.txt (must be the first tar member) to get basedir
    prefixes, then strips those prefixes from each file member to produce
    the flattened output path. The matching prefix is kept so coverage can be
    checked per manifest line, the same way packaging_utils.py does (#9049).

    Returns:
        [(manifest_prefix, flattened_path), ...]
    """
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

        # Strip prefixes from each file member to produce flattened paths.
        # For example:
        #   prefix:     math-libs/BLAS/hipBLAS/stage
        #   member:     math-libs/BLAS/hipBLAS/stage/include/hipblas/hipblas.h
        #   flattened:  include/hipblas/hipblas.h
        while member := tf.next():
            if member.isdir():
                continue
            name = member.name
            for prefix in prefixes:
                prefix_slash = prefix + "/"
                if name.startswith(prefix_slash):
                    flattened_path = name[len(prefix_slash) :]
                    if flattened_path:
                        entries.append((prefix, flattened_path))
                    break

    return entries


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
    # manifest prefix line -> flattened paths it contributes to THIS archive.
    # Prefixes that ship nothing for this component are absent, so they are
    # never flagged.
    prefix_paths: dict[str, set[str]]

    @functools.cached_property
    def flattened_paths(self) -> set[str]:
        return set().union(*self.prefix_paths.values())


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
    """Components declared by package.json."""

    # Flat (non-Artifact_Subdir) entries: artifact -> {components}.
    flat: dict[str, set[str]]
    # artifact -> [(Name lowercased, {components})] for Artifact_Subdir entries.
    subdirs: dict[str, list[tuple[str, set[str]]]]

    def matching_subdirs(self, artifact: str, prefix: str) -> list[set[str]]:
        """Components of every Artifact_Subdir entry whose Name the packager
        would match against this manifest line.

        Mirrors packaging_utils.py: (Name + "/") in manifest_line.
        """
        line = prefix.lower()
        return [
            comps
            for name, comps in self.subdirs.get(artifact, [])
            if (name + "/") in line
        ]


def build_packaged_components(package_data: list) -> PackageCoverage:
    """Collect package.json coverage for both the nested ``Artifact_Subdir``
    structure and the older flat structure.

    Raises:
        ValueError: if an Artifact_Subdir entry has no "Name" (the packager
            would treat it as matching every manifest line).
    """
    flat: dict[str, set[str]] = defaultdict(set)
    subdirs: dict[str, list[tuple[str, set[str]]]] = defaultdict(list)

    for item in package_data:
        for pkg in item.get("Artifactory", []):
            artifact = (pkg.get("Artifact") or "").lower()
            if "Artifact_Subdir" in pkg:
                for subdir in pkg["Artifact_Subdir"]:
                    name = (subdir.get("Name") or "").lower()
                    if not name:
                        raise ValueError(
                            f"package.json: Artifact_Subdir entry under "
                            f"Artifact={artifact!r} has no 'Name'"
                        )
                    components = _clean_components(subdir.get("Components", ["run"]))
                    subdirs[artifact].append((name, set(components)))
            else:
                # Old flat structure without Artifact_Subdir.
                key = artifact or (pkg.get("Name") or "").lower()
                flat[key].update(_clean_components(pkg.get("Components", ["run"])))

    return PackageCoverage(flat=dict(flat), subdirs=dict(subdirs))


def _is_known_uncovered(artifact: str, subdir: str, component: str) -> bool:
    return (artifact, component) in KNOWN_UNCOVERED_COMPONENTS or (
        artifact,
        subdir,
        component,
    ) in KNOWN_UNCOVERED_COMPONENTS


def find_uncovered_components(
    archive_index: list["ArchiveInfo"], coverage: PackageCoverage
) -> list[str]:
    """Describe each artifact/subdir/component not covered by any package.

    Subdir-declared artifacts are checked per manifest line, so one subdir's
    gap cannot hide behind a sibling's entry (#9049).
    """
    uncovered: list[str] = []
    for info in archive_index:
        if info.component in IGNORED_COMPONENTS:
            continue
        artifact = info.artifact_name.lower()

        if artifact not in coverage.subdirs:
            if info.component not in coverage.flat.get(artifact, set()):
                if not _is_known_uncovered(artifact, "*", info.component):
                    uncovered.append(
                        f"{info.artifact_name} [{info.component}] "
                        f"({info.filename}): no package.json entry"
                    )
            continue

        for prefix in info.prefix_paths:
            subdir = subdir_from_prefix(prefix).lower()
            if _is_known_uncovered(artifact, subdir, info.component):
                continue
            matches = coverage.matching_subdirs(artifact, prefix)
            if any(info.component in comps for comps in matches):
                continue
            reason = (
                f"matching Artifact_Subdir does not list {info.component!r}"
                if matches
                else f"no Artifact_Subdir Name matches {prefix!r}"
            )
            uncovered.append(
                f"{info.artifact_name}/{subdir} [{info.component}] "
                f"({info.filename}): {reason}"
            )
    return sorted(uncovered)


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
            entries = list_archive_files(archive_path)
        except Exception:
            logger.exception("Failed to read %s", archive_name)
            skipped.append(archive_name)
            continue

        prefix_paths: dict[str, set[str]] = defaultdict(set)
        for prefix, fpath in entries:
            prefix_paths[prefix].add(fpath)

        index.append(
            ArchiveInfo(
                artifact_name=an.name,
                component=an.component,
                filename=archive_name,
                prefix_paths=dict(prefix_paths),
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

        for line in uncovered:
            logger.error("MISSING package coverage: %s", line)

        if uncovered:
            summary = "\n".join(f"  {line}" for line in uncovered)
            pytest.fail(
                f"Found {len(uncovered)} artifact component(s) not covered by "
                f"package.json (see https://github.com/ROCm/TheRock/issues/9049):"
                f"\n{summary}\n"
                f"Fix: add the component to the matching Artifact_Subdir in "
                f"build_tools/packaging/linux/package.json, or add a triaged "
                f"(artifact, subdir, component) entry to "
                f"KNOWN_UNCOVERED_COMPONENTS with a tracking issue."
            )

        logger.info(
            "Checked %d archives, all components covered by package.json",
            len(archive_index),
        )
