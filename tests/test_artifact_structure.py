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

Usage:
    THEROCK_ARTIFACTS_DIR=/path/to/archives \\
        python -m pytest tests/test_artifact_structure.py -v --log-cli-level=info

THEROCK_ARTIFACTS_DIR should point to a directory containing artifact archives
(*.tar.zst or *.tar.xz) as produced by the build pipeline.
"""

import dataclasses
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import pytest

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "build_tools"))

from _therock_utils.artifacts import ArtifactName, _open_archive_for_read

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = os.getenv("THEROCK_ARTIFACTS_DIR", "")


@pytest.fixture(scope="session")
def artifacts_dir() -> Path:
    if not ARTIFACTS_DIR:
        pytest.skip("THEROCK_ARTIFACTS_DIR not set")
    path = Path(ARTIFACTS_DIR).resolve()
    if not path.is_dir():
        pytest.fail(f"THEROCK_ARTIFACTS_DIR is not a directory: {path}")
    return path


def list_archive_files(archive_path: Path) -> tuple[list[str], list[str]]:
    """List the flattened file paths in an artifact archive.

    Reads artifact_manifest.txt (must be the first tar member) to get basedir
    prefixes, then strips those prefixes from each file member to produce
    the flattened output path.

    Returns:
        (manifest_prefixes, flattened_file_paths)
    """
    prefixes: list[str] = []
    flattened: list[str] = []

    with _open_archive_for_read(archive_path) as tf:
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
                        flattened.append(flattened_path)
                    break

    return prefixes, flattened


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
    flattened_paths: set[str]


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
            _prefixes, flattened_paths = list_archive_files(archive_path)
        except Exception:
            logger.exception("Failed to read %s", archive_name)
            skipped.append(archive_name)
            continue

        index.append(
            ArchiveInfo(
                artifact_name=an.name,
                component=an.component,
                filename=archive_name,
                flattened_paths=set(flattened_paths),
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
