# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Structural validation of artifact archives.

These tests scan artifact archives WITHOUT extracting them and check invariants
that the build system should maintain. They run on CPU (no GPU required) and
are designed to catch issues before artifacts are installed or tested on GPU
runners.

Tests:
  - Cross-artifact collisions: no two artifacts should produce files that
    flatten to the same path (causes silent overwrites or race conditions,
    see https://github.com/ROCm/TheRock/issues/3758).
  - Within-artifact component collisions: different components (lib, run,
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
        manifest_member = tf.next()
        if manifest_member is None or manifest_member.name != "artifact_manifest.txt":
            raise ValueError(
                f"{archive_path.name}: expected artifact_manifest.txt as first member, "
                f"got {manifest_member.name if manifest_member else 'empty archive'}"
            )
        with tf.extractfile(manifest_member) as mf:
            prefixes = [
                line for line in mf.read().decode().splitlines() if line.strip()
            ]

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
class FileCollision:
    """A flattened file path found in multiple artifacts."""

    path: str
    sources: list[tuple[str, str]]  # (artifact_name, archive_filename)


@dataclasses.dataclass
class ArtifactComponentOverlap:
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


def _format_collision_summary(collisions: list[FileCollision], limit: int = 20) -> str:
    """Format cross-artifact collisions into readable summary."""
    lines = []
    for collision in sorted(collisions, key=lambda c: c.path)[:limit]:
        lines.append(f"  {collision.path}")
        for label, archive in sorted(collision.sources):
            lines.append(f"    - {label} ({archive})")
    remaining = len(collisions) - limit
    if remaining > 0:
        lines.append(f"  ... and {remaining} more")
    return "\n".join(lines)


def _format_overlap_summary(overlaps: list[ArtifactComponentOverlap]) -> str:
    """Format within-artifact component overlaps into readable summary."""
    lines = []
    total = 0
    for overlap in sorted(overlaps, key=lambda o: o.artifact_name):
        total += len(overlap.overlaps)
        lines.append(f"  {overlap.artifact_name} ({len(overlap.overlaps)} files):")
        for fpath in sorted(overlap.overlaps)[:5]:
            comps = sorted(set(overlap.overlaps[fpath]))
            lines.append(f"    {fpath}  [{', '.join(comps)}]")
        if len(overlap.overlaps) > 5:
            lines.append(f"    ... and {len(overlap.overlaps) - 5} more")
    return total, "\n".join(lines)


class TestArtifactStructure:
    """Structural validation of artifact archives."""

    def test_no_cross_artifact_collisions(self, archive_index: list[ArchiveInfo]):
        """No two artifacts should contain files that flatten to the same path.

        This catches both same-basedir overlaps (like #3758) and the subtler
        cross-basedir case where different stage dirs install identically-named
        files (e.g., two subprojects both installing "bin/sequence.yaml").

        See https://github.com/ROCm/TheRock/issues/3796
        """
        # flattened_path -> { artifact_name: set of archive filenames }
        seen: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

        for info in archive_index:
            for fpath in info.flattened_paths:
                seen[fpath][info.artifact_name].add(info.filename)

        # Flag paths that appear in two or more different artifact names.
        collisions: list[FileCollision] = []
        for fpath, by_name in seen.items():
            if len(by_name) > 1:
                sources = []
                for name, archive_names in by_name.items():
                    for a in archive_names:
                        sources.append((name, a))
                collisions.append(FileCollision(path=fpath, sources=sources))

        if collisions:
            summary = _format_collision_summary(collisions)
            pytest.fail(
                f"Found {len(collisions)} cross-artifact collision(s) across "
                f"{len(archive_index)} archives "
                f"(see https://github.com/ROCm/TheRock/issues/3796):\n{summary}"
            )

        logger.info(
            "Checked %d unique paths across %d archives, no cross-artifact collisions",
            len(seen),
            len(archive_index),
        )

    def test_no_within_artifact_component_collisions(
        self, archive_index: list[ArchiveInfo]
    ):
        """Components of the same artifact should contain disjoint files.

        The component scanner (artifact_builder.py) enforces disjointness via
        the extends chain (lib -> run -> dbg -> dev -> doc). However, the
        "test" component has no extends, so it can re-claim files already taken
        by other components.

        See https://github.com/ROCm/TheRock/issues/3796
        """
        # Group archives by artifact name, merging target variants per component.
        # artifact_name -> component -> set of flattened paths
        by_artifact: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        for info in archive_index:
            by_artifact[info.artifact_name][info.component].update(info.flattened_paths)

        all_overlaps: list[ArtifactComponentOverlap] = []

        for artifact_name, comp_files in by_artifact.items():
            comp_names = sorted(comp_files.keys())
            # fpath -> list of component names that contain it
            artifact_overlaps: dict[str, list[str]] = {}
            for i in range(len(comp_names)):
                for j in range(i + 1, len(comp_names)):
                    c1, c2 = comp_names[i], comp_names[j]
                    for fpath in comp_files[c1] & comp_files[c2]:
                        artifact_overlaps.setdefault(fpath, []).extend([c1, c2])

            if artifact_overlaps:
                all_overlaps.append(
                    ArtifactComponentOverlap(
                        artifact_name=artifact_name, overlaps=artifact_overlaps
                    )
                )

        if all_overlaps:
            total, summary = _format_overlap_summary(all_overlaps)
            pytest.fail(
                f"Found within-artifact component collisions in "
                f"{len(all_overlaps)} artifact(s) ({total} total files). "
                f"Components should be disjoint "
                f"(see https://github.com/ROCm/TheRock/issues/3796):\n{summary}"
            )

        artifact_names = {a.artifact_name for a in archive_index}
        logger.info(
            "Checked %d artifacts, no within-artifact component collisions",
            len(artifact_names),
        )
