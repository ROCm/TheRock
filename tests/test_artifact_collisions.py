# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests that artifact archives don't contain files that collide after flattening.

When artifacts are installed (extracted and flattened), each file's basedir
prefix is stripped so that e.g. "core/clr/stage/lib/libfoo.so" becomes
"lib/libfoo.so". If two different archives contain files that flatten to the
same path, the second extraction silently overwrites the first (or races and
fails, as in https://github.com/ROCm/TheRock/issues/3758).

This test lists the contents of all artifact archives in a directory WITHOUT
extracting them, computes the flattened paths, and checks for duplicates.

Collisions between different components of the same artifact (e.g.,
foo_lib_generic and foo_run_generic) are expected — the component inheritance
model means "run" is a superset of "lib". Only collisions across different
artifact names are flagged.

Usage:
    THEROCK_ARTIFACTS_DIR=/path/to/archives \
        python -m pytest tests/test_artifact_collisions.py --log-cli-level=info

THEROCK_ARTIFACTS_DIR should point to a directory containing artifact archives
(*.tar.zst or *.tar.xz) as produced by the build pipeline. The test does not
require a GPU and can run on any CPU runner.
"""

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


class TestArtifactCollisions:
    """Checks for file path collisions across artifact archives."""

    def test_no_flattened_path_collisions(self, artifacts_dir: Path):
        """No two artifacts should contain files that flatten to the same path.

        This catches both same-basedir overlaps (like #3758) and the subtler
        cross-basedir case where different stage dirs install identically-named
        files (e.g., two subprojects both installing "bin/sequence.yaml").

        Collisions between components of the same artifact (e.g., lib and run)
        are expected due to component inheritance and are excluded.
        """
        archives = discover_archives(artifacts_dir)
        if not archives:
            pytest.skip(f"No artifact archives found in {artifacts_dir}")

        # flattened_path -> set of artifact names that contain it
        seen: dict[str, set[str]] = defaultdict(set)
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

            for fpath in flattened_paths:
                seen[fpath].add(an.name)

        # Only flag paths that appear in two or more different artifact names.
        collisions: dict[str, set[str]] = {
            fpath: artifact_names
            for fpath, artifact_names in seen.items()
            if len(artifact_names) > 1
        }

        if skipped:
            logger.warning("Skipped %d archives: %s", len(skipped), skipped)

        if collisions:
            # Show at most 20 collisions to keep output readable.
            lines = []
            for fpath, artifact_names in sorted(collisions.items())[:20]:
                lines.append(f"  {fpath}")
                for name in sorted(artifact_names):
                    lines.append(f"    - {name}")
            summary = "\n".join(lines)
            remaining = len(collisions) - 20
            if remaining > 0:
                summary += f"\n  ... and {remaining} more"
            pytest.fail(
                f"Found {len(collisions)} flattened path collision(s) across "
                f"{len(archives)} archives "
                f"(see https://github.com/ROCm/TheRock/issues/3758):\n{summary}"
            )

        logger.info(
            "Checked %d unique paths across %d archives, no collisions",
            len(seen),
            len(archives),
        )
