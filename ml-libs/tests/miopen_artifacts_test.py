# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check that Windows MIOpen system databases reach shipped library artifacts."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "build_tools"))
from _therock_utils.archive_util import open_archive_for_read


@pytest.mark.skipif(
    os.getenv("PLATFORM", "linux").lower() != "windows",
    reason="Windows MIOpen installs system databases under bin/",
)
def test_windows_miopen_system_databases():
    """Existing gfx110X find/perf pairs must survive in their per-arch libraries."""
    artifacts_dir = os.getenv("THEROCK_ARTIFACTS_DIR")
    if not artifacts_dir:
        pytest.skip("THEROCK_ARTIFACTS_DIR not set")
    root = Path(artifacts_dir)
    assert root.is_dir(), f"Artifact directory missing: {root}"

    # These pairs have shipped since before gfx1101 was added. Check the actual
    # archives; a present DB in another component does not make the runtime work.
    basenames = {"gfx1100": "gfx110060", "gfx1102": "gfx110220"}
    families = set(os.getenv("AMDGPU_FAMILIES", "").lower().split(";"))
    archives = {
        arch: next(
            (
                path
                for ext in ("zst", "xz")
                if (path := root / f"miopen_lib_{arch}.tar.{ext}").is_file()
            ),
            None,
        )
        for arch in basenames
    }
    if "gfx110x-all" not in families:
        basenames = {arch: base for arch, base in basenames.items() if archives[arch]}
        if not basenames:
            pytest.skip("No Windows gfx110X MIOpen library artifacts in this run")

    for arch, basename in basenames.items():
        archive = archives[arch]
        assert archive is not None, f"Missing MIOpen library artifact for {arch}"
        with open_archive_for_read(archive) as tf:
            manifest = tf.next()
            assert manifest is not None and manifest.name == "artifact_manifest.txt"
            manifest_file = tf.extractfile(manifest)
            assert manifest_file is not None
            prefixes = tuple(
                f"{line.rstrip('/')}/"
                for line in manifest_file.read().decode().splitlines()
                if line.strip()
            )
            flattened_paths = set()
            for member in tf:
                if member.isdir():
                    continue
                for prefix in prefixes:
                    if member.name.startswith(prefix):
                        flattened_paths.add(member.name[len(prefix) :])
                        break

        expected = {f"bin/{basename}.HIP.fdb.txt", f"bin/{basename}.db.txt"}
        missing = expected - flattened_paths
        if missing:
            pytest.fail(
                f"{archive.name}: missing Windows MIOpen system DBs {sorted(missing)}"
            )
