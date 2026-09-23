# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whole-tree invariant tests over the artifact-*.toml descriptors.

These tests are checks on the descriptors themselves rather than on any one
tool. Each loads every TheRock-owned `artifact-*.toml`, collects the basedirs
it declares, and asserts a property that must hold across the whole tree.
Descriptor discovery uses ``git ls-files`` so it does not recursively scan
generated build trees, caches, or the contents of submodules. Both tracked
files and non-ignored untracked files are included, so a newly created
descriptor is checked before it is staged.

`ArtifactDescriptorOverlapTest` — each basedir belongs to exactly one
descriptor. When two descriptors claim the same stage directory for any
component type, extracting both artifacts into the same output directory
causes file collisions. This was the root cause of
https://github.com/ROCm/TheRock/issues/3758, where both
`artifact-rocprofiler-sdk.toml` and `artifact-aqlprofile-tests.toml` included
`profiler/aqlprofile/stage`, causing concurrent extraction to race (and fail
with "file exists" errors) on overlapping files.

Cases the overlap test does NOT catch: two artifacts with *different* basedirs
whose installed files collide after flattening (basedir prefix stripped). If
two subprojects both install a file to the same relative path (e.g.,
"lib/libfoo.so" or "bin/sequence.yaml") in their respective stage dirs,
flattening produces duplicate paths even though the basedirs are distinct.
Current projects avoid this by using unique file names, but it's convention
rather than enforcement. Catching this requires inspecting actual build output.

`BootstrapMarkerBasedirTest` — the bootstrap ".prebuilt" marker derived from
each basedir names a directory the build actually checks. See
https://github.com/ROCm/TheRock/issues/7549.
"""

import subprocess
import sys
import tomllib
import unittest
from pathlib import Path

THEROCK_DIR = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _therock_utils.artifacts import STAGE_DIR_NAME, prebuilt_marker_relpath


def get_descriptor_paths() -> list[Path]:
    """Find TheRock-owned descriptors without crawling the whole workspace.

    Using a directory allow-list would need updating whenever descriptors move
    into a new source area, while a deny-list could easily miss a new generated
    or external tree. Git already knows the relevant boundary: ``--cached``
    lists files tracked by this repository (not files inside submodules), and
    ``--others --exclude-standard`` adds new, non-ignored files while excluding
    ignored build outputs and caches.
    """
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            ":(glob)**/artifact-*.toml",
        ],
        cwd=THEROCK_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    return [THEROCK_DIR / relpath for relpath in result.stdout.splitlines()]


def get_basedirs(descriptor_path: Path) -> set[str]:
    """Extract all basedir paths from an artifact descriptor.

    The TOML structure for basedirs is:
        [components.{component_name}."{basedir_path}"]

    Within a component's dict, string-keyed dict values are basedir entries.
    Non-dict values (like "extends") are component-level fields.
    """
    with open(descriptor_path, "rb") as f:
        data = tomllib.load(f)

    basedirs: set[str] = set()
    for _comp_name, comp_data in data.get("components", {}).items():
        if not isinstance(comp_data, dict):
            continue
        for key, value in comp_data.items():
            if isinstance(value, dict):
                basedirs.add(key)
    return basedirs


class ArtifactDescriptorOverlapTest(unittest.TestCase):
    """Verifies no two artifact descriptors claim the same stage directory."""

    def test_no_duplicate_basedirs_across_descriptors(self):
        """Each stage directory must belong to exactly one artifact descriptor.

        If two descriptors reference the same basedir, their tarballs will
        contain overlapping files, causing extraction failures.
        """
        # basedir -> first descriptor that claims it
        seen: dict[str, Path] = {}
        errors: list[str] = []

        descriptors = sorted(get_descriptor_paths())
        self.assertGreater(
            len(descriptors),
            0,
            f"No artifact descriptors found, check THEROCK_DIR ('{THEROCK_DIR}')",
        )

        for descriptor_path in descriptors:
            relpath = descriptor_path.relative_to(THEROCK_DIR)
            for basedir in get_basedirs(descriptor_path):
                if basedir in seen:
                    errors.append(
                        f"basedir '{basedir}' claimed by both "
                        f"{seen[basedir]} and {relpath}"
                    )
                else:
                    seen[basedir] = relpath

        if errors:
            self.fail(
                "Duplicate basedirs across artifact descriptors will cause "
                "extraction collisions (see "
                "https://github.com/ROCm/TheRock/issues/3758):\n"
                + "\n".join(f"  - {e}" for e in errors)
            )


class BootstrapMarkerBasedirTest(unittest.TestCase):
    """Verifies the bootstrap marker derived from each basedir names a stage dir.

    `--bootstrap` writes a ".prebuilt" marker per artifact manifest basedir, and
    therock_subproject.cmake only ever looks for "${_stage_dir}.prebuilt". A
    basedir nested below its stage dir (see `dctools/artifact-rdc.toml`) would
    otherwise produce a marker the build never checks.
    """

    def test_every_basedir_yields_a_stage_dir_marker(self):
        descriptors = sorted(get_descriptor_paths())
        self.assertGreater(
            len(descriptors),
            0,
            f"No artifact descriptors found, check THEROCK_DIR ('{THEROCK_DIR}')",
        )

        basedirs: set[str] = set()
        for descriptor_path in descriptors:
            basedirs |= get_basedirs(descriptor_path)

        # Any basedir whose marker is not simply "<basedir>.prebuilt" must have
        # been truncated to an enclosing stage dir, never anywhere else.
        rewritten: dict[str, str] = {}
        for basedir in sorted(basedirs):
            marker = prebuilt_marker_relpath(basedir)
            if marker == basedir + ".prebuilt":
                continue
            stage_dir = marker[: -len(".prebuilt")]
            self.assertTrue(
                basedir.startswith(stage_dir + "/"),
                f"Marker for '{basedir}' escaped its own basedir: '{marker}'",
            )
            self.assertTrue(
                stage_dir.endswith("/" + STAGE_DIR_NAME),
                f"Marker for '{basedir}' does not name a stage dir: '{marker}'",
            )
            rewritten[basedir] = marker

        # Only descriptors that declare a basedir below their stage dir are
        # affected. If this list grows, confirm the new marker is the one the
        # subproject's stage dir would produce.
        self.assertEqual(
            rewritten,
            {"dctools/rdc/stage/portable-rdc": "dctools/rdc/stage.prebuilt"},
        )


if __name__ == "__main__":
    unittest.main()
