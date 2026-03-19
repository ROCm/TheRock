# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests validating that artifact descriptors don't have overlapping basedirs.

When two artifact descriptors claim the same stage directory for any component
type, extracting both artifacts into the same output directory causes file
collisions.

This was the root cause of https://github.com/ROCm/TheRock/issues/3758, where
both `artifact-rocprofiler-sdk.toml` and `artifact-aqlprofile-tests.toml`
included `profiler/aqlprofile/stage`, causing concurrent extraction to race
(and fail with "file exists" errors) on overlapping files.

This test loads the real artifact-*.toml descriptors from the source tree and
checks that each stage directory (basedir) belongs to exactly one descriptor.

Limitations (cases this test does NOT catch):
  - Two artifacts with *different* basedirs whose installed files collide
    after flattening (basedir prefix stripped). If two subprojects both
    install a file to the same relative path (e.g., "lib/libfoo.so" or
    "bin/sequence.yaml") in their respective stage dirs, flattening produces
    duplicate paths even though the basedirs are distinct. Current projects
    avoid this by using unique file names, but it's convention rather than
    enforcement. Catching this requires inspecting actual build output.
"""

import tomllib
import unittest
from pathlib import Path

THEROCK_DIR = Path(__file__).resolve().parent.parent.parent


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

        descriptors = sorted(THEROCK_DIR.rglob("artifact-*.toml"))
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


if __name__ == "__main__":
    unittest.main()
