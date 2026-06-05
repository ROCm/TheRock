"""Full artifact file-accounting test.

Asserts that, across EVERY artifact-*.toml descriptor, every staged file is claimed
by exactly one component -- the check that `fileset_tool.py`'s `ComponentScanner.verify()`
performs but which is currently disabled in the normal artifact build. Unaccounted
files are otherwise silently dropped from artifacts and resurface as mysterious
rocm-sdk failures (this is what got origami reverted 3x: an unclaimed
share/doc/origami/LICENSE.md with no `doc` component entry).

Runs against a completed build tree pointed to by THEROCK_BINARY_DIR (the same root
the build passes to `fileset_tool.py artifact --root-dir`). Skips if unset so it is
safe in environments without a build. Intended to run as a post-build CI gate.

    THEROCK_BINARY_DIR=/path/to/TheRock/build python -m pytest tests/test_artifact_accounting.py -v
"""

import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "build_tools"))

import audit_artifact_accounting as audit  # noqa: E402


def _build_root() -> Path | None:
    for var in ("THEROCK_BINARY_DIR", "THEROCK_BUILD_DIR"):
        v = os.environ.get(var)
        if v and Path(v).is_dir():
            return Path(v)
    return None


@pytest.mark.skipif(
    _build_root() is None, reason="THEROCK_BINARY_DIR not set / not a build tree"
)
def test_all_staged_files_are_accounted():
    root = _build_root()
    descriptors = audit.discover_descriptors(_REPO)
    assert descriptors, "no artifact-*.toml descriptors discovered"

    failures: dict[str, list[str]] = {}
    audited = 0
    for toml in descriptors:
        name, undeclared, scanned, missing, error = audit.audit_descriptor(toml, root)
        assert error is None, f"{toml}: descriptor error: {error}"
        if not scanned:
            continue  # subproject not built in this configuration
        audited += 1
        if undeclared:
            failures[str(toml.relative_to(_REPO))] = undeclared

    assert audited > 0, f"no built stage dirs found under {root}"
    if failures:
        lines = [
            "Unaccounted staged files (claimed by no artifact component; would be",
            "silently dropped and break the rocm-sdk build). Add the right component",
            'entry to the descriptor (e.g. [components.doc."<stage>"] for share/doc),',
            "or allow-list via options.unmatched_exclude:",
        ]
        for desc, files in sorted(failures.items()):
            lines.append(f"  {desc}:")
            lines.extend(f"      {f}" for f in files)
        pytest.fail("\n".join(lines))
