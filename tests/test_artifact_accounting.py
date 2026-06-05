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
            continue
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
