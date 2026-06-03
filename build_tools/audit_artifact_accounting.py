#!/usr/bin/env python3
"""Full artifact file-accounting audit across every artifact-*.toml descriptor.

For each artifact descriptor in the repo, scan its component stage basedirs under a
build root and report any staged file that is **not claimed by exactly one
component** -- i.e. the exact check that `fileset_tool.py`'s
`ComponentScanner.verify()` performs but which is currently DISABLED in the normal
artifact build (see the commented-out `scanner.verify()` in fileset_tool.py).

Why this exists
---------------
Because verify() is off, an unclaimed staged file is **silently dropped** from all
artifacts instead of erroring -- it then resurfaces much later as a mysterious
rocm-sdk test/packaging failure. That is what caused origami to be reverted three
times (#2813/#3237/#3820): origami installed `share/doc/origami/LICENSE.md` but its
descriptor had no `doc` component, so the license was claimed by nobody. This tool
catches that class of bug for ANY subproject, up front, with a precise message.

Usage
-----
    # Audit the whole repo's descriptors against a completed build tree:
    python build_tools/audit_artifact_accounting.py --root-dir /path/to/TheRock/build

    # Audit specific descriptors:
    python build_tools/audit_artifact_accounting.py --root-dir BUILD \
        math-libs/BLAS/artifact-blas.toml

    # Exit non-zero on any unaccounted file (for CI):
    python build_tools/audit_artifact_accounting.py --root-dir BUILD --strict

Descriptors whose stage basedirs were not built (e.g. components disabled in this
configuration) are reported as SKIPPED rather than failed, so it works on partial
builds. In CI, run it after a full build to get repo-wide accounting.
"""

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "_therock_utils"))

import _therock_utils.artifact_builder as ab  # noqa: E402


def discover_descriptors(repo: Path) -> list[Path]:
    out = []
    for p in repo.rglob("artifact-*.toml"):
        s = str(p)
        if "/build/" in s or "/.claude/" in s or "/_therock_utils/" in s:
            continue
        out.append(p)
    return sorted(out)


def audit_descriptor(toml: Path, root: Path):
    """Return (artifact_name, undeclared_relpaths, scanned_basedirs, missing_basedirs, error)."""
    name = toml.stem
    if name.startswith("artifact-"):
        name = name[len("artifact-"):]
    try:
        ad = ab.ArtifactDescriptor.load_toml_file(toml, artifact_name=name)
        scanner = ab.ComponentScanner(root, ad)
    except Exception as e:  # malformed descriptor / scan error
        return name, [], [], [], f"{type(e).__name__}: {e}"

    missing = sorted(getattr(scanner, "missing_basedirs", []) or [])
    try:
        all_bd = sorted(scanner.all_basedirs)
    except Exception:
        all_bd = []
    scanned = [bd for bd in all_bd if bd not in missing]

    # Replicate verify()'s "undeclared unmatched" check exactly: an unmatched file
    # is a problem unless the descriptor allow-lists it via options.unmatched_exclude.
    undeclared = []
    try:
        for relpath, direntry in scanner.unmatched_files:
            if ad.options.unmatched_pattern.matches(relpath, direntry):
                undeclared.append(relpath)
    except Exception as e:
        return name, [], scanned, missing, f"unmatched-scan {type(e).__name__}: {e}"
    return name, sorted(undeclared), scanned, missing, None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root-dir", required=True, type=Path,
                    help="Build root the descriptor basedirs are relative to (THEROCK_BINARY_DIR).")
    ap.add_argument("--repo", type=Path, default=_HERE.parent,
                    help="Repo root to discover artifact-*.toml under (default: TheRock root).")
    ap.add_argument("descriptors", nargs="*", type=Path,
                    help="Specific descriptor files to audit (default: discover all).")
    ap.add_argument("--strict", action="store_true",
                    help="Exit non-zero if any unaccounted file is found.")
    ap.add_argument("--quiet", action="store_true", help="Only print problems + summary.")
    args = ap.parse_args(argv)

    root: Path = args.root_dir
    if not root.is_dir():
        print(f"ERROR: --root-dir does not exist: {root}", file=sys.stderr)
        return 2

    descriptors = args.descriptors or discover_descriptors(args.repo)
    total_undeclared = 0
    audited = skipped = errored = 0
    problems: list[tuple[str, Path, list[str]]] = []

    for toml in descriptors:
        name, undeclared, scanned, missing, error = audit_descriptor(toml, root)
        rel = toml.relative_to(args.repo) if str(toml).startswith(str(args.repo)) else toml
        if error:
            errored += 1
            print(f"[ERROR ] {rel}: {error}")
            continue
        if not scanned:
            skipped += 1
            if not args.quiet:
                print(f"[skip  ] {rel}  (no built stage dirs under root)")
            continue
        audited += 1
        if undeclared:
            total_undeclared += len(undeclared)
            problems.append((name, rel, undeclared))
            print(f"[FAIL  ] {rel}: {len(undeclared)} unaccounted file(s):")
            for r in undeclared:
                print(f"             {r}")
        elif not args.quiet:
            print(f"[ok    ] {rel}  ({len(scanned)} stage dir(s) accounted)")

    print()
    print(f"=== artifact accounting: {audited} audited, {skipped} skipped (not built), "
          f"{errored} descriptor error(s), {total_undeclared} unaccounted file(s) ===")
    if problems:
        print("UNACCOUNTED FILES are silently dropped from artifacts (verify() is disabled in")
        print("the build) and resurface as rocm-sdk failures. Fix by adding the right component")
        print("entry to the descriptor (e.g. a [components.doc.\"<stage>\"] for share/doc files),")
        print("or allow-list via options.unmatched_exclude if intentional.")
    if args.strict and (total_undeclared or errored):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
