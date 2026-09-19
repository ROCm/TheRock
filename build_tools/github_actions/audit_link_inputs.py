#!/usr/bin/env python
"""Audit linker response files after a failed build.

A recurring Windows CI flake makes `lld-link` report every symbol of one
translation unit as undefined, even though ccache wrote that object with a
plausible size before the link ran. The two remaining explanations are that
the object was not in the link's input list, or that it was on disk but
unreadable/empty.

CMake passes object lists to the linker in `CMakeFiles/<target>.rsp`, so the
list never appears in the build log. This dumps every response file and
reports any listed input that is missing or zero-length, which separates the
two cases in a single run.

Writes into `<build-dir>/logs/` so the existing stage log upload picks it up.
"""

import argparse
import shutil
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir",
        type=Path,
        required=True,
        help="Build tree to scan for CMakeFiles/*.rsp files.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Where to write the report (default: <build-dir>/logs).",
    )
    return parser.parse_args(argv)


def read_entries(rsp: Path) -> list[str]:
    """Returns the whitespace-separated entries of a response file."""
    try:
        text = rsp.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [f"<unreadable: {exc}>"]
    return [entry.strip('"') for entry in text.split() if entry.strip('"')]


def audit(build_dir: Path, log_dir: Path) -> int:
    rsp_files = sorted(build_dir.rglob("CMakeFiles/*.rsp"))
    rsp_copy_dir = log_dir / "rsp"
    rsp_copy_dir.mkdir(parents=True, exist_ok=True)

    report: list[str] = [
        f"scanned {len(rsp_files)} response files under {build_dir}",
        "",
    ]
    suspect_total = 0

    for rsp in rsp_files:
        # Entries are relative to the directory containing CMakeFiles/.
        base = rsp.parent.parent
        suspects: list[str] = []
        objects = 0
        for entry in read_entries(rsp):
            if not entry.lower().endswith((".obj", ".o")):
                continue
            objects += 1
            path = Path(entry)
            if not path.is_absolute():
                path = base / path
            try:
                size = path.stat().st_size
            except OSError as exc:
                suspects.append(f"MISSING  {entry}  ({exc.__class__.__name__})")
                continue
            if size == 0:
                suspects.append(f"EMPTY    {entry}")
        if not objects:
            continue

        rel = rsp.relative_to(build_dir)
        flat = str(rel).replace("\\", "_").replace("/", "_")
        shutil.copyfile(rsp, rsp_copy_dir / flat)

        status = f"{len(suspects)} suspect" if suspects else "ok"
        report.append(f"{rel}: {objects} objects, {status}")
        for line in suspects:
            report.append(f"    {line}")
        suspect_total += len(suspects)

    report.append("")
    report.append(f"TOTAL SUSPECT INPUTS: {suspect_total}")

    out = log_dir / "link_inputs_audit.txt"
    out.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report[-40:]))
    print(f"\n[audit_link_inputs] wrote {out}")
    print(
        f"[audit_link_inputs] copied {len(rsp_files)} response files to {rsp_copy_dir}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build_dir: Path = args.build_dir
    if not build_dir.is_dir():
        print(f"[audit_link_inputs] no build dir at {build_dir}, nothing to do")
        return 0
    log_dir: Path = args.log_dir or (build_dir / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    return audit(build_dir, log_dir)


if __name__ == "__main__":
    sys.exit(main())
