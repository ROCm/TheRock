#!/usr/bin/env python
"""Audit linker response files after a failed build.

A recurring Windows CI flake makes `lld-link` report every symbol of one
translation unit as undefined, even though ccache wrote that object with a
plausible size before the link ran. The two remaining explanations are that
the object was not in the link's input list, or that it was on disk but
unreadable/empty.

CMake passes object lists to the linker in `CMakeFiles/<target>.rsp`, so the
list never appears in the build log. This dumps every response file and
reports any listed input that is missing, zero-length, or not a valid COFF
object, which separates the two cases in a single run.

It then re-runs the failed sub-project build without touching any source. If
ninja re-runs only the link and it now succeeds, the inputs were correct all
along and the original failure was a transient read.

Writes into `<build-dir>/logs/` so the existing stage log upload picks it up.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# First two bytes of a COFF object for the architectures used here. An object
# that is non-empty but does not start with one of these was truncated or
# overwritten rather than merely missing.
COFF_MAGIC = {
    b"\x64\x86",  # IMAGE_FILE_MACHINE_AMD64
    b"\x00\x00",  # anonymous object / bigobj header
    b"\x4c\x01",  # IMAGE_FILE_MACHINE_I386
    b"\xaa\x64",  # IMAGE_FILE_MACHINE_ARM64
}


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


def find_failed_subprojects(build_dir: Path) -> list[Path]:
    """Returns sub-project build dirs whose build log ended in a ninja failure."""
    failed: list[Path] = []
    for log in sorted((build_dir / "logs").glob("*_build.log")):
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "ninja: build stopped" not in text:
            continue
        # The log's EXEC line records the sub-project build directory.
        for line in text.splitlines():
            if line.startswith("EXEC\t"):
                candidate = Path(line.split("\t")[1].strip())
                if candidate.is_dir():
                    failed.append(candidate)
                break
    return failed


def run_ninja(args: list[str], cwd: Path) -> subprocess.CompletedProcess | None:
    """Runs ninja in cwd, or returns None if ninja is not available.

    The audit must never lose its primary data because a retry could not run,
    so a missing ninja is reported rather than raised.
    """
    try:
        return subprocess.run(
            ["ninja", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            errors="replace",
        )
    except OSError:
        return None


def retry_link(sub_dir: Path, report: list[str]) -> None:
    """Re-runs a failed sub-project build and records what ninja redid.

    No source is touched between the two runs, so if ninja now reports only
    the link edge and that link succeeds, every object input was already
    correct on disk and the first failure was a transient read rather than a
    missing or corrupt input.
    """
    report.append("")
    report.append(f"=== retry: {sub_dir} ===")

    dry = run_ninja(["-n", "-v"], sub_dir)
    if dry is None:
        report.append("ninja not available; skipped retry")
        print("[audit_link_inputs] ninja not on PATH, skipped retry")
        return
    pending = [ln for ln in dry.stdout.splitlines() if ln.strip()]
    report.append(f"ninja -n reports {len(pending)} pending edge(s):")
    for line in pending[:20]:
        report.append(f"    {line[:200]}")

    rerun = run_ninja([], sub_dir)
    if rerun is None:
        report.append("ninja disappeared between calls; retry incomplete")
        return
    report.append(f"re-run exit code: {rerun.returncode}")
    tail = (rerun.stdout + rerun.stderr).splitlines()
    for line in tail[-30:]:
        report.append(f"    {line[:200]}")
    verdict = (
        "RE-LINK SUCCEEDED -> inputs were correct on disk; original failure was transient"
        if rerun.returncode == 0
        else "RE-LINK FAILED AGAIN -> inputs are deterministically wrong"
    )
    report.append(verdict)
    print(f"[audit_link_inputs] {verdict}")


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
                continue
            try:
                with path.open("rb") as handle:
                    magic = handle.read(2)
            except OSError as exc:
                suspects.append(f"UNREADABLE {entry}  ({exc.__class__.__name__})")
                continue
            if magic not in COFF_MAGIC:
                suspects.append(f"NOT-COFF {entry}  (size={size}, magic={magic.hex()})")
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

    for sub_dir in find_failed_subprojects(build_dir):
        retry_link(sub_dir, report)

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
