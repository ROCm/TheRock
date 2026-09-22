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

That covers failures whose inputs are objects. A second, distinct mode was
observed in CI run 35479909973: every object was valid and the re-link failed
identically, because the missing symbols were meant to come from an *import
library* rather than from an object. Object-only auditing reports "0 suspect
inputs" for that case and says nothing about the actual faulty input, so the
libraries named on the failed link line are audited too: each is checked for
the symbols the linker said were undefined, and its size/mtime/digest are
recorded so a bad build product can be told apart from a bad read.

Writes into `<build-dir>/logs/` so the existing stage log upload picks it up.
"""

import argparse
import hashlib
import os
import re
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

# A bigobj/anonymous object starts with IMAGE_FILE_MACHINE_UNKNOWN followed by
# 0xFFFF, so a run of zeros shares only its first two bytes. Checking two bytes
# alone therefore accepts an all-zero file as a valid object, which matters
# because that is precisely what a bad read produces: lld-link accepts such an
# object without complaint and silently omits everything it should have
# defined. Verified against lld-link directly -- an all-zero object of the
# original size links with exit 0 and drops the exports.
ANON_OBJECT_SIG2 = b"\xff\xff"


def classify_object(path: Path, size: int) -> str | None:
    """Returns a defect description for an object file, or None if it looks sound.

    Reads only the header plus enough of the body to tell an all-zero file
    apart from a merely unusual one.
    """
    try:
        with path.open("rb") as handle:
            head = handle.read(4)
            probe = handle.read(4096)
    except OSError as exc:
        return f"UNREADABLE ({exc.__class__.__name__})"

    magic = head[:2]
    if magic not in COFF_MAGIC:
        return f"NOT-COFF (size={size}, magic={magic.hex()})"

    # An all-zero header is only legitimate for an anonymous object, which must
    # carry 0xFFFF next. Without that, this is a zero-filled file wearing a
    # valid-looking magic.
    if magic == b"\x00\x00" and head[2:4] != ANON_OBJECT_SIG2:
        if not any(head) and not any(probe):
            return (
                f"ALL-ZEROS (size={size}) -- links cleanly but defines nothing; "
                "this is the signature of a bad read, not a bad compile"
            )
        return f"BAD-ANON-HEADER (size={size}, head={head.hex()})"
    return None


# `lld-link: error: undefined symbol: __declspec(dllimport) rocsolver_csytrs`
# The build log prefixes each line with an elapsed-time stamp, so this is
# deliberately not anchored to the start of the line.
UNDEFINED_RE = re.compile(
    r"undefined symbol:\s*(?:__declspec\(dllimport\)\s*)?(.+?)\s*$"
)

# `-LB:/path/to/lib` or `-L B:/path/to/lib`.
LIB_DIR_RE = re.compile(r"-L\s*([^\s\"]+)")

LIB_SUFFIXES = (".lib", ".dll", ".a", ".so", ".dll.a")


def parse_undefined_symbols(text: str) -> list[str]:
    """Returns the symbols lld reported as undefined, in first-seen order.

    C++ symbols are printed demangled and so will not match the mangled names
    that nm reports; they are still collected because a human reading the
    report needs to see everything the linker complained about.
    """
    seen: dict[str, None] = {}
    for line in text.splitlines():
        if "undefined symbol:" not in line:
            continue
        match = UNDEFINED_RE.search(line)
        if match:
            seen.setdefault(match.group(1), None)
    return list(seen)


def parse_link_libraries(text: str) -> tuple[list[str], list[str]]:
    """Returns (library search dirs, library tokens) from failed link lines.

    Only lines belonging to a failed link are considered, so an unrelated
    successful link earlier in the log cannot contribute libraries that were
    never part of the failure.
    """
    dirs: dict[str, None] = {}
    libs: dict[str, None] = {}
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if "FAILED:" not in line:
            continue
        # The command follows the FAILED: marker, usually on the next line.
        for command in lines[index : index + 3]:
            if "-fuse-ld" not in command and "lld-link" not in command:
                continue
            for hit in LIB_DIR_RE.finditer(command):
                dirs.setdefault(hit.group(1).replace("\\", "/"), None)
            for token in command.replace('"', " ").split():
                bare = token.strip(",")
                if bare.lower().endswith(LIB_SUFFIXES):
                    libs.setdefault(bare.replace("\\", "/"), None)
    return list(dirs), list(libs)


def find_llvm_tool(name: str, build_dir: Path, log_text: str) -> Path | None:
    """Locates an LLVM binary, preferring the toolchain the build actually used.

    The compiler path in the build log is authoritative: a tool from some other
    toolchain could disagree about what the file contains.
    """
    exe = f"{name}.exe" if sys.platform == "win32" else name
    for line in log_text.splitlines():
        match = re.search(r"(\S*[/\\]llvm[/\\]bin)[/\\]clang", line)
        if match:
            candidate = Path(match.group(1).replace("\\", "/")) / exe
            if candidate.is_file():
                return candidate
    for candidate in sorted(build_dir.glob(f"**/llvm/bin/{exe}")):
        return candidate
    found = shutil.which(name)
    return Path(found) if found else None


def library_symbols(lib: Path, nm: Path) -> tuple[set[str], str]:
    """Returns (symbol names, status) for one library.

    Import libraries list their exports as ordinary symbols, so a plain nm
    listing answers the question that matters here: does this library offer
    the symbol the linker could not find?
    """
    try:
        proc = subprocess.run(
            [str(nm), "--no-sort", str(lib)],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return set(), f"nm failed: {exc.__class__.__name__}"
    names: set[str] = set()
    for line in proc.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        # "<addr> <type> <name>" or "<type> <name>" for undefined entries.
        names.add(parts[-1])
    if not names and proc.returncode != 0:
        return set(), f"nm exit {proc.returncode}"
    return names, "ok"


def describe_file(path: Path) -> str:
    """Returns size/mtime/digest for a file, for telling builds apart."""
    try:
        stat = path.stat()
    except OSError as exc:
        return f"unstattable ({exc.__class__.__name__})"
    digest = "?"
    try:
        with path.open("rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()[:16]
    except OSError:
        pass
    return f"size={stat.st_size} mtime={int(stat.st_mtime)} sha256:{digest}"


def resolve_libraries(dirs: list[str], libs: list[str], build_dir: Path) -> list[Path]:
    """Maps library tokens from the link line onto files on disk."""
    resolved: dict[Path, None] = {}
    for token in libs:
        candidate = Path(token)
        if candidate.is_file():
            resolved.setdefault(candidate.resolve(), None)
            continue
        name = candidate.name
        for directory in dirs:
            probe = Path(directory) / name
            if probe.is_file():
                resolved.setdefault(probe.resolve(), None)
                break
        else:
            # A bare name with no matching -L dir: look inside the build tree
            # rather than dropping it, since that is where staged libs live.
            for probe in sorted(build_dir.rglob(name))[:1]:
                resolved.setdefault(probe.resolve(), None)
    return list(resolved)


def audit_libraries(build_dir: Path, report: list[str]) -> int:
    """Checks whether the libraries on the failed link line export the symbols.

    Returns the number of undefined symbols that no library provided. A
    non-zero count means the fault is in a build product rather than in a
    momentary failure to read a correct one.
    """
    logs = sorted((build_dir / "logs").glob("*_build.log"))
    text = ""
    for log in logs:
        try:
            candidate = log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "undefined symbol:" in candidate:
            text += candidate

    report.append("")
    report.append("=== library audit ===")
    if not text:
        report.append("no build log mentions an undefined symbol; skipped")
        return 0

    symbols = parse_undefined_symbols(text)
    dirs, lib_tokens = parse_link_libraries(text)
    libraries = resolve_libraries(dirs, lib_tokens, build_dir)
    report.append(f"undefined symbols reported: {len(symbols)}")
    report.append(f"libraries on the failed link line: {len(libraries)}")
    if not symbols:
        return 0
    if not libraries:
        report.append("could not resolve any library from the link line")
        return 0

    nm = find_llvm_tool("llvm-nm", build_dir, text)
    if nm is None:
        report.append("llvm-nm not found; cannot check exports")
        return 0
    report.append(f"using {nm}")

    exports: dict[Path, set[str]] = {}
    for lib in libraries:
        names, status = library_symbols(lib, nm)
        exports[lib] = names
        report.append(f"  {lib}  {describe_file(lib)}  symbols={len(names)} {status}")

    missing = 0
    report.append("")
    for symbol in symbols:
        providers = [lib.name for lib, names in exports.items() if symbol in names]
        if providers:
            report.append(f"  PRESENT  {symbol}  <- {', '.join(providers[:3])}")
        else:
            missing += 1
            report.append(f"  ABSENT   {symbol}  (no scanned library exports it)")

    report.append("")
    report.append(f"SYMBOLS NOT EXPORTED BY ANY SCANNED LIBRARY: {missing}")
    if missing:
        report.append(
            "MODE B: a library on the link line is missing symbols it should "
            "export -> a build product is wrong on disk, not merely misread"
        )
    return missing


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


def discover_alias_roots(build_dir: Path) -> list[Path]:
    """Finds other paths that expose the same build tree as build_dir.

    The Windows CI containers mount the build volume more than once (B:\\build,
    S:\\, and C:\\<GUID>\\build have all been observed live in one pod). Each
    alias is confirmed by writing a file through build_dir and reading it back
    through the candidate, so a directory that merely looks similar is not
    mistaken for the same storage.
    """
    marker = build_dir / f".alias_probe_{os.getpid()}"
    try:
        marker.write_bytes(b"alias-probe")
    except OSError:
        return []

    candidates: list[Path] = []
    try:
        for drive in "SBDEFGT":
            root = Path(f"{drive}:/")
            if not root.exists() or root == Path(f"{build_dir.drive}/"):
                continue
            candidates.append(root / build_dir.name)
            candidates.append(root)
        for entry in Path("C:/").glob("*-*-*"):
            if entry.is_dir():
                candidates.append(entry / build_dir.name)

        confirmed: list[Path] = []
        for candidate in candidates:
            try:
                probe = candidate / marker.name
                if probe.is_file() and probe.read_bytes() == b"alias-probe":
                    confirmed.append(candidate)
            except OSError:
                continue
        return confirmed
    finally:
        try:
            marker.unlink()
        except OSError:
            pass


def digest(path: Path) -> tuple[str, int] | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return hashlib.sha256(data).hexdigest()[:16], len(data)


def parse_link_objects(text: str, build_dir: Path) -> list[Path]:
    """Returns the object files named on failed link command lines.

    CMake only writes a response file when the command would otherwise exceed
    the command-line length limit, so short links pass their objects inline and
    leave no .rsp behind. Scanning only .rsp files therefore skips exactly the
    links that produced no file, which reads as "nothing to check" rather than
    as a gap. The failed command line is the one source that is present either
    way.
    """
    found: dict[Path, None] = {}
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if "FAILED:" not in line:
            continue
        for command in lines[index : index + 3]:
            if ".obj" not in command and ".o " not in command:
                continue
            for token in command.replace('"', " ").split():
                bare = token.strip(",").lstrip("@")
                if not bare.lower().endswith((".obj", ".o")):
                    continue
                path = Path(bare)
                if not path.is_absolute():
                    path = build_dir / bare
                found.setdefault(path, None)
    return list(found)


def compare_across_aliases(
    build_dir: Path, objects: list[Path], report: list[str]
) -> int:
    """Reads the same objects through every mount of the build volume.

    This is the decisive test for the mount-aliasing theory. If one alias
    returns different bytes than another for a file nothing is writing, the
    aliases are not coherent and that is the source of the bad reads. If every
    alias agrees, the divergence was momentary and had already healed, which
    points at caching during the link rather than at the mounts themselves.
    """
    report.append("")
    report.append("=== cross-alias comparison ===")

    roots = discover_alias_roots(build_dir)
    if not roots:
        report.append("no second mount of the build tree found; skipped")
        return 0
    for root in roots:
        report.append(f"alias: {root}")

    if not objects:
        # Saying "no divergence" here would report a clean result for a check
        # that never ran.
        report.append("NO OBJECTS TO COMPARE -> this check did not run; not a result")
        return 0

    divergent = 0
    checked = 0
    missing = 0
    for obj in objects[:400]:
        try:
            rel = obj.relative_to(build_dir)
        except ValueError:
            continue
        primary = digest(obj)
        if primary is None:
            missing += 1
            continue
        checked += 1
        for root in roots:
            other = digest(root / rel)
            if other is None:
                report.append(f"  UNREADABLE via {root}: {rel}")
                divergent += 1
            elif other != primary:
                divergent += 1
                report.append(
                    f"  DIVERGENT {rel}\n"
                    f"      {build_dir}: sha={primary[0]} size={primary[1]}\n"
                    f"      {root}: sha={other[0]} size={other[1]}"
                )

    report.append(
        f"compared {checked} object(s) across {len(roots)} alias(es)"
        + (f"; {missing} not readable via the primary path" if missing else "")
    )
    if not checked:
        report.append("NO OBJECTS TO COMPARE -> this check did not run; not a result")
        return 0
    if divergent:
        report.append(
            f"ALIAS DIVERGENCE: {divergent} -> the mounts are NOT coherent; "
            "this is the source of the bad reads"
        )
    else:
        report.append(
            "no divergence now -> if the link still failed, the bad read was "
            "momentary and has already healed"
        )
    return divergent


def audit(build_dir: Path, log_dir: Path) -> int:
    rsp_files = sorted(build_dir.rglob("CMakeFiles/*.rsp"))
    rsp_copy_dir = log_dir / "rsp"
    rsp_copy_dir.mkdir(parents=True, exist_ok=True)

    report: list[str] = [
        f"scanned {len(rsp_files)} response files under {build_dir}",
        "",
    ]
    suspect_total = 0
    all_objects: list[Path] = []

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
            all_objects.append(path)
            try:
                size = path.stat().st_size
            except OSError as exc:
                suspects.append(f"MISSING  {entry}  ({exc.__class__.__name__})")
                continue
            if size == 0:
                suspects.append(f"EMPTY    {entry}")
                continue
            defect = classify_object(path, size)
            if defect:
                suspects.append(
                    f"{defect.split(' ', 1)[0]:<8} {entry}  {defect.split(' ', 1)[1] if ' ' in defect else ''}"
                )
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

    # Short links pass their objects inline and leave no .rsp behind, so the
    # failed command line is the only record of what they consumed. Without
    # this the checks below silently have nothing to look at.
    inline_objects: list[Path] = []
    for log in sorted((build_dir / "logs").glob("*_build.log")):
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "undefined symbol:" in text:
            inline_objects.extend(parse_link_objects(text, build_dir))
    if inline_objects:
        known = set(all_objects)
        added = [o for o in inline_objects if o not in known]
        all_objects.extend(added)
        report.append(
            f"recovered {len(added)} object(s) from inline link command lines"
        )

    # Validate whatever the response files did not cover.
    extra_suspects = 0
    for path in inline_objects:
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size == 0:
            report.append(f"    EMPTY    {path}")
            extra_suspects += 1
            continue
        defect = classify_object(path, size)
        if defect:
            report.append(f"    {defect}  {path}")
            extra_suspects += 1
    if extra_suspects:
        report.append(f"TOTAL SUSPECT INLINE INPUTS: {extra_suspects}")

    # Run before the retry: the retry rebuilds, which can replace the very
    # library whose contents are the evidence.
    audit_libraries(build_dir, report)
    compare_across_aliases(build_dir, all_objects, report)

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
