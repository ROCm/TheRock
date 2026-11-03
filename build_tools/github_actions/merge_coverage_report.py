# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Turns the profraw files produced by a coverage test run into an lcov report.

Instrumented binaries emit one `.profraw` per process, so a sharded test job
produces many of them spread across downloaded artifact directories. This
merges them into a single `.profdata` index and exports lcov for the
instrumented objects.

Three renderings come out of the same index. lcov is what Codecov consumes, a
text table is what a reader wants when the question is just "what is the
number", and the HTML tree is the annotated line-by-line view. The last one is
the only one that reads the source files: the others report from the coverage
mappings alone.

The `llvm-profdata` and `llvm-cov` binaries must come from the same compiler
that built the instrumented objects; a version mismatch produces an
unhelpfully generic "malformed instrumentation profile data" error. Both live
under `lib/llvm/bin` of an installed ROCm distribution.

A report that measured nothing fails rather than publishing 0%. Most ways the
pipeline can break upstream of here still leave profraw files behind: an
uninstrumented library writes profiles with no counters, and a process that
crashes before its exit handler writes empty ones.

With --device-code the report covers the objects' GPU kernels as well. Their
counters arrive in separate device-side profiles, named after the GPU target,
which merge with the host ones; the code objects llvm-cov needs to map them
back to source are not in the host binaries of a kpack-split build, so they
are pulled out of the .kpack archives installed alongside.
"""

import argparse
import logging
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).resolve().parents[1]))

from _therock_utils.kpack_archive import read_kpack

logging.basicConfig(level=logging.INFO, format="%(message)s")

EXECUTABLE_SUFFIX = ".exe" if sys.platform == "win32" else ""

# The profile runtime names a device-side profile after the GPU it was read
# back from, "<target id>[.<n>].<LLVM_PROFILE_FILE name>"; the test job swaps
# the target id's colons for underscores before upload.
DEVICE_PROFILE_NAME = re.compile(r"^gfx[0-9a-z]+")
# Present in a code object only when its translation unit had instrumented
# device code. llvm-cov rejects objects without a coverage mapping.
COVERAGE_MAPPING_SECTION = "__llvm_covfun"


def find_llvm_tool(llvm_bin_dir: Path, tool_name: str) -> Path:
    """Locates an LLVM tool, preferring the ROCm distribution's own copy."""
    candidate = llvm_bin_dir / f"{tool_name}{EXECUTABLE_SUFFIX}"
    if candidate.is_file():
        return candidate

    fallback = shutil.which(tool_name)
    if fallback:
        logging.warning(
            "%s not found in %s, falling back to %s. A version mismatch with "
            "the compiler that produced the profiles may cause failures.",
            tool_name,
            llvm_bin_dir,
            fallback,
        )
        return Path(fallback)

    raise FileNotFoundError(
        f"Could not find '{tool_name}' in '{llvm_bin_dir}' or on PATH."
    )


def find_profraw_files(profraw_dir: Path) -> list[Path]:
    return sorted(profraw_dir.rglob("*.profraw"))


def resolve_objects(rocm_dir: Path, object_globs: list[str]) -> list[Path]:
    """Expands the per-project object globs into concrete instrumented files.

    Globs typically match a versioned family of symlinks (libfoo.so,
    libfoo.so.1, ...) that all resolve to one file, so results are deduplicated
    by real path to avoid handing llvm-cov the same object several times.

    A glob starting with '!' takes its matches back out. Header-only projects
    report against their test binaries, and those can share a directory with
    a sibling project's (rocPRIM's bin/test_* also matches hipCUB's tests).
    """
    excluded = {
        match
        for pattern in object_globs
        if pattern.startswith("!")
        for match in rocm_dir.glob(pattern[1:])
    }
    objects: dict[Path, Path] = {}
    for pattern in object_globs:
        if pattern.startswith("!"):
            continue
        matches = sorted(m for m in rocm_dir.glob(pattern) if m not in excluded)
        if not matches:
            logging.warning(
                "No files matched object glob '%s' under %s", pattern, rocm_dir
            )
        for match in matches:
            if match.is_dir():
                continue
            objects.setdefault(match.resolve(), match)
    return sorted(objects.values())


def count_device_profiles(profraw_files: list[Path]) -> int:
    return sum(1 for f in profraw_files if DEVICE_PROFILE_NAME.match(f.name))


def elf_section_names(image: bytes) -> set[str]:
    """Returns the section names of a 64-bit little-endian ELF image.

    Anything else, including a truncated image, yields no names.
    """
    if image[:4] != b"\x7fELF" or image[4:6] != b"\x02\x01":
        return set()
    try:
        (shoff,) = struct.unpack_from("<Q", image, 0x28)
        shentsize, shnum, shstrndx = struct.unpack_from("<HHH", image, 0x3A)

        def section(index: int) -> tuple:
            return struct.unpack_from("<IIQQQQIIQQ", image, shoff + index * shentsize)

        strtab_offset = section(shstrndx)[4]
        names = set()
        for index in range(shnum):
            start = strtab_offset + section(index)[0]
            names.add(image[start : image.index(b"\0", start)].decode())
        return names
    except (struct.error, ValueError, UnicodeDecodeError):
        return set()


def extract_device_objects(
    rocm_dir: Path, objects: list[Path], output_dir: Path
) -> list[Path]:
    """Writes out the instrumented GPU code objects of the given host objects.

    A kpack-split build keeps device code out of the host binaries, in
    archives under .kpack/ keyed "<stage prefix>/<binary>#<n>", one entry per
    translation unit and GPU target, so a binary's code objects are the
    entries whose key ends in its path under rocm_dir. Entries without a
    coverage mapping are skipped: every HIP translation unit gets a code
    object, instrumented device code or not.
    """
    rocm_root = rocm_dir.resolve()
    wanted = set()
    for obj in objects:
        try:
            wanted.add(obj.resolve().relative_to(rocm_root).as_posix())
        except ValueError:
            continue

    extracted = []
    for kpack_path in sorted(rocm_dir.glob("**/.kpack/*.kpack")):
        archive = read_kpack(kpack_path)
        with open(kpack_path, "rb") as kpack_file:
            for key, arch, entry in archive.entries():
                binary = key.rsplit("#", 1)[0]
                if not any(binary == w or binary.endswith("/" + w) for w in wanted):
                    continue
                image = archive.code_object(kpack_file, entry)
                if COVERAGE_MAPPING_SECTION not in elf_section_names(image):
                    continue
                name = re.sub(r"[^A-Za-z0-9._+-]", "_", f"{key}-{arch}") + ".co"
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / name).write_bytes(image)
                extracted.append(output_dir / name)
    return extracted


def merge_profraw(llvm_profdata: Path, profraw_files: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(llvm_profdata),
        "merge",
        "-sparse",
        "-o",
        str(output),
        *[str(f) for f in profraw_files],
    ]
    logging.info("Merging %d profraw file(s) into %s", len(profraw_files), output)
    subprocess.run(command, check=True)


def count_profiled_functions(llvm_profdata: Path, profdata: Path) -> int | None:
    """Returns how many functions the merged profile holds counters for.

    None if llvm-profdata's summary has no such line, so an unexpected format
    skips the check rather than failing a good report.
    """
    result = subprocess.run(
        [str(llvm_profdata), "show", str(profdata)],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"^Total functions: (\d+)$", result.stdout, re.MULTILINE)
    return int(match.group(1)) if match else None


def count_lcov_lines(lcov: Path) -> tuple[int, int]:
    """Returns (lines found, lines hit) summed over every file in an lcov report."""
    found = hit = 0
    with open(lcov) as lcov_file:
        for line in lcov_file:
            if line.startswith("LF:"):
                found += int(line[3:])
            elif line.startswith("LH:"):
                hit += int(line[3:])
    return found, hit


def build_cov_command(
    llvm_cov: Path,
    subcommand: str,
    profdata: Path,
    objects: list[Path],
    path_equivalence: str | None,
) -> list[str]:
    """Assembles the part of an llvm-cov command line the three outputs share."""
    # llvm-cov takes the first object positionally and the rest via -object.
    command = [str(llvm_cov), subcommand, str(objects[0])]
    for obj in objects[1:]:
        command.extend(["-object", str(obj)])
    command.append(f"-instr-profile={profdata}")
    if path_equivalence:
        command.append(f"-path-equivalence={path_equivalence}")
    return command


def export_lcov(
    llvm_cov: Path,
    profdata: Path,
    objects: list[Path],
    output: Path,
    path_equivalence: str | None = None,
) -> None:
    command = build_cov_command(llvm_cov, "export", profdata, objects, path_equivalence)
    command.append("--format=lcov")

    output.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Exporting lcov for %d object(s) to %s", len(objects), output)
    with open(output, "w") as lcov_file:
        subprocess.run(command, check=True, stdout=lcov_file)


def write_summary(
    llvm_cov: Path,
    profdata: Path,
    objects: list[Path],
    output: Path,
    path_equivalence: str | None = None,
) -> None:
    """Writes the per-file coverage table, and echoes the totals to the log.

    This reads nothing but the coverage mappings, so it is the one rendering
    that is complete even when the sources are not on the machine.
    """
    command = build_cov_command(llvm_cov, "report", profdata, objects, path_equivalence)

    output.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Writing coverage summary to %s", output)
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    output.write_text(result.stdout)

    for line in result.stdout.splitlines():
        if line.startswith("TOTAL"):
            logging.info("%s", line)


def write_html(
    llvm_cov: Path,
    profdata: Path,
    objects: list[Path],
    output_dir: Path,
    path_equivalence: str | None = None,
    project_title: str | None = None,
    demangler: Path | None = None,
) -> None:
    """Renders the browsable report into output_dir, entry point index.html.

    Unlike the other two this opens the source files, so anything not on the
    machine is reported by llvm-cov as uncovered rather than annotated. Files
    generated into the build tree are never in a source checkout, so a few
    misses here are expected and not worth failing over.
    """
    command = build_cov_command(llvm_cov, "show", profdata, objects, path_equivalence)
    command.extend(["--format=html", f"-output-dir={output_dir}"])
    if project_title:
        command.append(f"--project-title={project_title}")
    if demangler:
        command.append(f"-Xdemangler={demangler}")

    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Writing HTML report to %s", output_dir / "index.html")
    subprocess.run(command, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profraw-dir",
        type=Path,
        required=True,
        help="Directory searched recursively for .profraw files",
    )
    parser.add_argument(
        "--rocm-dir",
        type=Path,
        required=True,
        help="Directory holding the installed ROCm artifacts under test",
    )
    parser.add_argument(
        "--object-globs",
        type=str,
        required=True,
        help="Comma-separated globs, relative to --rocm-dir, matching the "
        "instrumented binaries to report on (e.g. 'lib/libhiprand.so*')",
    )
    parser.add_argument(
        "--llvm-bin-dir",
        type=Path,
        default=None,
        help="Directory holding llvm-profdata and llvm-cov "
        "(default: <rocm-dir>/lib/llvm/bin)",
    )
    parser.add_argument(
        "--profdata-output",
        type=Path,
        default=Path("coverage-report/coverage.profdata"),
        help="Path for the merged profdata index",
    )
    parser.add_argument(
        "--lcov-output",
        type=Path,
        default=Path("coverage-report/coverage.info"),
        help="Path for the generated lcov report",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Path for the plain text per-file coverage table. Needs no "
        "sources, so it is complete wherever it runs",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=None,
        help="Directory for the browsable HTML report, entry point "
        "index.html. Renders source, so the sources must be present and "
        "--path-equivalence set if they moved since the build",
    )
    parser.add_argument(
        "--path-equivalence",
        type=str,
        default=None,
        help="'<from>,<to>' remapping for source paths recorded at build "
        "time, e.g. '/__w/TheRock/TheRock,${GITHUB_WORKSPACE}'",
    )
    parser.add_argument(
        "--project-title",
        type=str,
        default=None,
        help="Heading for the HTML report's index page",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Exit successfully when no profraw files were collected, or "
        "none of them hold any counters. With --device-code, report host "
        "code alone when no device code or device-side profile was found",
    )
    parser.add_argument(
        "--device-code",
        action="store_true",
        help="Also report on the matched objects' GPU kernels: their "
        "instrumented code objects are taken from the kpack archives under "
        "--rocm-dir and handed to llvm-cov with the host objects",
    )
    parser.add_argument(
        "--device-code-dir",
        type=Path,
        default=Path("coverage-report/device-code"),
        help="Where --device-code writes the code objects it extracts",
    )
    args = parser.parse_args(argv)

    profraw_files = find_profraw_files(args.profraw_dir)
    if not profraw_files:
        message = f"No .profraw files found under {args.profraw_dir}"
        if args.allow_empty:
            logging.warning("%s, skipping report generation", message)
            return 0
        logging.error(
            "%s. The tests either never ran or the instrumented libraries were "
            "not the ones loaded at runtime.",
            message,
        )
        return 1

    llvm_bin_dir = args.llvm_bin_dir or (args.rocm_dir / "lib" / "llvm" / "bin")
    llvm_profdata = find_llvm_tool(llvm_bin_dir, "llvm-profdata")
    llvm_cov = find_llvm_tool(llvm_bin_dir, "llvm-cov")

    object_globs = [g.strip() for g in args.object_globs.split(",") if g.strip()]
    objects = resolve_objects(args.rocm_dir, object_globs)
    if not objects:
        logging.error(
            "None of the object globs (%s) matched anything under %s",
            ", ".join(object_globs),
            args.rocm_dir,
        )
        return 1

    if args.device_code:
        device_objects = extract_device_objects(
            args.rocm_dir, objects, args.device_code_dir
        )
        device_profiles = count_device_profiles(profraw_files)
        problem = None
        if not device_objects:
            problem = (
                f"No code object of the {len(objects)} object(s) in the kpack "
                f"archives under {args.rocm_dir} carries a coverage mapping. "
                "Either the kernels were built without device instrumentation, "
                "or the build was not kpack-split"
            )
        elif not device_profiles:
            problem = (
                f"None of the {len(profraw_files)} profraw file(s) is a "
                "device-side profile. The kernels never ran, the HIP runtime "
                "could not read their counters back, or the test job dropped "
                "the profiles"
            )
        if problem:
            if not args.allow_empty:
                logging.error("%s.", problem)
                return 1
            logging.warning("%s; reporting host code only.", problem)
        else:
            logging.info(
                "Reporting on %d device code object(s); %d of %d profile(s) "
                "are device-side",
                len(device_objects),
                device_profiles,
                len(profraw_files),
            )
            objects = objects + device_objects

    merge_profraw(llvm_profdata, profraw_files, args.profdata_output)
    if count_profiled_functions(llvm_profdata, args.profdata_output) == 0:
        message = f"None of the {len(profraw_files)} profraw file(s) hold any counters"
        if args.allow_empty:
            logging.warning("%s, skipping report generation", message)
            return 0
        logging.error(
            "%s. The processes that wrote them loaded no instrumented code: "
            "either the coverage flags never reached the compiler, or every "
            "process died before writing its profile.",
            message,
        )
        return 1

    try:
        export_lcov(
            llvm_cov,
            args.profdata_output,
            objects,
            args.lcov_output,
            args.path_equivalence,
        )
    except subprocess.CalledProcessError:
        logging.error(
            "llvm-cov could not export the %d object(s). If it found no "
            "coverage data, they carry no coverage mapping: the project's "
            "coverage option did not take effect in its build.",
            len(objects),
        )
        return 1
    logging.info("Wrote coverage report to %s", args.lcov_output)
    lines_found, lines_hit = count_lcov_lines(args.lcov_output)

    if args.summary_output:
        write_summary(
            llvm_cov,
            args.profdata_output,
            objects,
            args.summary_output,
            args.path_equivalence,
        )

    if args.html_output:
        # Demangling is what makes the C++ report readable; skip it rather
        # than fail if this distribution has no llvm-cxxfilt.
        demangler = llvm_bin_dir / f"llvm-cxxfilt{EXECUTABLE_SUFFIX}"
        write_html(
            llvm_cov,
            args.profdata_output,
            objects,
            args.html_output,
            args.path_equivalence,
            args.project_title,
            demangler if demangler.is_file() else None,
        )

    # Checked after the other renderings so the report still shows what was
    # measured.
    if lines_found and not lines_hit:
        logging.error(
            "None of the %d instrumented line(s) in the reported objects ran. "
            "The counters came from other binaries: the tests loaded a "
            "different copy of the project, or never reached it.",
            lines_found,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
