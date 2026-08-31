# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Turns the profraw files produced by a coverage test run into an lcov report.

Instrumented binaries emit one `.profraw` per process, so a sharded test job
produces many of them spread across downloaded artifact directories. This
merges them into a single `.profdata` index and exports lcov for the
instrumented objects.

The `llvm-profdata` and `llvm-cov` binaries must come from the same compiler
that built the instrumented objects; a version mismatch produces an
unhelpfully generic "malformed instrumentation profile data" error. Both live
under `lib/llvm/bin` of an installed ROCm distribution.
"""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

EXECUTABLE_SUFFIX = ".exe" if sys.platform == "win32" else ""


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
    """
    objects: dict[Path, Path] = {}
    for pattern in object_globs:
        matches = sorted(rocm_dir.glob(pattern))
        if not matches:
            logging.warning(
                "No files matched object glob '%s' under %s", pattern, rocm_dir
            )
        for match in matches:
            if match.is_dir():
                continue
            objects.setdefault(match.resolve(), match)
    return sorted(objects.values())


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


def export_lcov(
    llvm_cov: Path, profdata: Path, objects: list[Path], output: Path
) -> None:
    # llvm-cov takes the first object positionally and the rest via -object.
    command = [str(llvm_cov), "export", str(objects[0])]
    for obj in objects[1:]:
        command.extend(["-object", str(obj)])
    command.extend([f"-instr-profile={profdata}", "--format=lcov"])

    output.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Exporting lcov for %d object(s) to %s", len(objects), output)
    with open(output, "w") as lcov_file:
        subprocess.run(command, check=True, stdout=lcov_file)


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
        "--allow-empty",
        action="store_true",
        help="Exit successfully when no profraw files were collected",
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

    merge_profraw(llvm_profdata, profraw_files, args.profdata_output)
    export_lcov(llvm_cov, args.profdata_output, objects, args.lcov_output)
    logging.info("Wrote coverage report to %s", args.lcov_output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
