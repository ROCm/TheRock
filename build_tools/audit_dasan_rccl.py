#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify DASan evidence in the device code shipped with installed RCCL."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import msgpack
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection

ARCH_FLAGS = {"gfx942": 0x4C, "gfx950": 0x4F}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_device(path: Path, arch: str) -> dict:
    """Reject host images, wrong architectures, and missing DASan definitions."""
    with path.open("rb") as stream:
        elf = ELFFile(stream)
        flags = elf.header["e_flags"]
        if (
            elf.elfclass != 64
            or not elf.little_endian
            or elf.header["e_machine"] != "EM_AMDGPU"
            or flags & 0xFF != ARCH_FLAGS[arch]
        ):
            raise RuntimeError(f"Expected an ELF64LE {arch} device image: {path.name}")
        defined = set()
        undefined = set()
        for section in elf.iter_sections():
            if not isinstance(section, SymbolTableSection):
                continue
            for symbol in section.iter_symbols():
                name = symbol.name
                if name.startswith(("__dasan_", "__asan_")):
                    if symbol["st_shndx"] == "SHN_UNDEF":
                        undefined.add(name)
                    else:
                        defined.add(name)
    reports = sorted(n for n in defined if n.startswith("__dasan_report_"))
    marker = "__dasan_instrumented" in defined
    evidence = {
        "architecture": arch,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "elf_flags": hex(flags),
        "dasan_marker_defined": marker,
        "defined_report_functions": reports,
        "undefined_sanitizer_names": sorted(undefined),
    }
    if not marker or not reports or undefined:
        raise RuntimeError(f"Incomplete DASan device evidence: {json.dumps(evidence)}")
    return evidence


def read_reference(library: Path) -> dict | None:
    with library.open("rb") as stream:
        elf = ELFFile(stream)
        section = elf.get_section_by_name(".rocm_kpack_ref")
        return msgpack.unpackb(section.data(), raw=False) if section else None


def audit_packed(
    library: Path, rocm: Path, reference: dict, arch: str, scratch: Path
) -> list[dict]:
    from rocm_kpack.kpack import PackedKernelArchive

    kernel = reference["kernel_name"]
    paths = reference["kpack_search_paths"]
    if not kernel or not paths:
        raise RuntimeError("RCCL has an empty KPACK reference")
    found = []
    checked = set()
    for search in paths:
        archive_path = (library.parent / search.replace("@GFXARCH@", arch)).resolve()
        if not archive_path.is_relative_to(rocm):
            raise RuntimeError("RCCL KPACK reference escapes the installation")
        if archive_path in checked or not archive_path.is_file():
            continue
        checked.add(archive_path)
        archive = PackedKernelArchive.read(archive_path)
        archive_hash = sha256(archive_path)
        for name, targets in sorted(archive.toc.items()):
            if name != kernel and not (
                name.startswith(kernel + "#") and name[len(kernel) + 1 :].isdigit()
            ):
                continue
            for target in sorted(targets):
                if target.split(":", 1)[0] != arch:
                    continue
                payload = archive.get_kernel(name, target)
                if not payload:
                    raise RuntimeError(f"Empty RCCL KPACK payload: {name} {target}")
                object_path = scratch / "device.hsaco"
                object_path.write_bytes(payload)
                del payload
                evidence = inspect_device(object_path, arch)
                evidence.update(
                    source="kpack",
                    archive=str(archive_path.relative_to(rocm)),
                    archive_sha256=archive_hash,
                    kernel_name=name,
                    target=target,
                )
                found.append(evidence)
                object_path.unlink()
        if sha256(archive_path) != archive_hash:
            raise RuntimeError("RCCL KPACK archive changed during audit")
    if not found:
        raise RuntimeError(f"No packaged RCCL device objects for requested {arch}")
    return found


def audit_inline(library: Path, arch: str) -> list[dict]:
    from rocm_kpack.binutils import BundledBinary

    found = []
    # This parses regular and compressed inline HIP bundles. It requires no
    # .kpack archive and does not modify the installed library.
    with BundledBinary(library).unbundle(gfx_arch=arch) as unbundled:
        for target, filename in unbundled.target_list:
            evidence = inspect_device(unbundled.dest_dir / filename, arch)
            evidence.update(source="inline", target=target, object_name=filename)
            found.append(evidence)
    if not found:
        raise RuntimeError(f"No inline RCCL device objects for requested {arch}")
    return found


def audit(rocm_path: Path, arch: str, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"status": "failed", "requested_architecture": arch, "device_objects": []}
    try:
        rocm = rocm_path.resolve(strict=True)
        library = (rocm / "lib/librccl.so").resolve(strict=True)
        if not library.is_relative_to(rocm):
            raise RuntimeError(
                "Installed RCCL library resolves outside the installation"
            )
        library_hash = sha256(library)
        report.update(
            library=str(library.relative_to(rocm)), library_sha256=library_hash
        )
        reference = read_reference(library)
        if reference is not None:
            report["kpack_reference"] = reference
            with tempfile.TemporaryDirectory(prefix="dasan-rccl-device-") as temporary:
                report["device_objects"] = audit_packed(
                    library, rocm, reference, arch, Path(temporary)
                )
        else:
            report["device_objects"] = audit_inline(library, arch)
        if sha256(library) != library_hash:
            raise RuntimeError("Installed RCCL library changed during audit")
        report["status"] = "passed"
        return report
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        (output_dir / "audit.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocm-path", type=Path, required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--kpack-python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arch", choices=sorted(ARCH_FLAGS), default="gfx950")
    args = parser.parse_args()
    sys.path.insert(0, str(args.kpack_python.resolve()))
    os.environ["PATH"] = (
        str(args.llvm_bin.resolve()) + os.pathsep + os.environ.get("PATH", "")
    )
    report = audit(args.rocm_path, args.arch, args.output_dir)
    print(
        f"RCCL {args.arch}: verified {len(report['device_objects'])} DASan device objects"
    )


if __name__ == "__main__":
    main()
