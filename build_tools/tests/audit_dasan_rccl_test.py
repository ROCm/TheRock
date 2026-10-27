# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import contextlib
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audit_dasan_rccl as audit


def device_elf(*, arch=0x4F, machine=224, marker=True, reports=True, unresolved=False):
    """Small real ELF symbol table; no compiler or GPU needed for validation."""
    symbols = []
    if marker:
        symbols.append(("__dasan_instrumented", 0xFFF1))
    if reports:
        symbols.append(("__dasan_report_store8", 0xFFF1))
    if unresolved:
        symbols.append(("__dasan_report_load8", 0))
    strings = b"\0"
    entries = bytes(24)
    for name, index in symbols:
        offset = len(strings)
        strings += name.encode() + b"\0"
        entries += struct.pack("<IBBHQQ", offset, 0x12, 0, index, 0, 0)
    names = b"\0.symtab\0.strtab\0.shstrtab\0"
    symoff = 64
    stroff = symoff + len(entries)
    nameoff = stroff + len(strings)
    shoff = nameoff + len(names)
    header = struct.pack(
        "<16sHHIQQQIHHHHHH",
        b"\x7fELF\x02\x01\x01" + bytes(9),
        3,
        machine,
        1,
        0,
        0,
        shoff,
        arch,
        64,
        0,
        0,
        64,
        4,
        3,
    )
    sections = bytes(64)
    for name, kind, offset, size, link, entsize in (
        (1, 2, symoff, len(entries), 2, 24),
        (9, 3, stroff, len(strings), 0, 0),
        (17, 3, nameoff, len(names), 0, 0),
    ):
        sections += struct.pack(
            "<IIQQQQIIQQ", name, kind, 0, 0, offset, size, link, 0, 1, entsize
        )
    return header + entries + strings + names + sections


class RCCLDeviceAuditTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.rocm = self.root / "rocm"
        self.library = self.rocm / "lib/librccl.so"
        self.library.parent.mkdir(parents=True)
        self.library.write_bytes(b"host fixture")
        self.output = self.root / "logs"
        self.device = self.root / "device.hsaco"

    def test_real_elf_symbol_evidence_accepts_requested_arch_only(self):
        self.device.write_bytes(device_elf())
        evidence = audit.inspect_device(self.device, "gfx950")
        self.assertTrue(evidence["dasan_marker_defined"])
        self.assertEqual(
            evidence["defined_report_functions"], ["__dasan_report_store8"]
        )
        self.assertEqual(evidence["sha256"], audit.sha256(self.device))

    def test_rejects_missing_or_unresolved_instrumentation_and_wrong_machine(self):
        for options in (
            {"marker": False},
            {"reports": False},
            {"unresolved": True},
            {"arch": 0x4C},
            {"machine": 62},
        ):
            with self.subTest(options=options):
                self.device.write_bytes(device_elf(**options))
                with self.assertRaises(RuntimeError):
                    audit.inspect_device(self.device, "gfx950")

    def test_kpack_follows_host_reference_and_audits_every_requested_object(self):
        archive_path = self.rocm / ".kpack/rccl_gfx950.kpack"
        archive_path.parent.mkdir()
        archive_path.write_bytes(b"archive fixture")
        kernel = "comm-libs/rccl/stage/lib/librccl.so.1.0"
        reference = {
            "kernel_name": kernel,
            "kpack_search_paths": ["../.kpack/rccl_@GFXARCH@.kpack"],
        }
        archive = mock.Mock()
        archive.toc = {
            kernel + "#0": {"gfx950": {}},
            kernel + "#1": {"gfx950": {}},
            "unrelated.so#0": {"gfx950": {}},
        }
        archive.get_kernel.return_value = device_elf()
        archive_type = mock.Mock()
        archive_type.read.return_value = archive
        with (
            mock.patch.object(audit, "read_reference", return_value=reference),
            mock.patch.dict(
                sys.modules,
                {
                    "rocm_kpack": SimpleNamespace(),
                    "rocm_kpack.kpack": SimpleNamespace(
                        PackedKernelArchive=archive_type
                    ),
                },
            ),
        ):
            report = audit.audit(self.rocm, "gfx950", self.output)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(report["device_objects"]), 2)
        self.assertEqual(
            archive.get_kernel.call_args_list,
            [mock.call(kernel + "#0", "gfx950"), mock.call(kernel + "#1", "gfx950")],
        )
        self.assertEqual(
            report["device_objects"][0]["archive_sha256"], audit.sha256(archive_path)
        )

    def test_missing_referenced_archive_fails_and_writes_report(self):
        reference = {
            "kernel_name": "lib/librccl.so#0",
            "kpack_search_paths": ["../.kpack/missing_@GFXARCH@.kpack"],
        }
        with (
            mock.patch.object(audit, "read_reference", return_value=reference),
            mock.patch.dict(
                sys.modules,
                {
                    "rocm_kpack": SimpleNamespace(),
                    "rocm_kpack.kpack": SimpleNamespace(
                        PackedKernelArchive=mock.Mock()
                    ),
                },
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "No packaged RCCL"):
                audit.audit(self.rocm, "gfx950", self.output)
        self.assertEqual(
            json.loads((self.output / "audit.json").read_text())["status"], "failed"
        )

    def test_inline_payload_requires_no_kpack_archive(self):
        self.device.write_bytes(device_elf())
        unbundled = SimpleNamespace(
            dest_dir=self.root,
            target_list=[("hip-amdgcn-amd-amdhsa--gfx950", self.device.name)],
        )
        binary = mock.Mock()
        binary.return_value.unbundle.return_value = contextlib.nullcontext(unbundled)
        with (
            mock.patch.object(audit, "read_reference", return_value=None),
            mock.patch.dict(
                sys.modules,
                {
                    "rocm_kpack": SimpleNamespace(),
                    "rocm_kpack.binutils": SimpleNamespace(BundledBinary=binary),
                },
            ),
        ):
            report = audit.audit(self.rocm, "gfx950", self.output)
        binary.return_value.unbundle.assert_called_once_with(gfx_arch="gfx950")
        self.assertEqual(report["device_objects"][0]["source"], "inline")
        self.assertFalse((self.rocm / ".kpack").exists())

    def test_empty_inline_extraction_is_not_success(self):
        unbundled = SimpleNamespace(dest_dir=self.root, target_list=[])
        binary = mock.Mock()
        binary.return_value.unbundle.return_value = contextlib.nullcontext(unbundled)
        with (
            mock.patch.object(audit, "read_reference", return_value=None),
            mock.patch.dict(
                sys.modules,
                {
                    "rocm_kpack": SimpleNamespace(),
                    "rocm_kpack.binutils": SimpleNamespace(BundledBinary=binary),
                },
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "No inline RCCL"):
                audit.audit(self.rocm, "gfx950", self.output)


if __name__ == "__main__":
    unittest.main()
