# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import struct
import tempfile
import unittest
from pathlib import Path

import msgpack
import pyzstd

from _therock_utils.kpack_archive import (
    SCHEME_NONE,
    SCHEME_ZSTD,
    read_kpack,
    replace_entries,
)

ROCRAND = "math-libs/rocRAND/stage/lib/librocrand.so.1.1"
HIPRAND = "math-libs/hipRAND/stage/lib/libhiprand.so.1.1"


def write_reference_kpack(path: Path, kernels: dict, scheme: str = SCHEME_ZSTD):
    """Writes {(key, arch): code object} the way rocm_kpack.kpack does.

    Independent of the module under test, so reading is checked against the
    real layout rather than against its own writer.
    """
    toc = {}
    stored = []
    for ordinal, ((key, arch), data) in enumerate(kernels.items()):
        toc.setdefault(key, {})[arch] = {
            "type": "hsaco",
            "ordinal": ordinal,
            "original_size": len(data),
        }
        stored.append(pyzstd.compress(data) if scheme == SCHEME_ZSTD else data)

    blob_start = 64
    if scheme == SCHEME_ZSTD:
        blob = struct.pack("<I", len(stored)) + b"".join(
            struct.pack("<I", len(s)) + s for s in stored
        )
        layout = {"zstd_offset": blob_start, "zstd_size": len(blob)}
    else:
        blob = b"".join(stored)
        blobs, offset = [], blob_start
        for s in stored:
            blobs.append({"offset": offset, "size": len(s)})
            offset += len(s)
        layout = {"blobs": blobs}

    with open(path, "wb") as f:
        f.write(struct.pack("<4sIQ", b"KPAK", 1, 0))
        f.write(b"\0" * (blob_start - f.tell()))
        f.write(blob)
        toc_offset = f.tell()
        msgpack.pack(
            {
                "format_version": 1,
                "group_name": "rand_lib",
                "gfx_arch_family": "gfx942",
                "gfx_arches": ["gfx942"],
                "toc": toc,
                "compression_scheme": scheme,
                **layout,
            },
            f,
            use_bin_type=True,
        )
        f.seek(8)
        f.write(struct.pack("<Q", toc_offset))


def read_kernels(path: Path) -> dict:
    archive = read_kpack(path)
    with open(path, "rb") as f:
        return {
            (key, arch): archive.code_object(f, entry)
            for key, arch, entry in archive.entries()
        }


class KpackArchiveTest(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.root = Path(self._temp_dir.name)

    def test_reads_kernels_in_both_storage_schemes(self):
        kernels = {
            (f"{ROCRAND}#0", "gfx942"): b"roc0",
            (f"{ROCRAND}#1", "gfx942"): b"roc1" * 100,
            (f"{HIPRAND}#0", "gfx942"): b"hip0",
        }
        for scheme in (SCHEME_ZSTD, SCHEME_NONE):
            with self.subTest(scheme=scheme):
                path = self.root / f"{scheme}.kpack"
                write_reference_kpack(path, kernels, scheme)

                self.assertEqual(read_kpack(path).scheme, scheme)
                self.assertEqual(read_kernels(path), kernels)

    def test_a_file_that_is_not_a_kpack_is_rejected(self):
        path = self.root / "bogus.kpack"
        path.write_bytes(b"\x7fELF" + b"\0" * 60)

        with self.assertRaisesRegex(ValueError, "not a kpack archive"):
            read_kpack(path)

    def test_owned_keys_come_from_the_replacement_and_the_rest_stay(self):
        baseline = self.root / "baseline.kpack"
        replacement = self.root / "instrumented.kpack"
        write_reference_kpack(
            baseline,
            {
                (f"{ROCRAND}#0", "gfx942"): b"base-roc0",
                (f"{HIPRAND}#0", "gfx942"): b"base-hip0",
            },
        )
        write_reference_kpack(
            replacement,
            {
                (f"{ROCRAND}#0", "gfx942"): b"inst-roc0",
                (f"{ROCRAND}#1", "gfx942"): b"inst-roc1",
                (f"{HIPRAND}#0", "gfx942"): b"inst-hip0",
            },
        )
        output = self.root / "merged.kpack"

        swapped = replace_entries(
            read_kpack(baseline),
            read_kpack(replacement),
            lambda key: "/rocRAND/" in key,
            output,
        )

        self.assertEqual(swapped, 2)
        self.assertEqual(
            read_kernels(output),
            {
                (f"{HIPRAND}#0", "gfx942"): b"base-hip0",
                (f"{ROCRAND}#0", "gfx942"): b"inst-roc0",
                (f"{ROCRAND}#1", "gfx942"): b"inst-roc1",
            },
        )
        merged = read_kpack(output)
        # The runtime indexes frames by ordinal, so they have to be dense.
        self.assertEqual(
            sorted(entry["ordinal"] for _, _, entry in merged.entries()), [0, 1, 2]
        )
        self.assertEqual(merged.toc_data["group_name"], "rand_lib")
        self.assertEqual(merged.toc_data["gfx_arches"], ["gfx942"])

    def test_owned_keys_the_replacement_lacks_are_dropped(self):
        # A translation unit the instrumented build no longer has must not keep
        # running the baseline's copy under the instrumented host library.
        baseline = self.root / "baseline.kpack"
        replacement = self.root / "instrumented.kpack"
        write_reference_kpack(
            baseline,
            {
                (f"{ROCRAND}#0", "gfx942"): b"base-roc0",
                (f"{ROCRAND}#1", "gfx942"): b"base-roc1",
            },
        )
        write_reference_kpack(replacement, {(f"{ROCRAND}#0", "gfx942"): b"inst-roc0"})

        replace_entries(
            read_kpack(baseline),
            read_kpack(replacement),
            lambda key: "/rocRAND/" in key,
            baseline,
        )

        self.assertEqual(
            read_kernels(baseline), {(f"{ROCRAND}#0", "gfx942"): b"inst-roc0"}
        )

    def test_archives_with_different_schemes_are_not_combined(self):
        baseline = self.root / "baseline.kpack"
        replacement = self.root / "instrumented.kpack"
        write_reference_kpack(baseline, {(f"{ROCRAND}#0", "gfx942"): b"a"})
        write_reference_kpack(
            replacement, {(f"{ROCRAND}#0", "gfx942"): b"b"}, SCHEME_NONE
        )

        with self.assertRaisesRegex(ValueError, "compression schemes differ"):
            replace_entries(
                read_kpack(baseline),
                read_kpack(replacement),
                lambda key: True,
                self.root / "merged.kpack",
            )


if __name__ == "__main__":
    unittest.main()
