# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "packaging/linux/benchmark_package_preprocess.py"
)
spec = importlib.util.spec_from_file_location("package_preprocess", SCRIPT)
preprocess = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocess)


class PreprocessTest(unittest.TestCase):
    @unittest.skipUnless(
        sys.platform == "linux"
        and all(shutil.which(t) for t in ["cc", "readelf", "patchelf"]),
        "requires real Linux ELF tools",
    )
    def test_swallowed_patchelf_failure_rejects_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "work"
            root.mkdir()
            (base / "fixture.c").write_text("int fixture(void) { return 42; }\n")
            subprocess.run(
                [
                    "cc",
                    "-shared",
                    "-fPIC",
                    str(base / "fixture.c"),
                    "-Wl,-rpath,$ORIGIN",
                    "-o",
                    str(root / "fixture.so"),
                ],
                check=True,
            )
            (root / "rocm.cfg").write_text("--enable-new-dtags\n")
            sources = preprocess.prepare_sources(base)
            fakebin = base / "fakebin"
            fakebin.mkdir()
            fake = fakebin / "patchelf"
            fake.write_text("#!/bin/sh\nexit 7\n")
            fake.chmod(0o755)
            with patch.dict(
                os.environ, {"PATH": str(fakebin) + os.pathsep + os.environ["PATH"]}
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "patchelf/configuration errors"
                ):
                    preprocess.run_conversion(sources["A"], root, base / "pass", 30)
            self.assertTrue((base / "pass/conversion-errors.json").exists())

    def test_generated_sources_differ_only_in_the_bounded_probe(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = preprocess.prepare_sources(Path(temporary))
            original, candidate = (paths[k].read_text() for k in "AB")
            self.assertIn("filepath.read_bytes()[:4]", original)
            self.assertNotIn("filepath.read_bytes()[:4]", candidate)
            self.assertEqual(candidate.count("magic = stream.read(4)"), 1)
            self.assertEqual(
                original.split("            rpath =")[1],
                candidate.split("            rpath =")[1],
            )

    def test_manifest_checks_content_and_modes_but_separates_mtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "artifacts"
            root.mkdir()
            path = root / "text"
            path.write_text("abc")
            first = preprocess.manifest(
                root, base / "first.jsonl", time.monotonic() + 30
            )
            os.utime(
                path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1000000000)
            )
            second = preprocess.manifest(
                root, base / "second.jsonl", time.monotonic() + 30
            )
            self.assertEqual(first["content_sha256"], second["content_sha256"])
            self.assertNotEqual(first["mtime_sha256"], second["mtime_sha256"])
            path.write_text("def")
            third = preprocess.manifest(
                root, base / "third.jsonl", time.monotonic() + 30
            )
            self.assertNotEqual(first["content_sha256"], third["content_sha256"])

    @unittest.skipUnless(sys.platform == "linux", "Linux copy/link safety checks")
    def test_copies_are_independent_even_if_pristine_has_hardlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            pristine, work = base / "artifacts", base / "preprocess-work"
            pristine.mkdir()
            (pristine / "a").write_text("original")
            os.link(pristine / "a", pristine / "b")
            preprocess.copy_tree(pristine, work, 16, time.monotonic() + 30)
            self.assertEqual((work / "a").stat().st_nlink, 1)
            self.assertEqual((work / "b").stat().st_nlink, 1)
            (work / "a").write_text("changed")
            self.assertEqual((pristine / "a").read_text(), "original")
            self.assertEqual((work / "b").read_text(), "original")

    @unittest.skipUnless(sys.platform == "linux", "Linux link/deletion gates")
    def test_escaping_links_cycles_and_unowned_reset_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            pristine, work = base / "artifacts", base / "preprocess-work"
            pristine.mkdir()
            (pristine / "escape").symlink_to(base)
            with self.assertRaisesRegex(RuntimeError, "escapes"):
                preprocess.check_links(pristine)
            (pristine / "escape").unlink()
            (base / "external-file").write_text("must remain unchanged")
            (pristine / "safe-file-link").symlink_to(base / "external-file")
            preprocess.check_links(pristine)
            (pristine / "rocm.cfg").symlink_to(base / "external-file")
            with self.assertRaisesRegex(RuntimeError, "escapes"):
                preprocess.check_links(pristine)
            (pristine / "rocm.cfg").unlink()
            (pristine / "cycle").symlink_to(pristine)
            with self.assertRaisesRegex(RuntimeError, "cycle"):
                preprocess.check_links(pristine)
            (pristine / "cycle").unlink()
            work.mkdir()
            with self.assertRaisesRegex(RuntimeError, "not created"):
                preprocess.copy_tree(pristine, work, 0, time.monotonic() + 30)

    def test_disk_budget_fails_before_copying(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            pristine = base / "artifacts"
            pristine.mkdir()
            usage = type("Usage", (), {"free": 0})()
            with patch.object(shutil, "disk_usage", return_value=usage):
                with self.assertRaisesRegex(RuntimeError, "Insufficient free disk"):
                    preprocess.copy_tree(
                        pristine, base / "preprocess-work", 100, time.monotonic() + 30
                    )
            self.assertFalse((base / "preprocess-work").exists())

    @unittest.skipUnless(
        sys.platform == "linux"
        and all(shutil.which(t) for t in ["cc", "readelf", "patchelf"]),
        "requires real Linux ELF tools",
    )
    def test_actual_converter_abba_has_identical_outputs_and_pristine(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            pristine = base / "artifacts"
            (pristine / "lib").mkdir(parents=True)
            (base / "fixture.c").write_text("int fixture(void) { return 42; }\n")
            subprocess.run(
                [
                    "cc",
                    "-shared",
                    "-fPIC",
                    str(base / "fixture.c"),
                    "-Wl,-rpath,$ORIGIN",
                    "-o",
                    str(pristine / "lib/fixture.so"),
                ],
                check=True,
            )
            (pristine / "lib/link.so").symlink_to("fixture.so")
            (pristine / "alias").symlink_to("lib", target_is_directory=True)
            (pristine / "rocm.cfg").write_text("--enable-new-dtags\n")
            (pristine / "not-elf").write_text("text")
            original_tags = subprocess.check_output(
                ["readelf", "-d", str(pristine / "lib/fixture.so")], text=True
            )
            self.assertIn("(RUNPATH)", original_tags)
            output = base / "diagnostics"
            run = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--root",
                    str(pristine),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            summary = json.loads((output / "summary.json").read_text())
            self.assertTrue(summary["validated"])
            self.assertTrue(summary["pristine_unchanged"])
            self.assertEqual([r["method"] for r in summary["passes"]], list("ABBA"))
            self.assertTrue(
                all(
                    r["conversion"]["converted_log_lines"] == 2
                    for r in summary["passes"]
                )
            )
            self.assertEqual(
                (pristine / "rocm.cfg").read_text(), "--enable-new-dtags\n"
            )
            self.assertEqual(
                subprocess.check_output(
                    ["readelf", "-d", str(pristine / "lib/fixture.so")], text=True
                ),
                original_tags,
            )
            self.assertEqual(
                (base / "preprocess-work/rocm.cfg").read_text(), "--disable-new-dtags\n"
            )
            tags = subprocess.check_output(
                ["readelf", "-d", str(base / "preprocess-work/lib/fixture.so")],
                text=True,
            )
            self.assertIn("(RPATH)", tags)
            self.assertNotIn("(RUNPATH)", tags)


if __name__ == "__main__":
    unittest.main(verbosity=2)
