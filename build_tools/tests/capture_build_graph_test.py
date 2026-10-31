# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "capture_build_graph.py"
spec = importlib.util.spec_from_file_location("capture_build_graph", SCRIPT)
capture_build_graph = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture_build_graph)


class CaptureBuildGraphTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ninja"), "Ninja required")
    def test_expands_response_files_and_preserves_generated_rules(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tree = root / "build" / "example" / "build"
            tree.mkdir(parents=True)
            (tree / "build.ninja").write_text(
                "rule cc\n  command = clang @$out.rsp\n  rspfile = $out.rsp\n"
                "  rspfile_content = -c $in -o $out\n"
                "build hello.o: cc hello.cpp\n",
                encoding="utf-8",
            )
            output = root / "build" / "logs" / "graph.tar.gz"
            capture_build_graph.capture(root / "build", output)
            with tarfile.open(output) as archive:
                self.assertIn("generated/example/build/build.ninja", archive.getnames())
                compdb = json.load(archive.extractfile("example/build/compdb.txt"))
                self.assertIn("-c hello.cpp -o hello.o", compdb[0]["command"])
                self.assertNotIn("@hello.o.rsp", compdb[0]["command"])
                manifest = json.load(archive.extractfile("capture-manifest.json"))
                self.assertEqual(manifest["ninja_tree_count"], 1)
                self.assertEqual(manifest["failures"], [])


if __name__ == "__main__":
    unittest.main()
