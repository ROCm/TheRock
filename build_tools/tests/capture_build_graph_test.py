# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import importlib.util
import hashlib
import os
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch

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
            blocked = root / "build" / "blocked-source-entry"
            blocked.touch()
            is_file = Path.is_file

            def generated_file_only(path):
                if path == blocked:
                    raise PermissionError(
                        "Unrelated source entry must not be inspected"
                    )
                return is_file(path)

            with patch.object(
                Path, "is_file", autospec=True, side_effect=generated_file_only
            ):
                capture_build_graph.capture(root / "build", output)
            with tarfile.open(output) as archive:
                self.assertIn("generated/example/build/build.ninja", archive.getnames())
                compdb = json.load(archive.extractfile("example/build/compdb.txt"))
                self.assertIn("-c hello.cpp -o hello.o", compdb[0]["command"])
                self.assertNotIn("@hello.o.rsp", compdb[0]["command"])
                manifest = json.load(archive.extractfile("capture-manifest.json"))
                self.assertEqual(manifest["ninja_tree_count"], 1)
                self.assertEqual(manifest["failures"], [])

    @unittest.skipUnless(shutil.which("ninja"), "Ninja required")
    def test_failed_command_enumeration_cannot_claim_complete_empty_responses(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary) / "build"
            build.mkdir()
            (build / "build.ninja").write_text("not a valid Ninja edge\n")
            output = build / "logs/graph.tar.gz"
            capture_build_graph.capture(build, output)
            with tarfile.open(output) as archive:
                manifest = json.load(archive.extractfile("capture-manifest.json"))
            self.assertTrue(any(r["tool"] == "commands" for r in manifest["failures"]))
            self.assertEqual(manifest["custom_response_file_count"], 0)
            self.assertFalse(manifest["custom_response_files_complete"])

    @unittest.skipUnless(shutil.which("ninja"), "Ninja required")
    def test_actual_rccl_custom_link_response_body_is_preserved_not_compdb_expanded(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = root / "build"
            tree = build / "comm-libs/rccl/build"
            response = tree / "device_build/-gfx90c/link_objects.rsp"
            response.parent.mkdir(parents=True)
            payload = (
                b"src/CMakeFiles/rccl_device_gfx900.dir/__/hipify/gensrc/device/foo.o\n"
            )
            response.write_bytes(payload)
            (tree / "build.ninja").write_text(
                f"rule link\n  command = cd {tree}/src && python rccl-device-compile --link --arch=gfx90c @{response}\nbuild output.co: link source.o\n"
            )
            output = build / "logs/graph.tar.gz"
            capture_build_graph.capture(build, output)
            with tarfile.open(output) as archive:
                manifest = json.load(archive.extractfile("capture-manifest.json"))
                self.assertTrue(manifest["custom_response_files_complete"])
                row = manifest["custom_response_files"][0]
                self.assertEqual(row["sha256"], hashlib.sha256(payload).hexdigest())
                self.assertEqual(archive.extractfile(row["member"]).read(), payload)
                compdb = json.load(
                    archive.extractfile("comm-libs/rccl/build/compdb.txt")
                )
                self.assertIn("@" + str(response), compdb[0]["command"])
            response.unlink()
            capture_build_graph.capture(build, output)
            with tarfile.open(output) as archive:
                manifest = json.load(archive.extractfile("capture-manifest.json"))
                self.assertFalse(manifest["custom_response_files_complete"])
                self.assertEqual(
                    manifest["custom_response_files"][0]["status"],
                    "unavailable-or-rejected",
                )
                self.assertFalse(
                    any(
                        name.startswith("custom-response-files/")
                        for name in archive.getnames()
                    )
                )

    @unittest.skipUnless(shutil.which("ninja"), "Ninja required")
    def test_all_25_retained_executed_rccl_links_after_only_workspace_prefix_remap(
        self,
    ):
        fixture = json.loads(
            (SCRIPT.parent / "tests/fixtures/rccl_link_rsp_commands.json").read_text()
        )
        self.assertEqual(len(fixture["commands"]), 25)
        self.assertEqual(
            len({r["ninja_command_hash"] for r in fixture["commands"]}), 25
        )
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            build = workspace / "build"
            tree = build / "comm-libs/rccl/build"
            tree.mkdir(parents=True)
            expected = {}
            ninja = []
            for index, row in enumerate(fixture["commands"]):
                original = row["original_command"]
                self.assertEqual(
                    hashlib.sha256(original.encode()).hexdigest(), row["command_sha256"]
                )
                # Exactly one replacement operation preserves original argv/CD shape.
                remapped = original.replace(fixture["workspace_prefix"], str(workspace))
                response = Path(
                    row["response_path"].replace(
                        fixture["workspace_prefix"], str(workspace)
                    )
                )
                self.assertEqual(
                    capture_build_graph.custom_response_paths(remapped, tree),
                    {response},
                )
                response.parent.mkdir(parents=True, exist_ok=True)
                payload = f"inert-object-{index}.o\n".encode()
                response.write_bytes(payload)
                expected[
                    (
                        Path("custom-response-files") / response.relative_to(build)
                    ).as_posix()
                ] = payload
                ninja += [
                    f"rule original{index}",
                    f"  command = {remapped}",
                    f"build output{index}.co: original{index} input{index}.o",
                ]
            (tree / "build.ninja").write_text("\n".join(ninja) + "\n")
            output = build / "logs/graph.tar.gz"
            capture_build_graph.capture(build, output)
            with tarfile.open(output) as archive:
                manifest = json.load(archive.extractfile("capture-manifest.json"))
                self.assertTrue(manifest["custom_response_files_complete"])
                self.assertEqual(manifest["custom_response_file_count"], 25)
                self.assertEqual(
                    {r["member"] for r in manifest["custom_response_files"]},
                    set(expected),
                )
                for row in manifest["custom_response_files"]:
                    self.assertEqual(
                        archive.extractfile(row["member"]).read(),
                        expected[row["member"]],
                    )
                    self.assertEqual(
                        row["sha256"],
                        hashlib.sha256(expected[row["member"]]).hexdigest(),
                    )

    def test_custom_response_path_alias_bounds_and_mutation_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary).resolve()
            path = build / "device_build/gfx900/link_objects.rsp"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"object.o\n")
            self.assertEqual(
                capture_build_graph.read_custom_response(path, build), b"object.o\n"
            )
            with self.assertRaisesRegex(ValueError, "escapes"):
                capture_build_graph.read_custom_response(
                    build.parent / "outside/link_objects.rsp", build
                )
            with patch.object(
                capture_build_graph, "MAX_CUSTOM_RSP_BYTES", 4
            ), self.assertRaisesRegex(ValueError, "byte bound"):
                capture_build_graph.read_custom_response(path, build)
            alias = path.with_name("alias.rsp")
            os.link(path, alias)
            with self.assertRaisesRegex(ValueError, "single-link"):
                capture_build_graph.read_custom_response(path, build)
            alias.unlink()
            original_fstat = os.fstat
            calls = 0

            def modify_during_read(fd):
                nonlocal calls
                calls += 1
                if calls == 2:
                    with path.open("ab") as stream:
                        stream.write(b"changed.o\n")
                return original_fstat(fd)

            with patch.object(
                capture_build_graph.os, "fstat", side_effect=modify_during_read
            ), self.assertRaisesRegex(ValueError, "changed during capture"):
                capture_build_graph.read_custom_response(path, build)
            if os.name != "nt":
                path.unlink()
                path.symlink_to(build / "outside")
                with self.assertRaisesRegex(ValueError, "symlink"):
                    capture_build_graph.read_custom_response(path, build)


if __name__ == "__main__":
    unittest.main()
