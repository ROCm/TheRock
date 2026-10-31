# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "profile_rocshmem_links.py"
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("rocshmem_replay", SCRIPT)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
KPACK = importlib.util.find_spec("rocm_kpack") is not None
LINUX = sys.platform == "linux" and importlib.util.find_spec("psutil") is not None


class LinkReplayTest(unittest.TestCase):
    def args(self, target):
        return [
            "/compiler/bin/clang++",
            "-O3",
            "object.o",
            "-o",
            target,
            "librocshmem.a",
            "--hip-link",
            *["--offload-arch=" + a for a in replay.ARCHITECTURES],
            "-fgpu-rdc",
        ]

    def test_generated_edge_validation_retains_every_full_architecture_and_native_output(
        self,
    ):
        args = self.args(replay.TARGETS[1])
        self.assertEqual(
            replay.driver_arguments(
                ": && " + shlex.join(args) + " && :", replay.TARGETS[1]
            ),
            args,
        )
        for bad in [
            args + ["-v"],
            [a for a in args if a != "--offload-arch=gfx1250-strict"],
            args + ["&&", "echo", "unexpected"],
            args + ["-c"],
            args + ["source.cpp"],
            args + ["@unexpanded-link.rsp"],
        ]:
            with self.assertRaises(RuntimeError):
                replay.driver_arguments(shlex.join(bad), replay.TARGETS[1])

    def test_candidate_can_change_only_width_two_and_no_input_or_architecture(self):
        original = self.args(replay.TARGETS[1])
        replay.validate_candidate(original, [*original, "--offload-jobs=2"])
        for candidate in [
            [*original, "--offload-jobs=4"],
            [*original, "--offload-jobs=2", "--offload-jobs=2"],
            [*original[:-1], "--offload-jobs=2"],
            [a.replace("librocshmem.a", "other.a") for a in original]
            + ["--offload-jobs=2"],
        ]:
            with self.assertRaises(RuntimeError):
                replay.validate_candidate(original, candidate)

    def test_input_manifest_detects_same_size_same_mtime_byte_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.o"
            path.write_bytes(b"original")
            previous_time = path.stat().st_mtime_ns
            before = replay.input_manifest([path], time.monotonic() + 30)
            path.write_bytes(b"modified")
            os.utime(path, ns=(previous_time, previous_time))
            self.assertNotEqual(
                before, replay.input_manifest([path], time.monotonic() + 30)
            )

    def test_controller_rejects_other_repository_and_mismatched_checkout(self):
        for actions, repository in [
            ("false", "ROCm/TheRock"),
            ("true", "ROCm/rockrel"),
        ]:
            with patch.dict(
                os.environ, dict(GITHUB_ACTIONS=actions, GITHUB_REPOSITORY=repository)
            ):
                with self.assertRaisesRegex(RuntimeError, "only in ROCm/TheRock"):
                    replay.require_rock_ci()
        with patch.dict(
            os.environ,
            dict(
                GITHUB_ACTIONS="true",
                GITHUB_REPOSITORY="ROCm/TheRock",
                GITHUB_SHA="expected",
            ),
        ):
            with patch.object(subprocess, "check_output", return_value="other\n"):
                with self.assertRaisesRegex(RuntimeError, "Checkout"):
                    replay.require_rock_ci()

    @unittest.skipUnless(LINUX, "requires POSIX process groups and psutil")
    def test_concurrent_worker_releases_both_native_commands_and_retains_profiles(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            worker = base / "fixture.py"
            worker.write_text(
                """import pathlib,sys,time
mine,other,output=map(pathlib.Path,sys.argv[1:])
mine.write_text('started')
deadline=time.monotonic()+4
while not other.exists():
 if time.monotonic()>deadline: raise RuntimeError('other native command was not concurrent')
 time.sleep(.01)
output.write_bytes(b'unit fixture output')
"""
            )
            commands = {
                name: [
                    sys.executable,
                    str(worker),
                    str(base / name),
                    str(base / other),
                    str(base / (name + ".bin")),
                ]
                for name, other in [("info", "functional"), ("functional", "info")]
            }
            payload = base / "commands.json"
            payload.write_text(json.dumps(dict(cwd=str(base), commands=commands)))
            replay.pair_worker(payload, base / "output", 10)
            pair = json.loads((base / "output/pair.json").read_text())
            self.assertEqual(set(pair["links"]), {"info", "functional"})
            for name in commands:
                self.assertEqual(
                    (base / (name + ".bin")).read_bytes(), b"unit fixture output"
                )
                self.assertEqual(pair["links"][name]["exit_code"], 0)
                self.assertIsNotNone(pair["links"][name]["posix_waited_child_cpu"])

    @unittest.skipUnless(LINUX, "requires POSIX process groups and psutil")
    def test_one_native_failure_stops_other_process_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            commands = dict(
                info=[sys.executable, "-c", "raise SystemExit(7)"],
                functional=[sys.executable, "-c", "import time; time.sleep(30)"],
            )
            payload = base / "commands.json"
            payload.write_text(json.dumps(dict(cwd=str(base), commands=commands)))
            started = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "failed"):
                replay.pair_worker(payload, base / "output", 10)
            self.assertLess(time.monotonic() - started, 8)

    @unittest.skipUnless(
        LINUX and KPACK and all(shutil.which(t) for t in ["cc", "readelf", "objcopy"]),
        "requires Linux ELF tools and the pinned kpack parser fixture",
    )
    def test_native_binary_retention_and_per_image_hashes_use_existing_parser(self):
        from _therock_utils.archive_util import open_archive_for_read

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "fixture.c"
            source.write_text("int main(void) { return 0; }\n")
            native = base / "host-fixture"
            subprocess.run(["cc", str(source), "-o", str(native)], check=True)
            ids = ["hip-amdgcn-amd-amdhsa--" + arch for arch in replay.ARCHITECTURES]
            payloads = [
                ("synthetic code image " + arch).encode()
                for arch in replay.ARCHITECTURES
            ]
            magic = b"__CLANG_OFFLOAD_BUNDLE__"
            offset = len(magic) + 8 + sum(24 + len(i) for i in ids)
            header = bytearray(magic + struct.pack("<Q", len(ids)))
            for target_id, payload in zip(ids, payloads):
                header.extend(struct.pack("<QQQ", offset, len(payload), len(target_id)))
                header.extend(target_id.encode())
                offset += len(payload)
            bundle = base / "fixture.bundle"
            bundle.write_bytes(header + b"".join(payloads))
            tree = base / "build"
            for target in replay.TARGETS:
                path = tree / target
                path.parent.mkdir(parents=True)
                subprocess.run(
                    [
                        "objcopy",
                        "--add-section",
                        ".hip_fatbin=" + str(bundle),
                        str(native),
                        str(path),
                    ],
                    check=True,
                )
            output = base / "retained"
            output.mkdir()
            retained = replay.retain_outputs(tree, output, time.monotonic() + 30)
            for target in replay.TARGETS:
                images = {
                    image["architecture"]: image for image in retained[target]["images"]
                }
                self.assertEqual(set(images), set(replay.ARCHITECTURES))
                self.assertIn("gfx1250-strict", images)
                for arch, payload in zip(replay.ARCHITECTURES, payloads):
                    self.assertEqual(
                        images[arch]["sha256"], hashlib.sha256(payload).hexdigest()
                    )
            with open_archive_for_read(output / "linked-binaries.tar.zst") as archive:
                for target in replay.TARGETS:
                    self.assertEqual(
                        hashlib.sha256(archive.extractfile(target).read()).hexdigest(),
                        retained[target]["sha256"],
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
