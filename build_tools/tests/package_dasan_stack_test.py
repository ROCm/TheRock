# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import io
import hashlib
import http.server
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import threading
from types import SimpleNamespace
import unittest
import urllib.error
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import package_dasan_stack as package


class InventoryTest(unittest.TestCase):
    def test_openmpi_components_have_real_linux_packages(self):
        data = json.loads(
            (package.REPO / "build_tools/packaging/linux/package.json").read_text()
        )
        packages = {item["Package"]: item for item in data}
        self.assertIn("amdrocm-openmpi", packages)
        self.assertIn("amdrocm-openmpi-devel", packages)
        self.assertIn(
            "amdrocm-openmpi", packages["amdrocm-openmpi-devel"]["DEBDepends"]
        )
        owners = {}
        for item in data:
            for artifact in item.get("Artifactory", []):
                if artifact.get("Artifact") == "openmpi":
                    for subdir in artifact["Artifact_Subdir"]:
                        self.assertEqual(subdir["Name"], "openmpi")
                        for component in subdir["Components"]:
                            self.assertNotIn(component, owners)
                            owners[component] = item["Package"]
        self.assertEqual(
            owners,
            {
                "lib": "amdrocm-openmpi",
                "run": "amdrocm-openmpi",
                "dev": "amdrocm-openmpi-devel",
            },
        )

    def test_only_current_run_from_therock_is_accepted(self):
        with patch.dict("os.environ", {"GITHUB_RUN_ID": "123"}):
            package.validate_run_identity("123", "ROCm/TheRock")
            with self.assertRaisesRegex(RuntimeError, "own workflow run"):
                package.validate_run_identity("122", "ROCm/TheRock")
            with self.assertRaisesRegex(RuntimeError, "ROCm/TheRock"):
                package.validate_run_identity("123", "ROCm/rockrel")
            with self.assertRaisesRegex(RuntimeError, "numeric"):
                package.validate_run_identity("latest", "ROCm/TheRock")

    def test_requires_every_name_and_rejects_another_gpu(self):
        available = [
            "amd-llvm_run_generic.tar.zst",
            "rccl_lib_gfx942.tar.zst",
            "rccl_lib_gfx950.tar.zst.sha256sum",
            "blas_lib_gfx950-dcgpu.tar.zst",
        ]
        with self.assertRaisesRegex(RuntimeError, "rccl"):
            package.check_inventory(
                available, {"amd-llvm", "rccl", "blas"}, "gfx950-dcgpu"
            )
        available.append("rccl_lib_gfx950.tar.zst")
        available.append("rccl_lib_gfx950.tar.xz")
        selected = package.check_inventory(
            available, {"amd-llvm", "rccl", "blas"}, "gfx950-dcgpu"
        )
        self.assertEqual(len(selected), 3)
        self.assertNotIn("rccl_lib_gfx942.tar.zst", selected)
        self.assertNotIn("rccl_lib_gfx950.tar.xz", selected)

    def test_full_platform_inventory_includes_wsl_and_runtime_tests(self):
        required = package.required_artifacts(package.get_topology())
        self.assertTrue(
            {"wsl-rocdxg", "core-hiptests", "rocrtst", "rocjitsu", "rccl"} <= required
        )
        topology = SimpleNamespace(
            artifacts={"linux": object(), "windows": object(), "arm": object()},
        )
        linux, windows, arm = topology.artifacts.values()
        topology.is_artifact_disabled_on_platform = (
            lambda artifact, platform: artifact is windows
        )
        topology.is_artifact_disabled_on_processor = (
            lambda artifact, processor: artifact is arm
        )
        self.assertEqual(package.required_artifacts(topology), {"linux"})

    def test_inventory_exempts_only_verified_non_archive_nodes(self):
        required = package.required_artifacts(package.get_topology())
        self.assertTrue(
            {
                "core-hipinfo",
                "fmt",
                "grpc",
                "spdlog",
                "sysdeps-libmnl",
                "sysdeps-libnl",
            }.isdisjoint(required)
        )
        # OpenMPI is declared by CMake and must not be excused simply because
        # its feature was misconfigured or the archive failed to upload.
        self.assertTrue({"openmpi", "sysdeps", "hipdnn", "miopen", "rccl"} <= required)
        complete = [f"{name}_lib_generic.tar.zst" for name in required]
        self.assertEqual(
            len(package.check_inventory(complete, required, "gfx950-dcgpu")),
            len(required),
        )
        for real_artifact in ("openmpi", "sysdeps", "hipdnn", "miopen", "rccl"):
            incomplete = [
                filename
                for filename in complete
                if not filename.startswith(real_artifact + "_")
            ]
            with self.subTest(missing=real_artifact), self.assertRaisesRegex(
                RuntimeError, real_artifact
            ):
                package.check_inventory(incomplete, required, "gfx950-dcgpu")


class DownloadTest(unittest.TestCase):
    def test_exact_selected_checksum_and_all_metadata_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archives = [
                {
                    "name": "sdk-tests-123.tar.gz",
                    "bytes": 10,
                    "unpacked_bytes": 20,
                    "sha256": "a" * 64,
                },
                {
                    "name": "sdk-runtime-123.tar.gz",
                    "bytes": 5,
                    "unpacked_bytes": 10,
                    "sha256": "b" * 64,
                },
            ]
            manifest = {
                "run_id": "123",
                "repository": "ROCm/TheRock",
                "installation_archive": archives[0]["name"],
                "runtime_archive": archives[1]["name"],
                "archives": archives,
            }
            package.write_json(root / "source-config-manifest.json", manifest)
            (root / "README.txt").write_text("instructions")
            checksums = "".join(f"{a['sha256']}  {a['name']}\n" for a in archives)
            for name in ("README.txt", "source-config-manifest.json"):
                checksums += f"{package.sha256(root / name)}  {name}\n"
            (root / "SHA256SUMS").write_text(checksums)
            with patch.dict(os.environ, {"GITHUB_RUN_ID": "123"}):
                self.assertEqual(
                    package.selected_download(root)["name"], "sdk-tests-123.tar.gz"
                )
                (root / "README.txt").unlink()
                with self.assertRaisesRegex(RuntimeError, "Metadata missing"):
                    package.selected_download(root)

    def transfer(self, mode):
        payload = b"dasan transport\n" * 200000
        observed = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                offset = int(self.headers["Range"].split("=")[1].split("-")[0])
                observed.append((offset, self.headers.get("Authorization")))
                self.send_response(200 if mode == "ignore-range" else 206)
                start = offset + 1 if mode == "bad-range" else offset
                self.send_header(
                    "Content-Range", f"bytes {start}-{len(payload)-1}/{len(payload)}"
                )
                body = payload[offset:]
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if mode == "disconnect" and len(observed) == 1:
                    self.wfile.write(body[: 1024**2])
                    self.wfile.flush()
                    self.close_connection = True
                else:
                    try:
                        self.wfile.write(
                            (b"X" + body[1:]) if mode == "corrupt" else body
                        )
                    except (BrokenPipeError, ConnectionResetError):
                        # Rejected range tests deliberately close immediately.
                        pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        selected = {
            "name": "sdk-tests-123.tar.gz",
            "bytes": len(payload),
            "unpacked_bytes": 32,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "run_id": "123",
        }
        metadata = {
            "name": selected["name"],
            "size_in_bytes": len(payload),
            "workflow_run": {"id": 123},
            "expired": False,
        }

        def api(artifact_id, token, *, location=False):
            return (
                f"http://127.0.0.1:{server.server_port}/blob" if location else metadata
            )

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            package, "github_artifact_request", side_effect=api
        ), patch.object(package.time, "sleep"):
            root = Path(tmp)
            if mode == "ignore-range":
                partial = root / (selected["name"] + ".part")
                partial.write_bytes(payload[:1])
                package.write_json(
                    partial.with_name(partial.name + ".json"),
                    {
                        "artifact_id": "42",
                        "sha256": selected["sha256"],
                        "bytes": len(payload),
                    },
                )
            if mode in ("corrupt", "ignore-range", "bad-range"):
                with self.assertRaises(RuntimeError):
                    package.download_raw_archive(
                        selected, root, "42", "secret-api-token", timeout=10, margin=0
                    )
                self.assertFalse((root / selected["name"]).exists())
            else:
                result = package.download_raw_archive(
                    selected, root, "42", "secret-api-token", timeout=10, margin=0
                )
                self.assertEqual(result.read_bytes(), payload)
                self.assertTrue((root / "SELECTED_SHA256SUMS").exists())
        self.assertTrue(all(auth is None for offset, auth in observed))
        return observed

    def test_actual_http_disconnect_resumes_without_blob_credentials(self):
        observed = self.transfer("disconnect")
        self.assertEqual([offset for offset, auth in observed], [0, 1024**2])

    def test_corrupt_payload_never_becomes_complete_sdk(self):
        self.transfer("corrupt")

    def test_ignored_resume_range_is_rejected(self):
        self.transfer("ignore-range")

    def test_inconsistent_range_is_rejected(self):
        self.transfer("bad-range")

    def test_api_redirect_is_not_followed_with_credentials(self):
        self.assertIsNone(
            package.NoRedirect().redirect_request(
                None, None, 302, "redirect", {}, "https://blob.invalid/path"
            )
        )

    def test_api_authentication_stays_on_github_redirect_request(self):
        redirect = urllib.error.HTTPError(
            "https://api.github.com/",
            302,
            "redirect",
            {"Location": "https://blob.invalid/raw?private=not-logged"},
            None,
        )
        with patch.object(package.urllib.request, "build_opener") as factory:
            factory.return_value.open.side_effect = redirect
            result = package.github_artifact_request("42", "secret", location=True)
            request = factory.return_value.open.call_args.args[0]
            self.assertTrue(request.full_url.startswith("https://api.github.com/"))
            self.assertEqual(request.get_header("Authorization"), "Bearer secret")
            self.assertIs(factory.call_args.args[0], package.NoRedirect)
            self.assertTrue(result.startswith("https://blob.invalid/"))

    def test_disk_preflight_fails_before_blob_transfer(self):
        selected = {
            "name": "sdk-tests-123.tar.gz",
            "bytes": 10,
            "unpacked_bytes": 20,
            "sha256": "a" * 64,
            "run_id": "123",
        }
        artifact = {
            "name": selected["name"],
            "size_in_bytes": 10,
            "workflow_run": {"id": 123},
        }
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            package, "github_artifact_request", return_value=artifact
        ), patch.object(
            package.shutil, "disk_usage", return_value=SimpleNamespace(free=29)
        ), patch.object(
            package.urllib.request, "urlopen"
        ) as blob:
            with self.assertRaisesRegex(RuntimeError, "Insufficient disk"):
                package.download_raw_archive(
                    selected, Path(tmp), "42", "secret", margin=0
                )
            blob.assert_not_called()


class IntegrityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_checksum_proves_downloaded_bytes_and_accepts_standard_sidecar(self):
        archive = self.root / "payload.tar.zst"
        archive.write_bytes(b"complete payload")
        sidecar = self.root / "payload.tar.zst.sha256sum"
        digest = package.sha256(archive)
        for text in (digest, f"{digest}  {archive.name}\n"):
            sidecar.write_text(text)
            self.assertEqual(package.check_digest(archive, sidecar), digest)
        archive.write_bytes(b"truncated")
        with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            package.check_digest(archive, sidecar)
        sidecar.write_text("not a digest")
        with self.assertRaisesRegex(RuntimeError, "Invalid SHA256"):
            package.check_digest(archive, sidecar)

    def symlink(self, target, path, directory=False):
        try:
            path.symlink_to(target, target_is_directory=directory)
        except (OSError, NotImplementedError):
            self.skipTest("Creating symbolic links is unavailable")

    def sdk_aliases(self):
        (self.root / "lib/llvm/amdgcn").mkdir(parents=True)
        self.symlink("lib/llvm", self.root / "llvm", True)
        self.symlink("lib/llvm/amdgcn", self.root / "amdgcn", True)

    def test_sdk_aliases_cannot_escape_or_depend_on_build_tree(self):
        self.sdk_aliases()
        self.assertEqual(package.check_portable_links(self.root), 2)
        self.symlink("../../build-tree/librccl.so", self.root / "lib/librccl.so")
        with self.assertRaisesRegex(RuntimeError, "Nonportable"):
            package.check_portable_links(self.root)

    def test_missing_alias_and_dangling_link_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Missing portable SDK alias"):
            package.check_portable_links(self.root)
        self.sdk_aliases()
        self.symlink("missing.so", self.root / "lib/libmissing.so")
        with self.assertRaisesRegex(RuntimeError, "Dangling"):
            package.check_portable_links(self.root)

    def test_real_archive_roundtrip_has_flat_layout_and_executable_mode(self):
        archive = self.root / "sdk.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            entry = tarfile.TarInfo("bin/hipcc")
            entry.mode = 0o755
            content = b"#!/bin/sh\nexit 0\n"
            entry.size = len(content)
            stream.addfile(entry, io.BytesIO(content))
        output = self.root / "extracted"
        package.extract_sdk(archive, output)
        self.assertEqual((output / "bin/hipcc").read_bytes(), content)
        if sys.platform != "win32":
            self.assertEqual((output / "bin/hipcc").stat().st_mode & 0o777, 0o755)
        with self.assertRaisesRegex(RuntimeError, "clean directory"):
            package.extract_sdk(archive, output)

    def test_archive_cannot_write_outside_installation(self):
        archive = self.root / "malformed.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            entry = tarfile.TarInfo("../outside")
            entry.size = 1
            stream.addfile(entry, io.BytesIO(b"x"))
        with self.assertRaises(tarfile.OutsideDestinationError):
            package.extract_sdk(archive, self.root / "sdk")
        self.assertFalse((self.root / "outside").exists())


if __name__ == "__main__":
    unittest.main()
