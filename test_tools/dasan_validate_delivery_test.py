# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import hashlib
import http.server
import io
import json
import os
from pathlib import Path
import struct
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import dasan_validate_delivery as delivery


def fixture(wrong_hash=False):
    payload = b"complete SDK tar content" * 10000
    name = "sdk-tests-1.tar.gz"
    manifest = {
        "run_id": "1",
        "repository": "ROCm/TheRock",
        "therock_sha": "source-sha",
        "installation_archive": name,
    }
    small = {
        "source-config-manifest.json": json.dumps(manifest).encode(),
        "README.txt": b"instructions",
    }
    digest = "0" * 64 if wrong_hash else hashlib.sha256(payload).hexdigest()
    master = f"{digest}  {name}\n" + "".join(
        f"{hashlib.sha256(data).hexdigest()}  {filename}\n"
        for filename, data in small.items()
    )
    stream = io.BytesIO()
    with zipfile.ZipFile(
        stream, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
    ) as archive:
        with archive.open(name, "w", force_zip64=True) as member:
            member.write(payload)
        archive.writestr(
            "sdk-runtime-1.tar.gz",
            os.urandom(2 * 1024**2),
            compress_type=zipfile.ZIP_STORED,
        )
        for filename, data in small.items():
            archive.writestr(filename, data)
        archive.writestr("SHA256SUMS", master)
    data = stream.getvalue()
    end = struct.unpack("<4s4H2IH", data[-22:])
    count, cd_size, cd_offset = end[4:7]
    zip64 = struct.pack(
        "<4sQ2H2I4Q", b"PK\x06\x06", 44, 45, 45, 0, 0, count, count, cd_size, cd_offset
    )
    locator = struct.pack("<4sIQI", b"PK\x06\x07", 0, len(data) - 22, 1)
    legacy = struct.pack(
        "<4s4H2IH", b"PK\x05\x06", 0, 0, 65535, 65535, 0xFFFFFFFF, 0xFFFFFFFF, 0
    )
    return data[:-22] + zip64 + locator + legacy, payload, name


class RemoteDeliveryTest(unittest.TestCase):
    def server(self, data, mode="normal"):
        observed = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                text = self.headers["Range"].split("=")[1]
                start, end = map(int, text.split("-"))
                observed.append((start, end, self.headers.get("Authorization")))
                self.send_response(200 if mode == "ignore" else 206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
                self.send_header("Content-Length", str(end - start + 1))
                self.end_headers()
                body = data[start : end + 1]
                if mode == "disconnect" and len(observed) == 1:
                    body = body[: len(body) // 2]
                    self.close_connection = True
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        reader = delivery.RemoteZip(
            len(data),
            lambda: f"http://127.0.0.1:{server.server_port}/blob",
            block_size=128 * 1024,
            cache_blocks=2,
            timeout=10,
        )
        return reader, observed

    def test_zip64_deflate_only_selected_archive_and_metadata(self):
        data, payload, name = fixture()
        reader, observed = self.server(data)
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            delivery.shutil, "disk_usage", return_value=SimpleNamespace(free=10**15)
        ):
            result = delivery.download_selected(reader, Path(tmp), "1", "source-sha")
            self.assertEqual((Path(tmp) / name).read_bytes(), payload)
            self.assertFalse((Path(tmp) / "sdk-runtime-1.tar.gz").exists())
            self.assertEqual(result["original_build_run_id"], "1")
        self.assertLess(reader.network_bytes, len(data) // 2)
        self.assertLessEqual(len(reader.cache), 2)
        self.assertTrue(
            all(
                auth is None and end - start + 1 <= 128 * 1024
                for start, end, auth in observed
            )
        )

    def test_disconnect_retries_exact_range_before_advancing_position(self):
        data = os.urandom(300000)
        reader, observed = self.server(data, "disconnect")
        self.assertEqual(reader.read(4096), data[:4096])
        self.assertEqual(reader.tell(), 4096)
        self.assertEqual(observed[0][:2], observed[1][:2])

    def test_ignored_range_is_rejected_without_consuming_body(self):
        reader, _ = self.server(b"x" * 1000, "ignore")
        with self.assertRaisesRegex(RuntimeError, "exact bounded"):
            reader.read(1)
        self.assertEqual(reader.tell(), 0)
        self.assertEqual(reader.network_bytes, 0)

    def test_byte_cap_stops_before_read(self):
        reader, _ = self.server(b"x" * 1000)
        reader.byte_budget = 1
        with self.assertRaisesRegex(RuntimeError, "byte budget"):
            reader.read(1)
        self.assertEqual(reader.network_bytes, 0)

    def test_original_build_identity_must_match(self):
        data, _, _ = fixture()
        reader, _ = self.server(data)
        with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(reader) as archive:
            with self.assertRaisesRegex(RuntimeError, "original build"):
                delivery.metadata(archive, Path(tmp), "another-run", "source-sha")

    def test_hash_mismatch_never_publishes_selected_archive(self):
        data, _, name = fixture(wrong_hash=True)
        reader, _ = self.server(data)
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            delivery.shutil, "disk_usage", return_value=SimpleNamespace(free=10**15)
        ):
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                delivery.download_selected(reader, Path(tmp), "1", "source-sha")
            self.assertFalse((Path(tmp) / name).exists())

    def test_disk_preflight_fails_before_selected_member_transfer(self):
        data, _, _ = fixture()
        reader, observed = self.server(data)
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            delivery.shutil, "disk_usage", return_value=SimpleNamespace(free=0)
        ):
            with self.assertRaisesRegex(RuntimeError, "Insufficient disk"):
                delivery.download_selected(reader, Path(tmp), "1", "source-sha")
        self.assertTrue(all(start > 0 for start, end, auth in observed))


if __name__ == "__main__":
    unittest.main()
