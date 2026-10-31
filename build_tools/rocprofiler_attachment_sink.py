# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Private bounded loopback CDash sink; never forwards a submission."""

import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import threading
import urllib.parse
import uuid


class LoopbackSink:
    def __init__(self, output: Path):
        self.output = output
        self.output.mkdir()
        self.nonce = uuid.uuid4().hex
        self.records, self.errors = [], []
        self.total_bytes = 0
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                self.request.settimeout(2)
                self.capture_completed = False
                super().setup()

            def handle(self):
                try:
                    if len(owner.records) >= 64:
                        raise ValueError("submission count budget exceeded")
                    super().handle()
                finally:
                    if not self.capture_completed:
                        owner.errors.append("incomplete or rejected local HTTP request")

            def log_message(self, *args):
                pass

            def receive(self):
                parsed = urllib.parse.urlsplit(self.path)
                if (
                    parsed.path != "/" + owner.nonce
                    or self.client_address[0] != "127.0.0.1"
                ):
                    self.send_error(403)
                    return
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if (
                        size < 0
                        or size > 128 * 1024**2
                        or owner.total_bytes + size > 512 * 1024**2
                    ):
                        raise ValueError("submission size budget exceeded")
                    if self.headers.get("Transfer-Encoding"):
                        raise ValueError("unsupported streamed transfer encoding")
                    data = self.rfile.read(size)
                    if len(data) != size:
                        raise ValueError("incomplete submission")
                    owner.total_bytes += size
                    name = f"submission-{len(owner.records):04}.bin"
                    (owner.output / name).write_bytes(data)
                    owner.records.append(
                        dict(
                            method=self.command,
                            request_metadata={
                                key: value[0][:1024]
                                for key, value in urllib.parse.parse_qs(
                                    parsed.query
                                ).items()
                                if key in {"FileName", "MD5", "Type", "project"}
                            },
                            file=name,
                            bytes=size,
                            sha256=hashlib.sha256(data).hexdigest(),
                            scope="Original local CDash submission; no forwarding",
                        )
                    )
                    response = b'<?xml version="1.0"?><cdash version="3.1.0"><status>OK</status><buildId>0</buildId></cdash>'
                    self.send_response(200)
                    self.send_header("Content-Type", "application/xml")
                    self.send_header("Content-Length", str(len(response)))
                    self.end_headers()
                    self.wfile.write(response)
                    self.capture_completed = True
                except (OSError, ValueError) as error:
                    owner.errors.append(str(error))
                    self.send_error(400)

            do_PUT = receive
            do_POST = receive
            do_GET = receive

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.server.timeout = 5
        self.url = f"http://127.0.0.1:{self.server.server_port}/{self.nonce}?project=rocprofiler-sdk-alt"
        assert urllib.parse.urlsplit(self.url).hostname == "127.0.0.1"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self):
        self.thread.start()

    def close(self) -> dict:
        if self.thread.is_alive():
            self.server.shutdown()
            self.thread.join(timeout=10)
        self.server.server_close()
        result = dict(
            records=self.records,
            errors=self.errors,
            total_bytes=self.total_bytes,
            loopback_only=True,
            forwarded=False,
            complete=bool(self.records)
            and not self.errors
            and not self.thread.is_alive(),
        )
        (self.output / "index.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
