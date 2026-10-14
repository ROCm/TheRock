#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run actual fetcher publication concurrently with native Windows cache readers.

A barrier arranges simultaneous cache misses, as in the existing publication
unit test. No filesystem error is injected. Results distinguish reader errors
from publisher errors; zero errors do not rule out a rare interleaving.
"""

import argparse
import concurrent.futures
import importlib.util
import hashlib
import json
from pathlib import Path
import platform
import sys
import tempfile
import threading
import time
import traceback
import types
from unittest import mock

ROOT = Path(__file__).resolve().parent


def load_fetcher(variant):
    path = ROOT / variant / "build_tools/fetch_dvc_artifacts.py"
    name = "publication_" + variant
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    # Publication uses only Python filesystem APIs. Supply import-only stubs
    # for unused S3 dependencies so this experiment needs no package downloads.
    # An accidental S3 client request must fail instead of contacting a remote.
    boto = types.ModuleType("boto3")
    core = types.ModuleType("botocore")
    client = types.ModuleType("botocore.client")
    core.UNSIGNED = object()

    def forbidden_client(*args, **kwargs):
        raise RuntimeError("S3 is disabled in this filesystem-only experiment")

    boto.client = forbidden_client
    client.Config = forbidden_client
    with mock.patch.dict(
        sys.modules, {"boto3": boto, "botocore": core, "botocore.client": client}
    ):
        spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.iterations < 1 or args.workers < 2:
        parser.error("iterations must be positive and workers must be at least two")
    args.directory.mkdir(parents=True, exist_ok=True)
    real_exists = Path.exists
    records = []
    workers = args.workers
    payload = b"same verified tensor contents" * 100
    for variant in ("baseline", "candidate"):
        fda = load_fetcher(variant)
        for iteration in range(args.iterations):
            with tempfile.TemporaryDirectory(
                prefix="toga-7-publication-", dir=args.directory
            ) as tmp:
                root = Path(tmp)
                cache = root / "cache/key"
                sources = [root / f"src-{i}" for i in range(workers)]
                for path in sources:
                    path.write_bytes(payload)
                barrier = threading.Barrier(workers)
                state = threading.local()
                done = threading.Event()
                record = {
                    "variant": variant,
                    "iteration": iteration,
                    "reader_attempts": 0,
                    "reader_failures": 0,
                    "reader_examples": [],
                    "publisher_failures": [],
                    "fetcher_sha256": hashlib.sha256(
                        (
                            ROOT / variant / "build_tools/fetch_dvc_artifacts.py"
                        ).read_bytes()
                    ).hexdigest(),
                }

                def exists(path):
                    if path == cache and not getattr(state, "checked", False):
                        present = real_exists(path)
                        state.checked = True
                        barrier.wait(timeout=10)
                        return present
                    return real_exists(path)

                def reader():
                    while not done.is_set():
                        if real_exists(cache):
                            record["reader_attempts"] += 1
                            try:
                                if cache.read_bytes() != payload:
                                    raise RuntimeError("incorrect cache contents")
                            except Exception:
                                record["reader_failures"] += 1
                                if len(record["reader_examples"]) < 5:
                                    record["reader_examples"].append(
                                        traceback.format_exc()
                                    )
                        else:
                            time.sleep(0)

                thread = threading.Thread(target=reader)
                thread.start()
                try:
                    with mock.patch.object(Path, "exists", exists):
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=workers
                        ) as ex:
                            futures = [
                                ex.submit(fda._store_in_cache, src, cache)
                                for src in sources
                            ]
                            for future in futures:
                                try:
                                    future.result(timeout=15)
                                except Exception:
                                    record["publisher_failures"].append(
                                        traceback.format_exc()
                                    )
                finally:
                    done.set()
                    thread.join(timeout=5)
                try:
                    record["final_correct"] = cache.read_bytes() == payload
                except Exception:
                    record["final_correct"] = False
                    record["final_error"] = traceback.format_exc()
                record["staging_files"] = len(list(root.rglob("*.tmp")))
                records.append(record)
                # Preserve partial evidence if a later round or job times out.
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps({"records": records}, indent=2) + "\n"
                )
        print(
            json.dumps(
                {
                    "variant": variant,
                    "reader_failures": sum(
                        r["reader_failures"] for r in records if r["variant"] == variant
                    ),
                    "publisher_failures": sum(
                        len(r["publisher_failures"])
                        for r in records
                        if r["variant"] == variant
                    ),
                }
            ),
            flush=True,
        )
    result = {
        "platform": sys.platform,
        "python": platform.python_version(),
        "directory": str(args.directory),
        "records": records,
        "note": "Real rename/replace calls; synchronized misses, no injected errors.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
