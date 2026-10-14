#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify exact fixed fetcher with standard-library tests and synthetic pulls."""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import errno
from pathlib import Path
import platform
import sys
import tempfile
import threading
import unittest
from unittest import mock

from publication_stress import load_fetcher, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args()
    fda = load_fetcher("fixed")
    sys.modules["fetch_dvc_artifacts"] = fda
    real_pull = fda.pull

    def traced_pull(*args, **kwargs):
        messages = []
        original_log = kwargs.get("log", print)

        def log(message):
            messages.append(message)
            original_log(message)

        kwargs["log"] = log
        try:
            return real_pull(*args, **kwargs)
        except Exception:
            print("\n".join(messages))
            raise

    fda.pull = traced_pull
    spec = importlib.util.spec_from_file_location(
        "fixed_tests", ROOT / "fixed/fetch_dvc_artifacts_test.py"
    )
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(tests)
    )
    metadata = {
        "python": platform.python_version(),
        "platform": sys.platform,
        "source_sha256": hashlib.sha256(
            (ROOT / "fixed/build_tools/fetch_dvc_artifacts.py").read_bytes()
        ).hexdigest(),
        "unit_tests": result.testsRun,
        "unit_success": result.wasSuccessful(),
        "synthetic_rounds": 0,
        "synthetic_failures": [],
    }
    args.directory.mkdir(parents=True, exist_ok=True)
    payload = b"tiny deterministic tensor payload"
    md5 = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    manifest = json.dumps([{"md5": md5, "relpath": "child.bin"}]).encode()
    manifest_md5 = hashlib.md5(manifest, usedforsecurity=False).hexdigest() + ".dir"
    real_link = fda.os.link
    for mode in ("native", "manifest", "no-hardlinks", "source-cross-volume"):
        for iteration in range(100):
            with tempfile.TemporaryDirectory(
                prefix="toga-7-fixed-pull-", dir=args.directory
            ) as tmp:
                root = Path(tmp)
                projects = [root / "project-a", root / "project-b"]
                directory = mode == "manifest"
                out_md5 = manifest_md5 if directory else md5
                for project in projects:
                    (project / ".dvc").mkdir(parents=True)
                    (project / ".dvc/config").write_text(
                        "[core]\n remote = storage\n['remote \"storage\"']\n url = s3://b/pfx\n allow_anonymous_login = true\n"
                    )
                    for i in range(12):
                        (project / f"{i}.dvc").write_text(
                            f"outs:\n- md5: {out_md5}\n  size: {len(payload)}\n  path: out-{i}\n"
                        )
                cache = root / "cache"
                if iteration % 2:
                    for hash_value, data in ((md5, payload), (manifest_md5, manifest)):
                        if hash_value == manifest_md5 and not directory:
                            continue
                        cache_file = fda._cache_path(cache, hash_value)
                        cache_file.parent.mkdir(parents=True, exist_ok=True)
                        cache_file.write_bytes(b"X" * len(data))

                class S3:
                    calls = []
                    guard = threading.Lock()

                    def download_fileobj(self, bucket, key, file):
                        with self.guard:
                            self.calls.append(key)
                        file.write(manifest if key.endswith(".dir") else payload)

                def link(source, target, *args, **kwargs):
                    if mode == "no-hardlinks":
                        raise OSError(errno.ENOTSUP, "simulated unsupported hardlinks")
                    if mode == "source-cross-volume" and Path(target).is_relative_to(
                        cache
                    ):
                        if not Path(source).is_relative_to(cache):
                            raise OSError(
                                errno.EXDEV, "simulated different source volume"
                            )
                    return real_link(source, target, *args, **kwargs)

                fake = S3()
                messages = []
                with (
                    mock.patch.object(fda, "_make_s3_client", return_value=fake),
                    mock.patch.object(fda.os, "link", side_effect=link),
                ):
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                        futures = [
                            ex.submit(
                                fda.pull,
                                project,
                                jobs=12,
                                cache_dir=cache,
                                log=messages.append,
                            )
                            for project in projects
                        ]
                        results = [future.result(timeout=15) for future in futures]
                passed = len(fake.calls) == (2 if directory else 1)
                passed &= sum(r.fetched for r in results) == 1
                passed &= sum(r.cached for r in results) == 23
                passed &= all(
                    (
                        project / f"out-{i}" / "child.bin"
                        if directory
                        else project / f"out-{i}"
                    ).read_bytes()
                    == payload
                    for project in projects
                    for i in range(12)
                )
                passed &= not list(root.rglob("*.tmp"))
                metadata["synthetic_rounds"] += 1
                if not passed:
                    metadata["synthetic_failures"].append(
                        {
                            "mode": mode,
                            "iteration": iteration,
                            "downloads": fake.calls,
                            "results": [vars(r) for r in results],
                            "messages": messages,
                        }
                    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0 if result.wasSuccessful() and not metadata["synthetic_failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
