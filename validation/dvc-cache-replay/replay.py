"""Compare real DVC pulls with baseline and PR fetchers; no ROCm build needed."""

import argparse
from dataclasses import asdict
import hashlib
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parent


def worker(args):
    variant = ROOT / args.variant
    source = variant / "build_tools" / "fetch_dvc_artifacts.py"
    sys.path.insert(0, str(source.parent))
    import fetch_dvc_artifacts as fda

    fixtures = ROOT / "fixtures" / args.fixture
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "platform": sys.platform,
        "python": platform.python_version(),
        "variant": args.variant,
        "revision": (variant / "revision.txt").read_text().strip(),
        "fetcher_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "fixture": args.fixture,
        "jobs": args.jobs,
        "passed": False,
    }
    started = time.monotonic()
    lock = threading.Lock()
    original = fda._process_pointer
    with (output / "pull.log").open("w", encoding="utf-8") as logfile:

        def log(message):
            with lock:
                print(message, file=logfile, flush=True)

        def process_pointer(*positional, **keywords):
            try:
                return original(*positional, **keywords)
            except Exception:
                log(traceback.format_exc())
                raise

        fda._process_pointer = process_pointer
        try:
            pointers = sorted(p for p in fixtures.rglob("*.dvc") if p.is_file())
            expected = [
                (p.parent.relative_to(fixtures) / out.path, out.md5, out.size)
                for p in pointers
                for out in fda._parse_dvc_pointer(p)
            ]
            metadata["pointers"] = len(pointers)
            metadata["outputs"] = len(expected)
            metadata["bytes"] = sum(size for _, _, size in expected)
            cache = output / "cache"
            for phase in ("cold", "warm"):
                dest = output / phase
                shutil.copytree(fixtures, dest)
                result = fda.pull(dest, jobs=args.jobs, cache_dir=cache, log=log)
                metadata[phase] = asdict(result)
                if result.fetched + result.cached != len(expected) or result.skipped:
                    raise RuntimeError(f"Incomplete {phase} pull: {result}")
                if phase == "warm" and result.fetched:
                    raise RuntimeError(f"Warm pull unexpectedly downloaded: {result}")
                for relpath, md5, size in expected:
                    for path in (dest / relpath, fda._cache_path(cache, md5)):
                        data = path.read_bytes()
                        if (
                            len(data) != size
                            or hashlib.md5(data, usedforsecurity=False).hexdigest()
                            != md5
                        ):
                            raise RuntimeError(f"Content verification failed: {path}")
            if list(output.rglob("*.tmp")):
                raise RuntimeError("Staging files remain after successful pulls")
            metadata["passed"] = True
        except Exception:
            metadata["error"] = traceback.format_exc()
            log(metadata["error"])
        finally:
            fda._process_pointer = original
    metadata["seconds"] = round(time.monotonic() - started, 3)
    (output / "result.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)
    return 0 if metadata["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument(
        "--fixture",
        choices=["oct06-batchnorm", "oct06-sdpa", "oct12-batchnorm"],
        default="oct12-batchnorm",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["baseline", "candidate", "incident-baseline"],
        default=["baseline", "candidate"],
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--variant", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.jobs < 1 or args.iterations < 1:
        parser.error("jobs and iterations must be positive")
    if args.worker:
        return worker(args)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for round_number in range(1, args.iterations + 1):
        # Alternate execution order to reduce order/host-load bias.
        variants = args.variants if round_number % 2 else list(reversed(args.variants))
        for variant in variants:
            output = args.output / f"round-{round_number:04d}-{variant}"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--variant",
                variant,
                "--fixture",
                args.fixture,
                "--output",
                str(output),
                "--jobs",
                str(args.jobs),
            ]
            result = subprocess.run(command, text=True, capture_output=True)
            if (output / "result.json").exists():
                entry = json.loads((output / "result.json").read_text())
            else:
                output.mkdir(parents=True, exist_ok=True)
                entry = {
                    "variant": variant,
                    "passed": False,
                    "error": result.stderr,
                    "exit_code": result.returncode,
                }
            (output / "process.log").write_text(result.stdout + result.stderr)
            results.append(entry)
            summary = {
                v: {
                    "rounds": sum(e["variant"] == v for e in results),
                    "failures": sum(
                        e["variant"] == v and not e["passed"] for e in results
                    ),
                }
                for v in args.variants
            }
            (args.output / "results.json").write_text(
                json.dumps(
                    {"platform": sys.platform, "summary": summary, "results": results},
                    indent=2,
                )
                + "\n"
            )
            print(
                json.dumps(
                    {
                        "round": round_number,
                        "variant": variant,
                        "passed": entry["passed"],
                        "summary": summary,
                    }
                ),
                flush=True,
            )
    return 1 if any(not r["passed"] for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
