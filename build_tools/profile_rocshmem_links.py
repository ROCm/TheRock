#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Controlled link-only rocSHMEM ABBA replay, exclusively in TheRock CI.

Build prerequisites once, retain identical objects/libraries, then compare the
generated native final link edges at offload width one versus two. Both final
links run concurrently in each case; two CI allocations balance 1,2 and 2,1.
No publication.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

from _therock_utils.archive_util import open_archive_for_write

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("tools/rocshmem_info", "tests/functional_tests/rocshmem_functional_tests")
ARCHITECTURES = "gfx900 gfx90c gfx906 gfx908 gfx90a gfx942 gfx950 gfx1010 gfx1011 gfx1012 gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036 gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151 gfx1152 gfx1153 gfx1170 gfx1171 gfx1200 gfx1201 gfx1250-strict gfx1250".split()
SYSTEMS_SHA = "a10dca087065a82f15ea89dc7d6fdd3c6bb2c246"
LLVM_SHA = "e095f1b3becf0c1bfdc5e7647003d60976a2460b"


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise RuntimeError("Link replay runtime budget exhausted")


def require_rock_ci() -> None:
    if (
        os.getenv("GITHUB_ACTIONS") != "true"
        or os.getenv("GITHUB_REPOSITORY") != "ROCm/TheRock"
    ):
        raise RuntimeError("Full link replay executes only in ROCm/TheRock CI")
    current = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if current != os.environ["GITHUB_SHA"]:
        raise RuntimeError("Checkout does not match triggering TheRock SHA")


def sha256(path: Path, deadline: float) -> str:
    hashed = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            check_deadline(deadline)
            hashed.update(chunk)
    return hashed.hexdigest()


def driver_arguments(command: str, target: str) -> list[str]:
    args = shlex.split(command)
    if args[:2] == [":", "&&"] and args[-2:] == ["&&", ":"]:
        args = args[2:-2]
    if (
        not args
        or Path(args[0]).name != "clang++"
        or any(a in ["&&", ";", "|", ">"] for a in args)
    ):
        raise RuntimeError("Expected one generated clang++ link edge")
    if "-c" in args or "-o" not in args or args[args.index("-o") + 1] != target:
        raise RuntimeError("Unexpected generated output or compile action")
    if any(
        a.lower().endswith((".cpp", ".cc", ".cxx", ".hip", ".cu", ".c")) for a in args
    ):
        raise RuntimeError(
            "Measured native link edge must not recompile source translation units"
        )
    if any(a.startswith("@") for a in args):
        raise RuntimeError(
            "Generated link response file was not expanded by Ninja compdb -x"
        )
    arches = [
        a.removeprefix("--offload-arch=")
        for a in args
        if a.startswith("--offload-arch=")
    ]
    if arches != ARCHITECTURES or "--hip-link" not in args or "-fgpu-rdc" not in args:
        raise RuntimeError(
            "Link must retain the exact baseline 31-target RDC configuration"
        )
    if any(a in ["-v", "-###"] for a in args):
        raise RuntimeError("Timed replay must not add verbose/dry-run driver options")
    return args


def validate_candidate(baseline: list[str], candidate: list[str]) -> None:
    if any(a.startswith("--offload-jobs=") for a in baseline):
        raise RuntimeError(
            "Expected the recorded baseline with no offload-jobs override"
        )
    if (
        candidate.count("--offload-jobs=2") != 1
        or [a for a in candidate if a != "--offload-jobs=2"] != baseline
    ):
        raise RuntimeError(
            "Candidate changed more than the generated width-two link flag"
        )


def capture_edges(tree: Path) -> dict[str, list[str]]:
    rows = json.loads(
        subprocess.check_output(
            ["ninja", "-C", str(tree), "-t", "compdb", "-x"], text=True
        )
    )
    edges = {}
    for row in rows:
        if row.get("output") in TARGETS:
            target = row["output"]
            if target in edges:
                raise RuntimeError(
                    "Duplicate final edge in generated compilation database"
                )
            edges[target] = driver_arguments(row["command"], target)
    if set(edges) != set(TARGETS):
        raise RuntimeError("Missing expected final rocSHMEM links")
    return edges


def stop_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_logged(command: list[str], cwd: Path, log: Path, deadline: float) -> float:
    check_deadline(deadline)
    started = time.monotonic()
    with log.open("wb") as stream:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=max(1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            stop_group(process)
            raise RuntimeError(f"Command exceeded deadline; see {log}")
    if code:
        stop_group(process)
        raise RuntimeError(f"Command failed with exit {code}; see {log}")
    return time.monotonic() - started


def explicit_inputs(tree: Path, edges: dict[str, list[str]]) -> list[Path]:
    paths = set()
    for args in edges.values():
        paths.add(Path(args[0]).resolve())
        for arg in args[1:]:
            if arg.startswith("-"):
                continue
            path = Path(arg) if Path(arg).is_absolute() else tree / arg
            if path.is_file() and any(
                token in path.name for token in [".o", ".a", ".so"]
            ):
                paths.add(path.absolute())
    return sorted(paths)


def input_manifest(paths: list[Path], deadline: float) -> list[dict[str, Any]]:
    return [
        dict(
            path=str(p),
            resolved=str(p.resolve()),
            size=p.stat().st_size,
            sha256=sha256(p, deadline),
        )
        for p in paths
    ]


def pair_worker(commands_file: Path, output: Path, timeout: float) -> None:
    """Small local unit fixtures may exercise this worker without the CI controller."""
    payload = json.loads(commands_file.read_text())
    cwd = Path(payload["cwd"])
    processes, logs = [], []

    def interrupted(signum, frame):
        for process in processes:
            stop_group(process)
        raise SystemExit(128 + signum)

    old_handler = signal.signal(signal.SIGTERM, interrupted)
    started = time.monotonic()
    try:
        for name, command in payload["commands"].items():
            directory = output / name
            directory.mkdir(parents=True)
            log = (directory / "stdout.log").open("wb")
            logs.append(log)
            profiler = [
                sys.executable,
                str(ROOT / "build_tools/profile_build.py"),
                "--output",
                str(directory / "profile"),
                "--",
                *command,
            ]
            processes.append(
                subprocess.Popen(
                    profiler,
                    cwd=cwd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        while any(p.poll() is None for p in processes):
            if time.monotonic() - started > timeout:
                raise RuntimeError("Concurrent link pair exceeded its timeout")
            if any(p.poll() not in [None, 0] for p in processes):
                raise RuntimeError("One concurrent link failed")
            time.sleep(0.2)
        if any(p.returncode for p in processes):
            raise RuntimeError("Concurrent link pair failed")
        summaries = {
            name: json.loads((output / name / "profile/summary.json").read_text())
            for name in payload["commands"]
        }
        save(
            output / "pair.json",
            dict(
                pair_wall_s=time.monotonic() - started,
                links=summaries,
                info_link_policy="Both native links use the case's width; measure their interference separately",
                cpu_scope="Per-link waited-descendant CPU excludes profiler self CPU; do not add pair/subtree CPU scopes",
            ),
        )
    except BaseException:
        for process in processes:
            stop_group(process)
        raise
    finally:
        signal.signal(signal.SIGTERM, old_handler)
        for log in logs:
            log.close()


def retain_outputs(tree: Path, output: Path, deadline: float) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "rocm-systems/shared/kpack/python"))
    from rocm_kpack.binutils import BundledBinary

    rows = {}
    with open_archive_for_write(
        output / "linked-binaries.tar.zst", "zstd", 3
    ) as archive:
        for target in TARGETS:
            path = tree / target
            row = dict(bytes=path.stat().st_size, sha256=sha256(path, deadline))
            archive.add(path, arcname=target, recursive=False)
            rows[target] = row
    save(output / "output-sha256-before-inspection.json", rows)
    for target in TARGETS:
        path = tree / target
        row = rows[target]
        images = []
        with BundledBinary(path).unbundle() as contents:
            for target_id, filename in contents.target_list:
                if not target_id.startswith("hip"):
                    continue
                image = contents.dest_dir / filename
                images.append(
                    dict(
                        target_id=target_id,
                        architecture=target_id.rpartition("--")[2],
                        bytes=image.stat().st_size,
                        sha256=sha256(image, deadline),
                    )
                )
        if sorted(image["architecture"] for image in images) != sorted(ARCHITECTURES):
            raise RuntimeError(
                "Retained native binary does not contain every full architecture ID"
            )
        row["images"] = sorted(images, key=lambda image: image["target_id"])
        for option in ["-h", "-SW", "-d", "-n", "-sW"]:
            run_logged(
                ["readelf", option, str(path)],
                tree,
                output / (Path(target).name + option.replace("-", "_") + ".txt"),
                deadline,
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build/comm-link-diagnostics"
    )
    parser.add_argument("--max-wall-seconds", type=float, default=12600)
    parser.add_argument("--order", choices=["1,2", "2,1"], default="1,2")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--worker-timeout", type=float, default=4200)
    args = parser.parse_args()
    if args.worker:
        pair_worker(args.worker, args.worker_output, args.worker_timeout)
        return
    require_rock_ci()
    if sys.platform != "linux":
        parser.error("Link replay requires Linux")
    output = args.output.resolve()
    if output != ROOT / "build/comm-link-diagnostics" or output.exists():
        parser.error("Use the fresh dedicated diagnostic output path")
    output.mkdir(parents=True)
    tree = ROOT / "build/comm-libs/rocshmem/build"
    started, deadline = time.monotonic(), time.monotonic() + args.max_wall_seconds
    preparation = []
    try:
        pins = {}
        for path, expected in [
            ("rocm-systems", SYSTEMS_SHA),
            ("compiler/amd-llvm", LLVM_SHA),
        ]:
            actual = subprocess.check_output(
                ["git", "rev-parse", f"HEAD:{path}"], cwd=ROOT, text=True
            ).strip()
            if actual != expected:
                raise RuntimeError(f"Unexpected pinned source revision for {path}")
            pins[path] = actual
        preparation.append(
            dict(
                operation="configure-rocshmem",
                wall_s=run_logged(
                    [
                        "cmake",
                        "--build",
                        str(ROOT / "build"),
                        "--target",
                        "rocshmem+configure",
                    ],
                    ROOT,
                    output / "prepare-configure.log",
                    deadline,
                ),
            )
        )
        source = ROOT / "rocm-systems/projects/rocshmem"
        generated = {}
        for width in [1, 2, 1]:
            label = f"configure-width-{width}-{len(preparation)}"
            preparation.append(
                dict(
                    operation=label,
                    wall_s=run_logged(
                        [
                            "cmake",
                            "-S",
                            str(source),
                            "-B",
                            str(tree),
                            f"-DHIP_CLANG_NUM_PARALLEL_JOBS={width}",
                        ],
                        ROOT,
                        output / (label + ".log"),
                        deadline,
                    ),
                )
            )
            edges = capture_edges(tree)
            if width == 2:
                generated["B"] = edges
            elif "A" in generated:
                if generated["A"] != edges:
                    raise RuntimeError(
                        "Restoring the width-one configuration changed its generated edges"
                    )
            else:
                generated["A"] = edges
        for target in TARGETS:
            validate_candidate(generated["A"][target], generated["B"][target])
        save(output / "generated-link-edges.json", generated)
        prerequisites = {"librocshmem.a"}
        for argv in generated["A"].values():
            prerequisites.update(
                a
                for a in argv
                if not a.startswith("-")
                and a.endswith(".o")
                and not Path(a).is_absolute()
            )
        preparation.append(
            dict(
                operation="build-identical-prerequisites-once",
                targets=sorted(prerequisites),
                wall_s=run_logged(
                    ["ninja", "-C", str(tree), *sorted(prerequisites)],
                    ROOT,
                    output / "prepare-objects.log",
                    deadline,
                ),
            )
        )
        dry = subprocess.check_output(
            ["ninja", "-C", str(tree), "-n", *sorted(prerequisites)], text=True
        )
        (output / "prerequisite-dry-run.txt").write_text(dry)
        if "no work to do" not in dry:
            raise RuntimeError(
                "Prerequisite graph is not unchanged after its one preparation build"
            )
        inputs = explicit_inputs(tree, generated["A"])
        tools_dir = Path(generated["A"][TARGETS[0]][0]).parent
        for executable in ["ld.lld", "clang-linker-wrapper", "clang-offload-bundler"]:
            path = tools_dir / executable
            if not path.is_file():
                raise RuntimeError(f"Required baseline link tool is missing: {path}")
            inputs.append(path.absolute())
        inputs = sorted(set(inputs))
        inputs.extend(tools_dir.glob("*.cfg"))
        for argv in generated["A"].values():
            for arg in argv:
                if arg.startswith("--hip-device-lib-path="):
                    inputs.extend(Path(arg.split("=", 1)[1]).rglob("*.bc"))
        inputs.extend(tools_dir.parent.glob("libLLVM*.so*"))
        inputs = sorted(set(inputs))
        for executable in ["clang++", "ld.lld"]:
            run_logged(
                [str(tools_dir / executable), "--version"],
                tree,
                output / (executable.replace("+", "p") + "-version.txt"),
                deadline,
            )
        original = input_manifest(inputs, deadline)
        save(output / "input-manifest.json", original)
        # Retain the exact local object/archive bytes; runtime/toolchain bytes are
        # pinned inbound artifacts and are individually hashed in the manifest.
        with open_archive_for_write(
            output / "local-link-inputs.tar.zst", "zstd", 3
        ) as archive:
            for path in inputs:
                if tree in path.parents:
                    archive.add(
                        path, arcname=path.relative_to(tree).as_posix(), recursive=False
                    )
        for label, edges in generated.items():
            for target, argv in edges.items():
                # Driver expansion is untimed and never used as the real link.
                run_logged(
                    [*argv, "-###"],
                    tree,
                    output / f"driver-expansion-{label}-{Path(target).name}.log",
                    deadline,
                )
        save(
            output / "metadata.json",
            dict(
                ci={
                    k: os.getenv(k)
                    for k in [
                        "GITHUB_REPOSITORY",
                        "GITHUB_RUN_ID",
                        "GITHUB_SHA",
                        "RUNNER_NAME",
                    ]
                },
                pins=pins,
                python=sys.version,
                python_packages={
                    name: importlib.metadata.version(name)
                    for name in [
                        "psutil",
                        "pyzstd",
                        "zstandard",
                        "msgpack",
                        "pyelftools",
                    ]
                },
                runner_label="aws-linux-scale-rocm-prod",
                container_digest="sha256:d666e4613c1ca919bc7d1a31a4c368740b66c34b5c19ef95ba15cd31db7783fb",
                reference_run=41189458499,
                recorded_source_head="5b7cb88373362ab80dc0f19605915e534f4158aa",
                reference_sha="82d5e8c176ce0cc00b5fd1cf631ce7790be8992b",
                architectures=ARCHITECTURES,
                preparation=preparation,
                trial="Both native final links concurrent, each width one versus two; two allocations balance opposite orders",
                order=args.order,
                cache_state="One prerequisite build, full input hashes before every pass; no cache dropping",
                limitation="No concurrent RCCL build and no original link-release stagger; project link-only experiment, not stage/CI saving",
            ),
        )
        rows = []
        for index, width in enumerate(args.order.split(",")):
            method = "A" if width == "1" else "B"
            check_deadline(deadline)
            if input_manifest(inputs, deadline) != original:
                raise RuntimeError(
                    "Objects or explicit libraries changed before link replay"
                )
            if shutil.disk_usage(output).free < 5 * 1024**3:
                raise RuntimeError(
                    "Insufficient disk reserve for retained linked outputs"
                )
            directory = output / f"pass-{index}-{method}"
            directory.mkdir()
            commands = dict(
                cwd=str(tree),
                commands={
                    "info": generated[method][TARGETS[0]],
                    "functional": generated[method][TARGETS[1]],
                },
            )
            save(directory / "commands.json", commands)
            timeout = min(4200, max(1, deadline - time.monotonic()))
            wall = run_logged(
                [
                    sys.executable,
                    str(__file__),
                    "--worker",
                    str(directory / "commands.json"),
                    "--worker-output",
                    str(directory),
                    "--worker-timeout",
                    str(timeout),
                ],
                ROOT,
                directory / "worker.log",
                min(deadline, time.monotonic() + timeout + 10),
            )
            verification_started = time.monotonic()
            outputs = retain_outputs(tree, directory, deadline)
            if input_manifest(inputs, deadline) != original:
                raise RuntimeError(
                    "Link replay changed its fixed input objects/libraries"
                )
            row = dict(
                index=index,
                method=method,
                pair=json.loads((directory / "pair.json").read_text()),
                controller_worker_wall_s=wall,
                outputs=outputs,
                retention_and_input_verification_wall_s=time.monotonic()
                - verification_started,
            )
            rows.append(row)
            save(output / "passes.json", rows)
            print(
                json.dumps(
                    dict(
                        index=index,
                        method=method,
                        pair_wall_s=row["pair"]["pair_wall_s"],
                    )
                ),
                flush=True,
            )
        identical = all(row["outputs"] == rows[0]["outputs"] for row in rows)
        save(
            output / "summary.json",
            dict(
                measurements_complete=True,
                all_linked_binary_bytes_equal=identical,
                validated=identical,
                wall_s=time.monotonic() - started,
                passes=rows,
                correctness_scope="Exact linked output bytes versus repeated baseline; no new GPU runtime tests",
                scope="Link-only result; stage and full CI gains remain unvalidated",
            ),
        )
        if not identical:
            raise RuntimeError(
                "Retained linked binaries differ; image/semantic validation required before speedup claim"
            )
    except Exception as error:
        save(
            output / "failure.json",
            dict(
                error=str(error),
                elapsed_s=time.monotonic() - started,
                preparation=preparation,
            ),
        )
        raise


if __name__ == "__main__":
    main()
