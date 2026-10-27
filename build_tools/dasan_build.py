#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build the experimental DASan compiler/runtime subset in TheRock's CI image."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parent.parent
IMAGE = "ghcr.io/rocm/therock_build_manylinux_x86_64@sha256:cf4f6d9909056906e4a1c0bc4c245658a0af4c6a6174082a86adfa2d518b7aeb"
PINS = {
    "compiler/amd-llvm": (
        "https://github.com/ROCm/llvm-project.git",
        "b07aa864f5b2bf648e6a61e2902cbfde83192481",
    ),
    "compiler/hipify": (
        "https://github.com/ROCm/HIPIFY.git",
        "d364ef407b2c3e20c6ceb3575bd852f0c7cebb8c",
    ),
    "compiler/spirv-llvm-translator": (
        "https://github.com/ROCm/SPIRV-LLVM-Translator.git",
        "5817038657a5d3166f753389ff7f4740bd8d7c16",
    ),
    "rocm-systems": (
        "https://github.com/jhuber6/rocm-systems.git",
        "f535db62a0dc73f2d56d4b93d9bbd4f3155e113b",
    ),
}


def capture(command: list[str], cwd: Path = REPO) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def run(
    command: list[str],
    log: Path,
    cwd: Path = REPO,
    env: dict[str, str] | None = None,
) -> None:
    print(f"++ {shlex.join(command)}", flush=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\n++ {shlex.join(command)}\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            env=env,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_sources(log: Path) -> None:
    # fetch_sources resets submodules: only permit preparation of a fresh checkout.
    existing = [path for path in PINS if (REPO / path / ".git").exists()]
    if existing:
        raise RuntimeError(
            "--prepare-sources requires a fresh checkout; already initialized: "
            + ", ".join(existing)
        )
    for path, (_, revision) in PINS.items():
        run(["git", "update-index", "--cacheinfo", f"160000,{revision},{path}"], log)
    run(["git", "config", "submodule.rocm-systems.url", PINS["rocm-systems"][0]], log)
    run(
        [
            sys.executable,
            "build_tools/fetch_sources.py",
            "--stage",
            "compiler-runtime",
            "--jobs",
            "12",
            "--depth",
            "1",
            "--no-apply-patches",
        ],
        log,
    )
    # The index now contains the experimental pins, so .smrev records their base
    # SHAs correctly. Apply the same compatibility patches used by regular CI.
    run(
        [
            sys.executable,
            "build_tools/fetch_sources.py",
            "--stage",
            "compiler-runtime",
            "--no-update-submodules",
        ],
        log,
    )


def record_sources(output: Path) -> None:
    sources = {}
    for path, (url, revision) in PINS.items():
        directory = REPO / path
        subprocess.check_call(
            ["git", "merge-base", "--is-ancestor", revision, "HEAD"], cwd=directory
        )
        if capture(["git", "status", "--porcelain", "--untracked-files=no"], directory):
            raise RuntimeError(
                f"Tracked source changes in {path}; commit them before building"
            )
        sources[path] = {
            "url": url,
            "requested_commit": revision,
            "effective_commit": capture(["git", "rev-parse", "HEAD"], directory),
        }
    manifest = {
        "therock_commit": capture(["git", "rev-parse", "HEAD"]),
        "therock_diff_sha256": hashlib.sha256(
            capture(
                [
                    "git",
                    "diff",
                    "HEAD",
                    "--",
                    ".",
                    ":(exclude)compiler/amd-llvm",
                    ":(exclude)compiler/hipify",
                    ":(exclude)compiler/spirv-llvm-translator",
                    ":(exclude)rocm-systems",
                ]
            ).encode()
        ).hexdigest(),
        "image": IMAGE,
        "sources": sources,
    }
    write_json(output / "sources.json", manifest)


def apply_dasan_patches(log: Path) -> None:
    compiler = REPO / "compiler" / "amd-llvm"
    patches = sorted((REPO / "patches" / "dasan" / "llvm-project").glob("*.patch"))
    if not patches:
        raise RuntimeError("Missing patches/dasan/llvm-project compatibility patches")
    if capture(["git", "status", "--porcelain", "--untracked-files=no"], compiler):
        raise RuntimeError(
            "Refusing to patch compiler with tracked uncommitted changes"
        )
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", PINS["compiler/amd-llvm"][1], "HEAD"],
        cwd=compiler,
    )
    # Later patches can change context needed to reverse earlier patches.
    # Walk backwards in an isolated index, leaving the real sources untouched.
    applied_count = len(patches)
    saw_applied = False
    with tempfile.TemporaryDirectory(prefix="therock-dasan-index-") as temporary:
        index_env = {**os.environ, "GIT_INDEX_FILE": str(Path(temporary) / "index")}
        subprocess.check_call(["git", "read-tree", "HEAD"], cwd=compiler, env=index_env)
        for index in reversed(range(len(patches))):
            patch = patches[index]
            reverse = subprocess.run(
                ["git", "apply", "--cached", "--reverse", str(patch)],
                cwd=compiler,
                env=index_env,
                capture_output=True,
                text=True,
            )
            if reverse.returncode == 0:
                saw_applied = True
            elif saw_applied:
                raise RuntimeError(
                    f"DASan patch series is out of order or modified at {patch.name}:\n"
                    + reverse.stderr
                )
            else:
                applied_count = index
        pending = patches[applied_count:]
        subprocess.check_call(["git", "read-tree", "HEAD"], cwd=compiler, env=index_env)
        for patch in pending:
            forward = subprocess.run(
                ["git", "apply", "--cached", str(patch)],
                cwd=compiler,
                env=index_env,
                capture_output=True,
                text=True,
            )
            if forward.returncode:
                raise RuntimeError("DASan compiler patch conflict:\n" + forward.stderr)
    if not pending:
        print("DASan compiler compatibility patches are already applied", flush=True)
    else:
        print(
            f"Applying {len(pending)} DASan patches after {applied_count} existing patches"
        )
        # Match fetch_sources.apply_patches without reapplying the existing
        # prefix or clearing other projects' .smrev metadata.
        run(
            [
                "git",
                "-c",
                "user.name=therockbot",
                "-c",
                "user.email=therockbot@amd.com",
                "am",
                "--whitespace=nowarn",
                "--no-gpg-sign",
                *map(str, pending),
            ],
            log,
            cwd=compiler,
            env={**os.environ, "GIT_COMMITTER_DATE": "Thu, 1 Jan 2099 00:00:00 +0000"},
        )
    run(
        ["git", "update-index", "--skip-worktree", "--", "compiler/amd-llvm"],
        log,
        cwd=REPO,
    )
    patches_hash = hashlib.sha1(usedforsecurity=False)
    for patch in patches:
        patches_hash.update(patch.read_bytes())
    origin, revision = PINS["compiler/amd-llvm"]
    state = f"{origin}\n{revision}+PATCHED:{patches_hash.hexdigest()}\n"
    smrev = compiler.with_name(".amd-llvm.smrev")
    if not smrev.exists() or smrev.read_text(encoding="utf-8") != state:
        smrev.write_text(state, encoding="utf-8")


def build(args: argparse.Namespace, extra: list[str]) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    log = output / "build.log"
    status = {"started_utc": datetime.now(timezone.utc).isoformat(), "state": "running"}
    write_json(output / "status.json", status)
    try:
        os.environ["CCACHE_DIR"] = str(output / "ccache")
        os.environ["PIP_CACHE_DIR"] = str(output / "pip-cache")
        # Subprojects invoke cmake --build separately from the outer Ninja build.
        os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = str(args.jobs)
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], log)
        if args.prepare_sources:
            prepare_sources(log)
        apply_dasan_patches(log)
        record_sources(output)
        if not args.prepare_only:
            from configure_stage import generate_cmake_args, get_topology

            stage_args = generate_cmake_args(
                "compiler-runtime",
                "",
                args.gpu_families or "",
                get_topology(),
                platform_name="linux",
                manylinux=True,
                artifact_names=["amd-llvm", "hipify", "core-runtime", "core-hip"],
            )
            # configure_stage renders shell quotes; subprocess uses argument arrays.
            stage_args = [shlex.split(arg)[0] for arg in stage_args]
            if not args.gpu_families:
                stage_args.append(f"-DTHEROCK_DIST_AMDGPU_TARGETS={args.gpu_targets}")
            command = [
                "cmake",
                "-S",
                str(REPO),
                "-B",
                str(output / "build"),
                "-GNinja",
                *stage_args,
                "-DCMAKE_BUILD_TYPE=Release",
                "-DTHEROCK_BUNDLE_SYSDEPS=ON",
                "-DTHEROCK_ENABLE_DASAN=ON",
                "-DTHEROCK_BUILD_TESTING=OFF",
                "-DTHEROCK_BUILD_COMGR_TESTS=OFF",
                "-DTHEROCK_BUILD_LLVM_TESTS=ON",
                "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_HIP_COMPILER_LAUNCHER=ccache",
                f"-DLLVM_PARALLEL_LINK_JOBS={args.link_jobs}",
                *extra,
            ]
            write_json(output / "configure.json", {"argv": command})
            run(command, log)
            run(
                [
                    "cmake",
                    "--build",
                    str(output / "build"),
                    "--parallel",
                    str(args.jobs),
                    "--target",
                    "stage-compiler-runtime",
                    "therock-artifacts",
                    "therock-dist",
                ],
                log,
            )
        status["state"] = "prepared" if args.prepare_only else "build-complete"
    except BaseException as exc:
        status.update(state="failed", error=str(exc))
        raise
    finally:
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "status.json", status)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prepare-sources",
        action="store_true",
        help="Initialize fresh submodules with experimental pins",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Record sources without configuring/building",
    )
    parser.add_argument("--jobs", type=int, default=192)
    parser.add_argument("--link-jobs", type=int, default=8)
    targets = parser.add_mutually_exclusive_group()
    targets.add_argument("--gpu-targets", default="gfx942;gfx950")
    targets.add_argument("--gpu-families")
    parser.add_argument(
        "--docker",
        default="docker",
        help="Docker executable; use --sudo for sudo -n docker",
    )
    parser.add_argument("--sudo", action="store_true")
    parser.add_argument("--in-container", action="store_true", help=argparse.SUPPRESS)
    argv = sys.argv[1:]
    separator = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:separator])
    extra = argv[separator + 1 :]
    if args.jobs < 1 or args.link_jobs < 1:
        parser.error("job counts must be positive")
    if args.in_container:
        build(args, extra)
        return
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    docker = (["sudo", "-n"] if args.sudo else []) + [args.docker]
    run([*docker, "pull", IMAGE], output / "docker.log")
    inner = [
        "python",
        "/therock/src/build_tools/dasan_build.py",
        "--in-container",
        "--output-dir",
        "/therock/output",
        "--jobs",
        str(args.jobs),
        "--link-jobs",
        str(args.link_jobs),
    ]
    if args.prepare_sources:
        inner.append("--prepare-sources")
    if args.prepare_only:
        inner.append("--prepare-only")
    inner += (
        ["--gpu-families", args.gpu_families]
        if args.gpu_families
        else ["--gpu-targets", args.gpu_targets]
    )
    run(
        [
            *docker,
            "run",
            "--rm",
            "--init",
            "--mount",
            f"type=bind,src={REPO},dst=/therock/src",
            "--mount",
            f"type=bind,src={output},dst=/therock/output",
            "-w",
            "/therock/src",
            "-e",
            "GIT_CONFIG_COUNT=1",
            "-e",
            "GIT_CONFIG_KEY_0=safe.directory",
            "-e",
            "GIT_CONFIG_VALUE_0=*",
            IMAGE,
            *inner,
            "--",
            *extra,
        ],
        output / "docker.log",
    )


if __name__ == "__main__":
    main()
