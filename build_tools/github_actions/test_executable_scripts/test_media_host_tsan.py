#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build and run exact device-free media parser inventories under host TSAN."""

import hashlib
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

from host_tsan_instrumentation import (
    native_host_tsan_environment,
    require_direct_clang_tsan,
    require_no_gpu_nodes,
)


COMPONENTS = {
    "rocdecode": {
        "source": "rocdecode_host_tsan.cpp",
        "library": "rocdecode",
        "data_dir": "share/rocdecode",
        "expected_names": (
            "rocdecode.bitstream.av1",
            "rocdecode.bitstream.avc",
            "rocdecode.bitstream.hevc",
            "rocdecode.bitstream.vp9",
        ),
        "inventory_sha256": (
            "a6fbc27d22c74debc172611ba86504cd642ec12366fcb668571297e79e7b01f6"
        ),
    },
    "rocjpeg": {
        "source": "rocjpeg_host_tsan.cpp",
        "library": "rocjpeg",
        "data_dir": "share/rocjpeg",
        "expected_names": (
            "rocjpeg.stream.invalid",
            "rocjpeg.stream.mug_400",
            "rocjpeg.stream.mug_420",
            "rocjpeg.stream.mug_422",
        ),
        "inventory_sha256": (
            "2926400830de2f94ea6c56c54e3cebd78d746f0d163f8c7394ba1dfadd2d8949"
        ),
    },
}


def _inventory_digest(names: list[str] | tuple[str, ...]) -> str:
    normalized = "".join(f"{name}\n" for name in sorted(names))
    return hashlib.sha256(normalized.encode()).hexdigest()


def _validate_inventory(component: str, names: list[str]) -> tuple[str, ...]:
    config = COMPONENTS[component]
    actual = tuple(sorted(names))
    expected = tuple(sorted(config["expected_names"]))
    digest = _inventory_digest(actual)
    if actual != expected or digest != config["inventory_sha256"]:
        raise RuntimeError(
            f"{component} host-TSAN media inventory changed: "
            f"expected count={len(expected)}, sha256={config['inventory_sha256']}; "
            f"got count={len(actual)}, sha256={digest}, names={actual}"
        )
    return actual


def _artifact_compiler(prefix: Path) -> Path:
    candidates = (
        prefix / "llvm" / "bin" / "amdclang++",
        prefix / "llvm" / "bin" / "clang++",
        prefix / "lib" / "llvm" / "bin" / "amdclang++",
        prefix / "lib" / "llvm" / "bin" / "clang++",
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError(f"artifact C++ compiler not found below {prefix}")


def _compile_command(
    compiler: Path,
    source: Path,
    output: Path,
    prefix: Path,
    library: str,
) -> list[str]:
    lib_dir = prefix / "lib"
    sysdeps_dir = lib_dir / "rocm_sysdeps" / "lib"
    return [
        str(compiler),
        "-std=c++17",
        "-D__HIP_PLATFORM_AMD__=1",
        "-fsanitize=thread",
        "-shared-libsan",
        "-fno-omit-frame-pointer",
        f"-I{prefix / 'include'}",
        str(source),
        "-o",
        str(output),
        f"-L{lib_dir}",
        f"-Wl,-rpath,{lib_dir}",
        f"-Wl,-rpath-link,{lib_dir}",
        f"-Wl,-rpath-link,{sysdeps_dir}",
        f"-l{library}",
    ]


def _run(command: list[str], env: dict[str, str], cwd: Path | None = None):
    print(f"++ Exec {shlex.join(command)}", flush=True)
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def main() -> int:
    try:
        component = os.environ["TEST_COMPONENT"]
        try:
            config = COMPONENTS[component]
        except KeyError as error:
            raise RuntimeError(
                f"unsupported media host-TSAN component: {component}"
            ) from error

        require_no_gpu_nodes()
        env = native_host_tsan_environment()
        prefix = Path(os.environ["THEROCK_BIN_DIR"]).resolve().parent
        script_dir = Path(__file__).resolve().parent
        source = script_dir / config["source"]
        library_path = prefix / "lib" / f"lib{config['library']}.so"
        data_dir = prefix / config["data_dir"]
        compiler = _artifact_compiler(prefix)

        if not source.is_file():
            raise RuntimeError(f"host-TSAN media harness source is missing: {source}")
        if not data_dir.is_dir():
            raise RuntimeError(f"host-TSAN media test data is missing: {data_dir}")
        require_direct_clang_tsan(library_path, env)

        with tempfile.TemporaryDirectory(prefix=f"{component}-host-tsan-") as temp:
            build_dir = Path(temp)
            executable = build_dir / f"{component}_host_tsan"
            command = _compile_command(
                compiler, source, executable, prefix, config["library"]
            )
            compiled = _run(command, env, build_dir)
            print(compiled.stdout + compiled.stderr, end="")
            require_direct_clang_tsan(executable, env)

            listed = _run([str(executable), "--list"], env, build_dir)
            if listed.stderr:
                print(listed.stderr, end="", file=sys.stderr)
            names = [
                line.strip()
                for line in listed.stdout.splitlines()
                if line.strip()
            ]
            inventory = _validate_inventory(component, names)

            for name in inventory:
                executed = _run(
                    [str(executable), "--case", name, str(data_dir)], env, build_dir
                )
                output = executed.stdout + executed.stderr
                print(output, end="")
                if f"PASS {name}" not in output:
                    raise RuntimeError(
                        f"{component} host-TSAN case did not report success: {name}"
                    )
        return 0
    except (
        KeyError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        if isinstance(error, subprocess.CalledProcessError):
            if error.stdout:
                print(error.stdout, end="")
            if error.stderr:
                print(error.stderr, end="", file=sys.stderr)
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
