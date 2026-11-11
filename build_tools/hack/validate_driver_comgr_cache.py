# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise the upstream Comgr cache regression with the driver DLL."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "win32":
        raise RuntimeError("This regression must run on Windows")

    source = Path(__file__).resolve().parents[2] / "compiler/amd-llvm/amd/comgr"
    stage = args.build_dir.resolve() / "compiler/amd-comgr/stage"
    with tempfile.TemporaryDirectory(prefix="comgr-cache-check-") as directory:
        work = Path(directory)
        exe = work / "compile-opencl-minimal.exe"
        subprocess.run(
            [
                "cl",
                "/nologo",
                "/MT",
                "/std:c11",
                f"/I{stage / 'include/amd_comgr'}",
                str(source / "test-lit/comgr-sources/compile-opencl-minimal.c"),
                str(stage / "lib/amd_comgr_3.lib"),
                f"/Fe:{exe}",
            ],
            cwd=work,
            check=True,
        )
        shutil.copy2(stage / "bin/amd_comgr_3.dll", work)
        cache = work / "cache-Gökalp"
        env = os.environ.copy()
        env.update(
            AMD_COMGR_CACHE="1",
            AMD_COMGR_CACHE_DIR=str(cache),
            AMD_COMGR_EMIT_VERBOSE_LOGS="1",
            AMD_COMGR_REDIRECT_LOGS="stdout",
        )
        outputs: list[bytes] = []
        for iteration, marker in enumerate(("stored entry", "found entry")):
            output = work / f"output-{iteration}.bin"
            result = subprocess.run(
                [
                    str(exe),
                    str(source / "test-lit/compile-minimal.cl"),
                    str(output),
                    "1.2",
                ],
                cwd=work,
                env=env,
                capture_output=True,
            )
            print(result.stdout.decode("utf-8", errors="replace"), flush=True)
            print(result.stderr.decode("utf-8", errors="replace"), flush=True)
            result.check_returncode()
            if f"Comgr cache: {marker}".encode() not in result.stdout:
                raise RuntimeError(f"Missing cache diagnostic: {marker}")
            if b"when pruning the cache" in result.stdout:
                raise RuntimeError("Cache pruning failed")
            # Upstream expects a cache tag and three cached compilation actions.
            if not cache.is_dir() or len(list(cache.iterdir())) != 4:
                raise RuntimeError("Expected four entries in the non-ASCII cache")
            data = output.read_bytes()
            if not data.startswith(b"\x7fELF"):
                raise RuntimeError("Compilation did not produce an ELF code object")
            outputs.append(data)
        if outputs[0] != outputs[1]:
            raise RuntimeError("Cached and uncached compilation outputs differ")
        print(
            "PASS: Windows non-ASCII cache path stores and reuses compilation results"
        )


if __name__ == "__main__":
    main()
