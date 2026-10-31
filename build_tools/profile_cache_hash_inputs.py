#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Private ccache 4.12.1 hash diagnostics. Compilation is always blocked.

Linux only. Input JSON contains directory, arguments, config_path and
compiler_check. Use a reviewed, expanded compilation database entry (no shell
launcher), the existing config path, and the same compiler-check algorithm
with its fingerprint-cache directory relocated to private diagnostic storage.
No cache backend, product object, or product dependency file is written.
The compiler is still allowed to preprocess; review its compiler_check command.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_keys(text):
    return {
        kind.lower().replace(" ", "_"): value
        for kind, value in re.findall(r"(Manifest key|Result key): ([a-z0-9]+)", text)
    }


def rewrite_outputs(argv, shared):
    """Retain inputs and flags, redirect only explicit -o/-MF destinations."""
    forbidden = (
        "@",
        "--ccache-",
        "-fplugin",
        "-fpass-plugin",
        "-Xclang",
        "-save-temps",
        "--save-temps",
        "-ftime-trace",
        "-MJ",
        "-Wp,",
        "-Xpreprocessor",
        "-mllvm",
        "-wrapper",
        "--config",
        "-fmodules",
        "-fmodule-",
        "-fimplicit-module",
        "-fprebuilt-module",
        "-fprofile",
        "-fcoverage",
        "--coverage",
        "-ftest-coverage",
        "-fcrash-diagnostics-dir",
        "-serialize-diagnostics",
    )
    if not argv or not Path(argv[0]).is_absolute():
        raise ValueError(
            "Use an absolute real compiler path, without ccache or shell launchers"
        )
    if not re.fullmatch(
        r"(?:clang\+\+|clang|gcc|g\+\+)(?:-\d+(?:\.\d+)*)?", Path(argv[0]).name
    ):
        raise ValueError("Expected a reviewed GCC or Clang compiler command")
    if any(arg.startswith(forbidden) for arg in argv[1:]):
        raise ValueError(
            "Expand response files; reject plugin, module, shell or extra-output options"
        )
    if "-c" not in argv or "-E" in argv or "-S" in argv:
        raise ValueError("Expected exactly one compile-only command")
    result, outputs = [argv[0]], {}
    index = 1
    while index < len(argv):
        arg = argv[index]
        key = next(
            (key for key in ("-o", "-MF") if arg == key or arg.startswith(key)), None
        )
        if key:
            if key in outputs:
                raise ValueError(f"Repeated {key}")
            if arg == key:
                index += 1
                if index == len(argv):
                    raise ValueError(f"Missing {key} value")
                outputs[key] = argv[index]
            else:
                outputs[key] = arg[len(key) :]
            result += [
                key,
                str(shared / ("never-produced.o" if key == "-o" else "preprocess.d")),
            ]
        else:
            result.append(arg)
        index += 1
    if "-o" not in outputs:
        raise ValueError("Explicit output required")
    return result, outputs


def run_probe(ccache, record, output, shared, timeout=300, env=None):
    if sys.platform != "linux":
        raise ValueError("This validated guard is Linux-only")
    output, shared = Path(output).absolute(), Path(shared).absolute()
    if output.exists():
        raise ValueError("Probe output directory must be new and private")
    if any(char.isspace() for char in str(output)):
        raise ValueError(
            "ccache's prefix parser requires a guard path without whitespace"
        )
    output.mkdir(parents=True)
    shared.mkdir(parents=True, exist_ok=True)
    original = record["arguments"]
    if not isinstance(original, list) or not all(
        isinstance(item, str) for item in original
    ):
        raise ValueError("arguments must be an expanded string array")
    cwd = Path(record["directory"])
    if not cwd.is_absolute() or not cwd.is_dir():
        raise ValueError("directory must exist and be absolute")
    argv, rewrites = rewrite_outputs(original, shared)
    if (shared / "never-produced.o").exists():
        raise ValueError("The shared diagnostic object path must not already exist")
    if (shared / "preprocess.d").exists():
        raise ValueError("The shared diagnostic dependency path must not already exist")
    config = Path(record["config_path"])
    if not config.is_absolute() or not config.is_file():
        raise ValueError(
            "Explicit original config_path is required to preserve hash settings"
        )
    if not isinstance(record["compiler_check"], str) or not record["compiler_check"]:
        raise ValueError(
            "A reviewed compiler_check with private fingerprint storage is required"
        )
    version = subprocess.check_output(
        [str(ccache), "--version"], text=True
    ).splitlines()[0]
    if version != "ccache version 4.12.1":
        raise ValueError(f"Unvalidated ccache version: {version}")
    marker = output / "compile-blocked.marker"
    guard = output / "compiler-guard"
    (output / "raw-preprocessor").mkdir()
    guard.write_text(
        "#!/bin/sh\nset -e\nfor input in "
        + shlex.quote(str(output / "tmp") + "/")
        + 'cpp_stdout*; do\n  if [ -f "$input" ]; then cp -- "$input" '
        + shlex.quote(str(output / "raw-preprocessor") + "/")
        + "; fi\ndone\n"
        + "printf 'blocked\\n' >> "
        + shlex.quote(str(marker))
        + "\nexit 97\n"
    )
    guard.chmod(0o700)
    (output / "cache").mkdir()
    (output / "tmp").mkdir()
    (output / "debug").mkdir()
    log = output / "ccache.log"
    settings = [
        f"cache_dir={output / 'cache'}",
        "remote_storage=",
        "remote_only=false",
        f"temporary_dir={output / 'tmp'}",
        f"log_file={log}",
        f"stats_log={output / 'stats.log'}",
        "debug=true",
        "debug_level=2",
        f"debug_dir={output / 'debug'}",
        f"prefix_command={guard}",
        "prefix_command_cpp=",
        "depend_mode=false",
        "direct_mode=true",
        "recache=false",
        "disable=false",
        "read_only=false",
        "read_only_direct=false",
        "compiler_check=" + record["compiler_check"],
    ]
    command = [str(ccache), *settings, *argv]
    environment = dict(os.environ if env is None else env)
    environment["CCACHE_CONFIGPATH"] = str(config)
    began = time.perf_counter()
    with (output / "stdout").open("wb") as stdout, (output / "stderr").open(
        "wb"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        timed_out = False
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
    text = log.read_text(errors="replace") if log.exists() else ""
    keys = parse_keys(text)
    files = [
        {
            "path": str(path.relative_to(output)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted((output / "debug").rglob("*"))
        if path.is_file()
    ]
    dependency_file = shared / "preprocess.d"
    if dependency_file.exists():
        (output / "preprocess.d").write_bytes(dependency_file.read_bytes())
        dependency_file.unlink()  # Only this invocation's redirected diagnostic output.
    sources = []
    for arg in original[1:]:
        candidate = Path(arg)
        if not arg.startswith("-") and candidate.suffix in (
            ".c",
            ".cc",
            ".cpp",
            ".cxx",
            ".hip",
        ):
            candidate = candidate if candidate.is_absolute() else cwd / candidate
            if candidate.is_file():
                sources.append({"path": str(candidate), "sha256": sha256(candidate)})
    produced = (shared / "never-produced.o").exists()
    valid = (
        code == 97
        and marker.exists()
        and not timed_out
        and not produced
        and "result_key" in keys
        and "Got result key from preprocessor" in text
    )
    result = {
        "valid_hash_probe": valid,
        "exit_code": code,
        "timed_out": timed_out,
        "wall_s": time.perf_counter() - began,
        "compile_guard_fired": marker.exists(),
        "object_exists": produced,
        "keys": keys,
        "debug_files": files,
        "directory": str(cwd),
        "original_arguments": original,
        "diagnostic_arguments": argv,
        "rewritten_output_destinations": rewrites,
        "original_config_path": str(config),
        "original_config_sha256": sha256(config),
        "diagnostic_compiler_check": record["compiler_check"],
        "source_files": sources,
        "raw_preprocessor_files": [
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted((output / "raw-preprocessor").iterdir())
        ],
        "compiler_sha256": sha256(argv[0]),
        "compiler_realpath": str(Path(argv[0]).resolve()),
        "ccache_version": version,
        "ccache_sha256": sha256(ccache),
        "scope": "Preprocessing/hash diagnostics only; neither a compile nor cache-hit benchmark",
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ccache", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--shared-output",
        type=Path,
        required=True,
        help="Same private absolute path for both cases; must contain no product files",
    )
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    result = run_probe(
        args.ccache.absolute(),
        json.loads(args.record.read_text()),
        args.output,
        args.shared_output,
        args.timeout,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("valid_hash_probe", "keys", "wall_s", "exit_code")
            }
        )
    )
    return 0 if result["valid_hash_probe"] else 1


if __name__ == "__main__":
    sys.exit(main())
