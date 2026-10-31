#!/usr/bin/env python3
"""Executable guard checks using a tiny C fixture; no production compilation.

Run on Linux: python test_probe_hash_inputs.py --ccache /path/ccache --scratch /new/private/path
Preserves the complete test evidence in scratch; does not delete a cache.
"""

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from profile_cache_hash_inputs import parse_keys, run_probe, sha256


def input_hashes(debug):
    result = {}
    for suffix in ("c", "d", "p"):
        paths = list(Path(debug).rglob("*.ccache-input-" + suffix))
        assert len(paths) == 1, (suffix, paths)
        result[suffix] = sha256(paths[0])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ccache", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    root, ccache = args.scratch.absolute(), args.ccache.absolute()
    assert sys.platform == "linux"
    assert not root.exists(), "Use a new private scratch directory"
    root.mkdir(parents=True)
    config = root / "fixture.conf"
    config.write_text("namespace = guard-fixture\ncompiler_type = gcc\n")
    compiler = root / "gcc"
    permit, attempts = (
        root / "allow-tiny-compilation",
        root / "forbidden-compile-attempts",
    )
    # The compiler spy itself forbids non-preprocessing invocation after the
    # one explicitly permitted tiny baseline. Its bytes remain unchanged.
    compiler.write_text(
        "#!/usr/bin/python3\nimport os, sys\nfrom pathlib import Path\n"
        + "if '-E' not in sys.argv and not Path("
        + repr(str(permit))
        + ").exists():\n"
        + "    Path("
        + repr(str(attempts))
        + ").write_text(repr(sys.argv))\n"
        + "    sys.exit(99)\n"
        + "os.execv("
        + repr(shutil.which("gcc"))
        + ", ['gcc', *sys.argv[1:]])\n"
    )
    compiler.chmod(0o700)
    source = root / "tiny.c"
    source.write_text("int tiny(void) { return 42; }\n")
    shared = root / "shared"
    shared.mkdir()
    argv = [
        str(compiler),
        "-O2",
        "-MD",
        "-MF",
        str(shared / "preprocess.d"),
        "-c",
        str(source),
        "-o",
        str(shared / "never-produced.o"),
    ]
    record = {
        "directory": str(root),
        "arguments": argv,
        "config_path": str(config),
        "compiler_check": "content",
    }
    baseline = root / "baseline"
    baseline.mkdir()
    environment = os.environ.copy()
    # Remove any unrelated shell/user ccache configuration from this fixture.
    environment = {
        key: value
        for key, value in environment.items()
        if not key.startswith("CCACHE_")
    }
    environment["CCACHE_CONFIGPATH"] = str(config)
    baseline_args = [
        str(ccache),
        f"cache_dir={baseline / 'cache'}",
        "remote_storage=",
        f"log_file={baseline / 'ccache.log'}",
        "debug=true",
        "debug_level=2",
        f"debug_dir={baseline / 'debug'}",
        "compiler_check=content",
        *argv,
    ]
    permit.write_text("Only the tiny ordinary-baseline compilation is allowed.\n")
    subprocess.run(
        baseline_args, cwd=root, env=environment, check=True, capture_output=True
    )
    assert (shared / "never-produced.o").is_file()
    (shared / "never-produced.o").unlink()
    (shared / "preprocess.d").unlink()
    permit.unlink()
    baseline_keys = parse_keys((baseline / "ccache.log").read_text())
    baseline_inputs = input_hashes(baseline / "debug")
    saved_environment = os.environ.copy()
    os.environ.clear()
    os.environ.update(environment)
    # Both must be overridden: preprocessing must work and no remote must be used.
    os.environ["CCACHE_PREFIX_CPP"] = "/bin/false"
    os.environ["CCACHE_REMOTE_STORAGE"] = "http://127.0.0.1:1"
    results = []
    try:
        for name in ("guard-a", "guard-b"):
            result = run_probe(ccache, record, root / name, shared, 20)
            assert result["valid_hash_probe"], result
            assert result["keys"] == baseline_keys, (result["keys"], baseline_keys)
            assert input_hashes(root / name / "debug") == baseline_inputs
            assert result["raw_preprocessor_files"]
            assert any(
                str(source) in path.read_text()
                for path in (root / name / "raw-preprocessor").iterdir()
            )
            assert (
                not attempts.exists()
            ), "Compiler spy saw a non-preprocessing invocation"
            results.append(
                {"case": name, "keys": result["keys"], "input_hashes": baseline_inputs}
            )
        for name, content in (
            ("preprocessor-error", '#include "definitely-missing-header.h"\n'),
            ("fallback-disabled-source", "// ccache:disable\nint x;\n"),
        ):
            bad_source = root / (name + ".c")
            bad_source.write_text(content)
            altered = dict(
                record,
                arguments=[
                    arg if arg != str(source) else str(bad_source) for arg in argv
                ],
            )
            result = run_probe(ccache, altered, root / name, shared, 20)
            assert result["exit_code"] == 97 and result["compile_guard_fired"]
            assert not result["valid_hash_probe"] and not result["object_exists"]
            assert "result_key" not in result["keys"]
            assert (
                not attempts.exists()
            ), "Fallback reached the non-preprocessing compiler spy"
            results.append(
                {"case": name, "guarded": True, "invalid_as_hash_comparison": True}
            )
        check_helper = (
            Path(__file__).resolve().parents[1] / "posix_ccache_compiler_check.py"
        )
        assert (
            check_helper.is_file()
        ), "Provide the reviewed TheRock compiler-check helper"
        check_comparison = []
        for name in ("compiler-check-cache-a", "compiler-check-cache-b"):
            # Actual TheRock fingerprint algorithm, actual GCC ELF+libraries;
            # vary only its private memoization-cache argument, not its output.
            command = shlex.join(
                [
                    sys.executable,
                    str(check_helper),
                    str(root / (name + "-memo")),
                    "%compiler%",
                ]
            )
            altered = dict(
                record,
                arguments=[str(Path(shutil.which("gcc")).absolute()), *argv[1:]],
                compiler_check=command,
            )
            result = run_probe(ccache, altered, root / name, shared, 20)
            assert result["valid_hash_probe"], result
            check_comparison.append(
                (result["keys"], input_hashes(root / name / "debug"))
            )
        assert check_comparison[0] == check_comparison[1]
        results.append(
            {
                "case": "compiler-check-memoization-paths",
                "keys_and_inputs_identical": True,
                "helper_sha256": sha256(check_helper),
            }
        )
    finally:
        os.environ.clear()
        os.environ.update(saved_environment)
    summary = {
        "passed": True,
        "fixture": "tiny GCC C source, ccache 4.12.1 on Linux",
        "checks": results,
        "ordinary_baseline_keys": baseline_keys,
        "no_unintended_compile_invocations": not attempts.exists(),
        "production_HIP_probe_executed": False,
    }
    (root / "test-results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
