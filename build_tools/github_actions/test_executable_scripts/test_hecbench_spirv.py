# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import json
import shlex
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent
SHARD_INDEX = int(os.getenv("SHARD_INDEX", "1")) - 1
TOTAL_SHARDS = int(os.getenv("TOTAL_SHARDS", "1"))

HECBENCH_BENCHMARKS = os.getenv("HECBENCH_BENCHMARKS", "")

COMPILE_TIMEOUT = int(os.getenv("HECBENCH_COMPILE_TIMEOUT", "300"))
RUN_TIMEOUT = int(os.getenv("HECBENCH_RUN_TIMEOUT", "300"))

logging.basicConfig(level=logging.INFO)


def _run(
    cmd: List[str],
    cwd: Path,
    env: Dict[str, str],
    capture_output: bool = False,
    timeout: Optional[int] = None,
):
    logging.info(f"++ Exec [{cwd}]$ {shlex.join(cmd)}")
    try:
        return subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            check=False,
            text=True,
            capture_output=capture_output,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logging.error("Command timed out after %d seconds: %s", timeout, shlex.join(cmd))
        raise


def _setup_env() -> Dict[str, str]:
    if not THEROCK_BIN_DIR:
        logging.error("THEROCK_BIN_DIR is not set.")
        sys.exit(1)

    therock_bin = Path(THEROCK_BIN_DIR).resolve()
    rocm_path = therock_bin.parent

    env = os.environ.copy()
    env["ROCM_PATH"] = str(rocm_path)
    env["PATH"] = f"{therock_bin}{os.pathsep}{env.get('PATH', '')}".rstrip(os.pathsep)

    if sys.platform.startswith("linux"):
        rocm_lib = str(rocm_path / "lib")
        ld_library_path = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = (
            f"{rocm_lib}{os.pathsep}{ld_library_path}" if ld_library_path else rocm_lib
        )

    return env


def _enable_spirv_in_makefile(makefile_path: Path, rocm_path: Path):
    amdgcnspirv_flag = " --offload-arch=amdgcnspirv"
    makefile_text = makefile_path.read_text(encoding="utf-8")

    # Inject SPIR-V flag
    if "hipcc" in makefile_text:
        logging.info("Injecting --offload-arch=amdgcnspirv into %s", makefile_path)
        makefile_text = makefile_text.replace("hipcc", "hipcc" + amdgcnspirv_flag)

    # Replace hardcoded /opt/rocm with actual ROCm path
    if "/opt/rocm/" in makefile_text:
        logging.info("Replacing /opt/rocm/ with %s in %s", rocm_path, makefile_path)
        makefile_text = makefile_text.replace("/opt/rocm/", f"{str(rocm_path)}/")

    makefile_path.write_text(makefile_text, encoding="utf-8")


def _select_benchmarks(src_dir: Path) -> List[Path]:
    if HECBENCH_BENCHMARKS:
        names = [bench.strip() for bench in HECBENCH_BENCHMARKS.split(",") if bench.strip()]
        selected = []
        for name in names:
            bench_dir = src_dir / name
            if bench_dir.is_dir():
                selected.append(bench_dir)
            else:
                logging.warning("Benchmark directory not found: %s", bench_dir)
        if not selected:
            raise RuntimeError(f"None of the specified benchmarks found: {names}")
        return selected

    benches = sorted(
        bench for bench in src_dir.iterdir() if bench.is_dir() and bench.name.endswith("-hip")
    )

    if not benches:
        raise RuntimeError(f"No *-hip benchmark directories found in {src_dir}")

    selected = benches[SHARD_INDEX::TOTAL_SHARDS]
    if not selected:
        raise RuntimeError(
            "No benchmarks selected for this shard; "
            f"SHARD_INDEX={SHARD_INDEX + 1}, TOTAL_SHARDS={TOTAL_SHARDS}"
        )

    return selected


def main() -> None:
    env = _setup_env()
    rocm_path = Path(env["ROCM_PATH"])

    with tempfile.TemporaryDirectory(prefix="hecbench_spirv_") as tmp:
        tmp_path = Path(tmp)
        clone_cmd = [
            "git",
            "clone",
            "https://github.com/zjin-lcf/HeCBench",
            "HeCBench",
        ]
        clone_proc = _run(clone_cmd, cwd=tmp_path, env=env)
        if clone_proc.returncode != 0:
            raise RuntimeError("Failed to clone HeCBench repository")

        src_dir = tmp_path / "HeCBench" / "src"
        selected_benchmarks = _select_benchmarks(src_dir)
        logging.info(
            "Selected %d benchmark(s): %s",
            len(selected_benchmarks),
            ", ".join(bench.name for bench in selected_benchmarks),
        )

        results: List[Dict[str, str]] = []
        failures = []
        assertion_failures = []

        for index, bench_dir in enumerate(selected_benchmarks, start=1):
            bench_name = bench_dir.name
            logging.info(
                "Running benchmark %d/%d: %s",
                index,
                len(selected_benchmarks),
                bench_name,
            )

            makefile_path = bench_dir / "Makefile"
            if not makefile_path.is_file():
                failures.append((bench_name, "missing Makefile"))
                results.append({"benchmark": bench_name, "status": "missing Makefile"})
                continue

            _enable_spirv_in_makefile(makefile_path, rocm_path)

            _run(["make", "clean"], cwd=bench_dir, env=env)

            try:
                compile_proc = _run(
                    ["make", "-j2"],
                    cwd=bench_dir,
                    env=env,
                    timeout=COMPILE_TIMEOUT,
                )
            except subprocess.TimeoutExpired:
                failures.append((bench_name, "compile timed out"))
                results.append({"benchmark": bench_name, "status": "compile timed out"})
                continue

            if compile_proc.returncode != 0:
                logging.error("Compilation failed for %s", bench_name)
                logging.error("stdout:\n%s", compile_proc.stdout)
                logging.error("stderr:\n%s", compile_proc.stderr)
                failures.append((bench_name, "compile failed"))
                results.append({"benchmark": bench_name, "status": "compile failed"})
                continue

            try:
                run_proc = _run(
                    ["make", "run"],
                    cwd=bench_dir,
                    env=env,
                    timeout=RUN_TIMEOUT,
                    capture_output=True,
                )
            except subprocess.TimeoutExpired as exception:
                logging.warning(
                    "Benchmark %s exceeded timeout, partial output:\n%s",
                    bench_name,
                    exception.output,
                )
                results.append({"benchmark": bench_name, "status": "timeout (partial results)"})
                failures.append((bench_name, "run timed out"))
                continue

            if "Device-side assertion" in run_proc.stderr:
                logging.error("Benchmark %s failed due to device-side assertion", bench_name)
                logging.error("stdout:\n%s", run_proc.stdout)
                logging.error("stderr:\n%s", run_proc.stderr)
                failures.append((bench_name, "device-side assertion"))
                assertion_failures.append(bench_name)
                results.append({"benchmark": bench_name, "status": "failed (device-side assertion)"})
                continue

            if run_proc.returncode != 0:
                logging.error("Run failed for %s", bench_name)
                logging.error("stdout:\n%s", run_proc.stdout)
                logging.error("stderr:\n%s", run_proc.stderr)
                failures.append((bench_name, "run failed"))
                results.append({"benchmark": bench_name, "status": "run failed"})
                continue

            logging.info("Completed %s successfully", bench_name)
            results.append({"benchmark": bench_name, "status": "passed"})

        summary = {
            "total": len(selected_benchmarks),
            "passed": sum(1 for result in results if result["status"] == "passed"),
            "failed": sum(1 for result in results if "failed" in result["status"] or "timed out" in result["status"] or "compile" in result["status"]),
            "assertion_failures": len(assertion_failures),
            "results": results,
        }
        logging.info("Results summary: %s", json.dumps(summary, indent=2))

        if failures:
            logging.error("hecbench_spirv had %d failure(s): %s", len(failures), failures)
            sys.exit(1)

        logging.info("hecbench_spirv completed successfully")


if __name__ == "__main__":
    main()
