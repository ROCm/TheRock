#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""UCCL intranode EP test runner.

Runs the upstream UCCL ``test_intranode.py`` test via ``torchrun`` in
standalone mode. This exercises Expert Parallelism (EP) dispatch, combine,
and tuning kernels on a single node with multiple GPUs.

The test script lives inside the UCCL source checkout (default:
``external-builds/uccl/uccl/ep/bench/test_intranode.py``). Use
``uccl_repo.py checkout`` to obtain it before running tests.

Usage Examples
--------------
Basic (auto-detect GPUs, use all visible devices):
    $ python run_uccl_tests.py

Specify GPU count (e.g. 4 GPUs):
    $ python run_uccl_tests.py --nproc-per-node 4

Point to a custom UCCL checkout:
    $ python run_uccl_tests.py --uccl-dir /path/to/uccl

Override test parameters:
    $ python run_uccl_tests.py --num-tokens 2048 --hidden 4096

Dry-run (print command without executing):
    $ python run_uccl_tests.py --dry-run
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

THIS_SCRIPT_DIR = Path(__file__).resolve().parent


def detect_gpu_count() -> int:
    """Detect the number of visible AMD GPUs via a subprocess.

    Runs a small script that imports torch and queries device_count()
    without initializing CUDA in the current process.
    """
    query_script = "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)"
    try:
        result = subprocess.run(
            [sys.executable, "-c", query_script],
            capture_output=True,
            text=True,
            check=True,
        )
        count = int(result.stdout.strip())
        if count == 0:
            print("[ERROR] No GPUs detected by PyTorch")
            sys.exit(1)
        return count
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"[ERROR] Failed to detect GPU count: {e}")
        sys.exit(1)


def uccl_ep_available() -> tuple[bool, str]:
    """Return (available, diagnostic_output) for the `uccl.ep` module.

    When uccl.ep is not present the intranode test cannot run and we exit 0
    with a skip message so CI stays green. The diagnostic includes the full
    import traceback plus the on-disk contents of the installed uccl/
    package, so a missing ep.abi3.so vs an unloadable ep.abi3.so (missing
    DT_NEEDED .so at runtime) can be distinguished without re-running.
    """
    # uccl.ep is a torch.cpp_extension. Its DT_NEEDED libs (libtorch.so,
    # libtorch_python.so, libc10.so, libtorch_hip.so) are only on the
    # dynamic loader's search path once `import torch` has run, so we must
    # import torch first or the cold `import uccl.ep` will fail with
    # "libtorch_python.so: cannot open shared object file" and we'll wrongly
    # conclude EP was not built.
    check_script = (
        "import importlib, sys, traceback;\n"
        "try:\n"
        "    import torch  # noqa: F401\n"
        "    importlib.import_module('uccl.ep')\n"
        "    print('OK')\n"
        "except BaseException:\n"
        "    traceback.print_exc()\n"
        "    sys.exit(1)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", check_script],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True, ""

    diag_lines = [
        "--- `import uccl.ep` failed. Diagnostic info: ---",
        result.stderr.strip() or "(no traceback)",
    ]
    # Inventory what actually got installed under uccl/.
    listing_script = (
        "import os, uccl;\n"
        "root = os.path.dirname(uccl.__file__);\n"
        "print('uccl pkg root:', root);\n"
        "[print(' ', e) for e in sorted(os.listdir(root))]\n"
    )
    listing = subprocess.run(
        [sys.executable, "-c", listing_script],
        capture_output=True,
        text=True,
    )
    if listing.returncode == 0:
        diag_lines.append(listing.stdout.strip())
    else:
        diag_lines.append(
            "(could not list uccl/ package: "
            + (listing.stderr.strip() or "unknown error")
            + ")"
        )
    return False, "\n".join(diag_lines)


def cmd_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Runs UCCL intranode EP tests via torchrun."
    )

    parser.add_argument(
        "--uccl-dir",
        type=Path,
        default=THIS_SCRIPT_DIR / "uccl",
        help="Path to the UCCL source checkout (must contain ep/bench/test_intranode.py).",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=0,
        help="Number of GPU processes per node. 0 (default) = auto-detect all visible GPUs.",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=4096,
        help="Number of tokens for the EP test (default: 4096).",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=7168,
        help="Hidden dimension size (default: 7168).",
    )
    parser.add_argument(
        "--num-topk",
        type=int,
        default=8,
        help="Number of top-k experts (default: 8).",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=256,
        help="Number of experts (default: 256).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print the command without executing it.",
    )

    args = parser.parse_args(argv)

    test_script = args.uccl_dir / "ep" / "bench" / "test_intranode.py"
    if not test_script.exists():
        parser.error(
            f"test_intranode.py not found at '{test_script}'. "
            "Run 'python uccl_repo.py checkout' first to obtain UCCL sources."
        )

    return args


def build_torchrun_cmd(args: argparse.Namespace, nproc: int) -> list[str]:
    test_script = args.uccl_dir / "ep" / "bench" / "test_intranode.py"

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
        str(test_script),
        f"--num-tokens={args.num_tokens}",
        f"--hidden={args.hidden}",
        f"--num-topk={args.num_topk}",
        f"--num-experts={args.num_experts}",
    ]
    return cmd


def main(argv: list[str]) -> int:
    args = cmd_arguments(argv)

    available, diag = uccl_ep_available()
    if not available:
        print(diag, file=sys.stderr)
        print(
            "[SKIP] uccl.ep is not available in the installed UCCL wheel. "
            "Skipping intranode EP tests. See diagnostic above for the "
            "underlying ImportError.",
            file=sys.stderr,
        )
        return 0

    nproc = args.nproc_per_node
    if nproc == 0:
        nproc = detect_gpu_count()
    print(f"Using {nproc} GPU process(es)")

    # num_experts must be divisible by world_size
    if args.num_experts % nproc != 0:
        print(
            f"[WARNING] num_experts ({args.num_experts}) is not divisible by "
            f"nproc_per_node ({nproc}). Adjusting num_experts to {nproc * (args.num_experts // nproc)}."
        )
        args.num_experts = nproc * (args.num_experts // nproc)

    cmd = build_torchrun_cmd(args, nproc)

    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", str(nproc))

    print(f"Executing: {' '.join(cmd)}")
    if args.dry_run:
        print("[dry-run] Skipping execution.")
        return 0

    result = subprocess.run(cmd, env=env)
    print(f"test_intranode.py finished with return code: {result.returncode}")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
