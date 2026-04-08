# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run LLVM lit tests (check-llvm, check-clang, check-lld) from pre-built artifacts.

This script is used in CI to execute LLVM lit tests on a runner machine that
received pre-built LLVM tools and lit site configurations as artifacts. The
source test files are obtained via a sparse checkout of the amd-llvm submodule.

Expected environment variables:
  OUTPUT_ARTIFACTS_DIR  - directory where artifacts were extracted (e.g. ./build)

The artifacts directory is expected to contain:
  lib/llvm/bin/         - LLVM tool binaries (llvm-lit, FileCheck, opt, etc.)
  test/lit.site.cfg.py  - LLVM lit site config (from build tree)
  tools/clang/test/lit.site.cfg.py - Clang lit site config
  tools/lld/test/lit.site.cfg.py   - LLD lit site config
"""

import logging
import os
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

IS_WINDOWS = platform.system() == "Windows"

SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR", "")
if not OUTPUT_ARTIFACTS_DIR:
    logging.error("OUTPUT_ARTIFACTS_DIR is not set")
    sys.exit(1)

ARTIFACTS_PATH = Path(OUTPUT_ARTIFACTS_DIR).resolve()
LLVM_TOOLS_DIR = ARTIFACTS_PATH / "lib" / "llvm" / "bin"
LLVM_LIBS_DIR = ARTIFACTS_PATH / "lib" / "llvm" / "lib"
LLVM_LIT = LLVM_TOOLS_DIR / ("llvm-lit.py" if IS_WINDOWS else "llvm-lit")


def run_cmd(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command with logging."""
    logging.info(f"++ Exec [{os.getcwd()}]$ {shlex.join(cmd)}")
    return subprocess.run(cmd, **kwargs)


def sparse_checkout_submodule() -> Path:
    """Sparse-checkout the amd-llvm source to get test directories.

    Uses a direct sparse clone of the llvm-project repo (rather than git
    submodule update) so we can set up sparse-checkout patterns before any
    blobs are fetched.

    Returns the path to the checked-out amd-llvm directory.
    """
    amd_llvm_dir = THEROCK_DIR / "compiler" / "amd-llvm"

    if (amd_llvm_dir / "llvm" / "test").exists():
        logging.info("amd-llvm source already present, skipping checkout")
        return amd_llvm_dir

    logging.info("Sparse-checking out amd-llvm source for test files...")

    # Read the submodule URL from .gitmodules
    result = run_cmd(
        [
            "git",
            "config",
            "-f",
            ".gitmodules",
            "submodule.compiler/amd-llvm.url",
        ],
        cwd=THEROCK_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    submodule_url = result.stdout.strip()
    logging.info(f"Submodule URL: {submodule_url}")

    # Get the pinned commit from the superproject tree
    result = run_cmd(
        ["git", "ls-tree", "HEAD", "compiler/amd-llvm"],
        cwd=THEROCK_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    target_commit = result.stdout.split()[2]
    logging.info(f"Target commit: {target_commit}")

    # Clone without checkout (partial + sparse), fetching only tree objects
    run_cmd(
        [
            "git",
            "clone",
            "--no-checkout",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            submodule_url,
            str(amd_llvm_dir),
        ],
        check=True,
    )

    # Configure which directories to include in the sparse checkout
    run_cmd(
        [
            "git",
            "-C",
            str(amd_llvm_dir),
            "sparse-checkout",
            "set",
            "llvm/test",
            "llvm/utils/lit",
            "clang/test",
            "lld/test",
            "lld/ELF",
            "lld/COFF",
            "lld/MachO",
            "lld/MinGW",
            "lld/include",
            "third-party/unittest",
        ],
        check=True,
    )

    # Fetch the exact commit pinned by TheRock and check it out
    run_cmd(
        [
            "git",
            "-C",
            str(amd_llvm_dir),
            "fetch",
            "--depth",
            "1",
            "origin",
            target_commit,
        ],
        check=True,
    )
    run_cmd(
        ["git", "-C", str(amd_llvm_dir), "checkout", target_commit],
        check=True,
    )

    return amd_llvm_dir


def fixup_lit_site_cfg(
    cfg_path: Path,
    *,
    llvm_src_root: Path,
    llvm_tools_dir: Path,
    llvm_libs_dir: Path,
    extra_replacements: dict[str, str] | None = None,
) -> None:
    """Fix up a lit.site.cfg.py to use correct absolute paths on the runner.

    The generated lit.site.cfg.py files contain relative paths (via the path()
    helper) computed on the builder machine. On the runner, the directory layout
    is different, so we replace the path() function with one that uses a
    path-mapping table to redirect known directories to their runner locations.
    """
    if not cfg_path.exists():
        logging.warning(f"lit site config not found: {cfg_path}")
        return

    content = cfg_path.read_text()

    old_path_func = (
        "def path(p):\n"
        "    if not p: return ''\n"
        "    return os.path.join(os.path.dirname(os.path.abspath(__file__)), p)"
    )

    path_map = {
        "llvm_src_root": str(llvm_src_root),
        "llvm_obj_root": str(cfg_path.parent.parent),
        "llvm_tools_dir": str(llvm_tools_dir),
        "llvm_lib_dir": str(llvm_libs_dir),
        "llvm_libs_dir": str(llvm_libs_dir),
        "llvm_shlib_dir": str(llvm_libs_dir),
        "lit_tools_dir": str(llvm_tools_dir),
    }
    if extra_replacements:
        path_map.update(extra_replacements)

    for var_name, new_value in path_map.items():
        pattern = (
            rf"(config\.{re.escape(var_name)}\s*=\s*)"
            rf'(?:lit_config\.substitute\()?path\(r"[^"]*"\)\)?'
        )
        replacement = rf'\g<1>r"{new_value}"'
        content = re.sub(pattern, replacement, content)

    # Fix Python executable path
    content = re.sub(
        r'config\.python_executable\s*=\s*"[^"]*"',
        f'config.python_executable = r"{sys.executable}"',
        content,
    )

    # Fix host compiler paths to use system defaults
    if IS_WINDOWS:
        host_cc, host_cxx = "cl", "cl"
    else:
        host_cc, host_cxx = "cc", "c++"
    content = re.sub(
        r'config\.host_cc\s*=\s*"[^"]*"',
        f'config.host_cc = "{host_cc}"',
        content,
    )
    content = re.sub(
        r'config\.host_cxx\s*=\s*"[^"]*"',
        f'config.host_cxx = "{host_cxx}"',
        content,
    )

    cfg_path.write_text(content)
    logging.info(f"Fixed up: {cfg_path}")


def run_lit_tests(test_dir: Path, label: str) -> int:
    """Run llvm-lit on a test directory and return the exit code."""
    if not test_dir.exists():
        logging.warning(f"Test directory not found: {test_dir}, skipping {label}")
        return 0

    lit_site_cfg = test_dir / "lit.site.cfg.py"
    if not lit_site_cfg.exists():
        logging.warning(f"No lit.site.cfg.py in {test_dir}, skipping {label}")
        return 0

    if IS_WINDOWS:
        cmd = [sys.executable, str(LLVM_LIT), str(test_dir), "-v", "--timeout=300"]
    else:
        cmd = [str(LLVM_LIT), str(test_dir), "-v", "--timeout=300"]
    logging.info(f"=== Running {label} ===")
    result = run_cmd(cmd)
    logging.info(f"=== {label} exited with code {result.returncode} ===")
    return result.returncode


def main() -> int:
    if not LLVM_LIT.exists():
        logging.error(f"llvm-lit not found at {LLVM_LIT}")
        return 1

    amd_llvm_dir = sparse_checkout_submodule()
    llvm_src = amd_llvm_dir / "llvm"
    clang_src = amd_llvm_dir / "clang"
    lld_src = amd_llvm_dir / "lld"

    # The lit.site.cfg.py files were extracted from the build tree artifact
    # to ARTIFACTS_PATH/{test/, tools/clang/test/, tools/lld/test/}.
    llvm_test_dir = ARTIFACTS_PATH / "test"
    clang_test_dir = ARTIFACTS_PATH / "tools" / "clang" / "test"
    lld_test_dir = ARTIFACTS_PATH / "tools" / "lld" / "test"

    # Fix up lit.site.cfg.py files with correct runner paths
    fixup_lit_site_cfg(
        llvm_test_dir / "lit.site.cfg.py",
        llvm_src_root=llvm_src,
        llvm_tools_dir=LLVM_TOOLS_DIR,
        llvm_libs_dir=LLVM_LIBS_DIR,
    )

    fixup_lit_site_cfg(
        clang_test_dir / "lit.site.cfg.py",
        llvm_src_root=llvm_src,
        llvm_tools_dir=LLVM_TOOLS_DIR,
        llvm_libs_dir=LLVM_LIBS_DIR,
        extra_replacements={
            "clang_obj_root": str(clang_test_dir.parent),
            "clang_src_dir": str(clang_src),
            "clang_tools_dir": str(LLVM_TOOLS_DIR),
            "clang_lib_dir": str(LLVM_LIBS_DIR),
            "llvm_external_lit": "",
        },
    )

    fixup_lit_site_cfg(
        lld_test_dir / "lit.site.cfg.py",
        llvm_src_root=llvm_src,
        llvm_tools_dir=LLVM_TOOLS_DIR,
        llvm_libs_dir=LLVM_LIBS_DIR,
        extra_replacements={
            "lld_obj_root": str(lld_test_dir.parent),
            "lld_libs_dir": str(LLVM_LIBS_DIR),
            "lld_tools_dir": str(LLVM_TOOLS_DIR),
        },
    )

    # Add llvm/utils/lit to PYTHONPATH so llvm-lit can find the lit package
    path_sep = ";" if IS_WINDOWS else ":"
    lit_python_path = amd_llvm_dir / "llvm" / "utils" / "lit"
    if lit_python_path.exists():
        env_pythonpath = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = (
            f"{lit_python_path}{path_sep}{env_pythonpath}"
            if env_pythonpath
            else str(lit_python_path)
        )

    # Set library search path so tests can find shared libraries
    if IS_WINDOWS:
        lib_path_var = "PATH"
    else:
        lib_path_var = "LD_LIBRARY_PATH"
    lib_path = os.environ.get(lib_path_var, "")
    os.environ[lib_path_var] = (
        f"{LLVM_LIBS_DIR}{path_sep}{lib_path}" if lib_path else str(LLVM_LIBS_DIR)
    )

    rc_llvm = run_lit_tests(llvm_test_dir, "check-llvm")
    rc_clang = run_lit_tests(clang_test_dir, "check-clang")
    rc_lld = run_lit_tests(lld_test_dir, "check-lld")

    logging.info(
        f"Results: check-llvm={rc_llvm}, check-clang={rc_clang}, check-lld={rc_lld}"
    )

    if rc_llvm != 0 or rc_clang != 0 or rc_lld != 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
