# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run LLVM lit tests (check-llvm, check-clang, check-lld) from pre-built artifacts.

This script is used in CI to execute LLVM lit tests on a runner machine that
received pre-built LLVM tools and lit site configurations as artifacts. The
source test files are obtained via a sparse checkout of the amd-llvm submodule.

Expected environment variables:
  OUTPUT_ARTIFACTS_DIR  - directory where artifacts were extracted (e.g. ./build)

The artifacts directory is expected to contain:
  lib/llvm/bin/         - LLVM tool binaries, examples (from amd-llvm_run)
  lib/llvm/lib/         - LLVM shared libraries (from amd-llvm_lib)
  test/                 - LLVM lit configs including Unit/ (from amd-llvm_test)
  tools/clang/test/     - Clang lit configs including Unit/
  tools/lld/test/       - LLD lit configs including Unit/
  bin/                  - Fuzzers, test tools (from amd-llvm_test build tree)
  lib/                  - Plugin .so files (from amd-llvm_test build tree)
  unittests/            - LLVM gtest binaries
  tools/clang/unittests/ - Clang gtest binaries
  tools/lld/unittests/  - LLD gtest binaries
"""

import logging
import os
import platform
import re
import shlex
import shutil
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
LLVM_LIT = LLVM_TOOLS_DIR / ("llvm-lit.py" if IS_WINDOWS else "llvm-lit.real")


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

    # Read the submodule URL from .gitmodules.
    # The submodule name ("llvm-project") differs from the path ("compiler/amd-llvm").
    result = run_cmd(
        ["git", "config", "-f", ".gitmodules", "submodule.llvm-project.url"],
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
            "llvm/include",
            "llvm/utils",
            "llvm/unittests/DebugInfo",
            "llvm/lib/Target/X86",
            "llvm/lib/Analysis/models",
            "llvm/tools/opt-viewer",
            "llvm/docs/CommandGuide",
            "clang/test",
            "clang/utils",
            "clang/include",
            "clang/lib/Headers",
            "clang/lib/Sema",
            "clang/docs/tools",
            "lld/test",
            "lld/unittests/AsLibELF",
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
    load_config_src_root: str | None = None,
    config_overrides: dict[str, str] | None = None,
) -> None:
    """Fix up a lit.site.cfg.py to use correct absolute paths on the runner.

    The generated lit.site.cfg.py files contain relative paths (via the path()
    helper) computed on the builder machine. On the runner, the directory layout
    is different, so we replace config variable assignments that use path() with
    correct absolute paths.

    If load_config_src_root is given, also fix inline path() calls in the
    lit_config.load_config() invocation (needed when the template uses a raw
    path() call instead of a config variable, as LLD does).

    If config_overrides is given, override boolean/string config values that
    were set on the builder but don't apply on the runner (e.g. features for
    tools we don't ship in artifacts).
    """
    if not cfg_path.exists():
        logging.warning(f"lit site config not found: {cfg_path}")
        return

    content = cfg_path.read_text()

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

    # Some templates (e.g. LLD) use an inline path() call in the
    # lit_config.load_config() invocation rather than a config variable.
    # Replace that path() with the correct source root.
    if load_config_src_root:
        content = re.sub(
            r"(os\.path\.join\()path\(r\"[^\"]*\"\)",
            rf'\g<1>r"{load_config_src_root}"',
            content,
        )

    # Override config values that don't match the runner environment.
    if config_overrides:
        for var_name, new_value in config_overrides.items():
            pattern = rf"(config\.{re.escape(var_name)}\s*=\s*).*"
            replacement = rf"\g<1>{new_value}"
            content = re.sub(pattern, replacement, content)

    cfg_path.write_text(content)
    logging.info(f"Fixed up: {cfg_path}")


def run_lit_tests(
    test_dir: Path,
    label: str,
    extra_args: list[str] | None = None,
) -> int:
    """Run llvm-lit on a test directory and return the exit code."""
    if not test_dir.exists():
        logging.warning(f"Test directory not found: {test_dir}, skipping {label}")
        return 0

    lit_site_cfg = test_dir / "lit.site.cfg.py"
    if not lit_site_cfg.exists():
        logging.warning(f"No lit.site.cfg.py in {test_dir}, skipping {label}")
        return 0

    cmd = [sys.executable, str(LLVM_LIT), str(test_dir), "-v", "--timeout=300"]
    if extra_args:
        cmd.extend(extra_args)
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
        load_config_src_root=str(lld_src),
    )

    # Fix up Unit test (gtest) lit.site.cfg.py files.  The key difference vs
    # the main configs is that *_obj_root must point to where the unit test
    # binaries were extracted (relative to ARTIFACTS_PATH), not to cfg_path's
    # parent like the default path_map assumes.
    fixup_lit_site_cfg(
        llvm_test_dir / "Unit" / "lit.site.cfg.py",
        llvm_src_root=llvm_src,
        llvm_tools_dir=LLVM_TOOLS_DIR,
        llvm_libs_dir=LLVM_LIBS_DIR,
        extra_replacements={
            "llvm_obj_root": str(ARTIFACTS_PATH),
            "shlibdir": str(LLVM_LIBS_DIR),
        },
    )

    fixup_lit_site_cfg(
        clang_test_dir / "Unit" / "lit.site.cfg.py",
        llvm_src_root=llvm_src,
        llvm_tools_dir=LLVM_TOOLS_DIR,
        llvm_libs_dir=LLVM_LIBS_DIR,
        extra_replacements={
            "llvm_obj_root": str(ARTIFACTS_PATH),
            "clang_obj_root": str(ARTIFACTS_PATH / "tools" / "clang"),
            "shlibdir": str(LLVM_LIBS_DIR),
        },
        load_config_src_root=str(clang_src),
    )

    fixup_lit_site_cfg(
        lld_test_dir / "Unit" / "lit.site.cfg.py",
        llvm_src_root=llvm_src,
        llvm_tools_dir=LLVM_TOOLS_DIR,
        llvm_libs_dir=LLVM_LIBS_DIR,
        extra_replacements={
            "lld_obj_root": str(ARTIFACTS_PATH / "tools" / "lld"),
            "lld_src_dir": str(lld_src),
            "shlibdir": str(LLVM_LIBS_DIR),
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

    # Remove the pip-installed lit package so that nested llvm-lit invocations
    # (e.g. inside update_cc_test_checks tests) import the source tree's lit
    # from PYTHONPATH instead of an older pip version that may lack attributes
    # such as LitConfig.update_tests.
    run_cmd(
        [sys.executable, "-m", "pip", "uninstall", "-y", "lit"],
        check=False,
    )

    # Set library search path so tests can find shared libraries.
    # Include rocm_sysdeps/lib for tests that copy binaries to temp dirs
    # (e.g. lld COFF tests) where RPATH $ORIGIN resolution breaks.
    if IS_WINDOWS:
        lib_path_var = "PATH"
    else:
        lib_path_var = "LD_LIBRARY_PATH"
    rocm_sysdeps_lib = ARTIFACTS_PATH / "lib" / "rocm_sysdeps" / "lib"
    extra_lib_dirs = [str(LLVM_LIBS_DIR)]
    if rocm_sysdeps_lib.is_dir():
        extra_lib_dirs.append(str(rocm_sysdeps_lib))
    lib_path = os.environ.get(lib_path_var, "")
    new_lib_path = path_sep.join(extra_lib_dirs)
    os.environ[lib_path_var] = (
        f"{new_lib_path}{path_sep}{lib_path}" if lib_path else new_lib_path
    )

    # Symlink build-tree test binaries (fuzzers, test tools) into the LLVM
    # tools directory so that lit can find them alongside installed tools.
    build_bin_dir = ARTIFACTS_PATH / "bin"
    if build_bin_dir.is_dir():
        for entry in build_bin_dir.iterdir():
            if not entry.is_file():
                continue
            dest = LLVM_TOOLS_DIR / entry.name
            if not dest.exists():
                try:
                    os.symlink(entry, dest)
                except OSError:
                    shutil.copy2(entry, dest)
                logging.info(f"Linked test tool: {entry.name}")

    # The installed llvm-lit is a bash wrapper that sets PYTHONPATH before
    # calling llvm-lit.real.  Some tests (e.g. update_cc_test_checks) invoke
    # ``python <llvm-lit>`` internally which fails because Python cannot parse
    # the bash script.  Replace the wrapper with a copy of the real script.
    llvm_lit_wrapper = LLVM_TOOLS_DIR / "llvm-lit"
    if not IS_WINDOWS and LLVM_LIT.exists() and llvm_lit_wrapper.exists():
        shutil.copy2(LLVM_LIT, llvm_lit_wrapper)
        logging.info("Replaced llvm-lit bash wrapper with Python script")

    # Symlink build-tree plugin shared libraries into the LLVM libs directory
    # so that %shlibdir substitution in lit tests resolves correctly.
    build_lib_dir = ARTIFACTS_PATH / "lib"
    if build_lib_dir.is_dir():
        for entry in build_lib_dir.iterdir():
            if not entry.is_file():
                continue
            dest = LLVM_LIBS_DIR / entry.name
            if not dest.exists():
                try:
                    os.symlink(entry, dest)
                except OSError:
                    shutil.copy2(entry, dest)
                logging.info(f"Linked plugin: {entry.name}")

    # Prevent the Clang driver from auto-discovering ROCm in the artifacts
    # layout.  Without this, driver tests that rely on --sysroot or
    # --rocm-path for self-contained discovery pick up libamdhip64.so from
    # the runner's merged install prefix instead.
    for var in [
        "ROCM_PATH",
        "HIP_PATH",
        "ROCM_DIR",
        "HIP_DIR",
        "HIP_CLANG_PATH",
    ]:
        os.environ.pop(var, None)

    # TestExecuteEmptyEnvironment spawns a child with a completely empty env
    # which means no LD_LIBRARY_PATH.  Since the gtest binary's RPATH points
    # to the builder's path (not the runner's), the child cannot load
    # libLLVM.so.  This is inherent to out-of-tree execution with dynamic
    # linking and cannot be fixed without patching RPATH on every binary.
    rc_llvm = run_lit_tests(
        llvm_test_dir,
        "check-llvm",
        extra_args=["--filter-out", "ProgramEnvTest"],
    )

    # The build targets AMDGPU;X86 only (CLANG_ENABLE_HLSL is off) so hlsl.h
    # is not installed.  Skip .hlsl tests to avoid ~500 false failures.
    # Also filter out Driver tests that are sensitive to the installed layout:
    # the Clang binary finds ROCm relative to its install prefix, which in
    # the merged artifacts tree differs from an in-tree build.
    rc_clang = run_lit_tests(
        clang_test_dir,
        "check-clang",
        extra_args=[
            "--filter-out",
            r"\.hlsl|Driver/A\+A\.c|Driver/rocm-detect\.hip"
            r"|Driver/hip-runtime-libs-linux\.hip"
            r"|Driver/rocm-not-found\.cl",
        ],
    )

    rc_lld = run_lit_tests(lld_test_dir, "check-lld")

    logging.info(
        f"Results: check-llvm={rc_llvm}, check-clang={rc_clang}, check-lld={rc_lld}"
    )

    if rc_llvm != 0 or rc_clang != 0 or rc_lld != 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
