#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Prepare relocatable DASan lit tests for an installed TheRock distribution.

The source root is the LLVM monorepo, and llvm-build is its TheRock build tree.
Only tests, lit's Python package, and relocated configuration are copied. Run
the resulting llvm-lit.py with -D rocm_root=/path/to/dist/rocm and suite/.
The existing dasan_smoke.py can use those paths as --llvm-lit and --lit-suite.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

PATH_FIELDS = {
    "llvm_src_root",
    "llvm_obj_root",
    "compiler_rt_src_root",
    "compiler_rt_obj_root",
    "llvm_tools_dir",
    "lit_tools_dir",
    "llvm_shlib_dir",
    "gnu_ld_executable",
    "gold_executable",
    "clang",
    "python_executable",
    "compiler_rt_output_dir",
    "compiler_rt_bindir",
    "compiler_rt_libdir",
    "hip_lib_dir",
}


SUPPORT = r"""import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def verify_sources(bundle):
    manifest = json.loads((bundle / "manifest.json").read_text())
    for relative, expected in manifest["source_sha256"].items():
        actual = hashlib.sha256((bundle / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError("Bundle source hash mismatch: " + relative)


def configure_paths(config, lit_config, bundle, target_triple):
    root_arg = lit_config.params.get("rocm_root") or os.environ.get("DASAN_ROCM_ROOT")
    if not root_arg:
        lit_config.fatal("Pass -D rocm_root=/path/to/dist/rocm (or set DASAN_ROCM_ROOT)")
    root = Path(root_arg).resolve()
    hip_parameter = lit_config.params.get("hip_lib_path")
    if hip_parameter and Path(hip_parameter).resolve() != (root / "lib").resolve():
        lit_config.fatal("hip_lib_path must point inside the selected distribution's lib directory")
    prefix = root / "lib/llvm"
    bindir = prefix / "bin"
    required = ("clang", "FileCheck", "not", "llvm-symbolizer", "llvm-readelf",
                "ld.lld", "clang-linker-wrapper", "clang-offload-bundler")
    for name in required:
        tool = bindir / name
        if not tool.is_file() or not tool.resolve().is_relative_to(root):
            lit_config.fatal("Missing installed tool (or external symlink): " + str(tool))
    clang = bindir / "clang"
    library_dirs = [root / "lib", prefix / "lib", root / "lib/rocm_sysdeps/lib"]
    library_dirs.extend(p for p in (prefix / "lib").glob("*-linux-*") if p.is_dir())
    loader_path = os.pathsep.join(str(p) for p in library_dirs if p.is_dir())
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = loader_path
    environment["PATH"] = str(bindir) + os.pathsep + "/usr/bin:/bin"
    environment["CLANG_NO_DEFAULT_CONFIG"] = "1"
    # Common lit configuration also invokes clang through subprocess helpers
    # that inherit the Python process environment rather than config.environment.
    for name in ("LD_LIBRARY_PATH", "PATH", "CLANG_NO_DEFAULT_CONFIG"):
        os.environ[name] = environment[name]

    def clang_path(flag):
        value = subprocess.check_output(
            [str(clang), "--target=" + target_triple, flag],
            env=environment, text=True).strip()
        result = Path(value).resolve()
        if not result.is_dir() or not result.is_relative_to(root):
            lit_config.fatal("Clang resolved outside installed distribution: " + value)
        return result

    resource = clang_path("-print-resource-dir")
    runtime = clang_path("-print-runtime-dir")
    for name in ("libclang_rt.dasan.a", "libclang_rt.dasan-preinit.a", "libclang_rt.dasan.so"):
        if not (runtime / name).is_file():
            lit_config.fatal("Missing host DASan runtime: " + str(runtime / name))
    device = resource / "lib/amdgpu-amd-amdhsa/libclang_rt.dasan.a"
    if not device.is_file():
        lit_config.fatal("Missing device DASan runtime: " + str(device))
    for path in (runtime / "libclang_rt.dasan.a", runtime / "libclang_rt.dasan.so", device):
        if not path.resolve().is_relative_to(root):
            lit_config.fatal("Runtime symlink leaves installed distribution: " + str(path))
    config.environment["LD_LIBRARY_PATH"] = loader_path
    config.environment["PATH"] = environment["PATH"]
    config.environment["CLANG_NO_DEFAULT_CONFIG"] = "1"
    config.environment["LLVM_SYMBOLIZER_PATH"] = str(bindir / "llvm-symbolizer")
    work = bundle / "work"
    work.mkdir(exist_ok=True)
    config.test_exec_root = str(work)
    lit_config.note("Installed DASan compiler: " + str(clang))
    lit_config.note("Installed DASan resource directory: " + str(resource))
    return {
        "llvm_src_root": str(bundle / "src/llvm"),
        "llvm_obj_root": str(prefix),
        "compiler_rt_src_root": str(bundle / "src/compiler-rt"),
        "compiler_rt_obj_root": str(work),
        "llvm_tools_dir": str(bindir),
        "lit_tools_dir": str(bindir),
        "llvm_shlib_dir": str(prefix / "lib"),
        "gnu_ld_executable": shutil.which("ld.bfd", path="/usr/bin:/bin") or "",
        "gold_executable": shutil.which("ld.gold", path="/usr/bin:/bin") or "",
        "clang": str(clang),
        "python_executable": sys.executable,
        "compiler_rt_output_dir": str(resource),
        "compiler_rt_bindir": str(bindir),
        "compiler_rt_libdir": str(runtime),
        "hip_lib_dir": str(root / "lib"),
    }
"""


RUNNER = """#!/usr/bin/env python3
from pathlib import Path
import sys

bundle = Path(__file__).resolve().parent
sys.path.insert(0, str(bundle / "python"))
from bundle_support import verify_sources
verify_sources(bundle)
from lit.main import main
main()
"""


class RelocateConfig(ast.NodeTransformer):
    """Change generated path expressions, preserving all feature/default logic."""

    def __init__(self, common: bool):
        self.common = common
        self.replaced_fields: set[str] = set()
        self.loads = 0

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if (
            self.common
            and isinstance(node.func, ast.Name)
            and node.func.id == "set_default"
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value in PATH_FIELDS
        ):
            key = node.args[0].value
            node.args[1] = ast.parse(f"_bundle_paths[{key!r}]", mode="eval").body
            self.replaced_fields.add(key)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "lit_config"
            and node.func.attr == "load_config"
        ):
            self.loads += 1
            if self.common:
                relative = "src/compiler-rt/test/lit.common.cfg.py"
            elif self.loads == 1:
                relative = "config/lit.common.configured"
            else:
                relative = "src/compiler-rt/test/dasan/lit.cfg.py"
            node.args[1] = ast.parse(f"str(_bundle / {relative!r})", mode="eval").body
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        if not self.common and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "config"
            and target.attr == "dasan_lit_source_dir"
            for target in node.targets
        ):
            node.value = ast.parse(
                'str(_bundle / "src/compiler-rt/test/dasan")', mode="eval"
            ).body
        return node


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relocate_config(source: Path, destination: Path, common: bool) -> None:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    transformer = RelocateConfig(common)
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    expected_loads = 1 if common else 2
    if transformer.loads != expected_loads:
        raise ValueError(f"Unexpected generated configuration load graph: {source}")
    preamble = (
        "# Relocated generated configuration; original preserved in provenance/.\n"
        "from pathlib import Path\n"
        "import sys\n"
        "_bundle = Path(__file__).resolve().parents[1]\n"
        'sys.path.insert(0, str(_bundle / "python"))\n'
    )
    if common:
        if transformer.replaced_fields != PATH_FIELDS:
            raise ValueError(
                f"Missing configured path fields: {PATH_FIELDS - transformer.replaced_fields}"
            )
        target_triple = next(
            ast.literal_eval(node.args[1])
            for node in tree.body
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            for node in [node.value]
            if isinstance(node.func, ast.Name)
            and node.func.id == "set_default"
            and ast.literal_eval(node.args[0]) == "target_triple"
        )
        preamble += (
            "from bundle_support import configure_paths\n"
            f"_bundle_paths = configure_paths(config, lit_config, _bundle, {target_triple!r})\n"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(preamble + ast.unparse(tree) + "\n", encoding="utf-8")
    compile(destination.read_text(encoding="utf-8"), str(destination), "exec")


def prepare(args: argparse.Namespace) -> None:
    source = args.source_root.resolve()
    build = args.llvm_build.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    host_test = build / "runtimes/runtimes-bins/compiler-rt/test"
    sites = list((host_test / "dasan").glob("*LinuxConfig/lit.site.cfg.py"))
    if len(sites) != 1:
        raise ValueError(
            f"Expected one Linux host DASan configuration, found {len(sites)}"
        )
    common = host_test / "lit.common.configured"
    required = [
        common,
        source / "compiler-rt/test/lit.common.cfg.py",
        source / "compiler-rt/test/Inputs/hip.h",
        source / "llvm/utils/lit/lit/main.py",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}

    def copy_file(original: Path, relative: Path) -> None:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
        original_hash = sha256(original)
        if sha256(destination) != original_hash:
            raise ValueError(f"Copy verification failed: {original}")
        copied[relative.as_posix()] = original_hash

    for tree, relative in (
        (source / "compiler-rt/test/dasan", Path("src/compiler-rt/test/dasan")),
        (source / "llvm/utils/lit/lit", Path("python/lit")),
    ):
        for original in sorted(tree.rglob("*")):
            if original.is_file() and "__pycache__" not in original.parts:
                copy_file(original, relative / original.relative_to(tree))
    copy_file(
        source / "compiler-rt/test/Inputs/hip.h",
        Path("src/compiler-rt/test/Inputs/hip.h"),
    )
    copy_file(
        source / "compiler-rt/test/lit.common.cfg.py",
        Path("src/compiler-rt/test/lit.common.cfg.py"),
    )
    copy_file(common, Path("provenance/lit.common.configured.original"))
    copy_file(sites[0], Path("provenance/lit.site.cfg.py.original"))
    (output / "src/llvm").mkdir(parents=True, exist_ok=True)
    relocate_config(common, output / "config/lit.common.configured", common=True)
    relocate_config(sites[0], output / "suite/lit.site.cfg.py", common=False)
    (output / "python/bundle_support.py").write_text(SUPPORT, encoding="utf-8")
    (output / "llvm-lit.py").write_text(RUNNER, encoding="utf-8")
    manifest = {
        "source_sha256": copied,
        "configuration": "Relocated generated host configuration; test sources unchanged",
        "source_root": str(source),
        "llvm_build": str(build),
        "initial_rocm_root": str(args.rocm_root.resolve()),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    command = [
        sys.executable,
        str(output / "llvm-lit.py"),
        "--show-tests",
        "-D",
        "rocm_root=" + str(args.rocm_root.resolve()),
        str(output / "suite"),
    ]
    print("Bundle prepared. Auditing installed tools and runtime paths...", flush=True)
    subprocess.run(command, check=True)
    print(f"Verified {len(copied)} source/configuration hashes; bundle: {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, required=True, help="LLVM monorepo source directory"
    )
    parser.add_argument(
        "--llvm-build",
        type=Path,
        required=True,
        help="TheRock compiler/amd-llvm/build directory",
    )
    parser.add_argument(
        "--rocm-root",
        type=Path,
        required=True,
        help="Built dist/rocm directory to audit",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New or empty test bundle directory"
    )
    args = parser.parse_args()
    try:
        prepare(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"DASan bundle preparation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
