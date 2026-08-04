"""TEST-ONLY (do not merge): diagnose which dependency of torch_hip.dll fails
to load on Windows.

Statically walks the DLL dependency tree via `dumpbin /dependents` and reports,
for every transitive dependency, where it resolves (torch/lib, a rocm_sdk bin
dir, a system dir) or whether it is missing entirely. Then runs an empirical
`import torch` test after adding every rocm_sdk bin dir to the DLL search path,
to determine whether the failure is a search-path / preload gap or a genuinely
absent DLL.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

SYSTEM_PREFIXES = ("api-ms-win-", "ext-ms-win-")
SYSTEM_DLLS = {
    "kernel32.dll", "kernelbase.dll", "ntdll.dll", "msvcp140.dll",
    "vcruntime140.dll", "vcruntime140_1.dll", "ucrtbase.dll", "user32.dll",
    "advapi32.dll", "ole32.dll", "shell32.dll", "combase.dll", "rpcrt4.dll",
    "sechost.dll", "gdi32.dll", "ws2_32.dll", "bcrypt.dll", "dbghelp.dll",
    "python312.dll", "python3.dll", "msvcp140_1.dll", "msvcp140_2.dll",
    "concrt140.dll", "vcomp140.dll", "oleaut32.dll", "crypt32.dll",
}


def torch_lib_dir():
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.origin:
        raise RuntimeError("torch not found via find_spec")
    return Path(spec.origin).parent / "lib"


def rocm_bin_dirs():
    dirs = set()
    try:
        import rocm_sdk
        from rocm_sdk import _dist_info

        shortnames = [n for n, e in _dist_info.ALL_LIBRARIES.items() if e.dll_pattern]
        for sn in shortnames:
            try:
                for p in rocm_sdk.find_libraries(sn):
                    dirs.add(Path(p).parent)
            except Exception:
                pass
    except Exception as e:
        print(f"  (could not enumerate rocm_sdk libraries: {e})")
    return dirs


def dumpbin_deps(dll_path):
    try:
        out = subprocess.run(
            ["dumpbin", "/dependents", str(dll_path)],
            capture_output=True, text=True, timeout=120,
        ).stdout
    except Exception as e:
        return [], f"dumpbin failed: {e}"
    deps = []
    in_section = False
    for line in out.splitlines():
        s = line.strip()
        if "following dependencies" in line:
            in_section = True
            continue
        if in_section:
            if s.lower().endswith(".dll"):
                deps.append(s)
            elif s.startswith("Summary"):
                break
    return deps, None


def locate(name, torch_lib, rocm_dirs):
    low = name.lower()
    if low.startswith(SYSTEM_PREFIXES) or low in SYSTEM_DLLS:
        return "system"
    if (torch_lib / name).exists():
        return "torch/lib"
    for d in rocm_dirs:
        if (d / name).exists():
            return f"rocm_sdk:{d}"
    sys32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    if (sys32 / name).exists():
        return "System32"
    return None


def walk(root_dll, torch_lib, rocm_dirs):
    seen = {}
    missing = []
    order = []
    stack = [(root_dll.name, root_dll)]
    while stack:
        name, fpath = stack.pop()
        key = name.lower()
        if key in seen:
            continue
        loc = locate(name, torch_lib, rocm_dirs)
        seen[key] = loc
        order.append((name, loc))
        if loc is None:
            missing.append(name)
            continue
        if loc in ("system", "System32"):
            continue
        if fpath is None:
            if loc == "torch/lib":
                fpath = torch_lib / name
            elif loc.startswith("rocm_sdk:"):
                fpath = Path(loc.split("rocm_sdk:", 1)[1]) / name
        if fpath is None:
            continue
        deps, err = dumpbin_deps(fpath)
        if err:
            print(f"  ! {name}: {err}")
        for d in deps:
            if d.lower() not in seen:
                stack.append((d, None))
    return order, missing


def empirical_import_test(rocm_dirs):
    add_lines = "\n".join(f"os.add_dll_directory(r'{d}')" for d in rocm_dirs)
    script = (
        "import os\n" + add_lines + "\nimport torch\nprint('IMPORT_OK', torch.__version__)\n"
    )
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def main():
    print("\n" + "=" * 72)
    print("TEST-ONLY Windows DLL dependency diagnostic for torch_hip.dll")
    print("=" * 72)
    torch_lib = torch_lib_dir()
    print(f"torch/lib: {torch_lib}")
    rocm_dirs = sorted(rocm_bin_dirs())
    print(f"rocm_sdk bin dirs ({len(rocm_dirs)}):")
    for d in rocm_dirs:
        print(f"  {d}")

    target = torch_lib / "torch_hip.dll"
    if not target.exists():
        print(f"ERROR: {target} does not exist")
        return

    order, missing = walk(target, torch_lib, set(rocm_dirs))
    print(f"\nDependency tree resolution ({len(order)} unique DLLs):")
    for name, loc in order:
        print(f"  {name:44s} -> {loc if loc is not None else '*** NOT FOUND ***'}")

    print("\n--- SUMMARY ---")
    if missing:
        print("UNRESOLVED DLLs (candidate root cause):")
        for m in missing:
            print(f"  {m}")
    else:
        print("Every DLL in the tree resolves to some directory (torch/lib, "
              "rocm_sdk bin, or system).")

    print("\nEmpirical test: add all rocm_sdk bin dirs to DLL search then import torch:")
    rc, out, err = empirical_import_test(rocm_dirs)
    print(f"  return code: {rc}")
    print(f"  stdout: {out.strip()}")
    if rc != 0:
        print(f"  stderr tail:\n{err.strip()[-2500:]}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        print("Diagnostic crashed (non-fatal):")
        traceback.print_exc()
