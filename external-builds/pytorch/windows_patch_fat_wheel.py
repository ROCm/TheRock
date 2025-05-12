"""Given a windows torch .whl, patches it so that it contains ROCm.

This lets us have a standalone PyTorch wheel (albeit a big one) that does not need
a separate ROCm installation. While we have better means of packaging, this approach
has the benefit of simplicity, since you can build ROCm separately, then build PyTorch,
then smash them together vs dealing with 'packaging' stuff explicitly.
"""

import argparse
from pathlib import Path
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "build_tools"))
from _therock_utils.pattern_match import PatternMatcher


def run(args):
    wheel_input_path: Path = args.wheel_path
    wheel_output_path: Path = args.output_path
    wheel_input_fname = wheel_input_path.name
    if not args.output_path:
        # Derive from the wheel path.
        distribution, version, *rest = wheel_input_fname.split("-")
        if "+" in version:
            version, version_extra = version.split("+", maxsplit=2)
            version_extra = f"rocm_{version_extra}"
        else:
            version_extra = "rocm"
        version = f"{version}+{version_extra}"
        wheel_output_fname = "-".join([distribution, version] + rest)
        wheel_output_path = wheel_input_path.with_name(wheel_output_fname)

    print(f"Processing {wheel_input_path} to {wheel_output_path}")
    with tempfile.TemporaryDirectory(dir=wheel_output_path.parent) as td:
        process_wheel(wheel_input_path, wheel_output_path, args.rocm_path, Path(td))


def process_wheel(
    wheel_input_path: Path, wheel_output_path: Path, rocm_path: Path, temp_dir: Path
):
    print("Extracting wheel...")
    with zipfile.ZipFile(wheel_input_path, "r") as zip_in:
        zip_in.extractall(temp_dir)

    init_py = temp_dir / "torch" / "__init__.py"
    print("Patching __init__.py")
    patch_init_py(init_py)

    print("Copying rocm")
    pm = PatternMatcher(
        excludes=[
            # The full compiler is big and not needed. Strip.
            "lib/llvm/**",
            # On windows, outside of clients, sysdeps are static.
            "lib/rocm_sysdeps/**",
            # Currently, outside of clients, we don't need host math libs.
            "lib/host-math/**",
            # Don't need any EXEs
            "bin/*.exe",
        ]
    )
    pm.add_basedir(rocm_path)
    pm.copy_to(destdir=temp_dir / "torch" / "lib" / "rocm")

    print(f"Saving wheel to {wheel_output_path}")
    if wheel_output_path.exists():
        wheel_output_path.unlink()
    dest_pm = PatternMatcher()
    dest_pm.add_basedir(temp_dir)
    with zipfile.ZipFile(wheel_output_path, "w") as zip_out:
        for relpath, direntry in dest_pm.matches():
            if not direntry.is_dir() and not direntry.is_symlink():
                zip_out.write(direntry.path, relpath)


def patch_init_py(init_py_path: Path):
    lines = init_py_path.read_text().splitlines(keepends=True)
    with open(init_py_path, "w") as out:
        for line in lines:
            if "for dll_path in dll_paths:" in line:
                indent_count = len(line) - len(line.lstrip())
                indent = line[0:indent_count]
                patch_line = f"{indent}dll_paths.insert(0, os.path.join(os.path.join(th_dll_path, 'rocm', 'bin')))\n"
                print("Insert:")
                print(f"+{patch_line}")
                print(f" {line}")
                out.write(patch_line)
            out.write(line)


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("wheel_path", type=Path, help="Path to wheel file to alter")
    p.add_argument(
        "rocm_path", type=Path, help="Path to build/dist/rocm or equiv to embed"
    )
    p.add_argument("--output-path", type=Path, help="Optional path of the output wheel")
    args = p.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
