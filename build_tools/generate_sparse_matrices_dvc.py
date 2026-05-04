#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""One-time script to generate pre-converted sparse test matrices and track them with DVC.

Downloads matrices from SuiteSparse, compiles mtx2csr from the rocsparse/hipsparse
deps, converts to .csr/.bin, and runs `dvc add` on each result.

Usage:
    python build_tools/generate_sparse_matrices_dvc.py [--push]

Run from the TheRock repo root. Requires: C++ compiler (g++ or amdclang++), tar, dvc.
Pass --push to also run `dvc push` after adding files.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

THEROCK_DIR = Path(__file__).resolve().parent.parent
ROCM_LIBRARIES_DIR = THEROCK_DIR / "rocm-libraries"

# fmt: off
# Matrices used by rocsparse (24 matrices -> .csr format)
ROCSPARSE_MATRICES = [
    ("SNAP/amazon0312",       "f567e5f5029d052e3004bc69bb3f13f5"),
    ("Muite/Chebyshev4",      "e39879103dafab21f4cf942e0fe42a85"),
    ("FEMLAB/sme3Dc",         "a95eee14d980a9cfbbaf5df4a3c64713"),
    ("Williams/webbase-1M",   "2d4c239daad6f12d66a1e6a2af44cbdb"),
    ("Bova/rma10",            "a899a0c48b9a58d081c52ffd88a84955"),
    ("JGD_BIBD/bibd_22_8",   "455d5b699ea10232bbab5bc002219ae6"),
    ("Williams/mac_econ_fwd500", "f1b0e56fbb75d1d6862874e3d7d33060"),
    ("Williams/mc2depi",      "8c8633eada6455c1784269b213c85ea6"),
    ("Hamm/scircuit",         "3e62f7ea83914f7e20019aefb2a5176f"),
    ("Sandia/ASIC_320k",      "fcfaf8a25c8f49b8d29f138f3c65c08f"),
    ("GHS_psdef/bmwcra_1",    "8a3cf5448a4fe73dcbdb5a16b326715f"),
    ("HB/nos1",               "b203f7605cb1f20f83280061068f7ec7"),
    ("HB/nos2",               "b0f812ffcc9469f0bf9be701205522c4"),
    ("HB/nos3",               "f185514062a0eeabe86d2909275fe1dc"),
    ("HB/nos4",               "04b781415202db404733ca0c159acbef"),
    ("HB/nos5",               "c98e35f1cfd1ee8177f37bdae155a6e7"),
    ("HB/nos6",               "c39375226aa5c495293003a5f637598f"),
    ("HB/nos7",               "9a6481268847e6cf0d70671f2ff1ddcd"),
    ("DNVS/shipsec1",         "73372e7d6a0848f8b19d64a924fab73e"),
    ("Cote/mplate",           "ad5963d0a39a943fcb0dc2b119d5b22a"),
    ("Bai/qc2534",            "fda33f178963fbb39dfc8c051fd0279e"),
    ("Chevron/Chevron2",      "c093666487879a4e44409eb7be1c0348"),
    ("Chevron/Chevron3",      "5e784e1f8c6341287a2842bd188b347a"),
    ("Chevron/Chevron4",      "01e49e63fa0ac2204baef0f5f33974ad"),
]

# Matrices used by hipsparse (19 matrices -> .bin format)
# Subset of rocsparse matrices (same source .mtx, different binary format)
HIPSPARSE_MATRICES = [
    ("SNAP/amazon0312",       "f567e5f5029d052e3004bc69bb3f13f5"),
    ("Muite/Chebyshev4",      "e39879103dafab21f4cf942e0fe42a85"),
    ("FEMLAB/sme3Dc",         "a95eee14d980a9cfbbaf5df4a3c64713"),
    ("Williams/webbase-1M",   "2d4c239daad6f12d66a1e6a2af44cbdb"),
    ("Bova/rma10",            "a899a0c48b9a58d081c52ffd88a84955"),
    ("JGD_BIBD/bibd_22_8",   "455d5b699ea10232bbab5bc002219ae6"),
    ("Williams/mac_econ_fwd500", "f1b0e56fbb75d1d6862874e3d7d33060"),
    ("Williams/mc2depi",      "8c8633eada6455c1784269b213c85ea6"),
    ("Hamm/scircuit",         "3e62f7ea83914f7e20019aefb2a5176f"),
    ("Sandia/ASIC_320k",      "fcfaf8a25c8f49b8d29f138f3c65c08f"),
    ("GHS_psdef/bmwcra_1",    "8a3cf5448a4fe73dcbdb5a16b326715f"),
    ("HB/nos1",               "b203f7605cb1f20f83280061068f7ec7"),
    ("HB/nos2",               "b0f812ffcc9469f0bf9be701205522c4"),
    ("HB/nos3",               "f185514062a0eeabe86d2909275fe1dc"),
    ("HB/nos4",               "04b781415202db404733ca0c159acbef"),
    ("HB/nos5",               "c98e35f1cfd1ee8177f37bdae155a6e7"),
    ("HB/nos6",               "c39375226aa5c495293003a5f637598f"),
    ("HB/nos7",               "9a6481268847e6cf0d70671f2ff1ddcd"),
    ("DNVS/shipsec1",         "73372e7d6a0848f8b19d64a924fab73e"),
]
# fmt: on

SUITESPARSE_URL = "https://sparse.tamu.edu/MM"
SUITESPARSE_MIRROR = "https://www.cise.ufl.edu/research/sparse/MM"


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_matrix(matrix_path: str, expected_md5: str, dest_dir: Path) -> Path:
    """Download a matrix .tar.gz, verify MD5, extract .mtx. Returns path to .mtx file."""
    name = matrix_path.split("/")[1]
    tar_path = dest_dir / f"{name}.tar.gz"
    mtx_path = dest_dir / f"{name}.mtx"

    if mtx_path.exists():
        return mtx_path

    # Try primary URL, then mirror
    for url_base in [SUITESPARSE_URL, SUITESPARSE_MIRROR]:
        url = f"{url_base}/{matrix_path}.tar.gz"
        print(f"  Downloading {url} ...")
        try:
            urllib.request.urlretrieve(url, tar_path)
            actual_md5 = md5_file(tar_path)
            if actual_md5 == expected_md5:
                break
            print(f"  MD5 mismatch (got {actual_md5}, expected {expected_md5}), trying mirror...")
            tar_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"  Download failed: {e}, trying mirror...")
            tar_path.unlink(missing_ok=True)
    else:
        raise RuntimeError(f"Failed to download {matrix_path} from all mirrors")

    # Extract
    with tarfile.open(tar_path) as tf:
        tf.extractall(path=dest_dir, filter="data")

    extracted_mtx = dest_dir / name / f"{name}.mtx"
    if not extracted_mtx.exists():
        raise RuntimeError(f"Expected {extracted_mtx} after extraction")
    extracted_mtx.rename(mtx_path)

    # Cleanup
    tar_path.unlink(missing_ok=True)
    extracted_dir = dest_dir / name
    if extracted_dir.is_dir():
        shutil.rmtree(extracted_dir)

    return mtx_path


def compile_mtx2csr(convert_source: Path, output: Path) -> None:
    """Compile the mtx2csr converter tool."""
    if output.exists():
        return
    cxx = os.environ.get("CXX", "g++")
    cmd = [cxx, str(convert_source), "-O3", "-o", str(output)]
    print(f"  Compiling mtx2csr: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def convert_matrix(mtx2csr: Path, mtx_path: Path, output_path: Path) -> None:
    """Convert .mtx to .csr or .bin using mtx2csr."""
    if output_path.exists():
        return
    name = mtx_path.stem
    print(f"  Converting {name} -> {output_path.name}")
    subprocess.check_call(
        [str(mtx2csr), f"{name}.mtx", output_path.name],
        cwd=mtx_path.parent,
    )
    if not (mtx_path.parent / output_path.name).exists():
        raise RuntimeError(f"Conversion failed: {output_path.name} not created")
    if mtx_path.parent != output_path.parent:
        shutil.move(str(mtx_path.parent / output_path.name), str(output_path))


def run_dvc_add(file_path: Path) -> None:
    """Run dvc add on a file from within the rocm-libraries directory."""
    rel = file_path.relative_to(ROCM_LIBRARIES_DIR)
    print(f"  dvc add {rel}")
    subprocess.check_call(["dvc", "add", str(rel)], cwd=ROCM_LIBRARIES_DIR)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--push", action="store_true", help="Run 'dvc push' after adding files"
    )
    parser.add_argument(
        "--skip-dvc",
        action="store_true",
        help="Only download and convert; skip dvc add/push",
    )
    args = parser.parse_args()

    if not ROCM_LIBRARIES_DIR.is_dir():
        print(f"ERROR: rocm-libraries not found at {ROCM_LIBRARIES_DIR}", file=sys.stderr)
        sys.exit(1)

    if not args.skip_dvc and shutil.which("dvc") is None:
        print("ERROR: 'dvc' not found on PATH. Install with: pip install dvc[s3]", file=sys.stderr)
        sys.exit(1)

    rocsparse_convert_src = (
        ROCM_LIBRARIES_DIR / "projects" / "rocsparse" / "deps" / "convert.cpp"
    )
    hipsparse_convert_src = (
        ROCM_LIBRARIES_DIR / "projects" / "hipsparse" / "deps" / "convert.cpp"
    )

    rocsparse_matrices_dir = (
        ROCM_LIBRARIES_DIR / "projects" / "rocsparse" / "clients" / "matrices"
    )
    hipsparse_matrices_dir = (
        ROCM_LIBRARIES_DIR / "projects" / "hipsparse" / "clients" / "matrices"
    )

    rocsparse_matrices_dir.mkdir(parents=True, exist_ok=True)
    hipsparse_matrices_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Compile converters
        print("=== Compiling mtx2csr converters ===")
        rocsparse_mtx2csr = tmpdir / "rocsparse_mtx2csr"
        hipsparse_mtx2csr = tmpdir / "hipsparse_mtx2csr"
        compile_mtx2csr(rocsparse_convert_src, rocsparse_mtx2csr)
        compile_mtx2csr(hipsparse_convert_src, hipsparse_mtx2csr)

        # Collect unique matrices to download (avoid re-downloading shared ones)
        all_matrices = {}
        for path, md5 in ROCSPARSE_MATRICES + HIPSPARSE_MATRICES:
            name = path.split("/")[1]
            all_matrices[name] = (path, md5)

        # Download all matrices
        print("=== Downloading matrices ===")
        mtx_dir = tmpdir / "mtx"
        mtx_dir.mkdir()
        for name, (path, md5) in all_matrices.items():
            download_matrix(path, md5, mtx_dir)

        # Convert rocsparse matrices (.mtx -> .csr)
        print("=== Converting rocsparse matrices to .csr ===")
        for path, _ in ROCSPARSE_MATRICES:
            name = path.split("/")[1]
            mtx_path = mtx_dir / f"{name}.mtx"
            csr_path = rocsparse_matrices_dir / f"{name}.csr"
            convert_matrix(rocsparse_mtx2csr, mtx_path, csr_path)

        # Convert hipsparse matrices (.mtx -> .bin)
        print("=== Converting hipsparse matrices to .bin ===")
        for path, _ in HIPSPARSE_MATRICES:
            name = path.split("/")[1]
            mtx_path = mtx_dir / f"{name}.mtx"
            bin_path = hipsparse_matrices_dir / f"{name}.bin"
            convert_matrix(hipsparse_mtx2csr, mtx_path, bin_path)

    if args.skip_dvc:
        print("=== Skipping DVC (--skip-dvc) ===")
        print(f"rocsparse matrices: {rocsparse_matrices_dir}")
        print(f"hipsparse matrices: {hipsparse_matrices_dir}")
        return

    # DVC add each file
    print("=== Running dvc add ===")
    for path, _ in ROCSPARSE_MATRICES:
        name = path.split("/")[1]
        run_dvc_add(rocsparse_matrices_dir / f"{name}.csr")

    for path, _ in HIPSPARSE_MATRICES:
        name = path.split("/")[1]
        run_dvc_add(hipsparse_matrices_dir / f"{name}.bin")

    if args.push:
        print("=== Running dvc push ===")
        subprocess.check_call(["dvc", "push"], cwd=ROCM_LIBRARIES_DIR)

    print("=== Done ===")
    print(f"rocsparse matrices: {rocsparse_matrices_dir}")
    print(f"hipsparse matrices: {hipsparse_matrices_dir}")
    print()
    print("Next steps:")
    print("  1. cd rocm-libraries")
    print("  2. git add projects/rocsparse/clients/matrices/*.csr.dvc")
    print("     git add projects/rocsparse/clients/matrices/.gitignore")
    print("     git add projects/hipsparse/clients/matrices/*.bin.dvc")
    print("     git add projects/hipsparse/clients/matrices/.gitignore")
    print("  3. git commit -m 'Add sparse test matrices tracked via DVC'")


if __name__ == "__main__":
    main()
