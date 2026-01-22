#!/usr/bin/env python
"""Promotes release candidate packages to final releases.

This script removes release candidate (rc) suffixes from package versions, such
as promoting 7.10.0rc1 to 7.10.0. It handles both Python wheels (.whl) and
source distributions (.tar.gz). This only updates the version strings in the packages but
does not change any other package content.

PREREQUISITES:
  - pip install -r ./build_tools/packaging/requirements.txt

SIDE EFFECTS:
  - Creates NEW promoted package files side-by-side with the original files
  - By default, DOES NOT delete original RC files (safe to run with no --delete flag)
  - With --delete-old-on-success flag, removes original RC files after promotion
  - If no arguments are given, it will promote all RC packages in the current directory.

TYPICAL USAGE:
  # Promote all RC packages in a directory (keeps original RC files):
  python ./build_tools/packaging/promote_from_rc_to_final.py --input-dir=./release_candidates/rc1/

  # Promote and delete original RC files on success:
  python ./build_tools/packaging/promote_from_rc_to_final.py --input-dir=./release_candidates/rc1/ --delete-old-on-success

  # Promote only specific files matching a pattern:
  python ./build_tools/packaging/promote_from_rc_to_final.py --input-dir=./release_candidates/ --match-files='*rc2*'

  # Promote some nightly for testing
  python ./build_tools/packaging/promote_from_rc_to_final.py --input-dir=./release_candidates/ --release-type="a"

TESTING:
  python ./build_tools/packaging/tests/promote_from_rc_to_final_test.py
"""

import argparse
import shutil
from packaging.version import Version
import pathlib
from pkginfo import Wheel
import sys
import tempfile
import tarfile
import fileinput
import os
import importlib.util

# Need dynamic load as change_wheel_version needs to be imported via parent directory
this_file = pathlib.Path(__file__).resolve()
build_tools_dir = this_file.parent.parent
# ../third_party/change_wheel_version/change_wheel_version.py
change_wheel_version_path = (
    build_tools_dir / "third_party" / "change_wheel_version" / "change_wheel_version.py"
)

spec = importlib.util.spec_from_file_location(
    "third_party_change_wheel_version", change_wheel_version_path
)
change_wheel_version = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(change_wheel_version)

assert hasattr(
    change_wheel_version, "change_wheel_version"
), "change_wheel_version module does not expose change_wheel_version function"


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description="""Promotes packages from release candidate to final release (e.g. 7.10.0rc1 --> 7.10.0).

Promotion works for for wheels and .tar.gz.
Wheels version is determined by python library to interact with the wheel.
For tar.gz., the version is extract from <.tar.gz>/PKG-INFO file.
""",
        usage="python ./build_tools/packaging/promote_from_rc_to_final.py --input-dir=./release_candidates/rc1/ --delete-old-on-success",
    )
    parser.add_argument(
        "--input-dir",
        help="Path to the directory that contains .whl and .tar.gz files to promote",
        type=pathlib.Path,
        required=True,
    )
    parser.add_argument(
        "--match-files",
        help="Limits selection in '--input-dir' to files matchings this argument. Use wild cards if needed, e.g. '*rc2*' (default '*' to promote all files in '--input-dir')",
        default="*",
    )
    parser.add_argument(
        "--delete-old-on-success",
        help="Deletes old file after successful promotion",
        action="store_true",
    )
    parser.add_argument(
        "--prerelease-type",
        help="Prerelease type to use instead of 'rc' (e.g. 'dev' or 'a')",
        nargs="+",
        default=["rc"],
        choices=["rc", "dev", "a"],
    )

    return parser.parse_args(argv)


def wheel_change_extra_files(new_dir_path: pathlib.Path, old_version, new_version):
    # extract "rocm_sdk_core" from /tmp/tmp3swrl25j/wheel/rocm_sdk_core-7.10.0
    package_name_no_version = new_dir_path.name.split(str(new_version))[0][:-1]

    # correct capitalization and hyphenation
    # of interest for amdgpu arch: wheels are all lower case
    # (e.g.  rocm_sdk_libraries_gfx94x_dcgpu-7.10.0rc1-py3-none-linux_x86_64.whl)
    # but inside we have to match to rocm_sdk_libraries_gfx94X-dcgpu/ with a capital "X-dcgpu" instead of "x_dcgpu"
    if "gfx" in package_name_no_version:
        files = list(new_dir_path.glob("*gfx*"))
        for file in files:
            if len(file.name) == len(package_name_no_version):
                package_name_no_version = file.name

    old_rocm_version = (
        str(old_version)
        if not "rocm" in str(old_version)
        else str(old_version).split("+rocm")[-1]
    )
    new_rocm_version = (
        str(new_version)
        if not "rocm" in str(new_version)
        else str(new_version).split("+rocm")[-1]
    )

    print("    Changing ROCm-specific files that contain the version")

    if not "torch" in new_dir_path.name:  # rocm packages
        files_to_change = [
            new_dir_path / package_name_no_version / "_dist_info.py",
        ]
    # only torch and NOT triton, torchaudio, torchvision
    elif "torch" == package_name_no_version:
        files_to_change = [
            new_dir_path / package_name_no_version / "_rocm_init.py",
            new_dir_path / package_name_no_version / "version.py",
        ]

        # special handling
        # we only want to change required-dist matching "rocm"
        metadata_path = (
            new_dir_path
            / f"{package_name_no_version}-{old_version}.dist-info"
            / "METADATA"
        )
        print(f"      {metadata_path}")
        with fileinput.input(
            files=(metadata_path),
            encoding="utf-8",
            inplace=True,
        ) as f:
            for line in f:
                if "Requires-Dist" in line:
                    if "rocm" in line:
                        print(line.replace(old_rocm_version, new_rocm_version), end="")
                        continue
                print(line, end="")
    # torchaudio, torchvision
    elif not "triton" in package_name_no_version:
        files_to_change = [
            new_dir_path / package_name_no_version / "version.py",
        ]
    # triton
    else:
        # no additional (rocm-specific) files needed to be changed that contain the version
        return

    for f in files_to_change:
        print(f"      {f}")
    with fileinput.input(files=(files_to_change), encoding="utf-8", inplace=True) as f:
        for line in f:
            print(line.replace(old_rocm_version, new_rocm_version), end="")

    print("    ...done")


def promote_wheel(filename: pathlib.Path, prerelease_type: str) -> bool:
    print(f"Promoting whl from rc to final: {filename}")

    original_wheel = Wheel(filename)
    original_version = Version(original_wheel.version)
    new_base_version = str(original_version.base_version)

    print(f"  Detected version: {original_version}")

    if original_version.local:  # torch packages
        if not prerelease_type in original_version.local:
            print(
                f"  [ERROR] Only prerelease versions of type '{prerelease_type}' can be promoted! Skipping!"
            )
            return False
        new_local_version = str(original_version.local).split(prerelease_type, 1)[0]
        new_base_version = str(original_version.public)
    else:  # rocm packages
        if not prerelease_type in str(original_version):
            print(
                f"  [ERROR] Only prerelease versions of type '{prerelease_type}' can be promoted! Skipping!"
            )
            return False
        new_local_version = None

    print(f"  New base version: {new_base_version}")
    print(f"  New local version: {new_local_version}")

    print("  Starting to execute version change")
    new_wheel_path = change_wheel_version.change_wheel_version(
        filename,
        new_base_version,
        new_local_version,
        callback_func=wheel_change_extra_files,
    )
    print("  Version change done")

    new_wheel = Wheel(new_wheel_path)
    new_version = Version(new_wheel.version)
    print(f"New wheel has {new_version} and path is {new_wheel_path}")
    return True


def promote_targz_sdist(filename: pathlib.Path, prerelease_type: str) -> bool:
    print(f"Found tar.gz: {filename}")

    base_dir = filename.parent
    package_name = filename.name.removesuffix(".tar.gz")  # removes .tar.gz

    with tempfile.TemporaryDirectory(prefix=package_name + "-") as tmp_dir:
        print(f"  Extracting tar file to {tmp_dir}", end="")

        tmp_path = pathlib.Path(tmp_dir)

        targz = tarfile.open(filename)
        targz.extractall(tmp_path)
        targz.close()
        print(" ...done")

        with open(tmp_path / f"{package_name}" / "PKG-INFO", "r") as info:
            for line in info.readlines():
                if line.startswith("Version"):
                    version = Version(line.removeprefix("Version:").strip())

        assert version, f"No version found in {filename}/PKG-INFO."

        print(f"  Detected version: {version}")

        if not prerelease_type in str(version):
            print(
                f"  [ERROR] Only prerelease versions of type '{prerelease_type}' can be promoted! Skipping!"
            )
            return False

        base_version = version.base_version

        print(
            f"  Editing files to change version from {version} to {base_version}",
            end="",
        )

        files_to_change = [
            tmp_path / f"{package_name}" / "src" / "rocm.egg-info" / "requires.txt",
            tmp_path / f"{package_name}" / "src" / "rocm.egg-info" / "PKG-INFO",
            tmp_path / f"{package_name}" / "src" / "rocm_sdk" / "_dist_info.py",
            tmp_path / f"{package_name}" / "PKG-INFO",
        ]

        with fileinput.input(
            files=(files_to_change), encoding="utf-8", inplace=True
        ) as f:
            for line in f:
                print(line.replace(str(version), str(base_version)), end="")

        print(" ...done")

        print("  Creating new archive for it", end="")
        # Rename temporary directory to package name with promoted version
        package_name_no_version = package_name.removesuffix(str(version))
        new_archive_name = package_name_no_version + str(base_version)
        os.rename(tmp_path / f"{package_name}", tmp_path / f"{new_archive_name}")

        print(f" {new_archive_name}", end="")

        with tarfile.open(f"{base_dir}/{new_archive_name}.tar.gz", "w|gz") as tar:
            tar.add(tmp_path / f"{new_archive_name}", arcname=new_archive_name)

        print(f" ...done")
        print(
            f"\nRepacked {package_name} as release {base_dir}/{new_archive_name}.tar.gz"
        )
        return True


def promote_targz_tarball(
    filename: pathlib.Path, delete: bool, prerelease_type: str
) -> bool:
    old_name = filename.name.removesuffix(".tar.gz")
    old_version = Version(old_name.split("-")[-1])

    print(f"Promoting tarball from rc to final: {filename.name}")
    print(f"  Detected version: {old_version}")

    if not prerelease_type in str(old_version):
        print(
            f"  [ERROR] Only prerelease versions of type '{prerelease_type}' can be promoted! Skipping!"
        )
        return False

    new_version = Version(old_version.base_version)
    new_name = old_name.replace(str(old_version), str(new_version)) + ".tar.gz"

    print(f"  New version: {new_version}")

    if delete:
        os.rename(filename, filename.parent / new_name)
        print(f"  Rename {filename.name} to {new_name}", end="")
    else:
        print(f"  Copy {filename.name} to {new_name}", end="")
        shutil.copy2(filename, filename.parent / new_name)
        print(" ...done")

    print(f"Repacked {filename.name} as release {filename.parent}/{new_name}")
    return True


def main(
    input_dir: pathlib.Path,
    match_files: str = "*",
    delete: bool = False,
    prerelease_type: str = "rc",
) -> None:
    print(f"Looking for .whl and .tar.gz in {input_dir}/{match_files}")

    files = input_dir.glob(match_files)

    for file in files:
        print("")
        if file.is_dir():
            print(f"Skipping directory: {file}")
            continue
        if file.suffix == ".whl":
            if promote_wheel(file, prerelease_type) and delete:
                print(f"Removing old wheel: {file}")
                os.remove(file)
        elif file.suffixes[-1] == ".gz" and file.suffixes[-2] == ".tar":
            if file.name.startswith("therock-dist"):
                promote_targz_tarball(file, delete, prerelease_type)
            else:
                if promote_targz_sdist(file, prerelease_type) and delete:
                    print(f"Removing old sdist .tar.gz: {file}")
                    os.remove(file)
        else:
            print(f"File found that cannot be promoted: {file}")


if __name__ == "__main__":
    print("Parsing arguments", end="")
    p = parse_arguments(sys.argv[1:])
    print(" ...done")

    main(p.input_dir, p.match_files, p.delete_old_on_success, p.prerelease_type[0])
