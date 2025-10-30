import argparse
import change_wheel_version_with_hook
from packaging.version import Version
import pathlib
from pkginfo import Wheel
import glob
import sys
import tempfile
import tarfile
import fileinput
import os
import glob


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description="""Promotes packages from release candidate to final release (e.g. 7.10.0rc1 --> 7.10.0).

Promotion works for for wheels and .tar.gz.
Wheels version is determined by python library to interact with the wheel.
For tar.gz., the version is extract from <.tar.gz>/PKG-INFO file.
"""
    )
    parser.add_argument(
        "--input_dir",
        help="Path to the directory that contains .whl and .tar.gz files to promote",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--match-files",
        help="Limits selection in '--input-dir' to files matchings this argument. Use wild cards if needed, e.g. '*rc2*' (default '*' to promote all files in '--input-dir')",
        default="*",
    )
    return parser.parse_args(argv)


def wheel_change_extra_files(new_dir_path, old_version, new_version):
    # extract "rocm_sdk_core" from /tmp/tmp3swrl25j/wheel/rocm_sdk_core-7.10.0
    package_name_no_version = (
        str(new_dir_path).rsplit("/", 1)[-1].split(str(new_version))[0][:-1]
    )

    # correct capitalization
    # of interest for amdgpu arch: wheels are all lower case
    # (e.g.  rocm_sdk_libraries_gfx94x_dcgpu-7.10.0a20251028-py3-none-linux_x86_64.whl)
    # but inside we have to match to rocm_sdk_libraries_gfx94X-dcgpu/ with a capital "X-dcgpu" instead of "x_dcgpu"
    for dir in glob.glob(str(new_dir_path) + "/*"):
        stripped_dir = str(dir).rsplit("/", 1)[-1]
        if (
            "gfx" in package_name_no_version
            and "gfx" in stripped_dir
            and len(package_name_no_version) == len(stripped_dir)
        ):
            package_name_no_version = stripped_dir
            break

    files_to_change = [
        f"{new_dir_path}/{package_name_no_version}/_dist_info.py",
    ]

    print("  Changing wheel-specific files that contain the version", end="")

    with fileinput.input(files=(files_to_change), encoding="utf-8", inplace=True) as f:
        for line in f:
            print(line.replace(str(old_version), str(new_version)), end="")

    print(" ...done")


def promote_wheel(filename):
    print(f"Promoting whl from rc to final: {filename}")

    original_wheel = Wheel(filename)
    original_version = Version(original_wheel.version)
    base_version = original_version.base_version

    print(f"  Detected version: {original_version}")
    print(f"  Base version: {base_version}")

    if str(base_version) == str(original_version):
        print("  Version is already a release version, skipping")
        return

    print(f"  Changing to base version: {base_version}")
    new_wheel_path = change_wheel_version_with_hook.change_wheel_version(
        pathlib.Path(filename),
        str(base_version),
        None,
        callback_func=wheel_change_extra_files,
    )

    new_wheel = Wheel(new_wheel_path)
    new_version = Version(new_wheel.version)
    print(f"New wheel has {new_version} and path is {new_wheel_path}")


def promote_targz(filename: str):
    print(f"Found tar.gz: {filename}")

    split = filename.removesuffix(".tar.gz").rsplit("/", 1)
    base_dir = split[0]
    dir_prefix = split[-1]
    with tempfile.TemporaryDirectory(prefix=dir_prefix + "-") as tmp_dir:
        print(f"  Extracting tar file to {tmp_dir}", end="")

        targz = tarfile.open(filename)
        targz.extractall(tmp_dir)
        targz.close()
        print(" ...done")

        with open(f"{tmp_dir}/{dir_prefix}/PKG-INFO", "r") as info:
            for line in info.readlines():
                if "Version" in line:
                    version = line.removeprefix("Version:").strip()

        assert version, f"No version found in {filename}/PKG-INFO."

        base_version = version.split("rc", 1)[0]

        if any(c.isalpha() for c in base_version):
            print(
                f"  Base version extraction not successful and letters still in the version {base_version}."
            )
            print("  Only release candidates (rc) can be promoted! Aborting!")
            return
        if base_version == version:
            print(
                f"  {version} and {base_version} are the same. Already the base version? Skipping..."
            )
            return

        print(
            f"  Editing files to change version from {version} to {base_version}",
            end="",
        )

        files_to_change = [
            f"{tmp_dir}/{dir_prefix}/src/rocm.egg-info/requires.txt",
            f"{tmp_dir}/{dir_prefix}/src/rocm.egg-info/PKG-INFO",
            f"{tmp_dir}/{dir_prefix}/src/rocm_sdk/_dist_info.py",
            f"{tmp_dir}/{dir_prefix}/PKG-INFO",
        ]

        with fileinput.input(
            files=(files_to_change), encoding="utf-8", inplace=True
        ) as f:
            for line in f:
                print(line.replace(version, base_version), end="")

        print(" ...done")

        print("  Creating new archive for it", end="")
        # rename tmp dir to the new version
        dir_prefix_no_version = dir_prefix.removesuffix(version)
        new_dir_name = dir_prefix_no_version + base_version
        os.rename(f"{tmp_dir}/{dir_prefix}", f"{tmp_dir}/{new_dir_name}")

        print(f" {new_dir_name}", end="")

        with tarfile.open(f"{base_dir}/{new_dir_name}.tar.gz", "w|gz") as tar:
            tar.add(f"{tmp_dir}/{new_dir_name}", arcname=new_dir_name)

        print(f" ...done")
        print(
            f"New tar.gz with version {base_version} written to {base_dir}/{new_dir_name}.tar.gz"
        )


if __name__ == "__main__":
    p = parse_arguments(sys.argv[1:])

    for file in glob.glob(str(p.input_dir) + "/" + p.match_files):
        print("")
        if file.endswith(".whl"):
            promote_wheel(file)
        elif file.endswith(".tar.gz"):
            promote_targz(file)
        else:
            print(f"File found that cannot be promoted: {file}")
