import argparse
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
    parser.add_argument(
        "--delete",
        help="Deletes old file after successful promotion",
        action="store_true",
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

    print("  Changing wheel-specific files that contain the version", end="")

    if not "torch" in str(new_dir_path):  # rocm packages
        files_to_change = [
            f"{new_dir_path}/{package_name_no_version}/_dist_info.py",
        ]
    elif (
        "torch" == package_name_no_version
    ):  # only exactly torch and not triton, torchaudio, ..
        print("torch")
        files_to_change = [
            f"{new_dir_path}/{package_name_no_version}/_rocm_init.py",
            f"{new_dir_path}/{package_name_no_version}/version.py",
        ]

        # special handling
        # we only want to change required-distr matching "rocm"
        with fileinput.input(
            files=(
                f"{new_dir_path}/{package_name_no_version}-{old_version}.dist-info/METADATA"
            ),
            encoding="utf-8",
            inplace=True,
        ) as f:
            for line in f:
                if "Requires-Dist" in line:
                    if "rocm" in line:
                        print(line.replace(old_rocm_version, new_rocm_version), end="")
                        continue
                print(line, end="")
    elif not "triton" in package_name_no_version:  # torchaudio, torchvision
        files_to_change = [
            f"{new_dir_path}/{package_name_no_version}/version.py",
        ]
    else:  # triton
        return

    with fileinput.input(files=(files_to_change), encoding="utf-8", inplace=True) as f:
        for line in f:
            print(line.replace(old_rocm_version, new_rocm_version), end="")

    print(" ...done")


def promote_wheel(filename):
    print(f"Promoting whl from rc to final: {filename}")

    original_wheel = Wheel(filename)
    original_version = Version(original_wheel.version)
    base_version = original_version.base_version

    print(f"  Detected version: {original_version}")

    if original_version.local:  # torch packages
        if not "rc" in original_version.local:
            print("  Only release candidates (rc) can be promoted! Aborting!")
            return False
        local_version = str(original_version.local).split("rc", 1)[0]
        base_version = original_version.public
    else:  # rocm packages
        if not "rc" in str(original_version):
            print("  Only release candidates (rc) can be promoted! Aborting!")
            return False
        local_version = None

    print(f"  New base version: {base_version}")
    print(f"  New local version: {local_version}")

    if str(base_version) == str(
        original_version
    ) or f"{base_version}+{local_version}" == str(original_version):
        print("  Version is already a release version, skipping")
        return False

    print("  Starting to execute version change")
    new_wheel_path = change_wheel_version.change_wheel_version(
        pathlib.Path(filename),
        str(base_version),
        local_version,
        callback_func=wheel_change_extra_files,
    )
    print("  Version change done")

    new_wheel = Wheel(new_wheel_path)
    new_version = Version(new_wheel.version)
    print(f"New wheel has {new_version} and path is {new_wheel_path}")
    return True


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
            return False
        if base_version == version:
            print(
                f"  {version} and {base_version} are the same. Already the base version? Skipping..."
            )
            return False

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
        return True


def main(input_dir, match_files="/*", delete=False):
    print(f"Looking for .whl and .tar.gz in {input_dir}/{match_files}")

    files = glob.glob(str(input_dir) + "/" + match_files)

    for file in files:
        print("")
        if file.endswith(".whl"):
            if promote_wheel(file) and delete:
                print(f"Removing old wheel: {file}")
                os.remove(file)
        elif file.endswith(".tar.gz"):
            if promote_targz(file) and delete:
                print(f"Removing old .tar.gz: {file}")
                os.remove(file)
        else:
            print(f"File found that cannot be promoted: {file}")


if __name__ == "__main__":
    print("Parsing arguments", end="")
    p = parse_arguments(sys.argv[1:])
    print(" ...done")

    main(p.input_dir, p.match_files, p.delete)
