import argparse
import shutil
import sys
import os
import glob
from pathlib import Path
import tempfile
from packaging.version import Version
from pkginfo import Wheel
import subprocess
import urllib

sys.path.insert(0, os.fspath(Path(__file__).parent.parent / "packaging"))
import promote_from_rc_to_final

def checkPromotedFileNames(dir_path, platform):
    if platform == "linux":
        expected_promoted_pkgs = [  "rocm-7.9.0.tar.gz",
                                    "rocm_sdk_core-7.9.0-py3-none-linux_x86_64.whl",
                                    "rocm_sdk_devel-7.9.0-py3-none-linux_x86_64.whl",
                                    "rocm_sdk_libraries_gfx94x_dcgpu-7.9.0-py3-none-linux_x86_64.whl",
                                    "pytorch_triton_rocm-3.3.1+rocm7.9.0-cp312-cp312-linux_x86_64.whl",
                                    "torch-2.7.1+rocm7.9.0-cp312-cp312-linux_x86_64.whl",
                                    "torchaudio-2.7.1a0+rocm7.9.0-cp312-cp312-linux_x86_64.whl",
                                    "torchvision-0.22.1+rocm7.9.0-cp312-cp312-linux_x86_64.whl"
                                 ] 
    else:
        expected_promoted_pkgs = ["rocm-7.9.0.tar.gz",
                                  "rocm_sdk_core-7.9.0-py3-none-win_amd64.whl",
                                  "rocm_sdk_devel-7.9.0-py3-none-win_amd64.whl",
                                  "rocm_sdk_libraries_gfx1151-7.9.0-py3-none-win_amd64.whl",
                                  "torch-2.9.0+rocm7.9.0-cp312-cp312-win_amd64.whl",
                                  "torchaudio-2.9.0+rocm7.9.0-cp312-cp312-win_amd64.whl",
                                  "torchvision-0.24.0+rocm7.9.0-cp312-cp312-win_amd64.whl",
                                  "pytorch_triton_rocm-3.3.1+rocm7.9.0-cp312-cp312-linux_x86_64.whl"
                                ]

    # get files and strip path from them
    files = glob.glob(dir_path + "/*")
    files = [file.rsplit("/", 1)[-1] for file in files]

    if len(files) != len(expected_promoted_pkgs):
        return False, f"Files found and expected promoted packages are not the same amount ({len(files)} vs {len(expected_promoted_pkgs)})"

    for file in files:
        if not file in expected_promoted_pkgs:
            return False, f"{file} not matching any of the expected package names"

    return True, ""


def checkAllWheelsSameVersion(dir_path, expected_version):
    for file in glob.glob(dir_path + "/*.whl"):
        wheel = Wheel(file)
        version = Version(wheel.version)

        if str(version) == str(expected_version) and version.local == None:  # rocm packages
            continue
        elif str(version.local) == "rocm" + str(expected_version):  # torch packages
            continue
        else:
            return False, f"{file} has version {version}, but expected version is {expected_version}"
    
    return True, ""

def checkInstallation(dir_path):
    try:
        packages = glob.glob(dir_path + "/*")
        proc = subprocess.run(["pip", "install"] + packages, capture_output=True, encoding="utf-8", check=True)
    except subprocess.CalledProcessError as e:
        return False, e.stderr
    return True, ""

def checkPromoteEverything(dir_path, expected_version, platform):
    print("")
    print("=================================================================================")
    print("TEST: Testing promotion of all packages")
    print("=================================================================================")
    success = False
    with tempfile.TemporaryDirectory(prefix="PromoteRcToFinalTest-PromoteEverything-") as tmp_dir:
        # make a copy in a separate dir to not pollute it
        for file in glob.glob(dir_path + "/*"):
            shutil.copy2(file, tmp_dir)

        promote_from_rc_to_final.main(tmp_dir, delete=True)
        success = True

        for func_name, res in [
            ("checkPromotedFileNames", checkPromotedFileNames(tmp_dir, platform)),
            ("checkAllWheelsSameVersion", checkAllWheelsSameVersion(tmp_dir, expected_version)),
            ("checkInstallation", checkInstallation(tmp_dir))
        ]:
            if not res[0]:
                print("")
                print(f"[ERROR] Failure to promote the packages (failure captured by {func_name}):")
                print(res[1])
                success = False
                break
    print("")
    print("=================================================================================")
    print("TEST DONE: Testing promotion of all packages. Result:" + (" SUCCESS" if success else " FAILURE"))
    print("=================================================================================")
    return success

def checkPromoteOnlyRocm(dir_path, expected_version, platform):  # should fail
    print("")
    print("=================================================================================")
    print("TEST: Testing promotion of only rocm packages")
    print("=================================================================================")
    success = False
    with tempfile.TemporaryDirectory(prefix="PromoteRcToFinalTest-PromoteOnlyRocm-") as tmp_dir:
        # make a copy in a separate dir to not pollute it
        for file in glob.glob(dir_path + "/*"):
            shutil.copy2(file, tmp_dir)

        promote_from_rc_to_final.main(tmp_dir, match_files="/rocm*", delete=True)

        success = True

        for func_name, res in [
            ("checkPromotedFileNames", checkPromotedFileNames(tmp_dir, platform)),
            ("checkAllWheelsSameVersion", checkAllWheelsSameVersion(tmp_dir, expected_version)),
            ("checkInstallation", checkInstallation(tmp_dir))
        ]:
            if res[0]:
                success = False
                print("")
                print(f"[ERROR] checkPromoteOnlyRocm: Promotion of packages successful, eventhough it shouldnt be")
                print("Function that succeeded (and should NOT have): " + func_name)
                proc = subprocess.run(["ls", tmp_dir], capture_output=True, encoding="utf-8")
                print(proc.stdout)
                break
    print("")
    print("=================================================================================")
    print("TEST DONE: Testing promotion of only rocm packages. Result:" + (" SUCCESS" if success else " FAILURE"))
    print("=================================================================================")
    return success

def checkPromoteOnlyTorch(dir_path, expected_version, platform):  # should fail
    print("")
    print("=================================================================================")
    print("TEST: Testing promotion of only PyTorch packages")
    print("=================================================================================")
    success = False
    with tempfile.TemporaryDirectory(prefix="PromoteRcToFinalTest-PromoteOnlyTorch-") as tmp_dir:
        # make a copy in a separate dir to not pollute it
        for file in glob.glob(dir_path + "/*"):
            shutil.copy2(file, tmp_dir)

        promote_from_rc_to_final.main(tmp_dir, match_files="/*torch*", delete=True)

        success = True

        for func_name, res in [
            ("checkPromotedFileNames", checkPromotedFileNames(tmp_dir, platform)),
            ("checkAllWheelsSameVersion", checkAllWheelsSameVersion(tmp_dir, expected_version)),
            ("checkInstallation", checkInstallation(tmp_dir))
        ]:
            if res[0]:
                success = False
                print("")
                print(f"[ERROR] checkPromoteOnlyTorch: Promotion of packages successful, eventhough it shouldnt be")
                print("Function that succeeded (and should NOT have): " + func_name)
                proc = subprocess.run(["ls", tmp_dir], capture_output=True, encoding="utf-8")
                print(proc.stdout)
                break
    print("")
    print("=================================================================================")
    print("TEST DONE: Testing promotion of only PyTorch packages. Result:" + (" SUCCESS" if success else " FAILURE"))
    print("=================================================================================")
    return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Tests promotion of packages from release candidate to final release (e.g. 7.10.0rc1 --> 7.10.0).
"""
    )
    parser.add_argument(
        "--platform",
        help="OS platform: either 'linux' (default) or 'win'",
        default="linux"
    )
    p = parser.parse_args(sys.argv[1:])
    platform = p.platform
    # make tmpdir
    with tempfile.TemporaryDirectory(prefix=f"PromoteRcToFinalTest-{platform}-") as tmp_dir:
        if platform == "linux":
        # download some version
            URL="https://rocm.prereleases.amd.com/whl/gfx94X-dcgpu/"
            version=Version("7.9.0rc1")
            expected_version=Version("7.9.0")
            packages=["rocm-7.9.0rc1.tar.gz",
                      "rocm_sdk_core-7.9.0rc1-py3-none-linux_x86_64.whl",
                      "rocm_sdk_devel-7.9.0rc1-py3-none-linux_x86_64.whl",
                      "rocm_sdk_libraries_gfx94x_dcgpu-7.9.0rc1-py3-none-linux_x86_64.whl",
                      "pytorch_triton_rocm-3.3.1+rocm7.9.0rc1-cp312-cp312-linux_x86_64.whl",
                      "torch-2.7.1+rocm7.9.0rc1-cp312-cp312-linux_x86_64.whl",
                      "torchaudio-2.7.1a0+rocm7.9.0rc1-cp312-cp312-linux_x86_64.whl",
                      "torchvision-0.22.1+rocm7.9.0rc1-cp312-cp312-linux_x86_64.whl"
            ]
        else:  # win
            URL="https://rocm.prereleases.amd.com/whl/gfx1151/"
            version=Version("7.9.0rc1")
            expected_version=Version("7.9.0")
            packages=["rocm-7.9.0rc1.tar.gz",
                      "rocm_sdk_core-7.9.0rc1-py3-none-win_amd64.whl",
                      "rocm_sdk_devel-7.9.0rc1-py3-none-win_amd64.whl",
                      "rocm_sdk_libraries_gfx1151-7.9.0rc1-py3-none-win_amd64.whl",
                      "torch-2.9.0+rocm7.9.0rc1-cp312-cp312-win_amd64.whl",
                      "torchaudio-2.9.0+rocm7.9.0rc1-cp312-cp312-win_amd64.whl",
                      "torchvision-0.24.0+rocm7.9.0rc1-cp312-cp312-win_amd64.whl",
                      "pytorch_triton_rocm-3.3.1+rocm7.9.0rc1-cp312-cp312-linux_x86_64.whl"
            ]


        print(f"Testing promotion of {version} to {expected_version} on platform {platform}")
        print(f"  Downloading packages from {URL}")
        for package in packages:
            print(f"  Downloading {package}")
            # otherwise it gets unhappy with the "+" in the URL
            url_safe_encoding = URL + urllib.parse.quote(package)
            print(url_safe_encoding)
            subprocess.run(["curl", "--output", tmp_dir + "/" + package, url_safe_encoding], check=True)


        res_everything = checkPromoteEverything(tmp_dir, expected_version, platform)
        res_rocm = checkPromoteOnlyRocm(tmp_dir, expected_version, platform)
        res_torch = checkPromoteOnlyTorch(tmp_dir, expected_version, platform)

        print("")
        print("")
        print("=================================================================================")
        print("SUMMARY")
        print("=================================================================================")
        print("checkPromoteEverything: " + ("SUCCESS" if res_everything else "FAILURE"))
        print("checkPromoteOnlyRocm: " + ("SUCCESS" if res_rocm else "FAILURE"))
        print("checkPromoteOnlyTorch: " + ("SUCCESS" if res_torch else "FAILURE"))
        print("=================================================================================")

