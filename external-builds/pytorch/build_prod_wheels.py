#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

r"""Builds production PyTorch wheels based on the rocm wheels.

This script is designed to be used from CI but should be serviceable for real
users. It is not optimized for providing a development experience for PyTorch.

Under Linux, it is standard to run this under an appropriate manylinux container
for producing portable binaries. On Windows, it should run in an environment
with suitable VC redistributables to use the rocm-sdk.

In both cases, it should be run from a venv.

## Building interactively

A full build consists of multiple steps (can be mixed/matched for multi version
builds, etc):

1. Checkout repositories:

The following commands check out custom patched versions into this directory,
which the script will use by default if they exist. Otherwise, checkout your
own and specify with `--pytorch-dir`, `--pytorch-audio-dir`, `--pytorch-vision-dir`
during the build step.

```
# On Linux, using default paths (nested under this folder):
# Note that triton must be checked out after pytorch as it depends on pins
# in the former.
python pytorch_torch_repo.py checkout
python pytorch_audio_repo.py checkout
python pytorch_apex_repo.py checkout
python pytorch_vision_repo.py checkout
python pytorch_triton_repo.py checkout

# On Windows, using shorter paths to avoid compile command length limits:
python pytorch_torch_repo.py checkout --checkout-dir C:/b/pytorch
python pytorch_audio_repo.py checkout --checkout-dir C:/b/audio
python pytorch_vision_repo.py checkout --checkout-dir C:/b/vision
```

2. Install rocm wheels:

You must have the `rocm[libraries,devel]` packages installed. The `install-rocm`
command gives a one-stop to fetch the latest nightlies from the CI or elsewhere.
Below we are using nightly rocm-sdk packages from the CI bucket. See `RELEASES.md`
for further options. Specific versions can be specified via `--rocm-sdk-version`
and `--no-pre` (to disable searching for pre-release candidates). The installed
version will be printed and subsequently will be embedded into torch builds as
a dependency. Such an arrangement is a head-on-head build (i.e. torch head on top
of ROCm head). Other arrangements are possible by passing pinned versions, official
repositories, etc.

You can also install in the same invocation as build by passing `--install-rocm`
to the build sub-command (useful for docker invocations).

```
# For therock-nightly-python
build_prod_wheels.py \
    install-rocm \
    --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/

# For therock-dev-python (unstable but useful for testing outside of prod)
build_prod_wheels.py \
    install-rocm \
    --index-url https://rocm.devreleases.amd.com/v2/gfx110X-all/
```

3. Build torch, torchaudio and torchvision for a single gfx architecture.

Typical usage to build with default architecture from rocm-sdk targets:

```
# On Linux, using default paths for each repository:
python build_prod_wheels.py build \
    --output-dir $HOME/tmp/pyout

# On Windows, using shorter custom paths:
python build_prod_wheels.py build ^
    --output-dir %HOME%/tmp/pyout ^
    --pytorch-dir C:/b/pytorch ^
    --pytorch-audio-dir C:/b/audio ^
    --pytorch-vision-dir C:/b/vision
```

4. Compiler caching (optional):

```
# Use ccache:
python build_prod_wheels.py build --use-ccache --output-dir ...

# Use sccache with ROCm compiler wrapping (caches host + HIP device code):
python build_prod_wheels.py build --use-sccache --output-dir ...

# Use sccache without compiler wrapping (caches host C/C++ only):
python build_prod_wheels.py build --use-sccache --sccache-no-wrap --output-dir ...
```

``--use-ccache`` and ``--use-sccache`` are mutually exclusive.
``--sccache-no-wrap`` is a modifier for ``--use-sccache`` that skips ROCm compiler
wrapping — useful for developers who want basic caching without modifying compiler
binaries. See ``build_tools/setup_sccache_rocm.py`` for details on the wrapping
mechanism.

## Building Linux portable wheels

On Linux, production wheels are typically built in a manylinux container and must have
some custom post-processing to ensure that system deps are bundled. This can be done
via the `build_tools/linux_portable_build.py` utility in the root of the repo.

Example (note that the use of `linux_portable_build.py` can be replaced with custom
docker invocations, but we keep this tool up to date with respect to mounts and image
versions):

```
./build_tools/linux_portable_build.py --docker=podman --exec -- \
    /usr/bin/env CCACHE_DIR=/therock/output/ccache \
    /opt/python/cp312-cp312/bin/python \
    /therock/src/external-builds/pytorch/build_prod_wheels.py \
    build \
        --install-rocm \
        --pip-cache-dir /therock/output/pip_cache \
        --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ \
        --clean \
        --output-dir /therock/output/cp312/wheels
```

TODO: Need to add an option to post-process wheels, set the manylinux tag, and
inline system deps into the audio and vision wheels as needed.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

from pytorch_build_env import (
    capture,
    directory_if_exists,
    do_install_rocm,
    find_built_wheel,
    find_dir_containing,
    get_installed_package_version,
    get_rocm_env,
    get_version_suffix_for_installed_rocm_package,
    is_windows,
    prepare_pytorch_build,
    remove_dir_if_exists,
    run_command,
    sanity_check_pytorch,
)

script_dir = Path(__file__).resolve().parent

# LLVM download URL for triton-windows
LLVM_BASE_URL = "https://oaitriton.blob.core.windows.net/public/llvm-builds"


def get_triton_windows_llvm_hash(triton_dir: Path) -> str:
    """Read the LLVM hash from triton-windows cmake/llvm-hash.txt."""
    hash_file = triton_dir / "cmake" / "llvm-hash.txt"
    if not hash_file.exists():
        raise RuntimeError(f"LLVM hash file not found: {hash_file}")
    return hash_file.read_text().strip()


def download_llvm_for_triton_windows(triton_dir: Path) -> Path:
    """Download and extract pre-built LLVM binaries for triton-windows.

    triton-windows requires a specific LLVM version that matches the hash
    in cmake/llvm-hash.txt. Pre-built binaries are hosted at oaitriton.blob.core.windows.net.
    """
    full_hash = get_triton_windows_llvm_hash(triton_dir)
    short_hash = full_hash[:8]

    llvm_dir = triton_dir.parent / f"llvm-{short_hash}-windows-x64"
    llvm_hash_marker = llvm_dir / ".llvm-hash"

    if llvm_hash_marker.exists():
        installed_hash = llvm_hash_marker.read_text().strip()
        if installed_hash == full_hash:
            print(f"LLVM already downloaded: {llvm_dir}")
            return llvm_dir

    if llvm_dir.exists():
        shutil.rmtree(llvm_dir)

    filename = f"llvm-{short_hash}-windows-x64.tar.gz"
    download_url = f"{LLVM_BASE_URL}/{filename}"

    print(f"Downloading LLVM for triton-windows...")
    print(f"  Hash: {short_hash}")
    print(f"  URL: {download_url}")

    with tempfile.TemporaryDirectory() as temp_dir:
        download_path = Path(temp_dir) / filename

        print("  Downloading (this may take a few minutes, ~500MB)...")
        try:
            urllib.request.urlretrieve(download_url, download_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download LLVM from {download_url}: {e}\n"
                "You may need to download manually and extract to "
                f"{llvm_dir}"
            )

        print("  Extracting...")
        with tarfile.open(download_path, "r:gz") as tar:
            tar.extractall(triton_dir.parent, filter="data")

        if not llvm_dir.exists():
            raise RuntimeError(f"Extracted LLVM directory not found: {llvm_dir}")

        llvm_hash_marker.write_text(full_hash)

    print(f"  LLVM downloaded to: {llvm_dir}")
    return llvm_dir


def copy_to_output(args: argparse.Namespace, src_file: Path):
    output_dir: Path = args.output_dir
    print(f"++ Copy {src_file} -> {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_file, output_dir)


def _do_build_wheels_core(
    args: argparse.Namespace,
    env: dict[str, str],
    triton_dir: Path | None,
    pytorch_dir: Path | None,
    pytorch_audio_dir: Path | None,
    pytorch_vision_dir: Path | None,
    apex_dir: Path | None,
) -> None:
    """Execute all wheel builds (triton, pytorch, audio, vision, apex)."""
    # Build triton.
    triton_requirement = None
    if args.build_triton or (args.build_triton is None and triton_dir):
        assert triton_dir, "Must specify --triton-dir if --build-triton"
        triton_requirement = do_build_triton(args, triton_dir, dict(env))
    else:
        print("--- Not building triton (no --triton-dir)")

    # Build pytorch.
    if pytorch_dir:
        do_build_pytorch(
            args, pytorch_dir, dict(env), triton_requirement=triton_requirement
        )
    else:
        print("--- Not building pytorch (no --pytorch-dir)")

    # Build pytorch audio.
    if args.build_pytorch_audio or (
        args.build_pytorch_audio is None and pytorch_audio_dir
    ):
        assert (
            pytorch_audio_dir
        ), "Must specify --pytorch-audio-dir if --build-pytorch-audio"
        do_build_pytorch_audio(args, pytorch_audio_dir, dict(env))
    else:
        print("--- Not build pytorch-audio (no --pytorch-audio-dir)")

    # Build pytorch vision.
    if args.build_pytorch_vision or (
        args.build_pytorch_vision is None and pytorch_vision_dir
    ):
        assert (
            pytorch_vision_dir
        ), "Must specify --pytorch-vision-dir if --build-pytorch-vision"
        do_build_pytorch_vision(args, pytorch_vision_dir, dict(env))
    else:
        print("--- Not build pytorch-vision (no --pytorch-vision-dir)")

    # Build apex.
    if args.build_apex or (args.build_apex is None and apex_dir):
        assert apex_dir, "Must specify --apex-dir if --build-apex"
        do_build_apex(args, apex_dir, dict(env))
    else:
        print("--- Not build apex (no --apex-dir)")

    print("--- Builds all completed")


def do_build(args: argparse.Namespace):
    if args.install_rocm:
        do_install_rocm(args)

    if not args.version_suffix:
        args.version_suffix = get_version_suffix_for_installed_rocm_package()

    triton_dir: Path | None = args.triton_dir
    pytorch_dir: Path | None = args.pytorch_dir
    pytorch_audio_dir: Path | None = args.pytorch_audio_dir
    pytorch_vision_dir: Path | None = args.pytorch_vision_dir
    apex_dir: Path | None = args.apex_dir

    rocm_dir, env = get_rocm_env(
        pytorch_rocm_arch=args.pytorch_rocm_arch, triton_dir=triton_dir
    )

    if args.use_ccache:
        if not shutil.which("ccache"):
            raise RuntimeError(
                "ccache not found but --use-ccache was specified. "
                "Please install ccache before building."
            )
        print("Building with ccache, clearing stats first")
        env["CMAKE_C_COMPILER_LAUNCHER"] = "ccache"
        env["CMAKE_CXX_COMPILER_LAUNCHER"] = "ccache"
        run_command(["ccache", "--zero-stats"], cwd=tempfile.gettempdir())
    elif args.use_sccache:
        build_tools_dir = Path(__file__).resolve().parent.parent.parent / "build_tools"
        sys.path.insert(0, str(build_tools_dir))

        from setup_sccache_rocm import (
            find_sccache,
            restore_rocm_compilers,
            setup_rocm_sccache,
        )

        sccache_path = find_sccache()
        if not sccache_path:
            raise RuntimeError(
                "sccache not found but --use-sccache was specified.\n"
                "Install: https://github.com/mozilla/sccache#installation\n"
                "For CI, sccache is pre-installed in the manylinux build image:\n"
                "  https://github.com/ROCm/TheRock/tree/main/dockerfiles"
            )

        sccache_wrapped = False
        if args.sccache_no_wrap:
            print("Setting up sccache (CMAKE launchers only, no compiler wrapping)...")
        else:
            print("Setting up sccache with ROCm compiler wrapping...")
            setup_rocm_sccache(rocm_dir, sccache_path)
            sccache_wrapped = True

    try:
        if args.use_sccache:
            env["CMAKE_C_COMPILER_LAUNCHER"] = str(sccache_path)
            env["CMAKE_CXX_COMPILER_LAUNCHER"] = str(sccache_path)

            try:
                run_command(
                    [str(sccache_path), "--start-server"], cwd=tempfile.gettempdir()
                )
            except subprocess.CalledProcessError:
                pass  # Server may already be running

            run_command([str(sccache_path), "--zero-stats"], cwd=tempfile.gettempdir())

        _do_build_wheels_core(
            args,
            env,
            triton_dir,
            pytorch_dir,
            pytorch_audio_dir,
            pytorch_vision_dir,
            apex_dir,
        )
    finally:
        if args.use_sccache:
            if sccache_wrapped:
                print("Restoring ROCm compilers after sccache build...")
                try:
                    restore_rocm_compilers(rocm_dir)
                except Exception as e:
                    print(f"Warning: Failed to restore compilers: {e}")
            sccache_stats = capture(
                [str(sccache_path), "--show-stats"], cwd=tempfile.gettempdir()
            )
            print(f"sccache --show-stats output:\n{sccache_stats}")

        if args.use_ccache:
            ccache_stats_output = capture(
                ["ccache", "--show-stats"], cwd=tempfile.gettempdir()
            )
            print(f"ccache --show-stats output:\n{ccache_stats_output}")


def build_triton_windows(args: argparse.Namespace, triton_dir: Path) -> str:
    """Build triton wheel for Windows using triton-windows repository."""
    print("Building Triton for Windows (using triton-windows repository)")

    llvm_build_dir = download_llvm_for_triton_windows(triton_dir)

    # Prepare environment for triton-windows build.
    # Note: MSVC environment (vcvars64.bat) must already be set up.
    windows_env = dict(os.environ)
    windows_env.update(
        {
            "PYTHONUTF8": "1",
            "LLVM_BUILD_DIR": str(llvm_build_dir),
            "LLVM_INCLUDE_DIRS": str(llvm_build_dir / "include"),
            "LLVM_LIBRARY_DIR": str(llvm_build_dir / "lib"),
            "LLVM_SYSPATH": str(llvm_build_dir),
            "TRITON_BUILD_PROTON": "OFF",
            "TRITON_APPEND_CMAKE_ARGS": "-DCMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH=FALSE",
            # Override package name to "triton" for consistency with Linux
            "TRITON_WHEEL_NAME": "triton",
        }
    )

    print("+++ Installing build dependencies:")
    run_command(
        [sys.executable, "-m", "pip", "install", "build", "wheel"],
        cwd=triton_dir,
    )

    remove_dir_if_exists(triton_dir / "dist")
    if args.clean:
        remove_dir_if_exists(triton_dir / "build")

    print("+++ Building triton:")
    run_command(
        [sys.executable, "-m", "build", "--wheel"],
        cwd=triton_dir,
        env=windows_env,
    )

    # Build produces wheel named "triton" (overridden via TRITON_WHEEL_NAME)
    built_wheel = find_built_wheel(triton_dir / "dist", "triton")
    print(f"Found built wheel: {built_wheel}")
    copy_to_output(args, built_wheel)

    wheel_version = built_wheel.stem.split("-")[1]
    return f"triton=={wheel_version}"


def build_triton_linux(
    args: argparse.Namespace, triton_dir: Path, env: dict[str, str]
) -> str:
    """Build triton wheel for Linux using ROCm/triton repository."""
    print("Building Triton for Linux (using ROCm/triton repository)")

    version_suffix = env.get("TRITON_WHEEL_VERSION_SUFFIX", "")

    # Triton's setup.py constructs the final version string by using
    # a few components:
    # * Base version: `3.3.1`
    # * Version suffix
    #
    # Version suffix itself consist of from following two parts:
    # * git hash suffix:
    #   * "+git<githash>" for development builds
    #   * empty string "" for builds made from git release branches
    # * Additional version information is passed by using environment variable
    #   TRITON_WHEEL_VERSION_SUFFIX
    #   For example:
    #       env["TRITON_WHEEL_VERSION_SUFFIX"] = "+rocm7.0.0rc20250728"
    #
    # Version suffix part of the version is allowed to have only a single
    # "+"-character. Therefore if there are multiple suffixes,
    # they are joined togeher with `-` characters
    # instead of `+` characters in Triton's setup.py so that
    # there is only a single `+` character after the base version.
    #
    # For example:
    # * PyTorch release/2.7 builds use Triton versions like:
    #    3.3.1+rocm7.0.0rc20250728
    # * PyTorch nightly builds use Triton versions like:
    #    3.4.0+git12345678-rocm7.0.0rc20250728
    version_suffix += str(args.version_suffix)
    env["TRITON_WHEEL_VERSION_SUFFIX"] = version_suffix

    triton_wheel_name = env.get("TRITON_WHEEL_NAME", "triton")
    print(f"+++ Uninstall {triton_wheel_name}")
    run_command(
        [sys.executable, "-m", "pip", "uninstall", triton_wheel_name, "-y"],
        cwd=tempfile.gettempdir(),
    )
    print("+++ Installing triton requirements:")
    pip_install_args = []
    if args.pip_cache_dir:
        pip_install_args.extend(["--cache-dir", args.pip_cache_dir])
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            triton_dir / "python" / "requirements.txt",
        ]
        + pip_install_args,
        cwd=triton_dir,
    )

    print("+++ Building triton:")
    # In early ~2.9, setup.py moved from the python/ dir to the root. Check both.
    triton_python_dir = find_dir_containing(
        "setup.py", triton_dir / "python", triton_dir
    )
    remove_dir_if_exists(triton_python_dir / "dist")
    if args.clean:
        remove_dir_if_exists(triton_python_dir / "build")
    run_command(
        [sys.executable, "setup.py", "bdist_wheel"], cwd=triton_python_dir, env=env
    )
    built_wheel = find_built_wheel(triton_python_dir / "dist", triton_wheel_name)
    print(f"Found built wheel: {built_wheel}")
    copy_to_output(args, built_wheel)

    print("+++ Installing built triton:")
    run_command(
        [sys.executable, "-m", "pip", "install", built_wheel], cwd=tempfile.gettempdir()
    )

    installed_triton_version = get_installed_package_version(triton_wheel_name)
    return f"{triton_wheel_name}=={installed_triton_version}"


def do_build_triton(
    args: argparse.Namespace, triton_dir: Path, env: dict[str, str]
) -> str:
    """Build triton wheel. Dispatches to platform-specific build functions."""
    if is_windows:
        return build_triton_windows(args, triton_dir)
    else:
        return build_triton_linux(args, triton_dir, env)


def do_build_pytorch(
    args: argparse.Namespace,
    pytorch_dir: Path,
    env: dict[str, str],
    *,
    triton_requirement: str | None,
):
    pytorch_build_version = prepare_pytorch_build(
        env,
        pytorch_dir,
        version_suffix=args.version_suffix,
        triton_requirement=triton_requirement,
        enable_flash_attention_linux=args.enable_pytorch_flash_attention_linux,
        enable_flash_attention_windows=args.enable_pytorch_flash_attention_windows,
        enable_fbgemm_genai_linux=args.enable_pytorch_fbgemm_genai_linux,
        pip_cache_dir=args.pip_cache_dir,
    )
    env["PYTORCH_BUILD_VERSION"] = pytorch_build_version
    env["PYTORCH_BUILD_NUMBER"] = args.pytorch_build_number

    print("+++ Building pytorch:")
    remove_dir_if_exists(pytorch_dir / "dist")
    if args.clean:
        remove_dir_if_exists(pytorch_dir / "build")
    run_command([sys.executable, "setup.py", "bdist_wheel"], cwd=pytorch_dir, env=env)
    built_wheel = find_built_wheel(pytorch_dir / "dist", "torch")
    print(f"Found built wheel: {built_wheel}")
    copy_to_output(args, built_wheel)

    print("+++ Installing built torch:")
    run_command(
        [sys.executable, "-m", "pip", "install", built_wheel], cwd=tempfile.gettempdir()
    )

    sanity_check_pytorch()


def do_build_pytorch_audio(
    args: argparse.Namespace, pytorch_audio_dir: Path, env: dict[str, str]
):
    # Compute version.
    build_version = (pytorch_audio_dir / "version.txt").read_text().strip()
    build_version += args.version_suffix
    print(f"  pytorch audio BUILD_VERSION: {build_version}")
    env["BUILD_VERSION"] = build_version
    env["BUILD_NUMBER"] = args.pytorch_build_number

    env.update(
        {
            "USE_ROCM": "1",
            "USE_CUDA": "0",
            "USE_FFMPEG": "1",
            "USE_OPENMP": "1",
            "BUILD_SOX": "0",
        }
    )

    if is_windows:
        env.update(
            {
                "DISTUTILS_USE_SDK": "1",
            }
        )

    remove_dir_if_exists(pytorch_audio_dir / "dist")
    if args.clean:
        remove_dir_if_exists(pytorch_audio_dir / "build")

    run_command(
        [sys.executable, "setup.py", "bdist_wheel"], cwd=pytorch_audio_dir, env=env
    )
    built_wheel = find_built_wheel(pytorch_audio_dir / "dist", "torchaudio")
    print(f"Found built wheel: {built_wheel}")
    copy_to_output(args, built_wheel)


def do_build_pytorch_vision(
    args: argparse.Namespace, pytorch_vision_dir: Path, env: dict[str, str]
):
    # Compute version.
    build_version = (pytorch_vision_dir / "version.txt").read_text().strip()
    build_version += args.version_suffix
    print(f"  pytorch vision BUILD_VERSION: {build_version}")
    env["BUILD_VERSION"] = build_version
    env["VERSION_NAME"] = build_version
    env["BUILD_NUMBER"] = args.pytorch_build_number

    env.update(
        {
            "FORCE_CUDA": "1",
            "TORCHVISION_USE_NVJPEG": "0",
            "TORCHVISION_USE_VIDEO_CODEC": "0",
        }
    )

    if is_windows:
        env.update(
            {
                "DISTUTILS_USE_SDK": "1",
            }
        )

    remove_dir_if_exists(pytorch_vision_dir / "dist")
    if args.clean:
        remove_dir_if_exists(pytorch_vision_dir / "build")

    run_command(
        [sys.executable, "setup.py", "bdist_wheel"], cwd=pytorch_vision_dir, env=env
    )
    built_wheel = find_built_wheel(pytorch_vision_dir / "dist", "torchvision")
    print(f"Found built wheel: {built_wheel}")
    copy_to_output(args, built_wheel)


def do_build_apex(args: argparse.Namespace, apex_dir: Path, env: dict[str, str]):
    # Compute version.
    build_version = (apex_dir / "version.txt").read_text().strip()
    build_version += args.version_suffix
    print(f"  Default apex BUILD_VERSION: {build_version}")
    env["BUILD_VERSION"] = build_version
    env["BUILD_NUMBER"] = args.pytorch_build_number

    remove_dir_if_exists(apex_dir / "dist")
    if args.clean:
        remove_dir_if_exists(apex_dir / "build")

    run_command(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "-C--build-option=--cpp_ext",
            "-C--build-option=--cuda_ext",
        ],
        cwd=apex_dir,
        env=env,
    )
    built_wheel = find_built_wheel(apex_dir / "dist", "apex")
    print(f"Found built wheel: {built_wheel}")
    copy_to_output(args, built_wheel)


def main(argv: list[str]):
    p = argparse.ArgumentParser(prog="build_prod_wheels.py")

    def add_common(p: argparse.ArgumentParser):
        p.add_argument("--index-url", help="Base URL of the Python Package Index.")
        p.add_argument(
            "--find-links",
            help="URL or path for pip --find-links (flat package index).",
        )
        p.add_argument("--pip-cache-dir", type=Path, help="Pip cache dir")
        # Note that we default to >1.0 because at the time of writing, we had
        # 0.1.0 release placeholder packages out on pypi and we don't want them
        # taking priority.
        p.add_argument(
            "--rocm-sdk-version",
            default=">1.0",
            help="rocm-sdk version to match (with comparison prefix)",
        )
        p.add_argument(
            "--pre",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Include pre-release packages (default True)",
        )
        p.add_argument(
            "--rocm-extras",
            default="",
            help=(
                "Comma-separated additional extras for rocm package install "
                "(e.g. 'device-gfx942,device-gfx943'). "
                "Added alongside the base 'libraries,devel' extras."
            ),
        )

    sub_p = p.add_subparsers(required=True)
    install_rocm_p = sub_p.add_parser(
        "install-rocm", help="Install rocm-sdk wheels to the current venv"
    )
    add_common(install_rocm_p)
    install_rocm_p.set_defaults(func=do_install_rocm)

    build_p = sub_p.add_parser("build", help="Build pytorch wheels")
    add_common(build_p)

    build_p.add_argument(
        "--install-rocm",
        action=argparse.BooleanOptionalAction,
        help="Install rocm-sdk before building",
    )
    build_p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to copy built wheels to",
    )
    cache_group = build_p.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--use-ccache",
        action="store_true",
        default=False,
        help="Use ccache as the compiler launcher",
    )
    cache_group.add_argument(
        "--use-sccache",
        action="store_true",
        default=False,
        help="Use sccache as the compiler launcher (with ROCm compiler wrapping on Linux)",
    )
    build_p.add_argument(
        "--sccache-no-wrap",
        action="store_true",
        default=False,
        help="With --use-sccache: skip compiler wrapping, only set CMAKE launchers "
        "(caches host C/C++ but not HIP device code)",
    )
    build_p.add_argument(
        "--pytorch-dir",
        default=directory_if_exists(script_dir / "pytorch"),
        type=Path,
        help="PyTorch source directory",
    )
    build_p.add_argument(
        "--pytorch-audio-dir",
        default=directory_if_exists(script_dir / "pytorch_audio"),
        type=Path,
        help="pytorch_audio source directory",
    )
    build_p.add_argument(
        "--pytorch-vision-dir",
        default=directory_if_exists(script_dir / "pytorch_vision"),
        type=Path,
        help="pytorch_vision source directory",
    )
    build_p.add_argument(
        "--triton-dir",
        default=directory_if_exists(script_dir / "triton"),
        type=Path,
        help="pinned triton directory",
    )
    build_p.add_argument(
        "--apex-dir",
        default=directory_if_exists(script_dir / "apex"),
        type=Path,
        help="apex source directory",
    )
    build_p.add_argument(
        "--pytorch-rocm-arch",
        help="gfx arch to build pytorch with (defaults to rocm-sdk targets)",
    )
    build_p.add_argument(
        "--pytorch-build-number", default="1", help="Build number to append to version"
    )
    build_p.add_argument(
        "--build-triton",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable building of triton (requires --triton-dir)",
    )
    build_p.add_argument(
        "--build-pytorch-audio",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable building of torch audio (requires --pytorch-audio-dir)",
    )
    build_p.add_argument(
        "--build-pytorch-vision",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable building of torch vision (requires --pytorch-vision-dir)",
    )
    build_p.add_argument(
        "--build-apex",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable building of apex (requires --apex-dir)",
    )
    build_p.add_argument(
        "--enable-pytorch-flash-attention-windows",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable building of torch flash attention on Windows (enabled by default for Linux)",
    )
    build_p.add_argument(
        "--enable-pytorch-flash-attention-linux",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable building of torch flash attention on Linux (enabled by default, sets USE_FLASH_ATTENTION=1)",
    )
    build_p.add_argument(
        "--enable-pytorch-fbgemm-genai-linux",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable building of torch fbgemm_genai on Linux (enabled by default, sets USE_FBGEMM_GENAI=ON)",
    )
    build_p.add_argument(
        "--version-suffix",
        help="Explicit PyTorch version suffix (e.g. `+rocm7.10.0a20251124`). Typically computed with build_tools/github_actions/determine_version.py. If omitted it will be derived from the installed rocm package",
    )
    build_p.add_argument(
        "--clean",
        action=argparse.BooleanOptionalAction,
        help="Clean build directories before building",
    )
    build_p.set_defaults(func=do_build)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
