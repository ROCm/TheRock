#!/usr/bin/env python
r"""Builds production PyTorch wheels based on the rocm-sdk wheels.

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
python pytorch_torch_repo.py checkout
python pytorch_torch_audio_repo.py checkout
python pytorch_torch_vision_repo.py checkout
```

Note that as of 2025-05-28, some small patches are needed to PyTorch's `__init__.py`
to enable library resolution from `rocm-sdk` wheels. We will aim to land this at head
in the PyTorch 2.8 timeframe.

2. Install rocm-sdk wheels:

You must have the `rocm-sdk[libraries,devel]` packages installed. The `install-rocm`
command gives a one-stop to fetch the latest nightlies from the CI or elsewhere.
Below we are using nightly rocm-sdk packages from the CI bucket. See `RELEASES.md`
for further options. Specific versions can be specified via `--rocm-sdk-version`
and `--no-pre` (to disable searching for pre-release candidates). The installed
version will be printed and subsequently will be embedded into torch builds as
a dependency. Such an arrangement is a head-on-head build (i.e. torch head on top
of ROCm head). Other arrangements are possible by passing pinned versions, official
repositories, etc.

You can also install in the same invocation as build by passing `--install-rocm` to the
build sub-command (useful for docker invocations).

```
build_prod_wheels.py
    --find-links https://therock-nightly-python.s3.us-east-2.amazonaws.com/gfx110X-dgpu/index.html \
    install-rocm
```

3. Build torch, torchaudio and torchvision for a single gfx architecture.

Typical usage to build a single arch from the host:

```
python build_prod_wheels.py build \
    --pytorch-rocm-arch=gfx1100 \
    --output-dir $HOME/tmp/pyout
```

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
    --pip-cache-dir /therock/output/pip_cache \
    --find-links https://therock-nightly-python.s3.us-east-2.amazonaws.com/gfx110X-dgpu/index.html \
    build \
        --install-rocm \
        --clean \
        --output-dir /therock/output/cp312/wheels \
        --pytorch-rocm-arch gfx1100
```

TODO: Need to add an option to post-process wheels, set the manylinux tag, and
inline system deps into the audio and vision wheels as needed.
"""

import argparse
from datetime import date
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import tempfile

script_dir = Path(__file__).resolve().parent


def exec(args: list[str | Path], cwd: Path, env: dict[str, str] | None = None):
    args = [str(arg) for arg in args]
    full_env = dict(os.environ)
    print(f"++ Exec [{cwd}]$ {shlex.join(args)}")
    if env:
        print(f":: Env:")
        for k, v in env.items():
            print(f"  {k}={v}")
        full_env.update(env)
    subprocess.check_call(args, cwd=str(cwd), env=full_env)


def capture(args: list[str | Path], cwd: Path) -> str:
    args = [str(arg) for arg in args]
    return subprocess.check_output(args, cwd=str(cwd)).decode()


def get_rocm_sdk_version() -> str:
    # TODO: We should have a `rocm-sdk version` command, but alas, we do not.
    freeze_lines = capture(
        [sys.executable, "-m", "pip", "freeze"], cwd=Path.cwd()
    ).splitlines()
    for line in freeze_lines:
        prefix = "rocm-sdk=="
        if line.startswith(prefix):
            return line[len(prefix) :]
    raise ValueError(f"No rocm-sdk found in {' '.join(freeze_lines)}")


def get_rocm_path(path_name: str) -> Path:
    return Path(
        capture(
            [sys.executable, "-m", "rocm_sdk", "path", f"--{path_name}"], cwd=Path.cwd()
        ).strip()
    )


def remove_dir_if_exists(dir: Path):
    if dir.exists():
        print(f"++ Removing {dir}")
        shutil.rmtree(dir)


def find_built_wheel(dist_dir: Path, dist_package: str) -> Path:
    glob = f"{dist_package}-*.whl"
    all_wheels = list(dist_dir.glob(glob))
    if not all_wheels:
        raise RuntimeError(f"No wheels matching '{glob}' found in {dist_dir}")
    if len(all_wheels) != 1:
        raise RuntimeError(f"Found multiple wheels matching '{glob}' in {dist_dir}")
    return all_wheels[0]


def copy_to_output(args: argparse.Namespace, src_file: Path):
    output_dir: Path = args.output_dir
    print(f"++ Copy {src_file} -> {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_file, output_dir)


def directory_if_exists(dir: Path) -> Path | None:
    if dir.exists():
        return dir
    else:
        return None


def do_install_rocm(args: argparse.Namespace):
    # Because the rocm-sdk package caches current GPU selection and such, we
    # always purge it to ensure a clean rebuild.
    exec([sys.executable, "-m", "pip", "cache", "remove", "rocm_sdk"], cwd=Path.cwd())

    # Do the main pip install.
    pip_args = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
    ]
    if args.pre:
        pip_args.extend(["--pre"])
    if args.find_links:
        pip_args.extend(["--find-links", args.find_links])
    if args.pip_cache_dir:
        pip_args.extend(["--cache-dir", args.pip_cache_dir])
    rocm_sdk_version = args.rocm_sdk_version if args.rocm_sdk_version else ""
    pip_args.extend([f"rocm-sdk[libraries,devel]{rocm_sdk_version}"])
    exec(pip_args, cwd=Path.cwd())
    print(f"Installed version: {get_rocm_sdk_version()}")


def do_build(args: argparse.Namespace):
    if args.install_rocm:
        do_install_rocm(args)

    pytorch_dir: Path | None = args.pytorch_dir
    pytorch_audio_dir: Path | None = args.pytorch_audio_dir
    pytorch_vision_dir: Path | None = args.pytorch_vision_dir

    rocm_sdk_version = get_rocm_sdk_version()
    cmake_prefix = get_rocm_path("cmake")
    bin_dir = get_rocm_path("bin")
    root_dir = get_rocm_path("root")

    print(f"rocm-sdk version {rocm_sdk_version}:")
    print(f"  PYTHON VERSION: {sys.version}")
    print(f"  CMAKE_PREFIX_PATH = {cmake_prefix}")
    print(f"  BIN = {bin_dir}")
    print(f"  ROCM_HOME = {root_dir}")

    system_path = str(bin_dir) + os.path.pathsep + os.environ.get("PATH", "")
    print(f"  PATH = {system_path}")

    pytorch_rocm_arch = args.pytorch_rocm_arch
    print(f"  PYTORCH_ROCM_ARCH: {pytorch_rocm_arch}")

    env: dict[str, str] = {
        "CMAKE_PREFIX_PATH": str(cmake_prefix),
        "ROCM_HOME": str(root_dir),
        "PYTORCH_EXTRA_INSTALL_REQUIREMENTS": f"rocm-sdk[libraries]=={rocm_sdk_version}",
        "PYTORCH_ROCM_ARCH": pytorch_rocm_arch,
        # TODO: Should get the supported archs from the rocm-sdk install.
        "PYTORCH_ROCM_ARCH": pytorch_rocm_arch,
        # TODO: Figure out what is blocking GLOO and enable.
        "USE_GLOO": "OFF",
        # TODO: Fix source dep on rocprofiler and enable.
        "USE_KINETO": "OFF",
        # Workaround GCC12 compiler flags.
        "CXXFLAGS": " -Wno-error=maybe-uninitialized -Wno-error=uninitialized -Wno-error=restrict",
        "CPPFLAGS": "  -Wno-error=maybe-uninitialized -Wno-error=uninitialized -Wno-error=restrict",
    }

    if pytorch_dir:
        do_build_pytorch(args, pytorch_dir, dict(env))
    else:
        print("--- Not building pytorch (no --pytorch-dir)")

    if pytorch_audio_dir:
        do_build_pytorch_audio(args, pytorch_audio_dir, dict(env))
    else:
        print("--- Not build pytorch-audio (no --pytorch-audio-dir)")

    if pytorch_vision_dir:
        do_build_pytorch_vision(args, pytorch_vision_dir, dict(env))
    else:
        print("--- Not build pytorch-vision (no --pytorch-vision-dir)")


def do_build_pytorch(args: argparse.Namespace, pytorch_dir: Path, env: dict[str, str]):
    # Compute version.
    pytorch_build_version = (pytorch_dir / "version.txt").read_text().strip()
    pytorch_build_version += args.version_suffix
    print(f"  Default PYTORCH_BUILD_VERSION: {pytorch_build_version}")
    env["PYTORCH_BUILD_VERSION"] = pytorch_build_version
    env["PYTORCH_BUILD_NUMBER"] = args.pytorch_build_number

    print("+++ Uninstalling pytorch:")
    exec([sys.executable, "-m", "pip", "uninstall", "torch"], cwd=tempfile.gettempdir())

    print("+++ Installing pytorch requirements:")
    pip_install_args = []
    if args.pip_cache_dir:
        pip_install_args.extend(["--cache-dir", args.pip_cache_dir])
    exec(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            pytorch_dir / "requirements.txt",
            # TODO: Remove cmake<4 pin once the world adapts (check at end of 2025).
            "cmake<4",
        ]
        + pip_install_args,
        cwd=pytorch_dir,
    )
    print("+++ Building pytorch:")
    remove_dir_if_exists(pytorch_dir / "dist")
    if args.clean:
        remove_dir_if_exists(pytorch_dir / "build")
    exec([sys.executable, "setup.py", "bdist_wheel"], cwd=pytorch_dir, env=env)
    built_wheel = find_built_wheel(pytorch_dir / "dist", "torch")
    print(f"Found built wheel: {built_wheel}")
    copy_to_output(args, built_wheel)

    print("+++ Installing built torch:")
    exec(
        [sys.executable, "-m", "pip", "install", built_wheel], cwd=tempfile.gettempdir()
    )


def do_build_pytorch_audio(
    args: argparse.Namespace, pytorch_audio_dir: Path, env: dict[str, str]
):
    # Compute version.
    build_version = (pytorch_audio_dir / "version.txt").read_text().strip()
    build_version += args.version_suffix
    print(f"  Default pytorch audio BUILD_VERSION: {build_version}")
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

    remove_dir_if_exists(pytorch_audio_dir / "dist")
    if args.clean:
        remove_dir_if_exists(pytorch_audio_dir / "build")

    exec([sys.executable, "setup.py", "bdist_wheel"], cwd=pytorch_audio_dir, env=env)
    built_wheel = find_built_wheel(pytorch_audio_dir / "dist", "torchaudio")
    print(f"Found built wheel: {built_wheel}")
    copy_to_output(args, built_wheel)


def do_build_pytorch_vision(
    args: argparse.Namespace, pytorch_vision_dir: Path, env: dict[str, str]
):
    # Compute version.
    build_version = (pytorch_vision_dir / "version.txt").read_text().strip()
    build_version += args.version_suffix
    print(f"  Default pytorch vision BUILD_VERSION: {build_version}")
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

    remove_dir_if_exists(pytorch_vision_dir / "dist")
    if args.clean:
        remove_dir_if_exists(pytorch_vision_dir / "build")

    exec([sys.executable, "setup.py", "bdist_wheel"], cwd=pytorch_vision_dir, env=env)
    built_wheel = find_built_wheel(pytorch_vision_dir / "dist", "torchvision")
    print(f"Found built wheel: {built_wheel}")
    copy_to_output(args, built_wheel)


def main(argv: list[str]):
    p = argparse.ArgumentParser(prog="build_prod_wheels.py")
    p.add_argument("--find-links", help="Pip find-links to pass to pip install")
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

    sub_p = p.add_subparsers(required=True)
    install_rocm_p = sub_p.add_parser(
        "install-rocm", help="Install rocm-sdk wheels to the current venv"
    )
    install_rocm_p.set_defaults(func=do_install_rocm)

    build_p = sub_p.add_parser("build", help="Build pytorch wheels")
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
        help="pytorch_audo source directory",
    )
    build_p.add_argument(
        "--pytorch-vision-dir",
        default=directory_if_exists(script_dir / "pytorch_vision"),
        type=Path,
        help="pytorch_vision source directory",
    )
    build_p.add_argument(
        "--pytorch-rocm-arch", required=True, help="gfx arch to build pytorch with"
    )
    build_p.add_argument(
        "--pytorch-build-number", default="1", help="Build number to append to version"
    )
    today = date.today()
    formatted_date = today.strftime("%Y%m%d")
    build_p.add_argument(
        "--version-suffix",
        default=f"+rocmsdk{formatted_date}",
        help="PyTorch version suffix",
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
