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
own and specify `--root-checkout-dir` or the more specific `--pytorch-dir`,
`--pytorch-audio-dir`, and `--pytorch-vision-dir` options during the build step.

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

3. Build torch, torchaudio and torchvision for one or more gfx architectures.

Target architectures are resolved in priority order from `--pytorch-rocm-arch`
(comma-separated), the `PYTORCH_ROCM_ARCH` environment variable, and finally
`rocm-sdk targets` from the installed rocm-sdk-core. Passing the flag or env
var explicitly is preferred; see TODO on get_rocm_sdk_targets.

```
# On Linux, using default paths for each repository:
python build_prod_wheels.py build \
    --pytorch-rocm-arch gfx942 \
    --output-dir $HOME/tmp/pyout

# On Windows, using shorter custom paths:
python build_prod_wheels.py build ^
    --pytorch-rocm-arch gfx1201 ^
    --output-dir %HOME%/tmp/pyout ^
    --pytorch-dir C:/b/pytorch ^
    --pytorch-audio-dir C:/b/audio ^
    --pytorch-vision-dir C:/b/vision
```

4. Compiler caching (optional):

```
# Use ccache:
python build_prod_wheels.py build --use-ccache --output-dir ...

# Use sccache (caches host + HIP device code via HIP_CLANG_LAUNCHER):
python build_prod_wheels.py build --use-sccache --output-dir ...

# Use sccache for host C/C++ only (no HIP device code caching):
python build_prod_wheels.py build --use-sccache --sccache-no-wrap --output-dir ...
```

``--use-ccache`` and ``--use-sccache`` are mutually exclusive.
``--use-sccache`` sets the CMake C/C++ compiler launchers and, on Linux, the
``HIP_CLANG_LAUNCHER`` environment variable so that ``hipcc`` routes its clang
invocations — including the ``-x hip --offload-arch`` device passes — through
sccache. The real clang binary is left in place, so compiler-detection probes
work normally. ``--sccache-no-wrap`` skips ``HIP_CLANG_LAUNCHER`` (host C/C++
caching only), for toolchains whose hipcc predates HIP_CLANG_LAUNCHER support
(ROCm < 7.13). See ``build_tools/setup_sccache_rocm.py``.

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
from importlib import metadata
import json
import os
from pathlib import Path
from packaging.specifiers import SpecifierSet
from packaging.version import parse
import platform
import shutil
import shlex
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import urllib.request

script_dir = Path(__file__).resolve().parent

is_windows = platform.system() == "Windows"

# LLVM download URL for triton-windows
LLVM_BASE_URL = "https://oaitriton.blob.core.windows.net/public/llvm-builds"

# List of library preloads for Linux to generate into _rocm_init.py.
# These are loaded with RTLD_GLOBAL on `import torch` via _rocm_init.py so
# that their symbols are available via dlsym(RTLD_DEFAULT, ...) without
# requiring a successful dlopen by unversioned name (which fails in wheel
# installs where only the versioned .so exists in the runtime package).
LINUX_LIBRARY_PRELOADS = [
    "amd_comgr",
    "amd_smi",
    "amdhip64",
    "rocprofiler-sdk",  # Linux only: needed by torch since kineto uses rocprofiler-sdk.
    "rocprofiler-sdk-roctx",  # Linux only for the moment.
    # TODO: Remove roctracer64 and roctx64 once fully switched to rocprofiler-sdk.
    "roctracer64",  # Linux only for the moment.
    "roctx64",  # Linux only for the moment.
    "hiprtc",
    "hipblas",
    "hipfft",
    "hiprand",
    "hipsparse",
    "hipsparselt",
    "hipsolver",
    "rccl",  # Linux only for the moment.
    "hipblaslt",
    "miopen",
    "hipdnn",
    "rocm_sysdeps_liblzma",
    "rocm-openblas",
    "rocm_smi64",
]

ASAN_SUPPORTED_ROCM = (10, 1)
ASAN_SUPPORTED_ARCH = "gfx942:xnack+"
ASAN_DEFAULT_OPTIONS = "detect_leaks=0:abort_on_error=1:print_stacktrace=1"
ASAN_CMAKE_ARGS = ("-DCMAKE_CXX_SCAN_FOR_MODULES=OFF",)
ASAN_REQUIRED_LOCAL_PACKAGES = {
    "rocm",
    "rocm-sdk-core",
    "rocm-sdk-devel",
    "rocm-sdk-device-gfx942",
    "rocm-sdk-libraries",
}
ASAN_BOOTSTRAP_REQUIREMENTS = {
    "setuptools": SpecifierSet(">=70.2"),
    "wheel": SpecifierSet(""),
}

# List of library preloads for Windows to generate into _rocm_init.py
WINDOWS_LIBRARY_PRELOADS = [
    "amd_comgr",
    "amdhip64",
    "hiprtc",
    "hipblas",
    "hipfft",
    "hiprand",
    "hipsparse",
    "hipsparselt",
    "hipsolver",
    "hipblaslt",
    "miopen",
    "hipdnn",
    "rocm-openblas",
]


def run_command(args: list[str | Path], cwd: Path, env: dict[str, str] | None = None):
    args = [str(arg) for arg in args]
    full_env = dict(os.environ)
    print(f"++ Exec [{cwd}]$ {shlex.join(args)}")
    if env:
        print(f":: Env:")
        for k, v in env.items():
            print(f"  {k}={v}")
        full_env.update(env)
    subprocess.check_call(args, cwd=str(cwd), env=full_env)


def capture(
    args: list[str | Path],
    cwd: Path,
    env: dict[str, str] | None = None,
) -> str:
    args = [str(arg) for arg in args]
    print(f"++ Capture [{cwd}]$ {shlex.join(args)}")
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    try:
        return subprocess.check_output(
            args,
            cwd=str(cwd),
            env=full_env,
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except subprocess.CalledProcessError as e:
        print(f"Error capturing output: {e}")
        print(f"Output from the failed command:\n{e.output}")
        return ""


def get_rocm_sdk_version() -> str:
    return capture(
        [sys.executable, "-m", "rocm_sdk", "version"], cwd=Path.cwd()
    ).strip()


# TODO(#4687): Remove this fallback once every caller passes --pytorch-rocm-arch
# (or PYTORCH_ROCM_ARCH) explicitly. Reading dist_amdgpu_targets from the
# installed rocm-sdk-core misreports targets in two known cases:
#   1. Multi-arch kpack-split builds share one superset rocm-sdk-core, so every
#      per-family job would compile torch for every arch in the superset.
#   2. Prebuilt-reuse flows where the installed rocm-sdk-core's targets do not
#      match the build intent (e.g. issue #4687).
# This fallback is kept for legacy CI and release workflows that have not yet
# been updated to plumb --pytorch-rocm-arch through from the caller.
def get_rocm_sdk_targets() -> str:
    # Run `rocm-sdk targets` to get the default architecture
    targets = capture([sys.executable, "-m", "rocm_sdk", "targets"], cwd=Path.cwd())
    if not targets:
        print("Warning: rocm-sdk targets returned empty or failed")
        return ""
    # Convert space-separated targets to comma-separated for PYTORCH_ROCM_ARCH
    return targets.replace(" ", ",")


def get_installed_package_version(dist_package_name: str) -> str:
    lines = capture(
        [sys.executable, "-m", "pip", "show", dist_package_name], cwd=Path.cwd()
    ).splitlines()
    if not lines:
        raise ValueError(f"Did not find installed package '{dist_package_name}'")
    prefix = "Version: "
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix) :]
    joined_lines = "\n".join(lines)
    raise ValueError(
        f"Did not find Version for installed package '{dist_package_name}' in output:\n{joined_lines}"
    )


def get_version_suffix_for_installed_rocm_package() -> str:
    rocm_version = get_installed_package_version("rocm")
    print(f"Computing version suffix for installed rocm package: {rocm_version}")
    # Compute a version suffix to be used as a local version identifier:
    # https://packaging.python.org/en/latest/specifications/version-specifiers/#local-version-identifiers
    # This logic is copied from build_tools/github_actions/determine_version.py.
    parsed_version = parse(rocm_version)
    base_name = "devrocm" if "dev" in rocm_version else "rocm"
    version_suffix = f"+{base_name}{str(parsed_version).replace('+','-')}"
    print(f"Version suffix is: {version_suffix}")
    return version_suffix


def validate_asan_rocm_version(rocm_version: str) -> None:
    """Reject a release or incompatible ROCm SDK before an ASAN build."""
    parsed_version = parse(rocm_version)
    if tuple(parsed_version.release[:2]) != ASAN_SUPPORTED_ROCM:
        raise RuntimeError(
            "--asan currently requires a ROCm 10.1 SDK; "
            f"found {rocm_version!r}"
        )
    local_parts = (parsed_version.local or "").split(".")
    if not local_parts or local_parts[0] != "asan" or len(local_parts) < 2:
        raise RuntimeError(
            "--asan requires a uniquely labelled ROCm ASAN SDK version "
            f"(expected 10.1.x+asan.<build-id>, found {rocm_version!r})"
        )


def get_asan_version_suffix(rocm_version: str) -> str:
    """Derive a collision-resistant torch local version from an ASAN SDK."""
    validate_asan_rocm_version(rocm_version)
    parsed_version = parse(rocm_version)
    major, minor = parsed_version.release[:2]
    return f"+rocm{major}.{minor}.{parsed_version.local}"


def resolve_asan_version_suffix(
    rocm_version: str, explicit_suffix: str | None
) -> str:
    expected_suffix = get_asan_version_suffix(rocm_version)
    if explicit_suffix and explicit_suffix != expected_suffix:
        raise RuntimeError(
            "--asan refuses an explicit torch version suffix that could "
            "collide with or misidentify the SDK: "
            f"expected {expected_suffix!r}, found {explicit_suffix!r}"
        )
    return expected_suffix


def validate_local_asan_index(find_links: str) -> str:
    """Validate the Phase 1 local index and return its coherent SDK version."""
    index_path = Path(find_links).expanduser()
    index_dir = index_path.parent if index_path.name == "index.html" else index_path
    manifest_path = index_dir / "index-manifest.json"
    if not index_dir.is_dir() or not manifest_path.is_file():
        raise ValueError(
            "--asan --install-rocm requires a local Phase 1 index directory "
            "(or its index.html) containing index-manifest.json; "
            f"not found at {index_dir}"
        )

    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid local ASAN index manifest {manifest_path}: {exc}")
    if manifest.get("index_kind") != "local-only":
        raise ValueError(
            f"ASAN index manifest must declare index_kind='local-only': {manifest_path}"
        )
    if manifest.get("relative_path") != "whl-asan/gfx942-all":
        raise ValueError(
            "ASAN index must be the gfx942 family index at "
            f"whl-asan/gfx942-all: {manifest_path}"
        )

    packages = manifest.get("packages")
    if not isinstance(packages, list):
        raise ValueError(f"ASAN index manifest has no package list: {manifest_path}")
    for package in packages:
        if not isinstance(package, dict):
            raise ValueError(
                f"ASAN index manifest contains a malformed package: {manifest_path}"
            )
        filename = package.get("filename")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not (index_dir / filename).is_file()
        ):
            raise ValueError(
                f"ASAN index package file is missing or invalid: {filename!r}"
            )
        declared_size = package.get("size")
        if (
            declared_size is not None
            and (index_dir / filename).stat().st_size != declared_size
        ):
            raise ValueError(
                f"ASAN index package size does not match its manifest: {filename}"
            )
    projects = {package.get("normalized_project") for package in packages}
    missing = sorted(ASAN_REQUIRED_LOCAL_PACKAGES - projects)
    if missing:
        raise ValueError(
            f"ASAN index is missing required packages {missing}: {manifest_path}"
        )
    versions = {package.get("version") for package in packages}
    if len(versions) != 1 or None in versions:
        raise ValueError(
            f"ASAN index packages do not have one coherent version: {manifest_path}"
        )
    version = versions.pop()
    if not isinstance(version, str):
        raise ValueError(f"ASAN index contains an invalid version: {version!r}")
    validate_asan_rocm_version(version)
    return version


def validate_asan_bootstrap_requirements() -> None:
    """Ensure the environment can build the local selector without isolation."""
    missing = []
    for distribution, requirement in ASAN_BOOTSTRAP_REQUIREMENTS.items():
        try:
            installed_version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            missing.append(f"{distribution}{requirement}")
            continue
        if requirement and not requirement.contains(
            installed_version, prereleases=True
        ):
            missing.append(f"{distribution}{requirement} (found {installed_version})")
    if missing:
        raise RuntimeError(
            "--asan local installation uses --no-build-isolation and requires "
            "bootstrap build dependencies in the current environment: "
            f"{', '.join(missing)}. Install them first with: "
            "python -m pip install 'setuptools>=70.2' wheel"
        )


def get_source_commit_short(source_dir: Path, length: int = 8) -> str:
    """Return the short git commit for a source dir, or "" if unresolved."""
    # Pass safe.directory via `-c` (scoped to this single git invocation)
    # instead of `git config --global` so we don't mutate global system state.
    # This keeps the lookup working even when the checkout is owned by another
    # user (e.g. in CI).
    commit = capture(
        [
            "git",
            "-c",
            f"safe.directory={source_dir}",
            "rev-parse",
            f"--short={length}",
            "HEAD",
        ],
        cwd=source_dir,
    )
    if not commit:
        print(f"WARNING: could not resolve source commit in '{source_dir}'")
    return commit


def compute_build_version(
    source_dir: Path, version_suffix: str, release_type: str
) -> str:
    """Compute a wheel version, tagging dev builds with the source commit.

    Reads `<source_dir>/version.txt` as the base version and appends
    `version_suffix` (a PEP 440 local identifier like `+rocm7.10.0`). For `dev`
    builds the 8-char source commit is merged into that single local segment,
    e.g. `2.12.0a0+git1a2b3c4d.rocm7.10.0`, so each wheel (torch, torchaudio,
    torchvision) records exactly which source commit produced it. PyTorch's
    setup.py validates the version as PEP 440, which only allows a commit hash
    in the local segment (after `+`).
    TODO(#5110): reconcile with generate_pytorch_source_manifest.py once
    upfront, manifest-based version computation lands so the built version
    always matches what the manifest records.
    """
    base_version = (source_dir / "version.txt").read_text().strip()
    build_version = base_version + version_suffix
    if release_type == "dev":
        commit = get_source_commit_short(source_dir)
        if commit:
            # version_suffix is a local identifier like `+rocm7.10.0`; merge the
            # commit into that single local segment (PEP 440 allows one `+`).
            local = version_suffix.lstrip("+")
            local_parts = [p for p in (f"git{commit}", local) if p]
            build_version = f"{base_version}+{'.'.join(local_parts)}"
    return build_version


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


def get_rocm_path(path_name: str) -> Path:
    return Path(
        capture(
            [sys.executable, "-m", "rocm_sdk", "path", f"--{path_name}"], cwd=Path.cwd()
        ).strip()
    )


def get_rocm_init_contents(args: argparse.Namespace):
    """Gets the contents of the _rocm_init.py file to add to the build."""
    sdk_version = get_rocm_sdk_version()
    library_preloads = (
        WINDOWS_LIBRARY_PRELOADS if is_windows else LINUX_LIBRARY_PRELOADS
    )
    library_preloads_formatted = ", ".join(f"'{s}'" for s in library_preloads)
    return textwrap.dedent(
        f"""
        def initialize():
            import rocm_sdk
            rocm_sdk.initialize_process(
                preload_shortnames=[{library_preloads_formatted}],
                check_version='{sdk_version}')
        """
    )


def remove_dir_if_exists(dir: Path):
    if dir.exists():
        print(f"++ Removing {dir}")
        shutil.rmtree(dir)


def find_built_wheel(dist_dir: Path, dist_package: str) -> Path:
    dist_package = dist_package.replace("-", "_")
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


def apply_root_checkout_dir(args: argparse.Namespace) -> None:
    """Default per-project source dirs from --root-checkout-dir."""
    root_checkout_dir: Path | None = args.root_checkout_dir
    if not root_checkout_dir:
        return

    if args.pytorch_dir is None:
        args.pytorch_dir = directory_if_exists(root_checkout_dir / "pytorch")
    if args.pytorch_audio_dir is None:
        args.pytorch_audio_dir = directory_if_exists(
            root_checkout_dir / "pytorch_audio"
        )
    if args.pytorch_vision_dir is None:
        args.pytorch_vision_dir = directory_if_exists(
            root_checkout_dir / "pytorch_vision"
        )
    if args.triton_dir is None:
        args.triton_dir = directory_if_exists(root_checkout_dir / "triton")
    if args.apex_dir is None:
        args.apex_dir = directory_if_exists(root_checkout_dir / "apex")


def validate_project_dir(
    parser: argparse.ArgumentParser,
    *,
    build_enabled: bool,
    source_dir: Path | None,
    build_option: str,
    dir_option: str,
) -> None:
    """Validate the source directory for an enabled project build."""
    if not build_enabled:
        return

    if source_dir is None:
        parser.error(
            f"{build_option} requires {dir_option} or a matching checkout "
            "under --root-checkout-dir"
        )
    if not source_dir.exists():
        parser.error(f"{dir_option} does not exist: {source_dir}")
    if not source_dir.is_dir():
        parser.error(f"{dir_option} is not a directory: {source_dir}")


def validate_build_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Resolve automatic project selections and validate build arguments."""
    # If a project dir exists, enable that project --build-* option by default.
    if args.build_triton is None:
        args.build_triton = args.triton_dir is not None
    if args.build_pytorch_audio is None:
        args.build_pytorch_audio = args.pytorch_audio_dir is not None
    if args.build_pytorch_vision is None:
        args.build_pytorch_vision = args.pytorch_vision_dir is not None
    if args.build_apex is None:
        args.build_apex = args.apex_dir is not None

    # If a build-* option is set, the *-dir option must point to a real directory.
    validate_project_dir(
        parser,
        build_enabled=args.build_triton,
        source_dir=args.triton_dir,
        build_option="--build-triton",
        dir_option="--triton-dir",
    )
    validate_project_dir(
        parser,
        build_enabled=args.build_pytorch_audio,
        source_dir=args.pytorch_audio_dir,
        build_option="--build-pytorch-audio",
        dir_option="--pytorch-audio-dir",
    )
    validate_project_dir(
        parser,
        build_enabled=args.build_pytorch_vision,
        source_dir=args.pytorch_vision_dir,
        build_option="--build-pytorch-vision",
        dir_option="--pytorch-vision-dir",
    )
    validate_project_dir(
        parser,
        build_enabled=args.build_apex,
        source_dir=args.apex_dir,
        build_option="--build-apex",
        dir_option="--apex-dir",
    )

    if (
        args.enable_pytorch_flash_attention
        and args.pytorch_dir is not None
        and not is_windows
        and not args.build_triton
        and not args.asan
    ):
        parser.error(
            "--enable-pytorch-flash-attention on Linux requires Triton; "
            "specify --triton-dir or disable Flash Attention"
        )

    if not args.asan:
        return
    if is_windows:
        parser.error("--asan is supported only on Linux x86_64")
    if platform.machine().lower() not in ("x86_64", "amd64"):
        parser.error("--asan is supported only on Linux x86_64")
    if args.pytorch_dir is None:
        parser.error("--asan requires a PyTorch checkout via --pytorch-dir")

    unsupported_builds = [
        option
        for enabled, option in (
            (args.build_triton, "--build-triton"),
            (args.build_pytorch_audio, "--build-pytorch-audio"),
            (args.build_pytorch_vision, "--build-pytorch-vision"),
            (args.build_apex, "--build-apex"),
        )
        if enabled
    ]
    if unsupported_builds:
        parser.error(
            "--asan Phase 2 builds torch only; disable " + ", ".join(unsupported_builds)
        )
    asan_extras = {
        extra.strip() for extra in args.rocm_extras.split(",") if extra.strip()
    }
    if asan_extras - {"device"}:
        parser.error(
            "--asan only supports the gfx942 single-target 'device' extra; "
            f"found {sorted(asan_extras)}"
        )

    requested_arch = args.pytorch_rocm_arch or os.environ.get("PYTORCH_ROCM_ARCH")
    if requested_arch is None:
        requested_arch = ASAN_SUPPORTED_ARCH
    if requested_arch.replace(",", ";") != ASAN_SUPPORTED_ARCH:
        parser.error(
            f"--asan Phase 2 requires --pytorch-rocm-arch {ASAN_SUPPORTED_ARCH}; "
            f"found {requested_arch!r}"
        )
    args.pytorch_rocm_arch = ASAN_SUPPORTED_ARCH

    if args.index_url:
        parser.error(
            "--asan local mode does not accept --index-url; use the Phase 1 "
            "index with --find-links to prevent mixing release packages"
        )
    if args.install_rocm:
        if not args.find_links:
            parser.error(
                "--asan --install-rocm requires --find-links pointing to the "
                "local Phase 1 whl-asan/gfx942-all index"
            )
        try:
            index_version = validate_local_asan_index(args.find_links)
            requested_versions = SpecifierSet(args.rocm_sdk_version)
        except (ValueError, RuntimeError) as exc:
            parser.error(str(exc))
        if not requested_versions.contains(index_version, prereleases=True):
            parser.error(
                f"--rocm-sdk-version {args.rocm_sdk_version!r} excludes local "
                f"ASAN SDK {index_version}"
            )
        # Install an exact coherent set even when the caller used the default
        # broad selector. This prevents a future local index addition from
        # silently changing the toolchain used by a retry.
        args.rocm_sdk_version = f"=={index_version}"
        args.asan_index_version = index_version


def do_install_rocm(args: argparse.Namespace):
    if getattr(args, "asan", False):
        validate_asan_bootstrap_requirements()

    # Because the rocm package caches current GPU selection and such, we
    # always purge it to ensure a clean rebuild.
    #
    # This can fail in environments where the pip cache is disabled or
    # unwritable (e.g. manylinux containers), which is fine — if there's no
    # cache, there's nothing stale to purge.
    cache_dir_args = (
        ["--cache-dir", str(args.pip_cache_dir)] if args.pip_cache_dir else []
    )
    try:
        run_command(
            [sys.executable, "-m", "pip", "cache", "remove", "rocm"] + cache_dir_args,
            cwd=Path.cwd(),
        )
    except subprocess.CalledProcessError:
        print("Warning: pip cache remove failed (cache may be disabled), continuing")

    # Do the main pip install.
    pip_args = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
    ]
    if getattr(args, "asan", False) or getattr(args, "no_index", False):
        pip_args.append("--no-index")
    if getattr(args, "asan", False):
        # The local Phase 1 index intentionally contains only the ROCm package
        # set, not generic build dependencies. Reuse the explicitly prepared
        # environment when pip builds the selector sdist.
        pip_args.append("--no-build-isolation")
    if args.pre:
        pip_args.extend(["--pre"])
    if args.index_url:
        pip_args.extend(["--index-url", args.index_url])
    if args.find_links:
        pip_args.extend(["--find-links", args.find_links])
    if args.pip_cache_dir:
        pip_args.extend(["--cache-dir", str(args.pip_cache_dir)])
    rocm_sdk_version = args.rocm_sdk_version if args.rocm_sdk_version else ""
    extras = ["libraries", "devel"]
    requested_extras = [
        extra.strip() for extra in args.rocm_extras.split(",") if extra.strip()
    ]
    if getattr(args, "asan", False):
        # A single-target selector exposes `device`, which resolves to the
        # gfx942 KPACK wheel. `device-gfx942` is only a multi-target extra.
        requested_extras.append("device")
    extras.extend(requested_extras)
    # Preserve order while avoiding `device,device` when an orchestrator makes
    # the automatic ASAN choice explicit.
    extras = list(dict.fromkeys(extras))
    pip_args.extend([f"rocm[{','.join(extras)}]{rocm_sdk_version}"])
    run_command(pip_args, cwd=Path.cwd())
    print(f"Installed version: {get_rocm_sdk_version()}")


def add_env_compiler_flags(env: dict[str, str], flagname: str, *compiler_flags: str):
    current = env.get(flagname, "")
    append = ""
    for compiler_flag in compiler_flags:
        append += f"{compiler_flag} "
    env[flagname] = f"{current}{append}"
    print(f"-- Appended {flagname}+={append}")


def find_dir_containing(file_name: str, *possible_paths: Path) -> Path:
    for path in possible_paths:
        if (path / file_name).exists():
            return path
    raise ValueError(f"No directory contains {file_name}: {possible_paths}")


def _setup_common_build_env(
    cmake_prefix: Path,
    bin_dir: Path,
    rocm_dir: Path,
    pytorch_rocm_arch: str,
    triton_dir: Path | None,
    is_windows: bool,
    asan: bool = False,
) -> dict[str, str]:
    """Construct the common environment dict shared by all wheel builds."""
    env: dict[str, str] = {
        "PYTHONUTF8": "1",  # Some build files use utf8 characters, force IO encoding
        "CMAKE_PREFIX_PATH": str(cmake_prefix),
        "ROCM_HOME": str(rocm_dir),
        "ROCM_PATH": str(rocm_dir),
        "PYTORCH_ROCM_ARCH": pytorch_rocm_arch,
        "USE_KINETO": os.environ.get("USE_KINETO", "ON" if not is_windows else "OFF"),
        # Make ROCm tools discoverable on all platforms and ROCm DLLs
        # discoverable by the Windows loader.
        "PATH": str(bin_dir) + os.path.pathsep + os.environ.get("PATH", ""),
    }

    env["USE_GLOO"] = "ON"

    # At checkout, we compute some additional env vars that influence the way that
    # the wheel is named/versioned.
    if triton_dir:
        triton_env_file = triton_dir / "build_env.json"
        if triton_env_file.exists():
            with open(triton_env_file, "r") as f:
                addl_triton_env = json.load(f)
                print(f"-- Additional triton build env vars: {addl_triton_env}")
            env.update(addl_triton_env)
        # With `CMAKE_PREFIX_PATH` set, `find_package(LLVM)` (called in
        # `MLIRConfig.cmake` shipped as part of the LLVM bundled with
        # trition) may pick up TheRock's LLVM instead of triton's.
        # Here, `CMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH` is set
        # and passed via `TRITON_APPEND_CMAKE_ARGS` to avoid this.
        # See also https://github.com/ROCm/TheRock/issues/1999.
        env["TRITON_APPEND_CMAKE_ARGS"] = (
            "-DCMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH=FALSE"
        )

    if is_windows:
        llvm_dir = rocm_dir / "lib" / "llvm" / "bin"
        env.update(
            {
                "HIP_CLANG_PATH": str(llvm_dir.resolve().as_posix()),
                "CC": str((llvm_dir / "clang-cl.exe").resolve()),
                "CXX": str((llvm_dir / "clang-cl.exe").resolve()),
            }
        )
    elif not asan:
        env.update(
            {
                # Workaround GCC12 compiler flags.
                "CXXFLAGS": " -Wno-error=maybe-uninitialized -Wno-error=uninitialized -Wno-error=restrict ",
                "CPPFLAGS": " -Wno-error=maybe-uninitialized -Wno-error=uninitialized -Wno-error=restrict ",
            }
        )

    # Workaround missing devicelib bitcode
    # TODO: When "ROCM_PATH" and/or "ROCM_HOME" is set in the environment, the
    # clang frontend ignores its default heuristics and (depending on version)
    # finds the wrong path to the device library. This is bad/annoying. But
    # the PyTorch build shouldn't even need these to be set. Unfortunately, it
    # has been hardcoded for a long time. So we use a clang env var to force
    # a specific device lib path to workaround the hack to get pytorch to build.
    # This may or may not only affect the Python wheels with their own quirks
    # on directory layout.
    # Obviously, this should be completely burned with fire once the root causes
    # are eliminted.
    hip_device_lib_path = rocm_dir / "lib" / "llvm" / "amdgcn" / "bitcode"
    if not hip_device_lib_path.exists():
        print(
            "WARNING: Default location of device libs not found. Relying on "
            "clang heuristics which are known to be buggy in this configuration"
        )
    else:
        env["HIP_DEVICE_LIB_PATH"] = str(hip_device_lib_path)

    # OpenBLAS path setup
    host_math_path = rocm_dir / "lib" / "host-math"
    if not host_math_path.exists():
        print(
            "WARNING: Default location of host-math not found. "
            "Will not build with OpenBLAS support."
        )
    else:
        env["BLAS"] = "OpenBLAS"
        env["OpenBLAS_HOME"] = str(host_math_path)
        env["OpenBLAS_LIB_NAME"] = "rocm-openblas"

    return env


def _setup_asan_build_env(rocm_dir: Path, pytorch_rocm_arch: str) -> dict[str, str]:
    """Validate the ROCm Clang/ASAN payload and construct ASAN build settings."""
    if is_windows or platform.machine().lower() not in ("x86_64", "amd64"):
        raise RuntimeError("--asan is supported only on Linux x86_64")
    if pytorch_rocm_arch != ASAN_SUPPORTED_ARCH:
        raise RuntimeError(
            f"--asan requires PYTORCH_ROCM_ARCH={ASAN_SUPPORTED_ARCH}; "
            f"found {pytorch_rocm_arch!r}"
        )

    llvm_bin = rocm_dir / "lib" / "llvm" / "bin"
    clang = llvm_bin / "clang"
    clangxx = llvm_bin / "clang++"
    for compiler in (clang, clangxx):
        if not compiler.is_file() or not os.access(compiler, os.X_OK):
            raise RuntimeError(
                f"--asan requires the executable ROCm compiler {compiler}"
            )

    hip_device_lib_path = rocm_dir / "lib" / "llvm" / "amdgcn" / "bitcode"
    if not hip_device_lib_path.is_dir() or not any(
        hip_device_lib_path.glob("*.bc")
    ):
        raise RuntimeError(
            "--asan requires ROCm device bitcode under "
            f"{hip_device_lib_path}"
        )

    runtime_name = f"libclang_rt.asan-{platform.machine().lower()}.so"
    runtime_text = capture(
        [clangxx, f"-print-file-name={runtime_name}"], cwd=rocm_dir
    )
    runtime_path = Path(runtime_text)
    if (
        not runtime_text
        or runtime_text == runtime_name
        or not runtime_path.is_absolute()
        or not runtime_path.is_file()
    ):
        raise RuntimeError(
            "ROCm clang++ did not resolve its shared ASAN runtime: "
            f"expected {runtime_name}, got {runtime_text!r}"
        )
    try:
        runtime_path.resolve().relative_to(rocm_dir.resolve())
    except ValueError:
        # Some Linux venvs materialize the same wheel under both `lib` and
        # `lib64`. The rocm-sdk root can be the lib64 copy while clang reports
        # its resource dir through the equivalent lib copy. Accept only when
        # the reported suffix also exists below this exact SDK payload root.
        try:
            payload_index = len(runtime_path.parts) - 1 - list(
                reversed(runtime_path.parts)
            ).index(rocm_dir.name)
            payload_relative = Path(*runtime_path.parts[payload_index + 1 :])
        except ValueError:
            payload_relative = Path()
        if (
            payload_relative.parts[:4] != ("lib", "llvm", "lib", "clang")
            or not (rocm_dir / payload_relative).is_file()
        ):
            raise RuntimeError(
                "ROCm clang++ resolved ASAN outside the installed ROCm SDK, "
                f"which could mix sanitizer runtimes: {runtime_path}"
            )

    inherited_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    ld_library_parts = [str(runtime_path.parent), str(rocm_dir / "lib")]
    if inherited_ld_library_path:
        ld_library_parts.append(inherited_ld_library_path)
    inherited_path = os.environ.get("PATH", "")
    path_parts = [str(llvm_bin)]
    if inherited_path:
        path_parts.append(inherited_path)
    inherited_cmake_args = os.environ.get("CMAKE_ARGS", "")
    asan_cmake_args = " ".join(ASAN_CMAKE_ARGS)
    cmake_args = (
        f"{inherited_cmake_args} {asan_cmake_args}"
        if inherited_cmake_args
        else asan_cmake_args
    )

    return {
        "USE_ASAN": "1",
        "USE_ROCM": "1",
        "USE_CUDA": "0",
        "USE_NINJA": "1",
        "CC": str(clang),
        "CXX": str(clangxx),
        "CMAKE_C_COMPILER": str(clang),
        "CMAKE_CXX_COMPILER": str(clangxx),
        "PYTORCH_ROCM_ARCH": pytorch_rocm_arch,
        "HIP_DEVICE_LIB_PATH": str(hip_device_lib_path),
        "ASAN_OPTIONS": os.environ.get("ASAN_OPTIONS", ASAN_DEFAULT_OPTIONS),
        "CFLAGS": "-fno-omit-frame-pointer ",
        "CXXFLAGS": "-fno-omit-frame-pointer ",
        "LDFLAGS": "-shared-libasan ",
        "LD_LIBRARY_PATH": os.path.pathsep.join(ld_library_parts),
        "PATH": os.path.pathsep.join(path_parts),
        # scikit-build-core forwards CMAKE_ARGS to its configure invocation.
        # CMake 4.4 otherwise enables C++20 dependency scanning automatically
        # and requires clang-scan-deps, which is intentionally absent from the
        # Phase 1 ROCm devel wheel.
        "CMAKE_ARGS": cmake_args,
        # Private hand-off to do_build. It is removed before the environment is
        # passed to any build subprocess and used only for the post-build import
        # probe's LD_PRELOAD.
        "_THEROCK_ASAN_RUNTIME_PATH": str(runtime_path),
    }


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
    if args.build_triton:
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
    if args.build_pytorch_audio:
        do_build_pytorch_audio(args, pytorch_audio_dir, dict(env))
    else:
        print("--- Not build pytorch-audio (no --pytorch-audio-dir)")

    # Build pytorch vision.
    if args.build_pytorch_vision:
        do_build_pytorch_vision(args, pytorch_vision_dir, dict(env))
    else:
        print("--- Not build pytorch-vision (no --pytorch-vision-dir)")

    # Build apex.
    if args.build_apex:
        do_build_apex(args, apex_dir, dict(env))
    else:
        print("--- Not build apex (no --apex-dir)")

    print("--- Builds all completed")


def do_build(args: argparse.Namespace):
    if args.install_rocm:
        do_install_rocm(args)

    triton_dir: Path | None = args.triton_dir
    pytorch_dir: Path | None = args.pytorch_dir
    pytorch_audio_dir: Path | None = args.pytorch_audio_dir
    pytorch_vision_dir: Path | None = args.pytorch_vision_dir
    apex_dir: Path | None = args.apex_dir

    rocm_sdk_version = get_rocm_sdk_version()
    if args.asan:
        validate_asan_rocm_version(rocm_sdk_version)
        args.version_suffix = resolve_asan_version_suffix(
            rocm_sdk_version, args.version_suffix
        )
        index_version = getattr(args, "asan_index_version", None)
        if index_version and index_version != rocm_sdk_version:
            raise RuntimeError(
                f"Installed ROCm SDK {rocm_sdk_version} does not match local "
                f"ASAN index {index_version}"
            )
    elif not args.version_suffix:
        args.version_suffix = get_version_suffix_for_installed_rocm_package()
    cmake_prefix = get_rocm_path("cmake")
    bin_dir = get_rocm_path("bin")
    rocm_dir = get_rocm_path("root")

    print(f"rocm version {rocm_sdk_version}:")
    print(f"  PYTHON VERSION: {sys.version}")
    print(f"  CMAKE_PREFIX_PATH = {cmake_prefix}")
    print(f"  BIN = {bin_dir}")
    print(f"  ROCM_HOME = {rocm_dir}")

    # Priority: --pytorch-rocm-arch > PYTORCH_ROCM_ARCH env > `rocm-sdk targets`
    # fallback (legacy; see TODO on get_rocm_sdk_targets()).
    pytorch_rocm_arch = args.pytorch_rocm_arch or os.environ.get("PYTORCH_ROCM_ARCH")
    if pytorch_rocm_arch:
        print(f"  Using provided PYTORCH_ROCM_ARCH: {pytorch_rocm_arch}")
    else:
        pytorch_rocm_arch = get_rocm_sdk_targets()
        print(
            f"  Using default PYTORCH_ROCM_ARCH from rocm-sdk targets: {pytorch_rocm_arch}"
        )

    if not pytorch_rocm_arch:
        raise ValueError(
            "No --pytorch-rocm-arch provided, PYTORCH_ROCM_ARCH not set, and "
            "rocm-sdk targets returned empty. "
            "Please specify --pytorch-rocm-arch (e.g., gfx942)."
        )

    # PyTorch's CMake consumes PYTORCH_ROCM_ARCH as a CMake-style list, so any
    # comma-separated input needs to be rewritten with semicolons before
    # CMake runs — otherwise the whole string is treated as one arch.
    pytorch_rocm_arch = pytorch_rocm_arch.replace(",", ";")

    env = _setup_common_build_env(
        cmake_prefix,
        bin_dir,
        rocm_dir,
        pytorch_rocm_arch,
        triton_dir,
        is_windows,
        asan=args.asan,
    )
    print(f"  PATH = {env['PATH']}")
    if args.asan:
        asan_env = _setup_asan_build_env(rocm_dir, pytorch_rocm_arch)
        args.asan_runtime_path = Path(asan_env.pop("_THEROCK_ASAN_RUNTIME_PATH"))
        # ROCm/TheRock#7210 provides the generic option and post-build archive
        # gate. ASAN builds always opt into that portable RPATH contract.
        args.pytorch_portable_rpath = True
        env.update(asan_env)

    if args.use_ccache:
        if not shutil.which("ccache"):
            raise RuntimeError(
                "ccache not found but --use-ccache was specified. "
                "Please install ccache before building."
            )
        print("Building with ccache, clearing stats first")
        env["CMAKE_C_COMPILER_LAUNCHER"] = "ccache"
        env["CMAKE_CXX_COMPILER_LAUNCHER"] = "ccache"
        if is_windows:
            # ccache does not support MSVC's /Zi flag. Embedded (/Z7) is needed.
            # See: https://github.com/ccache/ccache/issues/1040
            env["CMAKE_MSVC_DEBUG_INFORMATION_FORMAT"] = "Embedded"
        run_command(["ccache", "--zero-stats"], cwd=tempfile.gettempdir())
    elif args.use_sccache:
        build_tools_dir = Path(__file__).resolve().parent.parent.parent / "build_tools"
        sys.path.insert(0, str(build_tools_dir))

        from setup_sccache_rocm import find_sccache, sccache_build_env

        sccache_path = find_sccache()
        if not sccache_path:
            raise RuntimeError(
                "sccache not found but --use-sccache was specified.\n"
                "Install: https://github.com/mozilla/sccache#installation\n"
                "For CI, sccache is pre-installed in the manylinux build image:\n"
                "  https://github.com/ROCm/TheRock/tree/main/dockerfiles"
            )

    try:
        if args.use_sccache:
            # sccache_build_env sets the CMake C/C++ launchers and, unless
            # disabled, HIP_CLANG_LAUNCHER so hipcc routes its clang calls
            # (incl. the -x hip --offload-arch device passes) through sccache
            # without replacing the clang binary. See setup_sccache_rocm.py.
            hip_launcher = not args.sccache_no_wrap
            env.update(sccache_build_env(sccache_path, hip_launcher=hip_launcher))
            if hip_launcher and not is_windows:
                print(f"Setting up sccache via HIP_CLANG_LAUNCHER={sccache_path}")
            else:
                print("Setting up sccache (CMAKE launchers only)...")

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


def copy_msvc_libomp_to_torch_lib(pytorch_dir: Path):
    # When USE_OPENMP is set (it is by default), torch_cpu.dll depends on OpenMP.
    #
    # Typically implementations of OpenMP are:
    #   * Intel OpenMP, `libiomp`, which PyTorch upstream uses
    #   * MSVC OpenMP, `libomp140`, which we'll use here since we have MSVC already
    #   * (?) LLVM OpenMP (https://openmp.llvm.org/)?
    #
    # Torch's CMake build selects which OpenMP to use in `FindOpenMP.cmake`,
    # then the relevant .dll files must be copied into the torch/lib/ folder or
    # torch will fail to initialize. This feels like something that could be
    # handled upstream as part of the centralized setup.py and/or CMake build
    # processes, but given the varied scripts and build workflows upstream and
    # multiple choices for where to source an implementation, we handle it here.
    #
    # If we wanted to switch to Intel OpenMP, we could:
    #   1. Install Intel OpenMP (and/or MKL?)
    #   2. Set CMAKE_INCLUDE_PATH and CMAKE_LIBRARY_PATH (?) so `FindOpenMP.cmake` finds them
    #   3. Copy `libiomp5md.dll` to torch/lib
    # Then remove the rest of the code from this function.

    vc_tools_redist_dir = os.environ.get("VCToolsRedistDir", "")
    if not vc_tools_redist_dir:
        raise RuntimeError("VCToolsRedistDir not set, can't copy libomp to torch lib")

    omp_name = "libomp140.x86_64.dll"
    dll_paths = sorted(Path(vc_tools_redist_dir).rglob(omp_name))
    if not dll_paths:
        raise RuntimeError(
            f"Did not find '{omp_name}' under '{vc_tools_redist_dir}', can't copy libomp to torch lib"
        )

    omp_path = dll_paths[0]
    target_lib = pytorch_dir / "torch" / "lib"
    print(f"Copying libomp from '{omp_path}' to '{target_lib}'")
    shutil.copy2(omp_path, target_lib)


def copy_libuv_to_torch_lib(pytorch_dir: Path):
    libuv_root = os.environ.get("libuv_ROOT", "")
    if not libuv_root:
        return
    uv_dll = Path(libuv_root) / "bin" / "uv.dll"
    if not uv_dll.exists():
        raise RuntimeError(f"Did not find uv.dll at '{uv_dll}'")
    target_lib = pytorch_dir / "torch" / "lib"
    print(f"Copying libuv from '{uv_dll}' to '{target_lib}'")
    shutil.copy2(uv_dll, target_lib)


def resolve_pytorch_flash_attention(
    args: argparse.Namespace,
    env: dict[str, str],
    triton_requirement: str | None,
) -> bool:
    """Resolve Flash Attention without conflating Triton and AOTriton."""
    if args.asan:
        # Phase 2 does not build a separate Triton wheel. PyTorch's AOTriton
        # CMake integration selects its prebuilt +asan runtime/images when
        # USE_ASAN=1. Keep that proven gfx942 path enabled by default while
        # retaining an explicit opt-out for diagnostics.
        use_flash_attention = args.enable_pytorch_flash_attention is not False
        print(
            "ASAN AOTriton Flash Attention behavior: "
            f"{'enabled' if use_flash_attention else 'disabled'}"
        )
        return use_flash_attention
    if args.enable_pytorch_flash_attention is not None:
        use_flash_attention = args.enable_pytorch_flash_attention
        print(f"Flash Attention explicitly set to: {use_flash_attention}")
        return use_flash_attention
    if not is_windows and not triton_requirement:
        print("Disabling Flash Attention on Linux since triton is not built")
        return False

    # Enable aotriton by default if supported. AOTriton supports a subset of
    # GPU architectures. When at least one target arch is supported, let its
    # build system filter to supported targets. When none are supported its
    # configure step fails on the empty target list.
    aotriton_supported_arch_prefixes = (
        "gfx90a",
        "gfx942",
        "gfx950",
        "gfx11",
        "gfx12",
    )
    rocm_arch_list = env.get("PYTORCH_ROCM_ARCH", "").split(";")
    has_aotriton_supported_arch = any(
        arch.startswith(aotriton_supported_arch_prefixes) for arch in rocm_arch_list
    )
    print(
        f"Flash Attention default behavior: {has_aotriton_supported_arch}\n"
        f"  (has_aotriton_supported_arch: {has_aotriton_supported_arch})"
    )
    return has_aotriton_supported_arch


def get_pytorch_sanity_env(
    args: argparse.Namespace, build_env: dict[str, str]
) -> dict[str, str] | None:
    """Return environment overrides for the post-install torch import only."""
    if not args.asan:
        return None
    runtime_path = Path(args.asan_runtime_path)
    if not runtime_path.is_file():
        raise RuntimeError(
            f"Validated ASAN runtime disappeared before torch sanity check: {runtime_path}"
        )
    inherited_preload = os.environ.get("LD_PRELOAD", "")
    preload_parts = [str(runtime_path)]
    if inherited_preload:
        preload_parts.append(inherited_preload)
    sanity_env = {"LD_PRELOAD": os.path.pathsep.join(preload_parts)}
    # Keep ASAN's no-leak policy and the validated SDK library search path
    # scoped to the same subprocess. In particular, do not LD_PRELOAD the
    # sanitizer into pip, CMake, Ninja, or compiler processes.
    for env_name in ("ASAN_OPTIONS", "LD_LIBRARY_PATH"):
        if env_name in build_env:
            sanity_env[env_name] = build_env[env_name]
    return sanity_env


def sanity_check_installed_pytorch(
    args: argparse.Namespace, build_env: dict[str, str]
) -> None:
    print("+++ Sanity checking installed torch (unavailable is okay on CPU machines):")
    sanity_check_output = capture(
        [sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
        cwd=tempfile.gettempdir(),
        env=get_pytorch_sanity_env(args, build_env),
    )
    if not sanity_check_output:
        raise RuntimeError("torch package sanity check failed (see output above)")
    print(f"Sanity check output:\n{sanity_check_output}")


def do_build_pytorch(
    args: argparse.Namespace,
    pytorch_dir: Path,
    env: dict[str, str],
    *,
    triton_requirement: str | None,
):
    # Compute version (dev builds are tagged with the torch source commit).
    pytorch_build_version = compute_build_version(
        pytorch_dir, args.version_suffix, args.release_type
    )
    print(f"  Using PYTORCH_BUILD_VERSION: {pytorch_build_version}")

    env["USE_ROCM"] = "1" if args.asan else "ON"
    env["USE_CUDA"] = "0" if args.asan else "OFF"
    env["USE_MPI"] = "OFF"
    env["USE_NUMA"] = "OFF"
    env["PYTORCH_BUILD_VERSION"] = pytorch_build_version
    env["PYTORCH_BUILD_NUMBER"] = args.pytorch_build_number

    # Determine which install requirements to add.
    install_requirements = [
        f"rocm[libraries]=={get_rocm_sdk_version()}",
    ]
    if triton_requirement:
        install_requirements.append(triton_requirement)
    env["PYTORCH_EXTRA_INSTALL_REQUIREMENTS"] = "|".join(install_requirements)
    print(
        f"--- PYTORCH_EXTRA_INSTALL_REQUIREMENTS = {env['PYTORCH_EXTRA_INSTALL_REQUIREMENTS']}"
    )

    # Add the _rocm_init.py file.
    (pytorch_dir / "torch" / "_rocm_init.py").write_text(get_rocm_init_contents(args))

    # Enable/disable Flash Attention. ASAN uses prebuilt AOTriton +asan
    # artifacts and intentionally has no separate Triton wheel dependency.
    use_flash_attention = resolve_pytorch_flash_attention(
        args, env, triton_requirement
    )
    # Finally update the environment with the resolved setting.
    env.update(
        {
            "USE_FLASH_ATTENTION": ("ON" if use_flash_attention else "OFF"),
            "USE_MEM_EFF_ATTENTION": ("ON" if use_flash_attention else "OFF"),
        }
    )

    if is_windows:
        # Apply Windows-specific settings.
        copy_msvc_libomp_to_torch_lib(pytorch_dir)
        copy_libuv_to_torch_lib(pytorch_dir)

        env.update(
            {
                "DISTUTILS_USE_SDK": "1",
                # Workaround compile errors in 'aten/src/ATen/test/hip/hip_vectorized_test.hip'
                # on Torch 2.7.0: https://gist.github.com/ScottTodd/befdaf6c02a8af561f5ac1a2bc9c7a76.
                #   error: no member named 'modern' in namespace 'at::native'
                #     using namespace at::native::modern::detail;
                #   error: no template named 'has_same_arg_types'
                #     static_assert(has_same_arg_types<func1_t>::value, "func1_t has the same argument types");
                # We may want to fix that and other issues to then enable building tests.
                "BUILD_TEST": "0",
            }
        )
    else:
        # Apply Linux-specific settings.

        # Prepend the ROCm sysdeps dir so that we use bundled libraries.
        # While a decent thing to be doing, this is presently required because:
        # TODO: include/rocm_smi/kfd_ioctl.h is included without its advertised
        # transitive includes. This triggers a compilation error for a missing
        # libdrm/drm.h.
        rocm_dir = get_rocm_path("root")
        sysdeps_dir = rocm_dir / "lib" / "rocm_sysdeps"
        assert sysdeps_dir.exists(), f"No sysdeps directory found: {sysdeps_dir}"
        add_env_compiler_flags(env, "CXXFLAGS", f"-I{sysdeps_dir / 'include'}")
        # Add correct include path for roctracer.h (for Kineto)
        add_env_compiler_flags(
            env, "CXXFLAGS", f"-I{rocm_dir / 'include' / 'roctracer'}"
        )
        add_env_compiler_flags(env, "LDFLAGS", f"-L{sysdeps_dir / 'lib'}")

        # needed to find liblzma packaged by rocm as sysdep to build aotriton
        os.environ["PKG_CONFIG_PATH"] = f"{sysdeps_dir / 'lib' / 'pkgconfig'}"
        os.environ["LD_LIBRARY_PATH"] = f"{sysdeps_dir / 'lib'}"

    print("+++ Uninstalling pytorch:")
    run_command(
        [sys.executable, "-m", "pip", "uninstall", "torch", "-y"],
        cwd=tempfile.gettempdir(),
    )

    print("+++ Installing pytorch requirements:")
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
            pytorch_dir / "requirements.txt",
        ]
        + pip_install_args,
        cwd=pytorch_dir,
    )

    # PEP 517 build backend requirements. We build below with
    # `python -m build --wheel --no-isolation`, which (unlike an isolated build)
    # does not auto-install the backend declared in pyproject.toml's
    # [build-system]. PyTorch migrated its build backend from setuptools to
    # scikit-build-core (ROCm/TheRock#6523; setup.py no longer builds wheels on
    # recent checkouts). Newer checkouts ship requirements-build.txt (pinning
    # scikit-build-core>=1.0), older ones do not. `build` provides the
    # `python -m build` frontend itself.
    print("+++ Installing pytorch build backend requirements:")
    build_backend_install = [sys.executable, "-m", "pip", "install", "build"]
    pytorch_build_requirements = pytorch_dir / "requirements-build.txt"
    if pytorch_build_requirements.exists():
        build_backend_install += ["-r", pytorch_build_requirements]
    run_command(build_backend_install + pip_install_args, cwd=pytorch_dir)

    build_command = [sys.executable, "-m", "build", "--wheel", "--no-isolation"]
    pytorch_pyproject_text = (pytorch_dir / "pyproject.toml").read_text()
    if "scikit_build_core.build" in pytorch_pyproject_text:
        # scikit-build-core applies Git ignore rules when constructing the wheel,
        # dropping generated, gitignored runtime files. This workaround can be
        # removed once these fixes are merged and commonly available:
        # https://github.com/pytorch/pytorch/pull/191625
        # https://github.com/pytorch/pytorch/pull/191629
        build_command.append(
            "-Cwheel.force-include.torch/_rocm_init.py=torch/_rocm_init.py"
        )
        if is_windows:
            build_command.append(
                "-Cwheel.force-include.torch/lib/libomp140.x86_64.dll="
                "torch/lib/libomp140.x86_64.dll"
            )
            if use_flash_attention:
                # TODO: similar upstream fix for these and then drop here
                build_command.append(
                    "-Cwheel.force-include.torch/lib/aotriton_v2.dll="
                    "torch/lib/aotriton_v2.dll"
                )
                build_command.append(
                    "-Cwheel.force-include.torch/lib/liblzma.dll="
                    "torch/lib/liblzma.dll"
                )

    if is_windows:
        # The PyPI `ninja` package is unusable on Windows: 1.11.1 loops without
        # making progress and 1.13.0 has an MSVC link.exe RSP-file regression
        # (LNK1104/LNK1181), and no fixed version has been published
        # (scikit-build/ninja-python-distributions#308). requirements-build.txt
        # just pulled it in, so remove it; the runner provides a good system
        # ninja (>=1.13.1) on PATH that CMake uses instead.
        run_command(
            [sys.executable, "-m", "pip", "uninstall", "ninja", "-y"],
            cwd=pytorch_dir,
        )
        # With the PyPI ninja gone, `python -m build`'s PEP 517 dependency check
        # would abort with "Missing dependencies: ninja". Skip that check: the
        # backend drives CMake, which finds the system ninja on PATH.
        build_command.append("--skip-dependency-check")

    print("+++ Building pytorch:")
    remove_dir_if_exists(pytorch_dir / "dist")
    if args.clean:
        remove_dir_if_exists(pytorch_dir / "build")
    # `python -m build --wheel --no-isolation` is the standard replacement for
    # the removed `setup.py bdist_wheel` (ROCm/TheRock#6523), used on all
    # platforms. It drives the legacy setuptools backend and the new
    # scikit-build-core backend alike, and all build-configuration env vars
    # (USE_ROCM, PYTORCH_ROCM_ARCH, MAX_JOBS, ...) continue to be honored as they
    # now seed the CMake cache directly.
    run_command(build_command, cwd=pytorch_dir, env=env)
    built_wheel = find_built_wheel(pytorch_dir / "dist", "torch")
    print(f"Found built wheel: {built_wheel}")
    copy_to_output(args, built_wheel)

    print("+++ Installing built torch:")
    run_command(
        [sys.executable, "-m", "pip", "install", built_wheel], cwd=tempfile.gettempdir()
    )

    sanity_check_installed_pytorch(args, env)


def do_build_pytorch_audio(
    args: argparse.Namespace, pytorch_audio_dir: Path, env: dict[str, str]
):
    # Compute version (dev builds are tagged with the audio source commit).
    build_version = compute_build_version(
        pytorch_audio_dir, args.version_suffix, args.release_type
    )
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
    # Compute version (dev builds are tagged with the vision source commit).
    build_version = compute_build_version(
        pytorch_vision_dir, args.version_suffix, args.release_type
    )
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
        p.add_argument(
            "--no-index",
            action="store_true",
            default=False,
            help="Pass --no-index to pip. This is automatic in --asan mode.",
        )
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

    sub_p = p.add_subparsers(dest="command", required=True)
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
        "--asan",
        action="store_true",
        default=False,
        help="Build a ROCm 10.1 gfx942:xnack+ torch ASAN wheel from the local "
        "Phase 1 SDK index. Enables ROCm Clang and strict SDK/runtime preflight; "
        "Triton, sibling wheels, and remote package indexes are excluded.",
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
        help="Use sccache as the compiler launcher. Sets the CMake C/C++ "
        "launchers and (on Linux) HIP_CLANG_LAUNCHER so hipcc routes its clang "
        "invocations -- including the HIP device passes -- through sccache, "
        "caching host and device code. Requires hipcc with HIP_CLANG_LAUNCHER "
        "support (ROCm 7.13+).",
    )
    build_p.add_argument(
        "--sccache-no-wrap",
        action="store_true",
        default=False,
        help="With --use-sccache: set only the CMake C/C++ launchers and skip "
        "HIP_CLANG_LAUNCHER (caches host C/C++ but not HIP device code). Use "
        "when hipcc lacks HIP_CLANG_LAUNCHER support.",
    )
    build_p.add_argument(
        "--root-checkout-dir",
        default=script_dir,
        type=Path,
        help=(
            "Root directory containing PyTorch source checkouts named pytorch, "
            "pytorch_audio, pytorch_vision, triton, and apex. Explicit "
            "--pytorch-dir, --pytorch-audio-dir, --pytorch-vision-dir, "
            "--triton-dir, and --apex-dir arguments override this."
        ),
    )
    build_p.add_argument(
        "--pytorch-dir",
        default=None,
        type=Path,
        help="PyTorch source directory",
    )
    build_p.add_argument(
        "--pytorch-audio-dir",
        default=None,
        type=Path,
        help="pytorch_audio source directory",
    )
    build_p.add_argument(
        "--pytorch-vision-dir",
        default=None,
        type=Path,
        help="pytorch_vision source directory",
    )
    build_p.add_argument(
        "--triton-dir",
        default=None,
        type=Path,
        help="pinned triton directory",
    )
    build_p.add_argument(
        "--apex-dir",
        default=None,
        type=Path,
        help="apex source directory",
    )
    build_p.add_argument(
        "--pytorch-rocm-arch",
        help="Comma-separated gfx arches to build pytorch for (e.g. 'gfx942' or "
        "'gfx942,gfx1201'). May also be supplied via the PYTORCH_ROCM_ARCH "
        "environment variable. Falls back to `rocm-sdk targets` when unset "
        "(legacy; see TODO on get_rocm_sdk_targets).",
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
        "--enable-pytorch-flash-attention",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable building of torch flash attention (sets USE_FLASH_ATTENTION and USE_MEM_EFF_ATTENTION). Defaults to enabled if supported",
    )
    build_p.add_argument(
        "--version-suffix",
        help="Explicit PyTorch version suffix (e.g. `+rocm7.10.0a20251124`). Typically computed with build_tools/github_actions/determine_version.py. If omitted it will be derived from the installed rocm package",
    )
    build_p.add_argument(
        "--release-type",
        choices=["ci", "dev", "nightly", "prerelease"],
        default="nightly",
        help="Release type of the build. For `dev` builds the torch wheel "
        "version is tagged with the 8-char torch source commit in the local "
        "segment, e.g. `2.12.0a0+git1a2b3c4d.rocm7.10.0` (torch wheel only). "
        "The default is non-appending so other callers (CI, nightly, "
        "prerelease) keep their plain `<base>+<suffix>` versions.",
    )
    build_p.add_argument(
        "--clean",
        action=argparse.BooleanOptionalAction,
        help="Clean build directories before building",
    )
    build_p.set_defaults(func=do_build)

    args = p.parse_args(argv)
    if args.command == "build":
        apply_root_checkout_dir(args)
        validate_build_args(build_p, args)

    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
