#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Prepare JAX manylinux builds to use ROCm SDK LLVM.

Copies and registers the required XLA patches, removes the LLVM source-build
block from the manylinux Dockerfile, and configures JAX Bazel builds to use
ROCm SDK Clang and LLD for ROCm release wheels.
"""

import argparse
import re
import shutil
from pathlib import Path

PATCH_NAMES = ("0001-Add-clang-resource-dir-include-path.patch",)

DOCKERFILE_NAME = "Dockerfile.jax-manylinux_2_28-therock"

LLVM_BLOCK_START = "# Install LLVM 18 from source"
LLVM_BLOCK_END = "# ld.lld (hermetic linker)"

ROCM_RELEASE_CONFIG = "build:rocm_release_wheel"

HERMETIC_CLANG_CONFIG = "--config=rocm_clang_hermetic"
LOCAL_CLANG_CONFIG = "--config=rocm_clang_local"

LLD_RELEASE_CONFIG = f"{ROCM_RELEASE_CONFIG} --linkopt=-fuse-ld=lld"


def _apply_xla_patch(
    jax_source: Path,
    rocm_jax_source: Path,
    patch_name: str,
) -> None:
    """Copy and register one XLA patch in the JAX source tree."""

    source_patch = rocm_jax_source / "ci" / "patches" / "xla" / patch_name
    destination_patch = jax_source / "third_party" / "xla" / patch_name

    if not source_patch.is_file():
        raise FileNotFoundError(f"Missing XLA patch: {source_patch}")

    shutil.copy2(source_patch, destination_patch)

    patch_label = f"//third_party/xla:{patch_name}"
    workspace_path = jax_source / "third_party" / "xla" / "workspace.bzl"
    contents = workspace_path.read_text(encoding="utf-8")

    # Locate XLA's patch_file list while preserving its existing indentation.
    patch_list_pattern = re.compile(
        r"(?P<indent>^[ \t]*)patch_file\s*=\s*\[\n" r"(?P<body>.*?)" r"^(?P=indent)\],",
        re.MULTILINE | re.DOTALL,
    )
    match = patch_list_pattern.search(contents)

    if match is None:
        raise RuntimeError(f"Could not find patch_file list in {workspace_path}")

    if patch_label not in match.group("body"):
        entry_indent = match.group("indent") + "    "
        new_body = match.group("body") + f'{entry_indent}"{patch_label}",\n'
        contents = (
            contents[: match.start("body")] + new_body + contents[match.end("body") :]
        )
        workspace_path.write_text(contents, encoding="utf-8")

    build_path = jax_source / "third_party" / "xla" / "BUILD.bazel"
    build_contents = build_path.read_text(encoding="utf-8")
    export_statement = f'exports_files(["{patch_name}"])'

    if export_statement not in build_contents:
        build_path.write_text(
            build_contents.rstrip() + "\n\n" + export_statement + "\n",
            encoding="utf-8",
        )


def _prepare_dockerfile(
    jax_source: Path,
    rocm_jax_source: Path,
) -> None:
    """Create the JAX manylinux Dockerfile without building LLVM."""

    source = rocm_jax_source / "docker" / "manylinux" / DOCKERFILE_NAME
    destination = jax_source / DOCKERFILE_NAME

    if not source.is_file():
        raise FileNotFoundError(f"Missing upstream Dockerfile: {source}")

    contents = source.read_text(encoding="utf-8")

    if LLVM_BLOCK_START not in contents or LLVM_BLOCK_END not in contents:
        raise RuntimeError(
            "Could not find the expected LLVM source-build block " f"in {source}"
        )

    start = contents.index(LLVM_BLOCK_START)
    end = contents.index(LLVM_BLOCK_END, start)

    destination.write_text(
        contents[:start] + contents[end:],
        encoding="utf-8",
    )


def prepare(
    jax_source: Path,
    rocm_jax_source: Path,
) -> None:
    """Prepare a clean JAX checkout for a ROCm SDK LLVM build."""

    for patch_name in PATCH_NAMES:
        _apply_xla_patch(
            jax_source,
            rocm_jax_source,
            patch_name,
        )

    _prepare_dockerfile(
        jax_source,
        rocm_jax_source,
    )


def configure_build(
    jax_source: Path,
    rocm_root: Path,
) -> None:
    """Configure JAX Bazel settings to use ROCm SDK Clang and LLD."""

    clang = rocm_root / "lib" / "llvm" / "bin" / "clang"

    if not clang.is_file():
        raise FileNotFoundError(f"Missing ROCm Clang: {clang}")

    bazelrc_path = jax_source / "build" / "rocm" / "rocm.bazelrc"

    if not bazelrc_path.is_file():
        raise FileNotFoundError(f"Missing Bazel configuration: {bazelrc_path}")

    original = bazelrc_path.read_text(encoding="utf-8")
    lines = original.splitlines()

    updated_clang_paths = 0
    release_config_lines = 0

    for index, line in enumerate(lines):
        stripped = line.lstrip()

        if not stripped or stripped.startswith("#"):
            continue

        # Update the active ROCm configuration to use Clang from the
        # installed ROCm SDK rather than a separately built LLVM toolchain.
        if "CLANG_COMPILER_PATH=" in line:
            prefix = line.split(
                "CLANG_COMPILER_PATH=",
                maxsplit=1,
            )[0]

            lines[index] = f'{prefix}CLANG_COMPILER_PATH="{clang}"'
            updated_clang_paths += 1

        # Verify that the scoped release-wheel configuration exists before
        # adding the LLD linker option to it.
        stripped = lines[index].strip()

        if stripped == ROCM_RELEASE_CONFIG or stripped.startswith(
            f"{ROCM_RELEASE_CONFIG} "
        ):
            release_config_lines += 1

            # JAX 0.10.2+ defaults ROCm release wheels to the hermetic
            # toolchain. Use the local ROCm crosstool so CLANG_COMPILER_PATH
            # can point to Clang from the installed ROCm SDK.
            lines[index] = line.replace(
                HERMETIC_CLANG_CONFIG,
                LOCAL_CLANG_CONFIG,
            )

    if updated_clang_paths != 1:
        raise RuntimeError(
            "Expected exactly one active CLANG_COMPILER_PATH entry "
            f"in {bazelrc_path}; found {updated_clang_paths}"
        )

    if release_config_lines == 0:
        raise RuntimeError(
            "Missing an active rocm_release_wheel configuration " f"in {bazelrc_path}"
        )

    # Use LLD only for ROCm release-wheel links. Remove any previous exact
    # copy first so repeated configure-build calls remain idempotent.
    lines = [line for line in lines if line.strip() != LLD_RELEASE_CONFIG]

    while lines and not lines[-1].strip():
        lines.pop()

    lines.extend(
        [
            "",
            LLD_RELEASE_CONFIG,
        ]
    )

    contents = "\n".join(lines) + "\n"

    if contents != original:
        bazelrc_path.write_text(
            contents,
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--jax-source",
        type=Path,
        required=True,
    )
    prepare_parser.add_argument(
        "--rocm-jax-source",
        type=Path,
        required=True,
    )

    configure_parser = subparsers.add_parser("configure-build")
    configure_parser.add_argument(
        "--jax-source",
        type=Path,
        required=True,
    )
    configure_parser.add_argument(
        "--rocm-root",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    if args.command == "prepare":
        prepare(
            args.jax_source.resolve(),
            args.rocm_jax_source.resolve(),
        )
    elif args.command == "configure-build":
        configure_build(
            args.jax_source.resolve(),
            args.rocm_root.resolve(),
        )


if __name__ == "__main__":
    main()
