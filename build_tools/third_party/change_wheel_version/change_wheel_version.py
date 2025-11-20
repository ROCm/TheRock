#################################################################################
# original version from https://github.com/hauntsaninja/change_wheel_version
# (commit e28436ebdb6fe5a81cd29922f7276d00675d1bd3, Sep 13, 2024)
#
# modified to allow a hook to update the version in other files part of the wheel
# modified parts are marked with " # AMD-added "
#################################################################################

import argparse
import email.parser
import email.policy
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Optional, no_type_check

import installer.utils
import packaging.version
from packaging.tags import parse_tag

# AMD-added
from collections.abc import Callable


def version_replace(
    v: packaging.version.Version, **kwargs: Any
) -> packaging.version.Version:
    # yikes :-)
    self = packaging.version.Version.__new__(packaging.version.Version)
    self._version = v._version._replace(**kwargs)
    return packaging.version.Version(str(self))


class ExecutablePreservingZipfile(zipfile.ZipFile):
    @no_type_check
    def _extract_member(self, member, targetpath, pwd):
        if not isinstance(member, zipfile.ZipInfo):
            member = self.getinfo(member)

        targetpath = super()._extract_member(member, targetpath, pwd)

        mode = member.external_attr >> 16
        if mode != 0:
            os.chmod(targetpath, mode)
        return targetpath


def wheel_unpack(wheel: Path, dest_dir: Path, name_ver: str) -> None:
    # This is the moral equivalent of:
    # subprocess.check_output(
    #     [sys.executable, "-m", "wheel", "unpack", "-d", str(dest_dir), str(wheel)]
    # )
    # Except we need to preserve permissions
    # https://github.com/pypa/wheel/issues/505
    with ExecutablePreservingZipfile(wheel) as wf:
        wf.extractall(dest_dir / name_ver)


def change_platform_tag(
    wheel_path: Path, tag: str, parser: email.parser.BytesParser
) -> str:
    """Changes the WHEEL file to specify `tag`. Returns a canonicalized copy of that tag."""
    platform_tags = list(parse_tag(tag))
    if len(platform_tags) != 1:
        raise ValueError(
            f"Parsed '{tag}' as {len(platform_tags)}; there must be exactly one."
        )
    platform_tag = platform_tags[0]
    is_pure = platform_tag.abi == "none"
    if is_pure != (platform_tag.platform == "any"):
        raise ValueError(f"ABI and platform are inconsistent in '{platform_tag}'.")
    if is_pure != platform_tag.interpreter.startswith("py"):
        raise ValueError(
            f"Interpreter and platform are inconsistent in '{platform_tag}'."
        )
    with open(wheel_path, "rb") as f:
        msg = parser.parse(f)
    msg.replace_header("Tag", str(platform_tag))
    msg.replace_header("Root-Is-Purelib", str(is_pure).lower())
    with open(wheel_path, "wb") as f:
        f.write(msg.as_bytes())
    return str(platform_tag)


def change_wheel_version(
    wheel: Path,
    version: Optional[str],
    local_version: Optional[str],
    allow_same_version: bool = False,
    platform_tag: Optional[str] = None,
    callback_func: Optional[Callable] = None,  # AMD-added
) -> Path:
    old_parts = installer.utils.parse_wheel_filename(wheel.name)
    old_version = packaging.version.Version(old_parts.version)
    distribution = old_parts.distribution

    if version is None:
        if local_version is not None:
            # just replace the local version
            new_version = version_replace(
                old_version, local=packaging.version._parse_local_version(local_version)
            )
        else:
            new_version = old_version
    else:
        # replace the base version and (possibly) the local version
        new_version = packaging.version.Version(version)
        assert not new_version.local
        if local_version:
            new_version = version_replace(
                new_version, local=packaging.version._parse_local_version(local_version)
            )

    if version == old_version:
        if allow_same_version:
            return wheel
        raise ValueError(f"Version {version} is the same as the old version")

    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)
        dest_dir = tmpdir / "wheel"

        print("  Unpacking wheel...", end="")
        wheel_unpack(wheel, dest_dir, f"{distribution}-{old_version}")
        print(" done")

        old_slug = f"{distribution}-{old_version}"
        new_slug = f"{distribution}-{new_version}"
        assert (dest_dir / old_slug).exists()
        assert (dest_dir / old_slug / f"{old_slug}.dist-info").exists()

        # copy everything over
        print(f"  Move directory from {old_slug} to {new_slug}...", end="")
        shutil.move(dest_dir / old_slug, dest_dir / new_slug)
        print(" done")

        # AMD-added
        print("  Changing wheel-specific files that contain the version")
        if callback_func != None:
            callback_func(dest_dir / new_slug, old_version, new_version)

        # rename dist-info
        shutil.move(
            dest_dir / new_slug / f"{old_slug}.dist-info",
            dest_dir / new_slug / f"{new_slug}.dist-info",
        )
        print(f"    Rename directory from {old_slug}.dist-info to {new_slug}.dist-info")
        # rename data
        if (dest_dir / new_slug / f"{old_slug}.data").exists():
            shutil.move(
                dest_dir / new_slug / f"{old_slug}.data",
                dest_dir / new_slug / f"{new_slug}.data",
            )
            print(f"    Rename directory from {old_slug}.data to {new_slug}.data")

        metadata_path = dest_dir / new_slug / f"{new_slug}.dist-info" / "METADATA"
        wheel_path = dest_dir / new_slug / f"{new_slug}.dist-info" / "WHEEL"

        # This is actually a non-conformant email policy as per
        # https://packaging.python.org/en/latest/specifications/core-metadata/
        # However, it works around this bug in setuptools in cases where the version is really long
        # https://github.com/pypa/setuptools/issues/3808
        max_line_length = 200
        policy = email.policy.compat32.clone(max_line_length=200)
        parser = email.parser.BytesParser(policy=policy)
        version_str = str(new_version)
        if len(version_str) >= max_line_length:
            raise ValueError(f"Version {version_str} is too long")

        with open(metadata_path, "rb") as f:
            msg = parser.parse(f)
        msg.replace_header("Version", version_str)
        with open(metadata_path, "wb") as f:
            f.write(msg.as_bytes())
        
        print(f"    {metadata_path}")

        print("  ... done")
        

        if platform_tag:
            print("  Changing platform tag...", end="")
            # We don't need to sort, because we assume len(parse_tag(platform_tag)) == 1
            new_tag = change_platform_tag(wheel_path, platform_tag, parser)
            print(" done")
        else:
            print("  Generating platform tag...", end="")
            # Generate the tags that will be associated with the wheel after it is repacked.
            # `wheel pack` sorts the tags, so we need to do the same if we're not changing it.
            new_tag = "-".join(
                ".".join(sorted(t.split("."))) for t in old_parts.tag.split("-")
            )
            print(" done")

        # print is sometimes buffered, so we need to flush it, as repacking can take a while
        print("  Repacking wheel and rewriting RECORD file...", end="", flush=True)
        # wheel pack rewrites the RECORD file
        subprocess.check_output(
            [
                sys.executable,
                "-m",
                "wheel",
                "pack",
                "-d",
                str(wheel.parent),
                str(dest_dir / new_slug),
            ]
        )
        print(" done")

    new_parts = old_parts._replace(version=str(new_version), tag=new_tag)
    new_wheel_name = "-".join(p for p in new_parts if p) + ".whl"
    new_wheel = wheel.with_name(new_wheel_name)

    if not new_wheel.exists():
        raise RuntimeError(
            f"Failed to create new wheel {new_wheel}\n"
            f"Directory contents: {list(wheel.parent.iterdir())}"
        )
    return new_wheel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--local-version")
    parser.add_argument("--version")
    parser.add_argument("--delete-old-wheel", action="store_true")
    parser.add_argument("--allow-same-version", action="store_true")
    parser.add_argument(
        "--platform-tag",
        help="Force the platform tag to this value. For example, 'py3-none-any'. See "
        "https://packaging.python.org/en/latest/specifications/platform-compatibility-tags.",
    )
    args = parser.parse_args()

    new_wheel = change_wheel_version(
        wheel=args.wheel,
        version=args.version,
        local_version=args.local_version,
        allow_same_version=args.allow_same_version,
        platform_tag=args.platform_tag,
    )
    print(new_wheel)
    if args.delete_old_wheel:
        args.wheel.unlink()


if __name__ == "__main__":
    main()
