#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ROCgdb emulator wrapper.

Runs rocgdb inside a mirage session so the program being debugged sees an
emulated GPU. The profile comes from --profile, ROCGDB_EMULATE_PROFILE, the
program's offload target or the tree's dist info, in that order.

ROCGDB_WRAPPER_DEBUG=1 prints each decision and the final command without
running it.
"""

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

USAGE = """\
Usage: rocgdb-emulate [--profile PROFILE] [--] [rocgdb args...]
PROFILE: mi300x, mi350x, mi450x (or gfx942, gfx950, gfx1250)"""


class Profile(NamedTuple):
    """A mirage profile and the gfx target it emulates."""

    name: str
    arch: str


MI300X = Profile("mi300x", "gfx942")
MI350X = Profile("mi350x", "gfx950")
MI450X = Profile("mi450x", "gfx1250")

PROFILES_BY_ALIAS: dict[str, Profile] = {
    "mi300x": MI300X,
    "mi300": MI300X,
    "gfx942": MI300X,
    "mi350x": MI350X,
    "mi350": MI350X,
    "gfx950": MI350X,
    "mi450x": MI450X,
    "mi450": MI450X,
    "gfx1250": MI450X,
}

# gdb options that take their value as the next argument.
GDB_OPTIONS_WITH_VALUE = set(
    "ex iex x ix eval-command init-eval-command command init-command cd d "
    "directory s symbols e exec se c core p pid b l tty i interpreter D "
    "data-directory".split()
)

OFFLOAD_BUNDLE_PATTERN = re.compile(
    r"^Extracting offload bundle: .*-amdgcn-amd-amdhsa--(gfx[0-9a-z]+)", re.MULTILINE
)
TARGET_SEPARATOR_PATTERN = re.compile(r"[,; ]+")

# mirage lends the terminal only when the session first stops for it, and
# rocgdb restores the terminal to whichever group owned it at startup, so
# take the terminal before rocgdb starts or it hangs after the first stop.
TAKE_TERMINAL = '[ -t 0 ] && stty "$(stty -g)"; exec "$0" "$@"'


class EmulateError(Exception):
    """An error that ends the wrapper with a message and exit code 1."""


class WrapperArgs(NamedTuple):
    profile: str | None
    show_help: bool
    rocgdb_args: list[str]


def debug_enabled() -> bool:
    return os.environ.get("ROCGDB_WRAPPER_DEBUG", "0") == "1"


def log(message: str) -> None:
    if debug_enabled():
        print(f"[DEBUG] -> {message}", file=sys.stderr)


def parse_wrapper_args(argv: list[str]) -> WrapperArgs:
    """Splits off the wrapper's own leading options.

    argparse is not used because only options before the first rocgdb argument
    belong to the wrapper; everything from there on must reach rocgdb as is.
    """
    profile = os.environ.get("ROCGDB_EMULATE_PROFILE") or None
    args = list(argv)
    while args:
        arg = args[0]
        if arg == "--profile":
            if len(args) < 2:
                raise EmulateError("--profile needs a value")
            profile = args[1] or None
            del args[:2]
        elif arg.startswith("--profile="):
            profile = arg.removeprefix("--profile=") or None
            del args[0]
        elif arg == "--help-emulate":
            return WrapperArgs(profile=profile, show_help=True, rocgdb_args=[])
        elif arg == "--":
            del args[0]
            break
        else:
            break
    return WrapperArgs(profile=profile, show_help=False, rocgdb_args=args)


def normalize_profile(spec: str) -> Profile | None:
    return PROFILES_BY_ALIAS.get(spec.lower())


def find_program(args: list[str]) -> str | None:
    """Returns the program rocgdb will debug: the argument after --args, or the
    first argument that is neither an option nor an option's value."""
    skip_next = False
    for index, arg in enumerate(args):
        if skip_next:
            skip_next = False
        elif arg in ("--args", "-args"):
            return args[index + 1] if index + 1 < len(args) else None
        elif not arg.startswith("-"):
            return arg
        elif "=" not in arg:
            skip_next = arg.lstrip("-") in GDB_OPTIONS_WITH_VALUE
    return None


def offload_targets(program: str, rocm_root: Path) -> set[str]:
    """Returns the gfx targets the program carries device code for."""
    path = Path(program)
    if not path.is_file():
        found = shutil.which(program)
        if found is None:
            return set()
        path = Path(found)
    objdump = rocm_root / "lib" / "llvm" / "bin" / "llvm-objdump"
    if not os.access(objdump, os.X_OK):
        return set()
    # --offloading writes every bundle it finds next to the path it is given,
    # so give it a link inside a scratch directory.
    with tempfile.TemporaryDirectory() as scratch:
        (Path(scratch) / "program").symlink_to(path.resolve())
        result = subprocess.run(
            [str(objdump), "--offloading", "program"],
            cwd=scratch,
            capture_output=True,
            text=True,
        )
    return set(OFFLOAD_BUNDLE_PATTERN.findall(result.stdout))


def tree_targets(rocm_root: Path) -> list[str]:
    """Returns the gfx targets the ROCm tree was built for."""
    info = rocm_root / "share" / "therock" / "dist_info.json"
    if not info.is_file():
        return []
    try:
        loaded = json.loads(info.read_text())
    except (OSError, ValueError) as e:
        raise EmulateError(f"cannot read {info}: {e}") from e
    targets = (
        loaded.get("dist_amdgpu_targets", "") if isinstance(loaded, dict) else None
    )
    if not isinstance(targets, str):
        raise EmulateError(f"cannot read {info}: dist_amdgpu_targets is not a string")
    return [target for target in TARGET_SEPARATOR_PATTERN.split(targets) if target]


def profile_from_program(program: str, rocm_root: Path) -> Profile | None:
    targets = offload_targets(program, rocm_root)
    log(f"Offload targets in {program}: {' '.join(sorted(targets))}")
    if len(targets) != 1:
        return None
    (target,) = targets
    profile = normalize_profile(target)
    if profile is None:
        raise EmulateError(
            f"{program} is built for {target}, which has no emulator profile"
        )
    return profile


def check_program_targets(program: str, profile: Profile, rocm_root: Path) -> None:
    targets = offload_targets(program, rocm_root)
    log(f"Offload targets in {program}: {' '.join(sorted(targets))}")
    if targets and profile.arch not in targets:
        raise EmulateError(
            f"{program} has no device code for {profile.arch} (built for "
            f"{' '.join(sorted(targets))}); rebuild with "
            f"--offload-arch={profile.arch} or pick a matching --profile"
        )


def profile_from_tree(rocm_root: Path) -> Profile | None:
    profiles = {normalize_profile(target) for target in tree_targets(rocm_root)}
    profiles.discard(None)
    log(f"Emulatable targets in tree: {' '.join(p.name for p in profiles)}")
    return profiles.pop() if len(profiles) == 1 else None


def resolve_profile(args: WrapperArgs, rocm_root: Path) -> Profile:
    program = find_program(args.rocgdb_args)
    if args.profile:
        profile = normalize_profile(args.profile)
        if profile is None:
            raise EmulateError(
                f"unknown profile '{args.profile}'; expected mi300x, mi350x, "
                "mi450x (or gfx942, gfx950, gfx1250)"
            )
        log(f"Profile from --profile/ROCGDB_EMULATE_PROFILE: {profile.name}")
        if program:
            check_program_targets(program, profile, rocm_root)
        return profile

    profile = profile_from_program(program, rocm_root) if program else None
    if profile is not None:
        log(f"Profile from program: {profile.name}")
        return profile

    profile = profile_from_tree(rocm_root)
    if profile is not None:
        log(f"Profile from tree: {profile.name}")
        return profile

    print(USAGE, file=sys.stderr)
    raise EmulateError("could not tell which GPU to emulate; pass --profile")


def build_command(args: WrapperArgs, bin_dir: Path) -> list[str]:
    rocm_root = bin_dir.parent
    log(f"ROCm tree: {rocm_root}")
    profile = resolve_profile(args, rocm_root)

    built_for = tree_targets(rocm_root)
    if built_for and profile.arch not in built_for:
        raise EmulateError(
            f"{profile.name} emulates {profile.arch}, but {rocm_root} was built "
            f"for {' '.join(built_for)}"
        )

    mirage = bin_dir / "mirage"
    if not os.access(mirage, os.X_OK):
        raise EmulateError(
            f"no mirage in {bin_dir}; this ROCm tree was built without the "
            "emulation tools (-DTHEROCK_ENABLE_EMULATION=ON)"
        )
    return [
        str(mirage),
        *("run", "--profile", profile.name, "--"),
        *("sh", "-c", TAKE_TERMINAL, str(bin_dir / "rocgdb")),
        *args.rocgdb_args,
    ]


def main(argv: list[str]) -> int:
    try:
        args = parse_wrapper_args(argv)
        if args.show_help:
            print(USAGE)
            return 0
        command = build_command(args, bin_dir=Path(__file__).resolve().parent)
    except EmulateError as e:
        print(f"rocgdb-emulate: {e}", file=sys.stderr)
        return 1

    log(f"Executing: {shlex.join(command)}")
    if debug_enabled():
        return 0
    try:
        os.execv(command[0], command)
    except OSError as e:
        print(f"rocgdb-emulate: cannot run {command[0]}: {e}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
