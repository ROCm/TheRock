#!/usr/bin/env python
"""CLI tool for managing the 'prebuilt' state of project components.

In TheRock terminology, 'prebuilt' refers to a project component that has had
its staging install provided externally and it will not be configured/built
locally (but will still be available for dependents). Where such prebuilts
come from can vary:

* From a prior build invocation locally.
* From a central CI server.
* From a current build invocation where we just want to mark the project as
  not involved in the build any longer.

Where such prebuilts come from is presently outside of the scope of this
utility, but it may be expanded to pull automatically from the CI, etc, in the
future.

Basic Usage
-----------
The most basic usage of the tool is to enable/disable sub-projects from the
build after having bootstrapped (built once or obtained artifacts from elsewhere).
Two subcommands are provided for this: "enable" and "disable". Both take arguments:

* List of regular expressions to explicitly include (default to all).
* `--exclude` + list of regular expressions to exclude (default to none).

For example, if after an initial boostrap, you only want to iterate on
`ROCR-Runtime`, and you don't intend to change the public API and therefore
don't care to build dependents, run:
    python ./build_tools/buildctl.py enable ROCR

If you do want to do codevelopment with clr, you could also do:
    python ./build_tools/buildctl.py enable ROCR clr

To reset to building everything, just run the `enable` command with no arguments.

Similar usage can be made with `disable`. Let's say you would like to avoid
spurious builds of some math libraries:

    python ./build_tools/buildctl.py disable hipBLASLt rocSOLVER

A report will be printed and if any changes to project state were made,
"Reconfiguring..." will be printed and a CMake reconfigure of TheRock will be
done to pick up the changes.

What is going on under the covers
---------------------------------
Under the covers, the build system operates off of `stage/` subdirectories in
each project's tree. This represents the result of `cmake --install` of the
sub-project. If there is an adjacent file called `stage.prebuilt`, then the
build system will just trust that the `stage/` directory contents are correct,
skip build/install of it, and just use the `stage.prebuilt` file as an up to
date check (so if you touch this file, it will invalidate all dependents,
forcing them to rebuild). You can manage these files yourself with `find`,
`touch`, and `rm` but it is tedious. This tool aims to handle common workflows
without filesystem hacking.

TODO: Merge the functionality in bootstrap_build.py into this script.
"""

import argparse
from pathlib import Path, PosixPath
import os
import re
import shutil
import subprocess
import sys
import tempfile

from _therock_utils.artifacts import ArtifactCatalog, ArtifactPopulator, SkipPopulation
from fetch_artifacts import main as fetch_artifacts_main


def do_enable_disable(args: argparse.Namespace, enable_mode: bool):
    build_dir = resolve_build_dir(args)
    stage_dirs = find_stage_dirs(build_dir)
    selection = filter_selection(args, stage_dirs)
    changed = False
    print("Projects marked with an 'X' will be build enabled:")
    for rp, include in selection:
        stage_dir = build_dir / PosixPath(rp)
        prebuilt_file = stage_dir.with_name(stage_dir.name + ".prebuilt")
        is_empty = False
        if not is_valid_stage_dir(stage_dir):
            action = (True, "(EMPTY)")
            is_empty = True
        elif include:
            action = (True, "")
        else:
            action = (False, "")

        action_enable, message = action

        if not enable_mode:
            action_enable = not action_enable

        if action_enable:
            # Enable by deleting the prebuilt file.
            if prebuilt_file.exists():
                prebuilt_file.unlink()
                changed = True
        elif not is_empty:
            # Disable by touching the prebuilt file.
            # When disabling, we only do so if the stage directory is non
            # empty. This keeps us from setting up prebuilts for trivial
            # projects like header only deps, etc.
            if not prebuilt_file.exists():
                prebuilt_file.touch()
                changed = True
        is_enabled = not prebuilt_file.exists()
        print(f"[{'X' if is_enabled else ' '}] {rp} {message}")

    if changed or args.force_reconfigure:
        reconfigure(build_dir)


def do_status(args: argparse.Namespace):
    build_dir = resolve_build_dir(args)
    stage_relpaths = find_stage_dirs(build_dir)
    print("Projects marked with an 'X' will be build enabled:")
    for stage_relpath in stage_relpaths:
        stage_dir = build_dir / stage_relpath
        prebuilt_file = stage_dir.with_name(stage_dir.name + ".prebuilt")
        is_enabled = not prebuilt_file.exists()
        print(f"[{'X' if is_enabled else ' '}] {stage_relpath}")


def do_download(args: argparse.Namespace):
    build_dir = resolve_build_dir(args)
    stage_dirs = find_stage_dirs(build_dir)
    if not stage_dirs:
        raise CLIError(
            f"There are no stage directories in the build dir {build_dir}: "
            f"Has an initial configure been done yet?"
        )
    # Make sure the fetch temp dir is on the same file system as the build dir
    # so that moving works.
    temp_dir = build_dir / ".fetch_artifacts"
    temp_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=temp_dir, delete=False) as extract_dir_str:
        print(f"Extracting to temporary directory: {extract_dir_str}")
        extract_dir = Path(extract_dir_str)
        fetch_args = [
            "--output-dir",
            str(extract_dir),
            "--run-id",
            args.run_id,
            "--all",
            "--target",
            args.target,
        ]
        if args.exclude:
            fetch_args.append("--exclude")
            fetch_args.extend(args.exclude)
        fetch_args.extend(args.include)
        fetch_artifacts_main(fetch_args)

        catalog = ArtifactCatalog(extract_dir)
        prebuilt_stage_dirs: set[str] = set()

        class PrebuiltArtifactPopulator(ArtifactPopulator):
            def on_relpath(self, relpath):
                if relpath not in stage_dirs:
                    print(f"SKIP: {relpath} (not part of current build)")
                    raise SkipPopulation()
                if relpath in prebuilt_stage_dirs:
                    # Already encountered: let it copy as is
                    return
                prebuilt_stage_dirs.add(relpath)

                # First time: Delete and mark as prebuilt.
                print(f"++ Populating {relpath} as a prebuilt")
                build_stage_dir = build_dir / relpath
                if build_stage_dir.exists():
                    shutil.rmtree(build_stage_dir)
                prebuilt_file = build_stage_dir.with_name(
                    build_stage_dir.name + ".prebuilt"
                )
                print(f"++ Marking prebuilt {prebuilt_file}")
                prebuilt_file.touch()

        populator = PrebuiltArtifactPopulator(
            output_path=build_dir, verbose=True, flatten=False
        )
        populator(*catalog.all_artifact_dirs)

    do_status(args)
    reconfigure(build_dir)


def resolve_build_dir(args: argparse.Namespace) -> Path:
    build_dir: Path | None = args.build_dir
    if build_dir is None:
        build_dir = Path(__file__).resolve().parent.parent / "build"
    if not build_dir.exists() or not build_dir.is_dir():
        raise CLIError(
            f"Build directory {build_dir} not found: specify with --build-dir"
        )
    return build_dir


# The build system creates marker files named ".{something}stage.marker" for
# every stage directory in the build tree. This returns relative paths to all
# such stage directories.
def find_stage_dirs(build_dir: Path) -> list[str]:
    PREFIX = "."
    SUFFIX = ".marker"
    results: list[Path] = list()
    for current_dir, dirs, files in os.walk(build_dir.absolute()):
        for file in files:
            if file.startswith(PREFIX) and file.endswith(f"stage{SUFFIX}"):
                stage_dir_name = file[len(PREFIX) : -len(SUFFIX)]
                results.append(Path(current_dir) / stage_dir_name)
                # Prevent os.walk from recursing into subdirectories of this match
                try:
                    index = dirs.index(stage_dir_name)
                except ValueError:
                    ...
                else:
                    del dirs[index]

    relative_results = [d.relative_to(build_dir).as_posix() for d in results]
    relative_results.sort()
    return relative_results


# Applies filter arguments to a list of relative paths returning a list of
# (relpath, include).
def filter_selection(
    args: argparse.Namespace, relpaths: list[str]
) -> list[tuple[str, bool]]:
    def _filter(rp: str) -> bool:
        # If any includes, only pass if at least one matches.
        if args.include:
            for include_regex in args.include:
                pattern = re.compile(include_regex)
                if pattern.search(rp):
                    break
            else:
                return False
        # And if no excludes match.
        if args.exclude:
            for exclude_regex in args.exclude:
                pattern = re.compile(exclude_regex)
                if pattern.search(rp):
                    return False
        # Otherwise, pass.
        return True

    return [(rp, _filter(rp)) for rp in relpaths]


def is_valid_stage_dir(stage_dir: Path) -> bool:
    # Non existing are invalid.
    if not stage_dir.exists():
        return False

    # Empty stage directories are invalid.
    children = list(stage_dir.iterdir())
    if not children:
        return False
    return True


# Runs cmake reconfiguration.
def reconfigure(build_dir: Path):
    PREFIX = "CMAKE_COMMAND:INTERNAL="
    cache_file = build_dir / "CMakeCache.txt"
    cmake_command = None
    if not cache_file.exists():
        raise CLIError(f"Cannot reconfigure: cache file {cache_file} does not exist")
    cache_lines = cache_file.read_text().splitlines()
    for cache_line in cache_lines:
        if cache_line.startswith(PREFIX):
            cmake_command = cache_line[len(PREFIX) :]
            break
    else:
        raise CLIError(
            f"Could not find {PREFIX} in {cache_file}: Cannot automatically reconfigure"
        )

    print("Reconfiguring...", file=sys.stderr)
    try:
        subprocess.check_output(
            [cmake_command, str(build_dir)], stderr=subprocess.STDOUT, text=True
        )
    except subprocess.CalledProcessError as e:
        # Print combined output only if the command fails
        print(e.output, end="")
        raise CLIError(f"Project reconfigure failed")


class CLIError(Exception):
    ...


def main(cl_args: list[str]):
    p = argparse.ArgumentParser("buildctl.py", usage="buildctl.py {command} ...")
    sub_p = p.add_subparsers(required=True)

    def add_common_options(command_p: argparse.ArgumentParser, handler):
        command_p.set_defaults(func=handler)
        command_p.add_argument(
            "--build-dir",
            type=Path,
            help="Build directory (defaults to project level build/)",
        )

    def add_selection_options(command_p: argparse.ArgumentParser):
        command_p.add_argument(
            "--force-reconfigure",
            action="store_true",
            help="Reconfigure, even if not changed",
        )
        command_p.add_argument(
            "include", nargs="*", help="Regular expressions to include (all if empty)"
        )
        command_p.add_argument(
            "--exclude",
            nargs="*",
            help="Regular expressions to exclude (none if empty)",
        )

    # 'enable' command
    enable_p = sub_p.add_parser("enable", help="Enable subset of projects as buildable")
    add_common_options(enable_p, lambda args: do_enable_disable(args, enable_mode=True))
    add_selection_options(enable_p)

    # 'disable' command
    disable_p = sub_p.add_parser(
        "disable", help="Disable subset of projects as prebuilt"
    )
    add_common_options(
        disable_p, lambda args: do_enable_disable(args, enable_mode=False)
    )
    add_selection_options(disable_p)

    # 'status' command
    status_p = sub_p.add_parser("status", help="Show status of all pre-builts")
    add_common_options(status_p, lambda args: do_status(args))

    # 'download' command
    download_p = sub_p.add_parser(
        "download",
        help="Download artifacts from a CI run and disable them in the build",
    )
    download_p.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="GitHub run ID to retrieve artifacts from",
    )
    download_p.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target variant for specific GPU target",
    )
    download_p.add_argument(
        "include",
        nargs="*",
        help="Regular expression patterns of artifacts to include: "
        "if supplied one pattern must match for an artifact to be included",
    )
    download_p.add_argument(
        "--exclude",
        nargs="*",
        help="Regular expression patterns of artifacts to exclude",
    )
    add_common_options(download_p, do_download)

    args = p.parse_args(cl_args)
    try:
        args.func(args)
    except CLIError as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
