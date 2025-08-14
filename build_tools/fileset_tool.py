#!/usr/bin/env python
"""fileset_tool.py

Helper tool for manipulating filesets by listing matching files, copying,
archiving, etc. This is ultimately inspired by the fileset manipulation behavior
of Ant, which uses recursive glob include/exclude patterns rooted on some
base directory to manage artifact moving and packaging.

This is based on a limited form of the pathlib.Path pattern language introduced
in Python 3.13 (https://docs.python.org/3/library/pathlib.html#pattern-language)
with the following changes:

* It does not support character classes.
"""

from typing import Callable
import argparse
from pathlib import Path
import platform
import sys
import shutil
import tarfile

from _therock_utils.artifacts import ArtifactPopulator
from _therock_utils.hash_util import calculate_hash, write_hash
from _therock_utils.pattern_match import PatternMatcher


def evaluate_optional(optional_value) -> bool:
    """Returns true if the given value should be considered optional on this platform.

    It can be either a str, list of str, or a truthy value. If a str/list, then it will
    return true if any of the strings match the case insensitive
    `platform.system()`.
    """
    if optional_value is None:
        return False
    if isinstance(optional_value, str):
        optional_value = [optional_value]
    if isinstance(optional_value, list):
        system_name = platform.system().lower()
        for v in optional_value:
            if str(v).lower() == system_name:
                return True
        return False
    return bool(optional_value)


class ComponentDefaults:
    """Defaults for to apply to artifact merging by component name."""

    ALL: dict[str, "ComponentDefaults"] = {}

    def __init__(self, name: str = "", includes=(), excludes=()):
        self.includes = list(includes)
        self.excludes = list(excludes)
        if name:
            if name in ComponentDefaults.ALL:
                raise KeyError(f"ComponentDefaults {name} already defined")
            ComponentDefaults.ALL[name] = self

    @staticmethod
    def get(name: str) -> "ComponentDefaults":
        return ComponentDefaults.ALL.get(name) or ComponentDefaults(name)


# Debug components collect all platform specific dbg file patterns.
ComponentDefaults(
    "dbg",
    includes=[
        # Linux build-id based debug files.
        ".build-id/**/*.debug",
    ],
)

# Dev components include all static library based file patterns and
# exclude file name patterns implicitly included for "run" and "lib".
# Descriptors should explicitly include header file any package file
# sub-trees that do not have an explicit "cmake" or "include" path components
# in them.
ComponentDefaults(
    "dev",
    includes=[
        "**/*.a",
        "**/*.lib",
        "**/cmake/**",
        "**/include/**",
        "**/share/modulefiles/**",
        "**/pkgconfig/**",
    ],
    excludes=[],
)
# Lib components include shared libraries, dlls and any assets needed for use
# of shared libraries at runtime. Files are included by name pattern and
# descriptors should include/exclude non-standard variations.
ComponentDefaults(
    "lib",
    includes=[
        "**/*.dll",
        "**/*.dylib",
        "**/*.dylib.*",
        "**/*.so",
        "**/*.so.*",
    ],
    excludes=[],
)
# Run components layer on top of 'lib' components and also include executables
# and tools that are not needed by library consumers. Descriptors should
# explicitly include "bin" directory contents as needed.
ComponentDefaults("run")
ComponentDefaults("doc", includes=["**/share/doc/**"])

# To help layering, we make lib/dev/run default patterns exclude patterns
# that the others define. This makes it easier for one of these to do directory
# level includes and have the files sorted into the proper component.
ComponentDefaults.get("dev").excludes.extend(ComponentDefaults.get("lib").includes)
ComponentDefaults.get("dev").excludes.extend(ComponentDefaults.get("run").includes)
ComponentDefaults.get("dev").excludes.extend(ComponentDefaults.get("doc").includes)
ComponentDefaults.get("lib").excludes.extend(ComponentDefaults.get("dev").includes)
ComponentDefaults.get("lib").excludes.extend(ComponentDefaults.get("run").includes)
ComponentDefaults.get("lib").excludes.extend(ComponentDefaults.get("doc").includes)
ComponentDefaults.get("run").excludes.extend(ComponentDefaults.get("dev").includes)
ComponentDefaults.get("run").excludes.extend(ComponentDefaults.get("lib").includes)
ComponentDefaults.get("run").excludes.extend(ComponentDefaults.get("doc").includes)


def do_list(args: argparse.Namespace, pm: PatternMatcher):
    for relpath, direntry in pm.matches():
        print(relpath)


def do_copy(args: argparse.Namespace, pm: PatternMatcher):
    verbose = args.verbose
    destdir: Path = args.dest_dir
    pm.copy_to(
        destdir=destdir,
        verbose=verbose,
        always_copy=args.always_copy,
        remove_dest=args.remove_dest,
    )


def do_artifact(args):
    """Produces an 'artifact directory', which is a slice of installed stage/
    directories, split into components (i.e. run, dev, dbg, doc, test).

    The primary input is the artifact.toml file, which defines records like:

        "components" : dict of covered component names
            "{component_name}": dict of build/ relative paths to materialize
                "{stage_directory}":
                    "default_patterns": bool (default True) whether component default
                        patterns are used
                    "include": str or list[str] of include patterns
                    "exclude": str or list[str] of exclude patterns
                    "force_include": str or list[str] of include patterns that if
                        matched, force inclusion, regardless of whether they match
                        an exclude pattern.
                    "optional": if true and the directory does not exist, it
                      is not an error. Use for optionally built projects. This
                      can also be either a string or array of strings, which
                      are interpreted as a platform name. If the case-insensitive
                      `platform.system()` equals one of them, then it is
                      considered optional.

    Most sections can typically be blank because by default they use
    component specific include/exclude patterns (see `COMPONENT_DEFAULTS` above)
    that cover most common cases. Local deviations must be added explicitly
    in the descriptor.

    This is called once per component and will create a directory for that
    component.
    """
    descriptor = load_toml_file(args.descriptor) or {}
    component_name = args.component
    # Set up output dir.
    output_dir: Path = args.output_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get metadata for the component we are merging.
    try:
        component_record = descriptor["components"][component_name]
    except KeyError:
        # No components.
        component_record = {}

    all_basedir_relpaths = []
    for basedir_relpath, basedir_record in component_record.items():
        use_default_patterns = basedir_record.get("default_patterns", True)
        basedir = args.root_dir / Path(basedir_relpath)
        optional = evaluate_optional(basedir_record.get("optional"))
        if optional and not basedir.exists():
            continue
        all_basedir_relpaths.append(basedir_relpath)

        # Force includes.
        force_includes = _dup_list_or_str(basedir_record.get("force_include"))

        # Includes.
        includes = _dup_list_or_str(basedir_record.get("include"))
        if use_default_patterns:
            includes.extend(
                ComponentDefaults.ALL.get(component_name, ComponentDefaults()).includes
            )

        # Excludes.
        excludes = _dup_list_or_str(basedir_record.get("exclude"))
        if use_default_patterns:
            excludes.extend(
                ComponentDefaults.ALL.get(component_name, ComponentDefaults()).excludes
            )

        pm = PatternMatcher(
            includes=includes,
            excludes=excludes,
            force_includes=force_includes,
        )
        pm.add_basedir(basedir)
        pm.copy_to(
            destdir=output_dir,
            destprefix=basedir_relpath + "/",
            remove_dest=False,
        )

    # Write a manifest containing relative paths of all base directories.
    manifest_path = output_dir / "artifact_manifest.txt"
    manifest_path.write_text("\n".join(all_basedir_relpaths) + "\n")


def do_artifact_archive(args):
    output_path: Path = args.o
    if output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with _open_archive(output_path, args.compression_level) as arc:
        for artifact_path in args.artifact:
            manifest_path: Path = artifact_path / "artifact_manifest.txt"
            relpaths = manifest_path.read_text().splitlines()
            # Important: The manifest must be stored first.
            arc.add(manifest_path, arcname=manifest_path.name, recursive=False)
            for relpath in relpaths:
                if not relpath:
                    continue
                source_dir = artifact_path / relpath
                if not source_dir.exists():
                    continue
                pm = PatternMatcher()
                pm.add_basedir(source_dir)
                for subpath, dir_entry in pm.all.items():
                    fullpath = f"{relpath}/{subpath}"
                    arc.add(dir_entry.path, arcname=fullpath, recursive=False)

    if args.hash_file:
        digest = calculate_hash(output_path, args.hash_algorithm)
        write_hash(args.hash_file, digest)


def _open_archive(p: Path, compression_level: int) -> tarfile.TarFile:
    return tarfile.TarFile.open(p, mode="x:xz", preset=compression_level)


def _do_artifact_flatten(args):
    flattener = ArtifactPopulator(
        output_path=args.o, verbose=args.verbose, flatten=True
    )
    flattener(*args.artifact)
    relpaths = list(flattener.relpaths)
    relpaths.sort()
    if args.verbose:
        for relpath in relpaths:
            print(relpath)


def _dup_list_or_str(v: list[str] | str) -> list[str]:
    if not v:
        return []
    if isinstance(v, str):
        return [v]
    return list(v)


def load_toml_file(p: Path):
    try:
        import tomllib
    except ModuleNotFoundError:
        # Python <= 3.10 compatibility (requires install of 'tomli' package)
        import tomli as tomllib
    with open(p, "rb") as f:
        return tomllib.load(f)


def main(cl_args: list[str]):
    def add_pattern_matcher_args(p: argparse.ArgumentParser):
        p.add_argument("basedir", type=Path, nargs="*", help="Base directories to scan")
        p.add_argument("--include", nargs="+", help="Recursive glob pattern to include")
        p.add_argument("--exclude", nargs="+", help="Recursive glob pattern to exclude")
        p.add_argument("--verbose", action="store_true", help="Print verbose status")

    def pattern_matcher_action(
        action: Callable[[argparse.Namespace, PatternMatcher], None],
    ):
        def run_action(args: argparse.Namespace):
            if not args.basedir:
                # base dir is CWD
                args.basedir = [Path.cwd()]
            pm = PatternMatcher(args.include or [], args.exclude or [])
            for basedir in args.basedir:
                pm.add_basedir(basedir)
            action(args, pm)

        return run_action

    p = argparse.ArgumentParser(
        "fileset_tool.py", usage="fileset_tool.py {command} ..."
    )
    sub_p = p.add_subparsers(required=True)
    # 'copy' command
    copy_p = sub_p.add_parser("copy", help="Copy matching files to a destination dir")
    copy_p.add_argument("dest_dir", type=Path, help="Destination directory")
    copy_p.add_argument(
        "--always-copy", action="store_true", help="Always copy vs attempting to link"
    )
    copy_p.add_argument(
        "--remove-dest",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Remove the destination directory before copying",
    )
    add_pattern_matcher_args(copy_p)
    copy_p.set_defaults(func=pattern_matcher_action(do_copy))

    # 'list' command
    list_p = sub_p.add_parser("list", help="List matching files to stdout")
    add_pattern_matcher_args(list_p)
    list_p.set_defaults(func=pattern_matcher_action(do_list))

    # 'artifact' command
    artifact_p = sub_p.add_parser(
        "artifact", help="Merge artifacts based on a descriptor"
    )
    artifact_p.add_argument(
        "--output-dir", type=Path, required=True, help="Artifact output directory"
    )
    artifact_p.add_argument(
        "--root-dir",
        type=Path,
        required=True,
        help="Source directory to which all descriptor directories are relative",
    )
    artifact_p.add_argument(
        "--descriptor",
        type=Path,
        required=True,
        help="TOML file describing the artifact",
    )
    artifact_p.add_argument(
        "--component", required=True, help="Component within the descriptor to merge"
    )
    artifact_p.set_defaults(func=do_artifact)

    # 'artifact-archive' command
    artifact_archive_p = sub_p.add_parser(
        "artifact-archive",
        help="Creates an archive file from one or more artifact directories",
    )
    artifact_archive_p.add_argument(
        "artifact", nargs="+", type=Path, help="Artifact directory"
    )
    artifact_archive_p.add_argument(
        "-o", type=Path, required=True, help="Output archive name"
    )
    artifact_archive_p.add_argument(
        "--compression-level",
        type=int,
        default=6,
        help="LZMA compression preset level [0-9, default 6]",
    )
    artifact_archive_p.add_argument(
        "--hash-file",
        type=Path,
        help="Hash file to write representing the archive contents",
    )
    artifact_archive_p.add_argument(
        "--hash-algorithm", default="sha256", help="Hash algorithm"
    )
    artifact_archive_p.set_defaults(func=do_artifact_archive)

    # 'artifact-flatten' command
    artifact_flatten_p = sub_p.add_parser(
        "artifact-flatten",
        help="Flattens one or more artifact directories into one output directory",
    )
    artifact_flatten_p.add_argument(
        "artifact", nargs="+", type=Path, help="Artifact directory"
    )
    artifact_flatten_p.add_argument(
        "-o", type=Path, required=True, help="Output archive name"
    )
    artifact_flatten_p.add_argument(
        "--verbose", action="store_true", help="Print verbose status"
    )
    artifact_flatten_p.set_defaults(func=_do_artifact_flatten)

    args = p.parse_args(cl_args)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
