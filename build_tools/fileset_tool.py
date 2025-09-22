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
import sys
import shutil
import tarfile

from _therock_utils.artifacts import ArtifactPopulator
import _therock_utils.artifact_builder as artifact_builder
from _therock_utils.hash_util import calculate_hash, write_hash
from _therock_utils.pattern_match import PatternMatcher


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
    """
    descriptor = artifact_builder.ArtifactDescriptor.load_toml_file(args.descriptor)
    scanner = artifact_builder.ComponentScanner(args.root_dir, descriptor)
    # Disable strict verification temporarily until debug builds are tested/fixed.
    # scanner.verify()
    component_dirs = args.component_dirs
    # It is an alternating list of <component> <dir> so must be divisible by 2.
    if len(component_dirs) % 2:
        raise SystemExit(
            "Expected component dirs to be alternating list of component names and directories"
        )

    for i in range(len(component_dirs) // 2):
        component_name = component_dirs[i * 2]
        output_dir = Path(component_dirs[i * 2 + 1])

        # Setup output dir.
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            contents = scanner.components[component_name]
        except KeyError:
            return
        contents.write_artifact(output_dir)


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
        "component_dirs",
        nargs="+",
        help="Alternating list of component name and directory to write it to",
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
