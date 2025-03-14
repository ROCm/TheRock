#!/usr/bin/env python
"""Given ROCm artifacts directories, performs surgery to re-layout them for
distribution as Python packages. This does not actually do the Python packaging
but just prepares a directory structure that can be overlaid with
`packaging/python` to build all packages.

This process involves both a re-organization of the sources and some surgical
alterations.

Example
-------

```
./build_tools/linux_python_dist_split.py \
    --artifact-dir ./output-linux-portable/build/artifacts \
    --dest-dir $HOME/tmp/packages
```

Note that this does do some dynamic compilation of files and it performs
patching via patchelf. It is recommended to run this in the same portable
Linux container as was used to build the SDK (so as to avoid the possibility
of accidentally referencing too-new glibc symbols).

Relayout
--------
We generally split into the following top-level python packages (which correlate
to dist package names but include an additional version suffix):

* `rocm_sdk_core{_suffix}`: Target independent core libraries, including
  runtimes, bundled sysdeps, comgr and compiler shared libraries sufficient
  to use the RTC.
* `rocm_sdk_libraries[_target_family]{_suffix}`: Kernel libraries, including
  BLAS, FFT, RAND, MIOpen, etc. These contain target specific code. See
  below `Fat and Thin Target Packages`.
* `rocm_sdk_devel{_suffix}`: All remaining assets that are needed to compile
  arbitrary programs against the ROCm SDK. Note that this package is special
  in several ways and will be discussed separately.

Note that Python distributed packages (i.e. wheel files) are named with dashes,
per usual, and they do not include the version suffix nonce that we encode into
the on-disk directory layout in site-lib. We include this extra version suffix
nonce on disk because Python packaging versioning is not precise, and because
these pieces are not interchangeable with different released versions.
Embedding a nonce gives us a point of control to ensure that things that
directly link together actually show up at distribution time that way. If this
invariant were ever violated, the usual bevy of C++ linkage issues would
make things very hard to troubleshoot (whereas with embedded nonces, we can
even include Python level diagnostics and helpful suggestions). This way,
we can detect mismatches in Python code and triage accordingly.

Surgical Alterations
--------------------
Python packages have a number of idiosyncrasies that make them somewhat
hostile for distributing native software components. Chief among these is
that symlinks are simply materialized as discrete copies. This breaks a number
of assumptions regarding how Unix libraries are laid out.

The compromise we make is that the (majority of) libraries are only materialized
as their SONAME component in the non devel package. This is what the operating
system actually looks for at runtime and in the majority of cases, is
sufficient to operate the software. By and large, the ROCm software and LLVM
follows this convention closely enough to allow us to do this. We then
materialize all names/symlinks in the devel package, which we do not
distribute through symlink-unclean channels (there are other peculiarities of
the devel package which are out of scope of this topic -- see below).

The issue arises because a standard Unix library is symlinked like this:

```
libamd_comgr.so -> libamd_comgr.so.2
libamd_comgr.so.2 -> libamd_comgr.so.2.9
libamd_comgr.so.2.9
```

But the SONAME is `libamd_comgr.so.2` -- a symlink that we would like to
avoid (and the only file that matters at runtime). This makes some sense in an
OS context where you may have multiple patch versions of a library co-existing
and being arbitrated by symlink. But it doesn't fit our constraints. So we
"rotate" all link farms: In the normal packages, we only materialize the SONAME
variant (the middle one above) as a real file. In devel packages with shared
libraries that were materialized in a runtime package, we symlink back to the
runtime package version.

The ROCm and LLVM distribution also includes a number of executable symlink,
many of the form:

```
clang*
amdclang -> clang
```

Critically, all such symlinks are relative. We replace them with a stub
executable that we compile as part of the script which can perform the
dynamic path computation and exec() the intended target.

There are other alterations we have to make to accommodate our uniquely split
layout on disk.

Fat and Thin Target Packages
----------------------------
Target specific packages (aka `libraries`) can either be fat or thin:

* Fat target packages will contain independent host+device code in their own
  dedicated tree joined together. For multi-target packaging scenarios, there
  will be one per family, and the top-level `rocm-sdk` package will contain
  the meta-data mapping actual devices to subsets. Such packages will be
  named with a `_target_family` in the name. For development or single-host
  use, we support distribution of a single tree with no target-specific
  indirection (i.e. does not contain the `_target_family`).
* Thin target packages contain target-neutral host code in a non target specific
  directory tree, and then all device code/kernel databases are located
  separately and loaded via an interposer.

Thin target packages are currently in the conceptual phase but are intended to
be the final default packaging mechanism for multi-target builds.

Development Package
-------------------
The `-devel` package is unique because it is the catch-all for any distributed
files that are materialized in one of the above packages. For anything that is
materialized in a prior package, a relative symlink is generated across the
root directory to the named package directory. Since the development package
contains symlink farms, it cannot be included verbatim in a Python wheel (which
would try to materialize the symlinks). Instead, we *embed* it as a .tar.xz
file in a built wheel at Python package construction time. Then code that
needs to get the ROCm `cmake_prefix_path()` or run any tools will dynamically
decompress it to the HOME directory, setup root links, and run from there.

Note that most users never will need the development package: it exists to
build projects such as PyTorch, et al, for distribution against specific
`rocm-sdk` packages. If not building or using the CLI development tools, it
is not needed.

Since the development package is not meant to be used at runtime, it references
a fat target package for a default target, regardless of what hardware may be
physically installed on the machine. Since this is just used for linking, this
should be fine and allows builds to run reproducibly on non-GPU systems.
"""

import argparse
import functools
import magic
import re
import os
from pathlib import Path
import shutil
import subprocess
import sys

from _therock_utils.artifacts import ArtifactCatalog, ArtifactName
from _therock_utils.exe_stub_gen import generate_exe_link_stub
from _therock_utils.pattern_match import MatchPredicate, PatternMatcher

MAGIC_AR_MATCH = re.compile("ar archive")
MAGIC_EXECUTABLE_MATCH = re.compile("ELF[^,]+executable,")
MAGIC_SO_MATCH = re.compile("ELF[^,]+shared object,")


def run(args: argparse.Namespace):
    artifacts = ArtifactCatalog(args.artifact_dir)
    target_families = artifacts.all_target_families

    core_path = args.dest_dir / f"rocm_sdk_core{args.version_suffix}"
    libraries_path = args.dest_dir / f"rocm_sdk_libraries{args.version_suffix}"
    devel_path = args.dest_dir / f"rocm_sdk_devel{args.version_suffix}"

    # Where things go is a waterfall: each package we populate removes files
    # from consideration. Anything that is left goes in the devel package.
    # Relative path to materialized abs path.
    materialized_paths: dict[str, Path] = {}
    populate_core_package(args, core_path, artifacts, materialized_paths)

    # Emit libraries, one per artifact family that we have
    for target_family in target_families:
        libraries_path = (
            args.dest_dir / f"rocm_sdk_libraries_{target_family}{args.version_suffix}"
        )
        populate_libraries_package(
            args, target_family, libraries_path, artifacts, materialized_paths
        )

    populate_devel_package(args, devel_path, artifacts, materialized_paths)


def core_artifact_filter(an: ArtifactName) -> bool:
    core = an.name in [
        "amd-llvm",
        "base",
        "core-hip",
        "core-runtime",
        "rocprofiler-sdk",
        "sysdeps",
    ] and an.component in [
        "lib",
        "run",
    ]
    # hiprtc needs to be able to find HIP headers in its same tree.
    hip_dev = an.name in [
        "core-hip",
    ] and an.component in ["dev"]
    return core or hip_dev


def libraries_artifact_filter(target_family: str, an: ArtifactName) -> bool:
    libraries = (
        an.name
        in [
            "blas",
            "fft",
            "host-blas",
            "miopen",
            "prim",
            "rand",
            "rccl",
        ]
        and an.component
        in [
            "lib",
        ]
        and an.target_family == target_family
    )
    return libraries


def populate_core_package(
    args: argparse.Namespace,
    package_path: Path,
    all_artifacts: ArtifactCatalog,
    materialized_paths: dict[str, Path],
):
    # Setup.
    our_artifacts = ArtifactCatalog(
        all_artifacts.artifact_dir, filter=core_artifact_filter
    )
    our_artifacts.pm.predicate = MatchPredicate(
        # TODO: The base package is shoving CMake redirects into lib.
        excludes=["**/cmake/**"],
    )
    print(f"::: Populating core package {package_path}")
    for an in our_artifacts.artifact_names:
        print(f"  + {an}")
    if package_path.exists():
        shutil.rmtree(package_path)
    package_path.mkdir(parents=True, exist_ok=True)
    materialize_lib_package(our_artifacts.pm, package_path, materialized_paths)


def populate_libraries_package(
    args: argparse.Namespace,
    target_family: str,
    package_path: Path,
    all_artifacts: ArtifactCatalog,
    materialized_paths: dict[str, Path],
):
    # Setup.
    our_artifacts = ArtifactCatalog(
        all_artifacts.artifact_dir,
        filter=functools.partial(libraries_artifact_filter, target_family),
    )
    print(f"::: Populating libraries package {target_family} {package_path}")
    for an in our_artifacts.artifact_names:
        print(f"  + {an}")
    if package_path.exists():
        shutil.rmtree(package_path)
    package_path.mkdir(parents=True, exist_ok=True)
    materialize_lib_package(our_artifacts.pm, package_path, materialized_paths)


def populate_devel_package(
    args: argparse.Namespace,
    package_path: Path,
    all_artifacts: ArtifactCatalog,
    materialized_paths: dict[str, Path],
):
    print(f"::: Populating devel package {package_path}")
    for an in all_artifacts.artifact_names:
        print(f"  + {an}")
    if package_path.exists():
        shutil.rmtree(package_path)
    package_path.mkdir(parents=True, exist_ok=True)
    for relpath, dir_entry in all_artifacts.pm.matches():
        dest_path = package_path / relpath
        materialize_devel_file(
            relpath,
            dest_path,
            dir_entry,
            materialized_paths,
            root_output_dir=package_path.parent,
        )


# Materializes a "library" package. This is used for everything except the
# devel package, which is a catch-all of everything.
def materialize_lib_package(
    pm: PatternMatcher,
    package_dest_dir: Path,
    materialized_paths: dict[str, Path],
):
    # Handle each file.
    for relpath, dir_entry in pm.matches():
        if relpath in materialized_paths:
            continue
        file_type = get_file_type(dir_entry)
        dest_path = package_dest_dir / relpath
        if file_type == "symlink":
            maybe_materialize_lib_symlink(
                relpath, dest_path, dir_entry, materialized_paths
            )
        else:
            materialize_lib_file(relpath, dest_path, dir_entry, materialized_paths)


# Maybe materializes a symlink destined for a library package.
def maybe_materialize_lib_symlink(
    relpath: str,
    dest_path: Path,
    src_entry: os.DirEntry[str],
    materialized_paths: dict[str, Path],
):
    # Symlink handling is annoying because we can't have any :(
    # Here is what we do based on what it points to:
    #   1. Dangling or directory symlink: drop
    #   2. Shared library symlink: materialize if the symlink name is the SONAME
    #   3. Executable: Build a little executable launcher in place of the symlink (TODO)
    #   4. Copy it into place (this should work for scripts and such -- hopefully).
    resolved_path = Path(src_entry.path).resolve()
    # Case 1.
    if resolved_path.is_dir() or not resolved_path.exists():
        return

    target_file_type = get_file_type(resolved_path)

    # Case 2: Shared library.
    if target_file_type == "so":
        maybe_materialize_lib_so(relpath, dest_path, src_entry, materialized_paths)
        return

    # Case 3: Executable.
    if target_file_type == "exe":
        # Compile a standalone executable that dynamically emulates the symlink.
        link_target = os.readlink(src_entry.path)
        generate_exe_link_stub(dest_path, link_target)
        materialized_paths[relpath] = dest_path
        return

    # Case 4: Copy.
    materialize_file(relpath, dest_path, src_entry, materialized_paths)


# Materializes a shared library iff its name == the SONAME. This will resolve
# symlinks.
def maybe_materialize_lib_so(
    relpath: str,
    dest_path: Path,
    src_entry: os.DirEntry[str],
    materialized_paths: dict[str, Path],
):
    soname = get_soname(src_entry.path)
    if soname != src_entry.name:
        return

    if src_entry.is_symlink():
        # We're just going to "rotate" the symlinks so that the SONAME based
        # one is primary and everything else points to that.
        link_target = os.readlink(src_entry.path)
        if not link_target.count("/"):
            # It is just the normal libfoo.so.0 -> libfoo.so.0.1 style thing.
            # Note that this path was materialized to dest_path too.
            link_target_relpath = str(Path(relpath).parent / link_target)
            assert link_target_relpath not in materialized_paths
            materialized_paths[link_target_relpath] = dest_path
    materialize_file(
        relpath, dest_path, src_entry, materialized_paths, resolve_src=True
    )


# Materialize a regular file/directory entry (not a symlink) destined for a
# library package. This will only materialize a shared library if it matches
# its SONAME.
def materialize_lib_file(
    relpath: str,
    dest_path: Path,
    src_entry: os.DirEntry[str],
    materialized_paths: dict[str, Path],
):
    # Skip shared library without matching SONAME.
    file_type = get_file_type(src_entry.path)
    if file_type == "so":
        soname = get_soname(src_entry.path)
        if soname != dest_path.name:
            return
    materialize_file(relpath, dest_path, src_entry, materialized_paths)


def materialize_file(
    relpath: str,
    dest_path: Path,
    src_entry: os.DirEntry[str],
    materialized_paths: dict[str, Path],
    *,
    resolve_src: bool = True,
):
    is_symlink = src_entry.is_symlink()
    src_path = Path(src_entry.path)
    if resolve_src and is_symlink:
        src_path = src_path.resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # If it is a directory entry, just mkdir it.
    if src_entry.is_dir():
        dest_path.mkdir(parents=False, exist_ok=True)
        return

    # It is a regular file of some kind.
    if dest_path.exists():
        os.unlink(dest_path)
    # We have to patch many files, so we do not hard-link: always copy.
    print(f"  MATERIALIZE: {relpath} (from {src_path})")
    shutil.copy2(src_path, dest_path)
    if relpath in materialized_paths:
        print(f"WARNING: Path already materialized: {relpath}")
    materialized_paths[relpath] = dest_path


def materialize_devel_file(
    relpath: str,
    dest_path: Path,
    src_entry: os.DirEntry[str],
    materialized_paths: dict[str, Path],
    *,
    root_output_dir: Path,
):
    if src_entry.is_dir(follow_symlinks=False):
        dest_path.mkdir(parents=True, exist_ok=True)
        return

    if relpath in materialized_paths:
        # Materialize as a symlink to the original placement.
        original_path = materialized_paths[relpath]
        link_target = original_path.relative_to(dest_path.parent, walk_up=True)
        print(f"LINK: {relpath} -> {link_target}")
        if dest_path.exists(follow_symlinks=False):
            dest_path.unlink()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.symlink_to(link_target)
        return

    # If the source is a symlink, faithfully transcribe it.
    if src_entry.is_symlink():
        if dest_path.exists(follow_symlinks=False):
            dest_path.unlink()
        target_path = os.readlink(src_entry.path)
        print(f"LINK: {relpath} (to {target_path})")
        os.symlink(target_path, dest_path)
        return

    # Otherwise, no one else has emitted it, so just materialize verbatim.
    print(f"MATERIALIZE: {relpath} (from {src_entry.path})")
    if dest_path.exists(follow_symlinks=False):
        dest_path.unlink()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_entry.path, dest_path)


def get_file_type(dir_entry: os.DirEntry[str] | Path) -> str:
    if isinstance(dir_entry, os.DirEntry):
        path = Path(dir_entry.path)
    else:
        path = Path(dir_entry)

    if path.is_symlink():
        return "symlink"
    if path.is_dir():
        return "dir"

    # We only care about finding certain needles in the haystack, so exclude
    # text-like files from the get-go.
    path = str(path)
    if path.endswith(".txt") or path.endswith(".h") or path.endswith(".hpp"):
        return "text"
    desc = magic.from_file(path)
    if MAGIC_EXECUTABLE_MATCH.search(desc):
        return "exe"
    if MAGIC_SO_MATCH.search(desc):
        return "so"
    if MAGIC_AR_MATCH.search(desc):
        return "ar"
    return "other"


def get_soname(sofile: Path) -> str:
    return (
        subprocess.check_output(["patchelf", "--print-soname", str(sofile)])
        .decode()
        .strip()
    )


def main(argv: list[str]):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="Source artifacts/ dir from a build",
    )
    p.add_argument(
        "--dest-dir",
        type=Path,
        required=True,
        help="Destination directory in which to materialize packages",
    )
    p.add_argument(
        "--version-suffix",
        default="",
        help="Version suffix to append to package names on disk",
    )
    args = p.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
