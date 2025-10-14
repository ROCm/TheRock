#!/usr/bin/env python
"""bootstrap_build.py

Bootstraps a build directory using artifacts from a prior build invocation.

Usage:
  ./build_tools/bootstrap_build.py --build-dir build \
      --artifact-dir /path/to/prior/build/artifacts
  cmake -GNinja -Bbuild -S. -DTHEROCK_AMDGPU_FAMILIES=...

Presently, this will only bootstrap from a directory containing artifacts
(specified via `--artifact-dir`), but in the future, it will be able to stream
directly from a remote source.

This will only process "generic" artifacts currently. In the future, it can be
extended to allow bootstrapping from target specific artifacts.

This works because artifacts are set up to capture the "stage" installation
directories from each sub-project, sorting their contents into components, each
in their own directory/archive. By expanding each one of these component
archives back into the build directory, we end up with pre-populated stage
installations for each. This script then writes a "*.prebuilt" file for each
root in the archive, so that whereas in the original build, there would just
be "stage/" directories, in this secondary build, there will now also be a
"stage.prebuilt" marker file. The CMake subproject system, upon encountering
this marker file will skip emitting the configure/build/stage targets, instead
just writing their stamp files with a dependency on the "stage.prebuilt" file.

Note that currently the "*.prebuilt" files are just empty touch-files. But in
the future, we can populate them with a hash of the contents. For artifact
directories, this could come by hashing all files. For artifact archives,
there is a parallel .sha256sum that is pre-computed and can be the basis of
the hash. In this way, incremental builds will re-trigger if the build dir
is bootstrapped with different artifacts. It can also be used in scripting to
derive an appropriate hash key for the compiler, etc (i.e. setting up a ccache
namespace based on the hash of all bootstrapped "*.prebuilt" files in the tree).

TODO: Merge this into the buildctl.py script and remove this.
"""

import argparse
from pathlib import Path
import shutil
import sys

from _therock_utils.artifacts import ArtifactPopulator, ArtifactName


def _do_run(args: argparse.Namespace):
    class CleaningPopulator(ArtifactPopulator):
        def on_first_relpath(self, relpath: str):
            full_path = self.output_path / relpath
            if full_path.exists():
                print(f"CLEANING: {full_path}")
                shutil.rmtree(full_path)
            # Write the ".prebuilt" marker file that the build system uses to
            # indicate that the staging install has already been done.
            prebuilt_path = full_path.with_name(full_path.name + ".prebuilt")
            prebuilt_path.parent.mkdir(parents=True, exist_ok=True)
            prebuilt_path.touch()

        def on_artifact_dir(self, artifact_dir: Path):
            print(f"FLATTENING {artifact_dir.name}")

        def on_artifact_archive(self, artifact_archive: Path):
            print(f"EXPANDING {artifact_archive.name}")

    args.build_dir.mkdir(parents=True, exist_ok=True)
    flattener = CleaningPopulator(
        output_path=args.build_dir, verbose=args.verbose, flatten=False
    )
    artifact_names: set[ArtifactName] = set()
    for entry in args.artifact_dir.iterdir():
        an = ArtifactName.from_path(entry)
        if not an:
            continue
        if an.target_family != "generic":
            print(f"SKIP {entry.name}: Not generic target")
            continue
        if an in artifact_names:
            print(f"SKIP {entry.name}: Duplicate")
            continue
        artifact_names.add(an)
        flattener(entry)


def main(cl_args: list[str]):
    p = argparse.ArgumentParser(
        "bootstrap_build.py",
        usage="bootstrap_build.py --build-dir <dir> --artifact-dir <artifacts>",
    )
    p.add_argument(
        "--build-dir",
        type=Path,
        required=True,
        help="Path to the CMake build directory to populate",
    )
    p.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="Directory from which to source artifacts",
    )
    p.add_argument("--verbose", action="store_true", help="Print verbose status")
    args = p.parse_args(cl_args)
    _do_run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
