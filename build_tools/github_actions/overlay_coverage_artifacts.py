# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Overlays one project's instrumented binaries onto a baseline install tree.

Nightly coverage builds an instrumented stack in its own workflow run while the
regular nightly keeps publishing the normal one, so the two live under separate
run ids. A coverage test job wants exactly one instrumented project and
non-instrumented everything else: an instrumented dependency would emit its own
profiles and move the coverage denominator around whenever that dependency
changes.

The test job gets there in two steps. It first installs the whole baseline run
the usual way, then calls this script to fetch the instrumented artifact from
the coverage run and copy just the project under test over the baseline copy.

TheRock's artifacts are grouped -- `rand` carries both rocRAND and hipRAND --
so a whole artifact is too coarse to overlay. Each artifact keeps its files
under the subproject stage directory they were built in
(`math-libs/hipRAND/stage/...`), and that is what --artifact-relpaths names, so
sibling projects in the same artifact stay non-instrumented.
"""

import argparse
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).resolve().parents[1]))

from artifact_manager import main as artifact_manager_main

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Artifact components an instrumented project can contribute: `lib` the
# instrumented libraries, `test` the test binaries (instrumented in their own
# right for header-only projects), and `dev` the headers and CMake config that
# go with them. The rest only carry symbols and documentation.
EXCLUDED_COMPONENTS = ["run", "dbg", "doc"]

# artifact_manager.py fetch explodes each archive into this subdirectory of its
# output directory, one directory per artifact.
FETCH_SUBDIR = "artifacts"


def split_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def fetch_instrumented_artifacts(
    *,
    run_id: str,
    artifact_names: list[str],
    amdgpu_families: str,
    amdgpu_targets: str,
    output_dir: Path,
    run_github_repo: str | None,
) -> Path:
    """Downloads and explodes the named artifacts, returning their parent dir.

    Neither flattened nor bootstrapped: the overlay needs the per-subproject
    directory layout that flattening would collapse.
    """
    argv = [
        "fetch",
        "--run-id",
        run_id,
        # These artifacts are outputs of the stage that built them rather than
        # inbound to any stage, so the whole topology is the search space and
        # --artifact-names does the narrowing.
        "--stage",
        "all",
        "--artifact-names",
        ",".join(artifact_names),
        "--exclude-components",
        ",".join(EXCLUDED_COMPONENTS),
        "--amdgpu-families",
        amdgpu_families,
        "--output-dir",
        os.fspath(output_dir),
    ]
    if amdgpu_targets:
        argv.extend(["--amdgpu-targets", amdgpu_targets])
    if run_github_repo:
        argv.extend(["--run-github-repo", run_github_repo])

    logging.info("Fetching instrumented artifacts: %s", " ".join(argv))
    artifact_manager_main(argv)
    return output_dir / FETCH_SUBDIR


def overlay_relpaths(
    *, staging_dir: Path, relpaths: list[str], install_dir: Path
) -> int:
    """Copies each artifact's <relpath> subtree over the install tree.

    Returns the number of subtrees copied. Symlinks are preserved so the
    libfoo.so -> libfoo.so.1 chain the loader follows still points at the
    instrumented file.
    """
    if not staging_dir.is_dir():
        return 0

    copied = 0
    for artifact_dir in sorted(p for p in staging_dir.iterdir() if p.is_dir()):
        for relpath in relpaths:
            source = artifact_dir / relpath
            if not source.is_dir():
                continue
            logging.info("Overlaying %s onto %s", source, install_dir)
            shutil.copytree(
                source,
                install_dir,
                symlinks=True,
                dirs_exist_ok=True,
            )
            copied += 1
    return copied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        required=True,
        help="Coverage workflow run that published the instrumented artifacts",
    )
    parser.add_argument(
        "--amdgpu-families",
        required=True,
        help="Semicolon-separated GPU families to fetch (e.g. 'gfx94X-dcgpu')",
    )
    parser.add_argument(
        "--amdgpu-targets",
        default="",
        help="Comma-separated individual GPU targets for split artifacts",
    )
    parser.add_argument(
        "--artifact-names",
        required=True,
        help="Comma-separated artifact names holding the instrumented project "
        "(e.g. 'rand')",
    )
    parser.add_argument(
        "--artifact-relpaths",
        required=True,
        help="Comma-separated subproject stage directories to overlay "
        "(e.g. 'math-libs/hipRAND/stage')",
    )
    parser.add_argument(
        "--install-dir",
        type=Path,
        required=True,
        help="Baseline install tree to overlay the instrumented files onto",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help="Directory for the downloaded artifacts (default: a temporary directory)",
    )
    parser.add_argument(
        "--run-github-repo",
        default=None,
        help="GitHub repository for --run-id in 'owner/repo' format",
    )
    args = parser.parse_args(argv)

    artifact_names = split_list(args.artifact_names)
    relpaths = split_list(args.artifact_relpaths)
    if not artifact_names or not relpaths:
        logging.error(
            "--artifact-names and --artifact-relpaths must both be non-empty."
        )
        return 1

    install_dir = args.install_dir.resolve()
    if not install_dir.is_dir():
        logging.error(
            "Install directory %s does not exist. The baseline artifacts must "
            "be installed before the instrumented ones are overlaid.",
            install_dir,
        )
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        staging_dir = args.staging_dir or Path(tmp)
        staging_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir = fetch_instrumented_artifacts(
            run_id=args.run_id,
            artifact_names=artifact_names,
            amdgpu_families=args.amdgpu_families,
            amdgpu_targets=args.amdgpu_targets,
            output_dir=staging_dir,
            run_github_repo=args.run_github_repo,
        )
        copied = overlay_relpaths(
            staging_dir=artifacts_dir,
            relpaths=relpaths,
            install_dir=install_dir,
        )

    # Silently testing the baseline build would report coverage against
    # binaries that were never instrumented, so treat this as fatal.
    if not copied:
        logging.error(
            "None of the artifact relpaths (%s) were found in the artifacts "
            "fetched from run %s. The instrumented build did not produce the "
            "expected layout.",
            ", ".join(relpaths),
            args.run_id,
        )
        return 1

    logging.info("Overlaid %d instrumented subtree(s) onto %s", copied, install_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
