#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Multi-arch release artifact-ready gate.

Used by the multi-arch release workflows as a small synthetic
`build_artifacts` job that gates downstream package/test jobs on whichever
producer (source build or prebuilt copy) is active for this run.

The job graph looks like this::

    build_source         copy_prebuilt
            \\           /
             verify_artifacts_ready    <- this script
                       |
        +--------+-----+-----+----------+
        |        |           |          |
    tarballs  python    native      tests
              packages  packages

Mode selection:

- If `--prebuilt-prefix` is empty, the active producer is `build_source`
  and the script ignores `--copy-prebuilt-result`.
- If `--prebuilt-prefix` is non-empty, the active producer is
  `copy_prebuilt` and the script ignores `--build-source-result`.

Exit code:

- 0 if the active producer's GitHub Actions job result is `success`.
- 1 otherwise (failure, cancelled, or skipped).

Skipping the script entirely (e.g. `if: !always()`) would mask producer
failures. Always run it and let it gate explicitly.
"""

import argparse
import sys


# GitHub Actions job result strings.
RESULT_SUCCESS = "success"


def decide(
    prebuilt_prefix: str,
    build_source_result: str,
    copy_prebuilt_result: str,
) -> tuple[str, str]:
    """Return (producer_name, producer_result) for the active producer."""
    if prebuilt_prefix:
        return ("copy_prebuilt", copy_prebuilt_result)
    return ("build_source", build_source_result)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="verify_artifacts_ready.py")
    p.add_argument(
        "--prebuilt-prefix",
        default="",
        help=(
            "Value of the workflow's prebuilt_prefix input. Empty selects "
            "source-build mode; non-empty selects prebuilt-copy mode."
        ),
    )
    p.add_argument(
        "--build-source-result",
        default="",
        help="GitHub Actions result of the source-build job (success|failure|cancelled|skipped).",
    )
    p.add_argument(
        "--copy-prebuilt-result",
        default="",
        help="GitHub Actions result of the prebuilt-copy job (success|failure|cancelled|skipped).",
    )
    args = p.parse_args(argv)

    producer, result = decide(
        prebuilt_prefix=args.prebuilt_prefix.strip(),
        build_source_result=args.build_source_result.strip(),
        copy_prebuilt_result=args.copy_prebuilt_result.strip(),
    )

    if result == RESULT_SUCCESS:
        print(f"Active producer '{producer}' succeeded; artifacts are ready.")
        return 0

    print(
        f"ERROR: active producer '{producer}' has result '{result}' "
        f"(expected '{RESULT_SUCCESS}')."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
