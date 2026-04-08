#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolve ``PYTHON_CMD`` from ``--os-profile`` and optionally install that runtime.

Mapping (install packages + ``PYTHON_CMD``):

- ``ubuntu*`` / ``debian*`` -> apt: ``python3.12``, ``python3.12-venv``, ``python3-pip`` -> ``python3.12``
- ``sles*`` -> zypper: ``python313``, ``python313-pip`` -> ``python3.13``
- else (e.g. ``rhel10``) -> dnf: ``python3.12``, ``python3.12-pip`` -> ``python3.12``

Use ``--install-runtime`` in CI so Python install lives in this script (not the workflow
prerequisites). The workflow may run a tiny bootstrap if ``python3`` is missing before
calling this script.

``--output-format`` matches ``get_s3_config.py``: ``env``, ``json``, ``github``.

Sample usage
------------

CI (install + append ``PYTHON_CMD`` to ``GITHUB_ENV``)::

    python3 build_tools/packaging/linux/set_python_cmd.py \\
        --os-profile ubuntu2404 --install-runtime >> \"$GITHUB_ENV\"

Resolve only (no package manager)::

    python3 build_tools/packaging/linux/set_python_cmd.py --os-profile rhel10 --output-format json
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Literal

# Matches get_s3_config.py --output-format vocabulary.
OutputFormat = Literal["env", "json", "github"]


# Maps os_profile to the interpreter name; must stay aligned with install + workflow.
def resolve_python_cmd(os_profile: str) -> str:
    """Map ``os_profile`` to the interpreter command name (e.g. ``python3.12``).

    Must match packages installed by ``install_python_runtime`` and the workflow.
    """
    # Keep in sync with test_native_linux_packages_install.yml + install_python_runtime().
    if os_profile.startswith(("ubuntu", "debian")):
        return "python3.12"
    if os_profile.startswith("sles"):
        return "python3.13"
    return "python3.12"


# Installs Python + pip with apt/zypper/dnf;
def install_python_runtime(os_profile: str) -> None:
    """Install Python and pip for this OS using apt, zypper, or dnf.

    Idempotent: safe to run again if packages are already present.
    """
    if os_profile.startswith(("ubuntu", "debian")):
        # Non-interactive apt (CI containers); avoids debconf prompts.
        env = {**os.environ, "DEBIAN_FRONTEND": "noninteractive"}
        # Refresh package index so install sees current repos; -qq reduces log noise.
        subprocess.run(["apt-get", "update", "-qq"], check=True, env=env)
        subprocess.run(
            [
                "apt-get",
                "install",
                "-y",
                # Skip Recommended packages for smaller/faster CI images.
                "--no-install-recommends",
                "python3.12",
                "python3.12-venv",  # python3.12 -m venv
                "python3-pip",  # python3.12 -m pip (workflow requirements.txt)
            ],
            check=True,
            env=env,
        )
    elif os_profile.startswith("sles"):
        # zypper refresh ~= apt update (metadata before install).
        subprocess.run(
            ["zypper", "--non-interactive", "refresh"],
            check=True,
        )
        subprocess.run(
            [
                "zypper",
                "--non-interactive",
                "install",
                "-y",
                "python313",
                "python313-pip",
            ],
            check=True,
        )
    else:
        # RHEL UBI: --allowerasing resolves curl vs curl-minimal style conflicts when pulling deps.
        subprocess.run(
            [
                "dnf",
                "install",
                "-y",
                "--allowerasing",
                "python3.12",
                "python3.12-pip",
            ],
            check=True,
        )


def emit_output(cmd: str, output_format: OutputFormat) -> None:
    """Print ``PYTHON_CMD`` to stdout as JSON, ``NAME=value`` (github), or ``export`` (env)."""
    if output_format == "json":
        print(json.dumps({"python_cmd": cmd}))
    elif output_format == "github":
        # One NAME=value line per get_s3_config github format; often appended to GITHUB_ENV.
        print(f"PYTHON_CMD={cmd}")
    else:
        # Shell: eval "$(...)" or copy-paste.
        print(f"export PYTHON_CMD={cmd}")


# Parses CLI, optionally installs runtime, then emits PYTHON_CMD.
def main(argv: list[str]) -> int:
    """CLI entry: parse args, optionally install runtime, then emit ``PYTHON_CMD``."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--os-profile",
        required=True,
        help="e.g. ubuntu2404, rhel10, sles16",
    )
    parser.add_argument(
        "--install-runtime",
        action="store_true",
        help="Install OS packages for this profile (apt/zypper/dnf) before emitting output.",
    )
    parser.add_argument(
        "--output-format",
        choices=["env", "json", "github"],
        default="github",
        help="Like get_s3_config.py: env, json, or github (default: github for GITHUB_ENV)",
    )
    args = parser.parse_args(argv)

    # Install first so emitted PYTHON_CMD exists on PATH for later workflow steps.
    if args.install_runtime:
        install_python_runtime(args.os_profile)

    cmd = resolve_python_cmd(args.os_profile)
    emit_output(cmd, args.output_format)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
