#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolve Docker image references from the TheRock central image registry.

Reads docker_images.json at the repo root and formats full image references
suitable for use in GitHub Actions workflow container fields, Dockerfile FROM
lines, and environment variables.

docker_images.json format
-------------------------
The file is a JSON object. Keys starting with "_" (e.g. "_comment") are
ignored by this script and may be used for documentation. Every other key is a
logical image name (e.g. "therock_build_manylinux_x86_64") whose value is an
object with the following fields:

  Required:
    registry  (str)  Registry hostname and optional path prefix.
                     Examples: "ghcr.io/rocm", "docker.io", "quay.io/pypa"
    image     (str)  Image name within the registry, lowercase, may contain
                     "/" for namespaced images.
                     Examples: "therock_build_manylinux_x86_64", "ubi10/ubi"

  Pin fields (at least one must be non-null):
    sha       (str|null)  SHA-256 digest in the form "sha256:<64 hex chars>".
                          When set, the tag field is ignored during resolution.
    tag       (str|null)  Docker tag string, e.g. "3.12-slim" or "24.04".
                          Used only when sha is null.

  Informational (not used by this script):
    timestamp    (str|null)  Creation or publish timestamp of the image itself,
                             in "YYYY-MM-DD" or ISO 8601 form. Tells you how
                             old the running image is.
    last_updated (str|null)  Date this JSON entry was last changed, in
                             "YYYY-MM-DD" form. Tells you how long ago we
                             bumped the pin. Use git log for full history.

Reference resolution priority:
  1. sha present             -> registry/image@sha256:...   (immutable)
  2. sha null, tag present   -> registry/image:tag          (mutable)
  3. sha null, tag null      -> hard error; one must be set

To bump an image: update sha, set timestamp to the image's publish date, and
set last_updated to today's date.

Subcommands:
  get-image    Print the full reference for a named image key.
  list         List all known image keys and their current references.
  validate     Check all entries in docker_images.json for errors.

Examples:
  python3 build_tools/resolve_docker_image.py get-image therock_build_manylinux_x86_64
  python3 build_tools/resolve_docker_image.py list
  python3 build_tools/resolve_docker_image.py validate
  python3 build_tools/resolve_docker_image.py get-image --output gha no_rocm_image_ubuntu24_04
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_DOCKER_IMAGES_PATH = _REPO_ROOT / "docker_images.json"

# sha256: followed by exactly 64 lowercase hex digits
_SHA_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

# Docker tag: printable ASCII, no whitespace or forward-slash, max 128 chars
_TAG_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,128}$")

# Registry hostname (optional port) or hostname/path-prefix
_REGISTRY_RE = re.compile(
    r"^[a-zA-Z0-9]([a-zA-Z0-9\-\.]*[a-zA-Z0-9])?(:\d+)?(/[a-zA-Z0-9_.\-]+)*$"
)

# Image name: one or more path components separated by /
_IMAGE_RE = re.compile(r"^[a-z0-9_.\-]+(/[a-z0-9_.\-]+)*$")


@dataclass
class ValidationError:
    key: str
    message: str

    def __str__(self) -> str:
        return f"[{self.key}] {self.message}"


def validate_entry(key: str, entry: object) -> list[ValidationError]:
    """Return a list of validation errors for one docker_images.json entry.

    An empty list means the entry is valid.
    """
    errors: list[ValidationError] = []

    if not isinstance(entry, dict):
        errors.append(
            ValidationError(key, f"entry must be an object, got {type(entry).__name__}")
        )
        return errors

    def err(msg: str) -> None:
        errors.append(ValidationError(key, msg))

    registry = entry.get("registry")
    image = entry.get("image")
    sha = entry.get("sha")
    tag = entry.get("tag")

    # Required fields
    if not registry or not isinstance(registry, str):
        err("'registry' must be a non-empty string")
    elif not _REGISTRY_RE.match(registry):
        err(f"'registry' value {registry!r} is not a valid registry hostname/path")

    if not image or not isinstance(image, str):
        err("'image' must be a non-empty string")
    elif not _IMAGE_RE.match(image):
        err(
            f"'image' value {image!r} contains invalid characters (use lowercase, digits, _, ., -)"
        )

    # SHA validation
    if sha is not None:
        if not isinstance(sha, str):
            err(f"'sha' must be a string or null, got {type(sha).__name__}")
        elif not _SHA_RE.match(sha):
            err(
                f"'sha' value {sha!r} is not a valid sha256 digest (expected sha256:<64 hex chars>)"
            )

    # Tag validation
    if tag is not None:
        if not isinstance(tag, str):
            err(f"'tag' must be a string or null, got {type(tag).__name__}")
        elif not _TAG_RE.match(tag):
            err(f"'tag' value {tag!r} contains invalid characters or exceeds 128 chars")

    # Neither sha nor tag — no safe reference can be formed
    if sha is None and tag is None:
        errors.append(
            ValidationError(key, "entry has neither 'sha' nor 'tag'; one must be set")
        )

    return errors


def load_images(path: Path = _DOCKER_IMAGES_PATH) -> dict:
    """Load and return the docker_images.json registry, minus comment keys.

    Raises:
        FileNotFoundError: If docker_images.json does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with path.open() as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def resolve(entry: dict, key: str) -> str:
    """Return the full image reference string for a single registry entry.

    Args:
        entry: One object from docker_images.json.
        key: The image key, used only in error/warning messages.

    Returns:
        Full image reference string.

    Raises:
        ValueError: If the entry is missing required fields.
    """
    errs = validate_entry(key, entry)
    if errs:
        raise ValueError("; ".join(e.message for e in errs))

    registry = entry["registry"].rstrip("/")
    image = entry["image"]
    sha = entry.get("sha")
    tag = entry.get("tag")
    base = f"{registry}/{image}"

    if sha:
        return f"{base}@{sha}"
    if tag:
        return f"{base}:{tag}"

    raise ValueError(
        f"{key!r} has neither 'sha' nor 'tag' in docker_images.json; one must be set"
    )


def get_image_ref(key: str, path: Path = _DOCKER_IMAGES_PATH) -> str:
    """Return the full image reference for the given key.

    Raises:
        KeyError: If the key is not present in docker_images.json.
        ValueError: If the entry is malformed.
    """
    images = load_images(path)
    if key not in images:
        known = ", ".join(sorted(images))
        raise KeyError(f"Unknown image key {key!r}. Known keys: {known}")
    return resolve(images[key], key)


def cmd_get_image(args: argparse.Namespace) -> int:
    try:
        ref = get_image_ref(args.key, path=_DOCKER_IMAGES_PATH)
    except (KeyError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.output == "gha":
        sys.path.insert(0, str(_REPO_ROOT))
        from build_tools.github_actions.github_actions_api import gha_set_output

        gha_set_output({"image": ref})
    else:
        print(ref)

    return 0


def cmd_list(args: argparse.Namespace) -> int:
    try:
        images = load_images()
    except Exception as e:
        print(f"Error reading docker_images.json: {e}", file=sys.stderr)
        return 1

    width = max(len(k) for k in images)
    for key, entry in images.items():
        try:
            ref = resolve(entry, key)
        except ValueError as e:
            ref = f"<error: {e}>"
        print(f"{key:<{width}}  {ref}")

    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        images = load_images()
    except FileNotFoundError:
        print(f"Error: {_DOCKER_IMAGES_PATH} not found", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: docker_images.json is not valid JSON: {e}", file=sys.stderr)
        return 1

    errors: list[ValidationError] = []

    for key, entry in images.items():
        errors.extend(validate_entry(key, entry))

    for e in errors:
        print(f"error: {e}", file=sys.stderr)

    if errors:
        print(f"\n{len(errors)} error(s) found in docker_images.json.", file=sys.stderr)
        return 1

    print(f"docker_images.json: all {len(images)} entries are valid.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Resolve Docker image references from docker_images.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_get = subparsers.add_parser(
        "get-image",
        help="Print the full reference for a named image key.",
    )
    p_get.add_argument(
        "key",
        help="Image key from docker_images.json (e.g. therock_build_manylinux_x86_64).",
    )
    p_get.add_argument(
        "--output",
        choices=["print", "gha"],
        default="print",
        help="'print' writes to stdout (default); 'gha' writes image= to $GITHUB_OUTPUT.",
    )
    p_get.set_defaults(func=cmd_get_image)

    p_list = subparsers.add_parser(
        "list",
        help="List all known image keys and their resolved references.",
    )
    p_list.set_defaults(func=cmd_list)

    p_val = subparsers.add_parser(
        "validate",
        help="Check all entries in docker_images.json for errors and warnings.",
    )
    p_val.set_defaults(func=cmd_validate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
