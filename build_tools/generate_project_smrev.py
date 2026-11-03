#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generate and validate project source identities without a repository-HEAD fallback."""

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile


SCHEMA = "therock-project-smrev-v2"
CATALOG = Path(__file__).with_name("project_smrev_inputs.json")


class SourceIdentityError(RuntimeError):
    pass


def git(repo, *args, optional=False):
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True,
        env=dict(os.environ, GIT_OPTIONAL_LOCKS="0", LC_ALL="C"),
    )
    if result.returncode:
        if optional and result.returncode == 128 and any(
            message in result.stderr for message in (
                b"No names found", b"No tags can describe", b"no tag exactly matches"
            )
        ):
            return None
        raise SourceIdentityError(
            f"git {' '.join(args)} in {repo}: {result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout.decode("utf-8", "surrogateescape").rstrip("\n")


def source_path(value):
    if not isinstance(value, str) or not value or any(c in value for c in "\\\n\r;:"):
        raise SourceIdentityError(f"invalid source path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise SourceIdentityError(f"expected a normalized repository-relative path: {value!r}")
    return value


def check_clean(repo, paths):
    # Match upstream's whole-repository tracked-source rule. Gitlinks inside the
    # selected scope are validated separately; unrelated submodules need not be populated.
    if git(repo, "diff", "--name-only", "--ignore-submodules=all", "HEAD", "--"):
        raise SourceIdentityError(f"tracked source changes in {repo}; commit or restore them first")
    pathspecs = [f":(literal){path}" for path in paths if path != "."]
    if "." in paths:
        pathspecs = []
    if git(repo, "ls-files", "--others", "--exclude-standard", "--", *pathspecs):
        raise SourceIdentityError(f"untracked files under source inputs in {repo}")
    # Hidden tracked modifications must not pass git diff through index flags.
    for record in git(repo, "ls-files", "-v", "-z", "--", *pathspecs).split("\0"):
        if record and (record[0].islower() or record[0] == "S"):
            raise SourceIdentityError(
                f"assume-unchanged/skip-worktree input is unsupported: {repo / record[2:]}"
            )


def tree_entry(repo, revision, path):
    if path == ".":
        return {"path": ".", "type": "tree", "mode": "040000",
                "object_id": git(repo, "rev-parse", f"{revision}^{{tree}}")}
    output = git(repo, "ls-tree", "-z", revision, "--", f":(literal){path}")
    if not output:
        return {"path": path, "type": "absent"}
    records = [r for r in output.split("\0") if r]
    if len(records) != 1:
        raise SourceIdentityError(f"ambiguous source input: {path}")
    metadata, actual = records[0].split("\t", 1)
    mode, kind, object_id = metadata.split()
    if actual != path or kind not in ("tree", "blob"):
        raise SourceIdentityError(f"source root must be a tracked file or directory: {path}")
    return {"path": path, "type": kind, "mode": mode, "object_id": object_id}


def nested_sources(repo, entry):
    if entry["type"] != "tree":
        return []
    records = []
    prefix = "" if entry["path"] == "." else entry["path"] + "/"
    for line in git(repo, "ls-tree", "-rz", entry["object_id"]).split("\0"):
        if not line:
            continue
        metadata, relative = line.split("\t", 1)
        mode, kind, pin = metadata.split()
        if mode != "160000":
            continue
        relative = source_path(prefix + relative)
        checkout = repo / relative
        record = {"path": relative, "pin": pin}
        # Availability can change a conditional find_package/add_subdirectory.
        # Record absence explicitly; never invent a tree ID for missing sources.
        if not checkout.exists() or (checkout.is_dir() and not any(checkout.iterdir())):
            record["state"] = "uninitialized"
        else:
            if checkout.resolve() != checkout or not checkout.is_dir():
                raise SourceIdentityError(f"invalid nested checkout: {checkout}")
            if Path(git(checkout, "rev-parse", "--show-toplevel")).resolve() != checkout:
                raise SourceIdentityError(f"nested source is not an independent checkout: {checkout}")
            if git(checkout, "rev-parse", "HEAD") != pin:
                raise SourceIdentityError(f"nested source differs from its recorded pin: {checkout}")
            check_clean(checkout, ["."])
            record["state"] = "checked_out"
            # Nested projects may interpret their own tags. Preserve this
            # conservative input until a narrower nested policy is reviewed.
            record["tags"] = git(checkout, "for-each-ref", "--sort=refname",
                                  "--format=%(refname) %(objectname)", "refs/tags").splitlines()
            record["children"] = nested_sources(checkout, tree_entry(checkout, pin, "."))
        records.append(record)
    return records


def version_inputs(repo, revision, queries):
    values = []
    for query in queries:
        kind = query["kind"]
        if kind in ("nearest", "timestamp") and git(repo, "rev-parse", "--is-shallow-repository") == "true":
            raise SourceIdentityError(f"version input requires complete Git history in {repo}; unshallow it first")
        if kind == "latest":
            result = git(repo, "tag", "--list", "--sort=-version:refname", "--", query["pattern"])
            value = next(iter(result.splitlines()), None)
        elif kind == "nearest":
            arguments = ["describe", "--abbrev=0", "--match", query["pattern"]]
            if query.get("tags", True):
                arguments.append("--tags")
            value = git(repo, *arguments, revision, optional=True)
        elif kind == "exact":
            value = git(repo, "describe", "--tags", "--exact-match", revision, optional=True)
        elif kind == "timestamp":
            path = source_path(query["path"])
            value = git(repo, "log", "-1", "--format=%ct", revision, "--", f":(literal){path}")
        else:
            raise SourceIdentityError(f"unrecognized version query: {kind}")
        values.append({"query": query, "value": value})
    return values


def calculate(repo, profile, repository_name):
    """Identity deliberately excludes commit/branch/count provenance labels.

    The catalog retains tag-derived releases and semantic project timestamps.
    Reused artifacts keep their original producer's metadata; we never rewrite it.
    """
    paths = sorted({source_path(p) for p in profile["paths"]})
    optional = sorted({source_path(p) for p in profile.get("optional_paths", [])} - set(paths))
    selected = paths + optional
    check_clean(repo, selected)
    revision = git(repo, "rev-parse", "HEAD")
    entries = [tree_entry(repo, revision, p) for p in selected]
    for entry in entries:
        path = repo / entry["path"]
        if entry["type"] == "absent":
            if entry["path"] in paths or os.path.lexists(path):
                raise SourceIdentityError(f"missing or uncommitted source input: {path}")
        elif path.resolve() != path or not path.exists():
            raise SourceIdentityError(f"input root must be populated without symlink traversal: {path}")
    nested = [n for e in entries for n in nested_sources(repo, e)]
    versions = version_inputs(repo, revision, profile.get("versions", []))
    environment_names = set(profile.get("environment", []))
    environment_patterns = sorted(set(profile.get("environment_patterns", [])))
    for pattern in environment_patterns:
        environment_names.update(name for name in os.environ if re.fullmatch(pattern, name))
    environment = {name: os.environ.get(name) for name in sorted(environment_names)}
    identity = {
        "schema": SCHEMA, "repository": repository_name,
        "version_policy": "retain-producer-provenance-v1",
        "inputs": entries, "version_inputs": versions, "environment": environment,
        "nested_sources": nested, "environment_patterns": environment_patterns,
    }
    check_clean(repo, selected)
    if revision != git(repo, "rev-parse", "HEAD"):
        raise SourceIdentityError("source revision changed during identity generation; retry")
    if versions != version_inputs(repo, revision, profile.get("versions", [])):
        raise SourceIdentityError("version inputs changed during identity generation; retry")
    if nested != [n for e in entries for n in nested_sources(repo, e)]:
        raise SourceIdentityError("nested sources changed during identity generation; retry")
    return identity


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()


def write_changed(path, data):
    if path.exists() and path.read_bytes() == data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".smrev-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def owned_marker(path):
    try:
        return not path.is_symlink() and json.loads(path.read_text())["schema"] == SCHEMA
    except (OSError, ValueError, KeyError, TypeError):
        return False


def generate(target, source, output, *, check=False, catalog_path=CATALOG):
    output = output.absolute()
    # Never follow an output symlink, including one in a parent directory.
    if output.is_symlink() or output.resolve() != output:
        raise SourceIdentityError("source marker output must not traverse a symlink")
    if output.exists() and not owned_marker(output):
        raise SourceIdentityError(f"refusing to replace foreign marker: {output}")
    try:
        catalog = json.loads(catalog_path.read_text())
        if catalog.get("schema_version") != 1 or target not in catalog["targets"]:
            raise SourceIdentityError(f"no source-identity policy for target {target}")
        target_policy = catalog["targets"][target]
        source = source.resolve(strict=True)
        repo = Path(git(source, "rev-parse", "--show-toplevel")).resolve()
        if output.is_relative_to(repo):
            raise SourceIdentityError("source marker must be outside its source repository")
        if source.relative_to(repo).as_posix() != target_policy["entry"]:
            raise SourceIdentityError(f"{target}: source checkout/layout does not match the declared input policy")
        profile = catalog["profiles"][target_policy["profile"]]
        identity = calculate(repo, profile, target_policy["repository"])
        data = encoded(identity)
        if check:
            if not output.exists() or output.read_bytes() != data:
                raise SourceIdentityError(f"{target}: inputs changed since configuration; rerun cmake -S . -B <build>")
        else:
            write_changed(output, data)
        return identity
    except BaseException:
        if owned_marker(output):
            output.unlink()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--check-list", type=Path)
    parser.add_argument("--check-request", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.check_list or args.check_request:
            requests = ([str(args.check_request)] if args.check_request else
                        args.check_list.read_text().splitlines())
            for line in requests:
                request = json.loads(Path(line).read_text())
                generate(request["target"], Path(request["source"]), Path(request["output"]), check=True)
        else:
            if not all((args.target, args.source, args.output, args.request)):
                parser.error("--target, --source, --output and --request are required")
            generate(args.target, args.source, args.output)
            write_changed(args.request, encoded({
                "target": args.target, "source": str(args.source.resolve()),
                "output": str(args.output.absolute()),
            }))
            print(f"Project source identity: {args.target} -> {args.output}")
    except (OSError, ValueError, KeyError, SourceIdentityError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
