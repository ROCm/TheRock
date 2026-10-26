# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read emitted component identities with explicit component or bundle granularity."""

import json
from pathlib import Path
import re

from .manifest import ArtifactKey
from .diagnostics import trace

COMPONENTS = ("lib", "dev", "run", "dbg", "doc", "test")


def collector_module():
    if __package__.startswith("build_tools."):
        from . import collector as artifact_cache
    else:
        from . import collector as artifact_cache
    return artifact_cache


def parse_identity(identity):
    if "/bundle/" in identity:
        name, family = identity.split("/bundle/", 1)
        return (name, "bundle", family) if name and family and "/" not in name + family else None
    tokens = identity.split("_")
    for i in range(len(tokens) - 2, 0, -1):
        if tokens[i] in COMPONENTS:
            return "_".join(tokens[:i]), tokens[i], "_".join(tokens[i + 1:])
    return None


def expected_outputs(build, expected_manifest=None):
    """An explicit inventory is authoritative; discovery only describes disk."""
    if expected_manifest is not None:
        result = collector_module().load_expected(Path(expected_manifest))
        trace("DISCOVERY", mode="explicit_outputs", expected_file=Path(expected_manifest).absolute(),
              artifacts_directory=Path(build).absolute() / "artifacts", count=len(result))
        return result
    artifacts = Path(build) / "artifacts"
    if not artifacts.is_dir():
        raise ValueError(f"no artifact directory: {artifacts}")
    names = set()
    for path in artifacts.iterdir():
        if path.name.endswith(".fprint"):
            names.add(path.name[:-7])
        elif path.is_dir():
            names.add(path.name)
    trace("DISCOVERY", mode="observed_outputs", artifacts_directory=artifacts.absolute(),
          count=len(names), sources="immediate directories and .fprint filenames")
    return [{"artifact_identity": name, "fingerprint_path": name + ".fprint"}
            for name in sorted(names)]


def write_expected(path, expected):
    Path(path).write_text(json.dumps({"schema_version": 1, "artifacts": expected}),
                          encoding="utf-8")


def scan_build(build, expected_manifest=None, *, granularity="component"):
    if granularity not in ("component", "bundle"):
        raise ValueError("granularity must be component or bundle")
    from .local_store import safe_relative
    expected = expected_outputs(build, expected_manifest)
    authoritative = {item["artifact_identity"] for item in expected}
    if granularity == "bundle" and expected_manifest is None:
        from .bundles import declared_ninja_outputs
        authoritative = declared_ninja_outputs(build)
        discovered = {item["artifact_identity"]: item for item in expected}
        for name in authoritative:
            discovered.setdefault(name, {"artifact_identity": name, "fingerprint_path": name + ".fprint"})
        expected = [discovered[name] for name in sorted(discovered)]
    for item in expected:
        identity = safe_relative(item["artifact_identity"])
        if "/" in identity or item["fingerprint_path"] != identity + ".fprint":
            raise ValueError("local workflow requires emitted component directory/sidecar identities")
    records = collector_module().read_fingerprints(
        expected, Path(build) / "artifacts", "succeeded")
    if granularity == "bundle":
        from .bundles import group_records
        return group_records(records, authoritative=authoritative)
    result = [(record, ArtifactKey(record["artifact_identity"], record["fingerprint"])
               if record["fingerprint_state"] == "valid" else None) for record in records]
    for record, key in result:
        if key is not None:
            trace("KEY", artifact=key.artifact_identity, epoch=key.epoch,
                  fingerprint=key.fingerprint)
    return result


def configured_build(build, variant=None):
    from .local_store import regular_reader
    build = Path(build).expanduser().resolve(strict=True)
    with regular_reader(build / "CMakeCache.txt") as stream:
        lines = stream.read().decode("utf-8").splitlines()
    values = {}
    for line in lines:
        match = re.match(r"^(CMAKE_HOME_DIRECTORY|CMAKE_COMMAND|CMAKE_BUILD_TYPE):[^=]+=(.*)$", line)
        if match:
            values[match[1]] = match[2]
    if not all(values.get(k) for k in ("CMAKE_HOME_DIRECTORY", "CMAKE_COMMAND")):
        raise ValueError("build directory lacks CMake source/tool locations")
    if variant and values.get("CMAKE_BUILD_TYPE", "").lower() != variant.lower():
        raise ValueError("--variant must match the configured CMAKE_BUILD_TYPE")
    return build, Path(values["CMAKE_HOME_DIRECTORY"]).resolve(strict=True), values["CMAKE_COMMAND"]


class TopologyAdapter:
    """Map component outputs to stages using TheRock's existing topology parser."""

    def __init__(self, topology):
        self.topology = topology
        self.stages = list(topology.get_build_order())
        self.owners = {}
        for stage in self.stages:
            for name in topology.get_produced_artifacts(stage):
                if name in self.owners and self.owners[name] != stage:
                    raise ValueError(f"ambiguous topology owner: {name}")
                self.owners[name] = stage

    @classmethod
    def from_path(cls, path=None):
        import sys
        build_tools = str(Path(__file__).resolve().parents[2])
        if build_tools not in sys.path:
            sys.path.insert(0, build_tools)
        from _therock_utils.build_topology import get_topology
        return cls(get_topology(Path(path) if path else None))

    def stage_for(self, identity):
        parsed = parse_identity(identity)
        return self.owners.get(parsed[0]) if parsed else None

    def changed_names(self, paths):
        names = set()
        for path in paths:
            affected = self.topology.get_artifacts_for_path(path)
            if not affected:
                return set(self.owners)
            names.update(affected)
        return names
