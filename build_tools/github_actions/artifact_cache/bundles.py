# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whole-artifact reuse with explicit component membership and one raw fingerprint.

The display/DB identity is <artifact>/bundle/<target>. The sorted component list
is also part of ArtifactKey, so a subset cannot satisfy a larger requested unit.
This module never manufactures a fingerprint for split or dirty source outputs.
"""

from pathlib import Path
import re
import subprocess

from .diagnostics import trace


def validate_bundle(identity, components):
    from .inventory import parse_identity
    from .local_store import safe_relative
    if (not components or any(not isinstance(c, str) for c in components)
            or tuple(sorted(set(components))) != components):
        raise ValueError("bundle components must be nonempty, unique and sorted")
    for component in components:
        safe_relative(component)
        parsed = parse_identity(component)
        if "/" in component or parsed is None:
            raise ValueError("bundle members must be emitted component identities")
        name, _, family = parsed
        if identity != f"{name}/bundle/{family}":
            raise ValueError("bundle members must belong to the same artifact and target")


def declared_ninja_outputs(build):
    """Read configured custom-command outputs; do not infer completeness from disk.

    Missing component directories remain expected. Split architecture outputs
    discovered on disk are retained for diagnostics even when not explicit Ninja
    outputs. Other generators require --expected to establish a bundle contract.
    """
    from .local_store import regular_reader, safe_relative
    build = Path(build).resolve()
    with regular_reader(build / "CMakeCache.txt") as stream:
        text = stream.read().decode("utf-8")
    values = dict(re.findall(r"^(CMAKE_GENERATOR|CMAKE_MAKE_PROGRAM):[^=]+=(.*)$", text, re.M))
    if values.get("CMAKE_GENERATOR") != "Ninja" or not values.get("CMAKE_MAKE_PROGRAM"):
        return set()
    command = [values["CMAKE_MAKE_PROGRAM"], "-C", str(build), "-t", "targets", "all"]
    trace("BUNDLE_OUTPUT_DISCOVERY", command=command, source="configured Ninja outputs")
    completed = subprocess.run(command, check=True, text=True, encoding="utf-8",
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    names = set()
    artifacts = (build / "artifacts").as_posix() + "/"
    for line in completed.stdout.splitlines():
        output, separator, _ = line.rpartition(": ")
        if not separator:
            continue
        output = output.replace("\\", "/")
        for prefix in ("artifacts/", artifacts):
            if output.startswith(prefix):
                relative = output[len(prefix):].split("/")
                if len(relative) == 2 and relative[1] == "artifact_manifest.txt":
                    safe_relative(relative[0])
                    names.add(relative[0])
                break
    trace("BUNDLE_OUTPUTS_DECLARED", count=len(names))
    return names


def group_records(records, *, authoritative):
    """Group complete requested sets; fail closed on missing/mismatched fingerprints."""
    from .inventory import parse_identity
    from .manifest import ArtifactKey
    groups, result = {}, []
    for record in records:
        identity = record["artifact_identity"]
        parsed = parse_identity(identity)
        if not parsed:
            result.append(({**record, "fingerprint_state": "invalid", "fingerprint": None,
                            "detail": "cannot determine artifact/component/target identity",
                            "components": [identity], "component_records": [record]}, None))
            continue
        name, _, family = parsed
        groups.setdefault(f"{name}/bundle/{family}", []).append(record)
    for identity, members in sorted(groups.items()):
        members.sort(key=lambda r: r["artifact_identity"])
        components = tuple(r["artifact_identity"] for r in members)
        validate_bundle(identity, components)
        missing = [r["artifact_identity"] for r in members if r["fingerprint_state"] == "missing"]
        invalid = [r for r in members if r["fingerprint_state"] != "valid"]
        fingerprints = {r["fingerprint"] for r in members if r["fingerprint_state"] == "valid"}
        state, fingerprint, reason = "valid", None, None
        if invalid:
            state = "missing" if len(missing) == len(invalid) else "invalid"
            reason = "; ".join(f"{r['artifact_identity']}: {r.get('detail') or r['fingerprint_state']}"
                               for r in invalid)
        elif len(fingerprints) != 1:
            state, reason = "invalid", "components have different artifact fingerprints"
        elif not set(components).issubset(authoritative):
            state, reason = "invalid", "complete component list is unknown; supply --expected or configure with Ninja"
        else:
            fingerprint = next(iter(fingerprints))
        record = {"artifact_identity": identity, "fingerprint_path": None,
                  "fingerprint_state": state, "fingerprint": fingerprint, "detail": reason,
                  "components": list(components), "component_records": members}
        key = ArtifactKey(identity, fingerprint, components=components) if state == "valid" else None
        trace("BUNDLE_KEY" if key else "BUNDLE_SKIPPED", artifact=identity,
              components=list(components), fingerprint=fingerprint,
              epoch=key.epoch if key else None, reason=reason)
        result.append((record, key))
    return sorted(result, key=lambda item: item[0]["artifact_identity"])


def component_requests(record):
    """Flatten a bundle request for the managed builder's sidecar checks."""
    names = record.get("components") or [record["artifact_identity"]]
    return [{"artifact_identity": name, "fingerprint_path": name + ".fprint"} for name in names]
