#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Local build artifact cache — cold/warm artifact reuse from a REAL local build.

Phase 0 of the roadmap runs on one machine with no CI and no network: a real
local TheRock build populates a cache, a later local build reuses from it. This
driver consumes exactly what a real build emits under ``build/artifacts/``:

    build/artifacts/
        <slice>_<component>_<suffix>/        # component OUTPUT DIRECTORY
        <slice>_<component>_<suffix>.fprint  # its source fingerprint (configure-time)

where ``<suffix>`` is ``generic`` (target-neutral) or a GPU family bundle name
(e.g. ``gfx94X-dcgpu``). The ``.fprint`` sidecars are written by TheRock's real
``THEROCK_FPRINT`` logic (``cmake/therock_artifacts.cmake``); all components of a
slice share one fingerprint. The cached *object* for a unit is the component
directory (tarred locally) — in CI the same unit maps to the uploaded archive.

Known real-build gaps this surfaces (Phase 0 is here to flush them):
  * Split / per-arch artifacts emit NO ``.fprint`` (cmake TODO) → uncacheable today.
  * Archives are produced by a separate packaging step, so a fingerprint-keyed
    cache keys on the component dir, not the archive.

You run ONE command, the same way every time, after each local build::

    python -m fingerprint_cache.local_cache cache --build <build> --cache ~/.therock-cache

The first run stores every artifact. Run it again with nothing changed and every
artifact is reused (you could have skipped rebuilding it). Change one subproject
and only that artifact is stored again; the rest are reused.

When you don't have a real build handy, ``synth`` fabricates a build tree in the
real layout so the loop is runnable anywhere.
"""


import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
import tarfile
import tempfile

from .cache_key import ArtifactIdentity, compute_cache_key
from .cache_db import CacheDB, ReadPolicy
from .manifest import Provenance, TrustLevel, ValidationState
from .publish import publish_build_outputs
from .restore import restore_manifest
from .local_store import LocalObjectStore
from .history_ingest import DEFAULT_COMPONENTS
from .topology_view import BUNDLE_COMPONENT

_DB_NAME = "cache.sqlite"
_OBJ_DIR = "objects"


@dataclass
class LocalUnit:
    """One (slice, target) reuse unit discovered in a real build tree."""

    identity: ArtifactIdentity
    source_fingerprint: str
    component_dirs: list[Path]  # real component output directories


@dataclass
class ScanResult:
    units: list[LocalUnit] = field(default_factory=list)
    uncacheable: list[str] = field(default_factory=list)  # dirs with no .fprint


def _parse_stem(
    stem: str, components=DEFAULT_COMPONENTS
) -> tuple[str, str, str] | None:
    """Parse ``<slice>_<component>_<suffix>`` (a bare directory/fprint stem).

    Slice names may contain underscores, so the component token (from
    ``components``) is the split point. Returns (slice, component, target) where
    target is the suffix (``generic`` or a GPU family).
    """
    tokens = stem.split("_")
    for i in range(len(tokens) - 2, -1, -1):
        if tokens[i] in components:
            name = "_".join(tokens[:i])
            component = tokens[i]
            target = "_".join(tokens[i + 1 :])
            if name and target:
                return (name, component, target)
    return None


def scan_build(build_dir: Path) -> ScanResult:
    """Discover cacheable units (and uncacheable slices) from a build tree."""
    build_dir = Path(build_dir)
    artifacts = build_dir / "artifacts"
    if not artifacts.is_dir():
        artifacts = build_dir  # allow pointing straight at an artifacts dir

    # Index .fprint sidecars and component output directories.
    fprint_by_stem: dict[str, str] = {}
    dirs_by_stem: dict[str, Path] = {}
    for entry in sorted(artifacts.iterdir()):
        if entry.is_file() and entry.name.endswith(".fprint"):
            fprint_by_stem[entry.name[: -len(".fprint")]] = entry.read_text().strip()
        elif entry.is_dir():
            dirs_by_stem[entry.name] = entry

    # Group component dirs that HAVE a fingerprint into (slice, target) units.
    groups: dict[tuple[str, str], dict[str, tuple[Path, str]]] = {}
    for stem, comp_dir in dirs_by_stem.items():
        parsed = _parse_stem(stem)
        if parsed is None:
            continue
        name, component, target = parsed
        fp = fprint_by_stem.get(stem)
        if fp is None:
            # Component output with no fingerprint — e.g. a split/per-arch
            # artifact. Not cacheable until TheRock emits .fprint for these.
            continue
        groups.setdefault((name, target), {})[component] = (comp_dir, fp)

    result = ScanResult()
    for (name, target), comps in sorted(groups.items()):
        fps = {fp for _d, fp in comps.values()}
        # All components of a slice should share one fingerprint; flag if not.
        source_fp = sorted(fps)[0] if len(fps) == 1 else _combine(comps)
        result.units.append(
            LocalUnit(
                identity=ArtifactIdentity(name, BUNDLE_COMPONENT, target),
                source_fingerprint=source_fp,
                component_dirs=[d for _c, (d, _fp) in sorted(comps.items())],
            )
        )

    # Report component dirs that parsed but had no fingerprint (the split gap).
    for stem, comp_dir in sorted(dirs_by_stem.items()):
        if _parse_stem(stem) and stem not in fprint_by_stem:
            result.uncacheable.append(stem)
    return result


def _combine(comps: dict[str, tuple[Path, str]]) -> str:
    h = hashlib.sha256()
    for component in sorted(comps):
        h.update(f"{component}={comps[component][1]}\n".encode())
    return h.hexdigest()


def _tar_dir(src: Path, dest_tar: Path) -> Path:
    with tarfile.open(dest_tar, "w") as tf:
        tf.add(src, arcname=src.name)
    return dest_tar


def _provenance() -> Provenance:
    return Provenance(
        "local",
        "local",
        "local",
        "local",
        "local_cache",
        datetime.now(timezone.utc).isoformat(),
    )


# ===================== commands =====================
def cmd_cache(args: argparse.Namespace) -> int:
    """One command, run the same way every time after a local build.

    For each artifact the build produced:
      * already in the cache (unchanged since a previous build) -> REUSED
        (you could have skipped rebuilding it);
      * not in the cache (new, or its sources changed) -> STORED now, so the
        next build can reuse it.

    First run stores everything. Run it again with no changes and everything is
    reused. Change one subproject and only that artifact is stored again.
    """
    scan = scan_build(Path(args.build))
    store = LocalObjectStore(Path(args.cache) / _OBJ_DIR)
    policy = ReadPolicy(
        min_trust=TrustLevel.TRUSTED, min_validation=ValidationState.BUILD_PASSED
    )
    reused, stored = 0, 0
    with CacheDB(
        str(Path(args.cache) / _DB_NAME)
    ) as db, tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for unit in scan.units:
            key = compute_cache_key(
                source_fingerprint=unit.source_fingerprint, build_variant=args.variant
            )
            existing = db.lookup(key.value, policy)
            if (
                existing is not None
                and restore_manifest(existing, store, verify_digests=True).reusable
            ):
                reused += 1
                print(f"  REUSED  {unit.identity}  (cache hit, skip rebuild)")
                continue
            # New or changed -> store it for next time.
            outputs = []
            for d in unit.component_dirs:
                tarred = _tar_dir(d, tmp / f"{d.name}.tar")
                outputs.append(store.upload_file(tarred, key=f"{d.name}.tar"))
            publish_build_outputs(
                db=db,
                artifact=unit.identity,
                outputs=outputs,
                provenance=_provenance(),
                source_fingerprint=unit.source_fingerprint,
                build_variant=args.variant,
                trust=TrustLevel.TRUSTED,
                validation=ValidationState.BUILD_PASSED,
            )
            stored += 1
            print(f"  STORED  {unit.identity}  (new/changed, cached for next build)")
        total = db.count()
    n = len(scan.units)
    if stored and not reused:
        headline = f"First build: stored all {stored} artifacts in the cache."
    elif reused and not stored:
        headline = f"Nothing changed: reused all {reused} artifacts from the cache."
    else:
        headline = (
            f"Reused {reused} unchanged artifact(s); stored "
            f"{stored} new/changed one(s)."
        )
    print(f"\n{headline}  ({n} cacheable; cache holds {total} entries.)")
    if scan.uncacheable:
        print(
            f"  Not cacheable yet ({len(scan.uncacheable)}, no .fprint — "
            f"split/per-arch gap): {', '.join(scan.uncacheable)}"
        )
    return 0


def cmd_keys(args: argparse.Namespace) -> int:
    """Fingerprint + key decision ONLY — no cache store, no byte copying.

    This exercises just the cache-key side: read each real .fprint, group into
    units, compute the cache key, and report what is cacheable vs not. Needs
    only a configured/built tree (no --cache dir). Use it to validate the
    fingerprint->key->cacheability logic in isolation.
    """
    scan = scan_build(Path(args.build))
    for unit in scan.units:
        key = compute_cache_key(
            source_fingerprint=unit.source_fingerprint, build_variant=args.variant
        )
        print(
            f"  {str(unit.identity):<34} fprint={unit.source_fingerprint[:12]}  "
            f"key={key.short}  components={len(unit.component_dirs)}"
        )
    n = len(scan.units)
    total = n + len(scan.uncacheable)
    pct = (n / total * 100) if total else 0
    print(f"\nCacheable units: {n}/{total} ({pct:.0f}%).")
    if scan.uncacheable:
        print(
            f"Not cacheable ({len(scan.uncacheable)}, no .fprint — split/per-arch gap):"
        )
        for s in scan.uncacheable:
            print(f"  {s}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    import json

    with CacheDB(str(Path(args.cache) / _DB_NAME)) as db:
        print(json.dumps(db.stats(), indent=2))
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    """Fabricate a build tree in the REAL layout.

    Emits component output directories + ``.fprint`` sidecars exactly as a real
    build does, plus one split artifact with NO fingerprint to exercise the gap.
    ``--change NAME`` bumps a slice's revision so its fingerprint flips.
    """
    artifacts = Path(args.out) / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    changed = set(args.change or [])
    targets = args.targets.split(",") if args.targets else ["gfx94X-dcgpu"]
    # (slice, components, target_specific, split?)
    spec = [
        ("amd-llvm", ["run"], False, False),
        ("core-hip", ["lib", "dev"], False, False),
        ("base", ["run"], False, False),
        ("blas", ["lib", "dev"], True, False),
        ("fft", ["lib"], True, False),
        ("miopen", ["lib"], True, True),  # split artifact -> no .fprint (gap)
    ]
    units = 0
    for name, components, target_specific, split in spec:
        rev = "r2" if name in changed else "r1"
        families = targets if target_specific else ["generic"]
        for target in families:
            fp = hashlib.sha256(f"{name}:{target}:{rev}".encode()).hexdigest()
            for component in components:
                stem = f"{name}_{component}_{target}"
                comp_dir = artifacts / stem
                comp_dir.mkdir(exist_ok=True)
                (comp_dir / "payload.txt").write_text(f"{name} {component} {target}\n")
                if not split:
                    (artifacts / f"{stem}.fprint").write_text(fp)
                units += 1
    note = f" (changed: {', '.join(sorted(changed))})" if changed else ""
    print(
        f"Synthesized build tree under {artifacts}{note} "
        f"(miopen is split -> no .fprint, by design)"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="fingerprint_cache.local_cache", description=__doc__
    )
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("synth", help="fabricate a real-layout build tree (fixture)")
    s.add_argument("--out", required=True)
    s.add_argument("--targets", default="gfx94X-dcgpu")
    s.add_argument("--change", nargs="*", help="slice names whose fingerprint to flip")
    s.set_defaults(func=cmd_synth)
    c = sub.add_parser(
        "cache",
        help="reuse what's cached + store what's new " "(run after every build)",
    )
    c.add_argument("--build", required=True)
    c.add_argument("--cache", required=True)
    c.add_argument("--variant", default="release")
    c.set_defaults(func=cmd_cache)

    k = sub.add_parser(
        "keys", help="fingerprint+key decision only — no store, " "no --cache needed"
    )
    k.add_argument("--build", required=True)
    k.add_argument("--variant", default="release")
    k.set_defaults(func=cmd_keys)
    st = sub.add_parser("stats", help="print cache stats")
    st.add_argument("--cache", required=True)
    st.set_defaults(func=cmd_stats)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
