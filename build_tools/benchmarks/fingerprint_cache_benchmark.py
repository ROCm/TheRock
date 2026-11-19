#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

r"""Benchmark the fingerprint artifact cache: hit rate, timing, local vs cloud, gaps.

This is a throwaway measurement harness (not production code) for evaluating the
fingerprint-based CI artifact cache (issue #8651) before it is productionized.
It answers: how much does the cache reuse, how fast is a lookup locally vs. in
the cloud (ClickHouse), and -- most importantly for the go/no-go decision --
where does the cache miss when it should not, and which artifacts cannot be
cached at all.

Two modes, both producing the same Markdown + HTML report:

  simulate  (default, no builds; runs in seconds)
      Reads the fingerprints a real build already produced under build/artifacts,
      populates the local index and the cloud (ClickHouse) with them, then walks
      every component and simulates a source change to it -- the changed artifact
      (and, with --propagate, its topological consumers) gets a fresh fingerprint
      and should MISS while everything else HITs. Measures hit rate, lookup
      latency local vs. cloud, and the invalidation blast radius per component.

  build     (real; you run this locally -- it compiles)
      Cold baseline build (timed) -> publish fingerprints -> warm rerun (timed,
      expect ~100% hit) -> for each component: commit a trivial source change,
      reconfigure (+ optionally rebuild, timed), look up, record, revert.

Local vs. cloud: "local" is a JSON index on disk (apples-to-apples content with
the cloud, so the comparison isolates file-vs-network latency). "cloud" is the
team ClickHouse index via build_tools/github_actions/artifact_cache. ClickHouse
credentials come from the environment (CLICKHOUSE_URL/_USER/_PASSWORD); if unset,
the cloud columns are skipped and only the local measurements run.

Gaps the report highlights:
  * uncacheable artifacts -- a component directory under build/artifacts with no
    .fprint sidecar (split/per-arch artifacts today emit none);
  * false misses -- a fingerprint present in the index that a lookup failed to
    resolve (would indicate an indexing/upload bug);
  * invalidation blast radius -- how many artifacts a single component change
    forces to rebuild.

Examples (run from the repository root):
    # fast analysis over an existing build's fingerprints
    python build_tools/benchmarks/fingerprint_cache_benchmark.py simulate \
        --build build --platform linux --build-variant release \
        --output build/fingerprint_cache_report

    # real cold+warm+per-component timing sweep
    python build_tools/benchmarks/fingerprint_cache_benchmark.py build \
        --build build --families gfx110X-all \
        --components build_tools/benchmarks/components.example.json \
        --output build/fingerprint_cache_report
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_HEX = set("0123456789abcdef")
FPRINT_SUFFIX = ".fprint"


# --------------------------------------------------------------------------- #
# Reading what a build produced
# --------------------------------------------------------------------------- #
@dataclass
class BuildInventory:
    """What a build left under build/artifacts: fingerprints + the cache gap."""

    fingerprints: dict  # artifact_identity -> fingerprint (64-hex)
    uncacheable: list  # component dirs with no .fprint (split/per-arch gap)

    @property
    def cacheable(self) -> int:
        return len(self.fingerprints)

    @property
    def total_components(self) -> int:
        return self.cacheable + len(self.uncacheable)


def read_inventory(build_dir: Path) -> BuildInventory:
    artifacts = Path(build_dir) / "artifacts"
    if not artifacts.is_dir():
        raise SystemExit(
            f"no artifacts directory under {build_dir} (did the build run?)"
        )
    fingerprints: dict = {}
    for path in sorted(artifacts.glob("*" + FPRINT_SUFFIX)):
        identity = path.name[: -len(FPRINT_SUFFIX)]
        value = path.read_text(encoding="utf-8").strip()
        if len(value) == 64 and set(value) <= _HEX:
            fingerprints[identity] = value
    # A component is a directory of outputs; if it has no .fprint it cannot be
    # cached today (the split/per-arch gap).
    component_dirs = {p.name for p in artifacts.iterdir() if p.is_dir()}
    uncacheable = sorted(d for d in component_dirs if d not in fingerprints)
    return BuildInventory(fingerprints=fingerprints, uncacheable=uncacheable)


# --------------------------------------------------------------------------- #
# Local index (JSON on disk) -- the "local" cache backend
# --------------------------------------------------------------------------- #
class LocalIndex:
    """A file-backed fingerprint index, same content model as the cloud index."""

    def __init__(self, path: Path, platform: str, build_variant: str):
        self.path = Path(path)
        self.platform = platform
        self.build_variant = build_variant
        self._keys: set = set()
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._keys = {tuple(k) for k in data.get("keys", [])}

    def _key(self, artifact: str, fingerprint: str) -> tuple:
        return (self.platform, self.build_variant, artifact, fingerprint)

    def publish(self, fingerprints: dict) -> None:
        for artifact, fingerprint in fingerprints.items():
            self._keys.add(self._key(artifact, fingerprint))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"keys": sorted(self._keys)}, indent=0), encoding="utf-8"
        )

    def lookup(self, fingerprints: dict) -> tuple:
        """Return ({artifact: hit_bool}, elapsed_ms) for a batch of fingerprints."""
        start = time.perf_counter()
        hits = {
            artifact: self._key(artifact, fingerprint) in self._keys
            for artifact, fingerprint in fingerprints.items()
        }
        return hits, (time.perf_counter() - start) * 1000.0


# --------------------------------------------------------------------------- #
# Cloud index (ClickHouse) -- optional, via the branch package
# --------------------------------------------------------------------------- #
class CloudIndex:
    """Self-contained ClickHouse index over HTTP (no external deps).

    Credentials come from CLICKHOUSE_URL / CLICKHOUSE_USER / CLICKHOUSE_PASSWORD.
    Uses its own ReplacingMergeTree table so it never touches production data.
    """

    _TABLE = "fingerprint_benchmark"

    def __init__(self, platform: str, build_variant: str):
        import urllib.request  # noqa: PLC0415 (self-contained on purpose)

        self._urllib = urllib.request
        self.url = os.environ["CLICKHOUSE_URL"].rstrip("/")
        self.user = os.environ.get("CLICKHOUSE_USER", "default")
        self.password = os.environ.get("CLICKHOUSE_PASSWORD", "")
        self.platform = platform
        self.build_variant = build_variant
        self._post(
            f"CREATE TABLE IF NOT EXISTS {self._TABLE} ("
            "platform String, build_variant String, artifact String, "
            "fingerprint String, run_label String, "
            "version UInt64) ENGINE = ReplacingMergeTree(version) "
            "ORDER BY (platform, build_variant, artifact, fingerprint)"
        )

    def _post(self, sql: str) -> str:
        import base64

        token = base64.b64encode(f"{self.user}:{self.password}".encode()).decode()
        last = None
        for attempt in range(5):  # tolerate serverless wake / transient errors
            request = self._urllib.Request(self.url + "/", data=sql.encode("utf-8"))
            request.add_header("Authorization", f"Basic {token}")
            try:
                with self._urllib.urlopen(request, timeout=30) as response:
                    return response.read().decode("utf-8")
            except Exception as exc:  # noqa: BLE001 (best-effort, cloud optional)
                last = exc
                time.sleep(min(2**attempt, 10))
        raise RuntimeError(f"ClickHouse request failed: {last}")

    def publish(self, fingerprints: dict, run_label: str) -> None:
        if not fingerprints:
            return
        version = int(time.time() * 1000)
        rows = ",".join(
            "('{}','{}','{}','{}','{}',{})".format(
                self.platform, self.build_variant, a, f, run_label, version
            )
            for a, f in fingerprints.items()
        )
        self._post(
            f"INSERT INTO {self._TABLE} "
            "(platform,build_variant,artifact,fingerprint,run_label,version) "
            f"VALUES {rows}"
        )

    def lookup(self, fingerprints: dict) -> tuple:
        fps = ",".join("'" + f + "'" for f in set(fingerprints.values()))
        sql = (
            f"SELECT artifact, fingerprint FROM {self._TABLE} FINAL "
            f"WHERE platform='{self.platform}' AND build_variant='{self.build_variant}' "
            f"AND fingerprint IN ({fps}) FORMAT TabSeparated"
        )
        start = time.perf_counter()
        text = self._post(sql)
        elapsed = (time.perf_counter() - start) * 1000.0
        present = set()
        for line in text.splitlines():
            if "\t" in line:
                artifact, fingerprint = line.split("\t", 1)
                present.add((artifact, fingerprint))
        hits = {
            artifact: (artifact, fingerprint) in present
            for artifact, fingerprint in fingerprints.items()
        }
        return hits, elapsed


# --------------------------------------------------------------------------- #
# Scenario record + report
# --------------------------------------------------------------------------- #
@dataclass
class Scenario:
    name: str
    changed: str | None = None
    configure_seconds: float | None = None
    build_seconds: float | None = None
    cacheable: int = 0
    uncacheable: int = 0
    local_hits: int = 0
    cloud_hits: int | None = None
    local_lookup_ms: float = 0.0
    cloud_lookup_ms: float | None = None
    misses: list = field(default_factory=list)
    false_misses: list = field(default_factory=list)

    def hit_rate(self, hits: int | None) -> str:
        if hits is None or self.cacheable == 0:
            return "-"
        return f"{100.0 * hits / self.cacheable:.0f}%"


def _mutate(fingerprint: str) -> str:
    """Deterministic 'changed source' fingerprint (stays 64-hex)."""
    import hashlib

    return hashlib.sha256((fingerprint + ":changed").encode()).hexdigest()


def _evaluate(
    fingerprints: dict,
    local: LocalIndex,
    cloud: CloudIndex | None,
    inventory: BuildInventory,
    name: str,
    changed: str | None = None,
    configure_seconds: float | None = None,
    build_seconds: float | None = None,
) -> Scenario:
    local_hits, local_ms = local.lookup(fingerprints)
    scenario = Scenario(
        name=name,
        changed=changed,
        configure_seconds=configure_seconds,
        build_seconds=build_seconds,
        cacheable=len(fingerprints),
        uncacheable=len(inventory.uncacheable),
        local_hits=sum(local_hits.values()),
        local_lookup_ms=round(local_ms, 2),
        misses=sorted(a for a, hit in local_hits.items() if not hit),
    )
    if cloud is not None:
        cloud_hits, cloud_ms = cloud.lookup(fingerprints)
        scenario.cloud_hits = sum(cloud_hits.values())
        scenario.cloud_lookup_ms = round(cloud_ms, 2)
        # Disagreement between backends on the same content is a real finding.
        scenario.false_misses = sorted(
            a for a in fingerprints if local_hits.get(a) != cloud_hits.get(a)
        )
    return scenario


def render_report(scenarios: list, inventory: BuildInventory, meta: dict) -> str:
    lines = [
        "# Fingerprint cache benchmark",
        "",
        "> The DB (ClickHouse) is the index of record: a hit means a *previous* build already\n"
        "> produced that exact fingerprint, so it can be reused. The local JSON column is gone;\n"
        "> it only mirrored this run's own inserts. Local lookup time is kept below as a latency\n"
        "> reference.\n",
        "",
        f"- Mode: **{meta['mode']}**  |  platform: `{meta['platform']}`  |  "
        f"variant: `{meta['build_variant']}`  |  cloud: "
        f"`{'ClickHouse' if meta['cloud'] else 'disabled'}`",
        f"- Build components: **{inventory.total_components}** "
        f"(**{inventory.cacheable} cacheable**, "
        f"**{len(inventory.uncacheable)} uncacheable** — no `.fprint`)",
        "",
        "## Results",
        "",
        "| Scenario | Changed | Configure | Build | Cacheable | "
        "DB hit (ClickHouse) | DB lookup | Missed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in scenarios:
        cfg = f"{s.configure_seconds:.1f}s" if s.configure_seconds is not None else "-"
        bld = f"{s.build_seconds:.1f}s" if s.build_seconds is not None else "-"
        cloud_hit = s.hit_rate(s.cloud_hits)
        cloud_ms = f"{s.cloud_lookup_ms}ms" if s.cloud_lookup_ms is not None else "-"
        lines.append(
            f"| {s.name} | {s.changed or '-'} | {cfg} | {bld} | {s.cacheable} | "
            f"{cloud_hit} | {cloud_ms} | {len(s.misses)} |"
        )

    # Local vs cloud latency summary
    local_ms = [s.local_lookup_ms for s in scenarios]
    cloud_ms = [s.cloud_lookup_ms for s in scenarios if s.cloud_lookup_ms is not None]
    lines += ["", "## Local vs. cloud lookup latency", ""]
    lines.append(
        f"- Local (JSON on disk): mean **{statistics.mean(local_ms):.2f} ms**, "
        f"max {max(local_ms):.2f} ms over {len(local_ms)} lookups."
    )
    if cloud_ms:
        lines.append(
            f"- Cloud (ClickHouse): mean **{statistics.mean(cloud_ms):.1f} ms**, "
            f"max {max(cloud_ms):.1f} ms over {len(cloud_ms)} lookups."
        )
        factor = statistics.mean(cloud_ms) / max(statistics.mean(local_ms), 1e-6)
        lines.append(f"- Cloud lookups are ~**{factor:.0f}x** the local latency.")

    # Gaps
    lines += ["", "## Gaps (where the cache is missing)", ""]
    if inventory.uncacheable:
        lines.append(
            f"**{len(inventory.uncacheable)} uncacheable artifacts** — a build "
            "output directory with no `.fprint`. These are the split/per-arch "
            "artifacts and cannot be reused today:"
        )
        lines.append("")
        for name in inventory.uncacheable[:40]:
            lines.append(f"- `{name}`")
        if len(inventory.uncacheable) > 40:
            lines.append(f"- … and {len(inventory.uncacheable) - 40} more")
    else:
        lines.append("No uncacheable artifacts: every component emitted a `.fprint`.")
    false = sorted({a for s in scenarios for a in s.false_misses})
    lines += ["", "### Local/cloud disagreements (false misses)", ""]
    lines.append(
        "None — local and cloud agreed on every lookup."
        if not false
        else "These artifacts resolved differently local vs. cloud (indexing gap): "
        + ", ".join(f"`{a}`" for a in false[:30])
    )

    # Blast radius (per-component change scenarios)
    changes = [s for s in scenarios if s.changed]
    if changes:
        lines += ["", "## Invalidation blast radius (per component change)", ""]
        lines.append("| Changed component | Artifacts forced to rebuild |")
        lines.append("| --- | --- |")
        for s in sorted(changes, key=lambda x: -len(x.misses)):
            lines.append(f"| `{s.changed}` | {len(s.misses)} |")

    lines += [
        "",
        "## Notes",
        "",
        f"- Fingerprint-stability PRs applied in this checkout: {meta['stability']}.",
        "- `simulate` mode measures cache behavior over an existing build's "
        "fingerprints; it does not rebuild. `build` mode reports real timings.",
        "",
    ]
    return "\n".join(lines)


def _inline(text: str) -> str:
    """Escape, then render `code` and **bold** inline markdown."""
    import re

    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    return text


def write_html(markdown_path: Path, html_path: Path) -> None:
    """Render the report markdown (headers, tables, lists) to standalone HTML."""
    out: list = []
    table: list = []

    def flush_table() -> None:
        if not table:
            return
        rows = [r for r in table if set(r.strip()) - set("|-: ")]
        out.append("<table>")
        for i, row in enumerate(rows):
            cells = [c.strip() for c in row.strip().strip("|").split("|")]
            tag = "th" if i == 0 else "td"
            out.append(
                "<tr>"
                + "".join(f"<{tag}>{_inline(c)}</{tag}>" for c in cells)
                + "</tr>"
            )
        out.append("</table>")
        table.clear()

    for line in markdown_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("|"):
            table.append(line)
            continue
        flush_table()
        if line.startswith("### "):
            out.append(f"<h3>{_inline(line[4:])}</h3>")
        elif line.startswith("## "):
            out.append(f"<h2>{_inline(line[3:])}</h2>")
        elif line.startswith("# "):
            out.append(f"<h1>{_inline(line[2:])}</h1>")
        elif line.startswith("- "):
            out.append(f"<li>{_inline(line[2:])}</li>")
        elif line.strip():
            out.append(f"<p>{_inline(line)}</p>")
    flush_table()
    html = (
        "<!doctype html><meta charset='utf-8'><title>Fingerprint cache benchmark</title>"
        "<style>:root{color-scheme:light}body{font:14px/1.6 system-ui,sans-serif;"
        "max-width:64rem;margin:2rem auto;padding:0 1rem;color:#111}"
        "table{border-collapse:collapse;margin:1rem 0}td,th{border:1px solid #ccc;"
        "padding:4px 10px;text-align:left}th{background:#f4f4f4}"
        "code{background:#f4f4f4;padding:1px 4px;border-radius:3px}"
        "h1{border-bottom:2px solid #eee}li{margin:2px 0}</style>" + "\n".join(out)
    )
    html_path.write_text(html, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Build driving (build mode)
# --------------------------------------------------------------------------- #
def _timed(cmd: list, cwd: Path | None = None) -> float:
    start = time.perf_counter()
    subprocess.run(cmd, cwd=cwd, check=True)
    return time.perf_counter() - start


def detect_stability_prs(repo_root: Path) -> str:
    """Best-effort: report which fingerprint-stability PRs appear applied."""
    marks = []
    subproj = repo_root / "cmake" / "therock_subproject.cmake"
    text = subproj.read_text(encoding="utf-8") if subproj.exists() else ""
    if "RELATIVE_PATH" in text and "build-dir" not in text:
        pass
    if (repo_root / "cmake" / "tests" / "therock_source_identity_test.py").exists():
        marks.append("#9907 (git-tree scoping)")
    if (
        repo_root / "cmake" / "tests" / "therock_subproject_fingerprint_test.py"
    ).exists():
        marks.append("#9060 (build-dir determinism)")
    return ", ".join(marks) if marks else "unknown (none detected)"


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def discover_components(repo_root: Path, limit: int) -> list:
    """Auto-discover components to change: one tracked file per git submodule."""
    result = subprocess.run(
        ["git", "submodule", "--quiet", "foreach", "echo $sm_path"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    paths = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    components = []
    for path in paths:
        source_file = _pick_source_file(repo_root / path)
        if source_file is not None:
            components.append(
                {
                    "name": Path(path).name,
                    "source_dir": path,
                    "source_file": source_file,
                }
            )
        if limit and len(components) >= limit:
            break
    return components


def _pick_source_file(source_dir: Path) -> str | None:
    """A small tracked text file in a subproject to perturb (README/CMakeLists)."""
    if not (source_dir / ".git").exists() and not source_dir.is_dir():
        return None
    for candidate in ("README.md", "README", "CMakeLists.txt", "LICENSE"):
        if (source_dir / candidate).is_file():
            return candidate
    for path in sorted(source_dir.glob("*.txt")):
        return path.name
    return None


def run_simulate(args) -> None:
    inventory = read_inventory(Path(args.build))
    if not inventory.fingerprints:
        raise SystemExit("no .fprint files found; run a configure/build first")
    local = LocalIndex(Path(args.local_index), args.platform, args.build_variant)
    cloud = _maybe_cloud(args)

    # Baseline: this build's fingerprints become the cache.
    run_label = args.run_id
    local.publish(inventory.fingerprints)
    if cloud:
        cloud.publish(inventory.fingerprints, run_label)

    scenarios = [
        _evaluate(
            inventory.fingerprints,
            local,
            cloud,
            inventory,
            name="baseline / warm rerun (no change)",
        )
    ]

    # Per-component sweep: change each one, expect it to miss and the rest to hit.
    for artifact in sorted(inventory.fingerprints):
        changed = dict(inventory.fingerprints)
        changed[artifact] = _mutate(changed[artifact])
        scenarios.append(
            _evaluate(
                changed,
                local,
                cloud,
                inventory,
                name=f"change: {artifact}",
                changed=artifact,
            )
        )

    _emit(scenarios, inventory, args, mode="simulate", cloud=cloud is not None)


def discover_source_repos(repo_root: Path, limit: int) -> list:
    """Find subproject source repos to perturb (first-level .git worktrees).

    TheRock fetches sources with fetch_sources.py, not top-level submodules, so
    we locate each subproject's git worktree directly and prune once found.
    """
    repos = []
    skip = {"build", ".git", ".ccache", "__pycache__", ".github", "third-party"}
    for dirpath, dirnames, filenames in os.walk(repo_root):
        current = Path(dirpath)
        if current == repo_root:
            dirnames[:] = [d for d in dirnames if d not in skip]
            continue
        if ".git" in dirnames or ".git" in filenames:
            repos.append(current)
            dirnames[:] = []  # a subproject repo: do not descend further
            if limit and len(repos) >= limit:
                break
    return sorted(repos)


def _delta(base: dict, now: dict) -> list:
    """Artifacts whose fingerprint changed (or appeared/disappeared)."""
    return sorted(a for a in set(base) | set(now) if base.get(a) != now.get(a))


def run_verify(args) -> None:
    """Configure-only sweep: verify fingerprints are stable and invalidate correctly.

    No builds. Configure writes the .fprint sidecars, so one configure per change
    is all that is needed to see which fingerprints move. Steps:
      1. configure baseline, capture fingerprints;
      2. configure again with no change -> any fingerprint that moves is a
         reproducibility gap (the thing that must be fixed before reuse is safe);
      3. for each discovered subproject: commit a trivial change, reconfigure,
         diff the fingerprints (= invalidation blast radius), revert.
    """
    repo_root = Path.cwd()
    build_dir = Path(args.build)
    configure_cmd = [
        "cmake",
        "-B",
        str(build_dir),
        "-GNinja",
        ".",
        f"-DTHEROCK_AMDGPU_FAMILIES={args.families}",
    ]
    local = LocalIndex(Path(args.local_index), args.platform, args.build_variant)
    cloud = _maybe_cloud(args)

    # 1. Baseline configure.
    cfg = _timed(configure_cmd)
    inventory = read_inventory(build_dir)
    base = inventory.fingerprints
    local.publish(base)
    if cloud:
        cloud.publish(base, args.run_id)
    scenarios = [
        _evaluate(
            base,
            local,
            cloud,
            inventory,
            name="baseline (configure)",
            configure_seconds=cfg,
        )
    ]
    _emit(scenarios, inventory, args, mode="verify", cloud=cloud is not None)

    # 2. Stability: configure again, nothing changed -> fingerprints must match.
    cfg = _timed(configure_cmd)
    now = read_inventory(build_dir).fingerprints
    unstable = _delta(base, now)
    stability = Scenario(
        name="stability: reconfigure, no change",
        configure_seconds=cfg,
        cacheable=len(now),
        local_hits=len(now) - len(unstable),
        misses=unstable,  # should be empty; anything here is a reproducibility gap
    )
    scenarios.append(stability)
    _emit(scenarios, inventory, args, mode="verify", cloud=cloud is not None)

    # 3. Per-component invalidation sweep (configure only).
    repos = discover_source_repos(repo_root, args.max_components)
    print(f"[benchmark] discovered {len(repos)} subproject repos to perturb")
    for repo in repos:
        name = repo.relative_to(repo_root).as_posix()
        source_file = _pick_source_file(repo)
        if source_file is None:
            print(f"[benchmark] skip {name}: no perturbable file")
            continue
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if not head:
            print(f"[benchmark] skip {name}: not a git worktree")
            continue
        try:
            target = repo / source_file
            target.write_text(
                target.read_text() + f"\n# fingerprint benchmark {name}\n"
            )
            subprocess.run(["git", "-C", str(repo), "add", source_file], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "-c",
                    "user.email=b@b",
                    "-c",
                    "user.name=b",
                    "commit",
                    "-q",
                    "-m",
                    "fingerprint benchmark change",
                ],
                check=True,
            )
            cfg = _timed(configure_cmd)
            changed_inv = read_inventory(build_dir)
            changed = _delta(base, changed_inv.fingerprints)
            scenario = _evaluate(
                changed_inv.fingerprints,
                local,
                cloud,
                changed_inv,
                name=f"change: {name}",
                changed=name,
                configure_seconds=cfg,
            )
            # The invalidation blast radius is the fingerprint diff, not just the
            # lookup miss (identical semantics here, but record the diff directly).
            scenario.misses = changed
            scenarios.append(scenario)
            _emit(scenarios, inventory, args, mode="verify", cloud=cloud is not None)
        finally:
            subprocess.run(["git", "-C", str(repo), "reset", "-q", "--hard", head])


def _order_by_fingerprint_yield(repo_root, repos, build_dir):
    """Order components so those that actually produce a .fprint are swept first.

    Only components whose change moves a fingerprint are worth a (re)build, so we
    rank by how many of the baseline's cacheable artifacts a subproject plausibly
    produces: an exact/prefix name match, or -- for the monorepos that hold many
    subprojects -- a nonzero fixed weight. Components that match nothing sort last
    and are easy to cap away with --max-components.
    """
    cacheable = set(read_inventory(build_dir).fingerprints)
    names = {a.split("_")[0] for a in cacheable}  # artifact name prefixes

    def norm(text):
        return text.replace("-", "").replace("_", "").lower()

    def score(repo):
        base = norm(repo.name)
        matches = sum(
            1 for n in names if norm(n).startswith(base) or base.startswith(norm(n))
        )
        is_monorepo = repo.name in ("rocm-libraries", "rocm-systems")
        # Bucket: clean per-component subprojects that produce a fingerprint
        # first (fast, single-component signal), the coarse+slow monorepos next,
        # and components that map to nothing last.
        if not is_monorepo and matches:
            bucket = 2
        elif is_monorepo:
            bucket = 1
        else:
            bucket = 0
        return (bucket, matches)

    ranked = sorted(repos, key=score, reverse=True)
    return [
        {
            "name": r.relative_to(repo_root).as_posix(),
            "source_dir": r.relative_to(repo_root).as_posix(),
        }
        for r in ranked
    ]


def run_build(args) -> None:
    repo_root = Path.cwd()
    build_dir = Path(args.build)
    families = args.families
    local = LocalIndex(Path(args.local_index), args.platform, args.build_variant)
    cloud = _maybe_cloud(args)
    if args.components:
        components = json.loads(Path(args.components).read_text())
    else:
        repos = discover_source_repos(repo_root, 0)
        components = _order_by_fingerprint_yield(repo_root, repos, build_dir)
        if args.max_components:
            components = components[: args.max_components]
    print(f"[benchmark] sweeping {len(components)} components", flush=True)

    configure_cmd = [
        "cmake",
        "-B",
        str(build_dir),
        "-GNinja",
        ".",
        f"-DTHEROCK_AMDGPU_FAMILIES={families}",
    ]
    build_cmd = ["cmake", "--build", str(build_dir)]
    if args.build_target:
        build_cmd += ["--target", args.build_target]
    scenarios = []

    # Cold baseline.
    cfg = _timed(configure_cmd)
    bld = _timed(build_cmd) if not args.configure_only else None
    inventory = read_inventory(build_dir)
    local.publish(inventory.fingerprints)
    if cloud:
        cloud.publish(inventory.fingerprints, args.run_id)
    scenarios.append(
        _evaluate(
            inventory.fingerprints,
            local,
            cloud,
            inventory,
            name="baseline (cold)",
            configure_seconds=cfg,
            build_seconds=bld,
        )
    )

    # Warm rerun (no change) -> expect all hits + fast build.
    cfg = _timed(configure_cmd)
    bld = _timed(build_cmd) if not args.configure_only else None
    inv2 = read_inventory(build_dir)
    scenarios.append(
        _evaluate(
            inv2.fingerprints,
            local,
            cloud,
            inv2,
            name="warm (no change)",
            configure_seconds=cfg,
            build_seconds=bld,
        )
    )

    # Per-component: commit a change, reconfigure (+build), look up, revert.
    for component in components:
        name = component["name"]
        src = repo_root / component["source_dir"]
        source_file = component.get("source_file") or _pick_source_file(src)
        if source_file is None or not src.is_dir():
            print(f"[benchmark] skip {name}: no perturbable source file", flush=True)
            continue
        print(f"[benchmark] change: {name}", flush=True)
        head = subprocess.run(
            ["git", "-C", str(src), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        try:
            target = src / source_file
            target.write_text(target.read_text() + f"\n// benchmark change {name}\n")
            subprocess.run(["git", "-C", str(src), "add", source_file], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(src),
                    "-c",
                    "user.email=benchmark@local",
                    "-c",
                    "user.name=benchmark",
                    "commit",
                    "-q",
                    "-m",
                    f"benchmark: {name}",
                ],
                check=True,
            )
            cfg = _timed(configure_cmd)
            bld = _timed(build_cmd) if not args.configure_only else None
            inv = read_inventory(build_dir)
            scenarios.append(
                _evaluate(
                    inv.fingerprints,
                    local,
                    cloud,
                    inv,
                    name=f"change: {name}",
                    changed=name,
                    configure_seconds=cfg,
                    build_seconds=bld,
                )
            )
            local.publish(inv.fingerprints)
            if cloud:
                cloud.publish(inv.fingerprints, args.run_id)
        finally:
            subprocess.run(
                ["git", "-C", str(src), "reset", "-q", "--hard", head], check=True
            )

    _emit(scenarios, inventory, args, mode="build", cloud=cloud is not None)


def _maybe_cloud(args):
    if args.no_cloud or not os.environ.get("CLICKHOUSE_URL"):
        return None
    try:
        return CloudIndex(args.platform, args.build_variant)
    except Exception as exc:  # cloud is optional; never fail the benchmark on it
        print(f"[benchmark] cloud disabled ({exc})", file=sys.stderr)
        return None


def _emit(scenarios, inventory, args, mode, cloud):
    meta = {
        "mode": mode,
        "platform": args.platform,
        "build_variant": args.build_variant,
        "cloud": cloud,
        "stability": detect_stability_prs(Path.cwd()),
    }
    report = render_report(scenarios, inventory, meta)
    out = Path(args.output)
    md, html = out.with_suffix(".md"), out.with_suffix(".html")
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(report, encoding="utf-8")
    write_html(md, html)
    print(report)
    print(f"\nWrote {md} and {html}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("simulate", "build", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--build", default="build", help="build directory")
        p.add_argument("--platform", default="linux")
        p.add_argument("--build-variant", default="release")
        p.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", "benchmark"))
        p.add_argument(
            "--local-index",
            default="build/benchmark-local-index.json",
            help="JSON file used as the local cache backend",
        )
        p.add_argument("--no-cloud", action="store_true", help="skip ClickHouse")
        p.add_argument(
            "--output",
            default="build/fingerprint_cache_report",
            help="report path without extension (.md and .html are written)",
        )
    for mode_name in ("build", "verify"):
        mp = sub.choices[mode_name]
        mp.add_argument("--families", default="gfx110X-all")
        mp.add_argument(
            "--max-components",
            type=int,
            default=0,
            help="cap components to change (0 = all discovered)",
        )
    build_p = sub.choices["build"]
    build_p.add_argument(
        "--build-target",
        default=None,
        help="scope the build to one target (e.g. amd-llvm+stage); default: build all",
    )
    build_p.add_argument(
        "--components", help="JSON list of {name, source_dir, source_file} to sweep"
    )
    build_p.add_argument(
        "--configure-only",
        action="store_true",
        help="configure only (produces .fprint) without compiling, for a fast run",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "simulate":
        run_simulate(args)
    elif args.command == "verify":
        run_verify(args)
    else:
        run_build(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
