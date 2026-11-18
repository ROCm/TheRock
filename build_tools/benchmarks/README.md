# Fingerprint cache benchmark

Throwaway harness (not production) to evaluate the fingerprint-based CI artifact
cache (issue #8651) before it is productionized. It measures **cache hit rate**,
**lookup latency local vs. cloud**, and — the part management wants — **where the
cache misses and what cannot be cached at all**.

## Prerequisites

1. A configured/built TheRock so `build/artifacts/*.fprint` exist
   (`cmake -B build -GNinja . -DTHEROCK_AMDGPU_FAMILIES=gfx110X-all && cmake --build build`).
2. For the cloud comparison, ClickHouse credentials in the environment
   (`CLICKHOUSE_URL`, `CLICKHOUSE_USER`, `CLICKHOUSE_PASSWORD`). Omit them to run
   local-only.
3. Fingerprint-stability fixes applied, so fingerprints reproduce and hits are
   meaningful. Start from `main` (it already has merged **#9921** upload,
   **#9936**/**#9871** MLIR, llvm-project **#5411** hipcc), then add the two open
   stability PRs by **merge** (they contain merge commits, so `git cherry-pick`
   of the range fails -- `git merge` is clean and both stack):

   ```bash
   git checkout -b fprint-benchmark origin/main
   git fetch origin pull/9060/head:pr-9060 pull/9907/head:pr-9907
   git merge --no-edit pr-9060   # build-dir determinism
   git merge --no-edit pr-9907   # git-tree source scoping (per-subdir, not whole monorepo)
   ```

   #9907 changes how fingerprints are computed, so after adding it you must
   rebuild (new fingerprints) and let the benchmark re-publish the baseline. The
   report prints which stability PRs it detects.

## Modes

### `verify` (default — configure only, NO builds; run this)

Configure writes the `.fprint` sidecars, so verifying fingerprints needs only
`cmake -B build` (seconds), never a compile. This mode:

1. configures a baseline and captures every fingerprint;
2. **configures again with no change** — any fingerprint that moves is a
   reproducibility gap (what must be fixed before reuse is safe, e.g. HIPCC);
3. for each discovered subproject: commits a trivial source change, reconfigures,
   diffs the fingerprints (= **invalidation blast radius**), then reverts.

```bash
python3 build_tools/benchmarks/fingerprint_cache_benchmark.py verify \
    --build build --families gfx110X-all \
    --output build/fingerprint_cache_report
```

Subprojects are found automatically (their `.git` worktrees — no `.gitmodules`
needed). Runs in minutes. The report's **stability** row and per-component
**blast radius** are what you use to draft topology-based reuse rules.

## Build / simulate modes

### `simulate` (fast, no builds — run this first)

Reads the fingerprints an existing build produced, populates the local index and
ClickHouse, then changes each component one at a time (the changed one should
miss, the rest hit) and measures hit rate and local-vs-cloud latency:

```bash
python build_tools/benchmarks/fingerprint_cache_benchmark.py simulate \
    --build build --output build/fingerprint_cache_report
```

Runs in seconds. Produces the full hit-rate / latency / gaps report without
compiling anything.

### `build` (real timings — you run this locally; it compiles)

Cold baseline build → warm rerun → per-component: commit a trivial source
change, reconfigure (+ rebuild), measure real time and hit rate, revert:

```bash
python build_tools/benchmarks/fingerprint_cache_benchmark.py build \
    --build build --families gfx110X-all \
    --components build_tools/benchmarks/components.example.json \
    --output build/fingerprint_cache_report
```

Each component in the JSON is `{name, source_dir, source_file}`; the harness
commits a one-line change in `source_dir`, measures, then `git reset --hard`s it
back. Use `--configure-only` to skip compilation (fingerprints are written at
configure time) for a faster real run. Point `--components` at the real
subproject paths in your checkout.

## Output

`fingerprint_cache_report.md` and `.html` (open the HTML for management). Sections:

- **Results** — per scenario: configure/build time, cacheable count, local & cloud
  hit rate, local & cloud lookup latency, number missed.
- **Local vs. cloud latency** — file lookup vs. ClickHouse round-trip.
- **Gaps** — uncacheable artifacts (a `build/artifacts/<name>/` dir with no
  `.fprint`: the split/per-arch artifacts) and any local/cloud disagreements.
- **Invalidation blast radius** — how many artifacts each component change forces
  to rebuild.

## Interpreting

- High hit rate on the warm rerun = the cache reuses unchanged work.
- Uncacheable count = the split/per-arch gap (today these never cache).
- A component that misses when its source did **not** change = a reproducibility
  gap (the stability PRs above target these).
- Cloud latency >> local = the per-lookup network cost to weigh against batching.
