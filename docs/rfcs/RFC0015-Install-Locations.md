---
author: Saad Rahim (saadrahim)
created: 2026-06-09
status: draft
---

# ROCm On-Disk Install Locations

## Related RFCs

- **RFC0012 — repo.amd.com folder hierarchy**
  ([ROCm/TheRock#4414](https://github.com/ROCm/TheRock/pull/4414))
  — defines the stream subdomains (`<stream>.repo.amd.com`), the
  per-stream folder tree, and *Concurrent stream installation*. This
  RFC owns the on-disk install paths those streams resolve to; it was
  factored out of RFC0012 so the disk layout has a single home.
- **RFC0009 — TheRock Software Packaging Requirements** — source of
  the hyphenated `/opt/rocm/core-<ver>` directory-layout convention and
  the `core/` / `core-<major>` symlink policy referenced below.
- **RFC0013 — End-user projects / extras** — owns the
  `/opt/rocm/extras/` install model that this RFC defers to for extras.

## Install Locations (ROCm Core SDK)

Native packages install the ROCm Core SDK into a directory directly
under `/opt/rocm/`, following the hyphenated `/opt/rocm/core-<ver>`
naming convention defined in RFC0009 (*TheRock Software Packaging
Requirements* — Directory Layout).

`stable` and `lts` (and their `rc` / `ltsrc` candidates and ASAN
siblings) are placed the same way under both options — at their
released `<X.Y>` / `<YYYY.MM>` paths. **The two options below differ
only in where the pre-release development streams (`dev`, `nightly`,
`weekly`) land on disk**, and as a consequence whether a pre-release
build of an upcoming release can coexist with anything else.

This RFC presents both and does not yet pick a winner; **Option B is
the current state** and is a reasonable choice if per-build on-disk
coexistence is not required.

### Option A — Per-build coexistence (stream-scoped, version-stamped paths)

Each stream gets a **distinct, version-scoped** directory so multiple
streams — and multiple builds within a stream — coexist on the same
machine without colliding (see *Concurrent stream installation* in
RFC0012 Stream Subdomains).

| Stream    | Variant    | Install location                              | Example                                     |
| :-------- | :--------- | :-------------------------------------------- | :------------------------------------------ |
| `dev`     | standard   | `/opt/rocm/core-dev-<YYYYMMDD-sha>`           | `/opt/rocm/core-dev-20260602-a3f1c9`        |
| `nightly` | standard   | `/opt/rocm/core-<YYYYMMDD>`                   | `/opt/rocm/core-20260602`                   |
| `weekly`  | standard   | `/opt/rocm/core-<YYYY>W<WW>`                  | `/opt/rocm/core-2026W23`                    |
| `rc`      | standard   | `/opt/rocm/core-<X.Y>rc<N>`                   | `/opt/rocm/core-7.12rc1`                    |
| `rc`      | asan       | `/opt/rocm/core-<X.Y>rc<N>-asan`              | `/opt/rocm/core-7.12rc1-asan`               |
| `stable`  | standard   | `/opt/rocm/core-<X.Y>`                        | `/opt/rocm/core-7.12`                       |
| `stable`  | asan       | `/opt/rocm/core-<X.Y>-asan`                   | `/opt/rocm/core-7.12-asan`                  |
| `ltsrc`   | standard   | `/opt/rocm/core-<YYYY.MM>rc<N>`               | `/opt/rocm/core-2026.09rc1`                 |
| `lts`     | standard   | `/opt/rocm/core-<YYYY.MM>`                    | `/opt/rocm/core-2026.09`                    |
| `lts`     | asan       | `/opt/rocm/core-<YYYY.MM>-asan`               | `/opt/rocm/core-2026.09-asan`               |

Under this option `dev`, `nightly`, `weekly`, and `rc` use distinct
version syntaxes from `stable`/`lts` so a glob (`ls /opt/rocm/core-*`)
makes the stream obvious without inspecting metadata. Every build is
retained on disk until explicitly removed, which is the property teams
that need to A/B multiple develop-branch snapshots depend on.

**Notes on the `weekly` version format (`YYYY'W'WW`):**

- ISO 8601 calendar-week notation: `YYYY` is the **ISO year**
  (which differs from the calendar year for a handful of dates in
  early January / late December — using the ISO year here prevents
  cross-year collisions like a phantom `2026W01` that actually maps
  to 2025).
- `WW` is the ISO week number, zero-padded, 01–53.
- ISO weeks start on Monday; the cut for `weekly` is Monday 00:00 UTC.
- The literal `W` separator visually distinguishes `weekly` paths
  (`core-2026W23`) from `nightly` paths (`core-20260602`) so a glob
  (`ls /opt/rocm/core-*`) makes the stream obvious without inspecting
  metadata — same guarantee the existing `dev`/`nightly`/`rc`/`stable`
  syntaxes already provide.

### Option B — Shared next-release location (current state)

`dev`, `nightly`, `weekly`, and `rc` all install into the **final
release directory of the version they are leading up to** — i.e. the
same `/opt/rocm/core-<X.Y>` path the eventual `stable` release will
occupy — where `<X.Y>` is the next ROCm Core release the develop
branch (for `dev`/`nightly`/`weekly`) or the candidate (for `rc`) is
targeting. For example, while 7.14 is in development every `dev`,
`nightly`, `weekly`, and `rc` build installs to `/opt/rocm/core-7.14`,
and when 7.14 ships `stable` lands at that same path.

| Stream                         | Install location          | Example                |
| :----------------------------- | :------------------------ | :--------------------- |
| `dev` / `nightly` / `weekly`   | `/opt/rocm/core-<X.Y>`    | `/opt/rocm/core-7.14`  |
| `rc`                           | `/opt/rocm/core-<X.Y>`    | `/opt/rocm/core-7.14`  |
| `stable`                       | `/opt/rocm/core-<X.Y>`    | `/opt/rocm/core-7.14`  |
| `lts`                          | `/opt/rocm/core-<YYYY.MM>`| `/opt/rocm/core-2026.09`|

Consequences:

- **No coexistence between a pre-release stream and `stable`** for a
  given version — they are literally the same directory. A box can
  hold the in-development 7.14 *or* released 7.14, never both at once;
  the final release simply overwrites the pre-release in place. (The
  same applies between `dev`/`nightly`/`weekly`/`rc` of one version
  and an already-installed *different* `stable` only insofar as they
  occupy different `<X.Y>` paths — distinct released majors/minors
  still coexist; what cannot coexist is the pre-release and final of
  the **same** `<X.Y>`.)
- **Successive pre-release builds overwrite in place.** There is no
  on-disk retention of multiple `dev`/`nightly`/`weekly` snapshots;
  each new build replaces the previous one inside `core-<X.Y>`. If the
  develop branch retargets to a new `<X.Y>`, the previous directory is
  removed by the package upgrade.
- **Simplest layout, matches today's behavior.** Pre-release paths are
  identical in shape to the stable scheme, so tooling that hard-codes
  `/opt/rocm/core-<X.Y>` works unchanged across the promotion from
  pre-release to release.

A variant of this option uses a `-dev` suffix
(`/opt/rocm/core-<X.Y>-dev`) to keep the in-development tree on a
distinct path from the eventual release while still allowing only one
pre-release build at a time; this restores pre-release/`stable`
coexistence for a single version at the cost of no longer matching the
final path exactly.

### Trade-offs

Option A maximizes on-disk coexistence and per-build retention at the
cost of more directories and stream-specific path syntaxes. Option B
is the simplest and is what ships today, but a machine cannot hold a
pre-release and the final release of the same `<X.Y>` simultaneously,
and develop-branch snapshots are not retained. Choose A if downstream
CI needs to pin and compare multiple concurrent develop-branch builds;
choose B if the prevailing need is a stable, predictable path that
matches the final release.

### Rules (apply to both options)

- The trailing version component is **always present** — there is no
  unversioned install directory. `/opt/rocm/core/` (with trailing
  slash) and `/opt/rocm/core-<major>` are reserved by RFC0009 as
  **symlinks** to the latest installed core (e.g. `/opt/rocm/core/
  → /opt/rocm/core-7.12` and `/opt/rocm/core-7 →
  /opt/rocm/core-7.12`); they are not this RFC's to define. This RFC
  only specifies the install directory name; the symlink policy stays
  owned by RFC0009.
- ASAN-instrumented builds are published only for `stable`, `rc`, and
  `lts` — the streams where memory-debug rebuilds are worth the build
  cost. `dev`, `nightly`, `weekly`, and `ltsrc` ship the standard
  variant only; users who need ASAN on a pre-release path should use
  the matching `rc` build. The `standard` packages come from
  `packages/<distro>/`, `asan` packages from `packages/<distro>-asan/`
  (per RFC0009 and the *Linux Distros* section of RFC0012); the two
  install to parallel sibling directories with the `-asan` suffix
  appended to the standard path so they coexist on disk. Future
  variants reserved by RFC0009 (e.g. `rpath`, `debug`) follow the same
  `-<variant>` suffix pattern when introduced.
- The `rc` and `ltsrc` directories are short-lived: they are removed
  by the stable/LTS package's post-install (or by the user's package
  manager when the matching final release supersedes them). Under
  Option B this removal is implicit — the final release overwrites the
  shared `core-<X.Y>` path directly.
- Extras install under a single `/opt/rocm/extras/` tree per RFC0013
  (ROCm major carried by the binary suffix, e.g. `bin/rvs7`) and are
  **not** stream-scoped on disk — the stream-scoped path applies to
  Core SDK only. (Extras coexist across streams and ROCm majors through
  their own versioning per RFC0013 §2.)
- Patch releases land in-place under the same directory (`stable`
  `7.12.1` overwrites `/opt/rocm/core-7.12`); they do not create a
  new path. This matches RFC0009's "patch versions must be in place
  within the existing X.Y folder" rule.
- **HPC SDK** follows the same hyphenated convention from RFC0009 and
  uses the `<X.Y>` version of its pinned ROCm Core SDK release per
  [ROCm/TheRock#5613](https://github.com/ROCm/TheRock/pull/5613):
  installs at `/opt/rocm/hpc-<X.Y>` (e.g. `/opt/rocm/hpc-7.14`) with
  `/opt/rocm/hpc/ → /opt/rocm/hpc-<latest>` as the convenience
  symlink. HPC SDK patch releases (`<X.Y>.N`, e.g. `7.14.1`) land
  in-place inside the matching `hpc-<X.Y>` directory.
