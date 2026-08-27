---
author: Peter McLean (PeterCDMcLean)
created: 2026-08-27
modified: 2026-08-27
status: draft
---

# RFC0014: Subproject Fingerprint Coverage and Portability

TheRock computes `THEROCK_FPRINT` for each subproject. The digest covers the files, source
revisions and configuration values used to build it. No code in this repository reads
the generated `.fprint` files; the value propagates through the dependency graph and terminates in
files nothing opens. This RFC makes the fingerprint cover the inputs it claims to cover, makes it
comparable across machines, uses it as the key that authorizes artifact reuse, and publishes the
covered input set as a generated, drift-checked manifest so CI can resolve a changed path to the
subprojects it affects.

All line numbers cite TheRock at `9071e093` unless stated otherwise.

## Summary

### Subproject inputs

- **INPUTS-1** Emit the subproject inputs as a committed, drift-checked manifest keyed by
  platform configuration. The manifest is generated from the same list that computes the
  fingerprint, so it reports coverage rather than deciding it.
- **INPUTS-2** Emit a per-configure fingerprint dump carrying every input's label and
  digest, so a mismatch can be explained.

### Narrowing

- **FP-1** Fingerprint the git subtree a subproject consumes, and scope the dirty check to
  the same path, instead of the enclosing submodule's `HEAD`.
- **FP-2** Key fingerprint entries on repo-relative labels instead of basenames.
- **FP-3** Fingerprint a canonical record of the whole configure invocation: ordered
  `_cmake_args`, build environment arguments, generator and CMake version.
- **FP-4** Fingerprint source trees reached only through configure arguments, discovered by
  a lint that reports candidates for declaration.
- **SMREV-1** Keep `.smrev` precedence, unify its two writer formats, and give the
  ExternalProject writer a real identity: URL, `URL_HASH` and a patch-directory digest.

### CI improvement

- **DRIFT-1** Extract the consumer-graph drift checker into shared machinery, convert the
  existing checker to it, and add the input set check to the job that already configures the
  full tree.
- **CI-1** Expand changed gitlinks into submodule-relative paths, then resolve those
  through the subproject inputs file to stages.

### Bootstrap prebuilt

- **PRE-1** Carry one `.fprint` record inside each artifact archive, listing one fingerprint per
  subproject in the slice.
- **PRE-2** Write a single subproject's fingerprint into its `stage.prebuilt` marker, at
  the path `prebuilt_marker_relpath()` derives.
- **PRE-3** Compute the subproject input record before the marker is tested, so producer
  and consumer hash the same record.
- **PRE-4** Validate the marker at configure time against a defined state table, and
  quarantine the imported stage tree whenever the build falls back to source.
- **PRE-5** Accumulate component coverage in the marker across unpacks, and fail configure
  when two artifacts claim different fingerprints for one subproject.

### Portable fingerprints

- **PORT-1** Drop compiler launchers from the toolchain contribution.
- **PORT-2** Rebase build-tree absolute paths onto `THEROCK_BINARY_DIR`.
- **PORT-3** Key toolchain identity by provenance: a `BOOTSTRAP_ROOT` digest over a pinned image
  roots `rocm-cmake`, the bundled sysdeps and `amd-llvm`; everything they build keys on
  `amd-llvm@<fprint>`.
- **PORT-4** Record `THEROCK_BUNDLED_*` choices and, when bundling is off, the identity of
  the system replacement.
- **ABI-1** Derive ELF compatibility requirements from built artifacts and compare them by
  ordering, separately from the fingerprint.
- **ABI-2** Parameterize the same audit for PE/COFF, where imports carry no version
  information.

INPUTS-1 describes the fingerprint's inputs; it does not scope them. FP-* and SMREV-1 decide what
is fingerprinted, the manifest is generated from that decision, and DRIFT-1 keeps the published copy
honest. Reading the dependency the other way, with the fingerprint derived from a committed file,
would put
that file in charge of build correctness.

FP-* and SMREV-1 are independently verifiable. INPUTS and DRIFT items build on them. PRE items make
same-machine reuse trustworthy. PORT and ABI items extend that across machines and rest on
compatibility claims that are harder to demonstrate than anything above.

## Decision and scope

`THEROCK_FPRINT` becomes the key that authorizes artifact reuse. A prebuilt stage directory is used
only when its recorded fingerprint equals the fingerprint computed from the current source tree and
configuration. The set of inputs feeding that fingerprint is published as
`test_tools/therock_subproject_inputs.json`, drift-checked in CI, and consumed by `stage_impact.py`
to resolve changed paths to stages.

Same-machine reuse requires FP-1 through FP-4 and PRE-1 through PRE-5. It covers a developer
unpacking artifacts into a fresh build tree on the producing host, and CI reusing a baseline run's
artifacts on identical runner images. Those are testable by building something and comparing an
observable result. Cross-machine reuse additionally needs PORT-1 through PORT-4; without them the
recorded and computed fingerprints differ over compiler paths, build-directory paths and ccache
configuration, so the comparison returns mismatch on every host that did not produce the
artifact.

### Four decisions taken

**Toolchain trust root.** The bootstrap root covers several subprojects.

Everything built by `amd-llvm` inherits its identity through `amd-llvm@<fprint>`, so the host
compiler does not reach those subprojects. `amd-llvm` itself is not a usable cut point: it declares
`rocm-cmake` as a build dependency and the bundled `zlib`, `zstd`, `numactl`, `elfutils` and
`libdrm` as runtime dependencies (`compiler/CMakeLists.txt:134-141`), and those six are compiled by
the host toolchain. Their records would carry `external@<digest>`, which reaches `amd-llvm`'s
fingerprint through the ordinary `DEP=` roll-up at `therock_subproject.cmake:774-785`. Excluding
only `amd-llvm`'s own host closure would leave that path open; stripping the dependency records to
close it would let a host difference change those binaries, and the LLVM built against them, without
moving any key.

The bootstrap closure contains `rocm-cmake`, the five bundled sysdeps and `amd-llvm`. One digest
identifies the pinned toolchain or container image that built it. That
digest is the root key. Subprojects inside the closure key on it; subprojects outside it key on
`amd-llvm@<fprint>` and never see the host toolchain. Cross-machine reuse is refused when the roots
differ, so a consumer whose root differs rejects the artifact.

The root is a declared constant, so it introduces no dependency edge.
and nothing inside the closure depends on a subproject outside it.

An equal bootstrap root does not prove identical compiler output: that two images satisfying the
same declared root produce
interchangeable compilers. An equal root proves the declared build environment matches. It does not
prove identical compiler
output. V-8 requires the comparison before cross-machine reuse is enabled: build `amd-llvm` under
two roots
and compare emitted objects for a fixed input.

**Marker fallback quarantine.** Any decision to build from source after a marker check deletes the
stale `stage.prebuilt` marker and removes or quarantines the imported stage directory, so a build
never mixes imported and source-built output in one stage tree. The cost is that a mismatch discards
an unpack that may have taken minutes; the alternative is a stage directory with two provenances and
no way to tell which file came from where.

**Artifact fingerprint file.** Per-subproject fingerprints travel in one `.fprint` file per slice,
carried inside each of that slice's archives: no archive format change, no new manifest member, no
change to extraction. The schema carries the fingerprint algorithm and schema version, the artifact
component and target family, the build and platform configuration identity, and enough per-input
detail to explain a mismatch. Because artifacts are archived one component at a time (`ArtifactName`
carries a single component, `build_tools/_therock_utils/artifacts.py:36-60`), component coverage
accumulates in the marker across successive unpacks rather than being stated once.

**Platform-keyed manifests.** The subproject inputs file is emitted once per supported configuration and
merged under explicit keys; stage impact unions the input sets for the platforms selected in the
current workflow. This costs a Windows job on the drift workflow, which today runs only on
`ubuntu-24.04`. A Linux-only manifest cannot authorize Windows stage reuse: Windows-only source
references such as `rocm-systems/shared/amdgpu-windows-interop` sit behind `if(WIN32)` and never
reach a Linux-generated manifest.

### Non-goals

- Splitting artifacts in `BUILD_TOPOLOGY.toml`. Reuse is decided at the artifact and stage
  boundary, so a noisy subproject bundled with quiet ones carries them along.
- Bit-reproducible builds. The ABI work compares recorded requirements.
- Changing what `compiler/CMakeLists.txt:46-56` does with `.amd-llvm.smrev`, or how LLVM
  version strings are produced.
- A content-addressed artifact server. This RFC supplies a usable key; where it is looked up
  is out of scope.
- Making the dirty check consider untracked files.

### Unresolved decisions that block implementation

| ID | question | blocks |
|---|---|---|
| Q1 | Contents of the bootstrap-closure digest, and how a builder image is declared and verified. Until settled, cross-machine reuse is refused whenever roots differ. | PORT-3 |
| Q2 | Scope of `THEROCK_ABI_TARGET`: per-build global or per-artifact. Artifacts with no compiled output have no floor. | ABI-1 |
| Q3 | Whether kpack-split artifacts get .fprint files or an explicit exclusion. `therock_artifacts.cmake:172` already skips `.fprint` emission when `_should_split`. Silently having no fingerprint is the bad outcome. | PRE-1 |
| Q6 | How `buildctl disable` records operator intent: a separate `stage.disable` flag file, or a reserved `stage.prebuilt` content form. The name `stage.prebuilt` asserts provenance the disabled case does not have. | PRE-4 |
| Q4 | Whether `SourceSet.path_prefixes` is removed once the manifest supplies the same mapping, or retained as an override. | CI-1 |
| Q5 | Where the committed manifest lives. `test_tools/` beside `therock_consumer_graph.json` is consistent; neither file is a test tool. | INPUTS-1 |

## Current implementation

### Fingerprint data flow

`therock_cmake_subproject_activate` (`cmake/therock_subproject.cmake:637`, fingerprint body at
`:703-1199`) maintains two lists.

`_fprint_files` holds pre/post hooks (`:723-734`), the generated toolchain file (`:755`), the
dependency provider (`:772`), `fileset_tool.py` (`:946`), and glob expansions from
`FPRINT_FILE_GLOBS` or a non-hashed `FPRINT_SOURCE_DIR` (`:714-720`).

`_fprint_content` holds `SOURCE=<rev>` (`:707-712`), one `DEP <target>=<fprint>` line per build and
runtime dependency (`:774-785`), and four configure literals (`:1014-1020`).

`_therock_subproject_fprint_files` appends `basename=sha256` per file at `:1188`. `:1191` takes
`string(SHA256)` over the joined list. `:1194` stores the result as `THEROCK_FPRINT` when
`_fprint_is_valid` and leaves it empty otherwise.

`therock_provide_artifact` (`cmake/therock_artifacts.cmake:154-183`) builds a second digest from the
slice name, the descriptor hash, and one `<subproject>=<fprint>` line per `SUBPROJECT_DEPS`. It
writes that digest at configure time to
`build/artifacts/<slice>_<component><bundle_suffix>.fprint`, once per component, and only when the
artifact is not kpack-split.

Verified: `THEROCK_FPRINT` is read at exactly two `get_target_property` sites,
`therock_subproject.cmake:776` and `therock_artifacts.cmake:157`. The string `.fprint` occurs once
tree-wide, at `therock_artifacts.cmake:176`, which is the `file(WRITE)`.

### Source revision and `.smrev`

`_therock_subproject_fprint_source_dir` (`:1891-1962`) derives `<parent>/.<basename>.smrev`
(`:1893-1895`) and, if present, reads it verbatim into `fprint` (`:1897-1900`). Execution then
continues: `git rev-parse --git-dir` at `:1903-1915`, and inside `if(IN_GIT_REPO)` the `if(NOT
fprint)` guard at `:1919` covers only `git rev-parse HEAD`. The whole-repo dirty check at
`:1941-1955` runs regardless. `.smrev` bypasses only `git rev-parse HEAD`. A
`.smrev` file beside a dirty git repository still yields an empty, invalid fingerprint.

Two writers produce incompatible formats:

| writer | for | format |
|---|---|---|
| `build_tools/fetch_sources.py:655-662` | patched git submodules | two lines: `<url>`, then `<rev>+PATCHED:<patch-digest>` |
| `cmake/therock_subproject.cmake:204-207` | ExternalProject sources | one line: `"${_extra};${ARG_UNPARSED_ARGUMENTS}"` |

`compiler/CMakeLists.txt:51-54` parses the two-line shape with `file(STRINGS)` and `list(GET ...
1)`, feeding `LLVM_FORCE_VC_REPOSITORY` / `LLVM_FORCE_VC_REVISION` alongside `LLVM_APPEND_VC_REV=ON`
(`:67-74`). Those reach `clang --version`, so the file lands in shipped binaries as a string
constant. `patches/amd-mainline/` does not exist, so `apply_patches()` (`fetch_sources.py:597-600`)
skips and the git-submodule writer never runs in a default build; ExternalProject `.smrev` files are
written in every build. The LLVM version-string consumer is therefore dormant upstream, but is the
normal path for any downstream repository maintaining an active patch series.

### Marker behaviour

`stage.prebuilt` carries no content. Every writer creates it with `Path.touch()`:
`build_tools/buildctl.py:125` and `:272`, `build_tools/artifact_manager.py:364` — and CMake tests
only `EXISTS` (`therock_subproject.cmake:956`, `:1223`), using the file as an up-to-date input to
three `add_custom_command` stamps (`:959-979`). `buildctl.py:71-73` documents the contract: the
build system "will just trust that the `stage/` directory contents are correct".

`prebuilt_marker_relpath()` (`build_tools/_therock_utils/artifacts.py:278-300`) maps an
artifact-manifest relpath to the marker path the build reads, truncating at the innermost enclosing
`stage` component, which handles descriptors whose basedir sits below the stage directory such as
`dctools/rdc/stage/portable-rdc`. It is used at `buildctl.py:270` and `artifact_manager.py:362`, and
covered by `build_tools/tests/artifacts_test.py:632-676`. Marker placement is solved; this RFC uses
the helper and does not restate the derivation.

Artifacts do not partition subprojects: `build_tools/artifact_subprojects.json` has 46 artifacts
referencing 87 subproject slots over 72 distinct subprojects. `amd-comgr` is referenced by
`amd-llvm`, `amd-dbgapi` and `rocgdb`, so `build/compiler/amd-comgr/stage.prebuilt` may be written
during any of three unpacks, in fetch order. Empty markers are currently equivalent, so unpack order
does not affect their contents.

### CI change-path behaviour

`StageImpactAnalyzer._resolve_source_set` (`build_tools/github_actions/stage_impact.py:191-218`)
tries `get_source_set_for_submodule(item)` on the whole item (`:201`), then
`get_source_set_for_path(item)` against `path_prefixes` (`:205`), then
`get_source_set_for_submodule(part)` per path component (`:211`). No source set in
`BUILD_TOPOLOGY.toml` populates `path_prefixes` — the field exists in the parser
(`build_tools/_therock_utils/build_topology.py:80`, validated `:562-579`, queried `:956-975`) and is
set only in `stage_impact_test.py:385,390` — so the second step never matches in production.

Step ordering matters for gitlinks. A submodule bump changes exactly one parent-repo path, the bare
string `rocm-systems`, which the first step matches immediately. Populating `path_prefixes` or
supplying a manifest would not change that: the gitlink is not a subproject-level path, so there is
nothing finer to match. `rocm-systems` is listed in the `source_sets` of 17 of the 22 artifact
groups at `fc11b46e`; `rocm-libraries` in 3 (`math-libs`, `ml-libs`, `cv-libs`).

`stage_reuse_mode` defaults to `dry-run` (`.github/workflows/multi_arch_ci.yml:117`), which analyzes
reuse and reports it without applying it; explicitly requested `prebuilt_stages` are honoured in
dry-run (`configure_multi_arch_ci.py:940-945`). The setup job checks out with `fetch-depth: 2` and
sets no `submodules:` input (`.github/workflows/setup_multi_arch.yml:179-184`), so submodule working
trees are not initialised in the job that computes stage impact. `GitContext.from_external_repo`
(`configure_multi_arch_ci.py:329-342`) synthesises `changed_files=[<repo name>]` with no old/new
gitlink pair.

## Required invariants

Every proposal below is subject to these. A change that violates one is wrong regardless of what it
improves.

**INV-1 — no reuse on unknown input.** If any input to a subproject's identity cannot be determined,
the fingerprint is invalid and the subproject builds from source. Invalid never compares equal to
anything, including another invalid.

**INV-2 — producer and consumer hash identical records.** The record a producing build publishes and
the record a consuming build computes are byte-identical for the same source and configuration, and
are produced by the same code path. Presence or absence of a `stage.prebuilt` marker does not change
which records are appended.

**INV-3 — the key graph is acyclic and terminating.** Each subproject's key depends only on keys
already computed. Termination comes from the bootstrap closure: its members key on a declared
`BOOTSTRAP_ROOT` digest rather than on each other, and everything outside keys on
`amd-llvm@<fprint>`, so `COMPILER_TOOLCHAIN` edges introduce no cycle. Membership of the closure is
declared, not inferred, and a member that acquires a dependency outside the closure is rejected at
configure.

**INV-4 — malformed metadata rebuilds safely.** A truncated, unparseable or schema-mismatched
.fprint file or marker is treated as absent: build from source, quarantine any imported tree,
report.
Marker writes are atomic so a partial write is never read as valid.

**INV-5 — conflicts fail before extraction mutates the tree.** Two artifacts claiming different
fingerprints for the same subproject fail configuration, before any stage directory is populated
from the second archive.

**INV-6 — platform-specific data cannot authorize another platform.** A record carrying a platform
key authorizes reuse only for that platform.

## Defects

| ID | defect | effect |
|---|---|---|
| DEF-SOURCE | source identity is submodule-granular | over-reports: unrelated siblings disturb the fingerprint and re-run CI stages |
| DEF-LABELS | fingerprint entries key on basename | diagnosability: a mismatch cannot name which file moved |
| DEF-CONFIG | the configure invocation is not fingerprinted | under-reports: different options, identical fingerprint |
| DEF-EXT-PATHS | source trees named only in configure arguments are not fingerprinted | under-reports: a subcase of DEF-CONFIG |
| DEF-PREBUILT | the `CONFIGURE` block is appended only on the non-prebuilt path | producer and consumer digests differ structurally for identical input |
| DEF-PORTABILITY | the toolchain file is hashed verbatim, absolute paths and all | fingerprints computed on different hosts never match |

### DEF-SOURCE — submodule-granular source identity

`_therock_subproject_fprint_source_dir` runs `git rev-parse HEAD` in the subproject's source
directory. Git resolves to the nearest `.git`, giving the submodule's HEAD. For a single-project
submodule that is the right granularity — `llvm-project` is `compiler/amd-llvm`. For monorepo
submodules many independent subprojects share one value: `rev-parse HEAD` in
`rocm-systems/projects/rocr-runtime` and in `rocm-systems/emulation/rocjitsu` both return the
submodule pin, while `rev-parse HEAD:projects/rocr-runtime` and `rev-parse HEAD:emulation/rocjitsu`
return distinct tree ids that move independently. The dirty check at `:1941-1955` is whole-repo for
the same reason, so an edit anywhere in the submodule drives every subproject built from it to an
invalid fingerprint; a TODO at `:1941-1943` records this, referencing ROCm/TheRock#2432.

The build system already knows the right granularity. Every monorepo subproject declares a directory
via `EXTERNAL_SOURCE_DIR`: `rocm-systems/projects/rocr-runtime` (`core/CMakeLists.txt:125`),
`rocm-systems/projects/clr` (`:290`, `:502`), `rocm-systems/shared/kpack` (`:20`),
`rocm-systems/emulation/rocjitsu` (`emulation/CMakeLists.txt:18`), and roughly 25 more across
`rocm-libraries`.

### DEF-LABELS — basename keys

`_therock_subproject_fprint_files` (`:1964-1972`) records `basename=sha256`, so every
`CMakeLists.txt` keys under the same name. This causes no collision: entries are appended to an
ordered CMake list, so entries keep their position and the digest covers the joined list; exchanging
the
contents of `A/CMakeLists.txt` and `B/CMakeLists.txt` turns `CMakeLists.txt=h1;CMakeLists.txt=h2`
into `CMakeLists.txt=h2;CMakeLists.txt=h1` and the digest changes. The cost is that a mismatch
report cannot name the file that moved, and a file relocated without a content change is
indistinguishable from no change.

### DEF-CONFIG — the configure invocation is not fingerprinted

`_cmake_args` is retrieved at `:648`, finalized at `:934`, and passed to the configure command at
`:1037`. It is never appended to `_fprint_content`. The only configure values entering the digest
are the four literals at `:1014-1020`: `CMAKE_BUILD_TYPE`, the relative source dir, and the two
stage dirs. So the fingerprint does not move when a subproject's `CMAKE_ARGS` change, when the
declaring `CMakeLists.txt` in TheRock changes (it is not generally in `_fprint_files`), or when the
generator or install destination changes. A subproject can be reconfigured with different feature
toggles, sanitizer settings and cache values and keep an identical fingerprint. This is the largest
hole. `_build_env_pairs` (`:748`), which prefixes every configure and build command via
`cmake -E env`, is likewise absent.

### DEF-EXT-PATHS — source trees named only in arguments

```cmake
# compiler/CMakeLists.txt:117
-DLIBOMPTARGET_EXTERNAL_PROJECT_HSA_PATH=${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/rocr-runtime
# core/CMakeLists.txt:405       FPRINT_SOURCE_DIR -> rocr-runtime/libhsakmt
# core/CMakeLists.txt:255,472   _compute_pal_dir  -> rocm-systems/shared/amdgpu-windows-interop
```

libomptarget compiles against those headers. They enter no `_fprint_files` list, so `amd-llvm`'s
fingerprint does not move when `rocr-runtime` headers change. This under-reports, which is the
direction that authorizes a stale artifact.

Scanning `CMAKE_ARGS` for path-valued strings does not fix it: such a scan misses individual files,
relative paths, list-valued arguments, generator expressions, paths embedded in compiler flags,
symlinks, and paths computed at build time — `pre_hook_amd-llvm.cmake` sets
`LIBOMPTARGET_EXTERNAL_PROJECT_HSA_PATH` inside the hook rather than through `CMAKE_ARGS`, so that
instance is invisible to a scan. Under-reporting produces false matches, so the scan is not
correctness coverage anywhere in this document. Correctness comes from FP-3, which hashes the whole
invocation; FP-4's scan is a lint that proposes trees for declaration.

### DEF-PREBUILT — producer and consumer compute different records

The prebuilt branch is selected at `:956`. The `CONFIGURE` literals at `:1014-1020` sit inside the
`else` branch and are appended only when no marker is present. The digest is computed later, at
`:1188-1194`, from whatever accumulated. A tree configured without a marker includes the `CONFIGURE`
block; a tree configured with one omits it. The two digests differ structurally for the same source
at the same revision, before portability enters. Any scheme comparing a stored fingerprint against a
freshly computed one is broken until this is fixed. Nothing in `main` compares fingerprints, so this
is currently unobservable.

### DEF-PORTABILITY — the toolchain file is hashed verbatim

The generated toolchain file is fingerprinted at `:755`. Its contents
(`_therock_cmake_subproject_setup_toolchain`, `:1682-1885`) mix three kinds of value:

- Output-neutral: `CMAKE_C_COMPILER_LAUNCHER` / `CMAKE_CXX_COMPILER_LAUNCHER` (`:1735-1736`) hold the
  ccache path, so one developer running `eval "$(./build_tools/setup_ccache.py)"` and another not,
on
  the same tree, fingerprint differently for byte-identical output.
- Build-tree absolute: `AMD_LLVM_C_COMPILER` / `AMD_LLVM_CXX_COMPILER` resolve into
  `<build>/compiler/amd-llvm/dist/...`, and `--hip-path` / `--hip-device-lib-path` likewise
  (`:1872-1875`). These differ between build directories on one machine.
- Host absolute: `CMAKE_C_COMPILER` / `CMAKE_CXX_COMPILER` / `CMAKE_LINKER` (`:1728-1730`), plus
  sysroot and `CMAKE_LIBRARY_ARCHITECTURE` — `/usr/bin/g++-11` on one host,
  `/opt/rh/gcc-toolset-11/root/usr/bin/g++` on another, for the same compiler.

Two hosts with compatible toolchains never produce matching fingerprints.

Two related gaps. `COMPILER_TOOLCHAIN` is registered in the consumer graph at `:575-587` but is not
added to `THEROCK_BUILD_DEPS` at `:534`, so `amdsmi` selects `amd-llvm` as its compiler without
`amd-llvm`'s fingerprint ever rolling into `amdsmi`'s at `:774-785`. And while `THEROCK_BUNDLED_*`
choices already affect fingerprints when bundling is on — the variables resolve to subproject target
names in `BUILD_DEPS` / `RUNTIME_DEPS` whose fingerprints roll into consumers at `:774-785` —
nothing records the identity of the system replacement when bundling is off. There are 20 such
variables: `AMDMESA`, `BZIP2`, `ELFUTILS`, `EXPAT`, `GMP`, `HWLOC`, `LIBBACKTRACE`, `LIBCAP`,
`LIBDRM`, `LIBIBERTY`, `LIBLZMA`, `LIBMNL`, `LIBNL`, `MPFR`, `NCURSES`, `NUMACTL`, `SQLITE3`,
`UTIL_LINUX`, `ZLIB`, `ZSTD`.

## Data model

### Canonical input record

The fingerprint is a SHA-256 over a versioned byte encoding of an ordered record stream. The same
encoder produces the bytes that are hashed and the entries in the INPUTS-2 dump, so the dump always
explains the digest it accompanies. Encoding: UTF-8, one record per line, `\n`-terminated, with
`\\`, `\n` and `\r` escapes. Records appear in emission order and are never sorted — duplicate `-D`
arguments are order-sensitive to CMake, so sorting would make two different invocations hash equal.
The stream opens with `SCHEMA=<n>` and `SUBPROJECT=<logical target>`; bumping `<n>` invalidates
every stored fingerprint.

| record | meaning |
|---|---|
| `SOURCE=<tree-id>` / `SOURCE_SMREV=<sha256>` | FP-1 subtree object id, or the digest of `.smrev` bytes when the short-circuit fires |
| `FILE=<label>=<sha256>` | FP-2 label: repo-relative, or `<build>/`-relative for generated files |
| `SUBTREE=<label>=<tree-id>` / `REF=<label>=<tree-id>` | a declared directory input; `REF` is an FP-4 tree reached through configure arguments |
| `DEP=<target>=<fprint>` | dependency fingerprint, as today |
| `TOOLCHAIN=<key>` | PORT-3 provenance key |
| `BOOTSTRAP_ROOT=<algo>:<digest>` | PORT-3; the pinned image or toolchain rooting the bootstrap closure |
| `CONFIGURE_ARG=<i>|<arg>` / `BUILD_ENV=<i>|<arg>` | FP-3; ordered argv, index preserves position |
| `CMAKE_VERSION=<v>`, `GENERATOR=<g>` | FP-3 |
| `LITERAL=<key>=<value>` | the existing `:1014-1020` values, plus install destination |
| `SYSDEP=<name>=bundled` / `=system:<version>` | PORT-4 |

"Normalized" means exactly this encoding: labels rebased, absolute build-tree paths rebased onto
`THEROCK_BINARY_DIR`, launchers excluded, nothing reordered. Any record that cannot be produced sets
`_fprint_is_valid` false, and an invalid fingerprint is the empty string, which INV-1 forbids from
comparing equal to anything.

### Toolchain key graph and trust root

Keys are rooted in a declared bootstrap closure. `BOOTSTRAP_ROOT=<algorithm>:<digest>` is a record
in every subproject's canonical input, naming the pinned builder image or toolchain that produced
the closure. Members are `rocm-cmake`, the bundled `zlib`, `zstd`, `numactl`, `elfutils` and
`libdrm`, and `amd-llvm` itself — the set `compiler/CMakeLists.txt:134-141` requires. The
configuration lists every closure member, and a member acquiring a dependency outside the closure is
rejected
at configure. Ordinary `DEP=` records are retained inside the closure; what the root replaces is the
host toolchain's contribution, which would otherwise reach `amd-llvm` through those dependencies.
Membership can vary by platform, so the root is recorded per configuration.

Outside the closure the host compiler does not appear. `amd-llvm`'s key is its own source and
configure record plus `BOOTSTRAP_ROOT`;
`hip-clr`, `amdsmi`, `rocBLAS` and every other consumer carry `TOOLCHAIN=amd-llvm@<fprint>` in
theirs. `TOOLCHAIN=` takes one of two forms. `amd-llvm@<fprint>` for in-tree toolchains, resolved
through
`therock_compiler_toolchain_subproject()` (`:213-220`); this closes the
`COMPILER_TOOLCHAIN`-not-in-`BUILD_DEPS` gap without changing the dependency graph, because the
toolchain's identity enters the consumer's record directly. `external@<digest>` for toolchains
TheRock did not build, where the digest covers a declared closure that Q1 must settle. Compiler id,
version and target are not sufficient: two differently-patched builds of the same LLVM report
identical `CMAKE_CXX_COMPILER_ID`, `_VERSION` and `_TARGET` while generating different code, and the
tuple omits the linker, the resource directory, default search paths, the sysroot and the assembler.
Until Q1 is settled, `external@` subprojects are ineligible for cross-machine reuse.

### Sidecar and marker

The fingerprint record travels **inside** the artifact archive, as a root member named `.fprint`
written immediately after `artifact_manifest.txt`. One archive, one object: nothing else to upload,
fetch, expire or keep in sync. A separate file beside the archive would double the object count for
every artifact and put CI in the position of handling an archive whose companion is missing.

This is a deliberate archive-format change, and a small one at both ends:

- Writer: `do_artifact_archive` (`build_tools/fileset_tool.py:81-105`) adds `artifact_manifest.txt`
  first; the record is added next, before the manifest's relpath contents.
- Reader: `ArtifactPopulator` (`build_tools/_therock_utils/artifacts.py:187-204`) already requires
  `artifact_manifest.txt` as the first member, then matches each remaining member against a manifest
  prefix. A member matching no prefix is skipped without error, so archives carrying `.fprint` are
  readable by unmodified consumers; the populator gains an explicit check for it rather than relying
  on that.

It replaces the per-component `<slice>_<component><suffix>.fprint` files written at
`therock_artifacts.cmake:176`, which are build-tree artifacts nothing publishes.

It carries no component or target-family field. A subproject's fingerprint does not vary by
component — components are packaging slices of one stage directory, not separate builds — so one
record per subproject covers every archive in the slice. The bundle suffix in the filename
disambiguates target-specific slices, which is the only axis along which the fingerprints actually
differ.

```
# build/artifacts/amd-llvm_generic.fprint      — one key=value group per line
FPRINT_SCHEMA=2 | FPRINT_ALGO=sha256 | ARTIFACT=amd-llvm | DESCRIPTOR=<sha256 of the descriptor>
PLATFORM=linux-x86_64 | CONFIG=<build config digest> | DUMP=<relpath of the INPUTS-2 dump>
amd-comgr | compiler/amd-comgr/stage | 9f2a...
amd-llvm  | compiler/amd-llvm/stage  | 44c1...
hipcc     | compiler/hipcc/stage     | UNKNOWN

# build/compiler/amd-comgr/stage.prebuilt
FPRINT_SCHEMA=2 | SUBPROJECT=amd-comgr | FPRINT=9f2a...
COMPONENTS=lib,run | ARTIFACTS=amd-llvm,rocgdb
```

Each subproject record carries its `stage_relpath`. The unpacker needs to know that
`compiler/amd-comgr/stage` belongs to `amd-comgr`, and nothing today provides that mapping:
`prebuilt_marker_relpath()` yields a marker path, not a subproject name, and
`artifact_subprojects.json` maps artifacts to subprojects without basedirs. Shipping the mapping
beside the fingerprint avoids reconstructing it. On unpack the path must appear in
`artifact_manifest.txt`, must be relative, must not contain `..`, and must not be claimed by two
records.

**INV-6 — every artifact basedir lives under its subproject's stage directory.** One
`stage_relpath` per record is only sufficient if a subproject's basedirs all resolve to one marker.
Across all descriptors there are 116 distinct basedirs, and three do not end in `/stage`:
`dctools/rdc/stage/portable-rdc`, which is under a `stage` component and truncates correctly, and
`math-libs/hipthreads/build` and `math-libs/libhipcxx/build`, which are not.

Those two are already broken on `main`, independently of this RFC.
`prebuilt_marker_relpath()` documents that a relpath with no `stage` component is "returned
unchanged apart from the suffix", so bootstrapping those artifacts writes `build.prebuilt` while
CMake tests `stage.prebuilt` — the artifact is fetched and the subproject rebuilds from source
anyway. This is the same class of defect the helper was written to fix for `rdc`.

The RFC therefore enforces the invariant rather than modelling the exception: those two descriptors
are corrected to stage-relative basedirs, and the drift check rejects any descriptor declaring a
basedir outside its subproject's stage directory. Two descriptors change instead of every `.fprint`
record growing a second path field.

`UNKNOWN` means "no fingerprint available" — an invalid fingerprint, or a subproject absent from
this build. It never matches, including against another `UNKNOWN`, and a marker carrying it rebuilds
from source.

`buildctl disable` needs a separate mechanism. Its purpose is to *stop* a subproject building, so
any state that rebuilds would break it, and an empty marker can no longer carry that meaning once
empty means "unidentified". The mechanism is Q6: `stage.prebuilt` reads as "this was built
elsewhere", which is the wrong claim for "do not build this", so overloading its content is a naming
compromise rather than a natural fit. Whatever is chosen, operator intent must outrank every
fingerprint state, never be compared against a producer record, and never be deleted automatically.
This document writes `MODE=manual` where a token is needed, as a placeholder for Q6's outcome.

Each marker stores one subproject fingerprint. The artifact-level `.fprint` record may list
several. A marker carrying
artifact-scoped content would make `amd-comgr`'s marker describe `amd-llvm` and `hipcc` too, so
`amd-comgr` would be judged stale when `hipcc` changed, with the outcome depending on which artifact
unpacked last. Subproject scoping makes the marker artifact-independent, order-independent and
idempotent across overlapping artifacts, which the 46/87/72 overlap above requires. `COMPONENTS`
accumulates across unpacks, derived from the archive names the unpacker processed rather than
declared in .fprint file; `ARTIFACTS` is diagnostic. Neither file stores a marker path — it
comes from `prebuilt_marker_relpath()` applied to the artifact manifest relpath at unpack time.
Storing identity and location in one field would make the artifact fingerprint sensitive to
build-directory layout, reintroducing DEF-PORTABILITY in a new place.

### Configuration-keyed subproject inputs file

| file | contents | lifecycle |
|---|---|---|
| `test_tools/therock_subproject_inputs.json` | labels only, keyed by configuration | committed, drift-checked |
| `build/therock_input set_fprint.json` | labels, digests, fingerprint, validity | per-configure, gitignored |

They are separate because digests in the committed copy would change on every edit to any input, so
the drift check would fire on every PR and re-report what git history already says.

```json
{"schema": 1, "configurations": {
  "linux-x86_64/gfx94X-dcgpu/all": {
    "subprojects": {"ROCR-Runtime": {
      "files": ["core/CMakeLists.txt", "cmake/therock_subproject_dep_provider.cmake"],
      "source_dirs": ["rocm-systems/projects/rocr-runtime"],
      "referenced_paths": ["rocm-systems/projects/rocr-runtime/libhsakmt"],
      "deps": ["amd-llvm", "therock-simde"]}},
    "artifacts": {"core-runtime": {
      "subprojects": ["ROCR-Runtime", "rocminfo"],
      "descriptor": "core/artifact-core-runtime.toml",
      "artifact_deps": ["base", "sysdeps", "amd-llvm"]}}},
  "windows-x86_64/gfx110X-all/all": {"...": "..."}}}
```

Source directories are recorded as directories and never expanded; consumers key on the subtree
hash. The `GLOB_RECURSE` paths at `:714-720` barely expand today — `FPRINT_SOURCE_HASH` has two
users (`base/CMakeLists.txt:12`, `comm-libs/CMakeLists.txt:86`), neither a submodule, and
`FPRINT_FILE_GLOBS` has 23 users, all small in-tree wrapper directories with a largest expansion of
11 files — but collapsing directories keeps a future `FPRINT_SOURCE_HASH` on a submodule from
ballooning the manifest. The schema is normalized rather than denormalized per artifact: roughly 350
distinct labels exist tree-wide, and denormalizing the transitive closure into each of ~50 artifacts
would repeat those into ~150 KB per configuration and turn a one-file edit to `fileset_tool.py` into
a 50-artifact drift diff. Normalized lands near 25-30 KB per configuration with a one-line diff,
against the 14 KB / 117-subproject consumer graph.

## Algorithms

### Compute the fingerprint (FP-1, FP-3, PRE-3)

```
activate(target):
    record = new_record(SCHEMA, target)
    append source identity, file and subtree labels, DEP lines, TOOLCHAIN key
    finalize _cmake_args                                     # currently :934
    append CONFIGURE_ARG / BUILD_ENV / CMAKE_VERSION / GENERATOR / LITERAL
    fprint = sha256(encode(record));  set THEROCK_FPRINT

    if marker_present(stage_dir):                            # currently :956
        decision = validate_marker(target, fprint)
        instantiate stamp-touch or source-build commands per decision
    else:
        instantiate source-build commands
```

The record is complete before the marker is tested, and the marker branch selects only which build
commands are instantiated. This satisfies INV-2 and fixes DEF-PREBUILT. The `CONFIGURE` literals
move from inside the `else` branch at `:1014-1020` onto the common path.

### Source identity (FP-1)

Resolve the enclosing repository root with `git rev-parse --show-toplevel`, compute the directory's
path within it, and hash the subtree — `HEAD^{tree}` when the path is the repository root,
`HEAD:<subpath>` otherwise. Scope the dirty check to the same path with `git diff --quiet HEAD --
"<subpath>"` run from the repository root, replacing the whole-repo form at `:1945`. Untracked files
remain not-dirty, matching today's semantics. The returned value changes shape from a commit id to a
tree id; nothing displays or interprets it as a commit, and tree objects carry no committer
metadata, so this is more stable than commit hashing for patched checkouts.

`.smrev` precedence is unchanged, and the granularity works out without a precedence change. The
lookup path is `<parent>/.<basename>.smrev` (`:1893-1895`), so for a monorepo subdirectory such as
`rocm-systems/projects/rocr-runtime` the lookup is `rocm-systems/projects/.rocr-runtime.smrev`,
which nothing writes — `fetch_sources.py` emits one `.smrev` per submodule, beside the submodule
root. The short-circuit never fires and subtree hashing applies, which is the whole DEF-SOURCE win.
For a submodule root such as `compiler/amd-llvm` the short-circuit fires when patched, and the
source directory is the repository root, so `HEAD^{tree}` and the whole-tree identity coincide.
`.smrev` remains required for sources with no git repository (ExternalProject tarballs, written by
`therock_subproject.cmake:204-207`, active in every build), trees with `.git` removed, and patched
checkouts.

### SMREV-1 — ExternalProject source identity

The CMake-side writer emits `"${_extra};${ARG_UNPARSED_ARGUMENTS}"`, which is the argument list
rather than a source identity, and is a single line where the LLVM consumer expects two. Give it the
two-line shape and a real identity: the URL or origin on line one, then
`<URL_HASH>+PATCHDIR:<sha256 over the sorted patch-directory file digests>`. `URL_HASH` is already
declared for these sources and identifies the tarball content; `PATCHDIR` covers `PATCH_COMMAND`
inputs under `patches/third-party/`. This gives a non-git source a stable identity that changes when
the tarball or its patches change, and makes the two writer formats interchangeable so
`compiler/CMakeLists.txt:51-54` cannot fail on an ExternalProject-written file. The fingerprint path
is unaffected either way: `:1900` reads the whole file and hashes the bytes.

### FP-4 external-path coverage, and producing .fprint file (PRE-1)

Where `_cmake_args` is finalized, scan `-D<name>=<value>` for values resolving under
`THEROCK_SOURCE_DIR` and emit each as a lint finding naming the subproject, the argument and the
resolved path. Findings are reported by the DRIFT-1 job, not silently folded into the record.
Declared external trees become `REF=` records hashed as subtrees. Declaration is a new
`FPRINT_SOURCE_DEPS` argument on `therock_cmake_subproject_declare`, alongside the existing
`FPRINT_SOURCE_DIR`, `FPRINT_FILE_GLOBS` and `FPRINT_SOURCE_HASH`
(`therock_subproject.cmake:570-572`):

```cmake
therock_cmake_subproject_declare(amd-llvm
  FPRINT_SOURCE_DEPS
    "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/rocr-runtime"
)
```

Entries resolve under `THEROCK_SOURCE_DIR`, are hashed as subtrees by the FP-1 helper, appear in
INPUTS-1 as `referenced_paths`, and are deduplicated against `FPRINT_SOURCE_DIR`. Files are allowed
as well as directories; symlinks resolve before hashing. Declarations behind platform conditionals
are recorded per configuration, which is one reason INPUTS-1 is configuration-keyed.

The lint never creates records. Scanning under-reports — computed paths, relative paths, and paths
set inside `pre_hook_*.cmake` rather than through `CMAKE_ARGS` all evade it — so a fingerprint that
depended on scanner completeness would sit one missed pattern away from a false match. FP-3 supplies
the correctness floor by hashing the argument string itself; FP-4 adds the referenced tree's content
once someone has declared it.

At artifact-populate time, replace the existing per-component `.fprint` writes at
`therock_artifacts.cmake:172-181` with one `build/artifacts/<slice_name><bundle_suffix>.fprint`
carrying the schema above,
one line per `SUBPROJECT_DEPS` entry and `UNKNOWN` where `get_target_property(... THEROCK_FPRINT)`
is empty. Upload and fetch are unchanged: the record ships within the archive they already move.
The existing aggregate per-component `.fprint` files continue to be written; whether they retain a
purpose is left open.

### Validate and merge on unpack (PRE-2, PRE-4, PRE-5)

```
# phase one -- no filesystem mutation
plan = validate_all(selected_archives, fprint_files, manifests, existing_markers)
  for each archive A, component C, basedir r:
      rec  = fprint[A].record_for(r)          # stage_relpath -> subproject, fingerprint
      mark = prebuilt_marker_relpath(r)       # INV-6 guarantees this is the stage marker
      if existing[mark] is operator-disabled:      plan.skip(A, reason=disabled)
      elif existing[mark] absent:                  plan.write(mark, rec, [C], [A])
      elif existing[mark].FPRINT == rec.fprint != UNKNOWN:
                                                   plan.merge(mark, [C], [A])
      else:                                        plan.conflict(mark, A, existing[mark].ARTIFACTS)
  plan.require_coverage(caller_required_components)
if plan.has_conflicts or plan.coverage_incomplete:
    abort the fetch; nothing has been written
# phase two -- apply, atomically per marker
apply(plan)
```

Bootstrap runs in two phases. Phase one downloads and parses every selected `.fprint` record and
`artifact_manifest.txt`, then validates schema versions, stage-path ownership, required component
coverage and cross-archive identity agreement. Phase two extracts and writes markers atomically, by
temp file plus rename. Nothing reaches a stage directory until every check has passed, as INV-4 and
INV-5 require. `artifact_manager` extracts archives in parallel, so a check performed during
extraction would run after a peer worker had already mutated the tree. Phase-one failures terminate
the fetch command.

Each subproject entry carries its own validity field. One unfingerprintable subproject must not
prevent verification of the others sharing its artifact, and under a whole-submodule dirty check
that case is common.

Conflicts are reachable today. `buildctl` keeps one populator across the iteration and
`artifact_manager` shares a locked cleaned-path set across workers, so which archive lands first can
depend on directory iteration or thread scheduling. The bootstrapper compares full records before
merging and rejects two different fingerprints for one subproject.

### Marker decision procedure (PRE-4)

| marker state | local fprint | configure result | marker | stage dir | build action |
|---|---|---|---|---|---|
| absent | any | proceed | — | — | build from source |
| `MODE=manual` | any | proceed | keep | keep | suppress build (operator intent) |
| match | valid | proceed | keep | keep | use prebuilt |
| mismatch | valid | log differing records | delete | quarantine | build from source |
| `UNKNOWN` | any | proceed, warn | delete | quarantine | build from source |
| empty, malformed, or unknown schema | any | proceed, warn | delete | quarantine | build from source |
| any | invalid (dirty tree) | proceed, warn once | delete | quarantine | build from source |

Rows are evaluated top to bottom and the first match wins; `MODE=manual` outranks everything so a
deliberate `buildctl disable` is never overridden.

Every path that falls back to source deletes the marker and quarantines the imported stage tree, so
no build mixes imported and locally built files. This applies to the dirty-tree row as much as to a
mismatch: keeping a marker beside a stage that a dirty source build has written into leaves a
directory describing neither provenance, and once the edit is reverted the fingerprint matches again
and a later configure trusts contaminated output.

An empty marker is not trusted. Legacy markers carry no identity, and INV-1 does not admit an
exception for artifacts that predate this work; they rebuild once and then carry a fingerprint.

Coverage and conflict do not appear here. Both are resolved during bootstrap phase one, before
anything is extracted — by configure time a marker is either present and complete or absent.

Except for bootstrap conflicts, a validation failure causes a source build. Configure never fails on
marker state: conflicts and
incomplete coverage are settled during bootstrap phase one, before any stage tree is written, so a
marker reaching configure is already coherent.

Quarantine moves the imported stage tree aside to `<stage>.quarantine.<n>` or removes it, per
`THEROCK_PREBUILT_FALLBACK=quarantine|delete`; the default is `quarantine` so the discarded tree can
be inspected, and CI sets `delete`. Either way the stage directory the build writes into contains no
imported files. On mismatch the configure output names the differing records from the INPUTS-2
dump; a bare "out of date" is unactionable, and until PORT-* lands mismatches are the common case.

There is no mode that reports a mismatch and uses the prebuilt tree anyway. Such an option would
suspend INV-1 by design, and a reuse key that can be overridden on a warning provides no guarantee
worth stating. `stage_reuse_mode` is a workflow input governing CI stage reuse
(`off`/`dry-run`/`reuse-stage`) and has no bearing on marker validation.

### CI-1 — gitlink expansion

Stage impact sees only the gitlink path for a submodule bump, so the subproject inputs file alone
cannot
give it subproject-level paths. Per changed gitlink:

1. Read old and new object ids: `git diff --raw --abbrev=40 <base>..<head> -- <path>`. `--abbrev=40`
   is required; `git diff --raw` abbreviates by default and the abbreviated form cannot be fetched.
2. Obtain a submodule repository. The setup job checks out with `fetch-depth: 2` and does not set
   `submodules:` (`.github/workflows/setup_multi_arch.yml:179-184`), so no submodule working tree
   exists. Create a temporary bare repository under the runner temp directory, or
   `git submodule update --init --depth 1` for the affected path only. Bare is preferred: no working
   tree is materialised and the cost is one fetch.
3. Resolve the remote URL from the base commit's `.gitmodules` and validate it against an allowlist
   of ROCm organisation URLs. A URL from an untrusted pull request's `.gitmodules` must not be
   fetched.
4. Fetch both object ids from the validated remote, with a timeout.
5. `git -C <repo> diff --name-only <old> <new>`; prefix each result with the submodule path.
6. Resolve the prefixed paths through the subproject inputs file, unioned over the platforms
selected in
   the current workflow.

Any anomaly falls back to marking the whole source set affected, the pre-existing behaviour: `--raw`
output that does not parse, a fetch that fails or times out, an added or removed submodule rather
than a bump, a `.gitmodules` URL change, a URL failing the allowlist, or an empty diff.
External-repo CI is unsupported — `GitContext.from_external_repo`
(`configure_multi_arch_ci.py:329-342`) synthesises `changed_files=[<repo name>]` with no old/new
gitlink pair, so there is nothing to expand and those runs keep whole-source-set impact. This adds a
network fetch to a decision that is currently pure path matching; a bump spanning many commits
legitimately touches many subprojects, so the expansion pays off on narrow bumps and changes nothing
on broad ones.

Downstream of the expansion manifest matches resolve directly to subprojects and artifacts, without
passing through source-set
resolution.
`_resolve_source_set()` returns a single `SourceSet` (`stage_impact.py:191-218`), so routing an
expanded path through it yields `rocm-systems` again and re-marks all seventeen groups — the
precision won by expanding the diff would be discarded one call later. Expanded paths instead
resolve directly:

```
changed path -> subprojects (subproject inputs file)
             -> artifacts containing those subprojects
             -> artifact groups -> stages
```

Source-set resolution is retained only for inputs the manifest does not model, such as paths in
TheRock's own tree.

The mapping is derived from `EXTERNAL_SOURCE_DIR` declarations at emission time, so it cannot drift
the way a hand-populated `path_prefixes` would (Q4 decides whether `path_prefixes` is then removed
or
kept as an override).

A path that survives expansion but matches nothing in the manifest is treated as unmatched, and
unmatched input falls back to whole-source-set impact. Treating it as *unaffected* would be the more
aggressive reading and would deliver a larger win, but it converts manifest incompleteness into
silently skipped builds. A submodule may opt into the aggressive reading only after passing the
sparse-checkout inputs validation in Safety checks.

Resolution stays bounded by artifact and stage granularity: a path resolving to a subproject marks
its artifacts affected, and a stage re-runs if any artifact it builds is affected. In Example 1
below, `compiler` is unaffected but shares the `compiler-runtime` stage with eight affected groups,
so that stage re-runs regardless.

## Safety checks

Several proposals here can produce a false match, and each carries a distinct soundness obligation:
FP-1 if a narrowed subtree omits a real input; FP-4 if a declaration is incomplete, since scanning
under-reports; PORT-2 if a build path reaches the output; PORT-3 because the bootstrap root asserts
provenance and not equivalence; and any acceptance of a marker whose identity cannot be checked. The
remaining proposals over-report, where a mistake costs a rebuild. The checks below are ordered by
which obligation they discharge.

| # | escape class | caught by |
|---|---|---|
| 1 | path in `CMAKE_ARGS` | FP-4 lint, then declaration |
| 2 | path set inside a pre/post hook | static lint only; the hook is hashed, the tree it names is not |
| 3 | relative escape inside the subproject's own build files | static lint and the sparse build |
| 4 | monorepo-root files consumed implicitly | the sparse build |

Class 3 is not hypothetical. Several files under `rocm-systems/projects/rocr-runtime` contain
`../../../` escapes. At the pinned `rocm-systems@14f81ac4`,
`projects/rocr-runtime/libhsakmt/CMakeLists.txt:379` points `WKMI_INCLUDE_DIR` at
`../../../shared/amdgpu-windows-interop/wkmi` and `:380` appends `win/lib/wkmi.lib` to
`WKMI_LIB_PATH`; both are consumed at `:382` and `:387`. That tree is outside the declared subtree,
and the references are Windows-gated, so a Linux build does not reveal them.

### Static escape lint

Scan each subproject's build files for `../` sequences resolving outside its declared input set and
for references to the monorepo root. Runs in the DRIFT-1 job; an escape must be added to the input
set
or waived with a recorded reason. Not sound — it cannot see computed paths — but it catches the wkmi
case statically, on both platforms, which a Linux build cannot.

### Sparse-build content comparison

Per subproject `S`, materialize only `S`'s declared paths
(`git -C rocm-systems sparse-checkout init --no-cone` then `sparse-checkout set <declared paths>`)
and build `S+build`. `--no-cone` is required: cone mode cannot express an arbitrary path set, and it
implicitly materializes repository-root files, which would mask escape class 4.

Build success is too weak a signal — a missing path can leave a build succeeding with an optional
dependency not found and a conditional target skipped. Compare against a control full build by
content: per-file sha256, mode bits and symlink targets from each build must be equal. Inequality
means the missing paths changed what was produced. The comparison must not use
`artifact_manifest.txt`, which records names; a missing input that selects a fallback implementation
produces the same filenames with different bytes and would compare equal. The two builds run in
separate worktrees, because reconfiguring sparse-checkout in the shared submodule mutates state
other jobs may be reading.

Limits, which argue for a periodic job rather than a per-PR gate: it proves sufficiency only for the
configurations exercised, so the Windows-gated wkmi escape survives a Linux run; it costs a full
build per subproject, twice; and it needs the control run to make a failure attributable. Output
feeds FP-4 — a path the sparse build proves necessary but the lint did not find is either added to
the declared input set or is a bug in the subproject's build files.

### Shadow fingerprints, opt-in narrowing, tracing

Compute coarse and narrow fingerprints together and record both. Replay N past commits; wherever the
coarse value moved and the narrow one did not, build both revisions and compare artifact content.
Divergence means an input was missed. This quantifies residual risk rather than asserting safety,
and depends on PORT-1 and PORT-2, since comparing artifacts requires that equal inputs give equal
outputs. One confound to subtract first: `LLVM_APPEND_VC_REV=ON` with `LLVM_FORCE_VC_REVISION`
(`compiler/CMakeLists.txt:67-74`) embeds the source revision in `clang --version`, so LLVM binaries
differ byte-wise whenever the revision moves, independently of any missed input. Mask the version
string or pin `LLVM_FORCE_VC_REVISION` for replay runs, and expect other embedded-provenance cases.

Do not narrow globally. A submodule stays submodule-granular until listed as splittable, and is
listed only after the static lint and the sparse-build comparison pass and a replay has run clean.
Opt-in confines an incorrect declaration to one submodule. As a further periodic audit, build under
`strace`
or fanotify, record every path opened, and diff against the declared input set — sound, catches
computed paths, expensive.

### ABI compatibility checks

Separate build identity from ABI compatibility. "Same inputs, same output?" is answered by
fingerprint equality; "can this artifact be used here?" is answered by ordering over recorded
requirements. Fingerprints compare by equality; ABI requirements compare by ordering — an artifact
built against
glibc 2.28 runs on 2.35 and the reverse fails. Hashing the host glibc version is too strict; hashing
a declared floor is a claim that can produce a false match.

**ABI-1** extracts requirements from the built ELF objects: the highest required `GLIBC_x.y`,
`GLIBCXX_x.y.z` and `CXXABI_x.y.z` symbol versions, and every `DT_NEEDED` soname with its version
needs. This is the shape of auditwheel/manylinux, Debian `shlibs`/`symbols`, and RPM automatic
`Requires`. `THEROCK_ABI_TARGET` becomes a policy assertion checked against the derived requirements
rather than a fingerprint input.

Use binary inspection to reject incompatible artifacts, not to establish input equivalence. Presence
of `__cxx11` mangled
symbols can indicate the new libstdc++ ABI, but absence cannot prove the old one — a translation
unit may simply use no affected type. `DT_NEEDED` entries and symbol versions identify runtime
library requirements, not the header versions a translation unit compiled against, and say nothing
about header-only dependencies. Extraction detects a mismatch after the fact; it does not certify a
match. It does record what a system-provided sysdep supplied, as a soname and version need, without
anyone enumerating system packages, but PORT-4 still records `THEROCK_BUNDLED_*` values and the
configure-time version of each system replacement in the fingerprint, because those change what is
built rather than what it can run against. That is conservative — system drift disturbs the
fingerprint even when the output would have been identical — but it cannot produce a false match,
which a declared floor can.

**ABI-2** covers PE/COFF, where none of the ELF mechanisms exist:

| | Linux / ELF | Windows / PE-COFF |
|---|---|---|
| dependency record | `DT_NEEDED` soname | import table DLL names |
| version granularity | per-symbol `GLIBC_x.y` / `GLIBCXX_x.y.z` | none — imports are unversioned |
| C runtime | glibc, with a derivable floor | UCRT plus `vcruntime140` / `msvcp140`, no floor |
| C++ ABI selector | `_GLIBCXX_USE_CXX11_ABI` | MSVC toolset line (`_MSC_VER`), `/MD` vs `/MT` |
| extraction | `readelf` / pyelftools | `dumpbin` / pefile |

The ordering predicate therefore degenerates to set-containment over imported DLLs plus an
explicitly recorded toolset and CRT identity, so Windows reuse is closer to requiring equality than
a floor, and `THEROCK_ABI_TARGET` needs a Windows vocabulary rather than a single global string. CRT
linkage is unrecorded at the superproject level: TheRock sets neither `CMAKE_MSVC_RUNTIME_LIBRARY`
nor `/MD`/`/MT`, while pinned submodules do — `rocm-systems@14f81ac4`
`projects/clr/CMakeLists.txt:21,25` uses `/MT`, and `rocm-libraries`
`projects/hipdnn/cmake/Sanitizers.cmake:29` sets `CMAKE_MSVC_RUNTIME_LIBRARY`. Mixing `/MD` and
`/MT` artifacts is a link-time failure. `CMAKE_MSVC_DEBUG_INFORMATION_FORMAT` is already in the
toolchain contribution (`:1737`), which is the precedent to follow.

TheRock's superproject never sets `_GLIBCXX_USE_CXX11_ABI`, so each subproject inherits its
compiler's default, and two hosts with different defaults produce mutually unlinkable artifacts with
nothing recording the difference. Setting it explicitly is out of scope; recording it is not, and
ABI-1 detects it from the output. Windows already receives platform-specific normalization for
deterministic output — `:876-880` adds `LINKER:/Brepro` so `link.exe` and `lld-link` zero PE header
timestamps.

## Rollout phases

| phase | items | gate to the next phase |
|---|---|---|
| 1 | FP-2, FP-3, SMREV-1 | fingerprints move for configure-argument changes; `.smrev` consumers unchanged |
| 2 | INPUTS-2 dump, DRIFT-1 extraction, FP-4 lint | dump reproduces the digest; existing drift test passes plus new CLI tests |
| 3 | FP-1 behind opt-in narrowing, static escape lint, sparse-build comparison | lint clean or waived per submodule; sparse comparison clean for narrowed subprojects |
| 4 | INPUTS-1 manifest, platform-keyed emission, Windows drift job | two build directories produce byte-identical manifests |
| 5 | PRE-1..PRE-5 | same-machine reuse verified; conflict path exercised |
| 6 | CI-1 gitlink expansion | narrow bumps resolve to subproject paths; every anomaly falls back |
| 7 | PORT-1, PORT-2 | ccache on/off and two build directories fingerprint identically |
| 8 | PORT-3, PORT-4, ABI-1, ABI-2 | Q1 settled, or external toolchains declared ineligible |

Phase 7 is cheap to test in isolation and may land sooner if convenient. PORT-3 and the ABI items
are last because their correctness rests on compatibility reasoning rather than an observable
comparison, and because the Windows evidence is weaker than the Linux evidence. FP-1 lands before
PRE-*, so narrowing bakes while fingerprint equality still decides nothing; fingerprints being inert
today is a rollout asset, in that phases 1-4 cannot corrupt a build because nothing consumes their
output. That window closes at phase 5.

An alternative ordering lands PRE-* immediately for the local-rebuild case, accepting that
cross-machine comparison mismatches until phase 8. That is viable only with FP-3 in place. Without
it the check accepts a stale `amd-llvm` after `rocr-runtime` headers change, which converts a
visible staleness problem into an invisible one.

## Verification matrix

| ID | check | expected |
|---|---|---|
| V-1 | commit an edit under `emulation/rocjitsu`, then one under `projects/rocr-runtime` | each moves only its own subproject's fingerprint |
| V-2 | modify a tracked file under `emulation/rocjitsu` without committing | `ROCR-Runtime` stays `"valid": true` |
| V-3 | two subprojects exchange `CMakeLists.txt` contents | fingerprints differ; the dump names both paths (today the digest changes but the dump cannot name them) |
| V-4 | add a `-D` flag, reorder two duplicate `-D` args, change a `_build_env_pairs` entry | fingerprint changes in each case; none do today |
| V-5 | configure with and without a valid `stage.prebuilt` marker | identical fingerprint (INV-2) |
| V-6 | `amd-llvm` referenced paths | contains `rocm-systems/projects/rocr-runtime`; editing a header there moves `amd-llvm`'s fingerprint. It does not today |
| V-7 | ccache on vs off (PORT-1); two build directories (PORT-2); one compiler at two prefixes (PORT-3) | identical fingerprints in each case |
| V-8 | recompute sha256 over the dump's records | reproduces the `fingerprint` field and the verbose-configure `FPRINT` |
| V-9 | `--compare` two dumps from fresh build directories on a clean tree | no differing records; any `kind: generated` entry is a reproducibility bug |
| V-10 | manifest for `core-runtime` | `source_dirs` contains `rocm-systems/projects/rocr-runtime`, not `rocm-systems` or `emulation/rocjitsu` |
| V-11 | two configures into different build directories | byte-identical manifests; `amd-llvm.files` in the tens |
| V-12 | add a `list(APPEND _fprint_files ...)` without regenerating | drift check fails and names the path |
| V-13 | Linux-generated manifest against a Windows stage | refused (INV-6) |
| V-14 | static escape lint on a Linux host | flags `libhsakmt/CMakeLists.txt:379-380` at the pinned revision |
| V-15 | sparse vs full build of `core-runtime` | identical content digests, modes and symlink targets |
| V-16 | unpack two artifacts sharing `amd-comgr` with equal fingerprints, in both orders | identical marker; `COMPONENTS` is the union |
| V-17 | the same with different fingerprints | bootstrap aborts in phase one; no stage tree is written |
| V-18 | truncate a marker mid-record | treated as absent; stage quarantined; build from source |
| V-19 | operator disable, then configure | build is suppressed and the operator record survives (mechanism per Q6) |
| V-25 | build `amd-llvm` under two roots with equal `BOOTSTRAP_ROOT`, fixed input | emitted objects compare equal; unequal roots refuse cross-machine reuse |
| V-26 | descriptor declaring a basedir outside its subproject's stage dir | drift check rejects it (INV-6) |
| V-20 | legacy empty marker | deleted, stage quarantined, built from source; rebuilt marker carries a fingerprint |
| V-21 | mismatch with `THEROCK_PREBUILT_FALLBACK=quarantine` | marker deleted, stage moved aside, no imported file in the rebuilt stage |
| V-22 | narrow submodule bump through CI-1 | resolves to the bumped subproject's stages only |
| V-23 | CI-1 with an unfetchable object id, a `.gitmodules` URL outside the allowlist, or an added submodule | whole source set marked affected in each case |
| V-24 | existing `check_consumer_graph_drift_test.py` after DRIFT-1 | passes unchanged |

V-24 is evidence of behaviour preservation, not proof.
`build_tools/github_actions/tests/check_consumer_graph_drift_test.py:14` imports `check`,
`normalize`
and `write` by name, and DRIFT-1 keeps those bound. The test does not cover the CLI input set,
argument
defaults, diagnostics or malformed input, so CLI-level regression tests land with it.

## Evidence appendix

### Two automated submodule bumps

Both pull requests merged 2026-05-04 from the same automation
(`.github/workflows/bump_submodules.yml`, `cron: "0 */12 * * *"` plus a push trigger). The Actions
runs cited below were created 2026-06-03, a month after the merges, and are later runs on the merged
head SHAs rather than the original merge-time runs. Both report `run_attempt: 1` with an empty
`pull_requests` array, so the API does not establish that they re-execute the same workflow
definition against the same base/head pair — only that they ran the same head commit. They are used
here to show which stages a change of this shape exercises, not to reconstruct the exact cost of the
merge; absolute durations are indicative, since runner fleet and cache state differ by date.

#### Example 1 — `rocm-systems`, where the mapping is too coarse to recover much

TheRock#5011, "Bump rocm-systems from 986a18c to 79e85e1", merged as `c59566b9`. One upstream
commit, `79e85e1468f rocjitsu: Extract scalar execute generators to codegen/execute (#5718)`: 34
files, all under `experimental/rocjitsu/`. At that commit TheRock did not build rocjitsu — `git grep
rocjitsu c59566b9 -- '*/CMakeLists.txt' 'CMakeLists.txt' 'BUILD_TOPOLOGY.toml'` returns nothing; the
subproject was wired up later and the source has since moved to `emulation/`, and it declares
`artifact_deps = []` (`BUILD_TOPOLOGY.toml:863-868`). The bump changes one path, the `rocm-systems`
gitlink; `get_source_set_for_submodule` matches it at `stage_impact.py:201` against
`BUILD_TOPOLOGY.toml:99-101`; `_resolve_artifact_groups` (`:220-226`) returns the 17 groups listing
`rocm-systems`; `_resolve_stages` (`:227-235`) and `_expand_downstream_stages` (`:237-277`) close
over the rest. Actions run 26879008799:

| stage | duration | job |
|---|---|---|
| compiler-runtime (builds LLVM) | 53m09s | 79281831757 |
| math-libs `gfx110X-all` | 5h48m42s | 79295512896 |
| math-libs `gfx94X-dcgpu` | 3h28m01s | 79295512895 |
| math-libs `gfx120X-all` | 2h33m50s | 79295512925 |
| math-libs `gfx1151` | 1h55m26s | 79295512764 |
| comm-libs | 1h49m54s | 79295512519 |
| profiler-apps | 54m42s | 79295512613 |
| dctools-core | 29m05s | 79295512845 |
| iree-compiler | 15m59s | 79295512526 |
| fusilli-libs, debug-tools, foundation, media-libs | 24m02s combined | — |

Plus packaging: DEB 33m05s, RPM 34m33s, Python 10m46s, PyTorch 1h09m13s. Total across the jobs
listed: 21h00m27s of build and packaging machine time, before tests, for code the build did not
compile. The run contains further jobs not enumerated here, including rocblas and rocsolver shards
near an hour each, six hipsparselt shards, and the Windows suite. Enabling `reuse-stage` recovers
little: the change marks 17 of 22 groups affected, only `compiler`, `third-party-sysdeps`,
`third-party-libs`, `cv-libs` and `ml-libs` are untouched, and `compiler` shares the
`compiler-runtime` stage with eight affected groups, so that stage re-runs anyway. This is a
precision
failure, addressed by CI-1 plus INPUTS-1.

#### Example 2 — `rocm-libraries`, where the analysis is right and is not acted on

TheRock#5012, "Bump rocm-libraries from 2384d1b to 89b3fc4", merged as `72e8880f`. Two upstream
commits, both hipBLASLt (`89b3fc4c280`, `1d7f9059ade`), 119 files under `projects/hipblaslt/`.
`rocm-libraries` appears in the `source_sets` of three artifact groups, so the analysis correctly
marks `compiler-runtime`, `comm-libs`, `profiler-apps`, `dctools-core`, `debug-tools`, `media-libs`
and `iree-compiler` unaffected. Actions run 26879007290:

| stage | affected? | duration | job |
|---|---|---|---|
| compiler-runtime (Linux) | no | 49m25s | 79282088606 |
| compiler-runtime (Windows) | no | 45m58s | 79278532874 |
| comm-libs | no | 1h45m39s | 79295502616 |
| profiler-apps | no | 52m02s | 79295502717 |
| dctools-core | no | 24m33s | 79295502554 |
| iree-compiler | no | 11m17s | 79295502753 |
| debug-tools, foundation, media-libs, fusilli-libs | no | 21m55s combined | — |
| math-libs `gfx110X-all` (Linux) | yes | 5h47m21s | 79295502939 |
| math-libs `gfx110X-all` (Windows) | yes | 5h25m58s | 79291255042 |

5h10m49s of unaffected build time across both platforms. This is an enablement failure: acting
on the analysis means trusting that a reused stage's artifacts are current, which is what PRE-1
through PRE-5 supply.

### Churn measurements

Measured at the revisions `fc11b46e0` pins — `rocm-systems@14f81ac4`,
`rocm-libraries@985d8327` — over the 300 commits ending at each pin. A path's value is the count of
distinct `git rev-parse <commit>:<path>` tree ids across those commits; the union row is the count
of
distinct tuples over all eight runtime-core paths together. Reproduce with:

```bash
# runtime-core union: PATHS = projects/{rocr-runtime,clr,amdsmi,rocm-core,rocm-smi-lib,
#                              rocprofiler-register,rocminfo} shared/kpack
git -C <clone> log -300 --format=%H <pin> | while read c; do
  for p in $PATHS; do git -C <clone> rev-parse "$c:$p" 2>/dev/null; done | sha1sum
done | sort -u | wc -l          # drop the inner loop and sha1sum for a single path
```

| `rocm-systems` | / 300 | | `rocm-libraries` | / 300 |
|---|---|---|---|---|
| `emulation/rocjitsu` | 63 | | `projects/hipblaslt` | 66 |
| `projects/rocr-runtime` | 16 | | `projects/composablekernel` | 30 |
| `projects/clr` | 20 | | `projects/rocsparse` | 25 |
| `projects/rocprofiler-sdk` | 13 | | `projects/miopen` | 12 |
| `projects/amdsmi` | 10 | | `projects/rocfft` | 9 |
| `shared/kpack` | 1 | | `projects/rocrand` | 6 |
| `projects/rocm-core` | 1 | | `projects/rpp` | 5 |
| `projects/rocminfo` | 1 | | `projects/rocblas` | 4 |
| `projects/rocm-smi-lib` | 1 | | `shared/tensile` | 3 |
| `projects/rocprofiler-register` | 1 | | `projects/rocprim` | 2 |
| runtime-core union (8 paths) | 40 | | | |
| *whole-submodule HEAD* | *300* | | *whole-submodule HEAD* | *300* |

`emulation/rocjitsu` is the noisiest path in `rocm-systems` — four times `rocr-runtime`. Only
`rocjitsu-hotswap` consumes it (`BUILD_TOPOLOGY.toml:870-873`); the runtime, math, comm and profiler
artifacts do not. Under submodule-granular identity all 63 of those commits disturb every runtime
subproject's fingerprint and re-run its stages. Under FP-1 they disturb `rocjitsu` and
`rocjitsu-hotswap`, which is correct, and nothing else.

The eight runtime-core paths take 40 distinct tuple values across the 300 commits; comparing each
commit against its first parent, 43 commits change at least one of them, so 85.7% of `rocm-systems`
history leaves the runtime core untouched. The two figures answer different questions — how many
distinct states exist, and how often the state moves — and only the second bounds how often a
consumer must rebuild.

In `rocm-libraries`, hipBLASLt at 66 sits alongside rocBLAS at 4, rocPRIM at 2 and rocFFT at 9 — the
quiet subtrees are disturbed roughly sixteen times more often than their own contents change.

TheRock's own tree, last 200 `origin/main` commits, cumulative distinct inputs keys measured the
same way: 20 for `cmake/` + `CMakeLists.txt` + `fetch_sources.py` + `fileset_tool.py` +
`_therock_utils/`; 26 adding `third-party/`; 32 adding `compiler/`; 36 adding `base/`, `core/` and
`BUILD_TOPOLOGY.toml`; 200 for the whole-repo tree. `build_tools/` in full is 107/200, almost
entirely `build_tools/github_actions/` at 66/200, which cannot affect a compiler build.
`BUILD_TOPOLOGY.toml` alone is 8/200.

> These 200-commit figures were taken on a working branch rather than at `fc11b46e0` and have not
> been regenerated; an independent recomputation produced 18/23/29/33 and 106/67. Regenerate them
> against a stated commit before circulation. The ordering they establish is unaffected.

### Granularity ceiling

`blas` is one artifact spanning `hipBLAS`, `hipBLAS-common`, `hipBLASLt`, `hipSOLVER`, `hipSPARSE`,
`rocBLAS`, `rocRoller` and `rocSOLVER` (`build_tools/artifact_subprojects.json`). hipBLASLt's churn
is inside the artifact boundary, so subtree fingerprinting cannot help `blas`: rocBLAS at 4/300 is
bundled with the noisiest subtree in the repository. The win lands on artifacts whose subprojects
sit in quiet subtrees — `fft` (9), `prim` (2), `rand` (6), `rpp` (5), `miopen` (12). Splitting
`blas` is a `BUILD_TOPOLOGY.toml` question and out of scope.
