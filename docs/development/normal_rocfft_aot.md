# I-029 private rocFFT AOT transactions in normal CI

This is a default-off product support layer for the controlled normal workflow.
The workflow selects it explicitly with default-off `profile_normal_product`. The source patch is applied identically
in both arms; only the helper runtime transaction setting differs. Its scope is
Linux `math-libs`, exactly `gfx103X-all` (gfx1030 through gfx1036). Windows,
gfx94X, other stages and ordinary builds retain their original behavior.

The motivating four-run diagnostic used all seven original growing-cache helpers
and observed a 46.49% lower mean math-stage wall time, with 4.98% more aggregate
stage CPU, in two independent allocations per arm. Those measurements do not
validate this reduced-instrumentation support layer or the normal full workflow.
The normal comparison must establish its own source/tool/cache/output parity.
Related diagnostic runs: [A1](https://github.com/ROCm/TheRock/actions/runs/41207159659),
[B1](https://github.com/ROCm/TheRock/actions/runs/41209712996),
[B2](https://github.com/ROCm/TheRock/actions/runs/41211415109), and
[A2](https://github.com/ROCm/TheRock/actions/runs/41213215567).

## Source and runtime contract

Apply only after the normal source fetch has completed, before normal source-input
capture and configuration. The required rocm-libraries HEAD is
`f4e24d58b3f313f1a31068def1e35376e43c2f49`. `git apply` changes only three host
files; a separately declared header is copied into `projects/rocfft/shared/`.
There is no source commit, branch switch, history rewrite, gitlink change,
architecture change, compiler option or generator input change. Both arms need
exactly the same patch and header declarations. Preserve those source receipts
in the normal common input comparison, including the copied untracked header.

The selected job's normal configuration must produce all seven original AOT
commands and their sequential dependency edges. The helper argv, target names,
worker policy, output paths and final copy/cleanup calls are unchanged. The
controller checks generated Ninja commands and edges after the build.

Before the build, prepare exclusively creates an owned directory beside the
original temporary DB, a random 32-hex ownership marker, an empty receipt directory,
and a fresh private COMGR directory on the same filesystem. Existing directories,
temporary DBs or shipped cache outputs are rejected, never cleared or adopted.
The seven original helpers share one growing private SQLite DB. Both arms use
`AMD_COMGR_CACHE=1`, the same fresh-private-cache layout, and the pinned default
COMGR policy (an explicitly empty `AMD_COMGR_CACHE_POLICY`). This is a separate,
explicit contract from ccache's CI-readonly or private-cold regime. It does not
claim to reproduce a stock persistent COMGR cache. Actual compiler command keys,
hits/misses, SDK provider hashes and source metadata remain common-controls gates.

Baseline uses `ROCFFT_AOT_BATCH_TRANSACTION=0`. Candidate uses `all`, beginning
one `BEGIN IMMEDIATE` per architecture helper before its workers and committing
after they join, before the original shipped-cache copy. Neither changes WAL,
FULL synchronous mode, busy timeout, checkpoint policy or runtime cache durability.
Custom caller paths and normal runtime caches remain outside this opt-in.

The source receipt checks original generator input files and CMake commands.
The end-only helper receipt records ownership, actual COMGR controls, cache
existence, SQLite policy and successful BEGIN/COMMIT/output completion. Only
subprocess exception paths increment a bounded fallback error record; there are
no per-kernel success timers, reporter threads, phase counters or generated-source
logs. A fallback invalidates the comparison rather than changing fallback policy.

## Integration calls and gates

The normal workflow supplies the existing common nonsecret environment:
`GITHUB_ACTIONS=true`, `GITHUB_REPOSITORY=ROCm/TheRock`, `NORMAL_VALIDATION=true`,
`STAGE_NAME=math-libs`, `AMDGPU_FAMILIES=gfx103X-all`, `RELEASE_TYPE=ci`, plus the
current run ID/attempt. Use the same case for all calls in a job:

```bash
python build_tools/normal_rocfft_aot.py apply --enable --case baseline
# Run normal configure and its normal source/tool input capture in the reviewed order.
python build_tools/normal_rocfft_aot.py prepare --enable --case baseline --github-env
# In a SUBSEQUENT Actions step, run the complete normal stage build.
python build_tools/normal_rocfft_aot.py validate --enable --case baseline
```

For candidate change only `--case candidate`. `--github-env` appends known
nonsecret controls literally to Actions' `GITHUB_ENV`; they affect subsequent
steps, not the already-running shell. Without `--github-env`, the controller
saves `runtime-environment.json` for an explicit caller. It never evaluates shell
text. Every invocation without `--enable` is a no-op. An enabled invocation
outside the selected stage/family is also a no-op after repository/mode guards.

Later stage/install/publication continuation must depend on the successful build
**and the current validation command's successful exit**, with the current case,
run and attempt. Do not accept mere presence of a stale `validation.json`.
Always retain the diagnostic directory on failure; a failed build may have only
partial receipts and is not accepted. No source modification in this layer
changes workflow destinations or authorizes release publication.

Validation requires exactly seven complete receipts, no fallback errors,
unchanged WAL/FULL/autocommit policy, 17,204 entries per architecture, the pinned
generator checksum, SQLite quick checks, and 120,428 entries in the final growing
DB. It retains every key/code-blob hash, raw shipped DB hashes, helper hashes,
generated commands and dependencies, and complete private COMGR filename/size
metadata without the generic 100,000-entry cap. COMGR metadata is not content
hashing; compare shipped kernel payloads and actual SDK fingerprints separately.
The validator does not execute GPU code or replace numerical/component tests.

The copied-header declaration must be scoped to `math-libs` **and** `gfx103X-all`;
gfx94X has a different retained architecture set. Patch/header declarations must
use committed LF Git-blob hashes, matching Linux checkout/application bytes.
Register the patch as a tracked input and the copied target as an explicitly
hashed source input in both cases. The common-controls integration registers these inputs with product, stage and
family selectors; changing the source payload requires updating those exact hashes.

## Host/source checks

The fixture builds a tiny C++ program against real SQLite 3.51.3, not ROCm. It
checks FULL durability configuration, transaction visibility, denied COMMIT with
rollback before output copy, seven growing-cache outputs, ownership alias rejection,
legacy custom-path no-op behavior and end receipts. Python checks reject changed
run identity, fallback/commit failures, changed target sets and preexisting outputs.
A synthetic 100,005-entry inventory catches accidental reuse of a 100k cap.

```bash
export ROCFFT_PROFILE_TEST_CXX='["g++"]'
export ROCFFT_AOT_TEST_SQLITE_DIR=/path/to/sqlite-3510300
export ROCFFT_AOT_TEST_SQLITE_LIB=/path/to/sqlite3.o
# Optional: exact-f4 selected-source fixture repository (see below).
export ROCFFT_AOT_TEST_SOURCE_FIXTURE=/path/to/normal-rocfft-product-source
python3 -W error::ResourceWarning -m unittest build_tools.tests.normal_rocfft_aot_test -v
```

The source fixture contains original f4 host files, original AOT CMake and all 69
parsed generator checksum inputs, committed locally only to make a disposable Git
fixture. Its fixture HEAD is not product provenance. The test clones it with LF
checkout bytes, checks apply against those exact original files, preserves fixture
HEAD/branch/history, and rejects a mutated generator input. The CLI always requires
the real f4 product HEAD; only the internal host fixture can supply another expected
HEAD. Tests that lack explicitly declared host dependencies report a skip, not a
ROCm correctness pass. Normal end-to-end CI and numerical/artifact checks remain
pending for this support layer. No upstream product PR is created by this local
experiment; work tracking is this session's I-029 evidence and run links above.
