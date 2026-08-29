# Reproducible archives

TheRock's artifact archives are byte-reproducible: the same source, built twice,
produces the same `.tar.zst`. This lets the same generic artifact be uploaded
from parallel CI jobs without one silently overwriting the other with different
content (see [#4202](https://github.com/ROCm/TheRock/issues/4202)).

Two things would otherwise vary between builds of identical content:

- **Member order.** Filesystem iteration order is not stable, so members are
  added sorted.
- **Member metadata.** `tar` records the build time and the building user's
  uid/gid. These are replaced by `normalize_tarinfo()` in
  [`build_tools/_therock_utils/archive_util.py`](../../build_tools/_therock_utils/archive_util.py).

Permissions, sizes, symlink targets and hardlink identity are preserved.

## The timestamp

Every member gets the same mtime, resolved from the source rather than the
clock. It is deliberately **not** zero: extraction restores mtime — Python's
`tarfile.extract()`, `extractall(filter="tar")`, `extractall(filter="data")` and
`tar -x` all do, for root and non-root alike — so an epoch timestamp would reach
installed SDKs and make freshly upgraded headers look older than objects a
downstream project already built against the previous version. Timestamp-driven
build systems would then skip work that needed doing.

`build_tools/_therock_utils/source_date.py` resolves it:

```bash
python build_tools/_therock_utils/source_date.py --explain
```

1. TheRock's `HEAD` commit time. The superproject pins every submodule, so its
   HEAD identifies the whole source state. An artifact has no single source
   submodule to attribute to: a merged `dist/` tree spans a subproject plus its
   runtime deps, and several subprojects (zlib, zstd, bzip2, elfutils) are
   tarballs downloaded from S3 with no git history at all.
1. Plus the `HEAD` time of any submodule checked out past its pin, which is
   normal when developing against subproject sources in place.
1. Plus the mtime of the most recently touched uncommitted file, if the tree is
   dirty — not the current time, so that an untouched dirty tree resolves to the
   same value every time it is asked. Everything is combined with `max()`, so a
   file touched to an old date cannot drag the result back before its commit.

Each tool that writes archives resolves this **once at its own entry** —
`artifact_manager.py push`, `build_python_packages.py`, `build_tarballs.py` —
and passes it to everything it spawns. That is deliberate: these run *after* the
build, so resolving there describes the source being packaged. Resolving at
CMake configure instead would freeze a value that goes stale as soon as anything
is pulled, and the build writes no archives of its own.

The build can still be pinned, via
[`THEROCK_SOURCE_DATE_EPOCH`](build_system.md#therock_source_date_epoch), but
that is a separate opt-in concerned with making the *compilers* reproducible.

Consequence worth knowing: a docs-only commit advances the timestamp for every
artifact, so archive hashes change even when content is byte-identical. That is
fine for detecting conflicting uploads within a run, but archive hashes cannot
be used to dedupe across commits.

## `SOURCE_DATE_EPOCH` and its blast radius

`SOURCE_DATE_EPOCH` is honored, and outranks `THEROCK_SOURCE_DATE_EPOCH` — if it
is set, someone decided that deliberately.

TheRock does **not** set it for you, and that is on purpose. It is a
cross-ecosystem convention that many tools pick up automatically, so exporting
it changes far more than archive metadata:

| Tool                      | What changes here                                                                                                                                                             |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CMake `string(TIMESTAMP)` | Pinned. Mostly cosmetic copyright years, but `ROCKE_ENGINE_VERSION` embeds a build date into a shipped version string deliberately, to keep successive builds distinguishable |
| Doxygen                   | `HTML_TIMESTAMP`/`LATEX_TIMESTAMP` pinned, including the hip docs target `clr/hipamd/packaging` runs in `ALL`                                                                 |
| setuptools, wheel         | zip entry timestamps — see below, this is the one place opting in is load-bearing                                                                                             |
| GCC, Clang                | `__DATE__`/`__TIME__` rewritten. Near-harmless: `rocm-systems` has no real uses; those in `rocm-libraries` and `openmp` sit behind default-off flags                          |
| binutils                  | `ar.exp` **requires the variable to be unset** and fails if rocgdb's tests run with it exported                                                                               |
| `dpkg-deb`                | Clamps tar mtimes downward only, so it never repairs a stale timestamp                                                                                                        |
| CPython                   | `.pyc` invalidation switches to checked-hash. No component byte-compiles during the build, so this only matters at `pip install` time                                         |
| `rccl` changelog          | **Not** affected — shells out to `date -R`, which ignores the variable                                                                                                        |

Also note that pinning the build puts the value in every subproject's build
command, so the tree rebuilds whenever it changes.

To opt in, at configure time for the build itself:

```bash
cmake -B build -DTHEROCK_SOURCE_DATE_EPOCH=$(python build_tools/_therock_utils/source_date.py) ...
```

or for the archiving and packaging steps, which run outside the build graph:

```bash
python build_tools/artifact_manager.py push --export-source-date-epoch ...
python build_tools/build_python_packages.py --export-source-date-epoch ...
```

The wheel workflows expose it as an `export_source_date_epoch` input, off by
default, on both `workflow_dispatch` and `workflow_call`:

- `.github/workflows/build_portable_linux_python_packages.yml`
- `.github/workflows/build_windows_python_packages.yml`

This is the one place where opting in is doing more than hardening metadata.
`archive_util` only writes the `_devel.tar.xz` **inside** the
`rocm-sdk-devel` wheel; the `.whl` zip itself and the `rocm` sdist are produced
by `setup.py bdist_wheel`/`sdist`, and setuptools only makes them reproducible
when `SOURCE_DATE_EPOCH` is set. That path compiles nothing, so most of the
blast radius above does not apply to it.

## What this does not do

Reproducibility and timestamp-based rebuild correctness want opposite things
from a single mtime field, and no stored value satisfies both. A source commit
date can still predate a user's earlier build — a release branch cut weeks
before publication, or a user moving from a newer nightly to an older stable
release — in which case installed headers still look older than their existing
objects.

Guaranteeing *"a newly installed SDK file is newer than anything built against
the previous one"* requires re-stamping at install time, per distribution
channel. That is not done today.
