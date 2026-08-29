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
1. Plus the current time if anything is uncommitted. Archives from a dirty tree
   are not reproducible, which is inherent rather than a limitation here.

This is resolved **once**, at CMake configure time, because it costs several git
invocations and every writer must agree on the value. See
[`THEROCK_SOURCE_DATE_EPOCH`](build_system.md#therock_source_date_epoch) for the
build option. It reaches tools two ways:

- Exported as `THEROCK_SOURCE_DATE_EPOCH` into every subproject build
  environment, so any tool a build invokes sees it — not only the ones that
  write archives.
- Written to `<build_dir>/therock_source_date_epoch.txt` for tools that run
  outside the build graph. `artifact_manager.py push --build-dir` reads it from
  there, since it compresses artifacts long after CMake has finished.

Tools run with no configured build tree at all (a direct `fileset_tool` call)
fall back to resolving it themselves.

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

| Tool              | What changes                                                                                          |
| ----------------- | ----------------------------------------------------------------------------------------------------- |
| GCC, Clang        | `__DATE__` and `__TIME__` are rewritten in every translation unit                                     |
| CPython           | `.pyc` files switch from timestamp to checked-hash invalidation, changing bytes shipped inside wheels |
| setuptools, wheel | zip entry timestamps                                                                                  |
| `dpkg-deb`        | tar entry mtimes are clamped to it (it never raises older ones)                                       |
| Sphinx, Doxygen   | generated copyright/date strings                                                                      |

Those are usually what you want from a fully reproducible build, but they are a
much wider change than stamping archive metadata, and they apply to code TheRock
builds rather than to TheRock itself.

To opt in, at configure time for the build itself:

```bash
cmake -B build -DTHEROCK_EXPORT_SOURCE_DATE_EPOCH=ON ...
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
