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

Orchestrators resolve this **once** — `artifact_manager.py push` and
`build_python_packages.py` — because it costs several git invocations and every
writer must agree on the value. They hand it to workers in
`THEROCK_SOURCE_DATE_EPOCH`.

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

To opt in:

```bash
python build_tools/artifact_manager.py push --export-source-date-epoch ...
python build_tools/build_python_packages.py --export-source-date-epoch ...
```

That sets `SOURCE_DATE_EPOCH` for child processes in addition to
`THEROCK_SOURCE_DATE_EPOCH`. Note this only covers the archiving and packaging
steps; making the compile itself reproducible means setting it for the CMake
build too, which is a separate decision.

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
