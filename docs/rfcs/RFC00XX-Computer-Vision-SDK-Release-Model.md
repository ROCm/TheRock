---
author: Marco Grond (marco-grond)
created: 2026-06-04
modified: 2026-11-12
status: draft
---

# RFC00XX: AMD Computer Vision SDK release model

## Related RFCs

- **RFC0009** — OS packaging requirements: `amdrocm-<package>` naming,
  the `standard`/`asan` package types, and supported distros.
- **RFC0012** — repository structure: stream subdomains, the per-stream
  `rocm/` tree, and the artifact set each expansion publishes. Names
  `computer-vision-sdk/` as a top-level peer of `core/`.
- **RFC00XX (Repository Package)** — the `amdrocm-repo-*` tier packages
  and the repo configuration they install.
- **RFC00XX (ROCm SDK archive packaging)** — not yet landed. CV tarball
  names follow the currently published Core tarball names and will be
  realigned if that RFC lands with a different scheme.
- **RFC00XX (HPC SDK)** — also built on the ROCm Core SDK; the CV SDK
  follows the same structure.

## Overview

The AMD Computer Vision (CV) SDK is a ROCm expansion built **against a
released ROCm Core SDK** from its own repository, `rocm-vision`. It
consumes a published Core SDK (tarball, packages, or wheels) and never
rebuilds or re-publishes Core content. It is opt-in: the default
`amdrocm` meta-package does not pull it in.

Anything not specified here (stream definitions, install prefixes,
package naming, repo layout) is inherited from the Core SDK via RFC0009
and RFC0012.

## 1. Release policy

- **Cadence.** CV releases for every Core `<major.minor>` release, built
  against that Core release. `dev` and `nightly` CV builds track the
  matching Core builds. CV does not share Core's schedule, promotion
  flow, or GA date.
- **Streams.** Same streams as Core (`dev`, `nightly`, `weekly`, `rc`,
  `stable`) with Core's retention rules. No CV-specific streams.
- **Versioning.** CV packages carry the version of the Core release they
  are built against (`<X.Y>.<N>`, e.g. `10.2.0`). There is no separate
  CV version. Component API/ABI versions stay independent and are
  tracked in library SONAMEs.
- **Compatible Core versions.**
  - `rc`/`stable`: a CV release built against Core `X.Y.M` is valid for
    any Core `X.Y.N` with `N >= M`, so a Core patch release does not
    require a CV release. This relies on Core keeping ABI compatibility
    across patch releases within a `<major.minor>` line; without that
    commitment, CV re-releases for every Core patch.
  - `dev`/`nightly`/`weekly`: CV is valid only against the exact Core
    build it was built against, and its package and wheel dependencies
    pin that build.
  - A new Core `<major.minor>` requires a new CV release.
  - CV-only fixes are rebuilt against the latest Core patch in the line
    and take that version. If CV has already published that version,
    the fix bumps the package revision (native `<pkgrel>`, wheel
    `.postN`).

## 2. Deliverables

Linux only, for the same targets, distros, and streams as the Core SDK.
There is no CV-specific matrix. Only the `standard` package type
(RFC0009) is produced.

| Format                     | First release           |
| -------------------------- | ----------------------- |
| Native packages (deb, rpm) | Required                |
| Overlay tarball            | Required (sections 4–6) |
| Python wheels              | Required                |
| Windows zip / installers   | Deferred (section 6)    |
| Linux installers           | Deferred (section 6)    |
| ASAN variant               | Deferred (section 6)    |

## 3. Contents and dependencies

**Components** (first release, ROCm 10.2): MiVisionX, rocAL, rocCV,
rocPyDecode. The CV SDK team owns this list and reviews it each release.

**Meta-packages.** Mirroring Core's runtime/devel/test split (as
`amdrocm-hpc`, `amdrocm-hpc-sdk`, `amdrocm-hpc-tests` do):

- `amdrocm-vision<X.Y>` — all CV runtime components.
- `amdrocm-vision-sdk<X.Y>` — development content (headers, CMake
  config).
- `amdrocm-vision-tests<X.Y>` — test content.

Each has an unversioned alias (e.g. `amdrocm-vision`) that depends on
the newest versioned package, following Core's convention
(`amdrocm-blas` → `amdrocm-blas8.2`). None is a dependency of the
default `amdrocm` meta-package.

**Component packages** use Core's naming. The suffix is the ROCm
`<major.minor>`, not the component's API version:

```
amdrocm-<component><major.minor>-<version>-<pkgrel>.<arch>.<ext>
e.g. amdrocm-rocal10.2-10.2.0-1.x86_64.rpm
```

They are architecture-agnostic fat binaries carrying code objects for
every supported gfx target, with no per-target package split. If Core
makes its kpack host/device split
(`docs/packaging/nativepackage_dependency_tree.md`) the default for
component packages, CV adopts it.

**Core dependencies (native).** CV packages declare versioned
dependencies on the Core packages they use, e.g. `amdrocm-runtime10.2`,
`amdrocm-rpp10.2`, `amdrocm-decode10.2`, with a `>= X.Y.M` lower bound
for the Core release they were built against. The `<major.minor>` name
suffix keeps resolution within the line.

**Core dependencies (wheels).** The CV wheels (rocAL, rocCV,
rocPyDecode) take their native dependencies from Core's `rocm-sdk-*`
wheels (e.g. `rocm-sdk-core`, `rocm-sdk-libraries`), not from a system
ROCm install. They require `>= X.Y.M, < X.(Y+1)`, which also admits the
`.postN` releases permitted by `docs/packaging/versioning.md`.

**Bundled system dependencies.** Runtime deps the CV libraries need but
Core does not ship (e.g. protobuf, libjpeg-turbo, lmdb, libsndfile,
libtar) ship in a dedicated `amdrocm-vision-sysdeps` package and install
into Core's existing `lib/rocm_sysdeps/lib/` with private SONAMEs, so
they do not collide with host copies.

## 4. Installation

- **Native packages.** CV installs into the standard Core prefix (e.g.
  the `core-<major.minor>` tree), with no `vision-` prefix or separate
  tree. Use the version-qualified name to match the installed Core:

  ```
  apt install amdrocm-vision10.2
  dnf install amdrocm-vision10.2
  ```

- **Overlay tarball.** Contains only CV content, laid out like the Core
  SDK. Extract the Core SDK tarball first, then extract the CV overlay
  into the same prefix; without Core, the CV libraries have unresolved
  dependencies. Use the newest CV overlay whose version is `<=` the Core
  version within the same `<major.minor>`.

- **Python.** `pip install` the CV wheels from the stream's central
  `whl/` or `whl-next/` index; pip resolves the Core `rocm-sdk-*` wheels
  from the same index. Wheels are the supported Python import path;
  native packages do not add CV modules to the host interpreter.

## 5. Publication

- **Location.** Packages, the overlay tarball
  (`computer-vision-sdk/tarball/`), and wheels are published under the
  top-level `computer-vision-sdk/` folder of each stream subdomain's
  `rocm/` tree (RFC0012), a peer of `core/` and not under `core/` or
  `extras/`. The stream's central `whl/` and `whl-next/` indices
  reference the CV wheels.

- **Repo access.** Installing the standard `amdrocm-repo-<tier>` package
  must be enough to install `amdrocm-vision<X.Y>` and its Core
  dependencies. How the tier packages expose `computer-vision-sdk/` is
  defined by the Repository Package RFC.

- **Tarball naming.** The Core tarball name plus a `vision`
  discriminator:

  ```
  therock-dist-vision-linux-<target>-<version>.tar.gz   # CV overlay
  therock-dist-linux-<target>-<version>.tar.gz          # Core SDK
  ```

## 6. Exceptions and deferred support

Explicit departures from RFC0012, and deliverables deferred from the
first release:

- **Overlay instead of a self-contained tarball.** RFC0012 gives each
  expansion the same artifact set as `core/`, whose tarball is a
  complete install. The CV tarball is an overlay instead. A
  self-contained Core+CV tarball could be produced without rebuilding
  Core (unpack Core, add CV, repack), but it would pin one exact Core
  patch: it would need re-publishing for every Core patch and would
  duplicate Core's storage for every target and stream. The overlay
  avoids both and is what lets one CV release cover later Core patches
  (section 1). A self-contained tarball can be added later if a
  single-archive install is needed.
- **Windows.** RFC0012 expects Windows zip and installer artifacts. CV
  is Linux-only today; Windows artifacts and their target matrix will be
  specified when Windows support lands.
- **Linux installers.** No `.run` installer in the first release; users
  install via native packages, the overlay tarball, or wheels.
- **ASAN.** Not produced in the first release. When added, ASAN builds
  are built against Core's ASAN SDK and kept separate from standard
  builds as RFC0012 requires.
