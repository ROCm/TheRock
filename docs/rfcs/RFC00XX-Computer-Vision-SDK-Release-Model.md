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

- **Cadence.** CV releases once per Core `<major.minor>` line on
  `stable`. `dev`, `nightly`, `weekly`, and `rc` CV builds track the
  matching Core builds. CV does not share Core's schedule, promotion
  flow, or GA date.
- **Streams.** Same streams as Core (`dev`, `nightly`, `weekly`, `rc`,
  `stable`) with Core's retention rules. No CV-specific streams.
- **Versioning.** CV packages carry the version of the Core release they
  are built against. On `stable` that is the line's `X.Y.0` (e.g.
  `10.2.0`). There is no separate CV version. Component API/ABI versions
  stay independent and are tracked in library SONAMEs.
- **Compatible Core versions.**
  - `stable`: one CV release, built against Core `X.Y.0`, covers every
    Core patch `X.Y.N` without a rebuild. This requires Core patch
    releases to stay ABI-compatible within a `<major.minor>` line
    (section 6).
  - `dev`/`nightly`/`weekly`/`rc`: CV is valid only against the exact
    Core build it was built against, and its package and wheel
    dependencies pin that build.
  - A new Core `<major.minor>` requires a new CV release.
  - CV-only fixes keep the `X.Y.0` build base and bump the package
    revision (native `<pkgrel>`, wheel and tarball `.postN`).

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
every supported gfx target, with no per-target package split (section
6).

**Core dependencies (native).** CV packages declare versioned
dependencies on the Core packages they use, e.g. `amdrocm-runtime10.2`,
`amdrocm-rpp10.2`, `amdrocm-decode10.2`, with a `>= X.Y.0` lower bound.
The `<major.minor>` name suffix keeps resolution within the line.

**Core dependencies (wheels).** The CV wheels (rocAL, rocCV,
rocPyDecode) take their native dependencies from Core's `rocm-sdk-*`
wheels (e.g. `rocm-sdk-core`, `rocm-sdk-libraries`), not from a system
ROCm install. On `stable` they require `>= X.Y.0, < X.(Y+1)`, which
admits every Core patch and `.postN` release in the line (section 6).

**Bundled system dependencies.** Runtime deps the CV libraries need but
Core does not ship (e.g. protobuf, libjpeg-turbo, lmdb, libsndfile,
libtar) ship in a dedicated `amdrocm-vision-sysdeps` package and install
into Core's existing `lib/rocm_sysdeps/lib/` with private SONAMEs, so
they do not collide with host copies. The CV wheels bundle the same deps
with the same private SONAMEs.

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
  dependencies. On `stable`, the overlay for a `<major.minor>` line
  works with every Core patch tarball in that line.

- **Python.** Install the CV wheels from the stream's central `whl/` or
  `whl-next/` index together with Core's `rocm` selector and the device
  extra for the GPU, e.g. `pip install "rocm[libraries,device-gfx942]"`
  plus the CV wheels. The extra is needed because Core libraries CV uses,
  such as rpp, ship device code in `rocm-sdk-device-<target>` wheels.
  Wheels are the supported Python import path; native packages do not
  add CV modules to the host interpreter.

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
  discriminator. Because CV binaries cover every supported target,
  overlays for different `<target>` names carry the same CV code; the
  name pairs each overlay with its Core tarball.

  ```
  therock-dist-vision-linux-<target>-<version>.tar.gz   # CV overlay
  therock-dist-linux-<target>-<version>.tar.gz          # Core SDK
  ```

## 6. Exceptions and deferred support

Explicit departures from RFC0012 and Core packaging policy, and
deliverables deferred from the first release:

- **Compatibility across Core patches.** `docs/packaging/versioning.md`
  treats versions that differ by patch (e.g. `7.14.0` and `7.14.1`) as
  incompatible. CV deliberately depends on any Core patch in its
  `<major.minor>` line so one CV release covers `X.Y.x` without a
  rebuild. This requires Core to commit to ABI compatibility across
  patch releases within a line; Core owners need to confirm that
  commitment.
- **Overlay instead of a self-contained tarball.** RFC0012 gives each
  expansion the same artifact set as `core/`, whose tarball is a
  complete install. The CV tarball is an overlay instead. A
  self-contained Core+CV tarball could be produced without rebuilding
  Core (unpack Core, add CV, repack), but it would pin one exact Core
  patch: it would need re-publishing for every Core patch and would
  duplicate Core's storage for every target and stream. The overlay
  avoids both and is what lets one CV release cover every Core patch
  (section 1). A self-contained tarball can be added later if a
  single-archive install is needed.
- **Fat binaries instead of a per-target split.** Core builds split
  device code into per-target packages and wheels (kpack) by default.
  CV ships fat binaries covering all targets for now. If CV adopts the
  split later, it follows Core's naming
  (`docs/packaging/nativepackage_dependency_tree.md`).
- **Windows.** RFC0012 expects Windows zip and installer artifacts. CV
  is Linux-only today; Windows artifacts and their target matrix will be
  specified when Windows support lands.
- **Linux installers.** No `.run` installer in the first release; users
  install via native packages, the overlay tarball, or wheels.
- **ASAN.** Not produced in the first release. When added, ASAN builds
  are built against Core's ASAN SDK and kept separate from standard
  builds as RFC0012 requires.
