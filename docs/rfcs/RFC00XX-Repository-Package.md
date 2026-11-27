---
author: Saad Rahim (saadrahim)
created: 2026-06-09
status: draft
---

# ROCm repository-setup packages (`amdrocm-repo-*`)

## Related RFCs

- **RFC0012 — repo.amd.com folder hierarchy**
  ([ROCm/TheRock#4414](https://github.com/ROCm/TheRock/pull/4414))
  — defines the stream subdomains (`<stream>.repo.amd.com`), the
  per-stream folder tree, and the `amdrepos/` singleton folder on
  the bare `repo.amd.com` domain where the packages defined here
  are published. This RFC depends on RFC0012's *Stream Subdomains*
  and *Repository Structure* sections; it does not redefine them.
- **RFC0009 — native packaging conventions** — source of the
  `amdrocm<major>-<project>` package-naming convention used by the
  packages this RFC installs.

## Overview

Allow users to install ROCm repositories via convenient repo-setup
packages. Each package provides repository definitions for the
streams it covers — rpm repo files are installed under
`/etc/yum.repos.d/`, deb822 source files under
`/etc/apt/sources.list.d/`. Each package also installs (or shares)
the AMD GPG key (prompting the user to accept on install).
Updating a package refreshes both its repo definitions and the
GPG key.

The repo packages are published as the singleton `amdrepos/`
folder on the bare `repo.amd.com` domain — i.e.
`https://repo.amd.com/amdrepos/<distro>/` — per RFC0012. The
packages themselves are stream-agnostic at install time
(downloaded from the bare domain); each installed file points at
the relevant `<stream>.repo.amd.com` subdomain via its `baseurl`.

## Per-tier packages — no opt-in to lower-quality streams from a higher-quality package

The repo-setup packages are split along **quality-and-cadence tier
boundaries**, with the invariant that **installing a tier package
never registers a lower-tier stream's URL on the box, not even
`Enabled: no`**. This is a deliberate guard: a stable user
installing `amdrocm-repo-stable` must not end up with a disabled
`nightly` or `dev` stanza on disk that an `dnf --enablerepo=...`
flip (or a misconfigured automation step) could accidentally
activate. The only way to opt in to a lower-quality stream is to
*install the package for that tier* — an explicit, audit-visible
action.

A single "covers every stream" package was considered and rejected
because it broke this invariant, embedded per-stream driver pins
that churned the package even for users on a different stream, and
registered dev/nightly URLs (even if `Enabled: no`) on production
hosts that some regulated environments reject on principle.

The tier packages are:

| Package                      | Streams covered                                            | Cadence                                          |
| :--------------------------- | :--------------------------------------------------------- | :----------------------------------------------- |
| `amdrocm-repo-stable`        | `stable` (+ `stable-rpath` sibling, disabled by default)   | Per stable release                               |
| `amdrocm-repo-stablerc`      | `rc`                                                       | Per rc bump                                      |
| `amdrocm-repo-nightly`       | `nightly`, `weekly`                                        | Per weekly cut; nightly driver-pin bumps         |
| `amdrocm-repo-dev`           | `dev`                                                      | Per-commit-ish (tracks develop-branch builds)    |

Each tier ships in both families:

- **rpm-family** (RHEL, SLES, Azure Linux, …) — `amdrocm-repo-stable.rpm`,
  `amdrocm-repo-stablerc.rpm`, `amdrocm-repo-nightly.rpm`,
  `amdrocm-repo-dev.rpm`.
- **deb-family** (Debian, Ubuntu) — `amdrocm-repo-stable.deb`,
  `amdrocm-repo-stablerc.deb`, `amdrocm-repo-nightly.deb`,
  `amdrocm-repo-dev.deb`.

**LTS (`lts.repo.amd.com`, `ltsrc.repo.amd.com`) placement is
deferred.** Whether the LTS family ships as its own
`amdrocm-repo-lts` + `amdrocm-repo-ltsrc` pair, folds into a
renamed `amdrocm-repo-stable`, or takes some other shape is to be
decided when LTS actually goes live. Until then, the
`lts.repo.amd.com` and `ltsrc.repo.amd.com` subdomains are
reserved (per RFC0012 *Stream Subdomains*) but no repo-setup
package references them — `amdrocm-repo-stablerc` covers `rc`
only.

## Installed repo file names

Each tier package installs its repo definitions with deterministic
filenames so the on-disk presence of a stream is auditable
(`ls /etc/yum.repos.d/` or `ls /etc/apt/sources.list.d/` makes
each enabled tier visible without parsing file contents). The
`[section]` name inside an rpm `.repo` file matches the filename
stem; the deb822 `.sources` filename matches the same stem so
both package managers present a consistent listing.

**rpm — `/etc/yum.repos.d/`:**

| Tier package              | Installed file(s)                                                | `[section]` name(s)                          |
| :------------------------ | :--------------------------------------------------------------- | :------------------------------------------- |
| `amdrocm-repo-stable`     | `amdrocm-stable.repo`<br>`amdrocm-stable-rpath.repo` *(disabled)*<br>`amdrocm-amdrepos.repo` *(self-update)* | `amdrocm-stable`<br>`amdrocm-stable-rpath`<br>`amdrocm-amdrepos` |
| `amdrocm-repo-stablerc`   | `amdrocm-stablerc.repo`                                          | `amdrocm-stablerc`                           |
| `amdrocm-repo-nightly`    | `amdrocm-nightly.repo`<br>`amdrocm-weekly.repo`                  | `amdrocm-nightly`<br>`amdrocm-weekly`        |
| `amdrocm-repo-dev`        | `amdrocm-dev.repo`                                               | `amdrocm-dev`                                |

**deb — `/etc/apt/sources.list.d/`:**

deb822 `.sources` files do not have a literal `[section]` header
like rpm `.repo` files. Two stanza fields serve the same
identifying role and are pinned by this RFC for consistency with
the rpm side:

- **`X-Repo-Id:`** — a non-standard but widely tolerated header
  (apt ignores unknown `X-*` fields) used as the
  human-readable stanza identifier. Set equal to the filename
  stem so the rpm `[section]` ↔ deb stanza identifier mapping is
  1:1.
- **`Suites:`** — required by deb822. Set to the literal stream
  name (`stable`, `stablerc`, `nightly`, `weekly`, `dev`) so
  `apt`-side tooling that filters by suite (e.g. `apt-cache
  policy`) shows the stream cleanly.

| Tier package              | Installed file(s)                                                                  | `X-Repo-Id:` / `Suites:` per stanza                                  |
| :------------------------ | :--------------------------------------------------------------------------------- | :------------------------------------------------------------------- |
| `amdrocm-repo-stable`     | `amdrocm-stable.sources`<br>`amdrocm-stable-rpath.sources` *(rpm-only today — file omitted in deb)*<br>`amdrocm-amdrepos.sources` *(self-update)* | `amdrocm-stable` / `stable`<br>*(n/a — file not shipped)*<br>`amdrocm-amdrepos` / `amdrepos` |
| `amdrocm-repo-stablerc`   | `amdrocm-stablerc.sources`                                                         | `amdrocm-stablerc` / `stablerc`                                      |
| `amdrocm-repo-nightly`    | `amdrocm-nightly.sources`<br>`amdrocm-weekly.sources`                              | `amdrocm-nightly` / `nightly`<br>`amdrocm-weekly` / `weekly`         |
| `amdrocm-repo-dev`        | `amdrocm-dev.sources`                                                              | `amdrocm-dev` / `dev`                                                |

Example `amdrocm-stable.sources` stanza shipped by
`amdrocm-repo-stable` on Ubuntu 24.04:

```deb822
X-Repo-Id: amdrocm-stable
Types: deb
URIs: https://stable.repo.amd.com/rocm/core/packages/ubuntu2404/
Suites: stable
Components: main
Signed-By: /usr/share/keyrings/amdrocm-keyring.gpg
Enabled: yes
```

Example `amdrocm-nightly.sources` stanza shipped by
`amdrocm-repo-nightly` (disabled by default per the *Default
enablement* table below):

```deb822
X-Repo-Id: amdrocm-nightly
Types: deb
URIs: https://nightly.repo.amd.com/rocm/core/packages/ubuntu2404/
Suites: nightly
Components: main
Signed-By: /usr/share/keyrings/amdrocm-keyring.gpg
Enabled: no
```

Rules:

- **One file per stream per tier.** Even when a tier covers
  multiple streams (e.g. `amdrocm-repo-nightly` covers both
  `nightly` and `weekly`), each stream gets its own file so the
  user can `Enabled: yes/no` (or `enabled=1/0`) them
  independently without touching siblings.
- **No templating across files.** Each file pins exactly one
  stream's `baseurl` (`https://<stream>.repo.amd.com/...`). The
  `$amdrocm_release_stream` variable is used only for the
  *default-stream selection* inside the tier (e.g. when the tier
  ships multiple streams), not to make a single file
  multi-purpose.
- **`amdrocm-amdrepos`** is the self-update repo (points at
  `https://repo.amd.com/amdrepos/<distro>/`) so `yum update` /
  `apt upgrade` picks up new versions of the tier package itself.
  It is installed by **every** tier package — but the file is
  identical across tiers, so co-installing tiers does not produce
  conflicting copies (rpm/deb file-conflict on identical contents
  is benign).
- **Filename uniqueness across tiers.** The `amdrocm-*` prefix
  plus the stream suffix guarantees that two tier packages
  installed side-by-side never write to the same filename.
  `amdrocm-repo-stable` and `amdrocm-repo-nightly` co-installed
  produce `amdrocm-stable.repo` + `amdrocm-nightly.repo` +
  `amdrocm-weekly.repo` (+ shared `amdrocm-amdrepos.repo`); no
  overwrite, no conflict.
- **Default enablement.** Within a tier, the table below shows
  the default `enabled` state. The intent: the "primary" stream
  for the tier is enabled, secondary streams default to disabled
  so the user picks them explicitly.

  | Tier package              | Enabled by default                           | Disabled by default                  |
  | :------------------------ | :------------------------------------------- | :----------------------------------- |
  | `amdrocm-repo-stable`     | `amdrocm-stable`, `amdrocm-amdrepos`         | `amdrocm-stable-rpath`               |
  | `amdrocm-repo-stablerc`   | `amdrocm-stablerc`, `amdrocm-amdrepos`       | —                                    |
  | `amdrocm-repo-nightly`    | `amdrocm-weekly`, `amdrocm-amdrepos`         | `amdrocm-nightly`                    |
  | `amdrocm-repo-dev`        | `amdrocm-dev`, `amdrocm-amdrepos`            | —                                    |

  Rationale for `amdrocm-repo-nightly` defaulting to `weekly`
  enabled and `nightly` disabled: most downstream CI consumers
  want the weekly cadence and only enable `nightly` for
  per-day investigation. Users who actually want `nightly` as
  the default flip the enabled flags after install
  (`dnf config-manager --set-enabled amdrocm-nightly
  --set-disabled amdrocm-weekly`).

## Concurrent stream installation

Per RFC0012 *Concurrent stream installation*, each package
installs one repo definition per `<stream>` it covers (rpm:
distinct `.repo` files; deb: distinct `deb822` `.sources`
stanzas). `$amdrocm_release_stream` (rpm) and the deb822 `Enabled:
yes/no` field (deb) pick the active default within an installed
tier; additional streams *within the same tier* are opted into
with `dnf --enablerepo=...` or by flipping `Enabled:`. Crossing a
tier boundary requires installing the other tier's package —
there is no `--enablerepo=...` shortcut that exposes a stream the
user did not opt into via package install. Tiers compose: a user
wanting `stable` + `weekly` side-by-side installs both
`amdrocm-repo-stable` and `amdrocm-repo-nightly`. The four tier
packages do **not** conflict with each other.

## Stable rpath variant

The `amdrocm-repo-stable.rpm` package ships an additional
`amdrocm-stable-rpath.repo` file, **disabled by default**,
pointing at the rpath subpath under `stable.repo.amd.com`
(`stable.repo.amd.com/rocm/core/packages/<distro>-rpath/`). Users
who want the rpath build enable that repo
(`dnf config-manager --set-enabled amdrocm-stable-rpath`) and `dnf
install rocm` then resolves rpath-built packages. The rpath build
installs to a distinct `/opt/rocm/core-<X.Y>-rpath` directory (per
RFC0012 *Install Locations*) so it coexists on disk with the
standard build. This does **not** violate the "no opt-in to
lower-quality streams" rule: rpath is a build variant of `stable`
with the same QA bar — not a lower-quality stream — so shipping
it as a disabled-by-default sibling inside the stable tier is
consistent with the invariant. **Rpath is rpm-only today**, so
the deb `amdrocm-repo-stable` package simply omits this stanza —
there is no `amdrocm-repo-rpath` package, and rpath is
intentionally scoped to the stable tier (`amdrocm-repo-stable`)
only.

## Conflict with `amdgpu-install`

The `amdgpu-install` package provides the equivalent repo-setup
functionality on legacy ROCm releases (it writes the same
`/etc/yum.repos.d/` and `/etc/apt/sources.list.d/` files and
registers the AMD GPG key). Only the stable-tier package,
`amdrocm-repo-stable`, supersedes that role for ROCm 7.14 and
above and **conflicts** with `amdgpu-install` — the two cannot
be installed simultaneously because they manage overlapping repo
files and key entries. `amdrocm-repo-stable` declares
`Conflicts:`/`Breaks:` against `amdgpu-install` so the package
manager surfaces the collision at install time and the user
removes one before adding the other. `amdrocm-repo-stablerc`,
`amdrocm-repo-nightly`, and `amdrocm-repo-dev` install repo
files with distinct names and do not collide with `amdgpu-install`
— no conflict declaration is required on those tiers. When the
LTS repo-package decision is made, the chosen package will
inherit the same `Conflicts:` rule against `amdgpu-install`.

## GPG key handling

Each tier package is responsible for placing the AMD signing key on
disk and wiring its repo definitions to that key. Install **must
prompt the user to accept the key** — the key is never silently
trusted. Upgrades of the tier package refresh both the installed
repo definitions and the key file, so a normal `yum update` /
`apt upgrade` cycle is sufficient to roll forward to a rotated key
without manual intervention.

**rpm-family** (RHEL, SLES, Azure Linux, …):

- The key is installed at a fixed path, e.g.
  `/etc/pki/rpm-gpg/RPM-GPG-KEY-amdrocm`, and each `.repo` file
  carries `gpgcheck=1` plus `gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-amdrocm`.
- On first install of any tier package, `dnf` / `yum` discovers
  the new `gpgkey` reference when the user first runs a transaction
  that pulls from one of the tier's repos (e.g. `yum install
  rocm`) and **prompts interactively** to import and trust the key.
  Non-interactive flows must pass `-y` (or pre-import the key) — the
  package itself does not silently call `rpm --import`.
- Package upgrades that ship a rotated key install the new key
  file alongside the old one for one release cycle (so in-flight
  package signatures from the previous key still verify), then the
  next upgrade removes the superseded key. Users see the standard
  `dnf` key-import prompt the first time they install or upgrade
  through a transaction that references the new key.

**deb-family** (Debian, Ubuntu):

- The key is installed as a dearmored keyring at
  `/usr/share/keyrings/amdrocm-keyring.gpg`, and each `.sources`
  stanza carries `Signed-By: /usr/share/keyrings/amdrocm-keyring.gpg`
  (already shown in the deb822 examples above).
- The `Signed-By:` path-pinning ensures the AMD key only signs the
  AMD repos — it is not added to the system-wide trusted set, so a
  compromise of an unrelated repo cannot leverage this key.
- On install the package places the keyring file and prompts the
  user via the package manager's standard confirmation flow before
  the file is written. Subsequent `apt update` / `apt upgrade`
  cycles refresh the keyring file in place when the tier package
  ships a rotated key; no `apt-key add` is used (it is deprecated).
- As with rpm, the previous key is retained for one release cycle
  to keep in-flight package signatures verifiable across the
  rotation.

**Shared key file across tiers.** All four tier packages install
the *same* key file at the *same* path. Co-installing tiers does
not produce conflicting keyring copies (rpm/deb file-conflict on
identical contents is benign, matching the `amdrocm-amdrepos`
self-update repo handling).

### Verifying the key is AMD's

When the package manager prompts to import the key, the only useful
question the user can answer is "is this fingerprint actually
AMD's?" This RFC requires that AMD publish the answer in a place
the user can reach **independently of the package install** so the
verification does not loop through the same channel being
authenticated.

**Canonical key URLs on `repo.amd.com`.** The `amdrepos/`
singleton folder hosts a `gpg/` subdirectory (also referenced from
RFC0012 *Structure on `repo.amd.com` (bare domain)*) with the
following well-known files, each served over HTTPS with the
domain's TLS certificate as the transport-level trust anchor:

| URL                                                            | Contents                                                                       |
| :------------------------------------------------------------- | :----------------------------------------------------------------------------- |
| `https://repo.amd.com/amdrepos/gpg/RPM-GPG-KEY-amdrocm`        | ASCII-armored public key (the same bytes the rpm packages install).            |
| `https://repo.amd.com/amdrepos/gpg/amdrocm-keyring.gpg`        | Dearmored keyring (the same bytes the deb packages install at `Signed-By:`).   |
| `https://repo.amd.com/amdrepos/gpg/FINGERPRINT`                | Plain-text file containing the current key's 40-character fingerprint.         |
| `https://repo.amd.com/amdrepos/gpg/FINGERPRINT.txt.asc`        | The same fingerprint file signed by the *previous* key (rotation chain).       |
| `https://repo.amd.com/amdrepos/gpg/keys.json`                  | Machine-readable manifest: `{ current, previous[], rotated_at, algorithm }`.   |

The `gpg/` directory listing is also human-browsable from
`https://repo.amd.com/amdrepos/` so a user starting from the bare
domain can find it without prior knowledge of the exact path.

**Out-of-band publication.** Because `repo.amd.com` itself is the
channel being authenticated, the fingerprint must additionally be
discoverable from at least one independent location so a user does
not have to trust `repo.amd.com`'s TLS cert in isolation:

- The same fingerprint is published in the official ROCm
  documentation on `rocm.docs.amd.com` (under a stable URL — TBD
  during doc-site integration).
- The fingerprint and rotation history are mirrored in this RFC
  (and in RFC0012's release notes) when a rotation occurs, so the
  git history of the RFC repo is itself a verifiable trail.
- The current key is uploaded to the OpenPGP keyserver network
  (`keys.openpgp.org`) under an `@amd.com` UID so
  `gpg --recv-keys <fingerprint>` resolves to the same key.

A user who wants high assurance compares the fingerprint shown by
the package-manager prompt against **two** of these sources
(`rocm.docs.amd.com`, the keyserver, and this RFC) before
accepting.

**How to check.** Concrete verification commands:

```sh
# rpm-family — after the package downloads but BEFORE accepting the import prompt,
# inspect the key file the package shipped and compute its fingerprint:
gpg --show-keys --with-fingerprint \
    /etc/pki/rpm-gpg/RPM-GPG-KEY-amdrocm

# Or fetch the canonical key directly and compare:
curl -fsSL https://repo.amd.com/amdrepos/gpg/RPM-GPG-KEY-amdrocm \
  | gpg --show-keys --with-fingerprint

# Then compare the printed fingerprint against the published one:
curl -fsSL https://repo.amd.com/amdrepos/gpg/FINGERPRINT
```

```sh
# deb-family — inspect the keyring the package will install:
gpg --no-default-keyring \
    --keyring /usr/share/keyrings/amdrocm-keyring.gpg \
    --list-keys --with-fingerprint

# Or compare against the canonical dearmored keyring on repo.amd.com:
curl -fsSL https://repo.amd.com/amdrepos/gpg/amdrocm-keyring.gpg \
  | gpg --show-keys --with-fingerprint
```

```sh
# Cross-check against the public keyserver (independent of repo.amd.com):
gpg --keyserver hkps://keys.openpgp.org --recv-keys <fingerprint>
gpg --fingerprint <fingerprint>
```

**Verifying a signed package directly.** Users who want to
verify an individual rpm or deb without going through the package
manager can do so with the same key:

```sh
# rpm — checks the embedded signature against the imported key
rpm --checksig amdrocm-core-7.14.0-1.el9.x86_64.rpm

# deb — verify the Release file's detached signature
gpg --no-default-keyring \
    --keyring /usr/share/keyrings/amdrocm-keyring.gpg \
    --verify InRelease
```

**Rotation chain.** When the key is rotated, the new
`FINGERPRINT` file is signed by the **previous** key
(`FINGERPRINT.txt.asc`) so an automated verifier that already
trusts the previous fingerprint can confirm the new one without a
fresh out-of-band lookup. The chain is recorded in `keys.json`
(`previous[]`) so historical signatures remain verifiable. AMD
also announces the rotation in the ROCm release notes and updates
the keyserver-published UID.

## Package-manager mechanics

Repository stream selection must be implemented per package
manager.

For rpm packages, install a package-manager variable file, for
example `/etc/yum/vars/amdrocm_release_stream` or
`/etc/dnf/vars/amdrocm_release_stream`, and use
`$amdrocm_release_stream` in the repo `baseurl`.

For Debian-based systems, the repository package must not rely on
shell-style or yum-style variable expansion in APT source files.
It should install explicit deb822 `.sources` stanzas, or separate
`.sources` files, for each supported stream, using `Enabled:
yes/no` to control which stream is active.

## AMD GPU driver pinning

The repository packages include the amdgpu driver folder from
`repo.radeon.com`. **Temporarily**, each repo-setup package
includes links to **all** AMD GPU driver folders that support
ROCm 7.14 and above (including the `latest/` folder), so users
can install any ROCm-compatible driver version through the same
repo configuration. Once the AMD GPU drivers are consolidated
into `repo.amd.com`, this reduces to a single link to all driver
releases for a particular OS hosted on `repo.amd.com`.
Driver-version selection follows the stream, and each pin lives
in the package that owns that stream:

- `stable` *(in `amdrocm-repo-stable`)* → the GA amdgpu driver
  listed at `https://repo.radeon.com/amdgpu/latest/`. Pinned at
  repo-package build time; refreshed only by publishing a new
  `amdrocm-repo-stable` package.
- `rc` *(in `amdrocm-repo-stablerc`)* → the **pre-GA amdgpu
  driver paired with the candidate**, not `latest/`. The
  `amdrocm-repo-stablerc` package built for an `rc` line carries
  an explicit driver version corresponding to the ROCm candidate
  under QA. This prevents an in-flight `rc` from picking up a
  driver bump that the candidate has not been validated against.
- `lts` / `ltsrc` *(future)* → driver pin policy will be decided
  alongside the LTS repo-package decision noted above; expected
  to mirror the `stable` / `rc` rules respectively (GA driver for
  `lts`, paired candidate driver for `ltsrc`).
- `weekly` *(in `amdrocm-repo-nightly`)* → the GA amdgpu driver
  pinned at weekly-promotion time (the same driver `nightly` was
  tracking at the cut). Refreshed on every `amdrocm-repo-nightly`
  rebuild. This gives downstream CI a driver target that does not
  move within a calendar week.
- `nightly` *(in `amdrocm-repo-nightly`)* → tracks the driver
  version the develop branch is currently built against; updated
  whenever the nightly pipeline rebuilds `amdrocm-repo-nightly`.
- `dev` *(in `amdrocm-repo-dev`)* → tracks the driver version
  the develop branch is currently built against, same as
  `nightly`. Refreshed on every `amdrocm-repo-dev` rebuild (i.e.
  effectively per-commit). Intended for developer testing only;
  not a supported install target for end users.

In all cases the amdgpu URL is **pinned at repo-package build
time**, not resolved at install time, so a given installed
repo-setup package always points at one specific driver version
(per stream it covers) until the user updates the repo package
itself. Because the pin lives in the tier package that owns the
stream, a develop-branch driver bump only churns
`amdrocm-repo-nightly` / `amdrocm-repo-dev`; stable users on
`amdrocm-repo-stable` (and stablerc users on
`amdrocm-repo-stablerc`) see no version bump from that event.

## Publication and self-update

The repo packages are published in the **singleton** `amdrepos/`
folder hosted directly on the bare `repo.amd.com` domain (i.e.
`https://repo.amd.com/amdrepos/`), per RFC0012. This folder is
**not** replicated under any stream subdomain — there is one
canonical copy that serves all streams. Each tier package also
ships an `amdrocm-amdrepos` repo definition (file name above) so
future updates to the repo packages themselves come through the
normal `yum update` / `apt upgrade` flow.

The repo packages themselves are **stream-agnostic** — the
install URL has no stream subdomain because `amdrepos/` is the
bare-domain singleton. Stream selection happens **after** install
via the `$amdrocm_release_stream` package-manager variable (rpm)
or `Enabled: yes/no` flips (deb).

## Install commands

The packages are designed to work with the following commands
(stable tier shown; substitute `amdrocm-repo-stablerc`,
`amdrocm-repo-nightly`, or `amdrocm-repo-dev` for the other
tiers):

```
# stable tier (stable; + stable-rpath sibling disabled by default)
yum install https://repo.amd.com/amdrepos/$OS/amdrocm-repo-stable.rpm

# stablerc tier (rc) — install alongside stable for pre-release QA
yum install https://repo.amd.com/amdrepos/$OS/amdrocm-repo-stablerc.rpm

# integration tier (nightly / weekly) — for downstream CI / framework
# integrators
yum install https://repo.amd.com/amdrepos/$OS/amdrocm-repo-nightly.rpm

# developer tier (dev) — opt-in, for developer testing only
yum install https://repo.amd.com/amdrepos/$OS/amdrocm-repo-dev.rpm
```

The four tier packages can be installed in any combination; they
do not conflict with each other. Each install is the **explicit,
audit-visible** opt-in for that tier — no tier package
transitively registers another tier's stream URL.
