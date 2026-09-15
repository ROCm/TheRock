# Configuring the ROCm package repository (`amdrocm-repo`)

`amdrocm-repo` is a configuration package that points a system package manager
(`apt`, `dnf`, or `zypper`) at a public AMD ROCm repository. Once it is
installed, you can install and update ROCm packages with the native package
manager.

It configures the repository only. The ROCm *content* packages and their upload
paths are documented in [native_packaging.md](native_packaging.md).

## What the package installs

`amdrocm-repo` is generated per OS profile and per stream. Every profile writes a
single repo definition and, for signed streams, the repository signing key:

| OS profile        | Package manager | Repo definition                                    | Signing key                            |
| ----------------- | --------------- | -------------------------------------------------- | -------------------------------------- |
| `ubuntu2404`      | `apt`           | `/etc/apt/sources.list.d/amdrocm-<stream>.sources` | `/usr/share/keyrings/amdrocm.gpg`      |
| `rhel8`, `rhel10` | `dnf` / `yum`   | `/etc/yum.repos.d/amdrocm-<stream>.repo`           | `/etc/pki/rpm-gpg/RPM-GPG-KEY-amdrocm` |
| `sles16`          | `zypper`        | `/etc/zypp/repos.d/amdrocm-<stream>.repo`          | `/etc/pki/rpm-gpg/RPM-GPG-KEY-amdrocm` |

The filename and the rpm section id both carry the stream, so packages for two
different streams can be installed in turn without one silently replacing the
other's configuration.

The deb repo definition uses the deb822 `.sources` format (`X-Repo-Id`,
`Suites: stable`, `Components: main`, `Architectures: amd64`, `Enabled: yes`)
and references the shipped keyring via `Signed-By`. The rpm repo definition
references the shipped key via `gpgkey=file://`. These match the configuration
the [published ROCm install
instructions](https://github.com/ROCm/rocm-install-utils) write by hand, so a
system set up either way ends up with equivalent repository configuration.

## Streams

Each stream is served from its own subdomain, `<stream>.repo.amd.com`, and every
repository serves x86_64 (`amd64`) packages only.

| Stream    | Repository base                                   | Signed |
| --------- | ------------------------------------------------- | ------ |
| `stable`  | `https://stable.repo.amd.com/rocm/core/packages`  | yes    |
| `nightly` | `https://nightly.repo.amd.com/rocm/core/packages` | no     |

Both streams are published per distro. They differ only in whether a build
identifier appears in the path:

| Stream    | deb `URIs`                       | rpm `baseurl`                           |
| --------- | -------------------------------- | --------------------------------------- |
| `stable`  | `<base>/<os-profile>/`           | `<base>/<os-profile>/x86_64/`           |
| `nightly` | `<base>/<os-profile>/<date-id>/` | `<base>/<os-profile>/<date-id>/x86_64/` |

The signing key is at `<stream-root>/gpg/packages.gpg` — for example
`https://stable.repo.amd.com/rocm/gpg/packages.gpg`. Note that this is *not*
under the repository base: packages live under `<stream-root>/core/packages/`
while the key sits beside `core/`, so the two are supplied separately when
building the package.

> **Nightly is unsigned and expires.** The nightly repository publishes no
> signatures, so the nightly `amdrocm-repo` disables signature verification
> (`gpgcheck=0` / apt `Trusted: yes`) and ships no key.
>
> More importantly, a nightly repo file names **one specific build**. Nightly
> retention prunes old builds, so a nightly `amdrocm-repo` stops resolving once
> the build it points at is removed, and the package must be rebuilt to follow
> the stream. Use `stable` for anything beyond short-lived testing.

Other streams exist on `repo.amd.com` — `rc`, `dev` and `weekly` — but no
`amdrocm-repo` is built for them. `rc` and `weekly` are not yet serving content,
and `dev` is intended for developer testing rather than end users.

## Relationship to `amdgpu-install`

`amdrocm-repo` declares a conflict with `amdgpu-install`, so the two packages
cannot be installed at the same time. `amdgpu-install` ships its own ROCm
repository definition (`/etc/apt/sources.list.d/rocm.list`, or the `[rocm]`
entry in `/etc/yum.repos.d/rocm.repo`) pointing at `repo.radeon.com`, and gives
that repository a higher priority than the default. Without the conflict, both
would install and the system would end up with two ROCm repositories
configured, the `repo.radeon.com` one taking precedence.

Use one or the other. To switch, remove `amdgpu-install` first:

```bash
sudo apt purge amdgpu-install       # Ubuntu: purge, not remove (see the note below)
sudo dnf remove amdgpu-install      # RHEL / Rocky / Oracle Linux
sudo zypper remove amdgpu-install   # SLES
```

> **On Debian and Ubuntu, use `purge` rather than `remove`.** `amdgpu-install`
> ships its repository file, its signing key and its priority pin
> (`/etc/apt/preferences.d/repo-radeon-pin-600`) as conffiles, which
> `apt remove` deliberately keeps. After a plain `remove` the package is gone
> but the repository stays configured, still has its key, and still outranks
> this one. The same applies if you let `apt` resolve the conflict for you:
> installing `amdrocm-repo` over `amdgpu-install` removes that package but
> leaves those files behind. `dnf` and `zypper` refuse the install outright
> instead, and erase their repository files when the package is removed, so no
> extra step is needed there.
>
> This is not cosmetic. That pin sets priority 600 against the default 500 this
> repository uses, and apt resolves by priority before version, so **ROCm keeps
> installing from `repo.radeon.com`** even when this repository offers a newer
> release. Both repositories publish the same `amdrocm*` package names, so the
> substitution is silent.
>
> On Debian and Ubuntu the package checks for exactly this state at install
> time and warns, so you do not have to remember.
>
> To check afterwards:
>
> ```bash
> apt-cache policy amdrocm
> ```
>
> The candidate should come from the repository this package configures, not
> from `repo.radeon.com`.

## Building the package locally

`build_repo_package.py` builds `amdrocm-repo` for a single OS profile and
stream. It requires Python 3.12 or newer, matching the rest of this
repository, the `jinja2` Python package, and the native packaging tools for the
target format:

- deb (`ubuntu2404`): `debhelper`, `dpkg-dev`, `build-essential`
- rpm (`rhel8`, `rhel10`, `sles16`): `rpm-build`

Some rpm base images ship no system `python3` at all — UBI 8, for example — so
Python may need installing first. `setup_python_cmd.sh` does that per profile
and reports the interpreter to use.

For a signed stream the builder also invokes `gpg` to prepare the key, and
fetches it over the network (see below), so `gpg` and `ca-certificates` must be
present.

`setup_repo_build_deps.sh` installs that set for a given profile, picking `apt`,
`zypper` or `dnf` to match:

```bash
bash build_tools/packaging/linux/setup_repo_build_deps.sh --os-profile ubuntu2404
```

It does not install Python — use `setup_python_cmd.sh` for that, which knows the
per-distro package names (SLES, for instance, has no unversioned `python3` or
`python3-pip` to install).

```bash
python3 build_tools/packaging/linux/build_repo_package.py \
    --os-profile ubuntu2404 \
    --stream stable \
    --repo-base-url https://stable.repo.amd.com/rocm/core/packages \
    --gpg-key-url https://stable.repo.amd.com/rocm/gpg/packages.gpg \
    --rocm-version 10.0.0 \
    --dest-dir ./repo-package-out
```

`--gpg-key-url` is the full URL of the key, used verbatim. It is a separate
option because the key is not under `--repo-base-url` — packages live under
`<root>/core/packages/` and the key beside `core/` — so one cannot be derived
from the other. Taking it whole rather than assembling it from a root also keeps
where the key lives a property of the repository rather than an assumption in
this tool. The fetch is https-only, and the key must match the fingerprint
pinned in `build_repo_package.py`, so a wrong or tampered key fails the build
regardless of the URL it came from.

For a signed stream the key is fetched over the network at build time. To build
offline, or to pin a specific key, pass it explicitly instead:

```bash
python3 build_tools/packaging/linux/build_repo_package.py \
    --os-profile rhel10 \
    --stream stable \
    --repo-base-url https://stable.repo.amd.com/rocm/core/packages \
    --gpg-key-file ./packages.gpg \
    --rocm-version 10.0.0 \
    --dest-dir ./repo-package-out
```

The nightly stream is pinned to a single build; pass its identifier. No key is
needed, since the nightly repository is unsigned:

```bash
python3 build_tools/packaging/linux/build_repo_package.py \
    --os-profile ubuntu2404 \
    --stream nightly \
    --repo-base-url https://nightly.repo.amd.com/rocm/core/packages \
    --repo-sub-folder 20260722-123456789 \
    --rocm-version 10.0.0 \
    --dest-dir ./repo-package-out
```

Pass `--verify-repo-url` to fail the build unless the repository the package
configures is reachable. This catches a package that would install cleanly but
fail on the user's first metadata refresh:

```bash
python3 build_tools/packaging/linux/build_repo_package.py \
    --os-profile rhel10 \
    --stream stable \
    --repo-base-url https://stable.repo.amd.com/rocm/core/packages \
    --gpg-key-url https://stable.repo.amd.com/rocm/gpg/packages.gpg \
    --rocm-version 10.0.0 \
    --dest-dir ./repo-package-out \
    --verify-repo-url
```

`--verify-repo-url` has no effect for the nightly stream: its build folder is
published by the same run that builds the package, so it does not exist yet.

See `build_repo_package.py --help` for the full option list.

## Installing and using the repository

`amdrocm-repo` is not yet published to a public download URL, so build it for
your distro first using the previous section. Then install the resulting file,
refresh the package manager's metadata, and install ROCm packages.

The examples below assume `--dest-dir ./repo-package-out`. The built filename
carries the package version, so the glob below is what selects it.

Ubuntu:

```bash
sudo apt install ./repo-package-out/amdrocm-repo_*_all.deb
sudo apt-get update
sudo apt install amdrocm
```

RHEL / Rocky / Oracle Linux:

```bash
sudo dnf install ./repo-package-out/amdrocm-repo-*.noarch.rpm
sudo dnf makecache
sudo dnf install amdrocm
```

SLES:

```bash
sudo zypper install --allow-unsigned-rpm ./repo-package-out/amdrocm-repo-*.noarch.rpm
sudo zypper refresh
sudo zypper install amdrocm
```

`zypper` verifies the signature of the package file itself, and a locally built
`amdrocm-repo` is unsigned, so `--allow-unsigned-rpm` is required on SLES
whichever stream the package configures. `apt` and `dnf` install a local file
without it.

On a signed stream, `zypper` prompts you to trust the repository signing key the
first time it refreshes; accept it to continue. The package ships the key and
points `gpgkey=` at it, but trusting it is still the user's decision, so an
unattended run has to say so explicitly:

```bash
sudo zypper --non-interactive --gpg-auto-import-keys refresh
```

Without `--gpg-auto-import-keys`, `--non-interactive` answers the prompt with
"reject" and skips the repository, so a scripted install fails with no packages
found. `dnf` takes the same decision with `-y`, and `apt` needs no equivalent
because `Signed-By:` names the keyring directly.

### Switching streams

Installing a different stream's package over the current one switches the
configured repository. The install commands above are the same ones to use.

On Debian and Ubuntu, expect `apt` to describe a switch from `nightly` to
`stable` as a **downgrade**. The package version carries the stream so that two
streams never produce an identically named package, and a nightly version is
derived from its build date, which sorts above a ROCm version number. `apt`
prompts and proceeds normally when you confirm. Only non-interactive use is
affected:

```bash
# fails: "Packages were downgraded and -y was used without --allow-downgrades"
sudo apt install -y ./repo-package-out/amdrocm-repo_*_all.deb

# succeeds
sudo apt install -y --allow-downgrades ./repo-package-out/amdrocm-repo_*_all.deb
```

`dnf` and `zypper` are unaffected and install the new stream's package directly.

Refresh the package manager metadata afterwards, exactly as after a first
install.

### Verifying the repository

Confirm the repository is configured by resolving the `amdrocm` package:

```bash
apt-cache policy amdrocm    # Ubuntu
dnf info amdrocm            # RHEL / Rocky / Oracle Linux
zypper info amdrocm         # SLES
```

A candidate version means the repository is reachable and its metadata is
current. No candidate means the repository is not configured, or its metadata
has not been refreshed.

To browse everything the repository publishes, list by prefix instead — expect
on the order of a thousand entries, since each component ships separately:

```bash
apt list 'amdrocm*'      # Ubuntu
dnf list 'amdrocm*'      # RHEL / Rocky / Oracle Linux
zypper search amdrocm    # SLES
```

## Troubleshooting

- **`dnf` refuses with `conflicts with amdgpu-install ... conflicting requests`.**
  Remove `amdgpu-install` first — see
  [Relationship to `amdgpu-install`](#relationship-to-amdgpu-install). Do not
  reach for `--allowerasing`; removing it deliberately is the point.
- **`apt` says `The following packages will be REMOVED: amdgpu-install`.**
  That is the conflict being resolved, and it is not the whole story: the
  removal is not a purge, so that package's repository file and priority pin
  remain. See the note in
  [Relationship to `amdgpu-install`](#relationship-to-amdgpu-install).
- **`dpkg -i` refuses with `conflicting packages - not installing amdrocm-repo`.**
  `dpkg` will not resolve the conflict for you; purge `amdgpu-install` first.
- **`zypper` reports `Signature verification failed [6-File is unsigned]`.**
  The package file is unsigned; pass `--allow-unsigned-rpm`.
- **`apt` reports `Packages were downgraded and -y was used without --allow-downgrades`.** You are switching streams non-interactively. See
  [Switching streams](#switching-streams); add `--allow-downgrades`,
  or drop `-y` and confirm the prompt.
- **`amdrocm*` packages are not listed.** Refresh the package manager metadata
  (`sudo apt-get update`, `sudo dnf makecache`, or `sudo zypper refresh`) after
  installing `amdrocm-repo`.
- **`amdrocm` resolves to an unexpected version.** Another ROCm repository is
  configured and is winning. Check with `apt-cache policy amdrocm`, `dnf info amdrocm`, or `zypper info amdrocm`, which name the repository each candidate
  comes from. If it is `repo.radeon.com` and `amdgpu-install` is still present
  in a removed-but-not-purged state, `sudo apt purge amdgpu-install` clears it
  along with its priority pin. If instead the repository was configured by hand
  following older setup instructions, no package owns those files and they must
  be removed by hand — look for entries referencing `repo.radeon.com` under
  `/etc/apt/sources.list.d/` or `/etc/yum.repos.d/`. Remove only the ROCm
  entries: the same host also serves the AMDGPU driver repository, which is
  unrelated to this package and is usually still wanted.
- **`zypper` reports an untrusted signing key.** Accept the key when prompted on
  the first refresh of a signed stream. Non-interactively, pass
  `--gpg-auto-import-keys`; see
  [Installing and using the repository](#installing-and-using-the-repository).
