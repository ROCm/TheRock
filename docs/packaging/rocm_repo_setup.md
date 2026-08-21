# Configuring the ROCm package repository (`amdrocm-repo`)

`amdrocm-repo` is a configuration package that points a system package manager
(`apt`, `dnf`, or `zypper`) at a public AMD ROCm repository. Once it is
installed, you can install and update ROCm packages with the native package
manager.

It configures the repository only. The ROCm *content* packages and their upload
paths are documented in [native_packaging.md](native_packaging.md).

## What the package installs

`amdrocm-repo` is generated per OS profile. Every profile writes a single repo
definition and, for signed lines, the repository signing key:

| OS profile        | Package manager | Repo definition                           | Signing key                            |
| ----------------- | --------------- | ----------------------------------------- | -------------------------------------- |
| `ubuntu2404`      | `apt`           | `/etc/apt/sources.list.d/amdrocm.sources` | `/usr/share/keyrings/amdrocm.gpg`      |
| `rhel8`, `rhel10` | `dnf` / `yum`   | `/etc/yum.repos.d/amdrocm.repo`           | `/etc/pki/rpm-gpg/RPM-GPG-KEY-amdrocm` |
| `sles16`          | `zypper`        | `/etc/zypp/repos.d/amdrocm.repo`          | `/etc/pki/rpm-gpg/RPM-GPG-KEY-amdrocm` |

The deb repo definition uses the deb822 `.sources` format
(`Suites: stable`, `Components: main`, `Architectures: amd64`) and references
the shipped keyring via `Signed-By`. The rpm repo definition references the
shipped key via `gpgkey=file://`.

## Release lines

The configured repository depends on the release line. All repositories serve
x86_64 (`amd64`) packages only.

| Release line | Repository base                                        | Signed |
| ------------ | ------------------------------------------------------ | ------ |
| `prerelease` | `https://rocm.prereleases.amd.com/packages-multi-arch` | yes    |
| `release`    | `https://repo.amd.com/rocm/packages-multi-arch`        | yes    |
| `nightly`    | `https://rocm.nightlies.amd.com/packages-multi-arch`   | no     |

The lines do not share a repository layout. The `prerelease` and `release`
repositories are published per distro; the `nightly` repository is published per
package type under a dated sub-folder. So the repo file resolves to:

| Release line            | deb `URIs`              | rpm `baseurl`                  |
| ----------------------- | ----------------------- | ------------------------------ |
| `prerelease`, `release` | `<base>/<os-profile>/`  | `<base>/<os-profile>/x86_64/`  |
| `nightly`               | `<base>/deb/<date-id>/` | `<base>/rpm/<date-id>/x86_64/` |

The signing key is at `<base>/gpg/rocm.gpg` on both signed lines.

> **Nightly is unsigned.** The nightly repository is not signed, so the nightly
> `amdrocm-repo` disables signature verification (`gpgcheck=0` / apt
> `Trusted: yes`) and ships no key. Prefer a signed line (`prerelease` or
> `release`) for anything beyond testing. Nightly packages live in a dated
> sub-folder, so a nightly `amdrocm-repo` is pinned to its build date.

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
sudo apt purge amdgpu-install       # Ubuntu -- purge, not remove; see below
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

## Installing and using the repository

`amdrocm-repo` is not yet published to a public download URL. Build it for your
distro first (see [Building the package locally](#building-the-package-locally)),
then install the resulting file, refresh the package manager's metadata, and
install ROCm packages.

The examples below assume `--dest-dir ./repo-package-out`. The built filename
carries the package version, so the glob is what selects it.

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
whichever release line the package configures. `apt` and `dnf` install a local
file without it.

On a signed line, `zypper` prompts you to trust the repository signing key the
first time it refreshes; accept it to continue.

### Switching release lines

Installing a different line's package over the current one switches the
configured repository. The install commands above are the same ones to use.

On Debian and Ubuntu, expect `apt` to describe the change as a **downgrade**.
The package version carries the release line so that two lines never produce an
identically named package, which means a line change is rarely an increase in
`dpkg` version order — nightly versions are date-derived, and `~prerelease`
sorts below the plain release version by design. `apt` prompts and proceeds
normally when you confirm. Only non-interactive use is affected:

```bash
# fails: "Packages were downgraded and -y was used without --allow-downgrades"
sudo apt install -y ./repo-package-out/amdrocm-repo_*_all.deb

# succeeds
sudo apt install -y --allow-downgrades ./repo-package-out/amdrocm-repo_*_all.deb
```

`dnf` and `zypper` are unaffected and install the new line's package directly.

Refresh the package manager metadata after switching, as after a first install.

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
- **`apt` reports `Packages were downgraded and -y was used without --allow-downgrades`.** You are switching release lines non-interactively. See
  [Switching release lines](#switching-release-lines); add `--allow-downgrades`,
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
  the first refresh of a signed line.

## Building the package locally

`build_repo_package.py` builds `amdrocm-repo` for a single OS profile and
release line. It requires Python 3.12 or newer, matching the rest of this
repository, the `jinja2` Python package, and the native packaging tools for the
target format:

- deb (`ubuntu2404`): `debhelper`, `dpkg-dev`, `build-essential`
- rpm (`rhel8`, `rhel10`, `sles16`): `rpm-build`

Some rpm base images ship no system `python3` at all — UBI 8, for example — so
Python may need installing first. `setup_python_cmd.sh` does that per profile
and reports the interpreter to use.

For a signed line the builder also invokes `gpg` to prepare the key, and fetches
it over the network (see below), so `gpg` and `ca-certificates` must be present.

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
    --release-type prerelease \
    --repo-base-url https://rocm.prereleases.amd.com/packages-multi-arch \
    --rocm-version 7.14.0 \
    --dest-dir ./repo-package-out
```

For a signed line, the signing key is fetched from `<base>/gpg/rocm.gpg` over
the network at build time. To build offline, or to pin a specific key, pass it
explicitly:

```bash
python3 build_tools/packaging/linux/build_repo_package.py \
    --os-profile rhel10 \
    --release-type prerelease \
    --repo-base-url https://rocm.prereleases.amd.com/packages-multi-arch \
    --gpg-key-file ./rocm.gpg \
    --rocm-version 7.14.0 \
    --dest-dir ./repo-package-out
```

The nightly line is date-pinned; pass the dated sub-folder:

```bash
python3 build_tools/packaging/linux/build_repo_package.py \
    --os-profile ubuntu2404 \
    --release-type nightly \
    --repo-base-url https://rocm.nightlies.amd.com/packages-multi-arch \
    --repo-sub-folder 20260722-123456789 \
    --rocm-version 7.14.0 \
    --dest-dir ./repo-package-out
```

Pass `--verify-repo-url` to fail the build unless the repository the package
configures is reachable. This catches a package that would install cleanly but
fail on the user's first metadata refresh:

```bash
python3 build_tools/packaging/linux/build_repo_package.py \
    --os-profile rhel10 \
    --release-type release \
    --repo-base-url https://repo.amd.com/rocm/packages-multi-arch \
    --rocm-version 7.14.0 \
    --dest-dir ./repo-package-out \
    --verify-repo-url
```

See `build_repo_package.py --help` for the full option list.
