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

| OS profile        | Package manager | Repo definition                           | Signing key                         |
| ----------------- | --------------- | ----------------------------------------- | ----------------------------------- |
| `ubuntu2404`      | `apt`           | `/etc/apt/sources.list.d/amdrocm.sources` | `/usr/share/keyrings/rocm.gpg`      |
| `rhel8`, `rhel10` | `dnf` / `yum`   | `/etc/yum.repos.d/amdrocm.repo`           | `/etc/pki/rpm-gpg/RPM-GPG-KEY-rocm` |
| `sles16`          | `zypper`        | `/etc/zypp/repos.d/amdrocm.repo`          | `/etc/pki/rpm-gpg/RPM-GPG-KEY-rocm` |

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

## Download URLs

On the `prerelease` and `nightly` lines, `amdrocm-repo` is published as a
standalone file per OS profile, alongside the packages it configures:

- prerelease: `<base>/<format>/repo/<os-profile>/amdrocm-repo.<ext>`
- nightly: `<base>/<format>/<YYYYMMDD-id>/repo/<os-profile>/amdrocm-repo.<ext>`

where `<format>` is `deb` or `rpm` and `<ext>` matches. For example, on the
prerelease line:

| OS profile   | Download URL                                                                                |
| ------------ | ------------------------------------------------------------------------------------------- |
| `ubuntu2404` | `https://rocm.prereleases.amd.com/packages-multi-arch/deb/repo/ubuntu2404/amdrocm-repo.deb` |
| `rhel8`      | `https://rocm.prereleases.amd.com/packages-multi-arch/rpm/repo/rhel8/amdrocm-repo.rpm`      |
| `rhel10`     | `https://rocm.prereleases.amd.com/packages-multi-arch/rpm/repo/rhel10/amdrocm-repo.rpm`     |
| `sles16`     | `https://rocm.prereleases.amd.com/packages-multi-arch/rpm/repo/sles16/amdrocm-repo.rpm`     |

A distro's package is reachable once its line has published. The `release` line
is published through the ROCm release process rather than from this repository's
build, so its download location is not listed here. To use the release
repository before that package is available, build it locally with
`--release-type release` (see below).

## Installing and using the repository

Install the `amdrocm-repo` package for your distro, refresh the package
manager's metadata, then install ROCm packages.

Ubuntu:

```bash
curl -fsSL -O https://rocm.prereleases.amd.com/packages-multi-arch/deb/repo/ubuntu2404/amdrocm-repo.deb
sudo apt install ./amdrocm-repo.deb
sudo apt-get update
sudo apt install amdrocm
```

RHEL / Rocky / Oracle Linux:

```bash
sudo dnf install https://rocm.prereleases.amd.com/packages-multi-arch/rpm/repo/rhel10/amdrocm-repo.rpm
sudo dnf makecache
sudo dnf install amdrocm
```

SLES:

```bash
sudo zypper install https://rocm.prereleases.amd.com/packages-multi-arch/rpm/repo/sles16/amdrocm-repo.rpm
sudo zypper refresh
sudo zypper install amdrocm
```

On a signed line, `zypper` prompts you to trust the repository signing key the
first time it refreshes; accept it to continue.

Confirm the repository is configured by listing the `amdrocm*` packages it
publishes:

```bash
apt list 'amdrocm*'      # Ubuntu
dnf list 'amdrocm*'      # RHEL / Rocky / Oracle Linux
zypper search amdrocm    # SLES
```

The command returns the available packages (for example `amdrocm` and
`amdrocm-base`). An empty result means the repository is not configured, or its
metadata has not been refreshed. Install the components you need from that
list.

## Troubleshooting

- **404 when downloading the package.** The release line is not published for
  your distro yet. Use the `prerelease` line, or build the package locally (see
  below).
- **`amdrocm*` packages are not listed.** Refresh the package manager metadata
  (`sudo apt-get update`, `sudo dnf makecache`, or `sudo zypper refresh`) after
  installing `amdrocm-repo`.
- **`zypper` reports an untrusted signing key.** Accept the key when prompted on
  the first refresh of a signed line.

## Building the package locally

`build_repo_package.py` builds `amdrocm-repo` for a single OS profile and
release line. It requires Python 3.12 or newer (note that some rpm base images,
such as RHEL 8, do not ship a system `python3` at that version), the `jinja2`
Python package, and the native packaging tools for the target format:

- deb (`ubuntu2404`): `debhelper`, `dpkg-dev`, `build-essential`
- rpm (`rhel8`, `rhel10`, `sles16`): `rpm-build`

For a signed line the builder also invokes `gpg` to prepare the key, and fetches
it over the network (see below), so `gpg` and `ca-certificates` must be present.

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
