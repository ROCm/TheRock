# Legacy multi-arch releases

> [!CAUTION]
> This page documents historical multi-arch release locations. Current
> releases use the stream-specific `repo.amd.com` locations documented in
> [RELEASES.md](/RELEASES.md).

Releases published before the `repo.amd.com` product layout remain at their
original URLs. They are not mirrored into the new indexes, so use the URL that
matches the ROCm version you need.

The legacy locations apply to:

- nightly releases through ROCm 7.14;
- ROCm 10.0.0 release candidates; and
- stable releases before ROCm 10.0.

## Legacy release locations

| Release channel | Python aggregate index                           | ROCm Core tarballs                                   | ROCm Core native packages                             |
| --------------- | ------------------------------------------------ | ---------------------------------------------------- | ----------------------------------------------------- |
| Nightly         | https://rocm.nightlies.amd.com/whl-multi-arch/   | https://rocm.nightlies.amd.com/tarball-multi-arch/   | https://rocm.nightlies.amd.com/packages-multi-arch/   |
| Prerelease      | https://rocm.prereleases.amd.com/whl-multi-arch/ | https://rocm.prereleases.amd.com/tarball-multi-arch/ | https://rocm.prereleases.amd.com/packages-multi-arch/ |
| Stable          | https://repo.amd.com/rocm/whl-multi-arch/        | https://repo.amd.com/rocm/tarball-multi-arch/        | https://repo.amd.com/rocm/packages-multi-arch/        |

## Installing legacy Python packages

Use the aggregate index for the release channel. For example, install a ROCm
7.14 nightly with support for `gfx942` from the legacy nightly index:

```bash
pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
    "rocm[libraries,devel,device-gfx942]"
```

## Downloading legacy tarballs

Browse the tarball index for the release channel and select the file matching
the required ROCm version and GPU family. For example, ROCm 10.0.0 release
candidate tarballs remain under:

```text
https://rocm.prereleases.amd.com/tarball-multi-arch/
```

## Installing legacy native packages

The native package parent contains the distribution-specific DEB and RPM
repositories for each channel. For example, prerelease repositories remain
under:

```text
https://rocm.prereleases.amd.com/packages-multi-arch/
```

The older GPU-family-specific layouts are documented separately in
[Legacy per-family releases](legacy_per_family_releases.md).
