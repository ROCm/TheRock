# How to make a release available

The steps for the release are:

1. Download prerelease candidates (e.g. 7.10.0rc2) that need promotion (ROCm and PyTorch packages containing rocm version), and tarballs
1. Promote those packages to release (7.10.0rc2 → 7.10.0)
1. Upload the release packages to S3 release buckets
1. Update index files for release bucket (for PyPI compatibility)

## 1. Download prerelease candidate

### Python packages and Tarballs

Need:

- `build_tools/packaging/download_prerelease_packages.py`
- IAM role: read and list bucket access for therock-prerelease-python and therock-prerelease-tarball

Example: Download all prerelease candidates 7.10.0rc2 to ./promotion/download

```bash
# 1. (Optional) Check which architectures are available
python build_tools/packaging/download_prerelease_packages.py --version=7.10.0rc2 --list-archs

# 2. (Recommended) Check which packages are available and their sizes
#    Make sure you have enough disk space available for what you want to download!
python build_tools/packaging/download_prerelease_packages.py --version=7.10.0rc2 --list-packages-per-arch --include-tarballs

# 3. Download all ROCm/PyTorch packages that need promotion (all architectures)
python build_tools/packaging/download_prerelease_packages.py --version=7.10.0rc2 --output-dir=./promotion/download/ --include-tarballs
```

## 2. Promote prerelease candidates to release

Need:

- `build_tools/packaging/promote_packages.py`

```bash
# TODO this needs a nicer wrapper
# For each architecture (e.g., gfx1151, gfx950-dcgpu, etc.)
for arch in ./promotion/download/*; do
   echo "Promoting packages in $arch"
   python build_tools/packaging/promote_packages.py --input-dir="$arch" --delete-old-on-success
done
```

Or run manually for each arch-subdirectory

```bash
# For python packages (repeat for each arch)
python build_tools/packaging/promote_packages.py --input-dir=./promotion/download/<arch> --delete-old-on-success

# For tarballs
python build_tools/packaging/promote_packages.py --input-dir=./promotion/download/tarball --delete-old-on-success
```

### Promoting nightly (`a`) builds

Nightlies carry an `a<YYYYMMDD>` prerelease segment (e.g. `7.13.0a20260501`).
The promotion source defaults to `rc`; use `--src-version-type=a` to look for
`a<YYYYMMDD>` instead. The destination defaults to `release` (strip the
prerelease entirely) but can be overridden with `--dest-version`.

```bash
# Nightly -> release (e.g. 7.13.0a20260501 -> 7.13.0)
python build_tools/packaging/promote_packages.py \
   --input-dir=./promotion/download/<arch> \
   --src-version-type=a \
   --delete-old-on-success

# Nightly -> RC (e.g. 7.13.0a20260501 -> 7.13.0rc1)
python build_tools/packaging/promote_packages.py \
   --input-dir=./promotion/download/<arch> \
   --src-version-type=a \
   --dest-version=rc1 \
   --delete-old-on-success
```

`--dest-version` accepts `release`, `rc<N>` (e.g. `rc1`, `rc2`), or
`a<YYYYMMDD>` (e.g. `a20260501`). The downstream RC -> release flow above is
unchanged.

### Multi-arch packages: restricting which gfx targets ship

Multi-arch aggregator wheels (`rocm`, `torch`, `torchvision`, …) reference
several gfx targets via `Provides-Extra` / `Requires-Dist` entries, and the
download directory may contain per-gfx wheels (`rocm_sdk_device_gfx1010-…`,
`amd_torch_device_gfx1010-…`) for each of those targets.

If a release should only ship a subset of those archs, pass
`--keep-gfx-archs` (positive list — everything else is dropped):

```bash
# Promote the version AND drop per-gfx wheels / aggregator entries for
# archs not in the keep list.
python build_tools/packaging/promote_packages.py \
   --input-dir=./promotion/download/<multiarch> \
   --keep-gfx-archs=gfx1201,gfx1010 \
   --delete-old-on-success
```

Effects of `--keep-gfx-archs`:

- Per-gfx wheels for non-kept archs are skipped (and deleted with
  `--delete-old-on-success`).
- Multi-arch aggregator wheels lose `Provides-Extra: device-gfx<N>` /
  `Requires-Dist: ...-gfx<N>` entries for non-kept archs.
- Multi-arch `_dist_info.py` loses matching `AVAILABLE_TARGET_FAMILIES`
  entries; `DEFAULT_TARGET_FAMILY` is repointed at the first kept arch if it
  referenced a dropped one. The same repoint happens for the bare
  `extra == "device"` line in METADATA and the `[device]` section in
  `requires.txt`.
- Single-arch packages are detected automatically and pass through unchanged.

To run *only* the arch trim (no version rewrite — e.g. you already have
release-versioned multi-arch wheels and just want to narrow them), use
`--skip-version-promotion`:

```bash
python build_tools/packaging/promote_packages.py \
   --input-dir=./promotion/download/<multiarch> \
   --skip-version-promotion \
   --keep-gfx-archs=gfx1201,gfx1010
```

`--skip-version-promotion` is mutually exclusive with `--src-version-type` /
`--dest-version` and requires `--keep-gfx-archs`.

## 3. Upload release packages

Need:

- `build_tools/packaging/upload_release_packages.py`
- IAM role:
  - for testing: write access to therock-testing-bucket
  - for production: write access to therock-release-python and therock-release-tarball
- Same folder structure as created by `download_prerelease_packages.py`:

```
<input-dir>/
   <arch1>/
      package1.whl
      package2.whl
      rocm-*.tar.gz
      ...
   <arch2>/
      package1.whl
      ...
   tarball/  (if --include-tarballs was used)
      therock-dist-linux-<arch1>-<version>.tar.gz
      therock-dist-windows-<arch2>-<version>.tar.gz
      ...
```

```bash
# 1. Run a dry run (default - shows what would be uploaded)
python build_tools/packaging/upload_release_packages.py --input-dir ./promotion/download/ --upload-tarballs

# 2. (Optional) Test upload to therock-testing-bucket
python build_tools/packaging/upload_release_packages.py --input-dir ./promotion/download/ --upload-tarballs --execute

# 3. Upload to production release buckets
python build_tools/packaging/upload_release_packages.py --input-dir ./promotion/download/ --upload-tarballs --execute --use-release-buckets
```

### Upload options:

```bash
# Upload only Python packages (no tarballs)
python build_tools/packaging/upload_release_packages.py --input-dir ./promotion/download/ --execute --use-release-buckets

# Upload only tarballs (no Python packages)
python build_tools/packaging/upload_release_packages.py --input-dir ./promotion/download/ --no-upload-python --upload-tarballs --execute --use-release-buckets
```

## 4. Update index files for the release bucket

### Update Python package index

Need:

- `build_tools/third_party/s3_management/manage.py`
- IAM role: read and write access for therock-release-python

```bash
export S3_BUCKET_PY="therock-release-python"

# TODO
```

### Update tarball bucket index
