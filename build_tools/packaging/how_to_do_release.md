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

- `build_tools/packaging/promote_from_rc_to_final.py`

```bash
# TODO this needs a nicer wrapper
# For each architecture (e.g., gfx1151, gfx950-dcgpu, etc.)
for arch in ./promotion/download/*; do
   echo "Promoting packages in $arch"
   python build_tools/packaging/promote_from_rc_to_final.py --input-dir="$arch" --delete-old-on-success
done
```

Or run manually for each arch-subdirectory

```bash
# For python packages (repeat for each arch)
python build_tools/packaging/promote_from_rc_to_final.py --input-dir=./promotion/download/<arch> --delete-old-on-success

# For tarballs
python build_tools/packaging/promote_from_rc_to_final.py --input-dir=./promotion/download/tarball --delete-old-on-success
```

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
