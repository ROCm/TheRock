AMD ROCm Aggregate Repository — Setup Guide
===========================================

Overview
--------
Two aggregation scripts merge package metadata from multiple sources into a
single repository endpoint. Package files (.deb / .rpm) are never copied or
moved — only the metadata index files are generated. CloudFront routes each
prefixed package path to the correct origin bucket at install time.

Scripts
-------
  aggregate_deb_metadata.py  — Merge DEB package metadata (Ubuntu/Debian)
  aggregate_rpm_metadata.py  — Merge RPM package metadata (RHEL/AlmaLinux)


How It Works
------------
Each source's Packages.gz / primary.xml is fetched, package file paths are
rewritten with an owner prefix, all entries are merged, and a fresh
Release / repomd.xml is generated.

When a client installs a package, it resolves the prefixed path against the
aggregate baseurl. CloudFront routes each prefix to the correct origin:

  DEB:
    pool/core/<pkg>.deb  ->  core release bucket
    pool/rvs/<pkg>.deb   ->  RVS bucket

  RPM:
    core/<pkg>.rpm       ->  core release bucket
    rvs/<pkg>.rpm        ->  RVS bucket


Step 1: Generate Aggregate Metadata
-------------------------------------
No additional dependencies required for local generation.

Sources are passed as --source arguments (repeatable, no limit on count).

  DEB format: name,base_url,style,pool_prefix
    style is "dists" (standard APT layout) or "flat" (Packages.gz at root)

  RPM format: name,base_url,href_prefix

DEB — merge ROCm nightly + RVS:

  python aggregate_deb_metadata.py \
    --source "core,https://repo.amd.com/rocm/packages-multi-arch/ubuntu2404,dists,pool/core" \
    --source "rvs,https://d22tya8uodfbu6.cloudfront.net/nightly/rvs/deb,flat,pool/rvs" \
    --output-dir /tmp/deb-metadata \
    --suite stable \
    --component main \
    --architecture amd64

  Output files:
    /tmp/deb-metadata/dists/stable/Release
    /tmp/deb-metadata/dists/stable/main/binary-amd64/Packages
    /tmp/deb-metadata/dists/stable/main/binary-amd64/Packages.gz

RPM — merge ROCm nightly + RVS:

  python aggregate_rpm_metadata.py \
    --source "core,https://repo.amd.com/rocm/packages-multi-arch/rhel10/x86_64,core" \
    --source "rvs,https://d22tya8uodfbu6.cloudfront.net/nightly/rvs/rpm,rvs" \
    --output-dir /tmp/rpm-metadata

  Output files:
    /tmp/rpm-metadata/repodata/repomd.xml
    /tmp/rpm-metadata/repodata/primary.xml.gz

Adding a third source is just another --source flag:

  python aggregate_deb_metadata.py \
    --source "core,...,dists,pool/core" \
    --source "rvs,...,flat,pool/rvs" \
    --source "extras,https://example.com/rocm-extras/deb,flat,pool/extras" \
    --output-dir /tmp/deb-metadata


Step 2 (Optional): Upload Metadata to S3
-----------------------------------------
Requires boto3 and AWS credentials with s3:PutObject on the output bucket.

  pip install boto3
  aws configure

Upload the generated metadata by adding --output-bucket:

  python aggregate_deb_metadata.py \
    --source "core,https://repo.amd.com/rocm/packages-multi-arch/ubuntu2404,dists,pool/core" \
    --source "rvs,https://d22tya8uodfbu6.cloudfront.net/nightly/rvs/deb,flat,pool/rvs" \
    --output-bucket therock-deb-rpm-test \
    --output-prefix metadata/deb \
    --suite stable \
    --component main \
    --architecture amd64

  Uploads to S3:
    therock-deb-rpm-test/metadata/deb/dists/stable/Release
    therock-deb-rpm-test/metadata/deb/dists/stable/main/binary-amd64/Packages
    therock-deb-rpm-test/metadata/deb/dists/stable/main/binary-amd64/Packages.gz

  python aggregate_rpm_metadata.py \
    --source "core,https://repo.amd.com/rocm/packages-multi-arch/rhel10/x86_64,core" \
    --source "rvs,https://d22tya8uodfbu6.cloudfront.net/nightly/rvs/rpm,rvs" \
    --output-bucket therock-deb-rpm-test \
    --output-prefix metadata/rpm

  Uploads to S3:
    therock-deb-rpm-test/metadata/rpm/repodata/repomd.xml
    therock-deb-rpm-test/metadata/rpm/repodata/primary.xml.gz

Both --output-dir and --output-bucket can be specified together to write
locally and upload in a single run.


Step 3: Set Up CloudFront
--------------------------
Add cache behaviors to the repo.amd.com CloudFront distribution. List them
most-specific first (CloudFront matches top-down).

DEB (ubuntu2404):

  Path pattern                                  Origin / S3 prefix
  --------------------------------------------  ----------------------------------------
  /rocm/metadata/ubuntu2404/dists/*            therock-deb-rpm-test: metadata/deb/dists/
  /rocm/metadata/ubuntu2404/pool/core/*        therock-release-packages: v4/.../deb/
  /rocm/metadata/ubuntu2404/pool/rvs/*         amd-sharks-rvs: v3/.../deb/

RPM (rhel10):

  Path pattern                                  Origin / S3 prefix
  --------------------------------------------  ----------------------------------------
  /rocm/metadata/rhel10/x86_64/repodata/*      therock-deb-rpm-test: metadata/rpm/repodata/
  /rocm/metadata/rhel10/x86_64/core/*          therock-release-packages: v4/.../rpm/
  /rocm/metadata/rhel10/x86_64/rvs/*           amd-sharks-rvs: v3/.../rpm/

For each behavior:
  - Cache policy: CachingDisabled for metadata paths (dists/*, repodata/*)
                  CachingOptimized for package paths (pool/*, core/*, rvs/*)
  - Origin request policy: AllViewer
  - Viewer protocol: HTTPS only

After adding the behaviors, regenerate and upload metadata (Step 1 + Step 2).
The generated Filename:/href values already use the relative prefixes that
CloudFront routes correctly — no script changes needed.


Step 4: User Setup
-------------------

Ubuntu/Debian:

  echo "deb [trusted=yes] https://repo.amd.com/rocm/metadata/ubuntu2404 stable main" \
    | sudo tee /etc/apt/sources.list.d/amdrocm-aggregate.list
  sudo apt update
  sudo apt install amdrocm7-rvs

RHEL/AlmaLinux:

  sudo tee /etc/yum.repos.d/amdrocm-aggregate.repo <<EOF
  [amdrocm-aggregate]
  name=AMD ROCm Aggregate
  baseurl=https://repo.amd.com/rocm/metadata/rhel10/x86_64
  enabled=1
  gpgcheck=0
  repo_gpgcheck=0
  priority=50
  EOF
  sudo dnf clean all
  sudo dnf install amdrocm7.13 amdrocm7-rvs


GPG Signing (optional)
-----------------------
To allow clients to verify repo metadata integrity, sign with a GPG key.
Can also be set via environment variable: export AGGREGATE_SIGN_KEY=<KEY_ID>

DEB — produces Release.gpg (detached) + InRelease (clearsigned):

  python aggregate_deb_metadata.py ... --sign-key <KEY_ID>

  Client setup (no [trusted=yes] needed):
  echo "deb [signed-by=/etc/apt/keyrings/amdrocm.gpg] https://repo.amd.com/rocm/metadata/ubuntu2404 stable main" \
    | sudo tee /etc/apt/sources.list.d/amdrocm-aggregate.list

RPM — produces repomd.xml.asc (detached):

  python aggregate_rpm_metadata.py ... --sign-key <KEY_ID>

  Client .repo file additions:
    repo_gpgcheck=1
    gpgkey=https://repo.amd.com/rocm/rocm.gpg.key
