# S3 Buckets

TheRock uses Amazon S3 buckets to store CI build outputs (artifacts, logs,
python packages, etc.) and release artifacts. This page lists all buckets
and explains the authentication needed to upload to them.

## Table of contents

- [Authentication](#authentication)
- [Bucket inventory](#bucket-inventory)
  - [CI buckets](#ci-buckets): `therock-ci-artifacts`, `therock-ci-artifacts-external`
  - [Release buckets](#release-buckets): `therock-{dev,nightly,prerelease,release}-{artifacts,packages,python,tarball}`
  - [Build system buckets](#build-system-buckets): `rocm-third-party-deps`
  - [Cache buckets](#cache-buckets): `therock-*-pytorch-sccache`
  - [Legacy buckets](#legacy-buckets): `therock-artifacts`, `therock-artifacts-external`

## Authentication

Most buckets have public _read_ access for use by developers as well as CI/CD
systems.

To _write_ to most buckets, assuming an IAM role via
[`aws-actions/configure-aws-credentials`](https://github.com/aws-actions/configure-aws-credentials)
using OIDC is needed. This requires `id-token: write` in the job's `permissions` block.
The full ARN pattern is
`arn:aws:iam::692859939525:role/therock-{ci,dev,nightly,prerelease}`.

Use the
[`configure_aws_artifacts_credentials`](/.github/actions/configure_aws_artifacts_credentials/action.yml)
composite action to set up credentials. It determines the correct IAM role and
bucket from the repository, event type, and optional `release_type` input
by using [`build_tools/_therock_utils/s3_buckets.py`](/build_tools/_therock_utils/s3_buckets.py):

```yaml
jobs:
  build:
    runs-on: azure-linux-scale-rocm
    permissions:
      id-token: write
    # Linux containers only — mount runner baseline credentials
    env:
      AWS_SHARED_CREDENTIALS_FILE: /home/awsconfig/credentials.ini

    steps:
      # ... build steps ...

      # Credentials are short-lived — assume the role close to when it's needed.
      - name: Configure AWS Credentials
        uses: ./.github/actions/configure_aws_artifacts_credentials

      # ... upload steps that use the credentials ...
```

**Platform-specific details:**

- **Linux containers** mount runner credentials via
  `AWS_SHARED_CREDENTIALS_FILE: /home/awsconfig/credentials.ini` in the job's
  `env` block. These baseline credentials allow uploading to
  `therock-ci-artifacts-external` without OIDC.
- **Windows** jobs must pass `special-characters-workaround: true` to
  `aws-actions/configure-aws-credentials`. This retries credential fetching
  until the secret access key contains no special characters, which some
  Windows environments cannot tolerate. (The
  `configure_aws_artifacts_credentials` composite action mentioned above
  handles this automatically)

## Bucket inventory

### CI buckets

Our CI runners come with baseline credentials that allow uploading to
`therock-ci-artifacts-external` without any extra setup. Workflows in
downstream repos like `rocm-libraries`, `rocm-systems`, and `llvm-project`
upload to this bucket and do not need `aws-actions/configure-aws-credentials`.

| Bucket                                                                                     | Contents                                | IAM role                                          |
| ------------------------------------------------------------------------------------------ | --------------------------------------- | ------------------------------------------------- |
| [`therock-ci-artifacts`](https://therock-ci-artifacts.s3.amazonaws.com/)                   | Build outputs for `ROCm/TheRock`        | `therock-ci`                                      |
| [`therock-ci-artifacts-external`](https://therock-ci-artifacts-external.s3.amazonaws.com/) | Build outputs for forks and other repos | `therock-ci-external`, or runner base credentials |

### Release buckets

Each release type (`dev`, `nightly`, `prerelease`, `release`) has a matching
set of buckets.

The `dev`, `nightly`, and `prerelease` types are accessed via
the `therock-{release_type}` IAM role while stable `release` buckets are
manually promoted from prereleases via IAM user policies (see
[`how_to_do_release.md`](/build_tools/packaging/how_to_do_release.md)).

Python, tarball, and native package buckets are fronted by CloudFront CDNs —
prefer the CDN URLs for reading (e.g. `pip install --index-url`).

| Bucket                                                                                   | Contents        | IAM role             | CDN                                                                                                                         |
| ---------------------------------------------------------------------------------------- | --------------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| [`therock-dev-artifacts`](https://therock-dev-artifacts.s3.amazonaws.com/)               | Build outputs   | `therock-dev`        | —                                                                                                                           |
| [`therock-dev-packages`](https://therock-dev-packages.s3.amazonaws.com/)                 | Native packages | `therock-dev`        | [`rocm.devreleases.amd.com/deb/`](https://rocm.devreleases.amd.com/deb/), [`…/rpm/`](https://rocm.devreleases.amd.com/rpm/) |
| [`therock-dev-python`](https://therock-dev-python.s3.amazonaws.com/)                     | Python packages | `therock-dev`        | [`rocm.devreleases.amd.com/v2/`](https://rocm.devreleases.amd.com/v2/)                                                      |
| [`therock-dev-tarball`](https://therock-dev-tarball.s3.amazonaws.com/)                   | ROCm tarballs   | `therock-dev`        | [`rocm.devreleases.amd.com/tarball/`](https://rocm.devreleases.amd.com/tarball/)                                            |
| [`therock-nightly-artifacts`](https://therock-nightly-artifacts.s3.amazonaws.com/)       | Build outputs   | `therock-nightly`    | —                                                                                                                           |
| [`therock-nightly-packages`](https://therock-nightly-packages.s3.amazonaws.com/)         | Native packages | `therock-nightly`    | [`rocm.nightlies.amd.com/deb/`](https://rocm.nightlies.amd.com/deb/), [`…/rpm/`](https://rocm.nightlies.amd.com/rpm/)       |
| [`therock-nightly-python`](https://therock-nightly-python.s3.amazonaws.com/)             | Python packages | `therock-nightly`    | [`rocm.nightlies.amd.com/v2/`](https://rocm.nightlies.amd.com/v2/)                                                          |
| [`therock-nightly-tarball`](https://therock-nightly-tarball.s3.amazonaws.com/)           | ROCm tarballs   | `therock-nightly`    | [`rocm.nightlies.amd.com/tarball/`](https://rocm.nightlies.amd.com/tarball/)                                                |
| [`therock-prerelease-artifacts`](https://therock-prerelease-artifacts.s3.amazonaws.com/) | Build outputs   | `therock-prerelease` | —                                                                                                                           |
| `therock-prerelease-packages`                                                            | Native packages | `therock-prerelease` | [`rocm.prereleases.amd.com/packages/`](https://rocm.prereleases.amd.com/packages/)                                          |
| `therock-prerelease-python`                                                              | Python packages | `therock-prerelease` | [`rocm.prereleases.amd.com/whl/`](https://rocm.prereleases.amd.com/whl/)                                                    |
| `therock-prerelease-tarball`                                                             | ROCm tarballs   | `therock-prerelease` | [`rocm.prereleases.amd.com/tarball/`](https://rocm.prereleases.amd.com/tarball/)                                            |
| [`therock-release-artifacts`](https://therock-release-artifacts.s3.amazonaws.com/)       | Build outputs   | —                    | —                                                                                                                           |
| `therock-release-packages`                                                               | Native packages | —                    | [`repo.amd.com/rocm/packages/`](https://repo.amd.com/rocm/packages/)                                                        |
| `therock-release-python`                                                                 | Python packages | —                    | [`repo.amd.com/rocm/whl/`](https://repo.amd.com/rocm/whl/)                                                                  |
| `therock-release-tarball`                                                                | ROCm tarballs   | —                    | [`repo.amd.com/rocm/tarball/`](https://repo.amd.com/rocm/tarball/)                                                          |

### Build system buckets

We mirror third-party dependency files into S3 for use by the build system.

| Bucket                  | Contents                                                | Details                                                                                                                    |
| ----------------------- | ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `rocm-third-party-deps` | Mirrors for [`third_party/`](/third-party/) subprojects | See ["Updating a third-party mirror"](./git_chores.md#updating-a-third-party-mirror) in [`git_chores.md`](./git_chores.md) |

### Cache buckets

| Bucket                               | Contents                   | IAM role             |
| ------------------------------------ | -------------------------- | -------------------- |
| `therock-ci-pytorch-sccache`         | PyTorch CI sccache         | `therock-ci`         |
| `therock-dev-pytorch-sccache`        | PyTorch dev sccache        | `therock-dev`        |
| `therock-nightly-pytorch-sccache`    | PyTorch nightly sccache    | `therock-nightly`    |
| `therock-prerelease-pytorch-sccache` | PyTorch prerelease sccache | `therock-prerelease` |

### Legacy buckets

CI runs before 2025-11-11 (see [TheRock#2046](https://github.com/ROCm/TheRock/issues/2046))
used different bucket names. These are no longer written to but still contain
historical data. We may remove these once we implement a retention policy for
artifacts.

| Legacy bucket                                                                        | Replaced by                     | IAM role                     |
| ------------------------------------------------------------------------------------ | ------------------------------- | ---------------------------- |
| [`therock-artifacts`](https://therock-artifacts.s3.amazonaws.com/)                   | `therock-ci-artifacts`          | `therock-artifacts`          |
| [`therock-artifacts-external`](https://therock-artifacts-external.s3.amazonaws.com/) | `therock-ci-artifacts-external` | `therock-artifacts-external` |
