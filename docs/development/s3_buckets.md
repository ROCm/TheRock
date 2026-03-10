# S3 Buckets

TheRock uses Amazon S3 buckets to store CI build outputs (artifacts, logs,
manifests, python packages) and release artifacts. This page lists all buckets
and explains the authentication needed to upload to them.

## CI buckets

| Bucket                                                                                     | Contents                                                   | IAM role                                          |
| ------------------------------------------------------------------------------------------ | ---------------------------------------------------------- | ------------------------------------------------- |
| [`therock-ci-artifacts`](https://therock-ci-artifacts.s3.amazonaws.com/)                   | Build artifacts, logs, manifests for `ROCm/TheRock`        | `therock-ci`                                      |
| [`therock-ci-artifacts-external`](https://therock-ci-artifacts-external.s3.amazonaws.com/) | Build artifacts, logs, manifests for forks and other repos | `therock-ci-external`, or runner base credentials |

## Release buckets

Each release type (`dev`, `nightly`, `prerelease`) has a matching set of
buckets. All buckets for a given release type are accessed via the
`therock-{release_type}` IAM role.

| Bucket                                                                                   | Contents                          | IAM role             |
| ---------------------------------------------------------------------------------------- | --------------------------------- | -------------------- |
| [`therock-dev-artifacts`](https://therock-dev-artifacts.s3.amazonaws.com/)               | Build artifacts, logs, manifests  | `therock-dev`        |
| [`therock-dev-python`](https://therock-dev-python.s3.amazonaws.com/)                     | Python wheels and pip index pages | `therock-dev`        |
| [`therock-dev-tarball`](https://therock-dev-tarball.s3.amazonaws.com/)                   | ROCm SDK tarballs                 | `therock-dev`        |
| [`therock-nightly-artifacts`](https://therock-nightly-artifacts.s3.amazonaws.com/)       | Build artifacts, logs, manifests  | `therock-nightly`    |
| [`therock-nightly-python`](https://therock-nightly-python.s3.amazonaws.com/)             | Python wheels and pip index pages | `therock-nightly`    |
| [`therock-nightly-tarball`](https://therock-nightly-tarball.s3.amazonaws.com/)           | ROCm SDK tarballs                 | `therock-nightly`    |
| [`therock-prerelease-artifacts`](https://therock-prerelease-artifacts.s3.amazonaws.com/) | Build artifacts, logs, manifests  | `therock-prerelease` |
| [`therock-prerelease-python`](https://therock-prerelease-python.s3.amazonaws.com/)       | Python wheels and pip index pages | `therock-prerelease` |
| [`therock-prerelease-tarball`](https://therock-prerelease-tarball.s3.amazonaws.com/)     | ROCm SDK tarballs                 | `therock-prerelease` |

## Cache buckets

| Bucket                                                                                               | Contents                   | IAM role             |
| ---------------------------------------------------------------------------------------------------- | -------------------------- | -------------------- |
| [`therock-ci-pytorch-sccache`](https://therock-ci-pytorch-sccache.s3.amazonaws.com/)                 | PyTorch CI sccache         | `therock-ci`         |
| [`therock-dev-pytorch-sccache`](https://therock-dev-pytorch-sccache.s3.amazonaws.com/)               | PyTorch dev sccache        | `therock-dev`        |
| [`therock-nightly-pytorch-sccache`](https://therock-nightly-pytorch-sccache.s3.amazonaws.com/)       | PyTorch nightly sccache    | `therock-nightly`    |
| [`therock-prerelease-pytorch-sccache`](https://therock-prerelease-pytorch-sccache.s3.amazonaws.com/) | PyTorch prerelease sccache | `therock-prerelease` |

## Authentication

All buckets except `therock-ci-artifacts-external` require assuming an IAM
role via
[`aws-actions/configure-aws-credentials`](https://github.com/aws-actions/configure-aws-credentials)
using OIDC. This requires `id-token: write` in the job's `permissions` block.
The full ARN pattern is
`arn:aws:iam::692859939525:role/therock-{ci,dev,nightly,prerelease}`.

Our CI runners come with baseline credentials that allow uploading to
`therock-ci-artifacts-external` without any extra setup. Workflows in
downstream repos like `rocm-libraries`, `rocm-systems`, and `llvm-project`
upload to this bucket and do not need `aws-actions/configure-aws-credentials`.

## Legacy buckets

Runs before 2025-11-11 ([TheRock #2046](https://github.com/ROCm/TheRock/issues/2046))
used different bucket names. These are no longer written to but still contain
historical data.

| Legacy bucket                                                                        | Replaced by                     | IAM role                     |
| ------------------------------------------------------------------------------------ | ------------------------------- | ---------------------------- |
| [`therock-artifacts`](https://therock-artifacts.s3.amazonaws.com/)                   | `therock-ci-artifacts`          | `therock-artifacts`          |
| [`therock-artifacts-external`](https://therock-artifacts-external.s3.amazonaws.com/) | `therock-ci-artifacts-external` | `therock-artifacts-external` |
