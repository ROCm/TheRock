# S3 Buckets

TheRock uses Amazon S3 buckets to store CI build outputs (artifacts, logs,
manifests, python packages) and release artifacts. This page lists all buckets
and explains the authentication needed to upload to them.

## CI buckets

| Bucket                                                                                     | Used for                                                                      | Upload authentication methods                                                                               |
| ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| [`therock-ci-artifacts`](https://therock-ci-artifacts.s3.amazonaws.com/)                   | CI runs on `ROCm/TheRock`                                                     | <ul><li>`therock-ci` role (OIDC)</li></ul>                                                                  |
| [`therock-ci-artifacts-external`](https://therock-ci-artifacts-external.s3.amazonaws.com/) | CI runs from forks and other repos<br>(e.g. `rocm-libraries`, `rocm-systems`) | <ul><li>`therock-ci-external` role (OIDC)</li><li>Runner base credentials (no extra setup needed)</li></ul> |

## Release buckets

| Bucket                                                                                   | Used for                  | Upload authentication methods                      |
| ---------------------------------------------------------------------------------------- | ------------------------- | -------------------------------------------------- |
| [`therock-dev-artifacts`](https://therock-dev-artifacts.s3.amazonaws.com/)               | Release type `dev`        | <ul><li>`therock-dev` role (OIDC)</li></ul>        |
| [`therock-nightly-artifacts`](https://therock-nightly-artifacts.s3.amazonaws.com/)       | Release type `nightly`    | <ul><li>`therock-nightly` role (OIDC)</li></ul>    |
| [`therock-prerelease-artifacts`](https://therock-prerelease-artifacts.s3.amazonaws.com/) | Release type `prerelease` | <ul><li>`therock-prerelease` role (OIDC)</li></ul> |

## Authentication

Our CI runners come with baseline credentials that allow uploading to
`therock-ci-artifacts-external`. To upload to any other bucket, workflows must
assume an IAM role via
[`aws-actions/configure-aws-credentials`](https://github.com/aws-actions/configure-aws-credentials)
using OIDC (`id-token: write` permission). The role name follows the pattern
`arn:aws:iam::692859939525:role/therock-{ci,dev,nightly,prerelease}`.

Workflows in downstream repos like `rocm-libraries`, `rocm-systems`, and
`llvm-project` upload to `therock-ci-artifacts-external` and do not need to
run `aws-actions/configure-aws-credentials` at all.

## Legacy buckets

Runs before 2025-11-11 ([TheRock #2046](https://github.com/ROCm/TheRock/issues/2046))
used different bucket names. These are no longer written to but still contain
historical data.

| Legacy bucket                                                                        | Replaced by                     | Upload auth                                                |
| ------------------------------------------------------------------------------------ | ------------------------------- | ---------------------------------------------------------- |
| [`therock-artifacts`](https://therock-artifacts.s3.amazonaws.com/)                   | `therock-ci-artifacts`          | <ul><li>`therock-artifacts` role (OIDC)</li></ul>          |
| [`therock-artifacts-external`](https://therock-artifacts-external.s3.amazonaws.com/) | `therock-ci-artifacts-external` | <ul><li>`therock-artifacts-external` role (OIDC)</li></ul> |
