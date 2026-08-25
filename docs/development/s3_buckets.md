# S3 Buckets

TheRock uses Amazon S3 buckets to store CI build outputs (artifacts, logs,
python packages, etc.) and release artifacts. This page lists all buckets
and explains the authentication needed to upload to them.

## Table of contents

- [Authentication](#authentication)
- [Bucket inventory](#bucket-inventory)
  - [CI buckets](#ci-buckets): `therock-ci-artifacts`, `therock-ci-artifacts-external`
  - [Release buckets](#release-buckets): artifact handoff and `repo.amd.com` product buckets
  - [Build system buckets](#build-system-buckets): `rocm-third-party-deps`
  - [Cache buckets](#cache-buckets): `therock-pytorch-sccache-*`
  - [Legacy buckets](#legacy-buckets): `therock-artifacts`, `therock-artifacts-external`
- [Extending the inventory from another repository](#extending-the-inventory-from-another-repository)

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
    runs-on: aws-linux-scale-rocm-prod
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

Final publication to the `repo.amd.com` product buckets uses a separate role
in account `324352301041`. Use
[`configure_aws_product_publication_credentials`](/.github/actions/configure_aws_product_publication_credentials/action.yml)
with the product name. The role and bucket follow these patterns:

```text
arn:aws:iam::324352301041:role/therock-repo-<stream>-<product>
therock-repo-amd-<stream>-<product>
```

Artifact credentials and product-publication credentials are intentionally
separate. Do not use product credentials for intermediate artifact uploads.

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

**External repos and forks:**

- **External repos** (e.g., `rocm-libraries`) use OIDC with the
  `therock-ci-external` role to upload to `therock-ci-artifacts-external`.
- **Fork PRs** cannot use OIDC (no trust relationship). They fall back to
  runner base credentials.

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

Release publication has two stages:

1. Build workflows upload intermediate outputs to an artifact bucket in
   account `692859939525`.
1. Release workflows copy final outputs into product buckets in account
   `324352301041`, which are served through the stream-specific
   `repo.amd.com` domains.

Internal release types map to public streams as follows:

| Internal release type    | Public stream |
| ------------------------ | ------------- |
| `dev`                    | `dev`         |
| `nightly`                | `nightly`     |
| `prerelease`             | `rc`          |
| `dev-bkc`, `nightly-bkc` | `bkc`         |

#### Product release buckets

`<stream>` is one of `dev`, `nightly`, `rc`, or `bkc`.

| Bucket pattern                      | Contents                                                 | Object prefixes                                                        |
| ----------------------------------- | -------------------------------------------------------- | ---------------------------------------------------------------------- |
| `therock-repo-amd-<stream>-core`    | ROCm Core Python packages, tarballs, and native packages | `v5/rocm/core/{whl-next,tarball,tarball-asan,packages,packages-asan}/` |
| `therock-repo-amd-<stream>-pytorch` | PyTorch Python packages                                  | `v5/rocm/pytorch/whl-next/`                                            |
| `therock-repo-amd-<stream>-jax`     | JAX Python packages                                      | `v5/rocm/jax/whl-next/`                                                |

Public downloads use the matching stream hostname and omit the internal `v5`
prefix. For example, nightly Core tarballs are at
https://nightly.repo.amd.com/rocm/core/tarball/ and nightly native packages
are under https://nightly.repo.amd.com/rocm/core/packages/.

That rewrite is encoded, not just documented: each product bucket carries
`key_prefix="v5/"` and a `CdnRule` mapping `v5/` to
`https://<stream>.repo.amd.com/`, so `StorageLocation.public_url` derives the
address above from the S3 key. The rule is built from one
`f"https://{stream}.repo.amd.com/"` formula rather than a per-stream table,
because [RFC0012](../rfcs/RFC0012-Repo-Structure.md) gives every stream the same
hierarchy under its own subdomain — a stream that has not finished cutting over
needs no edit here when it does.

None of the product buckets are readable over raw S3 — they carry
`anonymous_s3_read=false`, and the stream CDN is the only public way in.

#### Index URLs

For "which URL does a user install this channel from", call these rather than
reading `cdn_rules` or writing the URL out:

```python
from _therock_utils.s3_buckets import (
    get_release_package_index_url,
    get_release_tarball_index_url,
    get_legacy_release_index_url,
)

get_release_package_index_url("nightly")  # pip --index-url
get_release_package_index_url("nightly", "pytorch")  # product-local index
get_release_tarball_index_url("release")  # where users browse tarballs
get_legacy_release_index_url("nightly")  # pre-RFC0012 layout
```

They accept `release` in addition to the publishable release types. The stable
channel is promoted by hand and has no automated upload credentials, so
`get_release_bucket_config` and `get_release_stream` keep rejecting it — but it
is the channel most users install from, so the read side accepts it. Keeping the
two sets separate is what lets one widen without widening the other.

Pip installs must use the aggregate index, such as
https://nightly.repo.amd.com/rocm/whl-next/. Product-local Python indexes are
publication and indexer inputs, not self-contained install entry points.

Stable releases are manually promoted and served from
https://stable.repo.amd.com/rocm/. The new layout begins with ROCm 10.1
nightlies and ROCm 10.0 stable releases. Older releases remain in the
[legacy multi-arch release locations](../packaging/legacy_multi_arch_releases.md).

#### Artifact and legacy release buckets

Artifact buckets remain the handoff point between build and release workflows.
The separate `packages`, `python`, and `tarball` buckets below contain releases
published with the legacy layout and remain available for historical releases.
Developer-facing current-release documentation should use the
stream-specific `repo.amd.com` URLs above. CI may read artifact S3 URLs
directly when consuming intermediate build outputs.

The mappings in the CDN column below are also encoded as `cdn_rules` on each
`S3BucketConfig` in
[`build_tools/_therock_utils/s3_buckets.py`](/build_tools/_therock_utils/s3_buckets.py),
so tooling can resolve them via `StorageLocation.public_url` instead of
re-deriving them. A bucket or key prefix with no rule falls back to its raw S3
URL. Not every cell below has a rule: the prerelease and release *package* CDNs
serve a distro-partitioned apt/dnf repository rather than a prefix rewrite of
the bucket, so no rule is emitted for them. **Keep the table and the rules in
sync when either changes.**

| Bucket                                                                                   | Contents        | IAM role             | CDN                                                                                                                                                                                     |
| ---------------------------------------------------------------------------------------- | --------------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`therock-dev-artifacts`](https://therock-dev-artifacts.s3.amazonaws.com/)               | Build outputs   | `therock-dev`        | —                                                                                                                                                                                       |
| [`therock-dev-packages`](https://therock-dev-packages.s3.amazonaws.com/)                 | Native packages | `therock-dev`        | [`rocm.devreleases.amd.com/packages-multi-arch/deb/`](https://rocm.devreleases.amd.com/packages-multi-arch/deb/), [`…/rpm/`](https://rocm.devreleases.amd.com/packages-multi-arch/rpm/) |
| [`therock-dev-python`](https://therock-dev-python.s3.amazonaws.com/)                     | Python packages | `therock-dev`        | [`rocm.devreleases.amd.com/whl-multi-arch/`](https://rocm.devreleases.amd.com/whl-multi-arch/)                                                                                          |
| [`therock-dev-tarball`](https://therock-dev-tarball.s3.amazonaws.com/)                   | ROCm tarballs   | `therock-dev`        | [`rocm.devreleases.amd.com/tarball-multi-arch/`](https://rocm.devreleases.amd.com/tarball-multi-arch/)                                                                                  |
| [`therock-nightly-artifacts`](https://therock-nightly-artifacts.s3.amazonaws.com/)       | Build outputs   | `therock-nightly`    | —                                                                                                                                                                                       |
| [`therock-nightly-packages`](https://therock-nightly-packages.s3.amazonaws.com/)         | Native packages | `therock-nightly`    | [`rocm.nightlies.amd.com/packages-multi-arch/deb/`](https://rocm.nightlies.amd.com/packages-multi-arch/deb/), [`…/rpm/`](https://rocm.nightlies.amd.com/packages-multi-arch/rpm/)       |
| [`therock-nightly-python`](https://therock-nightly-python.s3.amazonaws.com/)             | Python packages | `therock-nightly`    | [`rocm.nightlies.amd.com/whl-multi-arch/`](https://rocm.nightlies.amd.com/whl-multi-arch/)                                                                                              |
| [`therock-nightly-tarball`](https://therock-nightly-tarball.s3.amazonaws.com/)           | ROCm tarballs   | `therock-nightly`    | [`rocm.nightlies.amd.com/tarball-multi-arch/`](https://rocm.nightlies.amd.com/tarball-multi-arch/)                                                                                      |
| [`therock-prerelease-artifacts`](https://therock-prerelease-artifacts.s3.amazonaws.com/) | Build outputs   | `therock-prerelease` | —                                                                                                                                                                                       |
| `therock-prerelease-packages`                                                            | Native packages | `therock-prerelease` | [`rocm.prereleases.amd.com/packages-multi-arch/`](https://rocm.prereleases.amd.com/packages-multi-arch/)                                                                                |
| `therock-prerelease-python`                                                              | Python packages | `therock-prerelease` | [`rocm.prereleases.amd.com/whl-multi-arch/`](https://rocm.prereleases.amd.com/whl-multi-arch/)                                                                                          |
| `therock-prerelease-tarball`                                                             | ROCm tarballs   | `therock-prerelease` | [`rocm.prereleases.amd.com/tarball-multi-arch/`](https://rocm.prereleases.amd.com/tarball-multi-arch/)                                                                                  |
| [`therock-release-artifacts`](https://therock-release-artifacts.s3.amazonaws.com/)       | Build outputs   | —                    | —                                                                                                                                                                                       |
| `therock-release-packages`                                                               | Native packages | —                    | [`repo.amd.com/rocm/packages-multi-arch/`](https://repo.amd.com/rocm/packages-multi-arch/)                                                                                              |
| `therock-release-python`                                                                 | Python packages | —                    | [`repo.amd.com/rocm/whl-multi-arch/`](https://repo.amd.com/rocm/whl-multi-arch/)                                                                                                        |
| `therock-release-tarball`                                                                | ROCm tarballs   | —                    | [`repo.amd.com/rocm/tarball-multi-arch/`](https://repo.amd.com/rocm/tarball-multi-arch/)                                                                                                |

### Build system buckets

We mirror third-party dependency files into S3 for use by the build system.

| Bucket                  | Contents                                                | Details                                                                                                                    |
| ----------------------- | ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `rocm-third-party-deps` | Mirrors for [`third_party/`](/third-party/) subprojects | See ["Updating a third-party mirror"](./git_chores.md#updating-a-third-party-mirror) in [`git_chores.md`](./git_chores.md) |

### Cache buckets

Unlike the other buckets on this page, the PyTorch sccache buckets live in a
separate AWS account (`324352301041`) in region `us-east-1`. The IAM role names
are identical to the artifact-pipeline roles but resolve to a different account
ID; the corresponding role ARNs are
`arn:aws:iam::324352301041:role/therock-{ci,dev,nightly,prerelease}`. OIDC trust
for `therock-{dev,nightly}` also covers `repo:ROCm/rockrel:*` so reusable
workflow invocations from `rockrel` can assume the role.

| Bucket                               | Contents                   | IAM role             |
| ------------------------------------ | -------------------------- | -------------------- |
| `therock-pytorch-sccache-ci`         | PyTorch CI sccache         | `therock-ci`         |
| `therock-pytorch-sccache-dev`        | PyTorch dev sccache        | `therock-dev`        |
| `therock-pytorch-sccache-nightly`    | PyTorch nightly sccache    | `therock-nightly`    |
| `therock-pytorch-sccache-prerelease` | PyTorch prerelease sccache | `therock-prerelease` |

### Legacy buckets

CI runs before 2025-11-11 (see [TheRock#2046](https://github.com/ROCm/TheRock/issues/2046))
used different bucket names. These are no longer written to but still contain
historical data. We may remove these once we implement a retention policy for
artifacts.

| Legacy bucket                                                                        | Replaced by                     | IAM role                     |
| ------------------------------------------------------------------------------------ | ------------------------------- | ---------------------------- |
| [`therock-artifacts`](https://therock-artifacts.s3.amazonaws.com/)                   | `therock-ci-artifacts`          | `therock-artifacts`          |
| [`therock-artifacts-external`](https://therock-artifacts-external.s3.amazonaws.com/) | `therock-ci-artifacts-external` | `therock-artifacts-external` |

## Extending the inventory from another repository

Repositories that reuse TheRock's build tools against their own buckets can add
to the inventory above with a JSON registry file, instead of patching the
scripts. Point at it either way:

- `--bucket-config-file <path>`, on `artifact_manager.py` subcommands. Prefer
  this: it makes a single invocation self-describing, so the registry in use is
  visible in the command line rather than inherited unnoticed from a parent
  process.
- `THEROCK_S3_BUCKETS_FILE=<path>` in the environment, for wrapper scripts that
  shell out to several entry points and cannot pass a flag to each.

Both are process-wide: whichever is used sets one registry for the whole
invocation, read by every backend it builds. That is deliberate — the registry
answers "which buckets exist and where do they map", which does not vary between
the two ends of an `artifact_manager copy`. What does vary is the transport, and
that is a separate per-end flag (`--source-transport`).

The flag wins when both are set, and the choice is logged.

```json
{
  "version": 1,
  "buckets": [
    {
      "name": "therock-npi-artifacts",
      "iam_role": "therock-npi",
      "key_prefix": "v3/artifacts/",
      "anonymous_s3_read": false,
      "cdn_rules": [
        {
          "key_prefix": "v3/artifacts/",
          "url_prefix": "https://genesis.example.com/artifacts/"
        }
      ]
    }
  ],
  "artifacts_buckets": {
    "ci": "therock-npi-artifacts",
    "ci-external": "therock-npi-artifacts"
  }
}
```

`buckets` registers metadata. `key_prefix` is the prefix the bucket stores
everything under, and is folded into every key the tools generate; it is
normalized to a trailing `/` and must not begin with one. `cdn_rules` map a key
prefix to a public URL prefix, longest match first.

A rule maps a *prefix*, not a whole bucket, because the public path is usually
not the S3 path: `v5/rocm/core/tarball/` in `therock-repo-amd-nightly-core` is
served at `https://nightly.repo.amd.com/rocm/core/tarball/`. The rule carries
both the prefix to strip and the URL to substitute. For a bucket whose whole
contents map to one URL, use `"key_prefix": ""`.

`anonymous_s3_read` (default `true`) says whether the bucket can be read over
raw S3 without credentials. It selects which URL `StorageLocation.download_url`
hands to pip, apt/dnf, or a plain download: the raw S3 URL when reads there work
(CI avoids CloudFront data-transfer charges that way), the CDN when they do not.
Set it `false` for a private bucket — the prerelease and release buckets are
private in-tree, and a private bucket handed a raw S3 URL produces a link that
answers 403. A private bucket with no `cdn_rules` covering the key raises rather
than returning an unusable URL.

`namespace_external_repos` (default `false`) makes uploads to the bucket
additionally namespaced by `{owner}-{repo}/`. Set it on any bucket that more
than one repository writes to — `therock-ci-artifacts-external` is the in-tree
example. Without it, keys are namespaced only by GitHub run ID, and run IDs are
allocated per repository rather than globally, so two repositories sharing a
bucket will eventually collide on a run ID and overwrite each other's artifacts.

`artifacts_buckets`, `release_buckets` and `product_release_buckets` override
*selection* — which bucket the lookup functions choose. Registration alone is
not enough, because those functions compute a bucket name from a formula
(`therock-{release_type}-artifacts`, `therock-repo-amd-{stream}-{product}`) that
a downstream repository does not follow.

| Key                       | Slots                                                                         |
| ------------------------- | ----------------------------------------------------------------------------- |
| `artifacts_buckets`       | `ci`, `ci-external`, `dev`, `dev-bkc`, `nightly`, `nightly-bkc`, `prerelease` |
| `release_buckets`         | release type, then `tarball`, `python`, or `packages`                         |
| `product_release_buckets` | release type, then `core`, `pytorch`, or `jax`                                |

The BKC release types have their own `artifacts_buckets` slots even though they
share dev's and nightly's buckets in-tree, so a downstream repository can
redirect a BKC channel without also redirecting the channel it shares a bucket
with.

Note the two CI slots. A repository other than `ROCm/TheRock` selects
`ci-external` for **all** of its CI, not just fork PRs, so a downstream repo
normally sets both — to the same bucket, if fork PRs need no separate
destination. They are separate keys deliberately: overriding only `ci` must not
silently redirect untrusted fork uploads into a trusted bucket.

Merge rules:

- Additive. A bucket name that already exists in TheRock's inventory is
  **rejected** unless the entry sets `"override": true`, in which case it fully
  replaces the built-in entry (no field-wise merge) and the replacement is
  logged. Silently retargeting a production bucket from an inherited environment
  variable would be the worst failure this mechanism could have, so it is
  opt-in and loud.
- Unknown keys at any level are an error. A typo'd `cdn_rule` would otherwise
  mean "this bucket has no CDN", which is exactly the silently-wrong-URL
  outcome the registry exists to prevent.
- Selection overrides must name a bucket registered in the same file or in-tree.
- Every error message names the file being loaded.
