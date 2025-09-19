# s3_management

## Overview

These scripts are forked from https://github.com/pytorch/test-infra/tree/main/s3_management.

* [`update_dependencies.py`](./update_dependencies.py) mirrors dependency
  packages from PyPI to our S3 buckets
* [`manage.py`](./manage.py) generates
  [PEP 503](https://peps.python.org/pep-0503/)-compliant index.html files for
  each subdirectory in our S3 buckets, including packages we build
  (e.g. `rocm`, `torch`) and dependencies (uploaded by `update_dependencies.py`)

The Python package buckets we maintain are:

S3 bucket name | S3 URL | User-facing URLs
-- | -- | --
`therock-dev-python` | https://therock-dev-python.s3.amazonaws.com/ | <ul><li>https://d25kgig7rdsyks.cloudfront.net/v2/</li><li>https://d25kgig7rdsyks.cloudfront.net/v2-staging/</li></ul>
`therock-nightly-python` | https://therock-nightly-python.s3.amazonaws.com/ | <ul><li>https://rocm.nightlies.amd.com/v2/</li><li>https://rocm.nightlies.amd.com/v2-staging/</li></ul>

Each bucket has `v2` and `v2-staging` top level folders at the moment. This may
evolve with `v3` in the future. Within each folder there are subfolders for
each index we publish, currently corresponding to each GPU family that we
build releases for. See these other pages for more details:
* [Index page listing in `RELEASES.md`](https://github.com/ROCm/TheRock/blob/main/RELEASES.md#index-page-listing)
* [Gating releases with Pytorch tests in `external-builds/pytorch/README.md`](/external-builds/pytorch/README.md#gating-releases-with-pytorch-tests)

The user-facing URLs can be used with `pip install --index-url`. For example:

```bash
python -m pip install \
  --index-url https://rocm.nightlies.amd.com/v2/gfx94X-dcgpu/ \
  "rocm[libraries,devel]"
```

## Playbook for running the scripts

While these scripts do run as part of some automated workflows with
automatically assumed roles that grant CI/CD machines access, these scripts do
still need to be run manually by developers under certain conditions.

> [!NOTE]
> See also these pytorch/test-infra issues, which we could collaborate on since these scripts are forked:
> * [test-infra/issues/6097 - Automate upload/sync of dependencies to pytorch s3 index](https://github.com/pytorch/test-infra/issues/6097)
> * [test-infra/issues/6105 - Automate creation and population of pytorch index subfolder in s3](https://github.com/pytorch/test-infra/issues/6105)

### Obtaining credentials

You will need credentials to be able to upload to our S3 buckets.

The manual way to do this is to create a user account in our IAM and
grant it these policies:

* `therock-dev-releases-access`
* `therock-nightly-releases-access`

> [!WARNING]
> **Be careful** with the "nightly" access, since that is effectively "prod".
> Always test changes in "dev" first and consider revoking nightly access after
> you are done using it.
>
> We plan on setting up a separate repository with isolated access for more
> user-facing releases in place of these developer nightly releases, but the
> general principle will still apply.

### Adding a new package dependency

Let's say pytorch adds a new dependency and we want to include that dependency
on our python package index pages. Here's how we would do that:

First, edit [`update_dependencies.py`](./update_dependencies.py) with the new
package name, version, and project mapping.

Note which `--package {jax,torch,rocm}` your changes affect. Then, roll out
those changes to the dev bucket:

1. Sanity test your changes with a dry run:

    ```bash
    export S3_BUCKET_PY=therock-dev-python
    python ./build_tools/third_party/s3_management/update_dependencies.py --package rocm --dry-run
    ```

1. Update dependencies in the dev bucket:

    ```bash
    export S3_BUCKET_PY=therock-dev-python
    python ./build_tools/third_party/s3_management/update_dependencies.py --package rocm
    ```

1. Regenerate the index pages for the dev bucket:

    ```bash
    export S3_BUCKET_PY=therock-dev-python
    python ./build_tools/third_party/s3_management/manage.py all
    ```

1. Visit the URL to check that the index pages look as expected and include the
  new dep. For example: https://d25kgig7rdsyks.cloudfront.net/v2/gfx120X-all/.

Finally, repeat those steps for the nightly bucket:

1. Update dependencies in the nightly bucket:

    ```bash
    export S3_BUCKET_PY=therock-nightly-python
    python ./build_tools/third_party/s3_management/update_dependencies.py --package rocm
    ```

1. Regenerate the index pages for the nightly bucket:

    ```bash
    export S3_BUCKET_PY=therock-nightly-python
    python ./build_tools/third_party/s3_management/manage.py all
    ```

1. Visit the URL to check that the index pages look as expected and include the
  new dep. For example: https://rocm.nightlies.amd.com/v2/gfx120X-all/.

1. If no longer needed, revoke your access to `therock-nightly-releases-access`.

### If something goes wrong

Bucket Versioning is enabled in S3 for the `therock-nightly-python` bucket.
Changes can be rolled back to that bucket using that feature.

Bucket Versioning is **NOT** (yet?) enabled in S3 for the `therock-dev-python`
bucket.
