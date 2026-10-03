# Updating ROCm aggregate Python index ownership

[`rocm_whl_next_ownership.yaml`](rocm_whl_next_ownership.yaml) defines which
packages are exposed through `/rocm/whl-next/`, their owning product paths,
and the streams where they are available.

## 1. Update dependency policy when needed

For a new mirrored dependency, add its version policy to `DEPENDENCIES` in
[`mirror_python_dependencies.py`](mirror_python_dependencies.py). Mirroring
publishes to the core product index; it does not automatically expose the
package through the aggregate index.

See [Mirroring Third-Party Python Dependencies](../../../docs/packaging/python_packaging.md#mirroring-third-party-python-dependencies)
for the mirroring commands and publication behavior.

## 2. Add aggregate ownership

If the package should be available through the aggregate index, add its PEP 503
normalized name to the manifest. For example, `example_package` becomes
`example-package`:

```yaml
      # Mirrored dependencies use core, regardless of their policy's project.
      example-package:
        owner_path: core/whl-next
        # Select the streams where this package will actually be published.
        streams: [dev, nightly]
```

Omitting stream selection uses the default group, which currently excludes BKC.
`stream_group: all` includes all known streams, including BKC. Use the intended
publication policy to choose availability.

A package present in a product index without ownership for the selected stream
gets no aggregate root link or CloudFront route. Its aggregate URL returns 404.
Product indexes may intentionally contain packages outside aggregate ownership.

## 3. Publish and verify product content

Publish the package to the affected product buckets and wait for the product
indexer to create its package page and wheel links. Verify those links before
deploying a new aggregate route. For mirrored dependencies, the public package
page is `/rocm/core/whl-next/<normalized-package>/`.

## 4. Generate and deploy aggregate artifacts

Use `tools/cloudfront_functions/generate_rocm_python_index_and_router.py` in the infra
repository. The generator calls this repository's `aggregate_index.py` and the
CloudFront routing generator to produce matching HTML, route JSON, validation
reports, and CloudFront routing functions. The step-by-step procedure is in
`tools/cloudfront_functions/rocm-python-index-deployment.md` in the infra repository.

The procedure offers three validation levels:

| Mode                | Missing owned packages | Unowned product packages         |
| ------------------- | ---------------------- | -------------------------------- |
| Manifest only       | Not checked            | Not checked; no routes generated |
| Content validation  | Fail                   | Tolerated; no routes generated   |
| Strict completeness | Fail                   | Fail                             |

Content validation is useful when product indexes intentionally contain unowned
packages. Strict completeness additionally requires the checked product roots
to contain only packages exposed through the aggregate index for that stream.
Skipping strict completeness never adds ownership or routes.

### Optional content snapshot

Prepare a snapshot only when using content validation; manifest-only generation
does not require one.

Downloading a snapshot requires AWS credentials with read access to the selected
product buckets. Developers without this access can use manifest-only generation.

`--content-root` is a local directory containing downloaded product index HTML,
not a bucket name or URL. Prepare it separately; neither the validator nor the
infra generator downloads it.

Use the product indexes for the stream being validated:
`/rocm/core/whl-next/`, `/rocm/pytorch/whl-next/`, and `/rocm/jax/whl-next/`,
as applicable. Do not use the aggregate `/rocm/whl-next/` index: the validator
compares product content against aggregate ownership.

1. Choose one stream and a new snapshot directory. Include every product root
   referenced by active ownership for that stream. For strict completeness,
   also include existing product roots whose owners appear only in inactive
   manifest entries.
1. Download each product's root `index.html` and its linked package
   `index.html` pages. Preserve the public paths beneath the snapshot directory.
   If downloading from S3, omit the storage prefix `v5/` from local paths.
   Wheel files are not needed.
1. Keep the root HTML unmodified, including links to unowned packages. Filtering
   those links would hide mismatches from strict completeness validation.
1. Pass the snapshot directory to `--content-root` with the same `--stream`.
   Missing roots, missing owned package links, or empty owned package pages fail
   content validation. Linked wheel objects are not checked.

For example, download nightly HTML from the product buckets into a fresh local
snapshot. These commands read S3 and write local files only:

```bash
# Copy core index HTML, excluding wheels and other artifacts.
aws s3 sync s3://therock-repo-amd-nightly-core/v5/rocm/core/whl-next/ \
  /tmp/rocm-whl-next-nightly-snapshot/rocm/core/whl-next/ \
  --exclude '*' --include 'index.html' --include '*/index.html'

# Copy the matching nightly PyTorch index HTML.
aws s3 sync s3://therock-repo-amd-nightly-pytorch/v5/rocm/pytorch/whl-next/ \
  /tmp/rocm-whl-next-nightly-snapshot/rocm/pytorch/whl-next/ \
  --exclude '*' --include 'index.html' --include '*/index.html'

# Copy the matching nightly JAX index HTML.
aws s3 sync s3://therock-repo-amd-nightly-jax/v5/rocm/jax/whl-next/ \
  /tmp/rocm-whl-next-nightly-snapshot/rocm/jax/whl-next/ \
  --exclude '*' --include 'index.html' --include '*/index.html'
```

The resulting layout is:

```text
/tmp/rocm-whl-next-nightly-snapshot/
  rocm/core/whl-next/index.html
  rocm/core/whl-next/<package>/index.html
  rocm/pytorch/whl-next/index.html
  rocm/pytorch/whl-next/<package>/index.html
  rocm/jax/whl-next/index.html
  rocm/jax/whl-next/<package>/index.html
```

Use `--content-root /tmp/rocm-whl-next-nightly-snapshot --stream nightly`.
For other streams, adjust both the source buckets and local directory. Product
roots may change while downloading; if validation exposes inconsistent pages,
prepare a fresh snapshot after publication has completed.

### Deploy generated artifacts

For additions, deploy the CloudFront routing functions first and upload the
matching aggregate root HTML afterwards. Once the product page and wheels exist,
the routing functions enable pip access: pip requests the normalized package URL
directly without consulting the aggregate root listing.

A version-only change for an already owned package does not require regenerating
aggregate routes or root HTML.
