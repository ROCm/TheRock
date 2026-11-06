## Updating ROCm aggregate Python index ownership

[`rocm_whl_next_ownership.yaml`](rocm_whl_next_ownership.yaml) defines which
packages are exposed through `/rocm/whl-next/`, their owning product paths,
and the streams where they are available.

### Audience and required access

This workflow spans public source changes and internal deployment:

- Any contributor can complete steps 1 and 2 and submit the changes through a
  TheRock pull request. These steps do not require infrastructure access.
- An infrastructure maintainer completes steps 3 and 4. They need permission to
  publish product content, access the internal infrastructure repository, deploy
  CloudFront functions, and upload the aggregate root index.
- The HTTPS snapshot helper does not require AWS access. Preparing an
  authoritative snapshot directly from S3 requires read access to the
  corresponding product buckets.

Contributors without the deployment permissions should coordinate the handoff
with an infrastructure maintainer after updating the dependency policy and
ownership manifest.

### 1. Update dependency policy when needed

For a new mirrored dependency, add its version policy to `DEPENDENCIES` in
[`mirror_python_dependencies.py`](mirror_python_dependencies.py). Mirroring
publishes to the core product index; it does not automatically expose the
package through the aggregate index.

See [Mirroring Third-Party Python Dependencies](../../../docs/packaging/python_packaging.md#mirroring-third-party-python-dependencies)
for the mirroring commands and publication behavior.

### 2. Add aggregate ownership

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

### 3. Publish and verify product content

Publish the package to the affected product buckets and wait for the product
indexer to create its package page and wheel links. Verify those links before
deploying a new aggregate route. For mirrored dependencies, the public package
page is `/rocm/core/whl-next/<normalized-package>/`.

### 4. Generate and deploy aggregate artifacts

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

#### Optional content validation

Skip this section for manifest-only generation.

1. Download the product index HTML for the stream being validated into a new
   local snapshot directory. The helper downloads from a public HTTPS origin,
   so it does not require AWS credentials. The output directory must not already
   exist:

```bash
python build_tools/packaging/python/download_rocm_python_index_snapshot.py \
  --stream nightly \
  --origin https://nightly.repo.amd.com \
  --output-dir /tmp/rocm-whl-next-nightly-snapshot
```

2. Pass the snapshot directory to the infrastructure generator with the matching
   stream:

```bash
python tools/cloudfront_functions/generate_rocm_python_index_and_router.py \
  --therock-repo /path/to/TheRock \
  --stream nightly \
  --content-root /tmp/rocm-whl-next-nightly-snapshot \
  --output-dir /tmp/rocm-whl-next-output
```

##### Snapshot layout and requirements

- `--content-root` is a local directory, not a bucket name or URL.
- The helper downloads product indexes such as `/rocm/core/whl-next/`,
  `/rocm/pytorch/whl-next/`, and `/rocm/jax/whl-next/`. It does not download the
  aggregate `/rocm/whl-next/` index.
- A product root referenced by active ownership for the stream is required. A
  root referenced only by inactive ownership is included when present and
  skipped when it returns 404.
- The helper downloads each product's root `index.html` and every linked package
  `index.html`, including pages for unowned packages. Wheel files are not needed
  or checked.
- The helper preserves the public paths below `/rocm/` and publishes the output
  directory only after the complete snapshot has downloaded and parsed.
- Keeping unowned package links allows strict completeness validation to detect
  ownership gaps.
- Missing roots, missing owned package links, and empty owned package pages fail
  content validation.

The snapshot used in the example has this layout:

```text
/tmp/rocm-whl-next-nightly-snapshot/
  rocm/core/whl-next/index.html
  rocm/core/whl-next/<package>/index.html
  rocm/pytorch/whl-next/index.html
  rocm/pytorch/whl-next/<package>/index.html
  rocm/jax/whl-next/index.html
  rocm/jax/whl-next/<package>/index.html
```

For other streams, use that stream's public HTTPS origin and a new local
directory. The HTTPS view may be cached, and product roots may change while
downloading. If validation exposes inconsistent pages, prepare a fresh snapshot
after publication has completed. Future deployment automation may prepare the
same layout directly from S3 when authoritative bucket access is available.

#### Deploy generated artifacts

For additions, deploy the CloudFront routing functions first and upload the
matching aggregate root HTML afterwards. Once the product page and wheels exist,
the routing functions enable pip access: pip requests the normalized package URL
directly without consulting the aggregate root listing.

A version-only change for an already owned package does not require regenerating
aggregate routes or root HTML.
