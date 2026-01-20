## Motivation

This PR refactors the JAX wheels build workflow to support multiple Python versions and JAX branches through a matrix-based approach, bringing it into alignment with the PyTorch wheels release pattern. Previously, `release_portable_linux_packages.yml` directly triggered `build_linux_jax_wheels.yml` with a hardcoded Python 3.12 version, limiting flexibility for testing and releasing JAX wheels across different Python versions.

The new architecture enables:
- Building JAX wheels for Python 3.11, 3.12, and 3.13 simultaneously
- Support for multiple JAX branch references (currently `rocm-jaxlib-v0.8.0`, easily extensible)
- Consistent workflow patterns across PyTorch and JAX wheel releases
- Better job naming and traceability in CI/CD pipelines

## Technical Details

### 1. Created `release_portable_linux_jax_wheels.yml`
New workflow file that acts as a matrix orchestrator, similar to `release_portable_linux_pytorch_wheels.yml`:
- Defines matrix strategy for Python versions: `["3.11", "3.12", "3.13"]`
- Defines matrix strategy for JAX branches: `["rocm-jaxlib-v0.8.0"]`
- Accepts inputs for `amdgpu_family`, `release_type`, S3 configuration, CloudFront URLs, `rocm_version`, `tar_url`, and `ref`
- Triggers `build_linux_jax_wheels.yml` for each matrix combination

### 2. Updated `build_linux_jax_wheels.yml`
- **Added `jax_ref` input to `workflow_call`** section (previously only in `workflow_dispatch`)
- **Removed hardcoded matrix strategy** from the job (moved to calling workflow)
- **Updated JAX checkout reference** from `${{ matrix.jax_ref }}` to `${{ inputs.jax_ref }}` (line 129)
- **Enhanced job naming** to include JAX ref: `Build Linux JAX Wheels | {amdgpu_family} | Python {version} | {jax_ref}`

### 3. Updated `release_portable_linux_packages.yml`
- **Changed workflow trigger** from `build_linux_jax_wheels.yml` to `release_portable_linux_jax_wheels.yml`
- **Updated step name** to "Trigger release JAX wheels"
- **Removed hardcoded `python_version`** (now handled by matrix in release workflow)
- **Fixed S3 path handling**: Changed `s3_subdir` from `S3_STAGING_SUBDIR` to `S3_SUBDIR`
- **Added missing parameters**: `s3_staging_subdir`, `cloudfront_url`, `cloudfront_staging_url`, and `ref`

### Workflow Architecture
```
release_portable_linux_packages.yml
    ↓ triggers
release_portable_linux_jax_wheels.yml
    ↓ matrix: [py 3.11, 3.12, 3.13] × [jax branches]
build_linux_jax_wheels.yml
    ↓ calls
test_linux_jax_wheels.yml
```

## Test Plan

- [ ] Verify `release_portable_linux_jax_wheels.yml` can be manually triggered via workflow_dispatch
- [ ] Confirm matrix builds trigger for all Python versions (3.11, 3.12, 3.13)
- [ ] Validate that `release_portable_linux_packages.yml` successfully triggers the new release workflow
- [ ] Check that JAX wheels are built and uploaded to correct S3 paths (both staging and release)
- [ ] Ensure proper CloudFront URL parameters are passed through the workflow chain
- [ ] Verify backward compatibility: existing `build_linux_jax_wheels.yml` workflow_dispatch still works

## Test Result

- All workflow YAML files pass linter validation with no errors
- Changes are structurally consistent with existing PyTorch wheels workflow pattern
- Branch `users/kithumma/jax-branches-python-support` created and pushed with all changes

## Submission Checklist

- [ ] Look over the contributing guidelines at https://github.com/ROCm/ROCm/blob/develop/CONTRIBUTING.md#pull-requests.
