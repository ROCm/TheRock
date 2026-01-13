# External Organization Workflow Integration for EMU Repositories

This document provides guidance for setting up GitHub EMU (Enterprise Managed User) repositories to use TheRock's CI workflows for building and testing, while maintaining separate S3 buckets with different permissions.

## Overview

EMU repositories can leverage TheRock's existing CI infrastructure by calling TheRock's `ci.yml` workflow using the `workflow_call` trigger. This allows:

- **Reuse of TheRock's build infrastructure**: No need to duplicate complex build logic
- **Separate S3 buckets**: Artifacts stored in organization-specific buckets with different IAM permissions
- **Hidden repository details**: No external organization repository names or URLs stored in TheRock's codebase
- **Flexible source management**: EMU repos handle their own source checkout

## Architecture

```
EMU Repo PR/Push → EMU rocm_ci_caller.yml (with s3_bucket_override)
                         ↓
                   TheRock ci.yml (workflow_call)
                         ↓
              Build & Test Jobs (use overridden bucket)
                         ↓
              Upload/Download from Custom S3 Bucket
```

## Prerequisites

### 1. AWS Infrastructure Setup

**S3 Bucket:**
- Create a dedicated S3 bucket for your organization's artifacts (e.g., `therock-external-org-ci-artifacts`)
- Configure appropriate bucket policies and permissions

**IAM Role:**
- Create an IAM role for GitHub Actions OIDC (e.g., `therock-external-org-ci`)
- Configure trust relationship for your EMU organization
- Grant permissions to your organization's S3 bucket

Example trust policy:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::YOUR_ACCOUNT:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": "repo:YOUR_EMU_ORG/*:*"
        }
      }
    }
  ]
}
```

### 2. GitHub Secrets Configuration

In your EMU repository settings, configure:
- `AWS_ROLE_ARN`: ARN of the IAM role created above
- `S3_BUCKET_NAME`: Name of your organization's S3 bucket (optional, can be hardcoded)

## EMU Repository Setup

### Step 1: Create Workflow Caller

Create `.github/workflows/rocm_ci_caller.yml` in your EMU repository:

```yaml
name: ROCm CI Caller

on:
  pull_request:
    branches: [main, develop]
    types: [opened, reopened, synchronize]
  push:
    branches: [main, develop]
  workflow_dispatch:
    inputs:
      linux_amdgpu_families:
        type: string
        description: "GPU families to build/test (e.g., 'gfx94X,gfx110X')"
        default: "gfx94X"
      projects:
        type: string
        description: "Projects to build/test (leave blank for auto-detect)"
        default: ""

jobs:
  call-therock-ci:
    # Call TheRock's reusable CI workflow
    uses: ROCm/TheRock/.github/workflows/ci.yml@mainline
    secrets: inherit
    with:
      # Indicate this is an external repository
      external_source_checkout: true
      
      # TheRock ref to use (mainline, specific tag, or branch)
      therock_ref: "mainline"
      
      # Pass the calling repository name for proper identification
      repository_override: ${{ github.repository }}
      
      # CRITICAL: Override S3 bucket for organization artifacts
      # Use GitHub secret or hardcode your bucket name
      s3_bucket_override: ${{ secrets.S3_BUCKET_NAME }}
      # Alternative: s3_bucket_override: "therock-external-org-ci-artifacts"
      
      # Specify GPU families to build/test
      linux_amdgpu_families: ${{ inputs.linux_amdgpu_families || 'gfx94X,gfx110X' }}
      windows_amdgpu_families: "gfx110X"
      
      # Optional: specify projects explicitly or leave blank for auto-detection
      projects: ${{ inputs.projects }}
    permissions:
      contents: read
      id-token: write  # Required for AWS OIDC authentication
```

### Step 2: Configure AWS Credentials Step (if needed)

If your EMU organization uses different AWS credential configuration, you may need to add a custom AWS configure step. However, TheRock's workflows should automatically handle OIDC authentication if your IAM role is properly configured.

## How It Works

### Bucket Override Flow

1. **Workflow Call**: EMU repo calls TheRock's `ci.yml` with `s3_bucket_override` parameter
2. **Environment Propagation**: TheRock workflows set `THEROCK_S3_BUCKET_OVERRIDE` environment variable
3. **Bucket Selection**: Python scripts call `retrieve_bucket_info()` which checks for override first
4. **Artifact Storage**: All uploads/downloads use the overridden bucket

### Key Workflow Parameters

| Parameter | Purpose | Required |
|-----------|---------|----------|
| `external_source_checkout` | Indicates external repo | Yes |
| `therock_ref` | TheRock version to use | Yes |
| `repository_override` | Identifies calling repo | Yes |
| `s3_bucket_override` | Custom S3 bucket | **Yes (for EMU orgs)** |
| `linux_amdgpu_families` | GPU targets (Linux) | Optional |
| `windows_amdgpu_families` | GPU targets (Windows) | Optional |
| `projects` | Project paths to build/test | Optional |

## Project Detection

TheRock workflows can auto-detect which projects to build based on file changes:

- **Auto-detection**: Leave `projects` parameter empty
- **Explicit projects**: Set `projects: "projects/rocprim,projects/hipcub"`
- **All projects**: Set `projects: "all"`

**Note**: For EMU repos without project mapping in TheRock, explicit project specification is recommended.

## Testing Your Integration

### 1. Manual Test

```bash
# Trigger workflow manually
gh workflow run rocm_ci_caller.yml \
  --ref your-branch \
  -f linux_amdgpu_families=gfx94X \
  -f projects=projects/rocprim
```

### 2. Verify Bucket Usage

Check workflow logs for:
```
Retrieving bucket info...
Using S3 bucket override: therock-external-org-ci-artifacts
  github_repository: YOUR_EMU_ORG/your-repo
  external_repo: YOUR_EMU_ORG-your-repo/
  bucket (override): therock-external-org-ci-artifacts
```

### 3. Verify S3 Artifacts

Artifacts should appear in S3 at:
```
s3://therock-external-org-ci-artifacts/YOUR_EMU_ORG-your-repo/RUN_ID-linux/
s3://therock-external-org-ci-artifacts/YOUR_EMU_ORG-your-repo/RUN_ID-windows/
```

## Troubleshooting

### Issue: Bucket override not working

**Check:**
1. `s3_bucket_override` is passed to `ci.yml`
2. Environment variable `THEROCK_S3_BUCKET_OVERRIDE` is set in job logs
3. Python scripts are using `retrieve_bucket_info()` from `github_actions_utils.py`

### Issue: AWS authentication failure

**Check:**
1. IAM role trust policy includes your EMU organization
2. `id-token: write` permission is set in workflow
3. IAM role has permissions to your organization's S3 bucket

### Issue: Projects not being built

**Check:**
1. Explicitly specify projects using `projects` input
2. Verify project paths exist in your repository structure
3. Check configure_ci.py logs for project detection results

## Security Considerations

1. **No Hardcoded Repo Names**: External organization repository names never appear in TheRock's codebase
2. **Separate Buckets**: Artifacts isolated with different IAM permissions
3. **GitHub Secrets**: Store sensitive bucket names in GitHub secrets
4. **OIDC Authentication**: Use GitHub Actions OIDC for secure AWS access
5. **Least Privilege**: IAM roles should have minimal required permissions

## Maintenance

### Updating TheRock Version

Update `therock_ref` in your workflow caller:
```yaml
therock_ref: "mainline"  # or specific tag like "v1.2.3"
```

### Adding New GPU Families

Update the `linux_amdgpu_families` and `windows_amdgpu_families` parameters as needed.

## Example: Complete EMU Repo Structure

```
your-emu-repo/
├── .github/
│   └── workflows/
│       └── rocm_ci_caller.yml        # Calls TheRock CI
├── projects/
│   ├── rocprim/                       # Your projects
│   └── hipcub/
├── CMakeLists.txt                     # Your build config
└── README.md
```

## Support

For issues or questions about this integration:
- Check TheRock workflow logs for detailed error messages
- Verify AWS and S3 configurations
- Review TheRock's `ci.yml` workflow for latest parameter options

## Changelog

- **2026-01**: Added `s3_bucket_override` support for EMU repositories
- **2025-11**: Initial external repository workflow_call support for rocm-libraries/rocm-systems
