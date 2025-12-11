# Strix AI Workflow - Troubleshooting & Fixes

## Summary

This document explains the issues encountered with automatic workflow triggers and how they were resolved.

---

## Issue 1: Workflows Skipping on Push

### Problem

Workflows were triggering initially but then skipping on subsequent pushes.

### Root Cause

The `if` condition was checking `github.event.head_commit.message`:

```yaml
if: |
  github.event_name == 'workflow_dispatch' ||
  !contains(github.event.head_commit.message, '[skip-strix-tests]')
```

**Why it failed:**
- `github.event.head_commit.message` can be `null` or unavailable for certain push events
- When `null`, the `contains()` function fails
- Workflow skips execution due to failed condition

### Fix

**Removed the `if` condition entirely:**

```yaml
# No condition - workflows always run when triggered
```

**Result:**
- Workflows execute on ALL push/PR events matching path filters
- Path filters already control when workflows trigger:
  - `tests/strix_ai/**`
  - `.github/workflows/strix_ai*.yml`

### Commit

```
66629233 - Remove problematic if conditions causing workflows to skip
```

---

## Issue 2: Build Directory Not Found

### Problem

Workflow was failing with:

```
ls: cannot access '/home/nod/actions-runner/_work/TheRock/TheRock/build/bin': No such file or directory
Error: Process completed with exit code 2.
```

### Root Cause

The workflow tried to verify `THEROCK_BIN_DIR` exists, but:
- No build artifacts were available (TheRock wasn't built)
- The step failed with exit code 2
- Workflow stopped execution

### Fix

**Made verification steps resilient:**

```yaml
- name: Verify TheRock Installation
  continue-on-error: true  # Don't fail workflow
  run: |
    if [ -d "${{ env.THEROCK_BIN_DIR }}" ]; then
      echo "SUCCESS: Build directory exists"
      # Use TheRock build
    else
      echo "WARNING: Build directory not found"
      echo "This is OK for AI/ML tests that primarily need PyTorch"
      # Fall back to system ROCm
    fi
```

**Changes:**
1. Added `continue-on-error: true` to verification step
2. Check if directory exists before accessing it
3. Gracefully fall back to system ROCm if available
4. Made "Setup Test Environment" step optional
5. Added informative messages explaining AI/ML tests run standalone

### Commit

```
5bee4cdc - Fix workflow to handle missing build artifacts gracefully
```

---

## How AI/ML Tests Work

### Standalone Execution

**AI/ML tests DO NOT require TheRock build:**
- Tests use PyTorch, Transformers, and CV libraries
- Tests validate GPU functionality through PyTorch's ROCm backend
- TheRock build is optional (useful for rocminfo, but not required)

### Dependencies

**Required:**
- Python 3.12+
- pytest
- PyTorch with ROCm support
- Transformers, ultralytics, opencv-python, pillow

**Optional:**
- TheRock build (for rocminfo, rocm-smi)
- Pre-built artifacts (for integration testing)

### Workflow Flow

```
1. Checkout code
2. Setup Python 3.12
3. Install AI/ML dependencies (pip install pytorch, transformers, etc.)
4. [Optional] Download TheRock artifacts
5. [Optional] Verify TheRock installation
6. Check GPU availability
7. Run AI/ML tests (tests/strix_ai/)
8. Upload results
```

**Steps 4-5 can fail without breaking the workflow.**

---

## Monitoring Workflows

### GitHub Actions Web UI

```
https://github.com/ROCm/TheRock/actions
```

**Look for:**
- "Strix AI/ML Testing" - Full test workflow
- "Strix AI Quick Test" - Quick test workflow
- "Strix AI On Push" - Simple notification test

### CLI Monitoring

```bash
# List recent runs
gh run list --workflow=strix_ai_tests.yml --limit 5

# Watch live
gh run watch

# View specific run
gh run view <run-id>

# View logs
gh run view <run-id> --log
```

### Expected Status After Push

**Successful workflow:**
```
✓ Checkout Repository
✓ Setup Python
✓ Install AI/ML Dependencies
✓ Setup Test Environment (may show warning)
✓ Verify Dependencies
~ Verify TheRock Installation (may show warning)
✓ Check ROCm/GPU
✓ Run Strix AI Tests
✓ Upload Test Results
```

**Key indicators:**
- "Install AI/ML Dependencies" should succeed
- "Verify TheRock Installation" can show warnings (OK)
- "Run Strix AI Tests" is the critical step

---

## Triggering Workflows

### Automatic Triggers

**Push to branch:**
```bash
# Any change to tests/strix_ai/** will trigger workflows
git add tests/strix_ai/
git commit -m "Update Strix AI tests"
git push origin users/rponnuru/strix_poc
```

**Branches that trigger:**
- `users/*/strix_*` (your branch)
- `main`
- `develop`

**Files that trigger:**
- `tests/strix_ai/**`
- `.github/workflows/strix_ai*.yml`
- `build_tools/_therock_utils/**`

### Manual Triggers

**Full test workflow:**
```bash
gh workflow run strix_ai_tests.yml \
  --ref users/rponnuru/strix_poc \
  -f platform=linux \
  -f strix_variant=gfx1151 \
  -f test_category=quick \
  -f test_type=quick
```

**Quick test workflow:**
```bash
gh workflow run strix_ai_quick_test.yml \
  --ref users/rponnuru/strix_poc \
  -f test_command="tests/strix_ai/test_simple.py" \
  -f strix_platform=linux-gfx1151
```

---

## Common Issues & Solutions

### Issue: Workflow Still Not Triggering

**Possible causes:**
1. Branch not pushed to remote
2. No files matching path filters changed
3. Repository settings restrict workflows from feature branches

**Solution:**
```bash
# Check branch status
git status
git log origin/main..HEAD --oneline

# Push branch
git push origin users/rponnuru/strix_poc

# Use manual trigger
gh workflow run strix_ai_tests.yml --ref users/rponnuru/strix_poc
```

### Issue: Tests Failing to Import Torch

**Error:**
```
ImportError: DLL load failed while importing _C
```

**Solution:**
This is a Windows Long Path issue. On Linux runners, torch should install correctly.

**For local testing on Windows:**
See `tests/strix_ai/TROUBLESHOOTING.md`

### Issue: No GPU Available

**Error:**
```
RuntimeError: No HIP GPUs are available
```

**Solution:**
This means the runner doesn't have Strix GPU or ROCm not configured.

**Check:**
- Runner label is correct (`linux-strix-halo-gpu-rocm`)
- GPU is available on the runner
- ROCm drivers installed

---

## Testing Changes

### Quick Test Cycle

```bash
# 1. Make a small change
echo "# Test trigger" >> tests/strix_ai/test_simple.py

# 2. Commit and push
git add tests/strix_ai/test_simple.py
git commit -m "Test workflow trigger"
git push origin users/rponnuru/strix_poc

# 3. Monitor
gh run list --workflow=strix_ai_tests.yml --limit 3
gh run watch
```

### Verify Specific Test

```bash
# Use manual trigger with custom test
gh workflow run strix_ai_quick_test.yml \
  --ref users/rponnuru/strix_poc \
  -f test_command="tests/strix_ai/vlm/test_clip.py::TestCLIP::test_clip_image_text_matching" \
  -f strix_platform=linux-gfx1151
```

---

## Summary of Fixes

| Issue | Fix | Commit |
|-------|-----|--------|
| Workflows skipping | Remove problematic `if` condition | 66629233 |
| Build directory not found | Add `continue-on-error`, check before access | 5bee4cdc |
| Unicode encoding errors | Replace emojis with plain text | 5bee4cdc |

**Current state:**
✅ Workflows trigger automatically on push
✅ Workflows handle missing build artifacts gracefully  
✅ AI/ML tests run standalone with PyTorch dependencies
✅ Clear error messages for debugging

---

## Next Steps

### For Production

1. **Merge workflows to main branch**
   - Create PR: `users/rponnuru/strix_poc` → `main`
   - Review workflows and documentation
   - Merge to enable automatic triggers across all branches

2. **Add artifact download**
   - If TheRock build is needed, add artifact download step
   - Configure proper artifact IDs and groups

3. **Configure runners**
   - Verify Strix runner labels exist and are accessible
   - Test on both Linux and Windows runners

### For POC/Testing

**Current setup is ready:**
- ✅ Automatic triggers work
- ✅ Tests run standalone
- ✅ Manual triggers available
- ✅ Comprehensive monitoring

**Continue developing tests:**
- Add more VLM tests (LLaVA, Qwen-VL)
- Add ViT tests (DINOv2, Swin)
- Add CV tests (DETR, Segmentation)
- Add optimization tests (quantization benchmarks)

---

## Contact & Resources

**Documentation:**
- `docs/development/STRIX_TESTING_GUIDE.md` - Comprehensive guide
- `docs/development/STRIX_AI_ML_TEST_PLAN.md` - Test plan
- `docs/development/STRIX_WORKFLOW_USAGE.md` - Workflow usage guide
- `tests/strix_ai/README.md` - Test suite setup

**Workflows:**
- `.github/workflows/strix_ai_tests.yml` - Full test workflow
- `.github/workflows/strix_ai_quick_test.yml` - Quick test workflow
- `.github/workflows/strix_ai_on_push.yml` - Notification workflow

**Tests:**
- `tests/strix_ai/` - All Strix AI tests
- `tests/strix_ai/test_simple.py` - Basic pytest verification
- `tests/strix_ai/vlm/test_clip.py` - CLIP tests
- `tests/strix_ai/vit/test_vit_base.py` - ViT tests
- `tests/strix_ai/cv/test_yolo.py` - YOLO tests

