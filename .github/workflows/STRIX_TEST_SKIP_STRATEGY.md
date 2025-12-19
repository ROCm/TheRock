# Strix AI Test Skip Strategy

## Overview

All Strix AI/ML tests have been updated to skip gracefully when models or dependencies are unavailable, rather than failing. This provides a better developer experience and clearer CI/CD feedback.

## Changes Implemented

### 1. **Graceful Test Skipping**

All test files now skip with informative messages when:
- **Models not available**: Model hasn't been published to Hugging Face yet
- **Authentication required**: Model requires HF_TOKEN for access (gated models)
- **Missing dependencies**: Required libraries not installed
- **Insufficient resources**: Not enough GPU memory for the model
- **Tools not configured**: Profiling tools (rocprofv3) not available

### 2. **Workflow Execution Strategy**

The workflow now has two execution modes:

#### **Automatic Triggers (Push/Pull Request)**
- **Only Quick Smoke Tests run automatically**
- Fast feedback on basic functionality
- Won't clutter CI/CD with expected skips from unavailable models
- Runs on push to `users/rponnuru/strix_poc` branch

#### **Manual Triggers (workflow_dispatch)**
- **All test categories available**
- Select specific category: VLM, VLA, Omni, Instruct, etc.
- Select `all` to run all tests
- Choose market segment for benchmarks
- Control test type: functional, performance, or full

### 3. **Test Files Updated**

**24 test files** across all categories:

#### Vision-Language Models (VLM)
- `test_qwen25_vl.py` - Added `load_model_or_skip()` helper
- `test_qwen3_vl.py` - Skip with model availability message
- `test_clip_blip.py` - Skip with model availability message
- `test_clip.py` - Skip with model availability message

#### Vision-Language-Action (VLA)
- `test_openvla.py` - Skip with model availability message
- `test_pi0.py` - Skip with model availability message
- `test_action_prediction.py` - Skip with model availability message

#### Other Categories
- `test_qwen_omni.py` (Omni)
- `test_qwen3_instruct.py` (Instruct)
- `test_sam2.py` (Segmentation)
- `test_zipformer.py` (ASR)
- `test_crossformer.py` (ASR)
- `test_flux_sd.py` (Diffusion)
- `test_llama32.py` (LLM)
- `test_deepseek_r1.py` (LLM)
- `test_awq_quantization.py` (Optimization)
- `test_gptq_quantization.py` (Optimization)
- `test_quantization.py` (Optimization)
- `test_strix_rocprofv3.py` (Profiling)
- `test_strix_models_profiling.py` (Profiling)
- `test_profile_existing_tests.py` (Profiling)
- `test_vit_base.py` (ViT)
- `test_yolo.py` (CV)

### 4. **Utility Module Added**

Created `tests/strix_ai/test_utils.py` with:
- `skip_if_model_unavailable()` decorator
- `load_model_safe()` function
- Common error pattern detection
- Informative skip messages

### 5. **Workflow Job Conditions Fixed**

**Before:**
```yaml
if: github.event.inputs.test_category == 'vlm' || github.event_name != 'workflow_dispatch'
```
This caused **all jobs to run** on push/PR, even if models weren't available.

**After:**
```yaml
# Most jobs: Only run on manual trigger
if: github.event_name == 'workflow_dispatch' && (github.event.inputs.test_category == 'all' || github.event.inputs.test_category == 'vlm')

# Quick tests: Run automatically OR on manual trigger
if: github.event_name != 'workflow_dispatch' || github.event.inputs.test_category == 'quick' || github.event.inputs.test_category == 'all'
```

## Benefits

### ✅ Clear Feedback
Instead of cryptic failures, tests skip with clear messages:
```
SKIPPED: Qwen2.5-VL-Instruct-3B-AWQ model not yet available on Hugging Face.
         Test will be enabled when model is published and accessible.
```

### ✅ Reduced CI/CD Noise
- No false failures from unavailable models
- Only Quick tests run automatically
- Comprehensive tests run when explicitly requested

### ✅ Better Development Flow
- Developers know exactly why tests are skipped
- Can identify missing HF tokens vs. unpublished models
- Can see which dependencies are missing

### ✅ Flexible Testing
- Run all tests: `test_category: all`
- Run specific category: `test_category: vlm`
- Run only quick tests: `test_category: quick` (default)
- Filter by market segment for benchmarks

## Test Execution Examples

### Run All VLM Tests
1. Go to Actions → Strix AI/ML Comprehensive Testing
2. Click "Run workflow"
3. Select `test_category: vlm`
4. Click "Run workflow"

### Run Quick Smoke Tests (Automatic)
- Just push changes to `users/rponnuru/strix_poc`
- Quick tests run automatically
- Fast feedback in ~5-10 minutes

### Run Full Test Suite
1. Go to Actions → Strix AI/ML Comprehensive Testing
2. Click "Run workflow"
3. Select `test_category: all`
4. Select `test_type: full`
5. Click "Run workflow"

## Next Steps: Enabling Tests

When models become available, update the test files:

### Example: Enable Qwen2.5-VL-3B-AWQ

1. **Remove the skip** in `test_qwen25_vl.py`:
```python
# Current (skips):
def test_qwen25_vl_3b_awq_load(check_strix_gpu):
    model, processor = load_model_or_skip(model_id, "test_qwen25_vl_3b_awq_load")
    # ... test continues if model loads

# When model is available, the load_model_or_skip() function
# will automatically stop skipping and run the test
```

2. **Or add HF_TOKEN** to GitHub Secrets for gated models:
   - Go to Repository Settings → Secrets and variables → Actions
   - Add `HF_TOKEN` with your Hugging Face token
   - Update workflow to pass token: `HF_TOKEN: ${{ secrets.HF_TOKEN }}`

3. **Install missing dependencies** if needed:
   - Update the "Install Dependencies" step in workflow
   - Add required packages: `pip install package-name`

## Summary

All tests now skip gracefully with clear, actionable messages. The workflow only runs Quick tests automatically, preventing CI/CD noise from unavailable models. Developers can manually trigger comprehensive tests when needed, and enabling tests is straightforward once models become available.

