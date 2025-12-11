# Using Strix AI Test Workflows

## 🎯 Overview

We've created **two custom workflows** specifically for Strix AI testing that operate independently from the main CI pipeline:

1. **`strix_ai_tests.yml`** - Full-featured workflow with comprehensive options
2. **`strix_ai_quick_test.yml`** - Lightweight workflow for rapid testing

These workflows can be triggered **manually** and run on actual Strix hardware without modifying the main CI configuration.

---

## 🚀 Workflow 1: Full Strix AI Tests

### **File:** `.github/workflows/strix_ai_tests.yml`

**Purpose:** Comprehensive AI/ML testing with full control over test parameters

**Triggers:**
- ✅ **Automatic:** Runs on push/PR to Strix test files
- ✅ **Manual:** Can be triggered on-demand with custom parameters

### **Automatic Triggers**

The workflow runs automatically when you push changes to:

```yaml
Branches:
  - users/*/strix_*      # Any user's strix branch
  - main                 # Main branch
  - develop              # Develop branch

Paths (must change at least one):
  - tests/strix_ai/**                  # Any Strix AI test file
  - .github/workflows/strix_ai*.yml    # Workflow files
  - build_tools/_therock_utils/**      # Utility files
```

**Automatic Run Configuration:**
- Platform: Linux (default)
- Strix Variant: gfx1151 (Strix Halo)
- Test Category: quick (smoke tests)
- Test Type: quick

**Skip Automatic Run:**
Add `[skip-strix-tests]` to your commit message:
```bash
git commit -m "Update docs [skip-strix-tests]"
```

### **Manual Trigger (Override Defaults)**

#### **Via GitHub UI:**

1. Go to your repository on GitHub
2. Click **Actions** tab
3. Select **"Strix AI/ML Testing"** from the left sidebar
4. Click **"Run workflow"** button (top right)
5. Fill in the parameters:

| Parameter | Options | Description |
|-----------|---------|-------------|
| **platform** | linux, windows | Target platform |
| **strix_variant** | gfx1150 (Point), gfx1151 (Halo) | GPU variant |
| **test_category** | all, vlm, vla, vit, cv, optimization, quick | Which tests to run |
| **test_type** | smoke, quick, full | Test execution mode |
| **runner_label** | (optional) | Custom runner label |
| **artifact_run_id** | (optional) | Specific build artifacts |

6. Click **"Run workflow"**

#### **Via GitHub CLI:**

```bash
# Run VLM tests on Linux Strix Halo
gh workflow run strix_ai_tests.yml \
  -f platform=linux \
  -f strix_variant=gfx1151 \
  -f test_category=vlm \
  -f test_type=quick

# Run all AI tests on Windows Strix Halo
gh workflow run strix_ai_tests.yml \
  -f platform=windows \
  -f strix_variant=gfx1151 \
  -f test_category=all \
  -f test_type=full

# Run CV tests on Linux Strix Point
gh workflow run strix_ai_tests.yml \
  -f platform=linux \
  -f strix_variant=gfx1150 \
  -f test_category=cv \
  -f test_type=smoke
```

#### **Via REST API:**

```bash
curl -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/ROCm/TheRock/actions/workflows/strix_ai_tests.yml/dispatches \
  -d '{
    "ref": "users/rponnuru/strix_poc",
    "inputs": {
      "platform": "linux",
      "strix_variant": "gfx1151",
      "test_category": "vla",
      "test_type": "quick"
    }
  }'
```

### **Test Categories**

| Category | Tests Run | Typical Duration |
|----------|-----------|------------------|
| **quick** | Smoke tests only | 5-10 min |
| **vlm** | Vision Language Models (CLIP) | 10-15 min |
| **vla** | Vision Language Action (OWL-ViT) | 15-20 min |
| **vit** | Vision Transformers | 15-20 min |
| **cv** | Computer Vision (YOLO) | 20-30 min |
| **optimization** | Quantization tests | 10-15 min |
| **all** | All tests (non-slow) | 60-90 min |

### **Test Types**

| Type | Description | Use When |
|------|-------------|----------|
| **smoke** | Quick validation tests | Sanity checking |
| **quick** | Reduced test set | Regular validation |
| **full** | Complete test suite | Before merge, comprehensive validation |

---

## 🔄 **Automatic vs Manual Behavior**

| Aspect | Automatic (Push/PR) | Manual (workflow_dispatch) |
|--------|---------------------|----------------------------|
| **Trigger** | On file changes | On demand |
| **Platform** | Linux (default) | Choose: Linux or Windows |
| **Strix Variant** | gfx1151 (default) | Choose: gfx1150 or gfx1151 |
| **Test Category** | quick (default) | Choose: all, vlm, vla, vit, cv, etc. |
| **Test Type** | quick (default) | Choose: smoke, quick, or full |
| **Duration** | 5-10 min | Depends on selection |

**Example Automatic Run:**
```bash
# Push to your branch
git add tests/strix_ai/vlm/test_clip.py
git commit -m "Improve CLIP test"
git push

# Workflow automatically runs:
# - Platform: linux
# - Variant: gfx1151
# - Category: quick
# - Type: quick
```

**Example Manual Run:**
```bash
# Override all defaults
gh workflow run strix_ai_tests.yml \
  -f platform=windows \
  -f strix_variant=gfx1150 \
  -f test_category=all \
  -f test_type=full
```

---

## ⚡ Workflow 2: Quick Test

### **File:** `.github/workflows/strix_ai_quick_test.yml`

**Purpose:** Rapid testing of specific test files or functions - perfect for debugging

### **How to Run**

#### **Via GitHub UI:**

1. Go to **Actions** → **"Strix AI Quick Test"**
2. Click **"Run workflow"**
3. Enter:
   - **test_command**: Path to test (e.g., `tests/strix_ai/vlm/test_clip.py`)
   - **strix_platform**: `linux-gfx1151`, `windows-gfx1151`, or `linux-gfx1150`
4. Click **"Run workflow"**

#### **Examples:**

**Test a specific file:**
```bash
gh workflow run strix_ai_quick_test.yml \
  -f test_command='tests/strix_ai/vlm/test_clip.py' \
  -f strix_platform='linux-gfx1151'
```

**Test a specific function:**
```bash
gh workflow run strix_ai_quick_test.yml \
  -f test_command='tests/strix_ai/vla/test_action_prediction.py::TestVisionLanguageAction::test_vla_visual_grounding' \
  -f strix_platform='linux-gfx1151'
```

**Test with pytest markers:**
```bash
gh workflow run strix_ai_quick_test.yml \
  -f test_command='tests/strix_ai/ -m "quick and vlm"' \
  -f strix_platform='linux-gfx1151'
```

**Just verify pytest works:**
```bash
gh workflow run strix_ai_quick_test.yml \
  -f test_command='tests/strix_ai/test_simple.py' \
  -f strix_platform='linux-gfx1151'
```

---

## 📊 Viewing Results

### **1. Workflow Run Page**

After triggering, click on the workflow run to see:
- Live logs
- Job status
- Test output
- Errors and failures

### **2. Test Results Artifacts**

Test results are saved as JUnit XML:
- Go to workflow run
- Scroll to **Artifacts** section
- Download `strix-ai-test-results-{platform}-{variant}-{category}`
- View in any JUnit viewer

### **3. Logs**

Detailed logs show:
```
=== Strix AI Test Summary ===
Platform: linux
Strix Variant: gfx1151
Test Category: vlm
Test Type: quick
Runner: linux-strix-halo-gpu-rocm-001
Status: success

🧠 Loading CLIP model...
🔍 Processing inputs...
⚡ Running CLIP inference on Strix...
✅ Text matching probabilities: [0.95, 0.03, 0.02]
✅ CLIP test passed on gfx1151!
```

---

## 🎯 Common Use Cases

### **Use Case 1: Validate VLM Tests**

```bash
# Quick validation
gh workflow run strix_ai_tests.yml \
  -f platform=linux \
  -f strix_variant=gfx1151 \
  -f test_category=vlm \
  -f test_type=quick
```

### **Use Case 2: Debug Specific Test**

```bash
# Run single test with full output
gh workflow run strix_ai_quick_test.yml \
  -f test_command='tests/strix_ai/vla/test_action_prediction.py::TestVisionLanguageAction::test_vla_visual_grounding -v -s' \
  -f strix_platform='linux-gfx1151'
```

### **Use Case 3: Full Validation Before Merge**

```bash
# Run all tests on both platforms
gh workflow run strix_ai_tests.yml \
  -f platform=linux \
  -f strix_variant=gfx1151 \
  -f test_category=all \
  -f test_type=full

gh workflow run strix_ai_tests.yml \
  -f platform=windows \
  -f strix_variant=gfx1151 \
  -f test_category=all \
  -f test_type=full
```

### **Use Case 4: Test on Both Strix Variants**

```bash
# Test on Strix Point (gfx1150)
gh workflow run strix_ai_tests.yml \
  -f platform=linux \
  -f strix_variant=gfx1150 \
  -f test_category=quick \
  -f test_type=smoke

# Test on Strix Halo (gfx1151)
gh workflow run strix_ai_tests.yml \
  -f platform=linux \
  -f strix_variant=gfx1151 \
  -f test_category=quick \
  -f test_type=smoke
```

### **Use Case 5: Performance Benchmarking**

```bash
# Run optimization tests to measure performance
gh workflow run strix_ai_tests.yml \
  -f platform=linux \
  -f strix_variant=gfx1151 \
  -f test_category=optimization \
  -f test_type=full
```

---

## 🔧 Workflow Features

### **Environment Variables Set**

```yaml
THEROCK_BIN_DIR: ./build/bin          # ROCm binaries
AMDGPU_FAMILIES: gfx1151              # GPU family
TEST_TYPE: quick                      # Test mode
PYTHONUNBUFFERED: 1                   # Immediate output
```

### **Automatic Features**

- ✅ **Dependency installation** - transformers, torch, ultralytics, etc.
- ✅ **GPU validation** - Checks ROCm/GPU before tests
- ✅ **Artifact upload** - Test results saved automatically
- ✅ **Error handling** - Tests continue even if deps fail (for validation)
- ✅ **Platform detection** - Automatic shell and runner selection

### **Runners Used**

| Platform | Runner Label | GPU |
|----------|--------------|-----|
| **Linux Strix Halo** | `linux-strix-halo-gpu-rocm` | gfx1151 |
| **Windows Strix Halo** | `windows-strix-halo-gpu-rocm` | gfx1151 |
| **Linux Strix Point** | `linux-strix-point-gpu-rocm` | gfx1150 (if available) |

---

## 📋 Comparison: Custom Workflow vs Main CI

| Feature | Custom Workflow | Main CI |
|---------|----------------|---------|
| **Trigger** | Manual | Automatic (PR/push) |
| **Flexibility** | High - choose tests | Fixed - runs all |
| **Dependencies** | Installed per run | Pre-configured |
| **Runners** | Strix-specific | Multi-platform |
| **Integration** | Independent | Part of pipeline |
| **Speed** | Fast (targeted) | Slower (comprehensive) |
| **Use Case** | Development, validation | Release gating |

---

## 🎨 Advanced Usage

### **Custom Runner**

If you have a custom Strix runner:

```bash
gh workflow run strix_ai_tests.yml \
  -f platform=linux \
  -f strix_variant=gfx1151 \
  -f test_category=vlm \
  -f test_type=quick \
  -f runner_label='my-custom-strix-runner'
```

### **Specific Build Artifacts**

Test against specific build artifacts:

```bash
gh workflow run strix_ai_tests.yml \
  -f platform=linux \
  -f strix_variant=gfx1151 \
  -f test_category=vit \
  -f test_type=quick \
  -f artifact_run_id='1234567890'
```

### **Parallel Testing**

Run multiple categories in parallel by triggering multiple workflows:

```bash
# Start all in parallel
for category in vlm vla vit cv; do
  gh workflow run strix_ai_tests.yml \
    -f platform=linux \
    -f strix_variant=gfx1151 \
    -f test_category=$category \
    -f test_type=quick &
done
wait
```

---

## 🐛 Troubleshooting

### **Workflow Not Listed**

Make sure the workflow files are on your branch:
```bash
git add .github/workflows/strix_ai_*.yml
git commit -m "Add Strix AI test workflows"
git push
```

### **Runner Not Available**

If runner error occurs:
- Check runner label matches your infrastructure
- Use `runner_label` input to specify custom runner
- Contact infra team for runner access

### **Test Failures**

1. **Check logs** for specific error
2. **Download artifacts** for detailed JUnit results
3. **Run quick test** with verbose output to debug

### **Dependencies Failed**

This is expected on Windows (torch long paths). Tests will skip gracefully:
```
Warning: torch installation failed, tests may skip
```

---

## ✅ Benefits of Custom Workflows

1. **🚀 Fast Iteration** - Test only what you need
2. **🔧 Flexible** - Choose platform, variant, category
3. **🧪 Safe** - Doesn't affect main CI
4. **📊 Trackable** - Results saved as artifacts
5. **🎯 Targeted** - Debug specific tests easily
6. **🔄 Repeatable** - Easy to re-run with same parameters
7. **👥 Team-Friendly** - Anyone can trigger tests

---

## 📚 Next Steps

1. **Try quick test first:**
   ```bash
   gh workflow run strix_ai_quick_test.yml \
     -f test_command='tests/strix_ai/test_simple.py' \
     -f strix_platform='linux-gfx1151'
   ```

2. **Run VLM tests:**
   ```bash
   gh workflow run strix_ai_tests.yml \
     -f test_category=vlm \
     -f test_type=quick
   ```

3. **View results** in Actions tab

4. **Iterate** based on results

---

## 🔗 Related Documentation

- [STRIX_CI_WORKFLOW.md](./STRIX_CI_WORKFLOW.md) - Main CI integration
- [STRIX_TESTING_GUIDE.md](./STRIX_TESTING_GUIDE.md) - Test overview
- [STRIX_AI_ML_TEST_PLAN.md](./STRIX_AI_ML_TEST_PLAN.md) - Test plan
- [TROUBLESHOOTING.md](../../tests/strix_ai/TROUBLESHOOTING.md) - Common issues

---

**Ready to test?** Just go to Actions → Strix AI Testing → Run workflow! 🚀

