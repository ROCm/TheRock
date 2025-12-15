# Strix Profiling Guide: Do You Need a Separate `profiling/` Folder?

## 🤔 **Short Answer: NO! You can profile existing tests directly.**

## 📊 **Three Approaches to Profiling**

### **Approach 1: Profile Existing Tests Directly** ⭐ **RECOMMENDED**

**No code changes needed!** Just wrap pytest with rocprofv3:

```bash
# Profile the existing CLIP test
rocprofv3 --hip-trace --kernel-trace --memory-copy-trace \
          --output-format pftrace -d ./clip_traces -- \
          pytest tests/strix_ai/vlm/test_clip.py::TestCLIP::test_clip_image_text_matching -v -s

# Profile all VLM tests
rocprofv3 --hip-trace --kernel-trace --memory-copy-trace \
          --output-format pftrace -d ./vlm_traces -- \
          pytest tests/strix_ai/vlm/ -v -s

# Profile ViT tests
rocprofv3 --hip-trace --kernel-trace --memory-copy-trace \
          --output-format pftrace -d ./vit_traces -- \
          pytest tests/strix_ai/vit/ -v -s

# Profile YOLO tests
rocprofv3 --hip-trace --kernel-trace --memory-copy-trace \
          --output-format pftrace -d ./yolo_traces -- \
          pytest tests/strix_ai/cv/test_yolo.py -v -s
```

**✅ Pros:**
- No code duplication
- Profile any test anytime
- Tests remain simple and focused
- Works with ALL existing tests

**❌ Cons:**
- Manual command (but can be scripted)
- Profiles entire pytest framework overhead

---

### **Approach 2: Add Profiling to Existing Test Files**

Add profiling directly in `vlm/test_clip.py`, `vit/test_vit_base.py`, etc.:

```python
# tests/strix_ai/vlm/test_clip.py

@pytest.mark.profile  # ← Add this marker
def test_clip_with_profiling(strix_device, enable_profiling):
    """CLIP test with optional profiling"""
    from transformers import CLIPModel, CLIPProcessor
    
    if enable_profiling:
        print("🔍 Profiling mode enabled")
        # Could add extra profiling logic here
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(strix_device)
    # ... rest of test ...
```

**Run without profiling:**
```bash
pytest tests/strix_ai/vlm/test_clip.py -v -s
```

**Run with profiling:**
```bash
rocprofv3 --hip-trace --kernel-trace --memory-copy-trace \
          --output-format pftrace -d ./traces -- \
          pytest tests/strix_ai/vlm/test_clip.py -v -s --profile
```

**✅ Pros:**
- Single test for validation AND profiling
- Optional profiling flag
- No code duplication

**❌ Cons:**
- Tests become slightly more complex
- Still need rocprofv3 wrapper command

---

### **Approach 3: Separate `profiling/` Folder** (Current)

Create dedicated profiling tests in `profiling/` folder:

```bash
pytest tests/strix_ai/profiling/test_strix_rocprofv3.py -v -s
```

**✅ Pros:**
- Clean separation of concerns
- Dedicated profiling workflows
- Can customize workload for profiling

**❌ Cons:**
- Duplicates model code
- Maintains separate test files
- More code to maintain

---

## 🎯 **Recommendation: Use Approach 1**

### **Directory Structure (Simplified)**

```
tests/strix_ai/
├── vlm/
│   └── test_clip.py         ← Profile this directly!
├── vit/
│   └── test_vit_base.py     ← Profile this directly!
├── cv/
│   └── test_yolo.py         ← Profile this directly!
└── conftest.py              ← Shared fixtures
```

**No `profiling/` folder needed!**

### **Profiling Commands**

```bash
# Profile CLIP
rocprofv3 --hip-trace --kernel-trace --memory-copy-trace \
          --output-format pftrace -d ./clip_traces -- \
          pytest tests/strix_ai/vlm/test_clip.py -v -s

# Profile ViT
rocprofv3 --hip-trace --kernel-trace --memory-copy-trace \
          --output-format pftrace -d ./vit_traces -- \
          pytest tests/strix_ai/vit/test_vit_base.py -v -s

# Profile ALL AI tests
rocprofv3 --hip-trace --kernel-trace --memory-copy-trace \
          --output-format pftrace -d ./all_traces -- \
          pytest tests/strix_ai/ -v -s -m "not slow"
```

---

## 📝 **Profiling Script (Optional)**

Create a helper script to make profiling easier:

```bash
# profile_strix_tests.sh
#!/bin/bash

TEST_PATH=$1
OUTPUT_DIR=${2:-./strix_traces}

echo "Profiling: $TEST_PATH"
echo "Output dir: $OUTPUT_DIR"

rocprofv3 \
  --hip-trace \
  --kernel-trace \
  --memory-copy-trace \
  --output-format pftrace \
  -d "$OUTPUT_DIR" \
  -- \
  pytest "$TEST_PATH" -v -s

echo "✅ Profiling complete. Traces in: $OUTPUT_DIR"
echo "   View at: https://ui.perfetto.dev"
```

**Usage:**
```bash
chmod +x profile_strix_tests.sh

# Profile CLIP
./profile_strix_tests.sh tests/strix_ai/vlm/test_clip.py ./clip_traces

# Profile specific test
./profile_strix_tests.sh tests/strix_ai/vlm/test_clip.py::TestCLIP::test_clip_performance ./clip_perf
```

---

## 🔄 **Comparison Matrix**

| Feature | Approach 1<br>(Direct) | Approach 2<br>(Markers) | Approach 3<br>(Separate) |
|---------|------------------------|-------------------------|--------------------------|
| **Code Duplication** | ✅ None | ✅ None | ❌ High |
| **Ease of Use** | ✅ Simple | ⚠️ Need flag | ⚠️ Separate tests |
| **Maintenance** | ✅ Easy | ✅ Easy | ❌ High |
| **Flexibility** | ✅ Profile anything | ✅ Per-test control | ⚠️ Fixed tests |
| **Separation** | ⚠️ Command-based | ✅ Marker-based | ✅ Folder-based |

---

## 💡 **When to Use Each Approach**

### **Use Approach 1 (Direct Profiling) when:**
- ✅ You want to profile existing tests quickly
- ✅ You don't want to modify test code
- ✅ You want flexibility to profile any test
- ✅ You're doing ad-hoc profiling

### **Use Approach 2 (Markers) when:**
- ⚠️ You want profiling mode as a test option
- ⚠️ You need per-test profiling control
- ⚠️ You want integrated profiling in CI

### **Use Approach 3 (Separate Folder) when:**
- ❌ You need specialized profiling workloads
- ❌ You want profiling tests in CI workflow
- ❌ You need simplified profiling scripts

---

## 🚀 **Recommended CI Integration**

### **Option A: Profile in Separate Job (Clean)**

```yaml
# .github/workflows/strix_ai_tests.yml

jobs:
  test:
    name: Strix AI Tests
    runs-on: linux-strix-halo-gpu-rocm
    steps:
      - name: Run Tests
        run: pytest tests/strix_ai/ -v -s

  profile:
    name: Strix Profiling
    runs-on: linux-strix-halo-gpu-rocm
    needs: test  # Only if tests pass
    steps:
      - name: Profile CLIP
        run: |
          rocprofv3 --hip-trace --kernel-trace --memory-copy-trace \
                    --output-format pftrace -d ./clip_traces -- \
                    pytest tests/strix_ai/vlm/test_clip.py -v -s
      
      - name: Upload Traces
        uses: actions/upload-artifact@v4
        with:
          name: profiling-traces
          path: ./clip_traces/
```

### **Option B: Profile on Demand (Workflow Dispatch)**

```yaml
workflow_dispatch:
  inputs:
    enable_profiling:
      description: 'Enable profiling'
      type: boolean
      default: false
    
    test_path:
      description: 'Test to profile'
      type: string
      default: 'tests/strix_ai/vlm/test_clip.py'

jobs:
  test_with_profiling:
    steps:
      - name: Run Test with Profiling
        if: ${{ inputs.enable_profiling }}
        run: |
          rocprofv3 --hip-trace --kernel-trace --memory-copy-trace \
                    --output-format pftrace -d ./traces -- \
                    pytest ${{ inputs.test_path }} -v -s
```

---

## 📊 **Example: Profile All Test Categories**

```bash
#!/bin/bash
# profile_all_categories.sh

CATEGORIES=("vlm" "vit" "cv" "vla" "optimization")

for category in "${CATEGORIES[@]}"; do
  echo "🔍 Profiling category: $category"
  
  rocprofv3 \
    --hip-trace --kernel-trace --memory-copy-trace \
    --output-format pftrace \
    -d "./traces_${category}" \
    -- \
    pytest "tests/strix_ai/${category}/" -v -s
  
  echo "✅ $category profiled → ./traces_${category}/"
done

echo "🎉 All categories profiled!"
echo "   View at: https://ui.perfetto.dev"
```

---

## ✅ **Final Recommendation**

### **For Strix Tests:**

1. **❌ Remove the `profiling/` folder** (or mark as optional)
2. **✅ Profile existing tests directly** using rocprofv3 wrapper
3. **✅ Keep tests simple** - one purpose: validate functionality
4. **✅ Add profiling script** for convenience

### **Commands You Need:**

```bash
# Profile any existing test
rocprofv3 --hip-trace --kernel-trace --memory-copy-trace \
          --output-format pftrace -d ./traces -- \
          pytest <TEST_PATH> -v -s

# Examples:
# CLIP:  pytest tests/strix_ai/vlm/test_clip.py -v -s
# ViT:   pytest tests/strix_ai/vit/test_vit_base.py -v -s
# YOLO:  pytest tests/strix_ai/cv/test_yolo.py -v -s
```

**That's it! No separate profiling tests needed.** 🎉

