# Strix AI Profiling Tests

This directory contains profiling tests that use **Option 1 approach**: running `rocprofv3` on existing test files without code duplication.

## 🎯 **Philosophy**

**No code duplication!** Instead of duplicating CLIP/ViT/YOLO code here, we:
1. Keep existing tests in `vlm/`, `vit/`, `cv/`, etc.
2. Profile them using `rocprofv3` wrapper from this folder
3. Centralize all profiling tests in one place

## 📁 **File Organization**

```
tests/strix_ai/
├── vlm/test_clip.py              ← Existing CLIP test
├── vit/test_vit_base.py          ← Existing ViT test  
├── cv/test_yolo.py               ← Existing YOLO test
└── profiling/
    ├── test_profile_existing_tests.py  ← Profiles all existing tests
    ├── test_strix_rocprofv3.py         ← Advanced rocprofv3 tests (optional)
    └── README.md                        ← This file
```

## 🚀 **How to Run**

### **Run All Profiling Tests**

```bash
# Run all profiling tests (profiles VLM, ViT, CV, VLA)
pytest tests/strix_ai/profiling/test_profile_existing_tests.py -v -s

# Quick smoke test only
pytest tests/strix_ai/profiling/test_profile_existing_tests.py::TestProfileQuick -v -s
```

### **Run Specific Category Profiling**

```bash
# Profile CLIP (VLM)
pytest tests/strix_ai/profiling/test_profile_existing_tests.py::TestProfileVLM::test_profile_clip -v -s

# Profile ViT
pytest tests/strix_ai/profiling/test_profile_existing_tests.py::TestProfileViT::test_profile_vit_inference -v -s

# Profile YOLO (CV)
pytest tests/strix_ai/profiling/test_profile_existing_tests.py::TestProfileCV::test_profile_yolo -v -s

# Profile VLA
pytest tests/strix_ai/profiling/test_profile_existing_tests.py::TestProfileVLA::test_profile_vla -v -s

# Profile ALL categories
pytest tests/strix_ai/profiling/test_profile_existing_tests.py::TestProfileAll::test_profile_all_categories -v -s
```

## 📊 **What Gets Profiled**

| Test Class | Profiles | Existing Test File |
|------------|----------|-------------------|
| `TestProfileVLM` | CLIP model | `tests/strix_ai/vlm/test_clip.py` |
| `TestProfileViT` | ViT model | `tests/strix_ai/vit/test_vit_base.py` |
| `TestProfileCV` | YOLO detection | `tests/strix_ai/cv/test_yolo.py` |
| `TestProfileVLA` | OWL-ViT action | `tests/strix_ai/vla/test_action_prediction.py` |
| `TestProfileAll` | All categories | All test directories |
| `TestProfileQuick` | Quick smoke | Quick marked tests |

## 🔧 **How It Works**

Each profiling test runs this command internally:

```bash
rocprofv3 \
  --hip-trace \
  --kernel-trace \
  --memory-copy-trace \
  --output-format pftrace \
  -d OUTPUT_DIR \
  -- \
  pytest EXISTING_TEST_PATH -v -s
```

**Example:**
```python
# test_profile_clip() runs:
rocprofv3 [...] -- pytest tests/strix_ai/vlm/test_clip.py::TestCLIP::test_clip_image_text_matching -v -s
```

## ✅ **Benefits**

| Benefit | Explanation |
|---------|-------------|
| ✅ **No Duplication** | Reuse existing test code |
| ✅ **Single Source of Truth** | Tests in `vlm/`, `vit/`, `cv/` are canonical |
| ✅ **Easy Maintenance** | Update test once, profiling gets it automatically |
| ✅ **Organized** | All profiling tests in one folder |
| ✅ **Flexible** | Easy to add/remove profiled tests |

## 📝 **Adding New Profiling Tests**

To profile a new test category:

```python
@pytest.mark.strix
@pytest.mark.profiling
@pytest.mark.p1
class TestProfileNewCategory:
    """Profile New Category tests"""
    
    def test_profile_new_test(self, cleanup_gpu):
        """Profile existing new test using rocprofv3"""
        if not check_rocprofv3_available():
            pytest.skip("rocprofv3 not available")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "new_traces"
            output_dir.mkdir()
            
            # Profile the existing test
            test_path = "tests/strix_ai/new_category/test_new.py::TestNew::test_function"
            
            result = run_rocprofv3_on_test(test_path, output_dir, timeout=300)
            
            # Check for trace files
            trace_files = list(output_dir.glob("*"))
            print(f"Generated {len(trace_files)} trace files")
```

## 🔍 **Output**

Each profiling test creates temporary directories with trace files:

```
/tmp/tmpXXXXXX/
├── clip_traces/
│   ├── trace_0.pftrace
│   └── ...
├── vit_traces/
│   ├── trace_0.pftrace
│   └── ...
└── yolo_traces/
    ├── trace_0.pftrace
    └── ...
```

**Viewing traces:**
1. Open https://ui.perfetto.dev
2. Upload the `.pftrace` file
3. Analyze timeline, kernels, memory transfers

## 🎯 **CI Integration**

Add to `.github/workflows/strix_ai_tests.yml`:

```yaml
- name: Run Strix Profiling Tests
  if: env.TEST_CATEGORY == 'all' || env.TEST_CATEGORY == 'profiling'
  run: |
    echo "=== Running Profiling Tests ==="
    python3 -m pytest tests/strix_ai/profiling/test_profile_existing_tests.py -v -s \
      --junit-xml=test-results-profiling.xml
```

## 📚 **Related Documentation**

- **Option 1 Guide**: `PROFILING_GUIDE.md` - Complete guide on profiling approaches
- **rocprofv3 Guide**: `README_ROCPROFV3.md` - rocprofv3 command reference
- **Migration Guide**: `MIGRATION_GUIDE.md` - Migration from legacy rocprof

## 🔧 **Prerequisites**

- ✅ rocprofv3 installed (`rocprofv3 --version`)
- ✅ ROCm 6.2+ with rocprofiler-sdk
- ✅ Strix GPU (gfx1150 or gfx1151)
- ✅ Existing tests in `vlm/`, `vit/`, `cv/`, `vla/`

## 💡 **Tips**

1. **Timeout**: Profiling adds overhead, increase timeout if needed
2. **Cleanup**: Temporary directories are auto-deleted after tests
3. **Markers**: Use `-m profiling` to run only profiling tests
4. **Quick**: Use `TestProfileQuick` for fast validation

## ❓ **FAQ**

### Q: Why not duplicate test code in profiling/?
**A:** Reduces maintenance burden. One test file = one source of truth.

### Q: Can I still profile tests manually?
**A:** Yes! Just run: `rocprofv3 [...] -- pytest test_path -v -s`

### Q: What if a test fails?
**A:** Profiling test passes if rocprofv3 runs. Individual test failures don't fail profiling.

### Q: How do I profile only one test?
**A:** Run specific test class, e.g., `TestProfileVLM::test_profile_clip`

## 🎉 **Example Run**

```bash
$ pytest tests/strix_ai/profiling/test_profile_existing_tests.py::TestProfileVLM -v -s

============================================================
Profiling VLM: CLIP Test
============================================================

🔍 Profiling command:
   rocprofv3 --hip-trace --kernel-trace --memory-copy-trace --output-format pftrace -d /tmp/tmp.../clip_traces -- python3 -m pytest tests/strix_ai/vlm/test_clip.py::TestCLIP::test_clip_image_text_matching -v -s

============================================================
Profiling Result: ✅ SUCCESS
============================================================

✅ Generated 3 profiling trace file(s):
   - trace_0.pftrace (1.2 MB)
   - trace_1.pftrace (856 KB)
   - metadata.json (4 KB)

📂 Traces saved to: /tmp/tmp.../clip_traces

✅ CLIP profiling completed
PASSED
```

---

**For questions or issues:**
- [Strix Testing Guide](../../../docs/development/STRIX_TESTING_GUIDE.md)
- [TheRock Issues](https://github.com/ROCm/TheRock/issues)
