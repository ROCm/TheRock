# Migration Guide: rocprof → rocprofv3 for Strix

## ⚠️ **Important Change**

**For Strix GPUs (gfx1150/gfx1151), use rocprofv3 ONLY, not legacy rocprof.**

## 🔄 **What Changed**

### **Old Approach (Deprecated)**
```bash
# ❌ DON'T USE - Legacy rocprof (roctracer)
rocprof --stats --hip-trace -o results.csv -d output_dir python3 app.py
```

### **New Approach (Use This)**
```bash
# ✅ USE THIS - rocprofv3 for Strix
rocprofv3 --hip-trace --kernel-trace --memory-copy-trace \
          --output-format pftrace -d ./v3_traces -- python3 app.py
```

## 📦 **Test File Changes**

| Old Test File | Status | Use Instead |
|---------------|--------|-------------|
| `test_pytorch_profiling.py::test_rocprof_external_profile` | ❌ **DEPRECATED** | `test_strix_rocprofv3.py::test_rocprofv3_pytorch_inference` |
| `test_pytorch_profiling.py::test_rocprofv3_external_profile` | ⚠️ **MOVED** | `test_strix_rocprofv3.py::test_rocprofv3_pytorch_inference` |
| `test_ai_workload_profiling.py` (legacy tests) | ⚠️ **UPDATED** | Use `test_strix_rocprofv3.py` for CLIP profiling |

## 🎯 **Recommended Test File**

**Primary profiling tests:** `test_strix_rocprofv3.py`

```bash
# Run all Strix profiling tests
pytest tests/strix_ai/profiling/test_strix_rocprofv3.py -v -s

# Specific tests
pytest tests/strix_ai/profiling/test_strix_rocprofv3.py::TestStrixRocprofv3::test_rocprofv3_pytorch_inference -v -s
pytest tests/strix_ai/profiling/test_strix_rocprofv3.py::TestStrixRocprofv3::test_rocprofv3_clip_inference -v -s
pytest tests/strix_ai/profiling/test_strix_rocprofv3.py::TestStrixRocprofv3::test_rocprofv3_simple_kernel -v -s
```

## 🔧 **Why rocprofv3?**

| Feature | rocprof (legacy) | rocprofv3 (Strix) |
|---------|------------------|-------------------|
| **Tool** | roctracer | rocprofiler-sdk |
| **ROCm Version** | ROCm 5.x era | ROCm 6.2+ |
| **Output Format** | CSV | Perfetto traces (pftrace) |
| **Strix Support** | ⚠️ Legacy | ✅ **Optimized** |
| **Hardware Counters** | Limited | Enhanced |
| **Analysis** | Manual CSV parsing | Perfetto UI visualization |

## 📝 **Code Migration Examples**

### **Before (rocprof)**
```python
# Old test using rocprof
cmd = [
    "rocprof",
    "--stats",
    "--hip-trace",
    "-o", "results.csv",
    "-d", output_dir,
    "python3", script_path
]
```

### **After (rocprofv3)**
```python
# New test using rocprofv3
cmd = [
    "rocprofv3",
    "--hip-trace",
    "--kernel-trace",
    "--memory-copy-trace",
    "--output-format", "pftrace",
    "-d", output_dir,
    "--",  # Important separator!
    "python3", script_path
]
```

## ⚙️ **Installation**

```bash
# rocprofv3 is included in ROCm 6.2+
rocprofv3 --version

# Should output:
# ROCProfiler: v3.x.x

# If not found, update ROCm:
# pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ "rocm[libraries,devel]"
```

## 📊 **Analyzing Results**

### **Old (rocprof CSV)**
```bash
# Manual CSV parsing
cat results_stats.csv
cat results_hip_stats.csv
```

### **New (rocprofv3 Perfetto)**
```bash
# Visual analysis in Perfetto UI
# 1. Open https://ui.perfetto.dev
# 2. Upload the .pftrace file
# 3. Interactive timeline visualization!
```

## 🚀 **Quick Start**

```bash
# 1. Verify rocprofv3 is installed
rocprofv3 --version

# 2. Run Strix profiling tests
pytest tests/strix_ai/profiling/test_strix_rocprofv3.py -v -s

# 3. Profile your own script
rocprofv3 --hip-trace --kernel-trace --memory-copy-trace \
          --output-format pftrace -d ./my_traces -- python3 my_app.py

# 4. Analyze results
# Open https://ui.perfetto.dev and upload the .pftrace file
```

## 📚 **Documentation**

- **Primary Guide**: `README_ROCPROFV3.md`
- **Test Examples**: `test_strix_rocprofv3.py`
- **ROCm Docs**: https://rocm.docs.amd.com/projects/rocprofiler/en/latest/

## ❓ **FAQ**

### **Q: Can I still use rocprof on Strix?**
A: While rocprof may work, **rocprofv3 is recommended** for Strix. It provides better hardware counter support and modern trace formats.

### **Q: What about existing rocprof tests?**
A: They're marked as deprecated but not removed. New development should use `test_strix_rocprofv3.py`.

### **Q: Do I need to change my existing profiling scripts?**
A: For Strix profiling, yes - migrate to rocprofv3 for better results and support.

### **Q: What if rocprofv3 is not available?**
A: Install ROCm 6.2+ which includes rocprofiler-sdk. Check with: `rocprofv3 --version`

## 🎉 **Benefits of Migration**

✅ **Better Strix Support** - Optimized for RDNA 3.5 architecture  
✅ **Modern Trace Format** - Perfetto UI visualization  
✅ **Enhanced Metrics** - More detailed hardware counters  
✅ **Future-Proof** - Active development, new features  
✅ **Simpler Command** - Cleaner flag structure  

---

**For questions or issues, see:**
- [Strix Testing Guide](../../../docs/development/STRIX_TESTING_GUIDE.md)
- [TheRock Issues](https://github.com/ROCm/TheRock/issues)

