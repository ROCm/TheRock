# Strix AI Tests - Troubleshooting Guide

## 🔴 Common Issue: Torch Import Error on Windows

### **Error Message:**
```
ImportError: DLL load failed while importing _C: The specified module could not be found.
```

### **Root Cause:**
PyTorch installation failed due to **Windows Long Path limitations**. Windows has a 260-character path limit by default, and PyTorch has very long file paths that exceed this limit.

---

## ✅ **Solution: Enable Windows Long Path Support**

### **Method 1: Registry Editor (Recommended)**

1. **Press `Win + R`**, type `regedit`, press **Enter**

2. **Navigate to:**
   ```
   HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem
   ```

3. **Find or create `LongPathsEnabled`:**
   - Right-click in the right pane → **New** → **DWORD (32-bit) Value**
   - Name it: `LongPathsEnabled`
   - Double-click it and set **Value data** to `1`

4. **Click OK** and **close Registry Editor**

5. **Restart your computer** (required!)

6. **Reinstall PyTorch:**
   ```powershell
   python -m pip uninstall torch torchvision -y
   python -m pip cache purge
   python -m pip install torch torchvision --no-cache-dir
   ```

### **Method 2: Group Policy (Windows Pro/Enterprise)**

1. **Press `Win + R`**, type `gpedit.msc`, press **Enter**

2. **Navigate to:**
   ```
   Computer Configuration 
   → Administrative Templates 
   → System 
   → Filesystem
   ```

3. **Find:** "Enable Win32 long paths"

4. **Double-click** and set to **Enabled**

5. **Click OK** and **restart your computer**

6. **Reinstall PyTorch** (see step 6 above)

### **Method 3: PowerShell (Requires Admin)**

```powershell
# Run PowerShell as Administrator
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

# Restart required
Restart-Computer
```

After restart:
```powershell
python -m pip uninstall torch torchvision -y
python -m pip install torch torchvision
```

---

## 🧪 **Verify Installation**

After enabling long paths and reinstalling:

```powershell
# Test torch import
python -c "import torch; print(f'Torch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Expected output:**
```
Torch version: 2.9.1
CUDA available: True/False
```

---

## 🎯 **Running Tests After Fix**

### **1. Set Environment Variables**

```powershell
# Set ROCm paths
$env:THEROCK_BIN_DIR="C:\path\to\rocm\bin"
$env:AMDGPU_FAMILIES="gfx1151"  # or gfx1150
```

### **2. Run Tests**

```powershell
# Run all VLA tests
python -m pytest tests/strix_ai/vla/ -v

# Run specific VLA test
python -m pytest tests/strix_ai/vla/test_action_prediction.py::TestVisionLanguageAction::test_vla_visual_grounding -v -s

# Run all Strix AI tests
python -m pytest tests/strix_ai/ -v
```

---

## 📋 **Alternative: Test on Linux**

Since Strix hardware typically runs Linux, you may want to test there instead:

```bash
# On Linux with Strix hardware
export THEROCK_BIN_DIR=/opt/rocm/bin
export AMDGPU_FAMILIES=gfx1151

# Install dependencies
pip install pytest pytest-check transformers accelerate ultralytics opencv-python pillow torch torchvision timm einops

# Run tests
pytest tests/strix_ai/vla/ -v
```

---

## 🔍 **Debugging Steps**

### **Check if pytest works:**
```powershell
python -m pytest tests/strix_ai/test_simple.py -v
```
✅ Should pass - this verifies pytest is working

### **Check Python version:**
```powershell
python --version
```
✅ Should be Python 3.8 or higher

### **Check installed packages:**
```powershell
python -m pip list | findstr torch
python -m pip list | findstr transformers
```

### **Check PATH:**
```powershell
python -m site
```

---

## ⚠️ **Known Issues**

### **1. Windows Store Python**
If using Python from Windows Store, paths can be even longer.

**Solution:** Use standard Python installer from python.org

### **2. OneDrive/Sync Folders**
Installing in OneDrive or synced folders can cause issues.

**Solution:** Install Python in `C:\Python312` or similar local path

### **3. Antivirus Blocking**
Some antivirus software blocks long path operations.

**Solution:** Temporarily disable or add exception for Python directories

### **4. Virtual Environment Issues**
Virtual environments can have path issues.

**Solution:** Use `--system-site-packages` or install in base environment for testing

---

## 📦 **Full Dependency List**

If you want to install everything at once after fixing long paths:

```powershell
python -m pip install `
  pytest `
  pytest-check `
  transformers `
  accelerate `
  ultralytics `
  opencv-python `
  pillow `
  torch `
  torchvision `
  timm `
  einops `
  numpy `
  scipy `
  matplotlib
```

---

## 🆘 **Still Having Issues?**

### **Option 1: Use WSL2 (Windows Subsystem for Linux)**

```powershell
# Install WSL2
wsl --install

# Inside WSL2
pip install pytest transformers torch torchvision ultralytics opencv-python pillow
pytest tests/strix_ai/vla/ -v
```

### **Option 2: Use Docker**

```bash
# Build container with all dependencies
docker build -t strix-ai-tests .

# Run tests
docker run strix-ai-tests pytest tests/strix_ai/ -v
```

### **Option 3: Test on Actual Strix Hardware**

The tests are designed to run on Strix hardware which is typically Linux-based. Windows testing is for development only.

---

## 📝 **Quick Reference**

| Issue | Solution |
|-------|----------|
| `ImportError: DLL load failed` | Enable Windows Long Paths + reinstall torch |
| `pytest: command not found` | Use `python -m pytest` instead |
| `CUDA not available` | Normal on Windows dev machine, tests will skip |
| `transformers not found` | `python -m pip install transformers` |
| Tests skip with "not Strix" | Set `AMDGPU_FAMILIES=gfx1151` |

---

## ✅ **Success Checklist**

- [ ] Windows Long Paths enabled
- [ ] Computer restarted
- [ ] torch installed successfully
- [ ] `python -c "import torch"` works
- [ ] pytest runs without import errors
- [ ] Environment variables set (THEROCK_BIN_DIR, AMDGPU_FAMILIES)
- [ ] Tests run (may skip if not on Strix hardware)

---

## 📞 **Need Help?**

- Check the main README: `tests/strix_ai/README.md`
- Review test documentation: `docs/development/STRIX_AI_ML_TEST_PLAN.md`
- PyTorch Windows installation guide: https://pytorch.org/get-started/locally/
- Windows Long Paths info: https://pip.pypa.io/warnings/enable-long-paths

---

**Last Updated:** December 11, 2025  
**Issue:** Windows Long Path Support  
**Status:** Fixable with system configuration

