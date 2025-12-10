# Python Package Update Guide

**Created:** 2025-12-09
**For:** TheRock gfx1031 ROCm Build

## Overview

This guide covers the safe update of Python packages for your TheRock build and AI/LLM tools using the automated `update_pip_packages.sh` script.

## 📋 What Will Be Updated

### Stage 1: TheRock Build Requirements ✅

- **setuptools:** 78.1.1 → 80.9.0 (Required by requirements.txt)
- **PyYAML:** 6.0.2 → 6.0.3
- **meson:** 1.9.1 → 1.10.0
- **pytest:** 8.4.2 → 9.0.2
- **pytest-cmake:** 1.1.0 → 1.2.0

### Stage 2: PyTorch with ROCm Support 🚀

**CRITICAL:** Replace CUDA PyTorch with ROCm version

**Current:**

```
PyTorch 2.9.1+cu128 (CUDA - NOT using your AMD GPU!)
```

**Options:**

- ROCm 6.2 (Recommended - Most Stable)
- ROCm 6.3 (Balanced)
- ROCm 6.4 (Latest - Cutting Edge)

### Stage 3: AI/LLM Tools 🤖

- **open-interpreter:** 0.3.14 → 0.4.3 (Major version)
- **anthropic:** 0.37.1 → 0.75.0 (⚠️ MAJOR jump - breaking changes)
- **openai:** 2.8.1 → 2.9.0
- **litellm:** 1.80.7 → 1.80.9
- **huggingface-hub:** 1.1.6 → 1.2.1
- **tiktoken:** 0.7.0 → 0.12.0

### Stage 4: ML Libraries 📊

- **numpy:** 2.3.4 → 2.3.5
- **google-generativeai:** 0.7.2 → 0.8.5
- **ipython:** 9.7.0 → 9.8.0
- **ipykernel:** 6.31.0 → 7.1.0
- **rich:** 13.9.4 → 14.2.0

### Stage 5: Infrastructure & Tools 🔧

- **boto3:** 1.40.55 → 1.42.6 (AWS SDK)
- **grpcio:** 1.67.1 → 1.76.0
- **pre_commit:** 4.3.0 → 4.5.0
- **psutil:** 5.9.8 → 7.1.3

## 🚀 Quick Start

### 1. Run the Update Script

```bash
cd /home/hashcat/TheRock
./update_pip_packages.sh
```

The script will:

- ✅ Automatically activate your virtual environment
- ✅ Create backups before each stage
- ✅ Test after critical updates
- ✅ Ask for confirmation at each stage
- ✅ Provide rollback options if issues occur
- ✅ Log everything to `pip_backups/update_<timestamp>.log`

### 2. Follow the Prompts

The script is **interactive** and will:

- Ask before each major update
- Let you choose which ROCm version to install
- Confirm before potentially breaking changes
- Offer rollback if tests fail

### 3. Expected Duration

- **Stage 1-3:** ~5-10 minutes
- **Stage 2 (PyTorch):** ~10-15 minutes (large download)
- **Full update:** ~20-30 minutes total

## 📝 Detailed Stage Breakdown

### Stage 1: Build Requirements (Safe ✅)

**Risk Level:** LOW
**Rollback Available:** YES
**Testing:** Automatic import tests

This updates the core build tools needed by TheRock. Very safe.

**What happens:**

1. Updates setuptools to meet requirements.txt minimum
1. Updates PyYAML for YAML parsing
1. Updates meson build system
1. Updates pytest test framework
1. **Tests:** Imports all packages to verify

**If it fails:**

- Script offers automatic rollback
- Choose to rollback or continue

______________________________________________________________________

### Stage 2: PyTorch ROCm (IMPORTANT 🚨)

**Risk Level:** MEDIUM
**Rollback Available:** YES
**Testing:** GPU detection and computation test

**CRITICAL:** This replaces your CUDA PyTorch with ROCm version!

**What happens:**

1. Asks which ROCm version you want (6.2, 6.3, or 6.4)
1. Downloads PyTorch with ROCm support (~2-3GB)
1. **Tests:**
   - Checks if ROCm is detected
   - Attempts simple GPU computation
   - Reports GPU name and device count

**Expected Results:**

✅ **Success:**

```
PyTorch version: 2.x.x+rocm6.x
ROCm available: True
GPU Device: AMD Radeon RX 6700 XT
✓ GPU computation test passed
```

⚠️ **Partial Success (Common for gfx1031):**

```
PyTorch version: 2.x.x+rocm6.x
ROCm available: False
⚠ ROCm not available - GPU not detected
```

**If GPU not detected:**

- This is somewhat expected for gfx1031 (unofficial support)
- You can still continue - ROCm is installed
- You may need to set `HSA_OVERRIDE_GFX_VERSION=10.3.0`
- Script will ask if you want to continue or rollback

**ROCm Version Recommendations:**

| Version | Stability  | Features | Recommended For              |
| ------- | ---------- | -------- | ---------------------------- |
| **6.2** | ⭐⭐⭐⭐⭐ | Stable   | Production, first-time users |
| **6.3** | ⭐⭐⭐⭐   | Balanced | General use                  |
| **6.4** | ⭐⭐⭐     | Latest   | Testing, newest features     |

______________________________________________________________________

### Stage 3: AI/LLM Tools (MAJOR CHANGES ⚠️)

**Risk Level:** MEDIUM-HIGH
**Rollback Available:** YES
**Testing:** Open Interpreter version check

**Major version bumps - breaking changes likely!**

**Critical Updates:**

1. **Anthropic API (0.37 → 0.75)**

   - Script asks for confirmation before updating
   - **Breaking changes highly likely**
   - Your Claude API code may need updates
   - Review: https://github.com/anthropics/anthropic-sdk-python/releases

1. **Open Interpreter (0.3 → 0.4)**

   - Major version change
   - New features and API changes
   - Your OI configs should still work
   - Review: https://github.com/OpenInterpreter/open-interpreter/releases

**What to test after:**

```bash
# Test Open Interpreter
interpreter --version
interpreter --help

# Test if profiles still work
source ~/.bashrc
oi  # Your alias to start OI
```

______________________________________________________________________

### Stage 4: ML Libraries (Safe ✅)

**Risk Level:** LOW
**Rollback Available:** YES
**Testing:** None (non-critical)

Updates numpy and supporting libraries. Generally safe.

______________________________________________________________________

### Stage 5: Infrastructure (Safe ✅)

**Risk Level:** LOW
**Rollback Available:** YES
**Testing:** None (non-critical)

Updates AWS tools, grpcio, and other infrastructure. Safe.

**Note:** Script **skips** protobuf 4→6 update (too risky for dependencies)

______________________________________________________________________

## 🔄 Rollback Instructions

### Automatic Rollback

If a stage fails tests, the script will ask:

```
Stage X tests failed
Rollback? (y/n)
```

Choose **y** for automatic rollback to pre-stage state.

### Manual Rollback

All backups are saved in `pip_backups/`:

```bash
# List available backups
ls -lht pip_backups/

# Rollback to specific backup
pip install -r pip_backups/packages_initial_full_YYYYMMDD_HHMMSS.txt --force-reinstall

# Rollback to before entire update
pip install -r pip_backups/packages_initial_full_*.txt --force-reinstall
```

### Find Your Backups

```bash
cd /home/hashcat/TheRock/pip_backups

# View latest backups
ls -lt | head -10

# View backup from specific date
ls -l | grep "20251209"
```

______________________________________________________________________

## 🧪 Post-Update Testing

### 1. Test TheRock Build Tools

```bash
cd /home/hashcat/TheRock
source .venv/bin/activate

# Test imports
python3 -c "import yaml, setuptools, mesonbuild, pytest; print('✓ All imports OK')"

# Check versions
pip show setuptools PyYAML meson pytest
```

### 2. Test PyTorch ROCm

```bash
python3 << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"ROCm available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    # Quick test
    x = torch.rand(10, 10).cuda()
    print(f"✓ GPU tensor created: {x.shape}")
else:
    print("⚠ GPU not available (may need HSA_OVERRIDE_GFX_VERSION)")
EOF
```

### 3. Test Open Interpreter

```bash
# Check version
interpreter --version

# Quick test (will ask for API key if not configured)
echo "print('test')" | interpreter -y

# Or use your configured profile
source ~/.bashrc
oi  # Should launch with your settings
```

### 4. Test Anthropic API (if updated)

```bash
python3 << 'EOF'
try:
    from anthropic import Anthropic
    print("✓ Anthropic imported successfully")
    print(f"Version: {Anthropic.__version__ if hasattr(Anthropic, '__version__') else 'Unknown'}")
except Exception as e:
    print(f"✗ Import failed: {e}")
EOF
```

______________________________________________________________________

## 🐛 Troubleshooting

### Issue: "Virtual environment not found"

**Solution:**

```bash
cd /home/hashcat/TheRock
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./update_pip_packages.sh
```

### Issue: PyTorch not detecting GPU

**Symptoms:**

```python
torch.cuda.is_available()  # Returns False
```

**Solutions:**

1. **Check ROCm installation:**

   ```bash
   /opt/rocm/bin/rocminfo | grep gfx
   ```

1. **Set override (if needed for gfx1031):**

   ```bash
   export HSA_OVERRIDE_GFX_VERSION=10.3.0
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

1. **Add to ~/.bashrc:**

   ```bash
   echo 'export HSA_OVERRIDE_GFX_VERSION=10.3.0' >> ~/.bashrc
   ```

1. **Verify PyTorch built for correct ROCm:**

   ```bash
   python3 -c "import torch; print(torch.version.hip)"
   # Should show: 6.2, 6.3, or 6.4
   ```

### Issue: Anthropic API errors after update

**Symptoms:**

```
AttributeError: 'Anthropic' object has no attribute 'X'
```

**Solution:**

1. Check what changed: https://github.com/anthropics/anthropic-sdk-python/releases
1. Update your code to new API
1. Or rollback:
   ```bash
   pip install anthropic==0.37.1
   ```

### Issue: Open Interpreter won't start

**Solutions:**

1. **Check if installed:**

   ```bash
   which interpreter
   pip show open-interpreter
   ```

1. **Reinstall:**

   ```bash
   pip install --force-reinstall open-interpreter==0.4.3
   ```

1. **Check config:**

   ```bash
   ls -la ~/.config/open-interpreter/
   cat ~/.config/open-interpreter/profiles/default.yaml
   ```

### Issue: Import errors after update

**Quick fix:**

```bash
# Reinstall from requirements.txt
cd /home/hashcat/TheRock
pip install -r requirements.txt --force-reinstall

# Or rollback completely
pip install -r pip_backups/packages_initial_full_*.txt --force-reinstall
```

______________________________________________________________________

## 📊 Logs and Backups

### Log File Location

Every update creates a timestamped log:

```
/home/hashcat/TheRock/pip_backups/update_YYYYMMDD_HHMMSS.log
```

### View Latest Log

```bash
cd /home/hashcat/TheRock/pip_backups
tail -100 update_*.log | less  # View last 100 lines
cat update_*.log | grep ERROR  # Find errors
```

### Backup Files

Each stage creates backups:

```
packages_initial_full_YYYYMMDD_HHMMSS.txt  # Before any changes
packages_stage1_before_YYYYMMDD_HHMMSS.txt  # Before stage 1
packages_stage2_before_YYYYMMDD_HHMMSS.txt  # Before stage 2
...
packages_final_YYYYMMDD_HHMMSS.txt  # After all updates
```

### Compare Before/After

```bash
cd pip_backups

# List packages that changed
diff packages_initial_full_*.txt packages_final_*.txt | grep "^[<>]"

# Count changes
diff packages_initial_full_*.txt packages_final_*.txt | grep "^[<>]" | wc -l
```

______________________________________________________________________

## 🎯 Recommended Workflow

### For First-Time Update:

```bash
# 1. Update TheRock from git first
cd /home/hashcat/TheRock
git fetch origin
git merge origin/main

# 2. Run the update script
./update_pip_packages.sh

# 3. Test everything
python3 -c "import torch; print(torch.cuda.is_available())"
interpreter --version

# 4. Rebuild TheRock if needed
cmake --build build
```

### For Conservative Update:

```bash
# Run script but skip risky stages
./update_pip_packages.sh

# When prompted:
# - Stage 1: YES (safe)
# - Stage 2: YES (important for GPU)
# - Stage 3: NO (skip Anthropic if you use it heavily)
# - Stage 4: YES (safe)
# - Stage 5: YES (safe)

# Update Anthropic manually later after reviewing changes
```

### For Aggressive Update (All at Once):

```bash
# Say YES to everything
./update_pip_packages.sh
# Choose ROCm 6.4 when prompted
# Confirm all stages
# Test thoroughly after
```

______________________________________________________________________

## 📚 Additional Resources

### TheRock Updates

- After updating packages, update TheRock:
  ```bash
  git merge origin/main
  pip install -r requirements.txt
  cmake --build build
  ```

### PyTorch ROCm Documentation

- Official: https://pytorch.org/get-started/locally/
- ROCm compatibility: https://rocm.docs.amd.com/

### Open Interpreter

- Releases: https://github.com/OpenInterpreter/open-interpreter/releases
- Docs: https://docs.openinterpreter.com/

### Anthropic Claude

- SDK: https://github.com/anthropics/anthropic-sdk-python
- API Docs: https://docs.anthropic.com/

______________________________________________________________________

## ✅ Success Criteria

After successful update, you should see:

```bash
# 1. Updated versions
pip list | grep -E "setuptools|PyYAML|open-interpreter|torch"
setuptools        80.9.0
PyYAML            6.0.3
open-interpreter  0.4.3
torch             2.x.x+rocm6.x

# 2. PyTorch with ROCm
python3 -c "import torch; print(torch.version.hip)"
# Output: 6.2 or 6.3 or 6.4

# 3. Open Interpreter works
interpreter --version
# Output: Open Interpreter 0.4.3

# 4. No import errors
python3 -c "import yaml, torch, anthropic; print('✓ OK')"
```

______________________________________________________________________

## 🆘 Emergency Rollback

If everything breaks:

```bash
cd /home/hashcat/TheRock/pip_backups

# Find your initial backup
ls -lt packages_initial_full_*.txt | head -1

# Full rollback
pip install -r packages_initial_full_YYYYMMDD_HHMMSS.txt --force-reinstall

# Verify
pip list | grep -E "torch|interpreter|anthropic"
```

______________________________________________________________________

## 📞 Getting Help

If you encounter issues:

1. **Check the log file:**

   ```bash
   cat pip_backups/update_*.log | grep -A5 ERROR
   ```

1. **Review this guide's Troubleshooting section**

1. **Check if backup exists:**

   ```bash
   ls -lh pip_backups/
   ```

1. **Test individually:**

   ```bash
   python3 -c "import torch"
   python3 -c "import anthropic"
   interpreter --version
   ```

1. **Create an issue with:**

   - Error messages from log
   - Output of `pip list`
   - Python version: `python3 --version`
   - OS: Fedora 43

______________________________________________________________________

**Last Updated:** 2025-12-09
**Script Version:** 1.0
**Tested On:** Fedora 43, Python 3.14, TheRock gfx1031 build
