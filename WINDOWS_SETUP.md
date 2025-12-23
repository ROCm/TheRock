# Windows Setup Guide

## Quick Setup (Recommended)

Due to Windows Long Path limitations, we recommend installing without the Guardrails AI package:

### Step 1: Install Dependencies

```powershell
pip install -r requirements-minimal.txt
```

This installs everything you need **except** Guardrails AI, which has very long file paths that can cause issues on Windows.

### Step 2: Set Your API Key

```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

### Step 3: Run Your Analysis

```powershell
python performance_analysis.py "C:\Users\rponnuru\Downloads\SubTestCountsMatrixView.csv"
```

## What's the Difference?

| Feature | With Guardrails | Without Guardrails |
|---------|-----------------|-------------------|
| LangChain Framework | ✅ | ✅ |
| OpenAI Integration | ✅ | ✅ |
| Performance Analysis | ✅ | ✅ |
| Token Tracking | ✅ | ✅ |
| Reports Generation | ✅ | ✅ |
| Input Validation | ✅ Basic | ✅ Basic |
| Output Validation | ✅ Advanced | ⚠️ Basic |
| Toxic Language Check | ✅ | ❌ |
| Topic Restriction | ✅ | ❌ |

**Bottom Line:** The tool works perfectly without Guardrails! You just won't have the advanced output validation features.

## Alternative: Enable Windows Long Paths

If you really want Guardrails, you can enable Windows Long Paths support:

### Option A: Via Registry (Requires Admin)

1. Open Registry Editor (`Win + R`, type `regedit`)
2. Navigate to: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
3. Find or create: `LongPathsEnabled` (DWORD)
4. Set value to: `1`
5. Restart your computer

### Option B: Via Group Policy (Windows Pro/Enterprise)

1. Open Group Policy Editor (`Win + R`, type `gpedit.msc`)
2. Navigate to: Computer Configuration → Administrative Templates → System → Filesystem
3. Find: "Enable Win32 long paths"
4. Set to: "Enabled"
5. Restart your computer

### After Enabling Long Paths:

```powershell
pip install -r requirements.txt
```

## Troubleshooting

### "Module not found" Errors

```powershell
# Reinstall
pip install -r requirements-minimal.txt --force-reinstall
```

### API Key Issues

```powershell
# Check if set
echo $env:OPENAI_API_KEY

# Set it (replace with your actual key)
$env:OPENAI_API_KEY="sk-your-actual-key-here"
```

### Permission Errors

Run PowerShell as Administrator, or use:

```powershell
pip install --user -r requirements-minimal.txt
```

## Verify Installation

Test that everything is installed:

```powershell
python -c "import pandas, langchain, openai; print('✅ All dependencies installed!')"
```

## Ready to Run!

You're all set! See [QUICKSTART.md](QUICKSTART.md) for usage examples.

### Quick Test

```powershell
# Set API key
$env:OPENAI_API_KEY="your-key"

# Run analysis
python performance_analysis.py "C:\Users\rponnuru\Downloads\SubTestCountsMatrixView.csv"
```

## Cost Estimate

For your CSV file (~350 tests, 118 configs):
- Using `gpt-4o`: ~$0.10 - $0.30 per run
- Using `gpt-4o-mini`: ~$0.02 - $0.05 per run

Use the mini model for testing:

```powershell
python performance_analysis.py "data.csv" --model gpt-4o-mini
```

