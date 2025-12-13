# GitHub Actions Logging Demo

This demo shows how to use GitHub Actions-specific logging methods in TheRock.

## Files

| File | Purpose |
|------|---------|
| `sample_github_actions_logging.py` | Python script using github_info, github_warning, github_error, github_group |
| `.github/workflows/demo_github_actions_logging.yml` | GitHub Actions workflow that runs the demo |
| `GITHUB_ACTIONS_LOGGING_DEMO.md` | This file - instructions |

---

## What Does This Demo Show?

The demo script simulates a CI/CD pipeline with:

1. **Environment Checks** - Uses `github_group()` and `github_warning()`
2. **Component Build** - Uses `github_info()` and `github_error()`
3. **Test Suites** - Uses all annotation types
4. **Report Generation** - Uses `github_group()` and `github_info()`

---

## How to Run the Demo

### Option 1: In GitHub Actions (Recommended)

1. **Go to GitHub Actions Tab:**
   ```
   https://github.com/ROCm/TheRock/actions/workflows/demo_github_actions_logging.yml
   ```

2. **Click "Run workflow" button**
   - Select your branch: `users/rponnuru/logging_poc`
   - Click "Run workflow"

3. **Wait for workflow to complete** (~30 seconds)

4. **Check the Results:**

   **A. Annotations Tab:**
   - Click on the workflow run
   - Look for "Annotations" section
   - You'll see:
     - 🔵 Blue info badges: `✅ Python 3.12 detected`, `✅ Component 'core' built successfully`
     - 🟡 Yellow warnings: `Low disk space: 15GB available`
     - 🔴 Red errors: `Compilation error in runtime`

   **B. Job Logs:**
   - Expand the "Run GitHub Actions Logging Demo" step
   - You'll see collapsible groups:
     - 🔍 Environment Checks
     - 🔨 Building 3 Components
     - 🧪 Running Test Suites
     - 📊 Generating Reports
   - Click each to expand/collapse

   **C. Files Changed Tab (if on PR):**
   - Warning/error annotations appear next to code
   - Linked to specific file:line numbers

---

### Option 2: Run Locally (Testing)

```bash
cd build_tools/_therock_utils
python sample_github_actions_logging.py
```

**Note:** When run locally:
- GitHub Actions annotations are NOT created (no badges)
- Falls back to normal logging
- Still shows the workflow logic
- Useful for testing before pushing

---

## What Each Method Does

### 1. `github_info()` - Success Notifications

**Code:**
```python
logger.github_info("✅ Component 'core' built successfully")
```

**In GitHub Actions:**
- Creates blue "notice" badge in Annotations
- Visible without expanding logs
- Shows in workflow summary

**Local Behavior:**
```
INFO - ✅ Component 'core' built successfully
```

---

### 2. `github_warning()` - Warning Annotations

**Code:**
```python
logger.github_warning(
    "Low disk space: 15GB available (recommended: 20GB)",
    file="config/requirements.txt",
    line=5
)
```

**In GitHub Actions:**
- Creates yellow warning badge
- Links to `config/requirements.txt` line 5
- Appears in Files Changed tab with annotation marker
- Visible in Annotations tab

**Local Behavior:**
```
WARNING - Low disk space: 15GB available (recommended: 20GB)
```

---

### 3. `github_error()` - Error Annotations

**Code:**
```python
logger.github_error(
    "Compilation error in runtime: undefined reference to 'hipMalloc'",
    file="src/runtime/device_memory.cpp",
    line=142
)
```

**In GitHub Actions:**
- Creates red error badge
- Links to `src/runtime/device_memory.cpp` line 142
- Highly visible in workflow summary
- Can block PR merge if configured
- Appears in Files Changed tab

**Local Behavior:**
```
ERROR - Compilation error in runtime: undefined reference to 'hipMalloc'
```

---

### 4. `github_group()` - Collapsible Sections

**Code:**
```python
with logger.github_group("🔨 Building 3 Components"):
    build_component("core")
    build_component("runtime")
    build_component("tools")
```

**In GitHub Actions:**
- Creates collapsible section in logs
- Collapsed by default (keeps logs clean)
- Click to expand and see details
- Can be nested for hierarchy

**Local Behavior:**
- Normal logging (no visual grouping)
- All logs appear normally

---

## Example Output

### In GitHub Actions Logs:
```
🔍 Environment Checks                                   [Click to expand ▶]
🔨 Building 3 Components                                [Click to expand ▶]
🧪 Running Test Suites                                  [Click to expand ▶]
📊 Generating Reports                                   [Click to expand ▶]
```

### In Annotations Tab:
```
ℹ️ ✅ Python 3.12 detected
ℹ️ ✅ Component 'core' built successfully
⚠️ Low disk space: 15GB available (recommended: 20GB)
    config/requirements.txt:5
❌ Compilation error in runtime: undefined reference to 'hipMalloc'
    src/runtime/device_memory.cpp:142
❌ integration_tests: 2/20 tests failed
    tests/integration_tests/test_runner.py:156
```

---

## When to Use in Real Projects

### ✅ Use GitHub Actions Methods In:

1. **CI/CD Build Scripts**
   ```python
   def build_rocm():
       logger.github_group("Building ROCm Components")
       # ... build code
       logger.github_info("Build completed successfully")
   ```

2. **Test Runners**
   ```python
   def run_tests():
       results = run_test_suite()
       if results.failed > 0:
           logger.github_error(f"{results.failed} tests failed")
   ```

3. **Deployment Scripts**
   ```python
   def deploy():
       with logger.github_group("Deployment Steps"):
           deploy_to_staging()
           logger.github_info("Deployed to staging")
   ```

4. **Validation Scripts**
   ```python
   def validate_pr():
       issues = check_code_quality()
       for issue in issues:
           logger.github_warning(issue.message, file=issue.file, line=issue.line)
   ```

---

### ❌ Don't Use GitHub Actions Methods In:

1. **Sample/Demo Applications** - Use regular `info()`, `warning()`, `error()`
2. **Library Code** - Code used outside GitHub Actions
3. **Interactive Scripts** - Scripts run by users locally
4. **General Applications** - Apps not related to CI/CD

---

## Customizing for Your Workflow

### Add to Your Own Workflow:

1. **Copy the pattern from demo:**
   ```python
   from _therock_utils.logging_config import get_logger
   
   logger = get_logger(__name__)
   
   with logger.github_group("My Operation"):
       # Your code here
       logger.github_info("Operation successful")
   ```

2. **Add to your YAML workflow:**
   ```yaml
   - name: Run My Script
     run: python my_script.py
   ```

3. **That's it!** The script will:
   - Create annotations in GitHub Actions
   - Fall back to normal logging locally

---

## Troubleshooting

### Q: I don't see annotations in GitHub Actions UI

**A:** Check:
1. Workflow completed successfully?
2. Look in "Annotations" tab (not job logs)
3. Scroll down in workflow summary page

### Q: Annotations show but file links don't work

**A:** Make sure:
1. File paths are relative to repo root
2. Files actually exist in the repo
3. Line numbers are valid

### Q: How do I test without pushing to GitHub?

**A:** Run locally:
```bash
python sample_github_actions_logging.py
```
Output will be normal logs, but logic is the same.

---

## Learn More

- **Presentation Guide:** `LOGGING_PRESENTATION_GUIDE.md` - Detailed explanation of all methods
- **README:** `LOGGING_README.md` - Complete framework documentation
- **Samples:** `sample_package_installer.py`, `sample_build_system.py` - Core logging examples

---

## Quick Reference

```python
# Import
from _therock_utils.logging_config import get_logger

logger = get_logger(__name__)

# GitHub Actions Methods
logger.github_info("Success message")
logger.github_warning("Warning", file="path/to/file.py", line=42)
logger.github_error("Error", file="path/to/file.py", line=100)

with logger.github_group("Group Title"):
    # Your code here
    pass

# Regular Methods (always work)
logger.info("Message")
logger.warning("Warning")
logger.error("Error")
logger.log_exception(e, "Exception occurred")
```

---

**Ready to try it?**

1. Go to: https://github.com/ROCm/TheRock/actions/workflows/demo_github_actions_logging.yml
2. Click "Run workflow"
3. Check the Annotations tab!

🚀

