# Activating Strix AI Workflows

## ⚠️ Issue: Workflows Not Triggering Automatically

If your workflows show as "skipped" or don't run on push, it's because **GitHub Actions workflows must be on the default branch** to trigger automatically on feature branches.

---

## 🎯 Solution Options

### **Option 1: Merge Workflows to Main (Recommended for Production)**

**Pros:**
- ✅ Workflows trigger automatically for everyone
- ✅ Works on all branches
- ✅ Standard GitHub Actions behavior

**Cons:**
- ⚠️ Requires PR review and merge
- ⚠️ Affects everyone (should be desired behavior)

**Steps:**

1. **Create Pull Request:**
   ```bash
   # View your changes
   git log origin/main..users/rponnuru/strix_poc --oneline
   
   # Create PR on GitHub
   # URL: https://github.com/ROCm/TheRock/pull/new/users/rponnuru/strix_poc
   ```

2. **After PR is merged**, workflows will trigger automatically on all branches

---

### **Option 2: Use Manual Triggers Only (Recommended for POC)**

**Pros:**
- ✅ Works immediately (no merge needed)
- ✅ Full control over when tests run
- ✅ Safe for POC/experimental work
- ✅ Can test before merging to main

**Cons:**
- ⚠️ Requires manual trigger each time
- ⚠️ Won't run automatically on push

**How to Use:**

```bash
# Trigger full test workflow
gh workflow run strix_ai_tests.yml \
  --ref users/rponnuru/strix_poc \
  -f platform=linux \
  -f strix_variant=gfx1151 \
  -f test_category=quick \
  -f test_type=quick

# Trigger quick test workflow
gh workflow run strix_ai_quick_test.yml \
  --ref users/rponnuru/strix_poc \
  -f test_command='tests/strix_ai/test_simple.py' \
  -f strix_platform='linux-gfx1151'
```

**Or via GitHub UI:**
1. Go to **Actions** tab
2. Select workflow (e.g., "Strix AI/ML Testing")
3. Click **"Run workflow"**
4. **IMPORTANT**: Select branch `users/rponnuru/strix_poc`
5. Fill in parameters
6. Click **"Run workflow"**

---

### **Option 3: Copy Workflows to Main Temporarily**

**Pros:**
- ✅ Quick activation
- ✅ Works immediately

**Cons:**
- ⚠️ Requires direct push to main (if you have permissions)
- ⚠️ Bypasses PR review

**Steps:**

```bash
# Checkout main
git checkout main
git pull

# Cherry-pick workflow commits
git cherry-pick b294e406  # Add workflows
git cherry-pick 59210689  # Fix YAML
git cherry-pick 29a9e599  # Add auto triggers
git cherry-pick aad26368  # Fix quick test

# Push to main
git push origin main

# Go back to your branch
git checkout users/rponnuru/strix_poc
```

**⚠️ Note:** This requires push access to main and should follow your team's policies.

---

### **Option 4: Create .github/workflows on Main with Workflow Call**

**Pros:**
- ✅ Workflows on main can call workflows on feature branches
- ✅ Clean separation
- ✅ Standard GitHub Actions pattern

**Cons:**
- ⚠️ More complex setup
- ⚠️ Requires workflow_call support

**Implementation:**

Create on **main** branch:

**File:** `.github/workflows/strix_ai_poc_trigger.yml` (on main)

```yaml
name: Strix AI POC Trigger

on:
  push:
    branches:
      - 'users/*/strix_*'
    paths:
      - 'tests/strix_ai/**'

jobs:
  trigger_poc_tests:
    runs-on: ubuntu-latest
    steps:
      - name: Trigger POC Tests
        uses: actions/github-script@v7
        with:
          script: |
            await github.rest.actions.createWorkflowDispatch({
              owner: context.repo.owner,
              repo: context.repo.repo,
              workflow_id: 'strix_ai_tests.yml',
              ref: context.ref,
              inputs: {
                platform: 'linux',
                strix_variant: 'gfx1151',
                test_category: 'quick',
                test_type: 'quick'
              }
            });
```

This workflow on main will trigger your POC workflows on the feature branch.

---

## 🔍 Verify Workflow Visibility

### **Check if Workflows Are Visible:**

1. Go to **GitHub → Actions** tab
2. Look for:
   - "Strix AI/ML Testing"
   - "Strix AI Quick Test"

**If you see them:**
- ✅ Click "Run workflow"
- ✅ Select branch: `users/rponnuru/strix_poc`
- ✅ Trigger manually

**If you DON'T see them:**
- ❌ Workflows are not on default branch
- ❌ Need to merge to main OR use Option 4

---

## 🎯 Recommended Approach for POC

**For your POC phase, use Option 2 (Manual Triggers):**

### **Quick Test Command:**

```bash
# Test that pytest works
gh workflow run strix_ai_quick_test.yml \
  --ref users/rponnuru/strix_poc \
  -f test_command='tests/strix_ai/test_simple.py' \
  -f strix_platform='linux-gfx1151'
```

### **Full Test Command:**

```bash
# Run quick smoke tests
gh workflow run strix_ai_tests.yml \
  --ref users/rponnuru/strix_poc \
  -f platform=linux \
  -f strix_variant=gfx1151 \
  -f test_category=quick \
  -f test_type=quick
```

### **Via GitHub UI:**

1. **Actions** → **"Strix AI/ML Testing"** → **"Run workflow"**
2. **IMPORTANT:** Change "Use workflow from" to **`users/rponnuru/strix_poc`**
3. Fill in parameters
4. Click **"Run workflow"**

---

## 📝 Why This Happens

**GitHub Actions Behavior:**
- Workflows must be on the **default branch** (main) to trigger automatically on other branches
- This is a security feature to prevent malicious workflows
- Feature branch workflows only trigger if:
  - The workflow exists on the default branch, OR
  - You trigger them manually with `--ref` flag

**Your Situation:**
- Workflows are on `users/rponnuru/strix_poc` ✅
- Not yet on `main` ❌
- **Therefore:** Can only use manual triggers until merged

---

## 🚀 Quick Actions

### **Test Now (Manual):**

```bash
# Run simple validation test
gh workflow run strix_ai_quick_test.yml \
  --ref users/rponnuru/strix_poc \
  -f test_command='tests/strix_ai/test_simple.py' \
  -f strix_platform='linux-gfx1151'
```

### **When Ready to Enable Auto-Triggers:**

```bash
# Create PR to merge workflows to main
# After merge, automatic triggers will work for all branches
```

---

## ✅ Summary

**Issue:** Workflows skipped because they're not on default branch (main)

**Quick Fix:** Use manual triggers with `--ref users/rponnuru/strix_poc`

**Long-term Fix:** Merge workflows to main via PR

**For POC:** Manual triggers are fine! You control when tests run.

---

## 🎬 Try This Now:

```bash
# This should work - manually trigger the quick test
gh workflow run strix_ai_quick_test.yml \
  --ref users/rponnuru/strix_poc \
  -f test_command='tests/strix_ai/test_simple.py' \
  -f strix_platform='linux-gfx1151'

# Check status
gh run list --workflow=strix_ai_quick_test.yml --limit 5
```

Or via UI:
1. **Actions** → **"Strix AI Quick Test"**
2. **"Run workflow"**
3. **Change branch to:** `users/rponnuru/strix_poc`
4. Click **"Run workflow"**

