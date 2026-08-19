# Upstream Synchronization & Rebase Guide

This guide explains how to keep your fork repository ([`analogbox/TheRock`](https://github.com/analogbox/TheRock)) up-to-date whenever AMD releases a new version of [ROCm/TheRock](https://github.com/ROCm/TheRock), while preserving all your custom patches (Ubuntu 26.04, GCC 15, AMD Strix Halo `gfx1151`, and `therock-env` orchestrator).

---

## 🧭 Repository Roles & Concepts

* **`upstream`**: Official AMD repository (`https://github.com/ROCm/TheRock.git`)
* **`origin`**: Your personal fork repository (`https://github.com/analogbox/TheRock.git`)
* **`Rebase`**: Taking your custom commits, temporarily setting them aside, updating the base code to AMD's newest release, and re-applying your commits cleanly on top.

```
Before Rebase:
  upstream (AMD) : --- [7.14.0] --- [7.15.0] --- [7.16.0] (New Release)
  origin (Fork)  : --- [7.14.0] --- [My Patches: Ubuntu 26.04 + GCC 15 + gfx1151]

After Rebase:
  origin (Fork)  : --- [7.14.0] --- [7.15.0] --- [7.16.0] --- [My Patches]
```

---

## 🚀 Scenario 1: Upgrading Your Active Branch to Latest Upstream

Use this when you want your active feature branch to incorporate all new AMD updates.

### Step 1: Fetch the Latest Upstream Commits and Tags

```bash
# Fetch all branches and release tags from AMD upstream
git fetch upstream --tags
```

### Step 2: Checkout Your Working Branch

```bash
git checkout feature/ubuntu-26.04-gcc15-gfx1151
```

### Step 3: Rebase Your Commits onto the New Upstream Version

```bash
# Option A: Rebase onto AMD's latest main branch
git rebase upstream/main

# Option B: Rebase onto a specific AMD release tag (e.g. 7.16.0)
git rebase upstream/tags/7.16.0
```

### Step 4: Resolving Conflicts (If Any Occur)

Because our modifications are clean and modular, conflicts are rare. If Git pauses with a conflict:

1. Check which files need attention:
   ```bash
   git status
   ```
2. Open the conflicting file, choose the correct lines (look for `<<<<<<<` and `>>>>>>>`), and save.
3. Mark the resolved file:
   ```bash
   git add <resolved-file-path>
   ```
4. Continue the rebase:
   ```bash
   git rebase --continue
   ```
*(Note: To completely cancel the rebase at any point, run `git rebase --abort`)*

### Step 5: Update Submodules

```bash
# Update git submodules to match the new upstream commit points
git submodule sync --recursive
git submodule update --init --recursive
```

### Step 6: Push Updated History to Your GitHub Fork

```bash
# Push the updated rebased branch to your personal fork
git push origin feature/ubuntu-26.04-gcc15-gfx1151 --force-with-lease
```
*(Always use `--force-with-lease` instead of `--force` for safe, protected pushing).*

---

## 📦 Scenario 2: Creating a New Version Branch (Preserving Old Versions)

Use this when you want to create a brand new branch for a new ROCm release (e.g., `therock-7.16`) while **keeping your 7.14 branch completely untouched**.

### Step 1: Fetch Upstream Release Tags

```bash
git fetch upstream --tags
```

### Step 2: Create a New Branch from the Upstream Release

```bash
# Create and checkout a new branch based on AMD's new release (e.g. tag 7.16.0)
git checkout -b feature/7.16-ubuntu-26.04 upstream/tags/7.16.0
```

### Step 3: Apply Your Custom Commits onto the New Branch

You can reapply your customizations from the previous branch using `cherry-pick`:

```bash
# Identify your custom commit hashes from the previous branch:
git log --oneline feature/ubuntu-26.04-gcc15-gfx1151 -n 10

# Apply the commits onto your new 7.16 branch:
git cherry-pick <commit-hash-1> <commit-hash-2> <commit-hash-3>
```

### Step 4: Verify and Build

```bash
# Run automated build on the new branch
./therock-env build --preset llm --python 3.14
```

### Step 5: Push the New Branch to Your Fork

```bash
git push -u origin feature/7.16-ubuntu-26.04
```

---

## ⚡ Quick Reference Cheat Sheet

| Task | Command |
| :--- | :--- |
| **Check configured remotes** | `git remote -v` |
| **Fetch newest AMD changes** | `git fetch upstream --tags` |
| **Rebase active branch onto upstream** | `git rebase upstream/main` |
| **Check rebase status** | `git status` |
| **Continue rebase after conflict fix** | `git rebase --continue` |
| **Cancel / abort rebase** | `git rebase --abort` |
| **Push rebased branch safely** | `git push origin <branch> --force-with-lease` |
| **View commit history tree** | `git log --graph --oneline -n 15` |

---

## 🔒 Safety Tips

1. **Never edit `upstream` directly**: Always make commits on your local feature branches and push to `origin`.
2. **Worktrees & Backups**: If experimenting with major upstream refactors, you can create a backup branch before rebasing:
   ```bash
   git branch backup/pre-rebase-$(date +%Y%m%d)
   ```
