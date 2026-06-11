# Coverage Workflow - Handoff Instructions

## Setup

```bash
# Clone repositories from reboss fork
git clone git@github.com:reboss/TheRock.git ~/git/TheRock
git clone git@github.com:reboss/rocm-libraries.git ~/git/rocm-libraries

# Create your own feature branches (DO NOT commit to enable-codecov or therock-codecov)
cd ~/git/TheRock
git checkout enable-codecov
git checkout -b your-name/coverage-implementation

cd ~/git/rocm-libraries
git checkout therock-codecov
git checkout -b your-name/coverage-implementation

# Push to your own fork (not reboss)
git remote set-url origin git@github.com:YOUR-USERNAME/TheRock.git
# (repeat for rocm-libraries)
```

## Context

**Read these files in order:**
1. `coverage_workflow_summary.md` - Previous work and roadblocks
2. `coverage_design_proposal.md` - Design validation
3. `coverage_implementation_plan.md` - Step-by-step implementation

**Related PRs:**
- rocm-libraries: https://github.com/ROCm/rocm-libraries/pull/4441
- TheRock: https://github.com/ROCm/TheRock/pull/3761 and #4931

## Current State

**Already implemented (TheRock):**
- Coverage passthrough flags in `cmake/therock_subproject.cmake`
- LLVM coverage tools enabled in `compiler/CMakeLists.txt`
- LLVM tool exports in `compiler/pre_hook_amd-llvm.cmake`

**Already implemented (rocm-libraries):**
- Coverage workflow skeleton at `.github/workflows/therock-ci-coverage.yml`
- Basic config script at `.github/scripts/therock_configure_coverage.py`

**Next steps (Phase 1 - hipDNN PoC):**
1. Create `test_categories_coverage.yaml` (section 1.1 in implementation plan)
2. Create `export_coverage_metadata.py` (section 1.2)
3. Create `coverage_runner.py` (section 1.3)
4. Modify `therock-ci-coverage.yml` (section 1.4)
5. Test locally then in CI (section 1.5)

## Working with Claude

Start new session with:
```
I'm continuing coverage workflow implementation. Please read:
- coverage_workflow_summary.md
- coverage_design_proposal.md
- coverage_implementation_plan.md

I'm ready for Phase 1 implementation. Help me create the YAML config and Python scripts.
```
