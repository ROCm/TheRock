# Onboarding a New Component into TheRock  
*(For new component teams — use this as your checklist and reference)*

## Purpose  
This document describes the steps for bringing a new component into TheRock (ROCm build platform) from initial commit through CI integration, using the example of a previous component add-up.

## Pre-Requisites  
Before you start, ensure you have:  
- A publicly accessible upstream repository for your component (cloneable).  
- A reproducible build (preferably CMake) of your component.  
- A test suite (smoke and full/regression) that can be run non-interactively in CI.  
- Any patches needed for TheRock integration (packaged as `.patch` files).  
- Defined artifact outputs (what binaries or libraries will be produced).  
- A point of contact in your team for component maintenance.

## Step-by-Step Process  

### 1. Add the source into TheRock  
- Add your component under `math-libs/<component-name>` (or appropriate category) as a submodule or source directory.  
- Update `.gitmodules` accordingly if using submodules.  
- Commit with message: `Add <component-name> submodule`.

### 2. Create artifact descriptor  
- Create `artifact-<component-name>.toml` in the same category directory.  
- Define the artifacts your build will generate and how TheRock should fetch or publish them.

### 3. Integrate into CMake build graph  
- Update `CMakeLists.txt` in the category folder (e.g., `math-libs/CMakeLists.txt`) to add your component with `add_subdirectory()`.  
- If needed, update the top-level `CMakeLists.txt` to include your category changes.  
- Provide an `examples/CMakeLists.txt` if your component has sample builds.

### 4. Update fetch/checkout logic  
- Edit `build_tools/fetch_sources.py` (and/or related scripts) to include your component so TheRock can fetch the correct commit/tag upstream.  
- Specify correct version/branch and any special fetch instructions.

### 5. Add CI test script  
- Under `build_tools/github_actions/test_executable_scripts/`, add `test_<component-name>.py`.  
- The script should support:  
  - A “smoke” mode (quick > 5-10 mins)  
  - A “full” mode (longer, sharded)  
  - Timeout parameters from environment variables or CI config.  
- Ensure invocation is non-interactive and returns proper exit codes on failure.

### 6. Provide patches (if needed)  
- If integration requires modifications to upstream code, add `.patch` files under `patches/amd-mainline/<component-name>/`.  
- Document in your commit or README why each patch exists and whether upstreaming is planned.

### 7. Define GPU/target compatibility  
- If your component depends on specific AMD GPU targets/families, update `cmake/therock_amdgpu_targets.cmake` (or equivalent) to include your targets.  
- Test compilation against those targets.

### 8. Tune test strategy  
- For long running tests: enable sharding, set reasonable timeouts, and ensure CI jobs can run them without exceeding resource quotas.  
- Provide logic in test script (or CI config) to skip full tests unless explicitly requested.

### 9. Documentation & PR hygiene  
- Create or update a README in the component folder describing how to build/test the component in TheRock context.  
- Your PR should be broken into logical commits (e.g., add submodule, add artifact descriptor, add CMake changes, add CI script, add patches).  
- Provide a descriptive PR body covering what you added, testing done, known limitations / future work.

## Minimal Checklist  
- [ ] Upstream repo ready & tested locally  
- [ ] Source added to TheRock repo (submodule or directory)  
- [ ] `artifact-<component>.toml` created  
- [ ] CMake integration complete  
- [ ] fetch_sources logic updated  
- [ ] CI test script added  
- [ ] Patches included (if needed)  
- [ ] GPU/target mapping updated (if needed)  
- [ ] Documentation updated and PR prepared  

