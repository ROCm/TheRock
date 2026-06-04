# PR: Fix Windows `hipcc --help` with stale `ROCM_PATH` / `HIP_PATH`

Use this file as the GitHub pull request description.

## Summary

Fixes Windows `rocm-sdk test` failures on `hipcc --help` when self-hosted runners have stale `ROCM_PATH` / `HIP_PATH` pointing at an old system install instead of the venv wheel.

- **llvm patch 0007** — lower `ROCM_PATH` priority in `constructRoccmPath()` (same pattern as [#1201](https://github.com/ROCm/TheRock/pull/1201) / 0006 for `HIP_PATH`).

**This branch (patch-only experiment):** `_cli` env clearing is reverted so CI proves 0007 in rebuilt `hipcc.exe`. Do not use `prebuilt_stages=compiler-runtime`. Re-add `_cli` before merge if baselines may still ship old `hipcc`.

## Problem

On Windows GPU runners, `testConsoleScripts` runs `hipcc --help` and expects clang help text. The test fails with exit code 1 when:

- `ROCM_PATH` (and often `HIP_PATH`) point at an old tarball or HIP SDK tree on the machine, and
- `hipcc` resolves `clang` from that path instead of `_rocm_sdk_core/lib/llvm/bin` in the installed wheel.

`amdclang` and other tools that call into the wheel layout directly still pass; `hipcc` reads global env and was still preferring **`ROCM_PATH`** over the install tree next to `hipcc` (see [llvm-project #289](https://github.com/ROCm/llvm-project/pull/289) — review noted `rocm_path` is what finds clang).

This is often runner-specific (stale env on one machine in the pool), so rerunning tests without a fix is unreliable.

## Solution

### 1. llvm patch 0007 (`constructRoccmPath`)

Align ROCm path resolution with 0006 / documented intent:

```text
--rocm-path  →  ../lib/llvm/bin next to hipcc  →  ROCM_PATH env  →  HIP path
```

Applied via `fetch_sources.py` (`patches/amd-mainline/llvm-project/*.patch`) when `compiler-runtime` is built from source.

### 2. `rocm_sdk_core._cli` (console scripts)

`rocm-sdk test` invokes `Scripts\hipcc.exe`, which goes through `_cli` before `bin/hipcc.exe`. When `_rocm_sdk_core/lib/llvm/bin` exists, drop **`HIP_PATH`** and **`ROCM_PATH`** from the environment before spawn.

This is needed because CI often packages a **prebuilt** `hipcc.exe` (no fresh 0006/0007 in that binary yet). Clearing only `ROCM_PATH` is not enough when `HIP_PATH` still points at the stale install.

The layout check uses **`_get_core_module_path()`** so devel-expanded installs still key off the core tree where clang lives.

## Testing

- [ ] Windows `rocm-sdk test` / `testConsoleScripts` (`hipcc --help`) on `windows-gfx1151-gpu-rocm` (or your target runner)
- [ ] Full pipeline: **Build Python** runs on this branch so wheels include the updated `_cli`
- [ ] Local (optional): with stale `ROCM_PATH`/`HIP_PATH` set, `python -m unittest rocm_sdk.tests.core_test.ROCmCoreTest.testConsoleScripts` passes

**Note:** Merging the patch alone does not fix Test Python until new wheels are built. If `compiler-runtime` is satisfied via **prebuilt** artifacts, `_cli` is what unblocks the test until baseline artifacts include 0007.

## CI / artifacts

| Change | Affects |
|--------|---------|
| 0007 | `hipcc.exe` after a **compiler-runtime** rebuild (not prebuilt baseline) |
| `_cli` | `rocm-sdk-core` wheel after **Build Python** on this ref |

After merge, refresh Windows compiler baseline / avoid `prebuilt_stages` including `compiler-runtime` for PRs that only change llvm patches, so 0007 lands in published `hipcc` binaries. `_cli` can be removed later once prebuilts are rebuilt with 0006+0007.

## Related

- TheRock [#1201](https://github.com/ROCm/TheRock/pull/1201) — `HIP_PATH` / `constructHipPath` (0006)
- llvm-project [#289](https://github.com/ROCm/llvm-project/pull/289) — upstream `HIP_PATH` discussion; review requested the same for **`rocm_path`**
