# hipcc: prefer install layout over stale `ROCM_PATH` on Windows

Copy-paste PR body: [hipcc-rocm-path-windows-pr-description.md](./hipcc-rocm-path-windows-pr-description.md).

## Patch-only CI experiment

This branch intentionally ships **llvm patch 0007 only** (no `rocm_sdk_core._cli` env clearing). CI must **rebuild `compiler-runtime`** so `hipcc.exe` includes the patch.

| Do | Don't |
|----|--------|
| PR / Multi-Arch CI with empty `prebuilt_stages` | `prebuilt_stages=compiler-runtime` or `all` + old `baseline_run_id` |
| Wait for **compiler-runtime** + **Build Python** + **Test Python** | Expect green Test Python from patch file merge alone |

If **Test Python** passes on `windows-gfx1151-gpu-rocm`, 0007 is sufficient without `_cli`. Re-add `_cli` before merge if prebuilt baselines may still ship old `hipcc`.

## Issue

Windows `rocm-sdk test` fails on `hipcc --help` (exit 1) while `amdclang`, `hipconfig`, etc. pass. The venv wheel is fine; `hipcc` invokes `clang` from the wrong place.

## Root cause

`constructRoccmPath()` still preferred a host **`ROCM_PATH`** (old tarball, AMD SDK, etc.) over the wheel tree. [#1201](https://github.com/ROCm/TheRock/pull/1201) / patch 0006 already fixed that for **`HIP_PATH`** only. Failures are often runner-specific (bad `ROCM_PATH` on one machine in the pool).

## Fix

Patch **0007** updates `constructRoccmPath()` like 0006:

```text
--rocm-path  →  lib/llvm/bin layout  →  ROCM_PATH env  →  HIP path
```

Applied by `build_tools/fetch_sources.py` when `compiler-runtime` is built from source.

## Build / CI

| Step | Workflow |
|------|----------|
| Fetch + patch llvm | `fetch_sources.py` in **compiler-runtime** / `build_windows_artifacts` |
| Wheel | **Build Python** |
| Test | **Test Python** → `testConsoleScripts` / `hipcc --help` |

## Local repro

```powershell
$env:ROCM_PATH = "C:\develop\therock-tarball\install\..."
$env:HIP_PATH = $env:ROCM_PATH
# Direct binary (bypasses _cli) — validates patch in hipcc.exe:
& ".\venv\Lib\site-packages\_rocm_sdk_core\bin\hipcc.exe" --help
venv\Scripts\hipcc.exe --help
python -m unittest rocm_sdk.tests.core_test.ROCmCoreTest.testConsoleScripts
```
