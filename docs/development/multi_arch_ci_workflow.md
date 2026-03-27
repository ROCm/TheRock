# `multi_arch_ci.yml` — workflow map

## Purpose

`Multi-Arch CI` is a **staging orchestrator** for the sharded multi-arch pipeline. It mirrors `ci.yml` but routes Linux through `multi_arch_ci_linux.yml` (instead of `ci_linux.yml`). Comments in the file note that once validated, `ci.yml` will call the same multi-arch sub-workflows directly.

**Reference:** `.github/workflows/multi_arch_ci.yml` (lines 4–9).

---

## Entry triggers (what starts this workflow)

| Trigger | When |
|--------|------|
| `push` | Branches `main` and `multi_arch/**` |
| `pull_request` | Types: `labeled`, `opened`, `synchronize` |
| `workflow_dispatch` | Manual run with optional inputs: GPU families, test label filters, `prebuilt_stages` + `baseline_run_id` for reusing artifacts from another run |

**Concurrency:** `group: ${{ github.workflow }}-${{ github.event.number || github.sha }}` with `cancel-in-progress: true` — newer runs for the same PR or commit supersede older ones.

---

## Downstream reusable workflows (“YAML triggers”)

`multi_arch_ci.yml` does **not** use `workflow_run` to fan out to other top-level workflows. **Downstream** here means **reusable workflows** invoked via `jobs.<id>.uses:`.

### Graph (high level)

```
multi_arch_ci.yml
├── setup                    → .github/workflows/setup.yml
├── linux_build_and_test     → .github/workflows/multi_arch_ci_linux.yml  (matrix: linux variants)
├── windows_build_and_test   → .github/workflows/multi_arch_ci_windows.yml (matrix: windows variants)
└── ci_summary               → (inline job) workflow_summary.py
```

### `multi_arch_ci_linux.yml` →

| Job / stage | Reusable workflow |
|-------------|-------------------|
| `copy_prebuilt_stages` (optional) | Inline job using `artifact_manager.py copy` |
| `build_multi_arch_stages` | `multi_arch_build_portable_linux.yml` |
| `validate_artifact_structure` | `test_artifacts_structure.yml` |
| `test_artifacts_per_family` | `test_artifacts.yml` → `test_component.yml` |
| `build_python_packages` | `build_portable_linux_python_packages.yml` |
| `test_python_packages_per_family` | `test_rocm_wheels.yml` |
| `build_pytorch_wheels_per_family` (if `build_pytorch`) | `build_portable_linux_pytorch_wheels_ci.yml` |

### `multi_arch_ci_windows.yml` →

Same pattern with Windows-specific build/python/pytorch workflows:

| Job / stage | Reusable workflow |
|-------------|-------------------|
| `copy_prebuilt_stages` | Inline + `artifact_manager.py copy` |
| `build_multi_arch_stages` | `multi_arch_build_windows.yml` |
| `validate_artifact_structure` | `test_artifacts_structure.yml` |
| `test_artifacts_per_family` | `test_artifacts.yml` → `test_component.yml` |
| `build_python_packages` | `build_windows_python_packages.yml` |
| `build_pytorch_wheels_per_family` (if `build_pytorch`) | `build_windows_pytorch_wheels_ci.yml` |

### `multi_arch_build_portable_linux.yml` →

Chains **stage jobs** that each call `multi_arch_build_portable_linux_artifacts.yml` with a `stage_name`. Documented stages include: **foundation** → **compiler-runtime** → parallel **math-libs** / **comm-libs** / **debug-tools** / **dctools-core** / **profiler-apps** / **iree-compiler** / **media-libs**, then **fusilli-libs** after math + IREE (see file header and `needs:` edges).

**Reference:** `.github/workflows/multi_arch_build_portable_linux.yml` (lines 1–16).

### `multi_arch_build_windows.yml` →

Fewer stages: **foundation** → **compiler-runtime** → **math-libs** only (other stages disabled on Windows per `BUILD_TOPOLOGY` / comments).

**Reference:** `.github/workflows/multi_arch_build_windows.yml` (lines 6–14).

---

## Phases and scripts

Below, **Source sync** = repo + submodules + CI selection + (optional) copying prior run artifacts. **Build** = compile, package, tests that exercise built bits. **Publish** = uploading artifacts/logs to shared storage and summarizing the run.

---

### 1. Source sync

| Location | Script / action | Role |
|----------|-----------------|------|
| `setup.yml` | `actions/checkout` (`fetch-depth: 2`) | Checkout for merge-base / diff usage |
| `setup.yml` | `gh pr view` (PR only) | Load PR labels into `PR_LABELS` |
| `setup.yml` | `build_tools/github_actions/configure_ci.py` | Decides **whether** build jobs run, **Linux/Windows matrix variants**, test label filters, `multi_arch` grouping |
| `setup.yml` | `build_tools/compute_rocm_package_version.py --release-type=dev` | **ROCm package version** string for wheels/packages |
| Optional: `copy_prebuilt_stages` | `build_tools/artifact_manager.py copy` | Copy **prebuilt stage** archives from run `baseline_run_id` into the **current** run’s artifact prefix (skip building those stages) |
| Per-stage jobs (`*_artifacts.yml`) | `build_tools/artifact_manager.py fetch ... --bootstrap` | Pull **prior stages’** tarballs from S3 into `BUILD_DIR` so the stage is incremental |
| Per-stage jobs | `build_tools/fetch_sources.py --stage ${STAGE_NAME} --jobs 12 --depth 1` | **Git submodule fetch** scoped to the stage (shallow) |
| `build_portable_linux_*` / `build_windows_python_packages.yml` | `artifact_manager.py fetch` or `fetch_artifacts.py` | Sync **built ROCm artifact archives** for wheel builds (not git sources) |
| `test_artifacts_structure.yml` | `fetch_artifacts.py --no-extract` | Download archives for structure checks |
| `test_artifacts.yml` (configure) | `fetch_test_configurations.py` | Derives **which components** and **shard matrix** to test from topology + labels |

---

### 2. Build

| Location | Script / action | Role |
|----------|-----------------|------|
| Stage jobs | `build_tools/setup_ccache.py` | Configure **ccache** for compiler reuse |
| Stage jobs | `build_tools/health_status.py` | **Runner health** check before heavy work |
| Stage jobs | `build_tools/configure_stage.py` + `cmake` | **Stage-specific CMake args** (`--gha-output` feeds next steps) |
| Stage jobs | `build_tools/memory_monitor.py` + `cmake --build ... stage-${STAGE_NAME} therock-artifacts` | **Compile** the stage under memory monitoring |
| `test_artifacts_structure.yml` | `pytest tests/test_artifact_structure.py` | Validates **layout** of fetched archives |
| `test_artifacts.yml` → `test_component.yml` | `health_status.py`, `print_driver_gpu_info.py`, `install_additional_requirements.py`, `memory_monitor.py`, optional `reproduce_test_failure.py` | **GPU/component tests** against staged ROCm |
| `build_*_python_packages.yml` | `build_python_packages.py` | Build **Python wheels** from artifact tree |
| `test_rocm_wheels.yml` | `setup_venv.py` + `rocm-sdk test` | **Smoke-test** installed wheels |
| `build_*_pytorch_wheels_ci.yml` | `python_to_cp_version.py`, `determine_version.py` | Map Python version / naming for **PyTorch wheel** CI builds |

---

### 3. Publish (artifacts, indexes, logs, summary)

| Location | Script / action | Role |
|----------|-----------------|------|
| Stage jobs | `aws-actions/configure-aws-credentials` (non-fork `ROCm/TheRock`) | **OIDC** to push to CI buckets |
| Stage jobs | `build_tools/artifact_manager.py push` | Upload **stage** `.tar.xz` (and related) to **S3** keyed by `github.run_id` |
| Stage jobs | `build_tools/github_actions/post_stage_upload.py` | Upload **stage logs** / metadata for the run |
| `build_*_python_packages.yml` | `build_tools/github_actions/upload_python_packages.py` | Publish **wheel trees** + **pip index** URLs (`package_find_links_url` output) |
| PyTorch CI workflows | (build + upload steps in those YAMLs) | Publish **PyTorch** wheels tied to the ROCm index slice |
| `multi_arch_ci.yml` → `ci_summary` | `build_tools/github_actions/workflow_summary.py --needs-json` | **Aggregate pass/fail** from `needs` for the final job view |

---

## Inputs that cross-cut phases

From `workflow_dispatch` on `multi_arch_ci.yml`: **GPU family strings**, **test label filters**, **`prebuilt_stages`** + **`baseline_run_id`** skip selected **build** stages by **syncing** artifacts instead of compiling them.
