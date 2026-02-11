[200~# Kpack Build and Packaging Analysis

## 1. Build and Artifact Overview

### Build Analyzed

**Multi-arch build (CI):** [Change hip-tests and aqlprofile-tests to TARGET_NEUTRAL. #153](https://github.com/ROCm/TheRock/actions/runs/21854651990)

- **Workflow:** Multi-Arch CI  
- **Run ID:** 21854651990  
- **Branch:** `multi_arch/integration-kpack`  
- **Status:** Failure (e.g. Stage - Math Libs gfx120X-all, CI Summary)  
- **Duration:** ~2h 2m 27s  

### Fetch Command Used

```bash
python3 ./build_tools/artifact_manager.py fetch \
  --stage all \
  --amdgpu-families gfx1151 \
  --run-id 21854651990 \
  --output-dir ../artifact_dir_multi_gfx1151
```

### BLAS Lib Generic Artifact Contents
The analysis was focussed on BLAS artifact.
<details>
<summary>Expand full listing</summary>

```
blas_lib_generic/
blas_lib_generic/math-libs
blas_lib_generic/math-libs/BLAS
blas_lib_generic/math-libs/BLAS/hipSPARSE
blas_lib_generic/math-libs/BLAS/hipSPARSE/stage
blas_lib_generic/math-libs/BLAS/hipSPARSE/stage/lib
blas_lib_generic/math-libs/BLAS/hipSPARSE/stage/lib/libhipsparse.so.4
blas_lib_generic/math-libs/BLAS/hipSPARSE/stage/lib/libhipsparse.so
blas_lib_generic/math-libs/BLAS/hipSPARSE/stage/lib/libhipsparse.so.4.3.0
blas_lib_generic/math-libs/BLAS/rocRoller
blas_lib_generic/math-libs/BLAS/rocRoller/stage
blas_lib_generic/math-libs/BLAS/rocRoller/stage/lib
blas_lib_generic/math-libs/BLAS/rocRoller/stage/lib/librocroller.so.1
blas_lib_generic/math-libs/BLAS/rocRoller/stage/lib/librocroller.so
blas_lib_generic/math-libs/BLAS/rocRoller/stage/lib/librocroller.so.1.0.0
blas_lib_generic/math-libs/BLAS/hipSPARSELt
blas_lib_generic/math-libs/BLAS/hipSPARSELt/stage
blas_lib_generic/math-libs/BLAS/hipSPARSELt/stage/lib
blas_lib_generic/math-libs/BLAS/hipSPARSELt/stage/lib/libhipsparselt.so.0
blas_lib_generic/math-libs/BLAS/hipSPARSELt/stage/lib/hipsparselt
blas_lib_generic/math-libs/BLAS/hipSPARSELt/stage/lib/hipsparselt/library
blas_lib_generic/math-libs/BLAS/hipSPARSELt/stage/lib/hipsparselt/library/TensileLibrary_BB_BB_A_Bias_SAV_SPB_Type_BB_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx942.co
blas_lib_generic/math-libs/BLAS/hipSPARSELt/stage/lib/hipsparselt/library/TensileLibrary_I8I8_I8I8_A_Bias_SAV_SPA_Type_I8I8_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx942.dat
... (additional TensileLibrary_*_gfx942 files)
blas_lib_generic/math-libs/BLAS/hipSPARSELt/stage/lib/.kpack
blas_lib_generic/math-libs/BLAS/hipSPARSELt/stage/lib/.kpack/blas_lib.kpm
blas_lib_generic/math-libs/BLAS/rocSOLVER
blas_lib_generic/math-libs/BLAS/rocSOLVER/stage/.kpack/blas_lib.kpm
blas_lib_generic/math-libs/BLAS/rocBLAS
blas_lib_generic/math-libs/BLAS/rocBLAS/stage/.kpack/blas_lib.kpm
blas_lib_generic/math-libs/BLAS/hipBLAS
blas_lib_generic/math-libs/BLAS/rocSPARSE
blas_lib_generic/math-libs/BLAS/rocSPARSE/stage/.kpack/blas_lib.kpm
blas_lib_generic/math-libs/BLAS/hipSOLVER
blas_lib_generic/math-libs/BLAS/hipBLASLt
blas_lib_generic/artifact_manifest.txt
```

</details>

### BLAS Lib gfx1151 Artifact Contents

<details>
<summary>Expand full listing</summary>

```
blas_lib_gfx1151/
blas_lib_gfx1151/math-libs
blas_lib_gfx1151/math-libs/BLAS
blas_lib_gfx1151/math-libs/BLAS/rocBLAS
blas_lib_gfx1151/math-libs/BLAS/rocBLAS/stage
blas_lib_gfx1151/math-libs/BLAS/rocBLAS/stage/lib
blas_lib_gfx1151/math-libs/BLAS/rocBLAS/stage/lib/rocblas
blas_lib_gfx1151/math-libs/BLAS/rocBLAS/stage/lib/rocblas/library
... (TensileLibrary_*_gfx1151.co/.dat/.hsaco)
blas_lib_gfx1151/math-libs/BLAS/rocBLAS/stage/.kpack
blas_lib_gfx1151/math-libs/BLAS/rocBLAS/stage/.kpack/blas_lib_gfx1151.kpack
blas_lib_gfx1151/math-libs/BLAS/hipBLASLt
blas_lib_gfx1151/math-libs/BLAS/hipBLASLt/stage/lib/hipblaslt/library
... (TensileLibrary_*_gfx1151.co/.dat)
blas_lib_gfx1151/artifact_manifest.txt
```

</details>

---

## 2. Issues and Analysis

### 2.1 Issues

#### Issue #1: Generic artifact contains architecture-specific files

The generic hipSPARSELt artifact contains several **gfx-specific** files. In particular, it includes **gfx942** `.hsaco` and `.dat` files instead of being architecture-neutral.

**Examples:**

- `blas_lib_generic/math-libs/BLAS/hipSPARSELt/stage/lib/hipsparselt/library/Kernels.so-000-gfx942.hsaco`
- `blas_lib_generic/math-libs/BLAS/hipSPARSELt/stage/lib/hipsparselt/library/TensileLibrary_lazy_gfx942.dat`

**Full listing:** See [Section 1 – BLAS Lib Generic Artifact Contents](#blas-lib-generic-artifact-contents).

---

#### Issue #2: KPM index files are incorrect

The `.kpm` (kpack manifest) index is wrong in two ways:

1. **Architecture reference:** The kpm file points to a **gfx942** kpack file, not a generic or multi-arch index.
2. **Prefix path:** The prefix in the kpm is the **artifact path**. The final packaging path should be `lib/.kpack/blas_lib.kpm`.

**File:** `blas_lib_generic/math-libs/BLAS/rocBLAS/stage/.kpack/blas_lib.kpm`

**Contents:**

```
Component: blas_lib
Format Version: 1
Prefix: math-libs/BLAS/rocBLAS/stage

Available Architectures: 1

Architectures:
  - gfx942       → blas_lib_gfx942.kpack (3.94 MB, 4 kernels)
```

---

#### Issue #3: All kpack files in an artifact share the same name

Every kpack file within a BLAS artifact uses the **same filename** (e.g. `blas_lib_gfx1151.kpack`) for all modules. This prevents distinguishing rocBLAS, hipBLASLt, hipSPARSELt, etc. by kpack file name.

**Example:**

- `./blas_lib_gfx1151/math-libs/BLAS/rocBLAS/stage/.kpack/blas_lib_gfx1151.kpack`
- Same name used for other BLAS components in the same artifact.

---

### 2.2 Analysis and Suggestions

**Analysis**

- **Issues [#1](#issue-1-generic-artifact-contains-architecture-specific-files) and [#2](#issue-2-kpm-index-files-are-incorrect):** The behavior indicates that **gfx942** artifacts are being selected as the default architecture and packaged as the **generic** artifact. 

- **Issue [#3](#issue-3-all-kpack-files-in-an-artifact-share-the-same-name):** As discussed offline, artifacts will need to be **split per module** so that each component (e.g. rocBLAS, hipBLASLt, hipSPARSELt) can have distinct kpack filenames and be packaged correctly.

**Suggestions**

1. **Single fetch for multi-arch:** With multi-arch, all artifacts for all GFX architectures should be fetchable in a single fetch command, instead of the previous per–GFX-family approach.

2. **KPM should reference all GFX architectures:** The generic kpm file should reference **all** target GFX architectures, not just one. It should list every architecture for which a kpack exists, with a prefix that matches the final install path (e.g. `lib` so that kpack files resolve under `lib/.kpack/`).

   **Expected KPM contents (multiple/all GFX reference):**

   ```
   Component: blas_lib
   Format Version: 1
   Prefix: lib

   Available Architectures: 5

   Architectures:
     - gfx942       → blas_lib_gfx942.kpack (3.94 MB, 4 kernels)
     - gfx1100      → blas_lib_gfx1100.kpack (3.89 MB, 4 kernels)
     - gfx1101      → blas_lib_gfx1101.kpack (3.91 MB, 4 kernels)
     - gfx1150      → blas_lib_gfx1150.kpack (3.92 MB, 4 kernels)
     - gfx1151      → blas_lib_gfx1151.kpack (3.93 MB, 4 kernels)
   ```

   (Exact arch list and sizes depend on the build.)

