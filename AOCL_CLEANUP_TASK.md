# AOCL-BLAS Integration Cleanup Task

## Overview

Remove temporary workaround code that copies AOCL files to rocBLAS's build dependency location, replacing it with standard CMake `find_package()` discovery.

## Background

### Current Implementation (Workaround)

**TheRock PR #3038** (`users/todavis/add-aocl-dependency`) integrates AOCL-BLAS 5.2 as a CPU reference BLAS library for rocBLAS testing. To make this work with rocBLAS's `develop` branch (which doesn't yet have CMake package discovery for AOCL), we implemented a temporary workaround:

- During AOCL's install phase, files are copied to rocBLAS's expected build location:
  - `build/math-libs/BLAS/rocBLAS/build/deps/aocl/install_package/lib/libaocl.a`
  - `build/math-libs/BLAS/rocBLAS/build/deps/aocl/install_package/include/`
- rocBLAS's `clients/CMakeLists.txt` already checks `${BUILD_DIR}/deps/aocl/` for AOCL 5.x
- TheRock sets `-DBUILD_DIR=${rocBLAS_build_dir}` so rocBLAS can find bundled AOCL

This works, but it's a workaround that couples AOCL's build logic to rocBLAS's internal directory structure.

### Future Implementation (Clean)

**rocm-libraries branch** `users/todavis/rocblas-aocl-cmake-package` adds proper CMake package discovery to rocBLAS:

- rocBLAS's `clients/CMakeLists.txt` will call `find_package(AOCL)` first
- Uses standard CMake search paths and package config files
- Includes robustness improvements (case-sensitivity, IMPORTED_LOCATION fallbacks)
- Falls back to existing `${BUILD_DIR}/deps/` checks if package not found

Once this merges, we can remove the copy workaround and rely on standard CMake discovery.

## Cleanup Tasks

### Prerequisites

1. ✅ **TheRock PR #3038 merged** - AOCL-BLAS integration with workaround
2. ⏳ **rocm-libraries PR merged** - `users/todavis/rocblas-aocl-cmake-package` (or equivalent)
3. ⏳ **TheRock submodule bump** - Update `rocm-libraries` submodule to include CMake package discovery

### Code Changes

**File:** `third-party/aocl/CMakeLists.txt`

**Remove this section:**
```cmake
# Copy AOCL to the location rocBLAS searches: ${BUILD_DIR}/deps/aocl/install_package/
# This runs during AOCL's install phase, ensuring files exist before rocBLAS configures.
# rocBLAS's clients/CMakeLists.txt searches this location first, allowing it to discover
# TheRock's AOCL without modifications to rocm-libraries.
if(NOT WIN32)
  install(CODE "
    # Construct path to rocBLAS build directory from AOCL's stage directory
    # stage is at: build/third-party/aocl/stage
    # rocBLAS is at: build/math-libs/BLAS/rocBLAS/build
    set(_rocblas_deps_dir \"\${CMAKE_INSTALL_PREFIX}/../../../math-libs/BLAS/rocBLAS/build/deps/aocl/install_package\")
    get_filename_component(_rocblas_deps_dir \"\${_rocblas_deps_dir}\" ABSOLUTE)

    # Create destination directories
    file(MAKE_DIRECTORY \"\${_rocblas_deps_dir}/lib\")
    file(MAKE_DIRECTORY \"\${_rocblas_deps_dir}/include\")

    # Copy library
    file(COPY \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/libaocl.a\"
         DESTINATION \"\${_rocblas_deps_dir}/lib\")
    message(STATUS \"Copied AOCL library for rocBLAS: \${_rocblas_deps_dir}/lib/libaocl.a\")

    # Copy headers (from aocl subdirectory to root, matching AOCL's expected structure)
    file(GLOB_RECURSE _aocl_headers \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/aocl/*.h\" \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/aocl/*.hh\")
    foreach(_header \${_aocl_headers})
      file(RELATIVE_PATH _rel_path \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/aocl\" \"\${_header}\")
      get_filename_component(_header_dir \"\${_rel_path}\" DIRECTORY)
      if(_header_dir)
        file(MAKE_DIRECTORY \"\${_rocblas_deps_dir}/include/\${_header_dir}\")
      endif()
      file(COPY_FILE \"\${_header}\" \"\${_rocblas_deps_dir}/include/\${_rel_path}\")
    endforeach()
    message(STATUS \"Copied AOCL headers for rocBLAS: \${_rocblas_deps_dir}/include/\")
  ")
endif()
```

**Replace with a comment:**
```cmake
# AOCL is discovered via standard CMake find_package(AOCL) mechanism.
# rocBLAS's clients/CMakeLists.txt will find our AOCLConfig.cmake package
# installed at lib/host-math/lib/cmake/AOCL/AOCLConfig.cmake
```

**File:** `math-libs/BLAS/CMakeLists.txt`

**Optional consideration:** The `-DBUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}/rocBLAS/build` setting can remain (it might be useful for other bundled dependencies in the future), or be removed if not needed. Document this decision.

### Testing

After cleanup, verify the following:

1. **Fresh build works:**
   ```bash
   cmake -B build -GNinja -DTHEROCK_AMDGPU_FAMILIES=gfx90a -DTHEROCK_BUILD_TESTING=ON
   ninja -C build rocBLAS
   ```

2. **rocBLAS finds AOCL via find_package:**
   ```bash
   grep "Found AOCL\|find_package(AOCL)" build/logs/rocBLAS_configure.log
   ```
   Should show: `find_package(AOCL)` succeeding, not the old deps/ path message

3. **rocblas-bench links AOCL:**
   ```bash
   nm build/math-libs/BLAS/rocBLAS/build/clients/staging/rocblas-bench | grep -i blis
   ```
   Should show BLIS symbols

4. **Execution test:**
   ```bash
   build/math-libs/BLAS/rocBLAS/build/clients/staging/rocblas-bench -f gemm -m 128 -n 128 -k 128 -r f32_r
   ```
   Should run successfully

5. **CI passes:** Ensure Linux x86_64 CI builds complete successfully

## Expected Outcome

- **Cleaner integration:** Uses standard CMake `find_package()` mechanism
- **Less coupling:** AOCL build logic no longer knows about rocBLAS's internal directory structure
- **Maintainable:** Standard CMake patterns are easier to understand and maintain
- **Backwards compatible:** Existing AOCL installations work the same way from rocBLAS's perspective

## Related Links

- **TheRock PR:** https://github.com/ROCm/TheRock/pull/3038
- **TheRock branch:** `users/todavis/add-aocl-dependency`
- **rocm-libraries branch:** `users/todavis/rocblas-aocl-cmake-package`
- **rocm-libraries repo:** https://github.com/ROCm/rocm-libraries

## Notes

- This is a **refactoring task** - no functional changes expected
- Can be coordinated with a rocm-libraries submodule bump PR in TheRock
- Consider documenting the standard CMake discovery pattern in developer docs as a best practice
- The cleanup makes AOCL integration consistent with other TheRock dependencies that use CMake packages
