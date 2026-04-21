# TheRock OpenMP compatibility shim.
#
# Staged alongside amd-llvm's CONFIG export so that TheRock's dependency
# provider, which routes default-mode find_package() to CONFIG-style resolution,
# also satisfies consumers written against CMake's FindOpenMP module.
#
# LLVM's openmp-config.cmake defines only OpenMP::omp. Subprojects that call
# plain find_package(OpenMP) (for example dyninst inside rocprofiler-systems)
# expect the Module-mode target names OpenMP::OpenMP_C, OpenMP::OpenMP_CXX,
# and OpenMP::OpenMP_Fortran. This shim loads the upstream config and adds
# those targets as interface libraries that link to OpenMP::omp, which already
# carries -fopenmp compile/link options.

include("${CMAKE_CURRENT_LIST_DIR}/../openmp/openmp-config.cmake")

# Clear INTERFACE_INCLUDE_DIRECTORIES on OpenMP::omp. Upstream's config exports
# the full clang resource dir (lib/clang/N/include) as an interface include,
# which propagates to every OpenMP consumer and collides when:
#   - A different active compiler's resource dir is already in scope (the
#     exported path becomes a second -isystem pointing at non-matching headers).
#   - libstdc++15's <cstdint> gets resolved through the wrong resource dir's
#     stdint.h and fails with "no member named 'intmax_t' in the global
#     namespace". Consumer builds die mid-compile.
# The -fopenmp compile option (still exported) is enough for the active clang
# to locate omp.h via its own resource dir, so dropping this export is safe.
set_target_properties(OpenMP::omp PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")

foreach(_therock_openmp_lang IN ITEMS C CXX Fortran)
  if(NOT TARGET OpenMP::OpenMP_${_therock_openmp_lang})
    # GLOBAL makes the imported target visible outside the directory scope
    # where find_package(OpenMP) was called. Needed for consumers that add
    # subdirectories referencing OpenMP::OpenMP_* before the subproject's own
    # find_package() runs, or that propagate the target through an exported
    # config file.
    add_library(OpenMP::OpenMP_${_therock_openmp_lang} INTERFACE IMPORTED GLOBAL)
    set_target_properties(OpenMP::OpenMP_${_therock_openmp_lang} PROPERTIES
      INTERFACE_LINK_LIBRARIES OpenMP::omp
    )
  endif()
  set(OpenMP_${_therock_openmp_lang}_FOUND TRUE)
  set(OpenMP_${_therock_openmp_lang}_FLAGS "-fopenmp")
  set(OpenMP_${_therock_openmp_lang}_LIB_NAMES "omp")
endforeach()
unset(_therock_openmp_lang)

set(OpenMP_FOUND TRUE)
