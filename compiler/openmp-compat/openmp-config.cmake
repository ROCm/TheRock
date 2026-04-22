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

# Strip clang-only link options when the consumer is not clang. Upstream's
# config unconditionally appends -fno-openmp-implicit-rpath, which GCC rejects
# with an unrecognized-option error. rocprofiler-systems builds with GCC
# (enforced by its own CMake) and transitively links OpenMP::omp through
# Dyninst, so leaving this in place breaks that subproject.
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  get_target_property(_therock_openmp_link_opts OpenMP::omp INTERFACE_LINK_OPTIONS)
  if(_therock_openmp_link_opts)
    list(REMOVE_ITEM _therock_openmp_link_opts "-fno-openmp-implicit-rpath")
    set_target_properties(OpenMP::omp PROPERTIES INTERFACE_LINK_OPTIONS "${_therock_openmp_link_opts}")
  endif()
  unset(_therock_openmp_link_opts)
endif()

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
  # Guard legacy variables: defer to upstream if it ever starts defining them.
  # The authoritative compile/link options live on OpenMP::omp already; these
  # strings only matter for consumers that read the variables directly.
  if(NOT DEFINED OpenMP_${_therock_openmp_lang}_FLAGS)
    set(OpenMP_${_therock_openmp_lang}_FLAGS "-fopenmp")
  endif()
  if(NOT DEFINED OpenMP_${_therock_openmp_lang}_LIB_NAMES)
    set(OpenMP_${_therock_openmp_lang}_LIB_NAMES "omp")
  endif()
endforeach()
unset(_therock_openmp_lang)

set(OpenMP_FOUND TRUE)
