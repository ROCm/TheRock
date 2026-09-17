# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# The pinned rocm-systems revision has the host-only environment parser test
# sources, but predates the BUILD_HOST_TESTS CMake wiring added by
# rocm-systems#11582. Keep this target independent of rocSHMEM, HIP, HSA, and
# MPI so it is safe to package and execute on a CPU-only sanitizer runner.
if(BUILD_HOST_TESTS)
  find_package(GTest CONFIG REQUIRED)

  add_executable(rocshmem_envvar_test
    "${CMAKE_SOURCE_DIR}/tests/unit_tests/envvar_gtest.cpp"
    "${CMAKE_SOURCE_DIR}/src/envvar.cpp"
  )
  target_include_directories(rocshmem_envvar_test PRIVATE
    "${CMAKE_SOURCE_DIR}/tests/unit_tests"
    "${CMAKE_SOURCE_DIR}/tests"
    "${CMAKE_SOURCE_DIR}/src"
  )
  target_link_libraries(rocshmem_envvar_test PRIVATE
    GTest::gtest
    GTest::gtest_main
  )

  # The global post-subproject hook snapshots targets before it delegates to
  # this file. Register this late-created executable so the remaining RPATH,
  # sanitizer-runtime, build-ID, and split-debug post-processing sees it.
  list(APPEND THEROCK_EXECUTABLE_TARGETS rocshmem_envvar_test)

  install(TARGETS rocshmem_envvar_test
    COMPONENT tests
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  )
endif()
