# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# temporarily disable flaky test for all architectures
add_definitions(-DCK_SKIP_FLAKY_F8_TEST)
set(CK_SKIP_FLAKY_F8_TEST "ON")

if (USE_BITINT_EXTENSION_INT4)
  add_gtest_executable(test_int4 test_int4.cpp)
  if(result EQUAL 0)
    target_link_libraries(test_int4 PRIVATE utility)
  endif()
endif()

add_custom_target(test_fp8)

if (CK_USE_OCP_FP8)
  # add test for ocp data types
  add_gtest_executable(test_fp8_ocp test_fp8_ocp.cpp)
  if(result EQUAL 0)
    target_link_libraries(test_fp8_ocp PRIVATE utility)
    add_dependencies(test_fp8 test_fp8_ocp)
  endif()

  add_gtest_executable(test_bf8_ocp test_bf8_ocp.cpp)
  if(result EQUAL 0)
    target_link_libraries(test_bf8_ocp PRIVATE utility)
    add_dependencies(test_fp8 test_bf8_ocp)
  endif()

endif()

if (CK_USE_FNUZ_FP8)
  add_gtest_executable(test_fp8_fnuz test_fp8_fnuz.cpp)
  if(result EQUAL 0)
    target_link_libraries(test_fp8_fnuz PRIVATE utility)
  endif()

  add_gtest_executable(test_bf8_fnuz test_bf8_fnuz.cpp)
  if(result EQUAL 0)
    target_link_libraries(test_bf8_fnuz PRIVATE utility)
  endif()

  add_dependencies(test_fp8 test_fp8_fnuz)
  add_dependencies(test_fp8 test_bf8_fnuz)
endif()

if(GPU_TARGETS MATCHES "gfx950")
  add_custom_target(test_mx_data_types)

  add_gtest_executable(test_fp4 test_fp4.cpp)
  if(result EQUAL 0)
    target_link_libraries(test_fp4 PRIVATE utility)
  endif()
  add_dependencies(test_mx_data_types test_fp4)

  add_gtest_executable(test_fp6 test_fp6.cpp)
  if(result EQUAL 0)
    target_compile_options(test_fp6 PRIVATE -mavx512f)
    target_link_libraries(test_fp6 PRIVATE utility)
  endif()
  add_dependencies(test_mx_data_types test_fp6)

  add_gtest_executable(test_bf6 test_bf6.cpp)
  if(result EQUAL 0)
    target_compile_options(test_bf6 PRIVATE -mavx512f)
    target_link_libraries(test_bf6 PRIVATE utility)
  endif()
  add_dependencies(test_mx_data_types test_bf6)

  add_gtest_executable(test_mx_fp8 test_mx_fp8.cpp)
  if(result EQUAL 0)
    target_link_libraries(test_mx_fp8 PRIVATE utility)
  endif()
  add_dependencies(test_mx_data_types test_mx_fp8)

  add_gtest_executable(test_mx_bf8 test_mx_bf8.cpp)
  if(result EQUAL 0)
    target_link_libraries(test_mx_bf8 PRIVATE utility)
  endif()
  add_dependencies(test_mx_data_types test_mx_bf8)

  add_gtest_executable(test_mx_fp4 test_mx_fp4.cpp)
  if(result EQUAL 0)
    target_link_libraries(test_mx_fp4 PRIVATE utility)
  endif()
  add_dependencies(test_mx_data_types test_mx_fp4)

  add_gtest_executable(test_e8m0 test_e8m0.cpp)
  if(result EQUAL 0)
    target_link_libraries(test_e8m0 PRIVATE utility)
  endif()
  add_dependencies(test_mx_data_types test_e8m0)
endif()
add_gtest_executable(test_custom_type test_custom_type.cpp)
if(result EQUAL 0)
  target_link_libraries(test_custom_type PRIVATE utility)
endif()

add_gtest_executable(test_type_convert_const type_convert_const.cpp)
add_gtest_executable(test_bhalf test_bhalf.cpp)
add_gtest_executable(test_pk_i4 test_pk_i4.cpp)
