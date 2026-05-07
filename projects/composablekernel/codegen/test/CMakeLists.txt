# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
add_subdirectory(rtc)
file(GLOB TEST_SRCS CONFIGURE_DEPENDS *.cpp)

# TODO: These tests need to be refactored to remove dependency on main ck
# headers and device compilation.
set(TESTS_REQUIRE_DEVICE_COMPILE 
  grouped_conv_fwd_multiple_d_v1
  grouped_conv_fwd_multiple_d_v2
  grouped_conv_fwd_multiple_d_v3
  grouped_conv_fwd_multiple_d_v4
)
find_package(hip)

foreach(TEST_SRC ${TEST_SRCS})
  get_filename_component(BASE_NAME ${TEST_SRC} NAME_WE)
  rocm_add_test_executable(codegen_test_${BASE_NAME} ${TEST_SRC})
  target_link_libraries(codegen_test_${BASE_NAME} ck_rtc ck_host)
  target_include_directories(codegen_test_${BASE_NAME} PUBLIC include)
  if(BASE_NAME IN_LIST TESTS_REQUIRE_DEVICE_COMPILE)
    target_link_libraries(codegen_test_${BASE_NAME} hip::device)
    target_include_directories(codegen_test_${BASE_NAME} PUBLIC ${CK_ROOT}/include)
    target_include_directories(codegen_test_${BASE_NAME} PUBLIC ${CK_ROOT}/library/include)
  endif()
endforeach()
