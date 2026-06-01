# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

find_package(hip)
file(GLOB RTC_SOURCES CONFIGURE_DEPENDS src/*.cpp)
add_library(ck_rtc ${RTC_SOURCES})
target_include_directories(ck_rtc PUBLIC include)
target_link_libraries(ck_rtc PUBLIC hip::host)
target_link_libraries(ck_rtc PUBLIC -lstdc++fs)

option(USE_HIPRTC_FOR_CODEGEN_TESTS "Whether to enable hipRTC for codegen tests." ON)
if(USE_HIPRTC_FOR_CODEGEN_TESTS)
    target_compile_definitions(ck_rtc PUBLIC HIPRTC_FOR_CODEGEN_TESTS)
    message(STATUS "CK compiled with USE_HIPRTC_FOR_CODEGEN_TESTS set to ${USE_HIPRTC_FOR_CODEGEN_TESTS}")
endif()
