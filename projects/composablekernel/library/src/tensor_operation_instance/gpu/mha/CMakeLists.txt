# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(FMHA_CPP_FOLDER ${CMAKE_CURRENT_BINARY_DIR})
set(FMHA_SRC_FOLDER ${CMAKE_SOURCE_DIR}/example/ck_tile/01_fmha/)
set(CK_TILE_SRC_FOLDER ${CMAKE_SOURCE_DIR}/include/ck_tile/)

# Usage: for customized Python location cmake -DCK_USE_ALTERNATIVE_PYTHON="/opt/Python-3.8.13/bin/python3.8"
# CK Codegen requires dataclass which is added in Python 3.7
# Python version 3.8 is required for general good practice as it is default for Ubuntu 20.04
if(NOT CK_USE_ALTERNATIVE_PYTHON)
   find_package(Python3 COMPONENTS Interpreter Development)
else()
   message(STATUS "Using alternative python version")
   set(EXTRA_PYTHON_PATH)
   # this is overly restrictive, we may need to be more flexible on the following
   string(REPLACE "/bin/python3.8" "" EXTRA_PYTHON_PATH "${CK_USE_ALTERNATIVE_PYTHON}")
   message(STATUS "alternative python path is: ${EXTRA_PYTHON_PATH}")
   find_package(Python3 3.6 COMPONENTS Interpreter REQUIRED)
   add_definitions(-DPython3_EXECUTABLE="${CK_USE_ALTERNATIVE_PYTHON}")
   set(Python3_EXECUTABLE "${CK_USE_ALTERNATIVE_PYTHON}")
   set(PYTHON_EXECUTABLE "${CK_USE_ALTERNATIVE_PYTHON}")
   set(ENV{LD_LIBRARY_PATH} "${EXTRA_PYTHON_PATH}/lib:$ENV{LD_LIBRARY_PATH}")
endif()

rocm_install(DIRECTORY ${CK_TILE_SRC_FOLDER} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ck_tile)

file(GLOB MHA_HEADERS "${FMHA_SRC_FOLDER}/*.hpp")
rocm_install(FILES ${MHA_HEADERS} DESTINATION include/ck_tile/ops)
# headers for building lib
file(COPY ${MHA_HEADERS} DESTINATION ${FMHA_CPP_FOLDER})

set(FMHA_KNOWN_APIS "fwd,fwd_splitkv,fwd_appendkv,bwd")

# generate a list of kernels, but not actually emit files at config stage
# Note: The receipt 3 arg filters the generated backwards instances to reduce compilation time.
# With receipt 3 set, we are generating instances for datatype == {fp16 || bfp16}, bias == {no || alibi}, deterministic == off, and dpad == dvpad.
execute_process(
  COMMAND ${Python3_EXECUTABLE} ${FMHA_SRC_FOLDER}/generate.py
  --list_blobs ${FMHA_CPP_FOLDER}/blob_list.txt
  --api ${FMHA_KNOWN_APIS}
  --receipt 3
  RESULT_VARIABLE ret
)
if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile MHA FAILED to generate a list of kernels via Python.")
else()
  file(STRINGS ${FMHA_CPP_FOLDER}/blob_list.txt FMHA_GEN_BLOBS)
endif()

# actually generate the kernel content now
# Note: The receipt 3 arg filters the generated backwards instances to reduce compilation time.
# With receipt 3 set, we are generating instances for datatype == {fp16 || bfp16}, bias == {no || alibi}, deterministic == off, and dpad == dvpad.
add_custom_command(
  OUTPUT ${FMHA_GEN_BLOBS}
  COMMAND ${Python3_EXECUTABLE} ${FMHA_SRC_FOLDER}/generate.py
  --output_dir ${FMHA_CPP_FOLDER}
  --api ${FMHA_KNOWN_APIS}
  --receipt 3
  COMMENT "Generating mha kernel (cpp) files now ..."
  VERBATIM
)

# This is done to remove path info and just
# have filename. Since, it was cauing the cmake
# to throw "File name too long"
set(device_files)
foreach(filepath IN LISTS FMHA_GEN_BLOBS)
    get_filename_component(filename ${filepath} NAME)
    # Append the filename to the device_files list
    list(APPEND device_files ${filename})
endforeach()
add_custom_target(generate_cpp_files DEPENDS ${FMHA_GEN_BLOBS})

add_instance_library(device_mha_instance ${device_files})

if (TARGET device_mha_instance)
  add_dependencies(device_mha_instance generate_cpp_files)
endif()
