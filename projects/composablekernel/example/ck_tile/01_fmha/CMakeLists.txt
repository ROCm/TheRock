# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(INST_TARGETS ${SUPPORTED_GPU_TARGETS})
# Currently only gfx9 and gfx12 archs are supported by FMHA
list(FILTER INST_TARGETS INCLUDE REGEX "gfx9|gfx12")
if(NOT INST_TARGETS)
  message(WARNING "Skipping Tile Engine FMHA compilation: No supported GPU targets (gfx9, gfx12) found in SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")
  return()
endif()

# validate user-specified fmha_fwd API list
set(FMHA_FWD_KNOWN_APIS "fwd;fwd_splitkv;fwd_appendkv;pagedkv_prefill")
set(FMHA_FWD_ENABLE_APIS "fwd" CACHE STRING
  "semicolon-separated list of APIs to generate (${FMHA_FWD_KNOWN_APIS}) & link, or \"all\".")
if(BUILD_TESTING)
  # Build instances of all APIs for tests
  message(DEBUG "Enabling all FWD APIs of CK Tile FMHA for because testing is enabled")
  set(FMHA_FWD_ENABLE_APIS "all")
endif()
if(FMHA_FWD_ENABLE_APIS STREQUAL "all")
  set(FMHA_FWD_ENABLE_APIS ${FMHA_FWD_KNOWN_APIS})
endif()

foreach(api ${FMHA_FWD_ENABLE_APIS})
  if(NOT "${api}" IN_LIST FMHA_FWD_KNOWN_APIS)
    message(FATAL_ERROR "${api} isn't a known api: ${FMHA_FWD_KNOWN_APIS}.")
  endif()
endforeach()

# "fwd" is a must-have api for the fmha_fwd example, add it if not specified
if(NOT "fwd" IN_LIST FMHA_FWD_ENABLE_APIS)
  list(PREPEND FMHA_FWD_ENABLE_APIS "fwd")
endif()

file(GLOB_RECURSE CODE_GEN_SCRIPTS CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_LIST_DIR}/generate.py
  ${CMAKE_CURRENT_LIST_DIR}/codegen/*.py
)
# re-run execute_process `generate.py --list_blobs` if any of the codegen scripts change
set_directory_properties(PROPERTIES CMAKE_CONFIGURE_DEPENDS "${CODE_GEN_SCRIPTS}")

list(JOIN INST_TARGETS , FMHA_TARGETS_ARG)

string(REPLACE ";" "," FMHA_FWD_APIS "${FMHA_FWD_ENABLE_APIS}")
set(FMHA_FWD_CODE_GEN_COMMON_ARGS
  ${CMAKE_CURRENT_LIST_DIR}/generate.py
  --targets ${FMHA_TARGETS_ARG}
  --api ${FMHA_FWD_APIS}
  --optdim 32,64,80,128,256
  # --filter fmha_fwd...
)
set(FMHA_BWD_CODE_GEN_COMMON_ARGS
  ${CMAKE_CURRENT_LIST_DIR}/generate.py
  --targets ${FMHA_TARGETS_ARG}
  --api bwd
  --receipt 3
  --optdim 32,64,96,128,256
  # --filter fmha_bwd_dot...@fmha_bwd_convert...@fmha_bwd...
)

# Reduce building time by disabling instances that are not currently used in the gtests
# TODO: Consider to use a special receipt for testing only, or even two receipts: a small subset of
# instances for quick CI runs and a larger subset for scheduled runs (the tests skip tests when
# there is no corresponding instance for parameters).
if(BUILD_TESTING)
  # Filters are in the order of FMHA_FWD_KNOWN_APIS: fwd,fwd_splitkv_combine@fwd_splitkv,fwd_appendkv,pagedkv_prefill
  list(APPEND FMHA_FWD_CODE_GEN_COMMON_ARGS --filter *_nlogits*_nskip*_nsink*,*@*_nlogits*_nbias*_nsink*,*,*_nlogits*_nskip*_pagedkv*)
endif()

# generate a list of kernels, but not actually emit files at config sta
execute_process(
  COMMAND ${Python3_EXECUTABLE} ${FMHA_FWD_CODE_GEN_COMMON_ARGS}
  --list_blobs ${CMAKE_CURRENT_BINARY_DIR}/fwd_blob_list.txt
  RESULT_VARIABLE ret
)
if(ret AND NOT ret EQUAL 0)
  message(FATAL_ERROR "CK Tile FMHA FAILED to generate a list of FWD kernels via Python.")
endif()

execute_process(
  COMMAND ${Python3_EXECUTABLE} ${FMHA_BWD_CODE_GEN_COMMON_ARGS}
  --list_blobs ${CMAKE_CURRENT_BINARY_DIR}/bwd_blob_list.txt
  RESULT_VARIABLE ret
)
if(ret AND NOT ret EQUAL 0)
  message(FATAL_ERROR "CK Tile FMHA FAILED to generate a list of BWD kernels via Python.")
endif()

# NOTE: for cmake, the FMHA_FWD_GEN_BLOBS/FMHA_BWD_GEN_BLOBS files must be in the same directory
#       as current cmake list, otherwise will not figure out the dependency properly
file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/fwd_blob_list.txt FMHA_FWD_GEN_BLOBS)
file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/bwd_blob_list.txt FMHA_BWD_GEN_BLOBS)

add_custom_command(
  OUTPUT ${FMHA_FWD_GEN_BLOBS}
  COMMAND ${Python3_EXECUTABLE} ${FMHA_FWD_CODE_GEN_COMMON_ARGS}
  --output_dir ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CODE_GEN_SCRIPTS}
  COMMENT "Generate CK Tile FMHA FWD kernels"
)

add_custom_command(
  OUTPUT ${FMHA_BWD_GEN_BLOBS}
  COMMAND ${Python3_EXECUTABLE} ${FMHA_BWD_CODE_GEN_COMMON_ARGS}
  --output_dir ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CODE_GEN_SCRIPTS}
  COMMENT "Generate CK Tile FMHA BWD kernels"
)

set(FMHA_FWD_INSTANCES "tile_fmha_fwd_instances")
set(FMHA_BWD_INSTANCES "tile_fmha_bwd_instances")

message(DEBUG "adding instances ${FMHA_FWD_INSTANCES}")
# to save build time, exclude the target from "all" target of "01_fmha" directory and its ancestors
add_library(${FMHA_FWD_INSTANCES} OBJECT EXCLUDE_FROM_ALL)
target_include_directories(${FMHA_FWD_INSTANCES} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_sources(${FMHA_FWD_INSTANCES} PRIVATE ${FMHA_FWD_GEN_BLOBS})
set_source_files_properties(${FMHA_FWD_GEN_BLOBS} PROPERTIES LANGUAGE HIP)
set_property(TARGET ${FMHA_FWD_INSTANCES} PROPERTY HIP_ARCHITECTURES ${INST_TARGETS})

message(DEBUG "adding instances ${FMHA_BWD_INSTANCES}")
add_library(${FMHA_BWD_INSTANCES} OBJECT EXCLUDE_FROM_ALL)
target_include_directories(${FMHA_BWD_INSTANCES} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_sources(${FMHA_BWD_INSTANCES} PRIVATE ${FMHA_BWD_GEN_BLOBS})
set_source_files_properties(${FMHA_BWD_GEN_BLOBS} PROPERTIES LANGUAGE HIP)
set_property(TARGET ${FMHA_BWD_INSTANCES} PROPERTY HIP_ARCHITECTURES ${INST_TARGETS})

set(FMHA_FWD_PRIVATE_COMPILE_OPTIONS)
set(FMHA_BWD_PRIVATE_COMPILE_OPTIONS)
set(FMHA_FWD_INTERFACE_COMPILE_OPTIONS)
set(FMHA_BWD_INTERFACE_COMPILE_OPTIONS)

# NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
#       ... because they are auto-generated
list(APPEND FMHA_FWD_PRIVATE_COMPILE_OPTIONS -Wno-undefined-func-template)
list(APPEND FMHA_BWD_PRIVATE_COMPILE_OPTIONS -Wno-undefined-func-template)

# Allow comparing floating points directly in order to check sentinel values
list(APPEND FMHA_FWD_PRIVATE_COMPILE_OPTIONS -Wno-float-equal)
list(APPEND FMHA_BWD_PRIVATE_COMPILE_OPTIONS -Wno-float-equal)

# NOTE: this is dangerous since will change the whole kernel to flush denormals
#       WIP with compiler team for an exp2 intrinsic..., then remove this
if(NOT DEFINED FMHA_FWD_FAST_EXP2)
  set(FMHA_FWD_FAST_EXP2 ON)
endif()

if(FMHA_FWD_FAST_EXP2)
  list(APPEND FMHA_FWD_PRIVATE_COMPILE_OPTIONS -DCK_TILE_FMHA_FWD_FAST_EXP2=1 -fgpu-flush-denormals-to-zero)
else()
  list(APPEND FMHA_FWD_PRIVATE_COMPILE_OPTIONS -DCK_TILE_FMHA_FWD_FAST_EXP2=0)
endif()
list(APPEND FMHA_BWD_PRIVATE_COMPILE_OPTIONS -fgpu-flush-denormals-to-zero)

# conditionally enable call to the fwd_splitkv API in fmha_fwd example and tests
if("fwd_splitkv" IN_LIST FMHA_FWD_ENABLE_APIS)
  list(APPEND FMHA_FWD_INTERFACE_COMPILE_OPTIONS -DCK_TILE_FMHA_FWD_SPLITKV_API=1)
else()
  list(APPEND FMHA_FWD_INTERFACE_COMPILE_OPTIONS -DCK_TILE_FMHA_FWD_SPLITKV_API=0)
endif()

# conditionally enable call to the fwd_appendkv API in fmha_fwd example and tests
if("fwd_appendkv" IN_LIST FMHA_FWD_ENABLE_APIS)
  list(APPEND FMHA_FWD_INTERFACE_COMPILE_OPTIONS -DCK_TILE_FMHA_FWD_APPENDKV_API=1)
else()
  list(APPEND FMHA_FWD_INTERFACE_COMPILE_OPTIONS -DCK_TILE_FMHA_FWD_APPENDKV_API=0)
endif()

# conditionally enable call to the pagedkv_prefill API in fmha_fwd example and tests
if("pagedkv_prefill" IN_LIST FMHA_FWD_ENABLE_APIS)
  list(APPEND FMHA_FWD_INTERFACE_COMPILE_OPTIONS -DCK_TILE_FMHA_FWD_PAGEDKV_API=1)
else()
  list(APPEND FMHA_FWD_INTERFACE_COMPILE_OPTIONS -DCK_TILE_FMHA_FWD_PAGEDKV_API=0)
endif()

# conditionally specify the use of OCP_FP8
if(CK_USE_OCP_FP8)
  list(APPEND FMHA_FWD_PRIVATE_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
  list(APPEND FMHA_FWD_INTERFACE_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

# use RTN_ASM on float to bfloat16 conversion by default, align with FA upstream
list(APPEND FMHA_BWD_PRIVATE_COMPILE_OPTIONS -DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3)
list(APPEND FMHA_BWD_INTERFACE_COMPILE_OPTIONS -DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3)

target_compile_options(${FMHA_FWD_INSTANCES}
  PRIVATE ${FMHA_FWD_PRIVATE_COMPILE_OPTIONS}
  INTERFACE ${FMHA_FWD_INTERFACE_COMPILE_OPTIONS})
target_compile_options(${FMHA_BWD_INSTANCES} 
  PRIVATE ${FMHA_BWD_PRIVATE_COMPILE_OPTIONS}
  INTERFACE ${FMHA_BWD_INTERFACE_COMPILE_OPTIONS})

set(EXAMPLE_FMHA_FWD "tile_example_fmha_fwd")
set(EXAMPLE_FMHA_BWD "tile_example_fmha_bwd")

message(DEBUG "adding example ${EXAMPLE_FMHA_FWD}")
# not using add_example_executable() to add this target, since we don't want this to be included in
# "make all/install/check"
add_executable(${EXAMPLE_FMHA_FWD} EXCLUDE_FROM_ALL example_fmha_fwd.cpp)
target_link_libraries(${EXAMPLE_FMHA_FWD} ${FMHA_FWD_INSTANCES})
target_include_directories(${EXAMPLE_FMHA_FWD} PRIVATE ${CMAKE_CURRENT_LIST_DIR})

message(DEBUG "adding example ${EXAMPLE_FMHA_BWD}")
# not using add_example_executable() to add this target, since we don't want this to be included in
# "make all/install/check"
add_executable(${EXAMPLE_FMHA_BWD} EXCLUDE_FROM_ALL example_fmha_bwd.cpp)
target_link_libraries(${EXAMPLE_FMHA_BWD} ${FMHA_BWD_INSTANCES})
target_include_directories(${EXAMPLE_FMHA_BWD} PRIVATE ${CMAKE_CURRENT_LIST_DIR})

# TODO: we have to turn off this global prop, otherwise the progress bar generated
# by cmake will print too many files, execvp: /bin/sh: Argument list too long
# however, this property may affect global
# TODO: consider codegen a makefile by us
set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)
