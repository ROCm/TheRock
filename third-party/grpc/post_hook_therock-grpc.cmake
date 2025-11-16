set(_grpc_patch_script "${CMAKE_CURRENT_LIST_DIR}/patch_install.sh")
if(NOT EXISTS "${_grpc_patch_script}")
  message(FATAL_ERROR "gRPC post-install hook: patch_install.sh not found at ${_grpc_patch_script}")
endif()

if(NOT DEFINED THEROCK_SOURCE_DIR OR THEROCK_SOURCE_DIR STREQUAL "")
  message(FATAL_ERROR "gRPC post-install hook: THEROCK_SOURCE_DIR is not defined")
endif()

set(_zlib_stage_lib_dir "${THEROCK_GRPC_ZLIB_STAGE_LIB_DIR}")

set(_grpc_post_install_code [=[
  set(_grpc_patch_script "@_grpc_patch_script@")
  set(_zlib_stage_lib_dir "@_zlib_stage_lib_dir@")
  set(_cmake_command "@CMAKE_COMMAND@")
  set(_working_directory "@CMAKE_CURRENT_BINARY_DIR@")

  message(STATUS "Running gRPC post-install patch for ${CMAKE_INSTALL_PREFIX}")

  execute_process(
    COMMAND "${_cmake_command}" -E env
      "THEROCK_SOURCE_DIR=@THEROCK_SOURCE_DIR@"
      --
      bash "${_grpc_patch_script}" "${CMAKE_INSTALL_PREFIX}"
    WORKING_DIRECTORY "${_working_directory}"
    RESULT_VARIABLE _grpc_patch_result
    OUTPUT_VARIABLE _grpc_patch_output
    ERROR_VARIABLE _grpc_patch_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
  )

  if(NOT _grpc_patch_result EQUAL 0)
    message(FATAL_ERROR
      "gRPC post-install patch failed with exit code ${_grpc_patch_result}\n"
      "Output:\n${_grpc_patch_output}\n"
      "Error:\n${_grpc_patch_error}\n")
  endif()

  if(_grpc_patch_output)
    message(STATUS "${_grpc_patch_output}")
  endif()
  if(_grpc_patch_error)
    message(STATUS "${_grpc_patch_error}")
  endif()

  if(_zlib_stage_lib_dir AND EXISTS "${_zlib_stage_lib_dir}")
    set(_grpc_lib_dir "${CMAKE_INSTALL_PREFIX}/lib")
    if(EXISTS "${_grpc_lib_dir}")
      file(GLOB _zlib_shared_libs
        "${_zlib_stage_lib_dir}/librocm_sysdeps_z.so"
        "${_zlib_stage_lib_dir}/librocm_sysdeps_z.so.*"
      )
      foreach(_zlib_lib ${_zlib_shared_libs})
        get_filename_component(_lib_name "${_zlib_lib}" NAME)
        set(_dest "${_grpc_lib_dir}/${_lib_name}")
        if(NOT EXISTS "${_dest}")
          file(CREATE_LINK "${_zlib_lib}" "${_dest}" SYMBOLIC)
          message(STATUS "Linked ${_dest} -> ${_zlib_lib}")
        endif()
      endforeach()
    endif()
  endif()
]=])

string(CONFIGURE "${_grpc_post_install_code}" _grpc_post_install_code @ONLY)
install(CODE "${_grpc_post_install_code}")
