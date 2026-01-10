# Platform-specific post-install patches for gRPC
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(_grpc_patch_script "${CMAKE_CURRENT_LIST_DIR}/patch_install_linux.sh")
  if(NOT EXISTS "${_grpc_patch_script}")
    message(FATAL_ERROR "gRPC post-install patch script not found: ${_grpc_patch_script}")
  endif()

  set(_grpc_post_install_code [=[
    set(_grpc_patch_script "@_grpc_patch_script@")
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
  ]=])

  string(CONFIGURE "${_grpc_post_install_code}" _grpc_post_install_code @ONLY)
  install(CODE "${_grpc_post_install_code}")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  # TODO: Add Windows-specific post-install patch when needed
  message(STATUS "gRPC Windows post-install: no patches needed yet")
endif()
