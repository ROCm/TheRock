cmake_minimum_required(VERSION 3.18)

project(BOOST_BUILD)

include(ProcessorCount)
ProcessorCount(PROCESSOR_COUNT)

# Limit Boost B2 threads to 1 by default unless more than 8 cpu threads available
set(B2_NJOBS "1")
if(PROCESSOR_COUNT GREATER 8)
  set(B2_NJOBS "8")
endif()
message(STATUS "Setting b2 jobs to: ${B2_NJOBS} / ${PROCESSOR_COUNT} cpu threads")

set(_terminal_option)
if("$ENV{THEROCK_INTERACTIVE}")
  set(_terminal_option "USES_TERMINAL")
endif()

set(_b2_args -j "${B2_NJOBS}" link=static threading=multi variant=release)

if(WIN32)
  # The boost batch file does not handle --with-libraries and silently builds
  # everything. So we run the bootstrap command. And then we run another command
  # that we generate in order to add the libraries stanza.
  # Have I mentioned before how painful it is to integrate Boost?
  string(REPLACE "," ";" boost_libraries_list "${THEROCK_BOOST_LIBRARIES}")
  set(_config_content "echo libraries =")
  foreach(_boost_lib ${boost_libraries_list})
    string(APPEND _config_content " --with-${_boost_lib}")
  endforeach()
  string(APPEND _config_content " ; >> project-config.jam\n\n")
  file(CONFIGURE OUTPUT "${BOOST_SOURCE_DIR}/fix_libraries.bat"
    CONTENT "${_config_content}"
  )
  message(STATUS "Adding windows boost config: ${_config_content}")
  set(_bootstrap_commands
    COMMAND
      "cmd.exe" "/C" "bootstrap.bat"
    COMMAND
      "cmd.exe" "/C" "fix_libraries.bat"
  )

  if(MSVC)
    list(APPEND _b2_args msvc)
  endif()
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    # Don't generate x32 libraries.
    list(APPEND _b2_args "architecture=x86" "address-model=64")
  endif()
else()
  # The unix bootstrap script takes a --with-libraries parameter.
  list(APPEND _b2_args "cflags=-fPIC")
  set(_bootstrap_commands
    COMMAND
      bash "bootstrap.sh" "--with-libraries=${THEROCK_BOOST_LIBRARIES}"
  )
endif()

message(STATUS "b2 args: ${_b2_args}")

add_custom_target(
  bootstrap_and_build ALL
  ${_terminal_option}
  WORKING_DIRECTORY "${BOOST_SOURCE_DIR}"
  ${_bootstrap_commands}
  COMMAND
    "${BOOST_SOURCE_DIR}/b2" ${_b2_args}
      --stagedir="${CMAKE_BINARY_DIR}/b2_out"
      --prefix="${CMAKE_INSTALL_PREFIX}" install
)
