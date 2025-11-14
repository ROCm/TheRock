# therock_setup_linker.cmake
# Detect and configure LLD linker if available for Clang or GCC on Linux/Windows.

# Compiler and system info
set(_compiler_id "${CMAKE_CXX_COMPILER_ID}")
set(_system_name "${CMAKE_SYSTEM_NAME}")

set(LLD_PATH "")

# Function to try finding LLD
macro(find_lld)
    # First try CMake package
    find_package(LLD QUIET)
    if(LLD_FOUND AND EXISTS "${LLD_EXECUTABLE}")
        set(LLD_PATH "${LLD_EXECUTABLE}")
    else()
        # Fall back to system PATH
        if(WIN32)
            find_program(LLD_PATH NAMES lld-link)
        else()
            find_program(LLD_PATH NAMES ld.lld lld)
        endif()
    endif()
endmacro()

# Determine if we can use LLD
if(NOT WIN32 OR _compiler_id MATCHES "Clang")
    set(LLD_PATH)
    find_lld()

    if(LLD_PATH)
        message(STATUS "Configuring sub-project to use LLD: ${LLD_PATH}")

        # Set CMake linker path
        set(CMAKE_LINKER "${LLD_PATH}" CACHE FILEPATH "Path to system LLD linker" FORCE)

        if(NOT WIN32)
            set(CMAKE_EXE_LINKER_FLAGS_INIT "${CMAKE_EXE_LINKER_FLAGS_INIT} -fuse-ld=lld")
            set(CMAKE_SHARED_LINKER_FLAGS_INIT "${CMAKE_SHARED_LINKER_FLAGS_INIT} -fuse-ld=lld")
            set(CMAKE_MODULE_LINKER_FLAGS_INIT "${CMAKE_MODULE_LINKER_FLAGS_INIT} -fuse-ld=lld")
        endif()
    else()
        message(STATUS "LLD not found on system PATH, using default linker")
    endif()
else()
    message(STATUS "Compiler ${_compiler_id} does not support LLD — skipping")
endif()
