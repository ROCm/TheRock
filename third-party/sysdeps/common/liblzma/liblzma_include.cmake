message(STATUS "Customizing xz/liblzma options for TheRock")
set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "" FORCE)
set(CMAKE_INSTALL_LIBDIR "lib")  # No lib64 for us, thank you very much.

# Force shared library build on Linux
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
endif()
