# We want to be able to use the static build catch2 lib in shared libraries.
message(STATUS "Enabling PIC build")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
