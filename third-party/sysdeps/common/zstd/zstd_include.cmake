message(STATUS "Customizing zlib options for TheRock")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_LIBDIR "lib")  # No lib64 for us, thank you very much.
