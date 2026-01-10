If your CMake project links to some library which you would like to replace with Implib.so wrapper:
```
cmake_minimum_required(VERSION 3.21)

project(prog)

add_executable(prog
  prog.c
)

target_link_libraries(prog PRIVATE xcb)
```
you just need to add a custom command to generate wrapper code and link against `libdl`:
```
cmake_minimum_required(VERSION 3.21)

project(prog)

enable_language(ASM)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libxcb.so.tramp.S
         ${CMAKE_CURRENT_BINARY_DIR}/libxcb.so.init.c
  COMMAND implib-gen.py -q /usr/lib/x86_64-linux-gnu/libxcb.so
  DEPENDS /usr/lib/x86_64-linux-gnu/libxcb.so
)

add_executable(prog
  prog.c
  ${CMAKE_CURRENT_BINARY_DIR}/libxcb.so.tramp.S
  ${CMAKE_CURRENT_BINARY_DIR}/libxcb.so.init.c
)

target_link_libraries(prog PRIVATE dl)
```
