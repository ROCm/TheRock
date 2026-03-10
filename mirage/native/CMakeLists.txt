cmake_minimum_required(VERSION 3.25)

project(MirageSimCore LANGUAGES CXX)

enable_testing()

add_library(mirage_sim_core STATIC
  generated/gfx950_instruction_catalog.cc
  src/gfx950_binary_decoder.cc
  src/mirage_sim_core.cc
  src/gfx950_interpreter.cc
  src/single_gpu_simulator.cc
  src/virtual_gpu_device.cc
)

target_compile_features(mirage_sim_core PUBLIC cxx_std_20)
target_include_directories(mirage_sim_core PUBLIC
  "${CMAKE_CURRENT_LIST_DIR}/.."
)

add_executable(mirage_instruction_catalog_test
  tests/instruction_catalog_test.cc
)
target_link_libraries(mirage_instruction_catalog_test PRIVATE mirage_sim_core)

add_executable(mirage_gfx950_interpreter_test
  tests/gfx950_interpreter_test.cc
)
target_link_libraries(mirage_gfx950_interpreter_test PRIVATE mirage_sim_core)

add_executable(mirage_gfx950_binary_decoder_test
  tests/gfx950_binary_decoder_test.cc
)
target_link_libraries(mirage_gfx950_binary_decoder_test PRIVATE mirage_sim_core)

add_executable(mirage_sim_core_test
  tests/single_gpu_simulator_test.cc
)
target_link_libraries(mirage_sim_core_test PRIVATE mirage_sim_core)

add_executable(mirage_gfx950_runtime_test
  tests/gfx950_runtime_test.cc
)
target_link_libraries(mirage_gfx950_runtime_test PRIVATE mirage_sim_core)

add_test(NAME mirage_instruction_catalog_test COMMAND mirage_instruction_catalog_test)
add_test(NAME mirage_gfx950_binary_decoder_test COMMAND mirage_gfx950_binary_decoder_test)
add_test(NAME mirage_gfx950_interpreter_test COMMAND mirage_gfx950_interpreter_test)
add_test(NAME mirage_sim_core_test COMMAND mirage_sim_core_test)
add_test(NAME mirage_gfx950_runtime_test COMMAND mirage_gfx950_runtime_test)
