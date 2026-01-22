# See the artifact descriptor where we require these matrices for the test
# artifact. Consider installing as part of the main project.
# The client will only be built if testing is enabled. See hipSPARSE CMakeLists.txt.
if(THEROCK_BUILD_TESTING)
  install(
    DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/clients/matrices"
    DESTINATION "clients"
  )
endif()
