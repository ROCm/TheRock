# See the artifact descriptor where we require these matrices for the test
# artifact. Consider installing as part of the main project.
install(
  DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/clients/matrices"
  DESTINATION "clients"
)
