# Attempt (best effort) to return a list of user specified parameters cmake was invoked with
# NOTE: Even if the user specifies CMAKE_INSTALL_PREFIX on the command line, the parameter is
# not returned because it does not have the matching helpstring

function( append_cmake_cli_arguments initial_cli_args return_cli_args )

  # Retrieves the contents of CMakeCache.txt
  get_cmake_property( cmake_properties CACHE_VARIABLES )

  foreach( property ${cmake_properties} )
    get_property(help_string CACHE ${property} PROPERTY HELPSTRING )

    # Properties specified on the command line have boilerplate text
    if( help_string MATCHES "variable specified on the command line" )
      # message( STATUS "property: ${property}")
      # message( STATUS "value: ${${property}}")

      list( APPEND cli_args "-D${property}=${${property}}")
    endif( )
  endforeach( )

  # message( STATUS "get_command_line_arguments: ${cli_args}")
  set( ${return_cli_args} ${${initial_cli_args}} ${cli_args} PARENT_SCOPE )

endfunction( )
