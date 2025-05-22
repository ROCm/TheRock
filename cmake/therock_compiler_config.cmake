# Project wide compiler configuration.

# On win32 only support embedded debug databases project wide.
if(WIN32 AND NOT DEFINED CMAKE_MSVC_DEBUG_INFORMATION_FORMAT)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<$<CONFIG:Debug,RelWithDebInfo>:Embedded>")
endif()
