# Globals defined for the entire super-project.

# THEROCK_CONDITION_* variables
# Often-times in sub-project definitions we need to chain conditions to
# sub-project cache variables, we need a variable that exactly represents
# the condition (since expressions are not valid in expansions).
# For platform checks and other feature checks, define such condition
# variables here for use by the whole project.

# THEROCK_CONDITION_WINDOWS / THEROCK_CONDITION_IS_NON_WINDOWS
set(THEROCK_CONDITION_IS_WINDOWS OFF)
set(THEROCK_CONDITION_IS_NON_WINDOWS ON)
if(WIN32)
  set(THEROCK_CONDITION_IS_WINDOWS ON)
  set(THEROCK_CONDITION_IS_NON_WINDOWS OFF)
endif()

# THEROCK_CONDITION_IS_MACOS / THEROCK_CONDITION_IS_LINUX
set(THEROCK_CONDITION_IS_MACOS OFF)
set(THEROCK_CONDITION_IS_LINUX OFF)
if(APPLE)
  set(THEROCK_CONDITION_IS_MACOS ON)
endif()
if(UNIX AND NOT APPLE)
  set(THEROCK_CONDITION_IS_LINUX ON)
endif()
