# This finder resolves the virtual FFTW package for sub-projects.
# In reality the FFTW library installed in FFTW3 but we set also the
# FFTW_INCLUDE_DIRS and FFTW_LIBRARIES env variables without "3"
cmake_policy(PUSH)
cmake_policy(SET CMP0057 NEW)

message(STATUS "FindFFTW.cmake: Checking is FFTW in THEROCK_PROVIDED_PACKAGES")
if("FFTW3" IN_LIST THEROCK_PROVIDED_PACKAGES)
  cmake_policy(POP)
  message(STATUS "FindFFTW.cmake: Resolving FFTW3 and FFTW3f libraries from super-project")
  find_package(FFTW3 CONFIG REQUIRED COMPONENTS FLOAT DOUBLE)
  find_package(FFTW3f CONFIG REQUIRED)
  #set(FFTW_LIBRARY_DIRS ${FFTW3_LIBRARY_DIRS})
  #set(FFTW_INCLUDE_DIRS ${FFTW3_INCLUDE_DIRS})
  set(FFTW_FOUND TRUE)
  set(FFTWf_FOUND TRUE)
else()
  cmake_policy(POP)
  set(FFTW_FOUND FALSE)
  set(FFTW3_FOUND FALSE)
  set(FFTW3f_FOUND FALSE)
endif()
